"""Sequence model over full KataGo-evaluated game trajectories (one token per move, both players).

Token = per-move evaluation features (score/winrate before, point loss, policy prob/rank of the played move,
position sharpness, ...) + move location embedding + target/opponent flag.
Per-game rank classifier (CE); exports dev/test per-game log-probs and game embeddings.
usage: seq_model.py CHUNK_DIR OUT_DIR [--epochs 12] [--d 192] [--layers 4] [--L 320] [--predict-only]
"""
import sys, os, glob, json, argparse, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn

NFEAT = 24


def tokens(a, mv, m, L):
    T = min(len(a) - 1, L)
    s = a[:, 1].astype(np.float64); w = a[:, 0].astype(np.float64)
    full_T = len(a) - 1
    black_first = m["first_player"] == "B"
    mover_black = (np.arange(full_T) % 2 == 0) == black_first
    sign = np.where(mover_black, 1.0, -1.0)
    loss = sign * (s[:-1] - s[1:]); wrl = sign * (w[:-1] - w[1:])
    mine = mover_black == (m["color"] == "B")
    b = a[:-1]
    mvv = mv[:full_T].astype(np.int64)
    is_pass = mvv >= 361
    x = np.where(is_pass, 9, mvv % 19); y = np.where(is_pass, 9, mvv // 19)
    line = np.minimum(np.minimum(x, 18 - x), np.minimum(y, 18 - y))
    def lag(v, k):
        o = np.zeros_like(v); o[k:] = np.abs(v[k:] - v[:-k]); return o
    f = np.stack([
        sign * (w[:-1] - 0.5) * 2, np.clip(sign * s[:-1], -60, 60) / 20, np.clip(loss, -15, 40) / 5, np.clip(wrl, -1, 1) * 3,
        np.clip(loss, 0, 3), (loss > 5).astype(float), b[:, 2] / 20, b[:, 3] * 5, b[:, 4] / 3,
        np.log(b[:, 6] + 1e-6) / 5, np.log1p(b[:, 7]) / 3, (b[:, 7] == 0).astype(float), b[:, 8], b[:, 9] / 3,
        np.log(b[:, 6] + 1e-6) / 5 - np.log(b[:, 8] + 1e-6) / 5, np.log(b[:, 10] + 1e-6) / 10, b[:, 11] / 361,
        np.arange(full_T) / 200.0, mine.astype(float), mover_black.astype(float), line / 9.0,
        np.minimum(np.maximum(lag(x, 1), lag(y, 1)), 10) / 10, np.minimum(np.maximum(lag(x, 2), lag(y, 2)), 10) / 10,
        np.full(full_T, m.get("hd", 0), float)], 1)
    return f[:T].astype(np.float16), mvv[:T].astype(np.int16)


def load(chunk_dir, L, limit=None):
    X, M, LEN, metas = [], [], [], []
    files = sorted(glob.glob(chunk_dir + "/chunk_*.npz"))
    for f in files[:limit]:
        z = np.load(f); meta = json.load(open(f[:-4] + ".json"))
        feats, moves, off = z["feats"], z["moves"], z["offset"]
        for i, m in enumerate(meta):
            a = feats[off[i]:off[i + 1]]
            if len(a) < 12:
                continue
            t, mv = tokens(a, moves[off[i]:off[i + 1]], m, L)
            x = np.zeros((L, NFEAT), np.float16); x[:len(t)] = t
            mm = np.full(L, 362, np.int16); mm[:len(mv)] = mv
            X.append(x); M.append(mm); LEN.append(len(t)); metas.append(m)
    return np.stack(X), np.stack(M), np.array(LEN), metas


def d4_tables():
    idx = np.arange(361).reshape(19, 19)
    tabs = []
    for k in range(4):
        for flip in (False, True):
            g = np.rot90(idx, k)
            if flip: g = g[:, ::-1]
            t = np.zeros(363, np.int64); t[g.reshape(-1)] = idx.reshape(-1); t[361] = 361; t[362] = 362
            tabs.append(t)
    return torch.tensor(np.stack(tabs))


class GameNet(nn.Module):
    def __init__(self, d=192, layers=4, heads=4, L=320, de=128, drop=0.1, nf=NFEAT, mine_col=18, split=False):
        super().__init__()
        self.mc = mine_col; self.split = split
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        self.mv = nn.Embedding(363, d, padding_idx=362)
        self.pos = nn.Embedding(L, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.emb = nn.Sequential(nn.Linear(4 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, de), nn.GELU())
        self.out = nn.Linear(de, 10)
        if split:
            def head(): return nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, de), nn.GELU())
            self.emb_o, self.emb_p = head(), head(); self.out_o, self.out_p = nn.Linear(de, 10), nn.Linear(de, 10)

    def forward(self, x, mv, length):
        B, L, _ = x.shape
        pad = torch.arange(L, device=x.device)[None] >= length[:, None]
        h = self.inp(x) + self.mv(mv) + self.pos(torch.arange(L, device=x.device))[None]
        h = self.norm(self.enc(h, src_key_padding_mask=pad))
        valid = (~pad).float()[..., None]
        mc = self.mc; mine = x[..., mc:mc + 1] * valid; opp = (1 - x[..., mc:mc + 1]) * valid
        pm = (h * mine).sum(1) / mine.sum(1).clamp(min=1); po = (h * opp).sum(1) / opp.sum(1).clamp(min=1)
        hm = h.masked_fill(pad[..., None], -1e4)
        mx_m = hm.masked_fill(mine == 0, -1e4).max(1).values; mx_o = hm.masked_fill(opp == 0, -1e4).max(1).values
        if self.split:
            eo = self.emb_o(torch.cat([pm, mx_m], 1)); ep = self.emb_p(torch.cat([po, mx_o], 1))
            zo, zp = self.out_o(eo), self.out_p(ep)
            return zo + zp, torch.cat([eo, ep], 1), zo, zp
        e = self.emb(torch.cat([pm, po, mx_m, mx_o], 1))
        return self.out(e), e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks"); ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--L", type=int, default=320)
    ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--drop", default=None, help="comma list of feature columns to drop (ablation; never 18)")
    ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--split", action="store_true", help="separate own-side / opponent-side heads"); ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--player-frac", type=float, default=1.0); ap.add_argument("--games-per-player", type=int, default=0)
    ap.add_argument("--qtrain", action="store_true", help="loss on sets of games of one player (needs --split)")
    ap.add_argument("--qmin", type=int, default=3); ap.add_argument("--qmax", type=int, default=8); ap.add_argument("--qw", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--load", default=None, help="model.pt: predict every game of the dataset, no training")
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--smooth", type=float, default=0.0)
    a = ap.parse_args()
    ck = None
    if a.load:
        ck = torch.load(a.load, map_location="cpu", weights_only=False)
        for k_ in ("d", "layers", "dropout", "split", "drop"): setattr(a, k_, ck["args"].get(k_, getattr(a, k_)))
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")
    t0 = time.time()
    if a.chunks.endswith(".npz"):
        z = np.load(a.chunks, allow_pickle=True)
        X, M, LEN = z["X"], z["M"], z["LEN"]; metas = [json.loads(m) for m in z["meta"]]
        a.L = X.shape[1]
        if a.drop:
            keepf = [i for i in range(X.shape[2]) if i not in set(int(v) for v in a.drop.split(","))]
            X = X[:, :, keepf]
    else:
        X, M, LEN, metas = load(a.chunks, a.L, a.limit)
    NF = X.shape[2]
    mine_col = 18
    if a.chunks.endswith(".npz"):
        base_mc = int(z["mine_col"]) if "mine_col" in z.files else 18
        mine_col = keepf.index(base_mc) if a.drop else base_mc
    sp = np.array([m["split"] for m in metas]); y = np.array([m.get("rank_idx", -1) for m in metas])
    tr = np.where(sp == "train")[0]
    if ck is not None and len(tr) == 0: tr = np.arange(len(sp))
    if a.player_frac < 1 or a.games_per_player:
        import hashlib
        cntp = {}; keep_tr = []
        for i in tr:
            pid = metas[i]["player_id"]
            if int(hashlib.md5(pid.encode()).hexdigest()[:8], 16) / 16**8 >= a.player_frac: continue
            c = cntp.get(pid, 0); cntp[pid] = c + 1
            if a.games_per_player and c >= a.games_per_player: continue
            keep_tr.append(i)
        tr = np.array(keep_tr); print("train subset: players", len(cntp), "games", len(tr), flush=True)
    mu = X[tr[:20000]].astype(np.float32).reshape(-1, NF).mean(0); sd = X[tr[:20000]].astype(np.float32).reshape(-1, NF).std(0) + 1e-3
    keepraw = [mine_col]  # the target/opponent flag stays 0/1 (used for pooling)
    mu[keepraw] = 0; sd[keepraw] = 1
    if ck is not None: mu, sd = ck["mu"], ck["sd"]
    print("loaded", X.shape, "train/dev/test", (sp == "train").sum(), (sp == "dev").sum(), (sp == "test").sum(), round(time.time() - t0, 1), flush=True)
    Xt = torch.tensor(X); Mt = torch.tensor(M.astype(np.int64)); Lt = torch.tensor(LEN); yt = torch.tensor(y)
    mu_t = torch.tensor(mu, device=dev); sd_t = torch.tensor(sd, device=dev)
    tabs = d4_tables().to(dev)
    net = GameNet(a.d, a.layers, 4, a.L, nf=NF, drop=a.dropout, mine_col=mine_col, split=a.split).to(dev)
    steps = max(1, a.epochs * (len(tr) // a.bs))
    qscale = nn.Parameter(torch.zeros(2, device=dev))
    opt = torch.optim.AdamW([dict(params=net.parameters(), weight_decay=a.wd), dict(params=[qscale], weight_decay=0.0)], lr=a.lr)
    byp = {}
    for i in tr: byp.setdefault(metas[i]["player_id"], []).append(int(i))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)

    def batch(idx, aug):
        x = (Xt[idx].to(dev).float() - mu_t) / sd_t
        pad = torch.arange(a.L, device=dev)[None] >= Lt[idx].to(dev)[:, None]
        x = x.masked_fill(pad[..., None], 0.0)
        mv = Mt[idx].to(dev)
        if aug:
            mv = tabs[torch.randint(0, 8, (len(idx),), device=dev)[:, None], mv]
        return x, mv, Lt[idx].to(dev)

    def predict(idx):
        net.eval(); lp, em, zo, zp = [], [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx), 512):
                j = torch.tensor(idx[i:i + 512])
                o = [net(*batch(j, False)) for _ in range(1)]
                lp.append(torch.log_softmax(o[0][0].float(), 1).cpu().numpy()); em.append(o[0][1].float().cpu().numpy())
                if a.split:
                    zo.append(torch.log_softmax(o[0][2].float(), 1).cpu().numpy()); zp.append(torch.log_softmax(o[0][3].float(), 1).cpu().numpy())
        net.train()
        extra = dict(zown=np.concatenate(zo), zopp=np.concatenate(zp)) if a.split else {}
        return np.concatenate(lp), np.concatenate(em), extra

    if ck is not None:
        net.load_state_dict(ck["model"]); idx = np.arange(len(sp)); lp, em, extra = predict(idx)
        np.savez(f"{a.out}/all_pred.npz", logp=lp, **extra, meta=np.array([json.dumps(metas[i]) for i in idx])); print("saved", a.out, len(idx), flush=True); return
    dv = np.where(sp == "dev")[0]
    step = 0
    for ep in range(a.epochs):
        perm = np.random.RandomState(1000 * a.seed + ep).permutation(tr)
        tot = n = 0
        qrng = np.random.RandomState(1000 + ep)
        for i in range(0, len(perm) - a.bs + 1, a.bs):
            if a.qtrain:
                g = int(qrng.randint(a.qmin, a.qmax + 1)); npl = max(2, a.bs // g)
                ok = [p_ for p_ in byp if len(byp[p_]) >= g]
                ps = qrng.choice(len(ok), npl, replace=False)
                j = torch.tensor(np.concatenate([qrng.choice(byp[ok[pi]], g, replace=False) for pi in ps]))
            else:
                j = torch.tensor(perm[i:i + a.bs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(*batch(j, True)); logit = o[0]
            tgt = yt[j].to(dev)
            loss = Fn.cross_entropy(logit.float(), tgt, label_smoothing=a.smooth)
            if a.split and a.aux > 0:
                loss = loss + a.aux * (Fn.cross_entropy(o[2].float(), tgt) + Fn.cross_entropy(o[3].float(), tgt))
            if a.qtrain:
                zo = torch.log_softmax(o[2].float(), 1).view(npl, g, 10).sum(1); zp = torch.log_softmax(o[3].float(), 1).view(npl, g, 10).sum(1)
                ql = (qscale[0].exp() * zo + qscale[1].exp() * zp) / g ** 0.5
                loss = loss + a.qw * Fn.cross_entropy(ql, tgt.view(npl, g)[:, 0])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            tot += loss.item(); n += 1
        lp, _, _ = predict(dv)
        acc = (lp.argmax(1) == y[dv]).mean(); w1 = (np.abs(lp.argmax(1) - y[dv]) <= 1).mean()
        nll = -lp[np.arange(len(dv)), y[dv]].mean()
        if a.qtrain: print("  qscale", qscale.detach().exp().cpu().numpy().round(3).tolist(), flush=True)
        print(f"epoch {ep} train_loss {tot / n:.4f} dev_nll {nll:.4f} dev_game_acc {acc:.4f} within1 {w1:.4f} t {time.time() - t0:.0f}s", flush=True)
    torch.save(dict(model=net.state_dict(), mu=mu, sd=sd, args=vars(a), mine_col=mine_col), a.out + "/model.pt")
    for name in ["dev", "test", "train"]:
        idx = np.where(sp == name)[0]
        if len(idx) == 0: continue
        lp, em, extra = predict(idx)
        np.savez(f"{a.out}/{name}_pred.npz", logp=lp, emb=em.astype(np.float16), **extra, meta=np.array([json.dumps(metas[i]) for i in idx]))
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
