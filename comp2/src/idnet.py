"""Identity network: which labelled training account plays ONE SIDE of a game?  (moves only, raw orientation, no D4 augment)
Why: opponents in a player's games are often themselves training accounts (probe_opponent_identity.py).  Recognising the
opponent gives his known rank, and matchmaking ties it to the target's rank.
Training: owner side of all train-split games (HD<=1) -> softmax over train-split owners; a few games per owner are held out
to measure single-game identification.  Dev/test players are never classes (dev simulates the official test faithfully).
Outputs (OUT_DIR/idnet_pred.npz):
  dev_* / test_*: per game, for the OPPONENT side and the target's own side: rank mass (sum of account probabilities per
                  rank) [N,10], top-1 probability, top-1 account's rank;  keys, hd, target colour
  ho_*:           held-out owner-side games: top-1 correct, rank mass, owner rank
  tro_*:          opponent side of a sample of training games: rank mass, top-1 prob/rank, owner rank, hd, owner colour
usage: idnet.py MOVES.npz OUT_DIR [--epochs 8] [--L 200] [--d 256] [--layers 6] [--bs 512] [--lr 1e-3] [--train-n N] [--holdout 3]
"""
import sys, os, json, argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn


class Net(nn.Module):
    def __init__(self, ncls, d=256, layers=6, heads=4, L=200, de=384, drop=0.1, cos=0.0, rel=0, board=0, bl=4, bpatch=0, solo=False, trunk=0):
        super().__init__(); self.cos = cos                      # cos > 0: cosine classifier with this scale (open-set friendly embedding)
        self.solo = solo                                        # solo: the Transformer sees only the classified side's moves (the other side only as stones in the board patches)
        # rel = K: explicit board offset (dx, dy) of every move to each of the K previous moves (local answer / tenuki / shape habits)
        self.rel = nn.ModuleList([nn.Embedding(37 * 37 + 1, d) for _ in range(rel)])
        # board = C: small CNN over the stones on the board before every move (captures ignored), read out at the point played:
        # the local shape a move makes (the coordinate-only net cannot see it; moves 80-200 added nothing without it)
        self.board = board; self.bpatch = bpatch                 # bpatch = K: the CNN sees only the K x K neighbourhood of the move (plus an off-board plane)
        if board:
            layers_ = []; cin = 4 if bpatch else 3
            for _ in range(bl): layers_ += [nn.Conv2d(cin, board, 3, padding=1), nn.ReLU(inplace=True)]; cin = board
            self.cnn = nn.Sequential(*layers_); self.bproj = nn.Sequential(nn.Linear(2 * board, d), nn.GELU(), nn.Linear(d, d))
        # trunk = D: per-move embedding of a strong human-imitation net (HumanSL trunk, PCA D dims, extract_trunk*.py) added to the token
        self.tproj = nn.Sequential(nn.Linear(trunk, d), nn.GELU(), nn.Linear(d, d)) if trunk else None
        self.mv = nn.Embedding(363, d, padding_idx=362); self.pos = nn.Embedding(L, d); self.who = nn.Embedding(2, d); self.col = nn.Embedding(2, d); self.hd = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        self.emb = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, de)); self.cls = nn.Linear(de, ncls)

    def forward(self, mv, length, side_black, first_black, hd, tk=None):
        B, L = mv.shape; ar = torch.arange(L, device=mv.device); pad = ar[None] >= length[:, None]
        mover_black = (ar[None] % 2 == 0) == first_black[:, None]; mine = mover_black == side_black[:, None]
        h = self.mv(mv) + self.pos(ar)[None] + self.who(mine.long()) + self.col(mover_black.long()) + self.hd(hd.clamp(max=1))[:, None]
        if self.tproj is not None: h = h + self.tproj(tk.to(h.dtype))
        if len(self.rel):
            ok = mv < 361; x = mv % 19; yy = mv // 19
            for k, emb in enumerate(self.rel, 1):
                pk = torch.roll(ok, k, 1); pk[:, :k] = False; idx = (x - torch.roll(x, k, 1) + 18) * 37 + (yy - torch.roll(yy, k, 1) + 18)
                h = h + emb(torch.where(ok & pk, idx, torch.full_like(idx, 37 * 37)))
        if self.board:
            oh = torch.zeros(B, L, 362, device=mv.device, dtype=h.dtype).scatter_(2, mv.clamp(max=361)[..., None], 1.0)[..., :361]      # the move itself (pass / padding: empty)
            mb = mover_black[..., None].to(h.dtype); bk = torch.cumsum(oh * mb, 1) - oh * mb; wt = torch.cumsum(oh * (1 - mb), 1) - oh * (1 - mb)   # stones before the move
            own = torch.where(mover_black[..., None], bk, wt); opp = torch.where(mover_black[..., None], wt, bk)
            sel = (mine & ~pad).view(-1)                         # only the moves of the classified side go through the CNN (memory, speed)
            pl = torch.stack([own, opp, oh], 2).view(B * L, 3, 19, 19)[sel].clamp(max=1)
            if self.bpatch:
                K = self.bpatch; r = K // 2; pp = torch.cat([Fn.pad(pl, (r, r, r, r)), Fn.pad(torch.zeros_like(pl[:, :1]), (r, r, r, r), value=1.0)], 1)
                ms = mv.view(-1)[sel].clamp(max=360); ak = torch.arange(K, device=mv.device); n_ = torch.arange(len(ms), device=mv.device)
                pt = pp[n_[:, None, None], :, (ms // 19)[:, None, None] + ak[None, :, None], (ms % 19)[:, None, None] + ak[None, None, :]].permute(0, 3, 1, 2)
                f = self.cnn(pt.contiguous(memory_format=torch.channels_last)); loc = f[:, :, r, r]
            else:
                f = self.cnn(pl.contiguous(memory_format=torch.channels_last)); loc = (f * pl[:, 2:3]).sum((2, 3))
            bf = self.bproj(torch.cat([loc, f.mean((2, 3))], 1)); add = torch.zeros(B * L, bf.shape[1], device=h.device, dtype=bf.dtype); add[sel] = bf
            h = h + add.view(B, L, -1).to(h.dtype)
        pm = pad
        if self.solo: pm = pad | ~mine; pm[pm.all(1), 0] = False        # (a row without any move of the side keeps one token: no NaN)
        h = self.norm(self.enc(h, src_key_padding_mask=pm))
        m = (mine & ~pad).float()[..., None]
        e = self.emb(torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), h.masked_fill(m == 0, -1e4).max(1).values], 1))
        if self.cos > 0:                                        # logits = scale * cosine(embedding, class weight); returns the unit embedding
            with torch.autocast("cuda", enabled=False):
                ef = Fn.normalize(e.float(), dim=1); return self.cos * (ef @ Fn.normalize(self.cls.weight.float(), dim=1).t()), ef
        return self.cls(e), e


def load_trunk(dirs, keys, L, scale, load=True):
    """slot [N] (-1 = no embedding) and TK [n, L, D] float16 for the dataset rows; test keys are stored as qid#slot in the chunks, qid|slot in the dataset"""
    import glob
    pos = {str(k): i for i, k in enumerate(keys)}; slot = -np.ones(len(keys), np.int64); files = [f for f in sorted(sum([glob.glob(d + "/tchunk_*.npz") for d in dirs.split(",")], [])) if os.path.exists(f[:-4] + ".json")]; metas = []; n = 0      # the .json is written after the .npz is complete
    for f in files:
        ms = json.load(open(f[:-4] + ".json")); rows = []
        for m in ms:
            k = m["key"]; i = pos.get(k, pos.get(k.replace("#", "|")) if not k.startswith("train_") else None)
            if i is not None and slot[i] < 0: slot[i] = n; n += 1; rows.append(i)
            else: rows.append(-1)
        metas.append(rows)
    if not load: return slot, torch.zeros(1, 1, 1)
    D = np.load(files[0])["emb"].shape[1]; TK = torch.zeros(n, L, D, dtype=torch.float16)
    for f, rows in zip(files, metas):
        t = np.load(f); off = t["offset"]; emb = t["emb"]
        for j, i in enumerate(rows):
            if i >= 0: e = emb[off[j]:off[j + 1]][:L]; TK[slot[i], :len(e)] = torch.from_numpy((e.astype(np.float32) * scale).astype(np.float16))
    return slot, TK


def load_net(run_dir, dev):
    """model of a finished run (softmax or cosine classifier) in eval mode + its checkpoint dict"""
    ck = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=False); ar = ck["args"]
    net = Net(len(ck["owners"]), ar["d"], ar["layers"], 4, ar["L"], drop=ar["dropout"], cos=ar.get("cos", 0.0), rel=ar.get("rel", 0), board=ar.get("board", 0), bl=ar.get("board_layers", 4), bpatch=ar.get("board_patch", 0), trunk=64 if ar.get("trunk") and not ar.get("trunk_mask_only") else 0).to(dev); net.load_state_dict(ck["model"]); net.eval()
    return net, ck


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--epochs", type=float, default=8); ap.add_argument("--L", type=int, default=200); ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--train-n", type=int, default=0); ap.add_argument("--holdout", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--smooth", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--tro", type=int, default=40000); ap.add_argument("--holdout-players", default=None)
    # self-training: opponent sides of training games recognised by a teacher (idnet_pseudo.py) become extra samples of that account
    ap.add_argument("--pseudo", default=None); ap.add_argument("--pseudo-thr", type=float, default=0.5); ap.add_argument("--init", default=None)
    # final gallery: also train on the dev-split accounts (split codes 1 = dev block, 3 = their other games, build_moves_extra.py)
    ap.add_argument("--extra-split", default="")
    # cosine classifier (scale --cos, additive margin --margin on the true class): embedding usable for accounts that are not classes
    ap.add_argument("--cos", type=float, default=0.0); ap.add_argument("--margin", type=float, default=0.0)
    # question classes (build_moves_fs.py, split code 5 via --extra-split 5): no hold-out for accounts with fewer than --holdout-min games,
    # samples of split 5 repeated --fs-rep times per epoch
    ap.add_argument("--holdout-min", type=int, default=0); ap.add_argument("--fs-rep", type=int, default=1)
    ap.add_argument("--rel", type=int, default=0)      # board offsets to the K previous moves as extra token features
    ap.add_argument("--board", type=int, default=0); ap.add_argument("--board-layers", type=int, default=4)      # CNN over the stones before each move (channels, layers)
    ap.add_argument("--board-patch", type=int, default=0)                                                          # K: only the K x K neighbourhood of the move
    # --trunk DIR[,DIR2]: trunk embeddings per ply; only games that have them are used (training, hold-out, dev, x, test).
    # --trunk-mask-only: same game selection but the embeddings are NOT fed (control run on identical data).  --max-per N: at most N training games per account.
    ap.add_argument("--trunk", default=None); ap.add_argument("--trunk-scale", type=float, default=0.5); ap.add_argument("--trunk-mask-only", action="store_true"); ap.add_argument("--max-per", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); dev = torch.device("cuda"); t0 = time.time()
    z = np.load(a.data, allow_pickle=True)
    M = torch.tensor(z["M"][:, :a.L].astype(np.int64)); LEN = torch.tensor(np.minimum(z["LEN"].astype(np.int64), a.L)); CB = torch.tensor(z["color_black"].astype(bool))
    HDV = torch.tensor(z["hd"].astype(np.int64)); y = z["y"].astype(np.int64); sp = z["split"]; pid = z["pid"].astype(np.int64); keys = z["key"]
    # the first mover is black unless stated otherwise by the dataset (handicap games in this data still start with B)
    FB = torch.tensor(z["first_black"].astype(bool)) if "first_black" in z.files else torch.ones(len(y), dtype=torch.bool)
    tr_all = np.where(np.isin(sp, [0] + [int(s) for s in a.extra_split.split(",") if s]))[0]; dv = np.where(sp == 1)[0]; te = np.where(sp == 2)[0]
    slot = None; TK = None
    if a.trunk:
        slot, TK = load_trunk(a.trunk, keys, a.L, a.trunk_scale, not a.trunk_mask_only); hv = slot >= 0
        print(f"trunk embeddings: games {int(hv.sum())}  train-able {int(hv[tr_all].sum())} of {len(tr_all)}  dev {int(hv[dv].sum())} of {len(dv)}  test {int(hv[te].sum())} of {len(te)}  load {time.time() - t0:.0f}s", flush=True)
        tr_all = tr_all[hv[tr_all]]; dv = dv[hv[dv]]; te = te[hv[te]]
    rng = np.random.RandomState(a.seed)
    xs = np.zeros(0, np.int64)
    if a.holdout_players:                                    # players removed from the gallery; their games are predicted as block 'x'
        import csv
        rows = list(csv.DictReader(open(a.holdout_players, newline=""))); names = {str(n): i for i, n in enumerate(z["pid_names"])}
        hp = np.array(sorted({names[r["player_id"]] for r in rows if r["player_id"] in names})); kset = {r["key"] for r in rows}
        xs = np.array([i for i in np.where((sp == 0) | (sp == 3))[0] if str(keys[i]) in kset and (slot is None or slot[i] >= 0)]); tr_all = tr_all[~np.isin(pid[tr_all], hp)]
        print("held-out players", len(hp), "their games to predict", len(xs), flush=True)
    if a.train_n: tr_all = np.sort(rng.choice(tr_all, a.train_n, replace=False))
    if a.max_per:
        o_ = rng.permutation(tr_all); c_ = {}; keep = []
        for i in o_:
            if c_.get(pid[i], 0) < a.max_per: keep.append(i); c_[pid[i]] = c_.get(pid[i], 0) + 1
        tr_all = np.sort(np.array(keep))
    owners = np.unique(pid[tr_all]); cls_of = -np.ones(pid.max() + 1, np.int64); cls_of[owners] = np.arange(len(owners)); C = len(owners)
    cls_rank = np.zeros(C, np.int64); cls_rank[cls_of[pid[tr_all]]] = y[tr_all]
    R1 = torch.tensor(np.eye(10, dtype=np.float32)[cls_rank], device=dev)                      # [C,10]
    order = rng.permutation(tr_all); seen = {}; ho = []; ngames = np.bincount(pid[tr_all], minlength=pid.max() + 1)
    for i in order:
        c = seen.get(pid[i], 0)
        if c < a.holdout and ngames[pid[i]] >= a.holdout_min: ho.append(i); seen[pid[i]] = c + 1
    ho = np.array(sorted(ho)); hoset = set(ho.tolist()); tr = np.array([i for i in tr_all if i not in hoset])
    lab = torch.tensor(cls_of[pid])
    print(f"owners (classes) {C}  train games {len(tr)}  held-out {len(ho)}  dev {len(dv)}  test {len(te)}  load {time.time() - t0:.0f}s", flush=True)
    use_tk = TK is not None and not a.trunk_mask_only; tkof = (lambda j: TK[torch.as_tensor(slot[np.asarray(j)])].to(dev).float()) if use_tk else (lambda j: None)
    net = Net(C, a.d, a.layers, 4, a.L, drop=a.dropout, cos=a.cos, rel=a.rel, board=a.board, bl=a.board_layers, bpatch=a.board_patch, trunk=TK.shape[2] if use_tk else 0).to(dev)
    if a.board: net = net.to(memory_format=torch.channels_last)
    TI = tr.copy(); TF = np.zeros(len(tr), bool); TL = cls_of[pid[tr]]                          # training samples: game, flip to the opponent side?, class
    if a.fs_rep > 1:
        fs = tr[sp[tr] == 5]; TI = np.concatenate([TI] + [fs] * (a.fs_rep - 1)); TF = np.zeros(len(TI), bool); TL = cls_of[pid[TI]]; print(f"question-class games {len(fs)} repeated x{a.fs_rep}", flush=True)
    if a.pseudo:
        ps = np.load(a.pseudo); trset = np.zeros(len(y), bool); trset[tr] = True
        pc = cls_of[ps["pid"]]; ok = (ps["prob"] > a.pseudo_thr) & (pc >= 0) & trset[ps["idx"]] & (pc != cls_of[pid[ps["idx"]]])
        TI = np.concatenate([TI, ps["idx"][ok]]); TF = np.concatenate([TF, np.ones(int(ok.sum()), bool)]); TL = np.concatenate([TL, pc[ok]])
        print(f"pseudo-labelled opponent sides added: {int(ok.sum())} (threshold {a.pseudo_thr})  total samples {len(TI)}", flush=True)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False); sd = ck["model"]
        if not np.array_equal(ck["owners"], owners):            # different class set: copy the trunk and the rows of the classes both have
            mine = net.state_dict(); pos = {int(o): i for i, o in enumerate(ck["owners"])}; src = np.array([pos.get(int(o), -1) for o in owners]); has = torch.tensor(src >= 0)
            for k in ("cls.weight", "cls.bias"): new = mine[k].detach().cpu().clone(); new[has] = sd[k][torch.tensor(src[src >= 0])]; sd[k] = new
            sd["cls.bias"][~has] = sd["cls.bias"][has].mean(); print(f"class sets differ: {int(has.sum())} of {C} class rows copied", flush=True)
        net.load_state_dict(sd); print("initialised from", a.init, flush=True)
    TF_t = torch.tensor(TF); TL_t = torch.tensor(TL); tr = np.arange(len(TI))                  # from here on 'tr' indexes the sample arrays
    spe = len(tr) // a.bs; steps = int(a.epochs * spe)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)

    def run(idx, opp):
        """returns rank mass [n,10], top-1 prob, top-1 class for the owner side (opp=False) or the other side (opp=True)"""
        net.eval(); out = [[], [], []]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(idx), 1024):
                j = torch.tensor(idx[i:i + 1024]); sb = CB[j].to(dev)
                lo, _ = net(M[j].to(dev), LEN[j].to(dev), ~sb if opp else sb, FB[j].to(dev), HDV[j].to(dev), tkof(idx[i:i + 1024]))
                with torch.autocast("cuda", enabled=False):
                    p = torch.softmax(lo.float(), 1); tp, tc = p.max(1); mass = p @ R1
                out[0].append(mass.cpu().numpy()); out[1].append(tp.cpu().numpy()); out[2].append(tc.cpu().numpy())
        net.train(); return [np.concatenate(o) for o in out]

    step = 0; tot = n = 0; perm = rng.permutation(tr); pp = 0
    while step < steps:
        if pp + a.bs > len(perm): perm = rng.permutation(tr); pp = 0
        s = torch.tensor(perm[pp:pp + a.bs]); pp += a.bs; j = torch.tensor(TI[s.numpy()])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, _ = net(M[j].to(dev), LEN[j].to(dev), (CB[j] ^ TF_t[s]).to(dev), FB[j].to(dev), HDV[j].to(dev), tkof(j.numpy()))
        tgt = TL_t[s].to(dev); lo = lo.float()
        if a.cos > 0 and a.margin > 0: lo = lo - a.cos * a.margin * Fn.one_hot(tgt, lo.shape[1])
        loss = Fn.cross_entropy(lo, tgt, label_smoothing=a.smooth)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        step += 1; tot += loss.item(); n += 1
        if step % a.eval_every == 0 or step == steps:
            sub = ho[:: max(1, len(ho) // 6000)]; mass, tp, tc = run(sub, False)
            print(f"step {step}/{steps} ep {step / spe:.2f} train_loss {tot / n:.4f}  held-out owner top-1 {(tc == cls_of[pid[sub]]).mean():.4f}  rank of top-1 correct {(cls_rank[tc] == y[sub]).mean():.4f}"
                  f"  rank-mass argmax correct {(mass.argmax(1) == y[sub]).mean():.4f}  t {time.time() - t0:.0f}s", flush=True)
            tot = n = 0
    torch.save(dict(model=net.state_dict(), args=vars(a), owners=owners, cls_rank=cls_rank), a.out + "/model.pt")
    res = {}
    mass, tp, tc = run(ho, False); res.update(ho_mass=mass, ho_top=tp, ho_correct=(tc == cls_of[pid[ho]]), ho_toprank=cls_rank[tc], ho_rank=y[ho])
    tr0 = TI[~TF]; tro = np.sort(rng.choice(tr0, min(a.tro, len(tr0)), replace=False)); mass, tp, tc = run(tro, True)
    res.update(tro_mass=mass, tro_top=tp, tro_toprank=cls_rank[tc], tro_rank=y[tro], tro_hd=z["hd"][tro], tro_black=z["color_black"][tro], tro_self=(tc == cls_of[pid[tro]]))
    for name, idx in (("dev", dv), ("test", te)) + ((("x", xs),) if len(xs) else ()):
        for side, opp in (("opp", True), ("own", False)):
            mass, tp, tc = run(idx, opp); res.update({f"{name}_{side}_mass": mass, f"{name}_{side}_top": tp, f"{name}_{side}_toprank": cls_rank[tc]})
        res.update({f"{name}_keys": keys[idx], f"{name}_hd": z["hd"][idx], f"{name}_black": z["color_black"][idx], f"{name}_rank": y[idx], f"{name}_pid": pid[idx]})
    np.savez(a.out + "/idnet_pred.npz", **res)
    print(f"held-out: owner top-1 {res['ho_correct'].mean():.4f}  rank via top-1 {(res['ho_toprank'] == res['ho_rank']).mean():.4f}")
    hd0 = res["tro_hd"] == 0
    print(f"training games, OPPONENT side (HD0): top-1 prob mean {res['tro_top'][hd0].mean():.3f}  top-1 account has owner's rank {(res['tro_toprank'] == res['tro_rank'])[hd0].mean():.4f}"
          f"  within 1 {(np.abs(res['tro_toprank'] - res['tro_rank']) <= 1)[hd0].mean():.4f}  predicted as the owner himself {res['tro_self'][hd0].mean():.4f}")
    for thr in (0.3, 0.6, 0.9):
        k = hd0 & (res["tro_top"] > thr)
        if k.sum(): print(f"   top-1 prob > {thr}: share {k.mean() / hd0.mean():.3f}  same rank {(res['tro_toprank'] == res['tro_rank'])[k].mean():.3f}  within 1 {(np.abs(res['tro_toprank'] - res['tro_rank']) <= 1)[k].mean():.3f}")
    d0 = res["dev_hd"] == 0
    print(f"dev games, OPPONENT side (HD0): top-1 prob mean {res['dev_opp_top'][d0].mean():.3f}  top-1 account has the target's rank {(res['dev_opp_toprank'] == res['dev_rank'])[d0].mean():.4f}"
          f"   own side: top-1 prob mean {res['dev_own_top'][d0].mean():.3f}  rank {(res['dev_own_toprank'] == res['dev_rank'])[d0].mean():.4f}")
    print(f"test games, OPPONENT side: top-1 prob mean {res['test_opp_top'].mean():.3f}   own side {res['test_own_top'].mean():.3f}")
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
