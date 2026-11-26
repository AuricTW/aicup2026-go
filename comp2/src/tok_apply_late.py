"""Apply frozen competition-1 LATE token nets (trainer v5, tokhead_vm_20261118_v5.py: 90 early decisions of plies 1-180 + up to 60
decisions of plies 181-300, move coordinates, 31 ply buckets) to the opponent-side tokens of comp2 question games.
Model code = a copy of TokNet of the v5 trainer; eval mode, training BN statistics.  Early move codes from the payload SGF (as
tok_apply.py), late move codes from the late dir's tok_act (act2code of the trainer); both are checked against each other / the
pass flag.  Output as tok_apply.py: top-K cosine logits (s = 30) and the comp2 account name of every class.
usage: tok_apply_late.py MODEL.pt CLASSMAP.npz OUT.npz "RESULTS_DIR|GAMES.jsonl|KEYS.json|LATE_DIR" [...] [--k 1024] [--emb] [--no-late]
       [--ext] (net with external background classes)  [--device cpu]  [--like FINISHED_MODEL.pt] (MODEL.pt is a ckpt.pt)
"""
import sys, re, json, numpy as np, torch
from torch import nn
import torch.nn.functional as F
MV = re.compile(r';([BW])\[([a-s]{0,2})\]'); PAD, PASS = 362, 361


def mv_codes(sgf, plies, n, width):
    t = np.array([(ord(c[0]) - 97) * 19 + (ord(c[1]) - 97) if len(c) == 2 else PASS for _, c in MV.findall(sgf)], np.int64)
    out = np.full(width, PAD, np.int64); k = int(n); pl = np.asarray(plies[:k]).astype(np.int64); ok = (pl >= 1) & (pl <= len(t)); out[:k][ok] = t[pl[ok] - 1]; return out


def act2code(a):
    a = np.asarray(a).astype(np.int64); return np.where(a >= 361, a, (a % 19) * 19 + a // 19)


class TokNet(nn.Module):
    def __init__(self, ncls, d, layers, drop, s=30., m=.2, de=512):
        super().__init__(); self.s, self.m = s, m; self.mv = nn.Embedding(363, d, padding_idx=362); self.inp = nn.Sequential(nn.Dropout(.3), nn.Linear(1162, d), nn.GELU(), nn.Dropout(drop)); self.ply = nn.Embedding(31, d); self.col = nn.Embedding(2, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True, activation='gelu'), layers, enable_nested_tensor=False) if layers else None
        self.att = nn.Linear(d, 1); self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, de), nn.BatchNorm1d(de)); self.W = nn.Parameter(torch.randn(ncls, de) * .01)

    def embed(self, x, ply, valid, black, mv):
        pad = ~valid; hh = self.inp(x) + self.ply((ply.clamp(1, 300) - 1) // 10) + self.col(black.long())[:, None] + self.mv(mv.long())
        if self.enc is not None: hh = self.enc(hh, src_key_padding_mask=pad)
        m = (~pad).float()[..., None]; mean = (hh * m).sum(1) / m.sum(1); a = torch.softmax(self.att(hh).masked_fill(pad[..., None], -1e4), 1); return self.out(torch.cat([mean, (hh * a).sum(1)], 1))


def main():
    args = sys.argv[1:]; K = 1024; EMB = "--emb" in args; NOLATE = "--no-late" in args; EXT = "--ext" in args; dev = "cuda"; LIKE = None
    for f in ("--emb", "--no-late", "--ext"):
        if f in args: args.remove(f)
    if "--k" in args: K = int(args[args.index("--k") + 1]); del args[args.index("--k"):args.index("--k") + 2]
    if "--device" in args: dev = args[args.index("--device") + 1]; del args[args.index("--device"):args.index("--device") + 2]
    # --like MODEL.pt: MODEL.pt is a training CHECKPOINT (ckpt.pt: no args / classes); take both from this finished model of the
    # same recipe, plus (with --ext) the external classes first_external_y .. (tooling tests on a net that is still training)
    if "--like" in args: LIKE = args[args.index("--like") + 1]; del args[args.index("--like"):args.index("--like") + 2]
    ck = torch.load(args[0], map_location="cpu", weights_only=False); cm = np.load(args[1], allow_pickle=True)["names"]
    if LIKE:
        lk = torch.load(LIKE, map_location="cpu", weights_only=False); ck["args"] = lk["args"]; nW = ck["model"]["W"].shape[0]; c0 = np.asarray(lk["classes"])
        ck["classes"] = np.concatenate([c0, len(cm) + np.arange(nW - len(c0))]); print(f"checkpoint at step {ck.get('step')}: classes = {len(c0)} of {LIKE} + {nW - len(c0)} external", flush=True); del lk
    a = ck["args"]; cls = np.asarray(ck["classes"]).astype(np.int64)
    # --ext: a comp1 net trained with EXTERNAL accounts as extra classes (y >= len(class map); comp1 extfox_merge: 'first_external_y').
    # They have no comp2 account and NO label here: name '' = open-set background for tok_evidence.py (--bg).  Nothing of the
    # external accounts (name, rank, games) is read by comp2; no competition game is matched against external data.
    comp = cls < len(cm); names = np.where(comp, cm[np.minimum(cls, len(cm) - 1)], ""); assert (names[comp] != "").all(), "unmapped class"; assert comp.all() or EXT, "classes beyond the class map: pass --ext"
    assert len(cls) < 32768, "topi is int16"; assert a.get("late_dirs") and a.get("moves"), "not a v5 (late) model"; KL = int(a.get("late_k", 60))
    assert ck["model"]["ply.weight"].shape[0] == 31 and "o1.weight" not in ck["model"]
    if EXT: print(f"classes {len(cls)}: {int(comp.sum())} competition accounts + {int((~comp).sum())} external background classes", flush=True)
    net = TokNet(len(cls), a["d"], a["layers"], a["drop"]).to(dev); net.load_state_dict(ck["model"]); net.eval(); mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
    Wn = F.normalize(net.W.detach().float(), dim=1); allk, topv, topi, embs, nlate = [], [], [], [], []
    for spec in args[3:]:
        rd, gj, kj, ld = spec.split("|"); keys = json.load(open(kj)); G = [json.loads(l) for l in open(gj, encoding="utf-8")]; col = [g["color"] for g in G]
        idx = np.load(rd + "/indices.npy").astype(np.int64); assert len(idx) == len(keys) == len(col)
        Z = np.load(rd + "/tok_z.npy", mmap_mode="r"); A = np.load(rd + "/tok_a.npy", mmap_mode="r"); P = np.load(rd + "/tok_ply.npy", mmap_mode="r"); N = np.load(rd + "/tok_n.npy")
        li = {int(g["index"]): j for j, g in enumerate(G)}; assert all(li[int(idx[i])] == i for i in range(0, len(idx), 997)), "results rows are not in games.jsonl order"
        bl = torch.tensor([c == "B" for c in col])
        MVE = np.stack([mv_codes(G[i]["sgf"], P[i], N[i], 90) for i in range(len(idx))]); Aall = np.asarray(A); msk = np.arange(90)[None] < N[:, None]
        pf = float(((Aall[..., 9] > .5) == (MVE == PASS))[msk].mean()); del Aall
        LZ = np.load(ld + "/tok_z.npy", mmap_mode="r"); LA = np.load(ld + "/tok_a.npy", mmap_mode="r"); LP = np.load(ld + "/tok_ply.npy", mmap_mode="r"); LN = np.load(ld + "/tok_n.npy").astype(np.int64); LACT = np.load(ld + "/tok_act.npy"); lidx = np.load(ld + "/indices.npy").astype(np.int64)
        assert np.array_equal(lidx, idx), "late dir not aligned with the token dir"; assert LZ.shape[1] == KL
        MVL = act2code(LACT); lm = np.arange(KL)[None] < LN[:, None]
        chk = np.stack([mv_codes(G[i]["sgf"], LP[i], LN[i], KL) for i in range(len(idx))]); agree = float((chk == MVL)[lm].mean()) if lm.any() else 1.0
        plok = float(((np.asarray(LP)[lm] >= 181) & (np.asarray(LP)[lm] <= 300)).mean())
        print(f"{rd}: early pass-flag agreement {pf:.4f}; late move codes equal the SGF move at the token's ply {agree:.4f}; late plies in 181-300 {plok:.4f}; games with late decisions {np.mean(LN > 0):.3f}, mean {LN.mean():.1f}", flush=True)
        assert pf > 0.999 and agree > 0.999 and plok == 1.0
        if NOLATE: LN = np.zeros_like(LN)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            for i in range(0, len(keys), 384):
                s = slice(i, i + 384); z = torch.tensor(np.concatenate([np.asarray(Z[s]), np.asarray(LZ[s])], 1)).to(dev); aa = torch.tensor(np.concatenate([np.asarray(A[s]), np.asarray(LA[s])], 1)).to(dev)
                x = ((torch.cat([z.float(), aa.float()], 2) - mu) / sd).clamp(-8, 8)
                valid = np.concatenate([np.arange(90)[None] < N[s].astype(np.int64)[:, None], np.arange(KL)[None] < LN[s][:, None]], 1)
                mv = np.where(valid, np.concatenate([MVE[s], MVL[s]], 1), PAD); ply = np.concatenate([np.asarray(P[s]).astype(np.int64), np.asarray(LP[s]).astype(np.int64)], 1)
                e = net.embed(x, torch.tensor(ply).to(dev), torch.tensor(valid).to(dev), bl[s].to(dev), torch.tensor(mv).to(dev))
                en = F.normalize(e.float(), dim=1); lo = net.s * (en @ Wn.T); v, j = lo.topk(K, 1); topv.append(v.float().cpu().numpy().astype(np.float16)); topi.append(j.cpu().numpy().astype(np.int16))
                if EMB: embs.append(en.cpu().numpy().astype(np.float16))
        allk += keys; nlate.append(LN); print(rd, len(keys), flush=True)
    extra = {"emb": np.concatenate(embs)} if EMB else {}
    np.savez(args[2], keys=np.array(allk), topv=np.concatenate(topv), topi=np.concatenate(topi), names=names.astype(str), n_late=np.concatenate(nlate), **extra)
    tv = np.concatenate(topv).astype(np.float32); p1 = 1 / np.exp(tv - tv[:, :1]).sum(1); nl = np.concatenate(nlate)
    print("saved", args[2], "games", len(allk), " top-1 prob (s=30) mean", p1.mean().round(3), " > 0.5:", np.mean(p1 > 0.5).round(3), " | with late decisions:", np.mean(p1[nl > 0] > 0.5).round(3), " without:", np.mean(p1[nl == 0] > 0.5).round(3))


if __name__ == "__main__":
    main()
