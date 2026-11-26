"""Apply frozen competition-1 token nets (per-decision HumanSL tokens -> AM-softmax over the training-file owners; trained only on
labelled training-file games, which are byte-identical in both competitions) to the OPPONENT-side tokens of comp2 question games.
Model code = a copy of TokNet in the comp1 trainer (tokhead_vm_20261117_v2.py), eval mode, training BN statistics (no recalibration).
Output per game: top-K cosine logits (s * cos, s = 30) and the comp2 account NAME of each class (via tok_classmap.npz).
usage: tok_apply.py MODEL.pt CLASSMAP.npz OUT.npz RESULTS_DIR:GAMES.jsonl:KEYS.json [...] [--k 256]
"""
import sys, re, json, numpy as np, torch
from torch import nn
import torch.nn.functional as F

# move coordinates per token (comp1 trainer v3 '--moves' nets, e.g. t14): code = (col letter)*19 + (row letter), 361 pass, 362 padding,
# same convention as comp1 seqid_20261115_v1.tokens / tokmv_20261117_v1.codes (coordinate = move of the full game at the token's ply)
MV = re.compile(r';([BW])\[([a-s]{0,2})\]'); PAD, PASS = 362, 361


def mv_codes(sgf, plies, n, back=0):
    t = np.array([(ord(c[0]) - 97) * 19 + (ord(c[1]) - 97) if len(c) == 2 else PASS for _, c in MV.findall(sgf)], np.int64)
    out = np.full(90, PAD, np.int64); k = int(n); pl = np.asarray(plies[:k]).astype(np.int64) - back; ok = (pl >= 1) & (pl <= len(t)); out[:k][ok] = t[pl[ok] - 1]; return out


class TokNet(nn.Module):
    # offs=True: comp1 trainer v4 (tokhead_vm_20261117_v4.py): + offsets of the move to the opponent's last move (ply-1) and to the
    # own previous move (ply-2), two 1,370-way embeddings; mv input then has 3 channels (mv, prev1, prev2)
    def __init__(self, ncls, d, layers, drop, s=30., m=.2, de=512, moves=False, offs=False):
        super().__init__(); self.s, self.m = s, m; self.mv = nn.Embedding(363, d, padding_idx=362) if moves else None
        self.o1 = nn.Embedding(1370, d) if offs else None; self.o2 = nn.Embedding(1370, d) if offs else None; self.inp = nn.Sequential(nn.Dropout(.3), nn.Linear(1162, d), nn.GELU(), nn.Dropout(drop)); self.ply = nn.Embedding(19, d); self.col = nn.Embedding(2, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True, activation='gelu'), layers, enable_nested_tensor=False) if layers else None
        self.att = nn.Linear(d, 1); self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, de), nn.BatchNorm1d(de)); self.W = nn.Parameter(torch.randn(ncls, de) * .01)

    def embed(self, x, ply, n, black, mv=None):
        pad = torch.arange(x.shape[1], device=x.device)[None] >= n[:, None]; hh = self.inp(x) + self.ply((ply.clamp(1, 180) - 1) // 10) + self.col(black.long())[:, None]
        if self.mv is not None and self.o1 is not None:
            mv = mv.long(); m0, p1, p2 = mv[..., 0], mv[..., 1], mv[..., 2]
            def rel(a_, b_): ok_ = (a_ < 361) & (b_ < 361); return torch.where(ok_, (a_ // 19 - b_ // 19 + 18) * 37 + (a_ % 19 - b_ % 19 + 18), torch.full_like(a_, 1369))
            hh = hh + self.mv(m0) + self.o1(rel(m0, p1)) + self.o2(rel(m0, p2))
        elif self.mv is not None: hh = hh + self.mv(mv.long())
        if self.enc is not None: hh = self.enc(hh, src_key_padding_mask=pad)
        m = (~pad).float()[..., None]; mean = (hh * m).sum(1) / m.sum(1); a = torch.softmax(self.att(hh).masked_fill(pad[..., None], -1e4), 1); return self.out(torch.cat([mean, (hh * a).sum(1)], 1))


def main():
    args = sys.argv[1:]; K = 256; EMB = "--emb" in args
    if EMB: args.remove("--emb")        # also save the L2-normalised 512-d game embedding (float16)
    if "--k" in args: K = int(args[args.index("--k") + 1]); del args[args.index("--k"):args.index("--k") + 2]
    ck = torch.load(args[0], map_location="cpu", weights_only=False); a = ck["args"]; cls = np.asarray(ck["classes"]); cm = np.load(args[1], allow_pickle=True)["names"]
    names = cm[cls]; assert (names != "").all(), "unmapped class"
    MOVES = bool(a.get("moves")); OFFS = "o1.weight" in ck["model"]
    dev = "cuda"; net = TokNet(len(cls), a["d"], a["layers"], a["drop"], moves=MOVES, offs=OFFS).to(dev); net.load_state_dict(ck["model"]); net.eval(); mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
    Wn = F.normalize(net.W.detach().float(), dim=1)
    allk, topv, topi, embs = [], [], [], []
    for spec in args[3:]:
        rd, gj, kj = spec.split("|"); keys = json.load(open(kj)); G = [json.loads(l) for l in open(gj, encoding="utf-8")]; col = [g["color"] for g in G]
        idx = np.load(rd + "/indices.npy"); assert len(idx) == len(keys) == len(col)
        Z = np.load(rd + "/tok_z.npy", mmap_mode="r"); A = np.load(rd + "/tok_a.npy", mmap_mode="r"); P = np.load(rd + "/tok_ply.npy", mmap_mode="r"); N = np.load(rd + "/tok_n.npy")
        bl = torch.tensor([c == "B" for c in col])
        if MOVES:      # payload row i of the results = games.jsonl line indices[i]; check the pass flag and ply of every token
            li = {int(g["index"]): j for j, g in enumerate(G)}
            MVA = np.stack([mv_codes(G[li[int(idx[i])]]["sgf"], P[i], N[i]) for i in range(len(idx))]); Aall = np.asarray(A); msk = np.arange(90)[None] < N[:, None]
            pf = float(((Aall[..., 9] > .5) == (MVA == PASS))[msk].mean()); pa = float((MVA == PAD)[msk].mean()); print(f"{rd}: move codes, pass-flag agreement {pf:.4f}, padding inside tokens {pa:.4f}", flush=True)
            assert pf > 0.999 and pa < 0.001; del Aall
            if OFFS: MVA = np.stack([MVA] + [np.stack([mv_codes(G[li[int(idx[i])]]["sgf"], P[i], N[i], b) for i in range(len(idx))]) for b in (1, 2)], -1)
        with torch.no_grad():
            for i in range(0, len(keys), 512):
                z = torch.tensor(np.asarray(Z[i:i + 512])).to(dev); aa = torch.tensor(np.asarray(A[i:i + 512])).to(dev)
                x = ((torch.cat([z.float(), aa.float()], 2) - mu) / sd).clamp(-8, 8)
                mvb = torch.tensor(MVA[i:i + 512]).to(dev) if MOVES else None
                e = net.embed(x, torch.tensor(np.asarray(P[i:i + 512]).astype(np.int64)).to(dev), torch.tensor(N[i:i + 512].astype(np.int64)).to(dev), bl[i:i + 512].to(dev), mvb)
                en = F.normalize(e.float(), dim=1); lo = net.s * (en @ Wn.T); v, j = lo.topk(K, 1); topv.append(v.cpu().numpy().astype(np.float16)); topi.append(j.cpu().numpy().astype(np.int16))
                if EMB: embs.append(en.cpu().numpy().astype(np.float16))
        allk += keys; print(spec.split("|")[0], len(keys), flush=True)
    extra = {"emb": np.concatenate(embs)} if EMB else {}
    np.savez(args[2], keys=np.array(allk), topv=np.concatenate(topv), topi=np.concatenate(topi), names=names.astype(str), **extra)
    tv = np.concatenate(topv).astype(np.float32); p1 = 1 / np.exp(tv - tv[:, :1]).sum(1)
    print("saved", args[2], "games", len(allk), " top-1 prob (s=30) mean", p1.mean().round(3), " > 0.5:", np.mean(p1 > 0.5).round(3))


if __name__ == "__main__":
    main()
