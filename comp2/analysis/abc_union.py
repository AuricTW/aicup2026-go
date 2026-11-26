"""Unions of the v10w members (76 parts, hyb_multi.py groups) with the rank1 arms of abc_round.ps1 (B = + rank1 own side, C0 = + raw
rank1 opponent side, C = + converted), paired on the same OOF questions: full coverage, four matched mirrors, mirror mean (se by player).
usage: abc_union.py FULLDIRS HYBDIRS ABC_ROOT
"""
import sys, os, numpy as np
fdirs, hdirs, root = sys.argv[1].split(","), sys.argv[2].split(","), sys.argv[3]
P = ["p8", "p10", "pt1", "pf6a", "pgate"]; sx = ["", "_s1", "_s2"]; Q = "bhmn"
GR = dict(new=P, rg=[f"{p}_rg_s{s}" for p in P for s in range(3)], free=[t for q in Q for s in sx for t in (f"q{q}{s}", f"q{q}g{s}")],
          enr=[t for q in Q for s in sx for t in (f"q{q}c{s}", f"q{q}cg{s}")], rt=[t for q in Q for t in (f"q{q}r", f"q{q}gr")])
R = np.arange(10); K = np.eye(10) + np.exp(-1) * (np.abs(R[:, None] - R[None]) == 1)
NAT = np.array([890, 815, 1046, 1576, 427, 605, 611, 637, 656, 706], float)
tracks = ["m6", "s11", "s12", "s13"]
def ld(dirs, t):
    for d in dirs:
        p = os.path.join(d, f"go_rank_{t}_OOF_posteriors.npz")
        if os.path.exists(p): return np.load(p, allow_pickle=True)
    raise SystemExit(f"missing {t} in {dirs}")
def logs(dirs, abcdir):
    out = {}; meta = None
    for g, tags in GR.items():
        zs = [ld(dirs, t) for t in tags]; meta = meta or zs[0]
        for z in zs: assert (z["question"] == meta["question"]).all()
        out[g] = np.stack([np.log(np.clip(z["fused"], 1e-12, 1)) for z in zs])
    for arm in ("B", "C", "C0"):
        zs = [np.load(os.path.join(abcdir, f"go_rank_{arm}_{p}_OOF_posteriors.npz"), allow_pickle=True) for p in P]
        for z in zs: assert (z["question"] == meta["question"]).all()
        out[arm] = np.stack([np.log(np.clip(z["fused"], 1e-12, 1)) for z in zs])
    return out, meta
def score(L, comp, y):
    M = np.concatenate([L[g] for g in comp.split("+")]).mean(0); a = (np.exp(M - M.max(1, keepdims=True)) @ K).argmax(1); return K[a, y]
LF, meta = logs(fdirs, os.path.join(root, "full")); y = meta["y"].astype(int); up, inv = np.unique(meta["player_id"], return_inverse=True)
W = (NAT / NAT.sum() / (np.bincount(y, minlength=10) / len(y)))[y]; W /= W.mean()
LH = [logs([d], os.path.join(root, t))[0] for d, t in zip(hdirs, tracks)]
def cse(d): m = np.bincount(inv, weights=d) / np.bincount(inv); return m.std() / np.sqrt(len(m))
specs = [("v10w", "new+rg+free+rt+enr"), ("+B", "new+rg+free+rt+enr+B"), ("+C0", "new+rg+free+rt+enr+C0"), ("+B+C0", "new+rg+free+rt+enr+B+C0"),
         ("+B+C+C0", "new+rg+free+rt+enr+B+C+C0"), ("new", "new"), ("new+B+C0", "new+B+C0"), ("new+B", "new+B")]
s0f = score(LF, specs[0][1], y); s0h = [score(L, specs[0][1], y) for L in LH]
for name, comp in specs:
    sf = score(LF, comp, y); sh = [score(L, comp, y) for L in LH]; d = sf - s0f; dm = np.mean([a - b for a, b in zip(sh, s0h)], 0)
    print(f"{name:9s} full {sf.mean():.4f} ({d.mean():+.4f} se {cse(d):.4f}, nat {np.mean(d * W):+.4f}, changed {(d != 0).sum()})   mirrors " +
          " ".join(f"{s.mean():.4f}" for s in sh) + f"  mean {np.mean([s.mean() for s in sh]):.4f} ({dm.mean():+.4f} se {cse(dm):.4f}, nat {np.mean(dm * W):+.4f})")
