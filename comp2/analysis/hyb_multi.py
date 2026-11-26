"""Blend compositions at full coverage and on SEVERAL matched hybrid mirrors (drop seeds), paired on the same OOF questions (11-20).
Groups (tags as in hyb_m6_round.ps1): new = five prior parts; rg = their ridge twins, seeds 0-2; free = 24 free tuned-trunk parts;
enr = their 24 profile-enriched twins; rt = 8 ridge tuned parts.
For every composition: score, paired difference to the first one with the standard error clustered by player - at full coverage,
on each mirror, and for the MEAN over the mirrors (the per-question differences are averaged over the mirrors first, so the
drop-draw noise is averaged and the player-sampling noise is counted once); also re-weighted to the natural class mix.
usage: hyb_multi.py FULLDIR1,FULLDIR2,... HYBDIR1,HYBDIR2,... NAME=grp+grp [NAME=grp+grp ...]
"""
import sys, os, numpy as np
fdirs = sys.argv[1].split(","); hdirs = sys.argv[2].split(","); specs = [s.split("=", 1) for s in sys.argv[3:]]
P = ["p8", "p10", "pt1", "pf6a", "pgate"]; sx = ["", "_s1", "_s2"]; Q = "bhmn"
GR = dict(new=P, rg=[f"{p}_rg_s{s}" for p in P for s in range(3)], free=[t for q in Q for s in sx for t in (f"q{q}{s}", f"q{q}g{s}")],
          enr=[t for q in Q for s in sx for t in (f"q{q}c{s}", f"q{q}cg{s}")], rt=[t for q in Q for t in (f"q{q}r", f"q{q}gr")])
R = np.arange(10); K = np.eye(10) + np.exp(-1) * (np.abs(R[:, None] - R[None]) == 1)
def ld(dirs, t):
    for d in dirs:
        p = os.path.join(d, f"go_rank_{t}_OOF_posteriors.npz")
        if os.path.exists(p): return np.load(p, allow_pickle=True)
    raise SystemExit(f"missing {t} in {dirs}")
def logs(dirs):
    out = {}; meta = None
    for g, tags in GR.items():
        zs = [ld(dirs, t) for t in tags]; meta = meta or zs[0]
        for z in zs: assert (z["question"] == meta["question"]).all()
        out[g] = np.stack([np.log(np.clip(z["fused"], 1e-12, 1)) for z in zs])
    return out, meta
def score(L, comp, y):
    M = np.concatenate([L[g] for g in comp.split("+")]).mean(0); P_ = np.exp(M - M.max(1, keepdims=True)); P_ /= P_.sum(1, keepdims=True); a = (P_ @ K).argmax(1); return K[a, y], a
LFd, meta = logs(fdirs); y = meta["y"].astype(int); pl = meta["player_id"]; fold = meta["fold"].astype(bool); up, inv = np.unique(pl, return_inverse=True)
NAT = np.array([890, 815, 1046, 1576, 427, 605, 611, 637, 656, 706], float); W = (NAT / NAT.sum() / (np.bincount(y, minlength=10) / len(y)))[y]; W /= W.mean()
def cse(d): m = np.bincount(inv, weights=d) / np.bincount(inv); return m.std() / np.sqrt(len(m))
LH = []
for d in hdirs:
    L, m = logs([d]); assert (m["question"] == meta["question"]).all(); LH.append(L)
print(f"questions {len(y)}  players {len(up)}  mirrors {len(hdirs)}: {[os.path.basename(d) for d in hdirs]}")
print(f"members: " + ", ".join(f"{g} {len(t)}" for g, t in GR.items()))
ref = specs[0][1]; sf0, af0 = score(LFd, ref, y); sh0 = [score(L, ref, y)[0] for L in LH]
print(f"REF {specs[0][0]} = {ref}: full {sf0.mean():.4f}   mirrors " + " ".join(f"{s.mean():.4f}" for s in sh0) + f"   mean {np.mean([s.mean() for s in sh0]):.4f}")
for name, comp in specs[1:]:
    sf, af = score(LFd, comp, y); df = sf - sf0; dh = [score(L, comp, y)[0] - s0 for L, s0 in zip(LH, sh0)]; dm = np.mean(dh, 0); n = sum(len(GR[g]) for g in comp.split("+"))
    print(f"  {name:6s} = {comp:26s} ({n:3d} members)  FULL {df.mean():+.4f} (se {cse(df):.4f}; natural mix {np.average(df, weights=W):+.4f}; {int((af != af0).sum())} answers)"
          f"   HYBRID mean of {len(dh)} mirrors {dm.mean():+.4f} (se {cse(dm):.4f}; folds {dm[fold].mean():+.4f} / {dm[~fold].mean():+.4f}; natural mix {np.average(dm, weights=W):+.4f})   per mirror " + " ".join(f"{d.mean():+.4f}" for d in dh))
