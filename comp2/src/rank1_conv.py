"""Opponent-side RANK evidence converted to the TARGET player's rank (11-20, check of an external review).
The token rank net rank1 (tok_ranknet_vm.py) was trained on the moves of a player with THAT player's rank.  tok_rank_apply.py ran it
on the opponent side and stored the result as source 'dopp', which fuse_sources.py treats as evidence on the TARGET's rank (it only
centres it and fits one scale per G).  So the 11-17 test of dopp (OOF 0.7921 -> 0.7938, closed as noise) fed the OPPONENT's rank
distribution into target coordinates without the matchmaking conversion.
Conversion (Bayes, the same one the identity sources use): q = rank1 posterior of the opponent's rank under its training prior pi;
likelihood l_o = q_o / pi_o; evidence on the target rank t:  ev_t = log( sum_o K_c[t, o] l_o ),  K_c[t, o] = P(opponent rank o | owner
rank t) = the empirical matchmaking kernels of the reference identity file (idnet_eval2.emp_kernels: TRAINING-owner games with a
recognised opponent, c = HD0 / HD1 owner black / HD1 owner white).  No OOF, dev or test label is used for the conversion.
Outputs (fuse_sources.py --dir-x format, blocks dev and x; every key of the reference file present, missing games = zero evidence so
the OOF question set stays identical to the reference runs):
  OUT_own.npz  : own side only (source down)                 OUT_conv.npz : own + converted opponent side (down, dopp)
  OUT_raw.npz  : own + opponent side as before (the 11-17 form, for the conversion effect)
usage: rank1_conv.py RANK1_DIR.npz REF_ID.npz OUT_PREFIX                     (blocks dev and x, both from REF_ID: the x track)
       rank1_conv.py RANK1_DIR.npz - OUT_PREFIX dev=REF1.npz test=REF2.npz     (any blocks, each with its own reference file = the
       kernels fuse_sources.load_id uses for that block: full track dev = --id-full, test = --id-test)
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idnet_eval2 import emp_kernels
src, ref, outp = sys.argv[1:4]
BLK = [(b, ref) for b in ("dev", "x")] if ref != "-" else [tuple(s.split("=", 1)) for s in sys.argv[4:]]
PI = np.array([44074, 42248, 48726, 61219, 26581, 38138, 37026, 35079, 39152, 40612], float); PI /= PI.sum()    # rank1 training labels (tokres_rank_labels.npz)
def nk(k): return str(k).replace("|", "#")
D = np.load(src, allow_pickle=True)
# rank1 per-game outputs over all its blocks (dev / x / test) by key
own, opp = {}, {}
for b in ("dev", "x", "test"):
    for k, lo, lp in zip(D[f"{b}_keys"], D[f"{b}_own_logp"], D[f"{b}_opp_logp"]): own[nk(k)] = lo.astype(np.float64); opp[nk(k)] = lp.astype(np.float64)
res = {"own": {}, "conv": {}, "raw": {}}
for b, rf in BLK:
    R = np.load(rf, allow_pickle=True); Ks = emp_kernels(R, 0.5); print(f"{b}: reference {rf}, kernel HD0 diagonal", np.round(np.diag(Ks[0]), 2).tolist())
    keys = np.array([nk(k) for k in R[f"{b}_keys"]]); hd = R[f"{b}_hd"]; bl = R[f"{b}_black"].astype(bool); n = len(keys)
    have = np.array([k in own for k in keys]); LO = np.zeros((n, 10)); LP = np.zeros((n, 10))
    LO[have] = np.stack([own[k] for k in keys[have]]); LP[have] = np.stack([opp[k] for k in keys[have]])
    q = np.exp(LP - LP.max(1, keepdims=True)); q /= q.sum(1, keepdims=True); l = q / PI[None]; l /= l.sum(1, keepdims=True)
    EV = np.zeros((n, 10))
    for c, m in enumerate((hd == 0, (hd == 1) & bl, (hd == 1) & ~bl)):
        if m.any(): EV[m] = np.log(l[m] @ Ks[c].T + 1e-4)
    EV[~have] = 0.0; LO[~have] = 0.0; LP[~have] = 0.0
    for a in (LO, LP, EV): a -= a.mean(1, keepdims=True)
    print(f"{b}: reference games {n}, with rank1 output {have.mean():.4f};  hd>1 games {(hd > 1).mean():.4f} (kept as zero conversion)")
    for name, opp_part in (("own", None), ("conv", EV), ("raw", LP)):
        res[name][f"{b}_keys"] = keys; res[name][f"{b}_own_logp"] = LO.astype(np.float32)
        if opp_part is not None: res[name][f"{b}_opp_logp"] = opp_part.astype(np.float32)
for name, d in res.items(): np.savez(f"{outp}_{name}.npz", **d); print("saved", f"{outp}_{name}.npz")
