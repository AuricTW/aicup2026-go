"""Lower-variance stage 1 for fuse_sources.py (see stage1_bag.py / stage1_reg.py / stage1_xfit.py for the evidence).
The pipeline fits (k scales + 10 class biases) independently for each G on ~790 dev players: 56 parameters whose sampling noise is
common to every blend member (a bootstrap of the players flips ~32 of 800 official answers).
  fit_smooth : theta_G = alpha + beta * log(G / 10); the biases carry the known offset log(mix of the accounts with >= G games)
  fit_ridge  : per-G fit with a ridge toward the smooth values (lam_s on the scales, lam_b on the biases)
D: dict G -> dict(S = array [k, n, 10] of summed evidence, y = labels).  Returns dict G -> theta (k scales, then 10 biases),
the same layout as predict_test.fit / final_predict.prob.
"""
import os, numpy as np
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); _LM = {}
import sys; sys.path.insert(0, HERE); from comp2_paths import WORK


def log_mix(G, moves=None):
    """log label mix of the labelled accounts with >= G listed games (HD <= 1), all non-test accounts of moves_all3.npz"""
    if not _LM:
        z = np.load(moves or os.path.join(WORK, "moves_all3.npz"), allow_pickle=True); pid = z["pid"].astype(np.int64); y = z["y"].astype(int); m = (z["split"] != 2) & (z["hd"].astype(int) <= 1)
        n = pid.max() + 1; lab = np.full(n, -1); lab[pid[m]] = y[m]; _LM["lab"] = lab; _LM["cnt"] = np.bincount(pid[m], minlength=n)
    a = (_LM["lab"] >= 0) & (_LM["cnt"] >= G); h = np.bincount(_LM["lab"][a], minlength=10).astype(float); return np.log(h / h.sum())


def nll_G(t, S, y, k):
    zz = np.tensordot(t[:k], S, axes=(0, 0)) + t[k:]; zz = zz - zz.max(1, keepdims=True); p = np.exp(zz); p /= p.sum(1, keepdims=True); nq = len(y)
    val = -np.log(p[np.arange(nq), y]).mean(); d = p.copy(); d[np.arange(nq), y] -= 1; d /= nq
    return val, np.r_[np.tensordot(S, d, axes=([1, 2], [0, 1])), d.sum(0)]


def fit_smooth(D, l2=1e-3):
    GS = sorted(D); k = D[GS[0]]["S"].shape[0]; LG = {G: np.log(G / 10) for G in GS}; LM = {G: log_mix(G) for G in GS}

    def f(t):
        a, b = t[:k + 10], t[k + 10:]; val = 0; ga = np.zeros(k + 10); gb = np.zeros(k + 10)
        for G in GS:
            th = a + b * LG[G]; th[k:] += LM[G]; v, g = nll_G(th, D[G]["S"], D[G]["y"], k); val += v; ga += g; gb += g * LG[G]
        val += l2 * (a[k:] ** 2).sum(); ga[k:] += 2 * l2 * a[k:]; return val, np.r_[ga, gb]
    t = minimize(f, np.r_[np.ones(k) * 0.5, np.zeros(10), np.zeros(k + 10)], jac=True, method="L-BFGS-B", options=dict(maxiter=4000, ftol=1e-12, gtol=1e-9)).x; a, b = t[:k + 10], t[k + 10:]; out = {}
    for G in GS: th = a + b * LG[G]; th[k:] += LM[G]; out[G] = th
    return out


def fit_ridge(D, lam_s, lam_b):
    tg = fit_smooth(D); out = {}; GS = sorted(D); k = D[GS[0]]["S"].shape[0]
    if lam_s >= 1e5 and lam_b >= 1e5: return tg
    w = np.r_[np.full(k, lam_s), np.full(10, lam_b)]
    for G in GS:
        def f(t, G=G): v, g = nll_G(t, D[G]["S"], D[G]["y"], k); dd = t - tg[G]; return v + (w * dd ** 2).sum(), g + 2 * w * dd
        out[G] = minimize(f, tg[G].copy(), jac=True, method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-12, gtol=1e-9)).x
    return out
