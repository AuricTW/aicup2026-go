"""2026-11-22: opponent co-play graph component for comp1 (doc section 37.3).
Bank: 839,395 training games, owner X; opponent side recognised by the sequence net b1_full as training account Y (posterior p;
runs/oppbank_20261120_v2/full.npz).  Graph A[X, Y] += w(p) for p > thr (symmetric).  Official games: opponent recognised the same way.
og_<sim>_t<thr>_<u|w>(q, c) = mean over pairs (g in question, h in gallery of c, both recognised, Y_g != Y_h) of sim(Y_g, Y_h),
  sim = cos (cosine of the accounts' neighbourhood rows) or svd (cosine in a 64-d SVD embedding of the normalised adjacency);
  u = plain mean, w = pairs weighted by p_g * p_h (and edges weighted by p).  Exact same opponent is excluded (that is the rematch term).
Only anonymous competition records and gallery membership; never question labels or answers.
usage: oppgraph_20261122_v1.py val <old|mirror|mirror2> [variants]   -> extra/ogv_<pool>_d<r>.npz
       oppgraph_20261122_v1.py official <variant>                  -> runs/oppgraph_20261122_v1/official_<pool>.npy (q x c; original galleries)"""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import svds
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'runs/oppgraph_20261122_v1'
T = np.load(ROOT / 'runs/oppbank_20261120_v2/full.npz'); OWN = np.load(ROOT / 'runs/seqid_20261115_v1/bundle_g1.npz')['OWN'].astype(np.int64)
VARIANTS = [f'og_{s}_t{t}_{w}' for s in ('cos', 'svd') for t in ('0.3', '0.5', '0.7') for w in ('u', 'w')]
_G = {}


def graph(thr, weighted):
    key = (thr, weighted)
    if key not in _G:
        p, y = T['top'], T['owner']; m = (p > thr) & (y != OWN); n = int(max(OWN.max(), y.max())) + 1
        A = sp.coo_matrix((p[m] if weighted else np.ones(int(m.sum())), (OWN[m], y[m])), shape=(n, n)).tocsr(); A = (A + A.T).tocsr(); deg = np.asarray(A.sum(1)).ravel()
        Rn = sp.diags(1 / np.maximum(np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel()), 1e-9)) @ A; d = sp.diags(1 / np.sqrt(np.maximum(deg, 1e-9)))
        u, s, _ = svds((d @ A @ d).astype(np.float64), k=64); Es = u * s; Es /= np.maximum(np.linalg.norm(Es, axis=1, keepdims=True), 1e-9); _G[key] = (Rn.tocsr(), Es)
    return _G[key]


def matrix(variant, supports, questions):
    _, sim, t, w = variant.split('_'); thr = float(t[1:]); weighted = w == 'w'; Rn, Es = graph(thr, weighted); op, oo = T['off_top'], T['off_owner']; rec = op > thr
    out = np.zeros((len(questions), len(supports))); gal = [[(oo[h], op[h]) for h in s if rec[h]] for s in supports]
    for i, g in enumerate(questions):
        Yq = [(oo[x], op[x]) for x in g if rec[x]]
        if not Yq: continue
        for j, Yh in enumerate(gal):
            pr = [(a, b, pa * pb) for a, pa in Yq for b, pb in Yh if a != b]
            if not pr: continue
            a = np.array([x[0] for x in pr]); b = np.array([x[1] for x in pr]); ww = np.array([x[2] for x in pr]) if weighted else np.ones(len(pr))
            v = np.asarray(Rn[a].multiply(Rn[b]).sum(1)).ravel() if sim == 'cos' else (Es[a] * Es[b]).sum(1)
            out[i, j] = float((v * ww).sum() / ww.sum())
    return out


def val(which, variants):
    import suppsplit_eval_20261115_v2 as s2, mirror_val_20261121_v1 as mv
    if which == 'old': it = ((p, r, e) for p, r, e, f in s2.ss.each()); dest = s2.ss.RUN / 'extra'
    else:
        if which == 'mirror2': mv.MRUN = ROOT / 'runs/mirrorval2_20261122_v1'
        it = mv.each(); dest = mv.MRUN / 'extra'
    for pool, r, ep in it:
        qs = [z['indices'] for z in ep['questions']]; np.savez(dest / f'ogv_{pool}_d{r}.npz', **{v: matrix(v, ep['supports'], qs) for v in variants}); print(which, pool, r, flush=True)


def official(variant):
    import suppsplit_eval_20261115_v2 as s2
    OUT.mkdir(parents=True, exist_ok=True); games, pools, keys, pos = s2.ss.official(); info = {'variant': variant}
    for pool in pools:
        S = matrix(variant, [[pos[k] for k in pool.supports[c]] for c in pool.candidate_ids], [[pos[k] for k in z['keys']] for z in pool.questions]); assert S.shape == (400, 600)
        np.save(OUT / f'official_{pool.name}.npy', S); info[pool.name] = {'rows_with_signal': int((np.abs(S).sum(1) > 0).sum())}
    (OUT / 'INFO.json').write_text(json.dumps(info, indent=1)); print(info, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'val': val(sys.argv[2], sys.argv[3:] or VARIANTS)
    else: official(sys.argv[2])
