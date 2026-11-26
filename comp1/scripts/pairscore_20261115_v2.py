"""Learned colour-aware pair scorer v2: LINEAR listwise model instead of the tree model of v1.
v1 (HistGradientBoosting on pair features) ranked far worse than the plain cosine (0.46 vs 0.66): piecewise-constant trees cannot
resolve the small cosine differences among the top candidates.  v2 keeps the idea but fits a multinomial-logit over the 600 candidates:
  score(i, j) = sum_k [ a_k * z_all_k + b_k[n_same] * z_same_k + c_k[n_diff] * z_diff_k ]
k = embedding model (head d, head lin, optionally seq); z_* = cosine of game i with candidate j's centroid of all / same-colour /
other-colour gallery games, z-scored over candidates; b, c are free per count (1..10 games; 0 games -> term absent).
Loss = cross-entropy of the true owner over all 600 candidates (no negative sampling), L2 penalty; fitted with Adam.
Train: fresh equal-gallery draws (9/8/7/5 kept) in ONE official pool; test: stored suppsplit v2 episodes of the OTHER pool.
Question score = mean over the question's games.  Compared with b2e3 (v6b neural recipe).
Usage: run <tag> [models d,lin,seq] [draws]
"""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairscore_20261115_v1 as p1
import numpy as np
import torch
from threadpoolctl import threadpool_limits
ss = p1.ss; A = ss.A; p = ss.p; RUN = p.ROOT / 'runs/pairscore_20261115_v2'; SEED = 2026111509; NB = 11


def phi(M, black, qgames, sup):
    X, names = p1.features(M, black, np.asarray(qgames), sup); K = len(M); ns = X[:, :, -2].astype(int); nd = X[:, :, -1].astype(int); out = np.zeros((X.shape[0], X.shape[1], K * (1 + 2 * NB)), np.float32)
    for k in range(K):
        za, zs, zd = X[:, :, 6 * k + 1], np.nan_to_num(X[:, :, 6 * k + 3]), np.nan_to_num(X[:, :, 6 * k + 5]); o = k * (1 + 2 * NB); out[:, :, o] = za
        I, J = np.indices(ns.shape); out[I, J, o + 1 + np.clip(ns, 0, NB - 1)] = zs; out[I, J, o + 1 + NB + np.clip(nd, 0, NB - 1)] = zd
    return out


def fit(X, own, l2=1e-3, steps=400):
    Xt = torch.tensor(X); y = torch.tensor(own); w = torch.zeros(X.shape[2], requires_grad=True); opt = torch.optim.Adam([w], lr=.05)
    for s in range(steps):
        loss = torch.nn.functional.cross_entropy(Xt @ w, y) + l2 * (w ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().numpy(), float(loss)


def run(tag, models='d,lin', draws=1):
    out = RUN / tag; out.mkdir(parents=True, exist_ok=False); M, black = p1.load_models(models.split(',')); games, pools, keys, pos = ss.official(); rng = np.random.default_rng(SEED); W = {}; K = len(M)
    for pool in pools:
        full = [np.array([pos[k] for k in pool.supports[c]]) for c in pool.candidate_ids]; Xs, ys = [], []
        for r in range(draws):
            for held in (1, 2, 3, 5):
                sup, qg, own = [], [], []
                for j, s in enumerate(full): hj = rng.choice(10, held, replace=False); m = np.zeros(10, bool); m[hj] = True; sup.append(s[~m]); qg += s[m].tolist(); own += [j] * held
                Xs.append(phi(M, black, qg, sup)); ys += own
        W[pool.name], loss = fit(np.concatenate(Xs), np.array(ys)); del Xs; print(pool.name, 'fit loss', round(loss, 4), flush=True)
        for k, n in enumerate(M):
            o = k * (1 + 2 * NB); w = W[pool.name]; print(f'   {n}: all {w[o]:.2f} | same by count', np.round(w[o + 1:o + 1 + NB], 2).tolist(), '| diff by count', np.round(w[o + 1 + NB:o + 1 + 2 * NB], 2).tolist(), flush=True)
    np.savez(out / 'weights.npz', **W); eps = json.loads((ss.RUN / 'EPISODES.json').read_text()); data = []
    for pool in pools:
        other = [q for q in pools if q.name != pool.name][0].name
        for r, ep in enumerate(eps[pool.name]):
            qg = [i for z in ep['questions'] for i in z['indices']]; L = phi(M, black, qg, ep['supports']) @ W[other]; st = np.cumsum([0] + [z['q'] for z in ep['questions']]); S = np.stack([L[st[i]:st[i + 1]].mean(0) for i in range(len(ep['questions']))])
            V = np.load(ss.RUN / f'comps_{pool.name}_d{r}.npz'); data.append((pool.name, S, V['b2e3'], V['ens3'], np.array([z['target'] for z in ep['questions']]), np.array([z['q'] for z in ep['questions']]), ep.get('gallery'), V['base2'])); np.save(out / f'S_{pool.name}_d{r}.npy', S)
    Q = np.concatenate([d[5] for d in data]); G = np.concatenate([[d[6]] * len(d[5]) for d in data]); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    def evals(fn):
        bal = []
        for d in data:
            o = [e for e in data if e[0] != d[0]]; T = A.calibrate(A.z(np.vstack([fn(e) for e in o])), np.concatenate([e[4] for e in o]), np.concatenate([e[5] for e in o])); bal.append(A.sid(A.new_transform(fn(d), d[5], T), d[4]))
        return np.concatenate(bal)
    ref = evals(lambda d: d[2]); var = {'b2e3': lambda d: d[2], 'ens3': lambda d: d[3], 'pair': lambda d: d[1], 'b2+pair': lambda d: ss.BW * A.z(d[7]) + (1 - ss.BW) * A.z(d[1])}
    for w in (.3, .5, .7): var[f'ens3+pair_w{w}'] = (lambda d, w=w: (1 - w) * A.z(d[3]) + w * A.z(d[1]))
    for nm, fn in var.items():
        b = evals(fn); dl = b - ref; res[nm] = {'sinkhorn': float(b.mean()), 'delta_vs_b2e3': float(dl.mean()), 'CI95': np.quantile(dl[B].mean(1), [.025, .975]).round(4).tolist(), 'by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'by_gallery': {str(k): float(b[G == k].mean()) for k in (7, 5)}}
        print(f"{nm:16s} sink {b.mean():.4f} d {dl.mean():+.4f} {res[nm]['CI95']}", {k: round(v, 3) for k, v in res[nm]['by_q'].items()}, {k: round(v, 3) for k, v in res[nm]['by_gallery'].items()}, flush=True)
    (out / 'RESULT.json').write_text(json.dumps({'models': models, 'draws': draws, 'results': res}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=6): run(sys.argv[2], *(sys.argv[3:4]), *([int(sys.argv[4])] if len(sys.argv) > 4 else []))
