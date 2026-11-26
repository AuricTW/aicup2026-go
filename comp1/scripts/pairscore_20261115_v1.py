"""Learned colour-aware pair scorer, fitted on the OFFICIAL population (support-split inside the new pools; label = candidate id that
ships with the test files; no answer key, no question labels).
Why: with mixed colours the frozen embeddings identify a player far better against gallery games of the same colour, and the plain
centroid cosine cannot weigh a 2-game same-colour centroid against an 8-game other-colour centroid.
Unit = (held-out game i, candidate j).  Features per embedding model k (head d, head lin, seq-id): cosine of the game with the
candidate's centroid of ALL / SAME-colour / OTHER-colour gallery games, each also z-scored over the 600 candidates, plus the two counts.
Model: HistGradientBoosting (binary: is j the owner of game i).  Question score = mean logit over the question's games.
Training pairs: fresh equal-gallery draws (9/8/7/5 games kept) inside ONE pool; evaluation on the stored suppsplit v2 episodes of the
OTHER pool (model never saw those players); compared with b2e3 (v6b neural recipe) on the same episodes.
Usage: run <tag> [models comma: d,lin,seq] [draws]
"""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import suppsplit_eval_20261115_v2 as s2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
ss = s2.ss; A = ss.A; p = ss.p; g = ss.g; RUN = p.ROOT / 'runs/pairscore_20261115_v1'; SEED = 2026111508; V1 = p.ROOT / 'runs/suppsplit_20261115_v1'; SEQ = p.ROOT / 'runs/seqid_20261115_v1/a2/g1/f0/off_own.npy'


def nrm(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def zrow(a):
    m = np.nanmean(a, 1, keepdims=True); s = np.nanstd(a, 1, keepdims=True) + 1e-9; return (a - m) / s


def load_models(names):
    H = np.load(V1 / 'heademb_b3.npz'); M = {}
    for n in names: M[n] = [nrm(np.load(SEQ).astype(np.float64))] if n == 'seq' else [nrm(H[n][s].astype(np.float64)) for s in range(H[n].shape[0])]
    return M, H['black']


def features(M, black, qgames, sup):
    """qgames: list of game indices; sup: list (per candidate) of gallery index arrays -> X [len(qgames), ncand, F], feature names"""
    sup = [np.asarray(s) for s in sup]; nB = np.array([black[s].sum() for s in sup]); nW = np.array([len(s) - black[s].sum() for s in sup]); qb = black[qgames]; cols = []; names = []
    for n, embs in M.items():
        acc = [0, 0, 0]
        for Z in embs:
            cen = lambda idx: nrm(np.stack([Z[i].mean(0) if len(i) else np.zeros(Z.shape[1]) for i in idx]))
            call = cen(sup); cB = cen([s[black[s]] for s in sup]); cW = cen([s[~black[s]] for s in sup]); Q = Z[qgames]; sB = Q @ cB.T; sW = Q @ cW.T
            sB[:, nB == 0] = np.nan; sW[:, nW == 0] = np.nan; acc[0] = acc[0] + Q @ call.T; acc[1] = acc[1] + np.where(qb[:, None], sB, sW); acc[2] = acc[2] + np.where(qb[:, None], sW, sB)
        for a, tag in zip(acc, ('all', 'same', 'diff')): a = a / len(embs); cols += [a, zrow(a)]; names += [f'{n}_{tag}', f'{n}_{tag}_z']
    ns = np.where(qb[:, None], nB[None], nW[None]).astype(float); nd = np.where(qb[:, None], nW[None], nB[None]).astype(float); cols += [ns, nd]; names += ['n_same', 'n_diff']
    return np.stack(cols, 2), names


def train_pairs(M, black, pool, pos, draws, rng):
    full = [np.array([pos[k] for k in pool.supports[c]]) for c in pool.candidate_ids]; Xs, ys, ws = [], [], []
    for r in range(draws):
        for held in (1, 2, 3, 5):
            sup, qg, own = [], [], []
            for j, s in enumerate(full): hj = rng.choice(10, held, replace=False); m = np.zeros(10, bool); m[hj] = True; sup.append(s[~m]); qg += s[m].tolist(); own += [j] * held
            qg = np.array(qg); own = np.array(own); X, names = features(M, black, qg, sup); base = X[:, :, 0]
            for i in range(len(qg)):
                order = np.argsort(-base[i]); order = order[order != own[i]]; hard = order[:30]; rnd = rng.choice(order[30:], 30, replace=False)
                Xs.append(X[i, np.concatenate([[own[i]], hard, rnd])]); ys.append(np.r_[1, np.zeros(60)]); ws.append(np.r_[1., np.ones(30), np.full(30, (len(order) - 30) / 30)])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws), names


def run(tag, models='d,lin,seq', draws=2):
    out = RUN / tag; out.mkdir(parents=True, exist_ok=False); M, black = load_models(models.split(',')); games, pools, keys, pos = ss.official(); rng = np.random.default_rng(SEED); clf = {}; info = {}
    for pool in pools:
        X, y, w, names = train_pairs(M, black, pool, pos, draws, rng); c = HistGradientBoostingClassifier(max_iter=400, learning_rate=.06, max_leaf_nodes=31, l2_regularization=1.0, min_samples_leaf=50, random_state=0).fit(X, y, sample_weight=w); clf[pool.name] = c
        info[pool.name] = {'pairs': int(len(y)), 'features': names}; print(pool.name, 'trained on', len(y), 'pairs', flush=True)
    eps = json.loads((ss.RUN / 'EPISODES.json').read_text()); data = []
    for pool in pools:
        other = [q for q in pools if q.name != pool.name][0].name
        for r, ep in enumerate(eps[pool.name]):
            qg = np.array([i for z in ep['questions'] for i in z['indices']]); X, names = features(M, black, qg, ep['supports']); L = clf[other].decision_function(X.reshape(-1, X.shape[2])).reshape(X.shape[0], X.shape[1])
            st = np.cumsum([0] + [z['q'] for z in ep['questions']]); S = np.stack([L[st[i]:st[i + 1]].mean(0) for i in range(len(ep['questions']))]); V = np.load(ss.RUN / f'comps_{pool.name}_d{r}.npz')
            data.append((pool.name, S, V['b2e3'], V['ens3'], np.array([z['target'] for z in ep['questions']]), np.array([z['q'] for z in ep['questions']]), ep.get('gallery'))); np.save(out / f'S_{pool.name}_d{r}.npy', S); print(pool.name, r, flush=True)
    Q = np.concatenate([d[5] for d in data]); G = np.concatenate([[d[6]] * len(d[5]) for d in data]); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    def evals(fn):
        bal = []
        for d in data:
            o = [e for e in data if e[0] != d[0]]; T = A.calibrate(A.z(np.vstack([fn(e) for e in o])), np.concatenate([e[4] for e in o]), np.concatenate([e[5] for e in o])); bal.append(A.sid(A.new_transform(fn(d), d[5], T), d[4]))
        return np.concatenate(bal)
    ref = evals(lambda d: d[2]); var = {'b2e3': lambda d: d[2], 'ens3': lambda d: d[3], 'pair': lambda d: d[1]}
    for w in (.3, .5, .7): var[f'b2e3+pair_w{w}'] = (lambda d, w=w: (1 - w) * A.z(d[2]) + w * A.z(d[1]))
    for nm, fn in var.items():
        b = evals(fn); dl = b - ref; res[nm] = {'sinkhorn': float(b.mean()), 'delta_vs_b2e3': float(dl.mean()), 'CI95': np.quantile(dl[B].mean(1), [.025, .975]).round(4).tolist(), 'by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'by_gallery': {str(k): float(b[G == k].mean()) for k in (7, 5)}}
        print(f"{nm:16s} sink {b.mean():.4f} d {dl.mean():+.4f} {res[nm]['CI95']}", {k: round(v, 3) for k, v in res[nm]['by_q'].items()}, {k: round(v, 3) for k, v in res[nm]['by_gallery'].items()}, flush=True)
    (out / 'RESULT.json').write_text(json.dumps({'models': models, 'draws': draws, 'info': info, 'results': res}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=6): run(sys.argv[2], *(sys.argv[3:4]), *([int(sys.argv[4])] if len(sys.argv) > 4 else []))
