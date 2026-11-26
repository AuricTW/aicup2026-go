"""Cross-fit fresh evaluation inside the bigbank (never-evaluated identities) for representation choices.

3 folds over the 2105 bigbank players (label % 3). For fold k: fit PLDA on the other folds' players only;
build one official-like episode from fold-k players: 600 candidates x 10 support games, 400 questions with
400 distinct targets, 100 each of 1/2/3/5 query games (query games disjoint from supports).
Neural-only comparisons (no prior). Sinkhorn temps calibrated on the *other* folds' episodes.
"""
from pathlib import Path
import sys, argparse, json, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank90_20261114_v1 as bb
import assign_sinkhorn_20261113_v1 as A
import numpy as np
from threadpoolctl import threadpool_limits
s = bb.s; p = bb.p; old = bb.old
RUN = p.ROOT / 'runs/crossfit90_20261114_v1'; SEED = 2026111402; K = 3
VIEWS = {'v90': (90,), 'v64': (64,), 'v8': (8,), 'cat6490': (64, 90)}
DIMS = (256, 768, 1024)


def h(v): return hashlib.sha256((str(SEED) + '|' + str(v)).encode()).digest()


def raw_for(ids, views):
    return np.hstack([bb.lookup_views(ids, v) for v in views]) if len(views) > 1 else bb.lookup_views(ids, views[0])


def episodes():
    ids = np.load(bb.RUN / 'train_indices.npy'); lab = np.load(bb.RUN / 'train_labels.npy'); eps = []
    for k in range(K):
        players = [int(l) for l in np.unique(lab) if l % K == k]; players.sort(key=h)
        games = {l: sorted(ids[lab == l].tolist(), key=h) for l in players}
        cands = players[:600]; assert len(cands) == 600
        rng = np.random.default_rng(SEED + k); order = list(rng.permutation(cands)); tq = {}
        for n in (5, 3, 2, 1):
            got = 0
            for l in list(order):
                if got == 100: break
                if len(games[l]) >= 10 + n: tq[l] = n; order.remove(l); got += 1
            assert got == 100
        sup = [games[l][:10] for l in cands]; pos = {l: j for j, l in enumerate(cands)}
        qs = [{'id': f'f{k}_{l}', 'indices': games[l][10:10 + n], 'q': n, 'target': pos[l], 'player': str(l)} for l, n in sorted(tq.items(), key=lambda x: h(x[0]))]
        eps.append({'fold': k, 'candidates': [str(c) for c in cands], 'supports': sup, 'questions': qs})
    return eps


def fit_all():
    RUN.mkdir(exist_ok=True); eps = episodes(); p.b.write(RUN / 'EPISODES.json', eps)
    ids = np.load(bb.RUN / 'train_indices.npy'); lab = np.load(bb.RUN / 'train_labels.npy')
    for vname, views in VIEWS.items():
        raw = raw_for(ids, views)
        for k in range(K):
            tr = lab % K != k
            for d in DIMS:
                folder = RUN / f'models/f{k}_{vname}_d{d}'
                if folder.exists(): continue
                folder.mkdir(parents=True); x = old.pre(raw[tr], folder, True); r = p.fit_plda(x, lab[tr], folder / 'PLDA.npz', dimension=d); print(vname, k, d, r['seconds'])
            ep = eps[k]; used = sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']})
            sel = np.isin(ids, used); eids = ids[sel]; eraw = raw[sel]
            comps = {f'{vname}_d{d}': old.component(RUN / f'models/f{k}_{vname}_d{d}', eraw, eids, ep) for d in DIMS}
            np.savez(RUN / f'comps_f{k}_{vname}.npz', **comps)


def rrf(*xs): return sum((1 / len(xs)) / (60 + p.b.ranks(x)) for x in xs)


def variants(k):
    c = {}
    for vname in VIEWS:
        with np.load(RUN / f'comps_f{k}_{vname}.npz') as z: c.update({kk: z[kk] for kk in z.files})
    out = {name: c[name] for name in c}
    for vname in VIEWS: out[f'{vname}_big3'] = rrf(*[c[f'{vname}_d{d}'] for d in DIMS])
    out['v90+v64'] = rrf(*[c[f'{v}_d{d}'] for v in ('v90', 'v64') for d in DIMS])
    out['v90+v64+v8'] = rrf(*[c[f'{v}_d{d}'] for v in ('v90', 'v64', 'v8') for d in DIMS])
    out['v90+cat'] = rrf(*[c[f'{v}_d{d}'] for v in ('v90', 'cat6490') for d in DIMS])
    out['all12'] = rrf(*[c[f'{v}_d{d}'] for v in VIEWS for d in DIMS])
    return out


def report():
    eps = p.b.read(RUN / 'EPISODES.json'); V = [variants(k) for k in range(K)]
    tg = [np.array([z['target'] for z in e['questions']]) for e in eps]; ql = [np.array([z['q'] for z in e['questions']]) for e in eps]
    res = {}; ref = 'v90_big3'
    for name in V[0]:
        raw_sid = np.concatenate([A.sid(V[k][name], tg[k]) for k in range(K)])
        bal = []
        for k in range(K):
            others = [j for j in range(K) if j != k]
            temps = A.calibrate(A.z(np.vstack([V[j][name] for j in others])), np.concatenate([tg[j] for j in others]), np.concatenate([ql[j] for j in others]))
            bal.append(A.sid(A.new_transform(V[k][name], ql[k], temps), tg[k]))
        bal = np.concatenate(bal); res[name] = {'raw': raw_sid, 'bal': bal}
    Q = np.concatenate(ql); rng = np.random.default_rng(0); B = rng.integers(0, len(Q), (2000, len(Q))); out = {}
    for name, r in res.items():
        row = {}
        for mode in ('raw', 'bal'):
            d = r[mode] - res[ref][mode]; bs = d[B].mean(1)
            row[mode] = {'mean': float(r[mode].mean()), 'by_q': {str(n): float(r[mode][Q == n].mean()) for n in (1, 2, 3, 5)}, 'delta_vs_v90_big3': float(d.mean()), 'CI95': np.quantile(bs, [.025, .975]).round(4).tolist()}
        out[name] = row
        print(f"{name:14s} raw {row['raw']['mean']:.4f} d {row['raw']['delta_vs_v90_big3']:+.4f} {row['raw']['CI95']} | bal {row['bal']['mean']:.4f} d {row['bal']['delta_vs_v90_big3']:+.4f} {row['bal']['CI95']} q1 {row['bal']['by_q']['1']:.4f}")
    p.b.write(RUN / 'CROSSFIT_RESULTS.json', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['fit', 'report']); v = ap.parse_args()
    with threadpool_limits(limits=4): {'fit': fit_all, 'report': report}[v.stage]()
