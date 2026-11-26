"""Cross-fit v2 (same fresh fold episodes as v1): smaller PLDA dims, score-level vs RRF fusion, data-scaling slope.
  dims: v90 d64/d128/d192/d384 (+ v1 d256/d768/d1024)
  scaling: d256 fit on ONE other fold only (~700 players) vs both other folds (~1400) -> slope per doubling
  fusion: rrf vs zmean (mean of per-row z-scored PLDA log-LR) over dim sets
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crossfit90_20261114_v1 as c1
import numpy as np
from threadpoolctl import threadpool_limits
bb = c1.bb; p = c1.p; old = c1.old; A = c1.A; K = c1.K
RUN = c1.RUN; NEWD = (64, 128, 192, 384)


def fit():
    eps = p.b.read(RUN / 'EPISODES.json'); ids = np.load(bb.RUN / 'train_indices.npy'); lab = np.load(bb.RUN / 'train_labels.npy'); raw = bb.lookup_views(ids, 90)
    for k in range(K):
        ep = eps[k]; used = sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']}); sel = np.isin(ids, used); comps = {}
        jobs = [(f'f{k}_v90_d{d}', lab % K != k, d) for d in NEWD] + [(f'f{k}_half_v90_d256', lab % K == (k + 1) % K, 256), (f'f{k}_half2_v90_d256', lab % K == (k + 2) % K, 256)]
        for name, tr, d in jobs:
            folder = RUN / 'models' / name
            if not folder.exists():
                folder.mkdir(parents=True); x = old.pre(raw[tr], folder, True); r = p.fit_plda(x, lab[tr], folder / 'PLDA.npz', dimension=d); print(name, r['players'], r['seconds'])
            comps[name.split('_', 1)[1]] = old.component(folder, raw[sel], ids[sel], ep)
        np.savez(RUN / f'comps2_f{k}.npz', **comps)


def zmean(*xs): return sum(A.z(x) for x in xs) / len(xs)


def report():
    eps = p.b.read(RUN / 'EPISODES.json'); tg = [np.array([z['target'] for z in e['questions']]) for e in eps]; ql = [np.array([z['q'] for z in e['questions']]) for e in eps]
    V = []
    for k in range(K):
        c = {}
        with np.load(RUN / f'comps_f{k}_v90.npz') as z: c.update({kk: z[kk] for kk in z.files})
        with np.load(RUN / f'comps2_f{k}.npz') as z: c.update({kk: z[kk] for kk in z.files})
        v = {kk: c[kk] for kk in c}
        for nm, ds in {'256_768_1024': (256, 768, 1024), '128_256_384': (128, 256, 384), '192_256_384': (192, 256, 384), '128_256': (128, 256), '256_384': (256, 384), '64_128_256_384': (64, 128, 256, 384)}.items():
            v[f'rrf_{nm}'] = c1.rrf(*[c[f'v90_d{d}'] for d in ds]); v[f'z_{nm}'] = zmean(*[c[f'v90_d{d}'] for d in ds])
        v['half_avg_d256'] = None
        V.append(v)
    ref = 'rrf_256_768_1024'; res = {}
    for name in V[0]:
        if name == 'half_avg_d256': continue
        raw_sid = np.concatenate([A.sid(V[k][name], tg[k]) for k in range(K)]); bal = []
        for k in range(K):
            o = [j for j in range(K) if j != k]
            temps = A.calibrate(A.z(np.vstack([V[j][name] for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
            bal.append(A.sid(A.new_transform(V[k][name], ql[k], temps), tg[k]))
        res[name] = {'raw': raw_sid, 'bal': np.concatenate(bal)}
    Q = np.concatenate(ql); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); out = {}
    for name, r in sorted(res.items(), key=lambda x: -x[1]['bal'].mean()):
        row = {}
        for mode in ('raw', 'bal'):
            d = r[mode] - res[ref][mode]; row[mode] = {'mean': float(r[mode].mean()), 'by_q': {str(n): float(r[mode][Q == n].mean()) for n in (1, 2, 3, 5)}, 'delta': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist()}
        out[name] = row
        print(f"{name:20s} raw {row['raw']['mean']:.4f} d {row['raw']['delta']:+.4f} {row['raw']['CI95']} | bal {row['bal']['mean']:.4f} d {row['bal']['delta']:+.4f} {row['bal']['CI95']} | bal byq", {a: round(b, 3) for a, b in row['bal']['by_q'].items()})
    p.b.write(RUN / 'CROSSFIT2_RESULTS.json', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['fit', 'report']); v = ap.parse_args()
    with threadpool_limits(limits=4): {'fit': fit, 'report': report}[v.stage]()
