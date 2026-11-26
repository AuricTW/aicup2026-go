"""Fair cross-fit comparison bigbank2 vs bigbank1 on the crossfit90 fresh fold episodes.
Fold k: v1 model = existing crossfit models f{k}_v90_d{D} (v1 players of other folds, <=32 games).
        v2 model = bigbank2 games of owners NOT in v1 fold k, minus any game sharing a family with fold-k episode games.
Usage: run [tag]  (tag 'partial' = only games with features available now, e.g. after chunk0)
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank2_90_20261114_v1 as b2
import crossfit90_20261114_v2 as c2
import numpy as np
from threadpoolctl import threadpool_limits
c1 = c2.c1; bb = c1.bb; p = c1.p; old = c1.old; A = c1.A; K = c1.K
DIMS = (128, 256, 384, 768)


def available(indices):
    have = set()
    for ids in [np.load(b2.s.RUN / 'all_indices.npy'), np.load(bb.RUN / 'dev_results/indices.npy'), np.load(b2.old.RUN / 'dev_results/indices.npy')] + [np.load(d / 'indices.npy') for d in sorted(b2.RUN.glob('results_c*'))]: have.update(ids.tolist())
    return np.array([int(i) in have for i in indices])


def run(tag):
    out_dir = b2.RUN / f'eval_{tag}'; out_dir.mkdir(exist_ok=False); m = p.b.meta(); owners = m['owners'].astype(str)
    eps = p.b.read(c1.RUN / 'EPISODES.json'); ids1 = np.load(bb.RUN / 'train_indices.npy'); lab1 = np.load(bb.RUN / 'train_labels.npy')
    ids2 = np.load(b2.RUN / 'train_indices.npy'); lab2 = np.load(b2.RUN / 'train_labels.npy'); ok = available(ids2); ids2, lab2 = ids2[ok], lab2[ok]
    raw2 = b2.lookup90(ids2); raw1 = bb.lookup_views(ids1, 90); summary = {'games_used': int(ok.sum()), 'games_total': int(len(ok))}
    tg = [np.array([z['target'] for z in e['questions']]) for e in eps]; ql = [np.array([z['q'] for z in e['questions']]) for e in eps]; V = [dict() for _ in range(K)]
    for k in range(K):
        ep = eps[k]; used = sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']})
        heldowners = set(owners[ids1[lab1 % K == k]]); keep = ~np.isin(owners[ids2], sorted(heldowners))
        for f in p.b.FAMILIES: keep &= ~np.isin(m[f][ids2], np.unique(m[f][used]))
        summary[f'fold{k}'] = {'v2_games': int(keep.sum()), 'v2_players': int(len(np.unique(lab2[keep]))), 'v1_games': int((lab1 % K != k).sum()), 'v1_players': int(len(np.unique(lab1[lab1 % K != k])))}
        print(summary[f'fold{k}'])
        sel = np.isin(ids1, used); eraw, eids = raw1[sel], ids1[sel]
        for d in DIMS:
            folder = out_dir / f'models/f{k}_v2_d{d}'; folder.mkdir(parents=True); x = old.pre(raw2[keep], folder, True); p.fit_plda(x, lab2[keep], folder / 'PLDA.npz', dimension=d)
            V[k][f'v2_d{d}'] = old.component(folder, eraw, eids, ep)
            f1 = c1.RUN / f'models/f{k}_v90_d{d}'
            if f1.exists(): V[k][f'v1_d{d}'] = old.component(f1, eraw, eids, ep)
        for v in ('v1', 'v2'):
            V[k][f'{v}_z128_256_384'] = c2.zmean(*[V[k][f'{v}_d{d}'] for d in (128, 256, 384)])
        V[k]['v2_z256_384_768'] = c2.zmean(*[V[k][f'v2_d{d}'] for d in (256, 384, 768)]); V[k]['v2_z128_256_384_768'] = c2.zmean(*[V[k][f'v2_d{d}'] for d in (128, 256, 384, 768)])
        np.savez(out_dir / f'comps_f{k}.npz', **V[k])
    Q = np.concatenate(ql); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    def evals(nm):
        bal = []
        for k in range(K):
            o = [j for j in range(K) if j != k]; temps = A.calibrate(A.z(np.vstack([V[j][nm] for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
            bal.append(A.sid(A.new_transform(V[k][nm], ql[k], temps), tg[k]))
        return np.concatenate([A.sid(V[k][nm], tg[k]) for k in range(K)]), np.concatenate(bal)
    E = {nm: evals(nm) for nm in V[0]}; ref = E['v1_z128_256_384']
    for nm, (r, b) in sorted(E.items(), key=lambda x: -x[1][1].mean()):
        d = b - ref[1]; res[nm] = {'raw': float(r.mean()), 'bal': float(b.mean()), 'bal_by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'bal_delta_vs_v1_z': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist()}
        print(f"{nm:22s} raw {r.mean():.4f} bal {b.mean():.4f} d {d.mean():+.4f} {res[nm]['CI95']} byq", {a: round(v, 3) for a, v in res[nm]['bal_by_q'].items()})
    p.b.write(out_dir / 'RESULT.json', {'summary': summary, 'results': res})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('tag'); v = ap.parse_args()
    with threadpool_limits(limits=4): run(v.tag)
