"""Cross-fit evaluation of bigbank3 (<=160 games/owner, 234k games) vs bigbank2 (<=64, 158k) as the TRAINING bank,
for both components of the v6a recipe: PLDA6 z-mean base and MLP heads (d, lin; 2 seeds, same seeds as embcache).
Same episodes as before (g1 = crossfit90 v1; confirm = fresh-target split). Fold exclusion is by OWNER (labels differ between banks)
plus family exclusion against episode games. bigbank2 references come from saved comps / embcache.
Usage: run <split> [nseeds]   -> runs/bigbank3_90_20261114_v1/eval_<split>/
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlphead_crossfit_20261114_v2 as g
import mlphead_confirm_20261114_v1 as cf
import mlphead_scorevar_20261114_v1 as sv
import numpy as np
import torch
from threadpoolctl import threadpool_limits
h = g.h; b2 = g.b2; bb = g.bb; p = g.p; old = g.old; A = g.A; K = 3; DIMS = cf.DIMS; CFGS = ('d', 'lin'); BW = .2
# NOTE: do not import bigbank3_90_20261114_v1 here: it patches bigbank2 module globals (RUN) for its prepare/package stages.


class b3:
    RUN = p.ROOT / 'runs/bigbank3_90_20261114_v1'

    @staticmethod
    def lookup90(indices):
        srcs = [(np.load(b2.s.RUN / 'all_indices.npy'), np.load(b2.s.RUN / 'views/view90.npy', mmap_mode='r'))]
        for d in [bb.RUN / 'dev_results', b2.old.RUN / 'dev_results'] + sorted(b2.RUN.glob('results_c*')) + sorted(b3.RUN.glob('results_c*')): srcs.append((np.load(d / 'indices.npy'), np.load(d / 'view90.npy', mmap_mode='r')))
        pos = {}
        for n, (ids, _) in enumerate(srcs):
            for j, i in enumerate(ids): pos.setdefault(int(i), (n, j))
        miss = [int(i) for i in indices if int(i) not in pos]
        if miss: raise KeyError(f'{len(miss)} games without features')
        out = np.empty((len(indices), 2339), np.float32)
        for k, i in enumerate(indices): n, j = pos[int(i)]; out[k] = srcs[n][1][j]
        return out


def run(split, nseeds):
    out = b3.RUN / f'eval_{split}'; out.mkdir(exist_ok=False); m = p.b.meta(); owners = m['owners'].astype(str)
    ids3 = np.load(b3.RUN / 'train_indices.npy'); lab3 = np.load(b3.RUN / 'train_labels.npy'); raw3 = b3.lookup90(ids3); o3 = owners[ids3]
    ids2 = np.load(b2.RUN / 'train_indices.npy'); lab2 = np.load(b2.RUN / 'train_labels.npy'); pos2 = {int(i): j for j, i in enumerate(ids2)}
    if split == 'g1':
        eps = p.b.read(h.c1.RUN / 'EPISODES.json'); ids1 = np.load(bb.RUN / 'train_indices.npy'); lab1 = np.load(bb.RUN / 'train_labels.npy')
        drop = [set(owners[ids1[lab1 % K == k]]) for k in range(K)]; rawof = lambda e: bb.lookup_views(e, 90)
        base2 = [dict(np.load(h.BASE / f'comps_f{k}.npz'))[h.BASENAME] for k in range(K)]
    else:
        eps = p.b.read(cf.RUN / 'EPISODES.json'); _, fold = cf.episodes(ids2, lab2); o2 = owners[ids2]
        drop = [set(o2[np.array([fold[int(l)] == k for l in lab2])]) for k in range(K)]; raw2 = b2.lookup90(ids2); rawof = lambda e: raw2[[pos2[int(i)] for i in e]]
        base2 = [dict(np.load(cf.RUN / f'c1/comps_f{k}.npz'))['base'] for k in range(K)]
    tg = [np.array([z['target'] for z in ep['questions']]) for ep in eps]; ql = [np.array([z['q'] for z in ep['questions']]) for ep in eps]; V = [dict() for _ in range(K)]; info = {}
    for k in range(K):
        ep = eps[k]; used = sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']})
        keep = ~np.isin(o3, sorted(drop[k]))
        for f in p.b.FAMILIES: keep &= ~np.isin(m[f][ids3], np.unique(m[f][used]))
        assert not set(o3[keep]) & set(owners[np.array(used)])
        eids = np.array(sorted(used)); assert np.array_equal(eids, np.load(h.RUN / f'embcache_{split}/f{k}_eids.npy')); eraw = rawof(eids); ss, qs = sv.sets(ep, eids)
        info[k] = {'train_games': int(keep.sum()), 'players': int(len(np.unique(lab3[keep])))}; print(k, info[k], flush=True)
        folder = out / f'f{k}'; folder.mkdir(); xtr = old.pre(raw3[keep], folder, True); xe = old.pre(eraw, folder)
        for d in DIMS:
            mf = folder / f'plda{d}'; mf.mkdir(); (mf / 'preprocess.npz').write_bytes((folder / 'preprocess.npz').read_bytes())
            p.fit_plda(xtr, lab3[keep], mf / 'PLDA.npz', dimension=d); V[k][f'plda{d}'] = old.component(mf, eraw, eids, ep)
        V[k]['base3'] = g.zm([V[k][f'plda{d}'] for d in DIMS]); V[k]['base2'] = base2[k]; print(k, 'plda done', flush=True)
        E3, E2 = {}, {}
        for c in CFGS:
            cfg = dict(g.BASE, **g.CFGS[c]); coss = []
            for sd in range(nseeds):
                net = h.train_net(xtr, lab3[keep], cfg, 2026111420 + 97 * k + sd); ze = h.embed(net, xe); np.save(folder / f'ze_{c}_s{sd}.npy', ze.astype(np.float32)); coss.append(h.setscores(ze, eids, ep)[0]); del net; torch.cuda.empty_cache()
            E3[c] = g.zm(coss)
            E2[c] = g.zm([sv.score('plain', dict(np.load(f))['ze'], dict(np.load(f))['ztr'], None, ss, qs) for f in sorted((h.RUN / f'embcache_{split}').glob(f'f{k}_{c}_s*.npz'))[:nseeds]])
        n3 = g.zm([E3[c] for c in CFGS]); n2 = g.zm([E2[c] for c in CFGS]); V[k]['ens3'] = n3; V[k]['ens2'] = n2
        for bn in ('2', '3'):
            for en, ev in (('2', n2), ('3', n3)): V[k][f'b{bn}e{en}'] = BW * A.z(V[k][f'base{bn}']) + (1 - BW) * A.z(ev)
        V[k]['b3e23'] = BW * A.z(V[k]['base3']) + (1 - BW) * A.z(g.zm([n2, n3]))
        np.savez(out / f'comps_f{k}.npz', **V[k]); print(k, 'done', flush=True)
    Q = np.concatenate(ql); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    def evals(nm):
        bal = []
        for k in range(K):
            o = [j for j in range(K) if j != k]; T = A.calibrate(A.z(np.vstack([V[j][nm] for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
            bal.append(A.sid(A.new_transform(V[k][nm], ql[k], T), tg[k]))
        return np.concatenate(bal)
    E = {nm: evals(nm) for nm in V[0] if not nm.startswith('plda')}; ref = E['b2e2']
    for nm, b in sorted(E.items(), key=lambda x: -x[1].mean()):
        d = b - ref; res[nm] = {'bal': float(b.mean()), 'bal_by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'delta_vs_b2e2': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist()}
        print(f"{nm:8s} bal {b.mean():.4f} d {d.mean():+.4f} {res[nm]['CI95']}", {a: round(v, 3) for a, v in res[nm]['bal_by_q'].items()})
    p.b.write(out / 'RESULT.json', {'split': split, 'nseeds': nseeds, 'BW': BW, 'info': info, 'results': res, 'note': 'b{X}e{Y} = BW*z(PLDA6 of bank X) + (1-BW)*z(zm(d,lin) of bank Y); ref b2e2 = v6a recipe'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('split'); ap.add_argument('nseeds', nargs='?', type=int, default=2); v = ap.parse_args()
    with threadpool_limits(limits=4): run(v.split, v.nseeds)
