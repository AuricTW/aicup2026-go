"""Cross-fit evaluation of bigbank3 + bigbank4 (new players: 723 'hist' fit owners + old 'validation' split owners) as the training bank
for the v6b recipe components: MLP heads (d, lin; 2 seeds) and optionally PLDA6.  Same episodes/folds as bigbank3_eval (g1, confirm).
bigbank4 players are never episode players; they are added to every fold's training set (family exclusion against episode games still applies).
Uses whatever bigbank4 chunks have been accepted so far (recorded in RESULT.json).
Usage: run <split> <tag> [nseeds] [--plda]   -> runs/bigbank4_90_20261115_v1/eval_<split>_<tag>/
Reference keys come from bigbank3 eval comps: base2, ens3, b2e3 (= v6b recipe).
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank3_eval_20261114_v1 as ev3
import bigbank4_90_20261115_v1 as b4
import numpy as np
import torch
from threadpoolctl import threadpool_limits
g = ev3.g; h = ev3.h; b2 = ev3.b2; bb = ev3.bb; p = ev3.p; old = ev3.old; A = ev3.A; cf = ev3.cf; sv = ev3.sv; b3 = ev3.b3; K = 3; DIMS = ev3.DIMS; CFGS = ev3.CFGS; BW = ev3.BW


def bank45(use5):
    """bigbank4 (+ bigbank5 = holdout 840 + sealed 400, user-authorised for training 2026-11-15): indices, owners, features, chunk names, universe"""
    if not use5: return bank4() + (b4.Universe(),)
    import bigbank5_90_20261115_v1 as b5
    a = bank4(); b = bank4(b5.RUN); return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), np.concatenate([a[2], b[2]]), a[3] + ['b5_' + c for c in b[3]], b5.Universe()


def bank4(run=None):
    """accepted bigbank4 games: indices, owner strings, features"""
    run = b4.RUN if run is None else run
    ids = np.load(run / 'train_indices.npy'); own = np.load(run / 'train_owners.npy').astype(str); chunks = sorted(run.glob('results_c*')); pos = {}; arrs = []
    for n, d in enumerate(chunks):
        arrs.append(np.load(d / 'view90.npy', mmap_mode='r'))
        for j, i in enumerate(np.load(d / 'indices.npy')): pos[int(i)] = (n, j)
    have = np.array([int(i) in pos for i in ids]); ids = ids[have]; own = own[have]; raw = np.empty((len(ids), 2339), np.float32)
    for k, i in enumerate(ids): n, j = pos[int(i)]; raw[k] = arrs[n][j]
    return ids, own, raw, [c.name for c in chunks]


def run(split, tag, nseeds, plda, use5=False):
    out = b4.RUN / f'eval_{split}_{tag}'; out.mkdir(exist_ok=False); m = p.b.meta(); owners = m['owners'].astype(str)
    ids3 = np.load(b3.RUN / 'train_indices.npy'); raw3 = b3.lookup90(ids3); o3 = owners[ids3]; ids4, o4, raw4, chunks, U = bank45(use5); print('bank4/5 games', len(ids4), 'owners', len(set(o4)), chunks, flush=True)
    assert not set(o4) & set(o3); ids2 = np.load(b2.RUN / 'train_indices.npy'); lab2 = np.load(b2.RUN / 'train_labels.npy'); pos2 = {int(i): j for j, i in enumerate(ids2)}
    if split == 'g1':
        eps = p.b.read(h.c1.RUN / 'EPISODES.json'); ids1 = np.load(bb.RUN / 'train_indices.npy'); lab1 = np.load(bb.RUN / 'train_labels.npy')
        drop = [set(owners[ids1[lab1 % K == k]]) for k in range(K)]; rawof = lambda e: bb.lookup_views(e, 90)
    else:
        eps = p.b.read(cf.RUN / 'EPISODES.json'); _, fold = cf.episodes(ids2, lab2); o2 = owners[ids2]
        drop = [set(o2[np.array([fold[int(l)] == k for l in lab2])]) for k in range(K)]; raw2 = b2.lookup90(ids2); rawof = lambda e: raw2[[pos2[int(i)] for i in e]]
    tg = [np.array([z['target'] for z in ep['questions']]) for ep in eps]; ql = [np.array([z['q'] for z in ep['questions']]) for ep in eps]; V = [dict() for _ in range(K)]; info = {}
    fam4 = {f: U.get(f, ids4) for f in p.b.FAMILIES}
    for k in range(K):
        ep = eps[k]; used = sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']})
        keep3 = ~np.isin(o3, sorted(drop[k])); keep4 = ~np.isin(o4, sorted(drop[k]))
        for f in p.b.FAMILIES: uf = np.unique(m[f][used]); keep3 &= ~np.isin(m[f][ids3], uf); keep4 &= ~np.isin(fam4[f], uf)
        assert not (set(o3[keep3]) | set(o4[keep4])) & set(owners[np.array(used)])
        eids = np.array(sorted(used)); assert np.array_equal(eids, np.load(h.RUN / f'embcache_{split}/f{k}_eids.npy')); eraw = rawof(eids)
        otr = np.concatenate([o3[keep3], o4[keep4]]); _, ytr = np.unique(otr, return_inverse=True); rawtr = np.concatenate([raw3[keep3], raw4[keep4]])
        info[k] = {'train_games': int(len(ytr)), 'players': int(ytr.max() + 1), 'bank3_games': int(keep3.sum()), 'bank4_games': int(keep4.sum())}; print(k, info[k], flush=True)
        folder = out / f'f{k}'; folder.mkdir(); xtr = old.pre(rawtr, folder, True); xe = old.pre(eraw, folder); del rawtr
        R = dict(np.load(b3.RUN / f'eval_{split}/comps_f{k}.npz')); V[k].update({n: R[n] for n in ('base2', 'ens3', 'b2e3')})
        if plda:
            for d in DIMS:
                mf = folder / f'plda{d}'; mf.mkdir(); (mf / 'preprocess.npz').write_bytes((folder / 'preprocess.npz').read_bytes())
                p.fit_plda(xtr, ytr, mf / 'PLDA.npz', dimension=d); V[k][f'plda{d}'] = old.component(mf, eraw, eids, ep)
            V[k]['base4'] = g.zm([V[k][f'plda{d}'] for d in DIMS]); print(k, 'plda done', flush=True)
        E4 = {}
        for c in CFGS:
            cfg = dict(g.BASE, **g.CFGS[c]); coss = []
            for sd in range(nseeds):
                net = h.train_net(xtr, ytr, cfg, 2026111420 + 97 * k + sd); ze = h.embed(net, xe); np.save(folder / f'ze_{c}_s{sd}.npy', ze.astype(np.float32)); coss.append(h.setscores(ze, eids, ep)[0]); del net; torch.cuda.empty_cache()
            E4[c] = g.zm(coss); V[k][f'{c}4'] = E4[c]
        n4 = g.zm([E4[c] for c in CFGS]); V[k]['ens4'] = n4; V[k]['b2e4'] = BW * A.z(V[k]['base2']) + (1 - BW) * A.z(n4); V[k]['b2e34'] = BW * A.z(V[k]['base2']) + (1 - BW) * A.z(g.zm([V[k]['ens3'], n4]))
        if plda:
            V[k]['b4e4'] = BW * A.z(V[k]['base4']) + (1 - BW) * A.z(n4); V[k]['b4e3'] = BW * A.z(V[k]['base4']) + (1 - BW) * A.z(V[k]['ens3'])
            for w in (.1, .3, .4): V[k][f'b4e4_bw{w}'] = w * A.z(V[k]['base4']) + (1 - w) * A.z(n4)
        np.savez(out / f'comps_f{k}.npz', **V[k]); print(k, 'done', flush=True)
    Q = np.concatenate(ql); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    rk = np.concatenate([[m['ranks'][z['indices'][0]] for z in ep['questions']] for ep in eps]); ur, ui, uc = np.unique(rk, return_inverse=True, return_counts=True); W = (1 / len(ur)) / uc[ui]
    def evals(nm):
        bal = []
        for k in range(K):
            o = [j for j in range(K) if j != k]; T = A.calibrate(A.z(np.vstack([V[j][nm] for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
            bal.append(A.sid(A.new_transform(V[k][nm], ql[k], T), tg[k]))
        return np.concatenate(bal)
    E = {nm: evals(nm) for nm in V[0] if not nm.startswith('plda')}; ref = E['b2e3']
    for nm, b in sorted(E.items(), key=lambda x: -x[1].mean()):
        d = b - ref; res[nm] = {'bal': float(b.mean()), 'bal_rank_uniform': float((b * W).sum()), 'bal_by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'delta_vs_b2e3': float(d.mean()), 'delta_rank_uniform': float((d * W).sum()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist()}
        print(f"{nm:10s} bal {b.mean():.4f} d {d.mean():+.4f} {res[nm]['CI95']} ru {(b * W).sum():.4f} d {(d * W).sum():+.4f}", {a: round(v, 3) for a, v in res[nm]['bal_by_q'].items()})
    p.b.write(out / 'RESULT.json', {'split': split, 'nseeds': nseeds, 'BW': BW, 'bank4_chunks': chunks, 'info': info, 'results': res, 'note': 'e4/base4 = trained on bigbank3 + accepted bigbank4 games; ref b2e3 = v6b recipe; ru = rank-uniform weighted targets'})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('split'); ap.add_argument('tag'); ap.add_argument('nseeds', nargs='?', type=int, default=2); ap.add_argument('--plda', action='store_true'); ap.add_argument('--bank5', action='store_true'); v = ap.parse_args()
    with threadpool_limits(limits=4): run(v.split, v.tag, v.nseeds, v.plda, v.bank5)
