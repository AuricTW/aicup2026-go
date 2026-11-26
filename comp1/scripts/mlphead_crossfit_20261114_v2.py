"""MLP head grid around the strongly-regularized config 'c' of v1, with seed ensembles.
Same cross-fit folds/episodes as v1. For each cfg: n seeds, cos per seed, ens = zmean(cos seeds); PLDA on seed-0 embeddings.
Usage: run <tag> <cfg1,cfg2,...> <nseeds>
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlphead_crossfit_20261114_v1 as h
import numpy as np
import torch
from threadpoolctl import threadpool_limits
e = h.e; b2 = h.b2; c1 = h.c1; bb = h.bb; p = h.p; old = h.old; A = h.A; K = h.K
BASE = dict(hidden=1024, emb=256, drop=.5, indrop=.3, s=30., m=.20, epochs=20, bs=1024, lr=2e-3, wd=1e-2)
CFGS = {'c': {}, 'c_e10': dict(epochs=10), 'c_e30': dict(epochs=30), 'c_d6': dict(drop=.6, indrop=.4), 'c_in5': dict(indrop=.5), 'c_h2048': dict(hidden=2048),
        'c_h512': dict(hidden=512), 'c_emb128': dict(emb=128), 'c_emb512': dict(emb=512), 'c_m35': dict(m=.35), 'c_m10': dict(m=.10), 'c_wd05': dict(wd=5e-2),
        'd': dict(hidden=2048, emb=512, drop=.5, indrop=.5, s=30., m=.35, epochs=60, bs=512, lr=1e-3, wd=5e-2),
        'lin': dict(hidden=0, emb=512, drop=0., indrop=.1, m=.25, epochs=40, wd=5e-4), 'lin_in3': dict(hidden=0, emb=512, drop=0., indrop=.3, m=.25, epochs=40, wd=5e-4),
        'lin_in5': dict(hidden=0, emb=512, drop=0., indrop=.5, m=.30, epochs=40, wd=5e-4), 'lin_in3_e20': dict(hidden=0, emb=512, drop=0., indrop=.3, m=.25, epochs=20, wd=5e-4),
        'lin_in3_emb1024': dict(hidden=0, emb=1024, drop=0., indrop=.3, m=.25, epochs=40, wd=5e-4),
        'd_h4096': dict(hidden=4096, emb=512, drop=.5, indrop=.5, m=.35, epochs=60, bs=512, lr=1e-3, wd=5e-2),
        'd_e100': dict(hidden=2048, emb=512, drop=.5, indrop=.5, m=.35, epochs=100, bs=512, lr=1e-3, wd=5e-2),
        'd_wd1': dict(hidden=2048, emb=512, drop=.5, indrop=.5, m=.35, epochs=60, bs=512, lr=1e-3, wd=1e-1),
        'd_m45': dict(hidden=2048, emb=512, drop=.5, indrop=.5, m=.45, epochs=60, bs=512, lr=1e-3, wd=5e-2),
        'd_in6_d6': dict(hidden=2048, emb=512, drop=.6, indrop=.6, m=.35, epochs=60, bs=512, lr=1e-3, wd=5e-2),
        'c_in5_d6': dict(indrop=.5, drop=.6), 'c_in5_h2048': dict(indrop=.5, hidden=2048), 'c_in6_h2048_d6': dict(indrop=.6, hidden=2048, drop=.6)}


def zm(xs): return sum(A.z(x) for x in xs) / len(xs)


def run(tag, cfgs, nseeds):
    out = h.RUN / f'grid_{tag}'; out.mkdir(parents=True, exist_ok=False); m = p.b.meta(); owners = m['owners'].astype(str)
    eps = p.b.read(c1.RUN / 'EPISODES.json'); ids1 = np.load(bb.RUN / 'train_indices.npy'); lab1 = np.load(bb.RUN / 'train_labels.npy')
    ids2 = np.load(b2.RUN / 'train_indices.npy'); lab2 = np.load(b2.RUN / 'train_labels.npy'); raw2 = b2.lookup90(ids2); raw1 = bb.lookup_views(ids1, 90)
    tg = [np.array([z['target'] for z in ep['questions']]) for ep in eps]; ql = [np.array([z['q'] for z in ep['questions']]) for ep in eps]
    V = [{'base': dict(np.load(h.BASE / f'comps_f{k}.npz'))[h.BASENAME]} for k in range(K)]
    for k in range(K):
        ep = eps[k]; used = sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']})
        keep = ~np.isin(owners[ids2], sorted(set(owners[ids1[lab1 % K == k]])))
        for f in p.b.FAMILIES: keep &= ~np.isin(m[f][ids2], np.unique(m[f][used]))
        sel = np.isin(ids1, used); eraw, eids = raw1[sel], ids1[sel]; order = np.argsort(eids, kind='stable'); eraw, eids = eraw[order], eids[order]
        folder = out / f'f{k}'; folder.mkdir(); xtr = old.pre(raw2[keep], folder, True); xe = old.pre(eraw, folder)
        for name in cfgs:
            cfg = dict(BASE, **CFGS[name]); coss = []
            for sd in range(nseeds):
                net = h.train_net(xtr, lab2[keep], cfg, 2026111420 + 97 * k + sd); ze = h.embed(net, xe); c, ss, qs = h.setscores(ze, eids, ep); coss.append(c)
                if sd == 0:
                    ztr = h.embed(net, xtr); p.fit_plda(ztr, lab2[keep], folder / f'PLDA_{name}.npz', dimension=cfg['emb']); model = p.PLDA(folder / f'PLDA_{name}.npz'); zz = model.transform(ze)
                    sm, sn = p.summarize(zz, ss); qm, qn = p.summarize(zz, qs); V[k][f'{name}:plda'] = model.score(qm, qn, sm, sn); del ztr
                del net; torch.cuda.empty_cache()
            V[k][f'{name}:cos1'] = coss[0]; V[k][f'{name}:ens'] = zm(coss)
            V[k][f'{name}:base+ens'] = zm([V[k]['base'], V[k][f'{name}:ens']]); V[k][f'{name}:base+ens+plda'] = zm([V[k]['base'], V[k][f'{name}:ens'], V[k][f'{name}:plda']])
            print(k, name, 'done', flush=True)
        np.savez(out / f'comps_f{k}.npz', **V[k])
    Q = np.concatenate(ql); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    def evals(nm):
        bal = []
        for k in range(K):
            o = [j for j in range(K) if j != k]; temps = A.calibrate(A.z(np.vstack([V[j][nm] for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
            bal.append(A.sid(A.new_transform(V[k][nm], ql[k], temps), tg[k]))
        return np.concatenate([A.sid(V[k][nm], tg[k]) for k in range(K)]), np.concatenate(bal)
    E = {nm: evals(nm) for nm in V[0]}; ref = E['base']
    for nm, (r, b) in sorted(E.items(), key=lambda x: -x[1][1].mean()):
        d = b - ref[1]; res[nm] = {'raw': float(r.mean()), 'bal': float(b.mean()), 'bal_by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'bal_delta_vs_base': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist()}
        print(f"{nm:28s} raw {r.mean():.4f} bal {b.mean():.4f} d {d.mean():+.4f} {res[nm]['CI95']} byq", {a: round(v, 3) for a, v in res[nm]['bal_by_q'].items()})
    p.b.write(out / 'RESULT.json', {'base_cfg': BASE, 'cfgs': {n: CFGS[n] for n in cfgs}, 'nseeds': nseeds, 'results': res})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('tag'); ap.add_argument('cfgs'); ap.add_argument('nseeds', type=int); v = ap.parse_args()
    with threadpool_limits(limits=4): run(v.tag, v.cfgs.split(','), v.nseeds)
