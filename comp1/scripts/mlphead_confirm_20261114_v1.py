"""Confirmation split for MLP-head choices made on the crossfit90 v1 folds (grid g1).
New 3-fold partition over ALL bigbank2 players by a NEW hash seed; new official-like episodes per fold
(600 candidates x 10 support, 400 distinct targets, 100 each of 1/2/3/5 query games).
Fold-k training = bigbank2 games of players not in fold k, minus any game sharing a family with fold-k episode games.
Baseline = zmean of PLDA d128..1024 (current production neural). Candidates = given MLP cfgs (seed ensembles), fused with baseline.
Usage: run <tag> <cfg1,...> <nseeds>
"""
from pathlib import Path
import sys, argparse, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlphead_crossfit_20261114_v2 as g
import numpy as np
import torch
from threadpoolctl import threadpool_limits
h = g.h; b2 = g.b2; p = g.p; old = g.old; A = g.A; K = 3
SEED = 2026111499; DIMS = (128, 256, 384, 512, 768, 1024); RUN = h.RUN / 'confirm'


def hh(v): return hashlib.sha256((str(SEED) + '|' + str(v)).encode()).digest()


def episodes(ids, lab):
    used_targets = {int(q['player']) for e in p.b.read(h.c1.RUN / 'EPISODES.json') for q in e['questions']}  # targets already read on crossfit90 v1 -> distractor-only here
    eps = []; fold = {int(l): int.from_bytes(hh(l)[:4], 'big') % K for l in np.unique(lab)}
    for k in range(K):
        players = sorted([l for l, f in fold.items() if f == k], key=hh); games = {l: sorted(ids[lab == l].tolist(), key=hh) for l in players}
        cands = [l for l in players if len(games[l]) >= 10][:600]; assert len(cands) == 600
        rng = np.random.default_rng(SEED + k); order = list(rng.permutation(cands)); tq = {}
        for n in (5, 3, 2, 1):
            got = 0
            for l in list(order):
                if got == 100: break
                if len(games[l]) >= 10 + n and int(l) not in used_targets: tq[int(l)] = n; order.remove(l); got += 1
            assert got == 100
        sup = [games[l][:10] for l in cands]; pos = {l: j for j, l in enumerate(cands)}
        qs = [{'id': f'x{k}_{l}', 'indices': games[l][10:10 + n], 'q': n, 'target': pos[l], 'player': str(l)} for l, n in sorted(tq.items(), key=lambda x: hh(x[0]))]
        eps.append({'fold': k, 'candidates': [str(c) for c in cands], 'supports': sup, 'questions': qs})
    return eps, fold


def run(tag, cfgs, nseeds):
    out = RUN / tag; out.mkdir(parents=True, exist_ok=False); m = p.b.meta()
    ids2 = np.load(b2.RUN / 'train_indices.npy'); lab2 = np.load(b2.RUN / 'train_labels.npy'); raw2 = b2.lookup90(ids2)
    eps, fold = episodes(ids2, lab2)
    if not (RUN / 'EPISODES.json').exists(): p.b.write(RUN / 'EPISODES.json', eps)
    eps = p.b.read(RUN / 'EPISODES.json')
    pf = np.array([fold[int(l)] for l in lab2]); tg = [np.array([z['target'] for z in ep['questions']]) for ep in eps]; ql = [np.array([z['q'] for z in ep['questions']]) for ep in eps]; V = [dict() for _ in range(K)]
    pos2 = {int(i): j for j, i in enumerate(ids2)}
    for k in range(K):
        ep = eps[k]; used = sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']})
        keep = pf != k
        for f in p.b.FAMILIES: keep &= ~np.isin(m[f][ids2], np.unique(m[f][used]))
        eids = np.array(sorted(used)); eraw = raw2[[pos2[i] for i in eids]]
        print(f'fold {k}: train games {keep.sum()} players {len(np.unique(lab2[keep]))}', flush=True)
        folder = out / f'f{k}'; folder.mkdir(); xtr = old.pre(raw2[keep], folder, True); xe = old.pre(eraw, folder)
        for d in DIMS:
            mf = folder / f'plda{d}'; mf.mkdir(); (mf / 'preprocess.npz').write_bytes((folder / 'preprocess.npz').read_bytes())
            p.fit_plda(xtr, lab2[keep], mf / 'PLDA.npz', dimension=d); V[k][f'plda{d}'] = old.component(mf, eraw, eids, ep)
        V[k]['base'] = g.zm([V[k][f'plda{d}'] for d in DIMS])
        for name in cfgs:
            cfg = dict(g.BASE, **g.CFGS[name]); coss = []
            for sd in range(nseeds):
                net = h.train_net(xtr, lab2[keep], cfg, 2026111450 + 97 * k + sd); ze = h.embed(net, xe); c, ss, qs = h.setscores(ze, eids, ep); coss.append(c); del net; torch.cuda.empty_cache()
            V[k][f'{name}:cos1'] = coss[0]; V[k][f'{name}:ens'] = g.zm(coss); V[k][f'{name}:base+ens'] = g.zm([V[k]['base'], V[k][f'{name}:ens']])
            print(k, name, 'done', flush=True)
        if len(cfgs) > 1: V[k]['allens'] = g.zm([V[k][f'{n}:ens'] for n in cfgs]); V[k]['base+allens'] = g.zm([V[k]['base'], V[k]['allens']])
        np.savez(out / f'comps_f{k}.npz', **V[k])
    Q = np.concatenate(ql); B = np.random.default_rng(1).integers(0, len(Q), (2000, len(Q))); res = {}
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
    p.b.write(out / 'RESULT.json', {'cfgs': {n: g.CFGS[n] for n in cfgs}, 'nseeds': nseeds, 'results': res})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('tag'); ap.add_argument('cfgs'); ap.add_argument('nseeds', type=int); v = ap.parse_args()
    with threadpool_limits(limits=4): run(v.tag, v.cfgs.split(','), v.nseeds)
