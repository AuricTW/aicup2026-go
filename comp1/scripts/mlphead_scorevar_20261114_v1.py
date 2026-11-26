"""Offline scoring variants on cached MLP-head embeddings (mlphead_embcache_20261114_v1).
Fusion identical to v5: zm([base, zm([zm(d seeds), zm(lin seeds)])]); only the per-seed cosine scoring changes.
Variants: plain | ctr (train-mean centering) | ectr (episode-mean centering) | wht (shrunk PCA whitening from train emb)
          | asn (adaptive s-norm vs training-class cohort) | pair (mean pairwise game cosine) | combos.
Usage: eval <split> [variants comma]
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlphead_crossfit_20261114_v2 as g
import mlphead_confirm_20261114_v1 as cf
import numpy as np
h = g.h; p = g.p; A = g.A; K = 3; CFGS = ('d', 'lin')


def nrm(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)


def sets(ep, eids):
    pos = {int(i): j for j, i in enumerate(eids)}
    return [[pos[i] for i in s] for s in ep['supports']], [[pos[i] for i in q['indices']] for q in ep['questions']]


def score(variant, ze, ztr, ytr, ss, qs):
    z = ze.astype(np.float64); t = ztr.astype(np.float64)
    if 'wht' in variant:
        mu = t.mean(0); C = np.cov((t - mu).T); w, U = np.linalg.eigh(C); w = np.maximum(w, 1e-9); a = .5 if 'wht5' in variant else 1.
        W = U / (w + a * w.mean()) ** .5; z = (z - mu) @ W; t = (t - mu) @ W
    elif 'ectr' in variant: z = z - z.mean(0); t = t - ze.mean(0)
    elif 'ctr' in variant: mu = t.mean(0); z = z - mu; t = t - mu
    zn = nrm(z)
    sm = nrm(np.stack([zn[s].mean(0) for s in ss])); qm = nrm(np.stack([zn[q].mean(0) for q in qs]))
    if 'pair' in variant: S = np.stack([(zn[q] @ zn[np.concatenate(ss)].T).mean(0) for q in qs]); S = S.reshape(len(qs), len(ss), -1).mean(2) if all(len(s) == len(ss[0]) for s in ss) else None
    else: S = qm @ sm.T
    if 'asn' in variant:
        tn = nrm(t); labs = np.unique(ytr); cm = nrm(np.stack([tn[ytr == l].mean(0) for l in labs])); top = 300
        cq = np.sort(qm @ cm.T, axis=1)[:, -top:]; cs = np.sort(sm @ cm.T, axis=1)[:, -top:]
        S = .5 * ((S - cq.mean(1)[:, None]) / cq.std(1)[:, None] + (S - cs.mean(1)[None]) / cs.std(1)[None])
    return S


def run(split, variants):
    if split == 'g1': eps = p.b.read(h.c1.RUN / 'EPISODES.json'); base = [dict(np.load(h.BASE / f'comps_f{k}.npz'))[h.BASENAME] for k in range(K)]
    else: eps = p.b.read(cf.RUN / 'EPISODES.json'); base = [dict(np.load(cf.RUN / f'c1/comps_f{k}.npz'))['base'] for k in range(K)]
    d = h.RUN / f'embcache_{split}'; tg = [np.array([z['target'] for z in ep['questions']]) for ep in eps]; ql = [np.array([z['q'] for z in ep['questions']]) for ep in eps]
    V = [{'base': base[k]} for k in range(K)]
    for k in range(K):
        eids = np.load(d / f'f{k}_eids.npy'); ss, qs = sets(eps[k], eids); files = {c: sorted(d.glob(f'f{k}_{c}_s*.npz')) for c in CFGS}
        data = {c: [dict(np.load(f)) for f in fs] for c, fs in files.items()}
        for v in variants:
            enss = [g.zm([score(v, x['ze'], x['ztr'], x['ytr'], ss, qs) for x in data[c]]) for c in CFGS]
            V[k][f'{v}:base+ens'] = g.zm([base[k], g.zm(enss)]); V[k][f'{v}:ens'] = g.zm(enss)
            if v == 'plain':
                for w in (.1, .2, .3): V[k][f'plain:bw{w}'] = w * A.z(base[k]) + (1 - w) * A.z(g.zm(enss))
                V[k]['plain:d_only'] = enss[0]; V[k]['plain:lin_only'] = enss[1]; V[k]['plain:bw.2_d.6'] = .2 * A.z(base[k]) + .48 * A.z(enss[0]) + .32 * A.z(enss[1])
        print(k, {c: len(v) for c, v in files.items()}, flush=True)
    Q = np.concatenate(ql); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    def evals(nm):
        bal = []
        for k in range(K):
            o = [j for j in range(K) if j != k]; T = A.calibrate(A.z(np.vstack([V[j][nm] for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
            bal.append(A.sid(A.new_transform(V[k][nm], ql[k], T), tg[k]))
        return np.concatenate(bal)
    E = {nm: evals(nm) for nm in V[0]}; ref = E['plain:base+ens'] if 'plain:base+ens' in E else E['base']
    for nm, b in sorted(E.items(), key=lambda x: -x[1].mean()):
        dd = b - ref; res[nm] = {'bal': float(b.mean()), 'bal_by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'delta_vs_plain': float(dd.mean()), 'CI95': np.quantile(dd[B].mean(1), [.025, .975]).round(4).tolist()}
        print(f"{nm:20s} bal {b.mean():.4f} d {dd.mean():+.4f} {res[nm]['CI95']}", {a: round(v, 3) for a, v in res[nm]['bal_by_q'].items()})
    p.b.write(d / f'SCOREVAR_{"_".join(variants)}.json', {'split': split, 'results': res})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('split'); ap.add_argument('variants', nargs='?', default='plain,ctr,ectr,wht,wht5,asn,ctr_asn,ectr_asn'); v = ap.parse_args()
    run(v.split, v.variants.split(','))
