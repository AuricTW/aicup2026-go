"""Scoring-rule experiments on the official support-split validation (suppsplit_eval_20261115_v1): same frozen v6b head embeddings,
different ways of turning game embeddings into a question x candidate score.  No training.  Each variant replaces 'ens3' in the v6b
neural recipe (0.2 z(base2) + 0.8 z(ens)) and is compared with b2e3 after Sinkhorn (temperatures from the other pool).
Variants
  base        cos(mean of unit query embeddings, mean of unit support embeddings)                        (= v6b)
  colcen      subtract the pool-wide mean embedding of the game's colour (B / W) first
  colmatch_a  a * colour-matched score + (1-a) * base ; colour-matched = mean over query games of cos(game, mean of same-colour supports),
              falling back to all supports when the candidate has no support of that colour
  pair_topk   mean over query games of the mean of the k largest game-pair cosines to the candidate's supports
  gamez       z-normalise each query game's candidate scores (over candidates) before averaging over the query games
Usage: cache | run
"""
from pathlib import Path
import sys, json, os
sys.path.insert(0, str(Path(__file__).resolve().parent))
import suppsplit_eval_20261115_v1 as ss
if os.environ.get('SUPPSPLIT') == 'v2': import suppsplit_eval_20261115_v2      # equal-gallery episodes (sets ss.RUN)
import numpy as np
from threadpoolctl import threadpool_limits
A = ss.A; g = ss.g; p = ss.p; old = ss.old; BW = ss.BW; RUN = ss.RUN


def cache(root=None, name='heademb_b3'):
    games, pools, keys, pos = ss.official(); ids = np.load(old.RUN / 'official_results/indices.npy'); raw = np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r')); rowof = np.zeros(len(keys), int); rowof[ids] = np.arange(len(ids))
    embs = ss.head_embs(ss.b3.RUN / 'export' if root is None else root, raw); np.savez(RUN / f'{name}.npz', **{c: np.stack([e[rowof] for e in embs[c]]).astype(np.float32) for c in embs}, black=np.array([games[k].color == 'B' for k in keys]))


def nrm(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def score(Z, black, ep, kind, par=None):
    """Z: [N, D] game embeddings (pos order) -> [questions, candidates]"""
    if kind == 'colcen': Z = Z - np.where(black[:, None], Z[black].mean(0)[None], Z[~black].mean(0)[None])
    Z = nrm(Z); sup = [np.array(s) for s in ep['supports']]; qs = [np.array(z['indices']) for z in ep['questions']]
    sm = nrm(np.stack([Z[s].mean(0) for s in sup])); qm = nrm(np.stack([Z[q].mean(0) for q in qs])); base = qm @ sm.T
    if kind in ('base', 'colcen'): return base
    if kind == 'gamez':
        return np.stack([A.z(Z[q] @ sm.T).mean(0) for q in qs])
    if kind == 'colmatch':
        smc = {}
        for c in (True, False):
            m = np.stack([Z[s[black[s] == c]].mean(0) if (black[s] == c).any() else Z[s].mean(0) for s in sup]); smc[c] = nrm(m)
        cm = np.stack([np.mean([Z[i] @ smc[bool(black[i])].T for i in q], 0) for q in qs]); return par * A.z(cm) + (1 - par) * A.z(base)
    if kind == 'pair_topk':
        out = np.zeros((len(qs), len(sup)))
        for j, s in enumerate(sup):
            G = Z[s]
            for i, q in enumerate(qs): c = Z[q] @ G.T; out[i, j] = np.sort(c, 1)[:, -min(par, c.shape[1]):].mean()
        return out
    raise ValueError(kind)


def run(name='heademb_b3'):
    E = np.load(RUN / f'{name}.npz'); black = E['black']; cfgs = ('d', 'lin'); data = []; variants = [('base', None), ('colcen', None), ('gamez', None), ('colmatch', .3), ('colmatch', .5), ('colmatch', .7), ('colmatch', 1.0), ('pair_topk', 3), ('pair_topk', 10)]
    for pool, r, ep, f in ss.each():
        V = dict(np.load(f)); tg = np.array([z['target'] for z in ep['questions']]); q = np.array([z['q'] for z in ep['questions']]); S = {}
        for kind, par in variants: S[f'{kind}{"" if par is None else par}'] = g.zm([g.zm([score(E[c][sd].astype(np.float64), black, ep, kind, par) for sd in range(E[c].shape[0])]) for c in cfgs])
        assert np.allclose(S['base'], V['ens3'], atol=1e-4), np.abs(S['base'] - V['ens3']).max(); data.append((pool, V, S, tg, q)); print(pool, r, flush=True)
    Q = np.concatenate([d[4] for d in data]); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    def evals(fn):
        bal = []
        for pool, V, S, tg, q in data:
            other = [d for d in data if d[0] != pool]; T = A.calibrate(A.z(np.vstack([fn(d[1], d[2]) for d in other])), np.concatenate([d[3] for d in other]), np.concatenate([d[4] for d in other])); bal.append(A.sid(A.new_transform(fn(V, S), q, T), tg))
        return np.concatenate(bal)
    ref = evals(lambda V, S: V['b2e3'])
    for nm in data[0][2]:
        for label, fn in ((f'b2+{nm}', lambda V, S, nm=nm: BW * A.z(V['base2']) + (1 - BW) * A.z(S[nm])), (f'{nm} alone', lambda V, S, nm=nm: S[nm])):
            b = evals(fn); d = b - ref; res[label] = {'sinkhorn': float(b.mean()), 'delta_vs_b2e3': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist(), 'by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}}
            print(f"{label:22s} sink {b.mean():.4f} d {d.mean():+.4f} {res[label]['CI95']}", {k: round(v, 3) for k, v in res[label]['by_q'].items()}, flush=True)
    (RUN / f'SCORING_{name}.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        if sys.argv[1] == 'cache': cache(*sys.argv[2:])
        else: run(*sys.argv[2:])
