"""Colour-aware LISTWISE fusion of several embedding models (generalises pairscore_20261115_v2.py, which reached +0.035 over the v6b
neural recipe with bank3 heads + the board seq-id net, against +0.024 for the plain z-score fusion of the same models).
  score(game i, candidate j) = sum_k [ a_k z_all_k + b_k[n_same] z_same_k + c_k[n_diff] z_diff_k ]
  k = model group (mean cosine over the group's embeddings); z_* = cosine of the game with candidate j's centroid of ALL / SAME-colour /
  OTHER-colour gallery games, z-scored over the candidates; b, c free per number of gallery games of that colour (0 -> term absent).
  Question score = mean over the question's games.  Fit: multinomial logit over the 600 candidates (Adam, L2), labels = candidate ids of
  the gallery games that ship with the test files (support-split; no question labels, no answer key).
Model groups:  <group>=<src>+<src>,<group>=...   sources in REG (head embeddings in official pos order, seq-id off_own.npy)
  a group written <group>*=... gets two extra nearest-neighbour terms: mean of the 2 highest cosines between the game and the candidate's
  single gallery games (all colours / same colour only), z-scored over the candidates -- a game can match one particular gallery game.
Stages
  run <tag> <groups> [draws]     validation: fit on fresh support-split draws of ONE pool (held 1/2/3/5), score the stored validation v2
                                 episodes of the OTHER pool -> runs/suppsplit_20261115_v2/extra/pair_<tag>_<pool>_d<r>.npz (key pair_<tag>)
  official <tag> [draws]         final: fit on both pools (held 1/2/3 -> galleries 9/8/7), score the real questions (10-game galleries)
                                 -> runs/listfuse_20261115_v1/<tag>/official_<pool>.npy  (used by recipe_export_20261115_v1.py as V('pair_<tag>'))
"""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pairscore_20261115_v2 as p2
import numpy as np
import torch
from threadpoolctl import threadpool_limits
p1 = p2.p1; ss = p1.ss; A = ss.A; p = ss.p; old = ss.old; RUN = p.ROOT / 'runs/listfuse_20261115_v1'; SEED = 2026111511; MAXC = 9
V1 = p.ROOT / 'runs/suppsplit_20261115_v1'; PROTO = p.ROOT / 'runs/protohead_20261115_v1'; SQ = p.ROOT / 'runs/seqid_20261115_v1'; BB4 = p.ROOT / 'runs/bigbank4_90_20261115_v1'
REG = {'d3': ('npz', V1 / 'heademb_b3.npz', 'd'), 'lin3': ('npz', V1 / 'heademb_b3.npz', 'lin'),
       'a2f0': ('seq', SQ / 'a2/g1/f0/off_own.npy'), 'a2f2': ('seq', SQ / 'a2/g1/f2/off_own.npy'), 'b1f0': ('seq', SQ / 'b1/g1/f0/off_own.npy')}
for t in ('proto1', 'proto0_b345', 'proto1_b345', 'protoX1'):
    for c in ('d', 'lin'): REG[f'{t}:{c}'] = ('npy', PROTO / t / f'emb_{c}.npy')
for t in ('b345a', 'b345f'):
    for c in ('d', 'lin'): REG[f'{t}:{c}'] = ('heads', BB4 / f'export_{t}', c)


def source(name):
    if name not in REG and (SQ / name / 'off_own.npy').exists(): REG[name] = ('seq', SQ / name / 'off_own.npy')      # any seq model folder relative to runs/seqid_20261115_v1
    if name.startswith('tok:'): REG[name] = ('npz', p.ROOT / 'runs/tokhead_20261116_v1' / name[4:] / 'emb.npz', 'tok')      # token-net embeddings of tokhead_20261116_v1.py (official pos order)
    kind, path, *rest = REG[name]
    if kind == 'npz': return [p1.nrm(e.astype(np.float64)) for e in np.load(path)[rest[0]]]
    if kind == 'npy': return [p1.nrm(e.astype(np.float64)) for e in np.load(path)]
    if kind == 'seq': return [p1.nrm(np.load(path).astype(np.float64))]
    cache = RUN / '_heademb' / f'{path.name}.npz'      # export roots: embed the official games once (official_results order -> pos order)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True); games, pools, keys, pos = ss.official(); oid = np.load(old.RUN / 'official_results/indices.npy'); rowof = np.zeros(len(keys), int); rowof[oid] = np.arange(len(oid))
        e = ss.head_embs(path, np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r'))); np.savez(cache, **{c: np.stack(e[c])[:, rowof].astype(np.float32) for c in e})
    return [p1.nrm(e.astype(np.float64)) for e in np.load(cache)[rest[0]]]


NN = set()


def models(spec):
    M = {}
    for part in spec.split(','):
        g, srcs = part.split('=') if '=' in part else (part, part)
        if g.endswith('*'): g = g[:-1]; NN.add(g)
        M[g] = [e for s in srcs.split('+') for e in source(s)]
    return M, np.load(V1 / 'heademb_b3.npz')['black']


def phi(M, black, qg, sup):
    X, names = p1.features(M, black, np.asarray(qg), sup); K = len(M); NB = MAXC + 1; ns = np.clip(X[:, :, -2].astype(int), 0, MAXC); nd = np.clip(X[:, :, -1].astype(int), 0, MAXC); out = np.zeros((X.shape[0], X.shape[1], K * (1 + 2 * NB)), np.float32); I, J = np.indices(ns.shape)
    for k in range(K):
        o = k * (1 + 2 * NB); out[:, :, o] = X[:, :, 6 * k + 1]; out[I, J, o + 1 + ns] = np.nan_to_num(X[:, :, 6 * k + 3]); out[I, J, o + 1 + NB + nd] = np.nan_to_num(X[:, :, 6 * k + 5])
        out[:, :, o + 1] = 0; out[:, :, o + 1 + NB] = 0      # count 0 = no games of that colour: no term
    if NN:
        qg = np.asarray(qg); S = np.stack([np.asarray(s) for s in sup]); assert S.ndim == 2, 'nearest-neighbour terms need equal gallery sizes'; nc, gs = S.shape; same = black[qg][:, None, None] == black[S][None]; ext = []
        for g in M:
            if g not in NN: continue
            a = b = 0
            for Z in M[g]:
                C = (Z[qg] @ Z[S.ravel()].T).reshape(len(qg), nc, gs); a = a + np.sort(C, 2)[:, :, -2:].mean(2); Cs = np.where(same, C, -np.inf); top = np.sort(Cs, 2)[:, :, -2:]; fin = np.isfinite(top); cnt = fin.sum(2); val = np.where(fin, top, 0.).sum(2) / np.maximum(cnt, 1); val[cnt == 0] = np.nan; b = b + val
            ext += [p1.zrow(a / len(M[g])), np.nan_to_num(p1.zrow(b / len(M[g])))]
        out = np.concatenate([out, np.stack(ext, 2).astype(np.float32)], 2)
    return out


def fit(X, own, l2=1e-3, steps=400):
    dev = 'cpu'; Xt = torch.tensor(X, device=dev); y = torch.tensor(own, device=dev); w = torch.zeros(X.shape[2], requires_grad=True, device=dev); opt = torch.optim.Adam([w], lr=.05)
    for s in range(steps):
        loss = torch.nn.functional.cross_entropy(Xt @ w, y) + l2 * (w ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().cpu().numpy(), float(loss)


def draws(M, black, pool, pos, n, helds, rng):
    full = [np.array([pos[k] for k in pool.supports[c]]) for c in pool.candidate_ids]; Xs, ys = [], []
    for r in range(n):
        for held in helds:
            sup, qg, own = [], [], []
            for j, s in enumerate(full): hj = rng.choice(10, held, replace=False); m = np.zeros(10, bool); m[hj] = True; sup.append(s[~m]); qg += s[m].tolist(); own += [j] * held
            Xs.append(phi(M, black, qg, sup)); ys += own
    return np.concatenate(Xs), np.array(ys)


def show(M, w):
    NB = MAXC + 1
    for k, n in enumerate(M): o = k * (1 + 2 * NB); print(f'   {n}: all {w[o]:.2f} | same by count', np.round(w[o + 2:o + 1 + NB], 2).tolist(), '| other colour by count', np.round(w[o + 2 + NB:o + 1 + 2 * NB], 2).tolist(), flush=True)
    if NN: print('   nearest-neighbour terms (top-2 all, top-2 same colour) for', [g for g in M if g in NN], np.round(w[len(M) * (1 + 2 * NB):], 2).tolist(), flush=True)


def qscore(M, black, questions, sup, w):
    qg = [i for z in questions for i in z]; L = phi(M, black, qg, sup) @ w; st = np.cumsum([0] + [len(z) for z in questions]); return np.stack([L[st[i]:st[i + 1]].mean(0) for i in range(len(questions))])


def run(tag, spec, n=1):
    out = RUN / tag; out.mkdir(parents=True, exist_ok=False); M, black = models(spec); games, pools, keys, pos = ss.official(); rng = np.random.default_rng(SEED); W = {}
    for pool in pools:
        X, y = draws(M, black, pool, pos, n, (1, 2, 3, 5), rng); W[pool.name], loss = fit(X, y); del X; print(pool.name, 'fit loss', round(loss, 4), flush=True); show(M, W[pool.name])
    np.savez(out / 'weights_cross.npz', **W); (out / 'SPEC.json').write_text(json.dumps({'spec': spec, 'groups': {g: len(v) for g, v in M.items()}, 'draws': n, 'MAXC': MAXC}), encoding='utf-8'); eps = json.loads((ss.RUN / 'EPISODES.json').read_text()); (ss.RUN / 'extra').mkdir(exist_ok=True)
    for pool in pools:
        other = [q for q in pools if q.name != pool.name][0].name
        for r, ep in enumerate(eps[pool.name]): np.savez(ss.RUN / 'extra' / f'pair_{tag}_{pool.name}_d{r}.npz', **{f'pair_{tag}': qscore(M, black, [z['indices'] for z in ep['questions']], ep['supports'], W[other])})
    print('saved component', f'pair_{tag}', flush=True)


def official(tag, n=2):
    out = RUN / tag; spec = json.loads((out / 'SPEC.json').read_text())['spec']; M, black = models(spec); games, pools, keys, pos = ss.official(); rng = np.random.default_rng(SEED + 1); Xs, ys = [], []
    for pool in pools: X, y = draws(M, black, pool, pos, n, (1, 2, 3), rng); Xs.append(X); ys.append(y)
    w, loss = fit(np.concatenate(Xs), np.concatenate(ys)); del Xs; print('final fit on both pools, loss', round(loss, 4), flush=True); show(M, w); np.save(out / 'weights_final.npy', w)
    for pool in pools:
        S = qscore(M, black, [[pos[k] for k in z['keys']] for z in pool.questions], [np.array([pos[k] for k in pool.supports[c]]) for c in pool.candidate_ids], w); assert S.shape == (400, 600) and np.isfinite(S).all(); np.save(out / f'official_{pool.name}.npy', S, allow_pickle=False)
    print('official matrices saved in', out, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=6):
        if sys.argv[1] == 'run': run(sys.argv[2], sys.argv[3], *([int(sys.argv[4])] if len(sys.argv) > 4 else []))
        else: official(sys.argv[2], *([int(sys.argv[3])] if len(sys.argv) > 3 else []))
