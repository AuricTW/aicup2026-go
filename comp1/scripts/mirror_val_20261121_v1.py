"""2026-11-21: MIRROR validation for comp1 = validation v3 rebuilt on NEW support-split episodes (fresh seed) of the same official pools.
Why: every recipe choice since 11-17 (groups, rematch / MaxSim weights, models E-G, hubness) was selected on the 12 episodes of
runs/suppsplit_20261115_v2.  New random splits of the same galleries give a second estimate that none of those choices has seen
(same games, different roles -> partly, not fully, independent).  No answer key, no question labels: episodes come from the gallery
games that ship with the test files, exactly like v2.
Design = suppsplit_eval_20261115_v2.episodes (per pool and draw: g7 = 3 held out, q in {1,2,3}; g5 = 5 held out, q in {1,2,3,5};
400 distinct targets), seed 2026112101, 3 draws -> 6 episodes per pool.  Components are recomputed with the SAME frozen parts:
  pair <tag>        listfuse weights_cross.npz of that tag (weights fitted on the OTHER pool, as in listfuse run)
  ro <tag>          rematch evidence from runs/tokopp_20261117_v1/<tag>/posteriors_topk.npz (tokopp_20261117_v1.scores)
  ms <tag> <mdir>   decision-level MaxSim (tokmaxsim_20261117_v1.features / scores; GPU)
  hub               gallery hubness hg_top<k> / hgm_top<k> (hubgal_20261121_v1)
  tok <tag>         plain z-scored cosine of a registered token-net embedding (runs/tokhead_20261116_v1/<tag>/emb.npz) -- as tok_<tag>
  eval "<name>=<expr>" ...   v3 scoring (q=1 raw, Sinkhorn among q>=2, temperatures from the other pool) on the mirror episodes;
                             --both also reports the old 12 episodes and the 24 together.
Outputs: runs/mirrorval_20261121_v1/EPISODES.json, extra/<comp>_<pool>_d<r>.npz.  The original runs/suppsplit_20261115_v2 is never written."""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from threadpoolctl import threadpool_limits
import suppsplit_eval_20261115_v2 as s2
ss = s2.ss; A = ss.A; ROOT = Path(__file__).resolve().parents[1]; MRUN = ROOT / 'runs/mirrorval_20261121_v1'; SEED = 2026112101; POOLS = ('public2', 'private2')


def build(R=3):
    MRUN.mkdir(exist_ok=False); (MRUN / 'extra').mkdir(); games, pools, keys, pos = ss.official(); rng = np.random.default_rng(SEED); out = {}
    for pool in pools:
        out[pool.name] = []
        for r in range(R):
            for held, qset in ((3, (1, 2, 3)), (5, (1, 2, 3, 5))):
                full = [[pos[k] for k in pool.supports[c]] for c in pool.candidate_ids]; sup, ho = [], []
                for s in full: hj = set(rng.choice(10, held, replace=False).tolist()); sup.append([x for j, x in enumerate(s) if j not in hj]); ho.append([x for j, x in enumerate(s) if j in hj])
                tgt = rng.permutation(len(full))[:400]; qn = np.resize(np.array(qset), 400)[rng.permutation(400)]
                qs = [{'indices': sorted(rng.choice(ho[t], int(q), replace=False).tolist()), 'q': int(q), 'target': int(t)} for t, q in zip(tgt, qn)]
                out[pool.name].append({'supports': sup, 'questions': qs, 'gallery': 10 - held})
    (MRUN / 'EPISODES.json').write_text(json.dumps(out), encoding='utf-8')
    old = json.loads((ss.RUN / 'EPISODES.json').read_text()); same = sum(a['supports'] == b['supports'] for pl in POOLS for a in old[pl] for b in out[pl]); print('episodes built; identical galleries with the old set:', same, flush=True)


def each():
    eps = json.loads((MRUN / 'EPISODES.json').read_text())
    for pool in POOLS:
        for r, ep in enumerate(eps[pool]): yield pool, r, ep


def save(comp, pool, r, **arrs):
    np.savez(MRUN / 'extra' / f'{comp}_{pool}_d{r}.npz', **arrs)


def pair(tag):
    import listfuse_20261115_v1 as lf
    spec = json.loads((lf.RUN / tag / 'SPEC.json').read_text())['spec']; M, black = lf.models(spec); W = dict(np.load(lf.RUN / tag / 'weights_cross.npz'))
    for pool, r, ep in each():
        other = [p for p in POOLS if p != pool][0]; save(f'pair_{tag}', pool, r, **{f'pair_{tag}': lf.qscore(M, black, [z['indices'] for z in ep['questions']], ep['supports'], W[other])}); print('pair', tag, pool, r, flush=True)


def ro(tag):
    import tokopp_20261117_v1 as to
    games, pools, keys, pos = ss.official(); oid = np.load(ss.old.RUN / 'official_results/indices.npy'); rowof = np.zeros(len(keys), int); rowof[oid] = np.arange(len(oid)); P = np.load(to.RUN / tag / 'posteriors_topk.npz')
    for pool, r, ep in each():
        a, b = to.scores(P['top'], P['pt'], rowof, ep['supports'], [z['indices'] for z in ep['questions']]); save(f'tokopp_{tag}', pool, r, **{f'ro_{tag}': a, f'rh_{tag}': b}); print('ro', tag, pool, r, flush=True)


def ms(tag, mdir):
    import torch, tokmaxsim_20261117_v1 as tm
    torch.cuda.set_per_process_memory_fraction(0.3); games, pools, keys, pos = ss.official(); oid = np.load(tm.OFF / 'indices.npy'); rowof = np.zeros(len(keys), int); rowof[oid] = np.arange(len(oid)); H, Nn = tm.features(ROOT / mdir)
    for pool, r, ep in each():
        out = tm.scores(H, Nn, [np.array([rowof[g] for g in s]) for s in ep['supports']], [[rowof[g] for g in z['indices']] for z in ep['questions']]); save(f'maxsim_{tag}', pool, r, **{f'ms_{tag}': out}); print('ms', tag, pool, r, flush=True)


def hub():
    import hubgal_20261121_v1 as hg
    M, _ = hg.lf.models(hg.GROUPS)
    for pool, r, ep in each():
        q = np.array([z['q'] for z in ep['questions']]); save('hgv1', pool, r, **hg.comps(hg.hub(M, np.array(ep['supports'])), q)); print('hub', pool, r, flush=True)


def tok(tag):
    import listfuse_20261115_v1 as lf
    Zs = lf.source(f'tok:{tag}'); nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    for pool, r, ep in each():
        S = np.array(ep['supports']); qs = [z['indices'] for z in ep['questions']]; acc = 0
        for Z in Zs: C = nrm(Z[S].sum(1)); acc = acc + A.z(np.stack([(Z[np.array(g)] @ C.T).mean(0) for g in qs]))
        save(f'tokm_{tag}', pool, r, **{f'tok_{tag}': acc / len(Zs)}); print('tok', tag, pool, r, flush=True)


def load(run, eps_iter):
    data = {}
    for pool, r, ep in eps_iter:
        D = {}
        base = run / f'comps_{pool}_d{r}.npz'
        if base.exists(): D.update(dict(np.load(base)))
        for x in sorted((run / 'extra').glob(f'*_{pool}_d{r}.npz')): D.update(dict(np.load(x)))
        data[(pool, r)] = (D, np.array([z['target'] for z in ep['questions']]), np.array([z['q'] for z in ep['questions']]))
    return data


def evaluate(data, specs, label):
    keys = sorted(data); pools = sorted({k[0] for k in keys}); Q = np.concatenate([data[k][2] for k in keys]); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q)))
    def score(expr):
        S = {}
        for k in keys:
            D = data[k][0]; V = lambda n: A.z(D[n]); zm = lambda *a: A.z(sum(A.z(x) for x in a) / len(a)); S[k] = eval(expr, {'V': V, 'zm': zm, 'np': np, 'A': A, 'R': lambda n: D[n]})
        return S
    def evals(spec):
        e1, e2 = (spec.split('||') + [spec])[:2] if '||' in spec else (spec, spec)
        S1 = score(e1.strip()); S2 = S1 if e2 is e1 else score(e2.strip())
        T = {pl: A.calibrate(A.z(np.vstack([S2[j] for j in keys if j[0] != pl])), np.concatenate([data[j][1] for j in keys if j[0] != pl]), np.concatenate([data[j][2] for j in keys if j[0] != pl])) for pl in pools}
        out = []
        for k in keys:
            q = data[k][2]; tg = data[k][1]; m = q >= 2; s = np.empty(len(q)); s[~m] = A.sid(S1[k][~m], tg[~m]); s[m] = A.sid(A.new_transform(S2[k][m], q[m], T[k[0]]), tg[m]); out.append(s)
        return np.concatenate(out)
    print(f'--- {label}: {len(keys)} episodes, {len(Q)} questions', flush=True); base = None; res = {}
    for sp in specs:
        name, expr = sp.split('=', 1); b = evals(expr); res[name] = b; ab = ' '.join(f'q{q} {b[Q == q].mean():.4f}' for q in (1, 2, 3, 5))
        if base is None: base = b; print(f'{name:22s} {b.mean():.4f}  (reference)  {ab}', flush=True); continue
        d = b - base; ci = np.quantile(d[B].mean(1), [.025, .975]); byq = ' '.join(f'q{q} {d[Q == q].mean():+.4f}' for q in (1, 2, 3, 5))
        print(f'{name:22s} {b.mean():.4f}  paired {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]  {byq}', flush=True)
    return res


def main():
    st = sys.argv[1]; a = sys.argv[2:]
    if st == 'build': build(*[int(x) for x in a]); return
    if st == 'pair': [pair(t) for t in a]; return
    if st == 'ro': ro(a[0]); return
    if st == 'ms': ms(a[0], a[1]); return
    if st == 'hub': hub(); return
    if st == 'tok': [tok(t) for t in a]; return
    both = '--both' in a; specs = [x for x in a if x != '--both']
    mirror = load(MRUN, each()); evaluate(mirror, specs, 'MIRROR (new episodes)')
    if both:
        old = load(ss.RUN, ((p, r, e) for p, r, e, f in ss.each())); evaluate(old, specs, 'OLD validation v3')
        allv = {(k[0], k[1] + 100): v for k, v in mirror.items()}; allv.update(old); evaluate(allv, specs, 'OLD + MIRROR (24 episodes)')


if __name__ == '__main__':
    with threadpool_limits(limits=6): main()
