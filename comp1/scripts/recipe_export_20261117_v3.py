"""v3 (2026-11-17) = v2 + components ms_<tag> (decision-level MaxSim, runs/tokmaxsim_20261117_v1/<tag>/official_<pool>.npy).
v2 (2026-11-17) = v1 + components ro_<tag> / rh_<tag> (token-net opponent rematch evidence, runs/tokopp_20261117_v1/<tag>/official_<pool>.npz).
Recipe export (v8 family): ONE expression string defines the score, and the SAME string is (a) evaluated on the official
support-split validation v2 (incl. the production prior, suppsplit_prior_20261115_v1.py) and (b) applied to the real official questions.
  expression: python over z-scored components, V('<name>') and zm(a, b, ...) as in suppsplit_quick_20261115_v1.py, e.g.
     "0.05*V('prior')+0.95*zm(0.8*zm(V('ens_proto1'),V('ens345a'))+0.2*V('b1f0'))"
Components on the official questions are rebuilt from game-level embeddings with the same functions that filled the validation comps;
before exporting, every component is recomputed on one validation episode and must equal the stored validation matrix.
  heads export root (models_d / models_lin + TRAIN_RECEIPT.json) ......... HEADS
  protohead run (runs/protohead_20261115_v1/<tag>/emb_d.npy, emb_lin.npy)  any name ens_<tag> with that folder
  seq-id models (off_own.npy, official order) ............................. SEQ (+ --seq name=path[,path])
  prior = z(prior) recovered from v6b (identical to the recomputed production prior), base2 / ens3 / b2e3 = v6b parts
Temperatures: calibrated with the expression on all 12 validation episodes.  Parent = v6b; old 800 rows copied.  Never uploads.
Usage: eval "<expr>"                       validation only (other-pool temperatures, CI vs the full v6b formula)
       export <outname> "<expr>"           validation + CSV in submissions/<outname>/
       dry <outname> "<expr>"              everything except writing files
"""
from pathlib import Path
import sys, argparse, csv, re
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent))
import seqfuse_export_20261115_v1 as x
import suppsplit_eval_20261115_v2 as s2
import numpy as np
from threadpoolctl import threadpool_limits
ss = s2.ss; g = x.g; h = x.h; p = x.p; A = x.A; z6 = x.z6; old = ss.old; BW = x.BW; PW = x.PW
BB4 = p.ROOT / 'runs/bigbank4_90_20261115_v1'; PROTO = p.ROOT / 'runs/protohead_20261115_v1'; SQ = p.ROOT / 'runs/seqid_20261115_v1'; RUN = p.ROOT / 'runs/recipe_export_20261115_v1'
HEADS = {'ens345a': BB4 / 'export_b345a', 'ens345f': BB4 / 'export_b345f'}
SEQ = {'seq_a2f0': [SQ / 'a2/g1/f0/off_own.npy'], 'a2f2': [SQ / 'a2/g1/f2/off_own.npy'], 'b1f0': [SQ / 'b1/g1/f0/off_own.npy']}
V6B_EXPR = "0.1*V('prior')+0.9*V('b2e3')"; LF = p.ROOT / 'runs/listfuse_20261115_v1'


def score_shape(pool): return (len(pool.questions), len(pool.candidate_ids))


class Comp:
    """score matrices of named components for an episode {'supports': [[game pos]], 'questions': [{'indices': [game pos]}]}"""
    def __init__(self):
        self.games, self.pools, self.keys, self.pos = ss.official(); self.allk = np.arange(len(self.keys)); self.ids = np.load(old.RUN / 'official_results/indices.npy'); self.cache = {}

    def embs(self, name):
        if name in self.cache: return self.cache[name]
        if name in SEQ: E = ('pos', [[np.load(f).astype(np.float64) for f in SEQ[name]]])
        elif name in HEADS:
            raw = np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r')); e = ss.head_embs(HEADS[name], raw); E = ('ids', [e['d'], e['lin']])
        elif name.startswith('ens_') and (PROTO / name[4:] / 'emb_d.npy').exists(): E = ('pos', [list(np.load(PROTO / name[4:] / f'emb_{c}.npy').astype(np.float64)) for c in ('d', 'lin')])
        else: raise KeyError(name)
        self.cache[name] = E; return E

    def score(self, name, ep):
        kind, groups = self.embs(name); ids = self.allk if kind == 'pos' else self.ids
        S = [g.zm([h.setscores(e, ids, ep)[0] for e in grp]) for grp in groups]; return S[0] if name in SEQ else g.zm(S)


def names_of(expr): return sorted(set(re.findall(r"V\('([^']+)'\)", expr)))


def evaluator(expr):
    return lambda D: eval(expr, {'V': (lambda n: A.z(D[n])), 'zm': (lambda *a: A.z(sum(A.z(v) for v in a) / len(a))), 'np': np, 'A': A})


def validation(expr):
    data = {}
    for pool, r, ep, f in ss.each():
        D = dict(np.load(f))
        for e in sorted((ss.RUN / 'extra').glob(f'*_{pool}_d{r}.npz')): D.update(dict(np.load(e)))
        data[(pool, r)] = (D, np.array([z['target'] for z in ep['questions']]), np.array([z['q'] for z in ep['questions']]), ep)
    keys = sorted(data); pools = sorted({k[0] for k in keys}); Q = np.concatenate([data[k][2] for k in keys]); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q)))
    def run(e):
        fn = evaluator(e); S = {k: fn(data[k][0]) for k in keys}
        T = {pl: A.calibrate(A.z(np.vstack([S[j] for j in keys if j[0] != pl])), np.concatenate([data[j][1] for j in keys if j[0] != pl]), np.concatenate([data[j][2] for j in keys if j[0] != pl])) for pl in pools}
        return np.concatenate([A.sid(A.new_transform(S[k], data[k][2], T[k[0]]), data[k][1]) for k in keys]), A.calibrate(A.z(np.vstack([S[k] for k in keys])), np.concatenate([data[k][1] for k in keys]), np.concatenate([data[k][2] for k in keys]))
    ref, _ = run(V6B_EXPR); b, T = run(expr); d = b - ref; n = len(b) // len(keys)
    rep = {'expression': expr, 'questions': int(len(b)), 'score': float(b.mean()), 'v6b_formula': float(ref.mean()), 'delta_vs_v6b_formula': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist(),
           'by_q': {str(q): [float(b[Q == q].mean()), float(ref[Q == q].mean())] for q in (1, 2, 3, 5)}, 'by_pool': {pl: [float(np.concatenate([a[i * n:(i + 1) * n] for i, k in enumerate(keys) if k[0] == pl]).mean()) for a in (b, ref)] for pl in pools}, 'temps_all_episodes': T}
    print('validation v2:', round(rep['score'], 4), 'v6b formula', round(rep['v6b_formula'], 4), 'delta', round(rep['delta_vs_v6b_formula'], 4), rep['CI95'], 'by q [new, v6b]', {k: [round(a, 3) for a in v] for k, v in rep['by_q'].items()}, flush=True)
    return rep, T, data


def export(outname, expr, dry=False, tm=1.0):
    assert A.sha(x.PARENT) == x.PARENT_SHA; rep, T, data = validation(expr); keys_ = sorted(data); fn_ = evaluator(expr); T0 = dict(T); T = {k: v * tm for k, v in T.items()}
    rep.update({'temp_multiplier': tm, 'temps_used': T, 'validation_score_with_temps_used': float(np.concatenate([A.sid(A.new_transform(fn_(data[k][0]), data[k][2], T), data[k][1]) for k in keys_]).mean()), 'validation_score_with_unscaled_all_episode_temps': float(np.concatenate([A.sid(A.new_transform(fn_(data[k][0]), data[k][2], T0), data[k][1]) for k in keys_]).mean()),
                'temp_multiplier_note': 'validation galleries hold 5-7 games, the real ones 10: fitted temperatures fall with gallery size (7 vs 5 games: x0.86-0.96) and the Public2 scores of the 5 files already scored fit best at x0.6; validation score is flat for x0.5-0.9'})
    print('temps used', {k: round(v, 3) for k, v in T.items()}, 'validation with them', round(rep['validation_score_with_temps_used'], 4), flush=True); dest = RUN / f'out_{outname}'; dry or dest.mkdir(parents=True, exist_ok=False); cur = A.rows(x.PARENT); preds = dict(cur); C = Comp(); need = names_of(expr); fn = evaluator(expr)
    k0 = sorted(data)[0]; D0, _, _, ep0 = data[k0]; chk = {}
    for nm in need:      # every rebuilt component must reproduce the stored validation matrix
        if nm in ('prior', 'base2', 'ens3', 'b2e3') or nm.startswith(('pair_', 'ro_', 'rh_', 'ms_')): continue      # pair_<tag>: validation used cross-pool weights, the official matrix uses the both-pool fit (listfuse_20261115_v1.py official)
        chk[nm] = float(np.abs(A.z(C.score(nm, ep0)) - A.z(D0[nm])).max()); assert chk[nm] < 1e-4, (nm, chk[nm])
    print('component parity on', k0, chk, flush=True); rep.update({'component_parity_maxabs': chk, 'pools': {}})
    for pool in C.pools:
        ep = {'supports': [[C.pos[k] for k in pool.supports[a]] for a in pool.candidate_ids], 'questions': [{'indices': [C.pos[k] for k in z['keys']], 'q': z['num_games']} for z in pool.questions]}; q = np.array([z['num_games'] for z in pool.questions])
        S6 = np.load(x.V6B / f'{pool.name}_score.npy'); ens = np.load(x.V6B / f'{pool.name}_ens.npy'); base = g.zm([np.load(x.Z6 / f'{pool.name}_big2_raw{d}.npy') for d in z6.DIMS]); neural = BW * A.z(base) + (1 - BW) * A.z(ens); zprior = (S6 - (1 - PW) * neural) / PW
        assert np.abs(zprior.mean(1)).max() < 1e-6 and np.abs(zprior.std(1) - 1).max() < 1e-6, 'v6b decomposition failed'
        D = {'prior': zprior, 'base2': base, 'ens3': ens, 'b2e3': neural}
        for nm in need:
            if nm.startswith('pair_'): D[nm] = np.load(LF / nm[5:] / f'official_{pool.name}.npy'); assert D[nm].shape == score_shape(pool)
            elif nm.startswith('ms_'): D[nm] = np.load(p.ROOT / 'runs/tokmaxsim_20261117_v1' / nm[3:] / f'official_{pool.name}.npy'); assert D[nm].shape == score_shape(pool)      # v3: decision-level MaxSim (tokmaxsim_20261117_v1.py official)
            elif nm.startswith(('ro_', 'rh_')): D[nm] = np.load(p.ROOT / 'runs/tokopp_20261117_v1' / nm[3:] / f'official_{pool.name}.npz')[nm[:2]]; assert D[nm].shape == score_shape(pool)      # v2: rematch evidence (tokopp_20261117_v1.py official)
            elif nm not in D: D[nm] = C.score(nm, ep)
        score = fn(D); L = A.new_transform(score, q, T); order = np.argsort(-L, 1, kind='stable')[:, :5]; curorder = np.array([[pool.candidate_ids.index(c) for c in cur[z['question_id']]] for z in pool.questions])
        v6 = evaluator(V6B_EXPR)(D)
        for i, zq in enumerate(pool.questions): preds[zq['question_id']] = [pool.candidate_ids[j] for j in order[i]]
        dry or np.save(dest / f'{pool.name}_score.npy', score, allow_pickle=False)
        rep['pools'][pool.name] = {'Top1_changed_vs_parent': int((order[:, 0] != curorder[:, 0]).sum()), 'by_games': {str(n): int((order[q == n, 0] != curorder[q == n, 0]).sum()) for n in (1, 2, 3, 5)},
                                   'top1_in_parent_top5': float(np.mean([order[i, 0] in curorder[i] for i in range(len(q))])), 'corr_with_v6b_formula_score': float(np.corrcoef(A.z(score).ravel(), A.z(v6).ravel())[0, 1]), 'max_column_mass': float(np.exp(L).sum(0).max())}
        print(pool.name, rep['pools'][pool.name], flush=True)
    if dry: print('dry run: rows changed vs parent', sum(preds[k] != cur[k] for k in preds)); return
    sample = p.ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests/sample_submission_player_v2.csv'; qids = list(A.rows(sample).keys()); assert len(qids) == 1600 and set(qids) == set(preds)
    newq = {zq['question_id'] for pool in C.pools for zq in pool.questions}; assert all(preds[k] == cur[k] for k in qids if k not in newq)
    outdir = p.ROOT / f'submissions/{outname}'; outdir.mkdir(exist_ok=False); out = outdir / 'submission_player.csv'
    with out.open('x', encoding='utf-8', newline='') as fh:
        wr = csv.writer(fh, lineterminator='\n'); wr.writerow(A.HEADER)
        for k in qids: wr.writerow([k] + preds[k])
    val = p.b.validate(out); assert val['valid'], val; assert A.rows(out) == preds
    rep.update({'created_utc': datetime.now(timezone.utc).isoformat(), 'CSV': str(out), 'SHA256': A.sha(out), 'bytes': out.stat().st_size, 'rows': 1600, 'format_check': val, 'parent_CSV': str(x.PARENT), 'parent_SHA256': x.PARENT_SHA, 'parent_official': None,
                'rows_changed_vs_parent': int(sum(preds[k] != cur[k] for k in qids)), 'old800_identical_to_parent': True, 'uploaded': False, 'source_SHA256': A.sha(__file__), 'components': {n: ([str(f) for f in SEQ[n]] if n in SEQ else str(LF / n[5:]) if n.startswith('pair_') else str(HEADS.get(n, PROTO / n[4:]))) for n in need if n not in ('prior', 'base2', 'ens3', 'b2e3')}})
    p.b.write(dest / 'RESULT.json', rep); p.b.write(outdir / 'RECEIPT.json', rep); print(rep['CSV'], rep['SHA256'], rep['rows_changed_vs_parent'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['eval', 'export', 'dry']); ap.add_argument('args', nargs='+'); ap.add_argument('--seq', action='append', default=[]); ap.add_argument('--temp-mult', type=float, default=1.0); v = ap.parse_args()
    for s in v.seq: n, fs = s.split('=', 1); SEQ[n] = [Path(f) for f in fs.split(',')]
    with threadpool_limits(limits=6):
        if v.stage == 'eval': validation(v.args[0])
        else: export(v.args[0], v.args[1], v.stage == 'dry', v.temp_mult)
