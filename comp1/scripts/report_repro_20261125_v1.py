"""2026-11-25 report prep, phase A: reproducibility check of the comp1 final file v9r (SHA-256 b3df6a05...82ad).
Read-only on every existing run / submission; writes only runs/report_repro_20261125_v1/.
  L1   saved FINAL score matrices -> CSV -> SHA-256
       new pools: runs/recipe_export_20261115_v1/out_20261122_v9qt0_.../<pool>_score.npy + temps_used (RESULT.json);
                  single-game rows ranked by the raw score, Sinkhorn (cap 0.9) among q>=2 rows (nosink_20261119_v2.py)
       old pools: runs/oldpool_upgrade_20261123_v1/<pool>_S_blend_25H_75seq.npy -> Sinkhorn T=0.1 (2000 it.) -> Hungarian top-1 + Sinkhorn order
  L2o  old pools rebuilt from the saved game embeddings (8 seq models) + the two 11-05 H matrices -> blend matrix (max |diff|) -> rows
  L2n  new pools: v9qt0 score rebuilt from its components with recipe_export_20261122_v6 (pair_m39htg / ro_o5 / ms_mv17 loaded from their
       saved official matrices, hgm_top30 and og5q1_cos recomputed on the official galleries) -> max |diff| vs saved score -> rows
  L3o  (stage emb, GPU) old-pool game embeddings recomputed from the 8 seq checkpoints into this run folder -> blend -> rows
usage: report_repro_20261125_v1.py check | emb"""
from pathlib import Path
import sys, csv, json, hashlib, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'runs/report_repro_20261125_v1'
FINAL = ROOT / 'submissions/20261123_v9r_oldpool_seq8_h25_v9qt_new_v1/submission_player.csv'; FINAL_SHA = 'b3df6a05ddcc5614047691b44c0969fb60645b5aeab165f9c8d70de804fe82ad'
QT0 = ROOT / 'runs/recipe_export_20261115_v1/out_20261122_v9qt0_listfuse_m39htg_og5_rematch_maxsim_sinkhorn_v1'
SAMPLE = ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests/sample_submission_player_v2.csv'
SEQF = ('b1_full', 'b1cap_full', 'b1s1_full', 'b2_full'); SEQ3 = ('b1f0', 'a2f0', 'a2f2')


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def rows(f): return {r[0]: r[1:] for r in list(csv.reader(open(f, encoding='utf-8', newline='')))[1:]} if Path(f).exists() else None
def n_equal(pairs): return None if pairs is None else sum(pairs)


def new_rows(S, q, T, A):
    o = np.argsort(-S, 1, kind='stable')[:, :5]; m = q >= 2; o[m] = np.argsort(-A.new_transform(S[m], q[m], T), 1, kind='stable')[:, :5]; return o


def old_rows(blend, op):
    from scipy.optimize import linear_sum_assignment
    L = op.sinkhorn_bijection(blend, 0.1, iters=2000); _, h = linear_sum_assignment(-L); so = np.argsort(-L, axis=1, kind='stable')
    return [[int(h[i])] + [int(j) for j in so[i] if int(j) != int(h[i])][:4] for i in range(len(L))]


def blend_from(E, H, cg, qg, op):
    S = {m: op.z(op.score_matrix(E[m], cg, qg)) for m in SEQF + SEQ3}
    ens = op.z(.85 * op.z(sum(S[m] for m in SEQF) / 4) + .15 * op.z(sum(S[m] for m in SEQ3) / 3)); return op.z(.25 * op.z(H) + .75 * ens)


def write_csv(preds, name):
    qids = [r[0] for r in list(csv.reader(open(SAMPLE, encoding='utf-8', newline='')))[1:]]; f = OUT / name
    with open(f, 'w', encoding='utf-8', newline='') as h:
        w = csv.writer(h, lineterminator='\n'); w.writerow(['question_id', 'top_1', 'top_2', 'top_3', 'top_4', 'top_5'])
        for k in qids: w.writerow([k] + preds[k])
    return f


def check():
    OUT.mkdir(exist_ok=True); rep = {'final_sha_on_disk': sha(FINAL) if FINAL.exists() else None, 'final_sha_expected': FINAL_SHA}; fin = rows(FINAL); t0 = time.time()
    import assign_sinkhorn_20261113_v1 as A, oldpool_upgrade_20261123_v1 as op, suppsplit_eval_20261115_v2 as s2
    games, pools, keys, pos = s2.ss.official(); T = {int(k): float(v) for k, v in json.load(open(QT0 / 'RESULT.json', encoding='utf-8'))['temps_used'].items()}
    # ---- L1
    preds = {}
    for pool in pools:
        S = np.load(QT0 / f'{pool.name}_score.npy'); q = np.array([z['num_games'] for z in pool.questions]); o = new_rows(S, q, T, A)
        for i, z in enumerate(pool.questions): preds[z['question_id']] = [pool.candidate_ids[j] for j in o[i]]
    for pool, pre in (('public', 'pub'), ('private', 'priv')):
        R = old_rows(np.load(op.RUN / f'{pool}_S_blend_25H_75seq.npy'), op)
        for i in range(400): preds[f'{pre}_q_{i + 1:04}'] = [f'{pre}_p_{j + 1:04}' for j in R[i]]
    f1 = write_csv(preds, 'L1_from_saved_final_scores.csv'); rep['L1'] = {'csv': str(f1), 'sha256': sha(f1), 'identical_to_final': sha(f1) == FINAL_SHA, 'rows_equal': n_equal(None if fin is None else (preds[k] == fin[k] for k in fin))}
    print('L1', rep['L1'], f'{time.time() - t0:.0f}s', flush=True)
    # ---- L2o: old pools from saved embeddings + H
    rep['L2o'] = {}; L2pred = {}
    for pool, pre, Hp in (('public', 'pub', op.H_PUBLIC_PATH), ('private', 'priv', op.H_PRIVATE_PATH)):
        Z = dict(np.load(op.RUN / f'{pool}_moves_cap.npz')); role, own = Z['role'], Z['own']; cg = [np.flatnonzero((role == 0) & (own == c)) for c in range(400)]; qg = [np.flatnonzero((role == 1) & (own == c)) for c in range(400)]
        E = {}
        for m in SEQF + SEQ3: e = np.load(op.RUN / f'emb_{pool}_{m}.npy').astype(np.float32); E[m] = e / np.linalg.norm(e, axis=1, keepdims=True)
        B = blend_from(E, np.load(Hp).astype(np.float64), cg, qg, op); saved = np.load(op.RUN / f'{pool}_S_blend_25H_75seq.npy'); R = old_rows(B, op)
        for i in range(400): L2pred[f'{pre}_q_{i + 1:04}'] = [f'{pre}_p_{j + 1:04}' for j in R[i]]
        rep['L2o'][pool] = {'blend_maxabs_vs_saved': float(np.abs(B - saved).max()), 'rows_equal_final': n_equal(None if fin is None else ([f'{pre}_p_{j + 1:04}' for j in R[i]] == fin[f'{pre}_q_{i + 1:04}'] for i in range(400))), 'H_sha256': sha(Hp)}
    print('L2o', rep['L2o'], f'{time.time() - t0:.0f}s', flush=True)
    # ---- L2n: v9qt0 score rebuilt from components
    import recipe_export_20261122_v6 as v6
    v3 = v6.v3; expr = json.load(open(QT0 / 'RESULT.json', encoding='utf-8'))['expression']; fn = v3.evaluator(expr); C = v3.Comp(); rep['L2n'] = {'expression': expr}
    for pool in C.pools:
        ep = {'supports': [[C.pos[k] for k in pool.supports[a]] for a in pool.candidate_ids], 'questions': [{'indices': [C.pos[k] for k in z['keys']], 'q': z['num_games']} for z in pool.questions]}; q = np.array([z['num_games'] for z in pool.questions]); D = {}
        for nm in v3.names_of(expr):
            if nm.startswith('pair_'): D[nm] = np.load(v3.LF / nm[5:] / f'official_{pool.name}.npy')
            elif nm.startswith('ms_'): D[nm] = np.load(ROOT / 'runs/tokmaxsim_20261117_v1' / nm[3:] / f'official_{pool.name}.npy')
            elif nm.startswith(('ro_', 'rh_')): D[nm] = np.load(ROOT / 'runs/tokopp_20261117_v1' / nm[3:] / f'official_{pool.name}.npz')[nm[:2]]
            else: D[nm] = C.score(nm, ep)
        S = fn(D); saved = np.load(QT0 / f'{pool.name}_score.npy'); o = new_rows(S, q, T, A)
        for i, z in enumerate(pool.questions): L2pred[z['question_id']] = [pool.candidate_ids[j] for j in o[i]]
        rep['L2n'][pool.name] = {'score_maxabs_vs_saved': float(np.abs(S - saved).max()), 'rows_equal_final': n_equal(None if fin is None else ([pool.candidate_ids[j] for j in o[i]] == fin[z['question_id']] for i, z in enumerate(pool.questions)))}
    print('L2n', {k: v for k, v in rep['L2n'].items() if k != 'expression'}, f'{time.time() - t0:.0f}s', flush=True)
    f2 = write_csv(L2pred, 'L2_from_components.csv'); rep['L2'] = {'csv': str(f2), 'sha256': sha(f2), 'identical_to_final': sha(f2) == FINAL_SHA}
    print('L2', rep['L2'], flush=True)
    (OUT / 'CHECK.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')


def emb():
    """L3o: re-embed the old pools with the 8 seq checkpoints (same code path as oldpool_upgrade_20261123_v1.embed_pool, output redirected here)."""
    import torch, oldpool_upgrade_20261123_v1 as op
    OUT.mkdir(exist_ok=True); fin = rows(FINAL) or rows(OUT / 'L2_from_components.csv'); rep = {'compared_against': 'final CSV' if FINAL.exists() else 'L2_from_components.csv (run check first)'}
    for pool, pre, Hp in (('public', 'pub', op.H_PUBLIC_PATH), ('private', 'priv', op.H_PRIVATE_PATH)):
        Z = dict(np.load(op.RUN / f'{pool}_moves_cap.npz')); op.RUN = OUT / 'L3o'; op.RUN.mkdir(exist_ok=True); op.OLDPROXY = OUT / 'no_reuse'      # never reuse saved embeddings
        E = op.embed_pool(pool, Z); op.RUN = ROOT / 'runs/oldpool_upgrade_20261123_v1'
        role, own = Z['role'], Z['own']; cg = [np.flatnonzero((role == 0) & (own == c)) for c in range(400)]; qg = [np.flatnonzero((role == 1) & (own == c)) for c in range(400)]
        dE = {m: float(np.abs(E[m] - (lambda e: e / np.linalg.norm(e, axis=1, keepdims=True))(np.load(op.RUN / f'emb_{pool}_{m}.npy').astype(np.float32))).max()) for m in SEQF + SEQ3}
        B = blend_from(E, np.load(Hp).astype(np.float64), cg, qg, op); R = old_rows(B, op)
        rep[pool] = {'emb_maxabs_vs_saved': dE, 'blend_maxabs_vs_saved': float(np.abs(B - np.load(op.RUN / f'{pool}_S_blend_25H_75seq.npy')).max()), 'rows_equal_final': n_equal(None if fin is None else ([f'{pre}_p_{j + 1:04}' for j in R[i]] == fin[f'{pre}_q_{i + 1:04}'] for i in range(400))), 'checkpoints': {m: [str(op.MODELS[m][0]), sha(op.MODELS[m][0])] for m in SEQF + SEQ3}}
        print('L3o', pool, {k: v for k, v in rep[pool].items() if k != 'checkpoints'}, flush=True)
    (OUT / 'L3o.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    check() if sys.argv[1] == 'check' else emb()
