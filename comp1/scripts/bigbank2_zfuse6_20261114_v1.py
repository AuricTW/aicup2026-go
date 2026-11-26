"""bigbank2 (3987 players / 158k games, view90) full-bank PLDA d128/256/384/512/768/1024, z-mean of all 6 + 0.10 z(prior), capped Sinkhorn.
Choice evidence: runs/bigbank2_90_20261114_v1/eval_full_dims/RESULT.json (cross-fit fresh players):
  v2_z128_256_384_512_768_1024 balanced +0.0215 CI [0.0079, 0.0342] vs v1_z128_256_384 (which itself beat production rrf3 by +0.022).
Temperatures: calibrated on the cross-fit fold episodes (600 candidates, distinct targets; held-out players) using the v2 fold models'
  6-dim z-mean (scale dev identities are now in the bigbank2 training set, so scale dev is not used).
stages: train | export
"""
from pathlib import Path
import sys, argparse, csv
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank90_zfuse_20261114_v1 as zf
import bigbank2_90_20261114_v1 as b2
import numpy as np
from threadpoolctl import threadpool_limits
rw = zf.rw; ex = zf.ex; p = zf.p; old = zf.old; cap = zf.cap; A = zf.A
DIMS = (128, 256, 384, 512, 768, 1024); PRIOR_W = .10
MODELS = b2.RUN / 'models'; EVAL = b2.RUN / 'eval_full_dims'; CROSS = p.ROOT / 'runs/crossfit90_20261114_v1'
CUR = zf.CUR; CUR_SHA = zf.CUR_SHA
OUT = p.ROOT / 'submissions/20261114_v4_bigbank2_zfuse6_sinkhorn_v1'


def train():
    ids = np.load(b2.RUN / 'train_indices.npy'); labels = np.load(b2.RUN / 'train_labels.npy'); raw = b2.lookup90(ids); print('train', raw.shape, len(np.unique(labels)))
    for d in DIMS:
        folder = MODELS / f'big2_raw{d}_v90'; folder.mkdir(parents=True, exist_ok=False); x = old.pre(raw, folder, True)
        r = p.fit_plda(x, labels, folder / 'PLDA.npz', dimension=d); p.b.write(folder / 'FIT_RECEIPT.json', dict(r, training_only=True, games=int(len(ids)), players=int(len(np.unique(labels))))); print(d, r, flush=True)


def temps():
    eps = p.b.read(CROSS / 'EPISODES.json'); nm = 'v2_z128_256_384_512_768_1024'
    Z = np.vstack([A.z(np.load(EVAL / f'comps_f{k}.npz')[nm]) for k in range(3)])
    tg = np.concatenate([[z['target'] for z in ep['questions']] for ep in eps]); ql = np.concatenate([[z['q'] for z in ep['questions']] for ep in eps])
    return A.calibrate(Z, tg, ql)


def export():
    assert A.sha(CUR) == CUR_SHA; dest = b2.RUN / 'export_zfuse6'; dest.mkdir(exist_ok=False); cur = A.rows(CUR); preds = dict(cur)
    T = temps(); print('temps', T)
    ids = np.load(old.RUN / 'official_results/indices.npy'); raw = np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r')
    games, pools = p.b.load_official(); keys = p.b.read(p.b.RUN / 'official_stat/keys.json'); pos = {k: i for i, k in enumerate(keys)}; rep = {'temps': T, 'dims': DIMS, 'prior_weight': PRIOR_W, 'pools': {}}
    for pool in pools:
        ep = {'supports': [[pos[k] for k in pool.supports[a]] for a in pool.candidate_ids], 'questions': [{'indices': [pos[k] for k in z['keys']], 'q': z['num_games']} for z in pool.questions]}
        q = np.array([z['num_games'] for z in pool.questions]); comps = {}
        for d in DIMS:
            comps[d] = old.component(MODELS / f'big2_raw{d}_v90', raw, ids, ep); np.save(dest / f'{pool.name}_big2_raw{d}.npy', comps[d], allow_pickle=False)
        par = {n: np.load(cap.RUN / f'official_predictions/{pool.name}/{n}.npy') for n in ('fused', 'component', 'oldglobal64')}
        w = cap.countweights(ep); oldneural = .5 * ex.R(par['oldglobal64']) + .5 * ex.R(par['component']); prior = (par['fused'] - w * oldneural) / (1 - w)
        oldcomps = {f'big_raw{d}_v90': np.load(ex.RUN / pool.name / f'big_raw{d}_v90.npy') for d in (256, 768, 1024)}
        assert np.allclose(rw.fuse(prior, rw.big3(oldcomps), q, .85, .85), np.load(ex.RUN / pool.name / 'score.npy'), atol=1e-15, rtol=0)
        neural = sum(A.z(comps[d]) for d in DIMS) / len(DIMS); score = PRIOR_W * A.z(prior) + (1 - PRIOR_W) * neural
        L = A.new_transform(score, q, T); order = np.argsort(-L, 1, kind='stable')[:, :5]
        curorder = np.array([[pool.candidate_ids.index(c) for c in cur[z['question_id']]] for z in pool.questions])
        for i, z in enumerate(pool.questions): preds[z['question_id']] = [pool.candidate_ids[j] for j in order[i]]
        np.save(dest / f'{pool.name}_score.npy', score, allow_pickle=False); np.save(dest / f'{pool.name}_balanced_logP.npy', L, allow_pickle=False)
        rep['pools'][pool.name] = {'Top1_changed_vs_current_best': int((order[:, 0] != curorder[:, 0]).sum()), 'by_games': {str(n): int((order[q == n, 0] != curorder[q == n, 0]).sum()) for n in (1, 2, 3, 5)},
                                   'top1_in_current_top5': float(np.mean([order[i, 0] in curorder[i] for i in range(len(q))])), 'max_column_mass': float(np.exp(L).sum(0).max()),
                                   'raw_neural_top1_unique_frac': float(len(np.unique(neural.argmax(1))) / len(q))}
        print(pool.name, rep['pools'][pool.name], flush=True)
    sample = p.ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests/sample_submission_player_v2.csv'; qids = list(A.rows(sample).keys()); assert len(qids) == 1600 and set(qids) == set(preds)
    newq = {z['question_id'] for pool in pools for z in pool.questions}; assert all(preds[k] == cur[k] for k in qids if k not in newq)
    OUT.mkdir(exist_ok=False); out = OUT / 'submission_player.csv'
    with out.open('x', encoding='utf-8', newline='') as h:
        wr = csv.writer(h, lineterminator='\n'); wr.writerow(A.HEADER)
        for k in qids: wr.writerow([k] + preds[k])
    val = p.b.validate(out); assert val['valid'], val; assert A.rows(out) == preds
    rep.update({'created_utc': datetime.now(timezone.utc).isoformat(), 'CSV': str(out), 'SHA256': A.sha(out), 'bytes': out.stat().st_size, 'rows': 1600, 'format_check': val, 'parent_CSV': str(CUR), 'parent_SHA256': CUR_SHA,
                'parent_official': None, 'rows_changed_vs_parent': int(sum(preds[k] != cur[k] for k in qids)), 'old800_identical_to_parent': True, 'uploaded': False, 'source_SHA256': A.sha(__file__),
                'crossfit_evidence': p.b.read(EVAL / 'RESULT.json')['results']['v2_z128_256_384_512_768_1024']})
    p.b.write(dest / 'RESULT.json', rep); p.b.write(OUT / 'RECEIPT.json', rep); print(rep['CSV'], rep['SHA256'], rep['rows_changed_vs_parent'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); v = ap.parse_args()
    with threadpool_limits(limits=4): {'train': train, 'export': export, 'temps': lambda: print(temps())}[v.stage]()
