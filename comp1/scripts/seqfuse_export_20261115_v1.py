"""Export v5 (v7 family): v6b score with the moves-only sequence identity network fused into the neural part.
  neural' = (1-w) z(neural_v6b) + w z(seq),  seq = z-mean over the given seq models of cos(mean query emb, mean support emb)
  score   = 0.10 z(prior) + 0.90 * std_row(neural_v6b) * z(neural')        (prior weight relative to the neural part exactly as in v6b)
v6b parts are read back from its export folder (score, ens) and the bigbank2 PLDA6 files; z(prior) is recovered by subtraction and checked.
Seq models: cross-fit fold models (each trained without one third of the bigbank1 players; none of them contains official players) --
their official-game embeddings off_own.npy are used directly.  Temperatures: calibrated on the cross-fit folds that exist for <tag>
with the same fusion ((1-w) z(b2e3) + w z(seq)).  Parent = v6b; old 800 rows copied.  Never uploads.
Usage: export <tag> <w> <outname> <split/fold> [<split/fold> ...]      e.g. export a2 0.2 20261115_v7a_seqfuse_w20_v1 g1/f0
"""
from pathlib import Path
import sys, argparse, csv
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlphead_export_20261114_v2 as e2
import seqid_20261115_v2 as s2
import bigbank3_eval_20261114_v1 as ev3
import numpy as np
from threadpoolctl import threadpool_limits
g = e2.g; h = e2.h; p = e2.p; A = e2.A; z6 = e2.z6; Z6 = e2.Z6; b3 = ev3.b3; BW = ev3.BW; PW = z6.PRIOR_W
PARENT = p.ROOT / 'submissions/20261114_v6b_mlphead_bb3_bw20_sinkhorn_v1/submission_player.csv'; PARENT_SHA = '98896b18ec75efa258a6136c4aa65f55152562f0180bfbe2bf00ed70e88e5d0a'
V6B = h.RUN / 'export/out_20261114_v6b_mlphead_bb3_bw20_sinkhorn_v1'; RUN = s2.RUN / 'export'


def crossfit(tag, w):
    """fused cross-fit scores of every finished fold of <tag>: list of (split, fold, Z, targets, q)"""
    out = []
    for split, epf in (('g1', h.c1.RUN / 'EPISODES.json'), ('confirm', h.RUN / 'confirm/EPISODES.json')):
        folds = [k for k in range(3) if (s2.RUN / tag / split / f'f{k}/ev_own.npy').exists()]
        if not folds: continue
        eps = p.b.read(epf); Zb = np.load(s2.RUN / f'bundle_{split}.npz')
        for k in folds:
            ep = eps[k]; S = h.setscores(np.load(s2.RUN / tag / split / f'f{k}/ev_own.npy').astype(np.float64), Zb[f'EV_f{k}'], ep)[0]; V = np.load(b3.RUN / f'eval_{split}/comps_f{k}.npz')
            out.append((split, k, A.z((1 - w) * A.z(V['b2e3']) + w * A.z(S)), np.array([z['target'] for z in ep['questions']]), np.array([z['q'] for z in ep['questions']])))
    return out


def export(tag, w, outname, models):
    assert A.sha(PARENT) == PARENT_SHA; dest = RUN / f'out_{outname}'; dest.mkdir(parents=True, exist_ok=False); cur = A.rows(PARENT); preds = dict(cur)
    cf = crossfit(tag, w); assert cf; T = A.calibrate(np.vstack([c[2] for c in cf]), np.concatenate([c[3] for c in cf]), np.concatenate([c[4] for c in cf])); print('temps', T, 'from', [(c[0], c[1]) for c in cf])
    games, pools = p.b.load_official(); keys = p.b.read(p.b.RUN / 'official_stat/keys.json'); pos = {k: i for i, k in enumerate(keys)}; allk = np.arange(len(keys))
    embs = [np.load(s2.RUN / tag / m / 'off_own.npy').astype(np.float64) for m in models]; assert all(len(e) == len(keys) for e in embs)
    rep = {'seq_tag': tag, 'seq_models': models, 'w': w, 'BW': BW, 'prior_w': PW, 'temps': T, 'temps_from': [[c[0], c[1]] for c in cf], 'pools': {}}
    for pool in pools:
        ep = {'supports': [[pos[k] for k in pool.supports[a]] for a in pool.candidate_ids], 'questions': [{'indices': [pos[k] for k in z['keys']], 'q': z['num_games']} for z in pool.questions]}
        q = np.array([z['num_games'] for z in pool.questions]); S6 = np.load(V6B / f'{pool.name}_score.npy'); ens = np.load(V6B / f'{pool.name}_ens.npy')
        base = g.zm([np.load(Z6 / f'{pool.name}_big2_raw{d}.npy') for d in z6.DIMS]); neural = BW * A.z(base) + (1 - BW) * A.z(ens); zprior = (S6 - (1 - PW) * neural) / PW
        assert np.abs(zprior.mean(1)).max() < 1e-6 and np.abs(zprior.std(1) - 1).max() < 1e-6, 'v6b decomposition failed'      # recovered prior must be a row z-score
        seq = g.zm([h.setscores(e, allk, ep)[0] for e in embs]); fused = A.z((1 - w) * A.z(neural) + w * A.z(seq)); score = PW * zprior + (1 - PW) * neural.std(1, keepdims=True) * fused
        L = A.new_transform(score, q, T); order = np.argsort(-L, 1, kind='stable')[:, :5]; curorder = np.array([[pool.candidate_ids.index(c) for c in cur[z['question_id']]] for z in pool.questions])
        seqtop = np.argmax(seq, 1); oldtop = curorder[:, 0]
        for i, zq in enumerate(pool.questions): preds[zq['question_id']] = [pool.candidate_ids[j] for j in order[i]]
        np.save(dest / f'{pool.name}_seq.npy', seq, allow_pickle=False); np.save(dest / f'{pool.name}_score.npy', score, allow_pickle=False)
        rep['pools'][pool.name] = {'Top1_changed_vs_parent': int((order[:, 0] != oldtop).sum()), 'by_games': {str(n): int((order[q == n, 0] != oldtop[q == n]).sum()) for n in (1, 2, 3, 5)},
                                   'top1_in_parent_top5': float(np.mean([order[i, 0] in curorder[i] for i in range(len(q))])), 'seq_top1_equals_parent_top1': float((seqtop == oldtop).mean()),
                                   'seq_top1_in_parent_top5': float(np.mean([seqtop[i] in curorder[i] for i in range(len(q))])), 'max_column_mass': float(np.exp(L).sum(0).max())}
        print(pool.name, rep['pools'][pool.name], flush=True)
    sample = p.ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests/sample_submission_player_v2.csv'; qids = list(A.rows(sample).keys()); assert len(qids) == 1600 and set(qids) == set(preds)
    newq = {zq['question_id'] for pool in pools for zq in pool.questions}; assert all(preds[k] == cur[k] for k in qids if k not in newq)
    outdir = p.ROOT / f'submissions/{outname}'; outdir.mkdir(exist_ok=False); out = outdir / 'submission_player.csv'
    with out.open('x', encoding='utf-8', newline='') as fh:
        wr = csv.writer(fh, lineterminator='\n'); wr.writerow(A.HEADER)
        for k in qids: wr.writerow([k] + preds[k])
    val = p.b.validate(out); assert val['valid'], val; assert A.rows(out) == preds
    rep.update({'created_utc': datetime.now(timezone.utc).isoformat(), 'CSV': str(out), 'SHA256': A.sha(out), 'bytes': out.stat().st_size, 'rows': 1600, 'format_check': val, 'parent_CSV': str(PARENT), 'parent_SHA256': PARENT_SHA,
                'parent_official': None, 'rows_changed_vs_parent': int(sum(preds[k] != cur[k] for k in qids)), 'old800_identical_to_parent': True, 'uploaded': False, 'source_SHA256': A.sha(__file__)})
    p.b.write(dest / 'RESULT.json', rep); p.b.write(outdir / 'RECEIPT.json', rep); print(rep['CSV'], rep['SHA256'], rep['rows_changed_vs_parent'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('tag'); ap.add_argument('w', type=float); ap.add_argument('outname'); ap.add_argument('models', nargs='+'); v = ap.parse_args()
    with threadpool_limits(limits=4): export(v.tag, v.w, v.outname, v.models)
