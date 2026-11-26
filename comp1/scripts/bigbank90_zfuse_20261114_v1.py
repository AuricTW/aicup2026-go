"""Score-level (z-mean) fusion of bigbank90 PLDA dims, chosen on cross-fit fresh episodes
(runs/crossfit90_20261114_v1: z_128_256_384 +0.022 balanced vs rrf_256_768_1024, CI [0.006,0.039]).

stages: train (full bank d128/d384) | dev (scale screen+confirmation, post-Sinkhorn, temps cross-part) | export
Variants on dev: prod = fuse(prior, rrf3(256,768,1024), .85) [current official best],
                 zonly = zmean(128,256,384), zp15 = .15 z(prior) + .85 zmean, zp10 = .10 z(prior) + .90 zmean.
"""
from pathlib import Path
import sys, argparse, csv, json
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank90_reweight_20261114_v1 as rw
import export_bigbank90_sinkhorn_20261114_v1 as ex
import numpy as np
from threadpoolctl import threadpool_limits
f = rw.f; s = f.s; p = f.p; old = s.old; cap = f.cap; bb = f.bb; A = ex.A
RUN = bb.RUN / 'zfuse'; DIMS = (128, 256, 384)
CUR = p.ROOT / 'submissions/20261114_v2_bigbank90_reweight_sinkhorn_v1/submission_player.csv'; CUR_SHA = '7b92a84090f7de0753d659e4548d0176eda2b0d81f55dedb9c889a128115cd96'
OUT = p.ROOT / 'submissions/20261114_v3_bigbank90_zfuse_sinkhorn_v1'


def zmean(xs): return sum(A.z(x) for x in xs) / len(xs)


def train():
    ids = np.load(bb.RUN / 'train_indices.npy'); labels = np.load(bb.RUN / 'train_labels.npy'); raw = bb.lookup_views(ids, 90)
    for d in (128, 384):
        folder = bb.RUN / f'models/big_raw{d}_v90'; folder.mkdir(exist_ok=False); x = old.pre(raw, folder, True)
        r = p.fit_plda(x, labels, folder / 'PLDA.npz', dimension=d); p.b.write(folder / 'FIT_RECEIPT.json', dict(r, training_only=True)); print(d, r)


def variant(name, prior, comps, ql):
    if name == 'prod': return rw.fuse(prior, rw.big3(comps), ql, .85, .85)
    zn = zmean([comps[f'big_raw{d}_v90'] for d in DIMS])
    return {'zonly': zn, 'zp15': .15 * A.z(prior) + .85 * zn, 'zp10': .10 * A.z(prior) + .90 * zn}[name]


def dev():
    RUN.mkdir(exist_ok=True); parts = {}
    for part in ('screen', 'confirmation'):
        ep = p.b.read(s.RUN / f'EPISODE_{part}.json'); ids = np.load(s.RUN / 'cal_indices.npy'); allids = np.load(s.RUN / 'all_indices.npy'); raw = np.load(s.RUN / 'views/view90.npy', mmap_mode='r')[np.searchsorted(allids, ids)]
        comps = dict(np.load(bb.RUN / f'components_{part}.npz'))
        for d in (128, 384): comps[f'big_raw{d}_v90'] = old.component(bb.RUN / f'models/big_raw{d}_v90', raw, ids, ep)
        parts[part] = (ep, np.load(s.RUN / part / 'prior.npy'), comps, np.array([z['q'] for z in ep['questions']]), np.array([z['target'] for z in ep['questions']]))
    out = {}; names = ('prod', 'zonly', 'zp15', 'zp10'); S = {}
    for part, (ep, prior, comps, ql, tg) in parts.items():
        for n in names: S[part, n] = variant(n, prior, comps, ql)
    for part, (ep, prior, comps, ql, tg) in parts.items():
        other = 'confirmation' if part == 'screen' else 'screen'; oq, ot = parts[other][3], parts[other][4]
        for n in names:
            temps = A.calibrate(A.z(S[other, n]), ot, oq); L = A.new_transform(S[part, n], ql, temps)
            r_raw = cap.compare(S[part, n], S[part, 'prod'], ep, 'all'); r_bal = cap.compare(L, A.new_transform(S[part, 'prod'], ql, A.calibrate(A.z(S[other, 'prod']), ot, oq)), ep, 'all')
            out[f'{part}:{n}'] = {'raw': {k: r_raw[k] for k in ('macro_score_all', 'delta', 'CI95', 'by_games_delta')}, 'bal': {k: r_bal[k] for k in ('macro_score_all', 'baseline_macro_all', 'delta', 'CI95', 'by_games_delta')}}
            print(part, n, 'raw', round(r_raw['macro_score_all'], 4), 'd', round(r_raw['delta'], 4), '| bal', round(r_bal['macro_score_all'], 4), 'd', round(r_bal['delta'], 4), [round(x, 4) for x in r_bal['CI95']], {a: round(b, 4) for a, b in r_bal['by_games_delta'].items()})
    p.b.write(RUN / 'DEV_RESULTS.json', out)


def export(name):
    assert A.sha(CUR) == CUR_SHA; dest = RUN / f'export_{name}'; dest.mkdir(exist_ok=False); cur = A.rows(CUR); preds = dict(cur)
    Ss, tg, ql = [], [], []
    for part in ('screen', 'confirmation'):
        ep = p.b.read(s.RUN / f'EPISODE_{part}.json'); ids = np.load(s.RUN / 'cal_indices.npy'); allids = np.load(s.RUN / 'all_indices.npy'); raw = np.load(s.RUN / 'views/view90.npy', mmap_mode='r')[np.searchsorted(allids, ids)]
        comps = dict(np.load(bb.RUN / f'components_{part}.npz'))
        for d in (128, 384): comps[f'big_raw{d}_v90'] = old.component(bb.RUN / f'models/big_raw{d}_v90', raw, ids, ep)
        q = [z['q'] for z in ep['questions']]; Ss.append(variant(name, np.load(s.RUN / part / 'prior.npy'), comps, np.array(q))); tg += [z['target'] for z in ep['questions']]; ql += q
    temps = A.calibrate(A.z(np.vstack(Ss)), np.array(tg), np.array(ql)); print('temps', temps)
    ids = np.load(old.RUN / 'official_results/indices.npy'); raw = np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r')
    games, pools = p.b.load_official(); keys = p.b.read(p.b.RUN / 'official_stat/keys.json'); pos = {k: i for i, k in enumerate(keys)}; rep = {'variant': name, 'temps': temps, 'pools': {}}
    for pool in pools:
        ep = {'supports': [[pos[k] for k in pool.supports[a]] for a in pool.candidate_ids], 'questions': [{'indices': [pos[k] for k in z['keys']], 'q': z['num_games']} for z in pool.questions]}
        q = np.array([z['num_games'] for z in pool.questions]); comps = {}
        for d in (128, 256, 384, 768, 1024):
            c = ex.RUN / pool.name / f'big_raw{d}_v90.npy'
            comps[f'big_raw{d}_v90'] = np.load(c) if c.exists() else old.component(bb.RUN / f'models/big_raw{d}_v90', raw, ids, ep)
        par = {n: np.load(cap.RUN / f'official_predictions/{pool.name}/{n}.npy') for n in ('fused', 'component', 'oldglobal64')}
        w = cap.countweights(ep); oldneural = .5 * ex.R(par['oldglobal64']) + .5 * ex.R(par['component']); prior = (par['fused'] - w * oldneural) / (1 - w)
        assert np.allclose(rw.fuse(prior, rw.big3(comps), q, .85, .85), np.load(ex.RUN / pool.name / 'score.npy'), atol=1e-15, rtol=0)
        score = variant(name, prior, comps, q); L = A.new_transform(score, q, temps); order = np.argsort(-L, 1, kind='stable')[:, :5]
        curorder = np.array([[pool.candidate_ids.index(c) for c in cur[z['question_id']]] for z in pool.questions])
        for i, z in enumerate(pool.questions): preds[z['question_id']] = [pool.candidate_ids[j] for j in order[i]]
        np.save(dest / f'{pool.name}_score.npy', score, allow_pickle=False)
        rep['pools'][pool.name] = {'Top1_changed_vs_current_best': int((order[:, 0] != curorder[:, 0]).sum()), 'by_games': {str(n): int((order[q == n, 0] != curorder[q == n, 0]).sum()) for n in (1, 2, 3, 5)},
                                   'top1_in_current_top5': float(np.mean([order[i, 0] in curorder[i] for i in range(len(q))])), 'max_column_mass': float(np.exp(L).sum(0).max())}
        print(pool.name, rep['pools'][pool.name])
    sample = p.ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests/sample_submission_player_v2.csv'; qids = list(A.rows(sample).keys()); assert len(qids) == 1600 and set(qids) == set(preds)
    newq = {z['question_id'] for pool in pools for z in pool.questions}; assert all(preds[k] == cur[k] for k in qids if k not in newq)
    OUT.mkdir(exist_ok=False); out = OUT / 'submission_player.csv'
    with out.open('x', encoding='utf-8', newline='') as h:
        wr = csv.writer(h, lineterminator='\n'); wr.writerow(A.HEADER)
        for k in qids: wr.writerow([k] + preds[k])
    val = p.b.validate(out); assert val['valid'], val; assert A.rows(out) == preds
    rep.update({'created_utc': datetime.now(timezone.utc).isoformat(), 'CSV': str(out), 'SHA256': A.sha(out), 'bytes': out.stat().st_size, 'rows': 1600, 'format_check': val, 'parent_CSV': str(CUR), 'parent_SHA256': CUR_SHA,
                'parent_official': None, 'rows_changed_vs_parent': int(sum(preds[k] != cur[k] for k in qids)), 'old800_identical_to_parent': True, 'uploaded': False, 'source_SHA256': A.sha(__file__),
                'dev_results': p.b.read(RUN / 'DEV_RESULTS.json'), 'crossfit': str(p.ROOT / 'runs/crossfit90_20261114_v1/CROSSFIT2_RESULTS.json')})
    p.b.write(dest / 'RESULT.json', rep); p.b.write(OUT / 'RECEIPT.json', rep); print(rep['CSV'], rep['SHA256'], rep['rows_changed_vs_parent'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    with threadpool_limits(limits=4): {'train': train, 'dev': dev, 'export': lambda: export(*v.args)}[v.stage]()
