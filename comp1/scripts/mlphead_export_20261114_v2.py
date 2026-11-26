"""Export v2: same models/components as v5 (mlphead_export_20261114_v1), but neural = BW*z(PLDA6 zmean) + (1-BW)*z(zmean(d ens, lin ens)).
Evidence (mlphead_scorevar_20261114_v1, cached embeddings, 2 seeds/cfg): BW .2 vs .5 (v5) = g1 +0.0082 [0.002,0.015], confirm +0.0064 [0.0005,0.012].
Temperatures recalibrated on g1 grid + confirm c1 comps with the same combination. Parent = v5; old 800 rows copied.
Usage: export <cfgs comma> <nseeds> <grid_tag> <confirm_tag> <BW> <outname> [models_root]
"""
from pathlib import Path
import sys, argparse, csv
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlphead_export_20261114_v1 as e1
import numpy as np
import torch
from threadpoolctl import threadpool_limits
g = e1.g; h = e1.h; b2 = e1.b2; p = e1.p; old = e1.old; A = e1.A; cap = e1.cap; ex = e1.ex; z6 = e1.z6
RUN = h.RUN / 'export'; Z6 = e1.Z6
PARENT = p.ROOT / 'submissions/20261114_v5_mlphead_d_lin_s5_sinkhorn_v1/submission_player.csv'; PARENT_SHA = '801b606a8dae4d07690bfa07c6d49d64e3a55779579fe6e35ad98e83af4cda76'


def neural_of(base, enss, bw): return bw * A.z(base) + (1 - bw) * A.z(g.zm(enss))


def temps(cfgs, grid_tag, confirm_tag, bw):
    Zs, tg, ql = [], [], []
    for epf, comps in ((h.c1.RUN / 'EPISODES.json', h.RUN / f'grid_{grid_tag}'), (h.RUN / 'confirm/EPISODES.json', h.RUN / f'confirm/{confirm_tag}')):
        eps = p.b.read(epf)
        for k in range(3):
            V = np.load(comps / f'comps_f{k}.npz'); Zs.append(A.z(neural_of(V['base'], [V[f'{c}:ens'] for c in cfgs], bw))); tg += [z['target'] for z in eps[k]['questions']]; ql += [z['q'] for z in eps[k]['questions']]
    return A.calibrate(np.vstack(Zs), np.array(tg), np.array(ql))


def export(cfgnames, nseeds, grid_tag, confirm_tag, bw, outname, models_root):
    assert A.sha(PARENT) == PARENT_SHA; cfgs = cfgnames.split(','); dest = RUN / f'out_{outname}'; dest.mkdir(exist_ok=False); cur = A.rows(PARENT); preds = dict(cur)
    T = temps(cfgs, grid_tag, confirm_tag, bw); print('temps', T)
    ids = np.load(old.RUN / 'official_results/indices.npy'); raw = np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r'))
    embs = {}
    for c in cfgs:
        mdir = models_root / f'models_{c}'; rc = p.b.read(mdir / 'TRAIN_RECEIPT.json'); cfg = rc['cfg']; x = old.pre(raw, mdir); embs[c] = []
        for sd in range(nseeds):
            net = h.Net(x.shape[1], rc['players'], **cfg).cuda(); net.load_state_dict(torch.load(mdir / f'net_s{sd}.pt')); net.eval(); embs[c].append(h.embed(net, x)); del net
    games, pools = p.b.load_official(); keys = p.b.read(p.b.RUN / 'official_stat/keys.json'); pos = {k: i for i, k in enumerate(keys)}
    rep = {'cfgs': {c: p.b.read(models_root / f'models_{c}/TRAIN_RECEIPT.json')['cfg'] for c in cfgs}, 'models_root': str(models_root), 'nseeds': nseeds, 'BW': bw, 'temps': T, 'pools': {}}
    for pool in pools:
        ep = {'supports': [[pos[k] for k in pool.supports[a]] for a in pool.candidate_ids], 'questions': [{'indices': [pos[k] for k in z['keys']], 'q': z['num_games']} for z in pool.questions]}
        q = np.array([z['num_games'] for z in pool.questions])
        base = g.zm([np.load(Z6 / f'{pool.name}_big2_raw{d}.npy') for d in z6.DIMS])
        enss = [g.zm([h.setscores(e, ids, ep)[0] for e in embs[c]]) for c in cfgs]; ens = g.zm(enss); neural = neural_of(base, enss, bw)
        par = {n: np.load(cap.RUN / f'official_predictions/{pool.name}/{n}.npy') for n in ('fused', 'component', 'oldglobal64')}
        w = cap.countweights(ep); oldneural = .5 * ex.R(par['oldglobal64']) + .5 * ex.R(par['component']); prior = (par['fused'] - w * oldneural) / (1 - w)
        v4score = z6.PRIOR_W * A.z(prior) + (1 - z6.PRIOR_W) * (sum(A.z(np.load(Z6 / f'{pool.name}_big2_raw{d}.npy')) for d in z6.DIMS) / len(z6.DIMS))
        assert np.allclose(v4score, np.load(Z6 / f'{pool.name}_score.npy'), atol=1e-12, rtol=0)
        score = z6.PRIOR_W * A.z(prior) + (1 - z6.PRIOR_W) * neural; L = A.new_transform(score, q, T); order = np.argsort(-L, 1, kind='stable')[:, :5]
        curorder = np.array([[pool.candidate_ids.index(c) for c in cur[z['question_id']]] for z in pool.questions])
        for i, zq in enumerate(pool.questions): preds[zq['question_id']] = [pool.candidate_ids[j] for j in order[i]]
        np.save(dest / f'{pool.name}_ens.npy', ens, allow_pickle=False); np.save(dest / f'{pool.name}_score.npy', score, allow_pickle=False)
        rep['pools'][pool.name] = {'Top1_changed_vs_v5': int((order[:, 0] != curorder[:, 0]).sum()), 'by_games': {str(n): int((order[q == n, 0] != curorder[q == n, 0]).sum()) for n in (1, 2, 3, 5)},
                                   'top1_in_v5_top5': float(np.mean([order[i, 0] in curorder[i] for i in range(len(q))])), 'max_column_mass': float(np.exp(L).sum(0).max())}
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
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('cfgs'); ap.add_argument('nseeds', type=int); ap.add_argument('grid_tag'); ap.add_argument('confirm_tag'); ap.add_argument('bw', type=float); ap.add_argument('outname'); ap.add_argument('models_root', nargs='?', default=str(RUN)); v = ap.parse_args()
    with threadpool_limits(limits=4): export(v.cfgs, v.nseeds, v.grid_tag, v.confirm_tag, v.bw, v.outname, Path(v.models_root))
