"""Export: bigbank2 full-bank MLP head (cfg from mlphead_crossfit v2 CFGS) seed ensemble + v2 6-dim PLDA z-mean + 0.10 z(prior), capped Sinkhorn.
Neural = zmean([zmean6 PLDA, zmean(cos over seeds)]). Temperatures calibrated on the cross-fit episodes (g1 split + confirmation split),
using the same neural combination (held-out players). Old 800 rows copied from the current best.
Usage: train <cfg> <nseeds> | export <cfg> <nseeds> <grid_tag> <confirm_tag>
"""
from pathlib import Path
import sys, argparse, csv
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlphead_crossfit_20261114_v2 as g
import bigbank2_zfuse6_20261114_v1 as z6
import numpy as np
import torch
from threadpoolctl import threadpool_limits
h = g.h; b2 = g.b2; p = g.p; old = g.old; A = g.A; cap = z6.cap; ex = z6.ex; rw = z6.rw
RUN = h.RUN / 'export'; PARENT = p.ROOT / 'submissions/20261114_v4_bigbank2_zfuse6_sinkhorn_v1/submission_player.csv'; PARENT_SHA = '594a4bba7859b76069cebd30f5f41a5fd1b6bb4155a8e072cfb6037d8334bbd6'
Z6 = b2.RUN / 'export_zfuse6'


def train(cfgname, nseeds):
    dest = RUN / f'models_{cfgname}'; dest.mkdir(parents=True, exist_ok=False); cfg = dict(g.BASE, **g.CFGS[cfgname])
    ids = np.load(b2.RUN / 'train_indices.npy'); labels = np.load(b2.RUN / 'train_labels.npy'); x = old.pre(b2.lookup90(ids), dest, True)
    for sd in range(nseeds):
        net = h.train_net(x, labels, cfg, 2026111490 + sd); torch.save(net.state_dict(), dest / f'net_s{sd}.pt'); del net; torch.cuda.empty_cache()
    p.b.write(dest / 'TRAIN_RECEIPT.json', {'cfg': cfg, 'nseeds': nseeds, 'games': int(len(ids)), 'players': int(len(np.unique(labels))), 'created_utc': datetime.now(timezone.utc).isoformat()})


def neural_of(base, enss): return g.zm([base, g.zm(enss)])


def temps(cfgs, grid_tag, confirm_tag):
    Zs, tg, ql = [], [], []
    for epf, comps in ((h.c1.RUN / 'EPISODES.json', h.RUN / f'grid_{grid_tag}'), (h.RUN / 'confirm/EPISODES.json', h.RUN / f'confirm/{confirm_tag}')):
        eps = p.b.read(epf)
        for k in range(3):
            V = np.load(comps / f'comps_f{k}.npz'); Zs.append(A.z(neural_of(V['base'], [V[f'{c}:ens'] for c in cfgs]))); tg += [z['target'] for z in eps[k]['questions']]; ql += [z['q'] for z in eps[k]['questions']]
    return A.calibrate(np.vstack(Zs), np.array(tg), np.array(ql))


def export(cfgnames, nseeds, grid_tag, confirm_tag):
    assert A.sha(PARENT) == PARENT_SHA; cfgs = cfgnames.split(','); tagname = f"{'_'.join(cfgs)}_s{nseeds}"; dest = RUN / f'out_{tagname}'; dest.mkdir(exist_ok=False); cur = A.rows(PARENT); preds = dict(cur)
    T = temps(cfgs, grid_tag, confirm_tag); print('temps', T); ncls = int(len(np.unique(np.load(b2.RUN / 'train_labels.npy'))))
    ids = np.load(old.RUN / 'official_results/indices.npy'); raw = np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r'))
    embs = {}
    for c in cfgs:
        mdir = RUN / f'models_{c}'; cfg = p.b.read(mdir / 'TRAIN_RECEIPT.json')['cfg']; x = old.pre(raw, mdir); embs[c] = []
        for sd in range(nseeds):
            net = h.Net(x.shape[1], ncls, **cfg).cuda(); net.load_state_dict(torch.load(mdir / f'net_s{sd}.pt')); net.eval(); embs[c].append(h.embed(net, x)); del net
    games, pools = p.b.load_official(); keys = p.b.read(p.b.RUN / 'official_stat/keys.json'); pos = {k: i for i, k in enumerate(keys)}; rep = {'cfgs': {c: p.b.read(RUN / f'models_{c}/TRAIN_RECEIPT.json')['cfg'] for c in cfgs}, 'nseeds': nseeds, 'temps': T, 'pools': {}}
    for pool in pools:
        ep = {'supports': [[pos[k] for k in pool.supports[a]] for a in pool.candidate_ids], 'questions': [{'indices': [pos[k] for k in z['keys']], 'q': z['num_games']} for z in pool.questions]}
        q = np.array([z['num_games'] for z in pool.questions])
        base = g.zm([np.load(Z6 / f'{pool.name}_big2_raw{d}.npy') for d in z6.DIMS])
        enss = [g.zm([h.setscores(e, ids, ep)[0] for e in embs[c]]) for c in cfgs]; ens = g.zm(enss); neural = neural_of(base, enss)
        par = {n: np.load(cap.RUN / f'official_predictions/{pool.name}/{n}.npy') for n in ('fused', 'component', 'oldglobal64')}
        w = cap.countweights(ep); oldneural = .5 * ex.R(par['oldglobal64']) + .5 * ex.R(par['component']); prior = (par['fused'] - w * oldneural) / (1 - w)
        v4score = z6.PRIOR_W * A.z(prior) + (1 - z6.PRIOR_W) * (sum(A.z(np.load(Z6 / f'{pool.name}_big2_raw{d}.npy')) for d in z6.DIMS) / len(z6.DIMS))
        assert np.allclose(v4score, np.load(Z6 / f'{pool.name}_score.npy'), atol=1e-12, rtol=0)
        score = z6.PRIOR_W * A.z(prior) + (1 - z6.PRIOR_W) * neural; L = A.new_transform(score, q, T); order = np.argsort(-L, 1, kind='stable')[:, :5]
        curorder = np.array([[pool.candidate_ids.index(c) for c in cur[z['question_id']]] for z in pool.questions])
        for i, zq in enumerate(pool.questions): preds[zq['question_id']] = [pool.candidate_ids[j] for j in order[i]]
        np.save(dest / f'{pool.name}_ens.npy', ens, allow_pickle=False); np.save(dest / f'{pool.name}_score.npy', score, allow_pickle=False)
        rep['pools'][pool.name] = {'Top1_changed_vs_v4': int((order[:, 0] != curorder[:, 0]).sum()), 'by_games': {str(n): int((order[q == n, 0] != curorder[q == n, 0]).sum()) for n in (1, 2, 3, 5)},
                                   'top1_in_v4_top5': float(np.mean([order[i, 0] in curorder[i] for i in range(len(q))])), 'max_column_mass': float(np.exp(L).sum(0).max()),
                                   'ens_vs_base_top1_agree': float(np.mean(ens.argmax(1) == base.argmax(1)))}
        print(pool.name, rep['pools'][pool.name], flush=True)
    sample = p.ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests/sample_submission_player_v2.csv'; qids = list(A.rows(sample).keys()); assert len(qids) == 1600 and set(qids) == set(preds)
    newq = {zq['question_id'] for pool in pools for zq in pool.questions}; assert all(preds[k] == cur[k] for k in qids if k not in newq)
    outdir = p.ROOT / f'submissions/20261114_v5_mlphead_{tagname}_sinkhorn_v1'; outdir.mkdir(exist_ok=False); out = outdir / 'submission_player.csv'
    with out.open('x', encoding='utf-8', newline='') as fh:
        wr = csv.writer(fh, lineterminator='\n'); wr.writerow(A.HEADER)
        for k in qids: wr.writerow([k] + preds[k])
    val = p.b.validate(out); assert val['valid'], val; assert A.rows(out) == preds
    rep.update({'created_utc': datetime.now(timezone.utc).isoformat(), 'CSV': str(out), 'SHA256': A.sha(out), 'bytes': out.stat().st_size, 'rows': 1600, 'format_check': val, 'parent_CSV': str(PARENT), 'parent_SHA256': PARENT_SHA,
                'parent_official': None, 'rows_changed_vs_parent': int(sum(preds[k] != cur[k] for k in qids)), 'old800_identical_to_parent': True, 'uploaded': False, 'source_SHA256': A.sha(__file__),
                'evidence': {'g1_split_dlin_bal_delta': '+0.0451 [0.0339,0.0565]', 'confirm_split_base+allens': p.b.read(h.RUN / f'confirm/{confirm_tag}/RESULT.json')['results'].get('base+allens')}})
    p.b.write(dest / 'RESULT.json', rep); p.b.write(outdir / 'RECEIPT.json', rep); print(rep['CSV'], rep['SHA256'], rep['rows_changed_vs_parent'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    with threadpool_limits(limits=4):
        if v.stage == 'train': train(v.args[0], int(v.args[1]))
        else: export(v.args[0], int(v.args[1]), v.args[2], v.args[3])
