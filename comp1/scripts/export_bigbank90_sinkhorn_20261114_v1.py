"""Export: bigbank90 re-weighted fusion (confirmed) + per-length recalibrated capped Sinkhorn on new pools.

New pools (public2/private2): score = (1-w_q)*prior_official + w_q*big3_official, (w_Q1,w_Q235) from the
confirmed reweight winner; prior_official recovered exactly from the metric768 parent (same algebra as
export_complete90). Sinkhorn: temps recalibrated on the scale-run dev bank of the *new* score; beta 1, cap 0.9.
Old pools (pub/priv): rows copied unchanged from the current best CSV.
"""
from pathlib import Path
import sys, csv, json
from datetime import datetime, timezone
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank90_reweight_20261114_v1 as rw
import assign_sinkhorn_20261113_v1 as A
import numpy as np
from threadpoolctl import threadpool_limits
f = rw.f; s = f.s; p = f.p; old = s.old; cap = f.cap; bb = f.bb; e = old
CUR = p.ROOT / 'submissions/20261113_v2_sinkhorn_assign_v1/submission_player.csv'
CUR_SHA = 'a3b2e18c5b2b14c2f8aa87d48aeee7267351169cf3af0fb15d9c7bf2a1f24e50'
RUN = p.ROOT / 'runs/export_bigbank90_sinkhorn_20261114_v1'
OUT = p.ROOT / 'submissions/20261114_v2_bigbank90_reweight_sinkhorn_v1'
R = lambda x: 1 / (60 + p.b.ranks(x))


def dev_bank(w1, wn):
    S, tg, ql = [], [], []
    for part in ('screen', 'confirmation'):
        ep = p.b.read(s.RUN / f'EPISODE_{part}.json'); prior = np.load(s.RUN / part / 'prior.npy'); q = [z['q'] for z in ep['questions']]
        S.append(rw.fuse(prior, rw.big3(np.load(bb.RUN / f'components_{part}.npz')), q, w1, wn)); tg += [z['target'] for z in ep['questions']]; ql += q
    return A.z(np.vstack(S)), np.array(tg), np.array(ql)


def main():
    conf = p.b.read(rw.RUN / 'CONFIRMATION_RESULT.json'); assert conf['passed']; w1, wn = conf['w1'], conf['wn']
    assert A.sha(CUR) == CUR_SHA; RUN.mkdir(exist_ok=False)
    cur = A.rows(CUR); preds = dict(cur)
    Z, tg, ql = dev_bank(w1, wn); temps = A.calibrate(Z, tg, ql); sim = A.simulate(Z, tg, ql, temps)
    print('temps', temps); print('sim', json.dumps(sim, indent=1))
    rep = {'created_utc': datetime.now(timezone.utc).isoformat(), 'w1': w1, 'wn': wn, 'confirmation': {k: conf[k] for k in ('delta', 'CI95', 'by_games_delta', 'Top1', 'baseline_Top1', 'macro_score_all', 'baseline_macro_all', 'checks')},
           'sinkhorn': {'beta': A.NEW_BETA, 'cap': A.NEW_CAP, 'temps': temps, 'dev_simulation_vs_unbalanced_new_score': sim}, 'pools': {}}
    ids = np.load(e.RUN / 'official_results/indices.npy'); raw = np.load(e.RUN / 'official_results/view90.npy', mmap_mode='r')
    games, pools = p.b.load_official(); keys = p.b.read(p.b.RUN / 'official_stat/keys.json'); pos = {k: i for i, k in enumerate(keys)}
    for pool in pools:
        ep = {'supports': [[pos[k] for k in pool.supports[a]] for a in pool.candidate_ids], 'questions': [{'indices': [pos[k] for k in z['keys']], 'q': z['num_games']} for z in pool.questions]}
        q = np.array([z['num_games'] for z in pool.questions]); dd = RUN / pool.name; dd.mkdir()
        comps = {}
        for d in (256, 768, 1024):
            folder = bb.RUN / f'models/big_raw{d}_v90'; v = old.component(folder, raw, ids, ep)
            rev = np.arange(len(ids))[::-1]; assert np.array_equal(v, old.component(folder, raw[rev], ids[rev], ep))
            comps[f'big_raw{d}_v90'] = v; np.save(dd / f'big_raw{d}_v90.npy', v, allow_pickle=False)
        big3 = rw.big3(comps)
        par = {n: np.load(cap.RUN / f'official_predictions/{pool.name}/{n}.npy') for n in ('fused', 'component', 'oldglobal64')}
        w = cap.countweights(ep); oldneural = .5 * R(par['oldglobal64']) + .5 * R(par['component']); prior = (par['fused'] - w * oldneural) / (1 - w)
        pilot = np.load(e.RUN / f'official_predictions/{pool.name}/fused.npy'); pnew = np.load(e.RUN / f'official_predictions/{pool.name}/newneural.npy')
        recon = (1 - w) * prior + w * (.5 * oldneural + .5 * pnew); err = float(np.abs(recon - pilot).max()); assert err < 1e-12, err
        score = rw.fuse(prior, big3, q, w1, wn); np.save(dd / 'score.npy', score, allow_pickle=False)
        L = A.new_transform(score, q, temps); order = np.argsort(-L, 1, kind='stable')[:, :5]; raworder = np.argsort(-score, 1, kind='stable')[:, :5]
        curorder = np.array([[pool.candidate_ids.index(c) for c in cur[z['question_id']]] for z in pool.questions])
        pilotorder = np.argsort(-pilot, 1, kind='stable')[:, :5]
        for i, z in enumerate(pool.questions): preds[z['question_id']] = [pool.candidate_ids[j] for j in order[i]]
        np.save(dd / 'balanced_logP.npy', L, allow_pickle=False)
        byq = lambda a, b: {str(n): int((a[q == n, 0] != b[q == n, 0]).sum()) for n in (1, 2, 3, 5)}
        rep['pools'][pool.name] = {'prior_recon_maxabs': err, 'Top1_changed_vs_current_best': int((order[:, 0] != curorder[:, 0]).sum()), 'Top1_changed_vs_current_best_by_games': byq(order, curorder),
                                   'Top1_changed_raw_vs_pilot_unbalanced': int((raworder[:, 0] != pilotorder[:, 0]).sum()), 'Top1_changed_by_sinkhorn': int((order[:, 0] != raworder[:, 0]).sum()),
                                   'distinct_top1_raw': int(len(set(raworder[:, 0]))), 'distinct_top1_after': int(len(set(order[:, 0]))), 'max_column_mass': float(np.exp(L).sum(0).max()),
                                   'top1_in_current_top5': float(np.mean([order[i, 0] in curorder[i] for i in range(len(q))]))}
        print(pool.name, rep['pools'][pool.name])
    sample = p.ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests/sample_submission_player_v2.csv'
    qids = list(A.rows(sample).keys()); assert len(qids) == 1600 and set(qids) == set(preds)
    newq = {z['question_id'] for pool in pools for z in pool.questions}; assert len(newq) == 800
    assert all(preds[k] == cur[k] for k in qids if k not in newq)
    OUT.mkdir(exist_ok=False); out = OUT / 'submission_player.csv'
    with out.open('x', encoding='utf-8', newline='') as h:
        wr = csv.writer(h, lineterminator='\n'); wr.writerow(A.HEADER)
        for k in qids: wr.writerow([k] + preds[k])
    val = p.b.validate(out); assert val['valid'], val; assert A.rows(out) == preds
    rep.update({'CSV': str(out), 'bytes': out.stat().st_size, 'SHA256': A.sha(out), 'rows': 1600, 'format_check': val, 'parent_CSV': str(CUR), 'parent_SHA256': CUR_SHA, 'parent_official': None,
                'rows_changed_vs_parent': int(sum(preds[k] != cur[k] for k in qids)), 'old800_rows_identical_to_parent': True, 'uploaded': False, 'Public_labels_used_for_fit': False,
                'original_sealed400_untouched': True, 'train_bank': {'players': 2105, 'games': 57389}, 'source': str(Path(__file__)), 'source_SHA256': A.sha(__file__),
                'limitations': ['dev confirmation identities exposed: read once by bigbank90_fusion (failed, Q1 -0.0013) before this reweight gate', 'dev prior is clean proxy, production prior exact parent algebra',
                                'Sinkhorn official gain was ~1/3 of sim last time', 'no official delta forecast from local gain']})
    p.b.write(RUN / 'RESULT.json', rep); p.b.write(OUT / 'RECEIPT.json', rep)
    print(json.dumps({k: rep[k] for k in ('CSV', 'SHA256', 'rows_changed_vs_parent')}, indent=1))


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
