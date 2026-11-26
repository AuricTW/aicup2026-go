"""Pool-level capacity constraint (Sinkhorn) on top of the current best fused scores.

Leakage/structure used (observed, not officially guaranteed):
  * old Public 1.0 reference: 400 questions -> 400 distinct targets (perfect permutation);
  * new pools: 400 questions / 600 candidates; Top1 of 2/3-game questions falls inside the
    5-game Top1 set *below* chance -> consistent with distinct targets per question.
So each candidate should carry at most ~1 unit of probability mass across a pool.

New pools: per-length temperature calibrated (max true-target log-likelihood) on local
screen+confirmation dev episodes (600 candidates x 10 support, clean proxy fused scores), then
capped Sinkhorn (cap 0.9, beta 1.0).  Old pools: z-scored H scores, T=0.1, doubly-stochastic
Sinkhorn (cap 1).  Ranks within each question follow the balanced probabilities.
No model refit, no Public labels used for fitting; old Public 1.0 key only for reference eval.
"""
from pathlib import Path
import sys, csv, json, hashlib
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import complete90_pilot_20261112_v1 as e  # noqa: E402
p = e.p
RUN = ROOT / 'runs/assign_sinkhorn_20261113_v1'
PARENT = ROOT / 'submissions/20261112_v2_complete90_mean_a050_all_v1/submission_player.csv'
PARENT_SHA = 'f2780f4f700a5def6a285ac366eaf843409a5b9cd29a209cbb5da3564a47507d'
OUT = ROOT / 'submissions/20261113_v2_sinkhorn_assign_v1'
H_PUBLIC = ROOT / 'runs/fisher_policy_pilot_20261105_v2/public_inputs/H.npy'
H_PRIVATE = ROOT / 'runs/trainval_refinement_20261105_v1/release/Private_H_scores.npy'
PRIV_INPUTS = ROOT / 'runs/trainval_refinement_20261105_v1/release/PRIVATE_INPUTS.json'
NEW_BETA, NEW_CAP, OLD_T = 1.0, 0.9, 0.1
HEADER = ['question_id', 'top_1', 'top_2', 'top_3', 'top_4', 'top_5']


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def z(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-12)


def logsoft(L):
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))


def sinkhorn(logP, cap, iters=1000):
    P = np.exp(logP)
    for _ in range(iters):
        c = P.sum(0)
        P = P * np.where(c > cap, cap / c, 1.0)
        P /= P.sum(1, keepdims=True)
    return P


def sid(s, tgt):
    r = (s > s[np.arange(len(s)), tgt][:, None]).sum(1)
    return np.where(r < 5, np.exp(-r), 0.0)


def dev_bank():
    qs, S = [], []
    for part in ('screen', 'confirmation'):
        ep = json.loads((e.RUN / f'EPISODE_{part}.json').read_text())
        qs += ep['questions']; S.append(np.load(e.RUN / part / 'baseline.npy'))
    return z(np.vstack(S)), np.array([q['target'] for q in qs]), np.array([q['q'] for q in qs])


def calibrate(Z, tg, ql):
    temps = {}
    for n in (1, 2, 3, 5):
        m = ql == n; grid = np.geomspace(0.05, 2, 200)
        ll = [logsoft(Z[m] / t)[np.arange(m.sum()), tg[m]].mean() for t in grid]
        temps[n] = float(grid[int(np.argmax(ll))])
    return temps


def new_transform(S, ql, temps):
    tt = np.array([temps[int(n)] for n in ql])[:, None]
    return np.log(sinkhorn(logsoft(NEW_BETA * logsoft(z(S) / tt)), NEW_CAP) + 1e-300)


def simulate(Z, tg, ql, temps, reps=200, seed=20261113):
    rng = np.random.default_rng(seed); rows = []
    for _ in range(reps):
        used = set(); sel = []
        for n in (5, 3, 2, 1):
            k = 0
            for i in rng.permutation(np.flatnonzero(ql == n)):
                if tg[i] in used or k == 100:
                    continue
                sel.append(i); used.add(tg[i]); k += 1
        sel = np.array(sel); X, T, Q = Z[sel], tg[sel], ql[sel]
        b = sid(X, T); v = sid(new_transform(X, Q, temps), T)
        rows.append([v[Q == n].mean() - b[Q == n].mean() for n in (1, 2, 3, 5)] + [v.mean() - b.mean(), b.mean(), v.mean(), (X.argmax(1) == T).mean(), (v == 1).mean()])
    a = np.array(rows)
    return {'reps': reps, 'questions_per_rep': 400, 'distinct_targets_per_rep': 400, 'candidates': 600,
            'delta_by_games_mean': dict(zip(['1', '2', '3', '5'], a[:, :4].mean(0).round(5).tolist())),
            'delta_mean': float(a[:, 4].mean()), 'delta_q025_q975': np.quantile(a[:, 4], [.025, .975]).round(5).tolist(), 'delta_min': float(a[:, 4].min()),
            'fraction_reps_positive': float((a[:, 4] > 0).mean()), 'base_mean': float(a[:, 5].mean()), 'candidate_mean': float(a[:, 6].mean()),
            'top1_base': float(a[:, 7].mean()), 'top1_candidate': float(a[:, 8].mean()),
            'note': 'random subsets of the same 2033-question dev bank (500 exposed targets, shared 600 candidates); reps are not independent'}


def rows(f):
    with open(f, encoding='utf-8', newline='') as h:
        return {r['question_id']: [r[f'top_{i}'] for i in range(1, 6)] for r in csv.DictReader(h)}


def main():
    RUN.mkdir(exist_ok=False)
    assert sha(PARENT) == PARENT_SHA
    parent = rows(PARENT); preds = dict(parent); report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'parent_CSV': str(PARENT), 'parent_SHA256': PARENT_SHA,
                                                           'params': {'new_beta': NEW_BETA, 'new_cap': NEW_CAP, 'old_T': OLD_T, 'old_cap': 1.0}}
    Z, tg, ql = dev_bank(); temps = calibrate(Z, tg, ql); report['new_temps'] = temps
    report['dev_simulation'] = simulate(Z, tg, ql, temps)
    print('temps', temps); print('sim', json.dumps(report['dev_simulation'], indent=1))
    # new pools
    games, pools = p.b.load_official(); report['pools'] = {}
    for pool in pools:
        S = np.load(e.RUN / 'official_predictions' / pool.name / 'fused.npy'); q = np.array([z_['num_games'] for z_ in pool.questions])
        base = np.argsort(-S, 1, kind='stable')[:, :5]
        assert all(parent[z_['question_id']] == [pool.candidate_ids[j] for j in base[i]] for i, z_ in enumerate(pool.questions))
        L = new_transform(S, q, temps); order = np.argsort(-L, 1, kind='stable')[:, :5]
        P = np.exp(L)
        for i, z_ in enumerate(pool.questions):
            preds[z_['question_id']] = [pool.candidate_ids[j] for j in order[i]]
        np.save(RUN / f'{pool.name}_balanced_logP.npy', L, allow_pickle=False)
        report['pools'][pool.name] = {'Top1_changed': int((order[:, 0] != base[:, 0]).sum()), 'Top5_set_changed': int(sum(set(a) != set(b) for a, b in zip(order, base))),
                                      'Top1_changed_by_games': {str(n): int((order[q == n, 0] != base[q == n, 0]).sum()) for n in (1, 2, 3, 5)},
                                      'distinct_top1_before': int(len(set(base[:, 0]))), 'distinct_top1_after': int(len(set(order[:, 0]))), 'max_column_mass': float(P.sum(0).max())}
    # old pools
    priv = json.loads(PRIV_INPUTS.read_text())
    for name, f, qids, cids in (('public', H_PUBLIC, [f'pub_q_{i:04}' for i in range(1, 401)], [f'pub_p_{i:04}' for i in range(1, 401)]),
                                ('private', H_PRIVATE, priv['question_ids'], priv['candidate_ids'])):
        H = np.load(f); base = np.argsort(-H, 1, kind='stable')[:, :5]
        assert all(parent[qid] == [cids[j] for j in base[i]] for i, qid in enumerate(qids)), name
        P = np.exp(logsoft(z(H) / OLD_T))
        for _ in range(2000):
            P /= P.sum(0, keepdims=True); P /= P.sum(1, keepdims=True)
        L = np.log(P + 1e-300); order = np.argsort(-L, 1, kind='stable')[:, :5]
        for i, qid in enumerate(qids):
            preds[qid] = [cids[j] for j in order[i]]
        np.save(RUN / f'{name}_balanced_logP.npy', L, allow_pickle=False)
        r = {'Top1_changed': int((order[:, 0] != base[:, 0]).sum()), 'Top5_set_changed': int(sum(set(a) != set(b) for a, b in zip(order, base))),
             'distinct_top1_before': int(len(set(base[:, 0]))), 'distinct_top1_after': int(len(set(order[:, 0])))}
        report['pools'][name] = r
    # write CSV in sample order
    sample = ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests/sample_submission_player_v2.csv'
    qids = list(rows(sample).keys()); assert len(qids) == 1600 and set(qids) == set(preds)
    OUT.mkdir(exist_ok=False); out = OUT / 'submission_player.csv'
    with out.open('x', encoding='utf-8', newline='') as h:
        w = csv.writer(h, lineterminator='\n'); w.writerow(HEADER)
        for k in qids:
            w.writerow([k] + preds[k])
    validation = p.b.validate(out); assert validation['valid'], validation
    assert rows(out) == preds
    report.update({'CSV': str(out), 'bytes': out.stat().st_size, 'SHA256': sha(out), 'rows': 1600, 'format_check': validation,
                   'rows_changed_vs_parent': int(sum(preds[k] != parent[k] for k in qids)), 'uploaded': False,
                   'Public_labels_used_for_fit': False, 'original_sealed400_untouched': True,
                   'source': str(Path(__file__)), 'source_SHA256': sha(__file__)})
    (RUN / 'RESULT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (OUT / 'RECEIPT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('CSV', 'SHA256', 'rows_changed_vs_parent', 'pools')}, indent=1))


if __name__ == '__main__':
    main()
