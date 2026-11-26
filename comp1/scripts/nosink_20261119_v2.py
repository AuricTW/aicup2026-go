"""v2: new-pool rows from a recipe export: single-game queries ranked by the raw recipe score, Sinkhorn column cap applied among the
remaining (q >= 2) queries only. Sanity: the full transform on all rows must reproduce the base file exactly.
usage: nosink_20261119_v2.py <base submission dir> <export dir (<pool>_score.npy, RESULT.json)> <out dir name> [raw sizes, default 1]"""
import sys, csv, hashlib, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import suppsplit_eval_20261115_v2 as s2
import assign_sinkhorn_20261113_v1 as A
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def main(base, exp, out, raw='1'):
    raw = {int(x) for x in raw.split(',')}; rows = list(csv.reader(open(ROOT / 'submissions' / base / 'submission_player.csv', encoding='utf-8'))); head, rows = rows[0], rows[1:]; assert len(rows) == 1600; idx = {r[0]: i for i, r in enumerate(rows)}
    T = {int(k): float(v) for k, v in json.load(open(Path(exp) / 'RESULT.json', encoding='utf-8'))['temps_used'].items()}; games, pools, keys, pos = s2.ss.official(); info = {'base': base, 'export': str(exp), 'raw_sizes': sorted(raw), 'temps': T, 'pools': {}}
    for pool in pools:
        if not pool.name.endswith('2'): continue
        S = np.load(Path(exp) / f'{pool.name}_score.npy'); q = np.array([z['num_games'] for z in pool.questions]); ids = pool.candidate_ids
        full = np.argsort(-A.new_transform(S, q, T), 1, kind='stable')[:, :5]; same = sum([ids[j] for j in full[i]] == rows[idx[z['question_id']]][1:] for i, z in enumerate(pool.questions)); assert same == len(q), (pool.name, 'base file not reproduced', same)
        m = ~np.isin(q, list(raw)); o = np.argsort(-S, 1, kind='stable')[:, :5]; o[m] = np.argsort(-A.new_transform(S[m], q[m], T), 1, kind='stable')[:, :5]; t = 0
        for i, z in enumerate(pool.questions): new = [z['question_id']] + [ids[j] for j in o[i]]; t += new[1] != rows[idx[z['question_id']]][1]; rows[idx[z['question_id']]] = new
        info['pools'][pool.name] = {'top1_changed_vs_base': int(t), 'top1_changed_q>=2_vs_base': int((o[m, 0] != full[m, 0]).sum())}
    d = ROOT / 'submissions' / out; d.mkdir(exist_ok=False); f = d / 'submission_player.csv'
    with open(f, 'w', newline='', encoding='utf-8') as h: w = csv.writer(h, lineterminator='\n'); w.writerow(head); w.writerows(rows)
    assert all(len(set(r[1:])) == 5 for r in rows); info['sha256'] = hashlib.sha256(f.read_bytes()).hexdigest(); json.dump(info, open(d / 'INFO.json', 'w'), indent=1); print(info)


if __name__ == '__main__':
    main(*sys.argv[1:])
