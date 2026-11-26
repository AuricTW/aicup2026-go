"""Support-split validation v2: EQUAL gallery sizes for every candidate.
v1 flaw: only the targets lost games (gallery 10-q) while non-targets kept 10; the centroid cosine divides by the centroid norm, which is
larger for smaller galleries, so targets were penalised (q=5 scored below q=3) and methods with a different sensitivity to gallery size
were compared unfairly.  v2 per pool and draw builds two episodes in which ALL 600 candidates hold out the same number of games:
  g7: 3 games held out (gallery 7), 400 targets with q in {1,2,3};   g5: 5 held out (gallery 5), 400 targets with q in {1,2,3,5}.
Galleries are smaller than the real 10, so absolute scores are pessimistic; comparisons are fair.
Same stages / CLI as v1 (build R | heads | seq | report); outputs in runs/suppsplit_20261115_v2/.
"""
from pathlib import Path
import sys, json, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import suppsplit_eval_20261115_v1 as ss
import numpy as np
from threadpoolctl import threadpool_limits
p = ss.p; RUN = p.ROOT / 'runs/suppsplit_20261115_v2'; SEED = 2026111507


def episodes(R):
    RUN.mkdir(exist_ok=False); games, pools, keys, pos = ss.official(); rng = np.random.default_rng(SEED); out = {}
    for pool in pools:
        out[pool.name] = []
        for r in range(R):
            for held, qset in ((3, (1, 2, 3)), (5, (1, 2, 3, 5))):
                full = [[pos[k] for k in pool.supports[c]] for c in pool.candidate_ids]; sup, ho = [], []
                for s in full: hj = set(rng.choice(10, held, replace=False).tolist()); sup.append([x for j, x in enumerate(s) if j not in hj]); ho.append([x for j, x in enumerate(s) if j in hj])
                tgt = rng.permutation(len(full))[:400]; qn = np.resize(np.array(qset), 400)[rng.permutation(400)]
                qs = [{'indices': sorted(rng.choice(ho[t], int(q), replace=False).tolist()), 'q': int(q), 'target': int(t)} for t, q in zip(tgt, qn)]
                out[pool.name].append({'supports': sup, 'questions': qs, 'gallery': 10 - held})
    (RUN / 'EPISODES.json').write_text(json.dumps(out), encoding='utf-8'); return out


ss.RUN = RUN; ss.episodes = episodes

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    with threadpool_limits(limits=4):
        if v.stage == 'build': ss.build(int(v.args[0]))
        elif v.stage == 'heads': ss.heads(*v.args)
        elif v.stage == 'seq': ss.seq(v.args[0], v.args[1:])
        else: ss.report([float(x) for x in v.args] or [.05, .1, .2, .3], None)
