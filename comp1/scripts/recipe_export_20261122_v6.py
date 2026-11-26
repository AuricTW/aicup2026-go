"""v6 (2026-11-22) = recipe_export_20261122_v5.py + component og5q1_cos (opponent co-play graph with token-net o5 opponent posteriors,
oppgraph_20261122_v2.py; single-game questions only, rows of 2+-game questions are zero).  Doc section 37.3: OG1 replaced OG0 under the
pre-registered rule (>= 0 against OG0 on the 24 old + mirror episodes and on mirror2).
Extra stage: recipe_export_20261122_v6.py og5q1  -> extra/og5q1v_<pool>_d<r>.npz (key og5q1_cos) for old / mirror / mirror2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import recipe_export_20261122_v5 as v5
import oppgraph_20261122_v2 as og2
from threadpoolctl import threadpool_limits
v3 = v5.v3


class Comp6(v5.Comp5):
    def score(self, name, ep):
        if name != 'og5q1_cos': return super().score(name, ep)
        q = np.array([z['q'] for z in ep['questions']]); return v5.q1only(og2.matrix(ep['supports'], [z['indices'] for z in ep['questions']]), q)


v3.Comp = Comp6


def write_extras():
    import suppsplit_eval_20261115_v2 as s2, mirror_val_20261121_v1 as mv
    sets = [('old', [(p, r, e) for p, r, e, f in s2.ss.each()], s2.ss.RUN / 'extra')]
    for run in ('runs/mirrorval_20261121_v1', 'runs/mirrorval2_20261122_v1'):
        mv.MRUN = mv.ROOT / run; sets.append((run, list(mv.each()), mv.MRUN / 'extra'))
    for name, it, dest in sets:
        for pool, r, ep in it:
            S = np.load(dest / f'og5v_{pool}_d{r}.npz')['og5_cos']; q = np.array([z['q'] for z in ep['questions']]); np.savez(dest / f'og5q1v_{pool}_d{r}.npz', og5q1_cos=v5.q1only(S, q))
        print('og5q1 extras written for', name, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'og5q1': write_extras(); sys.exit()
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['eval', 'export', 'dry']); ap.add_argument('args', nargs='+'); ap.add_argument('--seq', action='append', default=[]); ap.add_argument('--temp-mult', type=float, default=1.0); v = ap.parse_args()
    for s in v.seq: n, fs = s.split('=', 1); v3.SEQ[n] = [Path(f) for f in fs.split(',')]
    with threadpool_limits(limits=6):
        if v.stage == 'eval': v3.validation(v.args[0])
        else: v3.export(v.args[0], v.args[1], v.stage == 'dry', v.temp_mult)
