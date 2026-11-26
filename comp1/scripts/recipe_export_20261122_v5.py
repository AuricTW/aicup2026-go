"""v5 (2026-11-22) = recipe_export_20261121_v4.py + components ogq1_<variant> (opponent co-play graph, oppgraph_20261122_v1.py,
applied to SINGLE-GAME questions only: rows of questions with 2+ games are zero, so after the row z-score the term vanishes there --
same device as hgm_ for hubness).  The component is rebuilt from the galleries of the episode being scored (validation episode for
the parity check, the official 10-game galleries for the real pools) with the function that wrote the validation extras.
Validation extras: `ogq1` stage writes extra/ogq1v_<pool>_d<r>.npz (keys ogq1_<variant>) from the ogv_ extras of the same episodes.
Usage and options are those of v3 / v4, plus:  recipe_export_20261122_v5.py ogq1 <variant> (old / mirror / mirror2 extras)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import recipe_export_20261121_v4 as v4
import oppgraph_20261122_v1 as og
from threadpoolctl import threadpool_limits
v3 = v4.v3


def q1only(S, q):
    S = np.array(S, dtype=np.float64); S[np.asarray(q) != 1] = 0.0; return S


class Comp5(v4.Comp4):
    def score(self, name, ep):
        if not name.startswith('ogq1_'): return super().score(name, ep)
        q = np.array([z['q'] for z in ep['questions']]); return q1only(og.matrix(name[5:], ep['supports'], [z['indices'] for z in ep['questions']]), q)


v3.Comp = Comp5


def write_extras(variant):
    import json, suppsplit_eval_20261115_v2 as s2, mirror_val_20261121_v1 as mv
    sets = [('old', ((p, r, e) for p, r, e, f in s2.ss.each()), s2.ss.RUN / 'extra')]
    for run in ('runs/mirrorval_20261121_v1', 'runs/mirrorval2_20261122_v1'):
        mv.MRUN = mv.ROOT / run; sets.append((run, list(mv.each()), mv.MRUN / 'extra'))
    for name, it, dest in sets:
        for pool, r, ep in it:
            S = np.load(dest / f'ogv_{pool}_d{r}.npz')[variant]; q = np.array([z['q'] for z in ep['questions']])
            np.savez(dest / f'ogq1v_{pool}_d{r}.npz', **{f'ogq1_{variant}': q1only(S, q)})
        print('ogq1 extras written for', name, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'ogq1': write_extras(sys.argv[2]); sys.exit()
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['eval', 'export', 'dry']); ap.add_argument('args', nargs='+'); ap.add_argument('--seq', action='append', default=[]); ap.add_argument('--temp-mult', type=float, default=1.0); v = ap.parse_args()
    for s in v.seq: n, fs = s.split('=', 1); v3.SEQ[n] = [Path(f) for f in fs.split(',')]
    with threadpool_limits(limits=6):
        if v.stage == 'eval': v3.validation(v.args[0])
        else: v3.export(v.args[0], v.args[1], v.stage == 'dry', v.temp_mult)
