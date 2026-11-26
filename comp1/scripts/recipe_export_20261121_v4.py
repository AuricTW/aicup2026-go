"""v4 (2026-11-21) = recipe_export_20261117_v3.py + components hg_top<k> / hgm_top<k> (gallery-based hubness, hubgal_20261121_v1.py).
The hubness components are computed from the galleries of the episode being scored (validation episode for the parity check, the
official galleries for the real pools) with the same function that wrote the validation extras; nothing else changes (v3 is imported
unchanged and only its Comp class is extended).  Usage and options are those of v3."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import recipe_export_20261117_v3 as v3
import hubgal_20261121_v1 as hg
from threadpoolctl import threadpool_limits


class Comp4(v3.Comp):
    M = None

    def score(self, name, ep):
        if not name.startswith(('hg_', 'hgm_')): return super().score(name, ep)
        if Comp4.M is None: Comp4.M, _ = hg.lf.models(hg.GROUPS)
        q = np.array([z['q'] for z in ep['questions']]); return hg.comps(hg.hub(Comp4.M, np.array(ep['supports'])), q)[name]


v3.Comp = Comp4

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['eval', 'export', 'dry']); ap.add_argument('args', nargs='+'); ap.add_argument('--seq', action='append', default=[]); ap.add_argument('--temp-mult', type=float, default=1.0); v = ap.parse_args()
    for s in v.seq: n, fs = s.split('=', 1); v3.SEQ[n] = [Path(f) for f in fs.split(',')]
    with threadpool_limits(limits=6):
        if v.stage == 'eval': v3.validation(v.args[0])
        else: v3.export(v.args[0], v.args[1], v.stage == 'dry', v.temp_mult)
