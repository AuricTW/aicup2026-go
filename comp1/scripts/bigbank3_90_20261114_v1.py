"""Bigbank v3: same players/exclusions as bigbank2, but up to PER_OWNER games per owner (bigbank2 capped at 64).
Reuses bigbank2's prepare/package/accept code with patched module globals (new RUN, larger cap, bigbank2 results as cache).
Only the extra games need replay. Stages: prepare | package CHUNK | accept CHUNK SHA SIZE
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank2_90_20261114_v1 as b2
import numpy as np
from threadpoolctl import threadpool_limits
p = b2.p; s = b2.s; bb = b2.bb; old = b2.old
B2RUN = b2.RUN; RUN = p.ROOT / 'runs/bigbank3_90_20261114_v1'; PER_OWNER = 160
b2.CACHES = b2.CACHES + sorted(B2RUN.glob('results_c*'))
b2.RUN = RUN; b2.PER_OWNER = PER_OWNER; b2.TIER = 64  # tier A = first 64 per owner (mostly bigbank2-cached), tier B = extra


def lookup90(indices):
    srcs = [(np.load(s.RUN / 'all_indices.npy'), np.load(s.RUN / 'views/view90.npy', mmap_mode='r'))]
    for d in [bb.RUN / 'dev_results', old.RUN / 'dev_results'] + sorted(B2RUN.glob('results_c*')) + sorted(RUN.glob('results_c*')): srcs.append((np.load(d / 'indices.npy'), np.load(d / 'view90.npy', mmap_mode='r')))
    pos = {}
    for n, (ids, _) in enumerate(srcs):
        for j, i in enumerate(ids): pos.setdefault(int(i), (n, j))
    miss = [int(i) for i in indices if int(i) not in pos]
    if miss: raise KeyError(f'{len(miss)} games without features')
    out = np.empty((len(indices), 2339), np.float32)
    for k, i in enumerate(indices): n, j = pos[int(i)]; out[k] = srcs[n][1][j]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': b2.prepare, 'package': lambda: b2.package(*v.args), 'accept': lambda: b2.accept(*v.args)}[v.stage]()
