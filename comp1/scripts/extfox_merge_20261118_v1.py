"""Merge the external (Fox) token chunks into the label / move-code files of the token-net trainer (tokhead_vm_20261118_v5.py).
External accounts become extra classes after the competition players (y = max competition y + 1 + dense account index); nothing is
matched against competition games.  Only chunks that are complete (RECEIPT.json + mv.npy, no LOCAL marker) are used; accounts with fewer
than --min-games extracted games are left out.
usage: extfox_merge_20261118_v1.py <labels_all.npz> <tokmv_all.npz> <labels_ext.npz> <out dir> <ext token dir> [...] [--min-games 4] [--max-games 0]
writes <out>/labels.npz (ids, y, black, off_black), <out>/tokmv.npz (ids sorted, mv [n, 90]), <out>/DIRS.txt (token dirs), <out>/LATE.txt, MERGE.json
"""
import sys, os, json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('labels'); ap.add_argument('tokmv'); ap.add_argument('ext'); ap.add_argument('out'); ap.add_argument('dirs', nargs='+'); ap.add_argument('--min-games', type=int, default=4); ap.add_argument('--max-games', type=int, default=0); v = ap.parse_args()
    os.makedirs(v.out, exist_ok=False); L = dict(np.load(v.labels)); M = dict(np.load(v.tokmv)); E = dict(np.load(v.ext)); row = {int(i): j for j, i in enumerate(E['ids'])}
    dirs = [d for d in v.dirs if os.path.exists(os.path.join(d, 'RECEIPT.json')) and os.path.exists(os.path.join(d, 'mv.npy')) and not os.path.exists(os.path.join(d, 'LOCAL')) and os.path.exists(os.path.join(d.rstrip('/\\') + '_late', 'RECEIPT.json'))]
    ids = np.concatenate([np.load(os.path.join(d, 'indices.npy')) for d in dirs]).astype(np.int64); mv = np.concatenate([np.load(os.path.join(d, 'mv.npy')) for d in dirs]); ne = np.concatenate([np.load(os.path.join(d, 'tok_n.npy')) for d in dirs])
    assert len(np.unique(ids)) == len(ids) and ids.min() > L['ids'].max(), 'external ids must be unique and above the competition ids'
    r = np.array([row[int(i)] for i in ids]); cls = E['cls'][r]; black = E['black'][r]; keep = ne > 0
    if v.max_games:      # the export is round-robin, so the first games of an account are the ones with the smallest ids
        o = np.lexsort((ids, cls)); rank = np.empty(len(ids), np.int64); c_sorted = cls[o]; first = np.r_[0, np.flatnonzero(c_sorted[1:] != c_sorted[:-1]) + 1]; rank[o] = np.arange(len(ids)) - np.repeat(first, np.diff(np.r_[first, len(ids)])); keep &= rank < v.max_games
    cnt = np.bincount(cls[keep], minlength=int(E['cls'].max()) + 1); keep &= cnt[cls] >= v.min_games; u, dense = np.unique(cls[keep], return_inverse=True); y0 = int(L['y'].max()) + 1
    y_ext = np.full(len(ids), -1, np.int32); y_ext[keep] = y0 + dense
    out_ids = np.concatenate([L['ids'].astype(np.int64), ids]); np.savez(os.path.join(v.out, 'labels.npz'), ids=out_ids, y=np.concatenate([L['y'].astype(np.int32), y_ext]), black=np.concatenate([L['black'], black]), off_black=L['off_black'])
    o = np.argsort(ids); assert (np.diff(M['ids']) > 0).all(); np.savez(os.path.join(v.out, 'tokmv.npz'), ids=np.concatenate([M['ids'].astype(np.int64), ids[o]]), mv=np.concatenate([M['mv'], mv[o]]))
    open(os.path.join(v.out, 'DIRS.txt'), 'w', newline='\n').write(' '.join(dirs)); open(os.path.join(v.out, 'LATE.txt'), 'w', newline='\n').write(' '.join(d.rstrip('/\\') + '_late' for d in dirs))
    g = np.bincount(dense) if len(dense) else np.zeros(1, np.int64); info = {'chunks': len(dirs), 'external_games_extracted': int(len(ids)), 'external_games_labelled': int(keep.sum()), 'external_classes': int(len(u)), 'games_per_class_min_median_max': [int(g.min()), float(np.median(g)), int(g.max())],
                                    'first_external_y': y0, 'competition_labelled_games': int((L['y'] >= 0).sum()), 'min_games': v.min_games, 'max_games': v.max_games}
    json.dump(info, open(os.path.join(v.out, 'MERGE.json'), 'w'), indent=1); print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
