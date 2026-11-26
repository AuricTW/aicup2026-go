"""Transductive gallery expansion on the REAL questions (validated by transgal_20261119_v1.py; doc section 35.5).
For each new pool: the multi-game questions (q >= 2) are assigned by the production score + Sinkhorn of a finished recipe export; a
question whose top probability is >= tau lends its games to the gallery of its top candidate; the pair-fusion scores of the
single-game questions are then recomputed against the expanded galleries (same weights).  Multi-game rows are left untouched.
No answer key is used anywhere: the lending decision is the model's own output.
usage: transgal_official_20261119_v1.py <base listfuse tag> <new tag> <export dir with <pool>_score.npy + RESULT.json> [tau=0.8]
writes runs/listfuse_20261115_v1/<new tag>/official_<pool>.npy (+ SPEC.json, weights_final.npy, TRANSGAL.json) and copies the base
tag's validation components as pair_<new tag> (the validation of this step is the separate episode design of transgal_20261119_v1.py)."""
import sys, json, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from threadpoolctl import threadpool_limits
import listfuse_20261115_v1 as lf
A = lf.A


def main(base, new, exp, tau=.8):
    src = lf.RUN / base; out = lf.RUN / new; out.mkdir(exist_ok=False); spec = json.loads((src / 'SPEC.json').read_text())['spec']; w = np.load(src / 'weights_final.npy'); M, black = lf.models(spec); games, pools, keys, pos = lf.ss.official()
    T = {int(k): float(v) for k, v in json.load(open(Path(exp) / 'RESULT.json', encoding='utf-8'))['temps_used'].items()}; info = {'base': base, 'export': str(exp), 'tau': tau, 'temps': T, 'pools': {}}
    for pool in pools:
        S = np.load(Path(exp) / f'{pool.name}_score.npy'); q = np.array([z['num_games'] for z in pool.questions]); P0 = np.load(src / f'official_{pool.name}.npy'); sup = [np.array([pos[k] for k in pool.supports[c]]) for c in pool.candidate_ids]; qs = [[pos[k] for k in z['keys']] for z in pool.questions]
        m = np.flatnonzero(q >= 2); s1 = np.flatnonzero(q == 1); P = np.exp(A.new_transform(S[m], q[m], T)); conf = P.max(1); arg = P.argmax(1); lend = conf >= tau
        chk = lf.qscore(M, black, [qs[i] for i in s1], sup, w); dev = float(np.abs(chk - P0[s1]).max()); assert dev < 1e-3, ('base matrix not reproduced', dev)
        sup2 = [s.copy() for s in sup]
        for i in np.flatnonzero(lend): sup2[arg[i]] = np.concatenate([sup2[arg[i]], np.array(qs[m[i]])])
        new1 = lf.qscore(M, black, [qs[i] for i in s1], sup2, w); Pn = P0.copy(); Pn[s1] = new1; assert np.isfinite(Pn).all(); np.save(out / f'official_{pool.name}.npy', Pn, allow_pickle=False)
        dup = int(len(arg[lend]) - len(np.unique(arg[lend]))); sizes = np.array([len(s) for s in sup2])
        info['pools'][pool.name] = {'multi_questions': int(len(m)), 'lent': int(lend.sum()), 'lent_by_q': {str(n): [int((lend & (q[m] == n)).sum()), int((q[m] == n).sum())] for n in (2, 3, 5)}, 'two_questions_lent_to_one_candidate': dup, 'galleries_expanded': int((sizes > 10).sum()), 'mean_expanded_size': float(sizes[sizes > 10].mean()),
                                    'base_reproduced_maxabs': dev, 'q1_top1_changed_pair_only': int((np.argmax(new1, 1) != np.argmax(P0[s1], 1)).sum()), 'q1_rows': int(len(s1))}
        print(pool.name, info['pools'][pool.name], flush=True)
    shutil.copy(src / 'SPEC.json', out / 'SPEC.json'); shutil.copy(src / 'weights_final.npy', out / 'weights_final.npy'); json.dump(info, open(out / 'TRANSGAL.json', 'w'), indent=1)
    for f in sorted((lf.ss.RUN / 'extra').glob(f'pair_{base}_*.npz')): np.savez(f.with_name(f.name.replace(f'pair_{base}_', f'pair_{new}_')), **{f'pair_{new}': np.load(f)[f'pair_{base}']})
    print('saved', out, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=6): main(sys.argv[1], sys.argv[2], sys.argv[3], *([float(sys.argv[4])] if len(sys.argv) > 4 else []))
