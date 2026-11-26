"""Wrapper around the standalone trainer seqid_core_20261115_v1.py (moves-only identity Transformer).
Training universe = every usable training player: SAFE meta (fit split incl. the 723 'hist' owners) + old 'validation' split (6,767 owners);
holdout (840) and the sealed 400 are not included.  Cross-fit folds exclude the fold's owners and the families of the episode games.
Stages
  bundle <split>          split in g1 | confirm | full -> runs/seqid_20261115_v1/bundle_<split>.npz (self-contained, for local or VM training)
  fuse <tag> [split ...]  reads runs/seqid_20261115_v1/<tag>/<split>/f<k>/ev_own.npy (+ev_opp.npy); compares against the v6b recipe comps (b2e3)
"""
from pathlib import Path
import sys, argparse, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import seqid_20261115_v1 as s1
import numpy as np
RUN = s1.RUN; K = 3


def bundle(split):
    p, u = s1.universe(); owners = u['owners'].astype(str); f = RUN / f'bundle_{split}.npz'; assert not f.exists(); _, own = np.unique(owners, return_inverse=True); base = u['hd'] <= 1
    O = np.load(RUN / 'official_moves.npz'); d = {'M': u['M'][:, :160], 'LEN': u['LEN'], 'FB': u['FB'], 'BLK': u['colors'] == 0, 'HD': u['hd'], 'OWN': own.astype(np.int32),
                                                'OFF_M': O['M'][:, :160], 'OFF_LEN': O['LEN'], 'OFF_FB': O['FB'], 'OFF_BLK': O['black'], 'OFF_HD': O['hd']}; info = {}
    if split == 'full': d['KEEP_ffull'] = base; info['full'] = {'games': int(base.sum()), 'owners': int(len(set(own[base])))}
    else:
        for k in range(K):
            ep, drop, ev3 = s1.fold_drop(split, k, p, owners); used = np.array(sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']})); keep = base & ~np.isin(owners, sorted(drop))
            for fam in p.b.FAMILIES: keep &= ~np.isin(u[fam], np.unique(u[fam][used]))
            assert not set(owners[keep]) & set(owners[used]); d[f'KEEP_f{k}'] = keep; d[f'EV_f{k}'] = used; info[k] = {'games': int(keep.sum()), 'owners': int(len(set(own[keep]))), 'episode_games': int(len(used))}
    np.savez_compressed(f, **d); (RUN / f'BUNDLE_{split}.json').write_text(json.dumps({'info': info, 'bytes': f.stat().st_size, 'SHA256': p.b.sha(f), 'source_SHA256': p.b.sha(__file__)}, indent=1), encoding='utf-8'); print(info, f.stat().st_size)


def nrm(a): return a / np.linalg.norm(a, axis=1, keepdims=True)


def fuse(tag, splits):
    import bigbank3_eval_20261114_v1 as ev3
    p = ev3.p; A = ev3.A; h = ev3.h; b3 = ev3.b3; rep = {}; mr = p.b.meta()['ranks']
    for split, epf in (('g1', h.c1.RUN / 'EPISODES.json'), ('confirm', h.RUN / 'confirm/EPISODES.json')):
        if splits and split not in splits: continue
        eps = p.b.read(epf); folds = [k for k in range(K) if (RUN / tag / split / f'f{k}/ev_own.npy').exists()]
        if not folds: continue
        Z = np.load(RUN / f'bundle_{split}.npz'); V, S, O, tg, ql = {}, {}, {}, {}, {}
        for k in folds:
            eids = Z[f'EV_f{k}']; ep = eps[k]; V[k] = dict(np.load(b3.RUN / f'eval_{split}/comps_f{k}.npz')); tg[k] = np.array([z['target'] for z in ep['questions']]); ql[k] = np.array([z['q'] for z in ep['questions']])
            S[k] = h.setscores(np.load(RUN / tag / split / f'f{k}/ev_own.npy').astype(np.float64), eids, ep)[0]
            zo = nrm(np.load(RUN / tag / split / f'f{k}/ev_opp.npy').astype(np.float64)); pos = {int(i): j for j, i in enumerate(eids)}   # opponent overlap: best opp-vs-opp game pair
            sup = [zo[[pos[i] for i in ss]] for ss in ep['supports']]; O[k] = np.array([[float((zo[[pos[i] for i in z['indices']]] @ sm.T).max()) for sm in sup] for z in ep['questions']])
        Q = np.concatenate([ql[k] for k in folds]); rk = np.concatenate([[mr[z['indices'][0]] for z in eps[k]['questions']] for k in folds]); ur, ui, uc = np.unique(rk, return_inverse=True, return_counts=True); W = (1 / len(ur)) / uc[ui]
        def evals(fn):
            bal = []
            for k in folds:
                o = [j for j in folds if j != k] or [k]; T = A.calibrate(A.z(np.vstack([fn(j) for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
                bal.append(A.sid(A.new_transform(fn(k), ql[k], T), tg[k]))
            return np.concatenate(bal)
        ref = evals(lambda k: A.z(V[k]['b2e3'])); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); rep[split] = {'folds': folds, 'in_fold_temps': len(folds) == 1}
        variants = {'ref_b2e3': lambda k: A.z(V[k]['b2e3']), 'seq_alone': lambda k: A.z(S[k]), 'opp_alone': lambda k: A.z(O[k]), 'ens3_alone': lambda k: A.z(V[k]['ens3'])}
        for w in (.1, .2, .3, .4, .5): variants[f'ref+seq_w{w}'] = (lambda k, w=w: (1 - w) * A.z(V[k]['b2e3']) + w * A.z(S[k]))
        for w in (.05, .1, .2): variants[f'ref+opp_w{w}'] = (lambda k, w=w: (1 - w) * A.z(V[k]['b2e3']) + w * A.z(O[k]))
        for nm, fn in variants.items():
            b = evals(fn); d = b - ref; rep[split][nm] = {'bal': float(b.mean()), 'bal_rank_uniform': float((b * W).sum()), 'delta': float(d.mean()), 'delta_rank_uniform': float((d * W).sum()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist(), 'by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}}
            print(split, folds, f"{nm:14s} bal {b.mean():.4f} d {d.mean():+.4f} {rep[split][nm]['CI95']} ru {(b * W).sum():.4f} d {(d * W).sum():+.4f}", {a: round(x, 3) for a, x in rep[split][nm]['by_q'].items()}, flush=True)
    (RUN / tag / 'FUSE.json').write_text(json.dumps(rep, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    if v.stage == 'bundle': bundle(v.args[0])
    else: fuse(v.args[0], v.args[1:])
