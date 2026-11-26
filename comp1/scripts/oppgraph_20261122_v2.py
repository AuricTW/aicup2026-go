"""v2 (2026-11-22) of the opponent co-play graph component: same bank graph as oppgraph_20261122_v1 (b1_full recognition, p > 0.5,
unweighted, neighbourhood cosine), but the OFFICIAL games' opponents come from the token-net rematch posteriors o5
(runs/tokopp_20261117_v1/o5/posteriors_topk.npz: top-20 training accounts per game, about 2x the identification accuracy of the
sequence net).  og5(q, c) = mean over game pairs (g in question, h in gallery of c) of sum_{a != b} p_g(a) p_h(b) cos(a, b)  (expected
neighbourhood cosine, low-confidence games contribute little; exact same account excluded = rematch term).
Token label ids -> bank owner index via runs/tokpilot_20261116_v1/labels_all.npz (meta row ids) and the sorted owner strings.
usage: oppgraph_20261122_v2.py val <old|mirror|mirror2>   -> extra/og5v_<pool>_d<r>.npz (key og5_cos)"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import oppgraph_20261122_v1 as g1
ROOT = g1.ROOT


def setup():
    import seqid_20261115_v1 as s1, suppsplit_eval_20261115_v2 as s2, bigbank5_90_20261115_v1 as b5
    p, u = s1.universe(); uo = np.unique(u['owners']); L = np.load(ROOT / 'runs/tokpilot_20261116_v1/labels_all.npz'); ids, y = L['ids'], L['y']
    U = b5.Universe(); lo = U.a['owners'][U.at(ids)]      # same id -> owner mapping as bigbank6_tok_20261117_v1.prepare
    k = np.searchsorted(uo, lo); ok = uo[np.minimum(k, len(uo) - 1)] == lo      # token classes outside the seq-net universe have no graph row
    ymap = -np.ones(int(y.max()) + 1, np.int64); ymap[y[ok]] = k[ok]; print('token classes mapped to the graph', int((ymap >= 0).sum()), 'of', len(ymap), flush=True)
    P = np.load(ROOT / 'runs/tokopp_20261117_v1/o5/posteriors_topk.npz'); oid = np.load(s2.ss.old.RUN / 'official_results/indices.npy'); rowof = np.zeros(len(oid), int); rowof[oid] = np.arange(len(oid))
    Rn, _ = g1.graph(0.5, False); top = ymap[P['top']]; pt = np.where(top >= 0, P['pt'], 0.0).astype(np.float64); top = np.maximum(top, 0)
    print('posterior mass on mapped accounts', round(float(pt.sum() / P['pt'].sum()), 3), flush=True); return top, pt, rowof, Rn


PS = VN = None


def matrix(supports, questions):
    """og5 = mean_g mean_h [ (sum_a p_g(a) RN[a]) . (sum_b p_h(b) RN[b]) - sum_a p_g(a) p_h(a) ]   (RN rows have unit norm, so the
    subtracted term removes exactly the a == b pairs)."""
    global PS, VN
    if PS is None:
        import scipy.sparse as sp
        top, pt, rowof, Rn = setup(); n = Rn.shape[0]; rows = np.repeat(np.arange(len(top)), top.shape[1])
        Pr = sp.csr_matrix((pt.ravel(), (rows, top.ravel())), shape=(len(top), n)); PS = Pr[rowof].tocsr()      # official pos order
        VN = np.asarray((PS @ Rn).todense(), dtype=np.float32)
    Qv = np.stack([VN[np.asarray(g)].mean(0) for g in questions]); Cv = np.stack([VN[np.asarray(s)].mean(0) for s in supports])
    Pq = np.vstack([np.asarray(PS[np.asarray(g)].mean(0)) for g in questions]); Pc = np.vstack([np.asarray(PS[np.asarray(s)].mean(0)) for s in supports])
    return (Qv @ Cv.T).astype(np.float64) - Pq @ Pc.T


def val(which):
    import suppsplit_eval_20261115_v2 as s2, mirror_val_20261121_v1 as mv
    if which == 'old': it = ((p, r, e) for p, r, e, f in s2.ss.each()); dest = s2.ss.RUN / 'extra'
    else:
        if which == 'mirror2': mv.MRUN = ROOT / 'runs/mirrorval2_20261122_v1'
        it = mv.each(); dest = mv.MRUN / 'extra'
    for pool, r, ep in it:
        np.savez(dest / f'og5v_{pool}_d{r}.npz', og5_cos=matrix(ep['supports'], [z['indices'] for z in ep['questions']])); print(which, pool, r, flush=True)


if __name__ == '__main__':
    val(sys.argv[2])
