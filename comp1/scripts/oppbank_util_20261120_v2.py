"""Utility of gallery expansion with opponent-CONDITIONED hidden bank games (2026-11-20; needs oppbank_tcond_20261120_v1.py outputs).
Real pools (600 candidates x 10 gallery games), full sequence net b1_full (never saw the candidates).  Per draw: 7 gallery games + 3 held
games as single-game questions.  T-set of a candidate = training accounts recognised (p > thr) on the opponent side of his 7 gallery
games; conditional set = the bank games listed under those accounts whose opponent is NOT recognised as a training account; a bank game
is taken for candidate c when its opponent-side embedding has cosine >= thr to c's 7-game centroid (optionally: and c is the nearest of
all 600 centroids).  Expanded centroid = 7 games + w x taken games.  The held games are never used for retrieval.
usage: oppbank_util_20261120_v2.py [draws=4]"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import listfuse_20261115_v1 as lf
ROOT = lf.p.ROOT; SQ = ROOT / 'runs/seqid_20261115_v1'; OUT = ROOT / 'runs/oppbank_20261120_v2'; dev = 'cuda'; rng = np.random.default_rng(3); R = int(sys.argv[1]) if len(sys.argv) > 1 else 4
Z = np.load(SQ / 'bundle_g1.npz'); OWN = Z['OWN'].astype(np.int64); N = len(OWN); T = np.load(OUT / 'full.npz'); rec_p, rec_o, off_p, off_o = T['top'], T['owner'], T['off_top'], T['off_owner']
EO = torch.tensor(np.load(OUT / 'opp_full.npy')).to(dev); EOFF = np.load(OUT / 'off_own_full.npy').astype(np.float32)
order = np.argsort(OWN, kind='stable'); start = np.searchsorted(OWN[order], np.arange(OWN.max() + 2))
def games_of(accs): return np.concatenate([order[start[a]:start[a + 1]] for a in accs]) if len(accs) else np.zeros(0, np.int64)
nz = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
def score(C, Q, tg):
    S = Q @ C.T; r = (S > S[np.arange(len(tg)), tg][:, None]).sum(1) + 1; return np.where(r <= 5, np.exp(-(r - 1.)), 0.)


games, pools, keys, pos = lf.ss.official(); res = {}; info = {}
VAR = [(tt, ko, thr, nn, w) for tt in (.5,) for ko in (.5,) for thr in (.4, .5, .55, .6, .65) for nn in (False, True) for w in (.5, 1.)] + [(.8, .5, .5, True, 1.), (.5, .8, .5, True, 1.), (.3, .5, .5, True, 1.)]
for pool in pools:
    G10 = np.stack([[pos[k] for k in pool.supports[c]] for c in pool.candidate_ids]); n = len(G10); E = EOFF[G10]      # [600, 10, d]
    for d in range(R):
        perm = np.stack([rng.permutation(10) for _ in range(n)]); gi = np.take_along_axis(G10, perm[:, :7], 1); Gs = np.take_along_axis(E, perm[:, :7, None], 1); Qg = np.take_along_axis(E, perm[:, 7:, None], 1).reshape(n * 3, -1); tg = np.repeat(np.arange(n), 3)
        C0 = nz(Gs.mean(1)); res.setdefault('base', []).append(score(C0, Qg, tg)); Ct = torch.tensor(C0).to(dev).half(); cache = {}
        for (tt, ko, thr, nn, w) in VAR:
            if (tt, ko) not in cache:
                L = []
                for j in range(n):
                    ts = np.unique(off_o[gi[j]][off_p[gi[j]] > tt]); S = games_of(ts); S = S[rec_p[S] <= ko]
                    if len(S) == 0: L.append((S, np.zeros(0, np.float32), np.zeros(0, bool))); continue
                    with torch.no_grad(): A = (EO[torch.tensor(S).to(dev)] @ Ct.t()).float(); s = A[:, j].cpu().numpy(); top = (A.argmax(1) == j).cpu().numpy()
                    L.append((S, s, top))
                cache[(tt, ko)] = L
            L = cache[(tt, ko)]; C = Gs.sum(1).copy(); cnt = np.zeros(n)
            for j, (S, s, top) in enumerate(L):
                m = (s >= thr) & (top if nn else True)
                if m.any(): C[j] += w * EO[torch.tensor(S[m]).to(dev)].float().sum(0).cpu().numpy(); cnt[j] = m.sum()
            key = f'T>{tt} known-other>{ko} cos>={thr} {"nearest" if nn else "any    "} w{w}'; res.setdefault(key, []).append(score(nz(C), Qg, tg)); info.setdefault(key, []).append(cnt)
        print(pool.name, 'draw', d, flush=True)
b = np.concatenate(res['base']); B = np.random.default_rng(0).integers(0, len(b), (1000, len(b))); print(f'baseline (b1_full, 7-game centroid, single-game questions) {b.mean():.4f}  n {len(b)}')
for k, v in res.items():
    if k == 'base': continue
    s = np.concatenate(v); dl = s - b; ci = np.quantile(dl[B].mean(1), [.025, .975]); cnt = np.repeat(np.concatenate(info[k]), 3); has = cnt > 0
    print(f'   {k:52s} {dl.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]   candidates expanded {has.mean():.3f} (games added, mean over those {cnt[has].mean() if has.any() else 0:.1f})   delta: expanded {dl[has].mean() if has.any() else 0:+.4f}  not expanded {dl[~has].mean():+.4f}', flush=True)
