"""2026-11-21: gallery-based hubness (label-free).  Every gallery game of candidate j is a pseudo-question for all OTHER candidates;
h(c) = mean of the top-k z-scored first-order similarities that candidate c receives from other candidates' gallery games (the owner's
own column is never read, so no leave-one-out is needed).  Uses only the galleries (known labels), never the questions, so it does not
depend on how question targets were drawn.  First-order similarity = sum over groups of row-z-scored cosine to the candidate centroid
(same as scratchpad secondorder.py / hubgal.py).
Components (rows = questions; every row = -h, so V() gives z(-h)):  hg_top<k>  all rows;  hgm_top<k>  q >= 2 rows only, q = 1 rows are 0
(A.z of a constant row is 0, so the term vanishes for single-game questions).
usage: hubgal_20261121_v1.py val [groups]        -> runs/suppsplit_20261115_v2/extra/hgv1_<pool>_d<r>.npz
       hubgal_20261121_v1.py official [groups]   -> runs/hubgal_20261121_v1/<component>/official_<pool>.npy (+ parity check on one episode)"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import listfuse_20261115_v1 as lf
from threadpoolctl import threadpool_limits
ss = lf.ss; A = lf.A; OUT = Path(__file__).resolve().parents[1] / 'runs/hubgal_20261121_v1'
GROUPS = 'seqf=b1_full/ffull+b1cap_full/ffull+b1s1_full/ffull+b2_full/ffull,tk=tok:r013,tkp=tok:p2,tkl=tok:tuned7e'; KS = (10, 30, 100)
nrm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True); zrow = lambda x: (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def hub(M, S):
    """S: [nc, gs] official game positions of the galleries -> {k: -h_k [nc]}"""
    nc, gs = S.shape; own = np.repeat(np.arange(nc), gs); sg = 0
    for g, Zs in M.items():
        cg = 0
        for Z in Zs: G = Z[S.ravel()]; C = nrm(Z[S].sum(1)); cg = cg + G @ C.T
        sg = sg + zrow(cg)
    sg = zrow(sg); sg[np.arange(nc * gs), own] = -1e9; srt = np.sort(sg, 0)
    return {k: -srt[-k:].mean(0) for k in KS}


def comps(h, q):
    out = {}
    for k, v in h.items():
        full = np.repeat(v[None].astype(np.float64), len(q), 0); out[f'hg_top{k}'] = full; m = full.copy(); m[q == 1] = 0.0; out[f'hgm_top{k}'] = m
    return out


def main(mode, groups=GROUPS):
    M, black = lf.models(groups); print({g: len(v) for g, v in M.items()}, flush=True)
    if mode == 'val':
        for pool, r, ep, f in ss.each():
            q = np.array([z['q'] for z in ep['questions']]); out = comps(hub(M, np.array(ep['supports'])), q)
            np.savez(ss.RUN / 'extra' / f'hgv1_{pool}_d{r}.npz', **out); print(pool, r, flush=True)
        return
    pool0, r0, ep0, _ = next(iter(ss.each())); stored = dict(np.load(ss.RUN / 'extra' / f'hgv1_{pool0}_d{r0}.npz')); q0 = np.array([z['q'] for z in ep0['questions']]); chk = comps(hub(M, np.array(ep0['supports'])), q0)
    par = {k: float(np.abs(chk[k] - stored[k]).max()) for k in chk}; assert max(par.values()) < 1e-9, par; print('parity on', pool0, r0, par, flush=True)
    games, pools, keys, pos = ss.official(); info = {'groups': groups, 'parity_maxabs': par, 'pools': {}}
    for pool in pools:
        S = np.array([[pos[k] for k in pool.supports[c]] for c in pool.candidate_ids]); q = np.array([z['num_games'] for z in pool.questions]); out = comps(hub(M, S), q)
        for k, v in out.items(): d = OUT / k; d.mkdir(parents=True, exist_ok=True); np.save(d / f'official_{pool.name}.npy', v, allow_pickle=False)
        info['pools'][pool.name] = {'gallery': list(S.shape), 'questions': int(len(q)), 'h_top30_mean_sd': [float(-out['hg_top30'][0].mean()), float(out['hg_top30'][0].std())]}; print(pool.name, info['pools'][pool.name], flush=True)
    (OUT / 'INFO.json').write_text(json.dumps(info, indent=1)); print('saved', OUT, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=6): main(*sys.argv[1:])
