"""Official-population validation without any answer key: SUPPORT-SPLIT episodes inside the official new pools.
Every new-pool candidate comes with 10 labelled support games (label = candidate id, part of the released test files).  Per pool and
draw: 400 random candidates become targets with q in {1,2,3,5} (100 each); q of their 10 games are held out as the query, the remaining
10-q stay as their gallery; the other 200 candidates keep all 10.  600 candidates, distinct targets, both colours -- the real task's
population, colours and candidate set; only the target galleries are a little smaller (5..9 games).
No training on these labels: frozen models only (v6b heads / PLDA, seq-id fold models).  Used to check whether a cross-fit gain exists
on test-population players before spending an upload (the v7a seq fusion: cross-fit +0.02, official -0.0046).
Stages
  build R                      -> runs/suppsplit_20261115_v1/EPISODES.json + comps_<pool>_d<r>.npz (base2, ens3, b2e3 = v6b neural recipe)
  heads <name> <models_root>   add z-mean head ensemble from another export root (models_d / models_lin with TRAIN_RECEIPT.json)
  seq <name> <off_own.npy>...  add seq-id cosine scores (z-mean over the given official embedding files)
  report [w ...]               all stored components + fusions against b2e3
"""
from pathlib import Path
import sys, argparse, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlphead_export_20261114_v2 as e2
import bigbank3_eval_20261114_v1 as ev3
import numpy as np
import torch
from threadpoolctl import threadpool_limits
g = e2.g; h = e2.h; p = e2.p; old = e2.old; A = e2.A; z6 = e2.z6; b3 = ev3.b3; BW = ev3.BW
RUN = p.ROOT / 'runs/suppsplit_20261115_v1'; SEED = 2026111506; POOLS = ('public2', 'private2')


def official():
    games, pools = p.b.load_official(); keys = p.b.read(p.b.RUN / 'official_stat/keys.json'); pos = {k: i for i, k in enumerate(keys)}; return games, pools, keys, pos


def episodes(R):
    RUN.mkdir(exist_ok=False); games, pools, keys, pos = official(); rng = np.random.default_rng(SEED); out = {}
    for pool in pools:
        out[pool.name] = []
        for r in range(R):
            tgt = rng.permutation(len(pool.candidate_ids))[:400]; qn = np.repeat([1, 2, 3, 5], 100)[rng.permutation(400)]; sup = [[pos[k] for k in pool.supports[c]] for c in pool.candidate_ids]; qs = []
            for t, q in zip(tgt, qn):
                held = rng.choice(10, int(q), replace=False); qs.append({'indices': [sup[t][j] for j in sorted(held)], 'q': int(q), 'target': int(t)}); sup[t] = [x for j, x in enumerate(sup[t]) if j not in set(held.tolist())]
            out[pool.name].append({'supports': sup, 'questions': qs})
    (RUN / 'EPISODES.json').write_text(json.dumps(out), encoding='utf-8'); return out


def head_embs(models_root, raw):
    embs = {}
    for c in ('d', 'lin'):
        mdir = Path(models_root) / f'models_{c}'; rc = p.b.read(mdir / 'TRAIN_RECEIPT.json'); x = old.pre(raw, mdir); embs[c] = []
        for sd in range(rc['nseeds']):
            net = h.Net(x.shape[1], rc['players'], **rc['cfg']).cuda(); net.load_state_dict(torch.load(mdir / f'net_s{sd}.pt')); net.eval(); embs[c].append(h.embed(net, x)); del net
    return embs


def ens_of(embs, ids, ep): return g.zm([g.zm([h.setscores(e, ids, ep)[0] for e in embs[c]]) for c in ('d', 'lin')])


def each():
    eps = json.loads((RUN / 'EPISODES.json').read_text())
    for pool in POOLS:
        for r, ep in enumerate(eps[pool]): yield pool, r, ep, RUN / f'comps_{pool}_d{r}.npz'


def save(f, **new):
    d = dict(np.load(f)) if f.exists() else {}; d.update(new); np.savez(f, **d)


def build(R):
    episodes(R); ids = np.load(old.RUN / 'official_results/indices.npy'); raw = np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r')); embs = head_embs(b3.RUN / 'export', raw)
    for pool, r, ep, f in each():
        base = g.zm([old.component(z6.MODELS / f'big2_raw{d}_v90', raw, ids, ep) for d in z6.DIMS]); ens = ens_of(embs, ids, ep)
        save(f, base2=base, ens3=ens, b2e3=BW * A.z(base) + (1 - BW) * A.z(ens)); print(pool, r, 'done', flush=True)


def heads(name, models_root):
    ids = np.load(old.RUN / 'official_results/indices.npy'); raw = np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r')); embs = head_embs(models_root, raw)
    for pool, r, ep, f in each(): save(f, **{name: ens_of(embs, ids, ep)})


def seq(name, files):
    E = [np.load(x).astype(np.float64) for x in files]; allk = np.arange(len(E[0]))
    for pool, r, ep, f in each(): save(f, **{name: g.zm([h.setscores(e, allk, ep)[0] for e in E])})


def report(ws, seqnames):
    data = {}
    for pool, r, ep, f in each(): data[(pool, r)] = (dict(np.load(f)), np.array([z['target'] for z in ep['questions']]), np.array([z['q'] for z in ep['questions']]))
    names = sorted(next(iter(data.values()))[0]); variants = {n: (lambda V, n=n: V[n]) for n in names}
    for sn in seqnames or [n for n in names if n.startswith('seq')]:
        for w in ws: variants[f'b2e3+{sn}_w{w}'] = (lambda V, sn=sn, w=w: (1 - w) * A.z(V['b2e3']) + w * A.z(V[sn]))
    for n in names:
        if n.startswith('ens') and n != 'ens3': variants[f'b2+{n}'] = (lambda V, n=n: BW * A.z(V['base2']) + (1 - BW) * A.z(V[n]))
    Q = np.concatenate([data[k][2] for k in sorted(data)]); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}; cache = {}
    def evals(fn):
        raw, bal = [], []
        for k in sorted(data):
            V, tg, q = data[k]; other = [j for j in sorted(data) if j[0] != k[0]]      # temperatures from the OTHER pool's episodes
            T = A.calibrate(A.z(np.vstack([fn(data[j][0]) for j in other])), np.concatenate([data[j][1] for j in other]), np.concatenate([data[j][2] for j in other]))
            S = fn(V); raw.append(A.sid(S, tg)); bal.append(A.sid(A.new_transform(S, q, T), tg))
        return np.concatenate(raw), np.concatenate(bal)
    ref = evals(variants['b2e3'])[1]
    for nm, fn in variants.items():
        raw, bal = evals(fn); d = bal - ref; res[nm] = {'raw': float(raw.mean()), 'sinkhorn': float(bal.mean()), 'delta_vs_b2e3': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist(), 'by_q': {str(n): float(bal[Q == n].mean()) for n in (1, 2, 3, 5)},
                                              'by_pool': {pl: float(np.concatenate([bal[i * 400:(i + 1) * 400] for i, k in enumerate(sorted(data)) if k[0] == pl]).mean()) for pl in POOLS}}
    for nm, r in sorted(res.items(), key=lambda x: -x[1]['sinkhorn']): print(f"{nm:22s} raw {r['raw']:.4f} sink {r['sinkhorn']:.4f} d {r['delta_vs_b2e3']:+.4f} {r['CI95']}", {k: round(v, 3) for k, v in r['by_q'].items()}, {k: round(v, 3) for k, v in r['by_pool'].items()})
    (RUN / 'REPORT.json').write_text(json.dumps({'questions': int(len(Q)), 'draws': sorted([list(k) for k in data]), 'results': res, 'note': 'support-split episodes inside the official new pools; temperatures from the other pool; bootstrap over questions (draws share candidates)'}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    with threadpool_limits(limits=4):
        if v.stage == 'build': build(int(v.args[0]))
        elif v.stage == 'heads': heads(*v.args)
        elif v.stage == 'seq': seq(v.args[0], v.args[1:])
        else: report([float(x) for x in v.args] or [.05, .1, .2, .3], None)
