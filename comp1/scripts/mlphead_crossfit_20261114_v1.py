"""Nonlinear MLP embedding head (AM-softmax over training players) on frozen view90 features, trained on the local RTX 3060.
Cross-fit protocol identical to bigbank2_eval (fold k: v2 games of owners not in v1 fold k, minus families of fold-k episode games).
Scores per fold: cosine(mean normalized embeddings) and PLDA on embeddings; fused with v2 6-dim PLDA z-mean from eval_full_dims.
Usage: run <cfg>   (cfg in CFGS)
"""
from pathlib import Path
import sys, argparse, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank2_eval_20261114_v1 as e
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
b2 = e.b2; c1 = e.c1; c2 = e.c2; bb = e.bb; p = e.p; old = e.old; A = e.A; K = e.K
RUN = p.ROOT / 'runs/mlphead_20261114_v1'; BASE = b2.RUN / 'eval_full_dims'; BASENAME = 'v2_z128_256_384_512_768_1024'
CFGS = {
    'a': dict(hidden=1536, emb=512, drop=.2, indrop=.1, s=30., m=.25, epochs=40, bs=1024, lr=2e-3, wd=5e-4),
    'b': dict(hidden=2048, emb=256, drop=.3, indrop=.2, s=30., m=.30, epochs=60, bs=1024, lr=2e-3, wd=1e-3),
    'c': dict(hidden=1024, emb=256, drop=.5, indrop=.3, s=30., m=.20, epochs=20, bs=1024, lr=2e-3, wd=1e-2),
    'd': dict(hidden=2048, emb=512, drop=.5, indrop=.5, s=30., m=.35, epochs=60, bs=512, lr=1e-3, wd=5e-2),
    'lin': dict(hidden=0, emb=512, drop=0., indrop=.1, s=30., m=.25, epochs=40, bs=1024, lr=2e-3, wd=5e-4),
}


class Net(nn.Module):
    def __init__(self, d, ncls, hidden, emb, drop, indrop, s, m, **_):
        super().__init__(); self.s, self.m = s, m; self.indrop = nn.Dropout(indrop)
        if hidden: self.body = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, emb), nn.BatchNorm1d(emb))
        else: self.body = nn.Sequential(nn.Linear(d, emb), nn.BatchNorm1d(emb))
        self.W = nn.Parameter(torch.randn(ncls, emb) * .01)

    def embed(self, x): return self.body(self.indrop(x))

    def forward(self, x, y):
        z = F.normalize(self.embed(x), dim=1); logits = z @ F.normalize(self.W, dim=1).T
        logits = logits - self.m * F.one_hot(y, logits.shape[1]); return F.cross_entropy(self.s * logits, y)


def train_net(x, y, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed); dev = 'cuda'
    X = torch.tensor(np.clip(x, -8, 8), dtype=torch.float32, device=dev); _, Y = np.unique(y, return_inverse=True); Y = torch.tensor(Y, device=dev)
    net = Net(X.shape[1], int(Y.max()) + 1, **cfg).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    steps = cfg['epochs'] * (len(X) // cfg['bs']); sch = torch.optim.lr_scheduler.OneCycleLR(opt, cfg['lr'], total_steps=steps, pct_start=.1); g = torch.Generator(device=dev); g.manual_seed(seed); t = time.time(); n = 0
    for ep in range(cfg['epochs']):
        perm = torch.randperm(len(X), device=dev, generator=g); tot = 0.
        for i in range(len(X) // cfg['bs']):
            b = perm[i * cfg['bs']:(i + 1) * cfg['bs']]
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = net(X[b], Y[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tot += float(loss); n += 1
        if ep % 10 == 9 or ep == cfg['epochs'] - 1: print(f'  ep {ep + 1} loss {tot / (i + 1):.3f} {time.time() - t:.0f}s', flush=True)
    net.eval(); return net


@torch.no_grad()
def embed(net, x):
    out = [net.embed(torch.tensor(np.clip(x[i:i + 8192], -8, 8), dtype=torch.float32, device='cuda')).float().cpu().numpy() for i in range(0, len(x), 8192)]
    return np.concatenate(out).astype(np.float64)


def setscores(z, eids, ep):
    pos = {int(i): j for j, i in enumerate(eids)}; ss = [[pos[i] for i in s] for s in ep['supports']]; qs = [[pos[i] for i in q['indices']] for q in ep['questions']]
    zn = z / np.linalg.norm(z, axis=1, keepdims=True)
    sm = np.stack([zn[s].mean(0) for s in ss]); qm = np.stack([zn[q].mean(0) for q in qs])
    return (qm / np.linalg.norm(qm, axis=1, keepdims=True)) @ (sm / np.linalg.norm(sm, axis=1, keepdims=True)).T, ss, qs


def run(cfgname):
    cfg = CFGS[cfgname]; out = RUN / cfgname; out.mkdir(parents=True, exist_ok=False); m = p.b.meta(); owners = m['owners'].astype(str)
    eps = p.b.read(c1.RUN / 'EPISODES.json'); ids1 = np.load(bb.RUN / 'train_indices.npy'); lab1 = np.load(bb.RUN / 'train_labels.npy')
    ids2 = np.load(b2.RUN / 'train_indices.npy'); lab2 = np.load(b2.RUN / 'train_labels.npy'); raw2 = b2.lookup90(ids2); raw1 = bb.lookup_views(ids1, 90)
    tg = [np.array([z['target'] for z in ep['questions']]) for ep in eps]; ql = [np.array([z['q'] for z in ep['questions']]) for ep in eps]
    V = [{'base': dict(np.load(BASE / f'comps_f{k}.npz'))[BASENAME]} for k in range(K)]
    for k in range(K):
        ep = eps[k]; used = sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']})
        keep = ~np.isin(owners[ids2], sorted(set(owners[ids1[lab1 % K == k]])))
        for f in p.b.FAMILIES: keep &= ~np.isin(m[f][ids2], np.unique(m[f][used]))
        sel = np.isin(ids1, used); eraw, eids = raw1[sel], ids1[sel]; order = np.argsort(eids, kind='stable'); eraw, eids = eraw[order], eids[order]
        folder = out / f'f{k}'; folder.mkdir(); xtr = old.pre(raw2[keep], folder, True); xe = old.pre(eraw, folder)
        print(f'fold {k}: train {xtr.shape} players {len(np.unique(lab2[keep]))}', flush=True)
        net = train_net(xtr, lab2[keep], cfg, 2026111410 + k); torch.save(net.state_dict(), folder / 'net.pt')
        ze = embed(net, xe); V[k]['cos'], ss, qs = setscores(ze, eids, ep)
        ztr = embed(net, xtr); p.fit_plda(ztr, lab2[keep], folder / 'PLDA.npz', dimension=min(cfg['emb'], 512)); model = p.PLDA(folder / 'PLDA.npz'); zz = model.transform(ze)
        sm, sn = p.summarize(zz, ss); qm, qn = p.summarize(zz, qs); V[k]['plda'] = model.score(qm, qn, sm, sn)
        for nm in ('cos', 'plda'):
            V[k][f'base+{nm}'] = .5 * A.z(V[k]['base']) + .5 * A.z(V[k][nm]); V[k][f'base7+{nm}'] = (6 * A.z(V[k]['base']) + A.z(V[k][nm])) / 7
            V[k][f'base.7+{nm}.3'] = .7 * A.z(V[k]['base']) + .3 * A.z(V[k][nm])
        V[k]['base+cos+plda'] = (A.z(V[k]['base']) + A.z(V[k]['cos']) + A.z(V[k]['plda'])) / 3
        np.savez(out / f'comps_f{k}.npz', **V[k]); del xtr, ztr; torch.cuda.empty_cache()
    Q = np.concatenate(ql); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); res = {}
    def evals(nm):
        bal = []
        for k in range(K):
            o = [j for j in range(K) if j != k]; temps = A.calibrate(A.z(np.vstack([V[j][nm] for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
            bal.append(A.sid(A.new_transform(V[k][nm], ql[k], temps), tg[k]))
        return np.concatenate([A.sid(V[k][nm], tg[k]) for k in range(K)]), np.concatenate(bal)
    E = {nm: evals(nm) for nm in V[0]}; ref = E['base']
    for nm, (r, b) in sorted(E.items(), key=lambda x: -x[1][1].mean()):
        d = b - ref[1]; res[nm] = {'raw': float(r.mean()), 'bal': float(b.mean()), 'bal_by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'bal_delta_vs_base': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist()}
        print(f"{nm:20s} raw {r.mean():.4f} bal {b.mean():.4f} d {d.mean():+.4f} {res[nm]['CI95']} byq", {a: round(v, 3) for a, v in res[nm]['bal_by_q'].items()})
    p.b.write(out / 'RESULT.json', {'cfg': cfg, 'results': res})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('cfg'); v = ap.parse_args()
    with threadpool_limits(limits=4): run(v.cfg)
