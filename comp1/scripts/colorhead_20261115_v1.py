"""Colour-aware embedding heads, judged on the official support-split validation v2 (equal galleries, test-population players).
Finding behind it: with mixed colours the target's score depends strongly on how many gallery games share the query game's colour
(q=1, gallery 5: SID 0.19 with none of the same colour -> 0.49 with 4-5); the old all-black pool scores 0.92 with the same models.
Variants (bigbank3 training games, AM-softmax as v6b, cfgs d and lin):
  base   v6b head retrained with the seeds used here (control)
  cc     colour-conditional first layer: separate first Linear for the player's black and white games, everything after is shared
  cstd   per-colour standardisation of the input features (training-bank statistics per colour), standard head
Writes scores into runs/suppsplit_20261115_v2/comps_*.npz as hd_<variant>_<cfg> and ens_<variant> (z-mean of d and lin);
then run: suppsplit_eval_20261115_v2.py report
Usage: run <variant> [nseeds]
"""
from pathlib import Path
import sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import suppsplit_eval_20261115_v2 as s2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
ss = s2.ss; g = ss.g; h = ss.h; p = ss.p; old = ss.old; A = ss.A; b3 = ss.b3; RUN = p.ROOT / 'runs/colorhead_20261115_v1'


class CNet(nn.Module):
    def __init__(self, d, ncls, hidden, emb, drop, indrop, s, m, **_):
        super().__init__(); self.s, self.m = s, m; self.indrop = nn.Dropout(indrop); w = hidden or emb; self.first = nn.ModuleList([nn.Linear(d, w), nn.Linear(d, w)])
        self.rest = nn.Sequential(nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, emb), nn.BatchNorm1d(emb)) if hidden else nn.BatchNorm1d(emb)
        self.W = nn.Parameter(torch.randn(ncls, emb) * .01)

    def embed(self, x, c): x = self.indrop(x); return self.rest(torch.where(c[:, None], self.first[0](x), self.first[1](x)))

    def forward(self, x, c, y):
        z = F.normalize(self.embed(x, c), dim=1); logits = z @ F.normalize(self.W, dim=1).T; return F.cross_entropy(self.s * (logits - self.m * F.one_hot(y, logits.shape[1])), y)


def train_c(x, c, y, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed); dev = 'cuda'; X = torch.tensor(np.clip(x, -8, 8), dtype=torch.float32, device=dev); C = torch.tensor(c, device=dev); _, Y = np.unique(y, return_inverse=True); Y = torch.tensor(Y, device=dev)
    net = CNet(X.shape[1], int(Y.max()) + 1, **cfg).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']); steps = cfg['epochs'] * (len(X) // cfg['bs'])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, cfg['lr'], total_steps=steps, pct_start=.1); gen = torch.Generator(device=dev); gen.manual_seed(seed); t = time.time()
    for ep in range(cfg['epochs']):
        perm = torch.randperm(len(X), device=dev, generator=gen)
        for i in range(len(X) // cfg['bs']):
            b = perm[i * cfg['bs']:(i + 1) * cfg['bs']]
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = net(X[b], C[b], Y[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if ep % 20 == 19 or ep == cfg['epochs'] - 1: print(f'  ep {ep + 1} loss {float(loss):.3f} {time.time() - t:.0f}s', flush=True)
    net.eval(); return net


@torch.no_grad()
def embed_c(net, x, c):
    return np.concatenate([net.embed(torch.tensor(np.clip(x[i:i + 8192], -8, 8), dtype=torch.float32, device='cuda'), torch.tensor(c[i:i + 8192], device='cuda')).float().cpu().numpy() for i in range(0, len(x), 8192)]).astype(np.float64)


def run(variant, nseeds):
    out = RUN / variant; out.mkdir(parents=True, exist_ok=False); m = p.b.meta(); ids3 = np.load(b3.RUN / 'train_indices.npy'); y = np.load(b3.RUN / 'train_labels.npy'); ctr = m['colors'][ids3] == 0
    games, pools, keys, pos = ss.official(); oid = np.load(old.RUN / 'official_results/indices.npy'); oraw = np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r')); rowof = np.zeros(len(keys), int); rowof[oid] = np.arange(len(oid))
    cof = np.array([games[k].color == 'B' for k in keys]); xtr = old.pre(b3.lookup90(ids3), out, True); xo = old.pre(oraw, out)[rowof]      # official features in pos order
    if variant == 'cstd':
        for c in (True, False): mu = xtr[ctr == c].mean(0); sd = xtr[ctr == c].std(0) + 1e-6; xtr[ctr == c] = (xtr[ctr == c] - mu) / sd; xo[cof == c] = (xo[cof == c] - mu) / sd
    allk = np.arange(len(keys)); E = {}
    for cn in ('d', 'lin'):
        cfg = dict(g.BASE, **g.CFGS[cn]); E[cn] = []
        for sd in range(nseeds):
            if variant == 'cc': net = train_c(xtr, ctr, y, cfg, 2026111570 + sd); E[cn].append(embed_c(net, xo, cof))
            else: net = h.train_net(xtr, y, cfg, 2026111570 + sd); E[cn].append(h.embed(net, xo))
            del net; torch.cuda.empty_cache(); print(variant, cn, sd, 'done', flush=True)
        np.save(out / f'emb_{cn}.npy', np.stack(E[cn]).astype(np.float32))
    for pool, r, ep, f in ss.each():
        S = {cn: g.zm([h.setscores(e, allk, ep)[0] for e in E[cn]]) for cn in E}; ss.save(f, **{f'hd_{variant}_d': S['d'], f'hd_{variant}_lin': S['lin'], f'ens_{variant}': g.zm([S['d'], S['lin']])})


if __name__ == '__main__':
    with threadpool_limits(limits=4): run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2)
