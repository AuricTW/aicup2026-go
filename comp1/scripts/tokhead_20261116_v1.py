"""Token pilot: does a LEARNED pooling over the per-decision HumanSL tokens beat the fixed mean/std aggregate (view90)?
Same training games for both arms (token replays of runs/tokpilot_20261116_v1/<chunk>_merged), same AM-softmax, judged on the official
support-split validation v2 (components written to runs/suppsplit_20261115_v2/extra/tok_<tag>_<pool>_d<r>.npz).
Arms
  agg   : v6b-style heads (cfg d and lin) on the 2339-d view90 aggregate of the same replay            -> key tokagg_<tag>
  tok   : token net = per-token MLP (1162 -> D) + phase / colour embeddings, L Transformer layers, masked mean + attention pooling,
          BatchNorm embedding, AM-softmax                                                               -> key tok_<tag>
Small-data caveat: a few hundred players only; the comparison is between arms, not with the production heads.
Usage: run <tag> <chunk>[,<chunk>...] [--layers 2 --d 256 --epochs 40 --seeds 2]
"""
from pathlib import Path
import sys, argparse, json, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import suppsplit_eval_20261115_v2 as s2
import bigbank4_eval_20261115_v1 as ev4
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ss = s2.ss; g = ss.g; h = ss.h; p = ss.p; old = ss.old; A = ss.A; TOK = p.ROOT / 'runs/tokpilot_20261116_v1'; RUN = p.ROOT / 'runs/tokhead_20261116_v1'


def load(name):
    d = TOK / f'{name}_merged'; return np.load(d / 'indices.npy'), np.load(d / 'tok_z.npy', mmap_mode='r'), np.load(d / 'tok_a.npy', mmap_mode='r'), np.load(d / 'tok_ply.npy'), np.load(d / 'tok_n.npy').astype(np.int64), np.load(d / 'view90.npy')


class TokNet(nn.Module):
    def __init__(self, ncls, d, layers, drop, s=30., m=.2, de=512):
        super().__init__(); self.s, self.m = s, m; self.inp = nn.Sequential(nn.Dropout(.3), nn.Linear(1162, d), nn.GELU(), nn.Dropout(drop)); self.ply = nn.Embedding(19, d); self.col = nn.Embedding(2, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True, activation='gelu'), layers, enable_nested_tensor=False) if layers else None
        self.att = nn.Linear(d, 1); self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, de), nn.BatchNorm1d(de)); self.W = nn.Parameter(torch.randn(ncls, de) * .01)

    def embed(self, x, ply, n, black):
        pad = torch.arange(x.shape[1], device=x.device)[None] >= n[:, None]; hh = self.inp(x) + self.ply((ply.clamp(1, 180) - 1) // 10) + self.col(black.long())[:, None]
        if self.enc is not None: hh = self.enc(hh, src_key_padding_mask=pad)
        m = (~pad).float()[..., None]; mean = (hh * m).sum(1) / m.sum(1); a = torch.softmax(self.att(hh).masked_fill(pad[..., None], -1e4), 1); return self.out(torch.cat([mean, (hh * a).sum(1)], 1))

    def forward(self, x, ply, n, black, y):
        z = F.normalize(self.embed(x, ply, n, black).float(), dim=1); lo = z @ F.normalize(self.W, dim=1).T; return F.cross_entropy(self.s * (lo - self.m * F.one_hot(y, lo.shape[1])), y)


def run(tag, chunks, v):
    out = RUN / tag; out.mkdir(parents=True, exist_ok=False); dev = 'cuda'; parts = [load(c) for c in chunks]; ids = np.concatenate([q[0] for q in parts]); n = np.concatenate([q[4] for q in parts]); ply = np.concatenate([q[3] for q in parts]); v90 = np.concatenate([q[5] for q in parts])
    bid, bown, _, _, U = ev4.bank45(True); o = {int(i): w for i, w in zip(bid, bown)}; own = np.array([o[int(i)] for i in ids]); black = U.get('colors', ids) == 0; _, y = np.unique(own, return_inverse=True); print('train games', len(ids), 'players', y.max() + 1, 'black share', black.mean().round(3), flush=True)
    Z = torch.cat([torch.cat([torch.tensor(np.asarray(q[1])), torch.tensor(np.asarray(q[2])).half()], 2) for q in parts]).to(dev)      # [N, 90, 1162] float16
    mu = torch.zeros(1162, device=dev); sq = torch.zeros(1162, device=dev); cnt = 0
    for i in range(0, len(Z), 2048): zz = Z[i:i + 2048].float(); mk = (torch.arange(90, device=dev)[None] < torch.tensor(n[i:i + 2048], device=dev)[:, None])[..., None]; mu += (zz * mk).sum((0, 1)); sq += (zz * zz * mk).sum((0, 1)); cnt += int(mk.sum())
    mu /= cnt; sd = (sq / cnt - mu * mu).clamp_min(1e-8).sqrt(); norm = lambda t: ((t.float() - mu) / sd).clamp(-8, 8)
    oi, oz, oa, oply, on, ov90 = load('official'); games, pools, keys, pos = ss.official(); oid = np.load(old.RUN / 'official_results/indices.npy'); assert np.array_equal(oi, oid); rowof = np.zeros(len(keys), int); rowof[oid] = np.arange(len(oid)); oblack = np.array([games[k].color == 'B' for k in keys])[oid]      # official_results order
    allk = np.arange(len(keys)); N = torch.tensor(n, device=dev); P = torch.tensor(ply.astype(np.int64), device=dev); B = torch.tensor(black, device=dev); Y = torch.tensor(y, device=dev); E = {'tok': [], 'd': [], 'lin': []}
    for sd_ in range(v.seeds):
        torch.manual_seed(2026111600 + sd_); net = TokNet(int(y.max()) + 1, v.d, v.layers, v.drop).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=v.lr, weight_decay=.05); spe = len(ids) // v.bs; sch = torch.optim.lr_scheduler.OneCycleLR(opt, v.lr, total_steps=v.epochs * spe, pct_start=.1); t0 = time.time()
        for ep in range(v.epochs):
            perm = torch.randperm(len(ids), device=dev)
            for i in range(spe):
                b = perm[i * v.bs:(i + 1) * v.bs]
                with torch.autocast('cuda', dtype=torch.bfloat16): loss = net(norm(Z[b]), P[b], N[b], B[b], Y[b])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            if ep % 10 == 9 or ep == v.epochs - 1: print(f'  tok seed {sd_} ep {ep + 1} loss {float(loss):.3f} {time.time() - t0:.0f}s', flush=True)
        net.eval(); e = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(oi), 1024): x = torch.cat([torch.tensor(np.asarray(oz[i:i + 1024])), torch.tensor(np.asarray(oa[i:i + 1024])).half()], 2).to(dev); e.append(net.embed(norm(x), torch.tensor(oply[i:i + 1024].astype(np.int64), device=dev), torch.tensor(on[i:i + 1024], device=dev), torch.tensor(oblack[i:i + 1024], device=dev)).float().cpu().numpy())
        E['tok'].append(np.concatenate(e)[rowof].astype(np.float64)); del net; torch.cuda.empty_cache()
    xtr = old.pre(v90, out, True); xo = old.pre(ov90, out)[rowof]
    for cn in ('d', 'lin'):
        for sd_ in range(v.seeds): net = h.train_net(xtr, y, dict(g.BASE, **g.CFGS[cn]), 2026111600 + sd_); E[cn].append(h.embed(net, xo)); del net; torch.cuda.empty_cache()
    (ss.RUN / 'extra').mkdir(exist_ok=True); np.savez(out / 'emb.npz', **{k: np.stack(e).astype(np.float32) for k, e in E.items()})
    for pool, r, ep, f in ss.each():
        S = {k: g.zm([h.setscores(e, allk, ep)[0] for e in E[k]]) for k in E}; np.savez(ss.RUN / 'extra' / f'tok_{tag}_{pool}_d{r}.npz', **{f'tok_{tag}': S['tok'], f'tokagg_{tag}': g.zm([S['d'], S['lin']]), f'tokaggd_{tag}': S['d'], f'tokagglin_{tag}': S['lin']})
    print('saved components', f'tok_{tag}', f'tokagg_{tag}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('tag'); ap.add_argument('chunks'); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--d', type=int, default=256); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--drop', type=float, default=.3); v = ap.parse_args(); run(v.tag, v.chunks.split(','), v)
