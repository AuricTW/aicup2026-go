"""Standalone (numpy + torch; runs on the 4090 VM): token net on the full-scale per-decision HumanSL tokens.
Same model as the pilot arm of tokhead_20261116_v1.py (per-token MLP 1162 -> d, ply-phase / colour embeddings, L Transformer layers,
masked mean + attention pooling, BN embedding, AM-softmax over players).  Tokens are memory-mapped (they do not fit in RAM): each "round"
loads a random set of contiguous game blocks, shuffles them together and trains on that buffer.
usage: python tokhead_vm_20261116_v1.py <out dir> <labels.npz> <official token dir> <token dir> [<token dir> ...]
       [--d 384 --layers 3 --epochs 30 --bs 256 --lr 1e-3 --drop .3 --seed 0 --block 512 --blocks 24]
token dir = folder with indices.npy, tok_z.npy, tok_a.npy, tok_ply.npy, tok_n.npy ; labels.npz: ids, y, black, off_black
writes <out>/off_emb.npy (official games, order of the official token dir), model.pt, LOG.json
"""
import sys, os, json, time, argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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


def opend(d): return {k: np.load(os.path.join(d, f'{k}.npy'), mmap_mode='r') for k in ('indices', 'tok_z', 'tok_a', 'tok_ply', 'tok_n')}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('labels'); ap.add_argument('official'); ap.add_argument('dirs', nargs='+'); ap.add_argument('--d', type=int, default=384); ap.add_argument('--layers', type=int, default=3); ap.add_argument('--epochs', type=float, default=30)
    ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--drop', type=float, default=.3); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--block', type=int, default=512); ap.add_argument('--blocks', type=int, default=24); ap.add_argument('--reuse', type=int, default=1); ap.add_argument('--mem-frac', type=float, default=0.); v = ap.parse_args()
    os.makedirs(v.out, exist_ok=False); dev = 'cuda'
    if v.mem_frac > 0: torch.cuda.set_per_process_memory_fraction(v.mem_frac)      # shared cards: cap this job's allocator
    torch.manual_seed(v.seed); rng = np.random.default_rng(v.seed); t0 = time.time(); L = np.load(v.labels); lab = {int(i): (int(y), bool(b)) for i, y, b in zip(L['ids'], L['y'], L['black'])}; D = [opend(d) for d in v.dirs]; blocks = []; total = 0; seen = set()
    for k, d in enumerate(D):
        ids = np.asarray(d['indices']); ok = np.array([int(i) in lab and int(i) not in seen for i in ids]); seen.update(int(i) for i in ids[ok]); d['y'] = np.array([lab[int(i)][0] if o else -1 for i, o in zip(ids, ok)]); d['black'] = np.array([lab[int(i)][1] if o else False for i, o in zip(ids, ok)]); total += int(ok.sum())
        blocks += [(k, s, min(s + v.block, len(ids))) for s in range(0, len(ids), v.block)]
    ys = np.concatenate([d['y'][d['y'] >= 0] for d in D]); cls = np.unique(ys); remap = -np.ones(int(L['y'].max()) + 1, np.int64); remap[cls] = np.arange(len(cls)); C = len(cls); steps = int(v.epochs * total / v.bs); print(f'games {total} classes {C} blocks {len(blocks)} steps {steps}', flush=True)
    def load(group):
        z, a, pl, n, y, b = [], [], [], [], [], []
        for k, s, e in group:
            d = D[k]; m = d['y'][s:e] >= 0
            if not m.any(): continue
            z.append(np.asarray(d['tok_z'][s:e])[m]); a.append(np.asarray(d['tok_a'][s:e])[m]); pl.append(np.asarray(d['tok_ply'][s:e])[m]); n.append(np.asarray(d['tok_n'][s:e])[m]); y.append(remap[d['y'][s:e][m]]); b.append(d['black'][s:e][m])
        return [torch.tensor(np.concatenate(q)) for q in (z, a, pl, n, y, b)]
    # normalisation statistics from a sample of blocks
    z, a, pl, n, y, b = load([blocks[i] for i in rng.choice(len(blocks), min(40, len(blocks)), replace=False)]); x = torch.cat([z.float(), a.float()], 2); mk = (torch.arange(90)[None] < n[:, None].long())[..., None]; mu = (x * mk).sum((0, 1)) / mk.sum(); sd = (((x - mu) ** 2 * mk).sum((0, 1)) / mk.sum()).clamp_min(1e-8).sqrt(); mu, sd = mu.to(dev), sd.to(dev); del x
    norm = lambda z_, a_: ((torch.cat([z_.float(), a_.float()], 2) - mu) / sd).clamp(-8, 8)
    net = TokNet(C, v.d, v.layers, v.drop).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=v.lr, weight_decay=.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, v.lr, total_steps=steps, pct_start=.05); step = 0; log = []; tot = cnt = 0
    while step < steps:
        order = rng.permutation(len(blocks))
        for g0 in range(0, len(order), v.blocks):
            if step >= steps: break
            z = a = pl = n = y = b = None; z, a, pl, n, y, b = [t.to(dev) for t in load([blocks[i] for i in order[g0:g0 + v.blocks]])];      # free the old buffer first (peak memory)
            perm = torch.cat([torch.randperm(len(y), device=dev) for _ in range(v.reuse)])      # --reuse: several passes over a loaded buffer (disk reads dominate on the VM)
            for i in range(0, len(perm) - v.bs + 1, v.bs):
                if step >= steps: break
                j = perm[i:i + v.bs]
                with torch.autocast('cuda', dtype=torch.bfloat16): loss = net(norm(z[j], a[j]), pl[j].long(), n[j].long(), b[j], y[j])
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); step += 1; tot += loss.item(); cnt += 1
                if step % 2000 == 0 or step == steps: log.append({'step': step, 'epoch': step * v.bs / total, 'loss': tot / cnt, 'sec': time.time() - t0}); print(json.dumps(log[-1]), flush=True); tot = cnt = 0
    net.eval(); O = opend(v.official); ob = L['off_black']; e = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(O['indices']), 1024): e.append(net.embed(norm(torch.tensor(np.asarray(O['tok_z'][i:i + 1024])).to(dev), torch.tensor(np.asarray(O['tok_a'][i:i + 1024])).to(dev)), torch.tensor(np.asarray(O['tok_ply'][i:i + 1024]).astype(np.int64)).to(dev), torch.tensor(np.asarray(O['tok_n'][i:i + 1024]).astype(np.int64)).to(dev), torch.tensor(ob[i:i + 1024]).to(dev)).float().cpu().numpy())
    np.save(os.path.join(v.out, 'off_emb.npy'), np.concatenate(e)); torch.save({'model': net.state_dict(), 'args': vars(v), 'mu': mu.cpu(), 'sd': sd.cpu(), 'classes': cls}, os.path.join(v.out, 'model.pt')); json.dump({'args': vars(v), 'games': total, 'classes': C, 'log': log, 'seconds': time.time() - t0}, open(os.path.join(v.out, 'LOG.json'), 'w'), indent=1); print('saved', v.out, flush=True)


if __name__ == '__main__':
    main()
