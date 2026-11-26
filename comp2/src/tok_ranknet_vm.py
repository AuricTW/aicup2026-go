"""Runs ON THE 4090 VM.  Token RANK net for comp2: the comp1 token-net backbone (per-token MLP 1162 -> d, ply-phase / colour
embeddings, Transformer, mean + attention pooling, BN embedding) with a 10-way RANK head (label smoothing 0.1) instead of the
AM-softmax over players.  Training games = comp1 token games (owner side) of comp2 training players with their rank label;
dev / OOF / lockbox players excluded (tok_rank_labels.py); held-out players ('va') for early stopping (best held-out NLL kept).
Token dirs are read only (memory-mapped, blocks loaded per round as in comp1's trainer).
usage: python tok_ranknet_vm.py OUT_DIR LABELS.npz TOKDIR [TOKDIR ...] [--d 384 --layers 3 --epochs 8 --bs 256 --lr 1e-3 --drop .3
       --seed 0 --block 512 --blocks 24 --reuse 2 --eval-every 1500 --va-games 6000 --mem-frac 0.4]
"""
import sys, os, json, time, argparse
import numpy as np, torch
from torch import nn
import torch.nn.functional as F


class RankNet(nn.Module):
    def __init__(self, d, layers, drop, de=512):
        super().__init__(); self.inp = nn.Sequential(nn.Dropout(.3), nn.Linear(1162, d), nn.GELU(), nn.Dropout(drop)); self.ply = nn.Embedding(19, d); self.col = nn.Embedding(2, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, drop, batch_first=True, norm_first=True, activation='gelu'), layers, enable_nested_tensor=False) if layers else None
        self.att = nn.Linear(d, 1); self.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Dropout(drop), nn.Linear(2 * d, de), nn.BatchNorm1d(de)); self.head = nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(de, 10))

    def forward(self, x, ply, n, black):
        pad = torch.arange(x.shape[1], device=x.device)[None] >= n[:, None]; hh = self.inp(x) + self.ply((ply.clamp(1, 180) - 1) // 10) + self.col(black.long())[:, None]
        if self.enc is not None: hh = self.enc(hh, src_key_padding_mask=pad)
        m = (~pad).float()[..., None]; mean = (hh * m).sum(1) / m.sum(1); a = torch.softmax(self.att(hh).masked_fill(pad[..., None], -1e4), 1)
        return self.head(self.out(torch.cat([mean, (hh * a).sum(1)], 1)))


def opend(d): return {k: np.load(os.path.join(d, f'{k}.npy'), mmap_mode='r') for k in ('indices', 'tok_z', 'tok_a', 'tok_ply', 'tok_n')}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('labels'); ap.add_argument('dirs', nargs='+'); ap.add_argument('--d', type=int, default=384); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--epochs', type=float, default=8); ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--drop', type=float, default=.3); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--block', type=int, default=512); ap.add_argument('--blocks', type=int, default=24); ap.add_argument('--reuse', type=int, default=2); ap.add_argument('--eval-every', type=int, default=1500); ap.add_argument('--va-games', type=int, default=6000); ap.add_argument('--mem-frac', type=float, default=0.4)
    v = ap.parse_args(); os.makedirs(v.out, exist_ok=True); dev = 'cuda'
    if v.mem_frac > 0: torch.cuda.set_per_process_memory_fraction(v.mem_frac)
    torch.manual_seed(v.seed); rng = np.random.default_rng(v.seed); t0 = time.time(); L = np.load(v.labels)
    lab = {int(i): (int(y), bool(b), bool(q)) for i, y, b, q in zip(L['ids'], L['y'], L['black'], L['va'])}; D = [opend(d) for d in v.dirs]; blocks = []; total = 0; seen = set(); vapos = []
    for k, d in enumerate(D):
        ids = np.asarray(d['indices']); y = -np.ones(len(ids), np.int64); bl = np.zeros(len(ids), bool); va = np.zeros(len(ids), bool)
        for j, i in enumerate(ids):
            r = lab.get(int(i))
            if r is None or int(i) in seen: continue
            seen.add(int(i)); bl[j] = r[1]
            if r[2]: va[j] = True
            else: y[j] = r[0]
            if r[2]: vapos.append((k, j, r[0], r[1]))
        d['y'] = y; d['black'] = bl; total += int((y >= 0).sum())
        blocks += [(k, s, min(s + v.block, len(ids))) for s in range(0, len(ids), v.block)]
    steps = int(v.epochs * total / v.bs); print(f'train games {total}  held-out games {len(vapos)}  blocks {len(blocks)}  steps {steps}', flush=True)
    def load(group):
        z, a, pl, n, y, b = [], [], [], [], [], []
        for k, s, e in group:
            d = D[k]; m = d['y'][s:e] >= 0
            if not m.any(): continue
            z.append(np.asarray(d['tok_z'][s:e])[m]); a.append(np.asarray(d['tok_a'][s:e])[m]); pl.append(np.asarray(d['tok_ply'][s:e])[m]); n.append(np.asarray(d['tok_n'][s:e])[m]); y.append(d['y'][s:e][m]); b.append(d['black'][s:e][m])
        if not z: return None
        return [torch.tensor(np.concatenate(q)) for q in (z, a, pl, n, y, b)]
    # held-out sample (kept in RAM)
    sel = [vapos[i] for i in rng.choice(len(vapos), min(v.va_games, len(vapos)), replace=False)]; sel.sort()
    VZ = torch.tensor(np.stack([np.asarray(D[k]['tok_z'][j]) for k, j, _, _ in sel])); VA = torch.tensor(np.stack([np.asarray(D[k]['tok_a'][j]) for k, j, _, _ in sel]))
    VP = torch.tensor(np.stack([np.asarray(D[k]['tok_ply'][j]) for k, j, _, _ in sel]).astype(np.int64)); VN = torch.tensor(np.array([int(D[k]['tok_n'][j]) for k, j, _, _ in sel]))
    VY = torch.tensor([r for _, _, r, _ in sel]); VB = torch.tensor([b for _, _, _, b in sel]); print('held-out sample', len(sel), f'{time.time() - t0:.0f}s', flush=True)
    z, a, pl, n, y, b = load([blocks[i] for i in rng.choice(len(blocks), min(40, len(blocks)), replace=False)]); x = torch.cat([z.float(), a.float()], 2); mk = (torch.arange(90)[None] < n[:, None].long())[..., None]
    mu = (x * mk).sum((0, 1)) / mk.sum(); sd = (((x - mu) ** 2 * mk).sum((0, 1)) / mk.sum()).clamp_min(1e-8).sqrt(); mu, sd = mu.to(dev), sd.to(dev); del x
    norm = lambda z_, a_: ((torch.cat([z_.float(), a_.float()], 2) - mu) / sd).clamp(-8, 8)
    net = RankNet(v.d, v.layers, v.drop).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=v.lr, weight_decay=.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, v.lr, total_steps=steps, pct_start=.05)
    def evaluate():
        net.eval(); lo = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(VY), 1024): lo.append(net(norm(VZ[i:i + 1024].to(dev), VA[i:i + 1024].to(dev)), VP[i:i + 1024].to(dev), VN[i:i + 1024].to(dev).long(), VB[i:i + 1024].to(dev)).float())
        net.train(); lo = torch.cat(lo); yy = VY.to(dev); nll = F.cross_entropy(lo, yy).item(); pr = lo.argmax(1)
        return nll, (pr == yy).float().mean().item(), ((pr - yy).abs() <= 1).float().mean().item()
    step = 0; log = []; tot = cnt = 0; best = (9e9, None)
    while step < steps:
        order = rng.permutation(len(blocks))
        for g0 in range(0, len(order), v.blocks):
            if step >= steps: break
            z = a = pl = n = y = b = None; got = load([blocks[i] for i in order[g0:g0 + v.blocks]])
            if got is None: continue
            z, a, pl, n, y, b = [t.to(dev) for t in got]
            perm = torch.cat([torch.randperm(len(y), device=dev) for _ in range(v.reuse)])
            for i in range(0, len(perm) - v.bs + 1, v.bs):
                if step >= steps: break
                j = perm[i:i + v.bs]
                with torch.autocast('cuda', dtype=torch.bfloat16): loss = F.cross_entropy(net(norm(z[j], a[j]), pl[j].long(), n[j].long(), b[j]).float(), y[j], label_smoothing=0.1)
                opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); step += 1; tot += loss.item(); cnt += 1
                if step % v.eval_every == 0 or step == steps:
                    nll, acc, w1 = evaluate(); log.append({'step': step, 'epoch': step * v.bs / total, 'train_loss': tot / cnt, 'va_nll': nll, 'va_acc': acc, 'va_within1': w1, 'sec': time.time() - t0}); print(json.dumps(log[-1]), flush=True); tot = cnt = 0
                    if nll < best[0]: best = (nll, step); torch.save({'model': net.state_dict(), 'args': vars(v), 'mu': mu.cpu(), 'sd': sd.cpu(), 'step': step}, os.path.join(v.out, 'model.pt'))
    json.dump({'args': vars(v), 'train_games': total, 'log': log, 'best_step': best[1], 'best_va_nll': best[0], 'seconds': time.time() - t0}, open(os.path.join(v.out, 'LOG.json'), 'w'), indent=1); print('done best', best, flush=True)


if __name__ == '__main__':
    main()
