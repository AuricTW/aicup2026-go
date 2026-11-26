"""Episodic (prototypical) training of the embedding heads, judged on the official support-split validation v2.
Why: the v6b heads are trained to classify ONE game into its training player; the task is to match the MEAN of 1-5 query games to the
MEAN of 10 gallery games with mixed colours, and cross-colour matching is where the score is lost.  Here each step adds a set-level
loss: P players, G gallery + q query games each (natural colour mix), prototypes = normalised mean of unit embeddings,
cross-entropy of every query mean against the P prototypes (scale s, no margin).  Loss = AM-softmax (as v6b) + lam * episodic loss.
Variants: proto<lam>  e.g. proto1, proto3 ; protoX<lam> = episodic loss with CROSS-COLOUR episodes only (query colour != gallery colour).
Writes hd_<variant>_<cfg>, ens_<variant> into runs/suppsplit_20261115_v2/comps_*.npz ; then: suppsplit_eval_20261115_v2.py report
Usage: run <variant> [nseeds] [bank: b3|b345]
"""
from pathlib import Path
import sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import colorhead_20261115_v1 as ch
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
ss = ch.ss; g = ch.g; h = ch.h; p = ch.p; old = ch.old; A = ch.A; b3 = ch.b3; RUN = p.ROOT / 'runs/protohead_20261115_v1'


def train_proto(x, black, y, cfg, seed, lam, cross, P=384):
    torch.manual_seed(seed); np.random.seed(seed); dev = 'cuda'; X = torch.tensor(np.clip(x, -8, 8), dtype=torch.float32, device=dev); _, yi = np.unique(y, return_inverse=True); Y = torch.tensor(yi, device=dev)
    net = h.Net(X.shape[1], int(yi.max()) + 1, **cfg).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd']); steps = cfg['epochs'] * (len(X) // cfg['bs'])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, cfg['lr'], total_steps=steps, pct_start=.1); gen = torch.Generator(device=dev); gen.manual_seed(seed); t = time.time()
    # padded per-class game lists, separately per colour (colour 0 = black)
    def table(mask):
        cnt = np.bincount(yi[mask], minlength=yi.max() + 1); T = -np.ones((yi.max() + 1, cnt.max()), np.int64); fill = np.zeros(yi.max() + 1, int)
        for i in np.flatnonzero(mask): T[yi[i], fill[yi[i]]] = i; fill[yi[i]] += 1
        return torch.tensor(T, device=dev), torch.tensor(cnt, device=dev)
    TA, CA = table(np.ones(len(yi), bool)); TB, CB = table(black); TW, CW = table(~black)
    def draw(T, classes, n):
        r = torch.rand(len(classes), T.shape[1], device=dev, generator=gen); r[T[classes] < 0] = 2.; return T[classes].gather(1, r.argsort(1)[:, :n])
    tot = [0., 0.]
    for ep in range(cfg['epochs']):
        perm = torch.randperm(len(X), device=dev, generator=gen)
        for i in range(len(X) // cfg['bs']):
            b = perm[i * cfg['bs']:(i + 1) * cfg['bs']]; G = int(torch.randint(5, 11, (1,), generator=gen, device=dev)); q = (1, 2, 3, 5)[int(torch.randint(0, 4, (1,), generator=gen, device=dev))]
            if cross:
                qb = bool(torch.rand(1, generator=gen, device=dev) < .5); Tq, Cq, Tg, Cg = (TB, CB, TW, CW) if qb else (TW, CW, TB, CB); ok = torch.nonzero((Cq >= q) & (Cg >= G)).squeeze(1); cl = ok[torch.randperm(len(ok), device=dev, generator=gen)[:P]]; gi = draw(Tg, cl, G); qi = draw(Tq, cl, q)
            else:
                ok = torch.nonzero(CA >= G + q).squeeze(1); cl = ok[torch.randperm(len(ok), device=dev, generator=gen)[:P]]; idx = draw(TA, cl, G + q); gi, qi = idx[:, :G], idx[:, G:]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lc = net(X[b], Y[b]); e = F.normalize(net.embed(X[torch.cat([gi.reshape(-1), qi.reshape(-1)])]).float(), dim=1); n = len(cl)
                pg = F.normalize(e[:n * G].view(n, G, -1).mean(1), dim=1); pq = F.normalize(e[n * G:].view(n, q, -1).mean(1), dim=1); lp = F.cross_entropy(net.s * (pq @ pg.T), torch.arange(n, device=dev)); loss = lc + lam * lp
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tot[0] += float(lc); tot[1] += float(lp)
        if ep % 10 == 9 or ep == cfg['epochs'] - 1: print(f'  ep {ep + 1} cls {tot[0] / (i + 1):.3f} proto {tot[1] / (i + 1):.3f} {time.time() - t:.0f}s', flush=True)
        tot = [0., 0.]
    net.eval(); return net


def bank(name):
    m = p.b.meta(); ids3 = np.load(b3.RUN / 'train_indices.npy'); owners = m['owners'].astype(str)
    if name == 'b3': return b3.lookup90(ids3), owners[ids3], m['colors'][ids3] == 0
    import bigbank4_eval_20261115_v1 as ev4
    ids4, o4, raw4, chunks, U = ev4.bank45(True); print('bank345 chunks', chunks, flush=True); return np.concatenate([b3.lookup90(ids3), raw4]), np.concatenate([owners[ids3], o4]), np.concatenate([m['colors'][ids3] == 0, U.get('colors', ids4) == 0])


def run(variant, nseeds, bankname='b3'):
    tag = variant + ('' if bankname == 'b3' else '_' + bankname); out = RUN / tag; out.mkdir(parents=True, exist_ok=False); raw, y, black = bank(bankname); cross = variant.startswith('protoX'); lam = float(variant.replace('protoX', '').replace('proto', '') or 0)
    games, pools, keys, pos = ss.official(); oid = np.load(old.RUN / 'official_results/indices.npy'); oraw = np.asarray(np.load(old.RUN / 'official_results/view90.npy', mmap_mode='r')); rowof = np.zeros(len(keys), int); rowof[oid] = np.arange(len(oid))
    xtr = old.pre(raw, out, True); del raw; xo = old.pre(oraw, out)[rowof]; allk = np.arange(len(keys)); E = {}
    for cn in ('d', 'lin'):
        cfg = dict(g.BASE, **g.CFGS[cn]); E[cn] = []
        for sd in range(nseeds):
            net = train_proto(xtr, black, y, cfg, 2026111570 + sd, lam, cross) if lam > 0 else h.train_net(xtr, y, cfg, 2026111570 + sd); E[cn].append(h.embed(net, xo)); torch.save(net.state_dict(), out / f'net_{cn}_s{sd}.pt'); del net; torch.cuda.empty_cache(); print(tag, cn, sd, 'done', flush=True)
        np.save(out / f'emb_{cn}.npy', np.stack(E[cn]).astype(np.float32))
    for pool, r, ep, f in ss.each():
        S = {cn: g.zm([h.setscores(e, allk, ep)[0] for e in E[cn]]) for cn in E}; ss.save(f, **{f'hd_{tag}_d': S['d'], f'hd_{tag}_lin': S['lin'], f'ens_{tag}': g.zm([S['d'], S['lin']])})


if __name__ == '__main__':
    with threadpool_limits(limits=4): run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2, sys.argv[4] if len(sys.argv) > 4 else 'b3')
