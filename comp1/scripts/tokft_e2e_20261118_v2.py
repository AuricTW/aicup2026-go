"""v2 = v1 + --init-head <model.pt of a trained v5/v6 token net> + --Kn <n>.
--init-head: the head starts from a trained full-scale token net (same TokNet; its d / layers / classes / mu / sd are taken over, all its
  player classes stay in the classifier, its embedding BN stays in eval mode) -> trunk and head are tuned JOINTLY from a strong model, the
  gradient that reaches the trunk is about the remaining (fine) confusions, and the result is directly usable: off_emb(bn).npy of this
  script = official embeddings of the joint model with ALL decisions of a game (no stage 2 needed to evaluate it).
--Kn: every game gets K decisions with gradient through the trainable blocks + Kn more decisions without gradient (forward only), so the
  head sees K + Kn decisions per game at the memory cost of K.
v1: Stage 1 of HumanSL trunk fine-tuning for player identity (runs on the 4090 VM; standalone: numpy + torch + the payload's vendor code).
Why: see tokcache_20261118_v1.py - everything downstream of the frozen HumanSL features is saturated; the trunk itself was only ever
trained to predict moves.  Here the top --blocks residual blocks (+ final norm) are trained with the identity loss.
Model: HumanSL net (eval mode, lower blocks and all heads frozen) -> per-decision token exactly as in the frozen pipeline
(local 384, board mean 384, local minus policy expectation 384, 10 scalars; policy detached) -> TokNet of trainer v5 (default without
Transformer layers: per-decision MLP + ply / colour / move-coordinate embeddings, masked mean + attention pooling, BN embedding,
AM-softmax over players).  Every step: G games x K random decisions of each game (plies 1-300) from the packed input cache.
Outputs: <out>/trunk_ft.pt (state of the trained blocks + final norm), model.pt (head, mu, sd, classes, args), off_emb.npy / off_embbn.npy
(official games in the order of --official-order, all decisions of a game), snapshots off_embbn_step<k>.npy, LOG.json; ckpt.pt = resume.
Stage 2 (not here): re-extract the tokens of all games with the tuned trunk and train the normal v5 head.
usage: tokft_e2e_20261118_v1.py <out> <labels.npz> <payload dir> <weight.pt> <official cache dir> <official order indices.npy> <cache dir> [...]
       [--blocks 6 --K 16 --G 64 --epochs 8 --lr-trunk 1e-4 --lr-head 1e-3 --d 384 --layers 0 --seed 0 --snap-every 12000 --mem-frac 0]
"""
import sys, os, json, time, argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def opencache(d): return {k: np.load(os.path.join(d, f'{k}.npy'), mmap_mode='r') for k in ('indices', 'px', 'g', 'legal', 'act', 'ply', 'n', 'black', 'hd')}


def act2code(a): return torch.where(a >= 361, a, (a % 19) * 19 + a // 19)       # worker action index -> tokmv code (pass 361, pad 362)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('labels'); ap.add_argument('payload'); ap.add_argument('weight'); ap.add_argument('official'); ap.add_argument('official_order'); ap.add_argument('dirs', nargs='+')
    ap.add_argument('--blocks', type=int, default=6); ap.add_argument('--K', type=int, default=16); ap.add_argument('--G', type=int, default=64); ap.add_argument('--epochs', type=float, default=8); ap.add_argument('--lr-trunk', type=float, default=1e-4); ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--d', type=int, default=384); ap.add_argument('--layers', type=int, default=0); ap.add_argument('--drop', type=float, default=.3); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--block', type=int, default=512); ap.add_argument('--buf', type=int, default=12); ap.add_argument('--reuse', type=int, default=2)
    ap.add_argument('--snap-every', type=int, default=12000); ap.add_argument('--ckpt-every', type=int, default=1000); ap.add_argument('--mem-frac', type=float, default=0.); ap.add_argument('--max-steps', type=int, default=0); ap.add_argument('--no-final', action='store_true'); ap.add_argument('--init-head', default=''); ap.add_argument('--Kn', type=int, default=0); ap.add_argument('--head-eval', action='store_true'); v = ap.parse_args(); KK = v.K + v.Kn
    assert not os.path.exists(os.path.join(v.out, 'LOG.json')), 'finished run exists'; os.makedirs(v.out, exist_ok=True); ck = os.path.join(v.out, 'ckpt.pt'); dev = 'cuda'
    if v.mem_frac > 0: torch.cuda.set_per_process_memory_fraction(v.mem_frac)
    for q in ('vendor', 'code', ''): sys.path.insert(0, os.path.join(os.path.abspath(v.payload), q))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import worker as W
    from katago.train.load_model import load_model
    from katago.train.model_pytorch import ExtraOutputs
    import tokhead_vm_20261118_v5 as tv

    class GradExtra(ExtraOutputs):      # as ExtraOutputs, but the reported tensor keeps its graph
        def report(self, name, value):
            if name in self.requested: self.returned[name] = value

    torch.manual_seed(v.seed); rng = np.random.default_rng(v.seed); t0 = time.time()
    model, swa, _ = load_model(v.weight, use_swa=False, device=torch.device(dev), pos_len=19); del swa; model.eval().requires_grad_(False); nb = len(model.blocks); trainable = []
    if v.blocks > 0:
        for m in list(model.blocks[nb - v.blocks:]) + [model.norm_trunkfinal]:
            for p_ in m.parameters(): p_.requires_grad_(True); trainable.append(p_)
    print('trunk blocks', nb, 'trainable top', v.blocks, 'params', sum(p_.numel() for p_ in trainable), flush=True)
    META = {b: torch.tensor(W.n.profile_metadata('rank_1d').get_metadata_row('B' if b else 'W', 361)).to(dev) for b in (True, False)}

    def tokens(px, g, legal, act, ply, black, hd, grad):
        """px [P,993] u8 (cpu numpy), g [P,19], legal [P,46] u8, act [P], ply [P], black [P] bool, hd [P] -> z [P,1152], a [P,10] (as the frozen pipeline)"""
        x = torch.from_numpy(np.unpackbits(px, axis=1)[:, :22 * 361].reshape(-1, 22, 19, 19)).to(dev).float(); lg = torch.from_numpy(np.unpackbits(legal, axis=1)[:, :362]).to(dev).bool(); gg = torch.from_numpy(np.ascontiguousarray(g)).to(dev).float()
        ac = torch.from_numpy(act.astype(np.int64)).to(dev); bl = torch.from_numpy(black).to(dev); meta = torch.where(bl[:, None], META[True][None], META[False][None]); pl = torch.from_numpy(ply.astype(np.float32)).to(dev); hdv = torch.from_numpy(hd.astype(np.float32)).to(dev)
        extra = GradExtra(['trunkfinal'])
        with torch.set_grad_enabled(grad), torch.autocast('cuda', dtype=torch.float16):
            out = model(x, gg, input_meta=meta, extra_outputs=extra)
        logits = out[0][0][:, 0, :].float().detach(); pi = torch.softmax(logits.masked_fill(~lg, -1e9), -1); h = extra.returned['trunkfinal'].float().flatten(2)
        local = h.gather(2, ac.clamp_max(360)[:, None, None].expand(-1, h.shape[1], 1)).squeeze(2) * (ac != 361)[:, None]; expected = torch.einsum('pcs,ps->pc', h, pi[:, :361]); z = torch.cat((local, h.mean(2), local - expected), 1)
        chosen = pi.gather(1, ac[:, None]).squeeze(1); rank = ((pi > chosen[:, None]) & lg).sum(1).float() / lg.sum(1); ent = -(pi * torch.log(pi.clamp_min(1e-12))).sum(1); top = pi.topk(2, dim=1).values
        a = torch.stack(((pl <= 60).float(), (pl > 60).float(), torch.log(chosen.clamp_min(1e-12)), rank, ent, torch.log(top[:, 0].clamp_min(1e-12)), top[:, 0] - top[:, 1], (pl / 180).clamp(max=1), hdv, (ac == 361).float()), 1)
        return z, a, ac

    L = np.load(v.labels); lab = {int(i): int(y) for i, y in zip(L['ids'], L['y'])}; D = [opencache(d) for d in v.dirs]; blocks = []; total = 0; seen = set()
    for k, d in enumerate(D):
        ids = np.asarray(d['indices']); n = np.asarray(d['n']); ok = np.array([int(i) in lab and int(i) not in seen and n[j] > 0 for j, i in enumerate(ids)]); seen.update(int(i) for i in ids[ok]); d['y'] = np.array([lab[int(i)] if o else -1 for i, o in zip(ids, ok)]); total += int(ok.sum())
        d['blackv'] = np.asarray(d['black']); d['hdv'] = np.asarray(d['hd']); d['nv'] = n; blocks += [(k, s, min(s + v.block, len(ids))) for s in range(0, len(ids), v.block)]
    cls = np.unique(np.concatenate([d['y'][d['y'] >= 0] for d in D])); ih = None
    if v.init_head:
        ih = torch.load(v.init_head, map_location='cpu', weights_only=False); assert np.isin(cls, np.asarray(ih['classes'])).all(), 'training players missing from the warm-start head'
        cls = np.asarray(ih['classes']); v.d, v.layers = ih['args']['d'], ih['args']['layers']; print('warm start from', v.init_head, 'd', v.d, 'layers', v.layers, 'classes', len(cls), flush=True)
    remap = -np.ones(int(L['y'].max()) + 1, np.int64); remap[cls] = np.arange(len(cls)); C = len(cls); steps = int(v.epochs * total / v.G)
    if v.max_steps: steps = min(steps, v.max_steps)
    print(f'games {total} classes {C} blocks {len(blocks)} steps {steps}', flush=True)

    def load(group):
        o = {q: [] for q in ('px', 'g', 'legal', 'act', 'ply', 'n', 'y', 'black', 'hd')}
        for k, s, e in group:
            d = D[k]; m = d['y'][s:e] >= 0
            if not m.any(): continue
            for q in ('px', 'g', 'legal', 'act', 'ply'): o[q].append(np.asarray(d[q][s:e])[m])
            o['n'].append(d['nv'][s:e][m]); o['y'].append(remap[d['y'][s:e][m]]); o['black'].append(d['blackv'][s:e][m]); o['hd'].append(d['hdv'][s:e][m])
        return {q: np.concatenate(x) for q, x in o.items()}

    def batch(B, sel):
        """K random valid decisions of the selected games (with repeats masked out if a game has fewer than K)"""
        n = B['n'][sel].astype(np.int64); r = rng.random((len(sel), B['px'].shape[1])); r[np.arange(r.shape[1])[None] >= n[:, None]] = 2.; idx = np.argsort(r, 1)[:, :KK]; valid = np.arange(KK)[None] < np.minimum(n, KK)[:, None]
        idx = np.where(valid, idx, idx[:, :1]); gi = np.repeat(sel, KK); di = idx.reshape(-1)
        return (B['px'][gi, di], B['g'][gi, di], B['legal'][gi, di], B['act'][gi, di], B['ply'][gi, di], B['black'][gi], B['hd'][gi]), valid

    net = tv.TokNet(C, v.d, v.layers, v.drop, moves=True).to(dev)
    if ih is not None: net.load_state_dict(ih['model'])

    def settrain():
        net.train()
        if v.head_eval: net.eval()             # --head-eval (use with --lr-head 0): the head is a fixed, deterministic judge; only the trunk learns
        if v.init_head: net.out[3].eval()      # warm start: the embedding BN keeps the statistics of the trained head

    def tok2(args_):
        """tokens of a [G*KK] batch: the first K decisions of every game with gradient, the other Kn without"""
        if not v.Kn or not trainable: return tokens(*args_, grad=bool(trainable))
        m = (np.arange(len(args_[0])) % KK) < v.K; zg, ag, acg = tokens(*[q[m] for q in args_], grad=True); zn, an, acn = tokens(*[q[~m] for q in args_], grad=False); mt = torch.from_numpy(m).to(dev)
        z = torch.empty((len(m), zg.shape[1]), device=dev, dtype=zg.dtype); a = torch.empty((len(m), ag.shape[1]), device=dev, dtype=ag.dtype); ac = torch.empty(len(m), device=dev, dtype=acg.dtype)
        z[mt] = zg; z[~mt] = zn; a[mt] = ag; a[~mt] = an; ac[mt] = acg; ac[~mt] = acn; return z, a, ac
    # normalisation statistics of the tokens with the frozen trunk
    mu = sd = None; ckst = torch.load(ck, map_location=dev, weights_only=False) if os.path.exists(ck) else None
    if ckst is None and ih is not None: mu, sd = ih['mu'].to(dev), ih['sd'].to(dev)
    elif ckst is None:
        B = load([blocks[i] for i in rng.choice(len(blocks), min(6, len(blocks)), replace=False)]); s1 = s2 = cn = 0.
        for _ in range(12):
            args_, valid = batch(B, rng.choice(len(B['y']), min(v.G, len(B['y'])), replace=False)); z, a, _ = tokens(*args_, grad=False); x = torch.cat([z, a], 1).double()[torch.from_numpy(valid.reshape(-1)).to(dev)]; s1 = s1 + x.sum(0); s2 = s2 + (x * x).sum(0); cn += len(x)
        mu = (s1 / cn).float(); sd = (s2 / cn - (s1 / cn) ** 2).clamp_min(1e-8).sqrt().float(); del B
    else: mu, sd = ckst['mu'].to(dev), ckst['sd'].to(dev)
    groups = [{'params': list(net.parameters()), 'lr': v.lr_head}] + ([{'params': trainable, 'lr': v.lr_trunk}] if trainable else [])
    opt = torch.optim.AdamW(groups, weight_decay=.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, [g_['lr'] for g_ in groups], total_steps=steps, pct_start=.05); scaler = torch.amp.GradScaler('cuda'); step = 0; log = []; tot = cnt = 0

    def trunk_state(): return {'blocks': [b.state_dict() for b in model.blocks[nb - v.blocks:]] if v.blocks else [], 'norm': model.norm_trunkfinal.state_dict(), 'first_block': nb - v.blocks}

    if ckst is not None:
        net.load_state_dict(ckst['net']); opt.load_state_dict(ckst['opt']); sch.load_state_dict(ckst['sch']); scaler.load_state_dict(ckst['scaler']); step = ckst['step']; log = ckst['log']; rng = np.random.default_rng(v.seed + step)
        for b, s in zip(model.blocks[nb - v.blocks:], ckst['trunk']['blocks']): b.load_state_dict(s)
        model.norm_trunkfinal.load_state_dict(ckst['trunk']['norm']); print('resumed at step', step, flush=True); del ckst

    def official():
        O = opencache(v.official); order = np.load(v.official_order); row = {int(i): j for j, i in enumerate(np.asarray(O['indices']))}; r = np.array([row[int(i)] for i in order]); n = np.asarray(O['n']); bl = np.asarray(O['black']); hd = np.asarray(O['hd']); Kk = O['px'].shape[1]
        was = net.training; net.eval(); bn = net.out[3]; e, f = [], []
        with torch.no_grad():
            for i in range(0, len(r), 8):
                rr = r[i:i + 8]; o = np.argsort(rr); rs = rr[o]; nn_ = n[rs].astype(np.int64); gi = np.repeat(np.arange(len(rs)), nn_); di = np.concatenate([np.arange(k) for k in nn_])
                px = np.asarray(O['px'][rs]); g = np.asarray(O['g'][rs]); lg = np.asarray(O['legal'][rs]); ac = np.asarray(O['act'][rs]); pl = np.asarray(O['ply'][rs])
                z, a, acg = tokens(px[gi, di], g[gi, di], lg[gi, di], ac[gi, di], pl[gi, di], bl[rs][gi], hd[rs][gi], grad=False); x = ((torch.cat([z, a], 1) - mu) / sd).clamp(-8, 8)
                X = torch.zeros((len(rs), Kk, x.shape[1]), device=dev); PL = torch.ones((len(rs), Kk), dtype=torch.long, device=dev); MV = torch.full((len(rs), Kk), 362, dtype=torch.long, device=dev); VA = torch.zeros((len(rs), Kk), dtype=torch.bool, device=dev)
                gi_t = torch.from_numpy(gi).to(dev); di_t = torch.from_numpy(di).to(dev); X[gi_t, di_t] = x; PL[gi_t, di_t] = torch.from_numpy(pl[gi, di].astype(np.int64)).to(dev); MV[gi_t, di_t] = act2code(acg); VA[gi_t, di_t] = True; VA[:, 0] |= ~VA.any(1)
                args_ = (X, PL, VA, torch.from_numpy(bl[rs]).to(dev), MV)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    ee = net.embed(*args_).float(); net.out[3] = nn.Identity(); ff = net.embed(*args_).float(); net.out[3] = bn
                inv = np.argsort(o); e.append(ee[inv]); f.append(ff[inv])
        e = torch.cat(e); f = torch.cat(f); g_ = (f - f.mean(0)) / torch.sqrt(f.var(0) + bn.eps) * bn.weight + bn.bias; settrain() if was else net.eval(); return e.cpu().numpy(), g_.detach().float().cpu().numpy()

    settrain(); del ih
    while step < steps:
        order = rng.permutation(len(blocks))
        for g0 in range(0, len(order), v.buf):
            if step >= steps: break
            B = None; B = load([blocks[i] for i in order[g0:g0 + v.buf]]); ng = len(B['y']); perm = np.concatenate([rng.permutation(ng) for _ in range(v.reuse)])
            for i in range(0, len(perm) - v.G + 1, v.G):
                if step >= steps: break
                sel = np.sort(perm[i:i + v.G]); args_, valid = batch(B, sel); z, a, ac = tok2(args_); x = ((torch.cat([z, a], 1) - mu) / sd).clamp(-8, 8).reshape(len(sel), KK, -1)
                va = torch.from_numpy(valid).to(dev); ply = torch.from_numpy(args_[4].astype(np.int64)).to(dev).reshape(len(sel), KK); mv = torch.where(va, act2code(ac).reshape(len(sel), KK), torch.full_like(ply, 362)); y = torch.from_numpy(B['y'][sel]).to(dev); bl = torch.from_numpy(B['black'][sel]).to(dev)
                with torch.autocast('cuda', dtype=torch.bfloat16): loss = net(x, ply, va, bl, y, mv)
                if not torch.isfinite(loss): print('non-finite loss at step', step, '- abort (v1 pilot: trunk lr 2e-4 diverged in the warm-up)', flush=True); sys.exit(3)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if trainable: nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt); scaler.update(); sch.step(); step += 1; tot += loss.item(); cnt += 1
                if step % 200 == 0 or step == steps: log.append({'step': step, 'epoch': step * v.G / total, 'loss': tot / cnt, 'sec': time.time() - t0}); print(json.dumps(log[-1]), flush=True); tot = cnt = 0
                if v.snap_every and step % v.snap_every == 0 and step < steps: E_, bnE = official(); np.save(os.path.join(v.out, f'off_embbn_step{step}.npy'), bnE); torch.save(trunk_state(), os.path.join(v.out, f'trunk_ft_step{step}.pt')); print('snapshot step', step, flush=True)
                if v.ckpt_every and step % v.ckpt_every == 0 and step < steps: torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'scaler': scaler.state_dict(), 'trunk': trunk_state(), 'step': step, 'log': log, 'mu': mu.cpu(), 'sd': sd.cpu()}, ck + '.tmp'); os.replace(ck + '.tmp', ck)
    torch.save(trunk_state(), os.path.join(v.out, 'trunk_ft.pt')); torch.save({'model': net.state_dict(), 'args': vars(v), 'mu': mu.cpu(), 'sd': sd.cpu(), 'classes': cls}, os.path.join(v.out, 'model.pt'))
    if not v.no_final: E_, bnE = official(); np.save(os.path.join(v.out, 'off_emb.npy'), E_); np.save(os.path.join(v.out, 'off_embbn.npy'), bnE)
    json.dump({'args': vars(v), 'games': total, 'classes': C, 'log': log, 'seconds': time.time() - t0}, open(os.path.join(v.out, 'LOG.json'), 'w'), indent=1); print('saved', v.out, flush=True)


if __name__ == '__main__':
    main()
