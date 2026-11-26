"""v4 = v3 + --captures : bundle built by seqcap_20261115_v1.py (CAPT / CAPP capture events); the board planes then show the real
position (captured stones removed) and a 5th plane marks the stones captured by the move itself.
v3 = v2 + --aug-d4 P : during training each game is, with probability P, rotated / mirrored by a random one of the 8 board symmetries
(all coordinates of the game together; pass and padding unchanged).  Embeddings for evaluation / official games use the original orientation.
     + --pk K [--proto-lam LAM] : PK batches (bs/K players, K games each) and a set-level loss on top of the AM-softmax -- the mean of g
       games of a player has to match the mean of the same player's other K-g games against the other players in the batch (both directions).
v2 of the standalone trainer: optional board CNN (--board C --board-layers N --board-patch K), otherwise identical to v1.
Board CNN (design after the comp2 session's idnet.py, read-only reference): stones on the board before each move of the classified side
(captures ignored; planes own / opponent / the move / off-board), K x K neighbourhood of the move, small CNN, read out at the centre.
Standalone trainer (numpy + torch only; runs locally or on the 4090 VM) for the moves-only identity Transformer.
Network layout after the comp2 session's idnet.py (cosine classifier with additive margin, pooled over the target side's moves).
usage: python seqid_core_20261115_v1.py BUNDLE.npz OUT_DIR --fold K [--epochs 20 --L 160 --d 256 --layers 6 ...]
bundle: M[N,Lmax] uint16, LEN, FB, BLK, HD, OWN (owner index per game), KEEP_f{K} (bool train mask), EV_f{K} (episode game indices, optional),
        OFF_M, OFF_LEN, OFF_FB, OFF_BLK, OFF_HD (official games)
writes OUT_DIR/f{K}/: model.pt, LOG.json, ev_own.npy, ev_opp.npy (order of EV_f{K}), off_own.npy, off_opp.npy
"""
import sys, os, json, argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
PAD = 362; PASS = 361


class Net(nn.Module):
    def __init__(self, ncls, d, layers, L, drop, de=384, board=0, bl=4, bpatch=9, captures=False):
        super().__init__(); self.board = board; self.bpatch = bpatch
        if board:
            cv = []; cin = 5 if captures else 4; self.captures = captures
            for _ in range(bl): cv += [nn.Conv2d(cin, board, 3, padding=1), nn.ReLU(inplace=True)]; cin = board
            self.cnn = nn.Sequential(*cv); self.bproj = nn.Sequential(nn.Linear(2 * board, d), nn.GELU(), nn.Linear(d, d))
        self.mv = nn.Embedding(363, d, padding_idx=PAD); self.pos = nn.Embedding(L, d); self.who = nn.Embedding(2, d); self.col = nn.Embedding(2, d); self.hd = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, drop, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
        self.emb = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, de)); self.W = nn.Parameter(torch.randn(ncls, de) * .01)

    def embed(self, mv, length, side_black, first_black, hd, capt=None, capp=None):
        B, Lx = mv.shape; ar = torch.arange(Lx, device=mv.device); pad = ar[None] >= length[:, None]
        mover_black = (ar[None] % 2 == 0) == first_black[:, None]; mine = mover_black == side_black[:, None]
        h = self.mv(mv) + self.pos(ar)[None] + self.who(mine.long()) + self.col(mover_black.long()) + self.hd(hd.clamp(max=1))[:, None]
        if self.board:
            oh = torch.zeros(B, Lx, 362, device=mv.device, dtype=h.dtype).scatter_(2, mv.clamp(max=361)[..., None], 1.0)[..., :361]; mb = mover_black[..., None].to(h.dtype)
            db = oh * mb; dw = oh * (1 - mb); planes = 3
            if self.captures:      # captured stones leave the board; a stone captured by a black move is white and vice versa
                tt = capt.long(); val = tt < Lx; co = torch.zeros(B, Lx, 362, device=mv.device, dtype=h.dtype); bi = torch.arange(B, device=mv.device)[:, None].expand_as(tt); co[bi[val], tt[val], capp.long()[val]] = 1.0; co = co[..., :361]; db = db - co * (1 - mb); dw = dw - co * mb; planes = 4
            bk = torch.cumsum(db, 1) - db; wt = torch.cumsum(dw, 1) - dw; own = torch.where(mover_black[..., None], bk, wt); opp = torch.where(mover_black[..., None], wt, bk)
            sel = (mine & ~pad).view(-1); pl = torch.stack([own, opp, oh] + ([co] if self.captures else []), 2).view(B * Lx, planes, 19, 19)[sel].clamp(0, 1); K = self.bpatch; r = K // 2
            pp = torch.cat([Fn.pad(pl, (r, r, r, r)), Fn.pad(torch.zeros_like(pl[:, :1]), (r, r, r, r), value=1.0)], 1); ms = mv.view(-1)[sel].clamp(max=360); ak = torch.arange(K, device=mv.device); n_ = torch.arange(len(ms), device=mv.device)
            pt = pp[n_[:, None, None], :, (ms // 19)[:, None, None] + ak[None, :, None], (ms % 19)[:, None, None] + ak[None, None, :]].permute(0, 3, 1, 2)
            f = self.cnn(pt.contiguous(memory_format=torch.channels_last)); bf = self.bproj(torch.cat([f[:, :, r, r], f.mean((2, 3))], 1)); add = torch.zeros(B * Lx, bf.shape[1], device=h.device, dtype=bf.dtype); add[sel] = bf; h = h + add.view(B, Lx, -1).to(h.dtype)
        h = self.norm(self.enc(h, src_key_padding_mask=pad)); m = (mine & ~pad).float()[..., None]
        return self.emb(torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), h.masked_fill(m == 0, -1e4).max(1).values], 1))

    def logits(self, e): return Fn.normalize(e.float(), dim=1) @ Fn.normalize(self.W.float(), dim=1).t()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bundle'); ap.add_argument('out'); ap.add_argument('--fold', default='0')
    ap.add_argument('--epochs', type=float, default=20); ap.add_argument('--L', type=int, default=160); ap.add_argument('--d', type=int, default=256); ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--bs', type=int, default=512); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=.05); ap.add_argument('--dropout', type=float, default=.1)
    ap.add_argument('--scale', type=float, default=30.); ap.add_argument('--margin', type=float, default=.2); ap.add_argument('--cap', type=int, default=300); ap.add_argument('--min-games', type=int, default=8)
    ap.add_argument('--mask', type=float, default=0.); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--eval-every', type=int, default=1000); ap.add_argument('--amp', default='bf16'); ap.add_argument('--board', type=int, default=0); ap.add_argument('--board-layers', type=int, default=4); ap.add_argument('--board-patch', type=int, default=9); ap.add_argument('--aug-d4', type=float, default=0.); ap.add_argument('--pk', type=int, default=0); ap.add_argument('--captures', action='store_true'); ap.add_argument('--color-classes', action='store_true'); ap.add_argument('--proto-lam', type=float, default=0.)
    v = ap.parse_args(); out = os.path.join(v.out, f'f{v.fold}'); ck = os.path.join(out, 'ckpt.pt'); assert not os.path.exists(out) or os.path.exists(ck), 'output exists without a checkpoint'; os.makedirs(out, exist_ok=True); torch.manual_seed(v.seed); rng = np.random.default_rng(v.seed); dev = 'cuda'; t0 = time.time()
    amp = torch.bfloat16 if v.amp == 'bf16' else torch.float16; Z = np.load(v.bundle); L = v.L
    M = torch.tensor(Z['M'][:, :L].astype(np.int64)); LEN = torch.tensor(np.minimum(Z['LEN'].astype(np.int64), L)); FB = torch.tensor(Z['FB']); BLK = torch.tensor(Z['BLK']); HD = torch.tensor(Z['HD'].astype(np.int64))
    CAP = (torch.tensor(Z['CAPT']), torch.tensor(Z['CAPP'].astype(np.int16))) if v.captures else ()
    own = Z['OWN'].astype(np.int64); keep = Z[f'KEEP_f{v.fold}'].copy(); nk = np.bincount(own[keep], minlength=own.max() + 1); keep &= nk[own] >= v.min_games
    cls_owner = np.flatnonzero(np.bincount(own[keep], minlength=own.max() + 1) > 0); cls_of = -np.ones(own.max() + 1, np.int64); cls_of[cls_owner] = np.arange(len(cls_owner)); Y = torch.tensor(cls_of[own]); C = len(cls_owner)
    if v.color_classes: Y = torch.where(Y >= 0, Y * 2 + BLK.long(), Y); C = 2 * C      # one class per (player, colour): the embedding is not asked to bring a player's black and white games together
    tr = np.flatnonzero(keep); order = rng.permutation(tr); seen = {}; ho = []
    for i in order:
        o = own[i]
        if nk[o] >= 20 and seen.get(o, 0) < 1: ho.append(i); seen[o] = 1
    ho = np.array(sorted(ho))[:4000]; tr = np.setdiff1d(tr, ho); n_own = np.bincount(own[tr], minlength=own.max() + 1)[own[tr]]; pkeep = np.minimum(1., v.cap / n_own)
    spe = int(pkeep.sum()) // v.bs; steps = int(v.epochs * spe); print(f'classes {C} train games {len(tr)} per-epoch {int(pkeep.sum())} held-out {len(ho)} steps {steps} load {time.time() - t0:.0f}s', flush=True)
    net = Net(C, v.d, v.layers, L, v.dropout, board=v.board, bl=v.board_layers, bpatch=v.board_patch, captures=v.captures).to(dev)
    if v.board: net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=v.lr, weight_decay=v.wd); sch = torch.optim.lr_scheduler.OneCycleLR(opt, v.lr, total_steps=steps, pct_start=.05)

    def emb(*ts):
        net.eval(); o = []
        with torch.no_grad(), torch.autocast('cuda', dtype=amp):
            for i in range(0, len(ts[0]), 512): o.append(net.embed(*[t[i:i + 512].to(dev) for t in ts]).float().cpu())
        net.train(); return torch.cat(o)

    TD4 = np.tile(np.arange(363), (8, 1)); r_, c_ = np.divmod(np.arange(361), 19)
    for k in range(8):
        rr, cc = (c_, r_) if k & 4 else (r_, c_); rr = 18 - rr if k & 1 else rr; cc = 18 - cc if k & 2 else cc; TD4[k, :361] = rr * 19 + cc
    TD4 = torch.tensor(TD4, device=dev)
    if v.pk:
        yt = cls_of[own[tr]]; o_ = np.argsort(yt, kind='stable'); cnt_ = np.bincount(yt, minlength=C); byc = np.split(tr[o_], np.cumsum(cnt_)[:-1]); elig = np.flatnonzero(cnt_ >= v.pk); print('PK sampling: eligible classes', len(elig), flush=True)
    step = 0; tot = n = 0; log = []
    if os.path.exists(ck):      # resume after a crash (data order differs from an uninterrupted run)
        st = torch.load(ck, map_location=dev, weights_only=False); net.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sch.load_state_dict(st['sch']); step = st['step']; log = st['log']; rng = np.random.default_rng(v.seed + step); print('resumed at step', step, flush=True)
    while step < steps:
        perm = rng.permutation(tr[rng.random(len(tr)) < pkeep])
        if v.pk:      # PK batches: bs/pk players (uniform over players with >= pk games), pk random games each, grouped by player
            perm = np.concatenate([np.concatenate([rng.choice(byc[c], v.pk, replace=False) for c in rng.choice(elig, v.bs // v.pk, replace=False)]) for _ in range(spe)])
        for i in range(0, len(perm) - v.bs + 1, v.bs):
            if step >= steps: break
            j = torch.as_tensor(perm[i:i + v.bs]); mv = M[j].to(dev); y = Y[j].to(dev)
            if v.aug_d4 > 0: kk = torch.randint(0, 8, (len(j),), device=dev) * (torch.rand(len(j), device=dev) < v.aug_d4); mv = TD4[kk[:, None], mv]
            cap = [c[j].to(dev) for c in CAP]
            if v.aug_d4 > 0 and cap: cap[1] = TD4[kk[:, None], cap[1].long()]
            if v.mask > 0: mv = torch.where((torch.rand(mv.shape, device=dev) < v.mask) & (mv != PAD), torch.full_like(mv, PASS), mv)
            with torch.autocast('cuda', dtype=amp): e = net.embed(mv, LEN[j].to(dev), BLK[j].to(dev), FB[j].to(dev), HD[j].to(dev), *cap)
            lo = net.logits(e); loss = Fn.cross_entropy(v.scale * (lo - v.margin * Fn.one_hot(y, C)), y)
            if v.pk and v.proto_lam > 0:      # set-level loss: mean of g games of a player must match the mean of that player's other games against the other players of the batch
                en = Fn.normalize(e.float(), dim=1).view(v.bs // v.pk, v.pk, -1); gsz = int(rng.integers(1, v.pk)); pg = Fn.normalize(en[:, :gsz].mean(1), dim=1); pq = Fn.normalize(en[:, gsz:].mean(1), dim=1); sim = v.scale * (pq @ pg.t()); ar_ = torch.arange(len(pg), device=dev)
                loss = loss + v.proto_lam * .5 * (Fn.cross_entropy(sim, ar_) + Fn.cross_entropy(sim.t(), ar_))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); step += 1; tot += loss.item(); n += 1
            if step % v.eval_every == 0 or step == steps:
                j = torch.as_tensor(ho)
                with torch.no_grad(): acc = float((net.logits(emb(M[j], LEN[j], BLK[j], FB[j], HD[j], *[c[j] for c in CAP]).to(dev)).argmax(1).cpu() == Y[j]).float().mean())
                log.append({'step': step, 'epoch': step / spe, 'loss': tot / n, 'heldout_top1': acc, 'sec': time.time() - t0}); print(json.dumps(log[-1]), flush=True); tot = n = 0; torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'step': step, 'log': log}, ck + '.tmp'); os.replace(ck + '.tmp', ck)
    torch.save({'model': net.state_dict(), 'args': vars(v), 'classes': cls_owner}, os.path.join(out, 'model.pt'))
    if f'EV_f{v.fold}' in Z.files:
        j = torch.as_tensor(Z[f'EV_f{v.fold}'].astype(np.int64))
        np.save(os.path.join(out, 'ev_own.npy'), emb(M[j], LEN[j], BLK[j], FB[j], HD[j], *[c[j] for c in CAP]).numpy()); np.save(os.path.join(out, 'ev_opp.npy'), emb(M[j], LEN[j], ~BLK[j], FB[j], HD[j], *[c[j] for c in CAP]).numpy())
    OM = torch.tensor(Z['OFF_M'][:, :L].astype(np.int64)); OL = torch.tensor(np.minimum(Z['OFF_LEN'].astype(np.int64), L)); OF = torch.tensor(Z['OFF_FB']); OB = torch.tensor(Z['OFF_BLK']); OH = torch.tensor(Z['OFF_HD'].astype(np.int64)); OC = (torch.tensor(Z['OFF_CAPT']), torch.tensor(Z['OFF_CAPP'].astype(np.int16))) if v.captures else ()
    np.save(os.path.join(out, 'off_own.npy'), emb(OM, OL, OB, OF, OH, *OC).numpy()); np.save(os.path.join(out, 'off_opp.npy'), emb(OM, OL, ~OB, OF, OH, *OC).numpy())
    json.dump({'args': vars(v), 'classes': C, 'train_games': int(len(tr)), 'log': log, 'seconds': time.time() - t0}, open(os.path.join(out, 'LOG.json'), 'w'), indent=1); print('saved', out, flush=True)


if __name__ == '__main__':
    main()
