"""Moves-only sequence identity network for competition 1 (end-to-end, no engine features).
Idea and network layout follow the comp2 session's idnet.py (Go Competition 2/runs/kata_quality_20261114_v1/idnet.py, read-only reference):
a Transformer over the raw move sequence (both sides, raw orientation), pooled over the TARGET side's moves, cosine classifier over
training owners with additive margin.  Here the unit embedding is used open-set: cosine between mean query and mean support embeddings.

Stages
  data                       -> runs/seqid_20261115_v1/moves.npz (all meta games, index-aligned), official_moves.npz, safe.npy
  train <split> <fold> <tag> -> runs/seqid_20261115_v1/<tag>/<split>_f<k>/ (model, episode embeddings ze.npy, seq score matrix)
  fuse <tag>                 -> cross-fit comparison against the v6b recipe comps (bigbank3 eval 'b2e3')
Fold exclusion = owners of the fold's players + the four families of every episode game (same rule as bigbank3_eval).
"""
from pathlib import Path
import sys, argparse, re, time, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RUN = ROOT / 'runs/seqid_20261115_v1'; LMAX = 200; PAD = 362; PASS = 361; K = 3
MV = re.compile(r';([BW])\[([a-s]{0,2})\]')


def tokens(sgf):
    mv = MV.findall(sgf); t = [(ord(c[0]) - 97) * 19 + (ord(c[1]) - 97) if len(c) == 2 else PASS for _, c in mv[:LMAX]]
    return t, len(mv), (mv[0][0] == 'B') if mv else True


def pack(sgfs):
    M = np.full((len(sgfs), LMAX), PAD, np.uint16); LEN = np.zeros(len(sgfs), np.uint16); FB = np.ones(len(sgfs), bool)
    for i, s in enumerate(sgfs):
        t, n, fb = tokens(s); M[i, :len(t)] = t; LEN[i] = min(n, 65535); FB[i] = fb
    return M, LEN, FB


def data():
    import bigbank90_20261114_v1 as bb
    p = bb.p; s = bb.s; old = bb.old; RUN.mkdir(exist_ok=False); m = p.b.meta(); owners = m['owners'].astype(str); N = len(owners)
    # same exclusion as bigbank prepare: exposed dev identities, historical validation identities and their families
    players = set(p.b.read(s.RUN / 'EPISODE_screen.json')['candidates']); previous = set(p.b.read(old.RUN / 'EPISODE_screen.json')['candidates']); earlier = set(p.b.read(p.b.RUN / 'calibration/ep0.json')['candidates'])
    hist = set(owners[np.load(p.DENSE.parent / 'validation_indices.npy')]); excluded = np.isin(owners, sorted(players | previous | earlier | hist)); safe = (~excluded) & (m['hd'] <= 1)
    for f in p.b.FAMILIES: safe &= ~np.isin(m[f], np.unique(m[f][excluded]))
    np.save(RUN / 'safe.npy', safe, allow_pickle=False); print('safe games', int(safe.sum()), 'owners', len(set(owners[safe])), flush=True)
    FB, keys = moves_meta(p, m), moves_official(p)
    p.b.write(RUN / 'DATA.json', {'meta_games': N, 'safe_games': int(safe.sum()), 'official_games': len(keys), 'LMAX': LMAX, 'first_black_share': float(FB.mean()), 'source_SHA256': p.b.sha(__file__)})


def moves_meta(p, m):
    """move tokens of every SAFE-meta game (index-aligned) -> moves.npz"""
    from goid_v4.pilot_scope_v1 import PointReader
    N = len(m['owners']); M = np.full((N, LMAX), PAD, np.uint16); LEN = np.zeros(N, np.uint16); FB = np.ones(N, bool); reader = PointReader(p.ROOT / 'data/processed/audit.sqlite'); t = time.time()
    try:
        for st in range(0, N, 512):
            ii = np.arange(st, min(N, st + 512)); texts = {r['row_id']: r['sgf_content'] for r in reader.sample_text(m['rows'][ii])}
            M[ii], LEN[ii], FB[ii] = pack([texts[int(m['rows'][i])] for i in ii])
            if st % 51200 == 0: print(st, f'{time.time() - t:.0f}s', flush=True)
    finally: reader.close()
    print('move-count mismatch share vs meta', float((LEN.astype(np.int64) != m['moves'].astype(np.int64)).mean()), flush=True)
    np.savez(RUN / 'moves.npz', M=M, LEN=LEN, FB=FB); return FB


def moves_official(p):
    """move tokens of the official test games in keys.json order -> official_moves.npz"""
    games, pools = p.b.load_official(); keys = p.b.read(p.b.RUN / 'official_stat/keys.json'); Mo, Lo, Fo = pack([games[k].sgf for k in keys])
    np.savez(RUN / 'official_moves.npz', M=Mo, LEN=Lo, FB=Fo, black=np.array([games[k].color == 'B' for k in keys]), hd=np.array([int(re.search(r'HD\[(\d+)\]', games[k].sgf).group(1)) for k in keys], np.uint8)); return keys


def safe2():
    """bigbank2/3 rule (production bank): only historical validation owners (incl. sealed 400) and their families are excluded."""
    import bigbank90_20261114_v1 as bb
    p = bb.p; m = p.b.meta(); owners = m['owners'].astype(str); hist = set(owners[np.load(p.DENSE.parent / 'validation_indices.npy')])
    excluded = np.isin(owners, sorted(hist)); safe = (~excluded) & (m['hd'] <= 1)
    for f in p.b.FAMILIES: safe &= ~np.isin(m[f], np.unique(m[f][excluded]))
    assert not set(owners[safe]) & hist; f = RUN / 'safe_v2.npy'; assert not f.exists(); np.save(f, safe, allow_pickle=False)
    uo, c = np.unique(owners[safe], return_counts=True); print('safe_v2 games', int(safe.sum()), 'owners', len(uo), 'owners>=8', int((c >= 8).sum()), 'games/owner q', np.quantile(c, [.1, .5, .9, .99]).tolist())


def data2():
    """move tokens of the old 'validation' split games (bigbank4 EXTMETA order) -> moves_ext.npz"""
    import bigbank4_90_20261115_v1 as b4
    e = b4.extmeta(); con = b4.db(); rows = [int(r) for r in e['rows']]; sg = []
    for st in range(0, len(rows), 500):
        rr = rows[st:st + 500]; t = dict(con.execute(f'SELECT row_id,sgf_content FROM games WHERE row_id IN ({",".join("?" for _ in rr)})', rr).fetchall()); sg += [t[r] for r in rr]
    con.close(); M, LEN, FB = pack(sg); f = RUN / 'moves_ext.npz'; assert not f.exists(); np.savez(f, M=M, LEN=LEN, FB=FB)
    print('ext games', len(rows), 'move-count mismatch share', float((LEN.astype(np.int64) != e['moves'].astype(np.int64)).mean()), 'first-black share', float(FB.mean()))


def universe():
    """all usable training games: SAFE meta (fit split, incl. the 723 'hist' owners) + old 'validation' split; holdout / sealed 400 are not here."""
    import bigbank4_90_20261115_v1 as b4
    p = b4.p; m = p.b.meta(); e = b4.extmeta(); Z = np.load(RUN / 'moves.npz'); X = np.load(RUN / 'moves_ext.npz')
    u = {k: np.concatenate([m[k], e[k].astype(m[k].dtype)]) for k in ('owners', 'ranks', 'colors', 'hd', 'raw', 'move', 'canonical', 'group')}
    u.update(M=np.concatenate([Z['M'], X['M']]), LEN=np.concatenate([Z['LEN'], X['LEN']]), FB=np.concatenate([Z['FB'], X['FB']]), n_meta=len(m['owners'])); return p, u


def build_net(ncls, d, layers, L, drop, de=384):
    import torch, torch.nn as nn, torch.nn.functional as Fn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mv = nn.Embedding(363, d, padding_idx=PAD); self.pos = nn.Embedding(L, d); self.who = nn.Embedding(2, d); self.col = nn.Embedding(2, d); self.hd = nn.Embedding(2, d)
            layer = nn.TransformerEncoderLayer(d, 4, 4 * d, drop, batch_first=True, norm_first=True, activation='gelu')
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.norm = nn.LayerNorm(d)
            self.emb = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, de)); self.W = nn.Parameter(torch.randn(ncls, de) * .01)

        def embed(self, mv, length, side_black, first_black, hd):
            B, Lx = mv.shape; ar = torch.arange(Lx, device=mv.device); pad = ar[None] >= length[:, None]
            mover_black = (ar[None] % 2 == 0) == first_black[:, None]; mine = mover_black == side_black[:, None]
            h = self.mv(mv) + self.pos(ar)[None] + self.who(mine.long()) + self.col(mover_black.long()) + self.hd(hd.clamp(max=1))[:, None]
            h = self.norm(self.enc(h, src_key_padding_mask=pad)); m = (mine & ~pad).float()[..., None]
            return self.emb(torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), h.masked_fill(m == 0, -1e4).max(1).values], 1))

        def logits(self, e):
            return Fn.normalize(e.float(), dim=1) @ Fn.normalize(self.W.float(), dim=1).t()
    return Net()


def fold_drop(split, k, p, owners):
    import bigbank3_eval_20261114_v1 as ev3
    h = ev3.h; bb = ev3.bb; b2 = ev3.b2; cf = ev3.cf
    if split == 'g1':
        eps = p.b.read(h.c1.RUN / 'EPISODES.json'); ids1 = np.load(bb.RUN / 'train_indices.npy'); lab1 = np.load(bb.RUN / 'train_labels.npy'); drop = set(owners[ids1[lab1 % K == k]])
    else:
        ids2 = np.load(b2.RUN / 'train_indices.npy'); lab2 = np.load(b2.RUN / 'train_labels.npy'); eps = p.b.read(cf.RUN / 'EPISODES.json'); _, fold = cf.episodes(ids2, lab2)
        drop = set(owners[ids2][np.array([fold[int(l)] == k for l in lab2])])
    return eps[k], drop, ev3


def train(v):
    import torch, torch.nn.functional as Fn
    p, m = universe(); owners = m['owners'].astype(str); out = RUN / v.tag / f'{v.split}_f{v.fold}'; out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(v.seed); rng = np.random.default_rng(v.seed); dev = 'cuda'; t0 = time.time()
    L = v.L; M = torch.tensor(m['M'][:, :L].astype(np.int64)); LEN = torch.tensor(np.minimum(m['LEN'].astype(np.int64), L)); FB = torch.tensor(m['FB'])
    BLK = torch.tensor(m['colors'] == 0); HD = torch.tensor(m['hd'].astype(np.int64)); keep = m['hd'] <= 1
    if v.bank == 'old': keep[m['n_meta']:] = False; keep[:m['n_meta']] &= np.load(RUN / 'safe_v2.npy')   # the 4,026-owner universe of bigbank2/3
    if v.split != 'full':
        ep, drop, ev3 = fold_drop(v.split, v.fold, p, owners); used = np.array(sorted({i for ss in ep['supports'] for i in ss} | {i for z in ep['questions'] for i in z['indices']}))
        keep &= ~np.isin(owners, sorted(drop))
        for f in p.b.FAMILIES: keep &= ~np.isin(m[f], np.unique(m[f][used]))
        assert not set(owners[keep]) & set(owners[used])
    uo, inv, cnt = np.unique(owners, return_inverse=True, return_counts=True); nk = np.bincount(inv[keep], minlength=len(uo)); keep &= nk[inv] >= v.min_games
    cls_owner = np.flatnonzero(np.bincount(inv[keep], minlength=len(uo)) > 0); cls_of = -np.ones(len(uo), np.int64); cls_of[cls_owner] = np.arange(len(cls_owner)); Y = torch.tensor(cls_of[inv]); C = len(cls_owner)
    tr = np.flatnonzero(keep); ho = []
    for c in rng.choice(cls_owner, min(2000, C), replace=False):
        ii = tr[inv[tr] == c]
        if len(ii) >= 20: ho += rng.choice(ii, 2, replace=False).tolist()
    ho = np.array(sorted(ho)); tr = np.setdiff1d(tr, ho); n_own = np.bincount(inv[tr], minlength=len(uo))[inv[tr]]; pkeep = np.minimum(1., v.cap / n_own)
    spe = int(pkeep.sum()) // v.bs; steps = int(v.epochs * spe); print(f'classes {C} train games {len(tr)} per-epoch {int(pkeep.sum())} held-out {len(ho)} steps {steps} load {time.time() - t0:.0f}s', flush=True)
    net = build_net(C, v.d, v.layers, L, v.dropout).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=v.lr, weight_decay=v.wd); sch = torch.optim.lr_scheduler.OneCycleLR(opt, v.lr, total_steps=steps, pct_start=.05)
    scaler = torch.amp.GradScaler('cuda')

    def emb(idx, Mx=M, Lx=LEN, Bx=BLK, Fx=FB, Hx=HD):
        net.eval(); o = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(idx), 1024):
                j = torch.as_tensor(idx[i:i + 1024]); o.append(net.embed(Mx[j].to(dev), Lx[j].to(dev), Bx[j].to(dev), Fx[j].to(dev), Hx[j].to(dev)).float().cpu())
        net.train(); return torch.cat(o)

    step = 0; tot = n = 0; log = []
    while step < steps:
        perm = rng.permutation(tr[rng.random(len(tr)) < pkeep])
        for i in range(0, len(perm) - v.bs + 1, v.bs):
            if step >= steps: break
            j = torch.as_tensor(perm[i:i + v.bs]); mv = M[j].to(dev); y = Y[j].to(dev)
            if v.mask > 0: mv = torch.where((torch.rand(mv.shape, device=dev) < v.mask) & (mv != PAD), torch.full_like(mv, PASS), mv)
            with torch.autocast('cuda', dtype=torch.float16): e = net.embed(mv, LEN[j].to(dev), BLK[j].to(dev), FB[j].to(dev), HD[j].to(dev))
            lo = net.logits(e); lo = v.scale * (lo - v.margin * Fn.one_hot(y, C)); loss = Fn.cross_entropy(lo, y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
            step += 1; tot += loss.item(); n += 1
            if step % v.eval_every == 0 or step == steps:
                with torch.no_grad(): acc = float((net.logits(emb(ho).to(dev)).argmax(1).cpu() == Y[ho]).float().mean())
                log.append({'step': step, 'epoch': step / spe, 'loss': tot / n, 'heldout_top1': acc, 'sec': time.time() - t0}); print(json.dumps(log[-1]), flush=True); tot = n = 0
    torch.save({'model': net.state_dict(), 'args': vars(v), 'classes': uo[cls_owner]}, out / 'model.pt'); res = {'args': vars(v), 'classes': C, 'train_games': int(len(tr)), 'log': log}
    if v.split != 'full':
        h = ev3.h; A = ev3.A; ze = emb(used).numpy().astype(np.float64); np.save(out / 'ze.npy', ze.astype(np.float32)); np.save(out / 'eids.npy', used); S = h.setscores(ze, used, ep)[0]; np.save(out / 'seq.npy', S)
        tg = np.array([z['target'] for z in ep['questions']]); q = np.array([z['q'] for z in ep['questions']]); s = A.sid(S, tg)
        res['seq_raw'] = {'all': float(s.mean()), **{str(a): float(s[q == a].mean()) for a in (1, 2, 3, 5)}}; print('seq alone (no Sinkhorn)', res['seq_raw'], flush=True)
    O = np.load(RUN / 'official_moves.npz'); zo = emb(np.arange(len(O['M'])), torch.tensor(O['M'][:, :L].astype(np.int64)), torch.tensor(np.minimum(O['LEN'].astype(np.int64), L)), torch.tensor(O['black']), torch.tensor(O['FB']), torch.tensor(O['hd'].astype(np.int64)))
    np.save(out / 'official_ze.npy', zo.numpy().astype(np.float32)); p.b.write(out / 'RESULT.json', res)


def oppemb(v):
    """Embeddings of the OPPONENT side of episode / official games with a finished fold model (who does this player meet?)."""
    import torch
    import bigbank90_20261114_v1 as bb
    p = bb.p; m = p.b.meta(); out = RUN / v.tag / f'{v.split}_f{v.fold}'; ck = torch.load(out / 'model.pt', map_location='cpu', weights_only=False); a = ck['args']; L = a['L']; dev = 'cuda'
    net = build_net(len(ck['classes']), a['d'], a['layers'], L, a['dropout']).to(dev); net.load_state_dict(ck['model']); net.eval()
    def emb(Mx, Lx, Bx, Fx, Hx):
        o = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for i in range(0, len(Mx), 1024):
                o.append(net.embed(*[torch.as_tensor(t[i:i + 1024]).to(dev) for t in (Mx, Lx, Bx, Fx, Hx)]).float().cpu())
        return torch.cat(o).numpy().astype(np.float32)
    Z = np.load(RUN / 'moves.npz'); O = np.load(RUN / 'official_moves.npz')
    if v.split != 'full':
        e = np.load(out / 'eids.npy'); np.save(out / 'ze_opp.npy', emb(Z['M'][e][:, :L].astype(np.int64), np.minimum(Z['LEN'][e].astype(np.int64), L), m['colors'][e] != 0, Z['FB'][e], m['hd'][e].astype(np.int64)))
    np.save(out / 'official_ze_opp.npy', emb(O['M'][:, :L].astype(np.int64), np.minimum(O['LEN'].astype(np.int64), L), ~O['black'], O['FB'], O['hd'].astype(np.int64)))


def fuse(v):
    import bigbank3_eval_20261114_v1 as ev3
    p = ev3.p; A = ev3.A; h = ev3.h; b3 = ev3.b3; rep = {}; mr = p.b.meta()['ranks']
    for split, epf in (('g1', h.c1.RUN / 'EPISODES.json'), ('confirm', h.RUN / 'confirm/EPISODES.json')):
        eps = p.b.read(epf); folds = [k for k in range(K) if (RUN / v.tag / f'{split}_f{k}/seq.npy').exists()]
        if not folds: continue
        V = {k: dict(np.load(b3.RUN / f'eval_{split}/comps_f{k}.npz')) for k in folds}; S = {k: np.load(RUN / v.tag / f'{split}_f{k}/seq.npy') for k in folds}
        tg = {k: np.array([z['target'] for z in eps[k]['questions']]) for k in folds}; ql = {k: np.array([z['q'] for z in eps[k]['questions']]) for k in folds}; Q = np.concatenate([ql[k] for k in folds])
        def score(k, w, ref='b2e3'): return (1 - w) * A.z(V[k][ref]) + w * A.z(S[k])
        def evals(w):
            bal = []
            for k in folds:
                o = [j for j in folds if j != k] or [k]; T = A.calibrate(A.z(np.vstack([score(j, w) for j in o])), np.concatenate([tg[j] for j in o]), np.concatenate([ql[j] for j in o]))
                bal.append(A.sid(A.new_transform(score(k, w), ql[k], T), tg[k]))
            return np.concatenate(bal)
        ref = evals(0.); B = np.random.default_rng(0).integers(0, len(Q), (2000, len(Q))); rep[split] = {'folds': folds, 'in_fold_temps': len(folds) == 1}
        # rank-uniform weights: official new pools look uniform over the 10 ranks, cross-fit targets are not
        rk = np.concatenate([[mr[z['indices'][0]] for z in eps[k]['questions']] for k in folds]); ur, ui, uc = np.unique(rk, return_inverse=True, return_counts=True); W = (1 / len(ur)) / uc[ui]
        for w in (0., .1, .2, .3, .4, .5, .6, .8, 1.):
            b = evals(w); d = b - ref; rep[split][f'w{w}'] = {'bal': float(b.mean()), 'bal_rank_uniform': float((b * W).sum()), 'delta_rank_uniform': float((d * W).sum()), 'by_q': {str(n): float(b[Q == n].mean()) for n in (1, 2, 3, 5)}, 'delta': float(d.mean()), 'CI95': np.quantile(d[B].mean(1), [.025, .975]).round(4).tolist()}
            print(split, folds, f"w {w:.1f} bal {b.mean():.4f} d {d.mean():+.4f} {rep[split][f'w{w}']['CI95']} rank-uniform {(b * W).sum():.4f} d {(d * W).sum():+.4f}", {a: round(x, 3) for a, x in rep[split][f'w{w}']['by_q'].items()}, flush=True)
    (RUN / v.tag / 'FUSE.json').write_text(json.dumps(rep, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('split', nargs='?'); ap.add_argument('fold', nargs='?', type=int, default=0); ap.add_argument('tag', nargs='?', default='t1')
    ap.add_argument('--epochs', type=float, default=12); ap.add_argument('--L', type=int, default=160); ap.add_argument('--d', type=int, default=256); ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--bs', type=int, default=512); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--wd', type=float, default=.05); ap.add_argument('--dropout', type=float, default=.1)
    ap.add_argument('--scale', type=float, default=30.); ap.add_argument('--margin', type=float, default=.2); ap.add_argument('--cap', type=int, default=300); ap.add_argument('--min-games', type=int, default=8)
    ap.add_argument('--mask', type=float, default=0.); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--eval-every', type=int, default=1000); ap.add_argument('--bank', default='all'); v = ap.parse_args()
    if v.stage == 'data': data()
    elif v.stage == 'data2': data2()
    elif v.stage == 'safe2': safe2()
    elif v.stage == 'train': train(v)
    elif v.stage == 'oppemb': oppemb(v)
    elif v.stage == 'fuse': v.tag = v.split; fuse(v)
