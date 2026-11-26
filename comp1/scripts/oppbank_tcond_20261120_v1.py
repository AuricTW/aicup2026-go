"""Hidden games in the training bank, second look (2026-11-20): retrieval CONDITIONED on a known opponent.
Section 35.4 searched the opponent side of all 839,395 bank games for a candidate's other games and found the precision too low (14-20 %).
New facts: (a) no game is listed twice in the bank (0 twins among 839,395), although 37 % of the bank games have a recognised training
account as the opponent -> every game is listed under ONE of its two players, so a game "T vs A" listed under T is a genuine extra game
of A; (b) players meet the same opponents again (the rematch term).  So the bank games of the training accounts T that a player A is
seen to play in his gallery are a small closed set (some hundreds of games) with a high prior of holding more games of A.
Pseudo test players = training accounts never seen by the fold-0 sequence net (as in 35.4).  Ground truth for them = the full model
recognises the opponent side as that account (p > .8).  Real candidates: same statistics without ground truth + a matched control
(T-sets of other candidates / random accounts) to see whether the conditional excess exists for them as well.
Only anonymous game records inside the competition data are linked; no external data, no real identities.
usage: oppbank_tcond_20261120_v1.py"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
import seqid_core_20261115_v4 as sc
import listfuse_20261115_v1 as lf
ROOT = lf.p.ROOT; SQ = ROOT / 'runs/seqid_20261115_v1'; OUT = ROOT / 'runs/oppbank_20261120_v2'; OUT.mkdir(exist_ok=True); dev = 'cuda'; rng = np.random.default_rng(20261120)
Z = np.load(SQ / 'bundle_g1.npz'); M = torch.tensor(Z['M'].astype(np.int64)); LEN = torch.tensor(np.minimum(Z['LEN'].astype(np.int64), 160)); FB = torch.tensor(Z['FB']); BLK = torch.tensor(Z['BLK']); HD = torch.tensor(Z['HD'].astype(np.int64)); OWN = Z['OWN'].astype(np.int64); N = len(OWN)
OM = torch.tensor(Z['OFF_M'].astype(np.int64)); OL = torch.tensor(np.minimum(Z['OFF_LEN'].astype(np.int64), 160)); OF = torch.tensor(Z['OFF_FB']); OB = torch.tensor(Z['OFF_BLK']); OH = torch.tensor(Z['OFF_HD'].astype(np.int64))


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False); a = ck['args']; net = sc.Net(len(ck['classes']), a['d'], a['layers'], a['L'], a['dropout'], board=a['board'], bl=a['board_layers'], bpatch=a['board_patch']).to(dev).to(memory_format=torch.channels_last)
    net.load_state_dict(ck['model']); net.eval(); return net, np.asarray(ck['classes'])


def emb(net, m, ln, side, fb, hd, bs=1024):
    o = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(m), bs): o.append(torch.nn.functional.normalize(net.embed(m[i:i + bs].to(dev), ln[i:i + bs].to(dev), side[i:i + bs].to(dev), fb[i:i + bs].to(dev), hd[i:i + bs].to(dev)).float(), dim=1).half().cpu())
    return torch.cat(o)


def post(net, classes, E):      # top-1 posterior over the net's training classes
    W = torch.nn.functional.normalize(net.W.float(), dim=1); top = np.zeros(len(E), np.float32); arg = np.zeros(len(E), np.int64)
    with torch.no_grad():
        for i in range(0, len(E), 8192): P = torch.softmax(30 * E[i:i + 8192].to(dev).float() @ W.t(), 1); a, b = P.max(1); top[i:i + 8192] = a.cpu().numpy(); arg[i:i + 8192] = b.cpu().numpy()
    return top, classes[arg]


t0 = time.time(); f = OUT / 'full.npz'
if not f.exists():
    full, cf = load(SQ / 'b1_full/ffull/model.pt'); E = emb(full, M, LEN, ~BLK, FB, HD); p, o = post(full, cf, E); EOo = emb(full, OM, OL, ~OB, OF, OH); po, oo = post(full, cf, EOo)
    np.savez(f, top=p, owner=o, off_top=po, off_owner=oo, classes=cf); np.save(OUT / 'opp_full.npy', E.numpy()); np.save(OUT / 'off_own_full.npy', emb(full, OM, OL, OB, OF, OH).numpy()); del full, E; torch.cuda.empty_cache()
T = np.load(f); rec_p, rec_o, off_p, off_o, cf = T['top'], T['owner'], T['off_top'], T['off_owner'], T['classes']; print(f'[1] bank {N}: opponent recognised p>.5 {np.mean(rec_p > .5):.3f} p>.8 {np.mean(rec_p > .8):.3f} | official games: p>.5 {np.mean(off_p > .5):.3f} p>.8 {np.mean(off_p > .8):.3f}   {time.time() - t0:.0f}s', flush=True)
fold, cfo = load(SQ / 'b1/g1/f0/model.pt'); f = OUT / 'fold.npz'
if not f.exists():
    E = emb(fold, M, LEN, ~BLK, FB, HD); p, o = post(fold, cfo, E); np.save(OUT / 'opp_fold0.npy', E.numpy()); np.savez(f, top=p, owner=o); del E
F = np.load(f); fo_p, fo_o = F['top'], F['owner']; EO = torch.tensor(np.load(OUT / 'opp_fold0.npy')).to(dev); print(f'    fold model: bank opponent recognised as one of ITS accounts p>.8 {np.mean(fo_p > .8):.3f}   {time.time() - t0:.0f}s', flush=True)
order = np.argsort(OWN, kind='stable'); start = np.searchsorted(OWN[order], np.arange(OWN.max() + 2))
def games_of(accs): return np.concatenate([order[start[a]:start[a + 1]] for a in accs]) if len(accs) else np.zeros(0, np.int64)
keep = Z['KEEP_f0']; cnt_all = np.bincount(OWN, minlength=OWN.max() + 1); cnt_keep = np.bincount(OWN[keep], minlength=OWN.max() + 1); H = np.flatnonzero((cnt_keep == 0) & (cnt_all >= 20) & np.isin(np.arange(len(cnt_all)), cf)); conf = rec_p > .8; print('pseudo test players (held-out training accounts, >= 20 games):', len(H), flush=True)
known_other = fo_p > .8      # the fold model is sure that the opponent is one of its own accounts -> cannot be a pseudo test player (label-free filter, usable for real candidates with the full model)


def study(name, cents, tsets, truth=None, ctrl_sets=None):
    """cents [n, d] normalised centroids (fold model); tsets: per player the array of opponent accounts seen in the gallery"""
    C = torch.tensor(cents).to(dev).half(); rows = []; best = []; bestc = []; nS = []; cover = []; base = []; lists = []
    for j in range(len(cents)):
        S = games_of(tsets[j]); nS.append(len(S))
        if truth is not None: g = truth[j]; cover.append((np.isin(g, S).sum(), len(g))); base.append((np.isin(S, g).sum(), len(S)))
        if len(S) == 0: best.append(np.nan); lists.append((S, np.zeros(0)));
        else:
            with torch.no_grad(): s = (EO[torch.tensor(S).to(dev)] @ C[j]).float().cpu().numpy()
            best.append(float(s.max())); lists.append((S, s))
        if ctrl_sets is not None:
            Sc = games_of(ctrl_sets[j])
            with torch.no_grad(): bestc.append(float((EO[torch.tensor(Sc).to(dev)] @ C[j]).float().max()) if len(Sc) else np.nan)
    nS = np.array(nS); best = np.array(best); print(f'[{name}] players {len(cents)}  with a non-empty T-set {np.mean(nS > 0):.3f}  accounts in the T-set (mean) {np.mean([len(t) for t in tsets]):.2f}  games in the conditional set: mean {nS.mean():.0f} median {np.median(nS):.0f}', flush=True)
    if truth is not None:
        cv = np.array(cover); bs = np.array(base); print(f'    ground-truth hidden games {cv[:, 1].sum()} ({cv[:, 1].mean():.1f} per player); inside the conditional set {cv[:, 0].sum()} = {cv[:, 0].sum() / cv[:, 1].sum():.3f} of all; share of the conditional set that is hidden games of the player (prior) {bs[:, 0].sum() / bs[:, 1].sum():.4f}; whole bank prior {cv[:, 1].sum() / (len(cents) * N):.6f}', flush=True)
    q = lambda v: ' '.join(f'{x:.3f}' for x in np.nanquantile(v, [.1, .25, .5, .75, .9]))
    print(f'    best cosine inside the conditional set: {q(best)}' + (f'   control (another player\'s T-set): {q(np.array(bestc))}   share of players with conditional best > control best {np.nanmean(best > np.array(bestc)):.3f}' if ctrl_sets is not None else ''), flush=True)
    for thr in (.45, .5, .55, .6, .65, .7):
        n = 0; h = 0; n2 = 0; h2 = 0; pl = 0
        for j, (S, s) in enumerate(lists):
            m = s >= thr; n += m.sum(); pl += m.any(); m2 = m & ~known_other[S] if len(S) else m; n2 += m2.sum()
            if truth is not None: h += np.isin(S[m], truth[j]).sum(); h2 += np.isin(S[m2], truth[j]).sum()
        print(f'    cos >= {thr}: retrieved per player {n / len(cents):.2f} (players with any {pl / len(cents):.3f})' + (f'  precision {h / max(n, 1):.3f}  recall of all hidden games {h / cv[:, 1].sum():.3f}' if truth is not None else '') + f' | without known-other games: {n2 / len(cents):.2f}' + (f'  precision {h2 / max(n2, 1):.3f}' if truth is not None else ''), flush=True)
    return lists, best


# ---- pseudo test players
own_rows = [rng.choice(np.flatnonzero(OWN == a), 10, replace=False) for a in H]; gi = np.concatenate(own_rows); Eg = emb(fold, M[gi], LEN[gi], BLK[gi], FB[gi], HD[gi]).float().numpy().reshape(len(H), 10, -1); Ch = Eg.mean(1); Ch /= np.linalg.norm(Ch, axis=1, keepdims=True)
for thr_t in (.5, .8):
    ts = [np.unique(rec_o[g][(rec_p[g] > thr_t) & (rec_o[g] != a)]) for a, g in zip(H, own_rows)]; truth = [np.flatnonzero(conf & (rec_o == a)) for a in H]; perm = rng.permutation(len(H)); print(f'--- opponent of a gallery game counted when p > {thr_t}')
    Lh, bh = study('pseudo test players', Ch, ts, truth, [ts[k] for k in perm])
np.savez(OUT / 'pseudo.npz', H=H, own_rows=np.stack(own_rows))
# ---- real candidates (fold model centroids for parity with the pseudo players)
games, pools, keys, pos = lf.ss.official(); EOFF = emb(fold, OM, OL, OB, OF, OH).float().numpy()
for pool in pools:
    G = [np.array([pos[k] for k in pool.supports[c]]) for c in pool.candidate_ids]; Cc = np.stack([EOFF[g].mean(0) for g in G]); Cc /= np.linalg.norm(Cc, axis=1, keepdims=True)
    for thr_t in (.5, .8):
        ts = [np.unique(off_o[g][off_p[g] > thr_t]) for g in G]; perm = rng.permutation(len(G)); print(f'--- opponent of a gallery game counted when p > {thr_t}')
        study(pool.name, Cc, ts, None, [ts[k] for k in perm])
print('done', round(time.time() - t0), 's', flush=True)
