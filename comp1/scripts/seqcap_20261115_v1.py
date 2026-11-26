"""Capture events for the seq-id bundles, so that the board planes of the board-CNN seq net show the REAL position.
seqid_core v2/v3 build the board from the move list only: a stone stays for ever once played (captures ignored), so every position
after a capture carries ghost stones.  Here a batched Go engine (GPU, all games of a chunk in parallel, one move per step; liberties by
iterated flood fill) replays the first L moves of every game and records which stones each move captures.
Output bundle = input bundle + CAPT [N, K] uint8 (move index of the capturing move, 255 = unused), CAPP [N, K] uint16 (captured point),
OFF_CAPT / OFF_CAPP for the official games.  K = number of events kept per game (events beyond K are dropped; share reported).
Rules: a move on an occupied point is ignored; after a move, opponent groups without liberties are removed; suicide is not removed;
handicap stones that are not in the move list are unknown (as before).
Stages: build <in.npz> <out.npz> [K=64]      check <out.npz> [games=300]   (independent replay with sgfmill)
"""
import sys, json, time
import numpy as np
import torch


def nb(x): return x.roll(1, 1) | x.roll(-1, 1) | x.roll(1, 2) | x.roll(-1, 2)


@torch.no_grad()
def events(M, FB, dev='cuda', chunk=60000):
    N, L = M.shape; G, T, P = [], [], []
    for s in range(0, N, chunk):
        m = torch.tensor(M[s:s + chunk].astype(np.int64), device=dev); B = len(m); fb = torch.tensor(FB[s:s + chunk], device=dev); ar = torch.arange(B, device=dev)
        board = torch.zeros(B, 21, 21, dtype=torch.int8, device=dev); board[:, 0, :] = 3; board[:, -1, :] = 3; board[:, :, 0] = 3; board[:, :, -1] = 3      # 0 empty, 1 black, 2 white, 3 wall
        for t in range(L):
            mv = m[:, t]; col = torch.where(fb == (t % 2 == 0), 1, 2).to(torch.int8); r = mv.clamp(max=360) // 19 + 1; c = mv.clamp(max=360) % 19 + 1; ok = (mv < 361) & (board[ar, r, c] == 0)
            if not ok.any(): continue
            board[ar[ok], r[ok], c[ok]] = col[ok]; opp = (board == (3 - col)[:, None, None]) & ok[:, None, None]; lib = opp & nb(board == 0)
            while True:
                new = opp & (lib | nb(lib))
                if torch.equal(new, lib): break
                lib = new
            cap = opp & ~lib
            if cap.any():
                i = cap.nonzero(); G.append((i[:, 0] + s).cpu().numpy()); T.append(np.full(len(i), t, np.uint8)); P.append(((i[:, 1] - 1) * 19 + i[:, 2] - 1).cpu().numpy()); board[cap] = 0
    return np.concatenate(G), np.concatenate(T), np.concatenate(P)


def pad(g, t, pnt, N, K):
    o = np.lexsort((t, g)); g, t, pnt = g[o], t[o], pnt[o]; first = np.searchsorted(g, np.arange(N)); rank = np.arange(len(g)) - first[g]; keep = rank < K
    CT = np.full((N, K), 255, np.uint8); CP = np.zeros((N, K), np.uint16); CT[g[keep], rank[keep]] = t[keep]; CP[g[keep], rank[keep]] = pnt[keep]; cnt = np.bincount(g, minlength=N)
    return CT, CP, {'events': int(len(g)), 'dropped_share': float(1 - keep.mean()) if len(g) else 0., 'games_with_capture': float((cnt > 0).mean()), 'mean_events': float(cnt.mean()), 'p99_events': float(np.quantile(cnt, .99)), 'max_events': int(cnt.max())}


def build(src, dst, K=64):
    t0 = time.time(); Z = dict(np.load(src)); rep = {}
    for pre in ('', 'OFF_'):
        g, t, pnt = events(Z[pre + 'M'], Z[pre + 'FB']); Z[pre + 'CAPT'], Z[pre + 'CAPP'], rep[pre or 'train'] = pad(g, t, pnt, len(Z[pre + 'M']), K); print(pre or 'train', rep[pre or 'train'], f'{time.time() - t0:.0f}s', flush=True)
    np.savez(dst, **Z); json.dump({'source': src, 'K': K, 'report': rep, 'seconds': time.time() - t0}, open(dst.replace('.npz', '_CAPTURES.json'), 'w'), indent=1); print('saved', dst, flush=True)


def check(dst, n=300):
    from sgfmill import boards
    Z = np.load(dst); M = Z['M']; FB = Z['FB']; CT = Z['CAPT']; CP = Z['CAPP']; rng = np.random.default_rng(0); bad = 0; tot = 0; K = CT.shape[1]
    for i in rng.choice(len(M), n, replace=False):
        b = boards.Board(19); ref = []
        for t, mv in enumerate(M[i]):
            if mv >= 361: continue
            r, c = divmod(int(mv), 19)
            if b.get(r, c) is not None: continue
            col = 'b' if FB[i] == (t % 2 == 0) else 'w'; before = {(x, y) for x in range(19) for y in range(19) if b.get(x, y) == ('w' if col == 'b' else 'b')}
            try: b.play(r, c, col)
            except Exception: continue
            after = {(x, y) for x in range(19) for y in range(19) if b.get(x, y) == ('w' if col == 'b' else 'b')}; ref += sorted((t, x * 19 + y) for x, y in before - after)
            if b.get(r, c) is None: break      # suicide removed by sgfmill: positions diverge from here, stop comparing this game
        mine = sorted((int(a), int(q)) for a, q in zip(CT[i], CP[i]) if a != 255); ref = sorted(ref)[:K] if len(ref) > K else sorted(ref); tot += 1
        if mine[:len(ref)] != ref[:len(mine)] or (len(mine) != len(ref) and len(ref) < K): bad += 1
    print('games compared', tot, 'mismatching', bad, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'build': build(sys.argv[2], sys.argv[3], *([int(sys.argv[4])] if len(sys.argv) > 4 else []))
    else: check(sys.argv[2], *([int(sys.argv[3])] if len(sys.argv) > 3 else []))
