"""Per-move HumanSL trunk embeddings for EVERY position of a game (VM side, frozen original b18c384nbt-humanv0).

For each turn: KataGo v14 input features of the position before the move (mover's perspective, ladder planes zero as in
the 11/13 fine-tune pipeline), one neutral rank context, trunk-final activation -> [board mean, board max, vector at the
played point] (1152 dims) -> PCA projection (fitted once on the first games, saved and reused) -> float16.
Output: OUT/tchunk_NNNNN.npz (emb [sum T, D], offset) + .json meta (same keys as the game list).
usage: extract_trunk.py GAMELIST.jsonl OUT_DIR --pca PCA.npz [--fit-games 400] [--dim 64] [--select SEL.json] [--workers 12] [--chunk 500]
"""
import os, sys, json, time, argparse, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp2_paths import THIRD_PARTY, path as _cfg_path
sys.path.insert(0, str(THIRD_PARTY))      # KataGo python modules: ../third_party/katago
os.environ["KATAGO_FLEX_ATTENTION"] = "0"
MV = re.compile(r";([BW])\[([a-s]{2}|tt|)\]"); HDRE = re.compile(r"HD\[(\d+)\]")
_FEAT = None
MAXMOVES = int(os.environ.get("TRUNK_MAX_MOVES", "0"))      # only the first N plies of a game (identity gallery: 80)


def all_positions(sgf):
    """packed binary planes [T, 22*361/8], global [T,19], action [T], mover_is_white [T]"""
    global _FEAT
    from katago.game.board import Board
    from katago.game.gamestate import GameState
    from katago.game.features import Features
    if _FEAT is None:
        Features.iterLadders = lambda self, board, f: None
        _FEAT = Features({"version": 14}, 19)
    moves = [(c, None if p in ("", "tt") else (ord(p[0]) - 97, ord(p[1]) - 97)) for c, p in MV.findall(sgf)]
    if MAXMOVES: moves = moves[:MAXMOVES]
    m = HDRE.search(sgf); hd = int(m.group(1)) if m else 0
    rules = GameState.RULES_JAPANESE.copy(); rules["whiteKomi"] = 6.5 if hd == 0 else 0.5
    gs = GameState(19, rules)
    T = len(moves)
    if T == 0: return None
    gs.board.pla = Board.BLACK if moves[0][0] == "B" else Board.WHITE
    gs.boards = [gs.board.copy()]
    binary = np.zeros((T, 22 * 361), np.uint8); glob = np.zeros((T, 19), np.float32); act = np.zeros(T, np.int16); white = np.zeros(T, np.uint8)
    n = 0
    for turn, (c, pt) in enumerate(moves):
        pla = Board.BLACK if c == "B" else Board.WHITE
        if gs.board.pla != pla: gs.board.pla = pla
        loc = Board.PASS_LOC if pt is None else gs.board.loc(pt[0], pt[1])
        b, g = gs.get_input_features(_FEAT)
        binary[turn] = b[0].reshape(-1).astype(np.uint8); glob[turn] = g[0]
        act[turn] = 361 if pt is None else pt[1] * 19 + pt[0]; white[turn] = c == "W"; n = turn + 1
        if pt is not None and not gs.board.would_be_legal(pla, loc): break
        gs.play(pla, loc)
    return np.packbits(binary[:n], axis=1), glob[:n], act[:n], white[:n], T


def work(line):
    g = json.loads(line); r = all_positions(g.pop("sgf"))
    return g, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gamelist"); ap.add_argument("out"); ap.add_argument("--pca", required=True); ap.add_argument("--fit-games", type=int, default=400)
    ap.add_argument("--dim", type=int, default=64); ap.add_argument("--select", default=None); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=500); ap.add_argument("--bs", type=int, default=768); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    from multiprocessing import Pool
    import katago.train.load_model as _lm
    from katago.train.load_model import load_model
    _lm.load_checkpoint = lambda f, map_location="cpu": torch.load(f, map_location=map_location, weights_only=False)
    from katago.game.sgfmetadata import SGFMetadata
    from katago.game.board import Board
    CONTEXT = {"inverseBRank": 9, "inverseWRank": 9, "bIsUnranked": False, "wIsUnranked": False, "bRankIsUnknown": False, "wRankIsUnknown": False,
               "bIsHuman": True, "wIsHuman": True, "gameIsUnrated": False, "gameRatednessIsUnknown": False, "tcIsUnknown": False, "tcIsNone": False,
               "tcIsAbsolute": False, "tcIsSimple": False, "tcIsByoYomi": True, "tcIsCanadian": False, "tcIsFischer": False, "mainTimeSeconds": 600,
               "periodTimeSeconds": 30, "byoYomiPeriods": 5, "canadianMoves": 0, "gameDate": "2022-07-01", "source": 2}
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    model, _, _ = load_model(str(_cfg_path("humansl_ckpt")), False, dev, 19, False)
    model.eval(); tap = {}
    model.act_trunkfinal.register_forward_hook(lambda m, i, o: tap.__setitem__("z", o))
    meta = {w: torch.tensor(SGFMetadata.of_dict(dict(CONTEXT)).get_metadata_row(pla, 361), dtype=torch.float32, device=dev)
            for pla, w in [(Board.BLACK, 0), (Board.WHITE, 1)]}
    shifts = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)

    def embed(packed, glob, act, white):
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for lo in range(0, len(act), a.bs):
                p = torch.from_numpy(packed[lo:lo + a.bs]).to(dev)
                x = ((p.unsqueeze(-1) >> shifts) & 1).reshape(len(p), -1)[:, :22 * 361].reshape(-1, 22, 19, 19).float()
                g = torch.from_numpy(glob[lo:lo + a.bs]).to(dev); ac = torch.from_numpy(act[lo:lo + a.bs].astype(np.int64)).to(dev)
                w = torch.from_numpy(white[lo:lo + a.bs].astype(np.int64)).to(dev)
                m = torch.where(w[:, None] > 0, meta[1].expand(len(p), -1), meta[0].expand(len(p), -1))
                model(x, g, input_meta=m)
                z = tap.pop("z").float().flatten(2)
                loc = z[torch.arange(len(ac), device=dev), :, ac.clamp_max(360)] * (ac != 361).float()[:, None]
                out.append(torch.cat([z.mean(-1), z.amax(-1), loc], 1))
        return torch.cat(out)

    sel = set(json.load(open(a.select))) if a.select else None
    lines = []
    with open(a.gamelist) as f:
        for i, line in enumerate(f):
            if sel is not None and i not in sel: continue
            lines.append(line)
            if a.limit and len(lines) >= a.limit: break
    print("games", len(lines), flush=True)
    pool = Pool(a.workers); t0 = time.time()
    if os.path.exists(a.pca):
        z = np.load(a.pca); mu = torch.tensor(z["mean"], device=dev); comp = torch.tensor(z["components"], device=dev)
    else:
        rng = np.random.RandomState(0); pick = [lines[i] for i in rng.choice(len(lines), min(a.fit_games, len(lines)), replace=False)]
        feats = []
        for g, r in pool.imap(work, pick, chunksize=4):
            if r is None: continue
            e = embed(*r[:4]); feats.append(e[torch.randperm(len(e), device=dev)[:80]])
        Fm = torch.cat(feats); mu = Fm.mean(0); U, S, V = torch.pca_lowrank(Fm - mu, q=a.dim, center=False, niter=6)
        comp = (V / (S / (len(Fm) - 1) ** 0.5)).T.contiguous()           # whitened components [dim, 1152]
        np.savez(a.pca, mean=mu.cpu().numpy(), components=comp.cpu().numpy(), explained=(S ** 2 / ((Fm - mu) ** 2).sum()).cpu().numpy())
        print("pca fitted on", len(Fm), "positions; explained variance", float((S ** 2).sum() / ((Fm - mu) ** 2).sum()), "t", round(time.time() - t0), flush=True)
    done = 0; npos = 0; nchunk = 0
    for c0 in range(0, len(lines), a.chunk):
        name = f"{a.out}/tchunk_{nchunk:05d}"; nchunk += 1
        if os.path.exists(name + ".npz"): done += min(a.chunk, len(lines) - c0); continue
        embs, metas = [], []
        for g, r in pool.imap(work, lines[c0:c0 + a.chunk], chunksize=4):
            if r is None: continue
            e = ((embed(*r[:4]) - mu) @ comp.T).clamp(-30, 30).half().cpu().numpy()
            g.update(n_moves=int(r[4]), n_emb=len(e)); embs.append(e); metas.append(g)
        off = np.cumsum([0] + [len(e) for e in embs])
        np.savez(name + ".npz", emb=np.concatenate(embs), offset=off); json.dump(metas, open(name + ".json", "w"))
        done += len(metas); npos += int(off[-1]); el = time.time() - t0
        print(f"done {done}/{len(lines)} positions {npos} {npos / el:.0f} pos/s elapsed {el:.0f}s", flush=True)
    json.dump(dict(done=done, positions=npos, elapsed=time.time() - t0), open(a.out + "/DONE.json", "w"))
    print("DONE", done, npos, flush=True)


if __name__ == "__main__":
    main()
