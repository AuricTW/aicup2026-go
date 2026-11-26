"""Teacher pass for self-training of the identity network: recognise the OPPONENT side of every train-split game
(games of --holdout-players are skipped, exactly as in the teacher's training), and save the top-1 account + probability.
Several teachers (comma separated run dirs with identical class sets) are averaged in probability.
usage: idnet_pseudo.py MOVES.npz RUN_DIR[,RUN_DIR2] OUT.npz [--holdout-players CSV]
"""
import sys, os, argparse, csv
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idnet import Net, load_trunk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--holdout-players", default=None)
    ap.add_argument("--splits", default="0")      # split codes whose games are labelled (test track with the whole gallery: 0,1,3,4 = every training game of that track)
    ap.add_argument("--trunk", default=None); ap.add_argument("--trunk-scale", type=float, default=0.5)      # for teachers trained with idnet.py --trunk; only embedded games are labelled
    ap.add_argument("--rank-prior", action="store_true")      # re-weight the accounts with the known rank of the game's owner (training games only)
    a = ap.parse_args(); dev = torch.device("cuda"); z = np.load(a.data, allow_pickle=True); sp = z["split"]; pid = z["pid"].astype(np.int64)
    idx = np.where(np.isin(sp, [int(s) for s in a.splits.split(",")]))[0]
    if a.holdout_players:
        names = {str(n): i for i, n in enumerate(z["pid_names"])}; hp = np.array(sorted({names[r["player_id"]] for r in csv.DictReader(open(a.holdout_players, newline="")) if r["player_id"] in names}))
        idx = idx[~np.isin(pid[idx], hp)]
    slot = TRK = None
    if a.trunk: slot, TRK = load_trunk(a.trunk, z["key"], 80, a.trunk_scale); idx = idx[slot[idx] >= 0]; print("embedded games to label", len(idx), flush=True)
    nets = []; owners = None
    for r in a.run.split(","):
        ck = torch.load(os.path.join(r, "model.pt"), map_location="cpu", weights_only=False); ar = ck["args"]
        if owners is None: owners = ck["owners"]
        assert np.array_equal(owners, ck["owners"])
        net = Net(len(owners), ar["d"], ar["layers"], 4, ar["L"], drop=ar["dropout"], cos=ar.get("cos", 0.0), rel=ar.get("rel", 0), board=ar.get("board", 0), bl=ar.get("board_layers", 4), bpatch=ar.get("board_patch", 0), trunk=TRK.shape[2] if ar.get("trunk") and not ar.get("trunk_mask_only") else 0).to(dev); net.load_state_dict(ck["model"]); net.eval(); nets.append((net, ar["L"]))
    CB = torch.tensor(z["color_black"].astype(bool)); FB = torch.tensor(z["first_black"].astype(bool)); HDV = torch.tensor(z["hd"].astype(np.int64)); LENa = z["LEN"].astype(np.int64)
    Ms = {L: torch.tensor(z["M"][:, :L].astype(np.int64)) for L in {l for _, l in nets}}
    top = np.zeros(len(idx), np.int64); prob = np.zeros(len(idx), np.float32)
    K = 20; TK = np.zeros((len(idx), K), np.int64) if a.rank_prior else None; PK = np.zeros((len(idx), K), np.float32) if a.rank_prior else None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), 2048):
            j = torch.tensor(idx[i:i + 2048]); p = 0
            for net, L in nets:
                tk = TRK[torch.as_tensor(slot[idx[i:i + 2048]])][:, :L].to(dev).float() if net.tproj is not None else None
                lo, _ = net(Ms[L][j].to(dev), torch.tensor(np.minimum(LENa[idx[i:i + 2048]], L)).to(dev), ~CB[j].to(dev), FB[j].to(dev), HDV[j].to(dev), tk)
                with torch.autocast("cuda", enabled=False): p = p + torch.softmax(lo.float(), 1) / len(nets)
            tp, tc = p.max(1); top[i:i + 2048] = tc.cpu().numpy(); prob[i:i + 2048] = tp.cpu().numpy()
            if a.rank_prior: v, c = p.topk(K, 1); TK[i:i + 2048] = c.cpu().numpy(); PK[i:i + 2048] = v.cpu().numpy()
            if (i // 2048) % 50 == 0: print(i, "/", len(idx), flush=True)
    print("plain: share > 0.3 / 0.5 / 0.7 / 0.9:", [round(float((prob > t).mean()), 3) for t in (0.3, 0.5, 0.7, 0.9)])
    if a.rank_prior:
        # The owner of a training game has a known rank and the server pairs equal ranks (one rank apart with one handicap stone):
        # P(account | moves, owner rank y) ~ P(account | moves) * P(y | rank of the account, handicap, colour) / P(y | handicap, colour).
        # The pairing table is counted from the confident plain recognitions (> 0.9); only the 20 best accounts of a game are re-weighted,
        # the remaining mass keeps factor 1.
        y = z["y"].astype(np.int64)[idx]; hd = np.minimum(z["hd"].astype(np.int64)[idx], 1); cb = z["color_black"].astype(np.int64)[idx]
        cls_rank = np.zeros(len(owners), np.int64); own_cls = -np.ones(pid.max() + 1, np.int64); own_cls[owners] = np.arange(len(owners))
        lab_rows = np.where(np.isin(pid, owners))[0]; cls_rank[own_cls[pid[lab_rows]]] = z["y"].astype(np.int64)[lab_rows]
        ctx = hd * 2 + cb; C = np.ones((4, 10, 10)); m = prob > 0.9; np.add.at(C, (ctx[m], cls_rank[top[m]], y[m]), 1.0)          # [context, account rank, owner rank]
        Py_r = C / C.sum(2, keepdims=True); Py = np.stack([np.bincount(y[ctx == c], minlength=10) + 1.0 for c in range(4)]); Py = Py / Py.sum(1, keepdims=True)
        LR = Py_r / Py[:, None, :]
        for c in range(4): print("context", c, "(hd, owner black) =", divmod(c, 2), " P(owner rank = account rank):", np.round(np.diag(Py_r[c]), 2), flush=True)
        f = LR[ctx[:, None], cls_rank[TK], y[:, None]]; w = PK * f; den = w.sum(1) + np.clip(1 - PK.sum(1), 0, 1); w = w / den[:, None]
        j = w.argmax(1); top2 = TK[np.arange(len(idx)), j]; prob2 = w[np.arange(len(idx)), j].astype(np.float32)
        print(f"rank prior: top-1 changed {float((top2 != top).mean()):.3f}   among plain > 0.5: {float((top2 != top)[prob > .5].mean()):.3f}   new > 0.5 that were <= 0.5: {float(((prob2 > .5) & (prob <= .5)).mean()):.3f}   dropped below: {float(((prob2 <= .5) & (prob > .5)).mean()):.3f}")
        top, prob = top2, prob2
    np.savez(a.out, idx=idx, pid=owners[top], prob=prob)
    print("saved", a.out, "games", len(idx), " share > 0.3 / 0.5 / 0.7 / 0.9:", [round(float((prob > t).mean()), 3) for t in (0.3, 0.5, 0.7, 0.9)])


if __name__ == "__main__":
    main()
