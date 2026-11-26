"""Moves-only dataset over ALL non-lockbox training games (no engine evaluation needed) + dev block-A games + test games.
Stores the first L moves as policy indices (y*19+x, 361 pass), target colour, handicap flag, rank, split, player index, key.
Lockbox players are never read.  Dev = the dev-player games already used everywhere else (keys from the KataGo chunk metas),
so per-game outputs can be fused with the other sources by key.
usage: build_moves_dataset.py OUT.npz [--L 200] [--max-hd 1]
"""
import sys, os, csv, re, json, glob, argparse
import numpy as np, pandas as pd

csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); from comp2_paths import ROOT, WORK
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
MOVE = re.compile(r";([BW])\[([a-s]{2}|tt|)\]"); HD = re.compile(r"HD\[(\d+)\]")


def enc(sgf, L):
    mv = MOVE.findall(sgf)
    first = mv[0][0] if mv else "B"
    a = np.full(L, 362, np.int16)
    for i, (c, p) in enumerate(mv[:L]):
        a[i] = 361 if p in ("", "tt") else (ord(p[1]) - 97) * 19 + (ord(p[0]) - 97)
    m = HD.search(sgf)
    return a, min(len(mv), L), len(mv), first, (int(m.group(1)) if m else 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--L", type=int, default=200); ap.add_argument("--max-hd", type=int, default=1)
    a = ap.parse_args()
    ix = pd.read_csv(os.path.join(ROOT, "data", "interim", "index.csv"), usecols=["source_file", "source_row", "player_id", "rank_idx", "split"])
    split = {(f, r): s for f, r, s in zip(ix.source_file, ix.source_row, ix.split)}
    devkeys = set()
    for f in glob.glob(os.path.join(WORK, "full", "chunk_*.json")):
        for m in json.load(open(f)):
            if m["split"] == "dev": devkeys.add(m["key"])
    print("dev keys", len(devkeys), flush=True)
    M, LEN, T, COL, FIRST, HDV, Y, SP, PID, KEY = [], [], [], [], [], [], [], [], [], []
    pidx = {}
    for ri, rk in enumerate(RANKS):
        fn = f"train_{rk}.csv"
        with open(os.path.join(ROOT, "data", "raw", "training_v1_1_20261104", "training", fn), newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                s = split[(fn, i)]
                if s == "lockbox": continue
                key = f"{fn}#{i}"
                if s == "dev" and key not in devkeys: continue
                m, n, t, first, hd = enc(r["sgf_content"], a.L)
                if hd > a.max_hd or t < 12: continue
                M.append(m); LEN.append(n); T.append(t); COL.append(r["color"].strip().upper().startswith("B")); FIRST.append(first == "B"); HDV.append(hd)
                Y.append(ri); SP.append(0 if s == "train" else 1); PID.append(pidx.setdefault(r["player_id"], len(pidx))); KEY.append(key)
        print(fn, len(M), flush=True)
    for sp in ["public", "private"]:
        with open(os.path.join(ROOT, "data", "raw", "official_rank_tests_20261104_v1", "tests", f"rank_prediction_test_{sp}.csv"), newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                for i in range(1, int(r["num_games"]) + 1):
                    m, n, t, first, hd = enc(r[f"sgf_{i}"], a.L)
                    M.append(m); LEN.append(n); T.append(t); COL.append(r[f"color_{i}"].strip().upper().startswith("B")); FIRST.append(first == "B"); HDV.append(hd)
                    Y.append(-1); SP.append(2); PID.append(-1); KEY.append(f"{r['question_id']}|{i}")
    np.savez(a.out, M=np.stack(M), LEN=np.array(LEN, np.int16), T=np.array(T, np.int16), color_black=np.array(COL), first_black=np.array(FIRST), hd=np.array(HDV, np.int8),
             y=np.array(Y, np.int8), split=np.array(SP, np.int8), pid=np.array(PID, np.int32), key=np.array(KEY), pid_names=np.array(sorted(pidx, key=pidx.get)))
    sp = np.array(SP); print("saved", a.out, "train/dev/test", (sp == 0).sum(), (sp == 1).sum(), (sp == 2).sum(), "first-move-black share", np.mean(FIRST))


if __name__ == "__main__":
    main()
