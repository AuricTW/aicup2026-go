"""Extend the moves dataset with the games of the sealed LOCKBOX accounts (split code 4), for the FINAL identity gallery only.
User decision 2026-11-15 16:15: the lockbox accounts may join the opponent gallery of the final submission; they are never
scored and never used for validation or calibration (no dev / OOF question is built from them).
Existing rows, their order and the player index stay untouched (trained networks and pseudo-label files store row / player
indices); new rows are appended.
usage: build_moves_lockbox.py OLD.npz OUT.npz [--max-hd 1]
"""
import sys, os, csv, argparse
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); from comp2_paths import ROOT, WORK
sys.path.insert(0, HERE)
from build_moves_dataset import enc, RANKS
csv.field_size_limit(10**9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("old"); ap.add_argument("out"); ap.add_argument("--max-hd", type=int, default=1)
    a = ap.parse_args(); z = np.load(a.old, allow_pickle=True); L = z["M"].shape[1]; have = set(str(k) for k in z["key"]); names = [str(n) for n in z["pid_names"]]; pidx = {n: i for i, n in enumerate(names)}
    assert 4 not in set(np.unique(z["split"]).tolist()), "split code 4 already present"
    ix = pd.read_csv(os.path.join(ROOT, "data", "interim", "index.csv"), usecols=["source_file", "source_row", "split"]); split = {(f, r): s for f, r, s in zip(ix.source_file, ix.source_row, ix.split)}
    M, LEN, T, COL, FIRST, HDV, Y, PID, KEY = [], [], [], [], [], [], [], [], []
    for ri, rk in enumerate(RANKS):
        fn = f"train_{rk}.csv"
        with open(os.path.join(ROOT, "data", "raw", "training_v1_1_20261104", "training", fn), newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                key = f"{fn}#{i}"
                if split[(fn, i)] != "lockbox" or key in have: continue
                m, n, t, first, hd = enc(r["sgf_content"], L)
                if hd > a.max_hd or t < 12: continue
                assert r["player_id"] not in pidx or pidx[r["player_id"]] >= len(z["pid_names"]), "lockbox player already in the dataset"
                if r["player_id"] not in pidx: pidx[r["player_id"]] = len(names); names.append(r["player_id"])
                M.append(m); LEN.append(n); T.append(t); COL.append(r["color"].strip().upper().startswith("B")); FIRST.append(first == "B"); HDV.append(hd); Y.append(ri); PID.append(pidx[r["player_id"]]); KEY.append(key)
        print(fn, len(M), flush=True)
    n = len(M); cat = lambda k, v, dt: np.concatenate([z[k], np.array(v, dt)])
    np.savez(a.out, M=np.concatenate([z["M"], np.stack(M)]), LEN=cat("LEN", LEN, np.int16), T=cat("T", T, np.int16), color_black=cat("color_black", COL, bool), first_black=cat("first_black", FIRST, bool),
             hd=cat("hd", HDV, np.int8), y=cat("y", Y, np.int8), split=cat("split", [4] * n, np.int8), pid=cat("pid", PID, np.int32), key=np.concatenate([z["key"], np.array(KEY)]), pid_names=np.array(names))
    print("saved", a.out, "lockbox games", n, "players total", len(names), "(was", len(z["pid_names"]), ")")


if __name__ == "__main__":
    main()
