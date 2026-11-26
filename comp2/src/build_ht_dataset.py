"""Append per-move HumanSL trunk embeddings (extract_trunk.py, whitened PCA) to an H token dataset.
Token t gets the embedding of the position before move t (zeros where none).  Games without embeddings are dropped
(reported per split; dev/test must be complete unless --allow-missing).
usage: build_ht_dataset.py H_DATASET.npz TRUNK_DIR[,DIR2] OUT.npz [--scale 0.5] [--allow-missing]
"""
import sys, os, glob, json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("h"); ap.add_argument("trunk"); ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--allow-missing", action="store_true"); ap.add_argument("--keys", default=None, help="CSV with a 'key' column: keep only these games")
    a = ap.parse_args()
    z = np.load(a.h, allow_pickle=True); X = z["X"]; M = z["M"]; LEN = z["LEN"]; meta = [json.loads(m) for m in z["meta"]]
    pos = {m["key"]: i for i, m in enumerate(meta)}; L = X.shape[1]
    files = sorted(sum([glob.glob(d + "/tchunk_*.npz") for d in a.trunk.split(",")], []))
    D = np.load(files[0])["emb"].shape[1]
    O = np.zeros((len(meta), L, X.shape[2] + D), np.float16); O[:, :, :X.shape[2]] = X; have = np.zeros(len(meta), bool)
    for f in files:
        t = np.load(f); off = t["offset"]; emb = t["emb"]
        for j, m in enumerate(json.load(open(f[:-4] + ".json"))):
            i = pos.get(m["key"])
            if i is None: continue
            e = emb[off[j]:off[j + 1]][:LEN[i]].astype(np.float32) * a.scale
            O[i, :len(e), X.shape[2]:] = e; have[i] = True
    sp = np.array([m["split"] for m in meta])
    for s in sorted(set(sp)): print(s, "games", int((sp == s).sum()), "with trunk", int(have[sp == s].sum()))
    if not a.allow_missing: assert have[sp != "train"].all(), "dev/test games without trunk embeddings"
    if a.keys:
        import csv
        want = {r["key"] for r in csv.DictReader(open(a.keys, newline=""))}; have &= np.array([m["key"] in want for m in meta])
        print("restricted to", a.keys, "->", int(have.sum()), "games")
    k = np.where(have)[0]
    np.savez(a.out, X=O[k], M=M[k], LEN=LEN[k], meta=z["meta"][k], mine_col=int(z["mine_col"]) if "mine_col" in z else 1)
    print("saved", a.out, O[k].shape)


if __name__ == "__main__":
    main()
