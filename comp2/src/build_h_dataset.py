"""Token dataset from HumanSL profile-likelihood chunks only (no strong-net features needed).
Moves are re-parsed from the local raw CSVs (train/dev: key 'train_X.csv#row'; test: question_id + slot).
Token features: 0 t/200, 1 target-player flag, 2 mover is black, 3 line/9, 4 lag1, 5 lag2, 6 handicap flag,
  then per profile logp/5 (P), log1p(policy rank)/3 (P), adjacent-profile logp contrasts (P-1),
  mover-perspective winrate (mean over profiles, centred) and its change after the move (2).
usage: build_h_dataset.py OUT.npz H_DIR[,H_DIR2...] [--L 320] [--splits train,dev,test] [--max-per-player N]
"""
import sys, os, glob, json, argparse, csv, re
import numpy as np

csv.field_size_limit(10**9)
HERE = os.path.dirname(os.path.abspath(__file__))
from comp2_paths import ROOT, WORK
MOVE = re.compile(r";([BW])\[([a-s]{2}|tt|)\]")
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]


def parse_idx(sgf):
    out = []
    for c, p in MOVE.findall(sgf):
        out.append(361 if p in ("", "tt") else (ord(p[1]) - 97) * 19 + (ord(p[0]) - 97))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("h"); ap.add_argument("--L", type=int, default=320)
    ap.add_argument("--splits", default="train,dev,test"); ap.add_argument("--max-per-player", type=int, default=10**9)
    a = ap.parse_args()
    want = set(a.splits.split(","))
    files = sorted(sum([glob.glob(d + "/hchunk_*.npz") for d in a.h.split(",")], []))
    metas_all = [json.load(open(f[:-4] + ".json")) for f in files]
    need = {}
    for ms in metas_all:
        for m in ms:
            if m["split"] in want and m["split"] != "test":
                fn, row = m["key"].split("#"); need.setdefault(fn, set()).add(int(row))
    sgf = {}
    for fn, rows in need.items():
        with open(os.path.join(ROOT, "data", "raw", "training_v1_1_20261104", "training", fn), newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                if i in rows: sgf[f"{fn}#{i}"] = r["sgf_content"]
    if "test" in want:
        for sp in ["public", "private"]:
            with open(os.path.join(ROOT, "data", "raw", "official_rank_tests_20261104_v1", "tests", f"rank_prediction_test_{sp}.csv"), newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    for i in range(1, int(r["num_games"]) + 1): sgf[(r["question_id"], i)] = r[f"sgf_{i}"]
    print("sgf loaded", len(sgf), flush=True)
    X, M, LEN, metas, seen, cnt = [], [], [], [], set(), {}
    bad = 0
    for f, ms in zip(files, metas_all):
        z = np.load(f); off = z["offset"]; lpA = z["logp"]; rkA = z["rank"]; wrA = z["wr"]; P = lpA.shape[1]
        for i, m in enumerate(ms):
            if m["split"] not in want or m["key"] in seen: continue
            k = (m["question_id"], m["slot"]) if m["split"] == "test" else m["key"]
            if m["split"] != "test":
                c = cnt.get(m["player_id"], 0)
                if c >= a.max_per_player: continue
            mv = np.array(parse_idx(sgf[k]), np.int64)
            lp = lpA[off[i]:off[i + 1]].astype(np.float32); T = len(lp)
            if len(mv) != T: bad += 1; continue
            if T < 12: continue
            rk = rkA[off[i]:off[i + 1]].astype(np.float32); wr = wrA[off[i]:off[i + 1]].astype(np.float32).mean(1)
            mover_black = (np.arange(T) % 2 == 0) == (m["first_player"] == "B")
            mine = mover_black == (m["color"] == "B")
            is_pass = mv >= 361
            x = np.where(is_pass, 9, mv % 19); y = np.where(is_pass, 9, mv // 19)
            line = np.minimum(np.minimum(x, 18 - x), np.minimum(y, 18 - y))
            def lag(v, kk):
                o = np.zeros(T); o[kk:] = np.abs(v[kk:] - v[:-kk]); return o
            wm = np.where(mover_black, wr, 1 - wr)                     # mover-perspective winrate before the move
            nxt = np.r_[wr[1:], wr[-1]]; wn = np.where(mover_black, nxt, 1 - nxt)  # same player's winrate after the move
            feat = np.concatenate([np.stack([np.arange(T) / 200.0, mine.astype(float), mover_black.astype(float), line / 9.0,
                                             np.minimum(np.maximum(lag(x, 1), lag(y, 1)), 10) / 10, np.minimum(np.maximum(lag(x, 2), lag(y, 2)), 10) / 10,
                                             np.full(T, float(m.get("hd", 0)))], 1),
                                   lp / 5, np.log1p(rk) / 3, lp[:, 1:] - lp[:, :-1], np.stack([(wm - 0.5) * 2, np.clip(wn - wm, -1, 1) * 3], 1)], 1)
            n = min(T, a.L)
            xx = np.zeros((a.L, feat.shape[1]), np.float16); xx[:n] = feat[:n]
            mm = np.full(a.L, 362, np.int16); mm[:n] = mv[:n]
            X.append(xx); M.append(mm); LEN.append(n); metas.append(m); seen.add(m["key"])
            if m["split"] != "test": cnt[m["player_id"]] = cnt.get(m["player_id"], 0) + 1
    X = np.stack(X)
    np.savez(a.out, X=X, M=np.stack(M), LEN=np.array(LEN), meta=np.array([json.dumps(m) for m in metas]), mine_col=1)
    sp = [m["split"] for m in metas]
    print("saved", a.out, X.shape, {s: sp.count(s) for s in set(sp)}, "length mismatches skipped", bad)


if __name__ == "__main__":
    main()
