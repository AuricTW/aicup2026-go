"""Build the ordered game list for kata_eval.py (VM side, stdlib only).

Lockbox players are never read. Only HD<=1 games (the official test contains only HD0/HD1).
Order (so that any prefix is usable):
  A dev players, first 20 selected games each
  B pilot train players (150 per rank), first 20 each
  T official test games (all)
  C remaining train players, first 20 each
  D every train/dev player, selected games 21..CAP
Per-player game selection is a fixed-seed shuffle of that player's HD<=1 games.
usage: make_gamelist.py OUT.jsonl [CAP=40]
"""
import csv, sys, os, json, re, random, hashlib

csv.field_size_limit(10**9)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from comp2_paths import get as _cfg
BASE = _cfg("gpu_data", ".")      # folder with raw/training, raw/tests, raw/split_index.csv (README)
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
HD = re.compile(r"HD\[(\d+)\]")


def col(c):
    c = c.strip().upper()
    return "B" if c.startswith("B") else "W"


def main():
    out = sys.argv[1]
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    split = {}
    with open(BASE + "/raw/split_index.csv") as f:
        for r in csv.DictReader(f):
            split[(r["source_file"], int(r["source_row"]))] = (r["split"], r["player_id"], int(r["rank_idx"]))
    players = {}  # pid -> (split, rank_idx, [game dict])
    nhd = {}
    for rk in RANKS:
        fn = f"train_{rk}.csv"
        with open(f"{BASE}/raw/training/{fn}", newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                sp, pid, ri = split[(fn, i)]
                assert pid == r["player_id"] and RANKS[ri] == r["rank"]
                if sp == "lockbox":
                    continue
                m = HD.search(r["sgf_content"]); hd = int(m.group(1)) if m else 0
                nhd[hd] = nhd.get(hd, 0) + 1
                if hd > 1:
                    continue
                players.setdefault(pid, (sp, ri, []))[2].append(
                    dict(key=f"{fn}#{i}", player_id=pid, rank_idx=ri, color=col(r["color"]), split=sp, sgf=r["sgf_content"]))
    print("HD histogram (train+dev):", sorted(nhd.items()))
    for pid, (sp, ri, gs) in players.items():
        random.Random(int(hashlib.md5(pid.encode()).hexdigest()[:8], 16)).shuffle(gs)
    dev = sorted(p for p, v in players.items() if v[0] == "dev")
    train = sorted((p for p, v in players.items() if v[0] == "train"), key=lambda p: hashlib.md5(("pilot" + p).encode()).hexdigest())
    cnt = {}
    pilot, rest = [], []
    for p in train:
        r = players[p][1]
        if cnt.get(r, 0) < 150:
            cnt[r] = cnt.get(r, 0) + 1; pilot.append(p)
        else:
            rest.append(p)
    n = 0
    stats = {}
    with open(out, "w") as fo:
        def emit(g, block):
            nonlocal n
            fo.write(json.dumps(dict(g, block=block)) + "\n"); n += 1
            stats[block] = stats.get(block, 0) + 1
        def rounds(pids, lo, hi, block):
            for k in range(lo, hi):
                for p in pids:
                    gs = players[p][2]
                    if k < len(gs):
                        emit(gs[k], block)
        rounds(dev, 0, 20, "A")
        rounds(pilot, 0, 20, "B")
        for sp in ["public", "private"]:
            with open(f"{BASE}/raw/tests/rank_prediction_test_{sp}.csv", newline="", encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    for i in range(1, int(r["num_games"]) + 1):
                        emit(dict(key=f"{r['question_id']}#{i}", question_id=r["question_id"], slot=i, color=col(r[f"color_{i}"]),
                                  split="test", sgf=r[f"sgf_{i}"]), "T")
        rounds(rest, 0, 20, "C")
        rounds(dev + pilot + rest, 20, cap, "D")
    print("games", n, stats, "players dev/pilot/rest", len(dev), len(pilot), len(rest))


if __name__ == "__main__":
    main()
