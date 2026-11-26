"""Game list for trunk-embedding extraction of the identity gallery (VM side, stdlib only).
Every account of the training files (all splits; lockbox accounts are gallery classes only, user decision 2026-11-15), HD <= 1,
>= 12 moves.  Games already embedded (trunk/abt, trunk/oof) are skipped.  Order = round robin over the accounts: the k-th game
(fixed-seed shuffle per account) of every account before any (k+1)-th game, so every prefix of the list is a balanced gallery
and the extraction can be stopped / continued at any time.
usage: make_gamelist_id.py OUT.jsonl DONE_DIR[,DONE_DIR2] [MAXPER=400]
"""
import csv, sys, os, json, re, random, glob
csv.field_size_limit(10**9)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from comp2_paths import get as _cfg
BASE = _cfg("gpu_data", ".")      # folder with raw/training (README)
RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]; HD = re.compile(r"HD\[(\d+)\]"); MV = re.compile(r";[BW]\[")
out = sys.argv[1]; maxper = int(sys.argv[3]) if len(sys.argv) > 3 else 400
have = set()
for d in sys.argv[2].split(","):
    for f in glob.glob(d + "/tchunk_*.json"): have.update(m["key"] for m in json.load(open(f)))
print("already embedded", len(have), flush=True)
players = {}
for ri, rk in enumerate(RANKS):
    fn = f"train_{rk}.csv"
    with open(f"{BASE}/raw/training/{fn}", newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            s = r["sgf_content"]; m = HD.search(s); hd = int(m.group(1)) if m else 0
            if hd > 1 or len(MV.findall(s)) < 12: continue
            players.setdefault(r["player_id"], []).append((f"{fn}#{i}", ri, r["color"].strip().upper()[:1], s))
    print(fn, len(players), flush=True)
rows = []
for pid in sorted(players):
    g = players[pid]; random.Random(pid).shuffle(g); k0 = sum(1 for x in g if x[0] in have)        # games already embedded count towards the account
    k = k0
    for key, ri, c, s in g:
        if key in have: continue
        if k >= maxper: break
        rows.append((k, pid, key, ri, c, s)); k += 1
rows.sort(key=lambda r: (r[0], r[1]))
with open(out, "w") as f:
    for k, pid, key, ri, c, s in rows: f.write(json.dumps({"key": key, "player_id": pid, "rank_idx": ri, "color": c, "ord": k, "sgf": s}) + "\n")
import collections
cum = collections.Counter(r[0] for r in rows); tot = 0; print("games", len(rows))
for k in sorted(cum):
    tot += cum[k]
    if k in (9, 19, 39, 59, 79, 119, 199, 399): print(f"  up to {k + 1} games per account: {tot} games in the list")
