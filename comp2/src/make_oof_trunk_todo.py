"""oof_games.csv rows whose trunk embeddings are not yet in TRUNK_DIR -> vm_out/oof_trunk_todo.csv (input of make_gamelist_keys.py)
usage: make_oof_trunk_todo.py TRUNK_DIR[,DIR2]
"""
import sys, os, glob, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); from comp2_paths import WORK
have = set()
for d in sys.argv[1].split(","):
    for f in glob.glob(d + "/tchunk_*.json"): have |= {m["key"] for m in json.load(open(f))}
rows = list(csv.DictReader(open(os.path.join(WORK, "oof_games.csv"), newline="")))
todo = [r for r in rows if r["key"] not in have]
with open(os.path.join(WORK, "oof_trunk_todo.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(todo)
print("oof games", len(rows), "already have trunk", len(rows) - len(todo), "todo", len(todo))
