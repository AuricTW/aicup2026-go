"""Select game-list line numbers: blocks A/B (dev + pilot train players), first N games per player.
usage: make_select.py GAMELIST OUT.json [N=10] [BLOCKS=AB]
"""
import json, sys

n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
blocks = sys.argv[4] if len(sys.argv) > 4 else "AB"
cnt = {}; sel = []; per = {}
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        g = json.loads(line)
        b = g["block"]
        if b not in blocks:
            continue
        if b == "T":
            sel.append(i); per[b] = per.get(b, 0) + 1; continue
        c = cnt.get(g["player_id"], 0)
        if c < n:
            sel.append(i); cnt[g["player_id"]] = c + 1; per[b] = per.get(b, 0) + 1
json.dump(sel, open(sys.argv[2], "w"))
print(len(sel), per, "players", len(cnt))
