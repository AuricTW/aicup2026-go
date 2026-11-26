"""Select game-list line numbers by per-block caps, excluding already processed lines.
usage: make_select2.py GAMELIST OUT.json SPEC [EXCLUDE.json ...]
SPEC e.g. A:20,T:all,C:10  (first N games per player in that block letter; D block continues the per-player count of A/B/C)
"""
import json, sys

spec = {}
for part in sys.argv[3].split(","):
    b, n = part.split(":"); spec[b] = 10**9 if n == "all" else int(n)
excl = set()
for f in sys.argv[4:]:
    excl |= set(json.load(open(f)))
cnt = {}; sel = []; per = {}
with open(sys.argv[1]) as f:
    for i, line in enumerate(f):
        g = json.loads(line)
        b = g["block"]
        if b not in spec:
            continue
        if b == "T":
            ok = True
        else:
            c = cnt.get(g["player_id"], 0); cnt[g["player_id"]] = c + 1
            ok = c < spec[b]
        if ok and i not in excl:
            sel.append(i); per[b] = per.get(b, 0) + 1
json.dump(sel, open(sys.argv[2], "w"))
print(len(sel), per)
