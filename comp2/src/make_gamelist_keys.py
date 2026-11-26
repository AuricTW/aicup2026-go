"""Build a game list (same schema as make_gamelist.py) for an explicit list of training-file rows.
usage: make_gamelist_keys.py GAMES.csv OUT.jsonl [BLOCK=O]     (GAMES.csv: key,source_file,source_row,player_id,rank_idx,color,split)
Lockbox rows are refused.
"""
import csv, sys, os, json

csv.field_size_limit(10**9)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from comp2_paths import get as _cfg
BASE = _cfg("gpu_data", ".")      # folder with raw/training (README)
want = {}
with open(sys.argv[1], newline="") as f:
    for r in csv.DictReader(f):
        assert r["split"] != "lockbox"
        want.setdefault(r["source_file"], {})[int(r["source_row"])] = r
block = sys.argv[3] if len(sys.argv) > 3 else "O"
n = 0
with open(sys.argv[2], "w") as out:
    for fn, rows in sorted(want.items()):
        with open(f"{BASE}/raw/training/{fn}", newline="", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                w = rows.get(i)
                if w is None: continue
                assert r["player_id"] == w["player_id"], (fn, i)
                col = "B" if r["color"].strip().upper().startswith("B") else "W"
                out.write(json.dumps(dict(key=w["key"], player_id=w["player_id"], rank_idx=int(w["rank_idx"]), color=col, split=w["split"], block=block,
                                          sgf=r["sgf_content"])) + "\n"); n += 1
print("wrote", n)
