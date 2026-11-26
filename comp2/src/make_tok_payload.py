"""Game lists for per-decision HumanSL token replay of the OPPONENT side of comp2 question games (frozen comp1 token worker;
target colour = the opponent of the question's player).  Runs under the comp1 KataGo venv (needs goid.sgf_features from a
payload's code dir for the canonical hash, exactly as in the training payloads).
Blocks: test (all 10,000 games), dev (split 1), oof (games of the clean same-question OOF queries).
usage: make_tok_payload.py MOVES3.npz COMP2_ROOT PAYLOAD_CODE_DIR OUT_DIR
writes OUT_DIR/{testdev,oof}.jsonl (worker records) and OUT_DIR/{testdev,oof}_keys.json (comp2 keys in the same order)
"""
import sys, os, json, csv, hashlib, numpy as np
csv.field_size_limit(10**9)
mv, root, code, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True); SIDE = sys.argv[5] if len(sys.argv) > 5 else "opp"     # opp | own
sys.path.insert(0, code + "/vendor"); sys.path.insert(0, code + "/code")
from goid.sgf_features import analyze_sgf
z = np.load(mv, allow_pickle=True); keys = z["key"].astype(str); sp = z["split"]; hd = z["hd"].astype(int); cb = z["color_black"].astype(bool)
pos = {k: i for i, k in enumerate(keys)}
# SGF texts
sgf = {}
for part in ("public", "private"):
    for r in csv.DictReader(open(f"{root}/data/raw/official_rank_tests_20261104_v1/tests/rank_prediction_test_{part}.csv", encoding="utf-8", newline="")):
        for g in range(1, int(r["num_games"]) + 1): sgf[f"{r['question_id']}|{g}"] = r[f"sgf_{g}"]
need_train = set(keys[sp == 1].tolist())
import pandas as pd
V = os.path.dirname(os.path.abspath(mv))
mp = pd.read_csv(os.path.join(V, "oof_dense_map.csv")); d2k = dict(zip(mp.dense, mp.key)); d2p = dict(zip(mp.dense, mp.player_id))
clean = set(pd.read_csv(os.path.join(V, "oof_games.csv")).player_id)
q = json.load(open(f"{root}/runs/fullsystem_oof_20261106_v1/input/queries_dense.json"))
oof = sorted(set(str(d2k[r]).replace("#", "#") for x in q if d2p[x["row_indices"][0]] in clean for r in x["row_indices"]))
oof = [k for k in oof if k in pos]; need_train |= set(oof)
for f in sorted(set(k.split("#")[0] for k in need_train)):
    rows = {int(k.split("#")[1]) for k in need_train if k.startswith(f + "#")}
    for j, r in enumerate(csv.DictReader(open(f"{root}/data/raw/training/{f}", encoding="utf-8", newline=""))):
        if j in rows: sgf[f"{f}#{j}"] = r["sgf_content"]
    print(f, len(rows), flush=True)
blocks = {"testdev": [k for k in keys[sp == 2]] + [k for k in keys[sp == 1]], "oof": oof}
base = 0
blocks = {(n if SIDE == "opp" else f"{n}_own"): ks for n, ks in blocks.items()}
for name, ks in blocks.items():
    with open(f"{out}/{name}.jsonl", "w", encoding="utf-8", newline="\n") as fo:
        for n, k in enumerate(ks):
            t = sgf[k]; i = pos[k]; tgt = ("W" if cb[i] else "B") if SIDE == "opp" else ("B" if cb[i] else "W"); assert hd[i] in (0, 1)
            p = analyze_sgf(t, tgt, include_features=False)
            fo.write(json.dumps({"index": base + n, "raw": hashlib.sha256(t.encode()).hexdigest(), "color": tgt, "hd": int(hd[i]), "canonical": p["canonical_hash"], "sgf": t}) + "\n")
    json.dump(ks, open(f"{out}/{name}_keys.json", "w")); print(name, len(ks), "games", flush=True); base += len(ks)
