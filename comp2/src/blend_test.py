"""Geometric-mean blend of test posteriors (fuse_sources.py *_x_D64_posteriors.npz: probabilities, question_ids) -> submission CSV with
expected-score decoding (same decoding as fuse_sources).  Prints the answers that differ from a reference CSV.
usage: blend_test.py OUT.csv REF.csv|- POST1.npz POST2.npz ...      (the final file: blend_final.py)
"""
import sys, os, csv, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from apply_scale import RANKS
E1 = np.exp(-1); K = np.array([[1 if i == j else (E1 if abs(i - j) == 1 else 0) for j in range(10)] for i in range(10)])
out, ref, posts = sys.argv[1], sys.argv[2], sys.argv[3:]
Z = [np.load(p, allow_pickle=True) for p in posts]; q = [str(x) for x in Z[0]["question_ids"]]
for z in Z[1:]: assert [str(x) for x in z["question_ids"]] == q
L = np.mean([np.log(np.clip(z["probabilities"].astype(np.float64), 1e-12, 1)) for z in Z], 0); P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
# DECODE_TAU (11-19, decode_temp.py): the expected-score decoding uses P^tau (tau < 1 = flatter); the saved posterior stays P
TAU = float(os.environ.get("DECODE_TAU", "1")); PD = P ** TAU; PD /= PD.sum(1, keepdims=True)
a = (PD @ K).argmax(1); pred = [RANKS[i] for i in a]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["question_id", "pred_rank"]); w.writerows(zip(q, pred))
np.savez(out.replace(".csv", "_posteriors.npz"), probabilities=P.astype(np.float32), question_ids=np.array(q), decode_tau=np.array(TAU))
if ref == "-":      # (repo) no reference file
    print(f"saved {out} (decode tau {TAU})")
else:
    r = pd.read_csv(ref).set_index("question_id")["pred_rank"]; d = [(x, r[x], p) for x, p in zip(q, pred) if r[x] != p]
    print(f"saved {out} (decode tau {TAU}): {len(d)} answers differ from {os.path.basename(ref)} ({sum(x.startswith('pub') for x, _, _ in d)} Public)")
print("RANKS", RANKS)
