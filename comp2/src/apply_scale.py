"""Class-total scaling of test posteriors (transductive prior correction), then expected-score decoding.
Within each official set (pub / priv, 400 questions) rescale P[i,c] *= v_c so that every rank's posterior mass is 40.
Dev simulation (sim_balanced.py): +0.011 if each set holds exactly 40 players per rank, about -0.000 if labels are only
multinomially balanced.  No labels are edited by hand; the output is a deterministic function of the input posteriors.
usage: apply_scale.py IN_posteriors.npz OUT_DIR TAG
"""
import sys, os, csv, json, hashlib
import numpy as np

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
EM1 = np.exp(-1)


def decode(P):
    U = P.copy(); U[:, 1:] += EM1 * P[:, :-1]; U[:, :-1] += EM1 * P[:, 1:]
    return U.argmax(1)


def scale(P, cap, it=500):
    v = np.ones(P.shape[1])
    for _ in range(it):
        Q = P * v; Q /= Q.sum(1, keepdims=True)
        v *= cap / np.maximum(Q.sum(0), 1e-9)
    Q = P * v
    return Q / Q.sum(1, keepdims=True), v


def scale_sets(P, qid):
    out = P.copy(); info = {}
    for s in ("pub", "priv"):
        m = np.array([str(q).startswith(s) for q in qid])
        out[m], v = scale(P[m], np.full(10, m.sum() / 10.0))
        info[s] = (v / v.mean()).round(3).tolist()
    return out, info


def main():
    z = np.load(sys.argv[1], allow_pickle=True); out_dir, tag = sys.argv[2], sys.argv[3]
    P = z["probabilities"].astype(np.float64); qid = [str(q) for q in z["question_ids"]]
    Q, info = scale_sets(P, qid)
    a, b = decode(P), decode(Q)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"go_rank_{tag}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["question_id", "pred_rank"])
        for q, p in zip(qid, b): w.writerow([q, RANKS[p]])
    np.savez(os.path.join(out_dir, f"go_rank_{tag}_posteriors.npz"), probabilities=Q, unscaled=P, question_ids=np.array(qid))
    rep = dict(source=os.path.abspath(sys.argv[1]), changed=int((a != b).sum()), changed_by_more_than_one=int((np.abs(a - b) > 1).sum()),
               class_multipliers=info, counts_before=np.bincount(a, minlength=10).tolist(), counts_after=np.bincount(b, minlength=10).tolist(),
               sha256=hashlib.sha256(open(path, "rb").read()).hexdigest().upper())
    json.dump(rep, open(os.path.join(out_dir, f"go_rank_{tag}_SCALE_REPORT.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
