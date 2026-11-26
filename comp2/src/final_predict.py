"""Calibrate fine-tuned per-position outputs on dev players and write the 800-question CSV.

Sources per game: own-decision vector and (optionally) opponent-decision vector, each the mean
of per-position 10-class log-probs. Per G: logits = b_own*sum(own) + b_opp*sum(opp) + bias,
fitted by NLL on dev queries (all dev players; 2-fold player cross-fit for the reported score).
Optional log-linear fusion with an existing 800x10 posterior file (weight given on CLI).
Decision: argmax expected official score.
usage: final_predict.py RUN_LOCAL_DIR OUT_DIR [--no-opp] [--fuse NPZ --fuse-w W]
"""
import sys, os, json, argparse, csv
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_queries as E

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
from comp2_paths import ROOT


def fit(S_list, y, l2=1e-3):
    k = len(S_list)
    def nll(t):
        z = sum(t[i] * S_list[i] for i in range(k)) + t[k:]
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean() + l2 * (t[k:] ** 2).sum()
    return minimize(nll, np.r_[np.ones(k) * 0.5, np.zeros(10)], method="L-BFGS-B").x


def prob(t, S_list):
    k = len(S_list)
    z = sum(t[i] * S_list[i] for i in range(k)) + t[k:]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--no-opp", action="store_true")
    ap.add_argument("--n-per", type=int, default=8)
    ap.add_argument("--fuse", default=None)
    ap.add_argument("--fuse-w", type=float, default=0.0)
    ap.add_argument("--tag", default="FT")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dg = json.load(open(f"{a.run}/dev_games.json"))["games"]
    tg = json.load(open(f"{a.run}/tests_games.json"))["games"]
    srcs = ["own"] if a.no_opp else ["own", "opp"]
    dv = {"own": E.game_vectors(np.load(f"{a.run}/dev_logp.npy"), dg)}
    tv = {"own": E.game_vectors(np.load(f"{a.run}/tests_logp.npy"), tg)}
    if not a.no_opp:
        dgo = json.load(open(f"{a.run}/dev_opp_games.json"))["games"]
        tgo = json.load(open(f"{a.run}/tests_opp_games.json"))["games"]
        assert [g["source_row"] for g in dgo] == [g["source_row"] for g in dg]
        assert [(g["question_id"], g["slot"]) for g in tgo] == [(g["question_id"], g["slot"]) for g in tg]
        dv["opp"] = E.game_vectors(np.load(f"{a.run}/dev_opp_logp.npy"), dgo)
        tv["opp"] = E.game_vectors(np.load(f"{a.run}/tests_opp_logp.npy"), tgo)
    # dev queries
    players = {}
    for i, g in enumerate(dg):
        players.setdefault(g["player_id"], (g["rank_idx"], []))[1].append(i)
    pids = sorted(players)
    half = set(np.random.RandomState(123).choice(pids, len(pids) // 2, replace=False))
    report = {}
    thetas = {}
    for seed in range(3):
        qs = E.make_queries(players, a.n_per, seed)
        for G in E.GS:
            q = [x for x in qs if x[2] == G]
            S = [np.stack([dv[s][x[3]].sum(0) for x in q]) for s in srcs]
            y = np.array([x[1] for x in q])
            fold = np.array([x[0] in half for x in q])
            P = np.zeros((len(q), 10))
            for f in [True, False]:
                t = fit([s[fold != f] for s in S], y[fold != f])
                P[fold == f] = prob(t, [s[fold == f] for s in S])
            rw = E.reward(y, E.decode(P))
            report.setdefault(G, []).append(dict(mean=rw.mean(), macro=np.mean([rw[y == r].mean() for r in range(10)]),
                                                 per_rank=[float(rw[y == r].mean()) for r in range(10)]))
            if seed == 0:
                thetas[G] = fit(S, y)  # final: all dev players
    summ = {G: dict(rank_macro=float(np.mean([r["macro"] for r in report[G]])), mean=float(np.mean([r["mean"] for r in report[G]])),
                    per_rank=np.mean([r["per_rank"] for r in report[G]], 0).round(3).tolist()) for G in E.GS}
    summ["overall_rank_macro"] = float(np.mean([summ[G]["rank_macro"] for G in E.GS]))
    summ["overall_mean"] = float(np.mean([summ[G]["mean"] for G in E.GS]))
    summ["thetas"] = {G: thetas[G].tolist() for G in E.GS}
    print(json.dumps({k: (v if k != "thetas" else "...") for k, v in summ.items()}, indent=1))
    # tests
    order = []
    with open(os.path.join(ROOT, "data/raw/official_rank_tests_20261104_v1/tests/sample_submission_rank.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            order.append(r["question_id"])
    byq = {}
    for i, g in enumerate(tg):
        byq.setdefault(g["question_id"], []).append(i)
    P = np.zeros((len(order), 10))
    for j, qid in enumerate(order):
        idx = byq[qid]
        G = len(idx)
        assert G in E.GS, (qid, G)
        P[j] = prob(thetas[G], [tv[s][idx].sum(0)[None] for s in srcs])[0]
    base = P.copy()
    if a.fuse:
        z = np.load(a.fuse, allow_pickle=True)
        assert list(z["question_ids"]) == order
        # geometric (log-linear) pool: (1-w) * this model + w * existing posterior
        lp = (1 - a.fuse_w) * np.log(np.clip(P, 1e-12, 1)) + a.fuse_w * np.log(np.clip(z["probabilities"], 1e-12, 1))
        P = np.exp(lp - lp.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    pred = E.decode(P)
    name = f"go_rank_{a.tag}.csv"
    with open(os.path.join(a.out, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "pred_rank"])
        for qid, p in zip(order, pred):
            w.writerow([qid, RANKS[p]])
    np.savez(os.path.join(a.out, f"go_rank_{a.tag}_posteriors.npz"), probabilities=P, model_only=base, question_ids=np.array(order))
    json.dump(summ, open(os.path.join(a.out, f"go_rank_{a.tag}_DEV_REPORT.json"), "w"), indent=1)
    print("wrote", os.path.join(a.out, name), "pred distribution", np.bincount(pred, minlength=10).tolist())


if __name__ == "__main__":
    main()
