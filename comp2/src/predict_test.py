"""Dev-calibrated test prediction from per-game log-prob sources (sequence models, fine-tuned net).

For each G in {5,10,15,20}: logits = sum_s scale_s * (sum over the question's games of source vector) + bias,
fitted by NLL on dev-player queries (2-fold player cross-fit for the reported score; final fit on all dev players).
Decoding: argmax of expected official score. Optional class-total scaling (see apply_scale.py) written as a second CSV.
usage: predict_test.py OUT_DIR TAG --seq DIR [DIR ...] [--ft RUN_DIR] [--split] [--gs 5,10,15,20]
  --seq dirs hold dev_pred.npz / test_pred.npz from seq_model.py; several dirs with the same games are averaged per group
  (use NAME=DIR,DIR to average seeds into one source; separate sources by spaces)
"""
import sys, os, json, csv, argparse, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_queries as E
from final_predict import prob
from scipy.optimize import minimize
from apply_scale import scale_sets, decode

RANKS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
BALANCED = False


def fit(S_list, y, l2=1e-3):
    """per-source scale + class bias by (optionally class-balanced) NLL"""
    k = len(S_list)
    w = 1.0 / np.bincount(y, minlength=10)[y] if BALANCED else np.ones(len(y)); w = w / w.sum()
    def nll(t):
        z = sum(t[i] * S_list[i] for i in range(k)) + t[k:]; z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -(w * lp[np.arange(len(y)), y]).sum() + l2 * (t[k:] ** 2).sum()
    return minimize(nll, np.r_[np.ones(k) * 0.5, np.zeros(10)], method="L-BFGS-B").x
from comp2_paths import ROOT


def load_seq(dirs, which, split):
    """average log-probs over dirs; returns {name_suffix: {key: vec}} and metas"""
    acc = {}; metas = {}
    fields = ["zown", "zopp"] if split else ["logp"]
    for d in dirs:
        z = np.load(os.path.join(d, f"{which}_pred.npz"), allow_pickle=True)
        ms = [json.loads(m) for m in z["meta"]]
        for f in fields:
            for i, m in enumerate(ms):
                k = m["key"] if which == "dev" else (m["question_id"], m["slot"])
                acc.setdefault(f, {}).setdefault(k, []).append(z[f][i].astype(np.float64))
                metas[k] = m
    return {f: {k: np.mean(v, 0) for k, v in d.items()} for f, d in acc.items()}, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("tag")
    ap.add_argument("--seq", nargs="+", required=True); ap.add_argument("--ft", default=None)
    ap.add_argument("--split", action="store_true"); ap.add_argument("--gs", default="5,10,15,20")
    ap.add_argument("--balanced", action="store_true", help="class-balanced calibration (uniform rank prior)")
    ap.add_argument("--n-per", type=int, default=8); ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()
    E.GS = [int(g) for g in a.gs.split(",")]
    global BALANCED; BALANCED = a.balanced
    os.makedirs(a.out, exist_ok=True)
    dsrc, tsrc, dmeta, tmeta = {}, {}, {}, {}
    for spec in a.seq:
        name, dirs = spec.split("=", 1) if "=" in spec else (os.path.basename(spec.rstrip("/\\")), spec)
        dirs = dirs.split(",")
        dv, dm = load_seq(dirs, "dev", a.split); dmeta.update(dm)
        for f, v in dv.items(): dsrc[f"{name}.{f}"] = v
        if not a.no_test:
            tv, tm = load_seq(dirs, "test", a.split); tmeta.update(tm)
            for f, v in tv.items(): tsrc[f"{name}.{f}"] = v
    if a.ft:
        for nm, lab in [("dev", "FTown"), ("dev_opp", "FTopp")]:
            g = json.load(open(f"{a.ft}/{nm}_games.json"))["games"]; v = E.game_vectors(np.load(f"{a.ft}/{nm}_logp.npy"), g)
            dsrc[lab] = {f"{gg['source_file']}#{gg['source_row']}": v[i] for i, gg in enumerate(g)}
        if not a.no_test:
            for nm, lab in [("tests", "FTown"), ("tests_opp", "FTopp")]:
                g = json.load(open(f"{a.ft}/{nm}_games.json"))["games"]; v = E.game_vectors(np.load(f"{a.ft}/{nm}_logp.npy"), g)
                tsrc[lab] = {(gg["question_id"], gg["slot"]): v[i] for i, gg in enumerate(g)}
    names = sorted(dsrc)
    keys = sorted(k for k in set.intersection(*[set(dsrc[n]) for n in names]) if k in dmeta)
    V = [np.stack([dsrc[n][k] for k in keys]) for n in names]
    players = {}
    for i, k in enumerate(keys):
        players.setdefault(dmeta[k]["player_id"], (dmeta[k]["rank_idx"], []))[1].append(i)
    pids = sorted(players); half = set(np.random.RandomState(123).choice(pids, len(pids) // 2, replace=False))
    print("sources", names, "dev games", len(keys), "dev players", len(players))
    report = {}; thetas = {}
    for seed in range(3):
        qs = E.make_queries(players, a.n_per, seed)
        for G in E.GS:
            q = [x for x in qs if x[2] == G]
            S = [np.stack([v[x[3]].sum(0) for x in q]) for v in V]; y = np.array([x[1] for x in q]); fold = np.array([x[0] in half for x in q])
            P = np.zeros((len(q), 10))
            for f in [True, False]:
                t = fit([s[fold != f] for s in S], y[fold != f]); P[fold == f] = prob(t, [s[fold == f] for s in S])
            rw = E.reward(y, E.decode(P))
            report.setdefault(G, []).append(dict(macro=float(np.mean([rw[y == r].mean() for r in range(10)])), mean=float(rw.mean()),
                                                 exact=float((E.decode(P) == y).mean()), per_rank=[float(rw[y == r].mean()) for r in range(10)]))
            if seed == 0: thetas[G] = fit(S, y)
    summ = {G: dict(rank_macro=float(np.mean([r["macro"] for r in report[G]])), mean=float(np.mean([r["mean"] for r in report[G]])),
                    exact=float(np.mean([r["exact"] for r in report[G]])), per_rank=np.mean([r["per_rank"] for r in report[G]], 0).round(3).tolist(),
                    scales=dict(zip(names, np.round(thetas[G][:len(names)], 4).tolist()))) for G in E.GS}
    summ["overall_rank_macro"] = float(np.mean([summ[G]["rank_macro"] for G in E.GS])); summ["sources"] = names; summ["balanced_calibration"] = bool(a.balanced)
    for G in E.GS:
        print(f"G{G}: rank-macro {summ[G]['rank_macro']:.4f} exact {summ[G]['exact']:.3f} scales {summ[G]['scales']}")
    print("overall rank-macro", round(summ["overall_rank_macro"], 4))
    json.dump(summ, open(os.path.join(a.out, f"go_rank_{a.tag}_DEV_REPORT.json"), "w"), indent=1)
    if a.no_test: return
    order = [r["question_id"] for r in csv.DictReader(open(os.path.join(ROOT, "data/raw/official_rank_tests_20261104_v1/tests/sample_submission_rank.csv"), encoding="utf-8-sig"))]
    byq = {}
    for k in tmeta: byq.setdefault(k[0], []).append(k)
    P = np.zeros((len(order), 10))
    for j, qid in enumerate(order):
        ks = byq[qid]; G = len(ks); assert G in thetas, (qid, G)
        P[j] = prob(thetas[G], [np.sum([tsrc[n][k] for k in ks], 0)[None] for n in names])[0]
    Q, info = scale_sets(P, order)
    for suffix, PP in [("", P), ("_scale40", Q)]:
        pred = decode(PP); path = os.path.join(a.out, f"go_rank_{a.tag}{suffix}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["question_id", "pred_rank"])
            for qid, p in zip(order, pred): w.writerow([qid, RANKS[p]])
        np.savez(os.path.join(a.out, f"go_rank_{a.tag}{suffix}_posteriors.npz"), probabilities=PP, question_ids=np.array(order))
        print("wrote", path, "counts", np.bincount(pred, minlength=10).tolist(), "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest().upper()[:16])
    print("scale multipliers", info, "answers changed by scaling", int((decode(P) != decode(Q)).sum()))


if __name__ == "__main__":
    main()
