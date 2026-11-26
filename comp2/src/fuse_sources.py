"""Final assembly: per-game sources -> calibrated question posterior (stage 1) -> log-linear fusion with D64 (stage 2).
Per-game sources (10 values each, summed over the games of a question):
  zopp, zown : H+trunk token model, opponent / own head              (PRED dirs; averaged over seeds)
  id         : identity network, rank mass of the recognised opponent (idnet.py; label-drift kernel, HD1 shift)
  nb         : fingerprint naive-Bayes identity evidence (idfp.py, oppAB)
Stage 1: per G, scale per source + class bias fitted on dev players (natural prior).
Stage 2: P = P_new^a * P_D64^b; (a, b) per G fitted on the identical out-of-fold questions of the 949 players that no source
         has seen (identity network trained WITHOUT them: --id-x), 2-fold cross-fit for the report.
Test: stage-1 parameters refitted on dev with the full-gallery identity network (--id-full), exponents shrunk toward their mean.
usage: fuse_sources.py --id-x IDX.npz --id-full IDFULL.npz --out-dir DIR --tag TAG [--sources zopp,zown,id,nb] [--shrink 0.5] [--no-test]
"""
import sys, os, json, csv, argparse, hashlib
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from comp2_paths import ROOT, WORK, D64_OOF, D64_Q, D64_TEST      # machine-specific paths (configs/paths.yaml)
import evaluate_queries as E
import predict_test as PT
from final_predict import prob
from idnet_eval import smooth, shift
from apply_scale import RANKS
from scipy.optimize import minimize
PT.BALANCED = False
GS = [5, 10, 15, 20]; V = str(WORK)


def nk(k): return str(k).replace("|", "#")


def load_model(dirs, name):
    acc = {}; metas = {}
    for d in dirs:
        z = np.load(os.path.join(d, name), allow_pickle=True); zo, zw = z["zopp"], z["zown"]
        for i, m in enumerate(z["meta"]):
            m = json.loads(m); acc.setdefault(nk(m["key"]), []).append(np.stack([zo[i], zw[i]]).astype(np.float64)); metas[nk(m["key"])] = m
    return {k: np.mean(v, 0) for k, v in acc.items()}, metas


ID_MODE = "emp"


def load_id(path, block, k=1, eps=0.02):
    """ID_MODE 'old': log(eps + kernel-smoothed rank mass), HD1 shift.
    ID_MODE 'emp': confidence-weighted mixture with the empirical drift matrix of the network's own training games
                   ev_r = log((1 - g) + g * 10 * sum_r' K[r, r'] lik_r'),  g = top-1 probability  (see idnet_eval2.py).
    path may list several networks (comma separated): rank masses and top-1 probabilities are averaged."""
    zs = [np.load(p, allow_pickle=True) for p in path.split(",")]; z = zs[0]
    if f"{block}_keys" not in z.files: return {}
    for zz in zs[1:]: assert (zz[f"{block}_keys"] == z[f"{block}_keys"]).all()
    mass = np.mean([zz[f"{block}_opp_mass"].astype(np.float64) for zz in zs], 0); h = z[f"{block}_hd"]; b = z[f"{block}_black"]
    if ID_MODE == "old":
        m0 = smooth(mass, k)
        ev = np.where((h == 0)[:, None], np.log(eps + m0), np.where(b[:, None], np.log(eps + shift(m0, 1)), np.log(eps + shift(m0, -1))))
    else:
        from idnet_eval2 import emp_kernels
        tp = np.mean([zz[f"{block}_opp_top"].astype(np.float64) for zz in zs], 0) ** float(os.environ.get("ID_GAMMA", "1"))      # gate exponent (experiment)
        # ID_CAP c (11-20): gate = c * top-1.  Even a 'certain' recognition can be a look-alike of an account outside the gallery
        # (about 18 % of the test opponents), so the trust in any single recognition is capped at c; "c2" form "CAP:CAP2" = (id, id2)
        if os.environ.get("ID_CAP"):
            cc = [float(v) for v in os.environ["ID_CAP"].split(":")]; tp = tp * (cc[-1] if ("tokeval" in path.replace("\\", "/") and len(cc) > 1) else cc[0])
        prior = z["rank_prior"].astype(np.float64) if "rank_prior" in z.files else np.bincount(z["ho_rank"], minlength=10).astype(np.float64)
        prior /= prior.sum(); Ks = emp_kernels(z, float(os.environ.get("ID_EMP_THR", "0.5")))
        lik = mass / prior[None]; lik /= lik.sum(1, keepdims=True); kl = np.zeros_like(mass)
        for c, m in enumerate((h == 0, (h == 1) & b, (h == 1) & ~b)): kl[m] = lik[m] @ Ks[c].T
        ev = np.log((1 - tp)[:, None] + tp[:, None] * 10 * kl + 1e-4)
        if ID_MODE == "acc":            # account-specific opponent profiles (idens.py --acc) for even games; handicap games keep the label form
            ac = np.mean([zz[f"{block}_opp_acc"].astype(np.float64) for zz in zs], 0); ev = np.where((h == 0)[:, None] | ("acc_hd1" in z.files), np.log((1 - tp)[:, None] + tp[:, None] * ac + 1e-4), ev)
    ev = ev - ev.mean(1, keepdims=True)
    return {nk(key): ev[i] for i, key in enumerate(z[f"{block}_keys"])}


DEVDUMP = []; DEVS = {}; DEVS_CALL = []
ZG =float(os.environ.get("ZOPP_GATE", "0")); TOPG = {}    # experiment: zopp per game x (1 - ZG * top-1 of the id2 recognition)
# ID_RE rho (11-20): random-effect damping of the identity evidence.  The games of a player are not quite independent (between-player
# variance of opponent label - own label 0.04 against 0.42 within): the summed identity evidence of a question is divided by
# (1 + rho * n), n = soft count of its recognised games (sum of the top-1 probabilities of the id2 source).
RE = float(os.environ.get("ID_RE", "0")); NREC = {}
def re_fac(ks): return 1.0 / (1.0 + RE * sum(NREC.get(k, 0.0) for k in ks)) if RE else 1.0


def set_gate(path, blocks):
    """top-1 recognition probability per game (mean over the nets of an id2 evidence file), for the zopp gate"""
    TOPG.clear(); NREC.clear()
    if RE and path:      # ID_RE: top-1 recognition probability of every game (soft count of recognised games per question)
        zs = [np.load(p, allow_pickle=True) for p in path.split(",")]
        for b in blocks:
            if f"{b}_keys" not in zs[0].files: continue
            tp = np.mean([zz[f"{b}_opp_top"].astype(np.float64) for zz in zs], 0)
            for key, t in zip(zs[0][f"{b}_keys"], tp): NREC[nk(key)] = t
    if not ZG or not path: return
    zs = [np.load(p, allow_pickle=True) for p in path.split(",")]
    for b in blocks:
        if f"{b}_keys" not in zs[0].files: continue
        tp = np.mean([zz[f"{b}_opp_top"].astype(np.float64) for zz in zs], 0)
        for key, t in zip(zs[0][f"{b}_keys"], tp): TOPG[nk(key)] = t


def load_nb(paths):
    out = {}
    for p in paths:
        z = np.load(p, allow_pickle=True); ab = z["oppAB"]
        for i, key in enumerate(z["keys"]): out[nk(key)] = ab[i]
    return out


def macro(y, P):
    rw = E.reward(y, E.decode(P)); return float(np.mean([rw[y == r].mean() for r in range(10) if (y == r).any()]))


def fit2(LA, LB, y):
    def nll(t):
        z = t[0] * LA + t[1] * LB; z = z - z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True)); return -lp[np.arange(len(y)), y].mean()
    return minimize(nll, np.array([0.5, 0.5]), method="L-BFGS-B", bounds=[(0, 3), (0, 3)]).x


def pool(LA, LB, a, b):
    lp = a * LA + b * LB; P = np.exp(lp - lp.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id-x", required=True); ap.add_argument("--id-full", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--sources", default="zopp,zown,id,nb"); ap.add_argument("--shrink", type=float, default=0.5); ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--dev-dirs", default=f"{V}/ht1_s1,{V}/ht1_s2"); ap.add_argument("--oof-dirs", default=f"{V}/oofpredHT_s1,{V}/oofpredHT_s2"); ap.add_argument("--kernel", type=int, default=1)
    ap.add_argument("--id-mode", default="emp", choices=["emp", "old", "acc"]); ap.add_argument("--id-test", default=None); ap.add_argument("--id-test-scale", type=float, default=1.0)
    # question-level source "self" (idself.py): identity nets whose embedding retrieves the question's player among the opponent sides
    ap.add_argument("--self-x", default=None); ap.add_argument("--self-full", default=None); ap.add_argument("--self-test", default=None); ap.add_argument("--self-bias", type=float, default=0.0)
    ap.add_argument("--moves", default=f"{V}/moves_all2.npz"); ap.add_argument("--holdout-players", default=f"{V}/oof_games.csv")
    # direct rank networks (oppnet.py): sources dopp / down = per-game log-probabilities of the opponent-side / own-side head
    ap.add_argument("--dir-x", default=None); ap.add_argument("--dir-full", default=None); ap.add_argument("--dir-test", default=None)
    # second identity source "id2" (e.g. token-net evidence, tok_evidence.py), same file format and tracks as --id-*
    ap.add_argument("--id2-x", default=None); ap.add_argument("--id2-full", default=None); ap.add_argument("--id2-test", default=None)
    ap.add_argument("--id3-x", default=None); ap.add_argument("--id3-full", default=None); ap.add_argument("--id3-test", default=None)
    ap.add_argument("--id2-test-scale", type=float, default=1.0)
    a = ap.parse_args(); SRC = a.sources.split(","); global ID_MODE; ID_MODE = a.id_mode
    SELF = "self" in SRC; SRC = [s for s in SRC if s != "self"]; NS = SRC + (["self"] if SELF else [])      # SRC: per-game sources; NS: all stage-1 sources
    if SELF:
        from idself import SelfEvidence
        base = np.load(a.id_full.split(",")[0], allow_pickle=True)["acc_base"]

        def self_ev(track):
            run, splits, hold = dict(x=(a.self_x, (0,), a.holdout_players), full=(a.self_full, (0,), None), test=(a.self_test or a.self_full, (0, 1, 3), None))[track]
            return SelfEvidence(a.moves, run, splits, hold, bias=a.self_bias, base=base)
    nb = load_nb([f"{V}/idfp_evidence.npz", f"{V}/idfp_evidence_oof.npz"]) if "nb" in SRC else {}
    dv, dmeta = load_model(a.dev_dirs.split(","), "dev_pred.npz")

    def load_dir(path, block):
        out = {}
        for p in (path.split(",") if path else []):
            zz = np.load(p, allow_pickle=True)
            for side in ("opp", "own"):
                if f"{block}_{side}_logp" in zz.files:
                    lp = zz[f"{block}_{side}_logp"].astype(np.float64); lp = lp - lp.mean(1, keepdims=True)
                    for i, key in enumerate(zz[f"{block}_keys"]): out.setdefault(nk(key), {}).setdefault("d" + side, []).append(lp[i])
        return {k: {s: np.mean(v, 0) for s, v in d.items()} for k, d in out.items()}

    DIRS = [s for s in SRC if s in ("dopp", "down")]

    def vec(k, mdl, idd, dd=None, idd2=None, idd3=None):
        out = []
        for s in SRC:
            if s == "zopp": out.append(mdl[k][0] * (1 - ZG * TOPG.get(k, 0.0)) if ZG else mdl[k][0])
            elif s == "zown": out.append(mdl[k][1])
            elif s == "id": out.append(idd[k])
            elif s == "id2": out.append(idd2[k])
            elif s == "id3": out.append(idd3[k])
            elif s == "nb": out.append(nb[k])
            elif s in ("dopp", "down"): out.append(dd[k][s])
        return np.stack(out)

    def dev_fit(id_path, report, se=None, dir_path=None, id2_path=None, id3_path=None):
        idd = load_id(id_path, "dev", a.kernel) if "id" in SRC else {}; dd = load_dir(dir_path, "dev") if DIRS else {}
        idd2 = load_id(id2_path, "dev", a.kernel) if "id2" in SRC else {}; idd3 = load_id(id3_path, "dev", a.kernel) if "id3" in SRC else {}
        keys = sorted(k for k in dv if dmeta[k].get("hd", 0) <= 1 and (not idd or k in idd) and (not idd2 or k in idd2) and (not idd3 or k in idd3) and (not nb or k in nb) and (not DIRS or k in dd))
        X = np.stack([vec(k, dv, idd, dd, idd2, idd3) for k in keys]); players = {}
        for i, k in enumerate(keys): players.setdefault(dmeta[k]["player_id"], (dmeta[k]["rank_idx"], []))[1].append(i)
        if os.environ.get("DEV_MONO"):      # single-colour dev questions (experiment): the OOF and the official test questions are single-colour
            players = {f"{p}|{c}": (r, [i for i in gl if dmeta[keys[i]]["color"] == c]) for p, (r, gl) in players.items() for c in "BW"}
            players = {p: v for p, v in players.items() if v[1]}
        up = sorted(players); half = set(np.random.RandomState(7).choice(up, len(up) // 2, replace=False)); theta = {}; sc = {}; SD = {}
        for G in GS:
            E.GS = [G]; qs = E.make_queries(players, int(os.environ.get("DEV_NQ", "8")), int(os.environ.get("DEV_SEED", "0"))); y = np.array([q[1] for q in qs]); f = np.array([q[0] in half for q in qs])
            S = [np.stack([X[q[3], j].sum(0) for q in qs]) for j in range(len(SRC))]
            if RE: fq = np.array([re_fac([keys[i] for i in q[3]]) for q in qs]); S = [s_ * fq[:, None] if SRC[j] in ("id", "id2") else s_ for j, s_ in enumerate(S)]
            if SELF: S.append(se.evidence([[keys[i] for i in q[3]] for q in qs]))
            theta[G] = PT.fit(S, y); SD[G] = dict(S=np.stack(S), y=y)
            if os.environ.get("SAVE_DEVS"): DEVS.setdefault(len(DEVS_CALL), {})[G] = dict(S=np.stack(S), y=y, player=np.array([str(q[0]) for q in qs]), theta=np.asarray(theta[G]), keys=[[keys[i] for i in q[3]] for q in qs])      # stage-1 inputs (calibration-noise experiments, stage1_bag.py)
            if report:
                for name, js in (("all sources", list(range(len(NS)))), ("without self", list(range(len(SRC)))), ("model only", [j for j, s in enumerate(SRC) if s.startswith("z")])):
                    P = np.zeros((len(qs), 10))
                    for side in (True, False):
                        th = PT.fit([S[j][f != side] for j in js], y[f != side]); P[f == side] = prob(th, [S[j][f == side] for j in js])
                    sc.setdefault(name, []).append(macro(y, P))
                    if name == "all sources" and os.environ.get("SAVE_DEV"):      # cross-fitted dev posteriors per question with the games of each question (population experiments)
                        DEVDUMP.append(dict(G=G, P=P, y=y, player=[str(q[0]) for q in qs], keys=[[keys[i] for i in q[3]] for q in qs]))
        if os.environ.get("S1_RIDGE"):      # lower-variance stage 1 (stage1_lib.py): "lam_s,lam_b" = ridge toward the smooth-in-log-G fit; "smooth" = the smooth fit itself
            import stage1_lib as SL
            theta = SL.fit_smooth(SD) if os.environ["S1_RIDGE"] == "smooth" else SL.fit_ridge(SD, *[float(v) for v in os.environ["S1_RIDGE"].split(",")])
            if report: print("   stage 1 replaced by", os.environ["S1_RIDGE"], "(G10):", dict(zip(NS, np.round(theta[10][:len(NS)], 3).tolist())))
        if os.environ.get("SAVE_DEVS"):
            import pickle; DEVS_CALL.append(id_path); pickle.dump(dict(calls=DEVS_CALL, data=DEVS, sources=NS), open(os.environ["SAVE_DEVS"], "wb"))
        if report and os.environ.get("SAVE_DEV") and DEVDUMP:
            import pickle; pickle.dump(DEVDUMP, open(os.environ["SAVE_DEV"], "wb")); print("dev posteriors saved to", os.environ["SAVE_DEV"])
        if report:
            for name, v in sc.items(): print(f"dev cross-fit  {name:12s} " + "  ".join(f"G{G} {x:.4f}" for G, x in zip(GS, v)) + f"   mean {np.mean(v):.4f}")
            print("   stage-1 scales (G10):", dict(zip(NS, np.round(theta[10][:len(NS)], 3).tolist())))
        return theta, sc

    # ---------- validation on identical out-of-fold questions ----------
    se_x = self_ev("x") if SELF else None
    set_gate(a.id2_x, ("dev", "x"))
    theta_x, dev_sc = dev_fit(a.id_x, True, se_x, a.dir_x, a.id2_x, a.id3_x)
    ov, ometa = load_model(a.oof_dirs.split(","), "all_pred.npz")
    for k, v in dv.items(): ov.setdefault(k, v)
    idx = {}
    if "id" in SRC:
        idx = load_id(a.id_x, "x", a.kernel)
        for k, v in load_id(a.id_x, "dev", a.kernel).items(): idx.setdefault(k, v)
    idx2 = {}
    if "id2" in SRC:
        idx2 = load_id(a.id2_x, "x", a.kernel)
        for k, v in load_id(a.id2_x, "dev", a.kernel).items(): idx2.setdefault(k, v)
    idx3 = {}
    if "id3" in SRC:
        idx3 = load_id(a.id3_x, "x", a.kernel)
        for k, v in load_id(a.id3_x, "dev", a.kernel).items(): idx3.setdefault(k, v)
    ddx = {}
    if DIRS:
        ddx = load_dir(a.dir_x, "x")
        for k, v in load_dir(a.dir_x, "dev").items(): ddx.setdefault(k, v)
    mp = pd.read_csv(os.path.join(V, "oof_dense_map.csv")); d2k = dict(zip(mp.dense, mp.key)); d2p = dict(zip(mp.dense, mp.player_id))
    clean = set(pd.read_csv(os.path.join(V, "oof_games.csv")).player_id)
    q = json.load(open(D64_Q)); z = np.load(D64_OOF, allow_pickle=True); PDall = z["count"]
    rows = []; skipped = dict(not_clean=0, missing=0)
    for i, x in enumerate(q):
        pid = d2p[x["row_indices"][0]]
        if pid not in clean: skipped["not_clean"] += 1; continue
        ks = [d2k[r] for r in x["row_indices"]]
        if any(k not in ov or (idx and k not in idx) or (idx2 and k not in idx2) or (idx3 and k not in idx3) or (nb and k not in nb) or (DIRS and k not in ddx) for k in ks): skipped["missing"] += 1; continue
        S = np.sum([vec(k, ov, idx, ddx, idx2, idx3) for k in ks], 0); G = x["num_games"]
        if RE: S = np.stack([s_ * re_fac(ks) if SRC[j] in ("id", "id2") else s_ for j, s_ in enumerate(S)])
        rows.append([pid, x["rank_idx"], G, S, PDall[i], ks, i])
    if SELF:
        sv = se_x.evidence([r[5] for r in rows]); del se_x
        for r, s in zip(rows, sv): r[3] = np.concatenate([r[3], s[None]])
    if os.environ.get("SAVE_S"):      # raw per-source evidence sums of the OOF questions (stage-1 experiments on OOF, oof_stage1.py)
        np.savez(os.environ["SAVE_S"], S=np.stack([r[3] for r in rows]), y=np.array([r[1] for r in rows]), G=np.array([r[2] for r in rows]), pid=np.array([r[0] for r in rows]),
                 D64=np.stack([r[4] for r in rows]), question=np.array([r[6] for r in rows]), sources=np.array(NS), theta=np.stack([theta_x[r[2]] for r in rows]), keys=np.array(["|".join(r[5]) for r in rows]))
    for r in rows: r[3] = prob(theta_x[r[2]], [s[None] for s in r[3]])[0]
    pid = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows]); Gq = np.array([r[2] for r in rows])
    LN = np.log(np.clip(np.stack([r[3] for r in rows]), 1e-9, 1)); LD = np.log(np.clip(np.stack([r[4] for r in rows]), 1e-9, 1))
    print(f"\nOOF questions used {len(rows)} skipped {skipped} players {len(set(pid))}")
    up = sorted(set(pid)); half = set(np.random.RandomState(7).choice(up, len(up) // 2, replace=False)); fold = np.array([p in half for p in pid]); exps = {}; res = {}; PF = np.zeros((len(rows), 10))
    for G in GS:
        m = Gq == G; f = fold[m]; P2 = np.zeros((m.sum(), 10))
        for side in (True, False):
            t = fit2(LN[m][f != side], LD[m][f != side], y[m][f != side]); P2[f == side] = pool(LN[m][f == side], LD[m][f == side], *t)
        exps[G] = fit2(LN[m], LD[m], y[m]); PF[m] = P2
        res[G] = dict(n=int(m.sum()), D64=macro(y[m], np.exp(LD[m])), NEW=macro(y[m], np.exp(LN[m])), fixed_05_05=macro(y[m], pool(LN[m], LD[m], .5, .5)), fixed_07_04=macro(y[m], pool(LN[m], LD[m], .7, .4)),
                      fitted2=macro(y[m], P2), exponents=[round(float(v), 3) for v in exps[G]])
        print(f"G{G}: n {res[G]['n']}  D64 {res[G]['D64']:.4f}  NEW {res[G]['NEW']:.4f}  0.5/0.5 {res[G]['fixed_05_05']:.4f}  0.7/0.4 {res[G]['fixed_07_04']:.4f}  fitted (2-fold) {res[G]['fitted2']:.4f}  (a,b)={res[G]['exponents']}")
    for k in ("D64", "NEW", "fixed_05_05", "fixed_07_04", "fitted2"): print(f"mean over G  {k:12s} {np.mean([res[G][k] for G in GS]):.4f}")
    os.makedirs(a.out_dir, exist_ok=True)
    # out-of-fold posteriors per question (cross-fitted fusion), for evidence that is added afterwards (idfs_apply.py)
    np.savez(os.path.join(a.out_dir, f"go_rank_{a.tag}_OOF_posteriors.npz"), question=np.array([r[6] for r in rows]), player_id=pid, y=y, G=Gq, fold=fold, fused=PF, new=np.exp(LN), d64=np.exp(LD))
    rep = dict(sources=NS, dev=dev_sc, oof={str(G): v for G, v in res.items()}, oof_mean={k: float(np.mean([res[G][k] for G in GS])) for k in ("D64", "NEW", "fixed_05_05", "fitted2")}, skipped=skipped)
    if a.no_test: json.dump(rep, open(os.path.join(a.out_dir, f"go_rank_{a.tag}_VALIDATION.json"), "w"), indent=1); return
    # ---------- official test ----------
    se_f = self_ev("full") if SELF else None
    set_gate(a.id2_full, ("dev",))
    theta_f, dev_full = dev_fit(a.id_full, SELF, se_f, a.dir_full, a.id2_full, a.id3_full); del se_f
    if SELF: rep["dev_full"] = dev_full
    tv = {}
    for d in a.dev_dirs.split(","):
        zt = np.load(os.path.join(d, "test_pred.npz"), allow_pickle=True)
        for i, m in enumerate(zt["meta"]): tv.setdefault(nk(json.loads(m)["key"]), []).append(np.stack([zt["zopp"][i], zt["zown"][i]]).astype(np.float64))
    # --id-test: test evidence from a larger gallery (dev-split accounts added; cannot be calibrated on dev, where those accounts are the targets)
    tv = {k: np.mean(v, 0) for k, v in tv.items()}; idt = load_id(a.id_test or a.id_full, "test", a.kernel) if "id" in SRC else {}
    if a.id_test_scale != 1.0: idt = {k: v * a.id_test_scale for k, v in idt.items()}     # hedge: trust the identity evidence less (or more) on the test set
    ddt = load_dir(a.dir_test or a.dir_full, "test") if DIRS else {}
    idt2 = load_id(a.id2_test or a.id2_full, "test", a.kernel) if "id2" in SRC else {}
    if a.id2_test_scale != 1.0: idt2 = {k: v * a.id2_test_scale for k, v in idt2.items()}
    set_gate(a.id2_test or a.id2_full, ("test",))
    idt3 = load_id(a.id3_test or a.id3_full, "test", a.kernel) if "id3" in SRC else {}
    zd = np.load(D64_TEST, allow_pickle=True); qd = [str(x) for x in zd["question_ids"]]; LDt = np.log(np.clip(zd["probabilities"].astype(np.float64), 1e-9, 1))
    Gt = {}
    for sp in ("public", "private"):
        with open(os.path.join(ROOT, "data", "raw", "official_rank_tests_20261104_v1", "tests", f"rank_prediction_test_{sp}.csv"), newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f): Gt[r["question_id"]] = int(r["num_games"])
    mean = np.mean([exps[G] for G in GS], 0); ex = {G: (1 - a.shrink) * exps[G] + a.shrink * mean for G in GS}
    Pn = []; Pf = []
    if os.environ.get("SAVE_T"):      # per-game per-source evidence of the official questions + fitted parameters (label-free population experiments, test_source_scan.py)
        import pickle; pickle.dump(dict(sources=NS, qid=qd, G=[Gt[x] for x in qd], ev=[np.stack([vec(f"{x}#{s}", tv, idt, ddt, idt2, idt3) for s in range(1, Gt[x] + 1)]) for x in qd],
                                        theta={G: np.asarray(theta_f[G]) for G in GS}, LD=LDt, ex={G: np.asarray(ex[G]) for G in GS}), open(os.environ["SAVE_T"], "wb")); print("test evidence saved to", os.environ["SAVE_T"])
    if SELF: se_t = self_ev("test"); svt = se_t.evidence([[f"{qid}#{s}" for s in range(1, Gt[qid] + 1)] for qid in qd]); del se_t
    for j, qid in enumerate(qd):
        G = Gt[qid]; S = np.sum([vec(f"{qid}#{s}", tv, idt, ddt, idt2, idt3) for s in range(1, G + 1)], 0)
        if RE: S = np.stack([s_ * re_fac([f"{qid}#{s}" for s in range(1, G + 1)]) if SRC[j] in ("id", "id2") else s_ for j, s_ in enumerate(S)])
        if SELF: S = np.concatenate([S, svt[j][None]])
        p = prob(theta_f[G], [s[None] for s in S])[0]
        Pn.append(p); Pf.append(pool(np.log(np.clip(p, 1e-9, 1))[None], LDt[j][None], *ex[G])[0])
    Pn = np.array(Pn); Pf = np.array(Pf)
    for name, P in ((a.tag + "_new", Pn), (a.tag + "_x_D64", Pf)):
        pred = E.decode(P); path = os.path.join(a.out_dir, f"go_rank_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["question_id", "pred_rank"]); [w.writerow([x, RANKS[p]]) for x, p in zip(qd, pred)]
        np.savez(os.path.join(a.out_dir, f"go_rank_{name}_posteriors.npz"), probabilities=P, question_ids=np.array(qd))
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest().upper(); rep[name] = dict(sha256=sha, counts=np.bincount(pred, minlength=10).tolist(), changed_vs_d64=int((pred != E.decode(np.exp(LDt))).sum()))
        print("wrote", path, rep[name])
    rep["exponents_used"] = {str(G): np.round(v, 4).tolist() for G, v in ex.items()}
    json.dump(rep, open(os.path.join(a.out_dir, f"go_rank_{a.tag}_VALIDATION.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
