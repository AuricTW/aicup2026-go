"""Dev evaluation of the identity-network evidence (idnet.py) on top of a per-game model.
Per game: rank mass of the recognised OPPONENT account (sum of account probabilities per rank).  Evidence for the target's
rank r:  log(eps + sum_r' K[r, r'] * mass_r')  with a label-drift kernel K (0 = none, 1 = [.2,.6,.2], 2 = [.1,.2,.4,.2,.1]);
HD1 games shift the opponent one rank up (target black) or down (target white).
usage: idnet_eval.py IDNET_PRED.npz PRED_DIR[,DIR2] [--nb IDFP_EVIDENCE.npz]
"""
import sys, os, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_queries as E
import predict_test as PT
from final_predict import prob
PT.BALANCED = False
KER = {0: [1.0], 1: [0.2, 0.6, 0.2], 2: [0.1, 0.2, 0.4, 0.2, 0.1]}


def smooth(mass, k):
    w = KER[k]; h = len(w) // 2; out = np.zeros_like(mass)
    for j, v in enumerate(w):
        s = j - h
        if s == 0: out += v * mass
        elif s > 0: out[:, s:] += v * mass[:, :-s]
        else: out[:, :s] += v * mass[:, -s:]
    return out


def shift(mass, s):
    out = np.zeros_like(mass)
    if s > 0: out[:, :-s] = mass[:, s:]          # target rank r <- opponent rank r + s
    elif s < 0: out[:, -s:] = mass[:, :s]
    else: out = mass
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("idp"); ap.add_argument("pred"); ap.add_argument("--nb", default=None); ap.add_argument("--eps", type=float, default=0.02)
    a = ap.parse_args()
    z = np.load(a.idp, allow_pickle=True); kpos = {str(k): i for i, k in enumerate(z["dev_keys"])}; hd = z["dev_hd"]; black = z["dev_black"]
    acc = {}; metas = {}
    for d in a.pred.split(","):
        q = np.load(os.path.join(d, "dev_pred.npz"), allow_pickle=True); zo, zw = q["zopp"], q["zown"]
        for i, m in enumerate(q["meta"]):
            m = json.loads(m); acc.setdefault(m["key"], []).append(np.stack([zo[i], zw[i]]).astype(np.float64)); metas[m["key"]] = m
    keys = sorted(k for k in acc if metas[k].get("hd", 0) <= 1 and k in kpos); ii = np.array([kpos[k] for k in keys])
    V = {"zopp": np.stack([np.mean(acc[k], 0)[0] for k in keys]), "zown": np.stack([np.mean(acc[k], 0)[1] for k in keys])}
    h = hd[ii]; b = black[ii]
    for side in ("opp", "own"):
        mass = z[f"dev_{side}_mass"][ii].astype(np.float64)
        for k in KER:
            m0 = smooth(mass, k); ev = np.log(a.eps + m0)
            if side == "opp":
                up = np.log(a.eps + shift(m0, 1)); dn = np.log(a.eps + shift(m0, -1))
                ev = np.where((h == 0)[:, None], ev, np.where(b[:, None], up, dn))       # HD1: black is the weaker side
            V[f"id_{side}_k{k}"] = ev - ev.mean(1, keepdims=True)
    tp = z["dev_opp_top"][ii]                                 # where does the gain come from: confident recognitions or the diffuse rest?
    V["id_hi"] = np.where((tp > 0.3)[:, None], V["id_opp_k1"], 0.0); V["id_lo"] = np.where((tp <= 0.3)[:, None], V["id_opp_k1"], 0.0)
    print(f"share of games with top-1 prob > 0.3: {(tp > 0.3).mean():.3f}")
    if a.nb:
        nb = np.load(a.nb, allow_pickle=True); npos = {k: i for i, k in enumerate(nb["keys"])}; V["nb_oppAB"] = np.stack([nb["oppAB"][npos[k]] if k in npos else np.zeros(10) for k in keys])
    print(f"dev games {len(keys)}; opponent side top-1 prob mean {z['dev_opp_top'][ii].mean():.3f}, own side {z['dev_own_top'][ii].mean():.3f}")
    players = {}
    for i, k in enumerate(keys): players.setdefault(metas[k]["player_id"], (metas[k]["rank_idx"], []))[1].append(i)
    up_ = sorted(players); half = set(np.random.RandomState(7).choice(up_, len(up_) // 2, replace=False))
    macro = lambda y, P: float(np.mean([E.reward(y, E.decode(P))[y == r].mean() for r in range(10) if (y == r).any()]))
    sets = [("model", ["zopp", "zown"])] + [(f"+id_opp_k{k}", ["zopp", "zown", f"id_opp_k{k}"]) for k in KER] + [("+id_opp_k1+id_own_k0", ["zopp", "zown", "id_opp_k1", "id_own_k0"]), ("id_opp_k1 only", ["id_opp_k1"]), ("id_own_k0 only", ["id_own_k0"]),
                                                                                                                         ("+id_hi (top>0.3 games)", ["zopp", "zown", "id_hi"]), ("+id_lo (other games)", ["zopp", "zown", "id_lo"]), ("+id_hi+id_lo", ["zopp", "zown", "id_hi", "id_lo"])]
    if a.nb: sets += [("+nb_oppAB", ["zopp", "zown", "nb_oppAB"]), ("+nb_oppAB+id_opp_k1", ["zopp", "zown", "nb_oppAB", "id_opp_k1"])]
    tot = {}
    for G in (5, 10, 15, 20):
        E.GS = [G]; qs = E.make_queries(players, 8, 0); y = np.array([q[1] for q in qs]); f = np.array([q[0] in half for q in qs]); line = f"G{G}:"
        S = {k: np.stack([v[q[3]].sum(0) for q in qs]) for k, v in V.items()}
        for name, src in sets:
            P = np.zeros((len(qs), 10))
            for side in (True, False):
                th = PT.fit([S[j][f != side] for j in src], y[f != side]); P[f == side] = prob(th, [S[j][f == side] for j in src])
            sc = macro(y, P); tot.setdefault(name, []).append(sc); line += f"  {name} {sc:.4f}"
        print(line, flush=True)
    print("mean over G:"); [print(f"   {k:26s} {np.mean(v):.4f}") for k, v in tot.items()]


if __name__ == "__main__":
    main()
