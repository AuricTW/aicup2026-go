"""Second-generation evidence forms for the identity network (idnet.py), dev cross-fit on top of the per-game model.
 emp      empirical drift matrix  P(opponent rank r' | owner rank r)  from the opponent side of training games (tro block,
          confident recognitions only), separately for HD0 / HD1-owner-black / HD1-owner-white:
              ev_r = log(eps + sum_r' Kemp[r, r'] * mass_r' / prior_r')
 bins     the evidence split into confidence bins (top-1 prob), each bin with its own fitted scale
usage: idnet_eval2.py IDNET_PRED.npz[,PRED2 (rank masses averaged)] PRED_DIR[,DIR2] [--eps 0.02]
"""
import sys, os, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_queries as E
import predict_test as PT
from final_predict import prob
from idnet_eval import smooth, shift
PT.BALANCED = False


def emp_kernels(z, thr=0.5, a0=1.0):
    """K[c][r, r'] = P(opponent rank r' | owner rank r), c = 0: HD0, 1: HD1 owner black, 2: HD1 owner white."""
    hd = z["tro_hd"]; bl = z["tro_black"]; ok = z["tro_top"] > thr; out = []
    for c, m in enumerate((hd == 0, (hd == 1) & bl, (hd == 1) & ~bl)):
        K = np.full((10, 10), 0.0)
        np.add.at(K, (z["tro_rank"][m & ok], z["tro_toprank"][m & ok]), 1.0)
        if c:                                             # few HD1 games: shrink towards the shifted HD0 matrix
            base = np.zeros((10, 10)); s = 1 if c == 1 else -1
            for r in range(10):
                for q in range(10):
                    if 0 <= q - s < 10: base[r, q] = out[0][r, q - s]
            K = K + 20.0 * base
        K = K + a0 * 0.1; out.append(K / K.sum(1, keepdims=True))
    return out


def evidence(mass, top, hd, black, Ks, prior, eps, mode="emp"):
    """mass [N,10] opponent rank mass; returns centred evidence [N,10] for the TARGET's rank."""
    lik = mass / prior[None]; lik = lik / lik.sum(1, keepdims=True)
    ev = np.zeros_like(mass)
    for c, m in enumerate((hd == 0, (hd == 1) & black, (hd == 1) & ~black)):
        if m.any(): ev[m] = np.log(eps + lik[m] @ Ks[c].T)
    return ev - ev.mean(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("idp"); ap.add_argument("pred"); ap.add_argument("--eps", type=float, default=0.02)
    a = ap.parse_args(); zs = [np.load(p, allow_pickle=True) for p in a.idp.split(",")]; z = zs[0]
    kpos = {str(k): i for i, k in enumerate(z["dev_keys"])}; hd = z["dev_hd"]; black = z["dev_black"]
    acc = {}; metas = {}
    for d in a.pred.split(","):
        q = np.load(os.path.join(d, "dev_pred.npz"), allow_pickle=True); zo, zw = q["zopp"], q["zown"]
        for i, m in enumerate(q["meta"]):
            m = json.loads(m); acc.setdefault(m["key"], []).append(np.stack([zo[i], zw[i]]).astype(np.float64)); metas[m["key"]] = m
    keys = sorted(k for k in acc if metas[k].get("hd", 0) <= 1 and k in kpos); ii = np.array([kpos[k] for k in keys])
    V = {"zopp": np.stack([np.mean(acc[k], 0)[0] for k in keys]), "zown": np.stack([np.mean(acc[k], 0)[1] for k in keys])}
    h = hd[ii]; b = black[ii]
    mass = np.mean([zz["dev_opp_mass"][ii].astype(np.float64) for zz in zs], 0); tp = np.mean([zz["dev_opp_top"][ii] for zz in zs], 0)
    prior = z["rank_prior"].astype(np.float64) if "rank_prior" in z.files else np.bincount(z["ho_rank"], minlength=10).astype(np.float64); prior /= prior.sum()   # prior mass per rank
    print("accounts per rank share:", np.round(prior, 3))
    # reference: the old form
    m1 = smooth(mass, 1); ev = np.log(a.eps + m1); up = np.log(a.eps + shift(m1, 1)); dn = np.log(a.eps + shift(m1, -1))
    ev = np.where((h == 0)[:, None], ev, np.where(b[:, None], up, dn)); V["old_k1"] = ev - ev.mean(1, keepdims=True)
    for thr in (0.3, 0.6):
        Ks = emp_kernels(z, thr)
        if thr == 0.6: print("empirical P(opp rank | owner rank), HD0, confident:"); print(np.round(Ks[0], 2))
        for eps in (0.005, 0.02, 0.05):
            V[f"emp{thr}_e{eps}"] = evidence(mass, tp, h, b, Ks, prior, eps)
    Ks = emp_kernels(z, 0.5); base = evidence(mass, tp, h, b, Ks, prior, 0.02); V["emp"] = base
    for lo, hi in ((0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01), (0.3, 1.01), (0.2, 1.01), (0.4, 1.01), (0.15, 0.3), (0, 0.15), (0, 0.2), (0.6, 1.01)):
        V[f"b{lo}-{hi}"] = np.where(((tp > lo) & (tp <= hi))[:, None], base, 0.0)
    # continuous confidence weighting:  ev_r = log((1 - g) + g * 10 * sum_r' Kemp[r, r'] lik_r'),  g = g(top-1 prob)
    lik = mass / prior[None]; lik /= lik.sum(1, keepdims=True); kl = np.zeros_like(mass)
    for c, m in enumerate((h == 0, (h == 1) & b, (h == 1) & ~b)): kl[m] = lik[m] @ Ks[c].T
    oh = np.eye(10)[z["dev_opp_toprank"][ii]]; kt = np.zeros_like(mass)                    # top-1 account only
    for c, m in enumerate((h == 0, (h == 1) & b, (h == 1) & ~b)): kt[m] = (oh[m] / prior[None]) @ Ks[c].T
    kt /= kt.sum(1, keepdims=True)
    GF = {"g=t": tp, "g=t^.5": tp ** 0.5, "g=t^2": tp ** 2, "g=min(1,2t)": np.minimum(1, 2 * tp), "g=sig(10(t-.3))": 1 / (1 + np.exp(-10 * (tp - 0.3))),
          "g=sig(20(t-.25))": 1 / (1 + np.exp(-20 * (tp - 0.25))), "g=clip((t-.1)/.4)": np.clip((tp - 0.1) / 0.4, 0, 1)}
    for n, g in GF.items():
        e = np.log((1 - g)[:, None] * 1.0 + g[:, None] * 10 * kl + 1e-4); V["c_" + n] = e - e.mean(1, keepdims=True)
        e = np.log((1 - g)[:, None] * 1.0 + g[:, None] * 10 * kt + 1e-4); V["t_" + n] = e - e.mean(1, keepdims=True)
    if float(z["dev_own_top"].max()) > 0:                                                  # the target himself recognised as (a twin of) a gallery account
        om = z["dev_own_mass"][ii].astype(np.float64); ot = z["dev_own_top"][ii].astype(np.float64); ol = om / prior[None]; ol /= ol.sum(1, keepdims=True)
        for nm, g in (("own_g=t", ot), ("own_g=t^2", ot ** 2), ("own_g=t>.5", np.where(ot > 0.5, ot, 0.0))):
            e = np.log((1 - g)[:, None] + g[:, None] * 10 * smooth(ol, 1) + 1e-4); V[nm] = e - e.mean(1, keepdims=True)
    if "dev_opp_acc" in z.files:                                                           # account-specific opponent profiles (idens.py --acc)
        ac = z["dev_opp_acc"][ii].astype(np.float64); hd0 = (h == 0)[:, None] | ("acc_hd1" in z.files)
        for n in ("g=t", "g=t^.5", "g=min(1,2t)", "g=clip((t-.1)/.4)"):
            g = GF[n]; e = np.log((1 - g)[:, None] + g[:, None] * ac + 1e-4); e = e - e.mean(1, keepdims=True); V["a_" + n] = np.where(hd0, e, V["c_" + n])
        e = np.log(ac + 1e-3); e = e - e.mean(1, keepdims=True); V["a_plain"] = np.where(hd0, e, V["c_g=t"])
    if "dev_opp_lin" in z.files:                                                           # learned profile (idens.py --lin)
        e = z["dev_opp_lin"][ii].astype(np.float64); e = e - e.mean(1, keepdims=True); V["a_lin"] = np.where((h == 0)[:, None], e, V["c_g=t"])
    players = {}
    for i, k in enumerate(keys): players.setdefault(metas[k]["player_id"], (metas[k]["rank_idx"], []))[1].append(i)
    up_ = sorted(players); half = set(np.random.RandomState(7).choice(up_, len(up_) // 2, replace=False))
    macro = lambda y, P: float(np.mean([E.reward(y, E.decode(P))[y == r].mean() for r in range(10) if (y == r).any()]))
    M = ["zopp", "zown"]
    sets = [("model", M), ("+old_k1", M + ["old_k1"])] + [(f"+c_g=t +{k}", M + ["c_g=t", k]) for k in V if k.startswith("own_")] + [(f"+{k}", M + [k]) for k in V if k[:2] in ("c_", "a_")] + ([("+a_g=t + c_g=t", M + ["a_g=t", "c_g=t"])] if "a_g=t" in V else []) + [("+c_g=t + lo bin", M + ["c_g=t", "b0-0.3"]),
        ("+b>0.3", M + ["b0.3-1.01"]), ("+b>0.2", M + ["b0.2-1.01"]), ("+b>0.4", M + ["b0.4-1.01"]),
        ("+3bins(.3,.6)", M + ["b0-0.3", "b0.3-0.6", "b0.6-1.01"]), ("+4bins", M + ["b0-0.3", "b0.3-0.6", "b0.6-0.9", "b0.9-1.01"]),
        ("+bins(.15,.3,.6)", M + ["b0-0.15", "b0.15-0.3", "b0.3-0.6", "b0.6-1.01"]), ("+bins(.2;.3+)", M + ["b0-0.2", "b0.2-1.01"]),
        ("+bins(>.3: .3-.6,.6-.9,.9+)", M + ["b0.3-0.6", "b0.6-0.9", "b0.9-1.01"])]
    tot = {}
    for G in (5, 10, 15, 20):
        E.GS = [G]; qs = E.make_queries(players, 8, 0); y = np.array([q[1] for q in qs]); f = np.array([q[0] in half for q in qs])
        S = {k: np.stack([v[q[3]].sum(0) for q in qs]) for k, v in V.items()}
        if os.environ.get("ID_INTER") and "a_g=t" in V:       # interaction: trust the style model less / the identity more when much is recognised
            u = np.array([tp[q[3]].mean() for q in qs]); u = (u - u.mean())[:, None]
            S["zopp*u"] = S["zopp"] * u; S["id*u"] = S["a_g=t"] * u; S["zown*u"] = S["zown"] * u
            if G == 5: sets += [("+a_g=t +zopp*u", M + ["a_g=t", "zopp*u"]), ("+a_g=t +id*u", M + ["a_g=t", "id*u"]), ("+a_g=t +zopp*u +id*u", M + ["a_g=t", "zopp*u", "id*u"])]
        for name, src in sets:
            P = np.zeros((len(qs), 10))
            for side in (True, False):
                th = PT.fit([S[j][f != side] for j in src], y[f != side]); P[f == side] = prob(th, [S[j][f == side] for j in src])
            tot.setdefault(name, []).append(macro(y, P))
            if name in ("model", "+a_g=t") and os.environ.get("ID_BINS"):          # score by how much of the question is recognised
                st = np.array([tp[q[3]].sum() for q in qs]) / G; rw = E.reward(y, E.decode(P)); ed = np.quantile(st, [0, .25, .5, .75, 1.0])
                print(f"   G{G} {name:8s} by mean top-1 prob of the question's games (quartiles): " + "  ".join(f"[{ed[i]:.2f}-{ed[i + 1]:.2f}] {rw[(st >= ed[i]) & (st <= ed[i + 1])].mean():.4f}" for i in range(4)))
        print("G", G, "done", flush=True)
    print("                               G5      G10     G15     G20     mean")
    for k, v in tot.items(): print(f"   {k:28s}" + "  ".join(f"{x:.4f}" for x in v) + f"  {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
