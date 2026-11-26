"""Query-level evaluation of per-position log-probs on player-disjoint dev players.

game vector v_g = mean over that game's positions of 10-class log-probs.
query logits = beta_G * sum_{g in query} v_g + bias_G (10), fitted by NLL on one half of
dev players and evaluated on the other half (2-fold player cross-fit), decision = argmax of
official expected score U(a) = p[a] + e^-1 (p[a-1] + p[a+1]).
Reported: per G, mean over ranks (rank-macro) and plain mean over queries.
"""
import json, sys, os
import numpy as np
from scipy.optimize import minimize

E1 = np.exp(-1.0)
GS = [5, 10, 15, 20]


def game_vectors(lp, games):
    v = np.zeros((len(games), 10))
    for i, g in enumerate(games):
        v[i] = lp[g["offset"]:g["offset"] + g["count"]].astype(np.float64).mean(0)
    return v


def decode(p):
    u = p.copy()
    u[:, 1:] += E1 * p[:, :-1]
    u[:, :-1] += E1 * p[:, 1:]
    return u.argmax(1)


def reward(y, a):
    d = np.abs(y - a)
    return np.where(d == 0, 1.0, np.where(d == 1, E1, 0.0))


def make_queries(players, n_per=4, seed=0):
    """players: dict pid -> (rank, [game indices]). Returns list of (pid, rank, G, idx array)."""
    rng = np.random.RandomState(seed)
    out = []
    for pid in sorted(players):
        r, gl = players[pid]
        for G in GS:
            if len(gl) < G:
                continue
            for _ in range(n_per):
                out.append((pid, r, G, rng.choice(gl, G, replace=False)))
    return out


def fit_G(S, y):
    """S: (n,10) summed game vectors; fit beta (scalar) + bias(10)."""
    def nll(theta):
        z = theta[0] * S + theta[1:]
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean() + 1e-3 * (theta[1:] ** 2).sum()
    res = minimize(nll, np.r_[1.0, np.zeros(10)], method="L-BFGS-B")
    return res.x


def apply(theta, S):
    z = theta[0] * S + theta[1:]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def evaluate(v, games, n_per=4, seed=0, return_detail=False):
    players = {}
    for i, g in enumerate(games):
        players.setdefault(g["player_id"], (g["rank_idx"], []))[1].append(i)
    pids = sorted(players)
    rng = np.random.RandomState(123)
    half = set(rng.choice(pids, len(pids) // 2, replace=False))
    qs = make_queries(players, n_per, seed)
    res = {}
    rows = []
    thetas = {}
    for G in GS:
        q = [x for x in qs if x[2] == G]
        S = np.stack([v[x[3]].sum(0) for x in q])
        y = np.array([x[1] for x in q])
        fold = np.array([x[0] in half for x in q])
        pred = np.zeros(len(q), int)
        P = np.zeros((len(q), 10))
        for f in [True, False]:
            th = fit_G(S[fold != f], y[fold != f])
            thetas[(G, f)] = th
            P[fold == f] = apply(th, S[fold == f])
        pred = decode(P)
        rw = reward(y, pred)
        per_rank = [rw[y == r].mean() for r in range(10) if (y == r).any()]
        res[G] = dict(n=len(q), mean=float(rw.mean()), rank_macro=float(np.mean(per_rank)),
                      acc=float((pred == y).mean()), per_rank=[round(float(x), 3) for x in per_rank])
        for x, pr, rr, pp in zip(q, pred, rw, P):
            rows.append((x[0], x[1], G, int(pr), float(rr)))
    res["overall_rank_macro"] = float(np.mean([res[G]["rank_macro"] for G in GS]))
    res["overall_mean"] = float(np.mean([res[G]["mean"] for G in GS]))
    if return_detail:
        return res, rows, thetas
    return res


if __name__ == "__main__":
    d = sys.argv[1]
    lp = np.load(os.path.join(d, "dev_logp.npy"))
    games = json.load(open(os.path.join(d, "dev_games.json")))["games"]
    v = game_vectors(lp, games)
    r = evaluate(v, games)
    print(json.dumps(r, indent=1))
