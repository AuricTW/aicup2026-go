"""GT lives here, never in inference. These functions do not load saved scores."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np
from .artifacts import ContractError
from .contracts import identifier, integer
from .scoring import top_five, finite_matrix


def rank_score(rank):
    if rank is None: return 0.0
    integer(rank)
    return math.exp(1 - rank) if rank <= 5 else 0.0


def evaluate_matrix(scores, question_order, candidate_order, truth, *, denominator=400,
                    expected_candidates=400):
    x = finite_matrix(scores)
    integer(denominator)
    if len(set(question_order)) != len(question_order) or set(question_order) != set(truth):
        raise ContractError("Question/GT sets differ")
    if len(question_order) != denominator or x.shape != (denominator, expected_candidates):
        raise ContractError("Fixed denominator/matrix mismatch")
    if not set(truth.values()) <= set(candidate_order):
        raise ContractError("GT outside candidate pool")
    rows = []
    for i, qid in enumerate(question_order):
        identifier(qid)
        pred = top_five(x[i], candidate_order, expected_candidates=expected_candidates)
        order = np.argsort(-x[i], kind="stable")
        rank = int(np.flatnonzero(np.asarray(candidate_order)[order] == truth[qid])[0]) + 1
        rows.append({"question_id": qid, "top5": pred, "answer_rank": rank,
                     "score": rank_score(rank), "top1": int(rank == 1), "top5_hit": int(rank <= 5)})
    return {"score": math.fsum(r["score"] for r in rows) / denominator,
            "top1": sum(r["top1"] for r in rows) / denominator,
            "top5": sum(r["top5_hit"] for r in rows) / denominator,
            "denominator": denominator, "rows": rows}


@dataclass(frozen=True)
class PairedEvent:
    player: str
    pool: str
    training_seed: int
    q: int
    question_id: str
    gain: float


def paired_cluster_intervals(events, *, family_size, repetitions=10000, seed,
                             quantile_method="linear"):
    """Conditional on fixed banks. Same player multiplicity across all cells.

    Average players within each pool/seed/q, then equal pools and seeds.
    Never infer that a training seed is another independent player.
    """
    if repetitions < 10000 or family_size not in (3, 4, 10):
        raise ContractError("Unregistered bootstrap budget/family")
    if quantile_method != "linear":
        raise ContractError("Only explicitly registered linear quantiles implemented")
    if not events:
        raise ContractError("No paired events")
    keys = set()
    for e in events:
        key = (e.pool, e.training_seed, e.q, e.question_id)
        identifier(e.player); identifier(e.pool); identifier(e.question_id)
        if key in keys or e.q not in (5, 10, 20) or not math.isfinite(e.gain) or not -1<=e.gain<=1:
            raise ContractError("Invalid or repeated paired event")
        keys.add(key)
    players = sorted({e.player for e in events})
    cells = sorted({(e.pool, e.training_seed) for e in events})
    pi = {p: i for i, p in enumerate(players)}
    ci = {c: i for i, c in enumerate(cells)}
    # Missing pool/seed combinations change the estimand, so reject them.
    pools = {p for p, _ in cells}; seeds = {s for _, s in cells}
    if len(cells) != len(pools) * len(seeds):
        raise ContractError("Incomplete pool/seed grid")
    numer = np.zeros((len(players), len(cells), 3), dtype=np.float64)
    denom = np.zeros_like(numer)
    qidx = {5: 0, 10: 1, 20: 2}
    player_grid = {}
    for e in events:
        i, j, k = pi[e.player], ci[(e.pool, e.training_seed)], qidx[e.q]
        numer[i, j, k] += e.gain; denom[i, j, k] += 1
        player_grid.setdefault((e.pool, e.training_seed, e.q), set()).add(e.player)
    for pool, s in cells:
        if any((pool, s, q) not in player_grid for q in qidx):
            raise ContractError("Incomplete q grid")
        if not (player_grid[pool, s, 5] == player_grid[pool, s, 10] == player_grid[pool, s, 20]):
            raise ContractError("Nested q endpoints need the same target players")
    point = (numer.sum(axis=0) / denom.sum(axis=0)).mean(axis=0)
    macro=(numer.sum(axis=1)/denom.sum(axis=1)).mean(axis=0)
    rng = np.random.default_rng(seed)
    samples, rejected, attempted = [], 0, 0
    # Bound retries for small/incompatible samples; never silently change estimand.
    while sum(len(s) for s in samples) < repetitions and attempted < repetitions * 20:
        n = min(256, repetitions - sum(len(s) for s in samples))
        weights = rng.multinomial(len(players), np.full(len(players), 1 / len(players)), size=n)
        ds = np.einsum("bp,pcq->bcq", weights, denom, optimize=False)
        ns = np.einsum("bp,pcq->bcq", weights, numer, optimize=False)
        valid = (ds > 0).all(axis=(1, 2))
        rejected += int((~valid).sum()); attempted += n
        if valid.any(): samples.append((ns[valid] / ds[valid]).mean(axis=1))
    if not samples or sum(len(s) for s in samples) < repetitions:
        return {"status": "inconclusive", "reason": "insufficient nonempty cluster resamples"}
    boot = np.concatenate(samples)[:repetitions]
    alpha = Fraction(1, 20 * family_size)
    lower = np.quantile(boot, float(alpha), axis=0, method=quantile_method)
    degenerate = np.ptp(boot, axis=0) <= 1e-15
    short_samples=boot[:,:2].mean(axis=1)
    return {"status": "inconclusive" if degenerate.any() else "computed_not_a_gate_pass",
            "q": [5, 10, 20], "point": point.tolist(), "lower": lower.tolist(),
            "unique_player_macro":macro.tolist(),
            "degenerate": degenerate.tolist(), "unique_players": len(players),
            "short_point":float(point[:2].mean()),
            "short_lower":float(np.quantile(short_samples,float(alpha),method=quantile_method)),
            "short_degenerate":bool(np.ptp(short_samples)<=1e-15),
            "training_seeds": sorted(seeds), "pools": sorted(pools),
            "repetitions": repetitions, "rejected_empty": rejected, "bootstrap_seed": seed,
            "family_size": family_size, "alpha_per_test_exact": str(alpha),
            "one_sided_coverage_exact": str(1 - alpha), "quantile_method": quantile_method,
            "conditional_on_fixed_candidate_banks": True,
            "adaptive_selection_bias_removed": False}
