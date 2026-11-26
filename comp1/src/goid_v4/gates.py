"""Prospective utility gates, distinct from historical board gate and official score."""
from __future__ import annotations
import math
from fractions import Fraction
from .artifacts import ContractError

SEEDS = (31415, 27182, 16180)
G0_KEYS = ("lawful_scope", "player_group_isolation", "rules_closed", "no_retired_reads",
           "no_GT_in_inference", "numeric_contract", "schema_contract")


def g0(evidence):
    if set(evidence) != set(G0_KEYS):
        raise ContractError("G0 evidence keys incomplete")
    if any(v is not None and type(v) is not bool for v in evidence.values()):
        raise ContractError("G0 requires explicit true/false/null facts")
    status = "passed_synthetic_declared_checks" if all(v is True for v in evidence.values()) else "pending_or_failed"
    return {"status": status, "checks": evidence, "quality_execution_allowed": False,
            "note": "Engineering check only; real scope receipts, environment and execution scope are not supplied here"}


def g1(per_seed):
    """One already-pinned main candidate. No score loading or checkpoint choice."""
    if set(per_seed) != set(SEEDS):
        raise ContractError("Need the three fixed training seeds")
    for rows in per_seed.values():
        if set(rows) != {5, 10, 20}:
            raise ContractError("G1 needs q5/10/20")
        for row in rows.values():
            if set(row) != {"score_gain", "top1_gain"} or not all(type(v) in (float, int) and math.isfinite(v) and -1 <= v <= 1 for v in row.values()):
                raise ContractError("Invalid gate input")
    scores = {q: sum(per_seed[s][q]["score_gain"] for s in SEEDS) / 3 for q in (5, 10, 20)}
    top1 = {q: sum(per_seed[s][q]["top1_gain"] for s in SEEDS) / 3 for q in (5, 10, 20)}
    seed_short = {s: (per_seed[s][5]["score_gain"] + per_seed[s][10]["score_gain"]) / 2 for s in SEEDS}
    checks = {"short_gain_at_least_01": (scores[5] + scores[10]) / 2 >= .01,
              "q5_nonnegative": scores[5] >= 0, "q10_nonnegative": scores[10] >= 0,
              "q20_guard": scores[20] >= -.005, "q5_top1_guard": top1[5] >= -.005,
              "q10_top1_guard": top1[10] >= -.005,
              "two_positive_seeds": sum(v > 0 for v in seed_short.values()) >= 2}
    return {"status": "passed_development_only" if all(checks.values()) else "failed",
            "checks": checks, "mean_score_gain": scores, "mean_top1_gain": top1,
            "seed_short_gain": seed_short, "independent_confirmation": False}


def g2(g1_result, intervals, *, route, minimum_targets, power_plan_locked,registered_family_size):
    if route not in ("pristine_400way", "prospective_blocked_internal", "fixed_distractor_conditional"):
        return {"status": "inconclusive", "reason": "no locked lawful evaluation route"}
    if minimum_targets is None or type(minimum_targets) is not int or minimum_targets <= 0 or not power_plan_locked:
        return {"status": "inconclusive", "reason": "target count/power plan not locked"}
    if route=='pristine_400way' and (minimum_targets<400 or intervals.get('unique_players',0)<400):
        return {'status':'inconclusive','reason':'400-way pristine confirmation needs at least400 new target identities'}
    if g1_result["status"] != "passed_development_only":
        return {"status": "failed", "reason": "G1 not reproduced"}
    if intervals.get("q") != [5, 10, 20] or len(intervals.get("lower", [])) != 3 or any(not math.isfinite(x) for x in intervals["lower"]):
        raise ContractError("G2 interval endpoints incomplete/nonfinite")
    if registered_family_size not in (3,4,10) or intervals.get('family_size')!=registered_family_size or intervals.get('one_sided_coverage_exact')!=str(1-Fraction(1,20*registered_family_size)) or intervals.get('repetitions',0)<10000:
        raise ContractError('G2 interval does not match the preregistered family/budget')
    if intervals.get("status") != "computed_not_a_gate_pass" or intervals["unique_players"] < minimum_targets:
        return {"status": "inconclusive", "reason": "insufficient or degenerate uncertainty"}
    checks = {str(q): lo > threshold for q, lo, threshold in zip((5, 10, 20), intervals["lower"], (0, 0, -.005))}
    return {"status": "passed" if all(checks.values()) else "not_demonstrated",
            "checks": checks, "route": route,
            "independent_confirmation": route == "pristine_400way"}


def p3_mechanism(per_seed_short_gain,*,interval=None,registered_family_size=None):
    if set(per_seed_short_gain)!=set(SEEDS) or any(not math.isfinite(v) or not -1<=v<=1 for v in per_seed_short_gain.values()):
        raise ContractError('P3 mechanism needs fixed3seed paired short gains vs same-phi raw-only')
    passed=sum(per_seed_short_gain.values())/3>0
    result={'g1_mechanism_passed':passed,'g2_mechanism':'inconclusive','identity_main_gate_separately_required':True}
    if interval is not None:
        if registered_family_size not in (4,10) or interval.get('family_size')!=registered_family_size or interval.get('one_sided_coverage_exact')!=str(1-Fraction(1,20*registered_family_size)) or interval.get('repetitions',0)<10000:
            raise ContractError('P3 mechanism comparison not in the locked family')
        if not interval.get('short_degenerate',True):
            lower=interval.get('short_lower')
            if lower is None or not math.isfinite(lower):raise ContractError('Invalid mechanism bound')
            result['g2_mechanism']='passed' if passed and lower>0 else 'not_demonstrated'
    return result
