"""Strict local metrics; this module has not been validated on official test data.

The competition score is the mean of exp(-(rank - 1)), for ranks 1 through 5.
A ground-truth player outside the submitted top five receives zero.  Evaluate
each pool separately with the official fixed denominator of 400.  A different
denominator must be supplied explicitly for synthetic tests.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _identifier(value: object, description: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{description} must be a nonempty, unpadded string")
    return value


def exponential_rank_score(rank: int | None) -> float:
    """Return exp(-(rank - 1)) for a one-based top-five rank, or zero for None.

    Use None to represent a miss.  Invalid ranks are errors rather than silent
    misses, so accidental zero-based ranks cannot change the metric unnoticed.
    """
    if rank is None:
        return 0.0
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 5:
        raise ValueError("rank must be an integer from 1 to 5, or None for a miss")
    return math.exp(-(rank - 1))


def evaluate(
    predictions: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, str],
    *,
    denominator: int = 400,
) -> dict[str, Any]:
    """Evaluate complete, five-player predictions against complete ground truth.

    Question sets must match exactly, and the ground-truth count must equal the
    explicitly fixed denominator.  Blank identifiers, repeated top-five players,
    missing/extra questions, and invalid ground truth raise ValueError.  This
    function does not know the candidate universe; use check_submission.py to
    validate candidates and public/private pool membership before submission.
    """
    if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
        raise ValueError("denominator must be a positive integer")
    if not isinstance(predictions, Mapping) or not isinstance(ground_truth, Mapping):
        raise ValueError("predictions and ground_truth must be mappings")

    for question_id, player_id in ground_truth.items():
        _identifier(question_id, "ground-truth question_id")
        _identifier(player_id, f"ground truth for {question_id!r}")
    for question_id in predictions:
        _identifier(question_id, "prediction question_id")

    missing = sorted(set(ground_truth) - set(predictions))
    extra = sorted(set(predictions) - set(ground_truth))
    if missing or extra:
        raise ValueError(f"question sets differ: missing={missing!r}; extra={extra!r}")
    if len(ground_truth) != denominator:
        raise ValueError(
            f"ground-truth count {len(ground_truth)} differs from fixed denominator {denominator}"
        )

    ranks: dict[str, int] = {str(rank): 0 for rank in range(1, 6)}
    misses = 0
    scores: list[float] = []
    for question_id, truth in ground_truth.items():
        top_five = predictions[question_id]
        if isinstance(top_five, (str, bytes)) or not isinstance(top_five, Sequence):
            raise ValueError(f"predictions for {question_id!r} must be a sequence of five players")
        if len(top_five) != 5:
            raise ValueError(f"predictions for {question_id!r} must contain exactly five players")
        players = [_identifier(value, f"player prediction for {question_id!r}") for value in top_five]
        if len(set(players)) != 5:
            raise ValueError(f"repeated top-five player for {question_id!r}")
        rank = players.index(truth) + 1 if truth in players else None
        scores.append(exponential_rank_score(rank))
        if rank is None:
            misses += 1
        else:
            ranks[str(rank)] += 1

    score_sum = math.fsum(scores)
    return {
        "questions": len(ground_truth),
        "denominator": denominator,
        "score": score_sum / denominator,
        "score_sum": score_sum,
        "top1_accuracy": ranks["1"] / denominator,
        "top5_accuracy": (denominator - misses) / denominator,
        "rank_counts": ranks,
        "misses": misses,
    }
