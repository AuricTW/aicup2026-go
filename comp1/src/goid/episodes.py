"""Deterministic, leakage-checked local player-identification episodes.

This module deliberately uses only the standard library. The original canonical
hash covers full main line, board size and setup stones under D4. When an audit
also supplies split_group_hash, its conservative near-duplicate family overrides
the full-game hash for splitting and deduplication. Neither key may encode player
ID or target colour. Unidentified partial/near duplicates remain a limitation.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class GameRecord:
    row_id: int
    player_id: str
    canonical_hash: str
    rank: str = ""
    color: str = ""
    split_group_hash: str = ""

    @property
    def group_hash(self) -> str:
        return self.split_group_hash or self.canonical_hash


@dataclass(slots=True)
class Partition:
    records: dict[str, list[GameRecord]]
    player_splits: dict[str, str]
    report: dict


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    player_id: str
    query_games: int
    query_row_ids: tuple[int, ...]


@dataclass(slots=True)
class EpisodePlan:
    candidate_ids: tuple[str, ...]
    support_row_ids: dict[str, tuple[int, ...]]
    questions: tuple[Question, ...]
    report: dict

    def serializable(self) -> dict:
        return {
            "candidate_ids": self.candidate_ids,
            "support_row_ids": self.support_row_ids,
            "questions": [asdict(q) for q in self.questions],
            "report": self.report,
        }


class EligibilityError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            f"Only {report['eligible_players']} players have at least "
            f"{report['minimum_unique_games_per_player']} globally unique games; "
            f"requested {report['requested_candidates']} candidates. "
            "Never duplicate games to fill an episode."
        )


class IncompleteAuditError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(
            "Audit is incomplete; refusing baseline evaluation: "
            f"parse_ok_rows={report['parse_ok_rows']}, "
            f"parse_failure_rows={report['parse_failure_rows']}, "
            f"unparsed_rows={report['unparsed_rows']}, "
            f"invalid_parse_status_rows={report['invalid_parse_status_rows']}"
        )


def stable_key(seed: int, value: str) -> bytes:
    return hashlib.blake2b(f"{seed}\0{value}".encode("utf-8"), digest_size=16).digest()


def read_audit(path: str | Path) -> tuple[list[GameRecord], dict]:
    """Load successful parses from the root audit's read-only SQLite database."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        required = {"row_id", "player_id", "canonical_hash", "rank", "color", "parse_ok"}
        actual = {row[1] for row in connection.execute("PRAGMA table_info(games)")}
        if not required <= actual:
            raise ValueError(f"games is missing required columns: {sorted(required - actual)}")
        grouping_expression = "COALESCE(split_group_hash,canonical_hash)" if "split_group_hash" in actual else "canonical_hash"
        total, parsed, failed, unparsed, invalid, minimum_row, maximum_row, unique_rows, invalid_rows = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(parse_ok=1),0), "
            "COALESCE(SUM(parse_ok=0),0), COALESCE(SUM(parse_ok IS NULL),0), "
            "COALESCE(SUM(parse_ok IS NOT NULL AND parse_ok NOT IN (0,1)),0), "
            "MIN(row_id),MAX(row_id),COUNT(DISTINCT row_id), "
            "COALESCE(SUM(typeof(row_id)!='integer'),0) FROM games"
        ).fetchone()
        report = {
            "audit_total_rows": total,
            "parse_ok_rows": parsed,
            "parse_failure_rows": failed,
            "unparsed_rows": unparsed,
            "invalid_parse_status_rows": invalid,
            # Preserve the public field, now counting only observed failures.
            "excluded_parse_failures": failed,
            "group_policy": (
                "COALESCE(split_group_hash,canonical_hash): conservative audited near-duplicate family when present; otherwise full-game D4 canonical"
                if "split_group_hash" in actual else "Full-game D4 canonical; no additional family column in audit"
            ),
        }
        override_rows, override_families = (connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT split_group_hash) FROM games WHERE split_group_hash IS NOT NULL"
        ).fetchone() if "split_group_hash" in actual else (0, 0))
        report["split_group_override_rows"] = override_rows
        report["split_group_override_families"] = override_families
        if unparsed or invalid:
            raise IncompleteAuditError(report)
        if invalid_rows or unique_rows != total or (total and (minimum_row != 0 or maximum_row != total - 1)):
            raise ValueError("Complete audit row_id mapping must be contiguous integers from 0 to audit_total_rows-1")
        result = []
        seen_rows = set()
        for row_id, player, canonical, rank, color, group_hash in connection.execute(
            f"SELECT row_id,player_id,canonical_hash,rank,color,{grouping_expression} "
            "FROM games WHERE parse_ok=1 ORDER BY row_id"
        ):
            if not isinstance(row_id, int) or row_id < 0 or row_id in seen_rows:
                raise ValueError(f"Invalid or repeated feature row_id: {row_id!r}")
            if not player or not canonical or not group_hash:
                raise ValueError(f"Successful parse has missing player/hash at row_id={row_id}")
            seen_rows.add(row_id)
            result.append(GameRecord(row_id, str(player), str(canonical), str(rank or ""), str(color or ""), str(group_hash)))
        return result, report
    finally:
        connection.close()


def partition_games(
    records: Iterable[GameRecord], *, seed: int = 20261025,
    fit_fraction: float = 0.6, validation_fraction: float = 0.25,
) -> Partition:
    """Split players before deduplication, reserving held-out effective groups.

    An effective group crossing splits belongs to holdout before validation before
    fit. Thus validation/holdout games appearing through an opponent view are
    removed from fit. Exactly one target view is then retained per group globally.
    Keys can be full-game D4 hashes or explicit conservative near-duplicate family
    hashes. Other target views/family members are discarded within each split.
    """
    if not (0 < fit_fraction < 1 and 0 < validation_fraction < 1 and fit_fraction + validation_fraction < 1):
        raise ValueError("fit_fraction and validation_fraction must be positive and sum to less than one")
    records = list(records)
    player_splits = {}
    for player in sorted({r.player_id for r in records}):
        value = int.from_bytes(stable_key(seed, "player:" + player), "big") / 2**128
        player_splits[player] = (
            "fit" if value < fit_fraction else
            "validation" if value < fit_fraction + validation_fraction else "holdout"
        )
    priority = {"holdout": 0, "validation": 1, "fit": 2}
    canonical_choice: dict[str, tuple[int, bytes, GameRecord]] = {}
    before = Counter()
    for record in records:
        split = player_splits[record.player_id]
        before[split] += 1
        choice = (priority[split], stable_key(seed, f"view:{record.row_id}"), record)
        current = canonical_choice.get(record.group_hash)
        if current is None or choice[:2] < current[:2]:
            canonical_choice[record.group_hash] = choice
    result = {split: [] for split in ("fit", "validation", "holdout")}
    removed_cross_split = Counter()
    removed_within_split = Counter()
    for record in records:
        chosen = canonical_choice[record.group_hash][2]
        split = player_splits[record.player_id]
        if chosen.row_id == record.row_id:
            result[split].append(record)
        elif split != player_splits[chosen.player_id]:
            removed_cross_split[split] += 1
        else:
            removed_within_split[split] += 1
    report = {
        "seed": seed,
        "fit_fraction": fit_fraction,
        "validation_fraction": validation_fraction,
        "holdout_fraction": 1 - fit_fraction - validation_fraction,
        "deduplication": "global effective game group (audited near-duplicate family override, otherwise full-game D4 canonical); exactly one view/member; holdout > validation > fit",
        "effective_game_groups": len(canonical_choice),
        "splits": {
            split: {
                "assigned_players": sum(value == split for value in player_splits.values()),
                "retained_players": len({r.player_id for r in result[split]}),
                "rows_before_dedup": before[split],
                "rows_retained": len(result[split]),
                "removed_cross_split_group": removed_cross_split[split],
                "removed_same_split_duplicate_view": removed_within_split[split],
            }
            for split in result
        },
        "remaining_limitations": [
            "Full-game D4 equality and supplied near-duplicate families do not guarantee all truncated or near-duplicate games are identified.",
            "Unlabelled opponent identities cannot be split by player ID.",
            "No temporal split is claimed without dates.",
        ],
    }
    return Partition(result, player_splits, report)


def build_episodes(
    records: Iterable[GameRecord], *, candidates: int = 400, support_games: int = 100,
    query_games: tuple[int, ...] = (5, 10, 20), seed: int = 20261025,
    allow_smaller_candidate_pool: bool = False,
) -> EpisodePlan:
    """Build one fixed pool and nested 5/10/20-game queries per candidate.

    The largest query determines eligibility so each query-size comparison uses
    the same players and supports. Games are sampled without replacement.
    """
    if candidates < 5 or support_games < 1 or not query_games or any(q < 1 for q in query_games):
        raise ValueError("Need at least five candidates, positive support/query games")
    if len(set(query_games)) != len(query_games):
        raise ValueError("query_games must be distinct")
    players: dict[str, list[GameRecord]] = defaultdict(list)
    seen_rows, seen_canonicals = set(), set()
    for record in records:
        if record.row_id in seen_rows or record.group_hash in seen_canonicals:
            raise ValueError("Episode input must contain globally unique row IDs and effective game groups")
        seen_rows.add(record.row_id)
        seen_canonicals.add(record.group_hash)
        players[record.player_id].append(record)
    required = support_games + max(query_games)
    eligible = [player for player, games in players.items() if len(games) >= required]
    eligible.sort(key=lambda p: stable_key(seed, "candidate:" + p))
    report = {
        "requested_candidates": candidates,
        "eligible_players": len(eligible),
        "minimum_unique_games_per_player": required,
        "support_games_per_candidate": support_games,
        "query_game_counts": list(query_games),
        "players_by_unique_game_threshold": {
            str(threshold): sum(len(games) >= threshold for games in players.values())
            for threshold in (2, 5, 20, 25, 50, 100, 105, 110, 120)
        },
    }
    if len(eligible) < candidates and (not allow_smaller_candidate_pool or len(eligible) < 5):
        raise EligibilityError(report)
    chosen = tuple(eligible[:candidates])
    supports, questions = {}, []
    for player in chosen:
        games = sorted(players[player], key=lambda r: stable_key(seed, f"game:{r.row_id}"))
        supports[player] = tuple(r.row_id for r in games[:support_games])
        reserved = games[support_games:required]
        for count in query_games:
            token = hashlib.sha256(player.encode()).hexdigest()[:16]
            questions.append(Question(f"val_q{count}_{token}", player, count, tuple(r.row_id for r in reserved[:count])))
    report.update({
        "actual_candidates": len(chosen),
        "questions_total": len(questions),
        "denominator_per_query_size": len(chosen),
        "official_shape": len(chosen) == 400 and support_games == 100 and all(5 <= q <= 20 for q in query_games),
        "official_test_comparability": "Local training-derived validation; not an official leaderboard score, even with matched shape.",
        "nested_queries_share_games": True,
        "all_candidates_have_equal_support": True,
    })
    return EpisodePlan(chosen, supports, tuple(questions), report)
