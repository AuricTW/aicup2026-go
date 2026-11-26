"""Auditable SGF replay, disk-backed features, and player-disjoint splits.

No player identity or SGF metadata value is used as a model feature. Coordinates
use sgfmill's (row from bottom, column from left) convention throughout. The
first 16 image channels are eight post-move target/opponent board pairs. All
image channels use 0..255 and must be divided by 255 by consumers.
"""
from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sgfmill import boards, sgf

LEVELS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
RANK_TO_ID = {rank: i for i, rank in enumerate(LEVELS)}
CSV_FIELDS = ("player_id", "game_id", "rank", "color", "sgf_content")
INDEX_FIELDS = (
    "row_idx", "row_index", "source_file", "source_row", "player_id", "game_id",
    "rank", "rank_idx", "color", "split", "game_hash", "d4_hash", "valid",
    "error", "excluded_reason", "num_moves", "pass_count", "setup_count",
    "metadata_properties", "has_rank_metadata", "reference_checked",
)
FEATURE_NAMES = [
    "num_moves", "target_move_count", "opponent_move_count", "target_pass_count",
    "opponent_pass_count", "target_capture_count", "opponent_capture_count",
    "target_color_black", "target_capture_per_move", "opponent_capture_per_move",
    "target_pass_fraction", "opponent_pass_fraction", "target_row_mean",
    "target_col_mean", "target_row_std", "target_col_std", "target_edge_mean",
    "target_edge_std",
] + [f"target_region_{r}_{c}" for r in range(3) for c in range(3)] + [
    "first16_target_row_mean", "first16_target_col_mean",
    "first16_target_row_std", "first16_target_col_std", "first16_target_edge_mean",
    "target_jump_mean", "target_jump_std", "final_target_stones",
    "final_opponent_stones",
]
N_FEATURES = len(FEATURE_NAMES)
NEIGHBORS = tuple(
    tuple(rr * 19 + cc for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
          if 0 <= rr < 19 and 0 <= cc < 19)
    for r in range(19) for c in range(19)
)


def _make_d4_lookup() -> np.ndarray:
    lookup = np.empty((8, 362), dtype=np.uint16)
    for point in range(361):
        r, c = divmod(point, 19)
        transformed = ((r, c), (c, 18-r), (18-r, 18-c), (18-c, r),
                       (r, 18-c), (18-c, 18-r), (18-r, c), (c, r))
        lookup[:, point] = [rr * 19 + cc for rr, cc in transformed]
    lookup[:, 361] = 361  # Pass is not a geometric coordinate.
    return lookup


D4_LOOKUP = _make_d4_lookup()


class ReplayError(ValueError):
    """An unsupported or inconsistent game; the row remains in the index."""


class _Group:
    __slots__ = ("color", "stones", "liberties")

    def __init__(self, color: int, stones: set[int], liberties: set[int]):
        self.color, self.stones, self.liberties = color, stones, liberties


class ReplayBoard:
    """Incremental groups, occupied-point, suicide, and simple-ko checks.

    Superko and the original game's ruleset cannot be inferred from anonymized
    move-only SGF. Suicide/simple-ko violations are classified as unsupported
    rule/replay cases rather than asserted to be illegal under every ruleset.
    """
    def __init__(self):
        self.cells = [0] * 361
        self.groups: list[_Group | None] = [None] * 361
        self.ko_point: int | None = None
        self.ko_color: int | None = None

    def setup(self, black: set[int], white: set[int], empty: set[int]) -> None:
        if black & white or black & empty or white & empty:
            raise ReplayError("overlapping_setup_points")
        for point in black | white | empty:
            if not 0 <= point < 361:
                raise ReplayError("setup_out_of_bounds")
        for points, color in ((empty, 0), (black, 1), (white, 2)):
            for point in points:
                self.cells[point] = color
        self.groups = [None] * 361
        for point, color in enumerate(self.cells):
            if not color or self.groups[point] is not None:
                continue
            stones, liberties, pending = set(), set(), [point]
            while pending:
                current = pending.pop()
                if current in stones:
                    continue
                stones.add(current)
                for neighbor in NEIGHBORS[current]:
                    if self.cells[neighbor] == 0:
                        liberties.add(neighbor)
                    elif self.cells[neighbor] == color and neighbor not in stones:
                        pending.append(neighbor)
            if not liberties:
                raise ReplayError("setup_group_without_liberty")
            group = _Group(color, stones, liberties)
            for stone in stones:
                self.groups[stone] = group
        self.ko_point = self.ko_color = None

    def play(self, point: int | None, color: int) -> int:
        if point is None:
            self.ko_point = self.ko_color = None
            return 0
        if not 0 <= point < 361:
            raise ReplayError("move_out_of_bounds")
        if self.cells[point]:
            raise ReplayError("occupied_point")
        if point == self.ko_point and color == self.ko_color:
            raise ReplayError("simple_ko_violation_rules_unspecified")
        own: set[_Group] = set()
        opposing: set[_Group] = set()
        liberties = set()
        for neighbor in NEIGHBORS[point]:
            group = self.groups[neighbor]
            if group is None:
                liberties.add(neighbor)
            elif group.color == color:
                own.add(group)
            else:
                opposing.add(group)
        stones = {point}
        for group in own:
            stones.update(group.stones)
            liberties.update(group.liberties)
        liberties.discard(point)
        merged = _Group(color, stones, liberties)
        self.cells[point] = color
        for stone in stones:
            self.groups[stone] = merged
        captured: list[int] = []
        for group in opposing:
            group.liberties.discard(point)
            if not group.liberties:
                captured.extend(group.stones)
        for stone in captured:
            self.cells[stone] = 0
            self.groups[stone] = None
        for stone in captured:
            for neighbor in NEIGHBORS[stone]:
                group = self.groups[neighbor]
                if group is not None:
                    group.liberties.add(stone)
        if not merged.liberties:
            raise ReplayError("suicide_rules_unspecified")
        if len(captured) == len(merged.stones) == len(merged.liberties) == 1:
            self.ko_point, self.ko_color = captured[0], 3 - color
        else:
            self.ko_point = self.ko_color = None
        return len(captured)


@dataclass
class ParsedGame:
    features: np.ndarray
    tabular: np.ndarray
    game_hash: str
    d4_hash: str
    num_moves: int
    pass_count: int
    setup_count: int
    metadata_properties: str
    has_rank_metadata: int
    checked_moves: int
    checked_snapshots: int


def normalized_hashes(events: list[tuple[int, list[int]]]) -> tuple[str, str]:
    """Hash move colors/order, setup and pass, excluding all metadata values.

    Setup point lists are re-sorted for every symmetry. A game and any D4
    rotation/reflection therefore share d4_hash without a second replay.
    """
    kinds = np.fromiter((kind for kind, points in events for _ in points), dtype=np.uint16)
    points = np.fromiter((point for _, group in events for point in group), dtype=np.intp)
    tokens = kinds[None, :] * 512 + D4_LOOKUP[:, points]
    offset = 0
    for kind, group in events:
        end = offset + len(group)
        if kind >= 3:
            tokens[:, offset:end].sort(axis=1)
        offset = end
    serializations = [b"go-moves-v1:19:" + tokens[symmetry].astype("<u2", copy=False).tobytes()
                      for symmetry in range(8)]
    return (hashlib.sha256(serializations[0]).hexdigest(),
            hashlib.sha256(min(serializations)).hexdigest())


def _reference_cells(reference: boards.Board) -> list[int]:
    cells = [0] * 361
    for color, (r, c) in reference.list_occupied_points():
        cells[r * 19 + c] = 1 if color == "b" else 2
    return cells


def parse_game(content: str, target_color: str, cross_check: bool = False) -> ParsedGame:
    """Parse and replay every main-line move once; capture eight real positions."""
    if target_color not in ("B", "W"):
        raise ReplayError("invalid_color")
    if not isinstance(content, str) or not content.strip():
        raise ReplayError("empty_sgf")
    game = sgf.Sgf_game.from_string(content)
    if game.get_size() != 19:
        raise ReplayError(f"unsupported_board_size:{game.get_size()}")
    sequence = list(game.get_main_sequence())
    actions: list[tuple[tuple[set[int], set[int], set[int]] | None, str | None, int | None]] = []
    events: list[tuple[int, list[int]]] = []
    properties: set[str] = set()
    num_moves = pass_count = setup_count = 0
    for node in sequence:
        node_properties = set(node.properties())
        properties.update(node_properties)
        if "B" in node_properties and "W" in node_properties:
            raise ReplayError("two_move_colors_in_node")
        setup = None
        if node.has_setup_stones():
            points = node.get_setup_stones()
            setup = tuple({r * 19 + c for r, c in group} for group in points)
            for kind, group in zip((3, 4, 5), setup):
                if group:
                    events.append((kind, sorted(group)))
                    setup_count += len(group)
        color, move = node.get_move()
        point = None if move is None else move[0] * 19 + move[1]
        if color is not None:
            num_moves += 1
            pass_count += move is None
            events.append((1 if color == "b" else 2, [361 if point is None else point]))
        if setup is not None and color is not None:
            raise ReplayError("setup_and_move_in_same_node")
        actions.append((setup, color, point))
    if not num_moves:
        raise ReplayError("no_moves")
    selected = np.rint(np.linspace(0, num_moves - 1, 8)).astype(int).tolist()
    frames_at: dict[int, list[int]] = defaultdict(list)
    for frame, move_index in enumerate(selected):
        frames_at[move_index].append(frame)
    output = np.zeros((18, 19, 19), dtype=np.uint8)
    target = 1 if target_color == "B" else 2
    board = ReplayBoard()
    reference = boards.Board(19) if cross_check else None
    move_counts, pass_counts, captures = [0, 0, 0], [0, 0, 0], [0, 0, 0]
    target_points: list[tuple[int, int]] = []
    checked_moves = checked_snapshots = 0
    move_index = -1
    for setup, color_text, point in actions:
        if setup is not None:
            board.setup(*setup)
            if reference is not None:
                reference_setup = [{divmod(p, 19) for p in points} for points in setup]
                if not reference.apply_setup(*reference_setup):
                    raise ReplayError("reference_setup_invalid")
                if board.cells != _reference_cells(reference):
                    raise ReplayError("reference_mismatch_setup")
        if color_text is None:
            continue
        move_index += 1
        color = 1 if color_text == "b" else 2
        move_counts[color] += 1
        pass_counts[color] += point is None
        capture_count = board.play(point, color)
        captures[color] += capture_count
        if color == target and point is not None:
            target_points.append(divmod(point, 19))
        if reference is not None:
            before = len(reference.list_occupied_points())
            if point is not None:
                reference.play(*divmod(point, 19), color_text)
            reference_state = _reference_cells(reference)
            reference_capture = before + (point is not None) - sum(bool(c) for c in reference_state)
            if board.cells != reference_state or capture_count != reference_capture:
                raise ReplayError(f"reference_mismatch_move:{move_index + 1}")
            checked_moves += 1
        if move_index in frames_at:
            cells = np.asarray(board.cells, dtype=np.uint8).reshape(19, 19)
            for frame in frames_at[move_index]:
                output[frame * 2] = (cells == target).astype(np.uint8) * 255
                output[frame * 2 + 1] = (cells == 3 - target).astype(np.uint8) * 255
                checked_snapshots += int(reference is not None)
    output[16].fill(255 if target == 1 else 0)
    output[17].fill(round(255 * min(num_moves, 500) / 500))
    opponent = 3 - target
    values = [num_moves, move_counts[target], move_counts[opponent],
              pass_counts[target], pass_counts[opponent], captures[target], captures[opponent],
              int(target == 1), captures[target] / max(1, move_counts[target]),
              captures[opponent] / max(1, move_counts[opponent]),
              pass_counts[target] / max(1, move_counts[target]),
              pass_counts[opponent] / max(1, move_counts[opponent])]
    coords = np.asarray(target_points, dtype=np.float32).reshape(-1, 2)
    if len(coords):
        normalized = coords / 18.0
        edge = np.minimum(coords, 18 - coords).min(axis=1) / 9.0
        values += [*normalized.mean(axis=0), *normalized.std(axis=0), edge.mean(), edge.std()]
        regions = np.minimum(coords.astype(int) // 7, 2)
        hist = np.bincount(regions[:, 0] * 3 + regions[:, 1], minlength=9) / len(coords)
        values += hist.tolist()
        early = normalized[:16]
        values += [*early.mean(axis=0), *early.std(axis=0), edge[:16].mean()]
        jumps = np.abs(np.diff(coords, axis=0)).sum(axis=1) / 36.0
        values += [jumps.mean() if len(jumps) else 0, jumps.std() if len(jumps) else 0]
    else:
        values += [0] * 22
    values += [board.cells.count(target), board.cells.count(opponent)]
    tabular = np.asarray(values, dtype=np.float32)
    if tabular.shape != (N_FEATURES,) or not np.isfinite(tabular).all():
        raise ReplayError("tabular_shape_or_finiteness")
    game_hash, d4_hash = normalized_hashes(events)
    return ParsedGame(output, tabular, game_hash, d4_hash, num_moves, pass_count, setup_count,
                      ",".join(sorted(properties)), int(bool(properties & {"BR", "WR", "PB", "PW"})),
                      checked_moves, checked_snapshots)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_source(path: Path, limit: int | None) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if header is None or len(header) != len(CSV_FIELDS) or set(header) != set(CSV_FIELDS):
            raise ValueError(f"Unexpected CSV schema in {path}: {header}")
        count = 0
        for _ in reader:
            count += 1
            if limit is not None and count >= limit:
                break
        return count


def _source_worker(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(task["source"])
    image = np.lib.format.open_memmap(task["features"], mode="r+")
    tabular = np.lib.format.open_memmap(task["tabular"], mode="r+")
    check_rows = set(task["reference_rows"])
    counters: Counter[str] = Counter()
    metadata: Counter[str] = Counter()
    references: list[dict[str, Any]] = []
    seen_game_ids: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as handle, \
            open(task["index_shard"], "w", encoding="utf-8", newline="") as output:
        reader = csv.DictReader(handle, strict=True)
        writer = csv.DictWriter(output, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for source_row, row in enumerate(reader):
            if source_row >= task["count"]:
                break
            global_row = task["offset"] + source_row
            errors = []
            player_id, game_id = (str(row.get(key) or "").strip() for key in ("player_id", "game_id"))
            rank, color = str(row.get("rank") or "").strip(), str(row.get("color") or "").strip()
            if None in row or any(row.get(key) is None for key in CSV_FIELDS):
                errors.append("csv_field_count")
            if not player_id:
                errors.append("missing_player_id")
            if not game_id:
                errors.append("missing_game_id")
            elif game_id in seen_game_ids:
                errors.append("duplicate_game_id_within_file")
            seen_game_ids.add(game_id)
            if rank not in RANK_TO_ID:
                errors.append("invalid_rank")
            elif rank != task["rank"]:
                errors.append("rank_source_mismatch")
            if color not in ("B", "W"):
                errors.append("invalid_color")
            entry: dict[str, Any] = dict.fromkeys(INDEX_FIELDS, "")
            entry.update(row_idx=global_row, row_index=global_row, source_file=source.name,
                         source_row=source_row, player_id=player_id, game_id=game_id,
                         rank=rank, rank_idx=RANK_TO_ID.get(rank, -1), color=color,
                         valid=0, num_moves=0, pass_count=0, setup_count=0,
                         has_rank_metadata=0, reference_checked=0)
            try:
                parsed = parse_game(row.get("sgf_content") or "", color if color in ("B", "W") else "B",
                                    cross_check=source_row in check_rows)
                entry.update(game_hash=parsed.game_hash, d4_hash=parsed.d4_hash,
                             num_moves=parsed.num_moves, pass_count=parsed.pass_count,
                             setup_count=parsed.setup_count,
                             metadata_properties=parsed.metadata_properties,
                             has_rank_metadata=parsed.has_rank_metadata,
                             reference_checked=int(source_row in check_rows))
                metadata.update(parsed.metadata_properties.split(","))
                counters["moves_replayed"] += parsed.num_moves
                counters["passes"] += parsed.pass_count
                counters["setup_stones"] += parsed.setup_count
                counters["rows_with_identity_or_rank_metadata"] += parsed.has_rank_metadata
                if not errors:
                    image[global_row] = parsed.features
                    tabular[global_row] = parsed.tabular
                    entry["valid"] = 1
                if source_row in check_rows:
                    references.append({"source_file": source.name, "source_row": source_row,
                                       "row_idx": global_row, "game_id": game_id,
                                       "ok": True, "moves_checked": parsed.checked_moves,
                                       "snapshots_checked": parsed.checked_snapshots})
            except Exception as exc:
                error = f"{type(exc).__name__}:{exc}"
                errors.append(error)
                if source_row in check_rows:
                    references.append({"source_file": source.name, "source_row": source_row,
                                       "row_idx": global_row, "game_id": game_id, "ok": False,
                                       "error": error, "reference_mismatch": "reference_" in error})
            entry["error"] = "|".join(errors)
            counters["rows"] += 1
            counters["valid"] += int(entry["valid"])
            for error in errors:
                counters["error:" + ":".join(error.split(":", 2)[:2])] += 1
            writer.writerow(entry)
            if (source_row + 1) % task["progress_every"] == 0:
                print(json.dumps({"source": source.name, "processed": source_row + 1,
                                  "total": task["count"], "valid": counters["valid"],
                                  "seconds": round(time.perf_counter() - started, 2)}), flush=True)
    image.flush()
    tabular.flush()
    if counters["rows"] != task["count"]:
        raise RuntimeError(f"Source row count changed: {source}, {counters['rows']} != {task['count']}")
    return {"source_file": source.name, "source_sha256": sha256_file(source),
            "counters": dict(counters), "metadata_property_rows": dict(metadata),
            "reference_checks": references, "seconds": time.perf_counter() - started,
            "index_shard": task["index_shard"]}


def build_splits_and_queries(database: Path, index_shards: list[Path], interim: Path,
                            seed: int = 42) -> dict[str, Any]:
    """Merge provenance on disk, split by player, remove cross-split duplicates."""
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    declarations = []
    integer_fields = {"row_idx", "row_index", "source_row", "rank_idx", "valid", "num_moves",
                      "pass_count", "setup_count", "has_rank_metadata", "reference_checked"}
    for field in INDEX_FIELDS:
        declarations.append(f"{field} {'INTEGER' if field in integer_fields else 'TEXT'}")
    connection.execute("CREATE TABLE records (" + ",".join(declarations) + ")")
    insert = "INSERT INTO records VALUES (" + ",".join("?" for _ in INDEX_FIELDS) + ")"
    for shard in index_shards:
        with shard.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            batch = []
            for row in reader:
                batch.append(tuple(int(row[f]) if f in integer_fields else row[f] for f in INDEX_FIELDS))
                if len(batch) >= 10_000:
                    connection.executemany(insert, batch)
                    batch.clear()
            if batch:
                connection.executemany(insert, batch)
        connection.commit()
    connection.execute("CREATE UNIQUE INDEX records_row_idx ON records(row_idx)")
    connection.execute("CREATE INDEX records_player ON records(player_id)")
    connection.execute("CREATE TABLE player_splits(player_id TEXT PRIMARY KEY, split TEXT, ranks INTEGER)")
    strata: dict[int, list[tuple[bytes, str, int]]] = defaultdict(list)
    for player, rank_idx, n_ranks in connection.execute(
            "SELECT player_id, MIN(rank_idx), COUNT(DISTINCT rank_idx) FROM records "
            "WHERE player_id != '' AND rank_idx >= 0 GROUP BY player_id"):
        order = hashlib.sha256(f"{seed}:{player}".encode()).digest()
        strata[rank_idx].append((order, player, n_ranks))
    assignments = []
    for rank_idx in sorted(strata):
        players = sorted(strata[rank_idx])
        first, second = int(len(players) * 0.8), int(len(players) * 0.9)
        # A stratum with one player must not produce an empty training stratum.
        first = max(1, first) if players else 0
        second = max(first, second)
        assignments.extend((player, "train" if i < first else "dev" if i < second else "lockbox", ranks)
                           for i, (_, player, ranks) in enumerate(players))
    connection.executemany("INSERT INTO player_splits VALUES (?,?,?)", assignments)
    del assignments, strata
    connection.execute("UPDATE records SET split=COALESCE((SELECT split FROM player_splits p "
                       "WHERE p.player_id=records.player_id),'unassigned')")
    connection.execute("CREATE INDEX records_d4 ON records(d4_hash) WHERE valid=1")
    connection.execute("CREATE TEMP TABLE holdout_hashes AS SELECT d4_hash, "
                       "MAX(CASE split WHEN 'lockbox' THEN 2 WHEN 'dev' THEN 1 ELSE 0 END) AS priority "
                       "FROM records WHERE valid=1 AND d4_hash!='' GROUP BY d4_hash")
    connection.execute("CREATE UNIQUE INDEX holdout_hash_idx ON holdout_hashes(d4_hash)")
    connection.execute("UPDATE records SET excluded_reason='cross_split_d4_duplicate' WHERE valid=1 AND "
                       "((split='train' AND (SELECT priority FROM holdout_hashes h WHERE h.d4_hash=records.d4_hash)>0) "
                       "OR (split='dev' AND (SELECT priority FROM holdout_hashes h WHERE h.d4_hash=records.d4_hash)=2))")
    # Remove repeated copies of the same player's same-color game within a split.
    connection.execute("CREATE TEMP TABLE retained_games AS SELECT MIN(row_idx) AS row_idx FROM records "
                       "WHERE valid=1 AND excluded_reason='' GROUP BY d4_hash,player_id,rank_idx,color,split")
    connection.execute("CREATE UNIQUE INDEX retained_idx ON retained_games(row_idx)")
    connection.execute("UPDATE records SET excluded_reason='duplicate_same_player_game_view' "
                       "WHERE valid=1 AND excluded_reason='' AND row_idx NOT IN (SELECT row_idx FROM retained_games)")
    connection.commit()
    summary: dict[str, Any] = {}
    summary["split_rows"] = [dict(zip(("split", "rows", "valid", "eligible"), row)) for row in connection.execute(
        "SELECT split,COUNT(*),SUM(valid),SUM(valid=1 AND excluded_reason='') FROM records GROUP BY split")]
    summary["split_players"] = dict(connection.execute("SELECT split,COUNT(*) FROM player_splits GROUP BY split"))
    summary["cross_rank_players"] = connection.execute("SELECT COUNT(*) FROM player_splits WHERE ranks>1").fetchone()[0]
    summary["excluded_reasons"] = dict(connection.execute(
        "SELECT excluded_reason,COUNT(*) FROM records WHERE excluded_reason!='' GROUP BY excluded_reason"))
    for column in ("game_hash", "d4_hash"):
        summary[column + "_duplicate_groups"] = connection.execute(
            f"SELECT COUNT(*) FROM (SELECT {column} FROM records WHERE valid=1 AND {column}!='' "
            f"GROUP BY {column} HAVING COUNT(*)>1)").fetchone()[0]
    summary["query_counts"] = {}
    for split in ("dev", "lockbox"):
        query_count, eligible_players = 0, 0
        rank_counts: Counter[str] = Counter()
        query_path = interim / f"{split}_queries.jsonl"
        grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
        for player, rank_idx, row_idx in connection.execute(
                "SELECT player_id,rank_idx,MIN(row_idx) FROM records WHERE split=? AND valid=1 "
                "AND excluded_reason='' GROUP BY player_id,rank_idx,d4_hash "
                "ORDER BY player_id,rank_idx,MIN(row_idx)", (split,)):
            grouped[(player, rank_idx)].append(row_idx)
        with query_path.open("w", encoding="utf-8") as handle:
            for (player, rank_idx), indices in sorted(grouped.items()):
                if len(indices) < 5:
                    continue
                eligible_players += 1
                order = np.random.default_rng(int.from_bytes(hashlib.sha256(
                    f"{seed}:{split}:{player}:{rank_idx}".encode()).digest()[:8], "little")).permutation(indices).tolist()
                # Nested 5/10/20 queries provide a fixed size curve. No row repeats
                # within any query; query dependence is explicit in the manifest.
                for num_games in (5, 10, 20):
                    if len(order) < num_games:
                        continue
                    question = f"{split}_{query_count:07d}_n{num_games}"
                    record = {"question_id": question, "query_id": question,
                              "player_id": player, "rank": LEVELS[rank_idx], "rank_idx": rank_idx,
                              "num_games": num_games, "row_indices": order[:num_games],
                              "queries_for_same_player_are_dependent": True}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    query_count += 1
                    rank_counts[LEVELS[rank_idx]] += 1
        summary["query_counts"][split] = {"queries": query_count, "players": eligible_players,
                                             "rank_counts": dict(rank_counts),
                                             "missing_query_classes": [r for r in LEVELS if not rank_counts[r]]}
    with (interim / "index.csv").open("w", encoding="utf-8", newline="") as handle, \
            (interim / "errors.csv").open("w", encoding="utf-8", newline="") as errors:
        writer = csv.writer(handle)
        writer.writerow(INDEX_FIELDS)
        error_writer = csv.writer(errors)
        error_writer.writerow(INDEX_FIELDS)
        for row in connection.execute("SELECT " + ",".join(INDEX_FIELDS) + " FROM records ORDER BY row_idx"):
            writer.writerow(row)
            if not row[INDEX_FIELDS.index("valid")]:
                error_writer.writerow(row)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    return summary


def prepare_data(training_dir: str | Path, output_dir: str | Path, workers: int = 4,
                 seed: int = 42, limit_per_file: int | None = None,
                 reference_per_file: int = 20, overwrite: bool = False,
                 progress_every: int = 10_000) -> dict[str, Any]:
    """Run source-file tasks with spawn. This function never modifies training CSVs."""
    started = time.perf_counter()
    training, output = Path(training_dir).resolve(), Path(output_dir).resolve()
    if workers < 1 or reference_per_file < 0 or progress_every < 1:
        raise ValueError("workers/progress must be positive; reference count must be nonnegative")
    if limit_per_file is not None and limit_per_file < 1:
        raise ValueError("limit_per_file must be positive")
    sources = [(rank, training / f"train_{rank}.csv") for rank in LEVELS]
    missing = [str(path) for _, path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing training sources: " + ", ".join(missing))
    interim, processed = output / "interim", output / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    features_path, tabular_path = processed / "features.npy", processed / "tabular.npy"
    database = interim / "index.sqlite"
    managed = [features_path, tabular_path, database, interim / "index.csv", interim / "audit.json",
               interim / "dev_queries.jsonl", interim / "lockbox_queries.jsonl"]
    existing = [str(path) for path in managed if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Prepared outputs exist; use a new output directory or --overwrite: " + ", ".join(existing))
    if overwrite:
        # Only explicit generated files, never recursively delete a directory.
        for path in (database, Path(str(database) + "-wal"), Path(str(database) + "-shm")):
            if path.exists():
                path.unlink()
    counts = [_count_source(path, limit_per_file) for _, path in sources]
    total = sum(counts)
    if not total:
        raise ValueError("All source files are empty")
    print(json.dumps({"stage": "allocate", "rows": total, "workers": min(workers, len(sources)),
                      "feature_bytes": total * 18 * 19 * 19, "tabular_features": N_FEATURES}), flush=True)
    image = np.lib.format.open_memmap(features_path, mode="w+", dtype=np.uint8, shape=(total, 18, 19, 19))
    table = np.lib.format.open_memmap(tabular_path, mode="w+", dtype=np.float32, shape=(total, N_FEATURES))
    image.flush()
    table.flush()
    del image, table
    (processed / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2) + "\n", encoding="utf-8")
    work = interim / ("prepare_work_" + time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    work.mkdir()
    tasks, offset = [], 0
    for (rank, source), count in zip(sources, counts):
        rng = np.random.default_rng(seed + RANK_TO_ID[rank])
        reference_rows = sorted(rng.choice(count, size=min(count, reference_per_file), replace=False).tolist()) if count else []
        tasks.append({"source": str(source), "rank": rank, "count": count, "offset": offset,
                      "features": str(features_path), "tabular": str(tabular_path),
                      "index_shard": str(work / f"index_{rank}.csv"),
                      "reference_rows": reference_rows, "progress_every": progress_every})
        offset += count
    source_results = []
    if workers == 1:
        source_results = [_source_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_source_worker, task): task["rank"] for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                source_results.append(result)
                print(json.dumps({"stage": "source_complete", "source": result["source_file"],
                                  "seconds": round(result["seconds"], 2),
                                  "counters": result["counters"]}), flush=True)
    print(json.dumps({"stage": "merge_splits_and_queries", "rows": total}), flush=True)
    split_summary = build_splits_and_queries(database, [Path(task["index_shard"]) for task in tasks], interim, seed)
    combined: Counter[str] = Counter()
    checks = []
    for result in source_results:
        combined.update(result["counters"])
        checks.extend(result["reference_checks"])
    mismatches = [check for check in checks if check.get("reference_mismatch")]
    (interim / "replay_reference_checks.json").write_text(json.dumps({
        "scope": "Each selected valid game: every main-line move board and capture count; all 8 snapshots.",
        "selected_rows": len(checks), "successful_games": sum(check["ok"] for check in checks),
        "mismatches": len(mismatches), "checks": checks,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1, "seed": seed, "training_dir": str(training), "output_dir": str(output),
        "complete": not mismatches, "is_limited_pilot": limit_per_file is not None,
        "limit_per_file": limit_per_file, "input_rows": total, "counters": dict(combined),
        "denominator_policy": "Every counted input row remains in index and feature arrays; invalid/excluded rows are flagged, never silently removed.",
        "split_policy": "Player-disjoint 80/10/10 by stable seeded player ordering within min-rank stratum; cross-rank player stays together. Lockbox > dev > train duplicate priority.",
        "query_policy": "Fixed per-player same-rank nested 5/10/20-game queries without repeated rows inside a query. Same-player queries are dependent. Classes with insufficient eligible players are reported.",
        "image_schema": {"shape": [total, 18, 19, 19], "dtype": "uint8", "normalization": "divide all channels by 255",
                         "channels": "0..15: eight equidistant post-move target/opponent pairs; 16: target black=255 white=0; 17: round(255*min(num_moves,500)/500)",
                         "snapshot_policy": "rint(linspace(0,num_moves-1,8)); pass counts as a move; short games repeat actual snapshots, never fabricated positions"},
        "tabular_schema": {"shape": [total, N_FEATURES], "dtype": "float32", "feature_names": FEATURE_NAMES,
                           "forbidden_model_inputs": "player_id/game_id/rank/source/order/metadata values/error flags are not tabular or image features"},
        "replay_validation": {"reference_selected": len(checks), "reference_successful": sum(c["ok"] for c in checks),
                              "reference_mismatches": len(mismatches), "reference_per_source_requested": reference_per_file},
        "limitations": ["Original rules/komi are absent in move-only SGF; positional/situational superko and historical game adjudication are not enforced.",
                        "Suicide and immediate simple-ko recaptures are excluded as unsupported rules/replay cases, not declared universally illegal.",
                        "Only the SGF main line is replayed. Metadata property names are audited; identity/rank-bearing values never enter model features.",
                        "Invalid and excluded rows retain zero or unused features; consumers must filter valid=1 and excluded_reason=''.",
                        "Lockbox manifest is generated for future one-time evaluation; this run's trainers must not read it."],
        "split_summary": split_summary, "source_results": source_results,
        "elapsed_seconds_before_hashing": time.perf_counter() - started,
        "artifact_sha256": {},
    }
    for path in (features_path, tabular_path, processed / "feature_names.json", interim / "index.csv",
                 interim / "dev_queries.jsonl", interim / "lockbox_queries.jsonl"):
        audit["artifact_sha256"][str(path.relative_to(output)).replace("\\", "/")] = sha256_file(path)
    audit["elapsed_seconds"] = time.perf_counter() - started
    (interim / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if mismatches:
        raise RuntimeError("Custom replay disagrees with sgfmill; audit saved. Do not train until resolved.")
    print(json.dumps({"stage": "complete", "rows": total, "valid": combined["valid"],
                      "seconds": round(audit["elapsed_seconds"], 2), "split_summary": split_summary}), flush=True)
    return audit
