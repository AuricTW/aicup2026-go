"""Read-only second-stage sequence audit; no training rows are removed.

Screen 100/200-ply prefixes under D4 (including size, setup and original
colours), then find proper exact prefixes of full games in RAW orientation.
A shared prefix is a screening signal, not proof that two games are copies.
Different games can share an opening. D4 truncations with fewer than 100
plies, non-prefix near matches and omitted/inserted moves are not exhaustively
detected. Raw-orientation exact truncation counts are exhaustive within the
valid positive-length sequences in this input, but records show only the
nearest shorter prefix and count all matching shorter variants.

The input SQLite connection is mode=ro and query_only. Large sorting/grouping
work happens in a separate near_duplicates.sqlite, with temp data on disk.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import struct
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PASS = 32767
WHITE = 32768
THRESHOLDS = (100, 200)
SCHEMA = "goid-near-duplicate-screen-v1"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


@lru_cache(maxsize=26)
def transforms(size):
    points = np.arange(size * size, dtype=np.uint16)
    row, col = points // size, points % size
    result = []
    for reflected in (False, True):
        rr, cc = row, size - 1 - col if reflected else col
        for _ in range(4):
            result.append(np.concatenate(((rr * size + cc).astype(np.uint16), np.asarray([PASS], dtype=np.uint16))))
            rr, cc = cc, size - 1 - rr
    return np.asarray(result)


def canonical_prefix_hashes(size, setup, moves):
    """Return threshold/hash pairs. Transform coordinates, never colours."""
    table = transforms(size)
    setup_points = setup & PASS
    transformed_setup = table[:, setup_points] | (setup & WHITE)[None, :]
    transformed_setup.sort(axis=1)
    prefix = b"goid-prefix-d4-v1\0" + struct.pack("<BH", size, len(setup))
    selected = moves[:max(THRESHOLDS)]
    points = selected & PASS
    points = np.where(points == PASS, size * size, points)
    transformed_moves = table[:, points] | (selected & WHITE)[None, :]
    setup_bytes = [transformed_setup[i].astype("<u2", copy=False).tobytes() for i in range(8)]
    output = []
    for threshold in THRESHOLDS:
        if len(moves) < threshold:
            continue
        header = prefix + struct.pack("<H", threshold)
        canonical = min(header + setup_bytes[i] + transformed_moves[i, :threshold].astype("<u2", copy=False).tobytes()
                        for i in range(8))
        output.append((threshold, hashlib.sha256(canonical).hexdigest()))
    return output


def create_derived(conn):
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-32768")
    conn.executescript("""
      CREATE TABLE sequences(
        row_id INTEGER PRIMARY KEY, player_id TEXT NOT NULL, board_size INTEGER NOT NULL,
        canonical_hash TEXT NOT NULL, setup_sequence BLOB NOT NULL,
        move_sequence BLOB NOT NULL, plies INTEGER NOT NULL
      );
      CREATE TABLE prefixes(
        threshold INTEGER NOT NULL, prefix_hash TEXT NOT NULL, row_id INTEGER NOT NULL,
        PRIMARY KEY(threshold,prefix_hash,row_id)
      ) WITHOUT ROWID;
    """)


def prefix_groups(conn, output, max_members):
    summary = {str(t): {"groups_with_multiple_full_games": 0, "groups_cross_player": 0,
                         "rows_in_groups": 0, "maximum_cluster_rows": 0,
                         "maximum_distinct_full_games": 0} for t in THRESHOLDS}
    query = """SELECT p.threshold,p.prefix_hash,COUNT(*),COUNT(DISTINCT s.canonical_hash),
               COUNT(DISTINCT s.player_id),MIN(s.plies),MAX(s.plies)
       FROM prefixes p JOIN sequences s ON s.row_id=p.row_id
       GROUP BY p.threshold,p.prefix_hash HAVING COUNT(DISTINCT s.canonical_hash)>1
       ORDER BY p.threshold,p.prefix_hash"""
    for threshold, key, rows, variants, players, shortest, longest in conn.execute(query):
        stats = summary[str(threshold)]
        stats["groups_with_multiple_full_games"] += 1
        stats["groups_cross_player"] += int(players > 1)
        stats["rows_in_groups"] += rows
        stats["maximum_cluster_rows"] = max(stats["maximum_cluster_rows"], rows)
        stats["maximum_distinct_full_games"] = max(stats["maximum_distinct_full_games"], variants)
        members = list(conn.execute("""SELECT s.row_id,s.player_id,s.canonical_hash,s.plies
            FROM prefixes p JOIN sequences s ON s.row_id=p.row_id
            WHERE p.threshold=? AND p.prefix_hash=? ORDER BY s.row_id LIMIT ?""", (threshold, key, max_members)))
        output.write(json.dumps(dict(kind="d4_prefix_screen", threshold=threshold, prefix_hash=key,
            rows=rows, distinct_full_canonical_games=variants, distinct_players=players,
            minimum_plies=shortest, maximum_plies=longest,
            members=[dict(row_id=r, player_id=p, canonical_hash=h, plies=n) for r, p, h, n in members],
            members_truncated=rows > len(members)), ensure_ascii=False) + "\n")
    for threshold in THRESHOLDS:
        summary[str(threshold)]["eligible_rows"] = conn.execute(
            "SELECT COUNT(*) FROM prefixes WHERE threshold=?", (threshold,)).fetchone()[0]
    return summary


def raw_truncations(conn, output):
    """Lexicographic BLOB order puts each proper prefix before descendants.

    A stack contains all shorter unique sequences that prefix the current
    sequence. Thus all variant and row-pair counts are exact, while one
    compact representative record is emitted per longer unique sequence.
    """
    query = """SELECT board_size,setup_sequence,move_sequence,COUNT(*),MIN(row_id),
               MIN(canonical_hash),COUNT(DISTINCT player_id),MIN(player_id)
        FROM sequences WHERE plies>0
        GROUP BY board_size,setup_sequence,move_sequence
        ORDER BY board_size,setup_sequence,move_sequence"""
    stats = dict(distinct_short_to_long_relations=0, ordered_row_pairs=0,
                 longer_variants_with_a_shorter_prefix=0, cross_player_row_pairs=0,
                 maximum_matching_shorter_variants=0, minimum_prefix_plies=None)
    previous_setup = None
    stack = []
    # Need player multiplicities only for the comparatively rare related
    # variants; aggregate duplicate copies are never loaded globally.
    for size, setup, moves, copies, row_id, canonical, players, representative_player in conn.execute(query):
        current_setup = (size, setup)
        if current_setup != previous_setup:
            stack.clear()
            previous_setup = current_setup
        while stack and not moves.startswith(stack[-1]["moves"]):
            stack.pop()
        if stack:
            stats["distinct_short_to_long_relations"] += len(stack)
            stats["ordered_row_pairs"] += copies * sum(item["copies"] for item in stack)
            stats["longer_variants_with_a_shorter_prefix"] += 1
            stats["maximum_matching_shorter_variants"] = max(stats["maximum_matching_shorter_variants"], len(stack))
            shortest = len(stack[0]["moves"]) // 2
            stats["minimum_prefix_plies"] = shortest if stats["minimum_prefix_plies"] is None else min(shortest, stats["minimum_prefix_plies"])
            # Count cross-player row pairs exactly without claiming separate
            # anonymous players imply independent physical games.
            current_players = dict(conn.execute("""SELECT player_id,COUNT(*) FROM sequences
                WHERE board_size=? AND setup_sequence=? AND move_sequence=? GROUP BY player_id""", (size, setup, moves)))
            cross_pairs = 0
            for item in stack:
                if item.get("players") is None:
                    item["players"] = dict(conn.execute("""SELECT player_id,COUNT(*) FROM sequences
                        WHERE board_size=? AND setup_sequence=? AND move_sequence=? GROUP BY player_id""", (size, setup, item["moves"])))
                same_player_pairs = sum(count * current_players.get(player, 0) for player, count in item["players"].items())
                cross_pairs += item["copies"] * copies - same_player_pairs
            stats["cross_player_row_pairs"] += cross_pairs
            nearest = stack[-1]
            output.write(json.dumps(dict(kind="raw_orientation_exact_truncation",
                board_size=size, shorter_representative_row_id=nearest["row_id"],
                longer_representative_row_id=row_id, shorter_plies=len(nearest["moves"]) // 2,
                longer_plies=len(moves) // 2, shorter_copies=nearest["copies"], longer_copies=copies,
                matching_shorter_variants=len(stack), cross_player_row_pairs_for_longer=cross_pairs,
                shorter_canonical_hash=nearest["canonical"], longer_canonical_hash=canonical), ensure_ascii=False) + "\n")
        stack.append(dict(moves=moves, copies=copies, row_id=row_id, canonical=canonical, players=None))
    return stats


def audit(args):
    started = time.perf_counter()
    source = sqlite3.connect(f"file:{args.input.resolve().as_posix()}?mode=ro", uri=True, timeout=60)
    source.execute("PRAGMA query_only=ON")
    totals = {"unparsed" if k is None else "ok" if k else "failed": count
              for k, count in source.execute("SELECT parse_ok,COUNT(*) FROM games GROUP BY parse_ok")}
    if totals.get("unparsed", 0):
        raise ValueError(f"Full replay must finish first; {totals['unparsed']} rows are unparsed")
    args.output.mkdir(parents=True, exist_ok=True)
    derived_path = args.output / "near_duplicates.sqlite"
    if derived_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior derived audit: {derived_path}")
    if derived_path.resolve() == args.input.resolve():
        raise ValueError("Derived database must be separate from input")
    derived = sqlite3.connect(derived_path)
    create_derived(derived)
    group_path = args.output / "near_duplicate_groups.jsonl"
    counter = Counter()
    first_colours = Counter()
    examples = []
    sequence_batch, prefix_batch = [], []
    last_progress = 0
    with group_path.open("x", encoding="utf-8") as output:
        query = """SELECT row_id,player_id,board_size,canonical_hash,setup_sequence,move_sequence,
                   color,mainline_moves,target_moves,passes,target_passes
                   FROM games WHERE parse_ok=1 ORDER BY row_id"""
        # This is the only streaming SELECT of full game sequences from input.
        for row in source.execute(query):
            row_id, player, size, canonical, setup_blob, moves_blob, colour, expected_moves, expected_target, expected_passes, expected_target_passes = row
            counter["scanned_rows"] += 1
            errors = []
            if (not isinstance(size, int) or not 2 <= size <= 26 or not canonical
                    or setup_blob is None or moves_blob is None or len(setup_blob) % 2 or len(moves_blob) % 2):
                errors.append("invalid_sequence_header_or_encoding")
            else:
                setup = np.frombuffer(setup_blob, dtype="<u2")
                moves = np.frombuffer(moves_blob, dtype="<u2")
                move_points = moves & PASS
                setup_points = setup & PASS
                if np.any((move_points >= size * size) & (move_points != PASS)) or np.any(setup_points >= size * size):
                    errors.append("point_outside_board")
                if len(np.unique(setup_points)) != len(setup_points):
                    errors.append("duplicate_or_mixed_colour_setup_point")
            if errors:
                counter["invalid_encoding_rows"] += 1
                output.write(json.dumps(dict(kind="sequence_consistency_error", row_id=row_id, errors=errors)) + "\n")
                continue
            white = (moves & WHITE) != 0
            target = white if colour == "W" else ~white
            if colour not in ("B", "W"):
                errors.append("invalid_target_color")
            expected = (expected_moves, expected_target, expected_passes, expected_target_passes)
            observed = (len(moves), int(target.sum()), int((move_points == PASS).sum()), int(((move_points == PASS) & target).sum()))
            for field, wanted, actual in zip(("mainline_moves", "target_moves", "passes", "target_passes"), expected, observed):
                if wanted != actual:
                    errors.append(f"{field}: stored={wanted}, observed={actual}")
            first = "none" if not len(moves) else "W" if white[0] else "B"
            first_colours[f"target_{colour}/first_{first}"] += 1
            consecutive = int((white[1:] == white[:-1]).sum())
            if consecutive:
                counter["rows_with_consecutive_same_colour"] += 1
                counter["consecutive_same_colour_transitions"] += consecutive
                if len(examples) < args.max_members:
                    examples.append(dict(row_id=row_id, transitions=consecutive, first_move_color=first))
            if not len(moves):
                counter["zero_move_rows"] += 1
            if errors:
                counter["count_mismatch_rows"] += 1
                output.write(json.dumps(dict(kind="sequence_consistency_error", row_id=row_id, errors=errors)) + "\n")
            # Mismatches remain in the derived screen; none is silently dropped.
            sequence_batch.append((row_id, player, size, canonical, setup_blob, moves_blob, len(moves)))
            prefix_batch.extend((threshold, key, row_id) for threshold, key in canonical_prefix_hashes(size, setup, moves))
            if len(sequence_batch) >= args.batch_size:
                derived.executemany("INSERT INTO sequences VALUES(?,?,?,?,?,?,?)", sequence_batch)
                derived.executemany("INSERT INTO prefixes VALUES(?,?,?)", prefix_batch)
                derived.commit()
                sequence_batch.clear()
                prefix_batch.clear()
            if counter["scanned_rows"] - last_progress >= 100000:
                last_progress = counter["scanned_rows"]
                print(json.dumps(dict(event="scan_progress", rows=last_progress, elapsed_seconds=time.perf_counter()-started)), flush=True)
        if sequence_batch:
            derived.executemany("INSERT INTO sequences VALUES(?,?,?,?,?,?,?)", sequence_batch)
            derived.executemany("INSERT INTO prefixes VALUES(?,?,?)", prefix_batch)
            derived.commit()
        print("input scan complete; grouping prefixes and exact truncations", flush=True)
        # The index also makes bounded per-group player multiplicity lookups cheap.
        derived.execute("CREATE INDEX sequences_raw ON sequences(board_size,setup_sequence,move_sequence)")
        derived.commit()
        screen = prefix_groups(derived, output, args.max_members)
        truncations = raw_truncations(derived, output)
    source.close()
    derived.close()
    result = dict(schema=SCHEMA, status="complete", created_at=now(), input=str(args.input.resolve()),
        input_parse_counts=totals, derived_database=str(derived_path.resolve()), groups_file=str(group_path.resolve()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started, sequence_consistency=dict(counter),
        first_move_colours=dict(first_colours), consecutive_same_colour_examples=examples,
        d4_prefix_screen=screen, raw_orientation_exact_truncation=truncations,
        scope={"prefix_plies":list(THRESHOLDS), "prefix_includes":"board size, effective setup, colour-labelled moves and passes",
               "short_games":"excluded only from prefixes longer than their actual length",
               "exact_truncation":"positive-length full sequences, raw orientation, exact bytes, identical board/setup",
               "truncation_records":"nearest shorter representative per longer unique raw sequence; counts include all shorter prefixes",
               "not_exhaustive_for":"D4 truncations below 100 plies, edited/inserted/omitted moves, other near-duplicate notions",
               "interpretation":"screening evidence only; no automatic deduplication or removal"})
    path = args.output / "near_duplicate_summary.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def self_test():
    """Synthetic controls only; no competition database is accessed."""
    size = 19
    setup = np.asarray([35, WHITE | 76], dtype=np.uint16)
    moves = np.asarray([(WHITE if i % 2 else 0) | (i % 361) for i in range(210)], dtype=np.uint16)
    moves[10], moves[11] = PASS, WHITE | PASS
    original = canonical_prefix_hashes(size, setup, moves)
    for table in transforms(size):
        transformed_setup = table[setup & PASS] | (setup & WHITE)
        transformed_moves = table[np.where((moves & PASS) == PASS, size * size, moves & PASS)] | (moves & WHITE)
        assert canonical_prefix_hashes(size, transformed_setup, transformed_moves) == original
    assert canonical_prefix_hashes(size, setup, moves ^ WHITE) != original
    assert canonical_prefix_hashes(size, setup, moves[:99]) == []
    assert len(canonical_prefix_hashes(size, setup, moves[:100])) == 1
    import io
    conn = sqlite3.connect(":memory:")
    create_derived(conn)
    for row_id, player, sequence in ((0,"A",b"aa"),(1,"A",b"aa"),(2,"B",b"aabb"),(3,"C",b"aabbcc"),(4,"A",b"aadd")):
        conn.execute("INSERT INTO sequences VALUES(?,?,?,?,?,?,?)", (row_id, player, 19, str(row_id), b"", sequence, len(sequence)//2))
    output = io.StringIO()
    actual = raw_truncations(conn, output)
    assert actual["distinct_short_to_long_relations"] == 4, actual
    assert actual["ordered_row_pairs"] == 7, actual
    assert actual["cross_player_row_pairs"] == 5, actual
    assert actual["longer_variants_with_a_shorter_prefix"] == 3, actual
    conn.close()
    print(json.dumps(dict(status="synthetic_self_test_passed", d4_transforms=8, raw_truncation_counts=actual)))


def review_prefix_groups(args):
    """Bounded full-sequence D4 comparison of already-screened members.

    This reads only the row IDs listed in the previous groups JSONL. It does
    not scan the million-row input again and never changes its hashes.
    """
    started = time.perf_counter()
    group_path = args.output / "near_duplicate_groups.jsonl"
    summary_path = args.output / "near_duplicate_summary.json"
    groups = [json.loads(line) for line in group_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = [g for g in groups if g.get("kind") == "d4_prefix_screen"]
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    members = {m["row_id"]: m for g in groups for m in g["members"]}
    required_prefix = {}
    for g in groups:
        for pair in itertools.combinations(sorted(m["row_id"] for m in g["members"]), 2):
            required_prefix[pair] = max(required_prefix.get(pair, 0), g["threshold"])
    source = sqlite3.connect(f"file:{args.input.resolve().as_posix()}?mode=ro", uri=True, timeout=60)
    source.execute("PRAGMA query_only=ON")
    rows = {}
    for row_id in sorted(members):
        row = source.execute("""SELECT board_size,setup_sequence,move_sequence,player_id,color,canonical_hash
                    FROM games WHERE row_id=? AND parse_ok=1""", (row_id,)).fetchone()
        if row is None:
            raise ValueError(f"Screened row {row_id} is missing or is no longer parse_ok")
        size, setup, moves, player, colour, canonical = row
        if canonical != members[row_id]["canonical_hash"]:
            raise ValueError(f"Canonical hash changed for screened row {row_id}")
        rows[row_id] = (size, np.frombuffer(setup, dtype="<u2"), np.frombuffer(moves, dtype="<u2"), player, colour)
    source.close()
    records = []
    for (a, b), threshold in sorted(required_prefix.items()):
        size, setup_a, moves_a, player_a, colour_a = rows[a]
        size_b, setup_b, moves_b, player_b, colour_b = rows[b]
        if size != size_b:
            raise ValueError(f"Screened pair {a}/{b} has different board sizes")
        best, orientation = -1, None
        for index, table in enumerate(transforms(size)):
            transformed_setup = np.sort(table[setup_b & PASS] | (setup_b & WHITE))
            if not np.array_equal(transformed_setup, np.sort(setup_a)):
                continue
            transformed_moves = table[np.where((moves_b & PASS) == PASS, size * size, moves_b & PASS)] | (moves_b & WHITE)
            length = min(len(moves_a), len(transformed_moves))
            differences = np.flatnonzero(moves_a[:length] != transformed_moves[:length])
            common = int(differences[0]) if len(differences) else length
            if common > best:
                best, orientation = common, index
        if best < threshold:
            raise ValueError(f"Pair {a}/{b} failed its {threshold}-ply D4 prefix screen; observed {best}")
        records.append(dict(row_a=a, row_b=b, player_a=player_a, player_b=player_b,
            target_color_a=colour_a, target_color_b=colour_b, plies_a=len(moves_a), plies_b=len(moves_b),
            longest_common_prefix_d4_plies=best, best_transform=orientation,
            exact_d4_truncation=best == min(len(moves_a), len(moves_b)) and len(moves_a) != len(moves_b)))
    total_ok = source_summary["input_parse_counts"].get("ok", 0)
    all_at_least_100 = source_summary["d4_prefix_screen"]["100"]["eligible_rows"] == total_ok
    all_groups_complete = all(not g["members_truncated"] for g in groups if g["threshold"] == 100)
    result = dict(kind="bounded_eight_orientation_pair_review", status="complete", created_at=now(),
        input=str(args.input.resolve()), input_groups=str(group_path.resolve()),
        input_groups_sha256=hashlib.sha256(group_path.read_bytes()).hexdigest(),
        input_summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        unique_rows=len(rows), pairs_reviewed=len(records),
        exact_d4_truncation_pairs=sum(r["exact_d4_truncation"] for r in records),
        elapsed_seconds=time.perf_counter()-started,
        coverage=dict(all_successful_input_rows_at_least_100_plies=all_at_least_100,
            prefix100_member_lists_complete=all_groups_complete,
            exact_d4_truncation_candidate_coverage_complete=all_at_least_100 and all_groups_complete,
            limitation="Other edited, non-prefix or approximate game similarities are not exhaustively covered"),
        algorithm="For each screened pair, transform the second complete sequence under all eight D4 orientations, require identical effective setup, retain original colours/pass, maximise equal leading move codes. Proper truncation requires common prefix equal to the complete shorter sequence length.",
        pairs=records)
    path = args.output / "near_duplicate_pair_review.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"data/processed/audit.sqlite")
    parser.add_argument("--output", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--max-members", type=int, default=100)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--review-prefix-groups", action="store_true",
                        help="Review only existing screened rows under all eight D4 orientations; no full rescan")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_members < 1:
        parser.error("positive batch-size and max-members required")
    if args.self_test:
        self_test()
    elif args.review_prefix_groups:
        review_prefix_groups(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
