"""Streaming full-data index and bounded-process SGF replay/feature audit.

Run index first, replay the fixed 10k selection, then replay all remaining rows.
All raw inputs are read only. Feature row i always corresponds to games.row_id i.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import multiprocessing
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
HEADER = ["player_id", "game_id", "rank", "color", "sgf_content"]


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")
    return conn


def index(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    db_path = args.output / "audit.sqlite"
    if db_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit database: {db_path}")
    start = time.perf_counter()
    conn = connection(db_path)
    conn.executescript("""
      CREATE TABLE games (
        row_id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, source_row INTEGER NOT NULL,
        player_id TEXT NOT NULL, game_id TEXT NOT NULL, rank TEXT NOT NULL, color TEXT NOT NULL,
        sgf_content TEXT NOT NULL, raw_hash TEXT NOT NULL,
        parse_ok INTEGER, error TEXT, board_size INTEGER, mainline_moves INTEGER,
        target_moves INTEGER, passes INTEGER, target_passes INTEGER, setup_stones INTEGER,
        has_variations INTEGER, move_hash TEXT, canonical_hash TEXT,
        move_sequence BLOB, setup_sequence BLOB,
        UNIQUE(source_file, source_row)
      );
      CREATE TABLE calibration_selection (row_id INTEGER PRIMARY KEY, level TEXT NOT NULL, priority TEXT NOT NULL);
      CREATE TABLE audit_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.execute("INSERT INTO audit_meta VALUES (?,?)", ("index_status", "running"))
    total = 0
    batches = []
    heaps = {level: [] for level in LEVELS}
    file_stats = []
    for level in LEVELS:
        path = args.raw_dir / f"train_{level}.csv"
        rows = 0
        colors = Counter()
        raw_bytes = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != HEADER:
                raise ValueError(f"Unexpected header in {path}: {reader.fieldnames}")
            for number, row in enumerate(reader, 1):
                if set(row) != set(HEADER) or any(row[x] is None for x in HEADER):
                    raise ValueError(f"Malformed CSV row at {path}:{number}")
                if row["rank"] != level or row["color"] not in ("B", "W") or not row["player_id"] or not row["game_id"]:
                    raise ValueError(f"Invalid row labels at {path}:{number}")
                sgf = row["sgf_content"]
                if not sgf:
                    raise ValueError(f"Empty SGF at {path}:{number}")
                blob = sgf.encode("utf-8")
                raw_hash = hashlib.sha256(blob).hexdigest()
                key = f"goid-calibration-v1|{path.name}|{number}|{row['player_id']}|{raw_hash}"
                priority_hex = hashlib.sha256(key.encode()).hexdigest()
                priority = int(priority_hex, 16)
                item = (-priority, total, priority_hex)
                heap = heaps[level]
                if len(heap) < args.sample_per_level:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
                batches.append((total, path.name, number, row["player_id"], row["game_id"], level,
                                row["color"], sgf, raw_hash))
                colors[row["color"]] += 1
                raw_bytes += len(blob)
                total += 1
                rows += 1
                if len(batches) >= 5000:
                    conn.executemany("INSERT INTO games(row_id,source_file,source_row,player_id,game_id,rank,color,sgf_content,raw_hash) VALUES (?,?,?,?,?,?,?,?,?)", batches)
                    conn.commit()
                    batches.clear()
            file_stats.append({"file": path.name, "rows": rows, "colors": dict(colors),
                               "sgf_utf8_bytes": raw_bytes, "file_bytes": path.stat().st_size})
            if rows != 100000:
                raise ValueError(f"Expected 100000 rows in {path.name}, found {rows}")
        print(f"indexed {path.name}: {rows} rows, cumulative {total}", flush=True)
    if batches:
        conn.executemany("INSERT INTO games(row_id,source_file,source_row,player_id,game_id,rank,color,sgf_content,raw_hash) VALUES (?,?,?,?,?,?,?,?,?)", batches)
    for level, heap in heaps.items():
        conn.executemany("INSERT INTO calibration_selection VALUES (?,?,?)",
                         [(row_id, level, priority) for _, row_id, priority in heap])
    conn.executescript("""
      CREATE INDEX games_player ON games(player_id);
      CREATE INDEX games_raw ON games(raw_hash);
      CREATE INDEX games_source_key ON games(source_file,game_id,color);
      INSERT OR REPLACE INTO audit_meta VALUES ('index_status','complete');
    """)
    conn.commit()
    elapsed = time.perf_counter() - start
    record = metadata_summary(conn)
    record.update(status="complete", kind="full_csv_metadata_audit", created_at=now(),
                  rows=total, elapsed_seconds=elapsed, rows_per_second=total/elapsed,
                  files=file_stats, selection_method="lowest SHA256 priorities, 1000 per rank, goid-calibration-v1",
                  calibration_rows=sum(len(x) for x in heaps.values()))
    save_json(args.output / "metadata_summary.json", record)
    sample = [dict(row_id=r[0], level=r[1], priority=r[2]) for r in conn.execute("SELECT * FROM calibration_selection ORDER BY row_id")]
    save_json(args.output / "calibration_selection.json", {"version": "goid-calibration-v1", "rows": sample})
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


def metadata_summary(conn: sqlite3.Connection) -> dict:
    counts = [r[0] for r in conn.execute("SELECT COUNT(*) FROM games GROUP BY player_id")]
    counts.sort()
    quantiles = {}
    for label, q in [("min", 0), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1)]:
        quantiles[label] = counts[round(q * (len(counts)-1))] if counts else None
    return {
        "unique_players": len(counts), "games_per_player": quantiles,
        "players_at_least_games": {str(n): sum(c >= n for c in counts) for n in [2,5,20,100,105,110,120]},
        "players_in_multiple_ranks": conn.execute("SELECT COUNT(*) FROM (SELECT player_id FROM games GROUP BY player_id HAVING COUNT(DISTINCT rank)>1)").fetchone()[0],
        "duplicate_raw_hash_groups": conn.execute("SELECT COUNT(*) FROM (SELECT raw_hash FROM games GROUP BY raw_hash HAVING COUNT(*)>1)").fetchone()[0],
        "duplicate_source_game_color_groups": conn.execute("SELECT COUNT(*) FROM (SELECT source_file,game_id,color FROM games GROUP BY source_file,game_id,color HAVING COUNT(*)>1)").fetchone()[0],
        "colors": dict(conn.execute("SELECT color,COUNT(*) FROM games GROUP BY color")),
        "players_per_rank": dict(conn.execute("SELECT rank,COUNT(DISTINCT player_id) FROM games GROUP BY rank")),
    }


def parse_batch(batch: list[tuple]) -> list[dict]:
    from goid.sgf_features import analyze_sgf
    results = []
    for row_id, sgf, color in batch:
        try:
            result = analyze_sgf(sgf, color, include_features=True)
            result.update(row_id=row_id, parse_ok=1, error=None)
        except Exception as exc:
            result = {"row_id": row_id, "parse_ok": 0, "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
    return results


def parse(args: argparse.Namespace) -> None:
    import numpy as np
    import psutil
    from goid.sgf_features import FEATURE_DIM, FEATURE_GROUPS, FEATURE_SCHEMA
    source_sha_at_start = hashlib.sha256((ROOT/"src/goid/sgf_features.py").read_bytes()).hexdigest()
    db_path = args.output / "audit.sqlite"
    conn = connection(db_path)
    if conn.execute("SELECT value FROM audit_meta WHERE key='index_status'").fetchone() != ("complete",):
        conn.close()
        raise ValueError("Metadata indexing must complete first")
    count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    pinned = conn.execute("SELECT value FROM audit_meta WHERE key='feature_parser_sha256'").fetchone()
    if pinned is None:
        old_schema = args.output / "feature_schema.json"
        if old_schema.exists():
            previous_sha = json.loads(old_schema.read_text(encoding="utf-8")).get("source_sha256")
            if previous_sha and previous_sha != source_sha_at_start:
                conn.close()
                raise ValueError("Existing feature source differs; rebuild in a new output directory")
        conn.execute("INSERT INTO audit_meta VALUES (?,?)", ("feature_parser_sha256", source_sha_at_start))
        conn.commit()
    elif pinned[0] != source_sha_at_start:
        conn.close()
        raise ValueError("Pinned parser differs; do not mix cached features, rebuild in a new output directory")
    feature_path = args.output / "features.npy"
    if feature_path.exists():
        features = np.load(feature_path, mmap_mode="r+")
        if features.shape != (count, FEATURE_DIM) or features.dtype != np.float32:
            conn.close()
            raise ValueError("Existing features have incompatible shape/dtype")
    else:
        features = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float32, shape=(count, FEATURE_DIM))
    save_json(args.output / "feature_schema.json", {"schema": FEATURE_SCHEMA, "dimension": FEATURE_DIM,
               "dtype": "float32", "row_mapping": "games.row_id zero based", "groups": list(FEATURE_GROUPS),
               "source_sha256": hashlib.sha256((ROOT/"src/goid/sgf_features.py").read_bytes()).hexdigest()})
    if args.reparse_selection:
        if args.selection != "calibration":
            conn.close()
            raise ValueError("Explicit reparse currently supports calibration only")
        conn.execute("UPDATE games SET parse_ok=NULL WHERE row_id IN (SELECT row_id FROM calibration_selection)")
        conn.commit()
    clause = "g.parse_ok IS NULL"
    if args.selection == "calibration":
        clause += " AND g.row_id IN (SELECT row_id FROM calibration_selection)"
    pending = conn.execute(f"SELECT COUNT(*) FROM games g WHERE {clause}").fetchone()[0]
    if pending == 0:
        print(json.dumps({"status":"no_op", "selection":args.selection,
                          "reason":"all selected rows already attempted; original run summaries retained"}), flush=True)
        conn.close()
        return
    cursor = conn.execute(f"SELECT g.row_id,g.sgf_content,g.color FROM games g WHERE {clause} ORDER BY g.row_id")
    def batches():
        while rows := cursor.fetchmany(args.batch_size):
            yield rows
    tasks = iter(batches())
    start = time.perf_counter()
    log_path = args.output / f"parse_{args.selection}_events.jsonl"
    completed = failed = 0
    peak_parent_rss = peak_tree_rss = 0
    next_progress = 1000
    update_sql = """UPDATE games SET parse_ok=?,error=?,board_size=?,mainline_moves=?,target_moves=?,passes=?,
       target_passes=?,setup_stones=?,has_variations=?,move_hash=?,canonical_hash=?,move_sequence=?,setup_sequence=? WHERE row_id=?"""
    fields = ["parse_ok","error","board_size","mainline_moves","target_moves","passes","target_passes",
              "setup_stones","has_variations","move_hash","canonical_hash","move_sequence","setup_sequence"]
    process = psutil.Process()
    with log_path.open("a", encoding="utf-8") as log, ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        log.write(json.dumps({"event":"start","at":now(),"pending":pending,"workers":args.workers,"batch_size":args.batch_size})+"\n")
        active = {}
        for _ in range(args.workers * 2):
            batch = next(tasks, None)
            if batch is not None:
                active[executor.submit(parse_batch, batch)] = len(batch)
        while active:
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                del active[future]
                results = future.result()
                updates = []
                for result in results:
                    row_id = result["row_id"]
                    if result["parse_ok"]:
                        vector = np.asarray(result["features"], dtype=np.float32)
                        if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
                            raise ValueError(f"Invalid feature vector at row_id={row_id}")
                        features[row_id] = vector
                    else:
                        features[row_id] = 0
                        failed += 1
                    updates.append(tuple(result.get(field) for field in fields)+(row_id,))
                    completed += 1
                # Commit the SQL success flag only after the corresponding
                # feature pages have been flushed. This matters on resumption.
                features.flush()
                conn.executemany(update_sql, updates)
                conn.commit()
                rss = process.memory_info().rss
                tree_rss = rss
                for child in process.children():
                    try:
                        tree_rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_parent_rss = max(peak_parent_rss, rss)
                peak_tree_rss = max(peak_tree_rss, tree_rss)
                if completed >= next_progress:
                    event = {"event":"progress","at":now(),"completed":completed,"pending_at_start":pending,
                             "failed":failed,"elapsed_seconds":time.perf_counter()-start,"tree_rss_bytes":tree_rss}
                    log.write(json.dumps(event)+"\n")
                    log.flush()
                    print(json.dumps(event), flush=True)
                    next_progress += 1000 if args.selection == "calibration" else 10000
                batch = next(tasks, None)
                if batch is not None:
                    active[executor.submit(parse_batch, batch)] = len(batch)
        features.flush()
        elapsed = time.perf_counter()-start
        record = {"kind":"measured_sgf_replay_and_feature_extraction","selection":args.selection,
                  "status":"complete","finished_at":now(),"attempted_this_run":completed,
                  "failed_this_run":failed,"elapsed_seconds":elapsed,"games_per_second":completed/elapsed if elapsed else None,
                  "workers":args.workers,"batch_size":args.batch_size,
                  "peak_parent_rss_bytes":peak_parent_rss,"peak_process_tree_rss_bytes":peak_tree_rss,
                  "feature_dimension":FEATURE_DIM,"source_sha256":source_sha_at_start,
                  "source_sha256_at_finish":hashlib.sha256((ROOT/"src/goid/sgf_features.py").read_bytes()).hexdigest()}
        record["source_unchanged_during_run"] = record["source_sha256"] == record["source_sha256_at_finish"]
        if not record["source_unchanged_during_run"]:
            record["status"] = "source_changed_requires_reparse"
        log.write(json.dumps({"event":"complete",**record})+"\n")
    save_json(args.output / f"parse_{args.selection}_summary.json", record)
    conn.execute("CREATE INDEX IF NOT EXISTS games_canonical ON games(canonical_hash)")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print(json.dumps(record, indent=2), flush=True)
    if not record["source_unchanged_during_run"]:
        raise RuntimeError("Parser source changed during extraction; do not use these results before reparse")


def summary(args: argparse.Namespace) -> None:
    conn = sqlite3.connect(f"file:{(args.output/'audit.sqlite').as_posix()}?mode=ro", uri=True)
    totals = {"unparsed" if r[0] is None else "ok" if r[0] else "failed": r[1]
              for r in conn.execute("SELECT parse_ok,COUNT(*) FROM games GROUP BY parse_ok")}
    result = {"created_at":now(),"kind":"full_training_data_audit","parse_counts":totals,**metadata_summary(conn)}
    if totals.get("ok"):
        result.update(
            board_sizes=dict(conn.execute("SELECT board_size,COUNT(*) FROM games WHERE parse_ok=1 GROUP BY board_size")),
            move_statistics=dict(zip(["min","mean","max"],conn.execute("SELECT MIN(mainline_moves),AVG(mainline_moves),MAX(mainline_moves) FROM games WHERE parse_ok=1").fetchone())),
            games_with_pass=conn.execute("SELECT COUNT(*) FROM games WHERE passes>0").fetchone()[0],
            games_with_setup=conn.execute("SELECT COUNT(*) FROM games WHERE setup_stones>0").fetchone()[0],
            games_with_variations=conn.execute("SELECT COUNT(*) FROM games WHERE has_variations=1").fetchone()[0],
            duplicate_move_hash_groups=conn.execute("SELECT COUNT(*) FROM (SELECT move_hash FROM games WHERE parse_ok=1 GROUP BY move_hash HAVING COUNT(*)>1)").fetchone()[0],
            duplicate_canonical_groups=conn.execute("SELECT COUNT(*) FROM (SELECT canonical_hash FROM games WHERE parse_ok=1 GROUP BY canonical_hash HAVING COUNT(*)>1)").fetchone()[0],
            canonical_groups_multiple_players=conn.execute("SELECT COUNT(*) FROM (SELECT canonical_hash FROM games WHERE parse_ok=1 GROUP BY canonical_hash HAVING COUNT(DISTINCT player_id)>1)").fetchone()[0],
            players_with_105_unique_games=conn.execute("SELECT COUNT(*) FROM (SELECT player_id FROM games WHERE parse_ok=1 GROUP BY player_id HAVING COUNT(DISTINCT canonical_hash)>=105)").fetchone()[0],
            players_with_120_unique_games=conn.execute("SELECT COUNT(*) FROM (SELECT player_id FROM games WHERE parse_ok=1 GROUP BY player_id HAVING COUNT(DISTINCT canonical_hash)>=120)").fetchone()[0],
        )
    result["error_types"] = list(conn.execute("SELECT error,COUNT(*) FROM games WHERE parse_ok=0 GROUP BY error ORDER BY COUNT(*) DESC"))
    result["status"] = "complete" if not totals.get("unparsed", 0) else "partial"
    result["legality_scope"] = "Parser-specific checks documented in sgf_features.py; no inferred rule-dependent superko guarantee"
    save_json(args.output / "audit_summary.json", result)
    conn.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["index","parse","summary"])
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"data/raw/training")
    parser.add_argument("--output", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--sample-per-level", type=int, default=1000)
    parser.add_argument("--selection", choices=["calibration","all"], default="calibration")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reparse-selection", action="store_true", help="Explicitly reparse calibration rows after preserving old evidence")
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.sample_per_level < 1:
        parser.error("positive workers, batch-size and sample-per-level required")
    {"index":index,"parse":parse,"summary":summary}[args.stage](args)


if __name__ == "__main__":
    main()
