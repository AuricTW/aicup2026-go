"""Attach conservative family keys to derived metadata; never change raw games.

Only complete 100-ply screen memberships are accepted. Original canonical hashes
remain intact. Applying the same version again is idempotent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    processed = ROOT/"data/processed"
    source = processed/"near_duplicate_groups.jsonl"
    mapping = {}
    families = []
    for line in source.read_text(encoding="utf-8").splitlines():
        group = json.loads(line)
        if group["kind"] != "d4_prefix_screen" or group["threshold"] != 100:
            continue
        if group.get("members_truncated") or len(group["members"]) != group["rows"]:
            raise ValueError("Refusing incomplete family membership")
        key = "prefix100:" + group["prefix_hash"]
        families.append({"key":key, "rows":[m["row_id"] for m in group["members"]]})
        for member in group["members"]:
            row_id = member["row_id"]
            if row_id in mapping:
                raise ValueError("Overlapping family memberships need explicit union handling")
            mapping[row_id] = (key, member["canonical_hash"])
    conn = sqlite3.connect(processed/"audit.sqlite", timeout=60)
    try:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(games)")}
        if "split_group_hash" not in columns:
            conn.execute("ALTER TABLE games ADD COLUMN split_group_hash TEXT")
        for row_id, (key, canonical) in mapping.items():
            row = conn.execute("SELECT canonical_hash,parse_ok,split_group_hash FROM games WHERE row_id=?",(row_id,)).fetchone()
            if row is None or row[0] != canonical or row[1] != 1 or row[2] not in (None,key):
                raise ValueError(f"Family membership differs from source metadata: row_id={row_id}")
            conn.execute("UPDATE games SET split_group_hash=? WHERE row_id=?",(key,row_id))
        conn.execute("INSERT OR REPLACE INTO audit_meta VALUES (?,?)",("validation_grouping","full D4 canonical plus complete conservative 100-ply D4 prefix families"))
        conn.commit()
        tagged = conn.execute("SELECT COUNT(*) FROM games WHERE split_group_hash IS NOT NULL").fetchone()[0]
        count = conn.execute("SELECT COUNT(DISTINCT COALESCE(split_group_hash,canonical_hash)) FROM games WHERE parse_ok=1").fetchone()[0]
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    record = {"status":"complete", "policy":"Conservative D4 100-ply family grouping; original canonical hash and raw files untouched",
              "source_groups_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "families":families,"tagged_rows":tagged,"effective_unique_groups":count,
              "fully_parsed_rows":1000000,"expected_rows_removed_by_one_view_dedup":len(mapping)-len(families),
              "note":"Common prefixes may be different real games; this is a deliberately conservative validation policy."}
    (processed/"family_grouping.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    main()
