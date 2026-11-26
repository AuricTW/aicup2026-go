"""Prepare immutable training CSVs into audited disk-backed model inputs."""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # (repo) go_rank/ is in prev_system/
from go_rank.data_pipeline import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]      # (repo) comp2/
    parser.add_argument("--training-dir", type=Path, default=root / "data/raw/training")
    parser.add_argument("--output-dir", type=Path, default=root / "data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-file", type=int, default=None,
                        help="Pilot only: first N rows per source; never interpreted as a full audit")
    parser.add_argument("--reference-per-file", type=int, default=20,
                        help="Seeded samples per source compared with sgfmill on every move")
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--overwrite", action="store_true", help="Replace generated outputs only; raw CSVs are unchanged")
    args = parser.parse_args()
    prepare_data(**vars(args))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
