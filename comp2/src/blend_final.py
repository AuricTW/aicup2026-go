"""Final submission file = geometric mean of the 91 member posteriors -> expected-score decoding (stage R1 of README).
usage: blend_final.py [--parts DIR] [--out OUT.csv]
Prints the SHA-256 of the CSV; the final file of the competition has 1ADC681D...04CA4A (README).
"""
import argparse, hashlib, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from comp2_paths import path      # noqa: E402
from members import final_members      # noqa: E402

FINAL = "1adc681d208ef87977103cda1604d22aab3041a020f79471f13a35c59404ca4a"      # the final submission file (12,043 bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(path("parts", "work/parts"))); ap.add_argument("--out", default=str(path("work", "work") / "final" / "go_rank_final.csv"))
    a = ap.parse_args()
    files = [Path(a.parts) / f"go_rank_{m['tag']}_x_D64_posteriors.npz" for m in final_members()]
    miss = [f.name for f in files if not f.exists()]
    if miss: raise SystemExit(f"missing {len(miss)} member files, e.g. {miss[:3]}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    print("members", len(files), flush=True)
    subprocess.run([sys.executable, str(HERE / "blend_test.py"), str(out), "-"] + [str(f) for f in files], check=True)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print("SHA256", h.upper(), "(matches the final file)" if h == FINAL else "(differs from the final file)")


if __name__ == "__main__":
    main()
