"""Build the 91 member parts of the final file from the evidence files in WORK (stage R2 of README; CPU only).

For every member: (1) the id2 token-evidence files of its set on the three tracks (tok_evidence.py; skipped if present),
(2) fuse_sources.py with the member's settings -> PARTS/go_rank_<tag>_{x_D64_posteriors.npz, OOF_posteriors.npz, VALIDATION.json, ...}.
usage: run_parts.py [--parts DIR] [--only TAG[,TAG]] [--workers N] [--dry-run]
"""
import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from comp2_paths import WORK, path      # noqa: E402
from members import TOKSETS, final_members      # noqa: E402

PY = sys.executable
REF = {"x": WORK / "idnet_tx2_full" / "acc_6m.npz", "full": WORK / "idnet_bf1_p9c64" / "acc_3.npz", "test": WORK / "idnet_ta2lb" / "acc_6m.npz"}


def evidence(ev, dry):
    frozen, tuned, cnt = TOKSETS[ev]
    files = [str(WORK / "tokres" / f"{b}_{m}.npz") for m in frozen for b in ("td", "oof")] + [str(WORK / "xf" / f"{b}_{m}.npz") for m in tuned for b in ("td", "oof")]
    miss = [f for f in files if not os.path.exists(f)]
    for track, ref in REF.items():
        out = WORK / "tokeval" / f"{track}_{ev}_A1.0.npz"
        if out.exists(): continue
        if miss: raise SystemExit(f"missing token-net outputs for {ev}: {miss[:3]} ...")
        cmd = [PY, str(HERE / "tok_evidence.py"), str(ref), str(out), "1.5", "50", ",".join(files), "--prior", "1.0"]
        if cnt: cmd += ["--cnt-add", str(WORK / "tokeval" / f"cntadd_{track}.npz")]
        print(" ".join(cmd[1:3]), "...", flush=True)
        if not dry: out.parent.mkdir(parents=True, exist_ok=True); subprocess.run(cmd, check=True)


def fuse(m, parts, dry):
    ev = m["ev"]; t = WORK / "tokeval"
    cmd = [PY, str(HERE / "fuse_sources.py"), "--id-x", str(REF["x"]), "--id-full", str(REF["full"]), "--id-test", str(REF["test"]),
           "--id2-x", str(t / f"x_{ev}_A1.0.npz"), "--id2-full", str(t / f"full_{ev}_A1.0.npz"), "--id2-test", str(t / f"test_{ev}_A1.0.npz"),
           "--id-mode", "acc", "--sources", m.get("sources", "zopp,zown,id,id2"), "--out-dir", str(parts), "--tag", m["tag"]]
    if "rank" in m:
        r = WORK / "tokrank"
        cmd += ["--dir-x", str(r / f"r1_{m['rank']}.npz"), "--dir-full", str(r / f"r1full_{m['rank']}.npz"), "--dir-test", str(r / f"r1full_{m['rank']}.npz")]
    env = {k: v for k, v in os.environ.items() if k not in ("ZOPP_GATE", "S1_RIDGE", "DEV_SEED")}; env.update(m["env"])
    if dry: print(m["tag"], m["env"], " ".join(cmd[2:])); return
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode: raise SystemExit(f"{m['tag']} failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    print(m["tag"], [ln for ln in res.stdout.splitlines() if "mean over G  fitted2" in ln][-1:], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(path("parts", "work/parts"))); ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ms = final_members()
    if a.only: keep = set(a.only.split(",")); ms = [m for m in ms if m["tag"] in keep]
    for ev in dict.fromkeys(m["ev"] for m in ms): evidence(ev, a.dry_run)
    parts = Path(a.parts); parts.mkdir(parents=True, exist_ok=True)
    todo = [m for m in ms if not (parts / f"go_rank_{m['tag']}_x_D64_posteriors.npz").exists()]
    print(f"members {len(ms)}, to build {len(todo)}", flush=True)
    with ThreadPoolExecutor(max(1, a.workers)) as ex: list(ex.map(lambda m: fuse(m, parts, a.dry_run), todo))


if __name__ == "__main__":
    main()
