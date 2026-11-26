"""Full-trajectory KataGo raw-net evaluation (1 visit per position, every turn of every game).

Input : gamelist jsonl, one game per line: {"key":..., "sgf":..., ...meta}
Output: OUT/chunk_XXXXX.npz with
          feats  float32 [sum(T+1), NF]  one row per position (turn 0..T)
          moves  int16   [sum(T+1)]      policy index of the move played FROM that position (-1 for final)
          offset int64   [n_games+1]
        OUT/chunk_XXXXX.json  list of game meta (key, hd, n_moves, first_player, ...)
Row features (BLACK perspective for winrate / score):
  0 winrate  1 scoreLead  2 scoreStdev  3 rawStWrError  4 rawStScoreError  5 rawVarTimeLeft
  6 p(played)  7 rank(played) among legal (0=top)  8 max policy  9 policy entropy  10 p(pass)  11 n_legal
usage: kata_eval.py GAMELIST OUT [--start N] [--end N] [--chunk 2000] [--threads 256]
"""
import sys, os, json, re, time, subprocess, threading, queue, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp2_paths import katago_engine, path as _cfg_path
EXT = None      # (was the engine folder of the GPU host; now configs/paths.yaml: katago_exe, katago_lib_dir)
COLS = "ABCDEFGHJKLMNOPQRST"
MOVE = re.compile(r";([BW])\[([a-s]{2}|tt|)\]")
HD = re.compile(r"HD\[(\d+)\]")
NF = 12


def parse(sgf):
    hd = HD.search(sgf)
    hd = int(hd.group(1)) if hd else 0
    mv, idx = [], []
    for c, p in MOVE.findall(sgf):
        if p in ("", "tt"):
            mv.append([c, "pass"]); idx.append(361)
        else:
            x, y = ord(p[0]) - 97, ord(p[1]) - 97
            mv.append([c, COLS[x] + str(19 - y)]); idx.append(y * 19 + x)
    return hd, mv, idx


def cfg_text(threads, batch, logdir):
    return f"""logDir = {logdir}
logToStderr = false
numAnalysisThreads = {threads}
numSearchThreads = 1
nnMaxBatchSize = {batch}
nnCacheSizePowerOfTwo = 20
nnMutexPoolSizePowerOfTwo = 16
nnRandomize = false
nnRandSeed = 20261114
reportAnalysisWinratesAs = BLACK
cudaUseFP16 = true
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gamelist"); ap.add_argument("out")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=256); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--inflight", type=int, default=600)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = a.out + "/analysis.cfg"
    open(cfg, "w").write(cfg_text(a.threads, a.batch, a.out + "/katalog"))
    env = os.environ.copy()
    exe, env = katago_engine(env)
    proc = subprocess.Popen([exe, "analysis", "-config", cfg, "-model", str(_cfg_path("katago_main_model"))], env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(a.out + "/stderr.log", "a"),
                            text=True, bufsize=1 << 20)
    games = []
    with open(a.gamelist) as f:
        for i, line in enumerate(f):
            if i < a.start: continue
            if a.end >= 0 and i >= a.end: break
            games.append((i, line))
    print("games", len(games), flush=True)
    state = {}   # gid -> dict(arr, moves, left, meta)
    done_q = queue.Queue()
    sem = threading.Semaphore(a.inflight)
    errors = []

    def reader():
        for line in proc.stdout:
            r = json.loads(line)
            if "error" in r or "warning" in r:
                errors.append(line[:300])
                if "error" in r and "id" in r and r["id"] in state:
                    st = state.pop(r["id"]); st["failed"] = line[:300]; done_q.put(st); sem.release()
                continue
            st = state[r["id"]]
            t = r["turnNumber"]; ri = r["rootInfo"]; row = st["arr"][t]
            row[0] = ri["winrate"]; row[1] = ri["scoreLead"]; row[2] = ri.get("scoreStdev", 0.0)
            row[3] = ri.get("rawStWrError", 0.0); row[4] = ri.get("rawStScoreError", 0.0); row[5] = ri.get("rawVarTimeLeft", 0.0)
            pol = np.asarray(r["policy"], np.float32)
            legal = pol >= 0
            pl = pol[legal]
            m = st["moves"][t]
            if m >= 0:
                pm = pol[m]
                row[6] = pm; row[7] = float((pl > pm).sum())
            row[8] = pl.max(); row[9] = float(-(pl * np.log(pl + 1e-12)).sum()); row[10] = pol[361]; row[11] = legal.sum()
            st["left"] -= 1
            if st["left"] == 0:
                del state[r["id"]]; done_q.put(st); sem.release()
        done_q.put(None)

    def writer():
        for gi, line in games:
            g = json.loads(line)
            hd, mv, idx = parse(g.pop("sgf"))
            T = len(mv)
            gid = str(gi)
            g.update(gi=gi, hd=hd, n_moves=T, first_player=mv[0][0] if mv else "B")
            sem.acquire()
            state[gid] = dict(arr=np.zeros((T + 1, NF), np.float32), moves=np.array(idx + [-1], np.int16), left=T + 1, meta=g)
            req = dict(id=gid, moves=mv, initialPlayer=g["first_player"], rules="japanese", komi=6.5 if hd == 0 else 0.5,
                       boardXSize=19, boardYSize=19, analyzeTurns=list(range(T + 1)), maxVisits=1, includePolicy=True)
            proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
        proc.stdin.flush()
        proc.stdin.close()

    threading.Thread(target=reader, daemon=True).start()
    threading.Thread(target=writer, daemon=True).start()
    t0 = time.time(); buf = []; nchunk = 0; npos = 0; ndone = 0; fails = []

    def flush():
        nonlocal buf, nchunk
        if not buf: return
        buf.sort(key=lambda s: s["meta"]["gi"])
        off = np.cumsum([0] + [len(s["arr"]) for s in buf])
        name = f"{a.out}/chunk_s{a.start:07d}_{nchunk:05d}"
        np.savez(name + ".npz", feats=np.concatenate([s["arr"] for s in buf]),
                 moves=np.concatenate([s["moves"] for s in buf]), offset=off)
        json.dump([s["meta"] for s in buf], open(name + ".json", "w"))
        nchunk += 1; buf = []

    while ndone < len(games):
        st = done_q.get()
        if st is None: break
        ndone += 1
        if "failed" in st:
            fails.append(dict(st["meta"], error=st["failed"])); continue
        buf.append(st); npos += len(st["arr"])
        if len(buf) >= a.chunk: flush()
        if ndone % 500 == 0:
            el = time.time() - t0
            print(f"done {ndone}/{len(games)} pos {npos} {npos / el:.0f} pos/s elapsed {el:.0f}s errors {len(errors)}", flush=True)
    flush()
    json.dump(dict(done=ndone, positions=npos, elapsed=time.time() - t0, failures=fails, errors=errors[:50]), open(a.out + "/DONE.json", "w"))
    print("DONE", ndone, npos, time.time() - t0, len(fails), flush=True)
    proc.wait(timeout=60)


if __name__ == "__main__":
    main()
