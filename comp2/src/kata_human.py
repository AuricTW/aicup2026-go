"""HumanSL rank-profile likelihood of every played move (KataGo engine, human model as main model, 1 visit).

For each game and each profile r: log p_r(move actually played) at every turn.
Output: OUT/hchunk_sXXXXXXX_NNNNN.npz  logp float16 [sum(T), P], rank int16 [sum(T), P], wr float16 [sum(T), P], offset
        + .json meta list (same keys as the game list)
usage: kata_human.py GAMELIST OUT --profiles rank_15k,rank_7k,rank_2k,rank_3d,rank_8d [--start N --end N --stride 1]
"""
import sys, os, json, time, subprocess, threading, queue, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kata_eval import parse, cfg_text
from comp2_paths import katago_engine, path as _cfg_path

HUMAN = str(_cfg_path("humansl_bin"))      # engine-format HumanSL model (README, weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gamelist"); ap.add_argument("out")
    ap.add_argument("--profiles", default="rank_15k,rank_7k,rank_2k,rank_3d,rank_8d")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--chunk", type=int, default=2000); ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--inflight", type=int, default=150)
    ap.add_argument("--select", default=None, help="json file with list of game-list line numbers to process")
    a = ap.parse_args()
    profs = a.profiles.split(","); P = len(profs)
    os.makedirs(a.out, exist_ok=True)
    cfg = a.out + "/analysis.cfg"
    open(cfg, "w").write(cfg_text(a.threads, 256, a.out + "/katalog") + "humanSLProfile = rank_1d\n")
    exe, env = katago_engine(os.environ.copy())
    proc = subprocess.Popen([exe, "analysis", "-config", cfg, "-model", HUMAN], env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=open(a.out + "/stderr.log", "a"), text=True, bufsize=1 << 20)
    sel = set(json.load(open(a.select))) if a.select else None
    games = []
    with open(a.gamelist) as f:
        for i, line in enumerate(f):
            if i < a.start: continue
            if a.end >= 0 and i >= a.end: break
            if sel is not None and i not in sel: continue
            games.append((i, line))
    print("games", len(games), "profiles", profs, flush=True)
    state = {}; done_q = queue.Queue(); sem = threading.Semaphore(a.inflight); errors = []

    def reader():
        for line in proc.stdout:
            r = json.loads(line)
            if "error" in r or "warning" in r:
                errors.append(line[:300]); continue
            gid, pi = r["id"].split("|"); pi = int(pi)
            st = state[gid]; t = r["turnNumber"]
            pol = np.asarray(r["policy"], np.float32); m = st["moves"][t]
            pm = pol[m]
            st["logp"][t, pi] = np.log(max(pm, 1e-7)); st["rank"][t, pi] = int((pol > pm).sum()); st["wr"][t, pi] = r["rootInfo"]["winrate"]
            st["left"] -= 1
            if st["left"] == 0:
                del state[gid]; done_q.put(st); sem.release()
        done_q.put(None)

    def writer():
        for gi, line in games:
            g = json.loads(line)
            hd, mv, idx = parse(g.pop("sgf"))
            T = len(mv)
            if T == 0:
                done_q.put(dict(meta=dict(g, gi=gi, hd=hd, n_moves=0), empty=True)); continue
            gid = str(gi)
            g.update(gi=gi, hd=hd, n_moves=T, first_player=mv[0][0])
            sem.acquire()
            state[gid] = dict(logp=np.zeros((T, P), np.float16), rank=np.zeros((T, P), np.int16), wr=np.zeros((T, P), np.float16),
                              moves=np.array(idx, np.int16), left=T * P, meta=g)
            for pi, pr in enumerate(profs):
                req = dict(id=f"{gid}|{pi}", moves=mv, initialPlayer=g["first_player"], rules="japanese", komi=6.5 if hd == 0 else 0.5,
                           boardXSize=19, boardYSize=19, analyzeTurns=list(range(T)), maxVisits=1, includePolicy=True,
                           overrideSettings=dict(humanSLProfile=pr))
                proc.stdin.write(json.dumps(req, separators=(",", ":")) + "\n")
            proc.stdin.flush()  # without this a small in-flight window never fills the 1 MB pipe buffer and the job deadlocks
        proc.stdin.flush(); proc.stdin.close()

    threading.Thread(target=reader, daemon=True).start(); threading.Thread(target=writer, daemon=True).start()
    t0 = time.time(); buf = []; nchunk = 0; npos = 0; ndone = 0

    def flush():
        nonlocal buf, nchunk
        if not buf: return
        buf.sort(key=lambda s: s["meta"]["gi"])
        off = np.cumsum([0] + [len(s["logp"]) for s in buf])
        name = f"{a.out}/hchunk_s{a.start:07d}_{nchunk:05d}"
        np.savez(name + ".npz", logp=np.concatenate([s["logp"] for s in buf]), rank=np.concatenate([s["rank"] for s in buf]),
                 wr=np.concatenate([s["wr"] for s in buf]), offset=off, profiles=np.array(profs))
        json.dump([s["meta"] for s in buf], open(name + ".json", "w"))
        nchunk += 1; buf = []

    while ndone < len(games):
        st = done_q.get()
        if st is None: break
        ndone += 1
        if "empty" in st: continue
        buf.append(st); npos += len(st["logp"]) * P
        if len(buf) >= a.chunk: flush()
        if ndone % 200 == 0:
            el = time.time() - t0
            print(f"done {ndone}/{len(games)} evals {npos} {npos / el:.0f} evals/s elapsed {el:.0f}s errors {len(errors)}", flush=True)
    flush()
    json.dump(dict(done=ndone, evals=npos, elapsed=time.time() - t0, errors=errors[:50]), open(a.out + "/DONE.json", "w"))
    print("DONE", ndone, npos, time.time() - t0, len(errors), flush=True)
    proc.wait(timeout=60)


if __name__ == "__main__":
    main()
