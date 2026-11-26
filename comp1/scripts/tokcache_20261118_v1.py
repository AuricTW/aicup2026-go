"""Cache the HumanSL network INPUTS of every target decision (plies 1-300, <= 150 per game) - CPU only.
Why (2026-11-18 review): data scale, heads on the frozen HumanSL features and the fusion are saturated (more games per player: 0;
more seeds: 0; validation 0.778).  Error anatomy: top-1 0.738 but recall@5 0.906 -> fine discrimination between similar candidates.
The one large part of the pipeline never trained for identity is the HumanSL trunk itself (frozen move-prediction features).
Fine-tuning it needs the encoded positions many times; encoding (board replay + KataGo features) is the CPU-bound part of every
extraction so far, so it is done once here and stored packed.
Reuses the frozen worker of a payload dir exactly like toklate_20261118_v1.py (encode() unchanged; selection = all target decisions
with ply <= hi in ply order, at most k).
usage: tokcache_20261118_v1.py <payload dir> <games.jsonl> <out dir> [--hi 300 --k 150 --range a:b --workers 12]
writes <out>/{px [N,k,993] u8 (22 planes x 361 bits, packbits), g [N,k,19] f32, legal [N,k,46] u8 (362 bits), act [N,k] u16 (worker
       action index, 362 = pad), ply [N,k] i16, n [N] u8, indices [N]}.npy, ERRORS.json, RECEIPT.json
The per-decision scalars of the tokens are functions of (ply, hd, act) and the network output; colour and hd come from games.jsonl
(stored: black [N] bool, hd [N] u8).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, time, argparse, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
W = None


def setup():
    global W
    if W is not None or 'CACHE_PAYLOAD' not in os.environ: return
    sys.path.insert(0, os.environ['CACHE_PAYLOAD']); import worker as W_; W = W_
    hi, k = int(os.environ['CACHE_HI']), int(os.environ['CACHE_K'])

    def select(moves, target, canonical, quota=None):
        plies = [ply for ply, (color, _) in enumerate(moves, 1) if color == target.lower() and ply <= hi][:k]
        return set(plies if quota is None else plies[:quota])
    W.select = select


setup()


def encode_batch(items):
    out = []
    for item in items:
        index, rec, err = W.encode(item)
        out.append((index, None, err) if rec is None else (index, (rec[0], rec[1].astype(np.float32), rec[2], rec[3].astype(np.uint16), rec[7].astype(np.int16)), None))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('payload'); ap.add_argument('games'); ap.add_argument('out'); ap.add_argument('--hi', type=int, default=300); ap.add_argument('--k', type=int, default=150); ap.add_argument('--range', default=''); ap.add_argument('--workers', type=int, default=12); v = ap.parse_args()
    os.environ.update({'CACHE_PAYLOAD': str(Path(v.payload).resolve()), 'CACHE_HI': str(v.hi), 'CACHE_K': str(v.k)}); setup()
    out = Path(v.out); out.mkdir(parents=True, exist_ok=False); t0 = time.time(); records = [json.loads(s) for s in Path(v.games).read_text().splitlines()]
    if v.range: a_, b_ = (int(x) for x in v.range.split(':')); records = records[a_:b_]
    config = json.loads((Path(v.payload) / 'model_config.json').read_text()); ids = np.asarray([r['index'] for r in records]); pos = {int(i): j for j, i in enumerate(ids)}; N = len(ids); K = v.k
    np.save(out / 'black.npy', np.array([r['color'] == 'B' for r in records])); np.save(out / 'hd.npy', np.array([r['hd'] for r in records], np.uint8))
    tasks = [(r['index'], r['sgf'], r['raw'], r['hd'], r['color'], config) for r in records]; del records
    mm = lambda name, dt, shape: np.lib.format.open_memmap(out / f'{name}.npy', mode='w+', dtype=dt, shape=shape)
    C = {'px': mm('px', np.uint8, (N, K, 993)), 'g': mm('g', np.float32, (N, K, 19)), 'legal': mm('legal', np.uint8, (N, K, 46)), 'act': mm('act', np.uint16, (N, K)), 'ply': mm('ply', np.int16, (N, K)), 'n': mm('n', np.uint8, (N,))}
    C['act'][:] = 362; done = np.zeros(N, bool); errors = {}; it = iter([tasks[i:i + 24] for i in range(0, N, 24)]); last = 0
    with ProcessPoolExecutor(max_workers=v.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = set()
        for _ in range(v.workers * 2):
            b = next(it, None)
            if b: pending.add(pool.submit(encode_batch, b))
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in ready:
                for index, rec, err in fut.result():
                    j = pos[index]; assert not done[j]; done[j] = True
                    if rec is None: errors[str(index)] = err; continue
                    px, g, legal, act, ply = rec; k_ = len(ply); assert px.shape == (k_, 993) and g.shape == (k_, 19) and legal.shape == (k_, 46), (px.shape, g.shape, legal.shape)
                    C['px'][j, :k_] = px; C['g'][j, :k_] = g; C['legal'][j, :k_] = legal; C['act'][j, :k_] = act; C['ply'][j, :k_] = ply; C['n'][j] = k_
                b = next(it, None)
                if b: pending.add(pool.submit(encode_batch, b))
            if done.sum() - last >= 2000 or done.all(): last = int(done.sum()); print(json.dumps({'games': last, 'total': N, 'seconds': round(time.time() - t0)}), flush=True)
    assert done.all(); [x.flush() for x in C.values()]; np.save(out / 'indices.npy', ids, allow_pickle=False); n = np.asarray(C['n']); kinds = {}
    for e in errors.values(): kinds[e.split(':')[0]] = kinds.get(e.split(':')[0], 0) + 1
    json.dump(errors, open(out / 'ERRORS.json', 'w'))
    rc = {'games': N, 'hi': v.hi, 'k': K, 'range': v.range, 'mean_n': float(n.mean()), 'share_n0': float((n == 0).mean()), 'error_kinds': kinds, 'seconds': time.time() - t0, 'payload': str(v.payload), 'games_file': str(v.games)}
    json.dump(rc, open(out / 'RECEIPT.json', 'w'), indent=1); print(json.dumps(rc), flush=True)


if __name__ == '__main__':
    main()
