"""Late-game per-decision HumanSL tokens (decisions after ply 180).
Why: every token net so far sees only the target player's decisions in plies 1-180 (<= 90 decisions; 75% of official games hit the
cap).  Official games: median 224 plies, 74% longer than 180, on average 23.6 more target decisions per game are never seen.
Masking showed later decisions matter (plies <= 120 only: 0.663 vs 0.714 with <= 180).
This script reuses the frozen worker of an existing payload (board replay, feature encoding, HumanSL forward, token definition are the
worker's own functions); only the decision selection is replaced: all target decisions with lo <= ply <= hi, in ply order, at most k.
Games without such a decision get n = 0 (no fallback to another selection).
usage: toklate_20261118_v1.py <payload dir with worker.py/code/vendor/model_config.json> <games.jsonl> <weight .pt> <out dir>
                              [--lo 181 --hi 300 --k 60 --range a:b --workers 8]
writes <out>/{tok_z [N,k,1152] f16, tok_a [N,k,10] f32, tok_ply [N,k] i16, tok_n [N] u8, tok_act [N,k] u16 (worker action index
       (18-row)*19+col, 361 pass, 362 pad), indices [N]}.npy, ERRORS.json, RECEIPT.json
Parity: with --lo 1 --hi 180 --k 90 the selection equals the worker's (30 opening + <= 60 middle target decisions = all of them),
so tokens must equal the accepted ones (checked on official games before use).
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8'); os.environ.setdefault('OMP_NUM_THREADS', '4'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys, json, time, argparse, multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
W = None


def setup():
    """import the payload's frozen worker and replace its selection (runs in the parent and in every spawned child)"""
    global W
    if W is not None or 'LATE_PAYLOAD' not in os.environ: return
    sys.path.insert(0, os.environ['LATE_PAYLOAD']); import worker as W_; W = W_
    lo, hi, k = int(os.environ['LATE_LO']), int(os.environ['LATE_HI']), int(os.environ['LATE_K'])

    def select(moves, target, canonical, quota=None):
        plies = [ply for ply, (color, _) in enumerate(moves, 1) if color == target.lower() and lo <= ply <= hi][:k]
        return set(plies if quota is None else plies[:quota])
    W.select = select


setup()


def encode_batch(items):
    return [W.encode(item) for item in items]


def infer(model, rec):
    """= the worker's infer64 token part (same chunking of 32, same forward, same token / scalar definitions)"""
    import torch
    px, glob, legal, actions, aux, meta, small, plies = rec; count = len(actions); assert 1 <= count <= 90; zs = []; aa = []; device = next(model.parameters()).device
    for st in range(0, count, 32):
        real = min(32, count - st); ix = np.pad(np.arange(st, st + real), (0, 32 - real), constant_values=st)
        x = np.unpackbits(px[ix], axis=1)[:, :22 * 361].reshape(32, 22, 19, 19).astype(np.float32); mask = np.unpackbits(legal[ix], axis=1)[:, :362].astype(bool)
        pi, h = W.p.n.forward(model, torch.from_numpy(x).to(device), torch.from_numpy(glob[ix]).to(device), torch.from_numpy(np.repeat(meta[None], 32, axis=0)).to(device), torch.from_numpy(mask).to(device), amp=True)
        h = h.float().flatten(2); ac = torch.from_numpy(actions[ix]).to(device); local = h.gather(2, ac.clamp_max(360)[:, None, None].expand(-1, 384, 1)).squeeze(2); local[ac == 361] = 0; expected = (h * pi[:, :361, None].transpose(1, 2)).sum(2); z = torch.cat((local, h.mean(2), local - expected), 1).cpu().numpy()[:real]
        chosen = pi.gather(1, ac[:, None]).squeeze(1); rank = ((pi > chosen[:, None]) & torch.from_numpy(mask).to(device)).sum(1).float() / torch.from_numpy(mask.sum(1)).to(device); entropy = -(pi * torch.log(pi.clamp_min(1e-12))).sum(1); top = pi.topk(2, dim=1).values
        numbers = torch.stack((torch.log(chosen.clamp_min(1e-12)), rank, entropy, torch.log(top[:, 0].clamp_min(1e-12)), top[:, 0] - top[:, 1]), 1).cpu().numpy()[:real]; a = np.concatenate((aux[ix[:real], :2], numbers, aux[ix[:real], 2:]), 1); zs.append(z); aa.append(a)
    return np.concatenate(zs).astype(np.float16), np.concatenate(aa).astype(np.float32), np.asarray(plies, np.int16), np.asarray(actions, np.uint16)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('payload'); ap.add_argument('games'); ap.add_argument('weight'); ap.add_argument('out')
    ap.add_argument('--lo', type=int, default=181); ap.add_argument('--hi', type=int, default=300); ap.add_argument('--k', type=int, default=60); ap.add_argument('--range', default=''); ap.add_argument('--workers', type=int, default=8); v = ap.parse_args()
    assert v.k <= 90
    os.environ.update({'LATE_PAYLOAD': str(Path(v.payload).resolve()), 'LATE_LO': str(v.lo), 'LATE_HI': str(v.hi), 'LATE_K': str(v.k)}); setup()
    import torch
    from katago.train.load_model import load_model
    out = Path(v.out); out.mkdir(parents=True, exist_ok=False); t0 = time.time()
    records = [json.loads(s) for s in Path(v.games).read_text().splitlines()]
    if v.range: a_, b_ = (int(x) for x in v.range.split(':')); records = records[a_:b_]
    config = json.loads((Path(v.payload) / 'model_config.json').read_text()); ids = np.asarray([r['index'] for r in records]); pos = {int(i): j for j, i in enumerate(ids)}
    tasks = [(r['index'], r['sgf'], r['raw'], r['hd'], r['color'], config) for r in records]; del records
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    model, swa, _ = load_model(v.weight, use_swa=False, device=torch.device('cuda'), pos_len=19); del swa; model.eval().requires_grad_(False)
    N = len(ids); K = v.k
    TK = {'z': np.lib.format.open_memmap(out / 'tok_z.npy', mode='w+', dtype=np.float16, shape=(N, K, 1152)), 'a': np.lib.format.open_memmap(out / 'tok_a.npy', mode='w+', dtype=np.float32, shape=(N, K, 10)),
          'ply': np.lib.format.open_memmap(out / 'tok_ply.npy', mode='w+', dtype=np.int16, shape=(N, K)), 'n': np.lib.format.open_memmap(out / 'tok_n.npy', mode='w+', dtype=np.uint8, shape=(N,)),
          'act': np.lib.format.open_memmap(out / 'tok_act.npy', mode='w+', dtype=np.uint16, shape=(N, K))}
    TK['act'][:] = 362
    done = np.zeros(N, bool); errors = {}; it = iter([tasks[i:i + 24] for i in range(0, N, 24)]); last = 0
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
                    z, a, pl, ac = infer(model, rec); k_ = len(pl); TK['z'][j, :k_] = z; TK['a'][j, :k_] = a; TK['ply'][j, :k_] = pl; TK['n'][j] = k_; TK['act'][j, :k_] = ac
                b = next(it, None)
                if b: pending.add(pool.submit(encode_batch, b))
            if done.sum() - last >= 2000 or done.all(): last = int(done.sum()); print(json.dumps({'games': last, 'total': N, 'seconds': round(time.time() - t0)}), flush=True)
    assert done.all(); [x.flush() for x in TK.values()]; np.save(out / 'indices.npy', ids, allow_pickle=False)
    n = np.asarray(TK['n']); kinds = {}
    for e in errors.values(): kinds[e.split(':')[0]] = kinds.get(e.split(':')[0], 0) + 1
    json.dump(errors, open(out / 'ERRORS.json', 'w'))
    rc = {'games': N, 'lo': v.lo, 'hi': v.hi, 'k': K, 'range': v.range, 'mean_n': float(n.mean()), 'share_n0': float((n == 0).mean()), 'share_full': float((n == K).mean()), 'error_kinds': kinds, 'seconds': time.time() - t0,
          'GPU': torch.cuda.get_device_name(), 'payload': str(v.payload), 'games_file': str(v.games), 'weight': str(v.weight)}
    json.dump(rc, open(out / 'RECEIPT.json', 'w'), indent=1); print(json.dumps(rc), flush=True)


if __name__ == '__main__':
    main()
