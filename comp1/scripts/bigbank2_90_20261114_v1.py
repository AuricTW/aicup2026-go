"""Bigbank v2: production training bank for view90 PLDA. More players AND more games per player.

Cross-fit evidence (runs/crossfit90_20261114_v1): 2x players ~ +0.025-0.04, 2x games/player (16->32) ~ +0.02.
Players: every owner except the historical validation owners (incl. sealed 400) -- i.e. now INCLUDES the
scale/pilot/earlier dev-eval identities (production bank; evaluation moves to cross-fit fold episodes).
Families of excluded (hist) games removed; families disjoint across picked games (as bigbank v1).
Up to 64 games/owner, min 4. Replay chunks (<=33k games each): tier A (first 32 per owner) first, then tier B.
Stages: prepare | package CHUNK | accept CHUNK SHA SIZE | train | crossfit
"""
from pathlib import Path
import sys, argparse, json, shutil, tarfile, hashlib
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank90_20261114_v1 as bb
import numpy as np
from threadpoolctl import threadpool_limits
s = bb.s; p = bb.p; old = bb.old
RUN = p.ROOT / 'runs/bigbank2_90_20261114_v1'; SEED = 2026111403; PER_OWNER = 64; MIN_GAMES = 4; TIER = 32; CHUNK = 33000
CACHES = [s.RUN / 'views', bb.RUN / 'dev_results', old.RUN / 'dev_results']


def h(v): return hashlib.sha256((str(SEED) + '|' + str(v)).encode()).digest()


def cache_index():
    out = {}
    for c in CACHES:
        ids = np.load(c / ('../all_indices.npy' if c.name == 'views' else 'indices.npy'))
        for j, i in enumerate(ids): out.setdefault(int(i), (str(c), j))
    return out


def prepare():
    RUN.mkdir(exist_ok=False); m = p.b.meta(); owners = m['owners'].astype(str)
    hist = set(owners[np.load(p.DENSE.parent / 'validation_indices.npy')])
    excluded = np.isin(owners, sorted(hist)); safe = (~excluded) & (m['hd'] <= 1)
    for f in p.b.FAMILIES: safe &= ~np.isin(m[f], np.unique(m[f][excluded]))
    by = defaultdict(list)
    for i in np.flatnonzero(safe): by[owners[i]].append(int(i))
    cached = cache_index(); used = {f: set() for f in p.b.FAMILIES}; train, labels, tier = [], [], []; k = 0
    for o in sorted(by, key=h):
        picked = []; local = {f: set() for f in p.b.FAMILIES}; cols = {0: [], 1: []}
        for i in by[o]: cols[int(m['colors'][i])].append(i)
        for c in (0, 1): cols[c].sort(key=lambda i: (i not in cached, h(m['group'][i])))
        order = [x for pair in zip(cols[0], cols[1]) for x in pair] + cols[0][len(cols[1]):] + cols[1][len(cols[0]):]
        for i in order:
            if any(m[f][i] in used[f] or m[f][i] in local[f] for f in p.b.FAMILIES): continue
            picked.append(i)
            for f in p.b.FAMILIES: local[f].add(m[f][i])
            if len(picked) == PER_OWNER: break
        if len(picked) < MIN_GAMES: continue
        for f in p.b.FAMILIES: used[f].update(local[f])
        train += picked; labels += [k] * len(picked); tier += [0 if j < TIER else 1 for j in range(len(picked))]; k += 1
    train = np.asarray(train, np.int64); labels = np.asarray(labels, np.int64); tier = np.asarray(tier, np.int8)
    assert not set(owners[train]) & hist
    for f in p.b.FAMILIES: assert not set(m[f][train]) & set(m[f][excluded])
    need = np.array([int(i) not in cached for i in train]); chunks = []
    for t in (0, 1):
        ii = train[need & (tier == t)]; ii = np.array(sorted(ii.tolist(), key=h), np.int64)
        for st in range(0, len(ii), CHUNK): chunks.append(ii[st:st + CHUNK])
    for n, c in enumerate(chunks): np.save(RUN / f'replay_chunk{n}.npy', np.sort(c), allow_pickle=False)
    np.save(RUN / 'train_indices.npy', train, allow_pickle=False); np.save(RUN / 'train_labels.npy', labels, allow_pickle=False); np.save(RUN / 'train_tier.npy', tier, allow_pickle=False)
    gp = np.bincount(labels)
    r = {'owners': int(k), 'games': int(len(train)), 'tierA_games': int((tier == 0).sum()), 'tierB_games': int((tier == 1).sum()), 'cached_reused': int((~need).sum()), 'replay_total': int(need.sum()),
         'replay_chunks': [int(len(c)) for c in chunks], 'chunk_tiers': ['A' if t == 0 else 'B' for t in (0, 1) for _ in range(-(-int((need & (tier == t)).sum()) // CHUNK))],
         'games_per_owner_q': np.quantile(gp, [0, .1, .5, .9, 1]).tolist(), 'owners_ge32': int((gp >= 32).sum()), 'owners_ge15': int((gp >= 15).sum()),
         'includes_dev_eval_identities': True, 'excluded': 'historical validation owners (incl sealed 400) + their families', 'per_owner_max': PER_OWNER, 'min_games': MIN_GAMES, 'seed': SEED, 'source_SHA256': p.b.sha(__file__)}
    p.b.write(RUN / 'SCOPE.json', r); print(json.dumps(r, indent=1))


def package(chunk):
    import remote_metric64_capacity_20261112_v1 as remote
    chunk = int(chunk); out = RUN / f'payload_c{chunk}'; out.mkdir(exist_ok=False); source = s.RUN / 'payload_dev'
    for f in ('worker.py', 'basis64.py', 'model_config.json'): shutil.copy2(source / f, out / f)
    for f in ('vendor', 'code'): shutil.copytree(source / f, out / f, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert p.b.sha(out / 'worker.py') == '3be16a208e7c804bb906a1dc4e2135ec335115901a95713b7b02d231abb7650b'
    with (out / 'supervisor.py').open('x', encoding='utf-8', newline='\n') as f: f.write(remote.SUPERVISOR)
    ids = np.load(RUN / f'replay_chunk{chunk}.npy'); m = p.b.meta(); from goid_v4.pilot_scope_v1 import PointReader
    reader = PointReader(p.ROOT / 'data/processed/audit.sqlite')
    try:
        with (out / 'games.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for st in range(0, len(ids), 256):
                ii = ids[st:st + 256]; texts = {r['row_id']: r['sgf_content'] for r in reader.sample_text(m['rows'][ii])}
                for i in ii:
                    text = texts[int(m['rows'][i])]; assert hashlib.sha256(text.encode()).hexdigest() == m['raw'][i].decode()
                    stream.write(json.dumps({'index': int(i), 'sgf': text, 'raw': m['raw'][i].decode(), 'color': 'BW'[int(m['colors'][i])], 'hd': int(m['hd'][i]), 'canonical': m['canonical'][i].decode()}) + '\n')
    finally: reader.close()
    p.b.write(out / 'MANIFEST.json', {'files': {f.relative_to(out).as_posix(): p.b.sha(f) for f in out.rglob('*') if f.is_file()}, 'weight_SHA256': p.b.sha(p.n.RUN / 'HumanSL_state_only.pt'), 'role': f'bigbank2_90_chunk{chunk}_forward_only', 'games': len(ids), 'fit_or_optimizer': False, 'worker_exact_priorcomplete90': True})
    archive = RUN / f'PAYLOAD_c{chunk}.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=3) as tf:
        for f in out.rglob('*'):
            if f.is_file(): tf.add(f, arcname=f.relative_to(out).as_posix(), recursive=False)
    r = {'archive': str(archive), 'SHA256': p.b.sha(archive), 'bytes': archive.stat().st_size, 'games': len(ids)}; p.b.write(RUN / f'PAYLOAD_c{chunk}.json', r); print(r)


def accept(chunk, expected, size):
    archive = RUN / f'RETURN_c{chunk}.tar.gz'; assert p.b.sha(archive) == expected and archive.stat().st_size == int(size); out = RUN / f'return_c{chunk}'; out.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf: tf.extractall(out, filter='data')
    manifest = p.b.read(out / 'RETURN_MANIFEST.json'); assert all(p.b.sha(out / f) == hh for f, hh in manifest.items())
    assert p.b.read(out / 'REMOTE_JOB.json')['archive_SHA256'] == p.b.read(RUN / f'PAYLOAD_c{chunk}.json')['SHA256']
    rel = p.b.read(out / 'GPU_RELEASE.json'); assert rel['worker_exit'] == 0 and rel['owned_PID_absent']
    dest = RUN / f'results_c{chunk}'; dest.mkdir(exist_ok=False)
    for f in (out / 'results').iterdir(): shutil.copy2(f, dest / f.name)
    assert p.b.read(dest / 'RECEIPT.json')['all_valid']; assert np.array_equal(np.sort(np.load(dest / 'indices.npy')), np.load(RUN / f'replay_chunk{chunk}.npy'))
    p.b.write(RUN / f'RETURN_c{chunk}_VERIFIED.json', {'SHA256': expected, 'bytes': int(size), 'release': rel})


def lookup90(indices):
    srcs = [(np.load(s.RUN / 'all_indices.npy'), np.load(s.RUN / 'views/view90.npy', mmap_mode='r'))]
    for d in [bb.RUN / 'dev_results', old.RUN / 'dev_results'] + sorted(RUN.glob('results_c*')): srcs.append((np.load(d / 'indices.npy'), np.load(d / 'view90.npy', mmap_mode='r')))
    pos = {}
    for n, (ids, _) in enumerate(srcs):
        for j, i in enumerate(ids): pos.setdefault(int(i), (n, j))
    miss = [int(i) for i in indices if int(i) not in pos]
    if miss: raise KeyError(f'{len(miss)} games without features')
    out = np.empty((len(indices), 2339), np.float32)
    for k, i in enumerate(indices): n, j = pos[int(i)]; out[k] = srcs[n][1][j]
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'package': lambda: package(*v.args), 'accept': lambda: accept(*v.args)}[v.stage]()
