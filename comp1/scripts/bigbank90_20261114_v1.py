"""Big training bank for the frozen90 representation: many more players (all safe owners) to refit PLDA.

Stages: prepare (select indices, exclusions) | package (4090 payload, exact frozen90 worker) |
accept SHA SIZE | assemble | train | screen (dev eval on scale-run episodes, component + fusion).
Evaluation episodes = complete90_scale screen/confirmation (600 exposed dev identities); all fits exclude
those identities and their four families, plus previous/earlier/historical eval pools (same rule as scale run).
"""
from pathlib import Path
import sys, argparse, json, shutil, tarfile, hashlib
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import complete90_scale_20261112_v1 as s
import numpy as np
from threadpoolctl import threadpool_limits
p = s.p; old = s.old; cap = s.cap
RUN = p.ROOT / 'runs/bigbank90_20261114_v1'; SEED = 2026111401
PER_OWNER = 32; MIN_GAMES = 12


def h(v): return hashlib.sha256((str(SEED) + '|' + str(v)).encode()).digest()


def prepare():
    RUN.mkdir(exist_ok=False); m = p.b.meta(); owners = m['owners'].astype(str)
    scale_ep = p.b.read(s.RUN / 'EPISODE_screen.json'); players = set(scale_ep['candidates'])
    previous = set(p.b.read(old.RUN / 'EPISODE_screen.json')['candidates']); earlier = set(p.b.read(p.b.RUN / 'calibration/ep0.json')['candidates'])
    hist = set(owners[np.load(p.DENSE.parent / 'validation_indices.npy')])
    excluded = np.isin(owners, sorted(players | previous | earlier | hist)); safe = (~excluded) & (m['hd'] <= 1)
    for f in p.b.FAMILIES: safe &= ~np.isin(m[f], np.unique(m[f][excluded]))
    by = defaultdict(list)
    for i in np.flatnonzero(safe): by[owners[i]].append(int(i))
    cachedset = set(np.load(s.RUN / 'all_indices.npy').tolist())
    used = {f: set() for f in p.b.FAMILIES}; train = []; labels = []; k = 0
    for o in sorted(by, key=h):
        picked = []; local = {f: set() for f in p.b.FAMILIES}
        cols = {0: [], 1: []}
        for i in by[o]: cols[int(m['colors'][i])].append(i)
        for c in (0, 1): cols[c].sort(key=lambda i: (i not in cachedset, h(m['group'][i])))
        order = [x for pair in zip(cols[0], cols[1]) for x in pair] + cols[0][len(cols[1]):] + cols[1][len(cols[0]):]
        for i in order:
            if any(m[f][i] in used[f] or m[f][i] in local[f] for f in p.b.FAMILIES): continue
            picked.append(i)
            for f in p.b.FAMILIES: local[f].add(m[f][i])
            if len(picked) == PER_OWNER: break
        if len(picked) < MIN_GAMES: continue
        for f in p.b.FAMILIES: used[f].update(local[f])
        train += picked; labels += [k] * len(picked); k += 1
    train = np.asarray(train, np.int64); labels = np.asarray(labels, np.int64)
    for f in p.b.FAMILIES: assert not set(m[f][train]) & set(m[f][excluded])
    assert not set(owners[train]) & (players | previous | earlier | hist)
    cached = np.load(s.RUN / 'all_indices.npy'); replay = np.setdiff1d(train, cached)
    np.save(RUN / 'train_indices.npy', train, allow_pickle=False); np.save(RUN / 'train_labels.npy', labels, allow_pickle=False); np.save(RUN / 'replay_indices.npy', replay, allow_pickle=False)
    r = {'owners': int(k), 'games': len(train), 'cached_reused': int(len(train) - len(replay)), 'replay': int(len(replay)), 'per_owner_max': PER_OWNER, 'min_games': MIN_GAMES,
         'B_fraction': float((m['colors'][train] == 0).mean()), 'excluded_eval_identities': True, 'families_disjoint': True, 'seed': SEED,
         'source_SHA256': p.b.sha(__file__)}
    p.b.write(RUN / 'SCOPE.json', r); print(r)


def package():
    import remote_metric64_capacity_20261112_v1 as remote
    out = RUN / 'payload_dev'; out.mkdir(exist_ok=False); source = s.RUN / 'payload_dev'
    for f in ('worker.py', 'basis64.py', 'model_config.json'): shutil.copy2(source / f, out / f)
    for f in ('vendor', 'code'): shutil.copytree(source / f, out / f, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert p.b.sha(out / 'worker.py') == '3be16a208e7c804bb906a1dc4e2135ec335115901a95713b7b02d231abb7650b'
    with (out / 'supervisor.py').open('x', encoding='utf-8', newline='\n') as f: f.write(remote.SUPERVISOR)
    ids = np.load(RUN / 'replay_indices.npy'); m = p.b.meta(); from goid_v4.pilot_scope_v1 import PointReader
    reader = PointReader(p.ROOT / 'data/processed/audit.sqlite')
    try:
        with (out / 'games.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for st in range(0, len(ids), 256):
                ii = ids[st:st + 256]; texts = {r['row_id']: r['sgf_content'] for r in reader.sample_text(m['rows'][ii])}
                for i in ii:
                    text = texts[int(m['rows'][i])]; assert hashlib.sha256(text.encode()).hexdigest() == m['raw'][i].decode()
                    stream.write(json.dumps({'index': int(i), 'sgf': text, 'raw': m['raw'][i].decode(), 'color': 'BW'[int(m['colors'][i])], 'hd': int(m['hd'][i]), 'canonical': m['canonical'][i].decode()}) + '\n')
    finally: reader.close()
    p.b.write(out / 'MANIFEST.json', {'files': {f.relative_to(out).as_posix(): p.b.sha(f) for f in out.rglob('*') if f.is_file()}, 'weight_SHA256': p.b.sha(p.n.RUN / 'HumanSL_state_only.pt'), 'role': 'originalsafe_bigbank90_forward_only', 'games': len(ids), 'fit_or_optimizer': False, 'worker_exact_priorcomplete90': True})
    archive = RUN / 'DEV_PAYLOAD.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=3) as tf:
        for f in out.rglob('*'):
            if f.is_file(): tf.add(f, arcname=f.relative_to(out).as_posix(), recursive=False)
    r = {'archive': str(archive), 'SHA256': p.b.sha(archive), 'bytes': archive.stat().st_size, 'games': len(ids)}; p.b.write(RUN / 'DEV_PAYLOAD.json', r); print(r)


def accept(expected, size):
    archive = RUN / 'DEV_RETURN.tar.gz'; assert p.b.sha(archive) == expected and archive.stat().st_size == int(size); out = RUN / 'dev_return'; out.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf: tf.extractall(out, filter='data')
    manifest = p.b.read(out / 'RETURN_MANIFEST.json'); assert all(p.b.sha(out / f) == hh for f, hh in manifest.items())
    assert p.b.read(out / 'REMOTE_JOB.json')['archive_SHA256'] == p.b.read(RUN / 'DEV_PAYLOAD.json')['SHA256']
    rel = p.b.read(out / 'GPU_RELEASE.json'); assert rel['worker_exit'] == 0 and rel['owned_PID_absent']
    dest = RUN / 'dev_results'; dest.mkdir(exist_ok=False)
    for f in (out / 'results').iterdir(): shutil.copy2(f, dest / f.name)
    assert p.b.read(dest / 'RECEIPT.json')['all_valid']; p.b.write(RUN / 'DEV_RETURN_VERIFIED.json', {'SHA256': expected, 'bytes': int(size), 'release': rel})


def lookup_views(indices, view):
    """view rows for indices from scale-run assembled views or this run's replay."""
    a = np.load(s.RUN / 'all_indices.npy'); va = np.load(s.RUN / f'views/view{view}.npy', mmap_mode='r'); pa = {int(i): j for j, i in enumerate(a)}
    b = np.load(RUN / 'dev_results/indices.npy') if (RUN / 'dev_results').exists() else np.zeros(0, np.int64)
    vb = np.load(RUN / f'dev_results/view{view}.npy', mmap_mode='r') if len(b) else None; pb = {int(i): j for j, i in enumerate(b)}
    out = np.empty((len(indices), 2339), np.float32)
    for k, i in enumerate(indices):
        i = int(i); out[k] = va[pa[i]] if i in pa else vb[pb[i]]
    return out


def train(dims=(256, 768, 1024)):
    ids = np.load(RUN / 'train_indices.npy'); labels = np.load(RUN / 'train_labels.npy'); dest = RUN / 'models'; dest.mkdir(exist_ok=True)
    raw = lookup_views(ids, 90)
    for dim in dims:
        folder = dest / f'big_raw{dim}_v90'; folder.mkdir(exist_ok=False); x = old.pre(raw, folder, True)
        r = p.fit_plda(x, labels, folder / 'PLDA.npz', dimension=dim); p.b.write(folder / 'FIT_RECEIPT.json', dict(r, training_only=True)); print(dim, r)


def screen():
    res = {}
    for part in ('screen', 'confirmation'):
        ep = p.b.read(s.RUN / f'EPISODE_{part}.json'); ids = np.load(s.RUN / 'cal_indices.npy'); allids = np.load(s.RUN / 'all_indices.npy'); at = np.searchsorted(allids, ids)
        raw = np.load(s.RUN / 'views/view90.npy', mmap_mode='r')[at]
        comps = {}
        for folder in sorted((RUN / 'models').iterdir()):
            comps[folder.name] = old.component(folder, raw, ids, ep)
        for name in ('diverse1024_raw256_v90', 'diverse1024_raw768_v90'):
            comps[name] = old.component(s.RUN / 'models' / name, raw, ids, ep)
        for k, v in comps.items():
            r = cap.compare(v, comps['diverse1024_raw768_v90'], ep, 'all')
            res[f'{part}:{k}'] = {'score': r['macro_score_all'], 'delta_vs_div768': r['delta'], 'CI95': r['CI95'], 'by_games': r['by_games_delta'], 'Top1': r['Top1']}
            print(part, k, round(r['macro_score_all'], 4), round(r['delta'], 4), [round(x, 4) for x in r['CI95']], {a: round(b, 4) for a, b in r['by_games_delta'].items()})
        np.savez(RUN / f'components_{part}.npz', **comps)
    p.b.write(RUN / 'SCREEN_COMPONENTS.json', res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'package': package, 'assemble': lambda: None, 'train': train, 'screen': screen}.get(v.stage, lambda: accept(*v.args))()
