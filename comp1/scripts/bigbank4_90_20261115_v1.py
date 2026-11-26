"""Bigbank v4: training players that no bank has used so far (view90 replay on the 4090, same frozen worker).
 (a) 'hist': 723 fit-split owners that bigbank2/3 excluded as historical validation identities (320k games, inside SAFE meta).
 (b) 'val' : 1,993 owners of the old 'validation' player split (251k games, NOT in SAFE meta; read from audit.sqlite).
Untouched: 'holdout' (840 owners) and 'holdout_confirmation' (the sealed 400).  Split source: runs/holdout_protocol_v1/effective_player_splits.json.
Index space: SAFE-meta index for (a); EXT + sqlite row_id for (b).  EXTMETA.npz holds meta-like arrays for ALL parse-ok 'val' games.
Up to PER_OWNER games per owner (colour-alternating, family-disjoint against bigbank3 picks and each other), min 4.
Replay tiers: first 16 per owner, then up to 40, then up to 64; chunks of <= CHUNK games.
Stages: prepare | package CHUNK | accept CHUNK SHA SIZE
"""
from pathlib import Path
import sys, argparse, json, shutil, tarfile, hashlib, sqlite3, re
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank2_90_20261114_v1 as b2
import numpy as np
from threadpoolctl import threadpool_limits
s = b2.s; p = b2.p; old = b2.old; bb = b2.bb
B3RUN = p.ROOT / 'runs/bigbank3_90_20261114_v1'; RUN = p.ROOT / 'runs/bigbank4_90_20261115_v1'; SEED = 2026111501; PER_OWNER = 64; MIN_GAMES = 4; TIERS = (16, 40, 64); CHUNK = 22000; EXT = 1_000_000
DB = p.ROOT / 'data/processed/audit.sqlite'; SPLITS = p.ROOT / 'runs/holdout_protocol_v1/effective_player_splits.json'


def h(v): return hashlib.sha256((str(SEED) + '|' + str(v)).encode()).digest()


def opaque(player): return hashlib.sha256(('goid-v4-player-v1|' + player).encode()).hexdigest()


def db(): return sqlite3.connect(DB.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)


def extmeta():
    with np.load(RUN / 'EXTMETA.npz', allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


class Universe:
    """meta-like arrays addressed by the mixed index space (SAFE index or EXT + row_id)."""
    def __init__(self):
        m = p.b.meta(); e = extmeta(); self.n = len(m['owners']); self.pos = {int(r): self.n + j for j, r in enumerate(e['rows'])}
        self.a = {k: np.concatenate([m[k], e[k].astype(m[k].dtype)]) for k in ('owners', 'ranks', 'colors', 'hd', 'raw', 'move', 'canonical', 'group')}
        self.rows = np.concatenate([m['rows'], e['rows']])

    def at(self, ids): return np.array([int(i) if int(i) < EXT else self.pos[int(i) - EXT] for i in ids], np.int64)

    def get(self, key, ids): return self.a[key][self.at(ids)]


def extmeta_build():
    """first step of prepare: EXTMETA.npz = meta-like arrays of every parse-ok game of the old 'validation' owners"""
    m = p.b.meta(); owners = m['owners'].astype(str); sp = json.loads(SPLITS.read_text(encoding='utf-8'))
    val = sorted(k for k, v in sp.items() if v == 'validation'); assert len(val) == 1993; con = db(); rows = []
    for st in range(0, len(val), 500):
        q = val[st:st + 500]
        rows += con.execute(f'SELECT row_id,player_id,rank,color,raw_hash,move_hash,canonical_hash,COALESCE(split_group_hash,canonical_hash),mainline_moves,sgf_content FROM games WHERE parse_ok=1 AND player_id IN ({",".join("?" for _ in q)})', q).fetchall()
    con.close(); rows.sort(key=lambda r: r[0]); hdre = re.compile(r'HD\[(\d+)\]')
    e = {'rows': np.array([r[0] for r in rows], np.int64), 'owners': np.array([opaque(r[1]) for r in rows], 'S64'), 'ranks': np.array([r[2] for r in rows], '<U2'), 'colors': np.array([0 if r[3] == 'B' else 1 for r in rows], np.uint8),
         'raw': np.array([r[4] for r in rows], 'S64'), 'move': np.array([r[5] for r in rows], 'S64'), 'canonical': np.array([r[6] for r in rows], 'S64'), 'group': np.array([r[7] for r in rows], 'S64'), 'moves': np.array([r[8] for r in rows], np.uint16),
         'hd': np.array([int(x.group(1)) if (x := hdre.search(r[9][:40])) else 0 for r in rows], np.uint8)}
    assert all(r[3] in 'BW' for r in rows) and not set(e['owners'].astype(str)) & set(owners); np.savez(RUN / 'EXTMETA.npz', **e); return m, owners, e


def prepare():
    RUN.mkdir(exist_ok=False); m, owners, e = extmeta_build()
    hist =set(owners[np.load(p.DENSE.parent / 'validation_indices.npy')]); U = Universe(); uo = U.a['owners'].astype(str)
    # candidates: hist owners (meta index) and val owners (EXT + row_id), HD <= 1
    by = defaultdict(list)
    for i in np.flatnonzero(np.isin(owners, sorted(hist)) & (m['hd'] <= 1)): by[owners[i]].append(int(i))
    for j in np.flatnonzero(e['hd'] <= 1): by[uo[U.n + j]].append(EXT + int(e['rows'][j]))
    b3 = np.load(B3RUN / 'train_indices.npy'); used = {f: set(m[f][b3].tolist()) for f in p.b.FAMILIES}; cached = b2.cache_index(); train, own, rank_in_owner = [], [], []
    for o in sorted(by, key=h):
        picked = []; local = {f: set() for f in p.b.FAMILIES}; cols = {0: [], 1: []}; at = U.at(by[o]); fam = {f: U.a[f][at] for f in p.b.FAMILIES}; col = U.a['colors'][at]; grp = U.a['group'][at]
        for n, i in enumerate(by[o]): cols[int(col[n])].append(n)
        for c in (0, 1): cols[c].sort(key=lambda n: (by[o][n] not in cached, h(grp[n])))
        order = [x for pair in zip(cols[0], cols[1]) for x in pair] + cols[0][len(cols[1]):] + cols[1][len(cols[0]):]
        for n in order:
            if any(fam[f][n] in used[f] or fam[f][n] in local[f] for f in p.b.FAMILIES): continue
            picked.append(by[o][n])
            for f in p.b.FAMILIES: local[f].add(fam[f][n])
            if len(picked) == PER_OWNER: break
        if len(picked) < MIN_GAMES: continue
        for f in p.b.FAMILIES: used[f].update(local[f])
        train += picked; own += [o] * len(picked); rank_in_owner += list(range(len(picked)))
    train = np.asarray(train, np.int64); rk = np.asarray(rank_in_owner); tier = np.searchsorted(np.array(TIERS), rk, side='right').astype(np.int8)
    assert len(set(train.tolist())) == len(train) and not set(train.tolist()) & set(b3.tolist())
    need = np.array([int(i) not in cached for i in train]); chunks = []; ctier = []
    for t in range(len(TIERS)):
        ii = np.array(sorted(train[need & (tier == t)].tolist(), key=h), np.int64)
        for st in range(0, len(ii), CHUNK): chunks.append(ii[st:st + CHUNK]); ctier.append(t)
    for n, c in enumerate(chunks): np.save(RUN / f'replay_chunk{n}.npy', np.sort(c), allow_pickle=False)
    np.save(RUN / 'train_indices.npy', train, allow_pickle=False); np.save(RUN / 'train_owners.npy', np.array(own, 'S64'), allow_pickle=False); np.save(RUN / 'train_tier.npy', tier, allow_pickle=False)
    gp = np.unique(own, return_counts=True)[1]; ishist = np.array([o in hist for o in own])
    r = {'owners': int(len(gp)), 'owners_hist': len(set(np.array(own)[ishist])), 'owners_val': len(set(np.array(own)[~ishist])), 'games': int(len(train)), 'games_by_tier': np.bincount(tier, minlength=len(TIERS)).tolist(), 'cached_reused': int((~need).sum()),
         'replay_total': int(need.sum()), 'replay_chunks': [int(len(c)) for c in chunks], 'chunk_tiers': ctier, 'games_per_owner_q': np.quantile(gp, [0, .1, .5, .9, 1]).tolist(), 'ext_val_games_all': int(len(e['rows'])),
         'untouched_splits': 'holdout (840 owners) and holdout_confirmation (sealed 400)', 'per_owner_max': PER_OWNER, 'min_games': MIN_GAMES, 'tiers': TIERS, 'seed': SEED, 'source_SHA256': p.b.sha(__file__)}
    p.b.write(RUN / 'SCOPE.json', r); print(json.dumps(r, indent=1))


def package(chunk):
    import remote_metric64_capacity_20261112_v1 as remote
    chunk = int(chunk); out = RUN / f'payload_c{chunk}'; out.mkdir(exist_ok=False); source = s.RUN / 'payload_dev'
    for f in ('worker.py', 'basis64.py', 'model_config.json'): shutil.copy2(source / f, out / f)
    for f in ('vendor', 'code'): shutil.copytree(source / f, out / f, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert p.b.sha(out / 'worker.py') == '3be16a208e7c804bb906a1dc4e2135ec335115901a95713b7b02d231abb7650b'
    with (out / 'supervisor.py').open('x', encoding='utf-8', newline='\n') as f: f.write(remote.SUPERVISOR)
    ids = np.load(RUN / f'replay_chunk{chunk}.npy'); U = Universe(); at = U.at(ids); con = db()
    try:
        with (out / 'games.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for st in range(0, len(ids), 500):
                a = at[st:st + 500]; rr = [int(x) for x in U.rows[a]]; texts = dict(con.execute(f'SELECT row_id,sgf_content FROM games WHERE row_id IN ({",".join("?" for _ in rr)})', rr).fetchall()); assert len(texts) == len(rr)
                for i, x, r in zip(ids[st:st + 500], a, rr):
                    text = texts[r]; assert hashlib.sha256(text.encode()).hexdigest() == U.a['raw'][x].decode()
                    stream.write(json.dumps({'index': int(i), 'sgf': text, 'raw': U.a['raw'][x].decode(), 'color': 'BW'[int(U.a['colors'][x])], 'hd': int(U.a['hd'][x]), 'canonical': U.a['canonical'][x].decode()}) + '\n')
    finally: con.close()
    p.b.write(out / 'MANIFEST.json', {'files': {f.relative_to(out).as_posix(): p.b.sha(f) for f in out.rglob('*') if f.is_file()}, 'weight_SHA256': p.b.sha(p.n.RUN / 'HumanSL_state_only.pt'), 'role': f'bigbank4_90_chunk{chunk}_forward_only', 'games': len(ids), 'fit_or_optimizer': False, 'worker_exact_priorcomplete90': True})
    archive = RUN / f'PAYLOAD_c{chunk}.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=3) as tf:
        for f in out.rglob('*'):
            if f.is_file(): tf.add(f, arcname=f.relative_to(out).as_posix(), recursive=False)
    r = {'archive': str(archive), 'SHA256': p.b.sha(archive), 'bytes': archive.stat().st_size, 'games': len(ids)}; p.b.write(RUN / f'PAYLOAD_c{chunk}.json', r); print(r)


def accept(chunk, expected, size):
    b2.RUN = RUN; b2.accept(chunk, expected, size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('args', nargs='*'); v = ap.parse_args()
    with threadpool_limits(limits=4):
        {'prepare': prepare, 'package': lambda: package(*v.args), 'accept': lambda: accept(*v.args)}[v.stage]()
