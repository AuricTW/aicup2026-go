"""Bigbank v5: the last unused training players -- old 'holdout' split (840 owners) and 'holdout_confirmation' (the sealed 400).
User authorisation 2026-11-15 ("add them if you think it is appropriate"): they join the FINAL training banks only.
They are never used as evaluation targets and no result on them is read; the sealed-400 confirmation records stay untouched.
Index space: EXT + sqlite row_id (same convention as bigbank4).  EXTMETA.npz = meta-like arrays for all parse-ok games of these owners.
Up to PER_OWNER games per owner (colour-alternating, family-disjoint against bigbank3/4 picks and each other), min 4.
Stages: prepare | package CHUNK | accept CHUNK SHA SIZE
"""
from pathlib import Path
import sys, argparse, json, shutil, tarfile, hashlib, re
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank4_90_20261115_v1 as b4
import numpy as np
from threadpoolctl import threadpool_limits
b2 = b4.b2; s = b4.s; p = b4.p; EXT = b4.EXT
RUN = p.ROOT / 'runs/bigbank5_90_20261115_v1'; SEED = 2026111503; PER_OWNER = 64; MIN_GAMES = 4; TIERS = (16, 40, 64); CHUNK = 22000; SPLITNAMES = ('holdout', 'holdout_confirmation')


def h(v): return hashlib.sha256((str(SEED) + '|' + str(v)).encode()).digest()


def extmeta():
    with np.load(RUN / 'EXTMETA.npz', allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}


class Universe:
    """SAFE meta + bigbank4 EXTMETA + bigbank5 EXTMETA, addressed by SAFE index or EXT + row_id."""
    KEYS = ('owners', 'ranks', 'colors', 'hd', 'raw', 'move', 'canonical', 'group')

    def __init__(self):
        m = p.b.meta(); e4 = b4.extmeta(); e5 = extmeta(); self.n = len(m['owners']); rows = np.concatenate([e4['rows'], e5['rows']]); assert len(set(rows.tolist())) == len(rows)
        self.pos = {int(r): self.n + j for j, r in enumerate(rows)}; self.a = {k: np.concatenate([m[k], e4[k].astype(m[k].dtype), e5[k].astype(m[k].dtype)]) for k in self.KEYS}; self.rows = np.concatenate([m['rows'], rows])

    def at(self, ids): return np.array([int(i) if int(i) < EXT else self.pos[int(i) - EXT] for i in ids], np.int64)

    def get(self, key, ids): return self.a[key][self.at(ids)]


def extmeta_build():
    """first step of prepare: EXTMETA.npz = meta-like arrays of every parse-ok game of the holdout + sealed-400 owners"""
    m = p.b.meta(); sp = json.loads(b4.SPLITS.read_text(encoding='utf-8')); pl = sorted(k for k, v in sp.items() if v in SPLITNAMES); assert len(pl) == 1240; con = b4.db(); rows = []
    for st in range(0, len(pl), 500):
        q = pl[st:st + 500]
        rows += con.execute(f'SELECT row_id,player_id,rank,color,raw_hash,move_hash,canonical_hash,COALESCE(split_group_hash,canonical_hash),mainline_moves,sgf_content FROM games WHERE parse_ok=1 AND player_id IN ({",".join("?" for _ in q)})', q).fetchall()
    con.close(); rows.sort(key=lambda r: r[0]); hdre = re.compile(r'HD\[(\d+)\]')
    e = {'rows': np.array([r[0] for r in rows], np.int64), 'owners': np.array([b4.opaque(r[1]) for r in rows], 'S64'), 'ranks': np.array([r[2] for r in rows], '<U2'), 'colors': np.array([0 if r[3] == 'B' else 1 for r in rows], np.uint8),
         'raw': np.array([r[4] for r in rows], 'S64'), 'move': np.array([r[5] for r in rows], 'S64'), 'canonical': np.array([r[6] for r in rows], 'S64'), 'group': np.array([r[7] for r in rows], 'S64'), 'moves': np.array([r[8] for r in rows], np.uint16),
         'hd': np.array([int(x.group(1)) if (x := hdre.search(r[9][:40])) else 0 for r in rows], np.uint8), 'sealed400': np.array([sp[r[1]] == 'holdout_confirmation' for r in rows])}
    assert all(r[3] in 'BW' for r in rows); np.savez(RUN / 'EXTMETA.npz', **e); return m, e


def prepare():
    RUN.mkdir(exist_ok=False); m, e = extmeta_build(); U = Universe(); uo = U.a['owners'].astype(str); n5 = len(e['rows']); base = len(uo) - n5
    assert not set(uo[base:]) & set(uo[:base]); by = defaultdict(list)
    for j in np.flatnonzero(e['hd'] <= 1): by[uo[base + j]].append(EXT + int(e['rows'][j]))
    prior = np.concatenate([np.load(b4.B3RUN / 'train_indices.npy'), np.load(b4.RUN / 'train_indices.npy')]); used = {f: set(U.get(f, prior).tolist()) for f in p.b.FAMILIES}; train, own, rank_in_owner = [], [], []
    for o in sorted(by, key=h):
        picked = []; local = {f: set() for f in p.b.FAMILIES}; cols = {0: [], 1: []}; at = U.at(by[o]); fam = {f: U.a[f][at] for f in p.b.FAMILIES}; col = U.a['colors'][at]; grp = U.a['group'][at]
        for n, i in enumerate(by[o]): cols[int(col[n])].append(n)
        for c in (0, 1): cols[c].sort(key=lambda n: h(grp[n]))
        order = [x for pair in zip(cols[0], cols[1]) for x in pair] + cols[0][len(cols[1]):] + cols[1][len(cols[0]):]
        for n in order:
            if any(fam[f][n] in used[f] or fam[f][n] in local[f] for f in p.b.FAMILIES): continue
            picked.append(by[o][n])
            for f in p.b.FAMILIES: local[f].add(fam[f][n])
            if len(picked) == PER_OWNER: break
        if len(picked) < MIN_GAMES: continue
        for f in p.b.FAMILIES: used[f].update(local[f])
        train += picked; own += [o] * len(picked); rank_in_owner += list(range(len(picked)))
    train = np.asarray(train, np.int64); tier = np.searchsorted(np.array(TIERS), np.asarray(rank_in_owner), side='right').astype(np.int8); assert len(set(train.tolist())) == len(train) and not set(train.tolist()) & set(prior.tolist()); chunks = []; ctier = []
    for t in range(len(TIERS)):
        ii = np.array(sorted(train[tier == t].tolist(), key=h), np.int64)
        for st in range(0, len(ii), CHUNK): chunks.append(ii[st:st + CHUNK]); ctier.append(t)
    for n, c in enumerate(chunks): np.save(RUN / f'replay_chunk{n}.npy', np.sort(c), allow_pickle=False)
    np.save(RUN / 'train_indices.npy', train, allow_pickle=False); np.save(RUN / 'train_owners.npy', np.array(own, 'S64'), allow_pickle=False); np.save(RUN / 'train_tier.npy', tier, allow_pickle=False); gp = np.unique(own, return_counts=True)[1]
    sealed = set(uo[base:][e['sealed400']]); r = {'owners': int(len(gp)), 'owners_sealed400': len(set(own) & sealed), 'owners_holdout': len(set(own) - sealed), 'games': int(len(train)), 'games_by_tier': np.bincount(tier, minlength=len(TIERS)).tolist(), 'replay_chunks': [int(len(c)) for c in chunks], 'chunk_tiers': ctier,
         'games_per_owner_q': np.quantile(gp, [0, .1, .5, .9, 1]).tolist(), 'ext_games_all': int(n5), 'authorisation': 'user 2026-11-15: holdout 840 + sealed 400 may join final training', 'per_owner_max': PER_OWNER, 'min_games': MIN_GAMES, 'tiers': TIERS, 'seed': SEED, 'source_SHA256': p.b.sha(__file__)}
    p.b.write(RUN / 'SCOPE.json', r); print(json.dumps(r, indent=1))


def package(chunk):
    import remote_metric64_capacity_20261112_v1 as remote
    chunk = int(chunk); out = RUN / f'payload_c{chunk}'; out.mkdir(exist_ok=False); source = s.RUN / 'payload_dev'
    for f in ('worker.py', 'basis64.py', 'model_config.json'): shutil.copy2(source / f, out / f)
    for f in ('vendor', 'code'): shutil.copytree(source / f, out / f, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert p.b.sha(out / 'worker.py') == '3be16a208e7c804bb906a1dc4e2135ec335115901a95713b7b02d231abb7650b'
    with (out / 'supervisor.py').open('x', encoding='utf-8', newline='\n') as f: f.write(remote.SUPERVISOR)
    ids = np.load(RUN / f'replay_chunk{chunk}.npy'); U = Universe(); at = U.at(ids); con = b4.db()
    try:
        with (out / 'games.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for st in range(0, len(ids), 500):
                a = at[st:st + 500]; rr = [int(x) for x in U.rows[a]]; texts = dict(con.execute(f'SELECT row_id,sgf_content FROM games WHERE row_id IN ({",".join("?" for _ in rr)})', rr).fetchall()); assert len(texts) == len(rr)
                for i, x, r in zip(ids[st:st + 500], a, rr):
                    text = texts[r]; assert hashlib.sha256(text.encode()).hexdigest() == U.a['raw'][x].decode()
                    stream.write(json.dumps({'index': int(i), 'sgf': text, 'raw': U.a['raw'][x].decode(), 'color': 'BW'[int(U.a['colors'][x])], 'hd': int(U.a['hd'][x]), 'canonical': U.a['canonical'][x].decode()}) + '\n')
    finally: con.close()
    p.b.write(out / 'MANIFEST.json', {'files': {f.relative_to(out).as_posix(): p.b.sha(f) for f in out.rglob('*') if f.is_file()}, 'weight_SHA256': p.b.sha(p.n.RUN / 'HumanSL_state_only.pt'), 'role': f'bigbank5_90_chunk{chunk}_forward_only', 'games': len(ids), 'fit_or_optimizer': False, 'worker_exact_priorcomplete90': True})
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
