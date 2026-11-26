"""Bigbank v6 (token replay): MORE games for players already in the training banks.
Why: the per-decision token net is data-hungry (40k games 0.638 -> 379k games 0.714 on validation v2) and only 421k of the ~1.0M
games of the 8,007 training owners have been used (banks capped owners at 16..160 games).  558k more family-disjoint HD<=1 games
exist, concentrated on 2,527 owners.  Here: up to EXTRA_CAP more games per owner (colour-alternating, hash order, family-disjoint
against every labelled game and each other), replay chunks of <= CHUNK games with the SAME frozen worker; tokens are added by
tokfull_vm_20261116_v1.py (VM) or tokpilot_20261116_v1.py (local).  Labels: same owner -> class mapping as labels_all.npz.
Stages: prepare [EXTRA_CAP] | package CHUNK
"""
from pathlib import Path
import sys, json, shutil, hashlib
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank5_90_20261115_v1 as b5
import numpy as np
from threadpoolctl import threadpool_limits
b4 = b5.b4; p = b5.p; s = b4.s; EXT = 1_000_000; RUN = p.ROOT / 'runs/bigbank6_tok_20261117_v1'; SEED = 2026111706; CHUNK = 22000; LAB = p.ROOT / 'runs/tokpilot_20261116_v1/labels_all.npz'


def h(v): return hashlib.sha256((str(SEED) + '|' + str(v)).encode()).digest()


def prepare(extra_cap=100):
    extra_cap = int(extra_cap); RUN.mkdir(exist_ok=False); U = b5.Universe(); n = len(U.rows); nm = U.n; idx = np.array([i if i < nm else EXT + int(U.rows[i]) for i in range(n)], np.int64)
    L = np.load(LAB); lab = L['ids']; labset = set(lab.tolist()); owners = U.a['owners'].astype(str); lo = owners[U.at(lab)]
    u = np.unique(lo); ymap = {o: k for k, o in enumerate(u)}; assert np.array_equal(np.array([ymap[o] for o in lo]), L['y']), 'owner -> class mapping differs from labels_all'
    used = {f: set(U.a[f][U.at(lab)].tolist()) for f in p.b.FAMILIES}; by = defaultdict(list)
    for i in np.flatnonzero((U.a['hd'] <= 1) & np.isin(owners, u)):
        if idx[i] not in labset: by[owners[i]].append(int(i))
    picked, own = [], []
    for o in sorted(by, key=h):
        ii = by[o]; cols = {0: [], 1: []}
        for i in ii: cols[int(U.a['colors'][i])].append(i)
        for c in (0, 1): cols[c].sort(key=lambda i: h(U.a['group'][i]))
        order = [x for pair in zip(cols[0], cols[1]) for x in pair] + cols[0][len(cols[1]):] + cols[1][len(cols[0]):]; k = 0
        for i in order:
            if any(U.a[f][i] in used[f] for f in p.b.FAMILIES): continue
            for f in p.b.FAMILIES: used[f].add(U.a[f][i])
            picked.append(int(idx[i])); own.append(o); k += 1
            if k == extra_cap: break
    picked = np.array(picked, np.int64); order = np.array(sorted(range(len(picked)), key=lambda j: h(int(picked[j])))); picked = picked[order]; own = np.array(own)[order]
    chunks = [np.sort(picked[st:st + CHUNK]) for st in range(0, len(picked), CHUNK)]
    for k, c in enumerate(chunks): np.save(RUN / f'replay_chunk{k}.npy', c, allow_pickle=False)
    blk = U.a['colors'][U.at(picked)] == 0; yy = np.array([ymap[o] for o in own], np.int32)
    np.savez(RUN / 'labels_extra.npz', ids=picked, y=yy, black=blk); np.savez(RUN / 'labels_all6.npz', ids=np.concatenate([L['ids'], picked]), y=np.concatenate([L['y'], yy]), black=np.concatenate([L['black'], blk]), off_black=L['off_black'])
    gp = np.bincount(yy, minlength=len(u)); r = {'extra_cap': extra_cap, 'extra_games': int(len(picked)), 'owners_with_extra': int((gp > 0).sum()), 'extra_per_owner_q': np.quantile(gp[gp > 0], [0, .1, .5, .9, 1]).tolist(), 'chunks': [int(len(c)) for c in chunks], 'black_share': float(blk.mean()), 'seed': SEED, 'source_SHA256': p.b.sha(__file__)}
    p.b.write(RUN / 'SCOPE.json', r); print(json.dumps(r, indent=1))


def package(chunk):
    import remote_metric64_capacity_20261112_v1 as remote
    chunk = int(chunk); out = RUN / f'payload_c{chunk}'; out.mkdir(exist_ok=False); source = s.RUN / 'payload_dev'
    for f in ('worker.py', 'basis64.py', 'model_config.json'): shutil.copy2(source / f, out / f)
    for f in ('vendor', 'code'): shutil.copytree(source / f, out / f, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    assert p.b.sha(out / 'worker.py') == '3be16a208e7c804bb906a1dc4e2135ec335115901a95713b7b02d231abb7650b'
    with (out / 'supervisor.py').open('x', encoding='utf-8', newline='\n') as f: f.write(remote.SUPERVISOR)
    ids = np.load(RUN / f'replay_chunk{chunk}.npy'); U = b5.Universe(); at = U.at(ids); con = b4.db()
    try:
        with (out / 'games.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for st in range(0, len(ids), 500):
                a = at[st:st + 500]; rr = [int(x) for x in U.rows[a]]; texts = dict(con.execute(f'SELECT row_id,sgf_content FROM games WHERE row_id IN ({",".join("?" for _ in rr)})', rr).fetchall()); assert len(texts) == len(rr)
                for i, x, r in zip(ids[st:st + 500], a, rr):
                    text = texts[r]; assert hashlib.sha256(text.encode()).hexdigest() == U.a['raw'][x].decode()
                    stream.write(json.dumps({'index': int(i), 'sgf': text, 'raw': U.a['raw'][x].decode(), 'color': 'BW'[int(U.a['colors'][x])], 'hd': int(U.a['hd'][x]), 'canonical': U.a['canonical'][x].decode()}) + '\n')
    finally: con.close()
    p.b.write(out / 'MANIFEST.json', {'files': {f.relative_to(out).as_posix(): p.b.sha(f) for f in out.rglob('*') if f.is_file()}, 'weight_SHA256': p.b.sha(p.n.RUN / 'HumanSL_state_only.pt'), 'role': f'bigbank6_tok_chunk{chunk}_forward_only', 'games': len(ids), 'fit_or_optimizer': False, 'worker_exact_priorcomplete90': True}); print('packaged', chunk, len(ids), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        if sys.argv[1] == 'prepare': prepare(*sys.argv[2:])
        else:
            for c in sys.argv[2:]: package(c)
