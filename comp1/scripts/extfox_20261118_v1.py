"""External identities for the token nets: Fox game records (GitHub featurecat/go-dataset, GPL-3.0; registered in docs/SOURCES.md).
Why (docs/20261115_fullbank_seqid.md section 29): the number of training PLAYERS is the one data axis that still scales (about +0.02 per
doubling, single model) and every competition training player is already used.
Compliance: the files are untrusted data and are parsed as BYTES only (never executed, never decoded as text).  Account names are never
stored or printed: the raw name bytes go straight into a salted SHA-256 and only the first 8 bytes (uint64) are kept.  External players
are used ONLY as extra training classes; nothing here (or downstream) compares competition games with external games.
Archives are streamed (Windows tar.exe = libarchive reads 7z; `tar -xOf` concatenates the member files) - no millions of small files.
Stages
  scan <archive.7z> <meta.npz>      one record per game in stream order: pb, pw (uint64 hashes), br, wr (rank code: kyu = -k, dan = +d,
                                    pro = 10 + d, 0 unknown), year, ha, setup (AB/AW stones), sz19, nmv (moves), bad (a move outside
                                    a..s incl. pass, or broken record)
  select <out dir> <meta.npz>...    choose accounts and games (see select()) -> PLAN.json + per-archive ordinals
  export <out dir> <archive.7z>...  second stream pass: write the chosen games as games.jsonl records of the existing pipeline
                                    ({index, sgf '(;SZ[19];B[..];W[..]...)', raw, color, hd, canonical}) + labels
"""
import sys, re, os, json, hashlib, subprocess, time
from pathlib import Path
import numpy as np
SALT = b'fox20261118|'; TAR = __import__('comp1_paths').get('tar_exe', 'tar'); START = b'(;GM['
PROP = re.compile(rb'([A-Z]{1,2})\[((?:[^\]\\]|\\.)*)\]'); MV = re.compile(rb';([BW])\[([^\]]*)\]'); GOOD = re.compile(rb'^[a-s][a-s]$')


def h64(b): return int.from_bytes(hashlib.sha256(SALT + b).digest()[:8], 'little')


def rank(b):     # b'9\xe6\xae\xb5' = 9 dan, b'5\xe7\xba\xa7' = 5 kyu, b'P3\xe6\xae\xb5' = pro 3 dan
    m = re.match(rb'(P?)(\d+)', b)
    if not m: return 0
    n = int(m.group(2))
    if m.group(1): return 10 + min(n, 9)
    return min(n, 9) if b'\xe6\xae\xb5' in b else -min(n, 30) if b'\xe7\xba\xa7' in b else 0


def records(archive):
    """yield the raw bytes of every game record of the archive, in stream order"""
    p = subprocess.Popen([TAR, '-xOf', str(archive)], stdout=subprocess.PIPE, bufsize=1 << 22); buf = b''
    while True:
        chunk = p.stdout.read(1 << 22)
        if not chunk: break
        buf += chunk; parts = buf.split(START); buf = START + parts[-1] if len(parts) > 1 else buf
        for q in parts[1:-1]: yield START + q
    if buf.startswith(START): yield buf
    assert p.wait() == 0, 'tar failed'


def parse(rec):
    m = MV.search(rec); head = rec[:m.start()] if m else rec; d = {}
    for k, v in PROP.findall(head): d.setdefault(k, v)
    mv = MV.findall(rec); alt = all(c == (b'B', b'W')[i % 2] for i, (c, _) in enumerate(mv)); bad = (not mv) or (not alt) or any(not GOOD.match(x) for _, x in mv) or not d.get(b'PB') or not d.get(b'PW')
    y = d.get(b'DT', b'')[:4]; ha = d.get(b'HA', b'0')
    return d, mv, {'pb': h64(d.get(b'PB', b'')), 'pw': h64(d.get(b'PW', b'')), 'br': rank(d.get(b'BR', b'')), 'wr': rank(d.get(b'WR', b'')), 'year': int(y) if y.isdigit() else 0, 'ha': int(ha) if ha.isdigit() and int(ha) < 250 else 255,
                   'setup': b'AB' in d or b'AW' in d, 'sz19': d.get(b'SZ') == b'19', 'nmv': min(len(mv), 32000), 'bad': bool(bad)}


def scan(archive, out):
    t0 = time.time(); cols = {k: [] for k in ('pb', 'pw', 'br', 'wr', 'year', 'ha', 'setup', 'sz19', 'nmv', 'bad')}
    for rec in records(archive):
        _, _, r = parse(rec)
        for k in cols: cols[k].append(r[k])
    dt = {'pb': np.uint64, 'pw': np.uint64, 'br': np.int8, 'wr': np.int8, 'year': np.int16, 'ha': np.uint8, 'setup': bool, 'sz19': bool, 'nmv': np.int16, 'bad': bool}
    A = {k: np.asarray(v, dt[k]) for k, v in cols.items()}; np.savez(out, **A)
    ok = (~A['bad']) & (~A['setup']) & A['sz19'] & (A['ha'] <= 1) & (A['nmv'] >= 60)
    print(json.dumps({'archive': Path(archive).name, 'games': int(len(ok)), 'usable': int(ok.sum()), 'bad': int(A['bad'].sum()), 'setup': int(A['setup'].sum()), 'ha>1': int((A['ha'] > 1).sum()), 'short': int((A['nmv'] < 60).sum()),
                      'median_moves': float(np.median(A['nmv'])), 'seconds': round(time.time() - t0)}), flush=True)


BANDS = (('k3-', -30, -3), ('k2-k1', -2, -1), ('d1-d2', 1, 2), ('d3-d4', 3, 4), ('d5-d6', 5, 6), ('d7+', 7, 9))


def select(out, metas, games_per=16, accounts=24000, seed=20261118):
    """accounts with >= games_per usable game-sides (19x19, no setup stones, HA <= 1, >= 60 moves, only board moves, both names), professionals
    excluded; equal quota per rank band (band = median rank of the account's usable sides; shortfall of a band goes to the others in
    proportion); games_per random sides per account.  Writes <out>/PLAN.npz (arch, ordinal, white, cls, band per chosen game-side)."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False); rng = np.random.default_rng(seed); acc, arch, ordi, white, rk = [], [], [], [], []
    for a, f in enumerate(metas):
        M = np.load(f); ok = (~M['bad']) & (~M['setup']) & M['sz19'] & (M['ha'] <= 1) & (M['nmv'] >= 60); o = np.flatnonzero(ok)
        for w, (pk, rk_) in enumerate((('pb', 'br'), ('pw', 'wr'))): acc.append(M[pk][o]); arch.append(np.full(len(o), a, np.int16)); ordi.append(o.astype(np.int32)); white.append(np.full(len(o), bool(w))); rk.append(M[rk_][o])
    acc, arch, ordi, white, rk = map(np.concatenate, (acc, arch, ordi, white, rk)); u, inv, cnt = np.unique(acc, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind='stable'); start = np.r_[0, np.cumsum(cnt)]; med = np.array([np.median(rk[order[start[i]:start[i + 1]]]) for i in np.flatnonzero(cnt >= games_per)]); elig = np.flatnonzero(cnt >= games_per)
    band = np.full(len(elig), -1); pro = med >= 10
    for b, (_, lo, hi) in enumerate(BANDS): band[(med >= lo) & (med <= hi) & ~pro] = b
    avail = np.array([(band == b).sum() for b in range(len(BANDS))]); quota = np.minimum(avail, accounts // len(BANDS))
    while quota.sum() < min(accounts, avail.sum()):
        room = avail - quota; quota = quota + np.minimum(room, np.ceil((accounts - quota.sum()) * room / max(room.sum(), 1)).astype(int)); quota = np.minimum(quota, avail)
        if quota.sum() > accounts: over = quota.sum() - accounts; quota[np.argmax(quota)] -= over
    P = {k: [] for k in ('arch', 'ordinal', 'white', 'cls', 'band')}; c = 0
    for b in range(len(BANDS)):
        for i in rng.permutation(np.flatnonzero(band == b))[:quota[b]]:
            e = elig[i]; rows = order[start[e]:start[e + 1]]; take = rng.choice(rows, games_per, replace=False)
            P['arch'].append(arch[take]); P['ordinal'].append(ordi[take]); P['white'].append(white[take]); P['cls'].append(np.full(games_per, c)); P['band'].append(np.full(games_per, b)); c += 1
    P = {k: np.concatenate(v) for k, v in P.items()}; np.savez(out / 'PLAN.npz', **P)
    info = {'metas': [Path(f).name for f in metas], 'game_sides_usable': int(len(acc)), 'accounts': int(len(u)), 'eligible': int(len(elig)), 'professional_excluded': int(pro.sum()), 'games_per': games_per,
            'available_by_band': {BANDS[b][0]: int(avail[b]) for b in range(len(BANDS))}, 'chosen_by_band': {BANDS[b][0]: int(quota[b]) for b in range(len(BANDS))}, 'classes': c, 'game_sides': int(len(P['cls'])),
            'distinct_games': int(len(np.unique(P['arch'].astype(np.int64) * 10 ** 8 + P['ordinal']))), 'seed': seed}
    json.dump(info, open(out / 'PLAN.json', 'w'), indent=1); print(json.dumps(info), flush=True)


def export(out, archives, chunk=20000, base=200_000_000, seed=20261118):
    """second pass over the archives (same order as the metas of select): games.jsonl records of the existing pipeline, in random order,
    <out>/games/extfox_cNN.jsonl (chunk games each) + <out>/labels_ext.npz (ids, cls (0-based external class), black, band)"""
    out = Path(out); P = dict(np.load(out / 'PLAN.npz')); (out / 'games').mkdir(exist_ok=True); assert not list((out / 'games').glob('*.jsonl')), 'games already exported'; rng = np.random.default_rng(seed); n = len(P['cls']); recs = [None] * n; t0 = time.time()
    # round-robin order: the k-th game of EVERY account comes before any account's (k+1)-th game (accounts in random order inside a round),
    # so the chunks can be processed one after another and the run can stop at any time with the same number of games per account
    _, first = np.unique(P['cls'], return_index=True); k = np.arange(n) - np.repeat(np.sort(first), np.diff(np.r_[np.sort(first), n])); slot = np.empty(n, np.int64); slot[np.lexsort((rng.random(n), k))] = np.arange(n)
    for a, f in enumerate(archives):
        at = np.flatnonzero(P['arch'] == a); want = {}
        for j in at: want.setdefault(int(P['ordinal'][j]), []).append(j)
        for o, rec in enumerate(records(f)):
            if o not in want: continue
            d, mv, r = parse(rec); assert not r['bad'] and not r['setup'] and r['ha'] <= 1
            text = '(;SZ[19]' + ''.join(f';{c.decode()}[{x.decode()}]' for c, x in mv) + ')'; raw = hashlib.sha256(text.encode()).hexdigest()
            for j in want[o]: recs[slot[j]] = (text, raw, 'W' if P['white'][j] else 'B', int(r['ha']), int(P['cls'][j]), int(P['band'][j]))
        print('exported', Path(f).name, len(at), round(time.time() - t0), 's', flush=True)
    assert all(r is not None for r in recs); ids = base + np.arange(n)
    for s in range(0, n, chunk):
        with open(out / 'games' / f'extfox_c{s // chunk:02d}.jsonl', 'w', newline='\n') as stream:
            for i in range(s, min(s + chunk, n)): t, raw, col, hd, _, _ = recs[i]; stream.write(json.dumps({'index': int(ids[i]), 'sgf': t, 'raw': raw, 'color': col, 'hd': hd, 'canonical': raw}) + '\n')
    np.savez(out / 'labels_ext.npz', ids=ids, cls=np.array([r[4] for r in recs]), black=np.array([r[2] == 'B' for r in recs]), band=np.array([r[5] for r in recs]))
    json.dump({'game_sides': n, 'chunks': (n + chunk - 1) // chunk, 'base_index': base, 'seconds': time.time() - t0}, open(out / 'EXPORT.json', 'w'), indent=1); print('done', n, flush=True)


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'scan': scan(sys.argv[2], sys.argv[3])
    elif st == 'select':
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('out'); ap.add_argument('metas', nargs='+'); ap.add_argument('--games-per', type=int, default=16); ap.add_argument('--accounts', type=int, default=24000); ap.add_argument('--dry', action='store_true'); v = ap.parse_args()
        select(v.out, v.metas, v.games_per, v.accounts)
    elif st == 'export': export(sys.argv[2], sys.argv[3:])
    else: raise SystemExit('unknown stage ' + st)
