"""External identities, batch 2 (same source and compliance rules as extfox_20261118_v1.py: bytes only, salted hashes, external
accounts are extra training classes only, nothing is compared with competition games).
Why: heads A/B (24,000 accounts x 10 / 16 games) help the single-game questions under validation v3, and 10 vs 16 games per account made no
difference -> spend the same extraction budget on MORE accounts with FEWER games: 48,000 new accounts x 8 game-sides.
Stages
  select <out dir> <previous plan dir> <meta.npz>...   accounts with >= 16 usable sides that are NOT in the previous plan; equal quota per
                                                        rank band; 8 random sides each; classes continue after the previous plan's
  export <out dir> <previous plan dir> <archive.7z>... games/extfox_cNN.jsonl with NN and the game indices continuing after the previous
                                                        export, labels_ext.npz (this batch) and labels_ext_all.npz (previous + this batch)
"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extfox_20261118_v1 as x

GAMES_PER = 8; ACCOUNTS = 48000; MIN_SIDES = 16; SEED = 20261119


def usable(M): return (~M['bad']) & (~M['setup']) & M['sz19'] & (M['ha'] <= 1) & (M['nmv'] >= 60)


def select(out, prev, metas):
    out = Path(out); out.mkdir(parents=True, exist_ok=False); prev = Path(prev); rng = np.random.default_rng(SEED); PP = np.load(prev / 'PLAN.npz'); pj = json.loads((prev / 'PLAN.json').read_text())
    assert pj['metas'] == [Path(f).name for f in metas], 'metas must be the ones of the previous plan, in the same order'
    acc, arch, ordi, white, rk, used = [], [], [], [], [], []
    for a, f in enumerate(metas):
        M = np.load(f); o = np.flatnonzero(usable(M)); m = PP['arch'] == a
        used.append(np.where(PP['white'][m], M['pw'][PP['ordinal'][m]], M['pb'][PP['ordinal'][m]]))      # account hashes of the previous plan
        for w, (pk, rk_) in enumerate((('pb', 'br'), ('pw', 'wr'))): acc.append(M[pk][o]); arch.append(np.full(len(o), a, np.int16)); ordi.append(o.astype(np.int32)); white.append(np.full(len(o), bool(w))); rk.append(M[rk_][o])
    acc, arch, ordi, white, rk = map(np.concatenate, (acc, arch, ordi, white, rk)); used = np.unique(np.concatenate(used)); assert len(used) == pj['classes'], (len(used), pj['classes'])
    u, inv, cnt = np.unique(acc, return_inverse=True, return_counts=True); order = np.argsort(inv, kind='stable'); start = np.r_[0, np.cumsum(cnt)]
    elig = np.flatnonzero((cnt >= MIN_SIDES) & ~np.isin(u, used)); med = np.array([np.median(rk[order[start[i]:start[i + 1]]]) for i in elig]); band = np.full(len(elig), -1); pro = med >= 10
    for b, (_, lo, hi) in enumerate(x.BANDS): band[(med >= lo) & (med <= hi) & ~pro] = b
    nb = len(x.BANDS); avail = np.array([(band == b).sum() for b in range(nb)]); quota = np.minimum(avail, ACCOUNTS // nb)
    while quota.sum() < min(ACCOUNTS, avail.sum()):
        room = avail - quota; quota = np.minimum(quota + np.minimum(room, np.ceil((ACCOUNTS - quota.sum()) * room / max(room.sum(), 1)).astype(int)), avail)
        if quota.sum() > ACCOUNTS: quota[np.argmax(quota)] -= quota.sum() - ACCOUNTS
    P = {k: [] for k in ('arch', 'ordinal', 'white', 'cls', 'band')}; c = pj['classes']
    for b in range(nb):
        for i in rng.permutation(np.flatnonzero(band == b))[:quota[b]]:
            e = elig[i]; rows = order[start[e]:start[e + 1]]; take = rng.choice(rows, GAMES_PER, replace=False)
            P['arch'].append(arch[take]); P['ordinal'].append(ordi[take]); P['white'].append(white[take]); P['cls'].append(np.full(GAMES_PER, c)); P['band'].append(np.full(GAMES_PER, b)); c += 1
    P = {k: np.concatenate(v) for k, v in P.items()}; np.savez(out / 'PLAN.npz', **P)
    info = {'metas': [Path(f).name for f in metas], 'previous_plan': prev.name, 'previous_classes': pj['classes'], 'eligible_new': int(len(elig)), 'games_per': GAMES_PER, 'min_usable_sides': MIN_SIDES,
            'available_by_band': {x.BANDS[b][0]: int(avail[b]) for b in range(nb)}, 'chosen_by_band': {x.BANDS[b][0]: int(quota[b]) for b in range(nb)}, 'classes_new': c - pj['classes'], 'classes_total': c,
            'game_sides': int(len(P['cls'])), 'distinct_games': int(len(np.unique(P['arch'].astype(np.int64) * 10 ** 8 + P['ordinal']))), 'seed': SEED}
    json.dump(info, open(out / 'PLAN.json', 'w'), indent=1); print(json.dumps(info), flush=True)


def export(out, prev, archives, chunk=20000):
    out = Path(out); prev = Path(prev); P = dict(np.load(out / 'PLAN.npz')); E0 = dict(np.load(prev / 'labels_ext.npz')); n0 = len(E0['ids']); base = int(E0['ids'].max()) + 1; c0 = (n0 + chunk - 1) // chunk      # the previous export's last chunk may be short (384,000 sides = 19 full chunks + 4,000)
    (out / 'games').mkdir(exist_ok=True); assert not list((out / 'games').glob('*.jsonl')), 'games already exported'; rng = np.random.default_rng(SEED); n = len(P['cls']); recs = [None] * n; t0 = time.time()
    _, first = np.unique(P['cls'], return_index=True); k = np.arange(n) - np.repeat(np.sort(first), np.diff(np.r_[np.sort(first), n])); slot = np.empty(n, np.int64); slot[np.lexsort((rng.random(n), k))] = np.arange(n)      # round-robin (see v1)
    for a, f in enumerate(archives):
        at = np.flatnonzero(P['arch'] == a); want = {}
        for j in at: want.setdefault(int(P['ordinal'][j]), []).append(j)
        for o, rec in enumerate(x.records(f)):
            if o not in want: continue
            d, mv, r = x.parse(rec); assert not r['bad'] and not r['setup'] and r['ha'] <= 1
            text = '(;SZ[19]' + ''.join(f';{c.decode()}[{m.decode()}]' for c, m in mv) + ')'; raw = hashlib.sha256(text.encode()).hexdigest()
            for j in want[o]: recs[slot[j]] = (text, raw, 'W' if P['white'][j] else 'B', int(r['ha']), int(P['cls'][j]), int(P['band'][j]))
        print('exported', Path(f).name, len(at), round(time.time() - t0), 's', flush=True)
    assert all(r is not None for r in recs); ids = base + np.arange(n)
    for s in range(0, n, chunk):
        with open(out / 'games' / f'extfox_c{c0 + s // chunk:02d}.jsonl', 'w', newline='\n') as stream:
            for i in range(s, min(s + chunk, n)): t, raw, col, hd, _, _ = recs[i]; stream.write(json.dumps({'index': int(ids[i]), 'sgf': t, 'raw': raw, 'color': col, 'hd': hd, 'canonical': raw}) + '\n')
    E = {'ids': ids, 'cls': np.array([r[4] for r in recs]), 'black': np.array([r[2] == 'B' for r in recs]), 'band': np.array([r[5] for r in recs])}; np.savez(out / 'labels_ext.npz', **E)
    np.savez(out / 'labels_ext_all.npz', **{k: np.concatenate([E0[k], E[k]]) for k in E})
    json.dump({'game_sides': n, 'first_chunk': c0, 'chunks': (n + chunk - 1) // chunk, 'base_index': base, 'seconds': time.time() - t0}, open(out / 'EXPORT.json', 'w'), indent=1); print('done', n, flush=True)


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'select': select(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif st == 'export': export(sys.argv[2], sys.argv[3], sys.argv[4:])
    else: raise SystemExit('unknown stage ' + st)
