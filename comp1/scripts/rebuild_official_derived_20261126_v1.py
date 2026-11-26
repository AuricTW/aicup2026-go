"""Rebuild the intermediates that re-encode official data (they are not in the Releases) from the official CSVs.

Inputs
  data/raw/...                         official training v1.0 + v1.1 and official tests v2 (see README section 2)
  data/processed/audit.sqlite          audit_training.py index/parse/summary -> audit_near_duplicates.py -> apply_family_groups.py
  runs/selection_20261126_v1/          index-only selection pack (shipped): which training games the banks picked
  runs/seqid_20261115_v1/b1_full/ffull/model.pt   shipped weight (GPU), only for the oppbank stage

Stages (run in this order; `all` runs every stage, `verify` compares SHA-256 with the original files)
  fitrows    runs/statistical_v1/fit_row_ids.npy          fit split = hash rule of goid_v4.pilot_scope_v1.fit_player
  splits     runs/holdout_protocol_v1/effective_player_splits.json   fit/validation/holdout hash rule + sealed 400
  scope      runs/research_v4/official_aligned_v1_20261104/scope_metadata.npz   (goid_v4.official_aligned_v1.capacity)
  keys       runs/v2_baseline_20261111_v2/official_stat/keys.json      sorted official test-game keys
  extmeta4   runs/bigbank4_90_20261115_v1/EXTMETA.npz     (bigbank4_90_20261115_v1.extmeta_build)
  extmeta5   runs/bigbank5_90_20261115_v1/EXTMETA.npz     (bigbank5_90_20261115_v1.extmeta_build)
  moves      runs/seqid_20261115_v1/moves.npz, official_moves.npz   (seqid_20261115_v1.moves_meta / moves_official)
  moves_ext  runs/seqid_20261115_v1/moves_ext.npz         (seqid_20261115_v1.data2)
  bundle     runs/seqid_20261115_v1/bundle_g1.npz         (seqid_20261115_v2.bundle('g1'), fold sets from the selection pack)
  labels     runs/tokpilot_20261116_v1/labels_all.npz     game id -> training-account class, colour
  oppbank    runs/oppbank_20261120_v2/full.npz            opponent side of every bank game recognised by b1_full (GPU)
  oldpool    runs/oldpool_upgrade_20261123_v1/{public,private}_moves_cap.npz   old-pool game bundles (oldpool_upgrade_20261123_v1.pack_pool, GPU)
usage: python scripts/rebuild_official_derived_20261126_v1.py all | verify | <stage> [<stage> ...]
"""
import sys, json, hashlib, sqlite3
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src'))
import numpy as np

SEL = ROOT / 'runs/selection_20261126_v1'; K = 3
EXPECTED = {
    'runs/research_v4/official_aligned_v1_20261104/scope_metadata.npz': '16e288236eb054214e65507753b9bb195b3165823db8a4422faa217144bb4a60',
    'runs/v2_baseline_20261111_v2/official_stat/keys.json': 'fb8a2eda6d2aca1ed32f3f13f41deb06223a01c59976ddd7056c83ca62d2caed',
    'runs/bigbank4_90_20261115_v1/EXTMETA.npz': '3257dac140781b0ec50a76a3fab9015ec92d42ae8bb5980fdbeb79a10b06b84a',
    'runs/bigbank5_90_20261115_v1/EXTMETA.npz': 'c04814f3fbbc22a40dd90d09783fc7ccc62c0ebb2c0840adfb5576029d9800fd',
    'runs/seqid_20261115_v1/moves.npz': '0f38299aaf3adf889771b0594164f35299e804348f253819d94ed57dc30f9229',
    'runs/seqid_20261115_v1/moves_ext.npz': '6257aa33263ce6d62b042fdbad7e4c69c09da2153499d86d372c9086311f63cc',
    'runs/seqid_20261115_v1/bundle_g1.npz': '156a9b93bc6c5e58f479e74ec7fc29bbd1fc327cf4c5cf4a36e9e366bf468eba',
    'runs/tokpilot_20261116_v1/labels_all.npz': 'acae73fada98989222bfda328038fb2059cca8643066417354b84d3807874d52',
    'runs/oppbank_20261120_v2/full.npz': 'e4da9e1650c188ea07b6d84e3c3f611df4e1609e1385b91d96c2d31914a984d3',
    'runs/oldpool_upgrade_20261123_v1/public_moves_cap.npz': '76bb52ed44c6ff3e1ea42cf92c7dfc2d630b5961de75effcf10fdafa63b730d9',
    'runs/oldpool_upgrade_20261123_v1/private_moves_cap.npz': '99dd07978ee8fe1f4a186bacae6246c8af2b467bd777f7d47ba19e18fcd6c113'}


def sha(f):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def parsed_players():
    con = sqlite3.connect((ROOT / 'data/processed/audit.sqlite').resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    try: return con.execute('SELECT row_id, player_id FROM games WHERE parse_ok=1 ORDER BY row_id').fetchall()
    finally: con.close()


def fitrows():
    """statistical_v1 fit split: every parse-ok game of a player with fit_player() true (60 % hash rule), row order"""
    from goid_v4.pilot_scope_v1 import fit_player
    f = ROOT / 'runs/statistical_v1/fit_row_ids.npy'; f.parent.mkdir(parents=True, exist_ok=True); assert not f.exists()
    np.save(f, np.array([r for r, p in parsed_players() if fit_player(p)], np.int64), allow_pickle=False)


def splits():
    """player split used by bigbank4/5: statistical_v1 hash rule (fit < .6 <= validation < .85 <= holdout); the sealed 400
    (holdout_confirmation, frozen on 2026-10-25 from 385 holdout and 15 validation players) come from the selection pack"""
    from goid_v4.pilot_scope_v1 import opaque
    sealed = set(np.load(SEL / 'sealed400_owners.npy').astype(str)); sp = {}
    for pl in sorted({p for _, p in parsed_players()}):
        u = int.from_bytes(hashlib.blake2b(f'20261025\0player:{pl}'.encode(), digest_size=16).digest(), 'big') / 2**128
        s = 'fit' if u < .6 else 'validation' if u < .85 else 'holdout'
        if opaque(pl) in sealed: assert s != 'fit'; s = 'holdout_confirmation'
        sp[pl] = s
    assert Counter(sp.values()) == {'fit': 4774, 'validation': 1993, 'holdout': 840, 'holdout_confirmation': 400}, Counter(sp.values())
    f = ROOT / 'runs/holdout_protocol_v1/effective_player_splits.json'; f.parent.mkdir(parents=True, exist_ok=True)
    with f.open('x', encoding='utf-8', newline='\n') as fh: json.dump(sp, fh, indent=2); fh.write('\n')


def scope():
    from goid_v4 import official_aligned_v1 as oa
    oa.capacity()


def keys():
    import build_v2_baseline_20261111_v2 as b
    games, _ = b.load_official(); b.write(b.RUN / 'official_stat/keys.json', sorted(games))


def extmeta4():
    import bigbank4_90_20261115_v1 as b4
    b4.RUN.mkdir(parents=True, exist_ok=True); b4.extmeta_build()


def extmeta5():
    import bigbank5_90_20261115_v1 as b5
    b5.RUN.mkdir(parents=True, exist_ok=True); b5.extmeta_build()


def moves():
    import seqid_20261115_v1 as s1, bigbank4_90_20261115_v1 as b4
    s1.RUN.mkdir(parents=True, exist_ok=True); s1.moves_meta(b4.p, b4.p.b.meta()); s1.moves_official(b4.p)


def moves_ext():
    import seqid_20261115_v1 as s1
    s1.data2()


def bundle():
    """seqid_20261115_v2.bundle('g1') unchanged; only fold_drop reads its fold sets from the selection pack
    (original: owners of the bigbank90 train labels with label % 3 == k, episode games of crossfit90 EPISODES.json fold k)"""
    import seqid_20261115_v1 as s1, seqid_20261115_v2 as s2
    ids1 = np.load(SEL / 'bigbank90_train_indices.npy'); lab1 = np.load(SEL / 'bigbank90_train_labels.npy')

    def fold_drop(split, k, p, owners):
        assert split == 'g1'
        used = np.load(SEL / f'g1_episode_games_f{k}.npy'); return {'supports': [used.tolist()], 'questions': []}, set(owners[ids1[lab1 % K == k]]), None
    s1.fold_drop = fold_drop; s2.bundle('g1')


def labels():
    """same arrays as the original label table: bigbank3 picks (SAFE index), then bigbank4 + bigbank5 picks (mixed index);
    class = rank of the owner hash; off_black = colour of each official game in keys.json order"""
    import bigbank5_90_20261115_v1 as b5
    p = b5.p; m = p.b.meta(); U = b5.Universe()
    ids3 = np.load(SEL / 'bigbank3_train_indices.npy'); o3 = m['owners'][ids3].astype(str); c3 = m['colors'][ids3] == 0
    ids4 = np.concatenate([np.load(SEL / 'bigbank4_train_indices.npy'), np.load(SEL / 'bigbank5_train_indices.npy')]); o4 = U.get('owners', ids4).astype(str); c4 = U.get('colors', ids4) == 0
    ids = np.concatenate([ids3, ids4]); own = np.concatenate([o3, o4]); blk = np.concatenate([c3, c4]); u, y = np.unique(own, return_inverse=True)
    games, _ = p.b.load_official(); kk = p.b.read(p.b.RUN / 'official_stat/keys.json'); ob = np.array([games[k].color == 'B' for k in kk])
    f = ROOT / 'runs/tokpilot_20261116_v1/labels_all.npz'; f.parent.mkdir(parents=True, exist_ok=True); assert not f.exists()
    np.savez(f, ids=ids.astype(np.int64), y=y.astype(np.int32), black=blk, off_black=ob); print('labelled games', len(ids), 'classes', len(u))


def oppbank():
    """first block of oppbank_tcond_20261120_v1.py (load / emb / post copied unchanged): full.npz only"""
    import torch, seqid_core_20261115_v4 as sc
    dev = 'cuda'; SQ = ROOT / 'runs/seqid_20261115_v1'; OUT = ROOT / 'runs/oppbank_20261120_v2'; OUT.mkdir(parents=True, exist_ok=True); f = OUT / 'full.npz'; assert not f.exists()
    Z = np.load(SQ / 'bundle_g1.npz'); M = torch.tensor(Z['M'].astype(np.int64)); LEN = torch.tensor(np.minimum(Z['LEN'].astype(np.int64), 160)); FB = torch.tensor(Z['FB']); BLK = torch.tensor(Z['BLK']); HD = torch.tensor(Z['HD'].astype(np.int64))
    OM = torch.tensor(Z['OFF_M'].astype(np.int64)); OL = torch.tensor(np.minimum(Z['OFF_LEN'].astype(np.int64), 160)); OF = torch.tensor(Z['OFF_FB']); OB = torch.tensor(Z['OFF_BLK']); OH = torch.tensor(Z['OFF_HD'].astype(np.int64))

    def load(path):
        ck = torch.load(path, map_location='cpu', weights_only=False); a = ck['args']; net = sc.Net(len(ck['classes']), a['d'], a['layers'], a['L'], a['dropout'], board=a['board'], bl=a['board_layers'], bpatch=a['board_patch']).to(dev).to(memory_format=torch.channels_last)
        net.load_state_dict(ck['model']); net.eval(); return net, np.asarray(ck['classes'])

    def emb(net, m, ln, side, fb, hd, bs=1024):
        o = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(m), bs): o.append(torch.nn.functional.normalize(net.embed(m[i:i + bs].to(dev), ln[i:i + bs].to(dev), side[i:i + bs].to(dev), fb[i:i + bs].to(dev), hd[i:i + bs].to(dev)).float(), dim=1).half().cpu())
        return torch.cat(o)

    def post(net, classes, E):
        W = torch.nn.functional.normalize(net.W.float(), dim=1); top = np.zeros(len(E), np.float32); arg = np.zeros(len(E), np.int64)
        with torch.no_grad():
            for i in range(0, len(E), 8192): P = torch.softmax(30 * E[i:i + 8192].to(dev).float() @ W.t(), 1); a, b = P.max(1); top[i:i + 8192] = a.cpu().numpy(); arg[i:i + 8192] = b.cpu().numpy()
        return top, classes[arg]

    full, cf = load(SQ / 'b1_full/ffull/model.pt'); E = emb(full, M, LEN, ~BLK, FB, HD); p, o = post(full, cf, E); EOo = emb(full, OM, OL, ~OB, OF, OH); po, oo = post(full, cf, EOo)
    np.savez(f, top=p, owner=o, off_top=po, off_owner=oo, classes=cf)


def oldpool():
    """old public / old private test pools: SGFs + capture events, packed straight from the official test CSVs"""
    import oldpool_upgrade_20261123_v1 as op
    op.RUN.mkdir(parents=True, exist_ok=True)
    for pool in ('public', 'private'):
        assert not (op.RUN / f'{pool}_moves_cap.npz').exists(); op.pack_pool(pool)


def verify():
    bad = 0
    for rel, want in EXPECTED.items():
        f = ROOT / rel; got = sha(f) if f.exists() else 'missing'; bad += got != want
        print(('OK  ' if got == want else 'DIFF'), rel, got)
    print('all identical' if not bad else f'{bad} file(s) differ'); return bad


STAGES = {'fitrows': fitrows, 'splits': splits, 'scope': scope, 'keys': keys, 'extmeta4': extmeta4, 'extmeta5': extmeta5, 'moves': moves,
          'moves_ext': moves_ext, 'bundle': bundle, 'labels': labels, 'oppbank': oppbank, 'oldpool': oldpool}
if __name__ == '__main__':
    todo = sys.argv[1:] or ['all']
    if todo == ['verify']: sys.exit(1 if verify() else 0)
    for st in (list(STAGES) if todo == ['all'] else todo):
        print('==', st, flush=True); STAGES[st]()
    if todo == ['all']: sys.exit(1 if verify() else 0)
