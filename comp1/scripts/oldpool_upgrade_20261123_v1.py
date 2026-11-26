"""2026-11-23 P1: Old-Pool (old_public 400 + old_private 400) Upgrade & Label-Free Validation.
Why: Since 2026-11-13 (assign_sinkhorn_20261113_v1.py), the 800 old-pool rows (priv_q_0001..0400 = 50% of final Private,
pub_q_0001..0400 = 50% of Public) have been frozen at the 2026-11-05 H matrix (8-decision HumanSL PLDA + stat PLDA + 40-move prefix)
plus Sinkhorn (T=0.1). None of the 2026-11-15+ sequence identity networks (b1_full, b2_full, b1s1_full, b1cap_full, b1aug_full,
b1f0, a2f0, a2f2) was ever applied to the old pools.
This script:
  1. Packs old_private SGFs (400 candidates x 100 games + 400 questions x 5..20 games = 45,000 games) exactly like oldproxy_20261115_v1.py
     packed old_public.
  2. Replays capture events (seqcap_20261115_v1) for old_public and old_private so b1cap_full can run alongside the other 7 seqid models.
  3. Embeds all 45,000 games of old_public and all 45,000 games of old_private with all 8 sequence models on local RTX 3060 (<2.5 GB VRAM).
  4. Evaluates strictly label-free validation on BOTH old_private and old_public:
     (a) Candidate-gallery support-split (suppsplit: 10 support games, q in {1,2,3,5}; and 80 support games, q in {5,10,15,20}).
     (b) Split-query self-consistency on the REAL query games of old_private and old_public (splitting each question's q games into
         disjoint halves A and B and checking Top-1 agreement against the 400 candidates, using zero answer keys).
     (c) Margin & Sinkhorn posterior confidence diagnostics on old_private and old_public.
  5. Exports a standalone candidate CSV
     that upgrades ONLY the 800 old-pool rows while keeping v9qt's 800 new-pool rows bit-for-bit identical.
"""
from pathlib import Path
import sys, csv, json, time, re, hashlib
from datetime import datetime, timezone
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import seqid_20261115_v1 as s1
import seqid_core_20261115_v2 as core_v2
import seqid_core_20261115_v4 as core_v4
import seqcap_20261115_v1 as scap

RUN = ROOT / 'runs/oldpool_upgrade_20261123_v1'
OLDPROXY = ROOT / 'runs/oldproxy_20261115_v1'
RAW = ROOT / 'data/raw/official_tests_20261111_v2/extracted/tests'
SQ = ROOT / 'runs/seqid_20261115_v1'
H_PUBLIC_PATH = ROOT / 'runs/fisher_policy_pilot_20261105_v2/public_inputs/H.npy'
H_PRIVATE_PATH = ROOT / 'runs/trainval_refinement_20261105_v1/release/Private_H_scores.npy'
V9QT_CSV = ROOT / 'submissions/20261122_v9qt_m39htg_og5_transgal_q1raw_sinkq2plus_v1/submission_player.csv'
V9QT_SHA = '23445c5ca7a6728e49d3519e268262b82a16d17d3d520d5144ede04da853f8b1'
SAMPLE_CSV = RAW / 'sample_submission_player_v2.csv'
HD_RE = re.compile(r'HD\[(\d+)\]')
HEADER = ['question_id', 'top_1', 'top_2', 'top_3', 'top_4', 'top_5']

MODELS = {
    'b1_full':   (SQ / 'b1_full/ffull/model.pt', False),
    'b1cap_full':(SQ / 'b1cap_full/ffull/model.pt', True),
    'b1s1_full': (SQ / 'b1s1_full/ffull/model.pt', False),
    'b2_full':   (SQ / 'b2_full/ffull/model.pt', False),
    'b1aug_full':(SQ / 'b1aug_full/ffull/model.pt', False),
    'b1f0':      (SQ / 'b1/g1/f0/model.pt', False),
    'a2f0':      (SQ / 'a2/g1/f0/model.pt', False),
    'a2f2':      (SQ / 'a2/g1/f2/model.pt', False),
}


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def z(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-12)


def logsoft(L):
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))


def sinkhorn_bijection(S, temp=0.1, iters=2000):
    P = np.exp(logsoft(z(S) / temp))
    for _ in range(iters):
        P /= P.sum(0, keepdims=True)
        P /= P.sum(1, keepdims=True)
    return np.log(P + 1e-300)


def sid(s, tgt):
    r = (s > s[np.arange(len(s)), tgt][:, None]).sum(1)
    return np.where(r < 5, np.exp(-r), 0.0)


def pack_pool(pool):
    """Pack SGFs + capture events for old public or old private."""
    out_npz = RUN / f'{pool}_moves_cap.npz'
    if out_npz.exists():
        return dict(np.load(out_npz))
    t0 = time.time()
    if pool == 'public' and (OLDPROXY / 'public_moves.npz').exists():
        Z = dict(np.load(OLDPROXY / 'public_moves.npz'))
    else:
        csv.field_size_limit(2 ** 31 - 1)
        sg, role, own, gid = [], [], [], []
        pre = 'pub' if pool == 'public' else 'priv'
        cids = [f'{pre}_p_{i:04}' for i in range(1, 401)]
        qids = [f'{pre}_q_{i:04}' for i in range(1, 401)]
        cpos = {c: i for i, c in enumerate(cids)}
        qpos = {q: i for i, q in enumerate(qids)}
        with open(RAW / f'player_identification_candidates_{pool}.csv', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                assert r['color'] == 'B'
                sg.append(r['sgf_content']); role.append(0); own.append(cpos[r['player_id']]); gid.append(r['game_id'])
        with open(RAW / f'player_identification_test_{pool}.csv', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                n = int(r['num_games'])
                for j in range(1, n + 1):
                    assert r[f'color_{j}'] == 'B'
                    sg.append(r[f'sgf_{j}']); role.append(1); own.append(qpos[r['question_id']]); gid.append(f"{r['question_id']}#{j}")
        M, LEN, FB = s1.pack(sg)
        hd = np.array([int(x.group(1)) if (x := HD_RE.search(s[:40])) else 0 for s in sg], np.uint8)
        Z = {'M': M, 'LEN': LEN, 'FB': FB, 'hd': hd, 'role': np.array(role, np.uint8), 'own': np.array(own, np.int32)}
        if pool == 'public':  # the public bundle was first packed by oldproxy_20261115_v1, which also kept game ids and SGF hashes (unused here)
            Z.update(gid=np.array(gid), raw=np.array([hashlib.sha256(s.encode()).hexdigest() for s in sg]))
    g, t, pnt = scap.events(Z['M'][:, :160], Z['FB'], dev='cuda')
    CT, CP, rep = scap.pad(g, t, pnt, len(Z['M']), 64)
    Z['CAPT'] = CT; Z['CAPP'] = CP
    np.savez(out_npz, **Z)
    print(f'Packed {pool}: {len(Z["M"])} games, captures={rep["games_with_capture"]:.3f} in {time.time()-t0:.1f}s', flush=True)
    return Z


def embed_pool(pool, Z):
    """Embed all 45,000 games of a pool with all 8 sequence models (saves normalized float16 embeddings)."""
    torch.cuda.set_per_process_memory_fraction(0.45)
    embs = {}
    for name, (ckpt_path, use_cap) in MODELS.items():
        ef = RUN / f'emb_{pool}_{name}.npy'
        if ef.exists():
            E = np.load(ef).astype(np.float32)
            E /= np.linalg.norm(E, axis=1, keepdims=True)
            embs[name] = E
            continue
        if pool == 'public' and name in ('b1_full', 'a2f0'):
            old_f = OLDPROXY / ('emb_b1full.npy' if name == 'b1_full' else 'emb_a2f0.npy')
            if old_f.exists():
                E = np.load(old_f).astype(np.float32)
                E /= np.linalg.norm(E, axis=1, keepdims=True)
                np.save(ef, E.astype(np.float16))
                embs[name] = E
                print(f'Reused {pool} {name} from oldproxy', flush=True)
                continue
        t0 = time.time()
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        a = ck['args']; L = a['L']
        if use_cap:
            net = core_v4.Net(len(ck['classes']), a['d'], a['layers'], L, a['dropout'],
                              board=a.get('board', 0), bl=a.get('board_layers', 4), bpatch=a.get('board_patch', 9),
                              captures=True).cuda()
        else:
            net = core_v2.Net(len(ck['classes']), a['d'], a['layers'], L, a['dropout'],
                              board=a.get('board', 0), bl=a.get('board_layers', 4), bpatch=a.get('board_patch', 9)).cuda()
        if a.get('board', 0):
            net = net.to(memory_format=torch.channels_last)
        net.load_state_dict(ck['model']); net.eval()
        M = torch.tensor(Z['M'][:, :L].astype(np.int64))
        LEN = torch.tensor(np.minimum(Z['LEN'].astype(np.int64), L))
        FB = torch.tensor(Z['FB'])
        BLK = torch.ones(len(M), dtype=torch.bool)
        HDv = torch.tensor(Z['hd'].astype(np.int64))
        CAP = (torch.tensor(Z['CAPT']), torch.tensor(Z['CAPP'].astype(np.int16))) if use_cap else ()
        out = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(M), 256):
                ts = [t[i:i + 256].cuda() for t in (M, LEN, BLK, FB, HDv) + CAP]
                e = net.embed(*ts).float()
                e = torch.nn.functional.normalize(e, dim=1).cpu().numpy()
                out.append(e)
        E = np.concatenate(out)
        np.save(ef, E.astype(np.float16))
        embs[name] = E
        del net, ck
        torch.cuda.empty_cache()
        print(f'Embedded {pool} {name}: shape={E.shape} in {time.time()-t0:.1f}s', flush=True)
    return embs


def score_matrix(E, c_idx_list, q_idx_list):
    C = np.stack([E[s].mean(0) for s in c_idx_list])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    Q = np.stack([E[q].mean(0) for q in q_idx_list])
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return (Q @ C.T).astype(np.float64)


def run_validation_and_export():
    RUN.mkdir(parents=True, exist_ok=True)
    Z_pub = pack_pool('public')
    Z_priv = pack_pool('private')
    E_pub = embed_pool('public', Z_pub)
    E_priv = embed_pool('private', Z_priv)
    H_pub = np.load(H_PUBLIC_PATH).astype(np.float64)
    H_priv = np.load(H_PRIVATE_PATH).astype(np.float64)

    # Group definitions matching m13g..m39:
    # seqf = 4 full board-CNN models (b1_full, b1cap_full, b1s1_full, b2_full)
    # seq  = 3 models (b1f0, a2f0, a2f2)
    # seq8 = all 8 models
    SEQF = ('b1_full', 'b1cap_full', 'b1s1_full', 'b2_full')
    SEQ3 = ('b1f0', 'a2f0', 'a2f2')

    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'label_free_suppsplit': {}, 'split_query_consistency': {}, 'margins': {}}

    for pool, Z, E_dict, H_mat in (('public', Z_pub, E_pub, H_pub), ('private', Z_priv, E_priv, H_priv)):
        role, own = Z['role'], Z['own']
        cg = [np.flatnonzero((role == 0) & (own == c)) for c in range(400)]
        qg = [np.flatnonzero((role == 1) & (own == q)) for q in range(400)]

        # 1. Label-free candidate-gallery suppsplit:
        #    (a) hard regime: 10 support games vs q in {1,2,3,5} (5 draws = 2,000 questions)
        #    (b) realistic regime: 80 support games vs q in {5,10,15,20} (5 draws = 2,000 questions)
        rng = np.random.default_rng(20261123)
        sup_res = {}
        for reg_name, n_sup, q_choices in (('sup10_q1235', 10, [1, 2, 3, 5]), ('sup80_q5_20', 80, [5, 10, 15, 20])):
            acc = {k: [] for k in list(MODELS.keys()) + ['seqf', 'seqf_plus_seq3', 'seqf_sink']}
            for r in range(5):
                q_lens = np.repeat(q_choices, 100)[rng.permutation(400)]
                sups, qrys = [], []
                for c in range(400):
                    perm = rng.permutation(cg[c])
                    sups.append(perm[:n_sup])
                    qrys.append(perm[n_sup:n_sup + int(q_lens[c])])
                tg = np.arange(400)
                S_models = {m: z(score_matrix(E_dict[m], sups, qrys)) for m in MODELS}
                for m, Sm in S_models.items():
                    acc[m].append(sid(Sm, tg).mean())
                S_seqf = z(sum(S_models[m] for m in SEQF) / len(SEQF))
                S_seq3 = z(sum(S_models[m] for m in SEQ3) / len(SEQ3))
                S_comb = z(0.85 * S_seqf + 0.15 * S_seq3)
                acc['seqf'].append(sid(S_seqf, tg).mean())
                acc['seqf_plus_seq3'].append(sid(S_comb, tg).mean())
                acc['seqf_sink'].append(sid(sinkhorn_bijection(S_comb, 0.1), tg).mean())
            sup_res[reg_name] = {k: round(float(np.mean(v)), 6) for k, v in acc.items()}
        report['label_free_suppsplit'][pool] = sup_res
        print(f'[{pool}] label-free suppsplit:', json.dumps(sup_res), flush=True)

        # 2. Full score matrices on real queries vs 100-game candidate galleries
        S_real = {m: z(score_matrix(E_dict[m], cg, qg)) for m in MODELS}
        S_seqf_real = z(sum(S_real[m] for m in SEQF) / len(SEQF))
        S_seq3_real = z(sum(S_real[m] for m in SEQ3) / len(SEQ3))
        S_seq_ens = z(0.85 * S_seqf_real + 0.15 * S_seq3_real)
        np.save(RUN / f'{pool}_S_seq_ens.npy', S_seq_ens)

        # 3. Label-free split-query consistency on REAL query games:
        #    Split each question's q games into Half-A (even indices) and Half-B (odd indices)
        qg_A = [q[::2] for q in qg]
        qg_B = [q[1::2] for q in qg]
        SA = z(0.85 * z(sum(z(score_matrix(E_dict[m], cg, qg_A)) for m in SEQF) / len(SEQF)) +
               0.15 * z(sum(z(score_matrix(E_dict[m], cg, qg_B if False else qg_A)) for m in SEQ3) / len(SEQ3)))
        SB = z(0.85 * z(sum(z(score_matrix(E_dict[m], cg, qg_B)) for m in SEQF) / len(SEQF)) +
               0.15 * z(sum(z(score_matrix(E_dict[m], cg, qg_B)) for m in SEQ3) / len(SEQ3)))
        LA = sinkhorn_bijection(SA, 0.1); LB = sinkhorn_bijection(SB, 0.1)
        q_lens_real = np.array([len(q) for q in qg])
        split_agree_raw = float((SA.argmax(1) == SB.argmax(1)).mean())
        split_agree_sink = float((LA.argmax(1) == LB.argmax(1)).mean())
        split_by_q = {str(n): {'raw_agree': float((SA[q_lens_real == n].argmax(1) == SB[q_lens_real == n].argmax(1)).mean()),
                               'sink_agree': float((LA[q_lens_real == n].argmax(1) == LB[q_lens_real == n].argmax(1)).mean())}
                      for n in (5, 10, 15, 20)}
        # Agreement between old H_mat (after Sinkhorn) and new S_seq_ens (after Sinkhorn)
        L_H = sinkhorn_bijection(H_mat, 0.1)
        L_seq = sinkhorn_bijection(S_seq_ens, 0.1)
        report['split_query_consistency'][pool] = {
            'halfA_vs_halfB_raw_top1_agree': round(split_agree_raw, 4),
            'halfA_vs_halfB_sink_top1_agree': round(split_agree_sink, 4),
            'by_num_games': split_by_q,
            'H_sink_vs_Seq_sink_top1_agree': int((L_H.argmax(1) == L_seq.argmax(1)).sum()),
            'H_raw_vs_Seq_raw_top1_agree': int((H_mat.argmax(1) == S_seq_ens.argmax(1)).sum()),
        }
        print(f'[{pool}] split-query self-consistency:', json.dumps(report['split_query_consistency'][pool]), flush=True)

        # 4. Margin and confidence diagnostics for H vs S_seq_ens vs pre-registered blend (0.25*z(H) + 0.75*z(S_seq_ens))
        S_blend = z(0.25 * z(H_mat) + 0.75 * S_seq_ens)
        np.save(RUN / f'{pool}_S_blend_25H_75seq.npy', S_blend)
        m_diag = {}
        for tag, Sm in (('H_old', H_mat), ('Seq_ens', S_seq_ens), ('Blend_25H_75Seq', S_blend)):
            zm = z(Sm)
            s_zm = np.sort(zm, axis=1)[:, ::-1]
            marg = s_zm[:, 0] - s_zm[:, 1]
            Lm = sinkhorn_bijection(Sm, 0.1)
            Pm = np.exp(Lm)
            top1_p = Pm.max(1)
            m_diag[tag] = {
                'raw_distinct_top1': int(len(set(Sm.argmax(1)))),
                'sink_distinct_top1': int(len(set(Lm.argmax(1)))),
                'median_z_margin': round(float(np.median(marg)), 4),
                'z_margin_lt_0.5': int((marg < 0.5).sum()),
                'z_margin_lt_1.0': int((marg < 1.0).sum()),
                'sink_prob_lt_0.8': int((top1_p < 0.8).sum()),
                'sink_prob_lt_0.9': int((top1_p < 0.9).sum()),
                'expected_correct_sum_maxP': round(float(top1_p.sum()), 2),
            }
        report['margins'][pool] = m_diag
        print(f'[{pool}] margins & diagnostics:', json.dumps(m_diag, indent=1), flush=True)

    # 5. Export standalone candidate CSV:
    #    ONLY old_public (400 rows) and old_private (400 rows) updated using pre-registered Blend (0.25*z(H) + 0.75*z(S_seq_ens)) + Sinkhorn (T=0.1) + Hungarian Top-1 guard
    #    New pools (public2 400 rows + private2 400 rows) copied bit-for-bit from v9qt!
    assert sha(V9QT_CSV) == V9QT_SHA, f'v9qt SHA mismatch: {sha(V9QT_CSV)}'
    with open(V9QT_CSV, encoding='utf-8', newline='') as f:
        v9qt_rows = {r['question_id']: [r[f'top_{i}'] for i in range(1, 6)] for r in csv.DictReader(f)}
    with open(SAMPLE_CSV, encoding='utf-8', newline='') as f:
        sample_qids = [r['question_id'] for r in csv.DictReader(f)]
    assert len(sample_qids) == 1600

    preds = dict(v9qt_rows)
    changes = {}
    for pool, pre in (('public', 'pub'), ('private', 'priv')):
        qids = [f'{pre}_q_{i:04}' for i in range(1, 401)]
        cids = [f'{pre}_p_{i:04}' for i in range(1, 401)]
        S_blend = np.load(RUN / f'{pool}_S_blend_25H_75seq.npy')
        L = sinkhorn_bijection(S_blend, 0.1, iters=2000)
        # Check if Sinkhorn Top-1 is already a 400-way bijection; use Hungarian for Top-1 to guarantee exact 400-way permutation, then fill Top-2..5 by L
        _, hung_top1 = linear_sum_assignment(-L)
        sink_order = np.argsort(-L, axis=1, kind='stable')
        top1_diff_vs_v9qt = 0
        any_diff_vs_v9qt = 0
        hung_vs_sink_diff = int((hung_top1 != sink_order[:, 0]).sum())
        for i, qid in enumerate(qids):
            t1 = int(hung_top1[i])
            rest = [int(j) for j in sink_order[i] if int(j) != t1][:4]
            row_cids = [cids[t1]] + [cids[j] for j in rest]
            assert len(row_cids) == 5 and len(set(row_cids)) == 5
            if row_cids[0] != v9qt_rows[qid][0]:
                top1_diff_vs_v9qt += 1
            if row_cids != v9qt_rows[qid]:
                any_diff_vs_v9qt += 1
            preds[qid] = row_cids
        changes[pool] = {
            'top1_changed_vs_v9qt': top1_diff_vs_v9qt,
            'any_top5_changed_vs_v9qt': any_diff_vs_v9qt,
            'hungarian_vs_sinkhorn_top1_diff': hung_vs_sink_diff,
            'distinct_top1': len({preds[qid][0] for qid in qids}),
        }

    # Verify new-pool 800 rows are 100% identical to v9qt
    for qid in sample_qids:
        if qid.startswith('pub2_') or qid.startswith('priv2_'):
            assert preds[qid] == v9qt_rows[qid], f'New pool row modified: {qid}'

    out_dir = ROOT / 'submissions/20261123_v9r_oldpool_seq8_h25_v9qt_new_v1'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'submission_player.csv'
    with open(out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(HEADER)
        for qid in sample_qids:
            w.writerow([qid] + preds[qid])

    # Format check
    import suppsplit_eval_20261115_v2 as s2
    val_chk = s2.p.b.validate(out_csv)
    assert val_chk['valid'], val_chk
    out_sha = sha(out_csv)
    report['candidate_export'] = {
        'CSV': str(out_csv),
        'SHA256': out_sha,
        'bytes': out_csv.stat().st_size,
        'rows': 1600,
        'parent_v9qt_SHA256': V9QT_SHA,
        'new800_identical_to_v9qt': True,
        'old_pool_changes_vs_v9qt': changes,
        'format_check': val_chk,
    }
    (RUN / 'RESULT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    (out_dir / 'RECEIPT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('=== EXPORT COMPLETE ===', flush=True)
    print(json.dumps(report['candidate_export'], indent=2), flush=True)


if __name__ == '__main__':
    run_validation_and_export()
