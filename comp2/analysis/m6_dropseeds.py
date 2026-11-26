"""More draws of the 20 % gallery drop for the six-net identity evidence of the x track, on CPU (11-20).
The hybrid validation so far has ONE random set of dropped accounts (idens.py --drop-seed 7).  idens.py recomputes everything on a GPU;
here the ensemble posterior q of the 17,686 held-out question games (opponent side) is computed once on CPU and cached, and the
evidence of any dropped set follows exactly as in idens.py (dropped classes set to 0, renormalised; mass, top-1, profile evidence
with the stored acc_cnt / acc_P0 / acc_base, kappa 5).
Checks: no drop -> the x block of the full-coverage file; seed 7 -> the existing dropped list and the x block of m6_d0.2.npz.
Output per seed: vm_out/dropsim/m6_hyb_s<SEED>.npz (dev block full coverage, x block dropped, 'dropped' = class indices; usable as
--id-x of fuse_sources.py and as --drop FILE:x of tok_evidence.py).
usage: m6_dropseeds.py SEED[,SEED...] [--threads 16]
"""
import sys, os, csv, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from idnet import Net, load_trunk
from comp2_paths import WORK; V = str(WORK); seeds = [int(s) for s in sys.argv[1].split(",")]; torch.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 16)
RUNS = ["id_v2x", "e3_ps05", "bx1_p9c64", "bx2_p9c64", "tx_full", "tx2_full"]; TR = ",".join(os.path.join(V, "trunk", d) for d in ("abt", "oof", "id"))
F = np.load(os.path.join(V, "idnet_tx2_full", "acc_6m.npz"), allow_pickle=True); D7 = np.load(os.path.join(V, "dropsim", "m6_d0.2.npz"), allow_pickle=True); H7 = np.load(os.path.join(V, "dropsim", "m6_hyb.npz"), allow_pickle=True)
z = np.load(os.path.join(V, "moves_all2.npz"), allow_pickle=True); sp = z["split"]; pid = z["pid"].astype(np.int64); keys = z["key"]; t0 = time.time()
rows = list(csv.DictReader(open(os.path.join(V, "oof_games.csv"), newline=""))); names = {str(n): i for i, n in enumerate(z["pid_names"])}
hp = np.array(sorted({names[r["player_id"]] for r in rows if r["player_id"] in names})); kset = {r["key"] for r in rows}
xs = np.array([i for i in np.where((sp == 0) | (sp == 3))[0] if str(keys[i]) in kset]); assert (keys[xs] == F["x_keys"]).all(), "x rows differ from the reference file"
tr_all = np.where(sp == 0)[0]; tr_all = tr_all[~np.isin(pid[tr_all], hp)]
slot_all, _ = load_trunk(TR, keys, 80, 0.5, load=False); tr_emb = tr_all[slot_all[tr_all] >= 0]      # idens.py with --trunk counts the embedded training games only
QF = os.path.join(V, "dropsim", "m6_qx.npy")
if os.path.exists(QF): Q = np.load(QF).astype(np.float64); owners = torch.load(os.path.join(V, "m6nets", RUNS[0], "model.pt"), map_location="cpu", weights_only=False)["owners"]
else:
    need = np.zeros(len(keys), bool); need[xs] = True; slot, TK = load_trunk(TR, np.array([str(k) if need[i] else "" for i, k in enumerate(keys)]), 80, 0.5)
    CB = torch.tensor(z["color_black"].astype(bool)); FB = torch.tensor(z["first_black"].astype(bool)); HDV = torch.tensor(z["hd"].astype(np.int64)); LENa = z["LEN"].astype(np.int64); Q = None; owners = None
    for r in RUNS:
        ck = torch.load(os.path.join(V, "m6nets", r, "model.pt"), map_location="cpu", weights_only=False); ar = ck["args"]; L = ar["L"]
        if owners is None: owners = ck["owners"]
        assert np.array_equal(owners, ck["owners"])
        net = Net(len(owners), ar["d"], ar["layers"], 4, L, drop=ar["dropout"], cos=ar.get("cos") or 0.0, rel=ar.get("rel") or 0, board=ar.get("board") or 0, bl=ar.get("board_layers") or 4, bpatch=ar.get("board_patch") or 0, trunk=64 if ar.get("trunk") else 0)
        net.load_state_dict(ck["model"]); net.eval(); M = torch.tensor(z["M"][xs][:, :L].astype(np.int64)); q = np.zeros((len(xs), len(owners)), np.float32)
        with torch.no_grad():
            for i in range(0, len(xs), 512):
                jn = xs[i:i + 512]; j = torch.tensor(jn); tk = TK[torch.as_tensor(slot[jn])][:, :L].float() if net.tproj is not None else None
                lo, _ = net(M[i:i + 512], torch.tensor(np.minimum(LENa[jn], L)), ~CB[j], FB[j], HDV[j], tk); q[i:i + len(jn)] = torch.softmax(lo.float(), 1).numpy()
        Q = q.astype(np.float64) if Q is None else Q + q; print(f"{r}: L {L}  mean top-1 {q.max(1).mean():.4f}  {time.time() - t0:.0f}s", flush=True)
    Q /= len(RUNS); np.save(QF, Q.astype(np.float32))
C = Q.shape[1]; rank = F["cls_rank"].astype(int); R1 = np.eye(10)[rank]; cnt = F["acc_cnt"].astype(np.float64); P0 = F["acc_P0"]; base = F["acc_base"]; n = cnt.sum(1); PT = ((cnt + 5.0 * base[rank]) / (n + 5.0)[:, None]) / P0[None]
def block(q): tc = q.argmax(1); return dict(x_opp_mass=(q @ R1).astype(np.float32), x_opp_top=q.max(1).astype(np.float32), x_opp_toprank=rank[tc], x_opp_topcls=tc, x_opp_acc=(q @ PT).astype(np.float32))
b0 = block(Q); print(f"CHECK no drop against the full-coverage file: top-1 max abs diff {np.abs(b0['x_opp_top'] - F['x_opp_top']).max():.4f} (mean {np.abs(b0['x_opp_top'] - F['x_opp_top']).mean():.5f}); same top class {np.mean(b0['x_opp_topcls'] == F['x_opp_topcls']):.4f}; acc mean abs diff {np.abs(b0['x_opp_acc'] - F['x_opp_acc']).mean():.5f}")
cls_of = -np.ones(pid.max() + 1, np.int64); cls_of[owners] = np.arange(C); ng = np.bincount(cls_of[pid[tr_emb]], minlength=C).astype(np.float64)
def dropped_of(seed): order_ = np.random.RandomState(seed).permutation(C); cum = np.cumsum(ng[order_]) / ng.sum(); return order_[:int(np.searchsorted(cum, 0.2)) + 1]
d7 = dropped_of(7); print(f"CHECK seed 7: {len(d7)} accounts, same set as the existing file: {set(d7.tolist()) == set(D7['dropped'].tolist())}")
def drop_block(dr): q = Q.copy(); q[:, dr] = 0; q /= q.sum(1, keepdims=True); return block(q)
b7 = drop_block(D7["dropped"]); print(f"CHECK seed-7 drop against m6_d0.2.npz: top-1 mean abs diff {np.abs(b7['x_opp_top'] - D7['x_opp_top']).mean():.5f}; same top class {np.mean(b7['x_opp_topcls'] == D7['x_opp_topcls']):.4f}; acc mean abs diff {np.abs(b7['x_opp_acc'] - D7['x_opp_acc']).mean():.5f}")
for s in seeds:
    dr = dropped_of(s); b = drop_block(dr); out = {k: H7[k] for k in H7.files}; out.update(b); out["dropped"] = dr
    np.savez(os.path.join(V, "dropsim", f"m6_hyb_s{s}.npz"), **out)
    print(f"seed {s}: {len(dr)} accounts dropped ({ng[dr].sum() / ng.sum():.3f} of the games; overlap with seed 7: {len(set(dr.tolist()) & set(d7.tolist()))})  x mean top-1 {b['x_opp_top'].mean():.4f} (full {b0['x_opp_top'].mean():.4f}, seed 7 {b7['x_opp_top'].mean():.4f})  > 0.5 {np.mean(b['x_opp_top'] > 0.5):.3f}", flush=True)
