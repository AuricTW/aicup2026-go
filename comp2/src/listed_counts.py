"""Extra counts for the account profiles of the identity evidence (11-20).
The profile of a gallery account a (idens.py --acc: acc_cnt) = labels of the OWNERS of the training games in which a is recognised
as the opponent (its hidden games).  The same matchmaking is seen from the other side in a's own LISTED games: the labels of the
recognised opponents.  This script counts them (pseudo file, recognition > THR, even games, opponent inside the gallery of REF) and
corrects for the different base rates of the two directions per label of a:  c[a, r] *= BH[l(a), r] / BL[l(a), r]  (row total kept),
BH = P(owner label | opponent label), BL = P(opponent label | owner label), both from the same recognised pairs.
Only labels of gallery accounts of REF are read, so the x track stays free of the held-out players (use the x-track pseudo file).
Output: OUT.npz with cnt [C, 10] in the class order of REF and class_names; used by tok_evidence.py --cnt-add OUT.npz[:W].
usage: listed_counts.py REF.npz PSEUDO.npz OUT.npz [THR 0.5]
"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); from comp2_paths import WORK
ref = np.load(sys.argv[1], allow_pickle=True); ps = np.load(sys.argv[2]); out = sys.argv[3]; thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
names = ref["class_names"].astype(str); C = len(names); rank = ref["cls_rank"].astype(int)
z = np.load(os.path.join(WORK, "moves_all3.npz"), allow_pickle=True); y = z["y"].astype(int); hd = z["hd"].astype(int); pid = z["pid"].astype(np.int64); sp = z["split"]; pn = z["pid_names"].astype(str)
cls_of = {nm: i for i, nm in enumerate(names)}; p2c = np.array([cls_of.get(x, -1) for x in pn] + [-1])
idx = ps["idx"]; ok = (ps["prob"] > thr) & (hd[idx] == 0) & (sp[idx] != 2); ro = idx[ok]; opc = p2c[ps["pid"][ok]]; owc = p2c[pid[ro]]; w = ps["prob"][ok].astype(np.float64)
assert (y[ro][owc >= 0] == rank[owc[owc >= 0]]).all(), "label of a gallery account differs between REF and the moves file"
both = (opc >= 0) & (owc >= 0); CL = np.zeros((C, 10)); np.add.at(CL, (owc[both], rank[opc[both]]), w[both])
BH = np.full((10, 10), 1e-3); np.add.at(BH, (rank[opc[both]], rank[owc[both]]), w[both]); BH /= BH.sum(1, keepdims=True)
BL = np.full((10, 10), 1e-3); np.add.at(BL, (rank[owc[both]], rank[opc[both]]), w[both]); BL /= BL.sum(1, keepdims=True)
CLc = CL * (BH[rank] / BL[rank]); CLc *= CL.sum(1, keepdims=True) / np.maximum(CLc.sum(1, keepdims=True), 1e-12)
n0 = ref["acc_cnt"].sum(1); n1 = CLc.sum(1)
print(f"gallery {C}  recognised even pairs {int(ok.sum())}  both sides in the gallery {int(both.sum())}")
print(f"profile counts per account: hidden median {np.median(n0):.1f} mean {n0.mean():.1f}   listed (added) median {np.median(n1):.1f} mean {n1.mean():.1f}   accounts with listed > 0: {(n1 > 0).mean():.3f}")
np.savez(out, cnt=CLc, class_names=names, thr=thr); print("saved", out)
