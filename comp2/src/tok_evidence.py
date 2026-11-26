"""Identity evidence file (idens.py --acc format, usable by fuse_sources.py --id-mode acc) from token-net outputs (tok_apply.py).
The gallery, the account profiles (acc_cnt / acc_P0 / acc_base, kappa) and all non-recognition arrays (keys, hd, colour, tro_* kernels)
are taken from a REFERENCE evidence file of the same track; only the recognition posterior q over that gallery comes from the
token nets (probabilities averaged over the nets; logits x T; classes outside the track's gallery and the question player's own
account removed, renormalised over the top-K classes kept by tok_apply).
--drop D.npz:BLOCKS  removes the accounts listed in D['dropped'] (indices in the reference class order) from the named blocks
(coverage simulation, as idens.py --drop-share).
usage: tok_evidence.py REF.npz OUT.npz T KAPPA TOK.npz[,TOK2.npz ...] [--drop D.npz:x]
"""
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from comp2_paths import WORK as _WORK
args = sys.argv[1:]; drop = None; OWN = None
if "--drop" in args: drop = args[args.index("--drop") + 1]; del args[args.index("--drop"):args.index("--drop") + 2]
# --own W: OWN-side tokens (the question player's own moves): q = look-alike accounts; acc = rank mass of the look-alikes smoothed by
# K[r', r] = exp(-|r - r'| / W), relative to the prior; used for every game (handicap included, 'acc_hd1' marker)
if "--own" in args: OWN = float(args[args.index("--own") + 1]); del args[args.index("--own"):args.index("--own") + 2]
# --kern K.npz: label-level kernels kern[c] = P(player rank | recognised opponent label) for c = HD0, HD1 player black, HD1 player
# white (rows = opponent label; tok_kernel.py) replace the account profiles; acc = q-rank-mass @ kern[c] / prior, every game
# --agree C2.npz:GAMMA  account-level product of experts with a second identity net (comp2 idens --acc file of the same track and
# class order): only its top-1 class cc and probability t2 are stored, so its posterior is approximated as t2 on cc and
# (1 - t2) / (C - 1) elsewhere -> the token posterior q is multiplied on cc by (1 + t2 (C - 1) / (1 - t2))^GAMMA (GAMMA < 1 damps
# for correlated errors) and renormalised
AGREE = None
if "--agree" in args: AGREE = args[args.index("--agree") + 1]; del args[args.index("--agree"):args.index("--agree") + 2]
# --prior A: activity prior over the accounts: logit += A * log(listed training games of the account) (moves_all3, split != 2);
# an account is met as an opponent about in proportion to its listed games (11-15: correlation 0.86)
PRIOR = 0.0
if "--prior" in args: PRIOR = float(args[args.index("--prior") + 1]); del args[args.index("--prior"):args.index("--prior") + 2]
# --bg N: nets with EXTERNAL background classes (tok_apply_late.py --ext, class name ''): keep them in the softmax as an open-set
# background (each with the activity prior of an account with N listed games); their probability is NOT assigned to any account or
# rank: q sums to < 1 and the rest counts as 'not recognised' (likelihood ratio 1, uniform rank mass).  Without --bg the external
# classes are removed and the gallery classes renormalised (closed set, as for every other net).
BG = None
if "--bg" in args: BG = float(args[args.index("--bg") + 1]); del args[args.index("--bg"):args.index("--bg") + 2]
# --cons geo|min[:LAM]: CONSENSUS of two net groups (group 0 = nets read from vm_out/tokres = frozen trunk, group 1 = vm_out/xf = tuned trunks):
# q = LAM * c + (1 - LAM) * (qA + qB) / 2 with c = sqrt(qA * qB) or min(qA, qB) per account, NOT renormalised: the mass the two groups do
# not share counts as 'not recognised' (likelihood ratio 1, uniform rank mass), as the background of --bg.  Open-set guard (11-19).
CONS = None; LAM = 1.0
if "--cons" in args:
    CONS = args[args.index("--cons") + 1]; del args[args.index("--cons"):args.index("--cons") + 2]
    if ":" in CONS: CONS, LAM = CONS.split(":")[0], float(CONS.split(":")[1])
# --em ALPHA[:BETA[:ITERS]]: TRANSDUCTIVE account prior (11-19): the accounts' frequencies among the sides of the SAME block (test
# sides for the test file) re-weight the recognition posterior: m_c = sum of q over the block, f_c = (m_c + ALPHA * pi0_c * M) /
# ((1 + ALPHA) * M) with pi0 = the activity prior, q <- q * (f / pi0)^BETA renormalised; ITERS passes.  Label-free.
EM = None
if "--em" in args:
    v_ = args[args.index("--em") + 1].split(":"); del args[args.index("--em"):args.index("--em") + 2]
    EM = (float(v_[0]), float(v_[1]) if len(v_) > 1 else 1.0, int(v_[2]) if len(v_) > 2 else 1)
# --multi TAU (11-20): MULTI-CANDIDATE recognition.  Default evidence = all recognition mass through the profiles, gated by the top-1
# probability alone.  With --multi the candidates with probability >= TAU form the recognition: the gate (stored as *_opp_top) is
# their TOTAL probability and the profile evidence is their normalised mixture; the diffuse tail below TAU is dropped.  A side with
# two or three plausible accounts then counts with their joint weight, and the other games of the question decide between them.
MULTI = None
if "--multi" in args: MULTI = float(args[args.index("--multi") + 1]); del args[args.index("--multi"):args.index("--multi") + 2]
# --rawp (11-20): the inputs hold PROBABILITIES (rerank_gallery.py: P(candidate is the opponent)), not logits: no temperature, no softmax,
# no activity prior; what is missing to 1 counts as 'not recognised' (likelihood ratio 1), like the background of --bg.
RAWP = False
if "--rawp" in args: RAWP = True; args.remove("--rawp")
# --cnt-add F.npz[:W]: extra profile counts (listed_counts.py: labels of the recognised opponents in the account's own listed games),
# added to acc_cnt with weight W before the kappa shrinkage (11-20)
CNTADD = None
if "--cnt-add" in args: CNTADD = args[args.index("--cnt-add") + 1]; del args[args.index("--cnt-add"):args.index("--cnt-add") + 2]
KERN = None
if "--kern" in args: KERN = np.load(args[args.index("--kern") + 1]); del args[args.index("--kern"):args.index("--kern") + 2]
ref = np.load(args[0], allow_pickle=True); out = args[1]; T = float(args[2]); kappa = float(args[3]); tpaths = args[4].split(","); toks = [np.load(p, allow_pickle=True) for p in tpaths]; TGRP = [int("xf" in p.replace("\\", "/").split("/")) if CONS else 0 for p in tpaths]
names = ref["class_names"].astype(str); C = len(names); rank = ref["cls_rank"]; pos = {n: j for j, n in enumerate(names)}
cnt = ref["acc_cnt"].astype(np.float64); P0 = ref["acc_P0"]; base = ref["acc_base"]
if CNTADD:
    f_, w_ = CNTADD, 1.0
    if not CNTADD.lower().endswith(".npz"): f_, w_ = CNTADD.rsplit(":", 1)[0], float(CNTADD.rsplit(":", 1)[1])
    ca = np.load(f_, allow_pickle=True); assert (ca["class_names"].astype(str) == names).all(), "--cnt-add: class order differs from the reference"
    print(f"profile counts: hidden {cnt.sum():.0f} + {w_} x listed {ca['cnt'].sum():.0f}", flush=True); cnt = cnt + w_ * ca["cnt"]
n = cnt.sum(1)
PT = ((cnt + kappa * base[rank]) / (n + kappa)[:, None]) / P0[None]; R1 = np.eye(10)[rank]
DROPPED = set()
if drop: f, blocks_ = drop.split(":"); DROPPED = (set(np.load(f)["dropped"].tolist()), set(blocks_.split(",")))
pn = np.load(os.path.join(str(_WORK), "moves_all3.npz"), allow_pickle=True)["pid_names"].astype(str) if any(f"{b}_pid" in ref.files for b in ("dev", "x")) else None
LOGN = np.zeros(C)
if PRIOR:
    z3 = np.load(os.path.join(str(_WORK), "moves_all3.npz"), allow_pickle=True); n3 = np.bincount(z3["pid"][z3["split"] != 2].astype(np.int64), minlength=len(z3["pid_names"]))
    cn = dict(zip(z3["pid_names"].astype(str), n3)); LOGN = np.log(np.maximum(np.array([cn.get(x, 1) for x in names], np.float64), 1.0)); del z3
    print(f"activity prior {PRIOR}: listed games per gallery account median {np.exp(np.median(LOGN)):.0f}, p90 {np.exp(np.quantile(LOGN, .9)):.0f}, max {np.exp(LOGN.max()):.0f}", flush=True)
T_ = []
for t, tg in zip(toks, TGRP):
    m = np.array([pos.get(x, -1) for x in t["names"].astype(str)]); row = {str(k): i for i, k in enumerate(t["keys"])}
    T_.append((t["topv"], t["topi"], m, row, t["names"].astype(str) == "", tg))
if BG is not None: assert AGREE is None and OWN is None and KERN is None, "--bg is implemented for the account-profile evidence only"; print(f"open-set background: external classes per net {[int(x[4].sum()) for x in T_]}, prior of {BG} listed games", flush=True)
res = {k: ref[k] for k in ref.files if not any(k.startswith(b + "_opp_") for b in ("dev", "x", "test"))}
for b in ("dev", "x", "test"):
    if f"{b}_keys" not in ref.files: continue
    keys = ref[f"{b}_keys"].astype(str); N = len(keys); mass = np.full((N, 10), 0.1); top = np.zeros(N); tcl = np.zeros(N, np.int64); acc = np.ones((N, 10)); found = 0; n_agree = 0
    if AGREE is not None:
        A2 = np.load(AGREE.rsplit(":", 1)[0], allow_pickle=True); AG_G = float(AGREE.rsplit(":", 1)[1])
        assert (A2[f"{b}_keys"].astype(str) == keys).all() and (A2["class_names"].astype(str) == names).all(), "second net: different keys or class order"
    own = np.full(N, -1)
    if pn is not None and f"{b}_pid" in ref.files:
        pid = ref[f"{b}_pid"]; own = np.array([pos.get(pn[p], -1) if p >= 0 else -1 for p in pid])
    dmask = np.zeros(C, bool)
    if drop and b in DROPPED[1]: dmask[list(DROPPED[0])] = True
    EMW = np.ones(C); NIT = EM[2] if EM else 0
    for emit in range(NIT + 1):
      last = emit == NIT; macc = np.zeros(C); found = 0
      for s in range(0, N, 2048):
          e = min(N, s + 2048); q = np.zeros((2 if CONS else 1, e - s, C + 1)); cntm = np.zeros((q.shape[0], e - s))      # column C = sink for removed classes
          for tv, ti, m, row, ext, gi in T_:
              r = np.array([row.get(k, -1) for k in keys[s:e]]); ok = r >= 0
              if not ok.any(): continue
              ci = ti[r[ok]].astype(np.int64); v = tv[r[ok]].astype(np.float64) * T; c = m[ci]
              bad = (c < 0) | (c == own[s:e][ok][:, None]) | (dmask[np.maximum(c, 0)])
              if RAWP:
                  p = tv[r[ok]].astype(np.float64).copy(); p[bad] = 0; rows = np.where(ok)[0]; q[gi, rows[:, None], np.where(bad, C, c)] += p; cntm[gi, rows] += 1; continue
              if PRIOR: v = v + PRIOR * LOGN[np.maximum(c, 0)]
              if BG is not None and ext.any():
                  bgm = ext[ci]; v = np.where(bgm, tv[r[ok]].astype(np.float64) * T + PRIOR * np.log(BG), v); v[bad & ~bgm] = -np.inf      # background stays in the softmax
              else: v[bad] = -np.inf
              v -= v.max(1, keepdims=True); p = np.exp(v); p /= p.sum(1, keepdims=True); p[bad] = 0
              rows = np.where(ok)[0]; q[gi, rows[:, None], np.where(bad, C, c)] += p; cntm[gi, rows] += 1      # (row, class) pairs are unique except the sink
          if CONS:
              qa, qb = q[0][:, :C], q[1][:, :C]; oa, ob = cntm[0] > 0, cntm[1] > 0; qa[oa] /= cntm[0][oa, None]; qb[ob] /= cntm[1][ob, None]; both = (oa & ob)[:, None]
              cq = np.sqrt(qa * qb) if CONS == "geo" else np.minimum(qa, qb)
              q = np.where(both, LAM * cq + (1 - LAM) * (qa + qb) / 2, qa + qb); ok = oa | ob; found += int(ok.sum()); del qa, qb, cq
          else: q = q[0][:, :C]; ok = cntm[0] > 0; found += int(ok.sum()); q[ok] /= cntm[0][ok, None]
          if AGREE is not None:
              cc = A2[f"{b}_opp_topcls"][s:e].astype(np.int64); t2 = np.clip(A2[f"{b}_opp_top"][s:e].astype(np.float64), 1e-6, 0.999)
              fac = (1 + t2 * (C - 1) / (1 - t2)) ** AG_G; rr = np.flatnonzero(ok)
              q[rr, cc[rr]] *= fac[rr]; q[ok] /= np.maximum(q[ok].sum(1, keepdims=True), 1e-300); n_agree += int((q[rr, cc[rr]] > 0).sum())
          if EM:
              q[ok] = q[ok] * EMW[None]; q[ok] /= np.maximum(q[ok].sum(1, keepdims=True), 1e-300)
              if not last: macc += q[ok].sum(0); continue
          rest = np.clip(1 - q.sum(1), 0, 1) if (BG is not None or CONS or RAWP) else np.zeros(e - s)      # background (unrecognised) probability
          mass[s:e][ok] = q[ok] @ R1 + 0.1 * rest[ok, None]; top[s:e][ok] = q[ok].max(1); tcl[s:e][ok] = q[ok].argmax(1); bgsum = (bgsum if s else 0.0) + rest[ok].sum()
          if KERN is not None:
              h_ = ref[f"{b}_hd"][s:e]; bl_ = ref[f"{b}_black"][s:e].astype(bool); c_ = np.where(h_ == 0, 0, np.where(bl_, 1, 2)); kk = KERN["kern"]; pr_ = KERN["prior"]
              for c in range(3):
                  m_ = ok & (c_ == c); acc[s:e][m_] = (mass[s:e][m_] @ kk[c]) / pr_[None]
          elif OWN is None and MULTI is not None:
              qm = q[ok] * (q[ok] >= MULTI); mm = qm.sum(1); has = mm > 0; a_ = np.ones((len(mm), 10)); a_[has] = (qm[has] @ PT) / mm[has, None]
              acc[s:e][ok] = a_; top[s:e][ok] = mm; nmulti = (nmulti if s else 0.0) + float(((qm > 0).sum(1) >= 2).sum())
          elif OWN is None: acc[s:e][ok] = q[ok] @ PT + rest[ok, None]
          else:
              Kr = np.exp(-np.abs(np.arange(10)[:, None] - np.arange(10)[None]) / OWN); pr = np.bincount(rank, minlength=10) / len(rank)
              acc[s:e][ok] = (mass[s:e][ok] @ Kr) / (pr @ Kr)[None]
      if EM and not last:
          pi0 = np.exp(PRIOR * LOGN); pi0[dmask] = 0; pi0 /= pi0.sum(); M = macc.sum(); f = (macc + EM[0] * pi0 * M) / ((1 + EM[0]) * M)
          EMW = np.where(pi0 > 0, (f / np.maximum(pi0, 1e-300)) ** EM[1], 0.0); print(f"{b}: EM pass {emit + 1}: weights quantiles 1/50/99 % {np.quantile(EMW[pi0 > 0], .01):.2f} {np.median(EMW[pi0 > 0]):.2f} {np.quantile(EMW[pi0 > 0], .99):.2f}  max {EMW.max():.1f}", flush=True)
    res.update({f"{b}_opp_mass": mass.astype(np.float32), f"{b}_opp_top": top.astype(np.float32), f"{b}_opp_toprank": rank[tcl], f"{b}_opp_topcls": tcl, f"{b}_opp_acc": acc.astype(np.float32)})
    print(f"{b}: games {N}  with tokens {found}  mean top-1 {top.mean():.3f}  > 0.5 {np.mean(top > 0.5):.3f}  > 0.8 {np.mean(top > 0.8):.3f}" + (f"  (second-net class kept in {n_agree} games)" if AGREE else "") + (f"  mean background probability {bgsum / max(found, 1):.3f}" if (BG is not None or CONS or RAWP) else "") + (f"  sides with >= 2 candidates {nmulti / max(found, 1):.3f}" if MULTI is not None and OWN is None and KERN is None else ""), flush=True)
if OWN is not None or KERN is not None: res["acc_hd1"] = np.array(1)
np.savez(out, **res); print("saved", out)
