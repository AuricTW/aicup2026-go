"""Temperature-calibrated ensemble of identity networks (idnet.py) -> opponent-side rank evidence file (idnet_pred format).
Each network is calibrated on its own held-out owner games (one temperature, minimum NLL): long training makes the raw
softmax over-confident, and the evidence uses the top-1 probability as the confidence of a recognition.
Networks must share the class set (same --holdout-players).  Probabilities are averaged after calibration.
usage: idens.py MOVES.npz RUN_DIR[,RUN2,...] OUT.npz [--holdout-players CSV] [--temp auto|1.0]
"""
import sys, os, argparse, csv
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idnet import Net, load_trunk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--holdout-players", default=None)
    ap.add_argument("--temp", default="auto"); ap.add_argument("--tro", type=int, default=60000); ap.add_argument("--extra-split", default="")
    ap.add_argument("--lin", action="store_true"); ap.add_argument("--lin-wd", type=float, default=1e-4); ap.add_argument("--lin-epochs", type=int, default=8)
    ap.add_argument("--acc-sym", type=float, default=0.0); ap.add_argument("--acc-hd1", action="store_true")
    ap.add_argument("--acc", action="store_true"); ap.add_argument("--acc-min", type=float, default=0.2); ap.add_argument("--acc-kappa", type=float, default=5.0)
    ap.add_argument("--trunk", default=None); ap.add_argument("--trunk-scale", type=float, default=0.5)
    # --drop-share F: simulate test-like gallery coverage.  A fixed random set of accounts holding a share F of all training games is removed
    # from the gallery for the dev / test / x blocks (their probability set to 0 and the rest renormalised, as for a net that never saw them).
    ap.add_argument("--drop-share", type=float, default=0.0); ap.add_argument("--drop-seed", type=int, default=7)      # embeddings for the networks trained with idnet.py --trunk (must cover every game used here)
    a = ap.parse_args(); dev = torch.device("cuda"); z = np.load(a.data, allow_pickle=True); sp = z["split"]; pid = z["pid"].astype(np.int64); keys = z["key"]; y = z["y"].astype(np.int64)
    tr_all = np.where(np.isin(sp, [0] + [int(s) for s in a.extra_split.split(",") if s]))[0]; xs = np.zeros(0, np.int64)
    if a.holdout_players:
        rows = list(csv.DictReader(open(a.holdout_players, newline=""))); names = {str(n): i for i, n in enumerate(z["pid_names"])}
        hp = np.array(sorted({names[r["player_id"]] for r in rows if r["player_id"] in names})); kset = {r["key"] for r in rows}
        xs = np.array([i for i in np.where((sp == 0) | (sp == 3))[0] if str(keys[i]) in kset]); tr_all = tr_all[~np.isin(pid[tr_all], hp)]
    CB = torch.tensor(z["color_black"].astype(bool)); FB = torch.tensor(z["first_black"].astype(bool)); HDV = torch.tensor(z["hd"].astype(np.int64)); LENa = z["LEN"].astype(np.int64)
    nets = []; owners = None; Ms = {}; slot = TK = None
    tr_raw = tr_all
    if a.trunk:          # with embeddings: profile pass, sample of training games and the trunk networks' own hold-out use only the embedded games (as idnet.py --trunk did)
        slot, TK = load_trunk(a.trunk, keys, 80, a.trunk_scale); print("trunk embeddings for", int((slot >= 0).sum()), "of", len(slot), "games", flush=True); tr_all = tr_all[slot[tr_all] >= 0]
    for r in a.run.split(","):
        ck = torch.load(os.path.join(r, "model.pt"), map_location="cpu", weights_only=False); ar = ck["args"]
        if owners is None: owners = ck["owners"]; cls_rank = ck["cls_rank"]
        assert np.array_equal(owners, ck["owners"]) and np.array_equal(owners, np.unique(pid[tr_all if ar.get("trunk") else tr_raw])), "class sets differ (check --holdout-players)"
        L = ar["L"]; net = Net(len(owners), ar["d"], ar["layers"], 4, L, drop=ar["dropout"], cos=ar.get("cos", 0.0), rel=ar.get("rel", 0), board=ar.get("board", 0), bl=ar.get("board_layers", 4), bpatch=ar.get("board_patch", 0), trunk=TK.shape[2] if ar.get("trunk") and not ar.get("trunk_mask_only") else 0).to(dev); net.load_state_dict(ck["model"]); net.eval()
        assert net.tproj is None or L <= 80
        if L not in Ms: Ms[L] = torch.tensor(z["M"][:, :L].astype(np.int64))
        rng = np.random.RandomState(ar["seed"]); assert not ar.get("train_n")                 # the network's own held-out games (same selection as idnet.py)
        seen = {}; ho = []
        for i in rng.permutation(tr_all if ar.get("trunk") else tr_raw):
            c = seen.get(pid[i], 0)
            if c < ar["holdout"]: ho.append(i); seen[pid[i]] = c + 1
        nets.append([net, L, np.array(sorted(ho)), 1.0, r])
    C = len(owners); cls_of = -np.ones(pid.max() + 1, np.int64); cls_of[owners] = np.arange(C); R1 = torch.tensor(np.eye(10, dtype=np.float32)[cls_rank], device=dev)

    def logits(net, L, ix, opp):
        with torch.no_grad():
            for i in range(0, len(ix), 2048):
                jn = ix[i:i + 2048]; j = torch.tensor(jn); sb = CB[j].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tk = None
                    if net.tproj is not None: sl = slot[jn]; assert (sl >= 0).all(), "game without trunk embedding"; tk = TK[torch.as_tensor(sl)][:, :L].to(dev).float()
                    lo, _ = net(Ms[L][j].to(dev), torch.tensor(np.minimum(LENa[jn], L)).to(dev), ~sb if opp else sb, FB[j].to(dev), HDV[j].to(dev), tk)
                yield i, lo.float()

    for n in nets:
        net, L, ho = n[0], n[1], n[2]; lg = torch.cat([lo for _, lo in logits(net, L, ho, False)]); lab = torch.tensor(cls_of[pid[ho]], device=dev)
        if a.temp == "auto":
            Ts = np.arange(0.6, 3.01, 0.05); nll = [float(torch.nn.functional.cross_entropy(lg / t, lab)) for t in Ts]; n[3] = float(Ts[int(np.argmin(nll))])
        else: n[3] = float(a.temp)
        p = torch.softmax(lg / n[3], 1); tp, tc = p.max(1); p1 = torch.softmax(lg, 1).max(1).values
        print(f"{n[4]}: held-out top-1 {(tc == lab).float().mean():.4f}   temperature {n[3]:.2f}   mean top-1 prob raw {p1.mean():.3f} -> calibrated {tp.mean():.3f}", flush=True)

    def ens(ix, opp):
        gens = [logits(n[0], n[1], ix, opp) for n in nets]
        for parts in zip(*gens):
            yield parts[0][0], sum(torch.softmax(lo / n[3], 1) for (_, lo), n in zip(parts, nets)) / len(nets)

    res = {}; rng = np.random.RandomState(0); hoall = np.unique(np.concatenate([n[2] for n in nets])); pool = np.setdiff1d(tr_all, hoall); tro = np.sort(rng.choice(pool, min(a.tro, len(pool)), replace=False))
    # ---- account-specific opponent profile: which ranks do the OWNERS have in the training games where account p is recognised as the opponent?
    # (a label is one of ten classes and drifts; the ranks of the people an account actually meets are a sharper description of it)
    PT = None
    if a.acc:
        Y1 = torch.tensor(np.eye(10, dtype=np.float32), device=dev); CNT = torch.zeros(C, 10, device=dev); hd0 = tr_all[z["hd"][tr_all] == 0]
        for i, q in ens(hd0, True):
            jn = hd0[i:i + 2048]; q[torch.arange(len(q)), torch.tensor(cls_of[pid[jn]], device=dev)] = 0; q = q * (q > a.acc_min)
            CNT += q.t() @ Y1[torch.tensor(y[jn], device=dev)]
            if a.acc_sym:      # the other direction: in account A's own games, the labels of the opponents recognised there also describe whom A meets
                CNT.index_add_(0, torch.tensor(cls_of[pid[jn]], device=dev), a.acc_sym * (q @ R1))
            if (i // 2048) % 100 == 0: print("profile pass", i, "/", len(hd0), flush=True)
        CNT = CNT.cpu().numpy().astype(np.float64); base = np.stack([CNT[cls_rank == r].sum(0) + 1.0 for r in range(10)]); base /= base.sum(1, keepdims=True)      # by label
        P0 = CNT.sum(0) / CNT.sum(); n = CNT.sum(1)
        prof = (CNT + a.acc_kappa * base[cls_rank]) / (n + a.acc_kappa)[:, None]; PT = torch.tensor(prof / P0[None], dtype=torch.float32, device=dev)
        res.update(acc_cnt=CNT, acc_P0=P0, acc_base=base)
        PT1 = None
        if a.acc_hd1:           # the same for handicap-1 games, separately for owner black (the weaker side) / owner white
            hd1 = tr_all[z["hd"][tr_all] == 1]; CN1 = torch.zeros(2, C, 10, device=dev); cbn = z["color_black"]
            for i, q in ens(hd1, True):
                jn = hd1[i:i + 2048]; q[torch.arange(len(q)), torch.tensor(cls_of[pid[jn]], device=dev)] = 0; q = q * (q > a.acc_min); yy = Y1[torch.tensor(y[jn], device=dev)]
                bl = torch.tensor(cbn[jn], device=dev); CN1[0] += (q * bl[:, None]).t() @ yy; CN1[1] += (q * (~bl)[:, None]).t() @ yy
            CN1 = CN1.cpu().numpy().astype(np.float64); PT1 = []
            for c in range(2):
                b1 = np.stack([CN1[c][cls_rank == r].sum(0) + 1.0 for r in range(10)]); b1 /= b1.sum(1, keepdims=True); p01 = CN1[c].sum(0) / CN1[c].sum(); n1 = CN1[c].sum(1)
                PT1.append(torch.tensor(((CN1[c] + a.acc_kappa * b1[cls_rank]) / (n1 + a.acc_kappa)[:, None]) / p01[None], dtype=torch.float32, device=dev))
                print(f"HD1 profiles ({'owner black' if c == 0 else 'owner white'}): soft count {n1.sum():.0f}  accounts with >= 5: {(n1 >= 5).sum()}   label-level mode by label:", b1.argmax(1), flush=True)
            res["acc_hd1"] = np.array(1)
        print(f"profiles: accounts with >= 5 recognised games {(n >= 5).sum()}  >= 50: {(n >= 50).sum()}  total soft count {n.sum():.0f}   P(owner rank = label of the account | label) by label:", np.round(np.diag(base), 2),
              "  same for accounts with >= 50 games (own profile):", np.round(np.mean([CNT[i, cls_rank[i]] / n[i] for i in np.where(n >= 50)[0]]), 3), flush=True)
    # ---- learned profile: owner rank of a training game ~ softmax(b + sum_k q_k * (B[label_k] + D[account_k]))  over the top-8 accounts of
    # the opponent-side posterior.  Replaces the hand-made mixture: the weight of a recognition and the profile of every account are fitted
    # by likelihood on the training games (owners split 90/10 for early stopping); D is shrunk to the label-level table B by weight decay.
    LIN = None
    if a.lin:
        hd0 = tr_all[z["hd"][tr_all] == 0]; KK = 8; TV = np.zeros((len(hd0), KK), np.float32); TC = np.zeros((len(hd0), KK), np.int64)
        for i, q in ens(hd0, True):
            jn = hd0[i:i + 2048]; q[torch.arange(len(q)), torch.tensor(cls_of[pid[jn]], device=dev)] = 0; v, c = q.topk(KK, 1); TV[i:i + len(v)] = v.cpu().numpy(); TC[i:i + len(v)] = c.cpu().numpy()
        yo = torch.tensor(y[hd0], device=dev); own = pid[hd0]; uo = np.unique(own); val = np.isin(own, np.random.RandomState(1).choice(uo, len(uo) // 10, replace=False))
        TVt = torch.tensor(TV, device=dev); TCt = torch.tensor(TC, device=dev); lab_t = torch.tensor(cls_rank, device=dev)
        Bp = torch.zeros(10, 10, device=dev, requires_grad=True); Dp = torch.zeros(C, 10, device=dev, requires_grad=True); bp = torch.zeros(10, device=dev, requires_grad=True)
        def lin(v, c): return bp[None] + (v[..., None] * (Bp[lab_t[c]] + Dp[c])).sum(1)
        opt = torch.optim.Adam([{"params": [Bp, bp], "weight_decay": 0.0}, {"params": [Dp], "weight_decay": a.lin_wd}], lr=0.02)
        tri = torch.tensor(np.where(~val)[0], device=dev); vai = torch.tensor(np.where(val)[0], device=dev); best = (9e9, None)
        for ep in range(a.lin_epochs):
            perm = tri[torch.randperm(len(tri), device=dev)]
            for i in range(0, len(perm), 4096):
                j = perm[i:i + 4096]; loss = torch.nn.functional.cross_entropy(lin(TVt[j], TCt[j]), yo[j]); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): vl = float(torch.nn.functional.cross_entropy(lin(TVt[vai], TCt[vai]), yo[vai])); v0 = float(torch.nn.functional.cross_entropy(bp[None].expand(len(vai), 10), yo[vai]))
            print(f"learned profile epoch {ep}: held-out-owner NLL {vl:.4f}  (prior only {v0:.4f})", flush=True)
            if vl < best[0]: best = (vl, (Bp.detach().clone(), Dp.detach().clone()))
        LIN = best[1][0][lab_t] + best[1][1]                                           # [C, 10]
    DROP = None
    if a.drop_share > 0:
        ng = np.bincount(cls_of[pid[tr_all]], minlength=C).astype(np.float64); order_ = np.random.RandomState(a.drop_seed).permutation(C); cum = np.cumsum(ng[order_]) / ng.sum()
        dropped = order_[:int(np.searchsorted(cum, a.drop_share)) + 1]; DROP = torch.zeros(C, dtype=torch.bool, device=dev); DROP[torch.tensor(dropped, device=dev)] = True
        print(f"drop-share {a.drop_share}: {len(dropped)} of {C} accounts removed from the dev / test / x galleries ({ng[dropped].sum() / ng.sum():.3f} of the training games)", flush=True); res["dropped"] = dropped
    for name, ix in [("dev", np.where(sp == 1)[0]), ("test", np.where(sp == 2)[0])] + ([("x", xs)] if len(xs) else []) + [("tro", tro)]:
        mass = []; top = []; tcl = []; acc = []; linv = []
        for i, q in ens(ix, True):
            if name == "tro": q[torch.arange(len(q)), torch.tensor(cls_of[pid[ix[i:i + 2048]]], device=dev)] = 0; q = q / q.sum(1, keepdims=True)
            elif DROP is not None: q = q.masked_fill(DROP[None], 0); q = q / q.sum(1, keepdims=True)
            tp, tc = q.max(1); mass.append((q @ R1).cpu().numpy()); top.append(tp.cpu().numpy()); tcl.append(tc.cpu().numpy())
            if PT is not None:
                ab = q @ PT
                if PT1 is not None and name != "tro":
                    jn = ix[i:i + 2048]; h1 = torch.tensor(z["hd"][jn] == 1, device=dev); bl = torch.tensor(z["color_black"][jn], device=dev)
                    ab = torch.where((h1 & bl)[:, None], q @ PT1[0], torch.where((h1 & ~bl)[:, None], q @ PT1[1], ab))
                acc.append(ab.cpu().numpy())
            if LIN is not None: v, c = q.topk(8, 1); linv.append((v[..., None] * LIN[c]).sum(1).cpu().numpy())
        mass = np.concatenate(mass); top = np.concatenate(top); tcl = np.concatenate(tcl)
        if PT is not None and name != "tro": res[f"{name}_opp_acc"] = np.concatenate(acc)
        if LIN is not None and name != "tro": res[f"{name}_opp_lin"] = np.concatenate(linv)
        if name == "tro": res.update(tro_mass=mass, tro_top=top, tro_toprank=cls_rank[tcl], tro_rank=y[ix], tro_hd=z["hd"][ix], tro_black=z["color_black"][ix])
        else:
            res.update({f"{name}_opp_mass": mass, f"{name}_opp_top": top, f"{name}_opp_toprank": cls_rank[tcl], f"{name}_opp_topcls": tcl, f"{name}_own_mass": np.full((len(ix), 10), 0.1, np.float32),
                        f"{name}_own_top": np.zeros(len(ix), np.float32), f"{name}_keys": keys[ix], f"{name}_hd": z["hd"][ix], f"{name}_black": z["color_black"][ix], f"{name}_rank": y[ix], f"{name}_pid": pid[ix]})
        print(f"{name}: games {len(ix)}  mean top-1 prob {top.mean():.3f}  share > 0.3 {(top > 0.3).mean():.3f}  > 0.6 {(top > 0.6).mean():.3f}", flush=True)
    res.update(ho_rank=cls_rank, class_names=z["pid_names"][owners], cls_rank=cls_rank, temps=np.array([n[3] for n in nets]))
    np.savez(a.out, **res); print("saved", a.out)


if __name__ == "__main__":
    main()
