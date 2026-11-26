"""Numbers for 陸、分析與結論 (comp2 final submission), from the saved OOF posteriors of the 91 members.
Validation questions: 7,592 OOF questions of 949 players; full coverage and the four matched mirrors (20 % of the gallery removed).
Scores use the official per-question score (exact 1, adjacent e^-1) after expected-score decoding, averaged over questions
(the OOF question set is balanced over ranks, as everywhere in the project).  Output: analysis_numbers.json next to this script.
usage: python analysis_numbers.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from comp2_paths import WORK, D64_Q, path      # noqa: E402
SUB = str(path("submissions", "work/submissions"))      # <SUB>/<run>/parts/*_OOF_posteriors.npz of the members (not in the release)
FDIRS = [os.path.join(SUB, d, "parts") for d in ("identity_fusion_20261118_v10i_prior", "identity_fusion_20261119_v10m_hedge",
                                                 "identity_fusion_20261119_v10p_ft", "identity_fusion_20261120_v10u_prof")]
HDIRS = [os.path.join(str(WORK), "tokeval", d) for d in ("hybparts_m6", "hybparts_m6_s11", "hybparts_m6_s12", "hybparts_m6_s13")]
ABC = os.path.join(str(WORK), "tokeval", "abc")
src = open(os.path.join(HERE, "abc_union.py"), encoding="utf-8").read()
helpers = src.split("LF, meta = logs(")[0].split("fdirs, hdirs, root = ")[0] + src.split("fdirs, hdirs, root = sys.argv[1].split(\",\"), sys.argv[2].split(\",\"), sys.argv[3]")[1].split("LF, meta = logs(")[0]
ns = {"__name__": "x", "sys": __import__("sys")}
exec(compile(helpers, "abc_union_helpers", "exec"), ns)
K = ns["K"]; tracks = ns["tracks"]
COMP = "new+rg+free+rt+enr+B+C+C0"   # = final 91 members (r1 = B, C, C0)


def blend(L):
    M = np.concatenate([L[g] for g in COMP.split("+")]).mean(0)
    P = np.exp(M - M.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    a = (P @ K).argmax(1)
    return P, a


LF, meta = ns["logs"](FDIRS, os.path.join(ABC, "full"))
y = meta["y"].astype(int); G = meta["G"].astype(int); pid = meta["player_id"]; qid = meta["question"]
n_members = sum(len(LF[g]) for g in COMP.split("+"))
PF, aF = blend(LF); sF = K[aF, y]
LH = [ns["logs"]([d], os.path.join(ABC, t))[0] for d, t in zip(HDIRS, tracks)]
sH = [K[blend(L)[1], y] for L in LH]; sHm = np.mean(sH, 0)
up, inv = np.unique(pid, return_inverse=True)


def ci(v):
    m = np.bincount(inv, weights=v) / np.bincount(inv); se = m.std() / np.sqrt(len(m))
    return [float(v.mean()), float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)]


out = {"n_questions": int(len(y)), "n_players": int(len(up)), "members": int(n_members),
       "full": ci(sF), "mirror_mean": ci(sHm), "mirrors": [float(s.mean()) for s in sH],
       "exact_full": float((aF == y).mean()), "adjacent_full": float((np.abs(aF - y) == 1).mean()),
       "far_full": float((np.abs(aF - y) >= 2).mean())}
out["by_G"] = {int(g): {"full": ci(sF * (G == g) / (G == g).mean()) if False else [float(sF[G == g].mean())],
                        "mirror": [float(sHm[G == g].mean())],
                        "exact": float((aF[G == g] == y[G == g]).mean())} for g in sorted(set(G))}
RN = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
out["by_rank"] = {RN[r]: {"score": float(sF[y == r].mean()), "exact": float((aF[y == r] == r).mean()),
                          "adjacent": float((np.abs(aF[y == r] - r) == 1).mean()), "n": int((y == r).sum())} for r in range(10)}
conf = np.zeros((10, 10), int)
for t, p in zip(y, aF):
    conf[t, p] += 1
out["confusion"] = conf.tolist()

# recognition level of a question: mean over its games of the opponent-side top-1 probability (x-track evidence files)
import csv
q_all = json.load(open(D64_Q, encoding="utf8"))
dmap = {}
with open(os.path.join(str(WORK), "oof_dense_map.csv"), newline="", encoding="utf8") as fh:
    for r in csv.DictReader(fh):
        dmap[int(r["dense"])] = r["key"]
        dmap.setdefault(("row", int(r["row_idx"])), r["key"])


def tops(path):
    z = np.load(path, allow_pickle=True)
    return {str(k).replace("|", "#"): float(t) for k, t in zip(z["x_keys"], z["x_opp_top"])}


T1 = tops(os.path.join(str(WORK), "idnet_tx2_full", "acc_6m.npz"))
T2 = tops(os.path.join(str(WORK), "tokeval", "x_p10_A1.0.npz"))
kpid = {}
with open(os.path.join(str(WORK), "oof_games.csv"), newline="", encoding="utf8") as fh:
    for r in csv.DictReader(fh):
        kpid[r["key"]] = r["player_id"]
rec = np.full(len(y), np.nan); miss = 0; badp = 0
for i, q in enumerate(qid):
    qq = q_all[int(q)]
    ks = [dmap[int(ri)] for ri in qq["row_indices"] if int(ri) in dmap]
    if len(ks) != len(qq["row_indices"]):
        miss += 1
    if any(kpid.get(k, pid[i]) != pid[i] for k in ks):
        badp += 1; continue
    v = [max(T1.get(k, 0.0), T2.get(k, 0.0)) for k in ks if (k in T1 or k in T2)]
    if v:
        rec[i] = np.mean(v)
out["recognition_missing_questions"] = int(np.isnan(rec).sum()); out["recognition_incomplete_games"] = miss; out["recognition_player_mismatch"] = badp
ok = ~np.isnan(rec)
if ok.sum() > 1000:
    edges = np.quantile(rec[ok], [0, 0.25, 0.5, 0.75, 1.0])
    bins = np.clip(np.searchsorted(edges, rec, side="right") - 1, 0, 3)
    out["by_recognition"] = [{"lo": float(edges[b]), "hi": float(edges[b + 1]),
                              "full": float(sF[ok & (bins == b)].mean()), "mirror": float(sHm[ok & (bins == b)].mean()),
                              "exact": float((aF[ok & (bins == b)] == y[ok & (bins == b)]).mean()),
                              "n": int((ok & (bins == b)).sum())} for b in range(4)]
    med = np.median(rec[ok]); hiR = ok & (rec >= med); loR = ok & (rec < med)
    def grp(mask):
        return {"n": int(mask.sum()), "score": float(sF[mask].mean()), "exact": float((aF[mask] == y[mask]).mean())}
    out["examples"] = {
        "edge_ranks_G20_highrec": grp(np.isin(y, [0, 9]) & (G == 20) & hiR),
        "edge_ranks_G5_lowrec": grp(np.isin(y, [0, 9]) & (G == 5) & loR),
        "1D2D_G20_highrec": grp(np.isin(y, [4, 5]) & (G == 20) & hiR),
        "1D2D_G5_lowrec": grp(np.isin(y, [4, 5]) & (G == 5) & loR),
        "all_G5_lowrec": grp((G == 5) & loR), "all_G20_highrec": grp((G == 20) & hiR)}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_numbers.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("confusion",)}, indent=1)[:4000])
