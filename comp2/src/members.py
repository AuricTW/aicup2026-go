"""The 91 members of the final submission file and how each one is built (Python form of the original PowerShell drivers
make_variants2/6/7/8.ps1, cnt_round.ps1 and blend_groups.ps1).

Group order and member order are the ones of the final blend ("new+rg3+fbf+fhf+fmf+fnf+rt+fbc+fhc+fmc+fnc+r1").
Each member is a dict: tag, ev (token-evidence set of the id2 source), env (environment variables of fuse_sources.py), and
for the r1 group the extra sources / rank files.
"""
P5 = ["p8", "p10", "pt1", "pf6a", "pgate"]
EV5 = {"p8": "p8", "p10": "p10", "pt1": "pt1", "pf6a": "pf6a", "pgate": "p8"}      # pgate = p8 evidence + ZOPP_GATE 1.0
RIDGE = "0.3,0.03"
SFX = ["", "_s1", "_s2"]

# token nets (comp1 heads).  frozen trunk: WORK/tokres/{td,oof}_<net>.npz ; fine-tuned trunks: WORK/xf/{td,oof}_<head>_<trunk>.npz
EARLY8 = ["vm_t1", "vm_t6", "full_s1", "full_big_s3", "full6a_s12", "vm_t4", "vm_t10", "vm_t2"]
P10 = EARLY8 + ["full_late_s24", "vm_t17late"]
HEADS10 = EARLY8 + ["t17late", "s24"]
TOKSETS = {      # name -> (frozen-trunk nets, tuned-trunk nets, with listed-game counts)
    "p8": (EARLY8, [], False), "p10": (P10, [], False), "pt1": (["vm_t1"], [], False), "pf6a": (["full6a_s12"], [], False),
    "qb": (P10, ["t17late_t6", "t17late_b3", "s24_t6", "s24_b3"], False),
    "qm": (P10, ["t17late_t6", "t17late_b3", "s24_t6", "s24_b3", "t17late_ha", "s24_ha"], False),
    "qh": (P10, [f"{h}_b3" for h in HEADS10] + [f"{h}_t6" for h in HEADS10], False),
    "qn": (P10, [f"{h}_b3" for h in HEADS10] + [f"{h}_t6" for h in HEADS10] + [f"{h}_ha" for h in HEADS10], False),
}
for _q in "bhmn": TOKSETS[f"q{_q}c"] = TOKSETS[f"q{_q}"][:2] + (True,)

R1_ARMS = {"B": ("zopp,zown,id,id2,down", "own"), "C": ("zopp,zown,id,id2,down,dopp", "conv"), "C0": ("zopp,zown,id,id2,down,dopp", "raw")}


def _seed_env(sfx):
    return {"DEV_SEED": sfx[2:]} if sfx else {}


def groups():
    g = {}
    g["new"] = [dict(tag=p, ev=EV5[p], env={"ZOPP_GATE": "1.0"} if p == "pgate" else {}) for p in P5]
    g["rg3"] = [dict(tag=f"{p}_rg_s{s}", ev=EV5[p], env=dict({"S1_RIDGE": RIDGE, "DEV_SEED": str(s), "OMP_NUM_THREADS": "4"}, **({"ZOPP_GATE": "1.0"} if p == "pgate" else {})))
                for s in range(3) for p in P5]
    for q in "bhmn":
        g[f"f{q}f"] = [m for x in SFX for m in (dict(tag=f"q{q}{x}", ev=f"q{q}", env=_seed_env(x)), dict(tag=f"q{q}g{x}", ev=f"q{q}", env=dict(_seed_env(x), ZOPP_GATE="1.0")))]
    g["rt"] = [m for q in "bhmn" for m in (dict(tag=f"q{q}r", ev=f"q{q}", env={"S1_RIDGE": RIDGE}), dict(tag=f"q{q}gr", ev=f"q{q}", env={"S1_RIDGE": RIDGE, "ZOPP_GATE": "1.0"}))]
    for q in "bhmn":
        g[f"f{q}c"] = [m for x in SFX for m in (dict(tag=f"q{q}c{x}", ev=f"q{q}c", env=_seed_env(x)), dict(tag=f"q{q}cg{x}", ev=f"q{q}c", env=dict(_seed_env(x), ZOPP_GATE="1.0")))]
    g["r1"] = [dict(tag=f"{a}_{p}", ev=EV5[p], env={"ZOPP_GATE": "1.0"} if p == "pgate" else {}, sources=R1_ARMS[a][0], rank=R1_ARMS[a][1]) for a in R1_ARMS for p in P5]
    return g


FINAL_BUILD = "new+rg3+fbf+fhf+fmf+fnf+rt+fbc+fhc+fmc+fnc+r1"


def final_members():
    g = groups()
    return [m for name in FINAL_BUILD.split("+") for m in g[name]]


if __name__ == "__main__":
    ms = final_members(); print(len(ms), "members"); print(" ".join(m["tag"] for m in ms))
