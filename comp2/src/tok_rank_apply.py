"""Apply a token RANK net (tok_ranknet_vm.py) to comp2 own-side and opponent-side tokens -> per-game rank log-probabilities in the
format of fuse_sources.py --dir-x / --dir-full / --dir-test (sources 'down' = own side, 'dopp' = opponent side).
Blocks: dev (split 1), x (OOF-query games), test.
usage: tok_rank_apply.py MODEL.pt[,MODEL2.pt] MOVES3.npz OUT.npz OWN_SPEC[;OWN_SPEC2] OPP_SPEC[;OPP_SPEC2]
       SPEC = results_dir|games.jsonl|keys.json
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok_ranknet_vm import RankNet


def run(models, specs):
    out = {}
    for spec in specs.split(";"):
        rd, gj, kj = spec.split("|"); keys = json.load(open(kj)); col = [json.loads(l)["color"] for l in open(gj, encoding="utf-8")]
        Z = np.load(rd + "/tok_z.npy", mmap_mode="r"); A = np.load(rd + "/tok_a.npy", mmap_mode="r"); P = np.load(rd + "/tok_ply.npy", mmap_mode="r"); N = np.load(rd + "/tok_n.npy"); bl = torch.tensor([c == "B" for c in col])
        acc = np.zeros((len(keys), 10))
        for net, mu, sd in models:
            with torch.no_grad():
                for i in range(0, len(keys), 512):
                    x = ((torch.cat([torch.tensor(np.asarray(Z[i:i + 512])).cuda().float(), torch.tensor(np.asarray(A[i:i + 512])).cuda().float()], 2) - mu) / sd).clamp(-8, 8)
                    lo = net(x, torch.tensor(np.asarray(P[i:i + 512]).astype(np.int64)).cuda(), torch.tensor(N[i:i + 512].astype(np.int64)).cuda(), bl[i:i + 512].cuda())
                    acc[i:i + 512] += torch.log_softmax(lo.float(), 1).cpu().numpy() / len(models)
        out.update({k: acc[j] for j, k in enumerate(keys)})
        print(rd, len(keys), flush=True)
    return out


def main():
    models = []
    for p in sys.argv[1].split(","):
        ck = torch.load(p, map_location="cpu", weights_only=False); a = ck["args"]; net = RankNet(a["d"], a["layers"], a["drop"]).cuda(); net.load_state_dict(ck["model"]); net.eval(); models.append((net, ck["mu"].cuda(), ck["sd"].cuda()))
    z = np.load(sys.argv[2], allow_pickle=True); keys = z["key"].astype(str); sp = z["split"]; y = z["y"]; pos = {k: i for i, k in enumerate(keys)}
    own = run(models, sys.argv[4]); opp = run(models, sys.argv[5]); res = {}
    blocks = {"dev": [k for k in keys[sp == 1]], "test": [k for k in keys[sp == 2]], "x": sorted(k for k in own if "#" in k and sp[pos[k]] != 1)}
    for b, ks in blocks.items():
        ks = [k for k in ks if k in own and k in opp]
        res[f"{b}_keys"] = np.array(ks); res[f"{b}_own_logp"] = np.stack([own[k] for k in ks]).astype(np.float32); res[f"{b}_opp_logp"] = np.stack([opp[k] for k in ks]).astype(np.float32)
        if b != "test":
            yy = y[[pos[k] for k in ks]]; po = res[f"{b}_own_logp"].argmax(1); pp = res[f"{b}_opp_logp"].argmax(1)
            print(f"{b}: games {len(ks)}  own side: per-game accuracy {np.mean(po == yy):.3f} within one {np.mean(np.abs(po - yy) <= 1):.3f} NLL {-res[f'{b}_own_logp'][np.arange(len(ks)), yy].mean():.3f}"
                  f"   opponent side (predicting the player's rank): accuracy {np.mean(pp == yy):.3f} within one {np.mean(np.abs(pp - yy) <= 1):.3f}")
    np.savez(sys.argv[3], **res); print("saved", sys.argv[3])


if __name__ == "__main__":
    main()
