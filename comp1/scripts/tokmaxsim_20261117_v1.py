"""Decision-level matching (MaxSim, no training) with a trained token net.
Per-decision vectors = the token net's Transformer outputs before pooling (normalised).  score(question, candidate) = mean over the
question's games of [ mean over the game's decisions of the max cosine to any decision of the candidate's gallery games ] (ColBERT style).
Why: on validation v2 it adds +0.0105 to the same model's pooled embedding and about +0.004 to the full fusion (pooling discards
decision-level correspondences).  Trained variants (tokli_20261117_v1.py) were worse; same-colour-only / phase-restricted matching worse.
Modes: val <model dir> <tag>       -> runs/suppsplit_20261115_v2/extra/maxsim_<tag>_<pool>_d<r>.npz (key ms_<tag>)
       official <model dir> <tag>  -> runs/tokmaxsim_20261117_v1/<tag>/official_<pool>.npy (all 10 support games, real questions)
Model dirs from tokhead_vm_20261117_v2/v3 (v3 models with move coordinates use runs/tokmv_20261117_v1/official_mv.npy).
"""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch, torch.nn.functional as F
import suppsplit_eval_20261115_v2 as s2
import tokhead_vm_20261117_v3 as tv
ss = s2.ss; ROOT = ss.p.ROOT; OFF = ROOT / 'runs/tokpilot_20261116_v1/official_merged'; RUN = ROOT / 'runs/tokmaxsim_20261117_v1'; dev = 'cuda'


def features(mdir):
    ck = torch.load(Path(mdir) / 'model.pt', map_location='cpu', weights_only=False); a = ck['args']; net = tv.TokNet(len(ck['classes']), a['d'], a['layers'], a['drop'], moves=bool(a.get('moves'))).to(dev); net.load_state_dict(ck['model']); net.eval()
    mu, sd = ck['mu'].to(dev), ck['sd'].to(dev); ob = np.load(ROOT / 'runs/tokpilot_20261116_v1/labels_all.npz')['off_black']; OMV = np.load(ROOT / 'runs/tokmv_20261117_v1/official_mv.npy') if a.get('moves') else None
    Z = np.load(OFF / 'tok_z.npy', mmap_mode='r'); Aa = np.load(OFF / 'tok_a.npy'); Pl = np.load(OFF / 'tok_ply.npy'); Nn = np.load(OFF / 'tok_n.npy'); H = torch.zeros((len(Nn), 90, a['d']), dtype=torch.float16)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(Nn), 256):
            x = ((torch.cat([torch.tensor(np.asarray(Z[i:i + 256])).to(dev).float(), torch.tensor(Aa[i:i + 256]).to(dev)], 2) - mu) / sd).clamp(-8, 8); ply = torch.tensor(Pl[i:i + 256].astype(np.int64)).to(dev); n = torch.tensor(Nn[i:i + 256].astype(np.int64)).to(dev); b = torch.tensor(ob[i:i + 256]).to(dev)
            pad = torch.arange(90, device=dev)[None] >= n[:, None]; hh = net.inp(x) + net.ply((ply.clamp(1, 180) - 1) // 10) + net.col(b.long())[:, None]
            if OMV is not None: hh = hh + net.mv(torch.tensor(OMV[i:i + 256].astype(np.int64)).to(dev))
            hh = net.enc(hh, src_key_padding_mask=pad); H[i:i + 256] = F.normalize(hh.float(), dim=-1).masked_fill(pad[..., None], 0).half().cpu()
    return H.to(dev), Nn


def scores(H, Nn, rows_sup, rows_q):
    """rows_sup: per candidate the official_merged rows of its gallery games; rows_q: per question the rows of its games"""
    C = len(rows_sup); G = torch.tensor(np.concatenate(rows_sup)).to(dev); own = torch.tensor(np.concatenate([[c] * len(s) for c, s in enumerate(rows_sup)])).to(dev); Nd = torch.tensor(Nn.astype(np.int64)).to(dev)
    GD = H[G].reshape(-1, H.shape[-1]); GO = own.repeat_interleave(90); valid = (torch.arange(90, device=dev)[None] < Nd[G][:, None]).reshape(-1); GD, GO = GD[valid], GO[valid]; out = np.zeros((len(rows_q), C))
    for qi, gs in enumerate(rows_q):
        acc = torch.zeros(C, device=dev)
        for r_ in gs:
            q = H[r_, :int(Nn[r_])]; sim = (q @ GD.T).float(); acc += torch.full((q.shape[0], C), -2., device=dev).scatter_reduce(1, GO[None].expand(q.shape[0], -1), sim, reduce='amax').mean(0)
        out[qi] = (acc / len(gs)).cpu().numpy()
    return out


def main(mode, mdir, tag):
    torch.cuda.set_per_process_memory_fraction(0.3); games, pools, keys, pos = ss.official(); oid = np.load(OFF / 'indices.npy'); assert np.array_equal(oid, np.load(ss.old.RUN / 'official_results/indices.npy')); rowof = np.zeros(len(keys), int); rowof[oid] = np.arange(len(oid))
    H, Nn = features(mdir)
    if mode == 'val':
        for pool, r, ep, f in ss.each():
            out = scores(H, Nn, [np.array([rowof[g] for g in s]) for s in ep['supports']], [[rowof[g] for g in z['indices']] for z in ep['questions']])
            np.savez(ss.RUN / 'extra' / f'maxsim_{tag}_{pool}_d{r}.npz', **{f'ms_{tag}': out}); print(pool, r, 'top1', float((out.argmax(1) == np.array([z['target'] for z in ep['questions']])).mean()), flush=True)
    else:
        d = RUN / tag; d.mkdir(parents=True, exist_ok=False)
        for P in pools:
            out = scores(H, Nn, [np.array([rowof[pos[k]] for k in P.supports[c]]) for c in P.candidate_ids], [[rowof[pos[k]] for k in z['keys']] for z in P.questions]); np.save(d / f'official_{P.name}.npy', out); print(P.name, out.shape, flush=True)
        json.dump({'model': str(mdir), 'tag': tag}, open(d / 'SOURCE.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
