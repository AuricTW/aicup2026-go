"""Re-estimate the final BatchNorm statistics of a token net and re-embed the official games.
Why: tokhead_vm_20261116_v1.py trains on buffers of contiguous 512-game blocks; the BN running statistics (momentum 0.1) therefore
reflect only the last few buffers, i.e. a narrow set of players.  Seed 2 (vm_t2) scored 0.676 vs 0.714 for seeds 0/1 with the same
config and a normal training loss, and its official embedding norms are far more uniform -> suspect the BN statistics.
Modes: train = statistics of the pre-BN features over a random sample of training games from all token dirs (small random blocks);
       off   = statistics of the official games themselves (transductive, label-free).
usage: tokbn_recal_20261117_v1.py <model dir> <out dir> <mode train|off> <official dir> <token dir> [<token dir> ...] [--games 40000]
writes <out dir>/off_emb.npy (official_results order, same as the model dir's) and RECAL.json
"""
from pathlib import Path
import sys, json, argparse, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tokhead_vm_20261116_v1 as tv
import numpy as np
import torch
from torch import nn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('out'); ap.add_argument('mode'); ap.add_argument('official'); ap.add_argument('dirs', nargs='*'); ap.add_argument('--games', type=int, default=40000); ap.add_argument('--chunk', type=int, default=128); ap.add_argument('--seed', type=int, default=0); v = ap.parse_args()
    out = Path(v.out); out.mkdir(parents=True, exist_ok=False); dev = 'cuda'; t0 = time.time(); ck = torch.load(Path(v.model) / 'model.pt', map_location='cpu', weights_only=False); a = ck['args']
    net = tv.TokNet(len(ck['classes']), a['d'], a['layers'], a['drop']).to(dev); net.load_state_dict(ck['model']); net.eval(); mu, sd = ck['mu'].to(dev), ck['sd'].to(dev); bn = net.out[3]; assert isinstance(bn, nn.BatchNorm1d); lab = np.load(Path(__file__).resolve().parents[1] / 'runs/tokpilot_20261116_v1/labels_all.npz')
    norm = lambda z_, a_: ((torch.cat([z_.float(), a_.float()], 2) - mu) / sd).clamp(-8, 8)
    def feats(D, sl, black, pre):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            x = norm(torch.tensor(np.asarray(D['tok_z'][sl])).to(dev), torch.tensor(np.asarray(D['tok_a'][sl])).to(dev)); pl = torch.tensor(np.asarray(D['tok_ply'][sl]).astype(np.int64)).to(dev); n = torch.tensor(np.asarray(D['tok_n'][sl]).astype(np.int64)).to(dev); b = torch.tensor(black).to(dev)
            if pre: bn_ = net.out[3]; net.out[3] = nn.Identity(); f = net.embed(x, pl, n, b); net.out[3] = bn_; return f.float()
            return net.embed(x, pl, n, b).float()
    O = tv.opend(v.official); ob = lab['off_black']; NO = len(O['indices'])      # official colours in official_results order
    if v.mode == 'off': F = torch.cat([feats(O, slice(i, i + 1024), ob[i:i + 1024], True) for i in range(0, NO, 1024)])
    else:
        col = {int(i): bool(b) for i, b in zip(lab['ids'], lab['black'])}; D = [tv.opend(d) for d in v.dirs]; sizes = np.array([len(d['indices']) for d in D]); rng = np.random.default_rng(v.seed); F = []
        for _ in range(v.games // v.chunk):
            k = rng.choice(len(D), p=sizes / sizes.sum()); s = int(rng.integers(0, sizes[k] - v.chunk)); sl = slice(s, s + v.chunk); ids = np.asarray(D[k]['indices'][sl]); F.append(feats(D[k], sl, np.array([col.get(int(i), True) for i in ids]), True))
        F = torch.cat(F)
    old = (bn.running_mean.clone(), bn.running_var.clone()); bn.running_mean.copy_(F.mean(0)); bn.running_var.copy_(F.var(0, unbiased=True))
    E = torch.cat([feats(O, slice(i, i + 1024), ob[i:i + 1024], False) for i in range(0, NO, 1024)]).cpu().numpy(); np.save(out / 'off_emb.npy', E)
    shift = float(((old[0] - bn.running_mean) / bn.running_var.sqrt()).abs().mean()); ratio = float((old[1] / bn.running_var).mean())
    json.dump({'model': str(v.model), 'mode': v.mode, 'games': int(len(F)), 'mean_abs_shift_in_sd': shift, 'mean_var_ratio_old_over_new': ratio, 'seconds': time.time() - t0}, open(out / 'RECAL.json', 'w'), indent=1); print(v.model, v.mode, 'shift', round(shift, 3), 'var ratio', round(ratio, 3), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
