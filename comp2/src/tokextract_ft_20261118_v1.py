"""Stage 2 of HumanSL trunk fine-tuning: re-extract the per-decision tokens of a cached dir with the tuned trunk
(tokft_e2e_20261118_v1.py -> trunk_ft.pt) from the packed input cache (tokcache_20261118_v1.py); GPU only, no board replay.
Output = the formats the existing trainers read:
  <out>/        indices, tok_z [N,90,1152] f16, tok_a [N,90,10] f32, tok_ply [N,90] i16, tok_n [N] u8          (plies 1-180)
  <out>_late/   indices, tok_z [N,60,1152] f16, tok_a [N,60,10] f32, tok_ply, tok_n, tok_act [N,60] u16         (plies 181-300)
--trunk none = frozen HumanSL trunk (parity with the accepted tokens).  --order <indices.npy>: write the games in that order
(official games: the official_merged order, so that official_mv.npy and the registration scripts stay aligned).
usage: tokextract_ft_20261118_v1.py <payload dir> <weight.pt> <trunk_ft.pt|none> <cache dir> <out dir> [--order idx.npy] [--pos 1024] [--mem-frac 0]
"""
import sys, os, json, time, argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('payload'); ap.add_argument('weight'); ap.add_argument('trunk'); ap.add_argument('cache'); ap.add_argument('out'); ap.add_argument('--order', default=''); ap.add_argument('--pos', type=int, default=1024); ap.add_argument('--mem-frac', type=float, default=0.); v = ap.parse_args()
    dev = 'cuda'
    if v.mem_frac > 0: torch.cuda.set_per_process_memory_fraction(v.mem_frac)
    for q in ('vendor', 'code', ''): sys.path.insert(0, os.path.join(os.path.abspath(v.payload), q))
    import worker as W
    from katago.train.load_model import load_model
    from katago.train.model_pytorch import ExtraOutputs
    model, swa, _ = load_model(v.weight, use_swa=False, device=torch.device(dev), pos_len=19); del swa; model.eval().requires_grad_(False)
    if v.trunk != 'none':
        st = torch.load(v.trunk, map_location=dev, weights_only=False); fb = st['first_block']
        for b, s in zip(model.blocks[fb:], st['blocks']): b.load_state_dict(s)
        model.norm_trunkfinal.load_state_dict(st['norm']); print('loaded tuned trunk from block', fb, flush=True)
    META = {b: torch.tensor(W.n.profile_metadata('rank_1d').get_metadata_row('B' if b else 'W', 361)).to(dev) for b in (True, False)}
    C = {k: np.load(os.path.join(v.cache, f'{k}.npy'), mmap_mode='r') for k in ('indices', 'px', 'g', 'legal', 'act', 'ply', 'n', 'black', 'hd')}; ids = np.asarray(C['indices']); n = np.asarray(C['n']).astype(np.int64); bl = np.asarray(C['black']); hd = np.asarray(C['hd']); PLY = np.asarray(C['ply']); ACT = np.asarray(C['act'])
    if v.order: order = np.load(v.order); row = {int(i): j for j, i in enumerate(ids)}; rows = np.array([row[int(i)] for i in order])
    else: order = ids; rows = np.arange(len(ids))
    N = len(rows); t0 = time.time(); late = v.out.rstrip('/\\') + '_late'; os.makedirs(v.out, exist_ok=False); os.makedirs(late, exist_ok=False)
    mm = lambda d, name, dt, shape: np.lib.format.open_memmap(os.path.join(d, f'{name}.npy'), mode='w+', dtype=dt, shape=shape)
    E = {'z': mm(v.out, 'tok_z', np.float16, (N, 90, 1152)), 'a': mm(v.out, 'tok_a', np.float32, (N, 90, 10)), 'ply': mm(v.out, 'tok_ply', np.int16, (N, 90)), 'n': mm(v.out, 'tok_n', np.uint8, (N,))}
    Lt = {'z': mm(late, 'tok_z', np.float16, (N, 60, 1152)), 'a': mm(late, 'tok_a', np.float32, (N, 60, 10)), 'ply': mm(late, 'tok_ply', np.int16, (N, 60)), 'n': mm(late, 'tok_n', np.uint8, (N,)), 'act': mm(late, 'tok_act', np.uint16, (N, 60))}; Lt['act'][:] = 362

    def tokens(px, g, legal, act, ply, black, hdv):
        x = torch.from_numpy(np.unpackbits(px, axis=1)[:, :22 * 361].reshape(-1, 22, 19, 19)).to(dev).float(); lg = torch.from_numpy(np.unpackbits(legal, axis=1)[:, :362]).to(dev).bool(); gg = torch.from_numpy(np.ascontiguousarray(g)).to(dev).float()
        ac = torch.from_numpy(act.astype(np.int64)).to(dev); b_ = torch.from_numpy(black).to(dev); meta = torch.where(b_[:, None], META[True][None], META[False][None]); pl = torch.from_numpy(ply.astype(np.float32)).to(dev); hh = torch.from_numpy(hdv.astype(np.float32)).to(dev)
        extra = ExtraOutputs(['trunkfinal'])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16): out = model(x, gg, input_meta=meta, extra_outputs=extra)
        logits = out[0][0][:, 0, :].float(); pi = torch.softmax(logits.masked_fill(~lg, -1e9), -1); h = extra.returned['trunkfinal'].float().flatten(2)
        local = h.gather(2, ac.clamp_max(360)[:, None, None].expand(-1, h.shape[1], 1)).squeeze(2) * (ac != 361)[:, None]; expected = (h * pi[:, None, :361]).sum(2); z = torch.cat((local, h.mean(2), local - expected), 1)
        chosen = pi.gather(1, ac[:, None]).squeeze(1); rank = ((pi > chosen[:, None]) & lg).sum(1).float() / lg.sum(1); ent = -(pi * torch.log(pi.clamp_min(1e-12))).sum(1); top = pi.topk(2, dim=1).values
        a = torch.stack(((pl <= 60).float(), (pl > 60).float(), torch.log(chosen.clamp_min(1e-12)), rank, ent, torch.log(top[:, 0].clamp_min(1e-12)), top[:, 0] - top[:, 1], (pl / 180).clamp(max=1), hh, (ac == 361).float()), 1)
        return z.half().cpu().numpy(), a.cpu().numpy()

    i = 0; last = 0
    while i < N:
        j = i; tot = 0
        while j < N and (j == i or tot + n[rows[j]] <= v.pos): tot += n[rows[j]]; j += 1
        rr = rows[i:j]; o = np.argsort(rr); rs = rr[o]; nn_ = n[rs]
        if nn_.sum() > 0:
            gi = np.repeat(np.arange(len(rs)), nn_); di = np.concatenate([np.arange(k) for k in nn_]); px = np.asarray(C['px'][rs]); g = np.asarray(C['g'][rs]); lg = np.asarray(C['legal'][rs])
            z, a = tokens(px[gi, di], g[gi, di], lg[gi, di], ACT[rs][gi, di], PLY[rs][gi, di], bl[rs][gi], hd[rs][gi]); st = np.concatenate([[0], np.cumsum(nn_)])
            for q in range(len(rs)):
                dst = i + int(o[q]); pl = PLY[rs[q], :nn_[q]]; ne = int((pl <= 180).sum()); nl = int(nn_[q]) - ne; assert ne <= 90 and nl <= 60; zz = z[st[q]:st[q + 1]]; aa = a[st[q]:st[q + 1]]
                E['z'][dst, :ne] = zz[:ne]; E['a'][dst, :ne] = aa[:ne]; E['ply'][dst, :ne] = pl[:ne]; E['n'][dst] = ne
                Lt['z'][dst, :nl] = zz[ne:]; Lt['a'][dst, :nl] = aa[ne:]; Lt['ply'][dst, :nl] = pl[ne:]; Lt['n'][dst] = nl; Lt['act'][dst, :nl] = ACT[rs[q], ne:ne + nl]
        i = j
        if i - last >= 4000 or i == N: last = i; print(json.dumps({'games': i, 'total': N, 'seconds': round(time.time() - t0)}), flush=True)
    for d_ in (E, Lt): [x.flush() for x in d_.values()]
    np.save(os.path.join(v.out, 'indices.npy'), np.asarray(order)); np.save(os.path.join(late, 'indices.npy'), np.asarray(order))
    rc = {'games': N, 'trunk': v.trunk, 'cache': v.cache, 'mean_n_early': float(np.asarray(E['n']).mean()), 'mean_n_late': float(np.asarray(Lt['n']).mean()), 'seconds': time.time() - t0, 'GPU': torch.cuda.get_device_name()}
    json.dump(rc, open(os.path.join(v.out, 'RECEIPT.json'), 'w'), indent=1); json.dump(rc, open(os.path.join(late, 'RECEIPT.json'), 'w'), indent=1); print(json.dumps(rc), flush=True)


if __name__ == '__main__':
    main()
