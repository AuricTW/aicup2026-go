"""Rematch evidence from token-net OPPONENT identification (comp1, new pools).
Why: many opponents of the official games are labelled training accounts (seq net: 36 % of opponent sides recognised > .5 vs 21 % false
positives on the players' own sides); two games of the same candidate share a recognised opponent ~20x more often than games of
different candidates.  The token nets identify training players from one game ~2x better than the seq net (comp2 session: top-1 0.50).
Input: opponent-side token replay of the 14,200 official games (payload official_opp_20261117 = colour flipped; tokens in official_results
order, like official_merged) and several trained token nets (model.pt with the training-account classifier W and the class list).
Steps: eval-mode embeddings -> cosine logits vs W (scale 30) -> softmax over training accounts, averaged over nets in label-id space ->
sparse top-K posteriors per game.  Components per validation v2 episode (extra/tokopp_<tag>_<pool>_d<r>.npz):
  ro_<tag>   = sum over query games g and gallery games h of the candidate of  p_g . p_h   (same-account probability)
  rh_<tag>   = count of (g, h) pairs with the same top-1 account, both with top-1 posterior > .8
Modes: val <opp token dir> <tag> <model dir> [<model dir> ...]   |   official <opp token dir> <tag> <model dir> ...  (writes runs/tokopp_20261117_v1/<tag>/official_<pool>.npz)
"""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import torch.nn.functional as F
import tokhead_vm_20261117_v2 as tv
ROOT = Path(__file__).resolve().parents[1]; LAB = ROOT / 'runs/tokpilot_20261116_v1/labels_all.npz'; RUN = ROOT / 'runs/tokopp_20261117_v1'; K = 20


def posteriors(odir, mdirs):
    O = {k: np.load(Path(odir) / f'{k}.npy', mmap_mode='r') for k in ('indices', 'tok_z', 'tok_a', 'tok_ply', 'tok_n')}; ob = ~np.load(LAB)['off_black'].astype(bool)      # opponent colour
    idx_ref = np.load(ROOT / 'runs/tokpilot_20261116_v1/official_merged/indices.npy'); assert np.array_equal(np.asarray(O['indices']), idx_ref)
    acc = None; dev = 'cuda'
    for md in mdirs:
        ck = torch.load(Path(md) / 'model.pt', map_location='cpu', weights_only=False); a = ck['args']; net = tv.TokNet(len(ck['classes']), a['d'], a['layers'], a['drop']).to(dev); net.load_state_dict(ck['model']); net.eval()
        mu, sd = ck['mu'].to(dev), ck['sd'].to(dev); W = F.normalize(net.W.detach(), dim=1); cls = np.asarray(ck['classes']); P = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for i in range(0, len(ob), 512):
                z = torch.tensor(np.asarray(O['tok_z'][i:i + 512])).to(dev); aa = torch.tensor(np.asarray(O['tok_a'][i:i + 512])).to(dev); x = ((torch.cat([z.float(), aa.float()], 2) - mu) / sd).clamp(-8, 8)
                e = net.embed(x, torch.tensor(np.asarray(O['tok_ply'][i:i + 512]).astype(np.int64)).to(dev), torch.tensor(np.asarray(O['tok_n'][i:i + 512]).astype(np.int64)).to(dev), torch.tensor(ob[i:i + 512]).to(dev))
                P.append(torch.softmax(30 * F.normalize(e.float(), dim=1) @ W.float().T, 1).cpu().numpy().astype(np.float32))
        P = np.concatenate(P); full = np.zeros((len(P), int(cls.max()) + 1 if acc is None else acc.shape[1]), np.float32)
        if acc is not None and cls.max() >= acc.shape[1]: acc = np.pad(acc, ((0, 0), (0, int(cls.max()) + 1 - acc.shape[1]))); full = np.zeros_like(acc)
        full[:, cls] = P; acc = full if acc is None else acc + full; print(md, 'top-1 posterior mean', round(float(P.max(1).mean()), 3), 'share > .5', round(float((P.max(1) > .5).mean()), 3), flush=True); del P
    acc /= len(mdirs); top = np.argsort(-acc, 1)[:, :K]; pt = np.take_along_axis(acc, top, 1); return top, pt


def scores(top, pt, rowof, sup, questions):
    """sup: candidate -> list of pos indices; questions: list of lists of pos indices"""
    C = len(sup); acct = {}
    for c, g in enumerate(sup):
        for h in g:
            r = rowof[h]
            for a, p in zip(top[r], pt[r]): acct.setdefault(int(a), []).append((c, float(p), bool(pt[r, 0] > .8 and a == top[r, 0])))
    ro = np.zeros((len(questions), C)); rh = np.zeros((len(questions), C))
    for qi, gs in enumerate(questions):
        for g in gs:
            r = rowof[g]; conf = pt[r, 0] > .8
            for k, (a, p) in enumerate(zip(top[r], pt[r])):
                for c, ph, hc in acct.get(int(a), []):
                    ro[qi, c] += p * ph
                    if k == 0 and conf and hc: rh[qi, c] += 1
    return ro, rh


def main(mode, odir, tag, mdirs):
    import suppsplit_eval_20261115_v2 as s2
    ss = s2.ss; games, pools, keys, pos = ss.official(); oid = np.load(ss.old.RUN / 'official_results/indices.npy'); rowof = np.zeros(len(keys), int); rowof[oid] = np.arange(len(oid))
    top, pt = posteriors(odir, mdirs); out = RUN / tag; out.mkdir(parents=True, exist_ok=True); np.savez(out / 'posteriors_topk.npz', top=top, pt=pt)
    if mode == 'val':
        for pool, r, ep, f in ss.each():
            ro, rh = scores(top, pt, rowof, ep['supports'], [z['indices'] for z in ep['questions']]); np.savez(ss.RUN / 'extra' / f'tokopp_{tag}_{pool}_d{r}.npz', **{f'ro_{tag}': ro, f'rh_{tag}': rh})
            tg = [z['target'] for z in ep['questions']]; print(pool, r, 'questions with a confident hit on the target', round(float(np.mean([rh[i, t] > 0 for i, t in enumerate(tg)])), 4), 'any confident hit', round(float((rh.sum(1) > 0).mean()), 4), flush=True)
    else:
        for P in pools:
            sup = [[pos[k] for k in P.supports[c]] for c in P.candidate_ids]; qs = [[pos[k] for k in z['keys']] for z in P.questions]; ro, rh = scores(top, pt, rowof, sup, qs); np.savez(out / f'official_{P.name}.npz', ro=ro, rh=rh)
            print(P.name, 'questions with any confident hit', round(float((rh.sum(1) > 0).mean()), 4), flush=True)
    json.dump({'mode': mode, 'opp_dir': str(odir), 'models': [str(m) for m in mdirs]}, open(out / f'RUN_{mode}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
