"""Second pre-registered gate: re-weight fusion toward the (much stronger) bigbank90 neural ensemble.

Motivation (screen only): big3 = RRF(big256,big768,big1024) alone beats the production-equivalent fused
baseline by +0.064 on screen; prior alone is weak (0.34). Fixed 0.4/0.7 neural weights dilute it.
Family (8 candidates): score = (1-w_q) * prior + w_q * big3, w_Q1 in {.55,.7,.85,1}, w_Q2/3/5 in {.85,1}.
Gate = cap.compare(all) checks vs baseline. Screen picks one; confirmation read once.
Caveat: confirmation already read once by bigbank90_fusion (component-level + one fusion winner).
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bigbank90_fusion_20261114_v1 as f
import numpy as np
from threadpoolctl import threadpool_limits
s = f.s; p = f.p; cap = f.cap; bb = f.bb
RUN = bb.RUN / 'reweight'
W1 = (.55, .7, .85, 1.); WN = (.85, 1.)


def big3(comps): return f.rrf(*[comps[f'big_raw{d}_v90'] for d in (256, 768, 1024)])


def fuse(prior, neural, ql, w1, wn):
    w = np.where(np.asarray(ql) == 1, w1, wn)[:, None]; return (1 - w) * prior + w * neural


def evaluate(part, keys=None):
    ep = p.b.read(s.RUN / f'EPISODE_{part}.json'); base = np.load(s.RUN / part / 'baseline.npy'); prior = np.load(s.RUN / part / 'prior.npy')
    neural = big3(np.load(bb.RUN / f'components_{part}.npz')); ql = [z['q'] for z in ep['questions']]; out = {}
    for w1 in W1:
        for wn in WN:
            key = f'w1_{w1:g}_wn_{wn:g}'
            if keys is not None and key not in keys: continue
            r = cap.compare(fuse(prior, neural, ql, w1, wn), base, ep, 'all'); r['passed'] = all(r['checks'].values()); r.update(w1=w1, wn=wn); out[key] = r
            print(part, key, round(r['baseline_macro_all'], 4), '->', round(r['macro_score_all'], 4), 'd', round(r['delta'], 4), [round(x, 4) for x in r['CI95']],
                  {a: round(b, 4) for a, b in r['by_games_delta'].items()}, 'Top1', r['Top1'], '/', r['baseline_Top1'], 'pass', r['passed'])
    return out


def screen():
    RUN.mkdir(exist_ok=False); res = evaluate('screen'); passed = [k for k, v in res.items() if v['passed']]
    winner = sorted(passed, key=lambda k: (-res[k]['macro_score_all'], -res[k]['w1'], -res[k]['wn']))[0] if passed else None
    p.b.write(RUN / 'SCREEN_RESULTS.json', res); p.b.write(RUN / 'WINNER_FREEZE.json', {'winner': winner, 'selected': res[winner] if winner else None, 'passing': passed, 'source_SHA256': p.b.sha(__file__)})
    print('WINNER', winner, 'passing', passed)


def confirm():
    fz = p.b.read(RUN / 'WINNER_FREEZE.json'); assert fz['winner'] and fz['source_SHA256'] == p.b.sha(__file__); assert not (RUN / 'CONFIRMATION_RESULT.json').exists()
    r = evaluate('confirmation', {fz['winner']})[fz['winner']]; r['winner'] = fz['winner']; p.b.write(RUN / 'CONFIRMATION_RESULT.json', r); print('CONFIRMATION passed', r['passed'], r['checks'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['screen', 'confirm']); v = ap.parse_args()
    with threadpool_limits(limits=4): {'screen': screen, 'confirm': confirm}[v.stage]()
