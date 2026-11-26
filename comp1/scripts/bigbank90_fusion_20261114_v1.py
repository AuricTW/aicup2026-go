"""Fusion-level gate for the bigbank90 PLDA models (screen -> single frozen winner -> confirmation).

Baseline = complete90_scale saved dev 'baseline' (= production-equivalent: prior + 0.4/0.7 * (.5 old64 + .5 old90)).
Candidate = scale.blend(base, old90, big90_mix, alpha, route): replace alpha of the old90 rank ensemble with big90.
Gate = cap.compare checks (macro, Top1, CI, each routed length). Screen picks one winner; confirmation read once.
"""
from pathlib import Path
import sys, argparse
sys.path.insert(0, str(Path(__file__).resolve().parent))
import complete90_scale_20261112_v1 as s
import bigbank90_20261114_v1 as bb
import numpy as np
from threadpoolctl import threadpool_limits
p = s.p; old = s.old; cap = s.cap
RUN = bb.RUN / 'fusion'
MIXES = {'big256_768': (256, 768), 'big768_1024': (768, 1024)}
ALPHAS = (.5, 1.)
ROUTES = ('Q1', 'all')


def rrf(*xs): return sum((1 / len(xs)) / (60 + p.b.ranks(x)) for x in xs)


def load(part):
    ep = p.b.read(s.RUN / f'EPISODE_{part}.json'); d = s.RUN / part
    base = np.load(d / 'baseline.npy'); old90 = np.load(d / 'old90.npy'); div = np.load(d / 'diverse1024_v90.npy')
    comps = np.load(bb.RUN / f'components_{part}.npz')
    mixes = {k: rrf(*[comps[f'big_raw{dim}_v90'] for dim in dims]) for k, dims in MIXES.items()}
    return ep, base, old90, div, mixes


def evaluate(part, keys=None):
    ep, base, old90, div, mixes = load(part); out = {}
    for mix in MIXES:
        for alpha in ALPHAS:
            for route in ROUTES:
                key = f'{mix}_a{alpha:g}_{route}'
                if keys is not None and key not in keys: continue
                value = s.blend(base, old90, mixes[mix], ep, alpha, route); r = cap.compare(value, base, ep, route)
                paired = cap.compare(s.blend(base, old90, mixes[mix], ep, alpha, route), s.blend(base, old90, div, ep, alpha, route), ep, route)
                r['passed'] = all(r['checks'].values()); r.update(mix=mix, alpha=alpha, route=route, vs_diverse1024_same_alpha={k: paired[k] for k in ('delta', 'CI95', 'by_games_delta')})
                out[key] = r
                print(part, key, round(r['baseline_macro_all'], 4), '->', round(r['macro_score_all'], 4), 'd', round(r['delta'], 4), [round(x, 4) for x in r['CI95']],
                      {a: round(b, 4) for a, b in r['by_games_delta'].items()}, 'Top1', r['Top1'], '/', r['baseline_Top1'], 'pass', r['passed'], '| vs div', round(paired['delta'], 4))
    return out


def screen():
    RUN.mkdir(exist_ok=False); res = evaluate('screen'); passed = [k for k, v in res.items() if v['passed']]
    winner = sorted(passed, key=lambda k: (-res[k]['macro_score_all'], 1 if res[k]['route'] == 'Q1' else 4, res[k]['alpha']))[0] if passed else None
    p.b.write(RUN / 'SCREEN_RESULTS.json', res); p.b.write(RUN / 'WINNER_FREEZE.json', {'winner': winner, 'selected': res[winner] if winner else None, 'passing': passed, 'confirmation_quality_unread': True, 'source_SHA256': p.b.sha(__file__)})
    print('WINNER', winner, 'passing', passed)


def confirm():
    fz = p.b.read(RUN / 'WINNER_FREEZE.json'); assert fz['winner'] and fz['source_SHA256'] == p.b.sha(__file__); assert not (RUN / 'CONFIRMATION_RESULT.json').exists()
    r = evaluate('confirmation', {fz['winner']})[fz['winner']]; r['winner'] = fz['winner']; p.b.write(RUN / 'CONFIRMATION_RESULT.json', r); print('CONFIRMATION passed', r['passed'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['screen', 'confirm']); v = ap.parse_args()
    with threadpool_limits(limits=4): {'screen': screen, 'confirm': confirm}[v.stage]()
