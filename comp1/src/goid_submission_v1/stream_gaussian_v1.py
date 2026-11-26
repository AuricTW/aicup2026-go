"""Bounded-memory version of the v2 missing-color Gaussian moment estimator.

No labels other than training player/color, no model selection, no adaptation.
All moments use all selected rows; PCA is not fitted to a convenience subsample.
"""
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh
from goid_v4.pilot_gaussian_v2 import JointRandomEffects


def fit_stream(features, colors, players, output, dimension=128, block=4096):
    output = Path(output)
    if output.exists():
        raise ValueError('New model path required')
    n, width = features.shape
    colors = np.asarray(colors)
    players = np.asarray(players)
    if n < 3 or len(colors) != n or len(players) != n or not np.isin(colors, [0, 1]).all():
        raise ValueError('Training shape/color mismatch')
    unique, codes = np.unique(players, return_inverse=True)
    start = time.perf_counter()
    center = np.zeros(width, np.float64)
    for first in range(0, n, block):
        x = np.asarray(features[first:first + block], np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite training feature')
        center += x.sum(0)
    center /= n
    scatter = np.zeros((width, width), np.float64)
    for first in range(0, n, block):
        x = np.asarray(features[first:first + block], np.float64) - center
        scatter += x.T @ x
    values, vectors = eigh(scatter / (n - 1), subset_by_index=[width - dimension, width - 1])
    if values.min() <= 0:
        raise ValueError('Nonpositive active PCA spectrum')
    # Sign canonicalization only; no rotation/selection based on accuracy.
    sign = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(dimension)])
    basis = vectors * sign[None] / np.sqrt(values)[None]
    counts = np.zeros((len(unique), 2), np.int64)
    sums = np.zeros((len(unique), 2, dimension), np.float64)
    for first in range(0, n, block):
        stop = min(first + block, n)
        z = (np.asarray(features[first:stop], np.float64) - center) @ basis
        np.add.at(counts, (codes[first:stop], colors[first:stop]), 1)
        np.add.at(sums, (codes[first:stop], colors[first:stop]), z)
    means = sums / np.maximum(counts, 1)[:, :, None]
    # A second residual pass avoids cancellation of raw second moments.
    residual = np.zeros((2, dimension), np.float64)
    for first in range(0, n, block):
        stop = min(first + block, n)
        cc = colors[first:stop]
        z = (np.asarray(features[first:stop], np.float64) - center) @ basis
        delta = z - means[codes[first:stop], cc]
        for c in (0, 1):
            residual[c] += np.square(delta[cc == c]).sum(0)
    model = JointRandomEffects()
    model.center, model.basis = center, basis
    model.prior_mean = np.zeros((2, dimension))
    model.noise = np.zeros((2, dimension))
    diagonal = np.zeros((2, dimension))
    for c in (0, 1):
        observed = counts[:, c] > 0
        df = int(np.maximum(counts[:, c] - 1, 0).sum())
        if observed.sum() < 2 or df <= 0:
            raise ValueError('Unidentifiable color moments')
        model.prior_mean[c] = means[observed, c].mean(0)
        model.noise[c] = np.maximum(residual[c] / df, 1e-6)
        diagonal[c] = np.maximum(means[observed, c].var(0, ddof=1) -
                                (model.noise[c] / counts[observed, c, None]).mean(0), 1e-5)
    paired = (counts > 0).all(1)
    if paired.sum() >= 2:
        a = means[paired, 0] - means[paired, 0].mean(0)
        b = means[paired, 1] - means[paired, 1].mean(0)
        shared = np.maximum((a * b).sum(0) / (paired.sum() - 1), 0)
    else:
        shared = np.zeros(dimension)
    model.K = np.zeros((2, 2, dimension))
    model.K[0, 0] = np.maximum(diagonal[0], shared + 1e-5)
    model.K[1, 1] = np.maximum(diagonal[1], shared + 1e-5)
    model.K[0, 1] = model.K[1, 0] = shared
    model.prior_scale, model.noise_scale = 2., 1.
    model.save(output)
    return {'rows': n, 'players': len(unique), 'paired_players': int(paired.sum()),
            'missing_color_players': int((counts == 0).any(1).sum()), 'dimension': dimension,
            'prior_scale': 2., 'noise_scale': 1., 'grid_or_quality_selection': False,
            'all_selected_rows_used_for_PCA_and_moments': True,
            'PCA_eigenvalue_min': float(values.min()), 'PCA_eigenvalue_max': float(values.max()),
            'seconds': time.perf_counter() - start}
