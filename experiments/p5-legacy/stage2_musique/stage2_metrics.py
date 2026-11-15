import numpy as np


def bootstrap_indices(n_questions, draws=1000, seed=0):
    if n_questions <= 0 or draws != 1000 or seed != 0:
        raise ValueError('protocol requires positive n, exactly 1000 draws, NumPy seed 0')
    return np.random.default_rng(seed).integers(0, n_questions, size=(draws, n_questions))


def mean_ci(values, indices):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or indices.ndim != 2 or indices.shape[1] != len(values):
        raise ValueError('bootstrap inputs have incompatible shapes')
    means = values[indices].mean(axis=1)
    return {'mean': float(values.mean()), 'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def paired_difference_ci(values_a, values_b, indices):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.shape != values_b.shape:
        raise ValueError('paired vectors must have equal shape')
    delta = values_a - values_b
    result = mean_ci(delta, indices)
    return {'mean_difference': result['mean'], 'ci95': result['ci95']}
