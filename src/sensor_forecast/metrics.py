"""Absolute error with a circular moving-block bootstrap over forecast origins."""

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.ndim != 2 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Need nonempty matching (origins, horizons) arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite observations and predictions")
    return np.abs(actual - predicted).mean(axis=0)


def bootstrap_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    seed: int = 42,
    resamples: int = 2000,
    block_size: int = 7,
) -> tuple:
    point = mae(actual, predicted)
    errors = np.abs(actual - predicted)
    n = len(errors)
    if resamples < 1 or not 1 <= block_size <= n:
        raise ValueError("Need positive resamples and block_size between 1 and origin count")
    rng = np.random.default_rng(seed)
    means = np.empty((resamples, errors.shape[1]))
    blocks = (n + block_size - 1) // block_size
    for i in range(resamples):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + np.arange(block_size)) % n).ravel()[:n]
        means[i] = errors[indices].mean(axis=0)
    low, high = np.quantile(means, [0.025, 0.975], axis=0)
    return point, low, high
