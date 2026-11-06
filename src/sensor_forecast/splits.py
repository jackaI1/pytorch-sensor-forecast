"""Integer index convention: origin is the FIRST unobserved hour."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    train_start: int
    cutoff: int
    test_end: int


def rolling_folds(
    n: int, folds: int, test_hours: int, train_hours: int, lookback: int, horizons: tuple[int, ...]
) -> list[Fold]:
    if min(folds, test_hours, train_hours, lookback, *horizons) < 1:
        raise ValueError("Split sizes and horizons must be positive")
    if test_hours < max(horizons) or train_hours < lookback + max(horizons):
        raise ValueError("Train/test windows too short for lookback and horizons")
    first = n - folds * test_hours
    if first < train_hours:
        raise ValueError("Not enough rows for requested training window and folds")
    return [Fold(c - train_hours, c, c + test_hours) for c in range(first, n, test_hours)]


def windows(
    values: np.ndarray, origins: np.ndarray, lookback: int, horizons: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """X ends at origin-1; y[h] is the observation at origin+h-1."""
    if len(origins) == 0 or lookback < 1 or min(horizons) < 1:
        raise ValueError("Need valid origins, lookback and positive horizons")
    if origins.min() < lookback or origins.max() + max(horizons) > len(values):
        raise ValueError("Window extends outside available observations")
    x = values[origins[:, None] + np.arange(-lookback, 0)]
    y = values[origins[:, None] + np.asarray(horizons) - 1]
    return x, y


def training_windows(
    values: np.ndarray, fold: Fold, lookback: int, horizons: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    # Purge every origin whose latest target would cross the fit cutoff.
    origins = np.arange(fold.train_start + lookback, fold.cutoff - max(horizons) + 1)
    return windows(values, origins, lookback, horizons)


def seasonal_naive(
    values: np.ndarray, origins: np.ndarray, horizons: tuple[int, ...], period: int = 24
) -> np.ndarray:
    if period < 1 or min(horizons) < 1 or origins.min() < period:
        raise ValueError("Need a complete past season and positive horizons")
    # Repeat the last observed day, including when horizons exceed one day.
    indices = origins[:, None] - period + (np.asarray(horizons) - 1) % period
    return values[indices]
