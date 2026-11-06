import numpy as np
import pytest

from sensor_forecast.benchmark import Config, run_benchmark
from sensor_forecast.splits import (
    Fold,
    rolling_folds,
    seasonal_naive,
    training_windows,
    windows,
)


def test_lags_are_strictly_past_and_targets_have_correct_horizons():
    values = np.arange(200)
    x, y = windows(values, np.array([60, 90]), 48, (1, 6, 24))
    np.testing.assert_array_equal(x[0], np.arange(12, 60))
    np.testing.assert_array_equal(y, [[60, 65, 83], [90, 95, 113]])
    assert (x.max(axis=1) < np.array([60, 90])).all()


def test_training_purges_labels_at_and_after_cutoff():
    values = np.arange(300, dtype=float)
    fold = Fold(10, 200, 248)
    x, y = training_windows(values, fold, 48, (1, 6, 24))
    assert x.min() == 10
    assert y.max() == 199
    changed = values.copy()
    changed[200:] = -999
    other_x, other_y = training_windows(changed, fold, 48, (1, 6, 24))
    np.testing.assert_array_equal(x, other_x)
    np.testing.assert_array_equal(y, other_y)


def test_rolling_test_blocks_disjoint_and_chronological():
    folds = rolling_folds(500, 3, 48, 200, 48, (1, 6, 24))
    assert folds[0] == Fold(156, 356, 404)
    for left, right in zip(folds, folds[1:]):
        assert left.test_end == right.cutoff
        assert left.train_start < right.train_start
    assert folds[-1].test_end == 500


@pytest.mark.parametrize(
    "arguments",
    [
        (100, 2, 48, 100, 48, (1, 24)),
        (500, 0, 48, 200, 48, (1, 24)),
        (500, 2, 12, 200, 48, (1, 24)),
        (500, 2, 48, 60, 48, (1, 24)),
    ],
)
def test_impossible_splits_rejected(arguments):
    with pytest.raises(ValueError):
        rolling_folds(*arguments)


def test_seasonal_baseline_never_reads_future_even_beyond_one_day():
    values = np.arange(100)
    actual = seasonal_naive(values, np.array([48]), (1, 6, 24, 25, 48))
    np.testing.assert_array_equal(actual, [[24, 29, 47, 24, 47]])


@pytest.mark.parametrize("origins", [np.array([10]), np.array([99]), np.array([], dtype=int)])
def test_out_of_bounds_windows_rejected(origins):
    with pytest.raises(ValueError):
        windows(np.arange(100), origins, 48, (1, 24))


def test_future_changes_cannot_change_first_forecast_or_fit_scaler(hourly):
    config = Config(train_hours=120, test_hours=48, folds=2, epochs=1, block_size=2, resamples=20)
    original, _, audit = run_benchmark(hourly, config)
    changed = hourly.copy()
    cutoff = len(hourly) - 96
    changed.loc[cutoff:, "value"] += 10000
    perturbed, _, other_audit = run_benchmark(changed, config)
    origin = str(hourly.timestamp.iloc[cutoff])
    a = original.loc[original.origin == origin, "predicted"]
    b = perturbed.loc[perturbed.origin == origin, "predicted"]
    np.testing.assert_array_equal(a, b)
    assert audit["folds"][0]["scaler_mean"] == other_audit["folds"][0]["scaler_mean"]
    assert audit["folds"][0]["scaler_mean"] == pytest.approx(
        hourly.value.iloc[cutoff - 120 : cutoff].mean()
    )
