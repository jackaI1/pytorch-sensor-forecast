import numpy as np
import pytest

from sensor_forecast.metrics import bootstrap_mae, mae


def test_mae_averages_origins_separately_for_each_horizon():
    np.testing.assert_array_equal(
        mae(np.array([[1, 4], [5, 8]]), np.array([[2, 1], [2, 3]])), [2, 4]
    )


def test_constant_errors_have_exact_bootstrap_intervals():
    actual, predicted = np.zeros((12, 3)), np.tile([1, 2, 3], (12, 1))
    for result in bootstrap_mae(actual, predicted, resamples=100, block_size=3):
        np.testing.assert_array_equal(result, [1, 2, 3])


def test_bootstrap_seed_and_block_resampling_match_hand_calculation():
    actual, predicted = np.zeros((8, 1)), np.arange(8).reshape(-1, 1)
    seed, resamples = 17, 50
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(resamples):
        starts = rng.integers(0, 8, size=2)
        errors = [int((start + offset) % 8) for start in starts for offset in range(4)]
        samples.append(np.mean(errors))
    result = bootstrap_mae(actual, predicted, seed, resamples, block_size=4)
    assert result[0][0] == 3.5
    np.testing.assert_allclose([result[1][0], result[2][0]], np.quantile(samples, [0.025, 0.975]))
    for a, b in zip(result, bootstrap_mae(actual, predicted, seed, resamples, 4)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "actual,predicted",
    [
        ([], []),
        ([[1]], [[1, 2]]),
        ([1], [2]),
        ([[np.nan]], [[1]]),
        ([[1]], [[np.inf]]),
    ],
)
def test_invalid_metrics_rejected(actual, predicted):
    with pytest.raises(ValueError):
        mae(np.asarray(actual), np.asarray(predicted))


@pytest.mark.parametrize("resamples,block_size", [(0, 1), (20, 0), (20, 5)])
def test_invalid_bootstrap_rejected(resamples, block_size):
    with pytest.raises(ValueError):
        bootstrap_mae(
            np.zeros((4, 2)), np.zeros((4, 2)), resamples=resamples, block_size=block_size
        )
