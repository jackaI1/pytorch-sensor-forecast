import random

import numpy as np
import torch

from sensor_forecast.models import LSTMForecaster, fit_boosting, seed_everything
from sensor_forecast.splits import windows


def test_seed_resets_python_numpy_and_torch():
    seed_everything(12)
    before = random.random(), np.random.rand(), torch.rand(1).item()
    seed_everything(12)
    assert before == (random.random(), np.random.rand(), torch.rand(1).item())


def test_training_is_reproducible_and_weights_learn():
    values = np.sin(np.arange(200) / 10).astype(float)
    x, y = windows(values, np.arange(48, 170), 48, (1, 6, 24))
    a = LSTMForecaster(seed=9, epochs=8).fit(x, y, values[:193])
    b = LSTMForecaster(seed=9, epochs=8).fit(x, y, values[:193])
    np.testing.assert_array_equal(a.predict(x[:4]), b.predict(x[:4]))
    assert a.losses[-1] < a.losses[0]
    c = LSTMForecaster(seed=10, epochs=8).fit(x, y, values[:193])
    assert not np.array_equal(a.predict(x[:4]), c.predict(x[:4]))
    assert a.predict(x[:4]).shape == (4, 3)


def test_boosting_fits_all_horizons_deterministically():
    values = np.sin(np.arange(200) / 10)
    x, y = windows(values, np.arange(48, 170), 48, (1, 6, 24))
    a, b = fit_boosting(x, y, 42), fit_boosting(x, y, 42)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))
    assert np.mean(np.abs(a.predict(x) - y)) < np.mean(np.abs(y - y.mean(axis=0)))
