import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly():
    n = 240
    t = np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2014-01-01", periods=n, freq="h"),
            "value": 5 + 3 * np.sin(t * np.pi / 12) + 0.02 * t,
        }
    )
