import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensor_forecast.benchmark import Config, run_benchmark, save_results


def test_all_models_write_auditable_finite_results(hourly, tmp_path):
    config = Config(train_hours=120, test_hours=48, folds=2, epochs=1, resamples=50, block_size=2)
    predictions, metrics, report = run_benchmark(hourly, config)
    assert len(predictions) == 2 * 2 * 3 * 3
    assert len(metrics) == 9
    assert metrics.origins.eq(4).all()
    assert np.isfinite(predictions.predicted).all()
    for row in metrics.itertuples():
        selected = predictions.loc[
            (predictions.model == row.model) & (predictions.horizon == row.horizon)
        ]
        assert row.mae == pytest.approx(np.abs(selected.actual - selected.predicted).mean())
    save_results(tmp_path, predictions, metrics, report)
    assert json.loads((tmp_path / "report.json").read_text())["device"] == "cpu"
    assert len(pd.read_csv(tmp_path / "predictions.csv")) == len(predictions)
    assert "95% bootstrap interval" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize(
    "changes",
    [
        {"epochs": 0},
        {"horizons": (24, 1)},
        {"horizons": (1, 1)},
        {"horizons": ()},
        {"horizons": (0, 24)},
        {"origin_step": 0},
        {"lookback": 12},
        {"block_size": 100},
        {"test_hours": 49},
        {"origin_step": 12},
    ],
)
def test_bad_benchmark_configuration_fails_early(hourly, changes):
    options = {"train_hours": 120, "test_hours": 48, "folds": 2, "epochs": 1, "block_size": 2}
    options.update(changes)
    with pytest.raises(ValueError):
        run_benchmark(hourly, Config(**options))


def test_cli_validate_and_reject_gap(tmp_path):
    fixture = Path(__file__).parent / "fixtures/weather.csv"
    result = subprocess.run(
        [sys.executable, "-m", "sensor_forecast.cli", "validate", str(fixture)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["rows"] == 240
    bad = pd.read_csv(fixture).drop(2)
    path = tmp_path / "gap.csv"
    bad.to_csv(path, index=False)
    result = subprocess.run(
        [sys.executable, "-m", "sensor_forecast.cli", "validate", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Gap" in result.stderr
