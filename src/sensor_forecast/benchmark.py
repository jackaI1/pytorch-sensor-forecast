"""Refit weekly, predict daily using the observations available at each origin."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import validate_hourly
from .metrics import bootstrap_mae
from .models import LSTMForecaster, fit_boosting
from .splits import rolling_folds, seasonal_naive, training_windows, windows


@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 6, 24)
    lookback: int = 48
    train_hours: int = 2160
    test_hours: int = 168
    folds: int = 8
    origin_step: int = 24
    epochs: int = 12
    seed: int = 42
    resamples: int = 2000
    block_size: int = 7


def run_benchmark(frame: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    quality = validate_hourly(frame)
    if (
        not config.horizons
        or tuple(sorted(set(config.horizons))) != config.horizons
        or min(config.horizons) < 1
    ):
        raise ValueError("Horizons must be unique, increasing positive integers")
    if min(config.origin_step, config.epochs, config.resamples, config.block_size) < 1:
        raise ValueError("Stride, epochs, resamples and block size must be positive")
    if config.lookback < 24:
        raise ValueError("Need at least 24 hours of history for the seasonal baseline")
    # Equal spacing makes a block of adjacent origins a consistent duration.
    if config.test_hours % config.origin_step:
        raise ValueError("Test hours must be divisible by the origin stride")
    if config.origin_step != max(config.horizons):
        raise ValueError("Origin stride must equal the maximum horizon for contiguous test blocks")
    values = frame.value.to_numpy(dtype=float)
    folds = rolling_folds(
        len(values),
        config.folds,
        config.test_hours,
        config.train_hours,
        config.lookback,
        config.horizons,
    )
    n_origins = config.folds * (config.test_hours // config.origin_step)
    if config.block_size > n_origins:
        raise ValueError("Bootstrap block size exceeds forecast origin count")
    rows, audit = [], []
    with threadpool_limits(limits=1):
        for fold_id, fold in enumerate(folds):
            train_x, train_y = training_windows(values, fold, config.lookback, config.horizons)
            origins = np.arange(
                fold.cutoff, fold.test_end - max(config.horizons) + 1, config.origin_step
            )
            test_x, actual = windows(values, origins, config.lookback, config.horizons)
            boosting = fit_boosting(train_x, train_y, config.seed)
            lstm = LSTMForecaster(config.seed, config.epochs).fit(
                train_x, train_y, values[fold.train_start : fold.cutoff]
            )
            predictions = {
                "seasonal_naive": seasonal_naive(values, origins, config.horizons),
                "gradient_boosting": boosting.predict(test_x),
                "lstm": lstm.predict(test_x),
            }
            audit.append(
                {
                    "fold": fold_id,
                    "train_start": str(frame.timestamp.iloc[fold.train_start]),
                    "fit_cutoff": str(frame.timestamp.iloc[fold.cutoff]),
                    "last_training_target": str(frame.timestamp.iloc[fold.cutoff - 1]),
                    "test_end_exclusive": str(
                        frame.timestamp.iloc[fold.test_end - 1] + pd.Timedelta(hours=1)
                    ),
                    "training_examples": len(train_x),
                    "forecast_origins": len(origins),
                    "scaler_mean": float(lstm.scaler.mean_[0]),
                    "lstm_training_mse_by_epoch": lstm.losses,
                }
            )
            for model, predicted in predictions.items():
                for i, origin in enumerate(origins):
                    for j, horizon in enumerate(config.horizons):
                        rows.append(
                            {
                                "fold": fold_id,
                                "model": model,
                                "origin": str(frame.timestamp.iloc[origin]),
                                "horizon": horizon,
                                "target_time": str(frame.timestamp.iloc[origin + horizon - 1]),
                                "actual": float(actual[i, j]),
                                "predicted": float(predicted[i, j]),
                            }
                        )
    forecasts = pd.DataFrame(rows)
    summaries = []
    for model in predictions:
        selected = forecasts.loc[forecasts.model == model]
        actual = selected.pivot(index="origin", columns="horizon", values="actual").to_numpy()
        predicted = selected.pivot(index="origin", columns="horizon", values="predicted").to_numpy()
        point, low, high = bootstrap_mae(
            actual, predicted, config.seed, config.resamples, config.block_size
        )
        for i, horizon in enumerate(config.horizons):
            summaries.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "mae": point[i],
                    "ci_low": low[i],
                    "ci_high": high[i],
                    "origins": len(actual),
                }
            )
    metadata = {
        "config": asdict(config),
        "data_quality": quality,
        "folds": audit,
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "torch")},
        "python": platform.python_version(),
        "device": "cpu",
        "input_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "interval_method": "95% percentile circular moving-block bootstrap over ordered origins",
        "interval_warning": "Descriptive uncertainty on this test period, not future forecast bounds",
    }
    return forecasts, pd.DataFrame(summaries), metadata


def save_results(output: Path, forecasts: pd.DataFrame, metrics: pd.DataFrame, metadata: dict):
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / "predictions.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "report.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Forecast errors",
        "",
        "Lower MAE (mean absolute error) is better.",
        "",
        "| Model | Hours ahead | MAE | 95% bootstrap interval | Origins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples():
        lines.append(
            f"| {row.model} | {row.horizon} | {row.mae:.3f} | "
            f"{row.ci_low:.3f} to {row.ci_high:.3f} | {row.origins} |"
        )
    lines += ["", metadata["interval_warning"], ""]
    (output / "summary.md").write_text("\n".join(lines))
