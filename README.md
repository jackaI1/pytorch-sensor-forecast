I built an hourly temperature forecast comparison because I wanted to know whether a trained model could beat simply reusing yesterday's readings.

I give three approaches the same past weather measurements and compare their forecasts with what actually happened 1, 6 and 24 hours later. I move the test period forward each week so that none of the models can learn from the future. The output is a table of errors, an estimate of how uncertain those averages are, and every individual forecast so I can check the numbers myself.

On 56 daily forecast times in November and December 2014, the tree model reduced the 1-hour average error from **2.66°C to 1.36°C**. Yesterday's readings still did best at 24 hours. My small neural network struggled at 6 hours, so I would choose the tree model for short forecasts in this comparison. These results describe one station and one test period.

To run it, install [uv](https://docs.astral.sh/uv/getting-started/installation/), a Python environment manager, then:

```sh
git clone https://github.com/jackaI1/pytorch-sensor-forecast.git
cd pytorch-sensor-forecast
uv sync --locked --extra dev --python 3.11
uv run sensor-forecast fetch
uv run sensor-forecast run
```

The download is public weather data, with no account needed. The run uses the CPU and saves its results in `outputs/benchmark/`. I also keep a [completed run](results/weather-2014/summary.md) in the repository, so the results are readable without installing anything.

## What I found

MAE means mean absolute error: how far the forecast was from the actual temperature on average, ignoring the direction of the mistake. Lower is better. Parentheses show a 95% bootstrap interval, estimated by repeatedly resampling groups of consecutive test days.

| Approach | 1 hour ahead, °C | 6 hours ahead, °C | 24 hours ahead, °C |
|---|---:|---:|---:|
| Repeat yesterday's hour | 2.661 (2.304 to 3.036) | 2.482 (1.786 to 3.304) | **2.357** (1.964 to 2.804) |
| Gradient boosting, a collection of small decision trees | **1.357** (1.050 to 1.823) | **2.193** (1.563 to 2.981) | 2.493 (1.770 to 3.336) |
| LSTM, a small neural network that reads a sequence | 1.437 (1.072 to 1.836) | 6.763 (5.728 to 7.784) | 2.773 (2.167 to 3.438) |

I kept the settings fixed rather than adjusting them after seeing the test scores. The LSTM's training loss fell in every fold, but that did not make it the best forecaster. Its 6-hour predictions were too warm by about 6.6°C on average. That is a useful reason to keep a simple reference forecast and inspect each horizon separately. The intervals describe uncertainty in average error on this period; they are not ranges for tomorrow's temperature, and overlapping intervals do not establish which model is better.

## What the commands produce

- `fetch` checks the source archive's checksum, validates its hourly records, and writes the 2014 temperatures to `data/weather.csv`.
- `validate data/weather.csv` checks for missing hours, duplicates, out-of-order records and invalid readings. I stop on bad data rather than silently filling or dropping it.
- `run` writes `metrics.csv` for the nine error scores and intervals, `predictions.csv` for all 504 predictions, `summary.md` for a readable table, and `report.json` for the settings, input hash, package versions and training boundaries.

The observations come from Song Chen's [Beijing PM2.5 dataset](https://doi.org/10.24432/C5JS49), published by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). I use its airport temperature measurements in Celsius, not its pollution column. I combine the calendar columns into station-time timestamps and select the complete 2014 year. The full archive has 43,824 hourly rows; my selected year has 8,760. Both pass the gap and duplicate checks. [Data attribution and methods](docs/methods.md) explain the transformations and limitations.

## Trying another hourly series

I accept a CSV with `timestamp` and `value` columns. Timestamps must increase by exactly one hour and values must be finite numbers. For example:

```csv
timestamp,value
2024-01-01 00:00:00,5.2
2024-01-01 01:00:00,5.0
```

That example shows the format only. The default comparison needs at least 3,504 consecutive hours: 90 days for training plus eight test weeks. Errors use the same units as the input values.

```sh
uv run sensor-forecast validate path/to/hourly.csv
uv run sensor-forecast run --data path/to/hourly.csv --output outputs/my-series
```

For a quick offline check on the included ten-day excerpt of real weather measurements:

```sh
uv run sensor-forecast run --data tests/fixtures/weather.csv \
  --train-hours 120 --test-hours 48 --folds 2 \
  --epochs 2 --block-size 2 --resamples 100 --output outputs/quick
```

That short run checks the pipeline; I use the full run above for the reported comparison. `uv run sensor-forecast run --help` lists the other settings.

## How I keep the comparison fair

Each forecast receives the previous 48 hourly readings. Seasonal naive repeats the corresponding hour from the previous day. I use scikit-learn's histogram gradient boosting to fit one tree ensemble per horizon. My PyTorch LSTM has 16 hidden units and predicts all three horizons together.

Rolling-origin validation means training on the past, testing on the next block of time, then moving forward and doing it again. I train from scratch on the previous 90 days for each of eight test weeks. I issue daily forecasts using only readings that have arrived by that forecast time. The model weights and the LSTM's scaling stay fixed within each week.

I exclude training examples if any of their targets reach the test period. The LSTM's mean and standard deviation come only from that fold's training observations. A test changes all future readings and checks that the first prediction from every model stays exactly the same. I use a seven-day moving-block bootstrap with 2,000 resamples to retain some day-to-day dependence in the uncertainty estimates. [The detailed methods](docs/methods.md) spell out the index boundaries, model settings and interval calculation.

## Code and checks

The package is in `src/sensor_forecast/`: `data.py` validates and downloads observations, `splits.py` builds past-only inputs, `models.py` fits the two learned models, `metrics.py` calculates error intervals, and `benchmark.py` writes the comparison and fold audit. `cli.py` connects these pieces to the commands above.

```sh
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

All **52 tests** pass locally. They cover future-data leakage, training cutoffs, deterministic seeding, actual learning, seasonal indexing, invalid inputs, hand-calculated errors, block-bootstrap sampling, report files and CLI failures. There were no existing tests before this project. GitHub Actions runs the suite on Python 3.11 and 3.12, then fits all three models through the CLI on the bundled public excerpt and uploads the resulting reports. CI does not depend on UCI being online.

I pin dependencies in `uv.lock` and record the actual versions in each report. The committed full run used Python 3.11.16, PyTorch 2.15.0 and scikit-learn 1.9.1 on CPU, with seed 42. Repeating the run in the same environment is deterministic; numerical details can differ across platforms. The code is MIT licensed; the attributed public observations retain their CC BY 4.0 license.
