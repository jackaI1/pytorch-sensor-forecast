I merged this into my existing [lifeos-open project](https://github.com/jackaI1/lifeos-open/tree/main/projects/pytorch-sensor-forecast). The current code is in `projects/pytorch-sensor-forecast/` there. I kept the code here so older links still work.

The move keeps the original layout, including `text_models/serving_check.py`, `text_models/jax_lm.py`, `text_models/__init__.py`, `text_models/serving_client.py`. I replaced the README with the report below and left out Git history, installed dependencies, caches and generated build output from the hub copy.

The project checks are configured in the hub at `.github/workflows/pytorch-sensor-forecast-text-models.yml`, `.github/workflows/pytorch-sensor-forecast-ci.yml`, with commands run from this project folder and changes filtered to its paths.

---

# pytorch-sensor-forecast

I built an hourly temperature forecast comparison to learn how the underlying idea works. This is a learning project.

I use public UCI Beijing weather measurements and public Tiny Shakespeare and IMDB text; attribution is kept in `docs/methods.md` and `docs/text-models.md`.

## Try it

From this project directory:

To run it, install [uv](https://docs.astral.sh/uv/getting-started/installation/), a Python environment manager, then:

```sh
uv sync --locked --extra dev --python 3.11
uv run sensor-forecast fetch
uv run sensor-forecast run
```

The download is public weather data, with no account needed. The run uses the CPU and saves its results in `outputs/benchmark/`. I also keep a [completed run](results/weather-2014/summary.md) in the repository, so the results are readable without installing anything.


## Text experiments

I also wanted to learn how small language models train and how a model can answer requests after training. These experiments use public text rather than the weather readings, so they do not change the temperature forecasts.

The first model learns to predict the next character in Shakespeare. It divides each batch among CPU devices, combines their updates, and checks its predictions on text held back from training. Perplexity is a measure of how uncertain the model is about the next character; lower is better. The second model learns whether an IMDB movie review is positive or negative. TensorFlow Serving runs the saved model as a small web service, and a client sends batches of token numbers to it.

The first command downloads [Tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare) and saves a checkpoint under `outputs/`. The second continues from the saved weights and random state. Only load checkpoints you trust. The script prints held-out loss and perplexity; the last 10% of the text is held out in order.

For the review model, use a separate environment with `pip install -r text_models/requirements-tf.txt`, then run:

```sh
python -m text_models.tf_classifier --limit 10000 --epochs 3
docker compose -f text_models/compose.yaml up -d
python -m text_models.serving_client --batch 32
docker compose -f text_models/compose.yaml down
```

The training command downloads the public [IMDB sentiment dataset](https://ai.stanford.edu/~amaas/data/sentiment/) through Keras and prints accuracy on its separate test set. The client reports mean and 95th percentile request time after a warm-up. It sends a zero-filled batch as a repeatable timing example; real reviews must first be converted to the same IMDB vocabulary and padded to 128 tokens. Docker packages the serving process so the saved model can be called over HTTP. The service is intended for local use, with no authentication configured.

These are small learning experiments, not production language models. Their scores depend on training length and hardware; CI prints its actual held-out scores and request timing rather than treating a short smoke run as a benchmark. More details are in [the text experiment notes](docs/text-models.md).

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

## Tests and limits

I kept 8 files of test code and supporting test helpers in this copy. This is a file inventory, not an execution result. This is a source snapshot, and running it on a new machine may need environment-specific setup.


Project checks run from this directory through the path-filtered hub workflows: `pytorch-sensor-forecast-text-models.yml`, `pytorch-sensor-forecast-ci.yml`.
