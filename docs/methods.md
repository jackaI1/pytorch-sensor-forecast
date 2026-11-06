# How I compare the forecasts

## Data and units

I use the `TEMP` column, in degrees Celsius, from Song Chen (2015),
[Beijing PM2.5](https://doi.org/10.24432/C5JS49), UCI Machine Learning Repository.
The archive contains hourly meteorological measurements from Beijing Capital
International Airport alongside pollution measurements from the US Embassy.
I forecast temperature, not pollution. Missing pollution values are irrelevant
because I never use that column.

The data is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
I combine year, month, day and hour into a timestamp and rename `TEMP` to `value`.
I preserve the station's time labels without claiming they are UTC. I validate
all 43,824 source hours, then select the complete 2014 calendar year (8,760 hours).
The downloader checks the archive's SHA-256 before reading it. `report.json`
also records a hash of the normalized input CSV, package versions and configuration.

Validation rejects duplicate timestamps, non-increasing timestamps, any spacing
other than one hour, missing timestamps, and non-finite or nonnumeric observations.
It does not repair gaps, sort records or silently discard bad rows. A replacement
dataset must pass the same validation. For a station with daylight-saving changes,
prepare a regular UTC timeline before using the tool.

## A forecast's boundary

The origin is the first hour I have not observed. For an origin at midnight,
the latest available observation is 23:00 on the previous day. The 1-hour target
is midnight, the 6-hour target is 05:00 and the 24-hour target is 23:00. Each
input has the 48 observed hours immediately before the origin. No centered
averages, future weather inputs or interpolated future values enter the models.

For a training cutoff `c`, a training example is allowed only if
`origin + max(horizons) - 1 < c`. I remove examples whose targets cross that
boundary, even if their inputs are in the past. My perturbation test changes
every observation from the cutoff onward and confirms the first forecast from
every model is identical. It also checks that the LSTM scaler stays unchanged.

## Rolling-origin evaluation

I reserve the final eight consecutive weeks of 2014. Each week is a new fold,
meaning a separate training and evaluation round. I fit both learned models
from scratch on the previous 90 days (2,160 hours), then issue one forecast
every 24 hours for seven days. Next week I move the training window forward
and refit. There are 56 forecast origins, each with all three horizons.

Observations that arrive during a test week become available for later forecasts
in that week. I do not update model weights or scaling during that week. This
matches a weekly retraining, daily forecasting schedule. Earlier test periods
may enter later training windows once their measurements have arrived, which is
the intended behavior of a rolling-origin evaluation. Test target blocks do not
overlap. The CLI requires the origin stride to equal the largest horizon and
to divide the test window evenly, so bootstrap blocks represent equal durations.

## Three approaches with the same information

- **Seasonal naive:** repeat the corresponding hour from the last observed day.
  It is a useful check on whether training a model buys anything.
- **Gradient boosting:** scikit-learn's `HistGradientBoostingRegressor`, one
  independent estimator per horizon through `MultiOutputRegressor`. Each receives
  the same 48 past readings, flattened into columns. I use 100 iterations,
  at most 15 leaves and L2 regularization of 1. Early stopping is disabled to
  avoid a random validation split inside the training window.
- **LSTM:** a PyTorch recurrent neural network that reads those 48 values in
  order, with one layer, 16 hidden units, and a linear head returning three
  predictions directly. I standardize with the mean and standard deviation of
  the current training window only. I train for 12 fixed epochs using Adam
  (learning rate 0.003), batch size 128, squared-error loss and gradient clipping
  at 1.0. Predictions are converted back to Celsius before scoring.

The LSTM is a deliberately small comparison, not a claim that this architecture
is optimal. Both learned models are trained without test-based parameter search.
I keep the default settings fixed and report their results even when they lose.

## Errors and uncertainty

MAE is the mean of the absolute difference between a forecast and the observed
temperature. I calculate it separately for each horizon over the same 56 origins.
Every individual prediction is in `predictions.csv`, so the summary can be checked
without fitting the models again.

For each model I take 2,000 bootstrap samples of the ordered daily error vectors.
Each sample selects consecutive blocks of seven origins, wraps at the end of the
series, and truncates to the original length. Resampling whole vectors preserves
the relationship among horizons; blocks retain some day-to-day dependence.
The 2.5th and 97.5th percentiles of the sample mean errors form a descriptive 95%
interval. These are uncertainty estimates for the measured average error, not
prediction intervals for an individual future temperature. With only eight weeks,
there is little evidence about long seasonal variation; block size is a modeling
choice and intervals do not include retraining or parameter-selection uncertainty.
I do not use interval overlap as a significance test between models.

## Reproducibility and limits

Python, NumPy and PyTorch random generators are seeded. PyTorch uses deterministic
algorithms on one CPU thread; scikit-learn thread pools are also limited to one.
Same-seed training and the bootstrap have regression tests. Floating-point results
can still vary across package versions and operating systems. `uv.lock` pins the
environment, and the report records the versions that actually ran.

This is an offline benchmark on one historical station and one late-year period.
It is useful for checking model choices and time boundaries, not for operational
weather forecasting or claims about all buildings, stations, seasons or seeds.
No fitted models are persisted. To forecast at another location, first provide
an hourly CSV and rerun all three comparisons on that location's measurements.
