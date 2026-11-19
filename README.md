# Maple

## Sliding-window Wasserstein verification

From the repository root (Python 3.10+):

```powershell
python -m pip install -r requirements.txt
python src/visualize.py --symbol XAUUSD --start 2026-11-12 --end 2026-11-13 --show
```

The default data root is `MarketData/axiory` in this repository. Override with
`--data-root C:/develop/MarketData/axiory` for external data.
Dates without an offset are JST; end is exclusive. Only M1 candles fully
closed by end are included. The chart labels candles at their close time,
one minute after the CSV's candle-open timestamp.

`src/libs/wasserstein_state.py` pools 30 M1 bars' OHLC into 120 equally weighted
price observations. Each minute it compares the window with the window ending
30 minutes earlier, using exact one-dimensional Wasserstein-1 distance.
It does not split a window into halves, center prices or use future bars.
The first comparison requires 60 consecutive M1 bars. Timestamp gaps and
nonfinite OHLC values invalidate affected comparisons. Historical warmup is
loaded before the requested chart start.

The self-contained HTML contains linked M1 candlesticks, OHLC regression slope (%/hour), and W1 as a
percentage of the current window mean. No threshold is selected by default.
`--threshold 0.2` optionally shades stable-distribution candidates in yellow
(W1 <= 0.2 price units). This is an illustrative threshold, not a calibrated
range classifier. Large oscillations can also give small distances, and
the default adjacent windows do not overlap. OHLC samples are correlated observations, not
120 independent trades or a time/volume-weighted price distribution.

Parameters: `--window-minutes 30`, `--slide-minutes 30` (comparison lag; results
still calculated every minute), `--output outputs/custom.html`, `--show`.
There are no entry/exit signals or H4/H1 filters in this verification step.

Run focused checks with `python -m unittest discover -s tests -v`.

## Percentile classes and hidden Markov forecasting

```powershell
python -m pip install -r requirements.txt
python src/train_hmm.py --start 2025-11-12 --end 2026-11-12
```

This uses one calendar year of XAUUSD M1 data, with 30-minute windows and a
30-minute comparison lag, evaluated every minute. It predicts the future
**feature classes**, not a price, return or trade entry. Each feature (W1 %,
signed slope %/hour) is split at training P20/P40/P60/P80 by default. Class 0
is lowest and class 4 highest; slope classes are empirical quantiles and do
not necessarily have zero as a boundary. Combine the pair into a categorical
symbol `w1_class * 5 + slope_class`, giving 25 observable symbols.

`libs/hmm_state.py` fits a `hmmlearn.hmm.CategoricalHMM` with four hidden states
and three initializations, selecting by training likelihood only. The hidden
state numbers are arbitrary; they are distinct from the 25 observed classes.
Forecasts are probabilities, not known future states. Inference uses only
forward filtering, `P(z_t | x_1..x_t)`, then `filtered @ A**h @ B` for the
future observation distribution. It never uses smoothed posteriors or a
full-series Viterbi path as a prediction.

The first 80% of the requested calendar interval is training; the remaining
20% is held out chronologically. Both percentile edges and HMM parameters
are fitted exclusively on training data. An additional window+lag embargo
at the test start prevents raw-window overlap with training. Market closures,
missing bars and invalid features split sequences, so no transition or
forecast target crosses a gap. Missing monthly files abort the run. Check
the reported first/last candle and sequence counts for finer coverage gaps.

Outputs under `outputs/hmm`:

- `report.json`: chronological test accuracy, log loss, multiclass Brier score
  and accuracy on cases where the target class actually changed, for 1/5/15/30
  minutes ahead. Baselines are current-class persistence (accuracy), empirical
  observable-class Markov transitions, and training class frequencies.
- `test_predictions_*m.csv`: origin/target JST close times, actual and predicted
  class pairs and all symbol probabilities. Each row's prediction uses only
  observations through its origin time.
- `model.json`: percentile boundaries, initial/transition/emission probabilities
  and feature settings in JSON (no executable pickle).
- `latest_forecast.json`: forecast at the last historical valid observation;
  the timestamp is explicit and is not live market data.

Parameters include `--bins 5`, `--states 4`, `--train-end 2026-08-31`,
`--iterations 100`, `--restarts 3`, `--horizons 1 5 15 30`, and `--output`.
Test results must not be used repeatedly to tune the model and still be
called out of sample. Overlapping feature windows make persistence strong;
high next-minute accuracy by itself is not evidence of predictive advantage.
No profitability claim is made without a separately defined trading test.

The slope uses `libs/technical.py:linear_regression()` on all OHLC values
within the current window. The four prices of each bar have the same minute
coordinate. Normalize the fitted price/minute slope as
`slope_pct_per_hour = slope * 60 / mean(window_OHLC) * 100`. Positive values
indicate rising prices, negative values falling prices. This is a linear rate,
not a compounded return. It requires only the current complete window,
independently of the previous window used by W1. No raw-price W1 panel is shown.
