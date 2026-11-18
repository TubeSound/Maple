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

`src/libs/wasserstein_state.py` pools 15 M1 bars' OHLC into 60 equally weighted
price observations. Each minute it compares the window with the window ending
one minute earlier, using exact one-dimensional Wasserstein-1 distance.
It does not split a window into halves, center prices or use future bars.
The first comparison requires 16 consecutive M1 bars. Timestamp gaps and
nonfinite OHLC values invalidate affected comparisons. Historical warmup is
loaded before the requested chart start.

The self-contained HTML contains linked M1 candlesticks, raw W1 and W1 as a
percentage of the current window mean. No threshold is selected by default.
`--threshold 0.2` optionally shades stable-distribution candidates in yellow
(W1 <= 0.2 price units). This is an illustrative threshold, not a calibrated
range classifier. Large oscillations can also give small distances, and
adjacent windows share 14 bars. OHLC samples are correlated observations, not
60 independent trades or a time/volume-weighted price distribution.

Parameters: `--window-minutes 15`, `--slide-minutes 1` (comparison lag; results
still calculated every minute), `--output outputs/custom.html`, `--show`.
There are no entry/exit signals or H4/H1 filters in this verification step.

Run focused checks with `python -m unittest discover -s tests -v`.
