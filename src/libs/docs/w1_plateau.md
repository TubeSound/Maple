# W1 decline followed by a low plateau

`libs/w1_plateau.py` detects W1's decline-to-flat shape. It is a preparation
signal, not a price bottom, a price breakout, or an entry signal.

```python
from libs.wasserstein_state import WassersteinState
from libs.w1_plateau import W1PlateauDetector, W1PlateauParams

features = WassersteinState(window_minutes=30, slide_minutes=30).calc(candles)
detector = W1PlateauDetector(W1PlateauParams(
    low_threshold=0.05,
    min_drop=0.05,
    lookback_bars=30,
    flat_bars=5,
    confirm_bars=3,
    max_flat_range=0.01,
    max_flat_slope=0.001,
    min_decline_slope=0.001,
))
result = detector.calc(features["distance_pct"], candles["jst"])
# result["signal"][i] == 1: newly confirmed, known at candle i CLOSE.
# result["plateau"][i] == 1: confirmed low/flat state continues.
```

For streaming use `detector.update(w1_pct, timestamp)` for each newly closed
M1 candle, once per candle. The timestamp may be either the open or close
label, consistently; a signal becomes available only after that candle closes.
`calc()` resets the detector, then produces exactly the same sequence as
`update()`. Omitting timestamps assumes evenly spaced samples; passing them
also detects missing bars. NaN/Inf or a timestamp gap resets detection.
Duplicate or decreasing timestamps and negative W1 values are rejected.

## Conditions and units

| Parameter | Default | Meaning |
|---|---:|---|
| `low_threshold` | 0.05 | Every W1 value in the flat window must be <= 0.05% |
| `lookback_bars` | 30 | Preceding peak-search sample count, excluding flat window |
| `min_drop` | 0.05 | Prior peak minus flat-window mean, in percentage points |
| `flat_bars` | 5 | Rolling window used to measure flatness |
| `max_flat_range` | 0.01 | Maximum W1 range in that window, in percentage points |
| `max_flat_slope` | 0.001 | Maximum absolute W1 regression slope, percentage points/minute |
| `min_decline_slope` | 0.001 | Required downward regression magnitude from prior peak to flat-window start, percentage points/minute |
| `confirm_bars` | 3 | Consecutive qualifying rolling windows before the first signal |
| `sample_minutes` | 1 | Sample spacing and time unit for regression |

These are raw W1-percent thresholds, not percentiles and not relative percent
changes in W1. For example, going from W1=0.10% to 0.03% is a drop of 0.07
percentage points. The W1 regression slope here is distinct from the existing
OHLC price `slope_pct_per_hour` feature. Both regressions use
`libs/technical.py:linear_regression()`.

There is no rebound requirement. Five flat samples followed by three
consecutive qualifying rolling windows require seven samples in total, not
fifteen. The first and last of those seven M1 samples are six minutes apart.
The detection index is the confirmation index; the code never backdates it
to the beginning of the flat window or a local minimum.

Once confirmed, `plateau` stays active while the low/range/slope conditions
continue; the historical peak is no longer required to stay in the lookback.
After the plateau ends, a new signal requires a new peak after the last active
sample, so an old large drop cannot trigger repeated preparation signals.

Diagnostics are `flat_slope`, `flat_range`, and `drop`; unavailable diagnostics
are NaN. `signal` and `plateau` are zero until confirmation. Unknown/missing
samples reset both to zero (not active).

The defaults are uncalibrated examples. On the repository's XAUUSD M1 data
for 2026-11-12, with 30/30-minute W1 and no preceding-day warmup, they confirm
at 04:48 and 12:38 JST candle close. These counts only check operation and do
not establish detection quality or trading profitability.

```powershell
python -m unittest discover -s tests -p test_w1_plateau.py -v
```
