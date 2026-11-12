# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# Python port of Pivot Point SuperTrend, © LonesomeTheBlue.
"""Confirmed-bar port of pivot_supertrend.pine; no backdated trading signals."""
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from .candle_arrays import validate_candles
from .supertrend import CandleDict, FloatArray, SupertrendResult


class PivotSupertrendResult(SupertrendResult):
    pivot_high: FloatArray
    pivot_low: FloatArray
    center: FloatArray
    true_range: FloatArray
    basic_up: FloatArray
    basic_down: FloatArray
    plot_line: FloatArray
    support: FloatArray
    resistance: FloatArray
    signal_price: FloatArray
    label_price: FloatArray


class WaveMetrics(TypedDict):
    wave_elapsed_bars: FloatArray
    wave_height: FloatArray
    wave_height_pct: FloatArray
    wave_net_move: FloatArray
    period_baseline: FloatArray
    height_baseline: FloatArray
    period_ratio: FloatArray
    height_ratio: FloatArray
    pivot_interval_bars: FloatArray
    pivot_height: FloatArray
    completed_wave_count: npt.NDArray[np.int64]


@dataclass(frozen=True)
class PivotSupertrendConfig:
    pivot_period: int = 2
    atr_factor: float = 3.0
    atr_period: int = 10

    def __post_init__(self):
        for name, upper in (("pivot_period", 50), ("atr_period", None)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1 or (upper and value > upper):
                raise ValueError(f"{name} must be a positive integer" + (" <= 50" if upper else ""))
        if not np.isfinite(self.atr_factor) or self.atr_factor < 1:
            raise ValueError("atr_factor must be finite and >= 1")


def _pivot(values, i, period, is_high):
    if i < 2 * period:
        return np.nan
    center = i - period
    value = values[center]
    left, right = values[center - period:center], values[center + 1:i + 1]
    valid = (np.all(value >= left) and np.all(value > right)) if is_high else (
        np.all(value <= left) and np.all(value < right))
    return value if valid else np.nan


def calc(candles: CandleDict, config: PivotSupertrendConfig | None = None) -> PivotSupertrendResult:
    """OHLC arrays -> Pine indicator arrays at the bar where values are known.

    pivot_high/low and support/resistance are on confirmation bars. Only
    display code may shift pivot markers back by pivot_period. Signals remain
    on their confirmation bars; signal_price is close, label_price is trail.
    """
    cfg = config or PivotSupertrendConfig()
    arrays = validate_candles(candles)
    opened, high, low, close = (arrays[k] for k in ("open", "high", "low", "close"))
    n = len(close)
    names = ("pivot_high", "pivot_low", "center", "true_range", "atr", "basic_up", "basic_down",
             "up", "down", "supertrend", "plot_line", "support", "resistance", "signal_price", "label_price")
    result = {name: np.full(n, np.nan) for name in names}
    result["trend"] = np.ones(n, dtype=np.int8)
    for name in ("buy_signal", "sell_signal", "change_signal"):
        result[name] = np.zeros(n, dtype=bool)
    for i in range(n):
        tr = high[i] - low[i] if i == 0 else max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        result["true_range"][i] = tr
        if i == cfg.atr_period - 1:
            result["atr"][i] = np.mean(result["true_range"][:cfg.atr_period])
        elif i >= cfg.atr_period:
            result["atr"][i] = (result["atr"][i - 1] * (cfg.atr_period - 1) + tr) / cfg.atr_period
        ph = result["pivot_high"][i] = _pivot(high, i, cfg.pivot_period, True)
        pl = result["pivot_low"][i] = _pivot(low, i, cfg.pivot_period, False)
        previous_center = result["center"][i - 1] if i else np.nan
        lastpp = ph if np.isfinite(ph) else pl  # Pine favors ph when both confirm
        result["center"][i] = (lastpp if np.isnan(previous_center) else (previous_center * 2 + lastpp) / 3
                                ) if np.isfinite(lastpp) else previous_center
        basic_up = result["basic_up"][i] = result["center"][i] - cfg.atr_factor * result["atr"][i]
        basic_down = result["basic_down"][i] = result["center"][i] + cfg.atr_factor * result["atr"][i]
        prev_up, prev_down = (result["up"][i - 1], result["down"][i - 1]) if i else (np.nan, np.nan)
        prev_close = close[i - 1] if i else np.nan
        result["up"][i] = np.maximum(basic_up, prev_up) if prev_close > prev_up else basic_up
        result["down"][i] = np.minimum(basic_down, prev_down) if prev_close < prev_down else basic_down
        prev_trend = result["trend"][i - 1] if i else 1
        trend = 1 if close[i] > prev_down else -1 if close[i] < prev_up else prev_trend
        result["trend"][i] = trend
        result["supertrend"][i] = result["up"][i] if trend == 1 else result["down"][i]
        # Pine hides the first bar and reversal bars' line color.
        if i and trend == prev_trend:
            result["plot_line"][i] = result["supertrend"][i]
        result["buy_signal"][i] = bool(i and trend == 1 and prev_trend == -1)
        result["sell_signal"][i] = bool(i and trend == -1 and prev_trend == 1)
        result["change_signal"][i] = bool(i and trend != prev_trend)
        if result["change_signal"][i]:
            result["signal_price"][i] = close[i]
            result["label_price"][i] = result["supertrend"][i]
        result["support"][i] = pl if np.isfinite(pl) else result["support"][i - 1] if i else np.nan
        result["resistance"][i] = ph if np.isfinite(ph) else result["resistance"][i - 1] if i else np.nan
    return result


def measure_waves(candles: CandleDict, result: PivotSupertrendResult, baseline_waves: int = 10) -> WaveMetrics:
    """Causal diagnostics, separate from Pine signals (no entry rules added).

    Wave = a trend-colored segment. Elapsed bars and running high-low range
    are compared to the median of preceding completed segments, never the
    eventual length/range of the current segment. Initial partial segment is
    excluded from the baseline. Actual pivot-to-pivot metrics are separate.
    """
    if isinstance(baseline_waves, bool) or not isinstance(baseline_waves, int) or baseline_waves < 1:
        raise ValueError("baseline_waves must be a positive integer")
    candles = validate_candles(candles)
    high, low = candles["high"], candles["low"]
    n = len(high)
    for key in ("supertrend", "change_signal", "pivot_high", "pivot_low"):
        if np.asarray(result[key]).shape != (n,):
            raise ValueError(f"result[{key}] must have shape ({n},)")
    names = ("wave_elapsed_bars", "wave_height", "wave_height_pct", "wave_net_move", "period_baseline", "height_baseline",
             "period_ratio", "height_ratio", "pivot_interval_bars", "pivot_height")
    metrics = {name: np.full(n, np.nan) for name in names}
    metrics["completed_wave_count"] = np.zeros(n, dtype=np.int64)
    periods, heights = [], []
    start, start_is_signal, previous_pivot = None, False, None
    hi = lo = np.nan
    close = np.asarray(candles["close"])
    for i in range(n):
        pp = result["pivot_high"][i] if np.isfinite(result["pivot_high"][i]) else result["pivot_low"][i]
        if np.isfinite(pp):
            if previous_pivot is not None:
                metrics["pivot_interval_bars"][i] = i - previous_pivot[0]
                metrics["pivot_height"][i] = abs(pp - previous_pivot[1])
            previous_pivot = (i, pp)
        if not np.isfinite(result["supertrend"][i]):
            continue
        if start is None or result["change_signal"][i]:
            if start is not None and start_is_signal:
                periods.append(i - start)
                heights.append(hi - lo)
            start, start_is_signal = i, bool(result["change_signal"][i])
            hi, lo = high[i], low[i]
        hi, lo = max(hi, high[i]), min(lo, low[i])
        metrics["wave_elapsed_bars"][i] = i - start
        metrics["wave_height"][i] = hi - lo
        # Normalize each segment by its own starting close, not the moving
        # current price or any future extreme. This is a nonnegative range.
        metrics["wave_height_pct"][i] = (hi - lo) / close[start] * 100.0
        metrics["wave_net_move"][i] = close[i] - close[start]
        metrics["completed_wave_count"][i] = len(periods)
        if periods:
            metrics["period_baseline"][i] = np.median(periods[-baseline_waves:])
            metrics["height_baseline"][i] = np.median(heights[-baseline_waves:])
            metrics["period_ratio"][i] = (i - start) / metrics["period_baseline"][i]
            if metrics["height_baseline"][i] > 0:
                metrics["height_ratio"][i] = (hi - lo) / metrics["height_baseline"][i]
    return metrics
