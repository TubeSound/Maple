# SPDX-License-Identifier: MPL-2.0
"""NumPy port of RallySeason with a fast-MA regression slope filter."""

from typing import TypedDict

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int8]
BoolArray = npt.NDArray[np.bool_]


class CandleDict(TypedDict):
    close: FloatArray


class RallySeasonResult(TypedDict):
    ma_quick: FloatArray
    ma_fast: FloatArray
    ma_middle: FloatArray
    ma_slow: FloatArray
    delta_fast: FloatArray
    delta_middle: FloatArray
    delta_slow: FloatArray
    gap_fast_middle: FloatArray
    gap_middle_slow: FloatArray
    gap_ratio: FloatArray
    ma_quick_slope_pct_per_hour: FloatArray
    ma_fast_slope_pct_per_hour: FloatArray
    ma_middle_slope_pct_per_hour: FloatArray
    ma_slow_slope_pct_per_hour: FloatArray
    balanced_gap: BoolArray
    positive_without_slope: BoolArray
    negative_without_slope: BoolArray
    slope_filter_pass: BoolArray
    positive: BoolArray
    negative: BoolArray
    signal: IntArray
    background_signal: IntArray


def _sma(array: FloatArray, period: int) -> FloatArray:
    """Calculate ``ta.sma`` for finite input values."""
    out = np.full(len(array), np.nan, dtype=np.float64)
    if len(array) < period:
        return out

    cumulative = np.cumsum(array, dtype=np.float64)
    out[period - 1] = cumulative[period - 1] / period
    out[period:] = (cumulative[period:] - cumulative[:-period]) / period
    return out


def _change(array: FloatArray, length: int) -> FloatArray:
    """Calculate Pine's ``value - value[length]`` expression."""
    out = np.full(len(array), np.nan, dtype=np.float64)
    if len(array) > length:
        out[length:] = array[length:] - array[:-length]
    return out


def _linear_regression_slope_pct_per_hour(
    array: FloatArray,
    window: int,
    timeframe_minutes: float,
) -> FloatArray:
    """Calculate rolling least-squares slopes normalized to ``%/hour``."""
    out = np.full(len(array), np.nan, dtype=np.float64)
    if len(array) < window:
        return out

    windows = np.lib.stride_tricks.sliding_window_view(array, window)
    elapsed_hours = (
        np.arange(window, dtype=np.float64) * timeframe_minutes / 60.0
    )
    centered_time = elapsed_hours - np.mean(elapsed_hours)
    time_variance = np.sum(centered_time * centered_time)

    valid = np.isfinite(windows).all(axis=1)
    price_slope = np.full(len(windows), np.nan, dtype=np.float64)
    price_slope[valid] = (windows[valid] @ centered_time) / time_variance

    mean_price = np.mean(windows, axis=1)
    valid_normalizer = valid & np.isfinite(mean_price) & (mean_price != 0.0)
    normalized = np.full(len(windows), np.nan, dtype=np.float64)
    np.divide(
        price_slope * 100.0,
        np.abs(mean_price),
        out=normalized,
        where=valid_normalizer,
    )
    out[window - 1:] = normalized
    return out


def rallyseason(
    candles: CandleDict,
    quick_window: int = 12,
    fast_window: int = 24,
    middle_window: int = 48,
    slow_window: int = 300,
    rate: float = 2.0,
    delta_length: int = 2,
    threshold: float = 7.0,
    signal_on: bool = True,
    slope_window: int = 5,
    slope_threshold: float = 0.0,
    timeframe_minutes: float = 15.0,
) -> RallySeasonResult:
    """Calculate the Pine conditions and apply a fast-MA slope filter.

    ``positive_without_slope`` and ``negative_without_slope`` preserve the
    original Pine conditions. ``signal`` contains the slope-filtered state
    (1=rise, -1=drop, 0=none).
    Pine's ``signalOn`` input only controls ``bgcolor``; therefore
    ``background_signal`` is suppressed when ``signal_on`` is false while the
    raw calculations remain available.
    """
    if "close" not in candles:
        raise KeyError("candles must contain 'close'")

    close = np.asarray(candles["close"], dtype=np.float64)
    if close.ndim != 1:
        raise ValueError("close must be a one-dimensional array")
    if not np.isfinite(close).all():
        raise ValueError("close must contain only finite values")

    windows = {
        "quick_window": quick_window,
        "fast_window": fast_window,
        "middle_window": middle_window,
        "slow_window": slow_window,
    }
    for name, window in windows.items():
        if window <= 0:
            raise ValueError(f"{name} must be at least 1")
    if rate < 0.01:
        raise ValueError("rate must be at least 0.01")
    if delta_length <= 0:
        raise ValueError("delta_length must be at least 1")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if slope_window < 2:
        raise ValueError("slope_window must be at least 2")
    if slope_threshold < 0:
        raise ValueError("slope_threshold must be non-negative")
    if timeframe_minutes <= 0:
        raise ValueError("timeframe_minutes must be positive")

    # Pine: PRICE = close
    ma_quick = _sma(close, quick_window)
    ma_fast = _sma(close, fast_window)
    ma_middle = _sma(close, middle_window)
    ma_slow = _sma(close, slow_window)

    # Pine: DELTA2/3/4 = MA - MA[length]
    delta_fast = _change(ma_fast, delta_length)
    delta_middle = _change(ma_middle, delta_length)
    delta_slow = _change(ma_slow, delta_length)

    # Pine: w1 = MA2 - MA3; w2 = MA3 - MA4; r = abs(w1 / w2)
    gap_fast_middle = ma_fast - ma_middle
    gap_middle_slow = ma_middle - ma_slow
    gap_ratio = np.full(len(close), np.nan, dtype=np.float64)
    valid_ratio = (
        np.isfinite(gap_fast_middle)
        & np.isfinite(gap_middle_slow)
        & (gap_middle_slow != 0.0)
    )
    np.divide(
        gap_fast_middle,
        gap_middle_slow,
        out=gap_ratio,
        where=valid_ratio,
    )
    gap_ratio = np.abs(gap_ratio)

    # Pine uses strict bounds and strict delta threshold comparisons.
    balanced_gap = (gap_ratio > 1.0 - rate) & (gap_ratio < 1.0 + rate)
    positive_without_slope = (
        (gap_fast_middle > 0.0)
        & (gap_middle_slow > 0.0)
        & (delta_fast > threshold)
        & (delta_middle > threshold)
        & (delta_slow > threshold)
        & balanced_gap
    )
    negative_without_slope = (
        (gap_fast_middle < 0.0)
        & (gap_middle_slow < 0.0)
        & (delta_fast < -threshold)
        & (delta_middle < -threshold)
        & (delta_slow < -threshold)
        & balanced_gap
    )

    ma_quick_slope_pct_per_hour = _linear_regression_slope_pct_per_hour(
        ma_quick,
        slope_window,
        timeframe_minutes,
    )
    ma_fast_slope_pct_per_hour = _linear_regression_slope_pct_per_hour(
        ma_fast,
        slope_window,
        timeframe_minutes,
    )
    ma_middle_slope_pct_per_hour = _linear_regression_slope_pct_per_hour(
        ma_middle,
        slope_window,
        timeframe_minutes,
    )
    ma_slow_slope_pct_per_hour = _linear_regression_slope_pct_per_hour(
        ma_slow,
        slope_window,
        timeframe_minutes,
    )
    positive_slope = ma_fast_slope_pct_per_hour > slope_threshold
    negative_slope = ma_fast_slope_pct_per_hour < -slope_threshold
    positive = positive_without_slope & positive_slope
    negative = negative_without_slope & negative_slope
    slope_filter_pass = positive_slope | negative_slope

    signal = np.zeros(len(close), dtype=np.int8)
    signal[positive] = 1
    signal[negative] = -1
    background_signal = (
        signal.copy()
        if signal_on
        else np.zeros(len(close), dtype=np.int8)
    )

    return {
        "ma_quick": ma_quick,
        "ma_fast": ma_fast,
        "ma_middle": ma_middle,
        "ma_slow": ma_slow,
        "delta_fast": delta_fast,
        "delta_middle": delta_middle,
        "delta_slow": delta_slow,
        "gap_fast_middle": gap_fast_middle,
        "gap_middle_slow": gap_middle_slow,
        "gap_ratio": gap_ratio,
        "ma_quick_slope_pct_per_hour": ma_quick_slope_pct_per_hour,
        "ma_fast_slope_pct_per_hour": ma_fast_slope_pct_per_hour,
        "ma_middle_slope_pct_per_hour": ma_middle_slope_pct_per_hour,
        "ma_slow_slope_pct_per_hour": ma_slow_slope_pct_per_hour,
        "balanced_gap": balanced_gap,
        "positive_without_slope": positive_without_slope,
        "negative_without_slope": negative_without_slope,
        "slope_filter_pass": slope_filter_pass,
        "positive": positive,
        "negative": negative,
        "signal": signal,
        "background_signal": background_signal,
    }
