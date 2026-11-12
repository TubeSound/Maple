"""Extract positive and negative MA-slope peaks using a fixed threshold."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real
from typing import TypedDict

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]
DirectionArray = npt.NDArray[np.int8]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SlopeSpikeParam:
    quick_window: int = 5
    quick_energy_window: int = 240
    fast_window: int = 18
    middle_window: int = 48
    slow_window: int = 720
    slope_window: int = 6
    threshold: float = 4.0
    dead_zone: float = 0.5
    pivot_window: int = 5
    burst_window_minutes: int = 30
    burst_min_spikes: int = 3
    max_gap_minutes: float = 5.0

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("quick_window", self.quick_window, 1),
            ("quick_energy_window", self.quick_energy_window, 1),
            ("fast_window", self.fast_window, 1),
            ("middle_window", self.middle_window, 1),
            ("slow_window", self.slow_window, 1),
            ("slope_window", self.slope_window, 2),
            ("pivot_window", self.pivot_window, 1),
            ("burst_window_minutes", self.burst_window_minutes, 1),
            ("burst_min_spikes", self.burst_min_spikes, 2),
        ):
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("threshold", "max_gap_minutes"):
            value = getattr(self, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            not isinstance(self.dead_zone, Real)
            or isinstance(self.dead_zone, bool)
            or not np.isfinite(self.dead_zone)
            or self.dead_zone < 0.0
        ):
            raise ValueError("dead_zone must be non-negative and finite")


def _sma(values: npt.ArrayLike, window: int) -> FloatArray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("values must be one-dimensional")
    out = np.full(len(source), np.nan, dtype=np.float64)
    if len(source) < window:
        return out
    cumulative = np.cumsum(source, dtype=np.float64)
    out[window - 1] = cumulative[window - 1] / window
    out[window:] = (cumulative[window:] - cumulative[:-window]) / window
    return out


def _rolling_rms(values: npt.ArrayLike, window: int) -> FloatArray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("values must be one-dimensional")
    out = np.full(len(source), np.nan, dtype=np.float64)
    if len(source) < window:
        return out

    finite = np.isfinite(source)
    squared = np.where(finite, source * source, 0.0)
    cumulative_sum = np.cumsum(squared, dtype=np.float64)
    cumulative_count = np.cumsum(finite, dtype=np.int64)
    window_sum = cumulative_sum[window - 1 :].copy()
    window_count = cumulative_count[window - 1 :].copy()
    if window > 1:
        window_sum[1:] -= cumulative_sum[:-window]
        window_count[1:] -= cumulative_count[:-window]
    valid = window_count == window
    values_out = np.full(len(window_sum), np.nan, dtype=np.float64)
    values_out[valid] = np.sqrt(window_sum[valid] / window)
    out[window - 1 :] = values_out
    return out


def regression_slope_pct_per_hour(
    values: npt.ArrayLike,
    window: int,
    timeframe_minutes: float,
) -> FloatArray:
    """Return rolling least-squares slope normalized to percent per hour."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1:
        raise ValueError("values must be one-dimensional")
    out = np.full(len(source), np.nan, dtype=np.float64)
    if (
        not isinstance(timeframe_minutes, Real)
        or isinstance(timeframe_minutes, bool)
        or not np.isfinite(timeframe_minutes)
        or timeframe_minutes <= 0.0
    ):
        raise ValueError("timeframe_minutes must be positive and finite")
    if len(source) < window:
        return out

    windows = np.lib.stride_tricks.sliding_window_view(source, window)
    x = np.arange(window, dtype=np.float64)
    centered_x = x - np.mean(x)
    denominator = np.sum(centered_x * centered_x)
    valid = np.isfinite(windows).all(axis=1)
    slope_per_bar = np.full(len(windows), np.nan, dtype=np.float64)
    slope_per_bar[valid] = (windows[valid] @ centered_x) / denominator
    mean_price = np.mean(windows, axis=1)
    normalizer_valid = valid & np.isfinite(mean_price) & (mean_price != 0.0)
    normalized = np.full(len(windows), np.nan, dtype=np.float64)
    np.divide(
        slope_per_bar * 100.0,
        (timeframe_minutes / 60.0) * np.abs(mean_price),
        out=normalized,
        where=normalizer_valid,
    )
    out[window - 1 :] = normalized
    return out


def calculate_ma_slopes(
    close: npt.ArrayLike,
    timeframe_minutes: float,
    param: SlopeSpikeParam | None = None,
) -> dict[str, FloatArray]:
    """Calculate the moving averages and slopes used by spike signals."""

    param = param or SlopeSpikeParam()
    result: dict[str, FloatArray] = {}
    for name, window in (
        ("quick", param.quick_window),
        ("fast", param.fast_window),
        ("middle", param.middle_window),
        ("slow", param.slow_window),
    ):
        moving_average = _sma(close, window)
        result[f"ma_{name}"] = moving_average
        result[f"{name}_slope"] = regression_slope_pct_per_hour(
            moving_average,
            param.slope_window,
            timeframe_minutes,
        )
    result["quick_slope_energy"] = _rolling_rms(
        result["quick_slope"],
        param.quick_energy_window,
    )
    return result


class SlopeSpikeResult(TypedDict):
    spike_metric: FloatArray
    spike_magnitude: FloatArray
    positive_threshold: FloatArray
    negative_threshold: FloatArray
    positive_candidate: BoolArray
    negative_candidate: BoolArray
    positive_peak: BoolArray
    negative_peak: BoolArray
    spike: BoolArray
    spike_direction: DirectionArray
    recent_spike_count: IntArray
    positive_recent_spike_count: IntArray
    negative_recent_spike_count: IntArray
    positive_burst: BoolArray
    negative_burst: BoolArray
    burst: BoolArray
    burst_start: BoolArray
    burst_spike: BoolArray


class SpikeTrendResult(TypedDict):
    trend_direction: DirectionArray
    trend_peak: BoolArray
    entry_pivot_peak: BoolArray
    entry_pivot_direction: DirectionArray
    neutral_peak: BoolArray


def _time_to_ns(value: object) -> int:
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("timestamps contain invalid values")
        return int(value.astype("datetime64[ns]").astype(np.int64))
    pandas_value = getattr(value, "value", None)
    if isinstance(pandas_value, Integral):
        return int(pandas_value)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return int(value.timestamp() * 1_000_000_000)
        return int((value - datetime(1970, 1, 1)).total_seconds() * 1_000_000_000)
    if isinstance(value, str):
        try:
            return _time_to_ns(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError as exc:
            raise ValueError(f"invalid timestamp: {value!r}") from exc
    raise TypeError(f"unsupported timestamp type: {type(value).__name__}")


def _as_time_ns(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    times = np.fromiter((_time_to_ns(value) for value in raw), dtype=np.int64)
    differences = np.diff(times)
    if np.any(differences < 0):
        raise ValueError("timestamps must be sorted in ascending order")
    if np.any(differences == 0):
        raise ValueError("timestamps must not contain duplicates")
    return times


def _pivot_peaks(
    times: npt.NDArray[np.int64],
    metric: FloatArray,
    positive_candidate: BoolArray,
    negative_candidate: BoolArray,
    param: SlopeSpikeParam,
) -> tuple[BoolArray, BoolArray]:
    positive_peak = np.zeros(len(metric), dtype=np.bool_)
    negative_peak = np.zeros(len(metric), dtype=np.bool_)
    window = param.pivot_window
    width = window * 2 + 1
    if len(metric) < width:
        return positive_peak, negative_peak

    max_gap_ns = int(param.max_gap_minutes * 60 * 1_000_000_000)
    rolling = np.lib.stride_tricks.sliding_window_view(metric, width)
    left = rolling[:, :window]
    right = rolling[:, window + 1 :]
    other_finite = np.isfinite(left).any(axis=1) | np.isfinite(right).any(axis=1)
    other_max = np.maximum(
        np.max(np.where(np.isfinite(left), left, -np.inf), axis=1),
        np.max(np.where(np.isfinite(right), right, -np.inf), axis=1),
    )
    other_min = np.minimum(
        np.min(np.where(np.isfinite(left), left, np.inf), axis=1),
        np.min(np.where(np.isfinite(right), right, np.inf), axis=1),
    )
    center_indices = np.arange(window, len(metric) - window)
    center_values = metric[center_indices]
    contiguous = (
        np.lib.stride_tricks.sliding_window_view(
            np.diff(times),
            width - 1,
        )
        <= max_gap_ns
    ).all(axis=1)
    valid = np.isfinite(center_values) & other_finite & contiguous
    positive_peak[center_indices] = (
        valid
        & positive_candidate[center_indices]
        & (center_values > other_max)
    )
    negative_peak[center_indices] = (
        valid
        & negative_candidate[center_indices]
        & (center_values < other_min)
    )
    return positive_peak, negative_peak


def _burst_state(
    times: npt.NDArray[np.int64],
    peaks: BoolArray,
    param: SlopeSpikeParam,
) -> tuple[IntArray, BoolArray]:
    recent_count = np.zeros(len(peaks), dtype=np.int32)
    burst = np.zeros(len(peaks), dtype=np.bool_)
    peak_times: deque[int] = deque()
    minute_ns = 60 * 1_000_000_000
    burst_window_ns = param.burst_window_minutes * minute_ns
    max_gap_ns = int(param.max_gap_minutes * minute_ns)
    previous_time: int | None = None

    for index, timestamp_value in enumerate(times):
        timestamp = int(timestamp_value)
        if previous_time is not None and timestamp - previous_time > max_gap_ns:
            peak_times.clear()
        cutoff = timestamp - burst_window_ns
        while peak_times and peak_times[0] < cutoff:
            peak_times.popleft()
        if peaks[index]:
            peak_times.append(timestamp)
        recent_count[index] = len(peak_times)
        burst[index] = len(peak_times) >= param.burst_min_spikes
        previous_time = timestamp
    return recent_count, burst


def detect_slope_spikes(
    timestamps: npt.ArrayLike,
    quick_slope: npt.ArrayLike,
    param: SlopeSpikeParam | None = None,
) -> SlopeSpikeResult:
    """Extract fixed-threshold slope pivots in both directions.

    The signed slope itself is ``spike_metric``. A point is a candidate when
    ``abs(spike_metric) >= threshold``. Candidates are emitted only when they
    are strict pivot highs/lows within ``pivot_window`` bars on both sides.
    """

    param = param or SlopeSpikeParam()
    times = _as_time_ns(timestamps)
    metric = np.asarray(quick_slope, dtype=np.float64)
    if metric.ndim != 1:
        raise ValueError("quick_slope must be one-dimensional")
    if len(times) != len(metric):
        raise ValueError("timestamps and quick_slope lengths must match")

    valid = np.isfinite(metric)
    magnitude = np.abs(metric)
    positive_candidate = valid & (metric >= param.threshold)
    negative_candidate = valid & (metric <= -param.threshold)
    positive_peak, negative_peak = _pivot_peaks(
        times,
        metric,
        positive_candidate,
        negative_candidate,
        param,
    )
    spike = positive_peak | negative_peak
    direction = np.zeros(len(metric), dtype=np.int8)
    direction[positive_peak] = 1
    direction[negative_peak] = -1

    positive_count, positive_burst = _burst_state(times, positive_peak, param)
    negative_count, negative_burst = _burst_state(times, negative_peak, param)
    recent_count = positive_count + negative_count
    burst = positive_burst | negative_burst
    max_gap_ns = int(param.max_gap_minutes * 60 * 1_000_000_000)
    gap = np.concatenate(([True], np.diff(times) > max_gap_ns))
    previous_not_burst = np.concatenate(([True], ~burst[:-1]))
    burst_start = burst & (previous_not_burst | gap)

    return {
        "spike_metric": metric.copy(),
        "spike_magnitude": magnitude,
        "positive_threshold": np.full(len(metric), param.threshold),
        "negative_threshold": np.full(len(metric), -param.threshold),
        "positive_candidate": positive_candidate,
        "negative_candidate": negative_candidate,
        "positive_peak": positive_peak,
        "negative_peak": negative_peak,
        "spike": spike,
        "spike_direction": direction,
        "recent_spike_count": recent_count,
        "positive_recent_spike_count": positive_count,
        "negative_recent_spike_count": negative_count,
        "positive_burst": positive_burst,
        "negative_burst": negative_burst,
        "burst": burst,
        "burst_start": burst_start,
        "burst_spike": spike & burst,
    }


def classify_spikes_by_fast_slope(
    timestamps: npt.ArrayLike,
    fast_slope: npt.ArrayLike,
    quick_slope: npt.ArrayLike,
    spike_direction: npt.ArrayLike,
    param: SlopeSpikeParam | None = None,
) -> SpikeTrendResult:
    """Classify trend spikes and Quick-slope counter-trend pivots.

    During a Fast-slope downtrend, a Quick-slope pivot high is an upward
    counter-trend pivot. During an uptrend, a Quick-slope pivot low is a
    downward counter-trend pivot. Middle slope is not used.
    """

    param = param or SlopeSpikeParam()
    times = _as_time_ns(timestamps)
    fast = np.asarray(fast_slope, dtype=np.float64)
    quick = np.asarray(quick_slope, dtype=np.float64)
    spike = np.asarray(spike_direction, dtype=np.int8)
    if any(values.ndim != 1 for values in (fast, quick, spike)):
        raise ValueError("slope and spike_direction arrays must be one-dimensional")
    if not (len(times) == len(fast) == len(quick) == len(spike)):
        raise ValueError("slope and spike_direction array lengths must match")
    if np.any(~np.isin(spike, (-1, 0, 1))):
        raise ValueError("spike_direction values must be -1, 0, or 1")

    trend = np.zeros(len(fast), dtype=np.int8)
    trend[np.isfinite(fast) & (fast > param.dead_zone)] = 1
    trend[np.isfinite(fast) & (fast < -param.dead_zone)] = -1
    peak = spike != 0
    trend_peak = peak & (trend != 0) & (spike == trend)
    finite_quick = np.isfinite(quick)
    upward_candidate = (
        (trend == -1)
        & finite_quick
    )
    downward_candidate = (
        (trend == 1)
        & finite_quick
    )
    upward_entry_pivot, downward_entry_pivot = _pivot_peaks(
        times,
        quick,
        upward_candidate,
        downward_candidate,
        param,
    )
    entry_pivot_peak = upward_entry_pivot | downward_entry_pivot
    entry_pivot_direction = np.zeros(len(fast), dtype=np.int8)
    entry_pivot_direction[upward_entry_pivot] = 1
    entry_pivot_direction[downward_entry_pivot] = -1
    neutral_peak = peak & (trend == 0)
    return {
        "trend_direction": trend,
        "trend_peak": trend_peak,
        "entry_pivot_peak": entry_pivot_peak,
        "entry_pivot_direction": entry_pivot_direction,
        "neutral_peak": neutral_peak,
    }


__all__ = [
    "SlopeSpikeParam",
    "SlopeSpikeResult",
    "SpikeTrendResult",
    "calculate_ma_slopes",
    "regression_slope_pct_per_hour",
    "detect_slope_spikes",
    "classify_spikes_by_fast_slope",
]
