"""Causal W1 decline-to-low-plateau detector for regularly sampled M1 data.

Input is W1 in percent (distance_pct), e.g. 0.05 means 0.05%.
No price-bottom, breakout, entry or future prediction is implied.
"""
from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .technical import linear_regression


@dataclass(frozen=True)
class W1PlateauParams:
    lookback_bars: int = 30
    flat_bars: int = 5
    confirm_bars: int = 3
    low_threshold: float = 0.05
    min_drop: float = 0.05
    max_flat_range: float = 0.01
    max_flat_slope: float = 0.001
    min_decline_slope: float = 0.001
    sample_minutes: float = 1.0

    def __post_init__(self):
        for name in ("lookback_bars", "flat_bars", "confirm_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.flat_bars < 2:
            raise ValueError("flat_bars must be >= 2")
        for name in ("low_threshold", "min_drop", "max_flat_range",
                     "max_flat_slope", "min_decline_slope", "sample_minutes"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.sample_minutes <= 0 or self.min_drop <= 0 or self.min_decline_slope <= 0:
            raise ValueError("sample_minutes, min_drop and min_decline_slope must be positive")


class W1PlateauDetector:
    """Detect a decline followed by low, narrow and approximately level W1.

    `signal=1` only on the confirmation bar; `plateau=1` while the confirmed
    low/flat condition continues. Nothing is backdated to a historical bottom.
    A new event requires a fresh peak AFTER the last active plateau sample.
    The thresholds are illustrative defaults, not optimized parameters.
    """

    def __init__(self, params=None):
        self.params = params if params is not None else W1PlateauParams()
        self.reset()

    def reset(self):
        p = self.params
        self._history = deque(maxlen=p.lookback_bars + p.flat_bars)
        self._index = -1
        self._last_seconds = None
        self._candidate_count = 0
        self._active = False
        self._used_through = -1
        self._peak = None

    def update(self, value, time=None):
        """Consume one confirmed W1 sample, optionally with its bar timestamp.

        Timestamps must increase; a sampling gap or NaN/Inf resets detection.
        Without timestamps, calls are assumed to be regularly spaced.
        Slopes are percentage POINTS/minute; range/drop are percentage points.
        """
        p = self.params
        seconds = None
        if time is not None:
            if str(time) == "NaT":
                raise ValueError("invalid timestamp")
            seconds = (time.timestamp() if hasattr(time, "timestamp") else
                       float(np.datetime64(time, "ns").astype("int64")) / 1e9)
            if not np.isfinite(seconds):
                raise ValueError("invalid timestamp")
            if self._last_seconds is not None:
                delta = seconds - self._last_seconds
                if delta <= 0:
                    raise ValueError("timestamps must be unique and increasing")
                if not np.isclose(delta, p.sample_minutes * 60, rtol=0, atol=1e-6):
                    self.reset()
        result = dict(signal=0, plateau=0, flat_slope=np.nan, flat_range=np.nan, drop=np.nan)
        if not np.isfinite(value):
            self.reset()
            self._last_seconds = seconds
            return result
        if value < 0:
            raise ValueError("W1 must be non-negative")
        self._last_seconds = seconds
        self._index += 1
        i = self._index
        self._history.append((i, float(value)))
        history = list(self._history)
        if len(history) < p.flat_bars:
            return result
        flat = np.array([v for _, v in history[-p.flat_bars:]])
        slope, _ = linear_regression(np.arange(p.flat_bars) * p.sample_minutes, flat)
        spread = float(np.ptp(flat))
        result.update(flat_slope=float(slope), flat_range=spread)
        is_flat = (np.max(flat) <= p.low_threshold and
                   abs(slope) <= p.max_flat_slope and spread <= p.max_flat_range)
        if self._active:
            if is_flat:
                self._used_through = i
                result.update(plateau=1, drop=float(self._peak - np.mean(flat)))
                return result
            self._active = False
            self._candidate_count = 0
            self._peak = None
            # The first non-flat sample may become a new peak, but no old
            # peak can arm another event for the same decline.
            return result
        if not is_flat:
            self._candidate_count = 0
            return result
        prior = [(j, v) for j, v in history[:-p.flat_bars] if j > self._used_through]
        if not prior:
            self._candidate_count = 0
            return result
        # Most recent tied maximum gives the shortest causal decline segment.
        peak_i, peak = max(prior, key=lambda pair: (pair[1], pair[0]))
        flat_begin = history[-p.flat_bars][0]
        falling = [(j, v) for j, v in history if peak_i <= j <= flat_begin]
        decline_slope, _ = linear_regression(
            np.array([j - peak_i for j, _ in falling]) * p.sample_minutes,
            np.array([v for _, v in falling]))
        drop = float(peak - np.mean(flat))
        result["drop"] = drop
        if drop < p.min_drop or decline_slope > -p.min_decline_slope:
            self._candidate_count = 0
            return result
        self._candidate_count += 1
        if self._candidate_count >= p.confirm_bars:
            self._active = True
            self._used_through = i
            self._peak = peak
            result.update(signal=1, plateau=1)
        return result

    def calc(self, values, times=None):
        """Reset and evaluate a complete vector, equivalent to online update."""
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if times is not None and len(times) != len(values):
            raise ValueError("timestamps and values must have the same length")
        self.reset()
        out = {name: np.full(len(values), np.nan) for name in
               ("signal", "plateau", "flat_slope", "flat_range", "drop")}
        for i, value in enumerate(values):
            result = self.update(value, None if times is None else times[i])
            for name, result_value in result.items():
                out[name][i] = result_value
        return out
