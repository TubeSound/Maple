"""Causal Wasserstein-1 distance between sliding M1 OHLC distributions.

Each window pools open/high/low/close with equal weight (15 bars = 60
observations). Prices retain their level; no centering or standardization.
Results at index i become available only when candle i has closed.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .technical import linear_regression


@dataclass(frozen=True)
class WassersteinState:
    window_minutes: int = 15
    slide_minutes: int = 1
    threshold: float | None = None

    def __post_init__(self):
        for name in ("window_minutes", "slide_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.threshold is not None and (
            not np.isfinite(self.threshold) or self.threshold < 0
        ):
            raise ValueError("threshold must be finite and non-negative")

    def calc(self, candles):
        """Accept aligned array dictionaries or DataFrames, including `jst`.

        Compare window ending at i with the window ending at i-slide_minutes,
        evaluated every minute. `distance` is in price units. `distance_pct`
        divides it by the current window's mean price. Optional `state` uses
        the raw-price threshold: 0=stable distribution, 1=changing, NaN=unknown.
        Stability alone does not establish a narrow or profitable range.
        Missing/nonfinite OHLC invalidates the comparison; timestamp gaps
        invalidate comparisons until the complete pair of windows is rebuilt.
        `slope_pct_per_hour` fits the current window OHLC against minute
        coordinates using technical.linear_regression, then scales by
        60 * 100 / current window mean price. It requires only one window.
        """
        arrays = [np.asarray(candles[k], dtype=float)
                  for k in ("open", "high", "low", "close")]
        if any(a.ndim != 1 for a in arrays) or len({len(a) for a in arrays}) != 1:
            raise ValueError("OHLC arrays must be one-dimensional and aligned")
        times = np.asarray(candles["jst"])
        n = len(arrays[0])
        if times.ndim != 1 or len(times) != n:
            raise ValueError("jst must be one-dimensional and aligned with OHLC")
        # Handle Python datetime, pandas Timestamp and numpy datetime64 without
        # converting or changing the timezone carried by the input.
        def seconds(t):
            if t is None or str(t) == "NaT":
                raise ValueError("jst timestamps must be valid")
            if hasattr(t, "timestamp"):
                return t.timestamp()
            return float(np.datetime64(t, "ns").astype("int64")) / 1e9

        epochs = np.array([seconds(t) for t in times])
        if not np.all(np.isfinite(epochs)) or (n > 1 and np.any(np.diff(epochs) <= 0)):
            raise ValueError("jst timestamps must be valid, unique and increasing")
        prices = np.column_stack(arrays)
        finite = np.isfinite(prices).all(axis=1)
        if np.any(prices[finite] <= 0):
            raise ValueError("OHLC prices must be positive")
        o, h, l, c = prices.T
        if np.any(finite & ((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l))):
            raise ValueError("inconsistent OHLC high/low")
        distance = np.full(n, np.nan)
        pct = np.full(n, np.nan)
        state = np.full(n, np.nan)
        slope_pct_per_hour = np.full(n, np.nan)
        w, s = self.window_minutes, self.slide_minutes
        # OHLC share the same minute coordinate, not four sequential times.
        # The slope is price/minute, normalized by the same window mean as W1.
        x = np.repeat(np.arange(w, dtype=float), 4)
        for i in range(w - 1, n):
            begin = i - w + 1
            if w < 2 or not finite[begin:i + 1].all():
                continue
            if not np.all(np.isclose(np.diff(epochs[begin:i + 1]), 60, rtol=0, atol=1e-6)):
                continue
            values = prices[begin:i + 1].reshape(-1)
            slope, _ = linear_regression(x, values)
            slope_pct_per_hour[i] = slope * 60 * 100 / np.mean(values)
        for i in range(w + s - 1, n):
            begin = i - w - s + 1
            if not finite[begin:i + 1].all():
                continue
            if not np.all(np.isclose(np.diff(epochs[begin:i + 1]), 60, rtol=0, atol=1e-6)):
                continue
            previous = prices[begin:begin + w].reshape(-1)
            current = prices[i - w + 1:i + 1].reshape(-1)
            # For equal-size equally weighted 1D samples, W1 is exactly the
            # mean absolute difference of their sorted order statistics.
            distance[i] = np.mean(np.abs(np.sort(current) - np.sort(previous)))
            pct[i] = 100 * distance[i] / np.mean(current)
            if self.threshold is not None:
                state[i] = float(distance[i] > self.threshold)
        return {"distance": distance, "distance_pct": pct, "state": state,
                "slope_pct_per_hour": slope_pct_per_hour}
