"""Single-SMA port of crescent.pine (%/hour).

Derived from the MPL-2.0 Pine source by © oIKUo.
https://mozilla.org/MPL/2.0/
Windows are observed bars, including across market gaps, as in Pine.
"""
from dataclasses import dataclass
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True, slots=True)
class CrescentParam:
    sma_window: int = 24
    slope_window: int = 5
    slope_threshold: float = .2

    def __post_init__(self):
        for name, minimum in (("sma_window", 1), ("slope_window", 2)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        value = self.slope_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError("slope_threshold must be finite and nonnegative")


def calc(close, param=None, *, bar_seconds=10):
    """SMA -> OLS slope/bar -> slope/hour / abs(mean SMA) * 100.

    The first sma_window+slope_window-2 bars have no slope. A zero
    denominator yields NaN. No centered windows or future values are used.
    """
    param = param or CrescentParam()
    if not isinstance(param, CrescentParam):
        raise TypeError("param must be CrescentParam")
    if isinstance(bar_seconds, bool) or not isinstance(bar_seconds, (int, float)) or not np.isfinite(bar_seconds) or bar_seconds <= 0:
        raise ValueError("bar_seconds must be positive and finite")
    close = np.asarray(close, dtype=float)
    if close.ndim != 1 or not np.isfinite(close).all():
        raise ValueError("close must be a finite one-dimensional array")
    n = len(close)
    sma, slope = np.full(n, np.nan), np.full(n, np.nan)
    if n >= param.sma_window:
        sma[param.sma_window-1:] = sliding_window_view(close, param.sma_window).mean(axis=1)
    first = param.sma_window + param.slope_window - 2
    if n > first:
        y = sliding_window_view(sma[param.sma_window-1:], param.slope_window)
        mean = y.mean(axis=1)
        x = np.arange(param.slope_window, dtype=float) - (param.slope_window-1)/2
        per_bar = ((y - mean[:, None]) @ x) / (x @ x)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = per_bar * (3600 / bar_seconds) / np.abs(mean) * 100
        slope[first:] = np.where(mean != 0, value, np.nan)
    return {"sma": sma, "sma_slope_pct_hour": slope,
            "long_exit": slope < -param.slope_threshold,
            "short_exit": slope > param.slope_threshold}
