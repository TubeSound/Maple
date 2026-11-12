"""Shared OHLC normalization for array-dictionary indicators."""
import numpy as np

from .supertrend import CandleDict


def validate_candles(candles: CandleDict) -> CandleDict:
    """Return float64 OHLC arrays without modifying the caller's arrays."""
    values = {key: np.asarray(candles[key], dtype=np.float64)
              for key in ("open", "high", "low", "close")}
    if any(a.ndim != 1 for a in values.values()) or len({len(a) for a in values.values()}) != 1:
        raise ValueError("OHLC arrays must be one-dimensional and have the same length")
    if any(not np.all(np.isfinite(a) & (a > 0)) for a in values.values()):
        raise ValueError("OHLC prices must be finite and positive")
    opened, high, low, close = (values[key] for key in ("open", "high", "low", "close"))
    if np.any(high < np.maximum(opened, close)) or np.any(low > np.minimum(opened, close)):
        raise ValueError("high/low must enclose open/close")
    return values
