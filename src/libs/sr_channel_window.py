"""Shared UTC array selection and JST calendar arithmetic for S/R APIs."""
from calendar import monthrange
from datetime import datetime, timedelta, timezone
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ArrayDict: TypeAlias = dict[str, npt.NDArray]
HOURS = {"D1": 24, "H4": 4, "H1": 1}
JST = timezone(timedelta(hours=9))

def cutoff_datetime(cutoff: datetime | str | np.datetime64) -> datetime:
    """Naive strings/datetimes are JST; numpy datetime64 values are UTC."""
    if isinstance(cutoff, np.datetime64):
        if np.isnat(cutoff):
            raise ValueError("cutoff cannot be NaT")
        cutoff = np.datetime_as_string(cutoff, unit="us") + "+00:00"
    if isinstance(cutoff, str):
        cutoff = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    if not isinstance(cutoff, datetime):
        raise TypeError("cutoff must be a datetime, ISO string or numpy datetime64")
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=JST)
    return cutoff.astimezone(JST)


def utc_time(value: datetime) -> np.datetime64:
    return np.datetime64(value.astimezone(timezone.utc).replace(tzinfo=None), "ns")


def window_start(timeframe: str, cutoff: datetime | str | np.datetime64) -> datetime | None:
    if timeframe not in HOURS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    cutoff = cutoff_datetime(cutoff)
    if timeframe == "H1":
        return cutoff - timedelta(days=5)
    if timeframe == "H4":
        year, month = (cutoff.year - 1, 12) if cutoff.month == 1 else (cutoff.year, cutoff.month - 1)
        return cutoff.replace(year=year, month=month, day=min(cutoff.day, monthrange(year, month)[1]))
    return None


def select_range(candles: ArrayDict, timeframe: str, cutoff: datetime | str | np.datetime64,
                 start: datetime | None = None) -> ArrayDict:
    """Select completed, sorted candles inside the inclusive opening-time range."""
    if timeframe not in HOURS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    arrays = {key: np.asarray(value) for key, value in candles.items()}
    times = arrays["jst"]
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("jst must be a one-dimensional datetime64 array of UTC instants")
    if any(a.ndim != 1 or len(a) != len(times) for a in arrays.values()):
        raise ValueError("candle arrays must be one-dimensional and have the same length")
    if np.isnat(times).any():
        raise ValueError("jst cannot contain NaT")
    times = times.astype("datetime64[ns]")
    end = utc_time(cutoff_datetime(cutoff))
    mask = times + np.timedelta64(HOURS[timeframe], "h") <= end
    if start is not None:
        mask &= times >= utc_time(start)
    order = np.flatnonzero(mask)
    order = order[np.argsort(times[order], kind="stable")]
    if not len(order):
        raise ValueError(f"{timeframe}: no completed candles in the requested window")
    if len(np.unique(times[order])) != len(order):
        raise ValueError(f"{timeframe}: duplicate candle timestamps")
    selected = {key: value[order].copy() for key, value in arrays.items()}
    selected["jst"] = times[order].copy()
    return selected



def select_window(candles: ArrayDict, timeframe: str, cutoff: datetime | str | np.datetime64) -> ArrayDict:
    """Existing composite windows: D1 all history, H4 one month, H1 five days."""
    return select_range(candles, timeframe, cutoff, window_start(timeframe, cutoff))


def subtract_period(end: datetime, years: int = 0, months: int = 0, days: int = 0) -> datetime:
    """Subtract calendar years/months with month-end clamping, then calendar days."""
    for name, value in (("years", years), ("months", months), ("days", days)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not (years or months or days):
        raise ValueError("period must be positive")
    month_index = end.year * 12 + end.month - 1 - int(years) * 12 - int(months)
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    if year < 1:
        raise ValueError("period extends before year 1")
    return end.replace(year=year, month=month, day=min(end.day, monthrange(year, month)[1])) - timedelta(days=int(days))
