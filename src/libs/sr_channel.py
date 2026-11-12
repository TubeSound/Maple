# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. https://mozilla.org/MPL/2.0/
# Python port of SRchannel.pine, © LonesomeTheBlue.
"""Sequential Support Resistance Channels, using confirmed OHLC bars only.

Inputs are aligned open/high/low/close arrays in chronological order. Results
are indexed by confirmation bar, never backdated to the pivot's origin bar.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from .candle_arrays import validate_candles
from .sr_channel_window import HOURS, cutoff_datetime, utc_time, select_range, subtract_period
from .supertrend import CandleDict, FloatArray, IntArray, BoolArray


class SRChannelResult(TypedDict):
    upper: FloatArray
    lower: FloatArray
    strength: FloatArray
    state: IntArray
    pivot_high: FloatArray
    pivot_low: FloatArray
    channel_width: FloatArray
    updated: BoolArray
    not_in_channel: BoolArray
    resistance_broken: BoolArray
    support_broken: BoolArray
    ma1: FloatArray
    ma2: FloatArray


@dataclass(frozen=True)
class SRChannelConfig:
    pivot_period: int = 10
    source: str = "High/Low"
    channel_width_pct: int = 5
    min_strength: int = 1
    max_channels: int = 6
    loopback: int = 290
    ma1_enabled: bool = False
    ma1_length: int = 50
    ma1_type: str = "SMA"
    ma2_enabled: bool = False
    ma2_length: int = 200
    ma2_type: str = "SMA"

    def __post_init__(self):
        for name, low, high in (("pivot_period", 4, 30),
                                ("channel_width_pct", 1, 8),
                                ("max_channels", 1, 10), ("loopback", 100, 400)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name in ("min_strength", "ma1_length", "ma2_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.source not in ("High/Low", "Close/Open"):
            raise ValueError("source must be High/Low or Close/Open")
        if self.ma1_type not in ("SMA", "EMA") or self.ma2_type not in ("SMA", "EMA"):
            raise ValueError("MA type must be SMA or EMA")


def _pivot(values, bar, period, high):
    if bar < 2 * period:
        return np.nan
    center = bar - period
    value = values[center]
    left = values[center - period:center]
    right = values[center + 1:bar + 1]
    # Equal older extremes are allowed; equal newer extremes disqualify it.
    valid = (np.all(value >= left) and np.all(value > right)) if high else (
        np.all(value <= left) and np.all(value < right))
    return value if valid else np.nan


def _channels(pivots, width, highs, lows, min_strength):
    candidates = []
    for seed in pivots:
        lo = hi = seed
        pivot_count = 0
        for price in pivots:  # newest first, as in Pine's array.unshift
            distance = hi - price if price <= hi else price - lo
            if distance <= width:
                if price <= hi:
                    lo = min(lo, price)
                else:
                    hi = max(hi, price)
                pivot_count += 1
        touches = int(np.count_nonzero(((lows >= lo) & (lows <= hi)) |
                                      ((highs >= lo) & (highs <= hi))))
        candidates.append([20 * pivot_count + touches, hi, lo])

    bounds = np.zeros((10, 2))  # upper, lower; Pine's zero sentinel
    strengths = np.zeros(10)
    selected = 0
    for _ in pivots:
        best_strength, best_index = -1, -1
        for index, (strength, _, _) in enumerate(candidates):
            if strength > best_strength and strength >= min_strength * 20:
                best_strength, best_index = strength, index
        if best_index < 0:
            continue
        strength, hi, lo = candidates[best_index]
        bounds[selected] = (hi, lo)
        strengths[selected] = strength
        # Preserve endpoint-based suppression; do not merge overlapping bands.
        for candidate in candidates:
            if lo <= candidate[1] <= hi or lo <= candidate[2] <= hi:
                candidate[0] = -1
        selected += 1
        if selected >= 10:
            break
    # Literal port of Pine's final sorting loop (including its stren update).
    for x in range(9):
        for y in range(x + 1, 10):
            if strengths[y] > strengths[x]:
                strengths[x] = strengths[y]
                bounds[[x, y]] = bounds[[y, x]]
    return bounds, strengths


def _ma(close, enabled, length, kind):
    result = np.full(len(close), np.nan)
    if not enabled:
        return result
    if kind == "EMA":
        alpha = 2 / (length + 1)
        for i, value in enumerate(close):
            result[i] = value if i == 0 else alpha * value + (1 - alpha) * result[i - 1]
    else:
        for i in range(length - 1, len(close)):
            result[i] = np.mean(close[i - length + 1:i + 1])
    return result


def calc(candles: CandleDict, config: SRChannelConfig | None = None) -> SRChannelResult:
    """Calculate historical S/R state, strength, colors and breakout alerts.

    upper/lower/strength/state have shape (bars, max_channels); state is
    1=support, -1=resistance, 0=in channel, 2=unused. Zero price levels are
    converted to NaN for consumers. Pass completed bars only. Appending future
    bars does not change any prior row. Pine's drawings extend both ways, but
    this history does not retroactively apply the final bands to earlier bars.
    """
    cfg = config or SRChannelConfig()
    values = validate_candles(candles)
    opened, high, low, close = (values[key] for key in ("open", "high", "low", "close"))
    n = len(close)
    shape = (n, cfg.max_channels)
    upper, lower, strength = (np.full(shape, np.nan) for _ in range(3))
    state = np.full(shape, 2, dtype=np.int8)
    ph, pl, width = (np.full(n, np.nan) for _ in range(3))
    resistance_broken, support_broken, updated = (np.zeros(n, dtype=bool) for _ in range(3))
    not_in_channel = np.ones(n, dtype=bool)
    src_high = high if cfg.source == "High/Low" else np.maximum(opened, close)
    src_low = low if cfg.source == "High/Low" else np.minimum(opened, close)
    pivots, locations = [], []
    bounds, scores = np.zeros((10, 2)), np.zeros(10)
    for i in range(n):
        width[i] = (np.max(high[max(0, i - 299):i + 1]) -
                    np.min(low[max(0, i - 299):i + 1])) * cfg.channel_width_pct / 100
        ph[i] = _pivot(src_high, i, cfg.pivot_period, True)
        pl[i] = _pivot(src_low, i, cfg.pivot_period, False)
        if np.isfinite(ph[i]) or np.isfinite(pl[i]):
            updated[i] = True
            # Pine stores only the high when both pivots confirm on one bar.
            pivots.insert(0, ph[i] if np.isfinite(ph[i]) else pl[i])
            locations.insert(0, i)  # confirmation index, not origin index
            while locations and i - locations[-1] > cfg.loopback:
                locations.pop()
                pivots.pop()
            # Inclusive 0..loopback, hence loopback + 1 bars for touches.
            start = max(0, i - cfg.loopback)
            bounds, scores = _channels(pivots, width[i], high[start:i + 1], low[start:i + 1], cfg.min_strength)
        for k, (hi, lo) in enumerate(bounds[:cfg.max_channels]):
            if hi != 0:
                upper[i, k], lower[i, k], strength[i, k] = hi, lo, scores[k]
                state[i, k] = -1 if lo > close[i] else 1 if hi < close[i] else 0
            if lo <= close[i] <= hi:
                not_in_channel[i] = False
        if i > 0 and not_in_channel[i]:
            # Both closes use today's bands, including on recalculation bars.
            for hi, lo in bounds[:cfg.max_channels]:
                resistance_broken[i] |= close[i - 1] <= hi < close[i]
                support_broken[i] |= close[i - 1] >= lo > close[i]
    return {"upper": upper, "lower": lower, "strength": strength, "state": state,
            "pivot_high": ph, "pivot_low": pl, "channel_width": width,
            "updated": updated, "not_in_channel": not_in_channel,
            "resistance_broken": resistance_broken, "support_broken": support_broken,
            "ma1": _ma(close, cfg.ma1_enabled, cfg.ma1_length, cfg.ma1_type),
            "ma2": _ma(close, cfg.ma2_enabled, cfg.ma2_length, cfg.ma2_type)}


class TimedCandleDict(CandleDict):
    jst: npt.NDArray[np.datetime64]  # UTC instants, datetime64[ns]


class SRLevelsResult(TypedDict):
    upper: FloatArray
    lower: FloatArray
    strength: FloatArray
    state: IntArray  # 1=support, -1=resistance, 0=in_channel
    rank: npt.NDArray[np.int64]
    as_of: npt.NDArray[np.datetime64]  # length-one UTC arrays below
    window_start: npt.NDArray[np.datetime64]
    reference_bar_open: npt.NDArray[np.datetime64]
    reference_price: FloatArray
    bars: npt.NDArray[np.int64]


def calc_sr(
    candles: TimedCandleDict,
    timeframe: str,
    as_of: datetime | str | np.datetime64 | None = None,
    *,
    years: int | None = None,
    months: int | None = None,
    days: int | None = None,
    start: datetime | str | np.datetime64 | None = None,
    config: SRChannelConfig | None = None,
) -> SRLevelsResult:
    """Return only the final S/R bands, rather than their historical evolution.

    Defaults: D1 one calendar year, H4 one calendar month, H1 five calendar days.
    as_of=None anchors the window at the end of the latest supplied candle.
    Explicit as_of uses completed bars only. Naive dates/strings mean JST;
    jst and numpy datetime64 parameters are UTC instants, as in the other APIs.
    years/months/days override the default duration; start specifies an exact
    inclusive opening-time boundary instead. Never initialize with older bars.
    state=0 is retained when the reference close is inside a band.
    """
    if timeframe not in HOURS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    times = np.asarray(candles["jst"])
    if times.ndim != 1 or not np.issubdtype(times.dtype, np.datetime64):
        raise ValueError("jst must be a one-dimensional datetime64 array of UTC instants")
    if np.isnat(times).any():
        raise ValueError("jst cannot contain NaT")
    if not len(times):
        raise ValueError(f"{timeframe}: no completed candles in the requested window")
    end = cutoff_datetime(as_of if as_of is not None else
                          times.max().astype("datetime64[ns]") + np.timedelta64(HOURS[timeframe], "h"))
    explicit_period = any(v is not None for v in (years, months, days))
    if start is not None:
        if explicit_period:
            raise ValueError("start cannot be combined with years/months/days")
        begin = cutoff_datetime(start)
        if begin >= end:
            raise ValueError("start must be before as_of")
    else:
        if not explicit_period:
            years, months, days = {"D1": (1, 0, 0), "H4": (0, 1, 0), "H1": (0, 0, 5)}[timeframe]
        begin = subtract_period(end, 0 if years is None else years,
                                0 if months is None else months, 0 if days is None else days)
    selected = select_range(candles, timeframe, end, begin)
    result = calc(selected, config)
    active = np.isfinite(result["upper"][-1])
    snapshot = {key: result[key][-1, active].copy() for key in ("upper", "lower", "strength", "state")}
    snapshot["rank"] = np.flatnonzero(active).astype(np.int64) + 1
    snapshot["as_of"] = np.array([utc_time(end)])
    snapshot["window_start"] = np.array([utc_time(begin)])
    snapshot["reference_bar_open"] = selected["jst"][-1:].copy()
    snapshot["reference_price"] = np.array([selected["close"][-1]], dtype=np.float64)
    snapshot["bars"] = np.array([len(selected["close"])], dtype=np.int64)
    return snapshot
