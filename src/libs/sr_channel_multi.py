"""D1/H4/H1 SRchannel calculations with flat dictionaries of NumPy arrays.

Input keys: D1_open/high/low/close/jst, H4_*, H1_*.
jst arrays are datetime64[ns] UTC instants (NumPy has no timezone).
Output values are arrays only; JSON and DataFrames belong in sr_channel_report.
"""
from datetime import datetime
from typing import TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

from .sr_channel import SRChannelConfig, SRChannelResult, calc as calc_channel
from .supertrend import FloatArray, IntArray, BoolArray

ArrayDict: TypeAlias = dict[str, npt.NDArray]
TimeArray: TypeAlias = npt.NDArray[np.datetime64]
from .sr_channel_window import HOURS, cutoff_datetime, utc_time, window_start, select_window


class ChannelTable(TypedDict):
    id: npt.NDArray[np.str_]
    timeframe: npt.NDArray[np.str_]
    rank: npt.NDArray[np.int64]
    lower: FloatArray
    upper: FloatArray
    strength: FloatArray
    state: IntArray


class OverlapTable(TypedDict):
    lower: FloatArray
    upper: FloatArray
    timeframes: BoolArray  # (regions, 3), columns in HOURS order
    timeframe_count: npt.NDArray[np.int64]
    channel_mask: BoolArray  # (regions, channels), same order as ChannelTable
    state: IntArray


def price_state(lower: FloatArray, upper: FloatArray, close: float) -> IntArray:
    """1=support, -1=resistance, 0=in channel, matching sr_channel.state."""
    return np.where(lower > close, -1, np.where(upper < close, 1, 0)).astype(np.int8)


def intersections(channels: ChannelTable, reference_price: float) -> OverlapTable:
    """Exact non-bridging overlaps as arrays with explicit membership masks."""
    lo, hi = np.asarray(channels["lower"]), np.asarray(channels["upper"])
    tf = np.asarray(channels["timeframe"])
    if lo.ndim != 1 or hi.shape != lo.shape or tf.shape != lo.shape:
        raise ValueError("channel lower/upper/timeframe arrays must have equal one-dimensional shapes")
    if not np.all(np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)):
        raise ValueError("channel bounds must be finite and lower <= upper")
    if not np.isin(tf, list(HOURS)).all():
        raise ValueError("unsupported channel timeframe")
    edges = np.unique(np.r_[lo, hi])
    regions, memberships = [], []
    for left, right in zip(edges, edges[1:]):
        midpoint = (left + right) / 2
        active = (lo <= midpoint) & (midpoint <= hi)
        present = np.array([np.any(active & (tf == name)) for name in HOURS], dtype=np.bool_)
        if present.sum() < 2:
            continue
        if regions and regions[-1][1] == left and np.array_equal(memberships[-1], active):
            regions[-1][1] = right
        else:
            regions.append([left, right])
            memberships.append(active)
    bounds = np.asarray(regions, dtype=np.float64).reshape(-1, 2)
    masks = np.asarray(memberships, dtype=np.bool_).reshape(len(regions), len(lo))
    presence = np.zeros((len(regions), len(HOURS)), dtype=np.bool_)
    for k, name in enumerate(HOURS):
        presence[:, k] = np.any(masks[:, tf == name], axis=1)
    return {"lower": bounds[:, 0], "upper": bounds[:, 1],
            "timeframes": presence, "timeframe_count": presence.sum(axis=1, dtype=np.int64),
            "channel_mask": masks, "state": price_state(bounds[:, 0], bounds[:, 1], reference_price)}


def calc(candles: ArrayDict, cutoff: datetime | str | np.datetime64,
         config: SRChannelConfig | None = None, *, starts: dict | None = None) -> ArrayDict:
    """Calculate independent windows and combined bands; all values are arrays.

    D1_*/H4_*/H1_* contain selected OHLC/time and indicator history (different
    lengths per timeframe). channel_* is the combined final band table;
    overlap_* is the exact overlap table. Metadata uses length-one arrays.
    """
    cfg = config or SRChannelConfig()
    end = cutoff_datetime(cutoff)
    output: ArrayDict = {"as_of": np.array([utc_time(end)])}
    rows = []
    for tf in HOURS:
        start = starts[tf] if starts is not None else window_start(tf, end)
        from .sr_channel_window import select_range
        data = select_range({key: candles[f"{tf}_{key}"] for key in ("open", "high", "low", "close", "jst")}, tf, end, start)
        result: SRChannelResult = calc_channel({key: data[key] for key in ("open", "high", "low", "close")}, cfg)
        output.update({f"{tf}_{key}": value for key, value in {**data, **result}.items()})
        output[f"{tf}_window_start"] = np.array([utc_time(start) if start else np.datetime64("NaT", "ns")])
        for k in range(cfg.max_channels):
            if np.isfinite(result["upper"][-1, k]):
                rows.append((f"{tf}:{k+1}", tf, k+1, result["lower"][-1, k],
                             result["upper"][-1, k], result["strength"][-1, k]))
    reference_price = float(output["H1_close"][-1])
    table: ChannelTable = {
        "id": np.array([r[0] for r in rows], dtype="<U5"),
        "timeframe": np.array([r[1] for r in rows], dtype="<U2"),
        "rank": np.array([r[2] for r in rows], dtype=np.int64),
        "lower": np.array([r[3] for r in rows], dtype=np.float64),
        "upper": np.array([r[4] for r in rows], dtype=np.float64),
        "strength": np.array([r[5] for r in rows], dtype=np.float64),
        "state": np.zeros(len(rows), dtype=np.int8)}
    table["state"] = price_state(table["lower"], table["upper"], reference_price)
    output.update({f"channel_{key}": value for key, value in table.items()})
    output.update({f"overlap_{key}": value for key, value in intersections(table, reference_price).items()})
    output["reference_price"] = np.array([reference_price], dtype=np.float64)
    output["reference_bar_open"] = output["H1_jst"][-1:].copy()
    return output


# Existing function name retained, with the new array-dictionary contract.
analyze_frames = calc
