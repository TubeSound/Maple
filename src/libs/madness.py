"""Madness: causal mud-market detection from M5/M15 Supertrend reversals.

"Mud" means that price repeatedly changes direction without travelling far.
The live detector uses only the preceding four hours.  The optional oracle
labels use the following four hours and are strictly for offline training.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd

try:
    from .supertrend import supertrend
except ImportError:  # pragma: no cover
    from supertrend import supertrend


class PriceRegime(IntEnum):
    UNCERTAIN = 0
    MUDDY = 1
    TREND = 2


@dataclass(frozen=True, slots=True)
class OracleParam:
    horizon_hours: float = 4.0
    muddy_efficiency_max: float = 0.25
    muddy_direction_ratio_max: float = 0.35
    muddy_balance_min: float = 0.20
    trend_efficiency_min: float = 0.45
    trend_direction_ratio_min: float = 0.55
    max_gap_minutes: float = 20.0


@dataclass(frozen=True, slots=True)
class MadnessParam:
    lookback_hours: float = 4.0
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    m5_flip_min: int = 3
    m15_flip_min: int = 1
    efficiency_max: float = 0.35
    direction_ratio_max: float = 0.45
    max_gap_minutes: float = 20.0

    def __post_init__(self) -> None:
        if self.lookback_hours <= 0.0 or not np.isfinite(self.lookback_hours):
            raise ValueError("lookback_hours must be positive and finite")
        if self.supertrend_period <= 0:
            raise ValueError("supertrend_period must be positive")
        if self.supertrend_multiplier < 0.0:
            raise ValueError("supertrend_multiplier must be non-negative")
        if self.m5_flip_min < 0 or self.m15_flip_min < 0:
            raise ValueError("flip thresholds must be non-negative")
        for name in ("efficiency_max", "direction_ratio_max"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.max_gap_minutes <= 0.0:
            raise ValueError("max_gap_minutes must be positive")


@dataclass(frozen=True, slots=True)
class MadnessGateParam:
    """Causal M15 reversal-cycle gate used by Mars entries."""

    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    swing_threshold_pct: float = 3.8
    short_cycle_hours: float = 7.0
    short_cycle_streak: int = 3
    large_swing_hold_hours: float = 168.0
    short_cycle_hold_hours: float = 24.0

    def __post_init__(self) -> None:
        if self.supertrend_period <= 0:
            raise ValueError("supertrend_period must be positive")
        for name in (
            "supertrend_multiplier",
            "swing_threshold_pct",
            "short_cycle_hours",
            "large_swing_hold_hours",
            "short_cycle_hold_hours",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if self.swing_threshold_pct == 0.0:
            raise ValueError("swing_threshold_pct must be positive")
        if self.short_cycle_hours == 0.0:
            raise ValueError("short_cycle_hours must be positive")
        if self.short_cycle_streak <= 0:
            raise ValueError("short_cycle_streak must be positive")


class MadnessGateResult(TypedDict):
    madness_range_state: npt.NDArray[np.bool_]
    madness_trend_state: npt.NDArray[np.bool_]
    madness_large_swing: npt.NDArray[np.bool_]
    madness_short_cycle: npt.NDArray[np.bool_]
    madness_cycle_hours: npt.NDArray[np.float64]
    madness_swing_pct: npt.NDArray[np.float64]
    madness_short_cycle_streak: npt.NDArray[np.int32]


def _cycle_gate_from_flips(
    close: npt.ArrayLike,
    flips: npt.ArrayLike,
    param: MadnessGateParam,
) -> dict[str, np.ndarray]:
    """Calculate the gate on completed, gap-free M15 candles."""

    prices = np.asarray(close, dtype=np.float64)
    changed = np.asarray(flips, dtype=np.bool_)
    if prices.ndim != 1 or changed.ndim != 1 or len(prices) != len(changed):
        raise ValueError("close and flips must be matching one-dimensional arrays")
    size = len(prices)
    blocked = np.zeros(size, dtype=np.bool_)
    large_swing = np.zeros(size, dtype=np.bool_)
    short_cycle = np.zeros(size, dtype=np.bool_)
    cycle_hours = np.full(size, np.nan, dtype=np.float64)
    swing_pct = np.full(size, np.nan, dtype=np.float64)
    streak_values = np.zeros(size, dtype=np.int32)
    large_hold_bars = int(np.ceil(param.large_swing_hold_hours / 0.25))
    short_hold_bars = int(np.ceil(param.short_cycle_hold_hours / 0.25))
    last_flip = -1
    last_flip_price = np.nan
    streak = 0
    remaining = 0
    current_cycle = np.nan
    current_swing = np.nan

    for index in range(size):
        if changed[index]:
            if last_flip >= 0 and np.isfinite(last_flip_price) and last_flip_price != 0.0:
                current_cycle = (index - last_flip) * 0.25
                current_swing = abs((prices[index] / last_flip_price - 1.0) * 100.0)
                is_short = current_cycle <= param.short_cycle_hours
                streak = streak + 1 if is_short else 0
                short_cycle[index] = is_short
                if current_swing >= param.swing_threshold_pct:
                    large_swing[index] = True
                    remaining = max(remaining, large_hold_bars)
                if streak >= param.short_cycle_streak:
                    remaining = max(remaining, short_hold_bars)
            last_flip = index
            last_flip_price = prices[index]
        cycle_hours[index] = current_cycle
        swing_pct[index] = current_swing
        streak_values[index] = streak
        blocked[index] = remaining > 0
        if remaining > 0:
            remaining -= 1

    return {
        "blocked": blocked,
        "large_swing": large_swing,
        "short_cycle": short_cycle,
        "cycle_hours": cycle_hours,
        "swing_pct": swing_pct,
        "short_cycle_streak": streak_values,
    }


def madness_entry_gate(
    candles: dict[str, Any],
    source_timeframe_minutes: float,
    param: MadnessGateParam | None = None,
) -> MadnessGateResult:
    """Return an M15-based entry gate aligned causally to source candles.

    Input timestamps denote candle starts.  Each M15 value becomes visible only
    at its closing boundary, so no incomplete higher-timeframe candle leaks
    into a Mars signal.
    """

    settings = param or MadnessGateParam()
    size = _validate_candles(candles, "candles")
    if not np.isfinite(source_timeframe_minutes) or source_timeframe_minutes <= 0.0:
        raise ValueError("source_timeframe_minutes must be positive and finite")
    if source_timeframe_minutes > 15.0 or 15.0 % source_timeframe_minutes != 0.0:
        raise ValueError("Madness requires a source timeframe that divides M15")

    source_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(candles["jst"], dtype=object), format="mixed"))
    if source_index.hasnans or np.any(np.diff(source_index.as_unit("ns").asi8) <= 0):
        raise ValueError("candle jst must be valid and strictly increasing")
    frame = pd.DataFrame(
        {
            "open": np.asarray(candles["open"], dtype=float),
            "high": np.asarray(candles["high"], dtype=float),
            "low": np.asarray(candles["low"], dtype=float),
            "close": np.asarray(candles["close"], dtype=float),
        },
        index=source_index,
    )
    expected = int(round(15.0 / source_timeframe_minutes))
    m15 = frame.resample("15min", label="right", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), bars=("close", "size"),
    )
    m15 = m15.loc[m15["bars"] == expected]
    if m15.empty:
        empty_bool = np.zeros(size, dtype=np.bool_)
        return {
            "madness_range_state": empty_bool,
            "madness_trend_state": ~empty_bool,
            "madness_large_swing": empty_bool.copy(),
            "madness_short_cycle": empty_bool.copy(),
            "madness_cycle_hours": np.full(size, np.nan),
            "madness_swing_pct": np.full(size, np.nan),
            "madness_short_cycle_streak": np.zeros(size, dtype=np.int32),
        }

    st = supertrend(
        {key: m15[key].to_numpy(dtype=float) for key in ("open", "high", "low", "close")},
        period=settings.supertrend_period,
        multiplier=settings.supertrend_multiplier,
    )
    native = _cycle_gate_from_flips(m15["close"].to_numpy(float), st["change_signal"], settings)
    m15_ns = m15.index.as_unit("ns").asi8
    source_completion = source_index.as_unit("ns").asi8 + int(round(source_timeframe_minutes * 60_000_000_000.0))
    indices = np.searchsorted(m15_ns, source_completion, side="right") - 1
    valid = indices >= 0
    first_visible = valid & (indices != np.r_[-2, indices[:-1]])

    def align(values: np.ndarray, fill: Any) -> np.ndarray:
        output = np.full(size, fill, dtype=values.dtype)
        output[valid] = values[indices[valid]]
        return output

    blocked = align(native["blocked"], False).astype(np.bool_)
    large_swing = align(native["large_swing"], False).astype(np.bool_) & first_visible
    short_cycle = align(native["short_cycle"], False).astype(np.bool_) & first_visible
    return {
        "madness_range_state": blocked,
        "madness_trend_state": ~blocked,
        "madness_large_swing": large_swing,
        "madness_short_cycle": short_cycle,
        "madness_cycle_hours": align(native["cycle_hours"], np.nan).astype(np.float64),
        "madness_swing_pct": align(native["swing_pct"], np.nan).astype(np.float64),
        "madness_short_cycle_streak": align(native["short_cycle_streak"], 0).astype(np.int32),
    }


class MadnessFeatures(TypedDict):
    jst: npt.NDArray[Any]
    m5_flip_count: npt.NDArray[np.int32]
    m15_flip_count: npt.NDArray[np.int32]
    efficiency: npt.NDArray[np.float64]
    direction_ratio: npt.NDArray[np.float64]
    bars_since_m5_flip: npt.NDArray[np.int32]
    timeframe_agreement: npt.NDArray[np.bool_]
    valid: npt.NDArray[np.bool_]


class MadnessResult(MadnessFeatures):
    muddy: npt.NDArray[np.bool_]
    mud_score: npt.NDArray[np.float64]


def _validate_candles(candles: dict[str, Any], name: str) -> int:
    missing = {"jst", "open", "high", "low", "close"}.difference(candles)
    if missing:
        raise KeyError(f"{name} is missing: {', '.join(sorted(missing))}")
    size = len(candles["jst"])
    if size == 0:
        raise ValueError(f"{name} must not be empty")
    if any(len(candles[key]) != size for key in ("open", "high", "low", "close")):
        raise ValueError(f"{name} candle array lengths must match")
    return size


def _times_ns(values: Any, name: str) -> npt.NDArray[np.int64]:
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(values, dtype=object), format="mixed"))
    if index.hasnans:
        raise ValueError(f"{name} contains an invalid timestamp")
    out = index.as_unit("ns").asi8
    if np.any(np.diff(out) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return out


def _window_metrics(
    times_ns: npt.NDArray[np.int64],
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    hours: float,
    max_gap_minutes: float,
    *,
    future: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Calculate path efficiency, direction/range ratio, and balance."""

    size = len(close)
    efficiency = np.full(size, np.nan)
    direction_ratio = np.full(size, np.nan)
    balance = np.full(size, np.nan)
    valid = np.zeros(size, dtype=np.bool_)
    span = int(round(hours * 3_600_000_000_000.0))
    tolerance = int(round(max_gap_minutes * 60_000_000_000.0))
    gaps = np.r_[False, np.diff(times_ns) > tolerance]
    gap_total = np.cumsum(gaps.astype(np.int64))

    for index in range(size):
        target = times_ns[index] + span if future else times_ns[index] - span
        if future:
            end = int(np.searchsorted(times_ns, target, side="right") - 1)
            begin = index
            complete = end > begin and target - times_ns[end] <= tolerance
        else:
            begin = int(np.searchsorted(times_ns, target, side="left"))
            end = index
            complete = end > begin and times_ns[begin] - target <= tolerance
        if not complete or gap_total[end] - gap_total[begin] > 0:
            continue
        segment = close[begin : end + 1]
        path = float(np.abs(np.diff(segment)).sum())
        width = float(np.max(high[begin : end + 1]) - np.min(low[begin : end + 1]))
        displacement = abs(float(close[end] - close[begin]))
        up = max(0.0, float(np.max(high[begin : end + 1]) - close[begin]))
        down = max(0.0, float(close[begin] - np.min(low[begin : end + 1])))
        efficiency[index] = displacement / path if path > 0.0 else 0.0
        direction_ratio[index] = displacement / width if width > 0.0 else 0.0
        larger = max(up, down)
        balance[index] = min(up, down) / larger if larger > 0.0 else 1.0
        valid[index] = True
    return efficiency, direction_ratio, balance, valid


def price_regime_oracle(
    m5_candles: dict[str, Any], param: OracleParam | None = None
) -> dict[str, np.ndarray]:
    """Create ex-post MUDDY/TREND labels from the following four hours."""

    settings = param or OracleParam()
    _validate_candles(m5_candles, "m5_candles")
    times = _times_ns(m5_candles["jst"], "m5 jst")
    high = np.asarray(m5_candles["high"], dtype=float)
    low = np.asarray(m5_candles["low"], dtype=float)
    close = np.asarray(m5_candles["close"], dtype=float)
    efficiency, direction_ratio, balance, valid = _window_metrics(
        times, high, low, close, settings.horizon_hours, settings.max_gap_minutes, future=True
    )
    regime = np.full(len(close), PriceRegime.UNCERTAIN, dtype=np.int8)
    muddy = (
        valid
        & (efficiency <= settings.muddy_efficiency_max)
        & (direction_ratio <= settings.muddy_direction_ratio_max)
        & (balance >= settings.muddy_balance_min)
    )
    trend = (
        valid
        & (efficiency >= settings.trend_efficiency_min)
        & (direction_ratio >= settings.trend_direction_ratio_min)
    )
    regime[muddy] = PriceRegime.MUDDY
    regime[trend] = PriceRegime.TREND
    return {
        "jst": np.asarray(m5_candles["jst"], dtype=object),
        "regime": regime,
        "future_efficiency": efficiency,
        "future_direction_ratio": direction_ratio,
        "future_balance": balance,
        "valid": valid,
    }


def _rolling_flip_count(
    times: npt.NDArray[np.int64], signal: npt.ArrayLike, hours: float
) -> npt.NDArray[np.int32]:
    values = np.asarray(signal, dtype=np.int32)
    cumulative = np.cumsum(np.insert(values, 0, 0), dtype=np.int64)
    span = int(round(hours * 3_600_000_000_000.0))
    begin = np.searchsorted(times, times - span, side="left")
    end = np.arange(len(times)) + 1
    return (cumulative[end] - cumulative[begin]).astype(np.int32)


def _bars_since(signal: npt.ArrayLike) -> npt.NDArray[np.int32]:
    output = np.zeros(len(signal), dtype=np.int32)
    age = 0
    for index, changed in enumerate(np.asarray(signal, dtype=np.bool_)):
        age = 0 if changed else age + 1
        output[index] = age
    return output


def _align_backward(source_times: np.ndarray, values: np.ndarray, target_times: np.ndarray, fill: Any) -> np.ndarray:
    indices = np.searchsorted(source_times, target_times, side="right") - 1
    output = np.full(len(target_times), fill, dtype=values.dtype)
    valid = indices >= 0
    output[valid] = values[indices[valid]]
    return output


def madness_features(
    m5_candles: dict[str, Any],
    m15_candles: dict[str, Any],
    param: MadnessParam | None = None,
) -> MadnessFeatures:
    """Calculate causal four-hour M5/M15 reversal and path features."""

    settings = param or MadnessParam()
    _validate_candles(m5_candles, "m5_candles")
    _validate_candles(m15_candles, "m15_candles")
    m5_times = _times_ns(m5_candles["jst"], "m5 jst")
    m15_times = _times_ns(m15_candles["jst"], "m15 jst")
    kwargs = {"period": settings.supertrend_period, "multiplier": settings.supertrend_multiplier}
    m5_st = supertrend(m5_candles, **kwargs)
    m15_st = supertrend(m15_candles, **kwargs)
    m5_flips = _rolling_flip_count(m5_times, m5_st["change_signal"], settings.lookback_hours)
    m15_native = _rolling_flip_count(m15_times, m15_st["change_signal"], settings.lookback_hours)
    m15_flips = _align_backward(m15_times, m15_native, m5_times, 0).astype(np.int32)
    m15_trend = _align_backward(m15_times, m15_st["trend"], m5_times, 0).astype(np.int8)
    efficiency, direction_ratio, _, valid = _window_metrics(
        m5_times,
        np.asarray(m5_candles["high"], dtype=float),
        np.asarray(m5_candles["low"], dtype=float),
        np.asarray(m5_candles["close"], dtype=float),
        settings.lookback_hours,
        settings.max_gap_minutes,
        future=False,
    )
    valid &= np.isfinite(m5_st["atr"]) & (m15_flips >= 0)
    return {
        "jst": np.asarray(m5_candles["jst"], dtype=object),
        "m5_flip_count": m5_flips,
        "m15_flip_count": m15_flips,
        "efficiency": efficiency,
        "direction_ratio": direction_ratio,
        "bars_since_m5_flip": _bars_since(m5_st["change_signal"]),
        "timeframe_agreement": (m5_st["trend"] == m15_trend) & (m15_trend != 0),
        "valid": valid,
    }


def classify_madness(features: MadnessFeatures, param: MadnessParam) -> MadnessResult:
    """Apply an interpretable muddy-range rule to precomputed features."""

    muddy = (
        features["valid"]
        & (features["m5_flip_count"] >= param.m5_flip_min)
        & (features["m15_flip_count"] >= param.m15_flip_min)
        & (features["efficiency"] <= param.efficiency_max)
        & (features["direction_ratio"] <= param.direction_ratio_max)
    )
    flip_score = 0.5 * np.minimum(features["m5_flip_count"] / max(param.m5_flip_min, 1), 2.0)
    flip_score += 0.5 * np.minimum(features["m15_flip_count"] / max(param.m15_flip_min, 1), 2.0)
    path_score = 1.0 - np.clip(features["efficiency"], 0.0, 1.0)
    mud_score = np.where(features["valid"], (flip_score + path_score) / 3.0, np.nan)
    result: MadnessResult = dict(features)  # type: ignore[assignment]
    result["muddy"] = muddy
    result["mud_score"] = mud_score
    return result


def optimize_madness(
    features: MadnessFeatures,
    oracle_regime: npt.ArrayLike,
    base: MadnessParam | None = None,
) -> tuple[MadnessParam, dict[str, float]]:
    """Fit rule thresholds to non-UNCERTAIN oracle labels by balanced accuracy."""

    settings = base or MadnessParam()
    labels = np.asarray(oracle_regime, dtype=np.int8)
    selected = features["valid"] & np.isin(labels, (PriceRegime.MUDDY, PriceRegime.TREND))
    if np.count_nonzero(selected) < 2 or len(np.unique(labels[selected])) < 2:
        raise ValueError("training data must contain both MUDDY and TREND labels")
    truth = labels[selected] == PriceRegime.MUDDY
    best: tuple[float, float, tuple[int, int, float, float], dict[str, float]] | None = None
    for m5_min, m15_min, efficiency_max, direction_max in product(
        range(1, 7), range(0, 4), np.arange(0.15, 0.61, 0.05), np.arange(0.20, 0.71, 0.05)
    ):
        prediction = (
            (features["m5_flip_count"][selected] >= m5_min)
            & (features["m15_flip_count"][selected] >= m15_min)
            & (features["efficiency"][selected] <= efficiency_max)
            & (features["direction_ratio"][selected] <= direction_max)
        )
        tp = int(np.count_nonzero(prediction & truth))
        tn = int(np.count_nonzero(~prediction & ~truth))
        fp = int(np.count_nonzero(prediction & ~truth))
        fn = int(np.count_nonzero(~prediction & truth))
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced = (recall + specificity) / 2.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        metrics = {
            "balanced_accuracy": balanced,
            "muddy_recall": recall,
            "muddy_precision": precision,
            "trend_specificity": specificity,
            "training_rows": float(np.count_nonzero(selected)),
        }
        candidate = (balanced, -fp / len(truth), (m5_min, m15_min, float(efficiency_max), float(direction_max)), metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    m5_min, m15_min, efficiency_max, direction_max = best[2]
    fitted = MadnessParam(
        lookback_hours=settings.lookback_hours,
        supertrend_period=settings.supertrend_period,
        supertrend_multiplier=settings.supertrend_multiplier,
        m5_flip_min=m5_min,
        m15_flip_min=m15_min,
        efficiency_max=efficiency_max,
        direction_ratio_max=direction_max,
        max_gap_minutes=settings.max_gap_minutes,
    )
    return fitted, best[3]


class Madness:
    def __init__(self, param: MadnessParam | None = None):
        self.param = param or MadnessParam()

    def calc(self, m5_candles: dict[str, Any], m15_candles: dict[str, Any]) -> MadnessResult:
        return classify_madness(madness_features(m5_candles, m15_candles, self.param), self.param)


__all__ = [
    "Madness",
    "MadnessFeatures",
    "MadnessGateParam",
    "MadnessGateResult",
    "MadnessParam",
    "MadnessResult",
    "OracleParam",
    "PriceRegime",
    "classify_madness",
    "madness_entry_gate",
    "madness_features",
    "optimize_madness",
    "price_regime_oracle",
]
