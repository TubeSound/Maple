# SPDX-License-Identifier: MPL-2.0
"""Offline OracleRegime labels for the Venus/Mars strategy.

This module intentionally looks into the future.  Its output is suitable for
research labels only and must never be consumed directly by live trading.

The MVP implements the design document's ``Continuous`` mode: Mars BASE and
Mars ALT are each simulated once over the full candle sequence.  For every
bar, their realized cumulative-gain increments over ``(t, t + H]`` are
compared for each requested horizon.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from datetime import datetime
from typing import Any, Final, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd


FloatArray = npt.NDArray[np.float64]
RegimeArray = npt.NDArray[np.int8]

REGIME_STOP: Final[int] = 0
REGIME_BASE: Final[int] = 1
REGIME_ALT: Final[int] = 2
REGIME_UNCERTAIN: Final[int] = 3

REGIME_NAMES: Final[dict[int, str]] = {
    REGIME_STOP: "STOP",
    REGIME_BASE: "BASE",
    REGIME_ALT: "ALT",
    REGIME_UNCERTAIN: "UNCERTAIN",
}

DEFAULT_HORIZONS: Final[tuple[float, ...]] = (1.0, 3.0, 6.0, 12.0, 24.0)
MODE_CONTINUOUS: Final[str] = "continuous"
EXECUTION_STRATEGY: Final[str] = "strategy"
EXECUTION_MATTERHORN: Final[str] = "matterhorn"
SCORE_GAIN: Final[str] = "gain"
SCORE_PROFIT: Final[str] = "profit"


class OracleRegimeResult(TypedDict):
    """Stable, nested portion of :meth:`OracleRegime.calc` output.

    ``calc`` also adds flattened keys such as ``base_future_gain_1h`` for the
    charting-oriented output proposed in the design document.
    """

    jst: npt.NDArray[Any]
    base_cumulative_gain: FloatArray
    alt_cumulative_gain: FloatArray
    base_future_gain: dict[float, FloatArray]
    alt_future_gain: dict[float, FloatArray]
    oracle_regime: dict[float, RegimeArray]


def _as_finite_real(value: object, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _normalize_horizons(horizons: Sequence[Real]) -> tuple[float, ...]:
    if isinstance(horizons, (str, bytes)):
        raise TypeError("horizons must be a sequence of hours")

    normalized: list[float] = []
    for value in horizons:
        hours = _as_finite_real(value, "horizon")
        if hours <= 0.0:
            raise ValueError("horizons must contain only positive hours")
        if hours in normalized:
            raise ValueError(f"duplicate horizon: {hours:g}h")
        normalized.append(hours)
    if not normalized:
        raise ValueError("horizons must not be empty")
    return tuple(normalized)


def _timestamps_ns(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1:
        raise ValueError("jst must be one-dimensional")
    if len(raw) == 0:
        raise ValueError("candles must contain at least one row")

    try:
        timestamps = pd.DatetimeIndex(pd.to_datetime(raw, format="mixed"))
    except (TypeError, ValueError) as exc:
        raise ValueError("jst contains an invalid timestamp") from exc
    if timestamps.hasnans:
        raise ValueError("jst contains an invalid timestamp")

    # pandas 3 may preserve the input timestamp unit (for example us), while
    # the horizon arithmetic below is explicitly in nanoseconds.
    nanoseconds = timestamps.as_unit("ns").asi8
    differences = np.diff(nanoseconds)
    if np.any(differences < 0):
        raise ValueError("jst must be sorted in ascending order")
    if np.any(differences == 0):
        raise ValueError("jst must not contain duplicates")
    return nanoseconds


def calculate_future_gain(
    timestamps: npt.ArrayLike,
    cumulative_gain: npt.ArrayLike,
    horizon_hours: Real,
) -> FloatArray:
    """Return realized gain over ``(t, t + horizon]`` for every timestamp.

    The horizon is measured in wall-clock hours rather than candle count, so
    market closures and missing bars do not silently turn a 1-hour label into
    a longer interval.  Rows without a complete future horizon are ``NaN``.
    """

    times_ns = _timestamps_ns(timestamps)
    cumulative = np.asarray(cumulative_gain, dtype=np.float64)
    if cumulative.ndim != 1:
        raise ValueError("cumulative_gain must be one-dimensional")
    if len(cumulative) != len(times_ns):
        raise ValueError("jst and cumulative_gain lengths must match")
    if not np.isfinite(cumulative).all():
        raise ValueError("cumulative_gain must contain only finite values")

    hours = _as_finite_real(horizon_hours, "horizon_hours")
    if hours <= 0.0:
        raise ValueError("horizon_hours must be positive")
    horizon_ns = int(round(hours * 60.0 * 60.0 * 1_000_000_000.0))
    if horizon_ns <= 0:
        raise ValueError("horizon_hours is too small")
    if int(times_ns[-1]) > np.iinfo(np.int64).max - horizon_ns:
        raise ValueError("horizon_hours causes timestamp overflow")

    target_ns = times_ns + horizon_ns
    endpoint = np.searchsorted(times_ns, target_ns, side="right") - 1
    complete = target_ns <= times_ns[-1]
    output = np.full(len(times_ns), np.nan, dtype=np.float64)
    output[complete] = cumulative[endpoint[complete]] - cumulative[complete]
    return output


def classify_oracle_regime(
    base_future_gain: npt.ArrayLike,
    alt_future_gain: npt.ArrayLike,
    *,
    minimum_gain: Real = 0.0,
    selection_margin: Real = 0.0,
    stop_threshold: Real = 0.0,
) -> RegimeArray:
    """Classify BASE, ALT, STOP, or UNCERTAIN from two future gains.

    STOP takes precedence when both candidates are at or below
    ``stop_threshold``.  This makes an intentionally raised stop threshold a
    conservative veto.  Missing/non-finite future gains remain UNCERTAIN.
    """

    base = np.asarray(base_future_gain, dtype=np.float64)
    alt = np.asarray(alt_future_gain, dtype=np.float64)
    if base.ndim != 1 or alt.ndim != 1:
        raise ValueError("future gain arrays must be one-dimensional")
    if len(base) != len(alt):
        raise ValueError("future gain array lengths must match")

    minimum = _as_finite_real(minimum_gain, "minimum_gain")
    margin = _as_finite_real(selection_margin, "selection_margin")
    stop = _as_finite_real(stop_threshold, "stop_threshold")
    if margin < 0.0:
        raise ValueError("selection_margin must be non-negative")

    regime = np.full(len(base), REGIME_UNCERTAIN, dtype=np.int8)
    valid = np.isfinite(base) & np.isfinite(alt)
    stopped = valid & (np.maximum(base, alt) <= stop)
    regime[stopped] = REGIME_STOP

    selectable = valid & ~stopped
    base_selected = (
        selectable
        & (base > minimum)
        & ((base - alt) > margin)
    )
    alt_selected = (
        selectable
        & (alt > minimum)
        & ((alt - base) > margin)
    )
    regime[base_selected] = REGIME_BASE
    regime[alt_selected] = REGIME_ALT
    return regime


def _horizon_suffix(hours: float) -> str:
    text = f"{hours:g}".replace(".", "_")
    return f"{text}h"


def _load_mars_dependencies() -> tuple[type[Any], Any]:
    """Load Mars lazily so label-only helpers stay lightweight."""

    try:
        from strategy.venus.evaluate import evaluate_from_data
        from strategy.venus.mars import MarsParam
    except ImportError:
        # Supports importing this file as ``src.libs.oracle_regime`` from the
        # repository root in addition to the project's usual ``src`` cwd.
        from src.strategy.venus.evaluate import evaluate_from_data
        from src.strategy.venus.mars import MarsParam
    return MarsParam, evaluate_from_data


class OracleRegime:
    """Generate offline future-performance labels for Mars BASE and ALT.

    Parameters are accepted either as ``MarsParam`` instances or mappings
    suitable for constructing one.  Numeric horizons are expressed in hours.
    Only Continuous mode is part of this MVP.
    """

    def __init__(
        self,
        base_params: Any,
        alt_params: Any,
        horizons: Sequence[Real] = DEFAULT_HORIZONS,
        minimum_gain: Real = 0.0,
        selection_margin: Real = 0.0,
        *,
        stop_threshold: Real = 0.0,
        symbol: str = "JP225",
        mode: str = MODE_CONTINUOUS,
        execution_mode: str = EXECUTION_STRATEGY,
        position_max: int | None = None,
        volume: Real = 1.0,
        score: str = SCORE_PROFIT,
        simulation_begin: datetime | None = None,
    ) -> None:
        mars_param_type, _ = _load_mars_dependencies()
        self.base_params = self._coerce_param(
            base_params, "base_params", mars_param_type
        )
        self.alt_params = self._coerce_param(
            alt_params, "alt_params", mars_param_type
        )
        if self.base_params.timeframe != self.alt_params.timeframe:
            raise ValueError("BASE and ALT timeframes must match")

        self.horizons = _normalize_horizons(horizons)
        self.minimum_gain = _as_finite_real(minimum_gain, "minimum_gain")
        self.selection_margin = _as_finite_real(
            selection_margin, "selection_margin"
        )
        self.stop_threshold = _as_finite_real(
            stop_threshold, "stop_threshold"
        )
        if self.selection_margin < 0.0:
            raise ValueError("selection_margin must be non-negative")

        self.symbol = str(symbol).strip().upper()
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        self.mode = str(mode).strip().lower()
        if self.mode != MODE_CONTINUOUS:
            raise NotImplementedError(
                "OracleRegime MVP supports only mode='continuous'"
            )
        self.execution_mode = str(execution_mode).strip().lower()
        if self.execution_mode not in (
            EXECUTION_STRATEGY,
            EXECUTION_MATTERHORN,
        ):
            raise ValueError(
                "execution_mode must be 'strategy' or 'matterhorn'"
            )
        if position_max is not None and (
            isinstance(position_max, bool)
            or not isinstance(position_max, int)
            or position_max <= 0
        ):
            raise ValueError("position_max must be a positive integer")
        self.position_max = position_max
        self.volume = _as_finite_real(volume, "volume")
        if self.volume <= 0.0:
            raise ValueError("volume must be positive")
        self.score = str(score).strip().lower()
        if self.score not in (SCORE_GAIN, SCORE_PROFIT):
            raise ValueError("score must be 'gain' or 'profit'")
        if simulation_begin is not None and not isinstance(
            simulation_begin, datetime
        ):
            raise TypeError("simulation_begin must be a datetime or None")
        self.simulation_begin = simulation_begin

    @staticmethod
    def _coerce_param(value: Any, name: str, param_type: type[Any]) -> Any:
        if isinstance(value, param_type):
            return value
        if isinstance(value, Mapping):
            try:
                return param_type(**dict(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {name}: {exc}") from exc
        raise TypeError(f"{name} must be a MarsParam or mapping")

    def _simulate_continuous(
        self,
        candles: dict[str, Any],
        params: Any,
    ) -> FloatArray:
        _, evaluate_from_data = _load_mars_dependencies()
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(np.asarray(candles["jst"], dtype=object), format="mixed")
        )
        begin = (
            timestamps[0].to_pydatetime()
            if self.simulation_begin is None
            else self.simulation_begin
        )
        end = timestamps[-1].to_pydatetime()
        _, profit_curve, _, _ = evaluate_from_data(
            self.symbol,
            params,
            candles,
            begin,
            end,
            include_profit_curve=True,
            execution_mode=self.execution_mode,
            position_max=self.position_max,
            volume=self.volume,
        )
        score_column = f"total_{self.score}"
        if score_column not in profit_curve:
            raise RuntimeError(
                f"Mars profit curve does not contain {score_column}"
            )
        cumulative = profit_curve[score_column].to_numpy(dtype=np.float64)
        if len(cumulative) != len(timestamps):
            raise RuntimeError("Mars profit curve length does not match candles")
        if not np.isfinite(cumulative).all():
            raise RuntimeError("Mars profit curve contains non-finite gains")
        return cumulative

    def calc(self, candles: dict[str, Any]) -> OracleRegimeResult:
        """Run BASE/ALT continuously and build labels for every horizon.

        The returned dictionary exposes both nested horizon dictionaries and
        flattened chart-friendly fields.  Labels near the end of the input,
        where a full future horizon is unavailable, are UNCERTAIN.
        """

        if not isinstance(candles, Mapping):
            raise TypeError("candles must be a mapping of candle arrays")
        required = {"jst", "open", "high", "low", "close"}
        missing = required.difference(candles)
        if missing:
            raise KeyError(f"candles must contain: {', '.join(sorted(missing))}")

        jst = np.asarray(candles["jst"], dtype=object)
        _timestamps_ns(jst)
        size = len(jst)
        for name in ("open", "high", "low", "close"):
            values = np.asarray(candles[name])
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if len(values) != size:
                raise ValueError("candle array lengths must match")

        candle_dict = dict(candles)
        base_cumulative = self._simulate_continuous(
            candle_dict, self.base_params
        )
        alt_cumulative = self._simulate_continuous(
            candle_dict, self.alt_params
        )

        base_future: dict[float, FloatArray] = {}
        alt_future: dict[float, FloatArray] = {}
        regimes: dict[float, RegimeArray] = {}
        result: dict[str, Any] = {
            "jst": jst.copy(),
            "base_cumulative_gain": base_cumulative,
            "alt_cumulative_gain": alt_cumulative,
            "base_future_gain": base_future,
            "alt_future_gain": alt_future,
            "oracle_regime": regimes,
        }

        for horizon in self.horizons:
            base_gain = calculate_future_gain(jst, base_cumulative, horizon)
            alt_gain = calculate_future_gain(jst, alt_cumulative, horizon)
            regime = classify_oracle_regime(
                base_gain,
                alt_gain,
                minimum_gain=self.minimum_gain,
                selection_margin=self.selection_margin,
                stop_threshold=self.stop_threshold,
            )
            base_future[horizon] = base_gain
            alt_future[horizon] = alt_gain
            regimes[horizon] = regime

            suffix = _horizon_suffix(horizon)
            result[f"base_future_gain_{suffix}"] = base_gain
            result[f"alt_future_gain_{suffix}"] = alt_gain
            result[f"oracle_regime_{suffix}"] = regime

        return result  # type: ignore[return-value]


__all__ = [
    "DEFAULT_HORIZONS",
    "EXECUTION_MATTERHORN",
    "EXECUTION_STRATEGY",
    "MODE_CONTINUOUS",
    "OracleRegime",
    "OracleRegimeResult",
    "REGIME_ALT",
    "REGIME_BASE",
    "REGIME_NAMES",
    "REGIME_STOP",
    "REGIME_UNCERTAIN",
    "SCORE_GAIN",
    "SCORE_PROFIT",
    "calculate_future_gain",
    "classify_oracle_regime",
]
