from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import numpy as np

from libs.madness import (
    MadnessGateParam,
    MadnessParam,
    PriceRegime,
    _align_backward,
    _cycle_gate_from_flips,
    _rolling_flip_count,
    classify_madness,
    optimize_madness,
    price_regime_oracle,
)


def candles(minutes: int, close: list[float]) -> dict[str, object]:
    values = np.asarray(close, dtype=float)
    return {
        "jst": [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes * i) for i in range(len(values))],
        "open": values.copy(),
        "high": values + 0.5,
        "low": values - 0.5,
        "close": values.copy(),
    }


class MadnessTest(unittest.TestCase):
    def test_large_swing_gate_is_causal_and_holds_for_market_bars(self) -> None:
        close = np.full(8, 100.0)
        close[3:] = 104.0
        flips = np.zeros(8, dtype=bool)
        flips[[1, 3]] = True
        result = _cycle_gate_from_flips(
            close,
            flips,
            MadnessGateParam(
                swing_threshold_pct=3.0,
                short_cycle_streak=99,
                large_swing_hold_hours=0.5,
                short_cycle_hold_hours=0.0,
            ),
        )

        np.testing.assert_array_equal(result["large_swing"], [0, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(result["blocked"], [0, 0, 0, 1, 1, 0, 0, 0])

    def test_consecutive_short_cycles_trigger_gate(self) -> None:
        close = np.full(9, 100.0)
        flips = np.zeros(9, dtype=bool)
        flips[[1, 3, 5]] = True
        result = _cycle_gate_from_flips(
            close,
            flips,
            MadnessGateParam(
                swing_threshold_pct=99.0,
                short_cycle_hours=1.0,
                short_cycle_streak=2,
                large_swing_hold_hours=0.0,
                short_cycle_hold_hours=0.75,
            ),
        )

        np.testing.assert_array_equal(result["blocked"], [0, 0, 0, 0, 0, 1, 1, 1, 0])
        self.assertEqual(int(result["short_cycle_streak"][5]), 2)

    def test_higher_timeframe_alignment_never_uses_future(self) -> None:
        actual = _align_backward(
            np.array([15, 30, 45]), np.array([1, 2, 3]), np.array([14, 15, 29, 30]), -1
        )
        np.testing.assert_array_equal(actual, [-1, 1, 1, 2])

    def test_rolling_flip_count_is_four_hour_trailing_only(self) -> None:
        times = np.arange(6, dtype=np.int64) * 3_600_000_000_000
        actual = _rolling_flip_count(times, [1, 0, 1, 0, 1, 0], 2.0)
        np.testing.assert_array_equal(actual, [1, 1, 2, 1, 2, 1])

    def test_oracle_labels_future_oscillation_as_muddy(self) -> None:
        close = [100.0 + (1.0 if index % 2 else -1.0) for index in range(120)]
        result = price_regime_oracle(candles(5, close))
        self.assertEqual(int(result["regime"][20]), PriceRegime.MUDDY)

    def test_oracle_labels_directional_path_as_trend(self) -> None:
        result = price_regime_oracle(candles(5, list(np.linspace(100.0, 200.0, 120))))
        self.assertEqual(int(result["regime"][20]), PriceRegime.TREND)

    def test_classifier_requires_reversals_and_low_efficiency(self) -> None:
        features = {
            "jst": np.array([0, 1], dtype=object),
            "m5_flip_count": np.array([4, 4], dtype=np.int32),
            "m15_flip_count": np.array([2, 2], dtype=np.int32),
            "efficiency": np.array([0.2, 0.8]),
            "direction_ratio": np.array([0.3, 0.8]),
            "bars_since_m5_flip": np.array([1, 1], dtype=np.int32),
            "timeframe_agreement": np.array([False, False]),
            "valid": np.array([True, True]),
        }
        result = classify_madness(features, MadnessParam())
        np.testing.assert_array_equal(result["muddy"], [True, False])

    def test_optimizer_finds_separating_rule(self) -> None:
        features = {
            "jst": np.arange(6, dtype=object),
            "m5_flip_count": np.array([4, 5, 3, 0, 1, 1], dtype=np.int32),
            "m15_flip_count": np.array([2, 2, 1, 0, 0, 1], dtype=np.int32),
            "efficiency": np.array([0.1, 0.2, 0.25, 0.8, 0.7, 0.6]),
            "direction_ratio": np.array([0.2, 0.3, 0.3, 0.8, 0.7, 0.6]),
            "bars_since_m5_flip": np.ones(6, dtype=np.int32),
            "timeframe_agreement": np.zeros(6, dtype=np.bool_),
            "valid": np.ones(6, dtype=np.bool_),
        }
        labels = np.array([1, 1, 1, 2, 2, 2], dtype=np.int8)
        fitted, metrics = optimize_madness(features, labels)
        result = classify_madness(features, fitted)
        np.testing.assert_array_equal(result["muddy"], [True, True, True, False, False, False])
        self.assertEqual(metrics["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
