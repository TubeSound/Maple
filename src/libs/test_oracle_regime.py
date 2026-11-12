import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from libs.oracle_regime import (
    OracleRegime,
    EXECUTION_MATTERHORN,
    REGIME_ALT,
    REGIME_BASE,
    REGIME_STOP,
    REGIME_UNCERTAIN,
    calculate_future_gain,
    classify_oracle_regime,
)


class OracleRegimeTest(unittest.TestCase):
    @staticmethod
    def times(count, *, minutes=60):
        start = datetime(2026, 8, 18, tzinfo=timezone.utc)
        return [start + timedelta(minutes=minutes * index) for index in range(count)]

    def test_future_gain_uses_wall_clock_horizon_and_marks_incomplete_tail(self):
        times = [
            self.times(1)[0],
            self.times(1)[0] + timedelta(minutes=30),
            self.times(1)[0] + timedelta(hours=1),
            self.times(1)[0] + timedelta(hours=2),
        ]

        result = calculate_future_gain(times, [0.0, 5.0, 7.0, 10.0], 1)

        np.testing.assert_allclose(result[:3], [7.0, 2.0, 3.0])
        self.assertTrue(np.isnan(result[3]))

    def test_classification_applies_stop_margin_and_unknown_rules(self):
        result = classify_oracle_regime(
            [1.0, 0.4, -1.0, 1.0, 0.0, np.nan],
            [0.0, 0.3, -2.0, 1.0, 1.0, 0.0],
            minimum_gain=0.2,
            selection_margin=0.15,
        )

        np.testing.assert_array_equal(
            result,
            [
                REGIME_BASE,
                REGIME_UNCERTAIN,
                REGIME_STOP,
                REGIME_UNCERTAIN,
                REGIME_ALT,
                REGIME_UNCERTAIN,
            ],
        )

    def test_raised_stop_threshold_is_a_conservative_veto(self):
        result = classify_oracle_regime(
            [0.5],
            [0.1],
            stop_threshold=0.5,
        )

        self.assertEqual(result[0], REGIME_STOP)

    def test_calc_returns_nested_and_flattened_horizon_outputs(self):
        base_curve = np.array([0.0, 2.0, 1.0, 1.0, 1.0])
        alt_curve = np.array([0.0, 0.0, 3.0, 3.0, 3.0])

        class StubOracle(OracleRegime):
            def _simulate_continuous(self, candles, params):
                if params is self.base_params:
                    return base_curve.copy()
                return alt_curve.copy()

        oracle = StubOracle(
            {"long_middle_window": 90},
            {"long_middle_window": 32},
            horizons=[1],
        )
        candles = {
            "jst": self.times(5),
            "open": np.full(5, 100.0),
            "high": np.full(5, 101.0),
            "low": np.full(5, 99.0),
            "close": np.full(5, 100.0),
        }

        result = oracle.calc(candles)

        expected = [
            REGIME_BASE,
            REGIME_ALT,
            REGIME_STOP,
            REGIME_STOP,
            REGIME_UNCERTAIN,
        ]
        np.testing.assert_array_equal(result["oracle_regime"][1.0], expected)
        self.assertIs(
            result["oracle_regime"][1.0],
            result["oracle_regime_1h"],
        )
        np.testing.assert_allclose(
            result["base_future_gain_1h"][:4],
            [2.0, -1.0, 0.0, 0.0],
        )
        self.assertTrue(np.isnan(result["base_future_gain_1h"][-1]))

    def test_calc_integrates_with_mars_continuous_simulation(self):
        count = 120
        times = self.times(count, minutes=1)
        close = np.full(count, 39_000.0)
        candles = {
            "jst": times,
            "open": close.copy(),
            "high": close + 5.0,
            "low": close - 5.0,
            "close": close,
        }
        oracle = OracleRegime(
            {},
            {"long_middle_window": 32},
            horizons=[0.5],
        )

        result = oracle.calc(candles)

        self.assertTrue((result["oracle_regime"][0.5][:-30] == REGIME_STOP).all())
        self.assertTrue(
            (result["oracle_regime"][0.5][-30:] == REGIME_UNCERTAIN).all()
        )
        np.testing.assert_array_equal(result["base_cumulative_gain"], 0.0)
        np.testing.assert_array_equal(result["alt_cumulative_gain"], 0.0)

    def test_simulation_uses_runtime_conditions_and_lot_weighted_score(self):
        times = self.times(2)
        candles = {
            "jst": times,
            "open": np.full(2, 100.0),
            "high": np.full(2, 101.0),
            "low": np.full(2, 99.0),
            "close": np.full(2, 100.0),
        }
        oracle = OracleRegime(
            {},
            {},
            horizons=[1],
            execution_mode=EXECUTION_MATTERHORN,
            position_max=10,
            volume=10.0,
            score="profit",
        )
        curve = pd.DataFrame(
            {
                "total_gain": [0.0, 2.0],
                "total_profit": [0.0, 20.0],
            }
        )
        with patch(
            "strategy.venus.evaluate.evaluate_from_data",
            return_value=({}, curve, {}, pd.DataFrame()),
        ) as evaluate:
            result = oracle._simulate_continuous(candles, oracle.base_params)

        np.testing.assert_array_equal(result, [0.0, 20.0])
        self.assertEqual(evaluate.call_args.kwargs["execution_mode"], "matterhorn")
        self.assertEqual(evaluate.call_args.kwargs["position_max"], 10)
        self.assertEqual(evaluate.call_args.kwargs["volume"], 10.0)

    def test_simulation_can_start_after_indicator_warmup_data(self):
        times = self.times(3)
        candles = {
            "jst": times,
            "open": np.full(3, 100.0),
            "high": np.full(3, 101.0),
            "low": np.full(3, 99.0),
            "close": np.full(3, 100.0),
        }
        simulation_begin = times[1]
        oracle = OracleRegime(
            {}, {}, horizons=[1], simulation_begin=simulation_begin
        )
        curve = pd.DataFrame(
            {"total_gain": [0.0, 0.0, 0.0], "total_profit": [0.0, 0.0, 0.0]}
        )
        with patch(
            "strategy.venus.evaluate.evaluate_from_data",
            return_value=({}, curve, {}, pd.DataFrame()),
        ) as evaluate:
            oracle._simulate_continuous(candles, oracle.base_params)

        self.assertEqual(evaluate.call_args.args[3], simulation_begin)

    def test_rejects_mismatched_timeframes_and_unsupported_flat_reset(self):
        with self.assertRaisesRegex(ValueError, "timeframes must match"):
            OracleRegime(
                {"timeframe": "M1"},
                {"timeframe": "M5"},
            )
        with self.assertRaisesRegex(NotImplementedError, "continuous"):
            OracleRegime({}, {}, mode="flat_reset")
        with self.assertRaisesRegex(ValueError, "position_max"):
            OracleRegime({}, {}, position_max=0)


if __name__ == "__main__":
    unittest.main()
