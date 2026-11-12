import unittest

import numpy as np

from src.libs.rallyseason import rallyseason


PINE_PARAMETERS = {
    "quick_window": 5,
    "fast_window": 12,
    "middle_window": 24,
    "slow_window": 48,
    "rate": 0.7,
    "delta_length": 1,
    "threshold": 0.1,
    "slope_window": 5,
    "timeframe_minutes": 15.0,
}


class RallySeasonTest(unittest.TestCase):
    def test_rising_and_falling_prices_match_pine_conditions(self):
        rising = rallyseason(
            {"close": np.arange(100.0, 260.0)},
            **PINE_PARAMETERS,
        )
        falling = rallyseason(
            {"close": np.arange(260.0, 100.0, -1.0)},
            **PINE_PARAMETERS,
        )

        self.assertTrue(rising["positive"][48:].all())
        self.assertFalse(rising["positive"][:48].any())
        self.assertTrue((rising["signal"][48:] == 1).all())
        self.assertTrue(falling["negative"][48:].all())
        self.assertTrue((falling["signal"][48:] == -1).all())

    def test_price_source_and_sma_match_pine(self):
        close = np.arange(1.0, 81.0)
        result = rallyseason({"close": close}, **PINE_PARAMETERS)

        self.assertAlmostEqual(result["ma_quick"][-1], np.mean(close[-5:]))
        self.assertAlmostEqual(result["ma_fast"][-1], np.mean(close[-12:]))
        self.assertAlmostEqual(result["ma_middle"][-1], np.mean(close[-24:]))
        self.assertAlmostEqual(result["ma_slow"][-1], np.mean(close[-48:]))
        self.assertAlmostEqual(result["gap_ratio"][-1], 0.5)

    def test_threshold_and_ratio_bounds_are_strict(self):
        close = np.arange(100.0, 260.0)
        threshold_boundary = rallyseason(
            {"close": close},
            **(PINE_PARAMETERS | {"threshold": 1.0}),
        )
        ratio_boundary = rallyseason(
            {"close": close},
            **(PINE_PARAMETERS | {"rate": 0.5}),
        )

        self.assertFalse(threshold_boundary["positive"].any())
        self.assertFalse(ratio_boundary["balanced_gap"].any())
        self.assertFalse(ratio_boundary["positive"].any())

    def test_signal_on_only_controls_background(self):
        result = rallyseason(
            {"close": np.arange(100.0, 260.0)},
            **(PINE_PARAMETERS | {"signal_on": False}),
        )

        self.assertTrue((result["signal"][48:] == 1).all())
        self.assertFalse(result["background_signal"].any())

    def test_ma_linear_regression_slopes_are_percent_per_hour(self):
        close = np.arange(100.0, 260.0)
        result = rallyseason({"close": close}, **PINE_PARAMETERS)

        for ma_name, slope_name in [
            ("ma_quick", "ma_quick_slope_pct_per_hour"),
            ("ma_fast", "ma_fast_slope_pct_per_hour"),
            ("ma_middle", "ma_middle_slope_pct_per_hour"),
            ("ma_slow", "ma_slow_slope_pct_per_hour"),
        ]:
            ma_window = result[ma_name][-5:]
            expected = 4.0 / np.mean(ma_window) * 100.0
            self.assertAlmostEqual(result[slope_name][-1], expected)

    def test_slope_threshold_filters_raw_signal(self):
        result = rallyseason(
            {"close": np.arange(100.0, 260.0)},
            **(PINE_PARAMETERS | {"slope_threshold": 100.0}),
        )

        self.assertTrue(result["positive_without_slope"][48:].all())
        self.assertFalse(result["positive"].any())

    def test_validates_input(self):
        with self.assertRaises(KeyError):
            rallyseason({})
        with self.assertRaises(ValueError):
            rallyseason({"close": np.array([1.0, np.nan])})
        with self.assertRaises(ValueError):
            rallyseason({"close": np.array([1.0])}, quick_window=0)
        with self.assertRaises(ValueError):
            rallyseason({"close": np.array([1.0])}, slope_window=1)
        with self.assertRaises(ValueError):
            rallyseason({"close": np.array([1.0])}, timeframe_minutes=0)


if __name__ == "__main__":
    unittest.main()
