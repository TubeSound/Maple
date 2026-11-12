import unittest

import numpy as np

from libs.atr_regime import (
    AtrRegimeParam,
    RegimeEvent,
    VolatilityPhase,
    VolatilityTier,
    _state_engine,
    atr_volatility_regime,
    rolling_percent_rank,
    true_range,
    wilder_rma,
)


class AtrRegimeTest(unittest.TestCase):
    def test_horizon_profiles_match_pine(self):
        intraday = AtrRegimeParam(horizon="Intraday")
        swing = AtrRegimeParam(horizon="Swing")
        long_term = AtrRegimeParam(horizon="Long-Term Hold")
        custom = AtrRegimeParam(
            horizon="Custom",
            atr_period=36,
            lookback=250,
            smoothing=7,
        )

        self.assertEqual(
            (
                intraday.effective_atr_period,
                intraday.effective_lookback,
                intraday.effective_smoothing,
                intraday.tier_hysteresis_up,
                intraday.tier_hysteresis_down,
            ),
            (14, 100, 3, 1, 2),
        )
        self.assertEqual(
            (
                swing.effective_atr_period,
                swing.effective_lookback,
                swing.effective_smoothing,
                swing.tier_hysteresis_up,
                swing.tier_hysteresis_down,
            ),
            (14, 200, 5, 1, 3),
        )
        self.assertEqual(
            (
                long_term.effective_atr_period,
                long_term.effective_lookback,
                long_term.effective_smoothing,
                long_term.tier_hysteresis_up,
                long_term.tier_hysteresis_down,
            ),
            (20, 300, 5, 2, 3),
        )
        self.assertEqual(
            (
                custom.effective_atr_period,
                custom.effective_lookback,
                custom.effective_smoothing,
            ),
            (36, 250, 7),
        )

    def test_true_range_and_wilder_rma(self):
        high = np.array([11.0, 12.0, 13.0, 14.0])
        low = np.array([9.0, 10.0, 11.0, 12.0])
        close = np.array([10.0, 11.0, 12.0, 13.0])

        tr = true_range(high, low, close)
        atr = wilder_rma(tr, 2)

        np.testing.assert_allclose(tr, [2.0, 2.0, 2.0, 2.0])
        np.testing.assert_allclose(
            atr,
            [np.nan, 2.0, 2.0, 2.0],
            equal_nan=True,
        )

    def test_percent_rank_uses_preceding_window(self):
        result = rolling_percent_rank(
            np.array([1.0, 2.0, 0.0, 3.0, 2.0]),
            2,
        )

        np.testing.assert_allclose(
            result,
            [np.nan, np.nan, 0.0, 100.0, 50.0],
            equal_nan=True,
        )

    def test_tier_hysteresis_moves_up_fast_and_down_slow(self):
        param = AtrRegimeParam(
            horizon="custom",
            lookback=1,
            smoothing=1,
        )
        rank = np.array([np.nan, 10.0, 80.0, 10.0, 10.0, 10.0])
        result = _state_engine(rank, rank, param)

        self.assertEqual(
            result["tier"].tolist(),
            [
                VolatilityTier.WARMUP,
                VolatilityTier.COMPRESSION,
                VolatilityTier.ELEVATED,
                VolatilityTier.ELEVATED,
                VolatilityTier.ELEVATED,
                VolatilityTier.COMPRESSION,
            ],
        )

    def test_extreme_touch_and_cooling_are_confirmed_later(self):
        param = AtrRegimeParam(
            horizon="custom",
            lookback=1,
            smoothing=1,
        )
        rank = np.array([np.nan, 20.0, 60.0, 90.0, 80.0, 70.0])
        result = _state_engine(rank, rank, param)

        self.assertTrue(result["rank_peak_confirmed"][4])
        self.assertEqual(result["phase"][4], VolatilityPhase.TOPPING)
        self.assertTrue(result["extreme_peak"][3])
        self.assertTrue(result["extreme_touch"][4])
        self.assertTrue(result["cooling"][5])
        self.assertEqual(result["last_event"][5], RegimeEvent.COOLING)

    def test_loaded_compression_releases_within_velocity_window(self):
        param = AtrRegimeParam(
            horizon="custom",
            lookback=1,
            smoothing=1,
        )
        rank = np.array([np.nan, *([10.0] * 10), 30.0, 55.0])
        result = _state_engine(rank, rank, param)

        self.assertEqual(result["bars_in_tier"][10], 10)
        self.assertFalse(result["compression_release"][11])
        self.assertTrue(result["compression_release"][12])
        self.assertEqual(
            result["last_event"][12],
            RegimeEvent.COMPRESSION_RELEASE,
        )

    def test_public_api_returns_aligned_numpy_arrays(self):
        size = 40
        close = 100.0 + np.sin(np.arange(size) / 3.0)
        candles = {
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
        }
        result = atr_volatility_regime(
            candles,
            AtrRegimeParam(
                horizon="custom",
                atr_period=3,
                lookback=5,
                smoothing=2,
            ),
        )

        self.assertTrue(all(len(values) == size for values in result.values()))
        self.assertTrue(np.isfinite(result["atr_percent"][4:]).all())
        self.assertTrue(result["warm"].any())


if __name__ == "__main__":
    unittest.main()
