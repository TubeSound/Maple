import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libs.wasserstein_state import WassersteinState


def candles(close):
    c = np.asarray(close, dtype=float)
    start = datetime(2026, 11, 12, tzinfo=timezone(timedelta(hours=9)))
    return dict(jst=np.array([start + timedelta(minutes=i) for i in range(len(c))]),
                open=c.copy(), high=c + .5, low=c - .5, close=c.copy())


class WassersteinTests(unittest.TestCase):
    def test_hourly_ohlc_slope(self):
        # Exactly 2 price units/minute; OHLC offsets average to zero.
        data = candles(1000 + 2 * np.arange(80))
        model = WassersteinState(30, 30)
        out = model.calc(data)
        self.assertTrue(np.isnan(out["slope_pct_per_hour"][:29]).all())
        expected = 2 * 60 * 100 / (1000 + 2 * (np.arange(29, 80) - 14.5))
        np.testing.assert_allclose(out["slope_pct_per_hour"][29:], expected)
        self.assertTrue(np.isnan(out["distance"][:59]).all())
        flat = model.calc(candles([1000.] * 60))
        np.testing.assert_allclose(flat["slope_pct_per_hour"][29:], 0, atol=1e-10)
        down = model.calc(candles(1000 - 2 * np.arange(80)))
        self.assertTrue((down["slope_pct_per_hour"][29:] < 0).all())

    def test_slope_gap_and_future_independence(self):
        data = candles(1000 + np.arange(100))
        model = WassersteinState(30, 30)
        prefix = {k: v[:70] for k, v in data.items()}
        np.testing.assert_allclose(model.calc(prefix)["slope_pct_per_hour"],
                                   model.calc(data)["slope_pct_per_hour"][:70])
        data["jst"][40:] += timedelta(minutes=1)
        slope = model.calc(data)["slope_pct_per_hour"]
        self.assertTrue(np.isnan(slope[40:69]).all())
        self.assertTrue(np.isfinite(slope[69]))

    def test_stationary_and_shift(self):
        steady = WassersteinState(threshold=0).calc(candles([100.] * 25))
        self.assertTrue(np.isnan(steady["distance"][:15]).all())
        np.testing.assert_array_equal(steady["distance"][15:], 0)
        np.testing.assert_array_equal(steady["state"][15:], 0)
        moved = WassersteinState().calc(candles(np.arange(25) + 100))
        np.testing.assert_allclose(moved["distance"][15:], 1)

    def test_matches_cdf_integral_with_overlapping_windows(self):
        # Independent CDF-integral oracle, including changed distribution shape.
        data = candles(np.random.default_rng(42).uniform(90, 110, 40))
        out = WassersteinState(15, 3).calc(data)["distance"]
        prices = np.column_stack([data[k] for k in ("open", "high", "low", "close")])
        for i in range(17, 40):
            a, b = prices[i-17:i-2].ravel(), prices[i-14:i+1].ravel()
            grid = np.unique(np.r_[a, b])
            mids = (grid[1:] + grid[:-1]) / 2
            ca = np.mean(a[:, None] <= mids, axis=0)
            cb = np.mean(b[:, None] <= mids, axis=0)
            self.assertAlmostEqual(out[i], np.sum(np.abs(ca-cb)*np.diff(grid)))

    def test_gap_and_nan_recovery(self):
        data = candles([100.] * 50)
        data["jst"][20:] += timedelta(minutes=1)
        out = WassersteinState().calc(data)["distance"]
        self.assertTrue(np.isnan(out[20:35]).all())
        self.assertEqual(out[35], 0)
        data = candles([100.] * 50)
        data["close"][20] = np.nan
        out = WassersteinState().calc(data)["distance"]
        self.assertTrue(np.isnan(out[20:36]).all())
        self.assertEqual(out[36], 0)

    def test_causal_prefix_and_validation(self):
        data = candles(np.arange(40) + 100)
        model = WassersteinState()
        prefix = {k: v[:25] for k, v in data.items()}
        np.testing.assert_allclose(model.calc(prefix)["distance"], model.calc(data)["distance"][:25])
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                WassersteinState(window_minutes=value)
        self.assertEqual(len(model.calc(candles([]))["distance"]), 0)
        data["jst"][10] = data["jst"][9]
        with self.assertRaises(ValueError):
            model.calc(data)


if __name__ == "__main__":
    unittest.main()
