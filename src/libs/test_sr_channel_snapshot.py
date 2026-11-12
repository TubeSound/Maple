"""Final-band snapshots, calendar lookbacks and completed-bar boundaries."""
import unittest

import numpy as np
import pandas as pd

from libs.sr_channel import SRChannelConfig, calc, calc_sr


def candles(hours=1, start="2024-01-01", periods=1000):
    t = pd.date_range(start, periods=periods, freq=f"{hours}h", tz="Asia/Tokyo")
    c = 100 + np.sin(np.arange(periods) * .23) * 8
    return {"jst": t.to_numpy(dtype="datetime64[ns]"),
            "open": c.copy(), "high": c + 1, "low": c - 1, "close": c.copy()}


def utc(stamp):
    return pd.Timestamp(stamp).to_datetime64()


class SRSnapshotTest(unittest.TestCase):
    def test_default_windows_and_final_values_match_manual_selection(self):
        end = pd.Timestamp("2026-10-04 12:30", tz="Asia/Tokyo")
        cfg = SRChannelConfig(pivot_period=4)
        for tf, hours, offset in (("D1", 24, pd.DateOffset(years=1)),
                                  ("H4", 4, pd.DateOffset(months=1)),
                                  ("H1", 1, pd.Timedelta(days=5))):
            with self.subTest(timeframe=tf):
                data = candles(hours, "2024-09-01", periods=20000 // hours)
                begin = end - offset
                mask = (data["jst"] >= utc(begin)) & (
                    data["jst"] + np.timedelta64(hours, "h") <= utc(end))
                selected = {key: value[mask] for key, value in data.items()}
                expected = calc(selected, cfg)
                actual = calc_sr(data, tf, end, config=cfg)
                valid = np.isfinite(expected["upper"][-1])
                for key in ("upper", "lower", "strength", "state"):
                    np.testing.assert_array_equal(actual[key], expected[key][-1, valid])
                np.testing.assert_array_equal(actual["rank"], np.flatnonzero(valid) + 1)
                self.assertEqual(actual["window_start"][0], utc(begin))
                self.assertEqual(actual["as_of"][0], utc(end))
                self.assertEqual(actual["bars"][0], mask.sum())
                self.assertEqual(actual["reference_price"][0], selected["close"][-1])
                self.assertTrue(all(isinstance(v, np.ndarray) for v in actual.values()))

    def test_custom_duration_and_exact_start(self):
        data = candles(4, "2026-06-01", 800)
        a = calc_sr(data, "H4", "2026-10-04 12:00", months=2, days=3)
        b = calc_sr(data, "H4", "2026-10-04 12:00", start="2026-08-01 12:00")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        c = calc_sr(candles(1, "2026-09-01", 1000), "H1", "2026-10-04 12:00", days=2)
        self.assertEqual(c["bars"][0], 48)

    def test_latest_anchors_to_latest_supplied_bar_end(self):
        data = candles(4, "2026-09-01", 220)
        latest_end = data["jst"].max() + np.timedelta64(4, "h")
        default = calc_sr(data, "H4")
        explicit = calc_sr(data, "H4", latest_end)
        for key in default:
            np.testing.assert_array_equal(default[key], explicit[key])

    def test_month_end_leap_year_and_timezone_equivalence(self):
        data = candles(24, "2022-01-01", 1000)
        a = calc_sr(data, "D1", "2024-02-29 09:00")
        self.assertEqual(a["window_start"][0], np.datetime64("2023-02-28T00:00", "ns"))
        b = calc_sr(data, "D1", "2024-02-29T00:00Z")
        c = calc_sr(data, "D1", np.datetime64("2024-02-29T00:00"))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(a[key], c[key])
        h4 = calc_sr(candles(4, "2026-01-01", 800), "H4", "2026-03-31 06:00")
        self.assertEqual(h4["window_start"][0], np.datetime64("2026-02-27T21:00", "ns"))

    def test_outside_window_and_unfinished_prices_have_no_effect(self):
        data = candles(1, "2026-09-01", 1000)
        original = {key: value.copy() for key, value in data.items()}
        a = calc_sr(data, "H1", "2026-10-04 12:30")
        outside = (data["jst"] < a["window_start"][0]) | (
            data["jst"] + np.timedelta64(1, "h") > a["as_of"][0])
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])
        for key in ("open", "high", "low", "close"):
            data[key][outside] *= 3
        b = calc_sr(data, "H1", "2026-10-04 12:30")
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_boundary_excludes_partial_bar_and_includes_exact_start(self):
        data = candles(1, "2026-10-01", 100)
        result = calc_sr(data, "H1", "2026-10-04 12:30", start="2026-10-03 12:00")
        self.assertEqual(result["bars"][0], 24)
        self.assertEqual(result["reference_bar_open"][0], np.datetime64("2026-10-04T02:00", "ns"))

    def test_empty_bands_and_short_warmup_are_arrays(self):
        data = candles(1, "2026-10-01", 10)
        for key in ("open", "high", "low", "close"):
            data[key][:] = 100.
        result = calc_sr(data, "H1")
        for key in ("upper", "lower", "strength", "state", "rank"):
            self.assertEqual(result[key].shape, (0,))
        self.assertEqual(result["state"].dtype, np.dtype("int8"))
        self.assertEqual(result["rank"].dtype, np.dtype("int64"))
        self.assertEqual(result["bars"][0], 10)

    def test_invalid_periods_ranges_and_timeframes(self):
        data = candles()
        for kwargs in ({"days": 0}, {"years": -1}, {"months": True}, {"days": 1.5},
                       {"start": "2026-01-01", "days": 5},
                       {"start": "2026-10-05"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                calc_sr(data, "H1", "2026-10-04", **kwargs)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            calc_sr(data, "M5")
        with self.assertRaisesRegex(ValueError, "no completed"):
            calc_sr(data, "H1", "2026-10-04")

    def test_bad_times_empty_arrays_and_duplicates(self):
        data = candles(1, "2026-10-01", 100)
        with self.assertRaisesRegex(ValueError, "no completed"):
            calc_sr({k: v[:0] for k, v in data.items()}, "H1")
        with self.assertRaisesRegex(ValueError, "datetime64"):
            calc_sr(data | {"jst": np.array(["2026-10-01"])}, "H1")
        bad = {k: v.copy() for k, v in data.items()}
        bad["jst"][0] = np.datetime64("NaT")
        with self.assertRaisesRegex(ValueError, "NaT"):
            calc_sr(bad, "H1")
        data["jst"][1] = data["jst"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            calc_sr(data, "H1")


if __name__ == "__main__":
    unittest.main()
