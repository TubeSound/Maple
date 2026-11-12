import unittest

import numpy as np
import pandas as pd

from libs.sr_channel import SRChannelConfig
from libs.sr_channel_multi import HOURS, calc, intersections, select_window, window_start


def frame(freq="h", start="2026-08-01", periods=1800):
    times = pd.date_range(start, freq=freq, periods=periods, tz="Asia/Tokyo")
    value = 100 + np.sin(np.arange(periods) * .22) * 5
    return {"jst": times.to_numpy(dtype="datetime64[ns]"), "open": value,
            "close": value + .1, "high": value + 1, "low": value - 1}


def times(data):
    return pd.DatetimeIndex(data["jst"]).tz_localize("UTC").tz_convert("Asia/Tokyo")


def frames():
    return {f"{tf}_{key}": array for tf, hours in HOURS.items()
            for key, array in frame(freq=f"{hours}h", periods=1800 // hours).items()}


class MultiChannelTest(unittest.TestCase):
    def assertArraysEqual(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for key in left:
            self.assertIsInstance(left[key], np.ndarray, key)
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)

    def test_calendar_month_and_completed_boundary(self):
        cutoff = pd.Timestamp("2026-03-31 06:00", tz="Asia/Tokyo")
        selected = select_window(frame(start="2026-02-01", freq="4h", periods=400), "H4", cutoff)
        start = window_start("H4", cutoff)
        self.assertEqual(start, pd.Timestamp("2026-02-28 06:00", tz="Asia/Tokyo"))
        self.assertTrue((times(selected) >= start).all())
        self.assertTrue((times(selected) + pd.Timedelta(hours=4) <= cutoff).all())

    def test_calendar_month_crosses_year_and_leap_day(self):
        self.assertEqual(window_start("H4", "2026-01-31 06:00").isoformat(), "2025-12-31T06:00:00+09:00")
        self.assertEqual(window_start("H4", "2024-03-31 06:00").isoformat(), "2024-02-29T06:00:00+09:00")

    def test_five_days_includes_start_and_excludes_partial_and_future(self):
        data = frame(start="2026-09-25", periods=250)
        selected = select_window(data, "H1", "2026-10-03 12:30")
        self.assertEqual(times(selected)[0].hour, 13)
        self.assertEqual(times(selected)[-1], pd.Timestamp("2026-10-03 11:00", tz="Asia/Tokyo"))
        exact = select_window(data, "H1", "2026-10-03 12:00")
        self.assertEqual(times(exact)[0], pd.Timestamp("2026-09-28 12:00", tz="Asia/Tokyo"))
        self.assertEqual(len(exact["close"]), 120)

    def test_daily_keeps_history_and_tz_equivalence(self):
        data = frame(freq="24h", periods=70)
        jst = select_window(data, "D1", "2026-10-03 09:00")
        utc = select_window(data, "D1", "2026-10-03T00:00Z")
        numpy_time = select_window(data, "D1", np.datetime64("2026-10-03T00:00"))
        self.assertIsNone(window_start("D1", "2026-10-03"))
        self.assertEqual(jst["jst"][0], data["jst"][0])
        self.assertArraysEqual(jst, utc)
        self.assertArraysEqual(jst, numpy_time)

    def test_exact_overlap_does_not_bridge_disjoint_timeframes(self):
        channels = {"id": np.array(["D1:1", "H4:1", "H1:1"]), "timeframe": np.array(["D1", "H4", "H1"]),
                    "lower": np.array([100., 105., 115.]), "upper": np.array([110., 120., 125.])}
        result = intersections(channels, 112)
        np.testing.assert_array_equal(result["lower"], [105, 115])
        np.testing.assert_array_equal(result["upper"], [110, 120])
        np.testing.assert_array_equal(result["timeframes"], [[True, True, False], [False, True, True]])
        np.testing.assert_array_equal(result["channel_mask"], [[True, True, False], [False, True, True]])
        np.testing.assert_array_equal(result["state"], [1, -1])

    def test_out_of_window_and_future_data_do_not_change_combined_channels(self):
        candles = frames()
        cutoff = pd.Timestamp("2026-10-03 12:00", tz="Asia/Tokyo")
        original = calc(candles, cutoff)
        for tf in HOURS:
            data = {key: candles[f"{tf}_{key}"] for key in ("jst", "open", "high", "low", "close")}
            stamp = times(data)
            outside = stamp + pd.Timedelta(hours=HOURS[tf]) > cutoff
            start = window_start(tf, cutoff)
            if start is not None:
                outside |= stamp < start
            for key in ("open", "high", "low", "close"):
                candles[f"{tf}_{key}"][outside] *= 3
        self.assertArraysEqual(original, calc(candles, cutoff))

    def test_empty_window_and_duplicate_timestamps_fail(self):
        with self.assertRaisesRegex(ValueError, "H1: no completed"):
            select_window(frame(start="2026-01-01", periods=50), "H1", "2026-10-03")
        data = frame(start="2026-10-01", periods=40)
        data["jst"][1] = data["jst"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select_window(data, "H1", "2026-10-03")

    def test_array_contract_shapes_dtypes_and_no_input_mutation(self):
        candles = frames()
        before = {k: v.copy() for k, v in candles.items()}
        result = calc(candles, "2026-10-03 12:00", SRChannelConfig(max_channels=3))
        self.assertTrue(all(isinstance(v, np.ndarray) for v in result.values()))
        for tf in HOURS:
            n = len(result[f"{tf}_close"])
            self.assertEqual(result[f"{tf}_upper"].shape, (n, 3))
            self.assertEqual(result[f"{tf}_state"].dtype, np.dtype("int8"))
            self.assertEqual(result[f"{tf}_upper"].dtype, np.dtype("float64"))
            self.assertEqual(result[f"{tf}_updated"].dtype, np.dtype("bool"))
        self.assertEqual(result["as_of"].dtype, np.dtype("datetime64[ns]"))
        self.assertEqual(result["reference_price"].shape, (1,))
        self.assertEqual(result["overlap_timeframes"].shape[1], 3)
        self.assertEqual(result["overlap_channel_mask"].shape[1], len(result["channel_id"]))
        self.assertArraysEqual(before, candles)

    def test_no_channels_have_stable_empty_table_shapes(self):
        candles = frames()
        for tf in HOURS:
            for key in ("open", "high", "low", "close"):
                candles[f"{tf}_{key}"][:] = 100.
        result = calc(candles, "2026-10-03 12:00")
        self.assertEqual(result["channel_upper"].shape, (0,))
        self.assertEqual(result["overlap_timeframes"].shape, (0, 3))
        self.assertEqual(result["overlap_channel_mask"].shape, (0, 0))

    def test_malformed_timestamps_and_arrays_rejected(self):
        data = frame(periods=10)
        data["jst"][0] = np.datetime64("NaT")
        with self.assertRaisesRegex(ValueError, "NaT"):
            select_window(data, "D1", "2026-10-03")
        with self.assertRaisesRegex(ValueError, "same length"):
            select_window(frame() | {"open": np.zeros(1)}, "D1", "2026-10-03")
        with self.assertRaisesRegex(ValueError, "datetime64"):
            select_window(frame() | {"jst": np.array(["2026-09-01"])}, "D1", "2026-10-03")

    def test_window_sorts_unsorted_input(self):
        data = frame(periods=10)
        shuffled = {key: array[::-1] for key, array in data.items()}
        selected = select_window(shuffled, "D1", "2026-10-03")
        self.assertArraysEqual(selected, data)


if __name__ == "__main__":
    unittest.main()
