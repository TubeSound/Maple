import unittest

import numpy as np

from libs.sr_channel import SRChannelConfig, _channels, _pivot, calc


def fixture():
    n = 53
    high, low, close = np.full(n, 104.), np.full(n, 90.), np.full(n, 100.)
    high[[10, 30]] = 105
    low[[20, 40]] = 89
    close[50], high[50] = 106, 107
    low[51] = 88
    close[52], low[52] = 88, 87
    return {"open": np.full(n, 100.), "high": high, "low": low, "close": close}


class SRChannelTest(unittest.TestCase):
    def test_array_dictionary_contract_and_input_immutability(self):
        data = fixture()
        before = {key: value.copy() for key, value in data.items()}
        result = calc(data, SRChannelConfig(max_channels=3))
        for key, value in result.items():
            self.assertIsInstance(value, np.ndarray, key)
            self.assertEqual(value.shape, (53, 3) if key in ("upper", "lower", "strength", "state") else (53,))
        self.assertEqual(result["state"].dtype, np.dtype("int8"))
        self.assertEqual(result["upper"].dtype, np.dtype("float64"))
        self.assertEqual(result["updated"].dtype, np.dtype("bool"))
        for key in data:
            np.testing.assert_array_equal(before[key], data[key])

    def test_confirmation_delay_and_equal_extremes(self):
        values = np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1.])
        self.assertTrue(np.isnan(_pivot(values, 8, 4, True)))
        self.assertEqual(_pivot(values, 9, 4, True), 5)
        self.assertEqual(_pivot(-values, 9, 4, False), -5)

    def test_width_and_score_hand_calculated(self):
        bands, scores = _channels([105, 105, 89, 89], 1, np.array([105, 105, 104]),
                                  np.array([89, 90, 89]), 1)
        np.testing.assert_array_equal(bands[:2], [[105, 105], [89, 89]])
        np.testing.assert_array_equal(scores[:2], [42, 42])

    def test_contact_count_once_and_inclusive_edges(self):
        bands, scores = _channels([100, 101], 1, np.array([101, 110]), np.array([100, 100]), 1)
        np.testing.assert_array_equal(bands[0], [101, 100])
        self.assertEqual(scores[0], 42)  # first candle counts once, second via low
        self.assertEqual(np.count_nonzero(scores), 1)

    def test_strength_threshold_includes_contacts(self):
        bands, _ = _channels([100], 1, np.full(20, 100.), np.full(20, 99.), 2)
        self.assertEqual(bands[0, 0], 100)  # 1 pivot*20 + 20 touches reaches 40

    def test_high_priority_on_simultaneous_confirmation(self):
        data = {"open": np.full(12, 100.), "close": np.full(12, 100.),
                "high": np.full(12, 101.), "low": np.full(12, 99.)}
        data["high"][5], data["low"][5] = 110, 90
        result = calc(data, SRChannelConfig(pivot_period=4))
        self.assertEqual(result["pivot_high"][9], 110)
        self.assertEqual(result["pivot_low"][9], 90)
        self.assertEqual(result["upper"][9, 0], 110)
        self.assertTrue(np.isnan(result["upper"][9, 1]))

    def test_future_append_does_not_change_history(self):
        data = fixture()
        cfg = SRChannelConfig(pivot_period=4)
        prefix = calc({k: v[:45] for k, v in data.items()}, cfg)
        full = calc(data, cfg)
        for key in prefix:
            np.testing.assert_array_equal(prefix[key], full[key][:45], err_msg=key)

    def test_alerts_and_colors(self):
        result = calc(fixture(), SRChannelConfig(pivot_period=4))
        self.assertTrue(result["resistance_broken"][50])
        self.assertTrue(result["support_broken"][52])
        # A zero-width 105 line becomes support after the upward break;
        # returning below it is also a support break in the original Pine.
        self.assertTrue(result["support_broken"][51])
        self.assertEqual(result["state"][49, 1], -1)
        self.assertEqual(result["state"][50, 1], 1)
        self.assertEqual(result["state"][52, 0], -1)

    def test_channels_hold_between_pivots(self):
        result = calc(fixture(), SRChannelConfig(pivot_period=4))
        for i in range(1, 53):
            if not result["updated"][i]:
                np.testing.assert_array_equal(result["upper"][i], result["upper"][i - 1])
                np.testing.assert_array_equal(result["strength"][i], result["strength"][i - 1])

    def test_body_source_ignores_wick_pivot_but_width_uses_wicks(self):
        data = fixture()
        result = calc(data, SRChannelConfig(pivot_period=4, source="Close/Open"))
        self.assertTrue(np.isnan(result["pivot_high"][14]))
        self.assertEqual(result["channel_width"][14], (105 - 90) * .05)

    def test_stale_band_is_not_removed_without_a_new_pivot(self):
        data = {"open": np.full(140, 100.), "close": np.full(140, 100.),
                "high": np.full(140, 101.), "low": np.full(140, 99.)}
        data["high"][5] = 110
        result = calc(data, SRChannelConfig(pivot_period=4, loopback=100))
        self.assertEqual(result["upper"][-1, 0], 110)
        self.assertFalse(result["updated"][-1])

    def test_width_uses_exactly_last_300_bars(self):
        data = {"open": np.full(301, 100.), "close": np.full(301, 100.),
                "high": np.full(301, 101.), "low": np.full(301, 99.)}
        data["high"][0] = 200
        result = calc(data)
        self.assertAlmostEqual(result["channel_width"][299], (200 - 99) * .05)
        self.assertAlmostEqual(result["channel_width"][300], (101 - 99) * .05)

    def test_empty_and_bad_input(self):
        result = calc({key: [] for key in ("open", "high", "low", "close")})
        self.assertEqual(result["upper"].shape, (0, 6))
        data = fixture()
        data["high"][0] = np.nan
        with self.assertRaises(ValueError):
            calc(data)
        with self.assertRaises(ValueError):
            SRChannelConfig(pivot_period=0)
        with self.assertRaises(ValueError):
            calc({"open": [1], "high": [2], "low": [1], "close": []})

    def test_optional_moving_averages(self):
        data = {key: np.arange(1, 11, dtype=float) for key in ("open", "high", "low", "close")}
        result = calc(data, SRChannelConfig(ma1_enabled=True, ma1_length=3,
                                         ma2_enabled=True, ma2_length=3, ma2_type="EMA"))
        self.assertTrue(np.isnan(result["ma1"][1]))
        self.assertEqual(result["ma1"][2], 2)
        self.assertEqual(result["ma2"][0], 1)
        self.assertEqual(result["ma2"][1], 1.5)


if __name__ == "__main__":
    unittest.main()
