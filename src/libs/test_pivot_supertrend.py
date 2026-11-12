import unittest

import numpy as np

from libs.pivot_supertrend import PivotSupertrendConfig, calc, measure_waves


def sample():
    close = np.array([10, 13, 11, 10, 8, 9, 11, 20.])
    return {"open": close.copy(), "close": close, "high": close + 1, "low": close - 1}


class PivotSupertrendTest(unittest.TestCase):
    def test_wave_amplitude_pct_uses_segment_start_price(self):
        # Different segment anchors deliberately distinguish start-close
        # normalization from dividing by the current close or the low.
        close = np.array([100., 110., 200., 210., 230.])
        candles = dict(open=close, close=close, high=close+2, low=close-2)
        result = {"supertrend": close, "change_signal": np.array([False, False, True, False, False]),
                  "pivot_high": np.full(5, np.nan), "pivot_low": np.full(5, np.nan)}
        metrics = measure_waves(candles, result)
        np.testing.assert_allclose(metrics["wave_height_pct"], [4., 14., 2., 7., 17.])
        for n in range(1, 6):
            prefix = measure_waves({k: v[:n] for k, v in candles.items()}, {k: v[:n] for k, v in result.items()})
            np.testing.assert_array_equal(prefix["wave_height_pct"], metrics["wave_height_pct"][:n])

    def test_array_dictionary_contract_and_input_immutability(self):
        data = sample()
        before = {key: value.copy() for key, value in data.items()}
        result = calc(data)
        metrics = measure_waves(data, result)
        for key, value in {**result, **metrics}.items():
            self.assertIsInstance(value, np.ndarray, key)
            self.assertEqual(value.shape, data["close"].shape, key)
        self.assertEqual(result["trend"].dtype, np.dtype("int8"))
        self.assertEqual(result["buy_signal"].dtype, np.dtype("bool"))
        self.assertEqual(result["atr"].dtype, np.dtype("float64"))
        self.assertEqual(metrics["completed_wave_count"].dtype, np.dtype("int64"))
        for key in data:
            np.testing.assert_array_equal(before[key], data[key])
        with self.assertRaisesRegex(ValueError, "shape"):
            measure_waves(data, result | {"supertrend": np.zeros(1)})

    def test_hand_calculated_center_and_signals(self):
        r = calc(sample(), PivotSupertrendConfig(1, 1, 1))
        self.assertEqual(r["pivot_high"][2], 14)
        self.assertEqual(r["pivot_low"][5], 7)
        self.assertEqual(r["center"][2], 14)
        self.assertAlmostEqual(r["center"][5], 35 / 3)
        self.assertTrue(r["sell_signal"][3])
        self.assertEqual(r["signal_price"][3], 10)
        self.assertEqual(r["label_price"][3], 16)
        self.assertTrue(r["buy_signal"][7])
        self.assertEqual(r["signal_price"][7], 20)
        self.assertAlmostEqual(r["label_price"][7], 26 / 3)
        self.assertTrue(np.isnan(r["plot_line"][3]))

    def test_atr_sma_seed_and_wilder_update(self):
        r = calc(sample(), PivotSupertrendConfig(1, 1, 3))
        np.testing.assert_allclose(r["true_range"][:4], [2, 4, 3, 2])
        self.assertTrue(np.isnan(r["atr"][1]))
        self.assertEqual(r["atr"][2], 3)
        self.assertAlmostEqual(r["atr"][3], 8 / 3)

    def test_simultaneous_pivots_favor_high_for_center(self):
        d = {"open": np.full(5, 100.), "close": np.full(5, 100.),
             "high": np.array([101, 101, 110, 101, 101.]),
             "low": np.array([99, 99, 90, 99, 99.])}
        r = calc(d, PivotSupertrendConfig(1, 1, 1))
        self.assertEqual(r["pivot_high"][3], 110)
        self.assertEqual(r["pivot_low"][3], 90)
        self.assertEqual(r["center"][3], 110)
        self.assertEqual(r["support"][3], 90)
        self.assertEqual(r["resistance"][3], 110)

    def test_equal_extremes_use_latest_and_wait_for_confirmation(self):
        close = np.array([10, 13, 13, 11, 10.])
        d = {"open": close, "close": close, "high": close + 1, "low": close - 1}
        r = calc(d, PivotSupertrendConfig(1, 1, 1))
        self.assertTrue(np.isnan(r["pivot_high"][2]))
        self.assertEqual(r["pivot_high"][3], 14)

    def test_constant_prices_no_fake_initial_buy(self):
        d = {k: np.full(20, 100.) for k in ("open", "high", "low", "close")}
        r = calc(d)
        self.assertFalse(r["buy_signal"].any())
        self.assertFalse(r["sell_signal"].any())
        self.assertTrue(np.isnan(r["center"]).all())

    def test_prefix_invariant_including_wave_metrics(self):
        close = 100 + np.sin(np.arange(200) * .3) * 10
        d = {"open": close, "close": close, "high": close + 1, "low": close - 1}
        p = {k: v[:100] for k, v in d.items()}
        full, prefix = calc(d), calc(p)
        for k in prefix:
            np.testing.assert_array_equal(prefix[k], full[k][:100], err_msg=k)
        full_metrics, prefix_metrics = measure_waves(d, full), measure_waves(p, prefix)
        for k in prefix_metrics:
            np.testing.assert_array_equal(prefix_metrics[k], full_metrics[k][:100], err_msg=k)

    def test_running_wave_uses_only_completed_baseline(self):
        close = np.arange(100, 120, dtype=float)
        d = {"open": close, "close": close, "high": close + 1, "low": close - 1}
        r = {"supertrend": close.copy(), "change_signal": np.zeros(20, dtype=bool),
             "pivot_high": np.full(20, np.nan), "pivot_low": np.full(20, np.nan)}
        r["change_signal"][[3, 7, 13]] = True
        r["pivot_high"][[4, 9]] = [105, 110]
        m = measure_waves(d, r)
        self.assertTrue(np.isnan(m["period_ratio"][6]))
        self.assertEqual(m["period_baseline"][7], 4)
        self.assertEqual(m["height_baseline"][7], 5)
        self.assertEqual(m["period_ratio"][11], 1)
        self.assertEqual(m["height_ratio"][11], 6 / 5)
        self.assertEqual(m["period_baseline"][13], 5)
        self.assertEqual(m["pivot_interval_bars"][9], 5)
        self.assertEqual(m["pivot_height"][9], 5)

    def test_empty_and_invalid_inputs(self):
        r = calc({k: [] for k in ("open", "high", "low", "close")})
        self.assertEqual(len(r["trend"]), 0)
        for kwargs in ({"pivot_period": 0}, {"atr_period": 0}, {"atr_factor": float("nan")}):
            with self.assertRaises(ValueError):
                PivotSupertrendConfig(**kwargs)
        d = sample()
        d["close"] = d["close"][:-1]
        with self.assertRaises(ValueError):
            calc(d)


if __name__ == "__main__":
    unittest.main()
