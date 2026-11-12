import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from libs.slope_spike import (
    SlopeSpikeParam,
    calculate_ma_slopes,
    classify_spikes_by_fast_slope,
    detect_slope_spikes,
)


class SlopeSpikeTest(unittest.TestCase):
    def setUp(self):
        self.param = SlopeSpikeParam(
            threshold=0.5,
            dead_zone=0.1,
            pivot_window=1,
            burst_window_minutes=12,
            burst_min_spikes=3,
            max_gap_minutes=5,
        )

    @staticmethod
    def times(count, *, start=datetime(2026, 4, 30, tzinfo=timezone.utc)):
        return [start + timedelta(minutes=index) for index in range(count)]

    def test_extracts_positive_and_negative_pivots(self):
        slope = np.zeros(35)
        slope[18] = 1.0
        slope[24] = -1.5

        result = detect_slope_spikes(self.times(len(slope)), slope, self.param)

        self.assertEqual(np.flatnonzero(result["positive_peak"]).tolist(), [18])
        self.assertEqual(np.flatnonzero(result["negative_peak"]).tolist(), [24])
        self.assertEqual(result["spike_direction"][18], 1)
        self.assertEqual(result["spike_direction"][24], -1)
        self.assertGreater(result["spike_metric"][18], self.param.threshold)
        self.assertLess(result["spike_metric"][24], -self.param.threshold)

    def test_ma_periods_and_slope_window_belong_to_spike_param(self):
        param = SlopeSpikeParam(
            quick_window=2,
            quick_energy_window=2,
            fast_window=3,
            middle_window=4,
            slow_window=5,
            slope_window=2,
        )
        result = calculate_ma_slopes(
            np.arange(1.0, 8.0),
            timeframe_minutes=1.0,
            param=param,
        )

        np.testing.assert_allclose(
            result["ma_quick"][1:],
            [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        )
        self.assertTrue(np.isnan(result["ma_slow"][:4]).all())
        self.assertTrue(np.isfinite(result["quick_slope"][2:]).all())
        quick = result["quick_slope"]
        expected = np.sqrt((quick[-2] ** 2 + quick[-1] ** 2) / 2.0)
        self.assertAlmostEqual(result["quick_slope_energy"][-1], expected)

    def test_threshold_candidates_are_reduced_to_pivot_peak(self):
        slope = np.zeros(30)
        slope[18:21] = [-1.0, -2.0, -1.2]

        result = detect_slope_spikes(self.times(len(slope)), slope, self.param)

        self.assertTrue(result["negative_candidate"][18:21].all())
        self.assertEqual(np.flatnonzero(result["negative_peak"]).tolist(), [19])

    def test_three_same_direction_peaks_form_burst(self):
        slope = np.zeros(35)
        slope[[18, 22, 26]] = [-1.0, -1.5, -2.0]

        result = detect_slope_spikes(self.times(len(slope)), slope, self.param)

        self.assertEqual(
            np.flatnonzero(result["negative_peak"]).tolist(),
            [18, 22, 26],
        )
        self.assertTrue(result["negative_burst"][26])
        self.assertTrue(result["burst_start"][26])
        self.assertEqual(result["negative_recent_spike_count"][26], 3)

    def test_gap_resets_recent_peak_count(self):
        first = self.times(20)
        second = self.times(
            4,
            start=datetime(2026, 4, 30, 11, 0, tzinfo=timezone.utc),
        )
        slope = np.zeros(24)
        slope[[16, 18, 21]] = -2.0

        result = detect_slope_spikes(first + second, slope, self.param)

        self.assertEqual(result["negative_recent_spike_count"][21], 1)
        self.assertFalse(result["negative_burst"][21])

    def test_metric_is_the_absolute_thresholded_slope(self):
        slope = np.array([0.1, -0.4, 0.5, -0.6, np.nan])
        result = detect_slope_spikes(self.times(len(slope)), slope, self.param)

        np.testing.assert_allclose(
            result["spike_metric"],
            slope,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            result["spike_magnitude"],
            np.abs(slope),
            equal_nan=True,
        )
        self.assertEqual(
            result["positive_candidate"].tolist(),
            [False, False, True, False, False],
        )
        self.assertEqual(
            result["negative_candidate"].tolist(),
            [False, False, False, True, False],
        )

    def test_fast_slope_trend_and_entry_pivot_classification(self):
        fast = np.array([0.2, 0.2, -0.3, -0.3, 0.05])
        quick = np.array([1.0, -2.0, -1.0, 2.0, 1.0])
        spike_direction = np.array([1, -1, -1, 1, 1])

        result = classify_spikes_by_fast_slope(
            self.times(5),
            fast,
            quick,
            spike_direction,
            param=self.param,
        )

        self.assertEqual(result["trend_direction"].tolist(), [1, 1, -1, -1, 0])
        self.assertEqual(result["trend_peak"].tolist(), [True, False, True, False, False])
        self.assertEqual(
            result["entry_pivot_peak"].tolist(),
            [False, True, False, True, False],
        )
        self.assertEqual(
            result["neutral_peak"].tolist(),
            [False, False, False, False, True],
        )

    def test_counter_trend_pivot_does_not_use_middle_slope(self):
        fast = np.full(3, -0.2)
        quick = np.array([-2.0, -1.0, -2.0])
        spike_direction = np.zeros(3, dtype=np.int8)

        result = classify_spikes_by_fast_slope(
            self.times(3),
            fast,
            quick,
            spike_direction,
            param=self.param,
        )

        self.assertTrue(result["entry_pivot_peak"][1])
        self.assertEqual(result["entry_pivot_direction"][1], 1)

    def test_entry_pivot_does_not_require_spike_threshold(self):
        fast = np.full(5, -0.2)
        quick = np.array([0.0, 0.1, 0.2, 0.1, 0.0])
        spike_direction = np.zeros(5)

        result = classify_spikes_by_fast_slope(
            self.times(5),
            fast,
            quick,
            spike_direction,
            param=self.param,
        )

        self.assertTrue(result["entry_pivot_peak"][2])
        self.assertEqual(result["entry_pivot_direction"][2], 1)


if __name__ == "__main__":
    unittest.main()
