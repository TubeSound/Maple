"""Regression checks for support/resistance width and segment-age direction."""
import unittest
import numpy as np

try:
    from .keltner_channel import keltner_channel, support_resistance_indicators
except ImportError:
    from keltner_channel import keltner_channel, support_resistance_indicators


class SupportResistanceIndicatorTests(unittest.TestCase):
    def test_segment_age_and_ratio(self):
        result = support_resistance_indicators(
            [0, 0, 110, 110, 120, 120, 120, 125],
            [0, 90, 90, 95, 95, 95, 100, 105], [100]*8)
        np.testing.assert_allclose(result['band_width'],
                                   [np.nan, np.nan, .20, .15, .25, .25, .20, .20], equal_nan=True)
        np.testing.assert_allclose(result['expansion_direction'],
                                   [np.nan, np.nan, 1, -1, 1, 1, -1, 0], equal_nan=True)

    def test_same_level_does_not_restart_or_require_widening(self):
        result = support_resistance_indicators([0, 110, 105, 105], [90]*4, [100]*4)
        np.testing.assert_allclose(result['expansion_direction'], [np.nan, 1, 1, 1], equal_nan=True)
        self.assertLess(result['band_width'][2], result['band_width'][1])

    def test_simultaneous_start_and_inverted_levels(self):
        result = support_resistance_indicators([90, 90], [110, 110], [100, 200])
        np.testing.assert_allclose(result['band_width'], [.2, .1])
        np.testing.assert_array_equal(result['expansion_direction'], [0, 0])

    def test_unavailable_and_zero_close(self):
        result = support_resistance_indicators([0, 110, np.nan, 120, 120],
                                               [90]*5, [100, 0, 100, 100, np.nan])
        np.testing.assert_allclose(result['band_width'], [np.nan, np.nan, np.nan, .3, np.nan], equal_nan=True)
        np.testing.assert_allclose(result['expansion_direction'], [np.nan, 1, np.nan, 1, 1], equal_nan=True)

    def test_empty_and_shape_validation(self):
        result = support_resistance_indicators([], [], [])
        self.assertEqual(result['band_width'].size, 0)
        self.assertEqual(result['expansion_direction'].size, 0)
        with self.assertRaises(ValueError):
            support_resistance_indicators([1], [], [1])

    def test_calculate_integration_and_causality(self):
        rng = np.random.default_rng(7)
        close = 60000 + np.cumsum(rng.normal(0, 80, 800))
        high, low = close+100, close-100
        full = keltner_channel(high, low, close)
        prefix = keltner_channel(high[:500], low[:500], close[:500])
        valid = (full['support'] != 0) & (full['resistance'] != 0)
        self.assertTrue(valid.any())
        np.testing.assert_allclose(full['band_width'][valid],
            np.abs(full['resistance'][valid]-full['support'][valid])/close[valid])
        for key in ('band_width', 'expansion_direction'):
            np.testing.assert_allclose(full[key][:500], prefix[key], equal_nan=True)
        self.assertTrue(np.isin(full['expansion_direction'][valid], [-1, 0, 1]).all())


if __name__ == '__main__':
    unittest.main()
