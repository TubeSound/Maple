import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libs.w1_plateau import W1PlateauDetector, W1PlateauParams


class PlateauTests(unittest.TestCase):
    def setUp(self):
        self.values = np.r_[np.linspace(.20, .02, 12), np.full(30, .02)]

    def test_decline_flat_confirms_without_rebound(self):
        out = W1PlateauDetector().calc(self.values)
        # First flat window is indices 11..15; three confirmations at 15..17.
        np.testing.assert_array_equal(np.flatnonzero(out["signal"]), [17])
        np.testing.assert_array_equal(out["plateau"][:17], 0)
        np.testing.assert_array_equal(out["plateau"][17:], 1)

    def test_flat_high_and_flat_without_decline_are_not_events(self):
        for values in (np.full(60,.02), np.r_[np.linspace(.4,.2,12),np.full(30,.2)],
                       np.linspace(.08,.001,42)):
            out = W1PlateauDetector().calc(values)
            self.assertEqual(out["signal"].sum(), 0)

    def test_prefix_matches_stream_and_full_result(self):
        detector = W1PlateauDetector()
        streamed = [detector.update(v) for v in self.values]
        full = W1PlateauDetector().calc(self.values)
        for name in full:
            np.testing.assert_allclose(full[name], [r[name] for r in streamed], equal_nan=True)
        for n in (15,17,18,25):
            prefix = W1PlateauDetector().calc(self.values[:n])
            for name in full:
                np.testing.assert_allclose(prefix[name], full[name][:n], equal_nan=True)

    def test_gaps_and_nan_do_not_join_decline_and_plateau(self):
        start = datetime(2026,11,12,tzinfo=timezone.utc)
        times = [start+timedelta(minutes=i) for i in range(len(self.values))]
        times[12:] = [t+timedelta(minutes=10) for t in times[12:]]
        self.assertEqual(W1PlateauDetector().calc(self.values,times)["signal"].sum(), 0)
        broken = self.values.copy()
        broken[12] = np.nan
        self.assertEqual(W1PlateauDetector().calc(broken)["signal"].sum(), 0)

    def test_rearm_requires_new_decline(self):
        # A tiny bump breaks the flat condition but cannot reuse the old peak.
        values = np.r_[self.values, [.055], np.full(12,.02),
                       np.linspace(.20,.02,12),np.full(12,.02)]
        out = W1PlateauDetector().calc(values)
        events = np.flatnonzero(out["signal"])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], 17)
        self.assertGreater(events[1], len(self.values)+13)

    def test_parameters_and_input_validation(self):
        for kwargs in ({"flat_bars":1},{"confirm_bars":0},{"min_drop":0},
                       {"max_flat_slope":-1},{"lookback_bars":True}):
            with self.assertRaises(ValueError):
                W1PlateauParams(**kwargs)
        with self.assertRaises(ValueError):
            W1PlateauDetector().calc([-1.])
        self.assertEqual(len(W1PlateauDetector().calc([])["signal"]), 0)


if __name__ == "__main__":
    unittest.main()
