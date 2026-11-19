import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from libs.hmm_state import (PercentileClasses, empirical_markov, filtered_probabilities,
                            fit_hmm, forecast_probabilities, probability_metrics, sequence_lengths)


class HMMTests(unittest.TestCase):
    def test_percentiles_frozen_on_training_data(self):
        values = np.column_stack([np.arange(100), np.arange(100)-50])
        q = PercentileClasses(5).fit(values)
        before = q.edges_.copy()
        cls, symbols = q.transform([[-100, -100], [10000, 10000]])
        np.testing.assert_array_equal(cls, [[0, 0], [4, 4]])
        np.testing.assert_array_equal(symbols, [0, 24])
        np.testing.assert_array_equal(q.edges_, before)

    def test_forward_filter_exact_and_no_future(self):
        m = SimpleNamespace(n_components=2, startprob_=np.array([.6, .4]),
                            transmat_=np.array([[.8,.2],[.3,.7]]),
                            emissionprob_=np.array([[.9,.1],[.2,.8]]))
        expected = m.startprob_ * m.emissionprob_[:, 0]
        expected /= expected.sum()
        out = filtered_probabilities(m, [0,1,0], [3])
        np.testing.assert_allclose(out[0], expected)
        prefix = filtered_probabilities(m, [0,1], [2])
        np.testing.assert_allclose(prefix, out[:2])
        np.testing.assert_allclose(forecast_probabilities(m, out, 2), out @ m.transmat_ @ m.transmat_ @ m.emissionprob_)
        np.testing.assert_allclose(filtered_probabilities(m, [0,1,0], [2,1])[-1], expected)

    def test_no_gap_transitions(self):
        start = datetime(2026,1,1,tzinfo=timezone.utc)
        times = [start+timedelta(minutes=i) for i in (0,1,5,6)]
        lengths = sequence_lengths(times)
        self.assertEqual(lengths, [2,2])
        # No 1->2 transition across the gap, so row 1 remains uniform.
        a = empirical_markov(np.array([0,1,2,2]), lengths, 3)
        np.testing.assert_allclose(a[1], [1/3]*3)

    def test_fit_forecast_and_metrics(self):
        seq = np.tile(np.r_[np.zeros(20,dtype=int), np.ones(20,dtype=int)], 10)
        m, attempts = fit_hmm(seq, [len(seq)], 2, n_states=2, n_iter=30, restarts=1)
        out = forecast_probabilities(m, filtered_probabilities(m, seq, [len(seq)]), 1)
        np.testing.assert_allclose(out.sum(axis=1), 1)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(len(attempts), 1)
        metrics = probability_metrics(out[:-1], seq[1:], seq[:-1])
        self.assertTrue(0 <= metrics["accuracy"] <= 1)
        self.assertTrue(metrics["log_loss"] >= 0)


if __name__ == "__main__":
    unittest.main()
