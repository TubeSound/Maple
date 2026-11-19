"""Percentile-coded (W1 %, slope %/hour) observations and causal HMM forecasts."""
from dataclasses import dataclass

import numpy as np


@dataclass
class PercentileClasses:
    n_bins: int = 5

    def fit(self, features):
        values = np.asarray(features, dtype=float)
        if type(self.n_bins) is not int or self.n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2")
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("features must be a finite (n, 2) matrix")
        if len(values) < self.n_bins:
            raise ValueError("not enough training observations")
        self.edges_ = np.quantile(values, np.arange(1, self.n_bins) / self.n_bins, axis=0).T
        if any(np.any(np.diff(e) <= 0) for e in self.edges_):
            raise ValueError("duplicate percentile edges: use fewer bins or more varied data")
        return self

    def transform(self, features):
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("features must be a finite (n, 2) matrix")
        classes = np.column_stack([np.searchsorted(edge, values[:, j], side="right")
                                   for j, edge in enumerate(self.edges_)])
        symbols = classes[:, 0] * self.n_bins + classes[:, 1]
        return classes, symbols.astype(int)

    def to_dict(self):
        return {"n_bins": self.n_bins, "edges": self.edges_.tolist(),
                "features": ["distance_pct", "slope_pct_per_hour"],
                "symbol_rule": "W1_class * n_bins + slope_class (zero based)"}


def sequence_lengths(times):
    """Split at every missing minute; do not learn transitions across gaps."""
    seconds = np.array([t.timestamp() for t in times])
    if len(seconds) == 0:
        return []
    if np.any(np.diff(seconds) <= 0):
        raise ValueError("timestamps must be unique and increasing")
    cuts = np.flatnonzero(~np.isclose(np.diff(seconds), 60, rtol=0, atol=1e-6)) + 1
    return np.diff(np.r_[0, cuts, len(seconds)]).tolist()


def fit_hmm(symbols, lengths, n_symbols, n_states=4, n_iter=100, restarts=3, seed=42):
    from hmmlearn.hmm import CategoricalHMM

    if n_states < 2 or n_iter < 1 or restarts < 1:
        raise ValueError("n_states >= 2, n_iter >= 1 and restarts >= 1 required")
    observations = np.asarray(symbols, dtype=int).reshape(-1, 1)
    candidates = []
    best, best_score = None, -np.inf
    for attempt in range(restarts):
        model = CategoricalHMM(n_components=n_states, n_features=n_symbols,
                               n_iter=n_iter, tol=.01, random_state=seed + attempt,
                               startprob_prior=1.1, transmat_prior=1.1,
                               emissionprob_prior=1.1, implementation="scaling")
        model.fit(observations, lengths)
        score = model.score(observations, lengths)
        if not np.isfinite(score):
            raise ValueError("nonfinite HMM training likelihood")
        candidates.append({"seed": seed + attempt, "train_log_likelihood": float(score),
                           "iterations": model.monitor_.iter,
                           "hit_iteration_limit": model.monitor_.iter >= n_iter})
        # Select initialization on training likelihood, never test performance.
        if score > best_score:
            best, best_score = model, score
    return best, candidates


def filtered_probabilities(model, symbols, lengths):
    """Forward filtering P(z_t | x_1..x_t); never backward smoothing/Viterbi."""
    symbols = np.asarray(symbols, dtype=int)
    if sum(lengths) != len(symbols) or any(n <= 0 for n in lengths):
        raise ValueError("sequence lengths must sum to the observation count")
    out = np.empty((len(symbols), model.n_components))
    begin = 0
    for length in lengths:
        belief = model.startprob_.copy()
        for i in range(begin, begin + length):
            if i > begin:
                belief = belief @ model.transmat_
            belief = belief * model.emissionprob_[:, symbols[i]]
            mass = belief.sum()
            if mass <= 0 or not np.isfinite(mass):
                raise ValueError("observation has zero or invalid probability")
            belief /= mass
            out[i] = belief
        begin += length
    return out


def forecast_probabilities(model, filtered, horizon=1):
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    return np.asarray(filtered) @ np.linalg.matrix_power(model.transmat_, horizon) @ model.emissionprob_


def empirical_markov(symbols, lengths, n_symbols, pseudocount=1.):
    """Observable-class Markov baseline fitted only to training sequences."""
    counts = np.full((n_symbols, n_symbols), pseudocount, dtype=float)
    begin = 0
    for length in lengths:
        seq = np.asarray(symbols[begin:begin + length], dtype=int)
        np.add.at(counts, (seq[:-1], seq[1:]), 1)
        begin += length
    return counts / counts.sum(axis=1, keepdims=True)


def probability_metrics(probabilities, actual, current):
    probabilities = np.asarray(probabilities)
    actual, current = np.asarray(actual), np.asarray(current)
    if len(actual) == 0:
        return {"count": 0}
    predicted = probabilities.argmax(axis=1)
    truth = probabilities[np.arange(len(actual)), actual]
    changed = actual != current
    return {"count": len(actual), "accuracy": float(np.mean(predicted == actual)),
            "log_loss": float(-np.mean(np.log(np.clip(truth, 1e-15, 1)))),
            "brier": float(np.mean(np.sum(probabilities**2, axis=1) - 2 * truth + 1)),
            "changed_count": int(changed.sum()),
            "changed_accuracy": float(np.mean(predicted[changed] == actual[changed])) if changed.any() else None}
