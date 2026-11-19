"""Train percentile-coded HMM on chronological data and evaluate future classes."""
import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from libs.data_loader import DataLoader, str2datetime_jst
from libs.hmm_state import (PercentileClasses, empirical_markov, filtered_probabilities,
                            fit_hmm, forecast_probabilities, probability_metrics, sequence_lengths)
from libs.wasserstein_state import WassersteinState

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", default="2025-11-12", help="JST, inclusive")
    p.add_argument("--end", default="2026-11-12", help="JST, exclusive")
    p.add_argument("--train-end", help="JST split; default first 80 percent of calendar interval")
    p.add_argument("--data-root", type=Path, default=ROOT / "MarketData" / "axiory")
    p.add_argument("--window-minutes", type=int, default=30)
    p.add_argument("--slide-minutes", type=int, default=30)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--states", type=int, default=4)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 15, 30])
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "hmm")
    args = p.parse_args(argv)
    start, end = str2datetime_jst(args.start), str2datetime_jst(args.end)
    split = str2datetime_jst(args.train_end) if args.train_end else start + (end-start)*.8
    if not start < split < end or min(args.horizons) < 1:
        p.error("require start < train-end < end and positive horizons")
    model_features = WassersteinState(args.window_minutes, args.slide_minutes)
    loader = DataLoader(args.symbol, str(args.data_root))
    month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while month < end:
        if not Path(loader.file_path("M1", month.year, month.month)).exists():
            p.error(f"Missing month: {month:%Y-%m}; cannot evaluate requested period")
        month = month.replace(year=month.year+1, month=1) if month.month == 12 else month.replace(month=month.month+1)
    print(f"Loading {args.symbol}: {start} to {end}", flush=True)
    candles = loader.load_data_numpy("M1", start, end)
    if len(candles["jst"]) < args.window_minutes + args.slide_minutes:
        p.error("not enough M1 candles")
    if candles["jst"][0] > start + timedelta(days=4) or candles["jst"][-1] < end - timedelta(days=4):
        p.error("data does not cover requested interval endpoints")
    print(f"Calculating features for {len(candles['jst'])} bars", flush=True)
    features = model_features.calc(candles)
    values = np.column_stack([features["distance_pct"], features["slope_pct_per_hour"]])
    closed = np.array([t + timedelta(minutes=1) for t in candles["jst"]], dtype=object)
    valid = np.isfinite(values).all(axis=1) & (closed <= end)
    values, times = values[valid], closed[valid]
    train_mask = times < split
    if train_mask.sum() < max(100, args.bins**2 * args.states):
        p.error("not enough training features")
    classifier = PercentileClasses(args.bins).fit(values[train_mask])
    classes, symbols = classifier.transform(values)
    train = symbols[train_mask]
    train_lengths = sequence_lengths(times[train_mask])
    print(f"Training HMM: {len(train)} observations, {len(train_lengths)} sequences", flush=True)
    hmm, attempts = fit_hmm(train, train_lengths, args.bins**2, args.states,
                            args.iterations, args.restarts, args.seed)
    lengths = sequence_lengths(times)
    filtered = filtered_probabilities(hmm, symbols, lengths)
    markov = empirical_markov(train, train_lengths, args.bins**2)
    frequency = (np.bincount(train, minlength=args.bins**2) + 1).astype(float)
    frequency /= frequency.sum()
    sequence_id = np.repeat(np.arange(len(lengths)), lengths)
    # Remove overlap between the raw data underlying training and test features.
    test_start = split + timedelta(minutes=args.window_minutes + args.slide_minutes)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"symbol": args.symbol, "requested_start": start.isoformat(), "requested_end": end.isoformat(),
              "first_candle_open": candles["jst"][0].isoformat(), "last_candle_open": candles["jst"][-1].isoformat(),
              "train_end": split.isoformat(), "test_start_after_embargo": test_start.isoformat(),
              "window_minutes": args.window_minutes, "slide_minutes": args.slide_minutes,
              "n_bins": args.bins, "n_states": args.states, "raw_bars": len(closed),
              "valid_features": len(times), "training_observations": len(train),
              "training_sequences": len(train_lengths), "initializations": attempts,
              "percentile_classes": classifier.to_dict(), "horizons": {},
              "target": "future (W1 percentile class, signed Slope percentile class), not price returns",
              "inference": "causal forward filtering; no future smoothing or Viterbi",
              "caution": "Overlapping rolling windows create persistence. Hidden state IDs have no intrinsic meaning."}
    report["hidden_state_profiles"] = []
    for state in range(args.states):
        weights = filtered[train_mask, state]
        report["hidden_state_profiles"].append({
            "state": state, "training_probability_mass": float(weights.sum()),
            "weighted_mean_w1_pct": float(np.average(values[train_mask, 0], weights=weights)),
            "weighted_mean_slope_pct_per_hour": float(np.average(values[train_mask, 1], weights=weights))})
    forecasts = {}
    for horizon in sorted(set(args.horizons)):
        probabilities = forecast_probabilities(hmm, filtered, horizon)
        forecasts[horizon] = probabilities
        src = np.arange(max(0, len(times)-horizon))
        dst = src + horizon
        ok = ((times[src] >= test_start) & (sequence_id[src] == sequence_id[dst]))
        src, dst = src[ok], dst[ok]
        if not len(src):
            p.error(f"no test observations for horizon {horizon}")
        actual, current = symbols[dst], symbols[src]
        baseline = np.linalg.matrix_power(markov, horizon)[current]
        evaluation = {
            "hmm": probability_metrics(probabilities[src], actual, current),
            "observable_markov": probability_metrics(baseline, actual, current),
            "training_frequency": probability_metrics(np.broadcast_to(frequency, (len(src), len(frequency))), actual, current),
            "persistence_accuracy": float(np.mean(actual == current)),
            "first_origin": times[src[0]].isoformat(), "last_origin": times[src[-1]].isoformat()}
        report["horizons"][str(horizon)] = evaluation
        with (args.output / f"test_predictions_{horizon}m.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["origin_close_jst", "target_close_jst", "current_w1_class", "current_slope_class",
                             "actual_w1_class", "actual_slope_class", "predicted_w1_class", "predicted_slope_class",
                             "predicted_probability", "filtered_hidden_state"] + [f"p_symbol_{j}" for j in range(args.bins**2)])
            for i, j in zip(src, dst):
                pred = int(probabilities[i].argmax())
                writer.writerow([times[i].isoformat(), times[j].isoformat(), *classes[i], *classes[j],
                                 pred // args.bins, pred % args.bins, probabilities[i, pred],
                                 filtered[i].argmax(), *probabilities[i]])
        print(f"{horizon}m: HMM accuracy={evaluation['hmm']['accuracy']:.4f}, "
              f"persistence={evaluation['persistence_accuracy']:.4f}, "
              f"Markov logloss={evaluation['observable_markov']['log_loss']:.4f}, "
              f"HMM logloss={evaluation['hmm']['log_loss']:.4f}", flush=True)
    latest = {"as_of_candle_close_jst": times[-1].isoformat(), "features": values[-1].tolist(),
              "classes": classes[-1].tolist(), "hidden_state_probabilities": filtered[-1].tolist(),
              "forecast_symbol_probabilities": {str(h): probs[-1].tolist() for h, probs in forecasts.items()}}
    latest["forecast_joint_class_probabilities"] = {
        str(h): probs[-1].reshape(args.bins, args.bins).tolist() for h, probs in forecasts.items()}
    latest["forecast_hidden_state_probabilities"] = {
        str(h): (filtered[-1] @ np.linalg.matrix_power(hmm.transmat_, h)).tolist() for h in forecasts}
    artifact = {"percentile_classes": classifier.to_dict(), "startprob": hmm.startprob_.tolist(),
                "transmat": hmm.transmat_.tolist(), "emissionprob": hmm.emissionprob_.tolist(),
                "window_minutes": args.window_minutes, "slide_minutes": args.slide_minutes,
                "train_end": split.isoformat()}
    write_json(args.output / "model.json", artifact)
    write_json(args.output / "report.json", report)
    write_json(args.output / "latest_forecast.json", latest)
    print(f"Saved {args.output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    main()
