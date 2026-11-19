"""Visualize sliding M1 OHLC Wasserstein distances using existing libraries."""
import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
from bokeh.io import save, show
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, Div, Range1d
from bokeh.resources import INLINE

from libs.candle_chart import CandleChart, TimeChart
from libs.data_loader import DataLoader, str2datetime_jst
from libs.wasserstein_state import WassersteinState

ROOT = Path(__file__).resolve().parents[1]


def build_chart(candles, result, symbol, model):
    # CSV timestamps label bar OPEN. Display at close, when OHLC and W1
    # are known, to avoid suggesting that the indicator is available early.
    times = np.array([t + timedelta(minutes=1) for t in candles["jst"]], dtype=object)
    price = CandleChart(f"{symbol} M1 | JST candle close time", None, 500, times)
    price.fig.yaxis.axis_label = "Price"
    price.plot_candle(*(candles[k] for k in ("open", "high", "low", "close")))
    slope = TimeChart(f"{model.window_minutes}-minute OHLC linear regression slope", None, 240,
                      times, ylabel="Slope (%/hour)")
    slope.line(result["slope_pct_per_hour"], color="#2266bb", line_width=2)
    slope.hline(0, "gray")
    pct = TimeChart("Wasserstein-1 / current window mean price", None, 180, times, ylabel="W1 (%)")
    pct.line(result["distance_pct"], color="#8a44aa", line_width=2)
    price.fig.x_range = Range1d(-1, len(times))
    for chart in (slope, pct):
        chart.fig.x_range = price.fig.x_range
    if model.threshold is not None:
        stable = result["state"] == 0
        # Shade contiguous stable runs rather than drawing thousands of spans.
        boundaries = np.diff(np.r_[False, stable, False].astype(int))
        for left, right in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)):
            for chart in (price, pct):
                chart.fig.add_layout(BoxAnnotation(left=float(left - .5), right=float(right - .5),
                                                  fill_color="gold", fill_alpha=.15, line_alpha=0))
    note = Div(text=(f"<b>Window: {model.window_minutes} min; comparison lag: {model.slide_minutes} min; "
                     "4 equally weighted OHLC prices/bar.</b> All times JST, labelled at candle close. "
                     "Gaps and incomplete windows have no distance. "
                     "Slope = OHLC regression price/minute × 60 / window mean price × 100 (%/hour). "
                     "Yellow (optional): W1 ≤ threshold; distribution stability is a range candidate, not a trade signal."))
    return column(note, price.fig, slope.fig, pct.fig, sizing_mode="stretch_width")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", default="2026-11-12", help="JST candle-open start (inclusive)")
    parser.add_argument("--end", default="2026-11-13", help="JST cutoff (exclusive); only closed candles")
    parser.add_argument("--data-root", type=Path, default=ROOT / "MarketData" / "axiory")
    parser.add_argument("--window-minutes", type=int, default=30)
    parser.add_argument("--slide-minutes", type=int, default=30, help="Comparison lag; evaluated every minute")
    parser.add_argument("--threshold", type=float, help="Optional W1 threshold in raw price units")
    parser.add_argument("--output", type=Path, help="HTML output path")
    parser.add_argument("--show", action="store_true", help="Also open browser")
    args = parser.parse_args(argv)
    model = WassersteinState(args.window_minutes, args.slide_minutes, args.threshold)
    start, end = str2datetime_jst(args.start), str2datetime_jst(args.end)
    if end <= start:
        parser.error("--end must be later than --start")
    loader = DataLoader(args.symbol, str(args.data_root))
    candles = loader.load_data_numpy("M1", start - timedelta(minutes=model.window_minutes + model.slide_minutes), end)
    # No incomplete final candle, even for a cutoff containing seconds.
    closed = np.array([t + timedelta(minutes=1) <= end for t in candles["jst"]], dtype=bool)
    candles = {k: v[closed] for k, v in candles.items()}
    result = model.calc(candles)
    visible = np.array([t >= start for t in candles["jst"]], dtype=bool)
    candles = {k: v[visible] for k, v in candles.items()}
    result = {k: v[visible] for k, v in result.items()}
    if len(candles["jst"]) < 2:
        parser.error(f"Not enough M1 data in [{start}, {end}); check --data-root and dates")
    output = args.output or ROOT / "outputs" / f"{args.symbol}_wasserstein_{start:%Y-%m-%d}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    chart = build_chart(candles, result, args.symbol, model)
    save(chart, filename=str(output), resources=INLINE, title=f"{args.symbol} Wasserstein state")
    if args.show:
        show(chart)
    valid = result["distance"][np.isfinite(result["distance"])]
    print(f"Saved {output.resolve()} | bars={len(candles['jst'])}, valid comparisons={len(valid)}")
    if len(valid):
        print("W1 price-unit quantiles (10/50/90%):", np.quantile(valid, [.1, .5, .9]))
    return output


if __name__ == "__main__":
    main()
