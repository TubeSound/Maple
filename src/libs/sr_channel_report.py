"""File outputs for the independent D1/H4/H1 SRchannel windows."""
from dataclasses import asdict
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .sr_channel import SRChannelResult
from .sr_channel_multi import HOURS, calc as calc_multi, price_state, window_start

STATE_NAMES = {-1: "resistance", 0: "in_channel", 1: "support"}
CANDLE_KEYS = ("open", "high", "low", "close", "jst")
RESULT_KEYS = tuple(SRChannelResult.__annotations__)


def candle_dict(frame):
    """DataFrame adapter at the file/report boundary; datetime arrays are UTC."""
    data = {key: frame[key].to_numpy() for key in ("open", "high", "low", "close")}
    data["jst"] = pd.to_datetime(frame["jst"], utc=True).to_numpy(dtype="datetime64[ns]")
    return data


def candle_frame(candles):
    frame = pd.DataFrame(candles)
    frame["jst"] = pd.to_datetime(frame["jst"], utc=True).dt.tz_convert("Asia/Tokyo")
    return frame


def analyze_frames_report(frames, cutoff, config, windows=None):
    """Adapt array-only calculations into serializable snapshots for CLI."""
    inputs = {f"{tf}_{key}": value for tf, frame in frames.items()
              for key, value in candle_dict(frame).items()}
    from .sr_channel_window import cutoff_datetime, subtract_period
    starts = None if windows is None else {tf: subtract_period(cutoff_datetime(cutoff), **window) for tf, window in windows.items()}
    arrays = calc_multi(inputs, cutoff, config, starts=starts)
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize("Asia/Tokyo") if cutoff.tzinfo is None else cutoff.tz_convert("Asia/Tokyo")
    snapshots, selected, results = {}, {}, {}
    channels = [
        {"id": str(arrays["channel_id"][i]), "timeframe": str(arrays["channel_timeframe"][i]),
         "rank": int(arrays["channel_rank"][i]), "upper": float(arrays["channel_upper"][i]),
         "lower": float(arrays["channel_lower"][i]), "strength": float(arrays["channel_strength"][i]),
         "state": STATE_NAMES[int(arrays["channel_state"][i])]}
        for i in range(len(arrays["channel_id"]))]
    for tf in HOURS:
        data = candle_frame({key: arrays[f"{tf}_{key}"] for key in CANDLE_KEYS})
        selected[tf] = data
        result = {key: arrays[f"{tf}_{key}"] for key in RESULT_KEYS}
        results[tf] = result
        start = starts[tf] if starts is not None else window_start(tf, cutoff)
        bands = []
        for band in channels:
            if band["timeframe"] == tf:
                local = dict(band)
                local["state"] = STATE_NAMES[int(price_state(
                    np.array([band["lower"]]), np.array([band["upper"]]), data.close.iloc[-1])[0])]
                bands.append(local)
        snapshots[tf] = {"timeframe": tf, "as_of": cutoff.isoformat(),
                         "window_start": start.isoformat() if start is not None else None,
                         "window": str(windows[tf]) if windows is not None else {"D1": "all_history", "H4": "1_calendar_month", "H1": "5_calendar_days"}[tf],
                         "first_bar_open": data.jst.iloc[0].isoformat(),
                         "last_bar_open": data.jst.iloc[-1].isoformat(), "bars": len(data),
                         "pivot_warmup_complete": len(data) >= 2 * config.pivot_period + 1,
                         "width_300_bars_available": len(data) >= 300,
                         "config": asdict(config), "channels": bands,
                         "channel_width": float(result["channel_width"][-1])}
    overlaps = [
        {"lower": float(arrays["overlap_lower"][i]), "upper": float(arrays["overlap_upper"][i]),
         "timeframes": [tf for k, tf in enumerate(HOURS) if arrays["overlap_timeframes"][i, k]],
         "timeframe_count": int(arrays["overlap_timeframe_count"][i]),
         "channel_ids": sorted(arrays["channel_id"][arrays["overlap_channel_mask"][i]].tolist()),
         "state": STATE_NAMES[int(arrays["overlap_state"][i])]}
        for i in range(len(arrays["overlap_lower"]))]
    combined = {"as_of": cutoff.isoformat(), "reference_price": float(arrays["reference_price"][0]),
                "reference_bar_open": selected["H1"].jst.iloc[-1].isoformat(),
                "snapshots": snapshots, "channels": channels, "overlaps": overlaps}
    return combined, selected, results




def analyze_multi(args, cutoff, config, windows=None):
    frames = {}
    for tf in HOURS:
        paths = sorted((args.data_root / args.symbol / tf).glob(f"{args.symbol}_{tf}_*.csv"))
        if not paths:
            raise ValueError(f"{tf}: no matching candle CSV files")
        frames[tf] = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        frames[tf]["jst"] = pd.to_datetime(frames[tf]["jst"], utc=True).dt.tz_convert("Asia/Tokyo")
    combined, selected, results = analyze_frames_report(frames, cutoff, config, windows)
    combined["symbol"] = args.symbol
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sr_channel_combined.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    pd.DataFrame(combined["channels"], columns=["id", "timeframe", "rank", "upper", "lower", "strength", "state"]).to_csv(
        args.output_dir / "sr_channel_combined.csv", index=False)
    overlap_rows = [{**r, "timeframes": "+".join(r["timeframes"]),
                     "channel_ids": "+".join(r["channel_ids"])} for r in combined["overlaps"]]
    pd.DataFrame(overlap_rows, columns=["lower", "upper", "timeframes", "timeframe_count", "channel_ids", "state"]).to_csv(
        args.output_dir / "sr_channel_overlaps.csv", index=False)
    styles = {"D1": ("-", 1.4), "H4": ("--", 1.0), "H1": (":", .9)}
    colors = {"support": "#16a878", "resistance": "#eb5260", "in_channel": "#888888"}
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), layout="constrained")
    for ax, tf in zip(axes, HOURS):
        data = selected[tf]
        result = results[tf]
        output = args.output_dir / tf
        output.mkdir(parents=True, exist_ok=True)
        (output / "sr_channel.json").write_text(json.dumps(combined["snapshots"][tf], indent=2), encoding="utf-8")
        history = pd.DataFrame({"jst": data.jst, "close": data.close})
        for key in ("pivot_high", "pivot_low", "channel_width", "updated", "not_in_channel",
                    "resistance_broken", "support_broken"):
            history[key] = result[key]
        for k in range(config.max_channels):
            for key in ("upper", "lower", "strength", "state"):
                history[f"{key}_{k + 1}"] = result[key][:, k]
        history.to_csv(output / "sr_channel_history.csv", index=False)
        shown = data.tail(args.plot_bars).reset_index(drop=True)
        for i, row in shown.iterrows():
            color = "#00a879" if row.close >= row.open else "#ec5861"
            ax.vlines(i, row.low, row.high, color=color, linewidth=.6)
            ax.add_patch(Rectangle((i - .3, min(row.open, row.close)), .6,
                                   max(abs(row.close - row.open), .01), facecolor=color))
        low, high = shown.low.min(), shown.high.max()
        padding = max((high - low) * .12, high * .002)
        ylo, yhi = low - padding, high + padding
        # Avoid squeezing the five-day H1 chart to include distant daily bands.
        for band in combined["channels"]:
            if band["upper"] < ylo or band["lower"] > yhi:
                continue
            style, line_width = styles[band["timeframe"]]
            color = colors[band["state"]]
            ax.axhspan(band["lower"], band["upper"], color=color, alpha=.08)
            for edge in (band["lower"], band["upper"]):
                ax.axhline(edge, color=color, linestyle=style, linewidth=line_width)
        for region in combined["overlaps"]:
            ax.axhspan(region["lower"], region["upper"], color=colors[region["state"]], alpha=.18)
        ax.axhline(combined["reference_price"], color="#3b73b9", linewidth=.7, alpha=.6)
        ticks = np.unique(np.linspace(0, len(shown) - 1, min(8, len(shown))).astype(int))
        label_format = "%Y-%m-%d" if tf == "D1" else "%m-%d\n%H:%M"
        ax.set_xticks(ticks, [shown.jst.iloc[i].strftime(label_format) for i in ticks])
        ax.set_xlim(-1, len(shown))
        ax.set_ylim(ylo, yhi)
        ax.grid(alpha=.15)
        ax.set_ylabel("Price")
        snapshot = combined["snapshots"][tf]
        ax.set_title(f'{tf}: {snapshot["window"]} | {len(data)} completed bars | '
                     f'{len(snapshot["channels"])} source channels | all timeframes overlaid', fontsize=10)
        ax.legend(handles=[Line2D([0], [0], color="#555", linestyle=s[0], linewidth=s[1], label=t)
                           for t, s in styles.items()], loc="upper left", ncol=3, fontsize=8)
    fig.suptitle(f"{args.symbol} SRchannel composite | cutoff {cutoff} (JST)\n"
                 "Final bands extended over history; dark fill = exact multi-timeframe overlap", fontsize=13)
    fig.savefig(args.output_dir / "sr_channel_combined.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"as_of": combined["as_of"], "reference_price": combined["reference_price"],
                      "timeframes": {tf: {k: combined["snapshots"][tf][k] for k in ("window", "bars", "channels")}
                                     for tf in HOURS}, "overlaps": combined["overlaps"]}, indent=2))
