import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import MetaTrader5 as mt5api

from src.libs import metatrader_simulator as simulator_module
from src.libs.data_loader import DataLoader
from src.libs.metatrader_simulator import BUY, SELL, MetaTraderSimulator
from src.libs.trade_manager import Signal


def ticks(*rows):
    return pd.DataFrame(rows, columns=["jst", "bid", "ask"])


class MetaTraderSimulatorStopLossTest(unittest.TestCase):
    def test_zero_stop_loss_is_treated_as_disabled(self):
        simulator = MetaTraderSimulator(
            "JP225",
            "M5",
            ticks=ticks(("2026-03-03 09:20:00+09:00", 57450.0, 57460.0)),
        )

        position = simulator.market_entry(SELL, sl=0)

        self.assertIsNone(position.sl)

    def test_sell_stop_must_be_above_current_ask(self):
        simulator = MetaTraderSimulator(
            "JP225",
            "M5",
            ticks=ticks(("2026-03-03 09:20:00+09:00", 57450.0, 57460.0)),
        )

        with self.assertRaisesRegex(ValueError, "invalid SELL stop loss"):
            simulator.market_entry(SELL, sl=57449.0)

    def test_buy_stop_must_be_below_current_bid(self):
        simulator = MetaTraderSimulator(
            "JP225",
            "M5",
            ticks=ticks(("2026-03-03 09:20:00+09:00", 57450.0, 57460.0)),
        )

        with self.assertRaisesRegex(ValueError, "invalid BUY stop loss"):
            simulator.market_entry(BUY, sl=57450.0)

    def test_sell_stop_gap_fills_at_current_ask(self):
        simulator = MetaTraderSimulator(
            "JP225",
            "M5",
            ticks=ticks(
                ("2026-03-03 09:20:00+09:00", 57450.0, 57460.0),
                ("2026-03-03 09:20:01+09:00", 57480.0, 57490.0),
            ),
        )
        position = simulator.market_entry(SELL, sl=57470.0)

        closed = simulator._process_tick(
            pd.Timestamp("2026-03-03 09:20:01+09:00"), 57480.0, 57490.0
        )

        self.assertEqual(closed, [position])
        self.assertEqual(position.exit_price, 57490.0)
        self.assertEqual(position.gain, -40.0)


class MetaTraderSimulatorDataTest(unittest.TestCase):
    def test_from_period_loads_both_half_month_pickle_files(self):
        with TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            tick_dir = data_root / "JP225" / "TICK"
            tick_dir.mkdir(parents=True)
            pd.DataFrame(
                [("2026-05-01 09:00:00+09:00", 38000.0, 38001.0)],
                columns=["jst", "bid", "ask"],
            ).to_pickle(tick_dir / "JP225_TICK_2026-05-1.pkl")
            pd.DataFrame(
                [("2026-05-20 09:00:00+09:00", 39000.0, 39001.0)],
                columns=["jst", "bid", "ask"],
            ).to_pickle(tick_dir / "JP225_TICK_2026-05-2.pkl")

            loader_factory = lambda symbol: DataLoader(symbol, data_root=str(data_root))
            with patch.object(simulator_module, "DataLoader", side_effect=loader_factory):
                simulator = MetaTraderSimulator.from_period(
                    "JP225",
                    "M1",
                    "2026-05-01 00:00:00+09:00",
                    "2026-05-31 23:59:59+09:00",
                )

        self.assertEqual(len(simulator.ticks), 2)
        self.assertEqual(simulator.ticks["bid"].tolist(), [38000.0, 39000.0])

    def test_entry_uses_signal_domain_for_bid_ask_and_order_side(self):
        simulator = MetaTraderSimulator(
            "JP225",
            "M1",
            ticks=ticks(("2026-05-01 09:00:00+09:00", 38000.0, 38001.0)),
        )

        long_position = simulator.entry(None, Signal.LONG, None, 1.0)[1]
        short_position = simulator.entry(None, Signal.SHORT, None, 1.0)[1]

        self.assertEqual(long_position.order_signal, Signal.LONG)
        self.assertEqual(long_position.entry_price, 38001.0)
        self.assertEqual(short_position.order_signal, Signal.SHORT)
        self.assertEqual(short_position.entry_price, 38000.0)

    def test_internal_sides_do_not_collide_with_mt5_order_types(self):
        simulator = MetaTraderSimulator(
            "JP225",
            "M1",
            ticks=ticks(("2026-05-01 09:00:00+09:00", 38000.0, 38001.0)),
        )

        self.assertNotEqual(BUY, mt5api.ORDER_TYPE_SELL)
        self.assertNotEqual(SELL, mt5api.ORDER_TYPE_BUY)
        with self.assertRaisesRegex(ValueError, "Unsupported side"):
            simulator.market_entry(mt5api.ORDER_TYPE_BUY)
        with self.assertRaisesRegex(ValueError, "Unsupported signal"):
            simulator.current_price(signal=mt5api.ORDER_TYPE_SELL)

    def test_advance_keeps_partial_candle_between_updates(self):
        times = pd.date_range("2026-05-01 09:20:00+09:00", periods=21, freq="s")
        tick_data = pd.DataFrame(
            {
                "jst": times,
                "bid": range(100, 121),
                "ask": range(101, 122),
            }
        )
        simulator = MetaTraderSimulator(
            "JP225", "M1", update_seconds=10, ticks=tick_data
        )

        first = simulator.advance()
        second = simulator.advance()

        self.assertEqual(first.candles.iloc[0]["open"], 100.5)
        self.assertEqual(first.candles.iloc[0]["close"], 110.5)
        self.assertEqual(second.candles.iloc[0]["jst"], first.candles.iloc[0]["jst"])
        self.assertEqual(second.candles.iloc[0]["open"], 100.5)
        self.assertEqual(second.candles.iloc[0]["close"], 120.5)

    def test_buy_stop_gap_fills_at_current_bid(self):
        simulator = MetaTraderSimulator(
            "JP225",
            "M5",
            ticks=ticks(
                ("2026-03-03 09:20:00+09:00", 57450.0, 57460.0),
                ("2026-03-03 09:20:01+09:00", 57420.0, 57430.0),
            ),
        )
        position = simulator.market_entry(BUY, sl=57440.0)

        closed = simulator._process_tick(
            pd.Timestamp("2026-03-03 09:20:01+09:00"), 57420.0, 57430.0
        )

        self.assertEqual(closed, [position])
        self.assertEqual(position.exit_price, 57420.0)
        self.assertEqual(position.gain, -40.0)


if __name__ == "__main__":
    unittest.main()
