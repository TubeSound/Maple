import unittest
from datetime import datetime, timezone

from libs.trade_manager import Param, PositionInfo, TradeManager


class TradeManagerLotTest(unittest.TestCase):
    def test_position_keeps_point_gain_separate_from_lot_profit(self):
        manager = TradeManager("JP225", Param())
        position = manager.create_position(
            1,
            datetime(2026, 8, 20, tzinfo=timezone.utc),
            10.0,
            1,
            39_000.0,
            0.0,
            0.0,
        )
        manager.add_position(position)
        manager.move_to_closed(
            1,
            datetime(2026, 8, 20, 1, tzinfo=timezone.utc),
            39_025.0,
            PositionInfo.EXIT_SIGNAL,
        )

        positions = manager.df_position()
        self.assertEqual(positions.iloc[0]["gain"], 25.0)
        self.assertEqual(positions.iloc[0]["profit"], 250.0)

        curve = manager.profit_curve(
            [
                datetime(2026, 8, 20, tzinfo=timezone.utc),
                datetime(2026, 8, 20, 1, tzinfo=timezone.utc),
            ],
            positions,
        )
        self.assertEqual(curve.iloc[-1]["total_gain"], 25.0)
        self.assertEqual(curve.iloc[-1]["total_profit"], 250.0)

        metrics = manager.performance(positions)
        self.assertEqual(metrics["total_gain"], 25.0)
        self.assertEqual(metrics["total_profit"], 250.0)


if __name__ == "__main__":
    unittest.main()
