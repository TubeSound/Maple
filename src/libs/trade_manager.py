from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Callable, Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
import MetaTrader5 as mt5api


class Param:
    position_max: int = 10
    mt5_sl: float = 0.0
    stop_loss: float = 100.0
class Signal:
    LONG = 1
    SHORT = -1

    BUY = 1
    SELL = -1

    order_types = {
        mt5api.ORDER_TYPE_BUY: 'Market Buy order',
        mt5api.ORDER_TYPE_SELL: 'Market Sell order',
        mt5api.ORDER_TYPE_BUY_LIMIT: 'Buy Limit pending order',
        mt5api.ORDER_TYPE_SELL_LIMIT: 'Sell Limit pending order',
        mt5api.ORDER_TYPE_BUY_STOP: 'Buy Stop pending order',
        mt5api.ORDER_TYPE_SELL_STOP: 'Sell Stop pending order',
        mt5api.ORDER_TYPE_BUY_STOP_LIMIT: 'Upon reaching the order price, a pending Buy Limit order is placed at the StopLimit price',
        mt5api.ORDER_TYPE_SELL_STOP_LIMIT: 'Upon reaching the order price, a pending Sell Limit order is placed at the StopLimit price',
        mt5api.ORDER_TYPE_CLOSE_BY: 'Order to close a position by an opposite one'
    }

    @staticmethod
    def order_type2signal(type):
        if type == mt5api.ORDER_TYPE_BUY:
            return Signal.LONG
        elif type == mt5api.ORDER_TYPE_BUY_LIMIT:
            return Signal.LONG
        elif type == mt5api.ORDER_TYPE_BUY_STOP:
            return Signal.LONG
        elif type == mt5api.ORDER_TYPE_BUY_STOP_LIMIT:
            return Signal.LONG
        elif type == mt5api.ORDER_TYPE_SELL:
            return Signal.SHORT
        elif type == mt5api.ORDER_TYPE_SELL_LIMIT:
            return Signal.SHORT
        elif type == mt5api.ORDER_TYPE_SELL_STOP:
            return Signal.SHORT
        elif type == mt5api.ORDER_TYPE_SELL_STOP_LIMIT:
            return Signal.SHORT
        else:
            return None
class PositionInfo:
    EXIT_SIGNAL =1
    TAKE_PROFIT = 2
    STOP_LOSS = 3
    TRAILING_STOP = 4
    TIME_UP = 5
    LOGICAL_TAKE_PROFIT = 6
    LOGICAL_STOP_LOSS = 7

    close_reason = {
        EXIT_SIGNAL: "Exit Signal",
        TAKE_PROFIT: "Take Profit",
        STOP_LOSS: "Stop Loss",
        TRAILING_STOP: "Trailing Stop",
        TIME_UP: "Time Up",
        LOGICAL_TAKE_PROFIT: "Logical Take Profit",
        LOGICAL_STOP_LOSS: "Logical Stop Loss",
    }



    def __init__(self, symbol, order_type: int, time: datetime, volume, ticket, price, param:Param, sl=None, tp=None):
        self.symbol = symbol
        self.order_type = order_type
        self.order_signal = Signal.order_type2signal(order_type)
        self.volume = volume
        self.ticket = ticket
        self.entry_time = time
        self.entry_price = float(price)
        self.sl = sl
        self.tp = tp
        self.param = param
        self.exit_time = None
        self.exit_price = None
        self.gain = None
        self.profit = None
        
        self.hit_level = 0
        self.profit_max = 0
        self.closed = False
        self.reason = None

    
    def calc_profit(self, price):
        gain = price - self.entry_price
        if self.order_signal == Signal.SHORT:
            gain *= -1.0
        profit = gain * float(self.volume)
        self.gain = gain
        self.profit = profit
        return profit

    
    def desc(self):
        type_str = Signal.order_types[self.order_type]
        s = 'symbol: ' + self.symbol + ' type: ' + type_str + ' volume: ' + str(self.volume) + ' ticket: ' + str(self.ticket)
        return s

    def array(self):
        reason = PositionInfo.close_reason.get(self.reason, str(self.reason))
        data = [
            self.symbol,
            Signal.order_type2signal(self.order_type),
            self.volume,
            self.ticket,
            self.entry_time,
            self.entry_price,
            self.sl,
            self.tp,
            self.exit_time,
            self.exit_price,
            self.gain,
            self.profit,
            self.closed,
            reason,
        ]
        return data

    @staticmethod
    def array_columns():
        return [
            'symbol',
            'type',
            'volume',
            'ticket',
            'entry_time',
            'entry_price',
            'sl',
            'tp',
            'exit_time',
            'exit_price',
            'gain',
            'profit',
            'closed',
            'reason',
        ]

class TradeManager:
    def __init__(self, symbol, param: Param):
        self.symbol = symbol
        self.param = param
        self.positions: Dict[int, PositionInfo] = {}
        self.positions_closed: Dict[int, PositionInfo] = {}

    def exit_all(self, exit_signal, time, price, reason):
        if exit_signal == 0:
           return 
        to_close = []
        for ticket, position in self.positions.items():
            if position.order_signal != exit_signal:
                to_close.append(ticket) 
        self.remove_positions(to_close, time, price, reason)        
        
    def stop_loss_sim(self, time, high, low, open_price=None):
        to_close = []
        prices = []
        for ticket, position in self.positions.items():
            if position.sl == 0:
                continue
            if position.order_signal == Signal.LONG:
                if position.sl >= low: 
                    to_close.append(ticket)
                    price = position.sl
                    if open_price is not None:
                        price = min(price, open_price)
                    prices.append(price)
            elif position.order_signal == Signal.SHORT:
                if position.sl <= high:
                    to_close.append(ticket)
                    price = position.sl
                    if open_price is not None:
                        price = max(price, open_price)
                    prices.append(price)
        for ticket, p in zip(to_close, prices):
            self.remove_positions([ticket], time, p, PositionInfo.STOP_LOSS)     
        return to_close
            
    def time_up_sim(self, time, price, time_up_minutes):
        to_close = {}
        for ticket, position in self.positions.items():
            dt = time - position.entry_time
            if dt > timedelta(minutes=time_up_minutes):
                position.calc_profit(price)
                to_close[ticket] = [time, price, PositionInfo.TIME_UP]
        for ticket, value in to_close.items():
            self.move_to_closed(ticket, value[0], value[1], value[2])
        return to_close
        
    def create_position(self, side:int, time: datetime, volume:float, ticket:int, price: float, tp: float, sl: float):
        if side == 1:
            order = mt5api.ORDER_TYPE_BUY
        elif side == -1:
            order = mt5api.ORDER_TYPE_SELL
        else:
            return None
        pos = PositionInfo(
                            self.symbol, 
                            order,
                            time,
                            volume,
                            ticket,
                            price,
                            self.param,
                            sl,
                            tp,
                           )
        return pos
                        
    def history_df(self):
        df = pd.DataFrame(data=self.pnl_history, columns=['Time', 'Profit(Unreal)', 'Profit(Realized)', 'count', 'win_rate', 'reason'])
        return df

    def add_position(self, position: PositionInfo):
        self.positions[position.ticket] = position

    def move_to_closed(self, ticket, time, price, reason):
        if ticket in self.positions.keys():
            pos = self.positions.pop(ticket)
            pos.exit_time = time
            pos.exit_price = price
            pos.reason = reason
            pos.calc_profit(price)
            pos.closed = True
            self.positions_closed[ticket] = pos
        else:
            print('move_to_closed, No tickt')

    def calc_profit_sum(self):
        current = 0
        win_num = 0
        for ticket, position in self.positions.items():
            if position.profit is not None:
                current += position.profit
                if position.profit > 0:
                    win_num += 1
        count = len(self.positions.items())
        if count == 0:
            win_rate = 0.0
        else:
            win_rate = float(win_num) / float(count)
        closed = 0.0
        for ticket, position in self.positions_closed.items():
            if position.profit is not None:
                closed += position.profit
        return current + closed, current, closed, count, win_rate

    def summary(self):
        out = []
        columns = PositionInfo.array_columns()
        for ticket, pos in list(self.positions.items()) + list(self.positions_closed.items()):
            d = pos.array()
            out.append(d)
        df = pd.DataFrame(data=out, columns=columns)
        return df

    def remove_positions(self, tickets, time, price, reason):
        for ticket in tickets:
            self.move_to_closed(ticket, time, price, reason)

    def open_positions(self):
        return self.positions

    def remove_position_auto(self, real_positions: list[PositionInfo], time, price, reason):
        remove_tickets = []
        for ticket, info in self.positions.items():
            found = False
            for pos in real_positions:
                if pos.ticket == ticket:
                    found = True
                    break
            if not found:
                remove_tickets.append(ticket)
        if len(remove_tickets):
            self.remove_positions(remove_tickets, time, price, reason)
            print('<Closed by Meta Trader Stoploss or Takeprofit> ', self.symbol, 'tickets:', remove_tickets)

    def add_position_auto(self, real_positions: PositionInfo):
        for position in real_positions:
            found = False
            for ticket, info in self.positions.items():
                if ticket == position.ticket:
                    found = True
            if not found:
                print(f"ポジションをマネージャーに追加します。 {position}")                                   
                self.add_position(position)


    def df_position(self):
        data = []
        columns = None
        for ticket, position in self.positions.items():
            r = position.array()
            columns = PositionInfo.array_columns()
            #data.append(r)
        for ticket, position in self.positions_closed.items():
            r = position.array()
            columns = PositionInfo.array_columns()
            data.append(r)
        df = pd.DataFrame(data=data, columns=columns if columns else PositionInfo.array_columns())
        return df

    def profit_curve(self, times, positions_df=None):
        if positions_df is None:
            positions_df = self.df_position()

        jst_index = pd.DatetimeIndex(pd.to_datetime(list(times)))
        out = pd.DataFrame({"jst": jst_index})
        out["profit"] = 0.0
        out["gain"] = 0.0
        out["trade_count"] = 0

        if len(out) == 0:
            out["total_profit"] = 0.0
            out["total_gain"] = 0.0
            return out

        if positions_df is None or len(positions_df) == 0:
            out["total_profit"] = out["profit"].cumsum()
            out["total_gain"] = out["gain"].cumsum()
            return out

        positions = positions_df.copy()
        if "closed" in positions.columns:
            positions = positions.loc[positions["closed"] == True]
        positions = positions.dropna(subset=["exit_time"])
        if "profit" not in positions.columns and "gain" not in positions.columns:
            raise ValueError("positions must contain profit or gain")
        if "profit" in positions.columns:
            positions["profit"] = pd.to_numeric(
                positions["profit"], errors="coerce"
            )
        else:
            positions["profit"] = pd.to_numeric(
                positions["gain"], errors="coerce"
            )
        if "gain" in positions.columns:
            positions["gain"] = pd.to_numeric(
                positions["gain"], errors="coerce"
            )
        else:
            positions["gain"] = positions["profit"]
        positions = positions.dropna(subset=["profit", "gain"])

        profit_col = out.columns.get_loc("profit")
        gain_col = out.columns.get_loc("gain")
        trade_count_col = out.columns.get_loc("trade_count")

        def normalize_timestamp(value):
            timestamp = pd.Timestamp(value)
            if jst_index.tz is None:
                if timestamp.tzinfo is not None:
                    return timestamp.tz_localize(None)
                return timestamp
            if timestamp.tzinfo is None:
                return timestamp.tz_localize(jst_index.tz)
            return timestamp.tz_convert(jst_index.tz)

        for position in positions.itertuples(index=False):
            exit_time = normalize_timestamp(getattr(position, "exit_time"))
            row_index = int(jst_index.searchsorted(exit_time, side="left"))
            if row_index >= len(out):
                row_index = len(out) - 1
            if row_index < 0:
                row_index = 0
            profit = float(getattr(position, "profit"))
            gain = float(getattr(position, "gain"))
            out.iat[row_index, profit_col] = float(out.iat[row_index, profit_col]) + profit
            out.iat[row_index, gain_col] = float(out.iat[row_index, gain_col]) + gain
            out.iat[row_index, trade_count_col] = int(out.iat[row_index, trade_count_col]) + 1

        out["total_profit"] = out["profit"].cumsum()
        out["total_gain"] = out["gain"].cumsum()
        return out

    @staticmethod
    def performance(summary_df):
        df = summary_df.copy()
        if len(df) == 0:
            return {
                "trades": 0,
                "trade_count": 0,
                "win_rate": 0.0,
                "profit_mean": 0.0,
                "profit_median": 0.0,
                "average_gain": 0.0,
                "average_loss": 0.0,
                "payoff_ratio": 0.0,
                "drawdown_max": 0.0,
                "max_drawdown": 0.0,
                "profit": 0.0,
                "total_profit": 0.0,
                "total_gain": 0.0,
                "profit_factor": 0.0,
                "ruin_risk": 999.0,
                "score": 0.0,
            }
        df["pnl"] = pd.to_numeric(df["profit"], errors="coerce")
        df = df.dropna(subset=["pnl"])
        if len(df) == 0:
            return TradeManager.performance(df)

        # A single signal can close multiple positions at the same time.  Treat
        # those fills as one portfolio event so drawdown does not depend on the
        # arbitrary row order within that timestamp.
        pnl_events = df["pnl"].reset_index(drop=True)
        if "exit_time" in df.columns:
            exit_time = pd.to_datetime(df["exit_time"], errors="coerce")
            if exit_time.notna().all():
                pnl_events = df.assign(exit_time=exit_time).groupby("exit_time", sort=True)["pnl"].sum()

        cumulative = pnl_events.cumsum()
        running_peak = cumulative.cummax().clip(lower=0.0)
        draw_down = float((cumulative - running_peak).min())
        profit = float(df["pnl"].sum())
        total_gain = (
            float(pd.to_numeric(df["gain"], errors="coerce").sum())
            if "gain" in df.columns
            else profit
        )
        wins = df.loc[df["pnl"] > 0.0, "pnl"]
        losses = df.loc[df["pnl"] < 0.0, "pnl"]
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
        average_gain = float(wins.mean()) if len(wins) else 0.0
        average_loss = float(losses.mean()) if len(losses) else 0.0
        payoff_ratio = abs(average_gain / average_loss) if average_loss < 0.0 else (999.0 if average_gain > 0.0 else 0.0)
        ruin_risk = abs(draw_down / profit) if profit > 0.0 else 999.0
        score = 0.0 if profit == 0.0 else draw_down / profit
        out = {
            "trades": len(df),
            "trade_count": len(df),
            "win_rate": float((df["pnl"] > 0).mean()),
            "profit_mean": float(df["pnl"].mean()),
            "profit_median": float(df["pnl"].median()),
            "average_gain": average_gain,
            "average_loss": average_loss,
            "payoff_ratio": payoff_ratio,
            "drawdown_max": draw_down,
            "max_drawdown": draw_down,
            "profit": profit,
            "total_profit": profit,
            "total_gain": total_gain,
            "profit_factor": profit_factor,
            "ruin_risk": ruin_risk,
            "score": score,
        }
        return out
        
        
        
        
