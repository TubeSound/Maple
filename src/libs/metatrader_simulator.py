from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import math
from pathlib import Path

import pandas as pd
from dateutil import tz

try:
    from data_loader import DataLoader
    from trade_manager import PositionInfo, Signal
except ImportError:  # pragma: no cover
    from src.libs.data_loader import DataLoader
    from src.libs.trade_manager import PositionInfo, Signal

try:
    import MetaTrader5 as mt5api
except ImportError:  # pragma: no cover
    class mt5api:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1


JST = tz.gettz("Asia/Tokyo")


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


BUY = OrderSide.BUY
SELL = OrderSide.SELL

_TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
}


@dataclass(slots=True)
class SimUpdate:
    start_time: datetime
    end_time: datetime
    ticks: pd.DataFrame
    candles: pd.DataFrame
    closed_positions: list[PositionInfo]


class MetaTraderSimulator:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        update_seconds: int = 10,
        ticks: pd.DataFrame | None = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe.upper()
        self.update_seconds = int(update_seconds)
        self.ticks = self._normalize_ticks(ticks if ticks is not None else pd.DataFrame())
        self.current_index = 0
        self.current_time = self.ticks["jst"].iloc[0] if len(self.ticks) else None
        self._ticket_seed = 1
        self.positions: dict[int, PositionInfo] = {}
        self.closed_positions: list[PositionInfo] = []
        self._candle_ticks = self.ticks.iloc[0:0].copy()

    def set_symbol(self, symbol: str) -> None:
        self.symbol = symbol

    @classmethod
    def from_data(
        cls,
        symbol: str,
        timeframe: str,
        year: int,
        month_begin: int,
        month_end: int,
        update_seconds: int = 10,
    ) -> "MetaTraderSimulator":
        ticks = DataLoader(symbol).load_tick(year, month_begin, month_end)
        return cls(symbol=symbol, timeframe=timeframe, update_seconds=update_seconds, ticks=ticks)

    @classmethod
    def from_period(
        cls,
        symbol: str,
        timeframe: str,
        start: datetime | str | pd.Timestamp,
        end: datetime | str | pd.Timestamp,
        update_seconds: int = 10,
    ) -> "MetaTraderSimulator":
        start_ts = cls._to_jst_timestamp(start)
        end_ts = cls._to_jst_timestamp(end)
        loader = DataLoader(symbol)
        dfs = []
        month = pd.Timestamp(year=start_ts.year, month=start_ts.month, day=1, tz=JST)
        last_month = pd.Timestamp(year=end_ts.year, month=end_ts.month, day=1, tz=JST)
        while month <= last_month:
            df = loader.load_tick(month.year, month.month, month.month)
            if not df.empty:
                dfs.append(df)
            month = month + pd.DateOffset(months=1)
        ticks = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        simulator = cls(symbol=symbol, timeframe=timeframe, update_seconds=update_seconds, ticks=ticks)
        simulator.ticks = simulator.ticks[(simulator.ticks["jst"] >= start_ts) & (simulator.ticks["jst"] <= end_ts)].reset_index(drop=True)
        simulator.current_index = 0
        simulator.current_time = simulator.ticks["jst"].iloc[0] if len(simulator.ticks) else None
        simulator._reset_candle_state()
        return simulator

    @staticmethod
    def _normalize_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
        if ticks.empty:
            return pd.DataFrame(columns=["jst", "bid", "ask"])
        df = ticks.copy()
        df["jst"] = pd.to_datetime(df["jst"], format="mixed")
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
        return df.dropna(subset=["jst", "bid", "ask"]).sort_values("jst").reset_index(drop=True)

    def set_time(self, time: datetime | str | pd.Timestamp) -> None:
        if self.ticks.empty:
            self.current_time = pd.Timestamp(time)
            self.current_index = 0
            self._reset_candle_state()
            return
        t = pd.Timestamp(time)
        if t.tzinfo is None:
            t = t.tz_localize(JST)
        self.current_time = t
        self.current_index = int(self.ticks["jst"].searchsorted(t, side="left"))
        self._reset_candle_state()

    def get_time(self):
        return self.current_time

    def current_price(
        self,
        symbol: str | None = None,
        signal: OrderSide | None = None,
    ) -> float | None:
        tick = self.current_tick()
        if signal == BUY:
            return float(tick["ask"])
        if signal == SELL:
            return float(tick["bid"])
        if signal is not None:
            raise ValueError(f"Unsupported signal: {signal}")
        return float((tick["bid"] + tick["ask"]) / 2.0)

    def current_tick(self) -> pd.Series:
        if self.ticks.empty:
            raise ValueError("No tick data loaded")
        if self.current_time is not None:
            index = int(self.ticks["jst"].searchsorted(self.current_time, side="right")) - 1
            index = max(index, 0)
        else:
            index = self.current_index
        index = min(max(index, 0), len(self.ticks) - 1)
        return self.ticks.iloc[index]

    def market_entry(
        self,
        side: OrderSide,
        volume: float = 1.0,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "",
    ) -> PositionInfo:
        if side not in (BUY, SELL):
            raise ValueError(f"Unsupported side: {side}")
        tick = self.current_tick()
        sl = self._normalize_stop_loss(sl)
        self._validate_stop_loss(side, sl, float(tick["bid"]), float(tick["ask"]))
        price = float(tick["ask"] if side == BUY else tick["bid"])
        ticket = self._ticket_seed
        self._ticket_seed += 1
        order_type = mt5api.ORDER_TYPE_BUY if side == BUY else mt5api.ORDER_TYPE_SELL
        position = PositionInfo(
            self.symbol,
            order_type,
            tick["jst"],
            float(volume),
            ticket,
            price,
            None,
            sl=sl,
            tp=tp,
        )
        position.comment = comment
        self.positions[ticket] = position
        return position

    def entry(
        self,
        symbol: str | None,
        signal: int,
        time: datetime | pd.Timestamp | None,
        volume: float,
        param=None,
        stoploss: float | None = 0,
        takeprofit: float | None = 0,
        deviation: int = 20,
    ) -> tuple[bool, PositionInfo]:
        if signal == Signal.LONG:
            side = BUY
        elif signal == Signal.SHORT:
            side = SELL
        else:
            raise ValueError(f"Unsupported signal: {signal}")
        price = self.current_price(symbol, side)
        sl = float(stoploss) if stoploss not in (None, 0) else None
        tp = None
        if takeprofit not in (None, 0):
            takeprofit = float(takeprofit)
            tp = price + takeprofit if side == BUY else price - takeprofit
        position = self.market_entry(side, volume=volume, sl=sl, tp=tp, comment="python script open")
        if time is not None:
            position.entry_time = time
        position.param = param
        return True, position

    def set_stop_loss(self, ticket: int, sl: float | None) -> None:
        position = self.positions[ticket]
        tick = self.current_tick()
        sl = self._normalize_stop_loss(sl)
        self._validate_stop_loss(
            BUY if position.order_signal == Signal.LONG else SELL,
            sl,
            float(tick["bid"]),
            float(tick["ask"]),
        )
        position.sl = sl

    def set_take_profit(self, ticket: int, tp: float | None) -> None:
        self.positions[ticket].tp = tp

    def modify_sl(self, symbol: str | None, ticket: int, sl_price: float | None, param=None):
        self.set_stop_loss(ticket, sl_price)
        return True, self.positions[ticket]

    def modify_tp(self, symbol: str | None, ticket: int, tp_price: float | None, param=None):
        self.set_take_profit(ticket, tp_price)
        return True, self.positions[ticket]

    def modify_sltp(
        self,
        symbol: str | None,
        ticket: int,
        sl_price: float | None = None,
        tp_price: float | None = None,
        param=None,
    ):
        if sl_price is not None:
            self.set_stop_loss(ticket, sl_price)
        if tp_price is not None:
            self.set_take_profit(ticket, tp_price)
        return True, self.positions[ticket]

    def get_positions(self, symbol: str | None = None):
        infos = [
            position
            for position in self.positions.values()
            if symbol is None or position.symbol == symbol
        ]
        if symbol is None:
            return infos
        return infos, infos

    def get_closed_positions(self) -> list[PositionInfo]:
        return list(self.closed_positions)

    def close_position(
        self,
        ticket: int | PositionInfo,
        reason: str = "manual",
        price: float | None = None,
        time: datetime | pd.Timestamp | None = None,
    ) -> PositionInfo:
        if isinstance(ticket, PositionInfo):
            ticket = ticket.ticket
        if ticket not in self.positions:
            raise KeyError(f"Position not found: {ticket}")
        tick = self.current_tick()
        position = self.positions.pop(ticket)
        close_time = tick["jst"] if time is None else time
        if price is None:
            price = float(
                tick["bid"]
                if position.order_signal == Signal.LONG
                else tick["ask"]
            )
        position.exit_time = close_time
        position.exit_price = float(price)
        position.reason = reason
        position.calc_profit(float(price))
        position.closed = True
        self.closed_positions.append(position)
        return position

    def close_all_positions(self, reason: str = "manual") -> list[PositionInfo]:
        closed = []
        for ticket in list(self.positions.keys()):
            closed.append(self.close_position(ticket, reason=reason))
        return closed

    def close_order_result(self, info: PositionInfo, param=None, deviation: int = 20):
        return True, self.close_position(info, reason="manual")

    def close_by_position_info(self, position_info: PositionInfo, param=None):
        return True, self.close_position(position_info, reason="manual")

    def close(self, typ, ticket, price, volume, param=None, deviation: int = 20):
        return True, self.close_position(ticket, reason="manual", price=price)

    def close_all_position(self, symbol: str | None = None, reason: str = "manual"):
        close_tickets = []
        failed_tickets = []
        for position in list(self.positions.values()):
            if symbol is not None and position.symbol != symbol:
                continue
            try:
                self.close_position(position.ticket, reason=reason)
                close_tickets.append(position.ticket)
            except Exception:
                failed_tickets.append(position.ticket)
        return close_tickets, failed_tickets

    def get_rates(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        length: int | None = None,
        count: int | None = None,
        start_time: datetime | str | pd.Timestamp | None = None,
        end_time: datetime | str | pd.Timestamp | None = None,
        price: str = "mid",
    ) -> pd.DataFrame:
        if timeframe is None and isinstance(symbol, str) and symbol.upper() in _TIMEFRAME_SECONDS:
            timeframe = symbol
            symbol = None
        if count is None and length is not None:
            count = length
        timeframe = (timeframe or self.timeframe).upper()
        if timeframe not in _TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        end = self.current_time if end_time is None else self._to_jst_timestamp(end_time)
        start = None if start_time is None else self._to_jst_timestamp(start_time)

        ticks = self.ticks[self.ticks["jst"] <= end].copy()
        if start is not None:
            ticks = ticks[ticks["jst"] >= start]
        rates = self._ticks_to_rates(ticks, timeframe=timeframe, price=price)
        if count is not None:
            rates = rates.tail(int(count)).reset_index(drop=True)
        return rates

    def get_rates_jst(
        self,
        symbol: str | None,
        timeframe: str,
        jst_begin,
        jst_end,
        price: str = "mid",
    ) -> pd.DataFrame:
        return self.get_rates(
            symbol=symbol,
            timeframe=timeframe,
            start_time=jst_begin,
            end_time=jst_end,
            price=price,
        )

    def get_rates_utc(
        self,
        symbol: str | None,
        timeframe: str,
        utc_begin,
        utc_end,
        price: str = "mid",
    ) -> pd.DataFrame:
        return self.get_rates_jst(symbol, timeframe, utc_begin, utc_end, price=price)

    def get_ticks_jst(self, symbol: str | None, jst_begin, jst_end) -> pd.DataFrame:
        begin = self._to_jst_timestamp(jst_begin)
        end = self._to_jst_timestamp(jst_end)
        return self.ticks[(self.ticks["jst"] >= begin) & (self.ticks["jst"] <= end)].copy().reset_index(drop=True)

    def get_ticks(self, symbol: str | None, utc_begin, utc_end) -> pd.DataFrame:
        return self.get_ticks_jst(symbol, utc_begin, utc_end)

    def get_ticks_from(self, symbol: str | None, utc_time, length: int = 10) -> pd.DataFrame:
        begin = self._to_jst_timestamp(utc_time)
        ticks = self.ticks[self.ticks["jst"] >= begin].head(int(length)).copy()
        return ticks.reset_index(drop=True)

    def advance(self, seconds: int | None = None) -> SimUpdate:
        if self.ticks.empty:
            raise ValueError("No tick data loaded")
        seconds = self.update_seconds if seconds is None else int(seconds)
        start_time = self.current_time
        if start_time is None:
            start_time = self.ticks["jst"].iloc[0]
        end_time = start_time + timedelta(seconds=seconds)

        start_index = self.current_index
        end_index = int(self.ticks["jst"].searchsorted(end_time, side="right"))
        chunk = self.ticks.iloc[start_index:end_index].copy()
        closed = []
        for tick in chunk.itertuples(index=False):
            closed.extend(self._process_tick(tick.jst, float(tick.bid), float(tick.ask)))

        self.current_index = end_index
        self.current_time = end_time
        candles = self._make_candles(chunk)
        return SimUpdate(start_time=start_time, end_time=end_time, ticks=chunk, candles=candles, closed_positions=closed)

    @staticmethod
    def _to_jst_timestamp(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
        t = pd.Timestamp(value)
        if t.tzinfo is None:
            return t.tz_localize(JST)
        return t.tz_convert(JST)

    def _ticks_to_rates(self, ticks: pd.DataFrame, timeframe: str, price: str = "mid") -> pd.DataFrame:
        if ticks.empty:
            return pd.DataFrame(columns=["jst", "open", "high", "low", "close", "volume"])
        df = ticks.copy()
        if price == "mid":
            df["price"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
        elif price in {"bid", "ask"}:
            df["price"] = df[price].astype(float)
        else:
            raise ValueError("price must be one of: mid, bid, ask")

        rule = f"{_TIMEFRAME_SECONDS[timeframe]}s"
        grouped = df.set_index("jst")["price"].resample(rule)
        rates = grouped.ohlc().dropna()
        rates["volume"] = grouped.count().reindex(rates.index).fillna(0).astype(int)
        return rates.reset_index()

    def _process_tick(self, time, bid: float, ask: float) -> list[PositionInfo]:
        closed = []
        for ticket, position in list(self.positions.items()):
            reason = None
            exit_price = None
            if position.order_signal == Signal.LONG:
                if position.sl is not None and bid <= position.sl:
                    reason = "stop_loss"
                    # A stop is a market order once triggered.  If the first
                    # available tick has already passed the level, it cannot
                    # be filled at the more favourable requested price.
                    exit_price = min(float(position.sl), bid)
                elif position.tp is not None and bid >= position.tp:
                    reason = "take_profit"
                    exit_price = position.tp
            else:
                if position.sl is not None and ask >= position.sl:
                    reason = "stop_loss"
                    exit_price = max(float(position.sl), ask)
                elif position.tp is not None and ask <= position.tp:
                    reason = "take_profit"
                    exit_price = position.tp

            if reason is None:
                continue

            del self.positions[ticket]
            position.exit_time = time
            position.exit_price = exit_price
            position.reason = reason
            position.calc_profit(exit_price)
            position.closed = True
            self.closed_positions.append(position)
            closed.append(position)
        return closed

    @staticmethod
    def _validate_stop_loss(
        side: OrderSide, sl: float | None, bid: float, ask: float
    ) -> None:
        if sl is None:
            return
        sl = float(sl)
        if side == BUY and sl >= bid:
            raise ValueError(
                f"invalid BUY stop loss: sl={sl} must be below current bid={bid}"
            )
        if side == SELL and sl <= ask:
            raise ValueError(
                f"invalid SELL stop loss: sl={sl} must be above current ask={ask}"
            )

    @staticmethod
    def _normalize_stop_loss(sl: float | None) -> float | None:
        if sl is None:
            return None
        sl = float(sl)
        if sl == 0.0:
            return None
        if not math.isfinite(sl):
            raise ValueError(f"stop loss must be finite: {sl}")
        return sl

    def _make_candles(self, ticks: pd.DataFrame) -> pd.DataFrame:
        if ticks.empty:
            return pd.DataFrame(columns=["jst", "open", "high", "low", "close"])
        combined = pd.concat([self._candle_ticks, ticks], ignore_index=True)
        combined = combined.sort_values("jst").reset_index(drop=True)
        rates = self._ticks_to_rates(combined, timeframe=self.timeframe, price="mid")
        if not rates.empty:
            current_bucket = rates["jst"].iloc[-1]
            self._candle_ticks = combined[combined["jst"] >= current_bucket].copy()
        return rates.loc[:, ["jst", "open", "high", "low", "close"]]

    def _reset_candle_state(self) -> None:
        self._candle_ticks = self.ticks.iloc[0:0].copy()

    def save_history_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_positions(output_path, self.closed_positions)
        return output_path


def save_positions(path: str | Path, positions: list[PositionInfo]) -> None:
    pd.DataFrame([position.array() for position in positions], columns=PositionInfo.array_columns()).to_csv(path, index=False)
