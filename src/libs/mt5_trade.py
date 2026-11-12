# mt5_trade  ver4.0


import MetaTrader5 as mt5api
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta
import numpy as np
from dateutil import tz
from dataclasses import asdict, dataclass, fields, field
import calendar

try:
    from .common import TimeFrame, Columns
    from .time_utils import TimeUtils
    from .trade_manager import PositionInfo, Signal, Param
except ImportError:  # pragma: no cover - used when running this file with src/libs on PYTHONPATH
    from common import TimeFrame, Columns
    from time_utils import TimeUtils
    from trade_manager import PositionInfo, Signal, Param

JST = tz.gettz('Asia/Tokyo')
UTC = tz.UTC
        
        
def now():
    t = datetime.now(tz=UTC)
    return t

# numpy timestamp -> pydatetime naive
def nptimestamp2pydatetime(npdatetime):
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    seconds_since_epoch = (npdatetime - unix_epoch) / one_second
    dt = datetime.utcfromtimestamp(seconds_since_epoch)
    #timestamp = (npdatetime - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    #dt2 = datetime.utcfromtimestamp(timestamp)
    return dt

def jst2utc(jst: datetime): 
    return jst.astimezone(UTC)

def utc2jst(utc: datetime): 
    return utc.astimezone(JST)

def utcstr2datetime(utc_str: str, format='%Y-%m-%d %H:%M:%S'):
    utc = datetime.strptime(utc_str, format)
    utc = utc.replace(tzinfo=UTC)
    return utc

def slice(df, ibegin, iend):
    new_df = df.iloc[ibegin: iend + 1, :]
    return new_df

def df2dic(df: pd.DataFrame):
    dic = {}
    for column in df.columns:
        dic[column] = df[column].to_numpy()
    return dic

def time_str_2_datetime(df, time_column, format='%Y-%m-%d %H:%M:%S'):
    time = df[time_column].to_numpy()
    new_time = [datetime.strptime(t, format) for t in time]
    df[time_column] = new_time


def get_first_monday(year: int, month: int) -> datetime:
    # JST タイムゾーンを設定
    # 指定年月の1日午前0時（JST aware）を作成
    first_day = datetime(year, month, 1, 0, 0, 0, tzinfo=JST)

    # 1日の曜日を取得（0: 月曜, 1: 火曜, ..., 6: 日曜）
    weekday = first_day.weekday()

    # 月曜日（0）までの経過日数を計算
    days_to_monday = (0 - weekday) % 7

    # 1日に日数を加算して最初の月曜日の datetime オブジェクトを返す
    return first_day + timedelta(days=days_to_monday)
    
def position_dic_array(positions):
    array = []
    for position in positions:
        d = {
            'ticket': position.ticket,
            'time': pd.to_datetime(position.time, unit='s'),
            'symbol': position.symbol,
            'type': position.type,
            'volume': position.volume,
            'price_open': position.price_open,
            'sl': position.sl,
            'tp': position.tp,
            'price_current': position.price_current,
            'profit': position.profit,
            'swap': position.swap,
            'comment': position.comment,
            'magic': position.magic}
        array.append(d)
    return array

@dataclass(frozen=True)
class Mt5ConnectionConfig:
    name: str = "default"
    path: str = None
    login: int = None
    password: str = None
    server: str = None
    timeout: int = None
    portable: bool = None

    @classmethod
    def from_dict(cls, name, data):
        data = data or {}
        login = data.get("login")
        if login in ("", None):
            login = None
        else:
            login = int(login)
        return cls(
            name=name,
            path=data.get("path") or None,
            login=login,
            password=data.get("password") or None,
            server=data.get("server") or None,
            timeout=data.get("timeout"),
            portable=data.get("portable"),
        )

    def initialize_kwargs(self):
        values = {
            "path": self.path,
            "login": self.login,
            "password": self.password,
            "server": self.server,
            "timeout": self.timeout,
            "portable": self.portable,
        }
        return {key: value for key, value in values.items() if value is not None}

    def key(self):
        return (
            self.path,
            self.login,
            self.password,
            self.server,
            self.timeout,
            self.portable,
        )
        
class TimeConversion:
    def __init__(self, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer):
        now = datetime.now(JST)
        dt, tz = TimeUtils.delta_hour_from_gmt(now, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer)
        self.delta_hour_from_gmt  = dt
        self.server_timezone = tz
        print('SeverTime GMT+', dt, tz)
        
    def utc_to_server_time(self, utc: datetime): 
        t = utc + self.delta_hour_from_gmt
        t = t.replace(tzinfo=UTC)
        return t
        
    def server_time_to_utc(self, time: datetime):
        t =  time - self.delta_hour_from_gmt
        t = t.replace(tzinfo=UTC)
        return t
    
    def jst_to_server_time(self, jst: datetime):
        utc = jst2utc(jst)
        return self.utc_to_server_time(utc)

    def server_time_to_jst(self, time: datetime):
        utc = self.server_time_to_utc(time)
        return utc2jst(utc) 

class Mt5Trade:
    def __init__(self, time_conversion: TimeConversion):
        self.time_conversion = time_conversion
        self.ticket = None
        self.symbol = None
        
        
    @staticmethod
    def connect():
        if mt5api.initialize():
            print('Connected to MT5 Version', mt5api.version())
        else:
            print('initialize() failed, error code = ', mt5api.last_error())
            
    @staticmethod
    def disconnect():
        mt5api.shutdown()

    def set_symbol(self, symbol):
        self.symbol = symbol
        
    def parse_order_result(self, result, time: datetime, stoploss, takeprofit, param):
        if result is None:
            print('Error')
            return False, result
        
        if takeprofit is None:
            takeprofit = 0
        code = result.retcode
        if code == 10009:
            #print("注文完了", self.symbol, 'type', result.request.type, 'volume', result.volume)
            position_info = PositionInfo(self.symbol, result.request.type, time, result.volume, result.order, result.price, param)
            return True, position_info
        elif code == 10013:
            print("無効なリクエスト")
            return False, result
        elif code == 10018:
            print("マーケットが休止中")
            return False, result       
        elif code == 10019:
            print("リクエストを完了するのに資金が不充分。")
            return False, result
        else:
            print('Entry error code', code)
            return False, result
        
    def current_price(self, symbol, signal):
        tick = mt5api.symbol_info_tick(symbol)
        if signal == Signal.LONG:
            return tick.ask
        elif signal == Signal.SHORT:
            return tick.bid 
        else:
            return None
        
    def entry(self, symbol, signal: Signal, time: datetime, volume: float, param: Param, stoploss=0, takeprofit=0, deviation=20, comment="python script open", *, price=None):        
        #point = mt5api.symbol_info(self.symbol).point
        if price is None:
            price = self.current_price(symbol, signal)
        if signal == Signal.LONG:
            typ =  mt5api.ORDER_TYPE_BUY
        elif signal == Signal.SHORT:
            typ =  mt5api.ORDER_TYPE_SELL
            
        #print('Entry symbol:', symbol, 'signal:', signal, 'Price:', price, 'volume:', volume, 'deviation:', deviation)
            
        request = {
            "action": mt5api.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": typ,
            "price": float(price),
            "deviation": deviation,# 許容スリップページ
            "magic":  234000,
            "comment": comment,
            "type_time": mt5api.ORDER_TIME_GTC,
            "type_filling": mt5api.ORDER_FILLING_IOC,
        }

        if stoploss > 0:
            if signal == Signal.LONG:
                request['sl'] = float(stoploss)
            elif signal == Signal.SHORT:
                request['sl'] = float(stoploss)
        
        if takeprofit is None:
            takeprofit = 0
        if takeprofit > 0:
            if signal == Signal.LONG:
                request['tp'] = float(price + takeprofit)
            elif signal == Signal.SHORT:
                request['tp'] = float(price - takeprofit)
        result = mt5api.order_send(request)
        #print('エントリー ', request)
        return self.parse_order_result(result, time, stoploss, takeprofit, param)
    
    def get_positions(self, symbol):
        mt5_positions = mt5api.positions_get(symbol=symbol)
        if mt5_positions is None:
            raise Exception('get position error')
        infos = []
        for pos in mt5_positions:
             time = pd.to_datetime(pos.time, unit='s')
             utc = self.time_conversion.server_time_to_utc(time)
             utc = utc.replace(tzinfo=UTC)
             jst = utc.astimezone(JST)
             info = PositionInfo(symbol, pos.type, jst, pos.volume, pos.ticket, pos.price_open, None)
             infos.append(info)
        return mt5_positions, infos

    def is_long(self, typ):
        if typ == mt5api.ORDER_TYPE_BUY or typ == mt5api.ORDER_TYPE_BUY_LIMIT or typ == mt5api.ORDER_TYPE_BUY_STOP_LIMIT:
            return True
        else:
            return False

    def is_short(self, typ):
        if typ == mt5api.ORDER_TYPE_SELL or typ == mt5api.ORDER_TYPE_SELL_LIMIT or typ == mt5api.ORDER_TYPE_SELL_STOP_LIMIT:
            return True
        else:
            return False

    def close_position(self, position: PositionInfo, param: Param, deviation=20):    
        tick = mt5api.symbol_info_tick(position.symbol)
        if self.is_long(position.type):
            price = tick.bid
            typ = mt5api.ORDER_TYPE_SELL
        elif self.is_short(position.type):
            price = tick.ask
            typ = mt5api.ORDER_TYPE_BUY
        return self.close(typ, position.ticket, price, position.volume, param, deviation=deviation)
    
    def close_order_result(self, info: PositionInfo, param: Param, deviation=20):     
        tick = mt5api.symbol_info_tick(info.symbol)
        if self.is_long(info.type):
            price = tick.bid
            typ = mt5api.ORDER_TYPE_SELL
        elif self.is_short(info.type):
            price = tick.ask
            typ = mt5api.ORDER_TYPE_BUY
        return self.close(typ, info.ticket, price, info.volume, param, deviation=deviation)

    def close_by_position_info(self, position_info: PositionInfo, param: Param):
        tick = mt5api.symbol_info_tick(position_info.symbol)            
        if position_info.order_signal == Signal.LONG:
            price = tick.bid
            typ = mt5api.ORDER_TYPE_SELL
        else:
            price = tick.ask
            typ = mt5api.ORDER_TYPE_BUY
        return self.close(typ, position_info.ticket, price, position_info.volume, param)

    def close(self, typ, ticket, price, volume, param: Param, deviation=20):
        request = {
            "action": mt5api.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": self.symbol,
            "volume": volume,
            "type": typ,
            "price": price,
            "deviation": deviation,
            "magic": 100,
            "comment": "python script close",
            "type_time": mt5api.ORDER_TIME_GTC,
            "type_filling": mt5api.ORDER_FILLING_IOC,
        }
        result = mt5api.order_send(request)
        #print('決済', request)
        return self.parse_order_result(result, None, None, None, param)
    
    def modify_sl(self, symbol, ticket, sl_price, param:Param):
        positions, infos =self.get_positions(symbol)
        found = False
        for position in positions:
            if position.ticket == ticket:
                found = True
                break
        
        if not found:
            print('Not found', symbol, ticket)
            return
    
        request = {
            "action": mt5api.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": float(sl_price),
            "tp": float(position.tp)
        }
        
        result = mt5api.order_send(request)
        #print('Set SL', request, result)
        return self.parse_order_result(result, None, None, None, param)
    
    def close_all_position(self, symbol):
        positions, infos = self.get_positions(symbol)
        close_tickets = []
        failed_tickets = []
        for position in positions:
            ret, _ = self.close_position(position, None)
            if ret:
                close_tickets.append(position.ticket)
            else:
                failed_tickets.append(position.ticket)
        return close_tickets, failed_tickets
    
    def get_ticks_jst(self, symbol, jst_begin, jst_end, *, as_numpy=False):
        t_begin = self.time_conversion.jst_to_server_time(jst_begin)
        t_end = self.time_conversion.jst_to_server_time(jst_end)
        return self.get_ticks(symbol, t_begin, t_end, as_numpy=as_numpy)

    # t0, t1: server_time
    def get_ticks(self, symbol, t0, t1, *, as_numpy=False):
        ticks = mt5api.copy_ticks_range(symbol, t0, t1, mt5api.COPY_TICKS_ALL)
        return self.parse_ticks_numpy(ticks) if as_numpy else self.parse_ticks(ticks)   
    
    def get_ticks_utc(self, symbol, utc_begin, utc_end, *, as_numpy=False):
        t0 = self.time_conversion.utc_to_server_time(utc_begin)
        t1 = self.time_conversion.utc_to_server_time(utc_end)
        ticks = mt5api.copy_ticks_range(symbol, t0, t1, mt5api.COPY_TICKS_ALL)
        return self.parse_ticks_numpy(ticks) if as_numpy else self.parse_ticks(ticks)    

    def get_ticks_from_jst(self, symbol,  jst, length=10, *, as_numpy=False):
        t = self.time_conversion.jst_to_server_time(jst)
        ticks = mt5api.copy_ticks_from(symbol, t, length, mt5api.COPY_TICKS_ALL)
        return self.parse_ticks_numpy(ticks) if as_numpy else self.parse_ticks(ticks)
    
    def get_ticks_from_utc(self, symbol,  utc_time, length=10, *, as_numpy=False):
        t = self.time_conversion.utc_to_server_time(utc_time)
        ticks = mt5api.copy_ticks_from(symbol, t, length, mt5api.COPY_TICKS_ALL)
        return self.parse_ticks_numpy(ticks) if as_numpy else self.parse_ticks(ticks)
        
    # Compatibility for callers using the former UTC method name.
    get_ticks_from = get_ticks_from_utc

    def parse_ticks_numpy(self, ticks):
        """Array dictionary; jst stores UTC instants as datetime64[ns].

        Apply the SAME server->UTC offset as parse_ticks exactly once. JST is
        the external display timezone; datetime64 cannot carry timezone info.
        """
        if ticks is None:
            raise RuntimeError(f"MT5 ticks failed: {mt5api.last_error()}")
        raw = np.asarray(ticks)
        fields = raw.dtype.names or ()
        if 'time' not in fields:
            if not len(raw):
                return {'jst': np.empty(0, dtype='datetime64[ns]'),
                        'time_msc': np.empty(0, dtype=np.int64),
                        'bid': np.empty(0), 'ask': np.empty(0)}
            raise ValueError("MT5 tick time missing")
        result = {name: raw[name].copy() for name in fields}
        msc = raw['time_msc'] if 'time_msc' in fields else raw['time'] * 1000
        delta_ms = int(self.time_conversion.delta_hour_from_gmt.total_seconds() * 1000)
        result['time_msc'] = np.asarray(msc, dtype=np.int64) - delta_ms
        result['jst'] = result['time_msc'].astype('datetime64[ms]').astype('datetime64[ns]')
        return result

    def parse_ticks(self, ticks):
        df = pd.DataFrame(ticks)
        if len(df) == 0:
            return df
        if 'time' not in df.columns:
            return pd.DataFrame()
        df = self.parse_rates(df)
        msec_col = None
        if 'time_msc' in df.columns:
            msec_col = 'time_msc'
        elif 'time_msec' in df.columns:
            msec_col = 'time_msec'

        if msec_col is not None:
            msec = pd.to_numeric(df[msec_col], errors='coerce') % 1000
            delta = pd.to_timedelta(msec.fillna(0), unit='ms')
            if Columns.UTC in df.columns:
                utc = pd.to_datetime(df[Columns.UTC]) + delta
                df[Columns.UTC] = [pd.Timestamp(value).to_pydatetime() for value in utc]
            if Columns.JST in df.columns:
                jst = pd.to_datetime(df[Columns.JST]) + delta
                df[Columns.JST] = [pd.Timestamp(value).to_pydatetime() for value in jst]
        return df
    
    def get_rates_jst(self, symbol, timeframe: TimeFrame, jst_begin, jst_end):
        begin = self.time_conversion.jst_to_server_time(jst_begin)
        end = self.time_conversion.jst_to_server_time(jst_end)
        rates = mt5api.copy_rates_range(symbol, TimeFrame.const(timeframe), begin, end)
        return self.parse_rates(rates)
    
    def get_rates_utc(self, symbol, timeframe, utc_begin, utc_end):
        begin = self.time_conversion.utc_to_server_time(utc_begin)
        end = self.time_conversion.utc_to_server_time(utc_end)
        rates = mt5api.copy_rates_range(symbol, TimeFrame.const(timeframe), begin, end)
        return self.parse_rates(rates)
        
    def get_rates(self, symbol, timeframe: str, length: int, *, as_numpy=False):
        #print(self.symbol, timeframe)
        rates = mt5api.copy_rates_from_pos(symbol, TimeFrame.const(timeframe), 0, length)
        if as_numpy:
            if rates is None:
                raise RuntimeError(f"MT5 rates failed: {mt5api.last_error()}")
            raw = np.asarray(rates)
            offset = int(self.time_conversion.delta_hour_from_gmt.total_seconds())
            times = (raw['time'].astype(np.int64) - offset).astype('datetime64[s]').astype('datetime64[ns]')
            return {'jst': times, **{key: raw[key].astype(float) for key in ('open', 'high', 'low', 'close')},
                    'volume': raw['tick_volume'].astype(float)}
        if rates is None:
            return None
        df0 = self.parse_rates(rates)
        df = df0[["jst", "open", "high", "low", "close"]]
        df["volume"] = df0["tick_volume"]
        return df

    def severtime2utc(self, time):
        return [t.replace(tzinfo=UTC) - self.time_conversion.delta_hour_from_gmt for t in time]

    def parse_rates(self, rates):
        if isinstance(rates, pd.DataFrame):
            df = rates.copy()
        else:
            df = pd.DataFrame(rates)

        if not 'time' in df.columns:
            return None
        
        time = pd.to_datetime(df['time'], unit='s').array.to_pydatetime()
        utc = self.severtime2utc(time)
        df[Columns.UTC] = [pd.Timestamp(t).to_pydatetime() for t in utc]
        jst = [pd.Timestamp(utc2jst(t)).to_pydatetime() for t in utc]
        df[Columns.JST] = jst
        return df

    def adjust(self, time):
        utc = []
        jst = []
        for i, ts in enumerate(time):
            if i == len(time) - 1:
                pass
            #t0 = pd.to_datetime(ts)
            t0 = ts.replace(tzinfo=UTC)
            t = self.time_conversion.server_time_to_utc(t0)
            utc.append(t)
            tj = t.astimezone(JST)
            jst.append(tj)
        return utc, jst
    
    def history(self, jst_from, jst_to):
        type2str = {0: 'buy', 1: 'sell', 2: 'balance'}
        entry2str = {0: 'open', 1: 'close'}
        
        utc_from = jst_from.astimezone(tz=UTC)
        t_from = self.time_conversion.utc_to_server_time(utc_from)
        utc_to = jst_to.astimezone(tz=UTC)
        t_to = self.time_conversion.utc_to_server_time(utc_to)
        deals = mt5api.history_deals_get(t_from, t_to)
        if deals is None:
            return None
        
        data = []
        for deal in deals:
            if deal.type > 2:
                continue
            t = pd.to_datetime(deal.time, unit='s')
            utc, jst = self.adjust([t])
            data.append([t, jst[0], deal.position_id, deal.symbol, type2str[deal.type], entry2str[deal.entry], deal.reason, deal.volume, deal.price, deal.profit])
        df = pd.DataFrame(data=data, columns = ['time', 'jst', 'ticket', 'symbol', 'type', 'entry', 'reason', 'volume', 'price', 'profit'])
        return df

    def deal_history(self, symbol, jst_begin, jst_end):
        if jst_end is None:
            jst_end = datetime.now().astimezone(JST)
        df0 = self.history(jst_begin, jst_end)
        df = df0[df0['symbol'] == symbol]
        data = []
        for ticket in df['ticket']:
            df1 = df[df['ticket'] == ticket]
            op = df1[df1['entry'] == 'open'].values
            cl = df1[df1['entry'] == 'close'].values
            if len(op) > 0 and len(cl) > 0:
                op = op[0]
                cl = cl[0]
                if op[4] == 'buy':
                    profit = cl[8] - op[8]
                elif op[4] == 'sell':
                    profit = op[8] - cl[8]
                data.append([op[4], op[7], op[1], op[8], cl[1], cl[8], profit])
        df_profits = pd.DataFrame(data=data, columns=['type', 'volume', 'time_entry', 'price_entry', 'time_exit', 'price_exit', 'profit'])
        df_profits = df_profits.sort_values('time_entry')
        return df_profits
    
    
    def open_positions(self, symbol, jst_begin):
        positions = mt5api.positions_get(symbol=symbol)
        if positions is None:
            return None
        df0 = self.history(jst_begin, datetime.now().astimezone(JST))
        dfs = []
        for position in positions:
            df1 = df0[df0['ticket'] == position.ticket]
            dfs.append(df1)
        df_positions = pd.concat(dfs)
        
        current = self.get_ticks_from(symbol, datetime.now().astimezone(UTC) - timedelta(minutes=1), length=1000)
        leveledge = mt5api.symbol_info(symbol).trade_contract_size
        for row in df_positions.itertuples():
            if row.profit != 0.0:
                continue
            if row.type == 'buy':
                price = current['bid'].values[-1]
                profit = (price - row.price) * leveledge * row.volume
            elif row.type == 'sell':
                price = current['ask'].values[-1]
                profit = (row.price - price) * leveledge * row.volume
            else:
                price = None
            df_positions.at[row.Index, 'price_current'] = price
            df_positions.at[row.Index, 'profit'] = profit
        return df_positions
        
class Mt5TradeSim:
    def __init__(self, symbol: str, files: dict):
        self.symbol = symbol
        self.load_data(files)
                
    def adjust_msec(self, df, time_column, msec_column):
        new_time = []
        for t, tmsec in zip(df[time_column], df[msec_column]):
            msec = tmsec % 1000
            dt = t + timedelta(milliseconds= msec)
            new_time.append(dt)
        df[time_column] = new_time
                
    def load_data(self, files):
        dic = {}
        for timeframe, file in files.items():
            df = pd.read_csv(file)
            time_str_2_datetime(df, "time")
            if timeframe == TimeFrame.TICK:
                self.adjust_msec(df, "time", 'time_msc')
            dic[timeframe] = df
        self.dic = dic
    
    def search_in_time(self, df, time_column, utc_time_begin, utc_time_end):
        time = list(df[time_column].values)
        if utc_time_begin is None:
            ibegin = 0
        else:
            ibegin = None
        if utc_time_end is None:
            iend = len(time) - 1
        else:
            iend = None
        for i, t in enumerate(time):
            dt = npdatetime2datetime(t)
            if ibegin is None:
                if dt >= utc_time_begin:
                    ibegin = i
            if iend is None:
                if dt > utc_time_end:
                    iend = i - 1
            if ibegin is not None and iend is not None:
                break
        slilced = slice(df, ibegin, iend)
        return slilced

    def get_rates(self, timeframe: str, utc_begin, utc_end):
        #print(self.symbol, timeframe)
        df = self.dic[timeframe]
        return self.search_in_time(df, "time", utc_begin, utc_end)

    def get_ticks(self, utc_begin, utc_end):
        df = self.dic[TimeFrame.TICK]
        return self.search_in_time(df, "time", utc_begin, utc_end)
    
    
def arrange(df):
    df_out = df[['jst', 'open', 'high', 'low', 'close']]
    return df_out
    
def history(symbol, jst_begin, jst_end):
    import os
    conversion = TimeConversion(3, 2, 11, 1, 3.0)
    mt5trade = Mt5Trade(conversion)
    Mt5Trade.connect()
    dir_path = "./backtest/histories/"
    os.makedirs(dir_path, exist_ok=True)
    if jst_end is None:
        jst_end = datetime.now().astimezone(JST)
    df = mt5trade.deal_history(symbol, jst_begin, jst_end)
    filename = f"{symbol}_{jst_begin.strftime('%Y%m%d')}-{jst_end.strftime('%Y%m%d')}.csv"
    filepath = os.path.join(dir_path, f"history_{filename}")
    df.to_csv(filepath, index=False, encoding='sjis')
    df = mt5trade.get_rates_jst(symbol, 'M1', jst_begin, jst_end)
    if df is None:
        return
    df = arrange(df)
    filepath = os.path.join(dir_path, f"price_{filename}")
    df.to_csv(filepath, index=False, encoding='sjis')


def main():
    t = datetime(2026, 1, 1).astimezone(JST)
    for symbol in ['JP225', 'XAUUSD', 'XAGUSD', 'US100', 'USDJPY']:
        history(symbol, t)

def date_str(dt):
    return dt.strftime("%Y%m%d")


def arrange_candle(df):
    df1 = df[['jst', 'open', 'high', 'low', 'close']]
    volume = df['tick_volume']
    df1['volume'] = volume
    return df1

def arrange_tick(df):
    try:
        df1 = df[['jst', 'bid', 'ask', 'flags']]
        return df1
    except:
        return df


ALL_SYMBOLS =  ["JP225", "US30", "US100", "UK100", "HKD50", "GER40", "USOIL", "NGAS", "UKOIL","XAUUSD", "XAGUSD", "XPDUSD", "XPTUSD", "XCUUSD", "CORN", "WHEAT", "SOYBEAN", "COFFEE.A", "SUGAR", "USDJPY", "USDMXN", "GBPJPY", "GBPAUD", "GBPNZD", "EURNZD", "EURAUD"]
DATA_ROOT = "c:/develop/MarketData/axiory"
def download_by_month():
   
    import os
    conversion = TimeConversion(3, 2, 11, 1, 3.0)
    mt5trade = Mt5Trade(conversion)
    Mt5Trade.connect()
    begin = datetime(2026, 10, 1, tzinfo=JST)
    end = datetime(2026, 12, 1, tzinfo=JST) - timedelta(seconds=1)
    
    for symbol in ALL_SYMBOLS:
        for timeframe in [ 'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']:
            year = begin.year
            month = begin.month
            t0 = datetime(year, month, 1, tzinfo=JST)
            while t0 < end:
                t0 = datetime(year, month, 1, tzinfo=JST)
                thalf = datetime(year, month, 16, tzinfo=JST) - timedelta(seconds=1) 
                if month == 12:
                    year += 1
                    month = 1
                else:
                    month += 1
                t1 = datetime(year, month, 1, tzinfo=JST) - timedelta(seconds=1)
                dir_path = os.path.join(DATA_ROOT, f"{symbol}/{timeframe}")
                os.makedirs(dir_path, exist_ok=True)
                if timeframe == "TICK":
                    df0 = mt5trade.get_ticks_jst(symbol, t0, thalf)
                    if len(df0) < 100:
                        continue
                    df = arrange_tick(df0)
                    path = f"{dir_path}/{symbol}_{timeframe}_{t0.year}-{str(t0.month).zfill(2)}-1.pkl"
                    df.to_pickle(path)
                    print(f"Download {symbol} {year}/{month} {timeframe}-1/2 len: {len(df)}")
                    df0 = mt5trade.get_ticks_jst(symbol, thalf, t1)
                    if len(df0) < 100:
                        continue
                    df = arrange_tick(df0)
                    path = f"{dir_path}/{symbol}_{timeframe}_{t0.year}-{str(t0.month).zfill(2)}-2.pkl"
                    df.to_pickle(path)
                    print(f"Download {symbol} {year}/{month} {timeframe}-2/2 len: {len(df)}")    
                else:
                    df0 = mt5trade.get_rates_jst(symbol, timeframe, t0, t1)
                    if df0 is None:
                        continue
                    df = arrange_candle(df0)
                    if len(df) > 0:
                        df.to_csv(f"{dir_path}/{symbol}_{timeframe}_{t0.year}-{str(t0.month).zfill(2)}.csv", index=False)
                        print(f"Download {symbol} {year}/{month} {timeframe} len: {len(df)}")    

def download_ticks():
    import os
    conversion = TimeConversion(3, 2, 11, 1, 3.0)
    mt5trade = Mt5Trade(conversion)
    Mt5Trade.connect()
    symbols = ['XAUUSD']    
    t0 = datetime(2026,  10, 1,  7, 0, tzinfo=JST)
    t1 = datetime(2026, 12, 1, tzinfo=JST) - timedelta(seconds=1)
    for symbol in ALL_SYMBOLS:
        dir_path = f"{DATA_ROOT}/{symbol}/TICK"
        os.makedirs(dir_path, exist_ok=True)
        begin = t0
        end = begin + timedelta(hours=24)
        count = 1
        while begin < t1:
            df0 = mt5trade.get_ticks_jst(symbol, begin, end)
            df = arrange_tick(df0)
            if len(df) > 100:
                jst = df0['jst'].to_list()
                name = f"{symbol}_TICK_{jst[0].year}-{str(jst[0].month).zfill(2)}-{str(jst[0].day).zfill(2)}.csv"
                df.to_csv(f"{dir_path}/{name}", index=False)
                print(f"Download {symbol} {begin.year}/{begin.month}/{begin.day} TICK-{count}/2 len: {len(df)}")
            begin = end
            end = begin + timedelta(hours=24)
            if end.month != begin.month:
                count = 1
            else:
                count += 1

def test():
    conversion = TimeConversion(3, 2, 11, 1, 3.0)
    mt5trade = Mt5Trade(conversion)
    Mt5Trade.connect()
    df = mt5trade.open_positions('USDJPY', datetime(2026, 5, 18).astimezone(JST))
    print(df)
    
if __name__ == '__main__':
    download_by_month()
    download_ticks()
    #history("XAUUSD", datetime(2026, 10, 9, 18, 0, tzinfo=JST), datetime(2026, 10, 10, 3, tzinfo=JST))
