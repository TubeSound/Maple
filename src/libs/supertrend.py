from typing import TypedDict
import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int8]
BoolArray = npt.NDArray[np.bool_]


class CandleDict(TypedDict):
    open: FloatArray
    high: FloatArray
    low: FloatArray
    close: FloatArray


class SupertrendResult(TypedDict):
    atr: FloatArray
    up: FloatArray
    down: FloatArray
    supertrend: FloatArray
    trend: IntArray
    buy_signal: BoolArray
    sell_signal: BoolArray
    change_signal: BoolArray


def true_range(
    high: FloatArray,
    low: FloatArray,
    close: FloatArray,
) -> FloatArray:
    """
    TradingViewのtr相当。

    先頭要素は前回終値が存在しないため high - low とする。
    """
    n = len(close)
    tr = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return tr

    tr[0] = high[0] - low[0]

    if n > 1:
        high_low = high[1:] - low[1:]
        high_prev_close = np.abs(high[1:] - close[:-1])
        low_prev_close = np.abs(low[1:] - close[:-1])

        tr[1:] = np.maximum.reduce(
            [high_low, high_prev_close, low_prev_close]
        )

    return tr


def sma(array: FloatArray, period: int) -> FloatArray:
    """
    Pine Scriptのsma()に近い単純移動平均。
    period本に満たない区間はnp.nan。
    """
    n = len(array)
    out = np.full(n, np.nan, dtype=np.float64)

    if period <= 0:
        raise ValueError("periodは1以上にしてください")

    if n < period:
        return out

    cumsum = np.cumsum(np.insert(array, 0, 0.0))
    out[period - 1:] = (
        cumsum[period:] - cumsum[:-period]
    ) / period

    return out


def rma(array: FloatArray, period: int) -> FloatArray:
    """
    TradingViewのRMA（Wilder移動平均）相当。

    初期値:
        最初のperiod本のSMA

    以降:
        RMA[i] = (RMA[i-1] * (period - 1) + array[i]) / period
    """
    n = len(array)
    out = np.full(n, np.nan, dtype=np.float64)

    if period <= 0:
        raise ValueError("periodは1以上にしてください")

    if n < period:
        return out

    initial = np.mean(array[:period])
    out[period - 1] = initial

    for i in range(period, n):
        out[i] = (
            out[i - 1] * (period - 1) + array[i]
        ) / period

    return out


def supertrend(
    candles: CandleDict,
    period: int = 10,
    multiplier: float = 3.0,
    change_atr: bool = True,
) -> SupertrendResult:
    """
    TradingView Pine Script v4のSupertrendをNumPyで再現する。

    Parameters
    ----------
    candles:
        {
            "open":  np.ndarray,
            "high":  np.ndarray,
            "low":   np.ndarray,
            "close": np.ndarray,
        }

    period:
        ATR Period

    multiplier:
        ATR Multiplier

    change_atr:
        True:
            TradingViewのatr(period)と同じWilder方式のRMA

        False:
            sma(tr, period)

    Returns
    -------
    {
        "atr": ATR,
        "up": 上昇トレンド用バンド,
        "down": 下降トレンド用バンド,
        "supertrend": 現在有効なSupertrendライン,
        "trend": 1=上昇、-1=下降,
        "buy_signal": 買い転換,
        "sell_signal": 売り転換,
        "change_signal": トレンド転換,
    }
    """
    required_keys = ("open", "high", "low", "close")

    for key in required_keys:
        if key not in candles:
            raise KeyError(f"candlesに'{key}'がありません")

    open_ = np.asarray(candles["open"], dtype=np.float64)
    high = np.asarray(candles["high"], dtype=np.float64)
    low = np.asarray(candles["low"], dtype=np.float64)
    close = np.asarray(candles["close"], dtype=np.float64)

    n = len(close)

    if not (
        len(open_) == n
        and len(high) == n
        and len(low) == n
    ):
        raise ValueError("open/high/low/closeの配列長が一致していません")

    if period <= 0:
        raise ValueError("periodは1以上にしてください")

    if multiplier < 0:
        raise ValueError("multiplierは0以上にしてください")

    atr = np.full(n, np.nan, dtype=np.float64)
    up = np.full(n, np.nan, dtype=np.float64)
    down = np.full(n, np.nan, dtype=np.float64)
    supertrend_line = np.full(n, np.nan, dtype=np.float64)

    trend = np.ones(n, dtype=np.int8)

    buy_signal = np.zeros(n, dtype=np.bool_)
    sell_signal = np.zeros(n, dtype=np.bool_)
    change_signal = np.zeros(n, dtype=np.bool_)

    if n == 0:
        return {
            "atr": atr,
            "up": up,
            "down": down,
            "supertrend": supertrend_line,
            "trend": trend,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "change_signal": change_signal,
        }

    # Pine Script:
    # src = hl2
    src = (high + low) / 2.0

    tr = true_range(high, low, close)

    if change_atr:
        # Pine Script: atr(Periods)
        atr[:] = rma(tr, period)
    else:
        # Pine Script: sma(tr, Periods)
        atr[:] = sma(tr, period)

    basic_up = src - multiplier * atr
    basic_down = src + multiplier * atr

    for i in range(n):
        if np.isnan(atr[i]):
            # ATR計算開始前。Pineのtrend初期値は1。
            if i > 0:
                trend[i] = trend[i - 1]
            continue

        if i == 0 or np.isnan(up[i - 1]):
            # Pine:
            # up1 = nz(up[1], up)
            previous_up = basic_up[i]
        else:
            previous_up = up[i - 1]

        if i == 0 or np.isnan(down[i - 1]):
            # Pine:
            # dn1 = nz(dn[1], dn)
            previous_down = basic_down[i]
        else:
            previous_down = down[i - 1]

        if i == 0:
            previous_close = close[i]
            previous_trend = 1
        else:
            previous_close = close[i - 1]
            previous_trend = trend[i - 1]

        # Pine:
        # up := close[1] > up1 ? max(up, up1) : up
        if previous_close > previous_up:
            up[i] = max(basic_up[i], previous_up)
        else:
            up[i] = basic_up[i]

        # Pine:
        # dn := close[1] < dn1 ? min(dn, dn1) : dn
        if previous_close < previous_down:
            down[i] = min(basic_down[i], previous_down)
        else:
            down[i] = basic_down[i]

        # Pine:
        # trend := trend == -1 and close > dn1 ? 1 :
        #          trend == 1 and close < up1 ? -1 : trend
        current_trend = previous_trend

        if previous_trend == -1 and close[i] > previous_down:
            current_trend = 1
        elif previous_trend == 1 and close[i] < previous_up:
            current_trend = -1

        trend[i] = current_trend

        buy_signal[i] = (
            current_trend == 1
            and previous_trend == -1
        )

        sell_signal[i] = (
            current_trend == -1
            and previous_trend == 1
        )

        change_signal[i] = current_trend != previous_trend

        if current_trend == 1:
            supertrend_line[i] = up[i]
        else:
            supertrend_line[i] = down[i]

    return {
        "atr": atr,
        "up": up,
        "down": down,
        "supertrend": supertrend_line,
        "trend": trend,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
        "change_signal": change_signal,
    }