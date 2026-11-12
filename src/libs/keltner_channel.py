"""NumPy port of the supplied Pine v3 'Keltner Channel signals'.

Dependencies: Python >= 3.10, numpy. No pandas / TA-Lib.

Usage::
    from keltner_channel import keltner_channel, Settings
    result = keltner_channel(high, low, close)
    buy_indices = np.flatnonzero(result['buy'])
    result = keltner_channel(high, low, close,
                            settings=Settings(moving_average_length=34))

All arrays are chronological (oldest first). Source defaults to close.
Returns KC mid/range/four boundaries; BB-like signal bands and bbr;
support/resistance, buy/sell and their marker prices; WaveTrend and background
conditions. Horizontal levels start at ZERO, as in the supplied script.
Buy/sell markers are on the DETECTION bar, while level prices use high/low[1].
They implement cross(level, level[1]), NOT simply 'level changed'.
WaveTrend does NOT filter buy/sell in the original script.

Additional derived indicators:
band_width = abs(resistance - support) / chart close (ratio, not percent).
expansion_direction = +1 when support's current horizontal segment began
before resistance's; -1 for the reverse, 0 for simultaneous starts.
A segment begins on the bar where its level first becomes nonzero or changes
price, not on the preceding high/low bar and not on the buy/sell marker.
Repeating the same level does not restart the segment. These indicators are
NaN until both levels exist. Direction compares segment ages only; it does
not additionally require that band_width increased. A zero close gives NaN
band_width without affecting the age-based direction.

Optional historical multi-timeframe mode::
    base = Bars(high, low, close, close_time=base_close_times)
    hourly = Bars(h1_high, h1_low, h1_close, close_time=h1_close_times)
    result = calculate(base, signal_bars=hourly)

Bars close_time must use the SAME units/timezone (e.g. UTC datetime64[ns] or
Unix seconds). Provide completed candles, including warmup history. kc_bars,
signal_bars and wt_bars replace the original three independent timeframe
switches. No timestamps or resampling/session guessing is needed in default
single-timeframe mode. For MTF, pass correctly session-aligned OHLC data;
confirmed values are merged by close_time, then forward-filled (gaps_off).
Only equal/higher timeframes are intended. Developing realtime HTF values,
lower-timeframe security sampling and TradingView nested-security quirks are
not emulated. In the combined KC+signal MTF case, KC mid is computed on kc_bars
and merged into signal_bars before constructing signal bands.

Fidelity: EMA starts at first non-NaN sample; TR first sample is NaN (Pine tr,
not tr(true)); rolling std uses ddof=0. KAMA/MAMA retain original constants
and zero seeds; MAMA's length parameter is intentionally unused in source.
Finite OHLC is required. NaNs generated during indicator warmup are preserved.
No export from TradingView was supplied, so numerical parity is unverified.
Results depend on history start, source and broker/session data.

Original source header credits:
https://in.tradingview.com/script/vXakWc27-Keltner-Channel-APEX/
Pine v3 security semantics:
https://www.tradingview.com/pine-script-docs/v3/essential/context-switching-the-security-function/
"""
from dataclasses import dataclass
import numpy as np


MA_TYPES = ('SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'MAMA', 'T3')


def _array(values):
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError('Expected a one-dimensional array')
    return out


def shift(values, bars=1):
    """Pine x[bars]; no wraparound."""
    x = _array(values)
    if not isinstance(bars, (int, np.integer)) or bars < 0:
        raise ValueError('bars must be a nonnegative integer')
    out = np.full(x.size, np.nan)
    if bars == 0:
        return x.copy()
    if bars < x.size:
        out[bars:] = x[:-bars]
    return out


def nz(x):
    return np.where(np.isnan(x), 0.0, x)


def divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=b != 0)
    return out


def _length(length):
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError('length must be a positive integer')
    return int(length)


def rolling(values, length, kind='mean'):
    """Pine-style MA/std: latest length non-NaN observations (NaNs ignored)."""
    x, length = _array(values), _length(length)
    out = np.full(x.size, np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if valid.size < length:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x[valid], length)
    if kind == 'mean':
        computed = windows.mean(axis=1)
    elif kind == 'std':
        computed = windows.std(axis=1, ddof=0)
    elif kind == 'wma':
        weights = np.arange(1, length + 1, dtype=float)
        computed = windows @ (weights / weights.sum())
    else:
        raise ValueError('Unknown rolling operation')
    out[valid[length - 1:]] = computed
    # Pine ignores NaN inputs; after initialization, retain latest rolling value.
    last = np.maximum.accumulate(np.where(~np.isnan(out), np.arange(x.size), -1))
    known = last >= 0
    out[known] = out[last[known]]
    return out


def ema(values, length):
    x, length = _array(values), _length(length)
    out = np.full(x.size, np.nan)
    alpha, previous = 2.0 / (length + 1), np.nan
    for i, value in enumerate(x):
        if not np.isnan(value):
            previous = value if np.isnan(previous) else alpha * value + (1-alpha) * previous
        out[i] = previous
    return out


def _pine_sum(values, length):
    """sum() propagates NaN within the window (unlike MA functions)."""
    x = _array(values)
    out = np.full(x.size, np.nan)
    if x.size >= length:
        out[length-1:] = np.lib.stride_tricks.sliding_window_view(x, length).sum(axis=1)
    return out


def kama(values, length):
    x, length = _array(values), _length(length)
    noise = _pine_sum(np.abs(x - shift(x)), length)
    efficiency = divide(np.abs(x - shift(x, length)), noise)
    efficiency = np.where(noise == 0, 0.0, efficiency)
    smoothing = (efficiency * (0.666 - 0.0645) + 0.0645) ** 2
    out = np.full(x.size, np.nan)
    for i in range(x.size):
        previous = 0.0 if i == 0 or np.isnan(out[i-1]) else out[i-1]
        out[i] = previous + smoothing[i] * (x[i] - previous)
    return out


def mama(values, length):
    """Literal supplied Ehlers implementation; length is unused in Pine."""
    x = _array(values)
    _length(length)
    n = x.size
    sp = (4*x + 3*shift(x) + 2*shift(x, 2) + shift(x, 3)) / 10
    names = ('dt', 'q1', 'i1', 'i2', 'q2', 're', 'im', 'p1', 'p3', 'p', 'phase', 'mama')
    state = {name: np.full(n, np.nan) for name in names}
    def prev(name, i, lag=1):
        value = state[name][i-lag] if i >= lag else np.nan
        return 0.0 if np.isnan(value) else value
    def hilbert(series, i):
        def old(lag):
            value = series[i-lag] if i >= lag else np.nan
            return 0.0 if np.isnan(value) else value
        return .0962*series[i] + .5769*old(2) - .5769*old(4) - .0962*old(6)
    pi = 3.1415926
    for i in range(n):
        factor = .075 * prev('p', i) + .54
        state['dt'][i] = hilbert(sp, i)*factor
        state['q1'][i] = hilbert(state['dt'], i)*factor
        state['i1'][i] = prev('dt', i, 3)
        ji = hilbert(state['i1'], i)*factor
        jq = hilbert(state['q1'], i)*factor
        i1, q1 = state['i1'][i], state['q1'][i]
        i2 = .2*(i1-jq) + .8*prev('i2', i)
        q2 = .2*(q1+ji) + .8*prev('q2', i)
        state['i2'][i], state['q2'][i] = i2, q2
        re = .2*(i2*prev('i2', i)+q2*prev('q2', i)) + .8*prev('re', i)
        im = .2*(i2*prev('q2', i)-q2*prev('i2', i)) + .8*prev('im', i)
        state['re'][i], state['im'][i] = re, im
        if np.isfinite(im) and np.isfinite(re) and im != 0 and re != 0:
            p1 = 2*pi/np.arctan(im/re)
        else:
            p1 = prev('p', i)
        state['p1'][i] = p1
        old_p1 = prev('p1', i)
        p2 = 1.5*old_p1 if p1 > 1.5*old_p1 else (.67*old_p1 if p1 < .67*old_p1 else p1)
        p3 = 6.0 if p2 < 6 else (50.0 if p2 > 50 else p2)
        state['p3'][i] = p3
        state['p'][i] = .2*p3 + .8*prev('p3', i)
        phase = 180/pi*np.arctan(q1/i1) if i1 != 0 else np.nan
        state['phase'][i] = phase
        dphase0 = prev('phase', i) - phase
        dphase = 1.0 if dphase0 < 1 else dphase0
        alpha0 = .5/dphase
        alpha = .05 if alpha0 < .05 else (.5 if alpha0 > .5 else alpha0)
        state['mama'][i] = alpha*x[i] + (1-alpha)*prev('mama', i)
    return state['mama']


def moving_average(values, length, kind='EMA'):
    x, length = _array(values), _length(length)
    kind = kind.upper()
    if kind == 'SMA':
        return rolling(x, length)
    if kind == 'WMA':
        return rolling(x, length, 'wma')
    if kind == 'TRIMA':
        return rolling(rolling(x, (length+1)//2), length//2+1)
    if kind == 'KAMA':
        return kama(x, length)
    if kind == 'MAMA':
        return mama(x, length)
    if kind not in MA_TYPES:
        raise ValueError(f'Unknown moving average: {kind}')
    e1 = ema(x, length)
    if kind == 'EMA':
        return e1
    e2, e3 = ema(e1, length), None
    if kind == 'DEMA':
        return 2*e1-e2
    e3 = ema(e2, length)
    if kind == 'TEMA':
        return 3*(e1-e2)+e3
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    b = .7
    return -b**3*e6 + (3*b*b+3*b**3)*e5 + (-6*b*b-3*b-3*b**3)*e4 + (1+3*b+b**3+3*b*b)*e3


def crossover(a, b):
    a, b = _array(a), _array(b)
    return (a > b) & (shift(a) <= shift(b))


def crossunder(a, b):
    a, b = _array(a), _array(b)
    return (a < b) & (shift(a) >= shift(b))


def level_signals(bbr, high, low, overbought=1.0, oversold=0.0):
    """Exact strict comparisons and delayed-level crossover from Pine."""
    bbr, high, low = map(_array, (bbr, high, low))
    if not (bbr.size == high.size == low.size):
        raise ValueError('Array lengths must match')
    upper_return = (shift(bbr) > overbought) & (bbr < overbought)
    lower_return = (shift(bbr) < oversold) & (bbr > oversold)
    resistance, support = np.zeros(bbr.size), np.zeros(bbr.size)
    for i in range(bbr.size):
        old_r = 0.0 if i == 0 or np.isnan(resistance[i-1]) else resistance[i-1]
        old_s = 0.0 if i == 0 or np.isnan(support[i-1]) else support[i-1]
        resistance[i] = high[i-1] if upper_return[i] else old_r
        support[i] = low[i-1] if lower_return[i] else old_s
    buy = crossover(support, shift(support)) | crossunder(support, shift(support))
    sell = crossover(resistance, shift(resistance)) | crossunder(resistance, shift(resistance))
    return dict(support=support, resistance=resistance, buy=buy, sell=sell,
                lower_return=lower_return, upper_return=upper_return)


def support_resistance_indicators(resistance, support, close):
    """NumPy width and segment-age direction, using the original zero seed.

    Segment starts are detected on the full history before display cropping.
    NaN/zero levels are unavailable; an invalid level restarts on reappearance.
    """
    resistance, support, close = map(_array, (resistance, support, close))
    if not (resistance.size == support.size == close.size):
        raise ValueError('Array lengths must match')
    indices = np.arange(close.size)
    starts, available = [], []
    for levels in (resistance, support):
        valid = np.isfinite(levels) & (levels != 0)
        previous = shift(levels)
        changed = valid & ((levels != previous) | ~np.isfinite(previous) | (previous == 0))
        starts.append(np.maximum.accumulate(np.where(changed, indices, -1)))
        available.append(valid)
    ready = available[0] & available[1]
    width = np.full(close.size, np.nan)
    np.divide(np.abs(resistance - support), close, out=width,
              where=ready & np.isfinite(close) & (close != 0))
    direction = np.where(ready, np.sign(starts[0] - starts[1]), np.nan)
    return dict(band_width=width, expansion_direction=direction)


@dataclass(frozen=True)
class Settings:
    moving_average_length: int = 34
    moving_average_type: str = 'EMA'
    atr_multiplier_min: float = 1.5
    atr_multiplier_max: float = 3.5
    atr_type: str = 'EMA'
    atr_length: int = 88
    length: int = 34
    multiplier: float = 2.0
    overbought: float = 1.0
    oversold: float = 0.0  # Preserve Pine default despite its inconsistent minval=1.
    wt_channel_len: int = 10
    wt_ma_len: int = 3
    wt_smooth_len: int = 3
    wt_overbought: float = 90.0
    wt_oversold: float = -90.0
    wt_show_cross: bool = False
    wt_show_oversold: bool = False
    wt_show_overbought: bool = False

    def __post_init__(self):
        for name in ('moving_average_length', 'atr_length', 'length', 'wt_channel_len', 'wt_ma_len', 'wt_smooth_len'):
            _length(getattr(self, name))
        for name in ('moving_average_type', 'atr_type'):
            if getattr(self, name).upper() not in MA_TYPES:
                raise ValueError(f'Invalid {name}')
        for name in ('atr_multiplier_min', 'atr_multiplier_max', 'multiplier', 'overbought', 'oversold', 'wt_overbought', 'wt_oversold'):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f'{name} must be finite')
        if min(self.atr_multiplier_min, self.atr_multiplier_max) < 0 or self.multiplier < 1:
            raise ValueError('ATR multipliers must be >= 0 and multiplier >= 1')


@dataclass(frozen=True)
class Bars:
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    source: np.ndarray | None = None
    close_time: np.ndarray | None = None

    def __post_init__(self):
        for name in ('high', 'low', 'close'):
            object.__setattr__(self, name, _array(getattr(self, name)))
        object.__setattr__(self, 'source', self.close if self.source is None else _array(self.source))
        for name in ('high', 'low', 'close', 'source'):
            arr = getattr(self, name)
            if arr.size != self.close.size or not np.all(np.isfinite(arr)):
                raise ValueError('OHLC/source must have equal lengths and finite values')
        if np.any(self.high < self.low) or np.any(self.close > self.high) or np.any(self.close < self.low):
            raise ValueError('Require low <= close <= high')
        if self.close_time is not None:
            t = np.asarray(self.close_time)
            if t.ndim != 1 or t.size != self.close.size or np.any(t[1:] <= t[:-1]):
                raise ValueError('close_time must match candles and be strictly increasing')
            if np.issubdtype(t.dtype, np.datetime64):
                if np.any(np.isnat(t)):
                    raise ValueError('close_time contains NaT')
            elif not np.issubdtype(t.dtype, np.number) or not np.all(np.isfinite(t)):
                raise ValueError('Use finite numeric or datetime64 close_time')
            object.__setattr__(self, 'close_time', t)


def _merge(values, origin, target):
    if origin is target:
        return values.copy()
    if origin.close_time is None or target.close_time is None:
        raise ValueError('MTF requires close_time on every Bars object')
    idx = np.searchsorted(origin.close_time, target.close_time, side='right') - 1
    out = np.full(target.close.size, np.nan)
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    # gaps_off: fill unavailable/NaN values with latest non-NaN mapped value.
    last = np.maximum.accumulate(np.where(~np.isnan(out), np.arange(out.size), -1))
    known = last >= 0
    out[known] = out[last[known]]
    return out


def _kc(bars, s):
    previous = shift(bars.close)
    tr = np.maximum(bars.high-bars.low,
                    np.maximum(np.abs(bars.high-previous), np.abs(bars.low-previous)))
    mid = moving_average(bars.source, s.moving_average_length, s.moving_average_type)
    width = moving_average(tr, s.atr_length, s.atr_type)
    return dict(kc_mid=mid, kc_range=width,
                kc_bottom_min=mid-width*s.atr_multiplier_min,
                kc_bottom_max=mid-width*s.atr_multiplier_max,
                kc_top_min=mid+width*s.atr_multiplier_min,
                kc_top_max=mid+width*s.atr_multiplier_max)


def _wt(bars, s):
    source = (bars.high+bars.low+bars.close)/3
    average = ema(source, s.wt_channel_len)
    channel = ema(np.abs(source-average), s.wt_channel_len)
    index = divide(source-average, .015*channel)
    wave1 = ema(index, s.wt_ma_len)
    return dict(wt1=wave1, wt2=rolling(wave1, s.wt_smooth_len))


def calculate(bars, settings=None, *, kc_bars=None, signal_bars=None, wt_bars=None):
    """Calculate on Bars; optional confirmed historical timeframe contexts."""
    s = Settings() if settings is None else settings
    kc_frame = bars if kc_bars is None else kc_bars
    signal_frame = bars if signal_bars is None else signal_bars
    wt_frame = bars if wt_bars is None else wt_bars
    kc_native = _kc(kc_frame, s)
    out = {k: _merge(v, kc_frame, bars) for k, v in kc_native.items()}
    basis = _merge(kc_native['kc_mid'], kc_frame, signal_frame)
    std = rolling(signal_frame.close, s.length, 'std')
    signal_native = dict(signal_close=signal_frame.close, signal_high=signal_frame.high,
                         signal_low=signal_frame.low, signal_basis=basis, signal_std=std)
    mapped = {k: _merge(v, signal_frame, bars) for k, v in signal_native.items()}
    out.update(mapped)
    dev = s.multiplier*out['signal_std']
    out['signal_upper'] = out['signal_basis']+dev
    out['signal_lower'] = out['signal_basis']-dev
    out['bbr'] = divide(out['signal_close']-out['signal_lower'], 2*dev)
    out.update(level_signals(out['bbr'], out['signal_high'], out['signal_low'], s.overbought, s.oversold))
    out.update(support_resistance_indicators(out['resistance'], out['support'], bars.close))
    out['buy_price'] = np.where(out['buy'], bars.low, np.nan)
    out['sell_price'] = np.where(out['sell'], bars.high, np.nan)
    out.update({k: _merge(v, wt_frame, bars) for k, v in _wt(wt_frame, s).items()})
    out['wt_cross_up'] = crossover(out['wt1'], out['wt2'])
    out['wt_overbought'] = out['wt2'] > s.wt_overbought
    out['wt_oversold'] = out['wt2'] < s.wt_oversold
    out['bg_cross'] = out['wt_cross_up'] & s.wt_show_cross
    out['bg_overbought'] = out['wt_overbought'] & s.wt_show_overbought
    out['bg_oversold'] = out['wt_oversold'] & s.wt_show_oversold
    return out


def keltner_channel(high, low, close, *, source=None, settings=None):
    """Single-timeframe NumPy-array convenience API."""
    return calculate(Bars(high, low, close, source=source), settings)

