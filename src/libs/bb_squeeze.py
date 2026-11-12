import numpy as np

def bb_squeeze(high, low, close, bb_period=20, bb_multiply=2.0, kc_period=20, kc_multiply=1.5, use_true_range=True):
    n = len(close)
    
    # 1. Bollinger Bands (ボリンジャーバンド) の計算
    basis = np.full(n, np.nan)
    dev = np.full(n, np.nan)
    for i in range(bb_period - 1, n):
        window = close[i - bb_period + 1:i + 1]
        basis[i] = np.mean(window)
        dev[i] = bb_multiply * np.std(window, ddof=0)
        
    bb_upper = basis + dev
    bb_lower = basis - dev

    # 2. Keltner Channels (ケルトナーチャネル) の計算
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
        
    range_val = tr if use_true_range else (high - low)
    
    ma = np.full(n, np.nan)
    rangema = np.full(n, np.nan)
    for i in range(kc_period - 1, n):
        ma[i] = np.mean(close[i - kc_period + 1:i + 1])
        rangema[i] = np.mean(range_val[i - kc_period + 1:i + 1])
        
    kc_upper = ma + rangema * kc_multiply
    kc_lower = ma - rangema * kc_multiply

    # 3. Squeeze State の判定 (1: スクイーズ中, 0: 非スクイーズ)
    sqzOn  = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    squeeze = np.zeros(n, dtype=int)
    squeeze[sqzOn] = 1

    # 4. Momentum Value (モメンタム値と線形回帰) の計算
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(kc_period - 1, n):
        highest_high[i] = np.max(high[i - kc_period + 1:i + 1])
        lowest_low[i] = np.min(low[i - kc_period + 1:i + 1])
        
    avg_hl = (highest_high + lowest_low) / 2.0
    avg_total = (avg_hl + ma) / 2.0
    source_diff = close - avg_total

    val = np.full(n, np.nan)
    val_upper = np.full(n, np.nan)
    val_lower = np.full(n, np.nan)
    
    x = np.arange(kc_period)
    for i in range(kc_period - 1, n):
        y = source_diff[i - kc_period + 1:i + 1]
        if not np.isnan(y).any():
            slope, intercept = np.polyfit(x, y, 1)
            val[i] = slope * (kc_period - 1) + intercept
            
            window_val = val[i - kc_period + 1:i + 1]
            if not np.isnan(window_val).any():
                val_upper[i] = np.max(window_val)
                val_lower[i] = np.min(window_val)

    # 5. モメンタムの色の状態 (state) の計算
    state = np.zeros(n, dtype=int)
    val_prev = np.full_like(val, np.nan)
    val_prev[1:] = val[:-1]

    for i in range(kc_period - 1, n):
        if np.isnan(val[i]) or np.isnan(val_prev[i]):
            continue
        
        if val[i] > 0:
            if val[i] > val_prev[i]:
                state[i] = 1   # 明るい緑 (lime)
            else:
                state[i] = 2   # 暗い緑 (green)
        else:
            if val[i] < val_prev[i]:
                state[i] = -1  # 明るい赤 (red)
            else:
                state[i] = -2  # 暗い赤 (maroon)

    # 6. val_upper / val_lower の補正処理
    is_pos = state > 0
    is_neg = state < 0

    is_pos_prev = np.full_like(is_pos, False, dtype=bool)
    is_pos_prev[1:] = is_pos[:-1]

    is_neg_prev = np.full_like(is_neg, False, dtype=bool)
    is_neg_prev[1:] = is_neg[:-1]

    cross_to_pos = is_pos & ~is_pos_prev
    cross_to_neg = is_neg & ~is_neg_prev

    # 補正済みの生数値包絡線
    val_upper_adj = np.full(n, np.nan)
    val_upper_adj[is_pos] = val_upper[is_pos]
    val_upper_adj[cross_to_pos] = 0.0

    val_lower_adj = np.full(n, np.nan)
    val_lower_adj[is_neg] = val_lower[is_neg]
    val_lower_adj[cross_to_neg] = 0.0

    # 7. 終値に対する%変換
    val_upper_pct = (val_upper / close) * 100
    val_lower_pct = (val_lower / close) * 100

    # 補正済みに対する%変換
    val_upper_adj_pct = (val_upper_adj / close) * 100
    val_lower_adj_pct = (val_lower_adj / close) * 100

    return {
        'val': val,
        'state': state,
        'squeeze': squeeze,              # 変更：1 (On), 0 (Off) のスクイーズ状態
        'ma': basis,
        'val_upper': val_upper_adj,
        'val_lower': val_lower_adj,
        'val_upper_pct': val_upper_pct,
        'val_lower_pct': val_lower_pct,
        'val_upper_adj_pct': val_upper_adj_pct,
        'val_lower_adj_pct': val_lower_adj_pct,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'kc_upper': kc_upper,
        'kc_lower': kc_lower,
    }