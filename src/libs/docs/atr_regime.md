# ATR Volatility Regime

`libs/atr_regime.py` は TradingView の
「ATR Volatility Regime [NovaLens]」の客観的な計算エンジンを
NumPyへ移植したライブラリである。

元コードのライセンスは MPL-2.0。

- 原版: <https://www.tradingview.com/script/HLMnKXog-ATR-Volatility-Regime-NovaLens/>
- TradingView固有の描画、テーブル、ローソク足着色、アラートは対象外
- Playbook Lensは解釈用レイヤーであり、レジーム計算には影響しないため対象外

## 使用例

```python
from libs.atr_regime import AtrRegimeParam, atr_volatility_regime

result = atr_volatility_regime(
    {
        "high": candles["high"],
        "low": candles["low"],
        "close": candles["close"],
    },
    AtrRegimeParam(horizon="swing"),
)
```

入力と出力は、すべて同じ長さの一次元NumPy配列である。
確定済み4時間足を渡す場合、Swingプロファイルでは
ATR 14本、順位計算200本、表示EMA 5本となる。

## プロファイル

| Horizon | ATR | 順位Lookback | 表示EMA | 上昇ヒステリシス | 下降ヒステリシス |
|---|---:|---:|---:|---:|---:|
| `intraday` | 14 | 100 | 3 | 1 | 2 |
| `swing` | 14 | 200 | 5 | 1 | 3 |
| `long-term hold` | 20 | 300 | 5 | 2 | 3 |
| `custom` | 指定値 | 指定値 | 指定値 | 1 | 3 |

## 出力の意味

`raw_rank` は現在のATR%を直前のLookback期間と比較した0～100の
パーセンタイル順位で、判定はこの値を使う。
`display_rank` は表示用にEMA平滑化した値であり、判定には使わない。

`tier` はヒステリシス適用後のボラティリティ水準を表す。

| 値 | 状態 | raw rank |
|---:|---|---|
| -1 | Warmup | 計算期間不足 |
| 0 | Compression | 0以上25未満 |
| 1 | Quiet | 25以上50未満 |
| 2 | Normal | 50以上75未満 |
| 3 | Elevated | 75以上 |

`phase` はATR順位の変化方向を表す。

| 値 | 状態 |
|---:|---|
| -1 | Warmup |
| 0 | Rising |
| 1 | Topping |
| 2 | Falling |
| 3 | Bottoming |

## イベント

- `compression_release`: Compressionが10本以上続いた後、離脱から5本以内に
  `raw_rank >= 50`へ到達
- `extreme_touch`: `raw_rank >= 90`のピークを次の足で確定
- `cooling`: Extreme Touch後10本以内に初めて`raw_rank < 75`
- `compression_entry`: ヒステリシス適用後にCompressionへ遷移

ピークそのものは`extreme_peak`、検出可能になった確定足は
`extreme_touch`に記録される。この2つは1本ずれる。

この指標は値動きの方向を判定しない。ElevatedやExtreme Touchは
上昇・下落のどちらでも発生するため、売買方向は価格トレンドなどの
別条件と組み合わせる。
