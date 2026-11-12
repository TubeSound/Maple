# Pivot Point SuperTrend Python版

元ファイルは `backend/src/libs/pivot_supertrend.pine`（Pine v4、MPL 2.0、© LonesomeTheBlue）。
`backend/src/libs/pivot_supertrend.py` はNumPyだけで計算する。scikit-learnは不要。

## 実行

Python 3.12、リポジトリルートで実行する。

```powershell
python -m pip install -r backend/src/requirements-sr-channel.txt
python backend/src/strategy/goldrush/visualize_pivot_supertrend.py
```

引数なしで現在JSTまでの確定済みM1を読み、利用可能な最新足から直近3暦日を表示する。VSCodeからも実行できる。
元PineとAPI・CLIの既定値はPivot Period=2、ATR Factor=3、ATR Period=10。
以前の画像の設定（2, 1.6, 10）で比較する場合：

```powershell
python backend/src/strategy/goldrush/visualize_pivot_supertrend.py --atr-factor 1.6 --start "2026-10-01 07:00" --as-of "2026-10-03 06:00"
```

このコマンドをCIの描画確認にも使う。データはリポジトリ内のAxiory XAUUSD M1。
元画像のOANDAとは配信元が異なり、TradingView上の数値との照合は未実施。
計算には表示開始前30暦日の足も使う（`--warmup-days`）。centerとトレーリングラインは状態を保持するため、
履歴開始位置が違うと初期値・その後の値が異なる場合がある。30日は完全一致の保証ではない。
`--as-of`はバー終了時刻で判定し、進行中の足を含めない。

## 計算と確定時点

- 左右それぞれperiod本のピボット。出現位置からperiod本後に確定。等値の極は右側優先。
- ピボット確定時、centerを初回の価格または `(previous_center * 2 + pivot) / 3` に更新。高値・安値が同時確定した場合は高値優先。
- ATRはTrue Range、初回period本のSMA、その後Wilder RMA。
- center ± factor × ATRから、前足の終値と前足のバンドでトレーリング上下線を更新。
- 前足の上下線を現在の終値が超えるとトレンド転換、Buy/Sellを確定。
- 緑・赤の色と転換足を隠すライン表示を移植。Buy/SellラベルのY位置はPine同様トレーリングライン。
- ラベル価格は約定価格ではない。CSVの `label_price` と `signal_price`（シグナル足終値）を分ける。売買損益は計算しない。

`calc(candles, PivotSupertrendConfig(...))` は同じ長さの有限・正値OHLC配列を受け取り、同じ長さの結果配列を返す。
`pivot_high/low`、support/resistanceは確定足に格納し、未来の情報を過去へ書き戻さない。
`--show-pivots`だけは元Pineのoffset同様、マーカーを発生位置へ戻す描画。取引時点には使わない。
`--show-sr`は確定時点のS/Rを表示し、元PineのS/R描画offset=-periodは適用しない。
`--show-center`でcenterを追加表示できる。

## 周期・値幅の観察

図の下2段は追加の診断値で、Pineの売買シグナルには影響しない。
色が続く区間を1つのトレンド区間とする。まだ完了していない区間をその時点で測る：

| 列 | 意味 |
|---|---|
| wave_elapsed_bars | 転換足から現在足までの経過バー数。転換時は0 |
| wave_height | 区間開始から現在までの最高値−最安値 |
| wave_net_move | 現在終値−区間開始終値 |
| period_ratio | 経過バー数 ÷ 直前10完了区間の長さの中央値 |
| height_ratio | 現在の値幅 ÷ 直前10完了区間の値幅の中央値 |
| pivot_interval_bars | 連続して確定したピボット間のバー数（確定足だけ値あり） |
| pivot_height | 同ピボット間の絶対価格差（確定足だけ値あり） |

比較対象の件数は `--baseline-waves` で指定。履歴先頭の途中区間は基準から除外。
基準がない間や値幅中央値が0のときは比率をNaNにする。
下2段の点線1は過去中央値と同じ規模。1超はその規模を上回ったことだけを表し、「真のトレンド」の確定条件ではない。
図の周期はピボット間周期そのものではなくトレンド区間の経過時間。ピボット間の実測値はCSVで別に保存する。
周期は実際に存在するバー数で数える。休場・欠損の時間は含めず、横軸の休場時間も圧縮する。
現区間の最終的な長さ・値幅を先読みしていない。

## 出力と検証

既定の `outputs/pivot_supertrend/` にPNG、全計算履歴CSV、転換シグナルCSV、設定・件数JSONを保存。
CSVにはウォームアップも含む。`in_display`で表示区間を選び、`jst`は足開始、`bar_close_time`は確定時刻。

PowerShell：

```powershell
$env:PYTHONPATH = "backend/src"
python -m unittest discover -s backend/src/libs -p "test_pivot_supertrend.py" -v
```

Bashは `export PYTHONPATH=backend/src`。9テストで手計算のcenter/転換/ATR、同時・等値ピボット、
初期化、入力検証、未来の足を追加しても過去の結果と診断値が変わらないことを確認。
GitHub Actionsは同じrequirementsとコマンドを使用し、引数なしと明示設定の描画も実行する。
PRのChecksで `Pivot SuperTrend tests and visualization`、Artifactsで `pivot-supertrend-preview` を確認できる。


## 配列辞書API

`calc(candles: CandleDict, config=None) -> PivotSupertrendResult`。
`CandleDict`は`supertrend.py`と共通。戻り値はすべて長さNのNumPy配列で、
`PivotSupertrendResult`は`SupertrendResult`を継承して追加列を定義する。
`measure_waves(...)->WaveMetrics`も配列辞書。診断へ渡す結果の配列長を検証する。
価格・比率はfloat64、trendはint8、シグナルはbool、completed_wave_countはint64。
OHLC入力は共通検証を通し、呼出側の配列を変更しない。
[単体・複数時間足のAPI例と移行手順](sr_channel_design.md)


## 上位足のSupport / Resistanceバンド

可視化CLIは既定で `sr_channel.calc_sr()` を使用し、M1と同じ `--as-of` 時点の最終バンドを価格チャートへ重ねる。
D1は直近1暦年・赤、H4は直近1暦月・青、H1は直近5暦日・黄色。
同じ価格へ重なる線を識別できるようD1=実線、H4=破線、H1=点線とする。
`MarketData/axiory/<symbol>/D1,H4,H1` のCSVを読み、それぞれの確定足で計算する。
表示期間が数日でもS/Rの計算期間は上記の独立した窓を使う。

価格チャートはM1の周辺範囲を表示し、遠い上位足の帯を含む全レベルは右の一覧図・表に表示する。
幅ゼロのバンドは水平線として描画する。色は時間足で決まり、Support/Resistanceは表のS/Rで区別する。
状態判定は各時間足の最終終値。バンドは基準時刻の最終スナップショットを表示期間へ延長したものなので、
過去の時点でそのバンドが確定していたことは意味しない。

`pivot_supertrend_sr_bands.csv` とJSONの `sr_bands` に全バンド・色・期間・順位・強度・状態・基準時刻・使用足数を保存。
`--no-sr-bands` で上位足のバンド表示を省略できる。
従来の `--show-sr` は元PineのピボットS/R点の追加表示で、この上位足バンドとは別。

