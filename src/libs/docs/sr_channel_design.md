# SRchannel：Pine ScriptからPythonへの移植

元コード：`backend/src/libs/SRchannel.pine`（Support Resistance Channels、作者 LonesomeTheBlue、MPL-2.0）。
移植先：`backend/src/libs/sr_channel.py`。DBSCANは使用せず、元スクリプトのピボットと強度計算を使う。
Python移植ファイルもMPL-2.0。元スクリプトの著作権表示を保持する。

## 計算仕様

| Pine入力 | Python設定 | 既定値 |
|---|---|---|
| prd | pivot_period | 10 |
| ppsrc | source | High/Low |
| ChannelW | channel_width_pct | 5 |
| minstrength | min_strength | 1 |
| maxnumsr（入力値） | max_channels | 6 |
| loopback | loopback | 290 |

1. 左右`pivot_period`本でピボットを判定。高値では過去側の同値を許容し、未来側の同値は許容しない（安値は逆）。
   ピボット値を**右側の本数が揃った確認足**へ記録する。表示用に元の極へ移す場合は`pivot_period`本戻すが、売買判定へ遡及使用しない。
2. 帯の最大幅は`(直近300本の最高high − 最低low) × channel_width_pct / 100`。
   価格そのものの5%ではない。`Close/Open`でもこの幅と接触強度はhigh/lowを使う。
3. ピボットを新しい順で保持。同じ足で高値・安値が両方確定した場合は**高値だけ**を候補に追加する。
   保持期間は極の発生足ではなく確認足から数える。古いピボットの除去は新しいピボットが来たときだけ行う。
4. 各ピボットを種にし、新しい順に、帯全体が最大幅に収まるピボットを追加する。
   強度は`20 × 所属ピボット数 + highまたはlowが帯内にある足数`。
   接触は直近`loopback + 1`本（当足を含む）で、highとlowが両方帯内でも1点。
   帯を跨いだだけでhighもlowも帯外の足は加点しない。
5. 強度`min_strength × 20`以上から最大10帯を強い順に選択。
   同点では元配列の先頭が優先。選択帯の中に上端または下端がある候補を除外する。
   区間の交差すべてを除外する処理や、帯を統合する処理には変更しない。
6. 出力・上下抜け判定は上位`max_channels`帯。帯の再計算は新しいピボットの確定時のみ。
   それ以外は価格帯と強度を保持し、当足終値による色と上下抜けだけ更新する。
7. 終値がいずれの表示帯にも属さない場合にだけ上下抜けを判定。
   `前足close <= 当足upper < 当足close`がレジスタンス上抜け、
   `前足close >= 当足lower > 当足close`がサポート下抜け。
   前足の帯と比較するのではなく、両終値を**当足の帯**と比較する。境界上は帯内。

原文の`stren`ソート代入もそのまま保持した。通常は選択時点で強度降順なので交換は発生しない。
`min_strength=1`では、ピボットが1点の幅ゼロ帯も採用される。
原文ツールチップの「最低2点」には合わせず、実際の`minstrength * 20`条件を移植した。
MA1/MA2のSMA・EMAも任意に計算できる。既定は無効。色・ボックス・アラートUIは数値出力と図へ置き換えた。

## API

```python
from libs.sr_channel import SRChannelConfig, calc

result = calc(candles, SRChannelConfig())
# candles: chronological aligned arrays open/high/low/close, completed bars only
upper = result["upper"]          # (N, max_channels)
lower = result["lower"]
strength = result["strength"]
state = result["state"]          # 1=support, -1=resistance, 0=in channel, 2=unused
break_up = result["resistance_broken"]
break_down = result["support_broken"]
```

未使用帯の価格・強度はNaN。OHLCは同じ長さの有限・正価格配列とし、high/lowはopen/closeを包む。
入力の時刻順序・確定足の選別は呼出側の責任。行を黙って削除せず、不正な価格はエラーにする。
確定足ベースの処理であり、TradingViewの未確定足内の更新・ロールバックは実装していない。

## ローカル実行とCI

Python 3.12、リポジトリルートからPowerShellで実行：

```powershell
python -m pip install -r backend/requirements-sr-channel.txt
$env:PYTHONPATH = "backend/src"
python -m unittest discover -s backend/src/libs -p "test_sr_channel*.py" -v
python backend/analyze_sr_channel.py --timeframe D1 --as-of "2026-10-03 23:05" --output-dir outputs/sr_channel/D1
```

Bashは`export PYTHONPATH=backend/src`とし、同じインストール・テストコマンドを使う。
CIはPRとmainへのpushで単体14件＋合成11件＋最終バンド9件の34テストを実行する。
`--as-of`は任意。省略時は実行時の現在JSTを使用する。
`python backend/analyze_sr_channel.py`だけで実行でき、VSCodeの引数なし実行にも対応する。
過去時点の再現には明示的に`--as-of`を指定する。
CIでは引数なし実行と明示cutoffでの実行も確認する。

引数なしは`--timeframe ALL`と同じで、D1・H4・H1を独立に計算して重ね合わせる。
単独解析は`--timeframe D1`、`H4`、`H1`。単独でも同じ期間の選別を適用する。
D1は従来の全確定履歴。H4はcutoffから暦の1カ月前以降、H1は5日前以降の開始時刻の足を使う。
休場日は期間に含める。全時間足で同一cutoffを使用し、公称時間足が終了した足だけを採用する。
H4/H1の窓の前の足はウォームアップにも使わない。したがって300本の幅計算と290本の接触窓は、存在する足だけで計算する。
ピボット左右10本では最低21本が必要。足が足りない場合は帯が空でも正常出力し、利用可能本数とwarmup情報をJSONへ記録する。
窓内に確定足がない、CSVがない、時刻重複がある場合は時間足名つきで停止する。
データ既定は`MarketData/axiory/<symbol>/<TF>/<symbol>_<TF>_YYYY-MM.csv`。
全CSVを時刻順に読み、`足開始 + 公称時間足 <= as_of`のみ採用。タイムスタンプはJST表示。
CSVのjstは時差つきを想定。日足の短縮セッションは公称24時間による近似。
日足の全履歴と、指定されたH4/H1の計算窓でそれぞれ最初から計算する。表示本数`--plot-bars`は計算期間を変えない。

出力は`sr_channel.json`（最終帯）、`sr_channel_history.csv`（各確認足の帯・強度・上下抜け）、
`sr_channel.png`（ローソク足と最終帯）。
PNGはPineの`extend.both`のように最終帯を過去にも延長したスナップショットで、当時その帯が既知だったことは意味しない。
時系列検証にはCSVを使う。

## 検証範囲

14テスト（配列辞書の形状・dtype・入力不変性を含む）：極の遅延と同値、手計算した幅・強度、境界接触と二重加点防止、接触を含む閾値、
同時極の高値優先、未来追加への不変性、上下抜けと色、更新間の保持、ボディソース、
古い帯の保持、300本窓、入力検証、SMA/EMA。

既存axiory XAUUSD日足453本（2026-10-02開始の足まで）で図・JSON・CSV出力を確認。
OANDAの添付画像とaxioryでは価格と日足区切りが異なるため、価格帯の完全一致は保証しない。
TradingView内部のピボット実装、とくに同値処理の数値照合には同一OHLCでのTradingView出力が必要。
現時点は原文ロジックと手計算テストの検証であり、TradingViewとの全行一致を検証済みとはしない。

## 3時間足の合成出力

- `sr_channel_combined.png`：D1・H4・H1の3段のローソク足へ全時間足の帯を重ねる。
  実線=D1、破線=H4、点線=H1。短期段の価格軸はローソクの周辺へ固定し、遠い日足の帯で潰さない。
- `sr_channel_combined.json`：計算期間・足数・最終足・設定・帯・共通区間。
- `sr_channel_combined.csv`：時間足識別つきの元の帯。各時間足の強度は合算しない。
- `sr_channel_overlaps.csv`：2時間足以上が重なる厳密な区間。間接的に繋がるだけの帯は一つにまとめない。
- `D1/H4/H1`各フォルダ：単独スナップショットJSONと各足の履歴CSV。

合成のサポート/レジスタンス判定と色は、最後の確定H1終値へ統一。
個別スナップショットの状態は各時間足の最後の終値を基準とする。
ゼロ幅ラインは元の帯に残すが、面積を持つ共通帯の計算には加えない。
共通区間がなければ空のCSV（列名つき）になる。最終帯の過去への延長はスナップショット表示である。

合成11テスト（配列契約・空配列・時刻型・整列・年境界も含む）：月末の暦月境界、5日境界と未確定足の除外、日足履歴とUTC/JST同値、
間接重なりを橋渡ししない共通区間、窓外と未来の値への不変性、空窓と時刻重複。

## 配列辞書API（supertrend.pyと共通）

単体APIは `CandleDict`（`supertrend.py`と同じTypedDict）を受け取る。
`PivotSupertrendResult`、`SRChannelResult`、`WaveMetrics`の値はすべてNumPy配列。
OHLCはfloat64へ正規化し、入力配列を書き換えない。設定は従来どおりConfigで渡す。

```python
import numpy as np
from libs.supertrend import CandleDict, supertrend
from libs.pivot_supertrend import calc as pivot_supertrend
from libs.sr_channel import calc as sr_channel

candles: CandleDict = {
    "open": np.array([100., 101., 102.]),
    "high": np.array([102., 103., 104.]),
    "low": np.array([99., 100., 101.]),
    "close": np.array([101., 102., 103.]),
}
standard = supertrend(candles)
pivot = pivot_supertrend(candles)
channel = sr_channel(candles)
```

浮動小数はfloat64、状態/trendはint8、シグナルはbool。
単体の時系列は(N,)。S/Rのupper/lower/strength/stateだけ(N, max_channels)。
空の入力でも同じ列と次元を持つ空配列を返す。

### 複数時間足

`sr_channel_multi.calc(candles, cutoff, config=None)`も
`dict[str, numpy.ndarray]`を入力・出力とする。ネストした辞書、DataFrame、タプルを返さない。
時間足ごとに足数が違うので、キーにD1/H4/H1接頭辞をつける。

```python
from libs.sr_channel_multi import calc

# d1/h4/h1は各時間足のOHLCとjstを持つ配列辞書
candles = {
    f"{tf}_{key}": array
    for tf, data in (("D1", d1), ("H4", h4), ("H1", h1))
    for key, array in data.items()
}
result = calc(candles, cutoff="2026-10-03 23:05")
h4_upper = result["H4_upper"]
combined_lower = result["channel_lower"]
overlap_membership = result["overlap_channel_mask"]
```

| キー | 配列の意味・形状 |
|---|---|
| D1_open/high/low/close/jst（H4/H1も同様） | 選別後の確定足、(時間足別N,) |
| D1_upper/lower/strength/state 等 | 単体S/R結果と同じ列、時間足別N |
| D1_window_start（H4/H1も同様） | 窓の開始時刻、(1,)。全履歴のD1はNaT |
| channel_id/timeframe/rank/lower/upper/strength/state | 共通H1終値を基準にした最終帯の列、(帯数B,) |
| overlap_lower/upper/state/timeframe_count | 厳密な共通区間、(区間数R,) |
| overlap_timeframes | (R, 3)のbool。列順D1,H4,H1 |
| overlap_channel_mask | (R, B)のbool。channel_*の行順に対応 |
| as_of/reference_bar_open/reference_price | 共通基準時刻・H1最終足開始・基準価格、(1,) |

合成のstateも1=support、-1=resistance、0=in_channel。
帯や共通区間が空でもfloat列(0,)、時間足マスク(0,3)、帯マスク(0,B)を返す。
未使用の単体帯は従来どおりstate=2とNaN。

NumPyのdatetime64はタイムゾーンを保持できないため、
`*_jst`と出力時刻は**UTCの瞬間を表すdatetime64[ns]**で統一する。
CSVの列名jstを継承しているが、配列値をJSTの壁時計時刻として解釈しない。
CSV→配列は `pd.to_datetime(frame["jst"], utc=True).to_numpy(dtype="datetime64[ns]")`、
JST表示は `pd.to_datetime(array, utc=True).tz_convert("Asia/Tokyo")`。
cutoffは日時/ISO文字列/np.datetime64を受け取り、タイムゾーンなし日時・文字列はJST、
np.datetime64はUTCと解釈する。H4の1暦月・H1の5暦日・確定足境界はJSTで従来どおり処理する。

### 呼び出し側の移行

- `analyze_frames`はcalcの別名として残すが、入力は接頭辞付き配列辞書、戻り値は配列辞書1個へ変更。
- `select_window`はOHLC/jstの配列辞書を受け取り、選別済み配列辞書だけを返す。
  開始時刻が必要なレポートは `window_start(timeframe, cutoff)` を使う。
- `intersections`も帯の配列辞書から共通区間の配列辞書へ変更。
- DataFrame変換・状態名への変換・設定等のJSONメタデータは `sr_channel_report.py` で処理。
  CLIのPNG/CSV/JSONと実行コマンドは従来どおり。

## calc_sr：指定時点の最終バンド

`calc()`は各足の履歴を返す。`calc_sr()`は期間を選別してから同じ計算を行い、
**最終時点の有効バンドだけ**を1次元のNumPy配列辞書で返す。

```python
from libs.sr_channel import calc_sr

# candles: open/high/low/close と jst の配列辞書
# jstはUTCのdatetime64[ns]。timeframeに対応する足を渡す。
daily = calc_sr(daily_candles, "D1", as_of="2026-10-04 09:00")
h4 = calc_sr(h4_candles, "H4", as_of="2026-10-04 09:00")
h1 = calc_sr(h1_candles, "H1", as_of="2026-10-04 09:00")

# 任意の期間：既定期間を置き換える
custom = calc_sr(h4_candles, "H4", as_of="2026-10-04 09:00", months=2)
exact = calc_sr(h1_candles, "H1", as_of="2026-10-04 09:00", start="2026-09-28 09:00")

support = h1["state"] == 1
resistance = h1["state"] == -1
support_bands = (h1["lower"][support], h1["upper"][support])
resistance_bands = (h1["lower"][resistance], h1["upper"][resistance])
```

| 時間足 | calc_srの既定期間 |
|---|---|
| D1 | 基準時刻から1暦年前以降 |
| H4 | 基準時刻から1暦月前以降 |
| H1 | 基準時刻から5暦日前以降 |

`as_of`省略時は、渡されたデータの最新足開始＋公称時間足を基準にする。
データの最新足が確定済みであることは呼出側で保証する。現在時刻までの確定足だけを使いたい場合は、
現在時刻を`as_of`へ明示する。タイムゾーンなし日時・文字列はJST、np.datetime64はUTC。
`years`/`months`/`days`で任意期間を指定でき、複数指定は暦年・月を引いた後に日を引く。
いずれも非負整数で、合計0はエラー。`start`との併用はエラー。
月末や閏日は存在する最終日に合わせる（2024-02-29から1年前は2023-02-28）。

足開始が期間開始以上、かつ足開始＋公称時間足が基準時刻以下の足だけを使用。
窓の前の足は初期化にも使わず、価格は選別後に検証する。
時刻は昇順へ整列し、窓内の時刻重複やNaTはエラー。
入力足が少なければその範囲で計算し、ピボット未確定等でバンドがなければ空配列を返す。
期間内に確定足が1本もない場合はValueError。
時間足の変換や再サンプリングは行わない。

| 出力キー | 形状・意味 |
|---|---|
| upper / lower / strength | (B,)のfloat64、最終バンド上端・下端・強度 |
| state | (B,)のint8、1=Support、-1=Resistance、0=価格がバンド内 |
| rank | (B,)のint64、元の強度順位（1始まり） |
| as_of / window_start / reference_bar_open | (1,)のUTC datetime64[ns]、基準・開始・最終足開始 |
| reference_price | (1,)のfloat64、最終確定足の終値 |
| bars | (1,)のint64、使用した足数 |

未使用のNaN帯は出力から除外。価格がバンド内の場合はSupport/Resistanceへ強制分類せずstate=0を残す。
バンドの順序・強度・状態は選別した足で`calc()`を実行した最後の行と一致する。
従来の`calc()`および合成CLIのD1全履歴という仕様は変更しない。
日足1年という既定期間はこの新しい`calc_sr()`に適用する。
