# Madness設計仕様

## 1. 目的

Madnessは、トレンドフォロー戦略が苦手とする「泥んこ相場」を検出し、一定期間の新規エントリーを停止する相場状態ゲートである。

回避対象は次の2種類。

1. 大きな価格スイングの直後
2. Supertrendの方向転換が短周期で繰り返される期間

Madnessは売買戦略でも損切り機能でもない。既存ポジションを決済せず、エントリー可否だけを出力する。

## 2. 本番用ゲートと研究用分類器

`src/libs/madness.py` には用途の異なる2系統がある。

| 系統 | API | 用途 | 最終状態 |
|---|---|---|---|
<<<<<<< HEAD
| 本番用 | `madness_entry_gate` / `MadnessGateParam` | Marsのレンジ・トレンド状態判定 | `madness_range_state` |
| 研究用 | `Madness` / `MadnessParam` | M5・M15特徴量による泥相場分類 | `muddy` |
| 教師ラベル用 | `price_regime_oracle` / `OracleParam` | 将来4時間から教師ラベルを作る | `PriceRegime` |

本番運用で使う最終状態名は **`madness_range_state`** と **`madness_trend_state`** である。`muddy` と `PriceRegime` は研究・教師データ用であり、本番の注文判定には使わない。
=======
| 本番用 | `madness_entry_gate` / `MadnessGateParam` | Marsの新規エントリー停止 | `madness_blocked` |
| 研究用 | `Madness` / `MadnessParam` | M5・M15特徴量による泥相場分類 | `muddy` |
| 教師ラベル用 | `price_regime_oracle` / `OracleParam` | 将来4時間から教師ラベルを作る | `PriceRegime` |

本番運用で使う最終状態名は **`madness_blocked`** である。`muddy` と `PriceRegime` は研究・教師データ用であり、本番の注文判定には使わない。
>>>>>>> 4fcca585e1c414b0babc00f003a2ccc2873491da

## 3. 本番用ゲートの処理

```mermaid
flowchart TD
    A["確定した元時間足"] --> B["完全なM15足へ集約"]
    B --> C["Supertrend反転検出"]
    C --> D["反転周期・変化率"]
    D --> E{"大スイング または 短周期連続"}
    E -->|Yes| F["停止時間を設定・延長"]
    E -->|No| G["残り時間を更新"]
<<<<<<< HEAD
    F --> H["madness_range_state"]
=======
    F --> H["madness_blocked"]
>>>>>>> 4fcca585e1c414b0babc00f003a2ccc2873491da
    G --> H
```

### 3.1 M15集約

`madness_entry_gate` は入力足を内部でM15へ集約する。

- 入力時刻は足の開始時刻
- `label="right", closed="left"` でM15 OHLCを生成
- 必要本数が揃わないM15足は破棄
- M15足は終了境界に到達してから元時間足へ反映
- 未確定M15足や未来データは使用しない

入力時間足は15分以下で、15分を割り切れる必要がある。現在の最適化済み運用はM1である。

### 3.2 Supertrend反転

完全なM15足に `src/libs/supertrend.py` の `supertrend` を適用し、`change_signal` を反転イベントとして使う。

反転が発生すると、前回反転から次を計算する。

\[
cycle\_hours = (現在のM15番号 - 前回反転のM15番号) \times 0.25
\]

\[
swing\_pct = \left|\frac{現在価格}{前回反転価格}-1\right| \times 100
\]

### 3.3 大きなスイング

次を満たすと大スイングイベントになる。

```text
swing_pct >= swing_threshold_pct
```

検出時点から `large_swing_hold_hours` の間、エントリーを停止する。

### 3.4 短周期の連続反転

次を満たす反転を短周期とする。

```text
cycle_hours <= short_cycle_hours
```

短周期なら連続回数を1増やし、長周期なら0へ戻す。連続回数が `short_cycle_streak` 以上になると、`short_cycle_hold_hours` の間、エントリーを停止する。

### 3.5 停止期間

停止時間はM15本数へ切り上げ変換する。

```text
hold_bars = ceil(hold_hours / 0.25)
```

停止中に新しい条件を検出した場合、現在の残り本数と新しい停止本数の大きい方を採用する。したがって、停止期間は短縮されず、必要に応じて延長される。

## 4. 入力仕様

```python
madness_entry_gate(
    candles: dict[str, Any],
    source_timeframe_minutes: float,
    param: MadnessGateParam | None = None,
) -> MadnessGateResult
```

`candles` に必要な配列：

| キー | 内容 |
|---|---|
| `jst` | 有効で厳密に昇順の足開始時刻 |
| `open` | 始値 |
| `high` | 高値 |
| `low` | 安値 |
| `close` | 終値 |

全配列は同じ長さでなければならない。空配列、重複時刻、逆順時刻、15分を割り切れない時間足はエラーになる。

## 5. パラメータ

本番用パラメータは `MadnessGateParam` に定義される。Marsでは同じ値を `madness_` 接頭辞付きで保持する。

| `MadnessGateParam` | Mars側 | 意味 |
|---|---|---|
| `supertrend_period` | `madness_supertrend_period` | SupertrendのATR期間 |
| `supertrend_multiplier` | `madness_supertrend_multiplier` | SupertrendのATR倍率 |
| `swing_threshold_pct` | `madness_swing_threshold_pct` | 大スイングとする変化率[%] |
| `short_cycle_hours` | `madness_short_cycle_hours` | 短周期とする最大反転間隔[h] |
| `short_cycle_streak` | `madness_short_cycle_streak` | 停止開始に必要な短周期連続回数 |
| `large_swing_hold_hours` | `madness_large_swing_hold_hours` | 大スイング後の停止時間[h] |
| `short_cycle_hold_hours` | `madness_short_cycle_hold_hours` | 短周期検出後の停止時間[h] |

`src/strategy/mars/params/mars_params.json` が銘柄別の最適値を保持する。

## 6. 出力仕様

戻り値は入力足と同じ長さのNumPy配列である。

| キー | 型 | 意味 |
|---|---|---|
<<<<<<< HEAD
| `madness_range_state` | bool | 最終レンジ状態。`True`なら新規エントリー停止 |
| `madness_trend_state` | bool | 最終トレンド状態。`not madness_range_state` |
=======
| `madness_blocked` | bool | 最終停止状態。`True`なら新規エントリー停止 |
| `madness_entry_allowed` | bool | `not madness_blocked` |
>>>>>>> 4fcca585e1c414b0babc00f003a2ccc2873491da
| `madness_large_swing` | bool | 大スイング検出イベント |
| `madness_short_cycle` | bool | 短周期反転イベント |
| `madness_cycle_hours` | float | 直近反転周期[h] |
| `madness_swing_pct` | float | 直近反転間の絶対変化率[%] |
| `madness_short_cycle_streak` | int | 短周期の連続回数 |

<<<<<<< HEAD
`madness_large_swing` と `madness_short_cycle` は、M15イベントが元時間足へ初めて見えた1本だけ `True` になる。`madness_range_state` は停止期間中ずっと `True` になる。

完全なM15足がまだ存在しない場合はフェイルオープンとなり、`madness_range_state=False`、`madness_trend_state=True` を返す。
=======
`madness_large_swing` と `madness_short_cycle` は、M15イベントが元時間足へ初めて見えた1本だけ `True` になる。`madness_blocked` は停止期間中ずっと `True` になる。

完全なM15足がまだ存在しない場合はフェイルオープンとなり、`madness_blocked=False`、`madness_entry_allowed=True` を返す。
>>>>>>> 4fcca585e1c414b0babc00f003a2ccc2873491da

## 7. Marsへの組み込み

Marsはゲート適用前後で別々にシグナル状態機械を計算する。

```text
mars_entries = Madness適用前のMarsエントリー
<<<<<<< HEAD
entries      = madness_trend_stateを適用したエントリー
```

`madness_range_state=True` でも既存ポジションの決済シグナルは抑止しない。停止対象は初回エントリーと積み増しの両方である。
=======
entries      = madness_entry_allowedを適用したエントリー
```

`madness_blocked=True` でも既存ポジションの決済シグナルは抑止しない。停止対象は初回エントリーと積み増しの両方である。
>>>>>>> 4fcca585e1c414b0babc00f003a2ccc2873491da

利用側は新規注文に `entries` を使う。`mars_entries` はMadnessが何を抑止したかを確認する監視用である。

## 8. 因果性と再計算

- 判定は確定した価格データだけを使う
- M15イベントはM15終了後にだけ見える
- 入力履歴が同じなら、再起動後も状態を再計算できる
<<<<<<< HEAD
- `madness_range_state` 自体を永続化する必要はない
=======
- `madness_blocked` 自体を永続化する必要はない
>>>>>>> 4fcca585e1c414b0babc00f003a2ccc2873491da
- 足履歴の先頭が変わると前回反転や連続回数も変わり得るため、十分なウォームアップを与える

## 9. 研究用API

### `Madness` / `MadnessParam`

M5・M15 Supertrendの過去4時間の反転回数、価格経路効率、方向比率を特徴量にし、`muddy` と `mud_score` を出力する。これは初期原理検証用であり、現在のMars本番ゲートとは別系統である。

### `price_regime_oracle`

将来4時間の価格経路を使用して `UNCERTAIN / MUDDY / TREND` の教師ラベルを作る。未来を参照するため、バックテストの売買判定や本番運用には使用禁止である。

## 10. 最適化方針

本番用ゲートは次の二段階で最適化した。

1. 直近半年を月ごとに走査し、各月の上位候補を抽出
2. 重複を除いた候補を全期間で再評価

最終順位は次の優先順。

1. 最低利益・最低取引数を満たす `eligible`
2. `quality`
3. `net_profit`

\[
quality = \frac{net\_profit}
{|max\_drawdown| + ulcer\_index + downside\_rms\sqrt{N}}
\]

正の利益ジャンプは罰せず、最大ドローダウン、長く続くドローダウン、損失ジャンプを抑える設計である。

## 11. テスト観点

1. M15未完成足が判定へ混入しない
2. 欠損を含むM15区間が除外される
3. 大スイング検出時に所定時間ブロックする
4. 短周期が規定回数続いた時だけブロックする
5. 長周期反転で短周期連続回数を0へ戻す
6. 停止中の再検出で残り時間を正しく延長する
7. 出力配列が入力と同じ長さ・時刻に整列する
<<<<<<< HEAD
8. `madness_trend_state == ~madness_range_state` が常に成立する
=======
8. `madness_entry_allowed == ~madness_blocked` が常に成立する
>>>>>>> 4fcca585e1c414b0babc00f003a2ccc2873491da
9. ブロック中もMarsの決済シグナルが維持される
10. 教師ラベル用Oracleが本番経路から呼ばれない

## 12. 運用上の注意

<<<<<<< HEAD
- 状態監視には `madness_range_state` を使う
=======
- 状態監視には `madness_blocked` を使う
>>>>>>> 4fcca585e1c414b0babc00f003a2ccc2873491da
- ブロック解除だけを理由に注文せず、新しい戦略シグナルを待つ
- 銘柄ごとに停止率、抑止率、取引数、取り逃した利益を監視する
- NGASの最適値は停止率が約97.6%なので、実運用前に再確認する
- 障害時は銘柄設定の `madness_enabled=false` でMars単体へ戻せる
