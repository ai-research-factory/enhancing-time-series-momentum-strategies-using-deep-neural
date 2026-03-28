# Enhancing Time Series Momentum Strategies Using Deep Neural Networks

## Project ID
proj_0f9d2516

## Taxonomy
StatArb, Transformer

## Current Cycle
2

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
This paper addresses the limitations of traditional time series momentum (TSMOM) strategies, which rely on manually defined, static rules for signal generation (e.g., lookback periods, moving average types). These heuristics may not be optimal or adaptive to changing market conditions. The paper proposes a 'Deep Momentum Network', a hybrid model that utilizes Long Short-Term Memory (LSTM) networks to learn complex temporal patterns from historical price data. The network is trained end-to-end by directly optimizing the portfolio's Sharpe ratio, allowing it to generate data-driven, dynamic trend estimation and position sizing signals, thereby aiming for superior risk-adjusted returns compared to conventional TSMOM approaches.

### Datasets
A diverse set of liquid ETFs from yfinance, representing major asset classes. Initial set: SPY (US Equity), TLT (US Treasuries), GLD (Gold), UUP (US Dollar Index), EFA (Int'l Equity). To be expanded in later phases.

### Targets
The primary optimization target is the portfolio's Sharpe ratio. The model's direct output is the position size (e.g., from -1 to +1) for each asset at each time step.

### Model
The core model is a Deep Momentum Network. It uses an LSTM to process a sequence of past returns for each asset. The LSTM's output is passed through a final layer to produce a position sizing signal. The entire network is trained using a custom loss function that is a differentiable proxy for the Sharpe ratio, enabling end-to-end optimization for risk-adjusted returns.

### Training
The model is trained and evaluated using a walk-forward methodology with expanding windows to prevent look-ahead bias. For each fold, the model is trained on the training set, with hyperparameters potentially tuned on a validation set. The final evaluation is performed on the unseen out-of-sample test set. The process is repeated across multiple folds to generate a continuous out-of-sample equity curve.

### Evaluation
The primary evaluation metric is the out-of-sample Sharpe Ratio. This will be supplemented by other standard portfolio metrics, including Annualized Return, Maximum Drawdown, Calmar Ratio, and Sortino Ratio. Performance will be evaluated both gross and net of transaction costs. The Deep Momentum Network's performance will be benchmarked against a traditional TSMOM strategy (e.g., 12-month moving average signal) and a buy-and-hold strategy on an equally weighted portfolio.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## ★ 今回のタスク (Cycle 2)


### Phase 2: データパイプライン構築 [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: yfinanceから実際のETFデータを取得し、前処理を行うパイプラインを構築する。

**具体的な作業指示**:
1. `src/data.py`に`DataLoader`クラスを作成します。2. このクラスに、`yfinance.download`を用いて複数のティッカー（'SPY', 'TLT', 'GLD'）の日足OHLCVデータを取得するメソッドを実装します。期間は`period='15y'`とします。3. 取得したデータから日次リターンを計算し、NaN値を処理（例：前方フィル）する前処理ステップを追加します。4. 処理済みデータを`data/processed/`ディレクトリにpickle形式またはcsv形式で保存する機能を実装します。5. `scripts/fetch_data.py`を作成し、このデータローダーを呼び出してデータを取得・保存します。

**期待される出力ファイル**:
- src/data.py
- data/processed/assets.pkl
- scripts/fetch_data.py

**受入基準 (これを全て満たすまで完了としない)**:
- `scripts/fetch_data.py`を実行すると`data/processed/assets.pkl`が生成されること。
- 生成されたデータにNaNが含まれていないこと。




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない







## 全体Phase計画 (参考)

✓ Phase 1: コアモデルと損失関数の実装 — 合成データ上で動作するLSTMモデルと微分可能Sharpe比損失関数を実装する。
→ Phase 2: データパイプライン構築 — yfinanceから実際のETFデータを取得し、前処理を行うパイプラインを構築する。
  Phase 3: 単一期間での学習と評価 — 実データを用いてモデルを学習させ、単一のテスト期間でパフォーマンスを評価する。
  Phase 4: 取引コストモデルの実装 — 取引コストを考慮したバックテストエンジンを実装し、ネットパフォーマンスを評価する。
  Phase 5: ウォークフォワード検証フレームワーク — 時系列データの評価に適したウォークフォワード検証を実装する。
  Phase 6: ベースライン戦略との比較 — 伝統的な時系列モメンタム戦略をベースラインとして実装し、DNNモデルと比較する。
  Phase 7: ハイパーパラメータ最適化 — Optunaを使い、論文で示唆される範囲近傍で主要なハイパーパラメータを最適化する。
  Phase 8: ロバスト性検証 — 取引コストとウォークフォワード分割数に対する戦略の感度を分析する。
  Phase 9: ユニバース拡大検証 — より多くの資産を含む拡大ユニバースでモデルのパフォーマンスを検証する。
  Phase 10: 代替モデルアーキテクチャの探求 — 論文で言及されていないTransformerベースのモデルを実装し、LSTMモデルと比較する。
  Phase 11: 統合レポートと可視化 — 全フェーズの結果をまとめた総合的なMarkdownレポートを生成する。
  Phase 12: 最終化とエグゼクティブサマリー — コードベースをクリーンアップし、非技術者向けの要約を作成する。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項
- 未来情報を特徴量やシグナルに使わない
- 全サンプル統計でスケーリングしない (train-onlyで)
- テストセットでハイパーパラメータを調整しない
- コストなしのgross PnLだけで判断しない
- 時系列データにランダムなtrain/test splitを使わない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_2/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_2/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新（セットアップ手順、主要な結果、使い方など）
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合（エラー、データ不足、期間の短さ等）
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
