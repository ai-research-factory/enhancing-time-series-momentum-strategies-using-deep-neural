# Cycle 3: Single-Period Training and Evaluation

## Overview
Implemented the Deep Momentum Network (LSTM-based) and trained it on real ETF data using a differentiable Sharpe ratio loss function. Evaluated on an out-of-sample test period using a simple 80/20 chronological split.

## Review Feedback Addressed (from Cycle 2)
1. **Future data filtering**: Added defensive filtering in `src/data.py` to exclude data beyond current date.
2. **Log returns**: Changed return calculation from simple `pct_change()` to `np.log(close / close.shift(1))`.
3. **Parquet format**: Switched from pickle to parquet for processed data storage.

## Model Architecture
- **DeepMomentumNetwork**: 2-layer LSTM (hidden_size=32) with tanh output
- **Input**: Past 20 days of log returns for 3 assets (SPY, TLT, GLD)
- **Output**: Position signals in [-1, 1] for each asset
- **Loss**: Negative Sharpe ratio (differentiable proxy)
- **Parameters**: 13,283 trainable parameters
- **Optimizer**: Adam (lr=1e-3) with gradient clipping (max_norm=1.0)

## Data Split
- **Train**: 3,001 samples (~2011-2023)
- **Test**: 751 samples (2023-03-30 to 2026-03-27)

## Results (Gross, No Transaction Costs)
| Metric | Value |
|---|---|
| Sharpe Ratio | 0.7648 |
| Annual Return | 4.23% |
| Max Drawdown | -6.72% |
| Hit Rate | 53.66% |
| Cumulative Return | 13.15% |

## Observations
- The model converges quickly; training loss stabilizes around epoch 20.
- The Sharpe ratio of 0.76 is reasonable for a gross (no-cost) single-period evaluation.
- The modest max drawdown of -6.72% suggests conservative position sizing.
- Hit rate above 50% indicates the model has learned directional signal.
- Transaction costs and walk-forward validation are deferred to later phases.

## Files Produced
- `src/model.py` — DeepMomentumNetwork and SharpeRatioLoss
- `scripts/train_single.py` — Training and evaluation script
- `reports/cycle_3/single_run_metrics.json` — Detailed metrics
- `reports/cycle_3/metrics.json` — Standard ARF metrics schema
- `reports/cycle_3/equity_curve_gross.png` — Cumulative return plot
