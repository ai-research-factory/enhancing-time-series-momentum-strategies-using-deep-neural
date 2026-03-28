# Cycle 4: Transaction Cost Model Implementation

## Objective
Implement a transaction cost model and evaluate the Deep Momentum Network's net-of-cost performance, comparing gross vs net metrics.

## Implementation

### Transaction Cost Model
Added `calculate_portfolio_costs()` to `src/backtest.py` to handle multi-asset portfolios:
- Computes per-asset turnover from position changes (including initial entry from zero)
- Applies proportional costs: `cost = turnover * (fee_bps + slippage_bps) / 10000`
- Aggregates costs across assets using equal-weight averaging (consistent with return aggregation)
- Returns gross returns, net returns, and trade count

### Cost Parameters
Following the ARF standard `BacktestConfig` defaults:
- Fee: 10 bps per trade (one-way)
- Slippage: 5 bps per trade (one-way)
- Total: 15 bps per unit of turnover

### Evaluation Pipeline
Extended `scripts/train_single.py` to:
1. Compute transaction costs on the test period using `calculate_portfolio_costs()`
2. Generate gross vs net equity curve comparison plot
3. Produce turnover analysis plot (daily turnover + cumulative cost drag)
4. Save cycle 4 metrics with full cost breakdown

## Results

### Performance Comparison (Test Period: 2023-03 to 2026-03)

| Metric | Gross | Net (15 bps) |
|---|---|---|
| Sharpe Ratio | 0.7648 | 0.7572 |
| Annual Return | 4.23% | 4.19% |
| Max Drawdown | -6.72% | -6.73% |
| Hit Rate | 53.66% | 53.53% |
| Cumulative Return | 13.15% | 13.01% |

### Turnover Analysis
- **Average daily turnover**: 0.0011 (per asset, per day)
- **Cumulative cost drag**: 0.15% over the full test period (~3 years)
- **Total trades** (position changes > 1%): 2 (initial entries for SPY and TLT)

### Key Observation
The LSTM learns extremely stable position signals (SPY ~0.88, TLT ~0.63, GLD ~0.00), with sub-1% daily position changes. This results in near-zero transaction costs. The gross-to-net Sharpe degradation is only 0.0076 (1.0%), indicating the model naturally produces low-turnover strategies.

This is consistent with the tanh activation on the output layer — it constrains positions to [-1, 1] and, when combined with the Sharpe ratio loss function, incentivizes consistent directional bets rather than frequent rebalancing.

## Limitations
- The current model's near-static positions may not fully exercise the cost model. Walk-forward validation (Phase 5) with retraining across windows will produce more realistic turnover patterns.
- Cost model assumes proportional costs only; does not model market impact for large positions.
- No bid-ask spread modeling or time-varying liquidity.

## Output Files
- `reports/cycle_4/metrics.json` — ARF standard metrics with transaction costs
- `reports/cycle_4/equity_curve_gross_vs_net.png` — Gross vs net cumulative returns
- `reports/cycle_4/turnover_analysis.png` — Daily turnover and cumulative cost drag
