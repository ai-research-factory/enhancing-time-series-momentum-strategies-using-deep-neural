# Open Questions

## Data
- The paper's design brief mentions 5 ETFs (SPY, TLT, GLD, UUP, EFA), but the Phase 2 task specifies only 3 (SPY, TLT, GLD). Phase 9 (Universe Expansion) will address adding more assets.
- ARF Data API returns ~15 years of data starting from 2011-03-29. The exact start date depends on ETF inception and API availability. This is sufficient for walk-forward validation.
- The paper may use different data sources or exact date ranges; this implementation uses ARF Data API as required by project rules.

## Model
- The paper's LSTM architecture specifics (exact hidden size, layers) are not fully specified. Using hidden_size=32, 2 layers as reasonable defaults.
- The lookback window of 20 days is a common choice; the paper may use different values. To be explored in hyperparameter optimization (Phase 7).
- Current evaluation uses a single 80/20 split. Walk-forward validation (Phase 5) is needed for robust assessment.

## Transaction Costs (Cycle 4)
- The model learns near-static positions, resulting in minimal turnover and negligible cost impact (Sharpe drops from 0.76 to 0.76 net). Walk-forward validation with retraining will produce more realistic turnover.
- Current cost model is proportional only (15 bps total). Does not model market impact, bid-ask spreads, or time-varying liquidity.
- The paper may use different cost assumptions; 10 bps fee + 5 bps slippage are ARF defaults.

## Walk-Forward Validation (Cycle 5)
- 5 windows may not provide enough statistical power. The paper may use more windows or different window sizing. Phase 8 (robustness) will test sensitivity to the number of splits.
- Window 4 (2021-01 to 2023-08) is the only negative Sharpe window. This period includes the 2022 simultaneous equity/bond drawdown — an unusual regime where traditional momentum signals on both SPY and TLT failed.
- The expanding window approach means Window 1 trains on ~2 years but Window 5 trains on ~12 years. This asymmetry could bias performance comparisons across windows.
- Combined OOS Sharpe (0.43) is lower than per-window average (0.68) due to the long negative Window 4 dragging down the aggregate series.
