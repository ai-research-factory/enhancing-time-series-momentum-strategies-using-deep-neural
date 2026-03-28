# Open Questions

## Data
- The paper's design brief mentions 5 ETFs (SPY, TLT, GLD, UUP, EFA), but the Phase 2 task specifies only 3 (SPY, TLT, GLD). Phase 9 (Universe Expansion) will address adding more assets.
- ARF Data API returns ~15 years of data starting from 2011-03-29. The exact start date depends on ETF inception and API availability. This is sufficient for walk-forward validation.
- The paper may use different data sources or exact date ranges; this implementation uses ARF Data API as required by project rules.

## Model
- The paper's LSTM architecture specifics (exact hidden size, layers) are not fully specified. Using hidden_size=32, 2 layers as reasonable defaults.
- The lookback window of 20 days is a common choice; the paper may use different values. To be explored in hyperparameter optimization (Phase 7).
- Current evaluation uses a single 80/20 split. Walk-forward validation (Phase 5) is needed for robust assessment.
- Gross Sharpe of 0.76 is preliminary; net-of-cost performance (Phase 4) will be the true test.
