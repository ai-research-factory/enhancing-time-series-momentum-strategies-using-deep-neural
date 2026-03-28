# Open Questions

## Data
- The paper's design brief mentions 5 ETFs (SPY, TLT, GLD, UUP, EFA), but the Phase 2 task specifies only 3 (SPY, TLT, GLD). Phase 9 (Universe Expansion) will address adding more assets.
- ARF Data API returns ~15 years of data starting from 2011-03-29. The exact start date depends on ETF inception and API availability. This is sufficient for walk-forward validation.
- The paper may use different data sources or exact date ranges; this implementation uses ARF Data API as required by project rules.

## Model
- No model questions at this phase (data pipeline only).
