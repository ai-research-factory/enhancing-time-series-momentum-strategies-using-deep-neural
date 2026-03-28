# Cycle 2: Data Pipeline Construction — Technical Findings

## Objective
Build a data pipeline to fetch, preprocess, and store ETF OHLCV data for SPY, TLT, and GLD using the ARF Data API.

## Implementation

### src/data.py — DataLoader class
- **fetch()**: Retrieves daily OHLCV data from the ARF Data API (`https://ai.1s.xyz/api/data/ohlcv`). Raw CSVs are cached in `data/raw/` to avoid redundant API calls.
- **preprocess()**: Aligns all tickers to a common date index, forward-fills gaps (handling differing ETF holidays), computes daily log-returns via `pct_change()`, and drops any residual NaN rows.
- **save()**: Persists the processed DataFrame to `data/processed/assets.pkl` using pickle.
- **run()**: Orchestrates the full pipeline (fetch → preprocess → save).

### scripts/fetch_data.py
Entry-point script that instantiates `DataLoader` with the target tickers and runs the pipeline. Includes a verification step that asserts zero NaN values in the output.

## Results

| Metric | Value |
|--------|-------|
| Tickers | SPY, TLT, GLD |
| Date range | 2011-03-29 to 2026-03-27 |
| Rows (trading days) | 3,772 |
| Columns | 6 (close + return per ticker) |
| NaN count | 0 |

### Output columns
- `SPY_close`, `SPY_return`
- `TLT_close`, `TLT_return`
- `GLD_close`, `GLD_return`

## Observations
1. The ARF Data API returns ~15 years of daily data for all three ETFs, providing a sufficiently long history for walk-forward validation.
2. All three ETFs have well-aligned trading dates with minimal gaps requiring forward-fill.
3. The data starts from 2011-03-29, which gives approximately 15 years of history as specified.
4. No synthetic data was used; all data comes from the ARF Data API.

## Acceptance Criteria Status
- [x] `scripts/fetch_data.py` runs and generates `data/processed/assets.pkl`
- [x] Generated data contains zero NaN values
