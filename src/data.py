"""
Data pipeline for fetching and preprocessing ETF OHLCV data.
Uses ARF Data API to retrieve historical price data.
"""
import os
import pickle
from pathlib import Path

import pandas as pd

API_BASE = "https://ai.1s.xyz/api/data/ohlcv"
DEFAULT_TICKERS = ["SPY", "TLT", "GLD"]
DEFAULT_INTERVAL = "1d"
DEFAULT_PERIOD = "15y"


class DataLoader:
    """Fetches and preprocesses ETF OHLCV data from ARF Data API."""

    def __init__(
        self,
        tickers: list[str] | None = None,
        interval: str = DEFAULT_INTERVAL,
        period: str = DEFAULT_PERIOD,
        data_dir: str = "data",
    ):
        self.tickers = tickers or DEFAULT_TICKERS
        self.interval = interval
        self.period = period
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    def fetch(self) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all tickers from ARF Data API.

        Returns:
            Dict mapping ticker to its OHLCV DataFrame.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        raw_data = {}
        for ticker in self.tickers:
            cache_path = self.raw_dir / f"{ticker}_{self.interval}.csv"
            if cache_path.exists():
                print(f"  Loading cached {ticker} from {cache_path}")
                df = pd.read_csv(cache_path, parse_dates=["timestamp"], index_col="timestamp")
            else:
                url = f"{API_BASE}?ticker={ticker}&interval={self.interval}&period={self.period}"
                print(f"  Fetching {ticker} from API...")
                df = pd.read_csv(url, parse_dates=["timestamp"], index_col="timestamp")
                df.to_csv(cache_path)
                print(f"  Saved raw data to {cache_path}")
            raw_data[ticker] = df
        return raw_data

    def preprocess(self, raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate daily returns and handle NaN values.

        Args:
            raw_data: Dict mapping ticker to OHLCV DataFrame.

        Returns:
            DataFrame with columns for each ticker's close price and daily return.
        """
        closes = pd.DataFrame({
            ticker: df["close"] for ticker, df in raw_data.items()
        })

        # Align all tickers to the same date index (intersection)
        closes = closes.dropna(how="all")
        # Forward-fill gaps (e.g., ETF holidays that differ)
        closes = closes.ffill()
        # Drop any remaining leading NaN rows
        closes = closes.dropna()

        # Calculate daily returns
        returns = closes.pct_change()
        # First row of returns will be NaN; drop it
        returns = returns.iloc[1:]

        result = pd.DataFrame(index=returns.index)
        for ticker in self.tickers:
            result[f"{ticker}_close"] = closes.loc[returns.index, ticker]
            result[f"{ticker}_return"] = returns[ticker]

        # Final safety check: forward-fill then drop any remaining NaN
        result = result.ffill().dropna()

        return result

    def save(self, df: pd.DataFrame, filename: str = "assets.pkl") -> Path:
        """Save processed data to pickle file.

        Args:
            df: Processed DataFrame.
            filename: Output filename.

        Returns:
            Path to the saved file.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.processed_dir / filename
        with open(out_path, "wb") as f:
            pickle.dump(df, f)
        print(f"  Saved processed data to {out_path}")
        return out_path

    def run(self) -> pd.DataFrame:
        """Full pipeline: fetch, preprocess, save.

        Returns:
            Processed DataFrame.
        """
        print("Step 1: Fetching raw OHLCV data...")
        raw_data = self.fetch()

        print("Step 2: Preprocessing (returns, NaN handling)...")
        processed = self.preprocess(raw_data)

        print("Step 3: Saving processed data...")
        self.save(processed)

        print(f"Done. Shape: {processed.shape}, NaN count: {processed.isna().sum().sum()}")
        return processed
