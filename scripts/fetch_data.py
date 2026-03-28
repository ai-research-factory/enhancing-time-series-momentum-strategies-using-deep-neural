#!/usr/bin/env python3
"""Fetch and preprocess ETF data using the DataLoader pipeline."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DataLoader


def main():
    loader = DataLoader(
        tickers=["SPY", "TLT", "GLD"],
        interval="1d",
        period="15y",
    )
    df = loader.run()

    # Verification
    nan_count = df.isna().sum().sum()
    print(f"\nVerification:")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  NaN count: {nan_count}")

    if nan_count > 0:
        print("ERROR: Processed data contains NaN values!")
        sys.exit(1)
    else:
        print("OK: No NaN values in processed data.")


if __name__ == "__main__":
    main()
