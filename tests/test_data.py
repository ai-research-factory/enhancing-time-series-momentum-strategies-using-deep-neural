"""Tests for data pipeline."""
import pandas as pd
import pytest

from src.data import DataLoader


def test_no_future_data():
    """Verify that preprocessed data contains no future dates."""
    loader = DataLoader()
    raw_data = loader.fetch()
    processed = loader.preprocess(raw_data)

    today = pd.Timestamp.now()
    assert processed.index.max() <= today, (
        f"Processed data contains future dates: max={processed.index.max()}, today={today}"
    )
