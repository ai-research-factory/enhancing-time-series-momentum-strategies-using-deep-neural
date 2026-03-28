"""Tests for backtest framework."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, calculate_portfolio_costs, compute_metrics


def test_calculate_portfolio_costs_basic():
    """Verify transaction cost calculation produces correct net returns."""
    config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
    cost_rate = 15.0 / 10000  # 0.0015

    # Static positions: no turnover after initial entry
    positions = np.array([[0.5, -0.3], [0.5, -0.3], [0.5, -0.3]])
    returns = np.array([[0.01, -0.01], [0.02, 0.01], [-0.01, 0.02]])

    gross, net, trades = calculate_portfolio_costs(positions, returns, config)

    # Gross: mean of (pos * ret) across assets
    expected_gross = (positions * returns).mean(axis=1)
    np.testing.assert_allclose(gross, expected_gross)

    # Only initial entry has turnover; subsequent days have zero turnover
    initial_cost = np.abs(positions[0]).mean() * cost_rate
    assert net[0] == pytest.approx(gross[0] - initial_cost, abs=1e-10)
    # Days 1,2 have no position change -> net == gross
    np.testing.assert_allclose(net[1:], gross[1:])


def test_calculate_portfolio_costs_with_turnover():
    """Verify costs increase with position changes."""
    config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)

    # Positions that change every day
    positions = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])
    returns = np.zeros((3, 2))

    gross, net, trades = calculate_portfolio_costs(positions, returns, config)

    # Gross should be zero (returns are zero)
    np.testing.assert_allclose(gross, 0.0)
    # Net should be negative (costs deducted)
    assert np.all(net <= 0)
    assert trades > 0


def test_compute_metrics_positive_returns():
    """Verify metrics computation with known positive returns."""
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.01, 252))
    metrics = compute_metrics(returns)

    assert metrics["sharpeRatio"] != 0
    assert metrics["annualReturn"] != 0
    assert metrics["maxDrawdown"] <= 0
    assert 0 < metrics["hitRate"] < 1
