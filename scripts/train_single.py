"""
Single-period training and evaluation with transaction cost analysis.

Loads preprocessed ETF data, trains a DeepMomentumNetwork with SharpeRatioLoss
on 80% of the data, evaluates on the remaining 20%, and produces:
  - Cycle 3: gross metrics and equity curve
  - Cycle 4: net-of-cost metrics and gross vs net equity curves
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import BacktestConfig, calculate_portfolio_costs, compute_metrics
from src.data import DataLoader as ETFDataLoader
from src.model import DeepMomentumNetwork, SharpeRatioLoss

# --- Configuration ---
LOOKBACK = 20         # Past N days of returns as input sequence
HIDDEN_SIZE = 32      # LSTM hidden size
NUM_LAYERS = 2        # LSTM layers
DROPOUT = 0.1
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 64
TRAIN_RATIO = 0.8
SEED = 42
REPORT_DIR = Path("reports/cycle_3")


def load_data() -> pd.DataFrame:
    """Load or fetch+preprocess ETF data, return processed DataFrame."""
    parquet_path = Path("data/processed/assets.parquet")
    if parquet_path.exists():
        print("Loading cached processed data from parquet...")
        return pd.read_parquet(parquet_path)

    print("No cached data found. Running data pipeline...")
    loader = ETFDataLoader()
    return loader.run()


def build_sequences(returns: np.ndarray, lookback: int):
    """Build (X, Y) pairs from return matrix.

    X[i] = returns[i-lookback:i]  (past returns)
    Y[i] = returns[i]             (next-step returns for position evaluation)
    """
    X, Y = [], []
    for i in range(lookback, len(returns)):
        X.append(returns[i - lookback : i])
        Y.append(returns[i])
    return np.array(X), np.array(Y)


def train_model(
    model: DeepMomentumNetwork,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Train the model and return per-epoch losses."""
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, Y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = SharpeRatioLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            positions = model(X_batch)
            loss = criterion(positions, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

    return losses


def evaluate(
    model: DeepMomentumNetwork,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate positions on test data and return (positions, portfolio_returns)."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        positions = model(X_t).cpu().numpy()

    # Portfolio return per day: mean of (position_i * return_i) across assets
    portfolio_returns = (positions * Y_test).mean(axis=1)
    return positions, portfolio_returns


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    df = load_data()
    tickers = [c.replace("_return", "") for c in df.columns if c.endswith("_return")]
    n_assets = len(tickers)
    print(f"Tickers: {tickers}, N={n_assets}, Data shape: {df.shape}")

    # Extract return columns as numpy array
    return_cols = [f"{t}_return" for t in tickers]
    returns = df[return_cols].values  # (T, n_assets)

    # 2. Build sequences
    X, Y = build_sequences(returns, LOOKBACK)
    print(f"Sequences: X={X.shape}, Y={Y.shape}")

    # 3. Train/test split (chronological)
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Get corresponding dates for the test period
    # Sequences start at index LOOKBACK in the original df
    test_dates = df.index[LOOKBACK + split_idx : LOOKBACK + split_idx + len(X_test)]

    # 4. Train model
    model = DeepMomentumNetwork(
        n_assets=n_assets,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print(f"\nTraining DeepMomentumNetwork ({sum(p.numel() for p in model.parameters())} params)...")
    train_losses = train_model(model, X_train, Y_train, EPOCHS, BATCH_SIZE, LR, device)

    # 5. Evaluate on test set
    print("\nEvaluating on test set...")
    positions, portfolio_returns = evaluate(model, X_test, Y_test, device)

    # 6. Compute cumulative returns and metrics (gross)
    cum_returns = np.cumprod(1 + portfolio_returns)
    returns_series = pd.Series(portfolio_returns, index=test_dates[:len(portfolio_returns)])
    metrics = compute_metrics(returns_series)

    print(f"\nTest Results (Gross):")
    print(f"  Sharpe Ratio:    {metrics['sharpeRatio']:.4f}")
    print(f"  Annual Return:   {metrics['annualReturn']:.4f}")
    print(f"  Max Drawdown:    {metrics['maxDrawdown']:.4f}")
    print(f"  Hit Rate:        {metrics['hitRate']:.4f}")
    print(f"  Cumulative Return: {cum_returns[-1]:.4f}")

    # 6b. Compute transaction costs and net metrics
    cost_config = BacktestConfig(fee_bps=10.0, slippage_bps=5.0)
    gross_rets, net_rets, total_trades = calculate_portfolio_costs(
        positions, Y_test, cost_config
    )
    net_cum_returns = np.cumprod(1 + net_rets)
    net_returns_series = pd.Series(net_rets, index=test_dates[:len(net_rets)])
    net_metrics = compute_metrics(net_returns_series)

    # Per-asset turnover stats
    pos_diff = np.abs(np.diff(positions, axis=0))
    first_trade = np.abs(positions[0:1, :])
    turnover = np.vstack([first_trade, pos_diff])
    avg_daily_turnover = float(turnover.mean())
    total_cost_drag = float(np.cumprod(1 + gross_rets)[-1] - np.cumprod(1 + net_rets)[-1])

    print(f"\nTest Results (Net of Costs — {cost_config.fee_bps}bps fee + {cost_config.slippage_bps}bps slippage):")
    print(f"  Net Sharpe Ratio:    {net_metrics['sharpeRatio']:.4f}")
    print(f"  Net Annual Return:   {net_metrics['annualReturn']:.4f}")
    print(f"  Net Max Drawdown:    {net_metrics['maxDrawdown']:.4f}")
    print(f"  Net Hit Rate:        {net_metrics['hitRate']:.4f}")
    print(f"  Net Cumulative Return: {net_cum_returns[-1]:.4f}")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Avg Daily Turnover:  {avg_daily_turnover:.4f}")
    print(f"  Cost Drag (cumul):   {total_cost_drag:.4f}")

    # 7. Save outputs
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save cycle 3 metrics
    single_run_metrics = {
        "grossSharpeRatio": metrics["sharpeRatio"],
        "annualReturn": metrics["annualReturn"],
        "maxDrawdown": metrics["maxDrawdown"],
        "hitRate": metrics["hitRate"],
        "cumulativeReturn": round(float(cum_returns[-1]), 4),
        "testPeriod": {
            "start": str(test_dates[0].date()) if len(test_dates) > 0 else "",
            "end": str(test_dates[-1].date()) if len(test_dates) > 0 else "",
            "nDays": len(portfolio_returns),
        },
        "modelConfig": {
            "lookback": LOOKBACK,
            "hiddenSize": HIDDEN_SIZE,
            "numLayers": NUM_LAYERS,
            "dropout": DROPOUT,
            "lr": LR,
            "epochs": EPOCHS,
            "batchSize": BATCH_SIZE,
        },
        "tickers": tickers,
    }
    metrics_path = REPORT_DIR / "single_run_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(single_run_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot equity curve (cycle 3)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates[:len(cum_returns)], cum_returns, linewidth=1.5, label="Deep Momentum Network")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Gross Cumulative Return — Deep Momentum Network (Single Period)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = REPORT_DIR / "equity_curve_gross.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved equity curve to {plot_path}")

    # Save model
    model_path = Path("models/dmn_single.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Cycle 3 standard metrics.json
    standard_metrics = {
        "sharpeRatio": metrics["sharpeRatio"],
        "annualReturn": metrics["annualReturn"],
        "maxDrawdown": metrics["maxDrawdown"],
        "hitRate": metrics["hitRate"],
        "totalTrades": total_trades,
        "transactionCosts": {"feeBps": 10, "slippageBps": 5, "netSharpe": net_metrics["sharpeRatio"]},
        "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
        "customMetrics": {
            "phase": "single-period-training",
            "grossCumulativeReturn": round(float(cum_returns[-1]), 4),
            "testStartDate": str(test_dates[0].date()) if len(test_dates) > 0 else "",
            "testEndDate": str(test_dates[-1].date()) if len(test_dates) > 0 else "",
            "lookback": LOOKBACK,
            "epochs": EPOCHS,
            "finalTrainLoss": round(float(train_losses[-1]), 4),
        },
    }
    std_metrics_path = REPORT_DIR / "metrics.json"
    with open(std_metrics_path, "w") as f:
        json.dump(standard_metrics, f, indent=2)
    print(f"Saved standard metrics to {std_metrics_path}")

    # --- Cycle 4: Transaction cost reports ---
    REPORT_DIR_4 = Path("reports/cycle_4")
    REPORT_DIR_4.mkdir(parents=True, exist_ok=True)

    # Plot gross vs net equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_dates = test_dates[:len(cum_returns)]
    ax.plot(plot_dates, cum_returns, linewidth=1.5, label="Gross Return")
    ax.plot(plot_dates, net_cum_returns, linewidth=1.5, label=f"Net Return ({cost_config.fee_bps:.0f}+{cost_config.slippage_bps:.0f} bps)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Cumulative Return — Gross vs Net of Transaction Costs")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    eq_path_4 = REPORT_DIR_4 / "equity_curve_gross_vs_net.png"
    fig.savefig(eq_path_4, dpi=150)
    plt.close(fig)
    print(f"Saved gross vs net equity curve to {eq_path_4}")

    # Plot turnover over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    daily_turnover = turnover.mean(axis=1)  # avg across assets
    axes[0].bar(plot_dates, daily_turnover, width=1.0, alpha=0.6, color="steelblue")
    axes[0].set_ylabel("Avg Daily Turnover")
    axes[0].set_title("Portfolio Turnover and Cost Impact")
    axes[0].grid(True, alpha=0.3)

    # Cumulative cost drag
    cost_bps = (cost_config.fee_bps + cost_config.slippage_bps) / 10000
    cumul_cost_drag = np.cumsum((turnover * cost_bps).mean(axis=1))
    axes[1].plot(plot_dates, cumul_cost_drag, linewidth=1.5, color="crimson")
    axes[1].set_ylabel("Cumulative Cost Drag")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    turnover_path = REPORT_DIR_4 / "turnover_analysis.png"
    fig.savefig(turnover_path, dpi=150)
    plt.close(fig)
    print(f"Saved turnover analysis to {turnover_path}")

    # Cycle 4 standard metrics.json
    cycle4_metrics = {
        "sharpeRatio": metrics["sharpeRatio"],
        "annualReturn": metrics["annualReturn"],
        "maxDrawdown": metrics["maxDrawdown"],
        "hitRate": metrics["hitRate"],
        "totalTrades": total_trades,
        "transactionCosts": {
            "feeBps": cost_config.fee_bps,
            "slippageBps": cost_config.slippage_bps,
            "netSharpe": net_metrics["sharpeRatio"],
        },
        "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
        "customMetrics": {
            "phase": "transaction-cost-model",
            "grossSharpe": metrics["sharpeRatio"],
            "netSharpe": net_metrics["sharpeRatio"],
            "netAnnualReturn": net_metrics["annualReturn"],
            "netMaxDrawdown": net_metrics["maxDrawdown"],
            "netHitRate": net_metrics["hitRate"],
            "grossCumulativeReturn": round(float(cum_returns[-1]), 4),
            "netCumulativeReturn": round(float(net_cum_returns[-1]), 4),
            "costDrag": round(total_cost_drag, 4),
            "avgDailyTurnover": round(avg_daily_turnover, 4),
            "totalTrades": total_trades,
            "testStartDate": str(test_dates[0].date()) if len(test_dates) > 0 else "",
            "testEndDate": str(test_dates[-1].date()) if len(test_dates) > 0 else "",
        },
    }
    cycle4_metrics_path = REPORT_DIR_4 / "metrics.json"
    with open(cycle4_metrics_path, "w") as f:
        json.dump(cycle4_metrics, f, indent=2)
    print(f"Saved cycle 4 metrics to {cycle4_metrics_path}")


if __name__ == "__main__":
    main()
