"""
Phase 3: Single-period training and evaluation.

Loads preprocessed ETF data, trains a DeepMomentumNetwork with SharpeRatioLoss
on 80% of the data, evaluates on the remaining 20%, and produces:
  - reports/cycle_3/single_run_metrics.json
  - reports/cycle_3/equity_curve_gross.png
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

from src.backtest import compute_metrics
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

    # 6. Compute cumulative returns and metrics
    cum_returns = np.cumprod(1 + portfolio_returns)
    returns_series = pd.Series(portfolio_returns, index=test_dates[:len(portfolio_returns)])
    metrics = compute_metrics(returns_series)

    print(f"\nTest Results (Gross):")
    print(f"  Sharpe Ratio:    {metrics['sharpeRatio']:.4f}")
    print(f"  Annual Return:   {metrics['annualReturn']:.4f}")
    print(f"  Max Drawdown:    {metrics['maxDrawdown']:.4f}")
    print(f"  Hit Rate:        {metrics['hitRate']:.4f}")
    print(f"  Cumulative Return: {cum_returns[-1]:.4f}")

    # 7. Save outputs
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics
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

    # Plot equity curve
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

    # Also produce the standard metrics.json
    standard_metrics = {
        "sharpeRatio": metrics["sharpeRatio"],
        "annualReturn": metrics["annualReturn"],
        "maxDrawdown": metrics["maxDrawdown"],
        "hitRate": metrics["hitRate"],
        "totalTrades": int(np.sum(np.abs(np.diff(positions, axis=0)) > 0.01)),
        "transactionCosts": {"feeBps": 10, "slippageBps": 5, "netSharpe": 0.0},
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


if __name__ == "__main__":
    main()
