"""
Deep Momentum Network: LSTM-based model for position sizing.
Trained end-to-end with a differentiable Sharpe ratio loss.
"""
import torch
import torch.nn as nn


class DeepMomentumNetwork(nn.Module):
    """LSTM-based deep momentum network.

    Processes a sequence of past returns for each asset and outputs
    a position signal in [-1, 1] via tanh activation.
    """

    def __init__(
        self,
        n_assets: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_assets = n_assets
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, n_assets) containing
               past return sequences.

        Returns:
            Position signals of shape (batch, n_assets) in [-1, 1].
        """
        # x: (batch, seq_len, n_assets)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        positions = torch.tanh(self.fc(last_hidden))  # (batch, n_assets)
        return positions


class SharpeRatioLoss(nn.Module):
    """Differentiable Sharpe ratio loss for end-to-end training.

    Computes negative Sharpe ratio of portfolio returns so that
    minimizing the loss maximizes the Sharpe ratio.
    """

    def __init__(self, annualization_factor: float = 252.0):
        super().__init__()
        self.annualization_factor = annualization_factor

    def forward(
        self, positions: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative Sharpe ratio.

        Args:
            positions: Position signals (batch, n_assets) in [-1, 1].
            returns: Actual asset returns (batch, n_assets).

        Returns:
            Scalar negative Sharpe ratio loss.
        """
        # Portfolio return = sum of position * asset return (equal weight across assets)
        portfolio_returns = (positions * returns).mean(dim=1)  # (batch,)

        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std()

        if std_ret < 1e-8:
            return -mean_ret

        sharpe = mean_ret / std_ret
        return -sharpe
