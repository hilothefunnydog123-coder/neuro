"""The forecasting network: an LSTM encoder with a linear projection head.

The LSTM consumes a sequence of feature vectors and the final hidden state is
projected to ``horizon`` future target values. It's deliberately simple and
fast to train on a laptop while still being a real recurrent net.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.Linear(hidden_size * directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # final timestep representation
        return self.head(last)  # (batch, horizon)

    @property
    def config(self) -> dict:
        """Hyperparameters needed to reconstruct this module from a checkpoint."""
        return {
            "n_features": self.n_features,
            "horizon": self.horizon,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
        }
