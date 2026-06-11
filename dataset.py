"""Windowing, scaling, and PyTorch ``Dataset`` for sequence forecasting.

The model sees ``lookback`` days of (scaled) features and predicts the next
``horizon`` days of the (scaled) target. Scalers are fit on the training split
only, to avoid look-ahead leakage into validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


@dataclass
class Scalers:
    """Holds the fitted feature/target scalers and column bookkeeping."""

    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_cols: List[str]
    target_col: str

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        return self.feature_scaler.transform(x)

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).reshape(-1, 1)
        return self.target_scaler.inverse_transform(y).ravel()


def make_windows(
    features: np.ndarray, target: np.ndarray, lookback: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice arrays into (X, y) sliding windows.

    X has shape (n, lookback, n_features); y has shape (n, horizon).
    """
    xs, ys = [], []
    n = len(features)
    for i in range(n - lookback - horizon + 1):
        xs.append(features[i : i + lookback])
        ys.append(target[i + lookback : i + lookback + horizon])
    if not xs:
        raise ValueError(
            f"Not enough rows ({n}) for lookback={lookback} + horizon={horizon}. "
            f"Use a longer history or smaller windows."
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def prepare_splits(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int,
    val_split: float,
) -> Tuple["SequenceDataset", "SequenceDataset", Scalers]:
    """Chronologically split, fit scalers on train, and build datasets.

    The split is done on the raw rows *before* windowing so that no validation
    row ever appears inside a training window.
    """
    n = len(df)
    n_val = int(n * val_split)
    # Guarantee both splits can produce at least one window.
    min_rows = lookback + horizon
    n_val = max(min_rows, min(n_val, n - min_rows))

    train_df = df.iloc[: n - n_val]
    # Carry lookback rows of context into validation so its first window is valid.
    val_df = df.iloc[n - n_val - lookback :]

    scalers = _fit_scalers(train_df, feature_cols, target_col)

    train_ds = SequenceDataset(train_df, scalers, lookback, horizon)
    val_ds = SequenceDataset(val_df, scalers, lookback, horizon)
    return train_ds, val_ds, scalers


def _fit_scalers(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Scalers:
    feature_scaler = StandardScaler().fit(df[feature_cols].values)
    target_scaler = StandardScaler().fit(df[[target_col]].values)
    return Scalers(feature_scaler, target_scaler, list(feature_cols), target_col)


class SequenceDataset(Dataset):
    """Wraps a DataFrame slice into scaled (X, y) sliding-window tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        scalers: Scalers,
        lookback: int,
        horizon: int,
    ):
        features = scalers.transform_features(df[scalers.feature_cols].values)
        target = scalers.target_scaler.transform(df[[scalers.target_col]].values).ravel()
        self.X, self.y = make_windows(features, target, lookback, horizon)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
