"""Network-free smoke tests for the windowing, dataset, and model plumbing.

These use a synthetic sine-wave "price" series so they run offline and fast.
Run with::  python -m pytest -q   (or)   python tests/test_pipeline.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data import apply_target_mode
from src.dataset import make_windows, prepare_splits
from src.model import LSTMForecaster


def _synthetic_frame(n: int = 400) -> pd.DataFrame:
    t = np.arange(n)
    close = 100 + 10 * np.sin(t / 12.0) + np.random.RandomState(0).normal(0, 0.5, n)
    return pd.DataFrame(
        {
            "Close": close,
            "Open": close + 0.1,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Volume": 1_000 + t,
        },
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def test_make_windows_shapes():
    feats = np.random.rand(50, 3).astype(np.float32)
    tgt = np.random.rand(50).astype(np.float32)
    X, y = make_windows(feats, tgt, lookback=10, horizon=3)
    assert X.shape == (50 - 10 - 3 + 1, 10, 3)
    assert y.shape == (50 - 10 - 3 + 1, 3)


def test_prepare_splits_and_forward():
    df = _synthetic_frame()
    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
    train_ds, val_ds, scalers = prepare_splits(
        df, feature_cols, "Close", lookback=20, horizon=5, val_split=0.2
    )
    assert len(train_ds) > 0 and len(val_ds) > 0

    model = LSTMForecaster(n_features=len(feature_cols), horizon=5, hidden_size=16, num_layers=1)
    x, y = train_ds[0]
    out = model(x.unsqueeze(0))
    assert out.shape == (1, 5)
    assert y.shape == (5,)


def test_inverse_target_roundtrip():
    df = _synthetic_frame()
    _, _, scalers = prepare_splits(
        df, ["Close"], "Close", lookback=20, horizon=5, val_split=0.2
    )
    raw = df["Close"].values[:5]
    scaled = scalers.target_scaler.transform(raw.reshape(-1, 1)).ravel()
    back = scalers.inverse_target(scaled)
    assert np.allclose(back, raw, atol=1e-4)


def test_logreturn_mode_reconstructs_near_last_price():
    """A naive (~zero return) forecast should land near the last price, not the
    multi-year average. This guards against the 'fake crash' regression."""
    df = _synthetic_frame()
    out, target_col = apply_target_mode(df, "Close", "logreturn")
    assert target_col == "__target__"
    assert "Close" in out.columns  # price stays available as a feature

    last_price = float(out["Close"].iloc[-1])
    hist_mean = float(out["Close"].mean())
    # Reconstruct price from ~zero predicted returns.
    horizon = 5
    preds = last_price * np.exp(np.cumsum(np.zeros(horizon)))
    assert np.allclose(preds, last_price)
    # The reconstruction is anchored to the present, not the historical mean.
    assert abs(preds[-1] - last_price) < abs(preds[-1] - hist_mean)


def test_price_mode_is_identity():
    df = _synthetic_frame()
    out, target_col = apply_target_mode(df, "Close", "price")
    assert target_col == "Close"
    assert len(out) == len(df)


if __name__ == "__main__":
    test_make_windows_shapes()
    test_prepare_splits_and_forward()
    test_inverse_target_roundtrip()
    test_logreturn_mode_reconstructs_near_last_price()
    test_price_mode_is_identity()
    print("All smoke tests passed.")
