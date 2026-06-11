"""Load a trained checkpoint and forecast the next ``horizon`` prices.

Also supports a simple back-test plot comparing predictions against the most
recent actual prices.
"""

from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .data import build_dataset
from .model import LSTMForecaster


def load_checkpoint(path: str, device: Optional[torch.device] = None):
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = LSTMForecaster(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    scalers = pickle.loads(payload["scalers"])
    return model, scalers, payload


def _price_col(payload) -> str:
    # "price_col" is the new key; fall back to the old "target_col" for
    # checkpoints trained before target modes existed.
    return payload.get("price_col", payload.get("target_col", "Close"))


def _latest_frame(payload, start: str = "2015-01-01") -> pd.DataFrame:
    """Re-download recent data using the same settings the model was trained on."""
    data = payload["data"]
    return build_dataset(
        ticker=data["ticker"],
        start=start,
        end=None,
        interval=data["interval"],
        features=data["features"],
        target=_price_col(payload),
        csv=data.get("csv"),
    )


def forecast(checkpoint: str, device: Optional[torch.device] = None) -> pd.DataFrame:
    """Predict the next ``horizon`` target values after the latest available data.

    Returns a DataFrame indexed by future (business-day) dates with the
    predicted price.
    """
    device = device or torch.device("cpu")
    model, scalers, payload = load_checkpoint(checkpoint, device)
    lookback = payload["lookback"]
    feature_cols: List[str] = payload["feature_cols"]

    df = _latest_frame(payload)
    if len(df) < lookback:
        raise ValueError(f"Need at least {lookback} rows; got {len(df)}.")

    window = df[feature_cols].values[-lookback:]
    x = scalers.transform_features(window).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, lookback, n_features)

    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().ravel()
    raw = scalers.inverse_target(pred_scaled)

    price_col = _price_col(payload)
    if payload.get("target_mode", "price") == "logreturn":
        # raw are predicted log returns; rebuild the price path from the last
        # actual price so the forecast is anchored to today, not the long-run mean.
        last_price = float(df[price_col].iloc[-1])
        preds = last_price * np.exp(np.cumsum(raw))
    else:
        preds = raw

    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=len(preds) + 1)[1:]
    result = pd.DataFrame({"predicted_" + price_col: preds}, index=future_dates)
    result.index.name = "Date"
    return result


def backtest_plot(checkpoint: str, out_path: str, device: Optional[torch.device] = None) -> str:
    """Plot the last stretch of actual prices plus the forward forecast."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = device or torch.device("cpu")
    _, _, payload = load_checkpoint(checkpoint, device)
    df = _latest_frame(payload)
    target = _price_col(payload)
    preds = forecast(checkpoint, device)

    recent = df[target].iloc[-120:]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(recent.index, recent.values, label=f"Actual {target}", color="#1f77b4")
    pred_col = "predicted_" + target
    ax.plot(preds.index, preds[pred_col].values, "o--", label="Forecast", color="#d62728")
    ax.axvline(df.index[-1], color="gray", linestyle=":", alpha=0.7)
    ax.set_title(f"{payload['data']['ticker']} — {len(preds)}-step forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
