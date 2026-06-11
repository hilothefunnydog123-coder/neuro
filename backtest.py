"""Honest accuracy measurement on held-out data.

This evaluates a trained checkpoint on the most recent slice of history that the
model did *not* train on (the chronological validation tail), and compares it
against a naive random-walk baseline ("tomorrow's price = today's price"). The
baseline matters: on near-efficient markets it is surprisingly hard to beat, so
any honest accuracy claim must be stated *relative to it*.

Metrics reported:
- RMSE / MAE in price units (lower is better), for the model and the baseline.
- Skill score = 1 - RMSE_model / RMSE_baseline. Positive means the model beats
  random walk; <= 0 means it does not.
- Directional accuracy: how often the 1-step-ahead up/down call is correct
  (0.5 = coin flip).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .predict import _latest_frame, _price_col, load_checkpoint


def run_backtest(
    checkpoint: str,
    device: Optional[torch.device] = None,
    eval_split: Optional[float] = None,
) -> dict:
    device = device or torch.device("cpu")
    model, scalers, payload = load_checkpoint(checkpoint, device)
    lookback = payload["lookback"]
    horizon = payload["horizon"]
    feature_cols = payload["feature_cols"]
    price_col = _price_col(payload)
    mode = payload.get("target_mode", "price")

    df = _latest_frame(payload)
    n = len(df)
    eval_split = eval_split if eval_split is not None else payload.get("val_split", 0.15)

    feats = scalers.transform_features(df[feature_cols].values).astype(np.float32)
    prices = df[price_col].values.astype(np.float64)

    # Evaluate on the held-out tail; never start before we have a full lookback.
    start = max(lookback, int(n * (1.0 - eval_split)))
    indices = list(range(start, n - horizon + 1))
    if not indices:
        raise ValueError("Not enough held-out data to backtest. Train on more history.")

    # Batch all evaluation windows through the model at once.
    batch = np.stack([feats[t - lookback : t] for t in indices])
    with torch.no_grad():
        raw = model(torch.from_numpy(batch).to(device)).cpu().numpy()

    pred_prices = np.empty_like(raw, dtype=np.float64)
    actual_prices = np.empty_like(raw, dtype=np.float64)
    last_prices = np.empty(len(indices), dtype=np.float64)

    for row, t in enumerate(indices):
        last_price = prices[t - 1]
        last_prices[row] = last_price
        actual_prices[row] = prices[t : t + horizon]
        if mode == "logreturn":
            pred_prices[row] = last_price * np.exp(np.cumsum(raw[row]))
        else:
            pred_prices[row] = raw[row]

    # Naive baseline: predict the last known price for every future step.
    naive = np.repeat(last_prices[:, None], horizon, axis=1)

    rmse_model = float(np.sqrt(np.mean((pred_prices - actual_prices) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((naive - actual_prices) ** 2)))
    mae_model = float(np.mean(np.abs(pred_prices - actual_prices)))
    mae_naive = float(np.mean(np.abs(naive - actual_prices)))
    skill = 1.0 - rmse_model / rmse_naive if rmse_naive > 0 else 0.0

    # 1-step directional accuracy (did we call up vs. down correctly?).
    pred_dir = np.sign(pred_prices[:, 0] - last_prices)
    actual_dir = np.sign(actual_prices[:, 0] - last_prices)
    moved = actual_dir != 0
    directional = float(np.mean(pred_dir[moved] == actual_dir[moved])) if moved.any() else float("nan")

    return {
        "ticker": payload["data"]["ticker"],
        "samples": len(indices),
        "horizon": horizon,
        "rmse_model": rmse_model,
        "rmse_naive": rmse_naive,
        "mae_model": mae_model,
        "mae_naive": mae_naive,
        "skill_score": skill,
        "directional_accuracy": directional,
    }


def format_report(m: dict) -> str:
    beats = "YES" if m["skill_score"] > 0 else "NO"
    lines = [
        f"Backtest — {m['ticker']}  ({m['samples']} held-out windows, horizon={m['horizon']})",
        "-" * 56,
        f"  RMSE   model={m['rmse_model']:.3f}   naive={m['rmse_naive']:.3f}",
        f"  MAE    model={m['mae_model']:.3f}   naive={m['mae_naive']:.3f}",
        f"  Skill score (vs random walk): {m['skill_score']:+.3f}   beats baseline: {beats}",
        f"  1-step directional accuracy:  {m['directional_accuracy']:.1%}  (0.5 = coin flip)",
        "-" * 56,
        "  Note: beating the random-walk baseline on price is genuinely hard.",
        "  A skill score near zero is the expected, honest result.",
    ]
    return "\n".join(lines)
