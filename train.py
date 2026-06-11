"""Training loop with validation, early stopping, and checkpointing.

A checkpoint bundles the model weights, its architecture config, and the fitted
scalers + column lists so that ``predict.py`` can reload everything it needs to
forecast on fresh data.
"""

from __future__ import annotations

import os
import pickle
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data import apply_target_mode, build_dataset
from .dataset import prepare_splits
from .model import LSTMForecaster


def resolve_device(choice: str = "auto") -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run_epoch(model, loader, criterion, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = criterion(pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def train(cfg: Config) -> dict:
    """Train a model end-to-end and write a checkpoint. Returns a summary dict."""
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"[train] device={device}")

    if cfg.data.csv:
        print(f"[train] loading data from {cfg.data.csv}")
    else:
        print(f"[train] downloading {cfg.data.ticker} ({cfg.data.start} -> {cfg.data.end or 'today'})")
    df = build_dataset(
        ticker=cfg.data.ticker,
        start=cfg.data.start,
        end=cfg.data.end,
        interval=cfg.data.interval,
        features=cfg.data.features,
        target=cfg.data.target,
        csv=cfg.data.csv,
    )
    # Derive the prediction target (raw price vs. log returns). The price
    # columns stay as input features either way.
    df, target_col = apply_target_mode(df, cfg.data.target, cfg.data.target_mode)
    feature_cols = [c for c in df.columns if c != target_col]
    print(
        f"[train] {len(df)} rows, mode={cfg.data.target_mode}, "
        f"target={target_col}, features={feature_cols}"
    )

    train_ds, val_ds, scalers = prepare_splits(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=cfg.window.lookback,
        horizon=cfg.window.horizon,
        val_split=cfg.train.val_split,
    )
    print(f"[train] windows: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = LSTMForecaster(
        n_features=len(feature_cols),
        horizon=cfg.window.horizon,
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
    ).to(device)

    # Optionally continue training from an existing checkpoint instead of
    # starting from random weights. Architecture must match (same feature count
    # and horizon); scalers are re-fit on the current data.
    if getattr(cfg.train, "resume", False) and os.path.exists(cfg.paths.checkpoint):
        prev = torch.load(cfg.paths.checkpoint, map_location=device, weights_only=False)
        if prev.get("model_config") == model.config:
            model.load_state_dict(prev["model_state"])
            print(f"[train] resumed weights from {cfg.paths.checkpoint}")
        else:
            print(
                "[train] WARNING: existing checkpoint architecture differs "
                "(feature count or horizon changed) -> starting from scratch"
            )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, cfg.train.patience // 2)
    )

    best_val = float("inf")
    best_state: Optional[dict] = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device, None)
        scheduler.step(val_loss)
        print(f"[train] epoch {epoch:3d}/{cfg.train.epochs}  train={tr_loss:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.patience:
                print(f"[train] early stopping at epoch {epoch} (best val={best_val:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = save_checkpoint(cfg, model, scalers, feature_cols, best_val)
    print(f"[train] saved checkpoint -> {cfg.paths.checkpoint}  (best val={best_val:.5f})")
    return summary


def save_checkpoint(cfg: Config, model, scalers, feature_cols, best_val: float) -> dict:
    os.makedirs(os.path.dirname(cfg.paths.checkpoint) or ".", exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "scalers": pickle.dumps(scalers),
        "feature_cols": feature_cols,
        "price_col": cfg.data.target,
        "target_mode": cfg.data.target_mode,
        "val_split": cfg.train.val_split,
        "lookback": cfg.window.lookback,
        "horizon": cfg.window.horizon,
        "data": {
            "ticker": cfg.data.ticker,
            "interval": cfg.data.interval,
            "features": cfg.data.features,
            "csv": cfg.data.csv,
        },
        "best_val": best_val,
    }
    torch.save(payload, cfg.paths.checkpoint)
    return {"checkpoint": cfg.paths.checkpoint, "best_val": best_val}
