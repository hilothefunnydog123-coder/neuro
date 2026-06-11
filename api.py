"""HTTP API that exposes the forecaster to a web frontend (e.g. a Lovable app).

Run locally:
    pip install -r requirements.txt
    uvicorn src.api:app --reload --port 8000

Endpoints
---------
GET  /health                  -> {"status": "ok"}
POST /forecast                -> trains/loads a model for a ticker and returns
                                 recent history, the forward forecast, and
                                 honest backtest metrics.

CORS is open so a browser app on another domain can call it. Tighten
`allow_origins` before any real deployment.
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .backtest import run_backtest
from .config import load_config
from .predict import _latest_frame, _price_col, forecast, load_checkpoint
from .train import resolve_device, train

app = FastAPI(title="YN Neuro Forecaster API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only; restrict to your frontend domain in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = os.environ.get("MODEL_CACHE_DIR", "checkpoints")
DISCLAIMER = (
    "Educational research tool. Forecasts are model estimates of short-term "
    "price movement, not predictions of actual prices. Not financial advice."
)


class ForecastRequest(BaseModel):
    ticker: str = "AAPL"
    horizon: int = 5
    lookback: int = 60
    epochs: int = 40
    retrain: bool = False  # force a fresh train even if a cached model exists


class Point(BaseModel):
    date: str
    price: float


class ForecastResponse(BaseModel):
    ticker: str
    history: List[Point]
    forecast: List[Point]
    metrics: dict
    disclaimer: str


def _checkpoint_path(ticker: str) -> str:
    safe = "".join(c for c in ticker.upper() if c.isalnum() or c in "-._")
    return os.path.join(CACHE_DIR, f"{safe}.pt")


def _ensure_model(req: ForecastRequest) -> str:
    """Train (and cache) a model for the ticker if we don't have one yet."""
    ckpt = _checkpoint_path(req.ticker)
    if os.path.exists(ckpt) and not req.retrain:
        return ckpt

    cfg = load_config()
    cfg.data.ticker = req.ticker
    cfg.window.horizon = req.horizon
    cfg.window.lookback = req.lookback
    cfg.train.epochs = req.epochs
    cfg.paths.checkpoint = ckpt
    os.makedirs(CACHE_DIR, exist_ok=True)
    train(cfg)
    return ckpt


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def make_forecast(req: ForecastRequest):
    device = resolve_device("auto")
    ckpt = _ensure_model(req)

    _, _, payload = load_checkpoint(ckpt, device)
    price_col = _price_col(payload)

    df = _latest_frame(payload).tail(120)
    history = [Point(date=str(d.date()), price=float(v)) for d, v in df[price_col].items()]

    preds = forecast(ckpt, device)
    pred_col = "predicted_" + price_col
    forecast_pts = [
        Point(date=str(d.date()), price=float(v)) for d, v in preds[pred_col].items()
    ]

    try:
        metrics = run_backtest(ckpt, device)
    except Exception as exc:  # backtest needs enough history; degrade gracefully
        metrics = {"error": str(exc)}

    return ForecastResponse(
        ticker=req.ticker.upper(),
        history=history,
        forecast=forecast_pts,
        metrics=metrics,
        disclaimer=DISCLAIMER,
    )
