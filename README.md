# neuro — a trainable neural network for stock / option price forecasting

A small, self-contained PyTorch project that trains an **LSTM** to forecast the
next *N* days of a stock (or option underlying) price from its recent history
and a handful of technical indicators.

> ⚠️ **Disclaimer:** This is an educational / research tool. Financial markets
> are noisy and largely unpredictable; no model here (or anywhere) can reliably
> predict prices. **Do not trade real money based on these outputs.** Nothing in
> this repo is financial advice.

## What's inside

| File | Purpose |
|------|---------|
| `src/data.py` | Downloads OHLCV data (Yahoo Finance) and builds technical features (returns, SMAs, MACD, RSI, volatility). |
| `src/dataset.py` | Sliding-window dataset + leak-free scaling (scalers fit on train only). |
| `src/model.py` | The `LSTMForecaster` network (LSTM encoder → MLP head → `horizon` outputs). |
| `src/train.py` | Training loop with validation, early stopping, LR scheduling, checkpointing. |
| `src/predict.py` | Loads a checkpoint and forecasts future prices; optional plot. |
| `src/cli.py` | `train` / `predict` command-line interface. |
| `config.yaml` | All hyperparameters in one place. |
| `tests/` | Offline smoke tests (no network needed). |

## Run it on a Chromebook (or any low-powered machine)

You don't need a powerful computer. Two easy options:

**A) Google Colab (recommended — free GPU, browser-only):**
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. File → Open notebook → **GitHub** tab → paste this repo's URL.
3. Open `notebooks/colab_quickstart.ipynb`.
4. (Optional) Runtime → Change runtime type → **GPU**.
5. Runtime → **Run all**. That's it — it downloads data, trains, and plots a forecast.

**B) ChromeOS Linux terminal (Crostini):**
Settings → Advanced → Developers → turn on **Linux development environment**.
Then open the Terminal app and follow the *Install* steps below.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

Defaults come from `config.yaml`; override anything on the command line:

```bash
# Train on Apple, 50 epochs, 60-day lookback, predict 5 days ahead
python -m src.cli train --ticker AAPL --epochs 50 --lookback 60 --horizon 5
```

This downloads the data, trains, and writes `checkpoints/model.pt` (best
validation weights, scalers, and config all bundled together).

**No network / your own data?** Train from a local CSV instead (must have a
`Date` column plus OHLCV columns):

```bash
python -m src.cli train --csv data/your_prices.csv --epochs 50
```

## Forecast

```bash
python -m src.cli predict --plot
```

Prints the next `horizon` predicted prices and (with `--plot`) saves
`outputs/<TICKER>_forecast.png`.

## Continual learning (resume)

Keep improving an existing model instead of starting fresh:

```bash
python -m src.cli train --resume --epochs 20
```

This loads `checkpoints/model.pt` and continues training. Useful when new data
arrives. Note: re-training on the *same* data plateaus — real gains come from
more data or more training budget, not from re-running alone.

## Measure accuracy honestly (backtest)

Never claim accuracy you haven't measured. The backtester evaluates the model on
recent held-out data it never trained on, and compares it to a naive
random-walk baseline ("tomorrow = today"):

```bash
python -m src.cli backtest
```

It reports RMSE/MAE for the model vs. the baseline, a **skill score**
(`1 - RMSE_model/RMSE_naive`; positive means it beats random walk), and 1-step
directional accuracy. Beating the baseline on price is genuinely hard — a skill
score near zero is the honest, expected result, and that's fine.

## Web app (API + Lovable frontend)

The model runs as a small HTTP API; a web frontend (built with
[Lovable](https://lovable.dev)) calls it. Lovable can't run Python, so the
split is: **Python API does the ML, Lovable builds the UI.**

Run the API locally:
```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
# POST http://localhost:8000/forecast  {"ticker": "AAPL"}
```

Deploy it with the included `Dockerfile` (Render / Railway / Fly.io / Hugging
Face Spaces), then build the frontend in Lovable using the files in `lovable/`:
- `lovable/PROJECT_BRIEF.md` — upload this (full API contract + UI spec).
- `lovable/PROMPT.md` — paste this into Lovable.

Point the Lovable app's `VITE_API_BASE_URL` at your deployed API URL.

## Forecasting options

Options are derived from their underlying, so the practical workflow is:

1. Forecast the **underlying** price path with this model (e.g. `--ticker SPY`).
2. Price the option from that forecast with a pricing model (Black–Scholes /
   binomial) using your strike, expiry, and an implied-volatility estimate.

A direct option-premium model would need an options data feed (strikes, IV
surface, greeks); the architecture here transfers directly once you have that
data — just point `config.yaml`'s `features`/`target` at those columns.

## Why predictions are returns, not raw prices

If you predict the **raw price level** on a long-trending stock, an uncertain
model drifts toward the *average price over its whole training history* — which
for something like AAPL is far below today's price. The forecast then looks like
a huge crash even though nothing is being "predicted." That's mean reversion on
a non-stationary series, not insight.

The default `target_mode: logreturn` instead predicts daily **log returns** and
rebuilds the price from the last actual price (`price_next = last × exp(return)`).
Returns are roughly stationary and centered on zero, so an uncertain forecast is
"roughly flat," anchored to today — not −50%. Set `target_mode: price` only for
short or stationary series.

## How it works

1. **Features** — OHLCV plus momentum/volatility indicators.
2. **Windowing** — each sample is `lookback` days of features → next `horizon`
   target values.
3. **Scaling** — `StandardScaler` fit on the training split only (no look-ahead).
4. **Model** — multi-layer LSTM; the final hidden state is projected to all
   `horizon` future values at once (direct multi-step forecasting).
5. **Training** — Adam + MSE, gradient clipping, `ReduceLROnPlateau`, and early
   stopping on validation loss.

## Tests

```bash
python -m pytest -q          # or: python tests/test_pipeline.py
```

## Ideas to extend

- Predict **returns** instead of raw price (often more stationary/learnable).
- Swap the LSTM for a Temporal Convolutional Network or a Transformer encoder.
- Add walk-forward / rolling back-testing with proper financial metrics.
- Quantile / probabilistic outputs to express forecast uncertainty.
