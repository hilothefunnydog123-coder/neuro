# YN Neuro — Web App Brief (upload this to Lovable)

Build a clean, modern web frontend for an existing stock-forecasting API. The
machine-learning model already exists and runs as a separate backend service —
**do not try to build or run the model in the browser.** This app only needs to
call the API and visualize the results.

## What the app does

A user types a stock ticker (e.g. AAPL), clicks "Forecast", and sees:
1. A line chart of recent price history plus the model's forward forecast.
2. A small panel of honest accuracy metrics (a "skill score" vs. a baseline).
3. A clear, always-visible disclaimer.

## The backend API (already built — just call it)

Base URL is configured via an env var `VITE_API_BASE_URL`
(default to `http://localhost:8000` for local dev).

### `POST /forecast`

Request JSON:
```json
{ "ticker": "AAPL", "horizon": 5, "lookback": 60, "epochs": 40, "retrain": false }
```
Only `ticker` is required; send the rest as sensible defaults.

Response JSON:
```json
{
  "ticker": "AAPL",
  "history":  [ { "date": "2024-01-02", "price": 184.2 }, ... ],
  "forecast": [ { "date": "2024-06-12", "price": 190.1 }, ... ],
  "metrics": {
    "samples": 142,
    "horizon": 5,
    "rmse_model": 3.21,
    "rmse_naive": 3.04,
    "mae_model": 2.6,
    "mae_naive": 2.4,
    "skill_score": -0.05,
    "directional_accuracy": 0.52
  },
  "disclaimer": "Educational research tool... Not financial advice."
}
```
Note: the **first** request for a new ticker may take 20–60s because the model
trains on the fly; show a loading state. Repeat requests are fast (cached).

### `GET /health` → `{ "status": "ok" }` (use for a connection indicator)

## Pages / UI

Single page is fine:
- **Header:** "YN Neuro" logo/title + one-line tagline:
  *"An open-source neural network that learns to forecast short-term price
  movement — with honest, backtested accuracy."*
- **Search bar:** ticker text input + "Forecast" button. Loading spinner while
  the request is in flight.
- **Chart:** one line for `history` and a second, visually distinct
  (dashed/different color) line for `forecast`, joined at the last history point.
  Use Recharts. Tooltips on hover. Label axes (Date / Price).
- **Metrics panel:** show `skill_score` prominently with a plain-English label:
  - skill_score > 0 → green "Beats the naive baseline"
  - skill_score ≤ 0 → neutral/amber "Does not beat the baseline (expected — this
    is hard and honest)"
  Also show RMSE (model vs naive) and directional accuracy as a percentage.
- **Disclaimer:** render the `disclaimer` string from the response in a
  persistent footer/banner. This must never be hidden.

## Design

- Modern, minimal, trustworthy fintech aesthetic. Dark mode default.
- Tailwind + shadcn/ui components. Rounded cards, subtle shadows.
- Mobile responsive.
- Accent color: a calm teal/blue. Use green/amber only for the skill-score badge.

## Honesty requirements (important — do not violate)

- Never describe forecasts as guaranteed, certain, or "the most accurate."
- Always show the disclaimer and the skill score together with the chart.
- Label the forecast line as "Model forecast (estimate)".

## Out of scope

- No user accounts, payments, or trading execution.
- No running the ML model in the browser — only call the API.
