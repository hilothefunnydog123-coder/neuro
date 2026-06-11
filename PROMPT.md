# Lovable Prompt (paste this into Lovable)

> Tip: also upload `PROJECT_BRIEF.md` alongside this prompt for the full API spec.

---

Build a single-page web app called **YN Neuro** — a clean, modern fintech-style
dashboard that visualizes stock-price forecasts from an existing backend API.
Do NOT build or run any machine-learning model in the browser; this app only
calls a REST API and displays the results.

**Tech:** React + Vite + TypeScript, Tailwind, shadcn/ui, and Recharts. Dark
mode by default. Mobile responsive. Read the API base URL from
`VITE_API_BASE_URL` (default `http://localhost:8000`).

**Layout (one page):**
1. Header: "YN Neuro" title and tagline "An open-source neural network that
   learns to forecast short-term price movement — with honest, backtested
   accuracy." Add a small green/red dot connection indicator that pings
   `GET /health`.
2. A ticker text input (default "AAPL") and a "Forecast" button.
3. On submit, `POST {VITE_API_BASE_URL}/forecast` with body
   `{ "ticker": <input>, "horizon": 5, "lookback": 60, "epochs": 40, "retrain": false }`.
   Show a loading spinner — the first call for a ticker can take up to 60s.
4. A Recharts line chart: one solid line for the `history` array and one dashed,
   differently-colored line labeled "Model forecast (estimate)" for the
   `forecast` array (each item is `{date, price}`). Connect them at the last
   history point. Hover tooltips, labeled axes.
5. A metrics card showing `metrics.skill_score` prominently:
   - if > 0: green badge "Beats the naive baseline"
   - if ≤ 0: amber badge "Does not beat baseline (expected — honest result)"
   Also show RMSE (model vs naive) and `directional_accuracy` as a percentage.
6. A persistent footer banner that renders the `disclaimer` string from the
   response. It must always be visible.

**Honesty rules:** never call forecasts guaranteed, certain, or "most accurate";
always show the disclaimer; label the forecast as an estimate.

Handle loading and error states gracefully (e.g. show a friendly message if the
API is unreachable). Make it look polished and trustworthy.
