"""Data download and feature engineering.

Prices are fetched from Yahoo Finance via ``yfinance``. On top of the raw OHLCV
columns we add a few classic technical-analysis features that help a model
forecast short-term moves. Everything here returns a tidy ``pandas.DataFrame``
indexed by date.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


def download_prices(
    ticker: str,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data for ``ticker`` from Yahoo Finance.

    Returns a DataFrame with columns Open, High, Low, Close, Volume indexed by
    date. Raises a clear error if nothing comes back (bad symbol / no network).
    """
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check the symbol, the "
            f"date range, or your network connection."
        )

    # yfinance can return a MultiIndex (column, ticker) when given one symbol.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.index.name = "Date"
    return df.dropna()


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a handful of momentum / volatility indicators.

    The new columns are appended; rows with NaNs introduced by rolling windows
    are dropped at the end so the result is model-ready.
    """
    out = df.copy()
    close = out["Close"]

    # Returns
    out["Return"] = close.pct_change()
    out["LogReturn"] = np.log(close).diff()

    # Moving averages and their ratio to price
    out["SMA_10"] = close.rolling(10).mean()
    out["SMA_30"] = close.rolling(30).mean()
    out["SMA_ratio"] = out["SMA_10"] / out["SMA_30"]

    # Exponential moving averages -> MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema_12 - ema_26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # Rolling volatility of returns
    out["Volatility_10"] = out["Return"].rolling(10).std()

    # Relative Strength Index (RSI, 14-day)
    out["RSI_14"] = _rsi(close, period=14)

    return out.dropna()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def load_csv(path: str) -> pd.DataFrame:
    """Load OHLCV data from a local CSV (offline / bring-your-own-data path).

    The CSV must have a date column (named Date, or the first column) and the
    usual OHLCV columns (case-insensitive). Useful when there's no network or
    you have your own historical data / option chain.
    """
    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.index.name = "Date"
    df.columns = [str(c).title() for c in df.columns]
    return df.dropna()


TARGET_COL = "__target__"


def apply_target_mode(df: pd.DataFrame, price_col: str, mode: str):
    """Return (df, target_col) prepared for the requested target mode.

    - "price": predict the raw price level directly (target_col == price_col).
    - "logreturn": add a ``__target__`` column of daily log returns of the
      price; the model predicts those and price is rebuilt at inference from the
      last actual price. This keeps the target stationary and anchored to the
      present, avoiding drift toward the multi-year average price.
    """
    df = df.copy()
    if mode == "price":
        return df, price_col
    if mode == "logreturn":
        df[TARGET_COL] = np.log(df[price_col]).diff()
        df = df.dropna()
        return df, TARGET_COL
    raise ValueError(f"Unknown target_mode '{mode}'. Use 'price' or 'logreturn'.")


def build_dataset(
    ticker: str,
    start: str,
    end: Optional[str],
    interval: str,
    features: List[str],
    target: str,
    with_technicals: bool = True,
    csv: Optional[str] = None,
) -> pd.DataFrame:
    """End-to-end: load (download or CSV), engineer features, select columns.

    The returned frame always contains every requested feature column plus the
    target column. If ``csv`` is given, data is read from that file instead of
    being downloaded from Yahoo Finance.
    """
    df = load_csv(csv) if csv else download_prices(ticker, start, end, interval)
    raw_cols = set(df.columns)
    if with_technicals:
        df = add_technical_features(df)

    # Keep the requested features + target, plus any engineered technical
    # indicators (everything that wasn't in the raw OHLCV frame) so the model
    # actually gets to use them as inputs.
    technical_cols = [c for c in df.columns if c not in raw_cols]
    cols = list(dict.fromkeys([*features, *technical_cols, target]))  # de-dupe, keep order

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Requested columns not available: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[cols].copy()
