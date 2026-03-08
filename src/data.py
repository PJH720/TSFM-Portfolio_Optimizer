"""
src/data.py
-----------
Local data loader. Reads from the static CSV built by scripts/build_dataset.py.
No real-time API calls — offline-first, deterministic, fast.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Canonical path to the static dataset (relative to project root)
_DATA_FILE = Path(__file__).parent.parent / "data" / "sp500_macro_dataset.csv"

# Cached DataFrame — loaded once per process
_cache: pd.DataFrame | None = None


def _load_cache() -> pd.DataFrame:
    """Load the full dataset into memory (once). Raises FileNotFoundError if missing."""
    global _cache
    if _cache is None:
        if not _DATA_FILE.exists():
            raise FileNotFoundError(
                f"Dataset not found at {_DATA_FILE}.\n"
                "Run:  python scripts/build_dataset.py\n"
                "to build the local dataset first."
            )
        log.info("Loading dataset from %s…", _DATA_FILE)
        _cache = pd.read_csv(_DATA_FILE, parse_dates=["Date"])
        log.info("Dataset loaded: %d rows, tickers: %s", len(_cache), sorted(_cache["Ticker"].unique()))
    return _cache


def get_available_tickers() -> list[str]:
    """Return sorted list of tickers present in the local dataset."""
    df = _load_cache()
    return sorted(df["Ticker"].unique().tolist())


def load_local_data(ticker: str) -> pd.DataFrame:
    """Load historical data for a single ticker from the local static CSV.

    Args:
        ticker: Ticker symbol (must exist in the dataset, e.g. "AAPL").

    Returns:
        DataFrame with columns: Date (DatetimeIndex), Close, Volume,
        plus any macro columns present (DGS10, VIXCLS, UNRATE).
        Sorted ascending by Date, NaN rows in Close dropped.

    Raises:
        ValueError: If the ticker is not found in the local dataset.
        FileNotFoundError: If the dataset CSV has not been built yet.
    """
    df = _load_cache()

    mask = df["Ticker"] == ticker.upper()
    if not mask.any():
        available = sorted(df["Ticker"].unique())
        raise ValueError(
            f"Ticker {ticker} not found in local dataset.\n"
            f"Available tickers: {available}"
        )

    ticker_df = (
        df[mask]
        .copy()
        .set_index("Date")
        .sort_index()
        .drop(columns=["Ticker"])
        .dropna(subset=["Close"])
    )

    log.debug("Loaded %d rows for %s (range: %s → %s)",
              len(ticker_df), ticker,
              ticker_df.index[0].date(), ticker_df.index[-1].date())

    return ticker_df


def get_close_series(ticker: str) -> pd.Series:
    """Convenience wrapper: returns just the Close price Series for a ticker.

    This is the primary input for TimesFM forecasting.
    """
    df = load_local_data(ticker)
    s = df["Close"]
    s.name = ticker
    return s
