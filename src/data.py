"""
src/data.py
-----------
Local data loader. Reads from the static CSV built by scripts/build_dataset.py.
No real-time API calls — offline-first, deterministic, fast.

v2.0 additions (Dual-Model Pipeline):
  - Sector / Industry columns now available
  - Extended macro columns: DGS10, VIXCLS, UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y, UMCSENT
  - get_sector_map()          → dict[ticker → sector]
  - get_industry_map()        → dict[ticker → industry]
  - get_covariate_series()    → DataFrame of macro covariates for a ticker's date range
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# Canonical path to the static dataset (relative to project root)
_DATA_FILE = Path(__file__).parent.parent / "data" / "sp500_macro_dataset.csv"

# All macro covariate columns produced by build_dataset.py v2.0
COVARIATE_COLS = [
    "DGS10",        # 10-Year Treasury rate
    "VIXCLS",       # VIX volatility index
    "UNRATE",       # Unemployment rate
    "CPIAUCSL",     # CPI (Consumer Price Index)
    "BAMLH0A0HYM2", # HY credit spread (market stress)
    "T10Y2Y",       # Yield curve spread (recession signal)
    "UMCSENT",      # Consumer sentiment
]

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
        log.info(
            "Dataset loaded: %d rows | %d tickers | columns: %s",
            len(_cache),
            _cache["Ticker"].nunique(),
            list(_cache.columns),
        )
    return _cache


def invalidate_cache() -> None:
    """Force reload on next access (useful after rebuilding the dataset)."""
    global _cache
    _cache = None


# ─────────────────────────────────────────────────────────────────────────────
# Ticker / sector discovery
# ─────────────────────────────────────────────────────────────────────────────

def get_available_tickers() -> list[str]:
    """Return sorted list of tickers present in the local dataset."""
    df = _load_cache()
    return sorted(df["Ticker"].unique().tolist())


def get_sector_map() -> dict[str, str]:
    """Return mapping of {Ticker: Sector} for all tickers in the dataset.

    Sector names follow GICS classification from sp500_companies.csv.
    Returns empty string for tickers where sector is unavailable.
    """
    df = _load_cache()
    if "Sector" not in df.columns:
        log.warning("Sector column not found — dataset may be from v1.0. Rebuild with v2.0.")
        return {t: "" for t in df["Ticker"].unique()}

    return (
        df.dropna(subset=["Sector"])
        .drop_duplicates("Ticker")
        .set_index("Ticker")["Sector"]
        .to_dict()
    )


def get_industry_map() -> dict[str, str]:
    """Return mapping of {Ticker: Industry} for all tickers in the dataset."""
    df = _load_cache()
    if "Industry" not in df.columns:
        return {t: "" for t in df["Ticker"].unique()}

    return (
        df.dropna(subset=["Industry"])
        .drop_duplicates("Ticker")
        .set_index("Ticker")["Industry"]
        .to_dict()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single-ticker data access
# ─────────────────────────────────────────────────────────────────────────────

def load_local_data(ticker: str) -> pd.DataFrame:
    """Load all available data for a single ticker from the local static CSV.

    Args:
        ticker: Ticker symbol (e.g. "AAPL"). Case-insensitive.

    Returns:
        DataFrame indexed by Date (ascending), with columns:
          Close, Volume, Sector, Industry (if available),
          + macro columns present in the dataset (DGS10, VIXCLS, etc.)
        NaN rows in Close are dropped.

    Raises:
        ValueError:           If the ticker is not in the local dataset.
        FileNotFoundError:    If the dataset CSV has not been built yet.
    """
    df = _load_cache()

    mask = df["Ticker"] == ticker.upper()
    if not mask.any():
        available = sorted(df["Ticker"].unique())
        raise ValueError(
            f"Ticker '{ticker}' not found in local dataset.\n"
            f"Available ({len(available)} tickers): {available}"
        )

    drop_cols = {"Ticker"}
    ticker_df = (
        df[mask]
        .copy()
        .set_index("Date")
        .sort_index()
        .drop(columns=[c for c in drop_cols if c in df.columns])
        .dropna(subset=["Close"])
    )

    log.debug(
        "Loaded %d rows for %s (%s → %s)",
        len(ticker_df), ticker,
        ticker_df.index[0].date(), ticker_df.index[-1].date(),
    )
    return ticker_df


def get_close_series(ticker: str) -> pd.Series:
    """Return the Close price Series for a ticker.

    Primary input for TimesFM and Chronos-2 forecasting.
    """
    df = load_local_data(ticker)
    s = df["Close"]
    s.name = ticker
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Covariate access for Chronos-2
# ─────────────────────────────────────────────────────────────────────────────

def get_covariate_series(
    ticker: str,
    cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Return macro covariate columns aligned to the ticker's date range.

    Args:
        ticker:  Ticker symbol (determines date range / alignment).
        cols:    Covariate columns to return.  Defaults to all available
                 COVARIATE_COLS.  Unknown columns are silently dropped.

    Returns:
        DataFrame indexed by Date with the requested covariate columns.
        All-NaN columns are dropped (indicates series not in dataset).

    Example:
        covariates = get_covariate_series("AAPL", cols=["DGS10", "VIXCLS"])
        # covariates.shape → (n_trading_days, 2)
    """
    if cols is None:
        cols = COVARIATE_COLS

    df = load_local_data(ticker)
    available = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]

    if missing:
        log.debug(
            "Covariate columns not in dataset (rebuild with v2.0): %s", missing
        )

    if not available:
        log.warning(
            "No covariate columns found for %s. "
            "Rebuild dataset: python scripts/build_dataset.py", ticker
        )
        return pd.DataFrame(index=df.index)

    cov_df = df[available].copy()

    # Drop columns that are entirely NaN (FRED download may have failed for them)
    all_nan = cov_df.columns[cov_df.isna().all()]
    if len(all_nan):
        log.debug("Dropping all-NaN covariate columns: %s", list(all_nan))
        cov_df = cov_df.drop(columns=all_nan)

    return cov_df


def get_available_covariates(ticker: str) -> list[str]:
    """Return list of covariate columns that have real data for the given ticker."""
    try:
        cov = get_covariate_series(ticker)
        return list(cov.columns)
    except Exception:
        return []
