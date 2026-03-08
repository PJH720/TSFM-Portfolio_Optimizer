"""
scripts/build_dataset.py
------------------------
Bulk-downloads daily OHLCV for the top 20 S&P 500 tickers and merges in
FRED macroeconomic indicators. Saves the result to data/sp500_macro_dataset.csv.

Usage:
    FRED_API_KEY=<your_key> python scripts/build_dataset.py

Outputs:
    data/sp500_macro_dataset.csv  — columns:
        Date, Ticker, Close, Volume, DGS10, VIXCLS, UNRATE
"""

import os
import sys
import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "LLY", "JNJ", "MA",
    "AVGO", "PG", "HD", "MRK", "COST",
]

PERIOD       = "5y"          # 5 years of history
DATA_DIR     = Path(__file__).parent.parent / "data"
OUTPUT_CSV   = DATA_DIR / "sp500_macro_dataset.csv"

FRED_SERIES  = {
    "DGS10":  "10-Year Treasury Constant Maturity Rate",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "UNRATE": "Unemployment Rate",
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Stock downloader ──────────────────────────────────────────────────────────

def download_stocks(tickers: list[str], period: str) -> pd.DataFrame:
    """Download daily Close + Volume for all tickers in one yfinance call.

    Returns a long-format DataFrame with columns: Date, Ticker, Close, Volume.
    """
    log.info("Downloading stock data for %d tickers (period=%s)…", len(tickers), period)
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
    )

    frames = []
    for ticker in tickers:
        try:
            df = raw[ticker][["Close", "Volume"]].copy()
        except KeyError:
            log.warning("Ticker %s returned no data — skipping.", ticker)
            continue
        df = df.dropna(subset=["Close"])
        df.index.name = "Date"
        df["Ticker"] = ticker
        frames.append(df.reset_index())

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    log.info("Stock data: %d rows across %d tickers.", len(combined), len(frames))
    return combined


# ── FRED downloader ───────────────────────────────────────────────────────────

def download_fred(series_ids: dict[str, str]) -> pd.DataFrame:
    """Download macro series from FRED. Requires FRED_API_KEY env var.

    Returns a wide DataFrame indexed by Date with one column per series.
    If FRED_API_KEY is missing, returns an empty DataFrame with NaN columns
    so the rest of the pipeline still works (graceful degradation).
    """
    api_key = os.getenv("FRED_API_KEY", "").strip()

    if not api_key:
        log.warning(
            "FRED_API_KEY not set — macro columns will be NaN. "
            "Set the env var and re-run to include macro data."
        )
        return pd.DataFrame(columns=list(series_ids.keys()))

    try:
        from fredapi import Fred
    except ImportError:
        log.error("fredapi not installed. Run: uv pip install fredapi")
        return pd.DataFrame(columns=list(series_ids.keys()))

    fred = Fred(api_key=api_key)
    frames = {}
    for sid, desc in series_ids.items():
        log.info("Fetching FRED series: %s (%s)…", sid, desc)
        try:
            s = fred.get_series(sid)
            frames[sid] = s
            time.sleep(0.5)   # be polite to FRED API
        except Exception as exc:
            log.warning("Failed to fetch %s: %s — column will be NaN.", sid, exc)

    if not frames:
        return pd.DataFrame(columns=list(series_ids.keys()))

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    log.info("FRED data: %d rows, series: %s", len(macro), list(frames.keys()))
    return macro


# ── Merger ────────────────────────────────────────────────────────────────────

def merge_and_fill(stocks: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro onto stocks by Date, then forward-fill macro gaps.

    Macro data (monthly/weekly) is forward-filled to match daily trading dates.
    Missing trailing values are backward-filled as a last resort.
    """
    if macro.empty:
        for col in FRED_SERIES:
            stocks[col] = float("nan")
        return stocks

    # Reindex macro to all stock dates, then ffill
    all_dates = stocks["Date"].sort_values().unique()
    macro_daily = (
        macro
        .reindex(pd.DatetimeIndex(all_dates).union(macro.index))
        .sort_index()
        .ffill()
        .bfill()
        .reindex(all_dates)
    )
    macro_daily.index.name = "Date"
    macro_daily = macro_daily.reset_index()

    merged = stocks.merge(macro_daily, on="Date", how="left")
    log.info("Merged dataset: %d rows, columns: %s", len(merged), list(merged.columns))
    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    stocks = download_stocks(TICKERS, PERIOD)
    macro  = download_fred(FRED_SERIES)
    merged = merge_and_fill(stocks, macro)

    # Canonical column order
    cols = ["Date", "Ticker", "Close", "Volume"] + list(FRED_SERIES.keys())
    merged = merged[[c for c in cols if c in merged.columns]]

    merged.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved → %s  (%d rows, %d tickers)", OUTPUT_CSV, len(merged), merged["Ticker"].nunique())

    # Quick sanity check
    sample = merged[merged["Ticker"] == "AAPL"].tail(3)
    log.info("AAPL sample (last 3 rows):\n%s", sample.to_string(index=False))


if __name__ == "__main__":
    main()
