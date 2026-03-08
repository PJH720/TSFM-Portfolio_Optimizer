"""
scripts/build_dataset.py  (v2.0 — Dual-Model Pipeline Edition)
--------------------------------------------------------------
Dataset selection rationale (printed at runtime — see _log_dataset_selection):

  ✅ andrewmvd/sp-500-stocks  [ALREADY LOCAL: data/raw/sp500_kaggle/]
     sp500_companies.csv  → dynamic ticker universe (502 tickers) + GICS Sector/Industry
     sp500_stocks.csv     → primary price source (2010-2024-12-20); yfinance fills gaps

  ❌ codebynadiia/s-and-p-500-companies-list-with-sectors  [SKIPPED]
     sp500_companies.csv already contains the identical Sector/Industry columns.
     Redundant download not warranted.

  ✅ FRED API  [7 covariates — Chronos-2 past_covariates]
     DGS10, VIXCLS, UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y, UMCSENT
     Preferred over eswaranmuthu/u-s-economic-vital-signs (static Kaggle snapshot)
     because FRED API provides fresher data and identical coverage.

  ❌ ziya07/financial-market-forecasting-dataset  [DEFERRED to Phase 2]
     Market stress already covered by VIXCLS + BAMLH0A0HYM2 from FRED.
     Sentiment features targeted for Phase 2 enrichment.

Pipeline:
  1. Extract ticker universe dynamically from sp500_companies.csv
     (sorted by index Weight; top-N configurable; Sector + Industry retained)
  2. Hybrid price loading:
        a. Parse Kaggle sp500_stocks.csv for tickers with <5% null Close (fast path)
        b. yfinance download for null/missing tickers + dates after 2024-12-20
  3. Attach Sector / Industry metadata from sp500_companies.csv
  4. Download 7 FRED macro series; ffill to daily trading dates
  5. Merge & save → data/sp500_macro_dataset.csv

Usage:
    python scripts/build_dataset.py                # top 30 tickers by index weight
    python scripts/build_dataset.py --top-n 50     # top 50
    python scripts/build_dataset.py --period 7y    # 7 years of history
    python scripts/build_dataset.py --no-fred      # skip FRED (macro cols = NaN)

Output columns:
    Date, Ticker, Sector, Industry, Close, Volume,
    DGS10, VIXCLS, UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y, UMCSENT
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR      = Path(__file__).parent.parent / "data"
RAW_DIR       = DATA_DIR / "raw"
KAGGLE_DIR    = RAW_DIR / "sp500_kaggle"
COMPANIES_CSV = KAGGLE_DIR / "sp500_companies.csv"
STOCKS_CSV    = KAGGLE_DIR / "sp500_stocks.csv"
OUTPUT_CSV    = DATA_DIR / "sp500_macro_dataset.csv"

# ── FRED macro series for Chronos-2 past_covariates ───────────────────────────

FRED_SERIES: dict[str, str] = {
    "DGS10":        "10-Year Treasury Constant Maturity Rate",
    "VIXCLS":       "CBOE Volatility Index (VIX)",
    "UNRATE":       "Unemployment Rate",
    "CPIAUCSL":     "Consumer Price Index (CPI, All Urban Consumers)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Spread (market stress)",
    "T10Y2Y":       "10Y-2Y Treasury Spread (recession indicator)",
    "UMCSENT":      "U. Michigan Consumer Sentiment Index",
}

# Quality threshold: Kaggle Close column null rate above this → fall back to yfinance
NULL_THRESHOLD = 0.05

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Dataset selection rationale
# ─────────────────────────────────────────────────────────────────────────────

def _log_dataset_selection() -> None:
    log.info("=" * 72)
    log.info("DATASET SELECTION — v2.0 Dual-Model Pipeline")
    log.info("=" * 72)
    log.info("")
    log.info("✅  SELECTED  andrewmvd/sp-500-stocks  [local: data/raw/sp500_kaggle/]")
    log.info("    ├─ sp500_companies.csv → dynamic ticker universe + GICS Sector/Industry")
    log.info("    └─ sp500_stocks.csv   → primary price source (Kaggle fast-path,")
    log.info("                            yfinance fallback for null tickers + post-2024)")
    log.info("")
    log.info("❌  SKIPPED   codebynadiia/s-and-p-500-companies-list-with-sectors")
    log.info("    └─ sp500_companies.csv already contains identical Sector/Industry data.")
    log.info("       Downloading a duplicate would waste bandwidth with no new information.")
    log.info("")
    log.info("✅  SELECTED  FRED API  (7 covariates for Chronos-2 past_covariates)")
    log.info("    ├─ DGS10        : 10-Year Treasury rate (risk-free benchmark)")
    log.info("    ├─ VIXCLS       : VIX (short-term market fear)")
    log.info("    ├─ UNRATE       : Unemployment rate (macro health)")
    log.info("    ├─ CPIAUCSL     : CPI inflation (monetary policy driver)")
    log.info("    ├─ BAMLH0A0HYM2 : HY credit spread (systemic credit risk)")
    log.info("    ├─ T10Y2Y       : 10Y-2Y spread (yield curve / recession signal)")
    log.info("    └─ UMCSENT      : Consumer sentiment (demand-side leading indicator)")
    log.info("    Preferred over eswaranmuthu/u-s-economic-vital-signs (static Kaggle")
    log.info("    snapshot) because FRED API is fresher and covers all 7 series.")
    log.info("")
    log.info("❌  DEFERRED  ziya07/financial-market-forecasting-dataset  [Phase 2]")
    log.info("    └─ Market stress already covered by VIXCLS + BAMLH0A0HYM2 from FRED.")
    log.info("       Sentiment/news scores targeted for Phase 2 covariate enrichment.")
    log.info("")
    log.info("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Dynamic ticker extraction from sp500_companies.csv
# ─────────────────────────────────────────────────────────────────────────────

def extract_ticker_universe(top_n: int) -> pd.DataFrame:
    """Extract top-N tickers by S&P 500 index weight from sp500_companies.csv.

    Returns DataFrame with columns: Symbol, Sector, Industry, Weight.
    """
    if not COMPANIES_CSV.exists():
        raise FileNotFoundError(
            f"sp500_companies.csv not found at {COMPANIES_CSV}.\n"
            "Expected from: andrewmvd/sp-500-stocks (data/raw/sp500_kaggle/)."
        )

    companies = pd.read_csv(COMPANIES_CSV)
    companies.columns = companies.columns.str.strip()

    required = {"Symbol", "Sector", "Industry", "Weight"}
    missing = required - set(companies.columns)
    if missing:
        raise ValueError(
            f"sp500_companies.csv is missing columns: {missing}\n"
            f"Found: {list(companies.columns)}"
        )

    companies["Symbol"] = companies["Symbol"].astype(str).str.upper().str.strip()
    companies["Weight"] = pd.to_numeric(companies["Weight"], errors="coerce").fillna(0.0)
    companies = companies.dropna(subset=["Symbol"]).drop_duplicates("Symbol")

    universe = (
        companies[["Symbol", "Sector", "Industry", "Weight"]]
        .sort_values("Weight", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    log.info(
        "Ticker universe: %d tickers selected (top %d by index weight)",
        len(universe), top_n,
    )
    log.info("Sector distribution:\n%s", universe["Sector"].value_counts().to_string())
    return universe


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Hybrid price loading: Kaggle CSV → yfinance fallback
# ─────────────────────────────────────────────────────────────────────────────

def _kaggle_cutoff() -> pd.Timestamp:
    """Latest date available in the local Kaggle stocks CSV (without reading all 92MB)."""
    tail = pd.read_csv(STOCKS_CSV, usecols=["Date"], skiprows=lambda i: i % 1000 != 0)
    return pd.to_datetime(tail["Date"]).max()


def _assess_kaggle_quality(
    tickers: list[str],
    start: pd.Timestamp,
) -> tuple[list[str], list[str]]:
    """Split tickers into Kaggle-OK vs yfinance-needed based on null rate.

    Returns (kaggle_ok, yfinance_needed).
    """
    if not STOCKS_CSV.exists():
        log.warning("sp500_stocks.csv not found — all tickers will use yfinance.")
        return [], list(tickers)

    log.info("Assessing Kaggle stock data quality for %d tickers…", len(tickers))
    stocks = pd.read_csv(STOCKS_CSV, usecols=["Date", "Symbol", "Close"])
    stocks["Date"] = pd.to_datetime(stocks["Date"], errors="coerce")
    stocks = stocks.dropna(subset=["Date"])

    ticker_set = set(t.upper() for t in tickers)
    recent = stocks[(stocks["Symbol"].isin(ticker_set)) & (stocks["Date"] >= start)]

    null_rates = recent.groupby("Symbol")["Close"].apply(
        lambda s: float(s.isna().mean())
    )

    kaggle_ok: list[str] = []
    yfinance_needed: list[str] = []

    for t in tickers:
        rate = null_rates.get(t, 1.0)
        if rate <= NULL_THRESHOLD:
            kaggle_ok.append(t)
        else:
            yfinance_needed.append(t)
            reason = "100% null" if rate == 1.0 else f"{rate:.0%} null"
            log.info("  %-8s → yfinance fallback  (%s in Kaggle CSV)", t, reason)

    log.info(
        "Kaggle OK: %d | yfinance fallback: %d",
        len(kaggle_ok), len(yfinance_needed),
    )
    return kaggle_ok, yfinance_needed


def _load_from_kaggle(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load Close + Volume from local Kaggle stocks CSV for the given tickers."""
    if not tickers:
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "Volume"])

    log.info("Loading %d tickers from Kaggle CSV…", len(tickers))
    stocks = pd.read_csv(STOCKS_CSV)
    stocks["Date"] = pd.to_datetime(stocks["Date"], errors="coerce")
    stocks = stocks.dropna(subset=["Date"])

    ticker_set = set(t.upper() for t in tickers)
    mask = stocks["Symbol"].isin(ticker_set) & (stocks["Date"] >= start) & (stocks["Date"] <= end)
    sub = stocks[mask][["Date", "Symbol", "Close", "Volume"]].copy()
    sub.columns = ["Date", "Ticker", "Close", "Volume"]
    sub = sub.dropna(subset=["Close"])
    sub["Ticker"] = sub["Ticker"].str.upper()

    log.info("Kaggle CSV: %d rows for %d tickers", len(sub), sub["Ticker"].nunique())
    return sub


def _load_from_yfinance(
    tickers: list[str],
    period: str,
) -> pd.DataFrame:
    """Download Close + Volume from yfinance for the given tickers."""
    if not tickers:
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "Volume"])

    log.info("Downloading %d tickers from yfinance (period=%s)…", len(tickers), period)
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=True,
        group_by="ticker",
    )

    frames: list[pd.DataFrame] = []
    # Single ticker returns flat MultiIndex; multiple returns nested
    if len(tickers) == 1:
        t = tickers[0]
        df = raw[["Close", "Volume"]].copy()
        df.index.name = "Date"
        df["Ticker"] = t
        df = df.dropna(subset=["Close"]).reset_index()
        frames.append(df)
    else:
        for t in tickers:
            try:
                df = raw[t][["Close", "Volume"]].copy()
            except KeyError:
                log.warning("yfinance: %s returned no data — skipping.", t)
                continue
            df = df.dropna(subset=["Close"])
            df.index.name = "Date"
            df["Ticker"] = t
            frames.append(df.reset_index())

    if not frames:
        log.warning("yfinance: no data returned for any ticker.")
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "Volume"])

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    log.info("yfinance: %d rows for %d tickers", len(combined), combined["Ticker"].nunique())
    return combined


def load_stock_prices(
    tickers: list[str],
    period: str,
) -> pd.DataFrame:
    """Hybrid price loader: Kaggle fast-path → yfinance fallback.

    For tickers where Kaggle data is null/incomplete, yfinance is used.
    If the Kaggle dataset ends before today, yfinance supplements recent dates.

    Returns long-format DataFrame: Date, Ticker, Close, Volume.
    """
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)

    # Parse period string → start timestamp
    period_map = {"1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "7y": 2555, "10y": 3650}
    days = period_map.get(period, 1825)
    start = now - pd.Timedelta(days=days)

    # Assess Kaggle quality
    kaggle_ok, yf_needed = _assess_kaggle_quality(tickers, start)

    # Determine Kaggle cutoff date
    kaggle_end = pd.Timestamp("2024-12-20")
    need_recent_supplement = (now - kaggle_end).days > 30

    frames: list[pd.DataFrame] = []

    # Load from Kaggle CSV for OK tickers
    if kaggle_ok:
        kaggle_df = _load_from_kaggle(kaggle_ok, start, kaggle_end)
        frames.append(kaggle_df)

    # yfinance for null tickers (full period)
    if yf_needed:
        yf_df = _load_from_yfinance(yf_needed, period)
        frames.append(yf_df)

    # Supplement Kaggle-OK tickers with yfinance for dates after Kaggle cutoff
    if kaggle_ok and need_recent_supplement:
        log.info(
            "Kaggle dataset ends %s — supplementing %d tickers with recent yfinance data…",
            kaggle_end.date(), len(kaggle_ok),
        )
        recent_days = (now - kaggle_end).days + 5
        recent_period = f"{max(recent_days // 30, 1)}mo"
        recent_df = _load_from_yfinance(kaggle_ok, recent_period)
        recent_df = recent_df[recent_df["Date"] > kaggle_end]
        if not recent_df.empty:
            frames.append(recent_df)
            log.info("Appended %d recent rows from yfinance.", len(recent_df))

    if not frames:
        raise RuntimeError("No stock data could be loaded from Kaggle or yfinance.")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = (
        combined
        .sort_values(["Ticker", "Date"])
        .drop_duplicates(subset=["Ticker", "Date"], keep="last")
        .reset_index(drop=True)
    )

    log.info(
        "Stock prices merged: %d rows | %d tickers | %s → %s",
        len(combined),
        combined["Ticker"].nunique(),
        combined["Date"].min().date(),
        combined["Date"].max().date(),
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Attach Sector / Industry metadata
# ─────────────────────────────────────────────────────────────────────────────

def attach_sector_metadata(
    stocks: pd.DataFrame,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join Sector and Industry onto the stock price DataFrame."""
    meta = universe[["Symbol", "Sector", "Industry"]].rename(columns={"Symbol": "Ticker"})
    merged = stocks.merge(meta, on="Ticker", how="left")
    missing = merged["Sector"].isna().sum()
    if missing:
        log.warning("%d rows have no sector metadata (tickers not in companies.csv).", missing)
    log.info("Sector metadata attached.")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FRED macro covariates for Chronos-2
# ─────────────────────────────────────────────────────────────────────────────

def download_fred(series_ids: dict[str, str], skip: bool = False) -> pd.DataFrame:
    """Download macro series from FRED API.

    Returns wide DataFrame indexed by Date.
    Graceful degradation: if FRED_API_KEY missing or skip=True → NaN columns.
    """
    covariate_cols = list(series_ids.keys())

    if skip:
        log.info("FRED download skipped (--no-fred). Macro columns will be NaN.")
        return pd.DataFrame(columns=covariate_cols)

    api_key = os.getenv("FRED_API_KEY", "").strip()

    # Try loading from .env if not in environment
    if not api_key:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("FRED_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not api_key:
        log.warning("FRED_API_KEY not found — macro columns will be NaN.")
        return pd.DataFrame(columns=covariate_cols)

    try:
        from fredapi import Fred
    except ImportError:
        log.error("fredapi not installed. Run: uv pip install fredapi")
        return pd.DataFrame(columns=covariate_cols)

    fred = Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}

    for sid, desc in series_ids.items():
        log.info("FRED ← %s: %s", sid, desc)
        try:
            s = fred.get_series(sid)
            frames[sid] = s
            time.sleep(0.4)
        except Exception as exc:
            log.warning("FRED %s failed: %s — column will be NaN.", sid, exc)

    if not frames:
        return pd.DataFrame(columns=covariate_cols)

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    log.info(
        "FRED: %d rows | series: %s",
        len(macro), list(frames.keys()),
    )
    return macro


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Merge & forward-fill macro onto daily trading dates
# ─────────────────────────────────────────────────────────────────────────────

def merge_macro(stocks: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro covariates onto stock price DataFrame.

    Monthly/weekly FRED series are forward-filled to daily trading dates.
    """
    macro_cols = list(FRED_SERIES.keys())

    if macro.empty or len(macro.columns) == 0:
        for col in macro_cols:
            stocks[col] = np.nan
        return stocks

    all_dates = stocks["Date"].sort_values().unique()
    all_date_idx = pd.DatetimeIndex(all_dates)

    macro_daily = (
        macro
        .reindex(all_date_idx.union(macro.index))
        .sort_index()
        .ffill()
        .bfill()
        .reindex(all_date_idx)
    )
    macro_daily.index.name = "Date"
    macro_daily = macro_daily.reset_index()

    merged = stocks.merge(macro_daily, on="Date", how="left")
    log.info(
        "Final dataset: %d rows | %d tickers | columns: %s",
        len(merged), merged["Ticker"].nunique(), list(merged.columns),
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="v2.0 — Build enriched S&P 500 + macro dataset for TSFM pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--top-n", type=int, default=30,
        help="Number of tickers to select (top by S&P 500 index weight).",
    )
    parser.add_argument(
        "--period", type=str, default="5y",
        choices=["1y", "2y", "3y", "5y", "7y", "10y"],
        help="Historical window for stock prices.",
    )
    parser.add_argument(
        "--no-fred", action="store_true",
        help="Skip FRED download (macro columns will be NaN). Faster for quick tests.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _log_dataset_selection()

    # 1. Dynamic ticker universe
    universe = extract_ticker_universe(top_n=args.top_n)
    tickers = universe["Symbol"].tolist()

    # 2. Hybrid stock price loading
    stocks = load_stock_prices(tickers, period=args.period)

    # 3. Sector / Industry metadata
    stocks = attach_sector_metadata(stocks, universe)

    # 4. FRED macro covariates
    macro = download_fred(FRED_SERIES, skip=args.no_fred)

    # 5. Merge macro → daily grid
    final = merge_macro(stocks, macro)

    # Canonical column order
    canonical = ["Date", "Ticker", "Sector", "Industry", "Close", "Volume"] + list(FRED_SERIES.keys())
    final = final[[c for c in canonical if c in final.columns]]

    final.to_csv(OUTPUT_CSV, index=False)

    n_tickers = final["Ticker"].nunique()
    log.info("")
    log.info("=" * 72)
    log.info("✅  SAVED  →  %s", OUTPUT_CSV)
    log.info("    Rows    : %d", len(final))
    log.info("    Tickers : %d", n_tickers)
    log.info("    Columns : %s", list(final.columns))
    log.info("    Date    : %s → %s", final['Date'].min().date(), final['Date'].max().date())
    log.info("=" * 72)

    # Sanity check — print one ticker sample
    sample_ticker = tickers[0]
    sample = final[final["Ticker"] == sample_ticker].tail(3)
    if not sample.empty:
        log.info("\n%s (last 3 rows):\n%s", sample_ticker, sample.to_string(index=False))

    # Coverage report
    log.info("\nCoverage per ticker:")
    coverage = final.groupby("Ticker")["Close"].count().sort_values(ascending=False)
    for t, cnt in coverage.items():
        log.info("  %-8s  %d rows", t, cnt)


if __name__ == "__main__":
    main()
