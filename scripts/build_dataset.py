"""
scripts/build_dataset.py  (v3.0 — Fully Automated Master Pipeline)
===================================================================
Produces:  data/sp500_macro_master.csv

Pipeline overview
-----------------
  Phase 0 · Environment & Kaggle auth
  Phase 1 · Automated Kaggle dataset downloads (idempotent)f
  Phase 2 · Dynamic ticker quality assessment & universe selection
  Phase 3 · Hybrid stock price loading  (Kaggle CSV → yfinance fallback)
  Phase 4 · Sector / Industry metadata join
  Phase 5 · FRED macro covariates (7 series via fredapi)
  Phase 6 · Kaggle macro dataset (eswaranmuthu — 6 additional series)
  Phase 7 · FRED + Kaggle macro merge  (FRED primary, Kaggle backup/extension)
  Phase 8 · Align all macro series to daily trading-date grid
  Phase 9 · Missing-value strategy  →  ffill → bfill → linear interpolate
  Phase 10· Final merge & save

Output schema
-------------
  Date, Ticker, Close, Volume, Sector, Industry,
  DGS10, VIXCLS, UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y, UMCSENT,  ← FRED
  FEDFUNDS_RATE, M2_MONEY_SUPPLY, SOFR, INFLATION_RATE               ← Kaggle macro

Usage
-----
  python scripts/build_dataset.py                  # top 100 tickers, 5y history
  python scripts/build_dataset.py --top-n 50       # top 50
  python scripts/build_dataset.py --period 7y      # 7 years
  python scripts/build_dataset.py --no-fred        # skip FRED (NaN macro cols)
  python scripts/build_dataset.py --force-download # re-download even if cached
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR          = Path(__file__).parent.parent / "data"
RAW_DIR           = DATA_DIR / "raw"
KAGGLE_STOCKS_DIR = RAW_DIR / "sp500_kaggle"
KAGGLE_MACRO_DIR  = RAW_DIR / "us_economic"

COMPANIES_CSV = KAGGLE_STOCKS_DIR / "sp500_companies.csv"
STOCKS_CSV    = KAGGLE_STOCKS_DIR / "sp500_stocks.csv"
KAGGLE_MACRO_CSV = KAGGLE_MACRO_DIR / "macro_data_25yrs.csv"

OUTPUT_CSV    = DATA_DIR / "sp500_macro_master.csv"

# Kaggle dataset slugs
_KG_STOCKS  = "andrewmvd/sp-500-stocks"
_KG_MACRO   = "eswaranmuthu/u-s-economic-vital-signs-25-years-of-macro-data"

# ── FRED series ───────────────────────────────────────────────────────────────   

FRED_SERIES: dict[str, str] = {
    "DGS10":        "10-Year Treasury Constant Maturity Rate",
    "VIXCLS":       "CBOE Volatility Index (VIX)", 
    "UNRATE":       "Unemployment Rate",
    "CPIAUCSL":     "Consumer Price Index (All Urban Consumers)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Spread (market stress)",
    "T10Y2Y":       "10Y-2Y Treasury Spread (recession indicator)",
    "UMCSENT":      "U. Michigan Consumer Sentiment Index",
}

# Kaggle macro columns → canonical output names
KAGGLE_MACRO_MAP: dict[str, str] = {
    "10Y Treasury Yield": "_KAG_DGS10",        # backup for FRED DGS10
    "CPI":                "_KAG_CPIAUCSL",     # backup for FRED CPIAUCSL
    "Fed Funds Rate":     "FEDFUNDS_RATE",
    "M2_Money_Supply":    "M2_MONEY_SUPPLY",
    "SOFR":               "SOFR",
    "Inflation_Rate_%":   "INFLATION_RATE",
}

# Final covariate column ordering (output)
COVARIATE_COLS = [
    # FRED primary
    "DGS10", "VIXCLS", "UNRATE", "CPIAUCSL",
    "BAMLH0A0HYM2", "T10Y2Y", "UMCSENT",
    # Kaggle macro extensions
    "FEDFUNDS_RATE", "M2_MONEY_SUPPLY", "SOFR", "INFLATION_RATE",
]

# Quality threshold for Kaggle CSV fast-path
_NULL_THRESH   = 0.05   # >5% null → yfinance fallback
_PERIOD_MAP    = {
    "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "7y": 2555, "10y": 3650
}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 0 — Environment & Kaggle authentication
# ═════════════════════════════════════════════════════════════════════════════

def _load_env() -> None:
    """Load .env from project root into os.environ (best-effort)."""
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def _kaggle_auth() -> "KaggleApi":  # type: ignore[name-defined]
    """Authenticate against Kaggle API using ~/.config/kaggle/kaggle.json.

    If kaggle.json is missing, writes it from KAGGLE_USERNAME / KAGGLE_KEY env vars.
    """
    kaggle_cfg = Path.home() / ".config" / "kaggle" / "kaggle.json"
    if not kaggle_cfg.exists():
        username = os.environ.get("KAGGLE_USERNAME", "")
        key      = os.environ.get("KAGGLE_KEY", "")
        if not username or not key:
            raise EnvironmentError(
                "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY "
                "in .env, or create ~/.config/kaggle/kaggle.json manually."
            )
        kaggle_cfg.parent.mkdir(parents=True, exist_ok=True)
        kaggle_cfg.write_text(f'{{"username":"{username}","key":"{key}"}}')
        kaggle_cfg.chmod(0o600)
        log.info("Wrote Kaggle credentials → %s", kaggle_cfg)

    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_cfg.parent)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    log.info("Kaggle API authenticated.")
    return api


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Automated Kaggle dataset downloads (idempotent)
# ═════════════════════════════════════════════════════════════════════════════

def _download_if_missing(
    api: "KaggleApi",
    dataset_slug: str,
    dest_dir: Path,
    sentinel_file: Path,
    force: bool = False,
) -> None:
    """Download a Kaggle dataset and unzip into dest_dir (skip if already present)."""
    if not force and sentinel_file.exists():
        log.info("  ✓ Already cached: %s  (%s)", dataset_slug, sentinel_file.name)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("  ↓ Downloading: %s → %s", dataset_slug, dest_dir)
    api.dataset_download_files(dataset_slug, path=str(dest_dir), unzip=True, quiet=False)
    log.info("  ✓ Downloaded: %s", dataset_slug)


def download_kaggle_datasets(api: "KaggleApi", force: bool = False) -> None:
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 1 — Kaggle Dataset Downloads")
    log.info("═" * 68)

    _download_if_missing(api, _KG_STOCKS, KAGGLE_STOCKS_DIR, STOCKS_CSV, force)
    _download_if_missing(api, _KG_MACRO,  KAGGLE_MACRO_DIR,  KAGGLE_MACRO_CSV, force)

    log.info("All Kaggle datasets ready.")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 — Dynamic ticker quality assessment & universe selection
# ═════════════════════════════════════════════════════════════════════════════

def assess_and_select_tickers(top_n: int, period: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Dynamically filter & rank tickers.

    Strategy:
      1. Compute Close null-rate per symbol over the requested history window.
      2. GOOD (null_rate < _NULL_THRESH)  → Kaggle CSV fast-path.
         FALLBACK                         → yfinance download.
      3. Load sp500_companies.csv for index Weight ranking & sector info.
      4. Return top-N by Weight regardless of source (GOOD + FALLBACK combined).

    Returns:
        (universe_df, kaggle_ok_tickers, yfinance_needed_tickers)
    """
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 2 — Dynamic Ticker Assessment (top-%d, period=%s)", top_n, period)
    log.info("═" * 68)

    days  = _PERIOD_MAP.get(period, 1825)
    start = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=days)

    # ── Load stock price index (just Date+Symbol+Close for speed) ─────────────
    log.info("Loading Kaggle stocks CSV for quality assessment (~1.9M rows)…")
    stocks = pd.read_csv(STOCKS_CSV, usecols=["Date", "Symbol", "Close"])
    stocks["Date"] = pd.to_datetime(stocks["Date"], errors="coerce")
    stocks.dropna(subset=["Date"], inplace=True)

    recent = stocks[stocks["Date"] >= start]
    log.info("  Window [%s → latest]: %d rows, %d unique symbols",
             start.date(), len(recent), recent["Symbol"].nunique())

    # ── Per-ticker null stats ─────────────────────────────────────────────────
    stats = recent.groupby("Symbol").agg(
        total =("Close", "size"),
        valid =("Close", lambda x: x.notna().sum()),
    )
    stats["null_rate"] = (stats["total"] - stats["valid"]) / stats["total"]
    stats["coverage"]  = stats["valid"] / stats["total"]

    n_good    = (stats["null_rate"] < _NULL_THRESH).sum()
    n_bad     = (stats["null_rate"] >= 0.95).sum()
    n_partial = len(stats) - n_good - n_bad
    log.info(
        "  Data quality: GOOD=%d (Kaggle fast-path)  BAD=%d (yfinance)  PARTIAL=%d",
        n_good, n_bad, n_partial,
    )

    # ── Load company metadata for Weight ranking ──────────────────────────────
    companies = pd.read_csv(COMPANIES_CSV)
    companies.columns = companies.columns.str.strip()
    companies["Symbol"] = companies["Symbol"].astype(str).str.upper().str.strip()
    companies["Weight"] = pd.to_numeric(companies.get("Weight", 0), errors="coerce").fillna(0.0)
    companies = companies.drop_duplicates("Symbol").set_index("Symbol")

    # ── Join quality stats onto company metadata ──────────────────────────────
    universe = companies.join(stats, how="left")
    universe["null_rate"] = universe["null_rate"].fillna(1.0)
    universe["valid"]     = universe["valid"].fillna(0).astype(int)
    universe["is_good"]   = universe["null_rate"] < _NULL_THRESH

    # ── Rank by S&P 500 index Weight → take top_n ─────────────────────────────
    ranked = universe.sort_values("Weight", ascending=False).head(top_n).copy()
    ranked.index.name = "Symbol"

    kaggle_ok  = ranked[ranked["is_good"]].index.tolist()
    yf_needed  = ranked[~ranked["is_good"]].index.tolist()

    log.info("  Selected top-%d tickers by S&P 500 index weight:", top_n)
    log.info("    Kaggle fast-path : %d tickers", len(kaggle_ok))
    log.info("    yfinance fallback: %d tickers", len(yf_needed))
    if yf_needed:
        log.info("    yfinance tickers : %s", yf_needed[:10],)

    # Sector distribution
    sector_counts = ranked["Sector"].value_counts()
    log.info("  Sector distribution:\n%s", sector_counts.to_string())

    universe_df = ranked[["Sector", "Industry", "Weight", "Marketcap", "is_good"]].reset_index()
    universe_df.rename(columns={"Symbol": "Ticker"}, inplace=True)

    return universe_df, kaggle_ok, yf_needed


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3 — Hybrid stock price loading
# ═════════════════════════════════════════════════════════════════════════════

def _kaggle_cutoff_date() -> pd.Timestamp:
    """Estimate Kaggle stocks CSV max date (sample every 10k rows to avoid full load)."""
    sample = pd.read_csv(STOCKS_CSV, usecols=["Date"],
                         skiprows=lambda i: i % 10_000 != 0)
    return pd.to_datetime(sample["Date"], errors="coerce").max()


def _load_from_kaggle(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "Volume"])

    log.info("  Loading %d tickers from Kaggle CSV…", len(tickers))
    raw = pd.read_csv(STOCKS_CSV)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw.dropna(subset=["Date"], inplace=True)

    ticker_set = {t.upper() for t in tickers}
    mask = raw["Symbol"].isin(ticker_set) & (raw["Date"] >= start) & (raw["Date"] <= end)
    sub  = raw.loc[mask, ["Date", "Symbol", "Close", "Volume"]].copy()
    sub.columns = ["Date", "Ticker", "Close", "Volume"]
    sub.dropna(subset=["Close"], inplace=True)
    sub["Ticker"] = sub["Ticker"].str.upper()

    n = sub["Ticker"].nunique()
    log.info("  Kaggle CSV → %d rows for %d/%d tickers", len(sub), n, len(tickers))
    return sub


def _load_from_yfinance(tickers: list[str], period: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "Volume"])

    log.info("  yfinance download: %d tickers (period=%s)…", len(tickers), period)
    raw = yf.download(
        tickers if len(tickers) > 1 else tickers[0],
        period=period,
        auto_adjust=True,
        progress=True,
        group_by="ticker" if len(tickers) > 1 else None,
    )

    frames: list[pd.DataFrame] = []
    if len(tickers) == 1:
        df = raw[["Close", "Volume"]].copy()
        df.index.name = "Date"
        df["Ticker"] = tickers[0]
        df.dropna(subset=["Close"], inplace=True)
        frames.append(df.reset_index())
    else:
        for t in tickers:
            try:
                df = raw[t][["Close", "Volume"]].copy()
            except KeyError:
                log.warning("  yfinance: %s returned no data — skipping.", t)
                continue
            df.dropna(subset=["Close"], inplace=True)
            df.index.name = "Date"
            df["Ticker"] = t
            frames.append(df.reset_index())

    if not frames:
        log.warning("  yfinance: no data returned.")
        return pd.DataFrame(columns=["Date", "Ticker", "Close", "Volume"])

    out = pd.concat(frames, ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    n = out["Ticker"].nunique()
    log.info("  yfinance → %d rows for %d tickers", len(out), n)
    return out


def load_stock_prices(
    kaggle_ok: list[str],
    yf_needed: list[str],
    period: str,
) -> pd.DataFrame:
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 3 — Hybrid Stock Price Loading")
    log.info("═" * 68)

    now   = pd.Timestamp.now(tz="UTC").tz_localize(None)
    days  = _PERIOD_MAP.get(period, 1825)
    start = now - pd.Timedelta(days=days)

    kaggle_end = _kaggle_cutoff_date()
    log.info("Kaggle CSV last date: %s | history start: %s", kaggle_end.date(), start.date())

    frames: list[pd.DataFrame] = []

    # A. Kaggle fast-path
    if kaggle_ok:
        frames.append(_load_from_kaggle(kaggle_ok, start, kaggle_end))

    # B. yfinance for broken Kaggle tickers (full period)
    if yf_needed:
        frames.append(_load_from_yfinance(yf_needed, period))

    # C. yfinance recency supplement for Kaggle-good tickers (post Kaggle cutoff)
    gap_days = (now - kaggle_end).days
    if kaggle_ok and gap_days > 10:
        supplement_period = f"{max(gap_days // 30 + 1, 2)}mo"
        log.info(
            "  Supplementing %d Kaggle tickers with recent yfinance data (%s)…",
            len(kaggle_ok), supplement_period,
        )
        recent_df = _load_from_yfinance(kaggle_ok, supplement_period)
        recent_df = recent_df[recent_df["Date"] > kaggle_end]
        if not recent_df.empty:
            log.info("  Recency supplement: +%d rows (post %s)", len(recent_df), kaggle_end.date())
            frames.append(recent_df)

    if not frames:
        raise RuntimeError("No stock data could be loaded from any source.")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    # Deduplicate (keep last for a given ticker+date — yfinance wins over Kaggle on overlap)
    combined.sort_values(["Ticker", "Date"], inplace=True)
    combined.drop_duplicates(subset=["Ticker", "Date"], keep="last", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    log.info(
        "Stock prices: %d rows | %d tickers | %s → %s",
        len(combined),
        combined["Ticker"].nunique(),
        combined["Date"].min().date(),
        combined["Date"].max().date(),
    )
    return combined


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 — Sector / Industry metadata join
# ═════════════════════════════════════════════════════════════════════════════

def attach_sector(stocks: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 4 — Sector / Industry Metadata Join")
    log.info("═" * 68)

    meta = universe[["Ticker", "Sector", "Industry"]].drop_duplicates("Ticker")
    out  = stocks.merge(meta, on="Ticker", how="left")
    missing = out["Sector"].isna().sum()
    if missing:
        log.warning("  %d rows have no sector metadata.", missing)
    log.info("  Sector metadata attached.")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5 — FRED macro covariates
# ═════════════════════════════════════════════════════════════════════════════

def download_fred(skip: bool = False) -> pd.DataFrame:
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 5 — FRED Macro Covariates")
    log.info("═" * 68)

    cols = list(FRED_SERIES.keys())

    if skip:
        log.info("  Skipped (--no-fred). FRED columns will be NaN.")
        return pd.DataFrame(columns=cols)

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        log.warning("  FRED_API_KEY not set — FRED columns will be NaN.")
        return pd.DataFrame(columns=cols)

    try:
        from fredapi import Fred
    except ImportError:
        log.error("  fredapi not installed. Run: uv pip install fredapi")
        return pd.DataFrame(columns=cols)

    fred   = Fred(api_key=api_key)
    frames: dict[str, pd.Series] = {}

    for sid, desc in FRED_SERIES.items():
        log.info("  FRED ← %s : %s", sid, desc)
        try:
            s = fred.get_series(sid)
            frames[sid] = s
            time.sleep(0.4)     # polite delay
        except Exception as exc:
            log.warning("  FRED %s failed: %s — column will be NaN.", sid, exc)

    if not frames:
        return pd.DataFrame(columns=cols)

    macro = pd.DataFrame(frames)
    macro.index = pd.to_datetime(macro.index)
    macro.index.name = "Date"
    log.info("  FRED: %d rows, series: %s", len(macro), list(frames.keys()))
    return macro


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6 — Kaggle macro dataset (eswaranmuthu)
# ═════════════════════════════════════════════════════════════════════════════

def load_kaggle_macro() -> pd.DataFrame:
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 6 — Kaggle Macro Dataset (eswaranmuthu)")
    log.info("═" * 68)

    if not KAGGLE_MACRO_CSV.exists():
        log.warning("  %s not found — Kaggle macro features skipped.", KAGGLE_MACRO_CSV)
        return pd.DataFrame()

    raw = pd.read_csv(KAGGLE_MACRO_CSV, parse_dates=["Date"])
    raw.set_index("Date", inplace=True)
    raw.index = pd.to_datetime(raw.index)
    raw.sort_index(inplace=True)

    # Rename to canonical names
    rename = {k: v for k, v in KAGGLE_MACRO_MAP.items() if k in raw.columns}
    raw.rename(columns=rename, inplace=True)

    # Keep only the columns we mapped
    keep = [v for v in KAGGLE_MACRO_MAP.values() if v in raw.columns]
    out  = raw[keep].copy()

    # Parse numeric (some may be string)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    log.info("  Kaggle macro: %d rows (%s → %s) | cols: %s",
             len(out), out.index.min().date(), out.index.max().date(), list(out.columns))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7 — FRED + Kaggle macro merge
# ═════════════════════════════════════════════════════════════════════════════

def merge_macro_sources(fred_df: pd.DataFrame, kaggle_macro: pd.DataFrame) -> pd.DataFrame:
    """Combine FRED and Kaggle macro into a single wide DataFrame.

    Strategy:
      • FRED primary for DGS10, CPIAUCSL (fill remaining NaN from Kaggle backup).
      • Kaggle-only columns (FEDFUNDS_RATE, M2_MONEY_SUPPLY, SOFR, INFLATION_RATE)
        are added directly.
      • Backup columns (_KAG_*) are used only to fill FRED gaps, then dropped.
    """
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 7 — FRED + Kaggle Macro Merge")
    log.info("═" * 68)

    # ── Combine on a unified date index ───────────────────────────────────────
    if fred_df.empty and kaggle_macro.empty:
        log.warning("  Both FRED and Kaggle macro are empty — no covariates will be set.")
        return pd.DataFrame()

    if fred_df.empty:
        combined = kaggle_macro.copy()
    elif kaggle_macro.empty:
        combined = fred_df.copy()
    else:
        combined = fred_df.join(kaggle_macro, how="outer", rsuffix="_kag")

    combined.sort_index(inplace=True)

    # ── Use Kaggle backup to fill FRED gaps ───────────────────────────────────
    for fred_col, kaggle_backup in [("DGS10", "_KAG_DGS10"), ("CPIAUCSL", "_KAG_CPIAUCSL")]:
        if fred_col in combined.columns and kaggle_backup in combined.columns:
            before = combined[fred_col].isna().sum()
            combined[fred_col] = combined[fred_col].fillna(combined[kaggle_backup])
            after  = combined[fred_col].isna().sum()
            if before - after:
                log.info("  Filled %d %s gaps using Kaggle backup.", before - after, fred_col)

    # ── Drop internal backup columns ─────────────────────────────────────────
    backup_cols = [c for c in combined.columns if c.startswith("_KAG_")]
    combined.drop(columns=backup_cols, inplace=True, errors="ignore")

    # ── Report final covariate set ────────────────────────────────────────────
    present = [c for c in COVARIATE_COLS if c in combined.columns]
    absent  = [c for c in COVARIATE_COLS if c not in combined.columns]
    log.info("  Covariate columns available (%d): %s", len(present), present)
    if absent:
        log.info("  Covariate columns absent     (%d): %s", len(absent), absent)

    return combined


# ═════════════════════════════════════════════════════════════════════════════
# Phase 8 — Align macro to daily trading-date grid
# ═════════════════════════════════════════════════════════════════════════════

def align_macro_to_trading_dates(
    stocks: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """Reindex macro series to the exact trading dates present in the stock data.

    Monthly/weekly series are forward-filled to fill daily gaps.
    """
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 8 — Align Macro to Daily Trading Grid")
    log.info("═" * 68)

    covariate_cols = [c for c in COVARIATE_COLS if c in macro.columns]

    if macro.empty or not covariate_cols:
        for col in COVARIATE_COLS:
            stocks[col] = np.nan
        return stocks

    trading_dates = pd.DatetimeIndex(sorted(stocks["Date"].unique()))

    # Reindex macro onto a superset of trading dates + its own dates, then ffill
    all_dates     = trading_dates.union(macro.index).sort_values()
    macro_aligned = (
        macro[covariate_cols]
        .reindex(all_dates)
        .ffill()   # forward-fill sparse monthly/weekly values
        .bfill()   # back-fill any leading NaN (e.g. SOFR before 2018)
        .reindex(trading_dates)
    )
    macro_aligned.index.name = "Date"
    macro_aligned_df = macro_aligned.reset_index()

    merged = stocks.merge(macro_aligned_df, on="Date", how="left")
    log.info(
        "  Aligned %d covariate series to %d unique trading dates.",
        len(covariate_cols), len(trading_dates),
    )
    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Phase 9 — Missing value strategy
# ═════════════════════════════════════════════════════════════════════════════

def apply_missing_value_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a three-pass missing value strategy to all covariate columns.

    Pass 1: forward-fill   (carry last known value forward)
    Pass 2: backward-fill  (handle leading NaN for short series like SOFR)
    Pass 3: linear interpolate (fill any remaining interior gaps)

    Stock-level columns (Close, Volume) use forward-fill only within each ticker.
    """
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 9 — Missing Value Strategy")
    log.info("═" * 68)

    covariate_cols = [c for c in COVARIATE_COLS if c in df.columns]

    # ── Covariate columns: global (same value across all tickers for same date) ──
    # They were already aligned in Phase 8; apply interpolation as safety net.
    for col in covariate_cols:
        before = df[col].isna().sum()
        if before == 0:
            continue
        df[col] = df[col].ffill().bfill()
        # Final pass: linear interpolation for any interior gaps
        df[col] = df[col].interpolate(method="linear", limit_direction="both")
        after = df[col].isna().sum()
        log.info(
            "  %-20s : %d NaN before → %d after fill/interpolate",
            col, before, after,
        )

    # ── Close / Volume: forward-fill within each ticker ───────────────────────
    for col in ("Close", "Volume"):
        if col not in df.columns:
            continue
        before = df[col].isna().sum()
        if before == 0:
            continue
        df[col] = df.groupby("Ticker")[col].transform(lambda s: s.ffill().bfill())
        after = df[col].isna().sum()
        log.info("  %-20s : %d NaN before → %d after ticker-level fill", col, before, after)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_remaining = df[covariate_cols + ["Close"]].isna().sum().sum()
    log.info("  Total remaining NaN in key columns: %d", total_remaining)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Phase 10 — Final merge & save
# ═════════════════════════════════════════════════════════════════════════════

def save_master(df: pd.DataFrame) -> None:
    log.info("")
    log.info("═" * 68)
    log.info("PHASE 10 — Save sp500_macro_master.csv")
    log.info("═" * 68)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Canonical column order
    meta_cols      = ["Date", "Ticker", "Close", "Volume", "Sector", "Industry"]
    covariate_cols = [c for c in COVARIATE_COLS if c in df.columns]
    final_cols     = meta_cols + covariate_cols

    out = df[[c for c in final_cols if c in df.columns]].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.date  # strip time component
    out.sort_values(["Ticker", "Date"], inplace=True)
    out.reset_index(drop=True, inplace=True)

    out.to_csv(OUTPUT_CSV, index=False)

    # ── Summary report ────────────────────────────────────────────────────────
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════════╗")
    log.info("║  ✅  SAVED → data/sp500_macro_master.csv                       ║")
    log.info("╠══════════════════════════════════════════════════════════════════╣")
    log.info("║  Rows      : %-52d ║", len(out))
    log.info("║  Tickers   : %-52d ║", out["Ticker"].nunique())
    log.info("║  Date range: %-52s ║",
             f"{out['Date'].min()} → {out['Date'].max()}")
    log.info("║  Columns   : %-52d ║", len(out.columns))
    log.info("╚══════════════════════════════════════════════════════════════════╝")
    log.info("  Columns: %s", list(out.columns))

    # Coverage check
    log.info("")
    log.info("Per-ticker row count:")
    coverage = out.groupby("Ticker")["Close"].count().sort_values(ascending=False)
    for ticker, cnt in coverage.items():
        sector = out[out["Ticker"] == ticker]["Sector"].iloc[0] if not out[out["Ticker"] == ticker].empty else "?"
        log.info("  %-8s  %d rows  [%s]", ticker, cnt, sector)

    # Covariate fill-rate
    log.info("")
    log.info("Covariate fill rate (% non-null):")
    for col in covariate_cols:
        fill_pct = out[col].notna().mean() * 100
        bar = "█" * int(fill_pct // 5) + "░" * (20 - int(fill_pct // 5))
        log.info("  %-22s [%s] %.1f%%", col, bar, fill_pct)


# ═════════════════════════════════════════════════════════════════════════════
# CLI & main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v3.0 — Fully automated S&P 500 + macro master dataset builder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--top-n",  type=int, default=100,
                   help="Number of tickers to select by S&P 500 index weight.")
    p.add_argument("--period", type=str, default="5y",
                   choices=list(_PERIOD_MAP.keys()),
                   help="Historical window for stock prices.")
    p.add_argument("--no-fred", action="store_true",
                   help="Skip FRED API calls (macro columns = NaN). Faster for tests.")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download Kaggle datasets even if cached.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("╔══════════════════════════════════════════════════════════════════╗")
    log.info("║   build_dataset.py  v3.0 — Fully Automated Master Pipeline      ║")
    log.info("║   top-n=%-4d  period=%-5s  fred=%-5s  force=%-5s              ║",
             args.top_n, args.period,
             "OFF" if args.no_fred else "ON",
             "YES" if args.force_download else "no")
    log.info("╚══════════════════════════════════════════════════════════════════╝")

    # Phase 0
    _load_env()
    api = _kaggle_auth()

    # Phase 1 — download Kaggle datasets
    download_kaggle_datasets(api, force=args.force_download)

    # Phase 2 — dynamic ticker selection
    universe, kaggle_ok, yf_needed = assess_and_select_tickers(args.top_n, args.period)

    # Phase 3 — hybrid price loading
    stocks = load_stock_prices(kaggle_ok, yf_needed, args.period)

    # Phase 4 — sector metadata
    stocks = attach_sector(stocks, universe)

    # Phase 5 — FRED
    fred_df = download_fred(skip=args.no_fred)

    # Phase 6 — Kaggle macro
    kaggle_macro = load_kaggle_macro()

    # Phase 7 — merge macro sources
    macro_wide = merge_macro_sources(fred_df, kaggle_macro)

    # Phase 8 — align to trading dates
    stocks = align_macro_to_trading_dates(stocks, macro_wide)

    # Phase 9 — missing value strategy
    stocks = apply_missing_value_strategy(stocks)

    # Phase 10 — save
    save_master(stocks)


if __name__ == "__main__":
    main()
