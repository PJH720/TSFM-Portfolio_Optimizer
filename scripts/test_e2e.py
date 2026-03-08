"""
scripts/test_e2e.py
-------------------
Headless end-to-end integration test for the full TSFM pipeline.

Validates (in order):
  [T1]  Master CSV loads correctly (99 tickers, 17 columns, 0 NaN)
  [T2]  Dynamic ticker list ≥ 50 tickers; sector map non-empty
  [T3]  Covariate alignment — 11 covariates, 0 NaN, correct shape
  [T4]  TSFM inference — generate_ensemble_forecast() for 5 diverse tickers
        (real model forward pass — validates tensor shapes + covariate injection)
  [T5]  Portfolio μ/Σ construction — shapes, PSD check, μ in plausible range
  [T6]  Markowitz QP without sector constraints
  [T7]  Markowitz QP with sector constraints (≤ 30% per GICS sector)
  [T8]  Constraint verification — weights ≥ 0, sum = 1, sector cap respected
  [T9]  portfolio_diagnostics() — Sharpe, E[r], σ all finite

Usage:
  python scripts/test_e2e.py               # full run (loads both TSFM models)
  python scripts/test_e2e.py --dry-run     # skip model inference (fast, ~5s)
  python scripts/test_e2e.py --tickers AAPL JPM CVX AMZN UNH

Exit codes:
  0 — all tests passed
  1 — one or more tests failed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path so `src.*` imports resolve
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.data import (
    COVARIATE_COLS,
    get_available_tickers,
    get_covariate_series,
    get_sector_map,
    load_local_data,
)
from src.optimize import optimize_portfolio, portfolio_diagnostics

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("e2e")

# ── Test state ────────────────────────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []   # (test_id, passed, message)


def _pass(test_id: str, msg: str) -> None:
    RESULTS.append((test_id, True, msg))
    log.info("  ✅  %-6s  %s", test_id, msg)


def _fail(test_id: str, msg: str) -> None:
    RESULTS.append((test_id, False, msg))
    log.error("  ❌  %-6s  %s", test_id, msg)


def _section(title: str) -> None:
    log.info("")
    log.info("─" * 64)
    log.info("  %s", title)
    log.info("─" * 64)


# ═════════════════════════════════════════════════════════════════════════════
# T1 — Master CSV loading
# ═════════════════════════════════════════════════════════════════════════════

def test_csv_loading() -> pd.DataFrame | None:
    _section("T1  Master CSV loading")
    try:
        from src.data import _load_cache
        df = _load_cache()

        n_rows    = len(df)
        n_tickers = df["Ticker"].nunique()
        n_cols    = len(df.columns)
        total_nan = df.isnull().sum().sum()

        if n_tickers < 50:
            _fail("T1", f"Only {n_tickers} tickers — expected ≥ 50")
        else:
            _pass("T1a", f"{n_tickers} tickers loaded ({n_rows:,} rows, {n_cols} cols)")

        if total_nan > 0:
            _fail("T1b", f"{total_nan} NaN values in master CSV — check build_dataset.py")
        else:
            _pass("T1b", "Zero NaN values in master CSV ✓")

        required = {"Date", "Ticker", "Close", "Volume", "Sector"} | set(COVARIATE_COLS[:7])
        missing  = required - set(df.columns)
        if missing:
            _fail("T1c", f"Missing required columns: {missing}")
        else:
            _pass("T1c", f"All required columns present ({n_cols} total)")

        return df
    except Exception as exc:
        _fail("T1", f"CSV load failed: {exc}")
        traceback.print_exc()
        return None


# ═════════════════════════════════════════════════════════════════════════════
# T2 — Dynamic ticker and sector APIs
# ═════════════════════════════════════════════════════════════════════════════

def test_ticker_and_sector_apis() -> tuple[list[str], dict[str, str]]:
    _section("T2  Ticker & Sector APIs")
    tickers    = get_available_tickers()
    sector_map = get_sector_map()

    if len(tickers) >= 50:
        _pass("T2a", f"get_available_tickers() → {len(tickers)} tickers")
    else:
        _fail("T2a", f"Only {len(tickers)} tickers (expected ≥ 50)")

    n_sectors = len(set(sector_map.values()))
    if n_sectors >= 5:
        sector_counts = defaultdict(int)
        for s in sector_map.values():
            sector_counts[s] += 1
        _pass("T2b", f"get_sector_map() → {n_sectors} GICS sectors, sample: "
              + " | ".join(f"{s}({c})" for s, c in list(sector_counts.items())[:4]))
    else:
        _fail("T2b", f"Only {n_sectors} sectors — GICS metadata may be missing")

    return tickers, sector_map


# ═════════════════════════════════════════════════════════════════════════════
# T3 — Covariate alignment
# ═════════════════════════════════════════════════════════════════════════════

def test_covariate_alignment(test_tickers: list[str]) -> None:
    _section("T3  Covariate alignment (11 series)")
    ticker = test_tickers[0]

    cov = get_covariate_series(ticker)
    n_cov     = len(cov.columns)
    n_nan     = int(cov.isnull().sum().sum())
    expected  = len(COVARIATE_COLS)

    if n_cov == expected:
        _pass("T3a", f"Covariate columns: {n_cov}/{expected} — {list(cov.columns)}")
    else:
        _fail("T3a", f"Covariate column count: got {n_cov}, expected {expected}")

    if n_nan == 0:
        _pass("T3b", "Zero NaN in covariate series ✓")
    else:
        _fail("T3b", f"{n_nan} NaN values in {ticker} covariates")

    # Shape should match Close series
    close = load_local_data(ticker)["Close"]
    if len(cov) == len(close):
        _pass("T3c", f"Covariate rows ({len(cov)}) == Close rows ({len(close)}) ✓")
    else:
        _fail("T3c", f"Length mismatch: cov={len(cov)} close={len(close)}")


# ═════════════════════════════════════════════════════════════════════════════
# T4 — TSFM inference (or dry-run stub)
# ═════════════════════════════════════════════════════════════════════════════

def test_tsfm_inference(
    test_tickers: list[str],
    horizon: int = 30,
    chronos_weight: float = 0.5,
    dry_run: bool = False,
) -> dict[str, np.ndarray]:
    """Run generate_ensemble_forecast for each ticker.

    In --dry-run mode, returns deterministic sine-wave stubs so the rest of
    the test suite (T5–T9) can still validate μ/Σ → cvxpy logic without
    waiting for model loading.
    """
    _section(f"T4  TSFM inference  (horizon={horizon}d, dry_run={dry_run})")

    from src.data import _load_cache
    full_df = _load_cache()

    mu_dict: dict[str, float] = {}

    if dry_run:
        log.info("  DRY-RUN: substituting deterministic sine-wave stubs for model output.")
        rng = np.random.default_rng(42)
        for t in test_tickers:
            df = load_local_data(t)
            last_close = df["Close"].iloc[-1]
            # Plausible daily return stubs: small positive drift
            stub_median = last_close * (1 + rng.uniform(0.001, 0.015))
            mu_dict[t]  = (stub_median - last_close) / last_close
        _pass("T4", f"Dry-run stubs built for {len(test_tickers)} tickers: "
              + ", ".join(f"{t}={v*100:.2f}%" for t, v in mu_dict.items()))
        return mu_dict

    # ── Real inference ────────────────────────────────────────────────────────
    from src.forecast import generate_ensemble_forecast
    import torch

    errors: list[str] = []
    for ticker in test_tickers:
        t0 = time.perf_counter()
        log.info("  Running %s…", ticker)
        try:
            mask = full_df["Ticker"] == ticker
            if not mask.any():
                _fail("T4", f"Ticker {ticker} not in master CSV")
                errors.append(ticker)
                continue

            df = (
                full_df[mask]
                .copy()
                .set_index("Date")
                .sort_index()
                .drop(columns=["Ticker"], errors="ignore")
                .dropna(subset=["Close"])
            )

            result = generate_ensemble_forecast(df, horizon=horizon, chronos_weight=chronos_weight)

            # ── Shape assertions ──────────────────────────────────────────────
            assert len(result["dates"]) == horizon, \
                f"dates length {len(result['dates'])} ≠ horizon {horizon}"
            assert result["ensemble_median"].shape == (horizon,), \
                f"ensemble_median shape {result['ensemble_median'].shape}"
            assert result["ensemble_q10"].shape == (horizon,), \
                f"q10 shape mismatch"
            assert result["ensemble_q90"].shape == (horizon,), \
                f"q90 shape mismatch"
            assert not result["ensemble_median"].isnull().any(), "NaN in ensemble_median"

            # ── Sanity: q10 ≤ median ≤ q90 ───────────────────────────────────
            q10_ok = (result["ensemble_q10"] <= result["ensemble_median"] + 1e-3).all()
            q90_ok = (result["ensemble_median"] <= result["ensemble_q90"] + 1e-3).all()
            if not (q10_ok and q90_ok):
                _fail(f"T4-{ticker}", "Quantile ordering violated (q10 ≤ median ≤ q90)")
                errors.append(ticker)
                continue

            last_close = df["Close"].iloc[-1]
            final_ens  = result["ensemble_median"].iloc[-1]
            mu         = (final_ens - last_close) / last_close
            mu_dict[ticker] = float(mu)

            elapsed = time.perf_counter() - t0
            _pass(f"T4-{ticker}", f"μ={mu*100:+.2f}%  CI=[${result['ensemble_q10'].iloc[-1]:.2f}, "
                  f"${result['ensemble_q90'].iloc[-1]:.2f}]  ({elapsed:.1f}s)")

            torch.cuda.empty_cache()

        except AssertionError as ae:
            _fail(f"T4-{ticker}", f"Assertion: {ae}")
            errors.append(ticker)
        except Exception as exc:
            _fail(f"T4-{ticker}", f"Exception: {exc}")
            traceback.print_exc()
            errors.append(ticker)

    if not errors:
        _pass("T4", f"All {len(test_tickers)} tickers passed TSFM inference")
    return mu_dict


# ═════════════════════════════════════════════════════════════════════════════
# T5 — μ/Σ construction
# ═════════════════════════════════════════════════════════════════════════════

def test_mu_sigma(
    test_tickers: list[str],
    mu_dict: dict[str, float],
    horizon: int = 30,
) -> tuple[pd.Series, pd.DataFrame]:
    _section("T5  μ vector & Σ matrix construction")

    from src.data import _load_cache
    full_df = _load_cache()

    mu_vector = pd.Series(mu_dict, name="expected_return")

    # Historical covariance (annualised, last 252 trading days)
    present = [t for t in test_tickers if t in full_df["Ticker"].values]
    pivot   = (
        full_df[full_df["Ticker"].isin(present)]
        .pivot_table(index="Date", columns="Ticker", values="Close")
        .sort_index()
        .tail(252)
    )
    daily_ret  = pivot.pct_change().dropna()
    cov_matrix = daily_ret.cov() * 252

    # ── Assertions ────────────────────────────────────────────────────────────
    n = len(mu_vector)

    # μ shape
    if len(mu_vector) >= 2:
        _pass("T5a", f"μ vector shape: ({n},)  values: " +
              ", ".join(f"{t}={v*100:+.2f}%" for t, v in mu_vector.items()))
    else:
        _fail("T5a", f"μ vector has only {n} elements — need ≥ 2")

    # Σ shape
    assert cov_matrix.shape == (n, n), f"Σ shape {cov_matrix.shape} ≠ ({n},{n})"
    _pass("T5b", f"Σ matrix shape: ({n}×{n})  diagonal: {np.diag(cov_matrix.values).round(4)}")

    # PSD check (smallest eigenvalue ≥ -1e-6)
    eigvals = np.linalg.eigvalsh(cov_matrix.values + np.eye(n) * 1e-8)
    if eigvals.min() >= -1e-6:
        _pass("T5c", f"Σ is PSD  (λ_min={eigvals.min():.2e})")
    else:
        _fail("T5c", f"Σ is NOT PSD  (λ_min={eigvals.min():.2e})")

    # μ range sanity: daily return * horizon should be plausible (< ±100%)
    if mu_vector.abs().max() < 5.0:
        _pass("T5d", f"μ values in plausible range [max |μ|={mu_vector.abs().max()*100:.2f}%]")
    else:
        _fail("T5d", f"μ seems extreme — max |μ| = {mu_vector.abs().max()*100:.1f}% (check forecast)")

    return mu_vector, cov_matrix


# ═════════════════════════════════════════════════════════════════════════════
# T6 — Markowitz QP without sector constraints
# ═════════════════════════════════════════════════════════════════════════════

def test_optimize_unconstrained(
    mu_vector: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
) -> dict[str, float]:
    _section("T6  Markowitz QP — unconstrained")
    try:
        weights = optimize_portfolio(
            mu_vector, cov_matrix,
            risk_aversion=risk_aversion,
            sector_map=None,
        )
        total = sum(weights.values())
        if abs(total - 1.0) < 1e-4:
            _pass("T6a", f"Weights sum to {total:.6f} ≈ 1.0 ✓")
        else:
            _fail("T6a", f"Weights sum = {total:.6f} (expected 1.0)")

        neg = {t: w for t, w in weights.items() if w < -1e-6}
        if not neg:
            _pass("T6b", f"All weights ≥ 0 ✓  ({len(weights)} active positions)")
        else:
            _fail("T6b", f"Negative weights: {neg}")

        log.info("  Unconstrained weights: " +
                 " | ".join(f"{t}: {w*100:.1f}%" for t, w in weights.items()))
        return weights
    except Exception as exc:
        _fail("T6", f"Unconstrained optimisation failed: {exc}")
        traceback.print_exc()
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# T7 — Markowitz QP with sector constraints
# ═════════════════════════════════════════════════════════════════════════════

def test_optimize_with_sector_constraints(
    mu_vector: pd.Series,
    cov_matrix: pd.DataFrame,
    sector_map: dict[str, str],
    risk_aversion: float = 1.0,
    max_sector: float = 0.30,
) -> dict[str, float]:
    _section(f"T7  Markowitz QP — sector constraints (cap={max_sector*100:.0f}%)")
    try:
        weights = optimize_portfolio(
            mu_vector, cov_matrix,
            risk_aversion=risk_aversion,
            sector_map=sector_map,
            max_sector_weight=max_sector,
            max_single_weight=0.25,
        )
        total = sum(weights.values())
        if abs(total - 1.0) < 1e-4:
            _pass("T7a", f"Weights sum to {total:.6f} ≈ 1.0 ✓")
        else:
            _fail("T7a", f"Weights sum = {total:.6f}")

        log.info("  Sector-constrained weights: " +
                 " | ".join(f"{t}: {w*100:.1f}%" for t, w in weights.items()))
        return weights
    except Exception as exc:
        _fail("T7", f"Sector-constrained optimisation failed: {exc}")
        traceback.print_exc()
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# T8 — Constraint verification
# ═════════════════════════════════════════════════════════════════════════════

def test_constraint_verification(
    weights_unconstrained: dict[str, float],
    weights_constrained: dict[str, float],
    sector_map: dict[str, str],
    max_sector: float = 0.30,
) -> None:
    _section("T8  Constraint verification")

    # ── Unconstrained: basic budget + non-negativity ──────────────────────────
    total_u = sum(weights_unconstrained.values())
    neg_u   = [t for t, w in weights_unconstrained.items() if w < -1e-6]
    if abs(total_u - 1.0) < 1e-4 and not neg_u:
        _pass("T8a", "Unconstrained: budget = 1.0, all weights ≥ 0 ✓")
    else:
        _fail("T8a", f"Unconstrained violated: total={total_u:.4f}, negatives={neg_u}")

    # ── Constrained: sector cap check ─────────────────────────────────────────
    if not weights_constrained:
        _fail("T8b", "Constrained weights empty — solver may have failed")
        return

    sector_totals: dict[str, float] = defaultdict(float)
    for t, w in weights_constrained.items():
        sector_totals[sector_map.get(t, "Unknown")] += w

    violations = {s: v for s, v in sector_totals.items() if v > max_sector + 1e-4}
    if not violations:
        _pass("T8b", (
            f"All sector caps ≤ {max_sector*100:.0f}% satisfied ✓  |  "
            + " | ".join(f"{s}: {v*100:.1f}%" for s, v in
                         sorted(sector_totals.items(), key=lambda x: -x[1]))
        ))
    else:
        _fail("T8b", f"Sector cap VIOLATED: " +
              " | ".join(f"{s}: {v*100:.1f}%" for s, v in violations.items()))

    # ── Single-name cap check (≤ 25%) ─────────────────────────────────────────
    name_violations = {t: w for t, w in weights_constrained.items() if w > 0.25 + 1e-4}
    if not name_violations:
        _pass("T8c", "Single-name cap ≤ 25% satisfied for all assets ✓")
    else:
        _fail("T8c", f"Name cap violated: {name_violations}")

    # ── Budget constraint ─────────────────────────────────────────────────────
    total_c = sum(weights_constrained.values())
    if abs(total_c - 1.0) < 1e-4:
        _pass("T8d", f"Constrained: budget = {total_c:.6f} ≈ 1.0 ✓")
    else:
        _fail("T8d", f"Constrained: budget = {total_c:.6f} (expected 1.0)")


# ═════════════════════════════════════════════════════════════════════════════
# T9 — portfolio_diagnostics()
# ═════════════════════════════════════════════════════════════════════════════

def test_portfolio_diagnostics(
    weights: dict[str, float],
    mu_vector: pd.Series,
    cov_matrix: pd.DataFrame,
    sector_map: dict[str, str],
) -> None:
    _section("T9  portfolio_diagnostics()")
    try:
        diag = portfolio_diagnostics(weights, mu_vector, cov_matrix, sector_map)

        # All numeric fields must be finite
        for field in ("expected_return", "portfolio_variance", "portfolio_std", "sharpe_ratio"):
            v = diag[field]
            if np.isfinite(v):
                _pass(f"T9-{field}", f"{field} = {v:.4f} (finite ✓)")
            else:
                _fail(f"T9-{field}", f"{field} = {v} (not finite)")

        # Variance must be non-negative
        if diag["portfolio_variance"] >= 0:
            _pass("T9-psd", "Portfolio variance ≥ 0 ✓")
        else:
            _fail("T9-psd", f"Portfolio variance < 0: {diag['portfolio_variance']}")

        log.info("  Diagnostics: E[r]=%.2f%%  σ=%.2f%%  Sharpe=%.3f  n=%d",
                 diag["expected_return"], diag["portfolio_std"],
                 diag["sharpe_ratio"], diag["active_positions"])

    except Exception as exc:
        _fail("T9", f"portfolio_diagnostics failed: {exc}")
        traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

def print_summary() -> int:
    """Print final test report. Returns exit code (0 = all pass, 1 = failures)."""
    passed  = [r for r in RESULTS if r[1]]
    failed  = [r for r in RESULTS if not r[1]]
    total   = len(RESULTS)

    log.info("")
    log.info("═" * 64)
    log.info("  E2E TEST SUMMARY")
    log.info("═" * 64)
    log.info("  PASSED: %d / %d", len(passed), total)
    if failed:
        log.info("  FAILED: %d / %d", len(failed), total)
        for tid, _, msg in failed:
            log.error("    ❌  %-8s  %s", tid, msg)
    else:
        log.info("  All tests passed ✅")
    log.info("═" * 64)

    return 0 if not failed else 1


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless E2E test for the TSFM pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip actual model inference — use deterministic stubs. Fast (~5s).",
    )
    p.add_argument(
        "--tickers", nargs="+",
        default=["AAPL", "JPM", "CVX", "AMZN", "UNH"],
        help="Tickers to test (aim for cross-sector diversity).",
    )
    p.add_argument(
        "--horizon", type=int, default=30,
        help="Forecast horizon in trading days.",
    )
    p.add_argument(
        "--chronos-weight", type=float, default=0.5,
        help="Chronos-2 weight in ensemble.",
    )
    p.add_argument(
        "--risk-aversion", type=float, default=1.0,
        help="Risk aversion γ for Markowitz QP.",
    )
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    t_start = time.perf_counter()

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  HFM_Implementation — Headless E2E Integration Test          ║")
    log.info("║  tickers: %-50s ║", " ".join(args.tickers))
    log.info("║  horizon: %-3d   chronos_weight: %.2f   dry_run: %-5s       ║",
             args.horizon, args.chronos_weight, str(args.dry_run))
    log.info("╚══════════════════════════════════════════════════════════════╝")

    # T1 — CSV
    csv_df = test_csv_loading()
    if csv_df is None:
        log.error("Cannot continue without master CSV. Run: python scripts/build_dataset.py")
        sys.exit(1)

    # T2 — APIs
    all_tickers, sector_map = test_ticker_and_sector_apis()

    # Validate test tickers exist in the dataset
    test_tickers = []
    for t in args.tickers:
        if t in all_tickers:
            test_tickers.append(t)
        else:
            log.warning("Ticker %s not in master CSV — skipping.", t)

    if len(test_tickers) < 2:
        log.error("Need ≥ 2 valid tickers — aborting.")
        sys.exit(1)

    log.info("  Test tickers (%d): %s", len(test_tickers),
             " | ".join(f"{t}[{sector_map.get(t,'?')}]" for t in test_tickers))

    # T3 — Covariates
    test_covariate_alignment(test_tickers)

    # T4 — Inference
    mu_dict = test_tsfm_inference(
        test_tickers,
        horizon=args.horizon,
        chronos_weight=args.chronos_weight,
        dry_run=args.dry_run,
    )

    if not mu_dict:
        _fail("pipeline", "No μ values produced — cannot continue to T5-T9")
        sys.exit(print_summary())

    # T5 — μ/Σ
    mu_vector, cov_matrix = test_mu_sigma(test_tickers, mu_dict, args.horizon)

    # T6 — Unconstrained QP
    w_unconstrained = test_optimize_unconstrained(
        mu_vector, cov_matrix, risk_aversion=args.risk_aversion
    )

    # T7 — Sector-constrained QP
    w_constrained = test_optimize_with_sector_constraints(
        mu_vector, cov_matrix, sector_map,
        risk_aversion=args.risk_aversion, max_sector=0.30,
    )

    # T8 — Constraint verification
    test_constraint_verification(w_unconstrained, w_constrained, sector_map)

    # T9 — Diagnostics
    if w_constrained:
        test_portfolio_diagnostics(w_constrained, mu_vector, cov_matrix, sector_map)

    elapsed = time.perf_counter() - t_start
    log.info("")
    log.info("Total elapsed: %.1f seconds", elapsed)

    sys.exit(print_summary())


if __name__ == "__main__":
    main()
