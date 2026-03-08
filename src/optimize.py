"""
src/optimize.py
---------------
Markowitz Mean-Variance portfolio optimizer using cvxpy.

Solves:  maximize  μᵀw  −  γ · wᵀΣw
Subject to:
    Σwᵢ = 1                          (fully invested)
    wᵢ ≥ 0                           (long-only, no short selling)
    Σ wᵢ ≤ max_sector_weight         (∀ sector — sector neutrality cap)
    wᵢ ≤ max_single_weight           (∀ i      — single-name concentration cap)
"""

from __future__ import annotations

import logging
from collections import defaultdict

import cvxpy as cp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Default constraint parameters ─────────────────────────────────────────────
_DEFAULT_MAX_SECTOR_WEIGHT = 0.30   # No single GICS sector > 30%
_DEFAULT_MAX_SINGLE_WEIGHT = 0.25   # No single name  > 25%


def _build_sector_index_map(
    tickers: list[str],
    sector_map: dict[str, str],
) -> dict[str, list[int]]:
    """Group ticker indices by sector label.

    Returns:
        {sector_name: [idx0, idx1, …]}  for every sector present in tickers.
    """
    index_map: dict[str, list[int]] = defaultdict(list)
    for i, ticker in enumerate(tickers):
        sector = sector_map.get(ticker, "Unknown")
        index_map[sector].append(i)
    return dict(index_map)


def optimize_portfolio(
    mu_vector: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    weight_threshold: float = 1e-3,
    sector_map: dict[str, str] | None = None,
    max_sector_weight: float = _DEFAULT_MAX_SECTOR_WEIGHT,
    max_single_weight: float = _DEFAULT_MAX_SINGLE_WEIGHT,
) -> dict[str, float]:
    """
    Solve the Markowitz Mean-Variance optimization with optional sector constraints.

    Args:
        mu_vector:          Expected returns per ticker (pd.Series, index = tickers).
        cov_matrix:         Annualised covariance matrix (pd.DataFrame, index = columns = tickers).
        risk_aversion:      Risk-aversion coefficient γ.  Higher → more conservative. Default 1.0.
        weight_threshold:   Weights below this threshold are treated as zero post-solve.
        sector_map:         Optional dict {ticker: sector_label}.  When provided, adds sector
                            concentration constraints to the QP.  Obtained via get_sector_map().
        max_sector_weight:  Maximum aggregate weight for any single GICS sector.  Default 0.30.
        max_single_weight:  Maximum weight for any individual ticker.  Default 0.25.

    Returns:
        dict mapping ticker → normalised optimal weight (values sum to 1.0).
        Falls back to equal-weight allocation if all solvers fail.

    Constraint summary (when sector_map provided):
        ∑ wᵢ  =  1                                      (budget)
        wᵢ  ≥  0  ∀ i                                   (long-only)
        wᵢ  ≤  max_single_weight  ∀ i                   (name cap)
        ∑_{i ∈ sector_s} wᵢ  ≤  max_sector_weight  ∀ s  (sector cap)
    """
    tickers = mu_vector.index.tolist()
    n = len(tickers)

    if n < 2:
        raise ValueError("Need at least 2 tickers to optimise a portfolio.")

    # ── Align Σ to mu order ───────────────────────────────────────────────────
    Sigma = cov_matrix.loc[tickers, tickers].values.astype(float)
    mu    = mu_vector.values.astype(float)

    # Regularise to guarantee PSD
    Sigma += np.eye(n) * 1e-8

    # ── cvxpy decision variable ───────────────────────────────────────────────
    w = cp.Variable(n)

    # ── Objective: Sharpe-proxy (maximise risk-adjusted return) ───────────────
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))

    # ── Base constraints ──────────────────────────────────────────────────────
    constraints: list = [
        cp.sum(w) == 1,          # fully invested
        w >= 0,                  # long-only
        w <= max_single_weight,  # single-name concentration cap
    ]

    # ── Sector constraints ────────────────────────────────────────────────────
    sector_constraint_count = 0
    if sector_map:
        sector_indices = _build_sector_index_map(tickers, sector_map)

        for sector, indices in sector_indices.items():
            idx  = np.array(indices)
            cap  = max_sector_weight

            # If all tickers in the portfolio belong to one sector we'd create an
            # infeasible problem — skip the constraint in that degenerate case.
            if len(indices) == n:
                log.warning(
                    "All %d tickers are in sector '%s' — skipping sector cap "
                    "(constraint would be infeasible with budget = 1).",
                    n, sector,
                )
                continue

            constraints.append(cp.sum(w[idx]) <= cap)
            sector_constraint_count += 1
            log.debug(
                "Sector constraint: Σw[%s] ≤ %.0f%%  (%d assets: %s)",
                sector, cap * 100, len(indices),
                [tickers[i] for i in indices],
            )

        log.info(
            "Sector constraints active: %d sectors capped at %.0f%%",
            sector_constraint_count, max_sector_weight * 100,
        )
    else:
        log.info("No sector_map provided — running unconstrained (long-only + budget + name cap).")

    # ── Solve ─────────────────────────────────────────────────────────────────
    problem = cp.Problem(objective, constraints)

    solvers = [
        (cp.OSQP,  {"warm_starting": True, "eps_abs": 1e-5, "eps_rel": 1e-5}),
        (cp.SCS,   {"eps": 1e-4}),
        (cp.ECOS,  {}),
    ]

    status = None
    for solver, kwargs in solvers:
        try:
            problem.solve(solver=solver, **kwargs)
            status = problem.status
            if status in ("optimal", "optimal_inaccurate") and w.value is not None:
                break
            log.warning("Solver %s status: %s — trying next.", solver, status)
        except Exception as exc:
            log.warning("Solver %s raised: %s — trying next.", solver, exc)

    if status not in ("optimal", "optimal_inaccurate") or w.value is None:
        log.error("All solvers failed (status=%s) — equal-weight fallback.", status)
        return {t: 1.0 / n for t in tickers}

    # ── Post-processing ───────────────────────────────────────────────────────
    raw = w.value

    # Clip numerical noise (negative near-zero weights)
    raw = np.clip(raw, 0.0, None)

    weights = {
        ticker: float(weight)
        for ticker, weight in zip(tickers, raw)
        if weight > weight_threshold
    }

    if not weights:
        log.warning("All weights below threshold — equal-weight fallback.")
        return {t: 1.0 / n for t in tickers}

    # Renormalise to exactly 1.0
    total = sum(weights.values())
    weights = {t: v / total for t, v in weights.items()}

    # ── Diagnostic report ─────────────────────────────────────────────────────
    log.info(
        "Optimisation complete: %d/%d assets allocated | γ=%.2f | solver status=%s",
        len(weights), n, risk_aversion, status,
    )

    if sector_map:
        # Report actual sector concentrations in final solution
        sector_actuals: dict[str, float] = defaultdict(float)
        for ticker, wt in weights.items():
            sector_actuals[sector_map.get(ticker, "Unknown")] += wt
        log.info("Actual sector weights:")
        for sec, wt in sorted(sector_actuals.items(), key=lambda x: -x[1]):
            flag = " ⚠️ EXCEEDS CAP" if wt > max_sector_weight + 1e-6 else ""
            log.info("  %-28s %.1f%%%s", sec, wt * 100, flag)

    return weights


def portfolio_diagnostics(
    weights: dict[str, float],
    mu_vector: pd.Series,
    cov_matrix: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> dict:
    """Compute portfolio-level performance diagnostics.

    Returns dict with:
        expected_return   — annualised portfolio expected return (%)
        portfolio_variance — annualised portfolio variance
        portfolio_std      — annualised portfolio volatility (%)
        sharpe_ratio       — expected_return / portfolio_std  (rf = 0)
        sector_weights     — dict {sector: aggregate_weight}
        active_positions   — number of non-trivial allocations
    """
    tickers  = list(weights.keys())
    w_arr    = np.array([weights[t] for t in tickers])
    mu_arr   = mu_vector.reindex(tickers).fillna(0.0).values
    Sigma    = cov_matrix.loc[tickers, tickers].values

    port_return = float(w_arr @ mu_arr) * 100          # %
    port_var    = float(w_arr @ Sigma @ w_arr)
    port_std    = float(np.sqrt(port_var)) * 100        # %
    sharpe      = port_return / port_std if port_std > 1e-8 else 0.0

    sector_weights: dict[str, float] = defaultdict(float)
    if sector_map:
        for t, wt in weights.items():
            sector_weights[sector_map.get(t, "Unknown")] += wt

    return {
        "expected_return":    round(port_return, 4),
        "portfolio_variance": round(port_var, 6),
        "portfolio_std":      round(port_std, 4),
        "sharpe_ratio":       round(sharpe, 4),
        "sector_weights":     dict(sector_weights),
        "active_positions":   len(weights),
    }
