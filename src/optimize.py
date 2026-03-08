"""
src/optimize.py
---------------
Markowitz Mean-Variance portfolio optimizer using cvxpy.

Solves:  maximize  μᵀw  -  γ · wᵀΣw
Subject to: Σwᵢ = 1  (fully invested)
             w  ≥ 0  (long-only, no short selling)
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import cvxpy as cp

log = logging.getLogger(__name__)


def optimize_portfolio(
    mu_vector: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
    weight_threshold: float = 1e-3,
) -> dict[str, float]:
    """
    Solve the Markowitz Mean-Variance optimization.

    Args:
        mu_vector:       Expected returns per ticker (pd.Series, index = tickers).
        cov_matrix:      Covariance matrix (pd.DataFrame, index = columns = tickers).
        risk_aversion:   Risk-aversion coefficient γ. Higher = more risk-averse. Default 1.0.
        weight_threshold: Weights below this value are treated as zero and filtered out.

    Returns:
        dict mapping ticker → normalized optimal weight (sums to 1.0).
        Falls back to equal-weight allocation if the solver fails.
    """
    tickers = mu_vector.index.tolist()
    n = len(tickers)

    if n < 2:
        raise ValueError("Need at least 2 tickers to optimize a portfolio.")

    # Align covariance matrix to mu_vector order
    Sigma = cov_matrix.loc[tickers, tickers].values.astype(float)
    mu    = mu_vector.values.astype(float)

    # Ensure Sigma is positive semi-definite (add small regularization)
    Sigma += np.eye(n) * 1e-8

    # ── cvxpy problem ─────────────────────────────────────────────────────────
    w = cp.Variable(n)
    objective   = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, Sigma))
    constraints = [cp.sum(w) == 1, w >= 0]
    problem     = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.OSQP, warm_starting=True)
    except Exception as e:
        log.warning("Primary solver (OSQP) failed: %s — trying SCS.", e)
        try:
            problem.solve(solver=cp.SCS)
        except Exception as e2:
            log.error("All solvers failed: %s. Falling back to equal weight.", e2)
            return {t: 1.0 / n for t in tickers}

    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        log.warning("Solver status: %s — falling back to equal weight.", problem.status)
        return {t: 1.0 / n for t in tickers}

    raw_weights = w.value

    # Filter near-zero weights and renormalize
    weights = {
        ticker: float(weight)
        for ticker, weight in zip(tickers, raw_weights)
        if weight > weight_threshold
    }

    if not weights:
        log.warning("All weights filtered out — equal weight fallback.")
        return {t: 1.0 / n for t in tickers}

    total = sum(weights.values())
    weights = {t: w / total for t, w in weights.items()}

    log.info(
        "Portfolio optimized: %d assets, γ=%.2f, status=%s",
        len(weights), risk_aversion, problem.status,
    )
    return weights
