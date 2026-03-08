"""
src/forecast.py
---------------
Dual-model ensemble forecasting: Chronos-2 (with macro covariates) + TimesFM 1.0.

Models are lazy-loaded as module-level singletons to avoid repeated initialization.
Chronos-2 leverages past covariates (DGS10, VIXCLS) when available in the dataset;
TimesFM operates univariate on Close prices only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

log = logging.getLogger(__name__)

# ── Lazy-cached model singletons ─────────────────────────────────────────────

_chronos_pipeline = None
_timesfm_model = None


def _get_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_chronos():
    """Load Chronos-2 pipeline (amazon/chronos-2). Cached after first call."""
    global _chronos_pipeline
    if _chronos_pipeline is None:
        from chronos import Chronos2Pipeline

        device = _get_device()
        log.info("Loading Chronos-2 pipeline on %s…", device)
        _chronos_pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map=device if device != "mps" else "cpu",  # Chronos uses device_map; MPS not supported
            torch_dtype=torch.float32,
        )
        torch.cuda.empty_cache()
        log.info("Chronos-2 loaded.")
    return _chronos_pipeline


def _load_timesfm():
    """Load TimesFM 1.0 200M (PyTorch). Cached after first call."""
    global _timesfm_model
    if _timesfm_model is None:
        import timesfm

        log.info("Loading TimesFM 1.0-200m-pytorch…")
        hparams = timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=128,
            num_layers=20,
            model_dims=1280,
            use_positional_embedding=True,   # Required for 1.0 model
            context_len=512,
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        )
        _timesfm_model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
        torch.cuda.empty_cache()
        log.info("TimesFM 1.0 loaded.")
    return _timesfm_model


# ── Individual model forecasters ─────────────────────────────────────────────


def get_chronos_forecast(
    df: pd.DataFrame, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Chronos-2 forecast with optional macro covariates.

    Args:
        df: DataFrame from load_local_data() — must have 'Close' column,
            may have 'DGS10' and 'VIXCLS' macro columns.
        horizon: Number of future trading days to predict.

    Returns:
        (median, q10, q90) as numpy arrays of length `horizon`.
    """
    pipeline = _load_chronos()

    # Target: close prices
    close = df["Close"].dropna().values.astype(np.float64)

    # Build input dict — always has target
    input_dict: dict = {"target": close}

    # Add past covariates if macro columns exist and are not all-NaN
    covariates = {}
    for col in ("DGS10", "VIXCLS"):
        if col in df.columns and not df[col].isna().all():
            # Forward-fill then back-fill to match target length, convert to float64
            vals = df[col].ffill().bfill().values.astype(np.float64)
            covariates[col] = vals

    if covariates:
        input_dict["past_covariates"] = covariates
        log.info("Chronos-2: using past covariates %s", list(covariates.keys()))
    else:
        log.info("Chronos-2: univariate mode (no macro covariates available)")

    # Run prediction
    results = pipeline.predict(
        inputs=[input_dict],
        prediction_length=horizon,
        limit_prediction_length=False,
    )

    # results[0] shape: (n_variates, n_quantiles, prediction_length)
    # n_quantiles=9, order: [0.1, 0.2, …, 0.9]
    # For univariate target: n_variates=1, so index [0]
    tensor = results[0]  # shape: (1, 9, horizon)
    median = tensor[0, 4, :].cpu().numpy()  # quantile 0.5
    q10 = tensor[0, 0, :].cpu().numpy()     # quantile 0.1
    q90 = tensor[0, 8, :].cpu().numpy()     # quantile 0.9

    return median, q10, q90


def get_timesfm_forecast(
    df: pd.DataFrame, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run TimesFM 1.0 univariate forecast on Close prices.

    Args:
        df: DataFrame from load_local_data() — must have 'Close' column.
        horizon: Number of future trading days to predict (max 128).

    Returns:
        (median, q10, q90) as numpy arrays of length `horizon`.
    """
    model = _load_timesfm()

    prices = df["Close"].dropna().values.astype(np.float32)

    # TimesFM forecast returns (point_forecasts, quantile_forecasts)
    # point_forecasts: shape (1, horizon_len)
    # quantile_forecasts: shape (1, horizon_len, 9)
    point_forecasts, quantile_forecasts = model.forecast(
        inputs=[prices],
        freq=[0],  # 0 = daily / high-frequency
    )

    # Slice to requested horizon (model may produce up to horizon_len=128)
    median = point_forecasts[0, :horizon]
    q10 = quantile_forecasts[0, :horizon, 0]   # quantile 0.1
    q90 = quantile_forecasts[0, :horizon, 8]   # quantile 0.9

    return median, q10, q90


# ── Ensemble forecaster ──────────────────────────────────────────────────────


def generate_ensemble_forecast(
    df: pd.DataFrame,
    horizon: int = 30,
    chronos_weight: float = 0.5,
) -> dict:
    """Run dual-model ensemble: Chronos-2 + TimesFM 1.0.

    Runs Chronos-2 first (with macro covariates if available), then TimesFM
    (univariate). Combines results via weighted average.

    Args:
        df: Full DataFrame from load_local_data() with Close + optional macro columns.
        horizon: Number of future trading days to forecast.
        chronos_weight: Weight for Chronos-2 predictions (0.0–1.0).
                        TimesFM weight = 1.0 - chronos_weight.

    Returns:
        Dict with keys:
            dates           — pd.DatetimeIndex of future business days
            ensemble_median — pd.Series (weighted average of medians)
            ensemble_q10    — pd.Series (weighted average of q10)
            ensemble_q90    — pd.Series (weighted average of q90)
            chronos_median  — pd.Series (Chronos-2 median only)
            timesfm_median  — pd.Series (TimesFM median only)
    """
    timesfm_weight = 1.0 - chronos_weight

    # --- Chronos-2 ---
    log.info("Running Chronos-2 forecast (horizon=%d)…", horizon)
    c_median, c_q10, c_q90 = get_chronos_forecast(df, horizon)
    torch.cuda.empty_cache()

    # --- TimesFM ---
    log.info("Running TimesFM forecast (horizon=%d)…", horizon)
    t_median, t_q10, t_q90 = get_timesfm_forecast(df, horizon)

    # --- Weighted ensemble ---
    ens_median = chronos_weight * c_median + timesfm_weight * t_median
    ens_q10 = chronos_weight * c_q10 + timesfm_weight * t_q10
    ens_q90 = chronos_weight * c_q90 + timesfm_weight * t_q90

    # --- Build future date index ---
    last_date = df.index[-1]
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon
    )

    return {
        "dates": future_dates,
        "ensemble_median": pd.Series(ens_median, index=future_dates, name="Ensemble Median"),
        "ensemble_q10": pd.Series(ens_q10, index=future_dates, name="Ensemble Q10"),
        "ensemble_q90": pd.Series(ens_q90, index=future_dates, name="Ensemble Q90"),
        "chronos_median": pd.Series(c_median, index=future_dates, name="Chronos-2 Median"),
        "timesfm_median": pd.Series(t_median, index=future_dates, name="TimesFM Median"),
    }
