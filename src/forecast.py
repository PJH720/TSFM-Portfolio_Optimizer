"""Zero-shot time-series forecasting using Google TimesFM 1.0 (200M, PyTorch)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import timesfm

_model: timesfm.TimesFm | None = None


def _load_model() -> timesfm.TimesFm:
    """Lazy-load TimesFM 1.0 200M (PyTorch). Cached after first call.

    Uses google/timesfm-1.0-200m-pytorch — compatible with timesfm v1.x on PyPI.
    Upgrade to timesfm-2.5-200m-pytorch once timesfm>=2.0 lands on PyPI.
    """
    global _model
    if _model is None:
        hf_config = timesfm.TimesFmHparams(
            backend="gpu",            # falls back to cpu if no CUDA
            per_core_batch_size=32,
            horizon_len=128,          # 1.0-200m max horizon
            num_layers=20,
            model_dims=1280,
            use_positional_embedding=True,   # required for 1.0 model
            context_len=512,          # 1.0-200m was trained with 512-token context
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        )
        _model = timesfm.TimesFm(hparams=hf_config, checkpoint=checkpoint)
    return _model


def generate_forecast(
    historical_prices: pd.Series,
    horizon: int = 30,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run zero-shot point + quantile forecast.

    Args:
        historical_prices: pd.Series of closing prices (DatetimeIndex, daily).
        horizon: Number of future trading days to forecast.

    Returns:
        (median_forecast, lower_80, upper_80) as pd.Series indexed by future dates.
    """
    model = _load_model()
    prices = historical_prices.dropna().values.astype(np.float32)

    point_forecasts, quantile_forecasts = model.forecast(
        inputs=[prices],
        freq=[0],  # 0 = daily / high-frequency
    )

    # Slice to requested horizon
    median = point_forecasts[0, :horizon]
    q10 = quantile_forecasts[0, :horizon, 0]   # 10th percentile
    q90 = quantile_forecasts[0, :horizon, 8]   # 90th percentile

    # Build future business-day date index
    last_date = historical_prices.dropna().index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    return (
        pd.Series(median, index=future_dates, name="Forecast (median)"),
        pd.Series(q10,    index=future_dates, name="Lower 80% CI"),
        pd.Series(q90,    index=future_dates, name="Upper 80% CI"),
    )
