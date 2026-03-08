"""
scripts/run_experiments.py
--------------------------
"나만의 한 끗" (Unique Edge) experiment suite for the FastCampus report.

Experiment 1 — TimesFM Context-Length Sweep
    Holds the ticker, horizon, and all other parameters constant while varying
    TimesFM's context_len parameter (512 → 256 → 128 → 64).
    Captures: MAPE, MAE, forecast shape, and quantile spread per context.
    Insight: How much historical context actually matters for zero-shot accuracy?

Experiment 2 — Chronos-2 NF4 Quantization vs FP32 VRAM Comparison
    Loads Chronos-2 twice:
        a) Standard  — torch.float32 (baseline)
        b) NF4 quant — bitsandbytes (4-bit NormalFloat via BitsAndBytesConfig)
    Captures: GPU VRAM before/after loading, inference latency, forecast delta.
    Insight: How much accuracy do we trade for ~4× memory reduction?

Experiment 3 — Ensemble Weight Sweep (Chronos-2 vs TimesFM)
    For a fixed ticker and horizon, sweeps chronos_weight in [0.0, 0.25, 0.5, 0.75, 1.0].
    Captures: how the ensemble forecast shape and CI width change with blending ratio.
    Insight: Is 50/50 optimal or does one model dominate?

Usage:
    python scripts/run_experiments.py --exp 1          # context sweep only
    python scripts/run_experiments.py --exp 2          # NF4 VRAM comparison
    python scripts/run_experiments.py --exp 3          # ensemble weight sweep
    python scripts/run_experiments.py                  # all experiments
    python scripts/run_experiments.py --ticker MSFT --horizon 20
    python scripts/run_experiments.py --exp 1 --no-save  # don't write CSV

Output:
    results/exp1_context_sweep.csv
    results/exp2_nf4_vram.csv
    results/exp3_weight_sweep.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_available_tickers, load_local_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("experiments")

RESULTS_DIR = Path(__file__).parent.parent / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _vram_mb() -> float:
    """Return current GPU allocated VRAM in MB (0.0 if no GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 ** 2
    except Exception:
        pass
    return 0.0


def _gpu_total_mb() -> float:
    """Return total GPU VRAM in MB (0.0 if no GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
    except Exception:
        pass
    return 0.0


def _last_n_close(df: pd.DataFrame, n: int) -> np.ndarray:
    """Return last n Close prices as float64 array."""
    return df["Close"].dropna().values[-n:].astype(np.float64)


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def _header(title: str) -> None:
    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  %-60s ║", title)
    log.info("╚══════════════════════════════════════════════════════════════╝")


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 1 — TimesFM Context-Length Sweep
# ═════════════════════════════════════════════════════════════════════════════

def exp1_context_sweep(
    ticker: str = "AAPL",
    horizon: int = 30,
    context_lengths: list[int] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Sweeps TimesFM context_len while holding everything else constant.

    Methodology:
      - Use the last (context_len + horizon) days of data.
      - Treat the final `horizon` days as a pseudo-holdout (walk-forward).
      - Compare TimesFM's forecast for those `horizon` days against actual Close.
      - Report MAPE, MAE, mean CI width, and inference time per context length.

    TimesFM 1.0 config note:
      context_len=512 is the model's trained maximum. Shorter contexts are padded
      internally by the model. Setting a very short context (64) forces the model
      to forecast with minimal historical signal — useful for ablation.
    """
    _header("Experiment 1 — TimesFM Context-Length Sweep")

    if context_lengths is None:
        context_lengths = [512, 256, 128, 64]

    df = load_local_data(ticker)
    log.info("Ticker: %s | %d rows | horizon: %d | contexts: %s",
             ticker, len(df), horizon, context_lengths)

    import timesfm
    import torch

    records: list[dict] = []

    for ctx in context_lengths:
        log.info("─── context_len = %d ───", ctx)

        # ── Instantiate a fresh TimesFM model with this context_len ──────────
        # NOTE: The TimesFM 1.0 model was trained with context_len=512.
        #       Using smaller ctx doesn't change model weights — it only controls
        #       how many context points are fed as input (the rest is zero-padded).
        hparams = timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=max(horizon, 128),  # always load full 128-head
            num_layers=20,
            model_dims=1280,
            use_positional_embedding=True,  # required for 1.0
            context_len=ctx,                # ← SWEEP VARIABLE
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
        )
        log.info("  Loading TimesFM (context_len=%d)…", ctx)
        model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)

        # ── Walk-forward pseudo-holdout ───────────────────────────────────────
        # Use last (ctx + horizon) rows: context window + holdout
        total_needed = ctx + horizon
        if len(df) < total_needed:
            log.warning("  Not enough data (%d rows < %d needed) — using all data.",
                        len(df), total_needed)
            prices_all = df["Close"].dropna().values.astype(np.float32)
        else:
            prices_all = df["Close"].dropna().values[-total_needed:].astype(np.float32)

        context_prices = prices_all[:ctx]       # feed as context
        actual_prices  = prices_all[ctx:ctx + horizon]   # pseudo-holdout

        t0 = time.perf_counter()
        point_fc, quant_fc = model.forecast(
            inputs=[context_prices],
            freq=[0],   # 0 = daily frequency
        )
        elapsed = time.perf_counter() - t0

        # Slice to requested horizon
        pred_median = point_fc[0, :horizon]
        pred_q10    = quant_fc[0, :horizon, 0]
        pred_q90    = quant_fc[0, :horizon, 8]
        ci_width    = float(np.mean(pred_q90 - pred_q10))

        # Metrics (compare forecast vs actual holdout)
        mape = _mape(actual_prices, pred_median)
        mae  = _mae(actual_prices, pred_median)

        vram = _vram_mb()

        record = {
            "ticker":         ticker,
            "horizon":        horizon,
            "context_len":    ctx,
            "mape_pct":       round(mape, 4),
            "mae_usd":        round(mae, 4),
            "mean_ci_width":  round(ci_width, 4),
            "inference_sec":  round(elapsed, 3),
            "vram_mb":        round(vram, 1),
        }
        records.append(record)
        log.info("  MAPE=%.2f%%  MAE=$%.2f  CI_width=$%.2f  time=%.2fs  VRAM=%.0fMB",
                 mape, mae, ci_width, elapsed, vram)

        del model
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(records)
    log.info("")
    log.info("Context sweep results:\n%s", results_df.to_string(index=False))

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "exp1_context_sweep.csv"
        results_df.to_csv(out, index=False)
        log.info("Saved → %s", out)

    return results_df


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Chronos-2 NF4 Quantization vs FP32
# ═════════════════════════════════════════════════════════════════════════════

def exp2_nf4_vram_comparison(
    ticker: str = "AAPL",
    horizon: int = 30,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compares Chronos-2 loaded in FP32 (baseline) vs NF4 (bitsandbytes).

    NF4 Quantization — How it works:
      bitsandbytes implements NF4 (NormalFloat4), a 4-bit quantisation scheme
      that assumes weights follow a normal distribution — well-suited to
      transformer weights. BitsAndBytesConfig wraps HuggingFace models:

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # nested 4-bit → extra compression
        )

      At 120M params, Chronos-2 uses:
        FP32:  120M × 4 bytes  ≈ 480 MB VRAM
        NF4 :  120M × 0.5 bytes ≈  60 MB VRAM  (≈8× reduction; 4× from NF4 + 2× double quant)

    Prerequisite:
        uv pip install bitsandbytes accelerate

    NOTE: NF4 requires a CUDA GPU. On CPU-only machines this test is skipped.
    """
    _header("Experiment 2 — Chronos-2 NF4 Quantization vs FP32")

    import torch
    if not torch.cuda.is_available():
        log.warning("No CUDA GPU detected — NF4 quantization requires CUDA. Skipping Exp 2.")
        return pd.DataFrame()

    try:
        import bitsandbytes  # noqa: F401
        from transformers import BitsAndBytesConfig
        log.info("bitsandbytes: available ✓")
    except ImportError:
        log.error(
            "bitsandbytes not installed.\n"
            "Install with: uv pip install --python .venv/bin/python bitsandbytes accelerate\n"
            "Skipping Exp 2."
        )
        return pd.DataFrame()

    from chronos import Chronos2Pipeline

    df     = load_local_data(ticker)
    prices = df["Close"].dropna().values[-512:].astype(np.float64)

    records: list[dict] = []

    # ── Mode A: FP32 (baseline) ───────────────────────────────────────────────
    for mode, quant_cfg in [
        ("fp32",   None),
        ("nf4",    BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )),
    ]:
        log.info("─── Mode: %s ───", mode.upper())
        torch.cuda.reset_peak_memory_stats()
        vram_before = _vram_mb()

        t_load = time.perf_counter()
        if quant_cfg is None:
            pipeline = Chronos2Pipeline.from_pretrained(
                "amazon/chronos-2",
                device_map="cuda",
                torch_dtype=torch.float32,
            )
        else:
            # BitsAndBytesConfig is passed via quantization_config kwarg
            pipeline = Chronos2Pipeline.from_pretrained(
                "amazon/chronos-2",
                device_map="cuda",
                quantization_config=quant_cfg,
            )
        load_time = time.perf_counter() - t_load

        vram_after     = _vram_mb()
        vram_peak_load = torch.cuda.max_memory_allocated() / 1024 ** 2
        vram_model_mb  = vram_after - vram_before

        log.info("  Load time: %.1fs | VRAM model: %.0f MB | VRAM peak: %.0f MB",
                 load_time, vram_model_mb, vram_peak_load)

        # ── Inference ─────────────────────────────────────────────────────────
        torch.cuda.reset_peak_memory_stats()
        t_inf = time.perf_counter()
        results = pipeline.predict(
            inputs=[{"target": prices}],
            prediction_length=horizon,
            limit_prediction_length=False,
        )
        inf_time = time.perf_counter() - t_inf
        vram_peak_inf = torch.cuda.max_memory_allocated() / 1024 ** 2

        tensor   = results[0]  # (1, n_quantiles, horizon)
        median   = tensor[0, 4, :].cpu().numpy()
        q10      = tensor[0, 0, :].cpu().numpy()
        q90      = tensor[0, 8, :].cpu().numpy()
        ci_width = float(np.mean(q90 - q10))

        log.info("  Inference time: %.2fs | VRAM peak (inf): %.0f MB | CI width: $%.2f",
                 inf_time, vram_peak_inf, ci_width)

        record = {
            "mode":             mode,
            "ticker":           ticker,
            "horizon":          horizon,
            "vram_model_mb":    round(vram_model_mb, 1),
            "vram_peak_load_mb":round(vram_peak_load, 1),
            "vram_peak_inf_mb": round(vram_peak_inf, 1),
            "load_time_sec":    round(load_time, 2),
            "inf_time_sec":     round(inf_time, 3),
            "median_day30":     round(float(median[-1]), 4),
            "mean_ci_width":    round(ci_width, 4),
            "gpu_total_mb":     round(_gpu_total_mb(), 0),
        }
        records.append(record)

        del pipeline
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(records)
    log.info("")
    log.info("NF4 comparison:\n%s", results_df.to_string(index=False))

    # ── Compute savings ────────────────────────────────────────────────────────
    if len(results_df) == 2:
        fp32 = results_df[results_df["mode"] == "fp32"].iloc[0]
        nf4  = results_df[results_df["mode"] == "nf4"].iloc[0]
        vram_savings_pct = (1 - nf4["vram_model_mb"] / fp32["vram_model_mb"]) * 100
        speed_delta_pct  = (fp32["inf_time_sec"] - nf4["inf_time_sec"]) / fp32["inf_time_sec"] * 100
        forecast_delta   = abs(nf4["median_day30"] - fp32["median_day30"])
        log.info("NF4 vs FP32: VRAM savings=%.1f%% | Speed change=%.1f%% | Forecast delta=$%.4f",
                 vram_savings_pct, speed_delta_pct, forecast_delta)

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "exp2_nf4_vram.csv"
        results_df.to_csv(out, index=False)
        log.info("Saved → %s", out)

    return results_df


# ═════════════════════════════════════════════════════════════════════════════
# Experiment 3 — Ensemble Weight Sweep
# ═════════════════════════════════════════════════════════════════════════════

def exp3_ensemble_weight_sweep(
    ticker: str = "AAPL",
    horizon: int = 30,
    chronos_weights: list[float] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Sweeps the Chronos-2 blending weight in [0, 0.25, 0.5, 0.75, 1.0].

    Captures how the ensemble median, CI width, and expected return change
    with different model proportions. Helps justify the 50/50 split or
    identify a data-driven optimal blend.
    """
    _header("Experiment 3 — Ensemble Weight Sweep")

    if chronos_weights is None:
        chronos_weights = [0.0, 0.25, 0.5, 0.75, 1.0]

    from src.data import _load_cache
    from src.forecast import generate_ensemble_forecast

    df      = load_local_data(ticker)
    full_df = _load_cache()

    mask = full_df["Ticker"] == ticker
    ticker_df = (
        full_df[mask]
        .copy()
        .set_index("Date")
        .sort_index()
        .drop(columns=["Ticker"], errors="ignore")
        .dropna(subset=["Close"])
    )

    last_close = ticker_df["Close"].iloc[-1]
    records: list[dict] = []

    for cw in chronos_weights:
        tw = 1.0 - cw
        log.info("─── Chronos weight = %.2f | TimesFM weight = %.2f ───", cw, tw)

        t0 = time.perf_counter()
        result = generate_ensemble_forecast(ticker_df, horizon=horizon, chronos_weight=cw)
        elapsed = time.perf_counter() - t0

        ens_final = result["ensemble_median"].iloc[-1]
        q10_final = result["ensemble_q10"].iloc[-1]
        q90_final = result["ensemble_q90"].iloc[-1]
        exp_ret   = (ens_final - last_close) / last_close * 100
        ci_width  = float(result["ensemble_q90"].mean() - result["ensemble_q10"].mean())

        log.info("  E[return]=%+.2f%%  final_price=$%.2f  CI_width=$%.2f  time=%.1fs",
                 exp_ret, ens_final, ci_width, elapsed)

        records.append({
            "ticker":            ticker,
            "horizon":           horizon,
            "chronos_weight":    cw,
            "timesfm_weight":    round(tw, 2),
            "last_close":        round(last_close, 4),
            "ens_median_final":  round(float(ens_final), 4),
            "ens_q10_final":     round(float(q10_final), 4),
            "ens_q90_final":     round(float(q90_final), 4),
            "expected_return_pct": round(exp_ret, 4),
            "mean_ci_width":     round(ci_width, 4),
            "elapsed_sec":       round(elapsed, 2),
        })

        import torch; torch.cuda.empty_cache()

    results_df = pd.DataFrame(records)
    log.info("")
    log.info("Ensemble weight sweep:\n%s",
             results_df[["chronos_weight", "expected_return_pct",
                          "ens_median_final", "mean_ci_width", "elapsed_sec"]]
             .to_string(index=False))

    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / "exp3_weight_sweep.csv"
        results_df.to_csv(out, index=False)
        log.info("Saved → %s", out)

    return results_df


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="'나만의 한 끗' experiment suite for the FastCampus TSFM report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--exp",    type=int, choices=[1, 2, 3],
                   help="Run a specific experiment (1/2/3). Default: all.")
    p.add_argument("--ticker", type=str, default="AAPL",
                   help="Primary ticker for experiments.")
    p.add_argument("--horizon", type=int, default=30,
                   help="Forecast horizon in trading days.")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write CSV result files.")
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    save  = not args.no_save
    run   = lambda n: args.exp is None or args.exp == n

    tickers = get_available_tickers()
    if args.ticker not in tickers:
        log.error("Ticker %s not in master CSV. Available: %s", args.ticker, tickers[:10])
        sys.exit(1)

    if run(1):
        exp1_context_sweep(ticker=args.ticker, horizon=args.horizon, save=save)

    if run(2):
        exp2_nf4_vram_comparison(ticker=args.ticker, horizon=args.horizon, save=save)

    if run(3):
        exp3_ensemble_weight_sweep(ticker=args.ticker, horizon=args.horizon, save=save)


if __name__ == "__main__":
    main()
