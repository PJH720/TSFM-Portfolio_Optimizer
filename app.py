"""
app.py
------
Gradio UI for AI-Powered Portfolio System.
Tab 1: Single-asset dual-model ensemble forecast (Chronos-2 + TimesFM).
Tab 2: Multi-asset Markowitz portfolio optimization (with sector constraints).

Ticker lists are loaded dynamically from data/sp500_macro_master.csv at startup.
No hardcoded ticker lists.

Run:
    python app.py
"""

import logging

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data import (
    _load_cache,
    get_available_tickers,
    get_sector_map,
    load_local_data,
)
from src.forecast import generate_ensemble_forecast, get_portfolio_forecasts_and_cov
from src.optimize import optimize_portfolio, portfolio_diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Startup — dynamic ticker & sector loading from master CSV
# ─────────────────────────────────────────────────────────────────────────────

def _load_tickers_and_sectors() -> tuple[list[str], dict[str, str]]:
    """Load tickers and sector map from master CSV.

    Returns (sorted_tickers, sector_map).
    Falls back gracefully if CSV not yet built, with an informative message.
    """
    try:
        tickers    = get_available_tickers()   # reads sp500_macro_master.csv
        sector_map = get_sector_map()
        log.info("Loaded %d tickers from master CSV.", len(tickers))
        return tickers, sector_map
    except FileNotFoundError:
        log.warning(
            "Master CSV not found — run: python scripts/build_dataset.py\n"
            "UI will show an empty ticker list until the dataset is built."
        )
        return [], {}


TICKERS, SECTOR_MAP = _load_tickers_and_sectors()


def _default_portfolio(n: int = 5) -> list[str]:
    """Pick n tickers that span as many sectors as possible for a diverse default."""
    if not TICKERS:
        return []
    by_sector: dict[str, str] = {}
    for t in TICKERS:
        sec = SECTOR_MAP.get(t, "Unknown")
        if sec not in by_sector:
            by_sector[sec] = t           # first ticker per sector
    picks = list(by_sector.values())[:n]
    # pad with weight-ordered tickers if we got fewer than n sectors
    for t in TICKERS:
        if len(picks) >= n:
            break
        if t not in picks:
            picks.append(t)
    return picks[:n]


DEFAULT_PORTFOLIO = _default_portfolio(5)
log.info("Default portfolio: %s", DEFAULT_PORTFOLIO)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Single-Asset Forecast
# ─────────────────────────────────────────────────────────────────────────────

def run_single_forecast(
    ticker: str,
    horizon: int,
    chronos_weight: float,
) -> tuple[go.Figure, str]:
    if not ticker:
        return go.Figure(), "⚠️ Please select a ticker."

    try:
        df = load_local_data(ticker)
    except (FileNotFoundError, ValueError) as e:
        return go.Figure(), f"❌ Data error: {e}"

    try:
        result = generate_ensemble_forecast(df, int(horizon), float(chronos_weight))
    except Exception as e:
        log.exception("Forecast error for %s", ticker)
        return go.Figure(), f"❌ Forecast error: {e}"

    timesfm_weight = 1.0 - float(chronos_weight)
    dates_fwd      = result["dates"]
    q10            = result["ensemble_q10"]
    q90            = result["ensemble_q90"]

    sector = SECTOR_MAP.get(ticker, "")
    title  = f"{ticker} — Dual-Model Ensemble Forecast"
    if sector:
        title += f"  [{sector}]"

    fig = go.Figure()

    # 1 · Historical Close — last 6 months, solid black
    hist = df["Close"].last("6ME")
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name="Historical Close",
        line=dict(color="black", width=2),
    ))

    # 2 · 80% CI band — red shaded fill
    fig.add_trace(go.Scatter(
        x=pd.concat([dates_fwd.to_series(), dates_fwd.to_series()[::-1]]),
        y=list(q90.values) + list(q10.values[::-1]),
        fill="toself",
        fillcolor="rgba(220,50,50,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="80% Prediction Interval",
        showlegend=True,
    ))

    # 3 · Ensemble Median — crimson dashed
    fig.add_trace(go.Scatter(
        x=dates_fwd, y=result["ensemble_median"].values,
        mode="lines", name=f"Ensemble (C={float(chronos_weight):.0%})",
        line=dict(color="crimson", width=2.5, dash="dash"),
    ))

    # 4 · Chronos-2 Median — blue dotted
    fig.add_trace(go.Scatter(
        x=dates_fwd, y=result["chronos_median"].values,
        mode="lines", name="Chronos-2",
        line=dict(color="royalblue", width=1.5, dash="dot"),
    ))

    # 5 · TimesFM Median — green dotted
    fig.add_trace(go.Scatter(
        x=dates_fwd, y=result["timesfm_median"].values,
        mode="lines", name="TimesFM 1.0",
        line=dict(color="seagreen", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=17)),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        height=540,
    )

    last_close  = df["Close"].iloc[-1]
    q10_final   = q10.iloc[-1]
    q90_final   = q90.iloc[-1]
    ens_final   = result["ensemble_median"].iloc[-1]
    exp_ret_pct = (ens_final - last_close) / last_close * 100
    status = (
        f"✅  {ticker}  |  Last close: ${last_close:.2f}  |  "
        f"Expected ({horizon}d): ${ens_final:.2f}  ({exp_ret_pct:+.1f}%)  |  "
        f"80% CI: [${q10_final:.2f}, ${q90_final:.2f}]  |  "
        f"Chronos={float(chronos_weight):.0%} / TimesFM={timesfm_weight:.0%}"
    )
    return fig, status


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Portfolio Optimization
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio_opt(
    tickers,
    horizon,
    chronos_weight,
    risk_aversion,
    enable_sector_constraints,
    progress=gr.Progress(),
) -> tuple[go.Figure, str]:
    if not tickers or len(tickers) < 2:
        return go.Figure(), "⚠️ Select at least 2 tickers."

    try:
        full_df = _load_cache()
    except FileNotFoundError as e:
        return go.Figure(), f"❌ Dataset missing — run: python scripts/build_dataset.py\n{e}"

    # ── TSFM batch forecasting ─────────────────────────────────────────────
    progress(0.0, desc="Starting batch TSFM forecasting…")

    def _prog(frac: float, desc: str) -> None:
        progress(frac * 0.82, desc=desc)

    try:
        mu_vector, cov_matrix = get_portfolio_forecasts_and_cov(
            full_df, list(tickers), int(horizon), float(chronos_weight), _prog
        )
    except Exception as e:
        log.exception("Batch forecast failed")
        return go.Figure(), f"❌ Forecast error: {e}"

    # ── Markowitz optimization (with optional sector constraints) ──────────
    progress(0.85, desc="Running Markowitz optimizer…")

    smap = SECTOR_MAP if enable_sector_constraints else None
    try:
        weights = optimize_portfolio(
            mu_vector,
            cov_matrix,
            risk_aversion=float(risk_aversion),
            sector_map=smap,
            max_sector_weight=0.30,
            max_single_weight=0.25,
        )
    except Exception as e:
        log.exception("Optimisation failed")
        return go.Figure(), f"❌ Optimisation error: {e}"

    diag = portfolio_diagnostics(weights, mu_vector, cov_matrix, smap)

    progress(0.95, desc="Building chart…")

    # ── Pie chart ──────────────────────────────────────────────────────────
    labels   = list(weights.keys())
    values   = [weights[t] for t in labels]
    sectors  = [SECTOR_MAP.get(t, "Unknown") for t in labels]
    mu_pct   = [f"{mu_vector.get(t, 0)*100:.2f}%" for t in labels]
    palette  = px.colors.qualitative.Set3

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        customdata=list(zip(mu_pct, sectors)),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Weight: %{percent}<br>"
            "Expected Return: %{customdata[0]}<br>"
            "Sector: %{customdata[1]}"
            "<extra></extra>"
        ),
        textinfo="label+percent",
        hole=0.38,
        marker=dict(
            colors=palette[: len(labels)],
            line=dict(color="white", width=2),
        ),
    ))

    constraint_tag = "  [Sector ≤30%]" if enable_sector_constraints else ""
    fig.update_layout(
        title=dict(
            text=(
                f"Optimal Portfolio — γ={float(risk_aversion):.1f} | "
                f"Horizon={horizon}d{constraint_tag}"
            ),
            font=dict(size=17),
        ),
        legend=dict(orientation="v", yanchor="middle", y=0.5),
        paper_bgcolor="white",
        height=500,
        annotations=[dict(
            text=f"E[r]={diag['expected_return']:.1f}%",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False,
        )],
    )

    # ── Status string ──────────────────────────────────────────────────────
    top3 = sorted(weights.items(), key=lambda x: -x[1])[:3]
    top3_str = "  |  ".join(f"{t}: {w*100:.1f}%" for t, w in top3)

    sector_lines = ""
    if enable_sector_constraints and diag["sector_weights"]:
        sector_lines = "\nSector allocations: " + "  |  ".join(
            f"{s}: {v*100:.0f}%" for s, v in
            sorted(diag["sector_weights"].items(), key=lambda x: -x[1])
        )

    status = (
        f"✅  Portfolio optimised  |  {len(weights)}/{len(tickers)} assets allocated\n"
        f"E[r]: {diag['expected_return']:.2f}%  |  "
        f"σ: {diag['portfolio_std']:.2f}%  |  "
        f"Sharpe: {diag['sharpe_ratio']:.3f}  |  γ={float(risk_aversion):.1f}\n"
        f"Top holdings: {top3_str}"
        f"{sector_lines}"
    )

    progress(1.0, desc="Done.")
    return fig, status


# ─────────────────────────────────────────────────────────────────────────────
# Gradio Blocks UI
# ─────────────────────────────────────────────────────────────────────────────

_NO_DATA_MSG = (
    "⚠️  No dataset found.  "
    "Run `python scripts/build_dataset.py` to build `data/sp500_macro_master.csv`."
    if not TICKERS else ""
)

with gr.Blocks(title="AI Portfolio System — TSFM Edition") as demo:

    gr.Markdown("""
# 📈 AI-Powered Portfolio System  ·  TSFM Edition
**Dual-Model Ensemble (Chronos-2 + TimesFM 1.0)  ·  Markowitz Mean-Variance  ·  Sector Constraints**

> Build dataset first: `python scripts/build_dataset.py --top-n 100 --period 5y`
""")

    if _NO_DATA_MSG:
        gr.Markdown(f"### {_NO_DATA_MSG}")

    with gr.Tabs():

        # ── Tab 1: Single Asset Forecast ─────────────────────────────────────
        with gr.Tab("🔍 Single Asset Forecast"):
            with gr.Row():
                with gr.Column(scale=1):
                    t1_ticker = gr.Dropdown(
                        label=f"Ticker  ({len(TICKERS)} available)",
                        choices=TICKERS,
                        value=TICKERS[0] if TICKERS else None,
                        filterable=True,
                    )
                    t1_horizon = gr.Slider(
                        label="Forecast Horizon (trading days)",
                        minimum=5, maximum=90, step=5, value=30,
                    )
                    t1_cw = gr.Slider(
                        label="Chronos-2 Weight  (TimesFM = 1 − this)",
                        minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                    )
                    t1_btn = gr.Button("🔮 Generate Forecast", variant="primary")

                with gr.Column(scale=3):
                    t1_plot   = gr.Plot(label="Forecast Chart")
                    t1_status = gr.Textbox(label="Status", lines=2, interactive=False)

            t1_btn.click(
                fn=run_single_forecast,
                inputs=[t1_ticker, t1_horizon, t1_cw],
                outputs=[t1_plot, t1_status],
            )

            gr.Examples(
                examples=[
                    ["AAPL",  30, 0.5],
                    ["MSFT",  60, 0.3],
                    ["NVDA",  14, 0.7],
                    ["JPM",   30, 0.5],
                    ["XOM",   45, 0.4],
                ],
                inputs=[t1_ticker, t1_horizon, t1_cw],
                label="Quick Examples",
            )

        # ── Tab 2: Portfolio Optimization ─────────────────────────────────────
        with gr.Tab("📊 Portfolio Optimization"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2_tickers = gr.Dropdown(
                        label=f"Tickers  (min 2, from {len(TICKERS)} available)",
                        choices=TICKERS,
                        value=DEFAULT_PORTFOLIO,
                        multiselect=True,
                        filterable=True,
                    )
                    t2_horizon = gr.Slider(
                        label="Forecast Horizon (trading days)",
                        minimum=5, maximum=90, step=5, value=30,
                    )
                    t2_cw = gr.Slider(
                        label="Chronos-2 Weight",
                        minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                    )
                    t2_ra = gr.Slider(
                        label="Risk Aversion γ  (higher = more conservative)",
                        minimum=0.1, maximum=10.0, step=0.1, value=1.0,
                    )
                    t2_sector = gr.Checkbox(
                        label="Enable Sector Constraints  (each GICS sector ≤ 30%)",
                        value=True,
                    )
                    t2_btn = gr.Button("⚙️ Optimise Portfolio", variant="primary")

                with gr.Column(scale=3):
                    t2_plot   = gr.Plot(label="Optimal Portfolio Weights")
                    t2_status = gr.Textbox(label="Status", lines=5, interactive=False)

            t2_btn.click(
                fn=run_portfolio_opt,
                inputs=[t2_tickers, t2_horizon, t2_cw, t2_ra, t2_sector],
                outputs=[t2_plot, t2_status],
            )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
