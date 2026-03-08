"""
app.py
------
Gradio + Plotly UI for dual-model ensemble stock forecasting.
Chronos-2 (with macro covariates) + TimesFM 1.0.

Run:
    python app.py
"""

import logging

import gradio as gr
import plotly.graph_objects as go
import pandas as pd

from src.data import get_available_tickers, load_local_data
from src.forecast import generate_ensemble_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Ticker list (populated at startup from local CSV) ─────────────────────────

def _get_tickers_safe() -> list[str]:
    """Return available tickers; fall back to a hardcoded list if CSV not built yet."""
    try:
        return get_available_tickers()
    except FileNotFoundError:
        return [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
            "META", "TSLA", "BRK-B", "JPM", "V",
            "UNH", "XOM", "LLY", "JNJ", "MA",
            "AVGO", "PG", "HD", "MRK", "COST",
        ]

TICKERS = _get_tickers_safe()


# ── Core handler ──────────────────────────────────────────────────────────────

def run_forecast(
    ticker: str, horizon: int, chronos_weight: float
) -> tuple[go.Figure, str]:
    """Gradio click handler. Loads local data → ensemble forecast → chart."""
    if not ticker:
        return None, "⚠️ Please select a ticker."

    try:
        df = load_local_data(ticker)
    except (FileNotFoundError, ValueError) as e:
        return None, f"❌ Data error: {e}"

    try:
        result = generate_ensemble_forecast(df, int(horizon), float(chronos_weight))
    except Exception as e:
        return None, f"❌ Forecast error: {e}"

    timesfm_weight = 1.0 - float(chronos_weight)

    # ── Build Plotly figure ──────────────────────────────────────────────
    fig = go.Figure()

    # 1. Historical Close — last 6 months, solid black
    hist = df["Close"].last("6ME")
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        mode="lines", name="Historical Close",
        line=dict(color="black", width=1.5),
    ))

    # 2. 80% Prediction Interval — red shaded fill
    dates_fwd = result["dates"]
    q10 = result["ensemble_q10"]
    q90 = result["ensemble_q90"]
    fig.add_trace(go.Scatter(
        x=pd.concat([dates_fwd.to_series(), dates_fwd.to_series()[::-1]]),
        y=list(q90.values) + list(q10.values[::-1]),
        fill="toself",
        fillcolor="rgba(220, 50, 50, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="80% Prediction Interval",
    ))

    # 3. Ensemble Median — red thick dashed
    fig.add_trace(go.Scatter(
        x=dates_fwd, y=result["ensemble_median"].values,
        mode="lines", name="Ensemble Median",
        line=dict(color="crimson", width=2.5, dash="dash"),
    ))

    # 4. Chronos-2 Median — blue dotted
    fig.add_trace(go.Scatter(
        x=dates_fwd, y=result["chronos_median"].values,
        mode="lines", name="Chronos-2 Median",
        line=dict(color="royalblue", width=1.5, dash="dot"),
    ))

    # 5. TimesFM Median — green dotted
    fig.add_trace(go.Scatter(
        x=dates_fwd, y=result["timesfm_median"].values,
        mode="lines", name="TimesFM Median",
        line=dict(color="seagreen", width=1.5, dash="dot"),
    ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Dual Model Ensemble Forecast", font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
        height=540,
    )

    # ── Status string ────────────────────────────────────────────────────
    last_close = df["Close"].iloc[-1]
    q10_last = q10.iloc[-1]
    q90_last = q90.iloc[-1]
    status = (
        f"✅ {ticker} | Last close: ${last_close:.2f} | Horizon: {int(horizon)}d | "
        f"Weights: Chronos={chronos_weight:.0%} / TimesFM={timesfm_weight:.0%} | "
        f"Ensemble range: ${q10_last:.2f} – ${q90_last:.2f}"
    )

    return fig, status


# ── Gradio Blocks UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="AI Stock Forecaster — Dual Model Ensemble", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📈 AI Stock Price Forecaster — Dual Model Ensemble
        **Chronos-2** (Amazon · macro covariates) + **TimesFM 1.0** (Google DeepMind · 200M params)
        Zero-shot forecasting — no fine-tuning required.

        > Data sourced from a local static dataset (S&P 500 top 20 + FRED macro).
        > Run `python scripts/build_dataset.py` to refresh.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            ticker_dropdown = gr.Dropdown(
                label="Select Ticker",
                choices=TICKERS,
                value=TICKERS[0] if TICKERS else "AAPL",
            )
            horizon_slider = gr.Slider(
                label="Forecast Horizon (days)",
                minimum=10, maximum=90, step=5, value=30,
            )
            chronos_weight = gr.Slider(
                label="Chronos-2 Weight (TimesFM = 1 − this)",
                minimum=0.0, maximum=1.0, step=0.05, value=0.5,
            )
            run_btn = gr.Button("🔮 Generate Ensemble Forecast", variant="primary")

        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Ensemble Forecast Chart")
            status_output = gr.Textbox(label="Status", lines=2, interactive=False)

    run_btn.click(
        fn=run_forecast,
        inputs=[ticker_dropdown, horizon_slider, chronos_weight],
        outputs=[plot_output, status_output],
    )

    gr.Examples(
        examples=[["AAPL", 30, 0.5], ["MSFT", 60, 0.3], ["NVDA", 14, 0.7]],
        inputs=[ticker_dropdown, horizon_slider, chronos_weight],
        label="Quick Examples",
    )

if __name__ == "__main__":
    demo.launch(share=False)
