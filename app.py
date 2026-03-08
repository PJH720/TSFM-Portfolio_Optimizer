"""
app.py
------
Gradio + Plotly UI for AI-powered stock price forecasting.
Reads from a local static dataset — no real-time API calls.

Run:
    python app.py
"""

import logging

import gradio as gr
import plotly.graph_objects as go
import pandas as pd

from src.data import get_available_tickers, get_close_series
from src.forecast import generate_forecast

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


# ── Visualization ─────────────────────────────────────────────────────────────

def build_figure(
    ticker: str,
    historical: pd.Series,
    median: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> go.Figure:
    """Compose Plotly figure: historical line + forecast median + 80% CI band."""
    fig = go.Figure()

    # Historical prices — solid black line
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        mode="lines",
        name="Historical Close",
        line=dict(color="black", width=1.5),
    ))

    # 80% CI shaded band (add before median so median renders on top)
    fig.add_trace(go.Scatter(
        x=pd.concat([upper.index.to_series(), lower.index.to_series()[::-1]]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself",
        fillcolor="rgba(220, 50, 50, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="80% Prediction Interval",
    ))

    # Forecast median — dashed red line
    fig.add_trace(go.Scatter(
        x=median.index,
        y=median.values,
        mode="lines",
        name="Forecast (median)",
        line=dict(color="crimson", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Zero-Shot Forecast (TimesFM 1.0)",
            font=dict(size=18),
        ),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#eee"),
        yaxis=dict(showgrid=True, gridcolor="#eee"),
        height=520,
    )
    return fig


# ── Core handler ──────────────────────────────────────────────────────────────

def run_forecast(ticker: str, horizon: int) -> tuple[go.Figure, str]:
    """Gradio click handler. Loads local data → forecast → chart."""
    if not ticker:
        return None, "⚠️ Please select a ticker."

    try:
        close = get_close_series(ticker)
    except (FileNotFoundError, ValueError) as e:
        return None, f"❌ Data error: {e}"

    try:
        median, lower, upper = generate_forecast(close, horizon=int(horizon))
    except Exception as e:
        return None, f"❌ Forecast error: {e}"

    # Show last 6 months of history for readability
    display_history = close.last("6ME")
    fig = build_figure(ticker, display_history, median, lower, upper)

    status = (
        f"✅ {ticker} | "
        f"Last close: ${close.iloc[-1]:.2f} ({close.index[-1].date()}) | "
        f"Horizon: {horizon} days | "
        f"Forecast range: ${lower.iloc[-1]:.2f} – ${upper.iloc[-1]:.2f}"
    )
    return fig, status


# ── Gradio Blocks UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="AI Stock Forecaster", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📈 AI Stock Price Forecaster
        **Zero-shot forecasting powered by [TimesFM 1.0](https://huggingface.co/google/timesfm-1.0-200m-pytorch)**
        (Google DeepMind · 200M parameters · No fine-tuning required)

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
                label="Forecast Horizon (trading days)",
                minimum=10,
                maximum=90,
                step=5,
                value=30,
            )
            run_btn = gr.Button("🔮 Generate Forecast", variant="primary")

        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Forecast Chart")
            status_output = gr.Textbox(
                label="Status", interactive=False, lines=2
            )

    run_btn.click(
        fn=run_forecast,
        inputs=[ticker_dropdown, horizon_slider],
        outputs=[plot_output, status_output],
    )

    gr.Examples(
        examples=[["AAPL", 30], ["MSFT", 60], ["NVDA", 14], ["TSLA", 45]],
        inputs=[ticker_dropdown, horizon_slider],
        label="Quick Examples",
    )

if __name__ == "__main__":
    demo.launch(share=False)
