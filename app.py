"""Gradio + Plotly UI for AI-powered stock price forecasting.

Run:
    python app.py
"""
import gradio as gr
import plotly.graph_objects as go
import pandas as pd

from src.data import fetch_stock_data
from src.forecast import generate_forecast


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

    # 80% CI shaded band (drawn before median so median overlays it)
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
            text=f"{ticker} — Zero-Shot Forecast (TimesFM 2.5)",
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
    """Gradio click handler. Returns (plotly figure, status string)."""
    ticker = ticker.strip().upper()
    if not ticker:
        return None, "⚠️ Please enter a ticker symbol."

    try:
        historical = fetch_stock_data(ticker, period="2y")
    except ValueError as e:
        return None, f"❌ Data error: {e}"

    try:
        median, lower, upper = generate_forecast(historical, horizon=int(horizon))
    except Exception as e:
        return None, f"❌ Forecast error: {e}"

    # Show last 6 months of history for chart readability
    display_history = historical.last("6ME")
    fig = build_figure(ticker, display_history, median, lower, upper)

    status = (
        f"✅ {ticker} | Last close: ${historical.iloc[-1]:.2f} | "
        f"Horizon: {horizon} days | "
        f"Forecast range: ${lower.iloc[-1]:.2f} – ${upper.iloc[-1]:.2f}"
    )
    return fig, status


# ── Gradio Blocks UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="AI Stock Forecaster", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📈 AI Stock Price Forecaster
        **Zero-shot forecasting powered by [TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch)**
        (Google DeepMind · 200M params · No fine-tuning required)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            ticker_input = gr.Textbox(
                label="Ticker Symbol",
                value="AAPL",
                placeholder="e.g. AAPL, MSFT, NVDA, TSLA",
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
        inputs=[ticker_input, horizon_slider],
        outputs=[plot_output, status_output],
    )

    gr.Examples(
        examples=[["AAPL", 30], ["MSFT", 60], ["NVDA", 14], ["TSLA", 45]],
        inputs=[ticker_input, horizon_slider],
        label="Quick Examples",
    )

if __name__ == "__main__":
    demo.launch(share=False)
