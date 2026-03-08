"""
app.py
------
Gradio UI for AI-Powered Portfolio System.
Tab 1: Single-asset dual-model ensemble forecast (Chronos-2 + TimesFM).
Tab 2: Multi-asset portfolio optimization (Markowitz Mean-Variance).

Run:
    python app.py
"""

import logging

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.data import get_available_tickers, get_close_series, load_local_data, _load_cache
from src.forecast import generate_ensemble_forecast, get_portfolio_forecasts_and_cov
from src.optimize import optimize_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Ticker list (populated at startup from local CSV) ─────────────────────────

def _get_tickers_safe() -> list[str]:
    try:
        return get_available_tickers()
    except FileNotFoundError:
        return ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","JPM","V",
                "UNH","XOM","LLY","JNJ","MA","AVGO","PG","HD","MRK","COST"]

TICKERS = _get_tickers_safe()
DEFAULT_PORTFOLIO = [t for t in ["AAPL","MSFT","NVDA","GOOGL","AMZN"] if t in TICKERS]


# ── Tab 1: Single Asset Forecast ──────────────────────────────────────────────

def run_single_forecast(
    ticker: str, horizon: int, chronos_weight: float
) -> tuple[go.Figure, str]:
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

    # 3. Ensemble Median — crimson dashed
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

    last_close = df["Close"].iloc[-1]
    q10_last = q10.iloc[-1]
    q90_last = q90.iloc[-1]
    cw = float(chronos_weight)
    tw = timesfm_weight
    status = (
        f"✅ {ticker} | Last: ${last_close:.2f} | Horizon: {int(horizon)}d | "
        f"Weights: Chronos={cw:.0%}/TimesFM={tw:.0%} | "
        f"Range: ${q10_last:.2f}–${q90_last:.2f}"
    )

    return fig, status


# ── Tab 2: Portfolio Optimization ─────────────────────────────────────────────

def run_portfolio_opt(tickers, horizon, chronos_weight, risk_aversion, progress=gr.Progress()):
    if not tickers or len(tickers) < 2:
        return None, "⚠️ Select at least 2 tickers to build a portfolio."

    try:
        full_df = _load_cache()
    except FileNotFoundError as e:
        return None, f"❌ Dataset not found. Run: python scripts/build_dataset.py\n{e}"

    progress(0, desc="Starting batch forecasting…")

    def prog(frac, desc):
        progress(frac * 0.85, desc=desc)

    try:
        mu_vector, cov_matrix = get_portfolio_forecasts_and_cov(
            full_df, list(tickers), int(horizon), float(chronos_weight), prog
        )
    except Exception as e:
        return None, f"❌ Forecast error: {e}"

    progress(0.87, desc="Running Markowitz optimizer…")

    try:
        weights = optimize_portfolio(mu_vector, cov_matrix, float(risk_aversion))
    except Exception as e:
        return None, f"❌ Optimization error: {e}"

    progress(0.95, desc="Building chart…")

    # ── Plotly Pie Chart ──────────────────────────────────────────────────────
    labels  = list(weights.keys())
    values  = [weights[t] for t in labels]
    mu_pct  = [f"{mu_vector.get(t, 0)*100:.2f}%" for t in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        customdata=mu_pct,
        hovertemplate="<b>%{label}</b><br>Weight: %{percent}<br>Expected Return: %{customdata}<extra></extra>",
        textinfo="label+percent",
        hole=0.35,
        marker=dict(
            colors=px.colors.qualitative.Set3[:len(labels)],
            line=dict(color="white", width=2),
        ),
    ))

    fig.update_layout(
        title=dict(
            text=f"Optimal Portfolio — γ={float(risk_aversion):.1f} | Horizon={horizon}d",
            font=dict(size=18),
        ),
        legend=dict(orientation="v", yanchor="middle", y=0.5),
        paper_bgcolor="white",
        height=500,
        annotations=[dict(
            text=f"γ={float(risk_aversion):.1f}",
            x=0.5, y=0.5,
            font_size=18,
            showarrow=False,
        )],
    )

    # ── Status string ─────────────────────────────────────────────────────────
    top = sorted(weights.items(), key=lambda x: -x[1])[:3]
    top_str = "  |  ".join(f"{t}: {w*100:.1f}%" for t, w in top)
    cov_str = (
        f"Weights: Chronos={float(chronos_weight):.0%} / TimesFM={1-float(chronos_weight):.0%}\n"
        f"Top holdings: {top_str}\n"
        f"μ range: {mu_vector.min()*100:.2f}% – {mu_vector.max()*100:.2f}%  |  "
        f"{len(weights)}/{len(tickers)} assets allocated"
    )
    progress(1.0, desc="Done.")
    return fig, f"✅ Portfolio optimized\n{cov_str}"


# ── Gradio Blocks UI ──────────────────────────────────────────────────────────

with gr.Blocks(title="AI Portfolio System") as demo:
    gr.Markdown("""
    # 📈 AI-Powered Portfolio System
    **Dual-Model Ensemble Forecasting (Chronos-2 + TimesFM) · Markowitz Mean-Variance Optimization**
    > Run `python scripts/build_dataset.py` first to build the local dataset.
    """)

    with gr.Tabs():

        # ── Tab 1: Single Asset Forecast ───────────────────────────────────────
        with gr.Tab("🔍 Single Asset Forecast"):
            with gr.Row():
                with gr.Column(scale=1):
                    t1_ticker   = gr.Dropdown(label="Ticker", choices=TICKERS, value=TICKERS[0] if TICKERS else "AAPL")
                    t1_horizon  = gr.Slider(label="Forecast Horizon (days)", minimum=10, maximum=90, step=5, value=30)
                    t1_cw       = gr.Slider(label="Chronos-2 Weight  (TimesFM = 1 - this)", minimum=0.0, maximum=1.0, step=0.05, value=0.5)
                    t1_btn      = gr.Button("🔮 Generate Forecast", variant="primary")
                with gr.Column(scale=3):
                    t1_plot   = gr.Plot(label="Forecast Chart")
                    t1_status = gr.Textbox(label="Status", lines=2, interactive=False)

            t1_btn.click(fn=run_single_forecast, inputs=[t1_ticker, t1_horizon, t1_cw], outputs=[t1_plot, t1_status])

            gr.Examples(
                examples=[["AAPL", 30, 0.5], ["MSFT", 60, 0.3], ["NVDA", 14, 0.7], ["TSLA", 45, 0.5]],
                inputs=[t1_ticker, t1_horizon, t1_cw],
                label="Quick Examples",
            )

        # ── Tab 2: Portfolio Optimization ──────────────────────────────────────
        with gr.Tab("📊 Portfolio Optimization"):
            with gr.Row():
                with gr.Column(scale=1):
                    t2_tickers = gr.Dropdown(
                        label="Select Tickers (min 2)",
                        choices=TICKERS,
                        value=DEFAULT_PORTFOLIO,
                        multiselect=True,
                    )
                    t2_horizon = gr.Slider(label="Forecast Horizon (days)", minimum=10, maximum=90, step=5, value=30)
                    t2_cw      = gr.Slider(label="Chronos-2 Weight", minimum=0.0, maximum=1.0, step=0.05, value=0.5)
                    t2_ra      = gr.Slider(label="Risk Aversion (γ)", minimum=0.1, maximum=10.0, step=0.1, value=1.0)
                    t2_btn     = gr.Button("⚙️ Optimize Portfolio", variant="primary")
                with gr.Column(scale=3):
                    t2_plot   = gr.Plot(label="Optimal Portfolio Weights")
                    t2_status = gr.Textbox(label="Status", lines=4, interactive=False)

            t2_btn.click(
                fn=run_portfolio_opt,
                inputs=[t2_tickers, t2_horizon, t2_cw, t2_ra],
                outputs=[t2_plot, t2_status],
            )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
