# HFM_Implementation — "Fast & Clean" MVP Build Plan

**Strategy:** Ship fast, meet all 4 criteria, treat portfolio optimizer as bonus.
**Hard deadline:** March 10 (Tue) 21:00 KST — report + demo uploaded
**Presenter selection:** March 10 (Tue) 17:00 KST

---

## Evaluation Criteria Checklist

| # | Criterion | Coverage |
|---|-----------|----------|
| 1 | HF model running (basic inference) | Day 1 MVP |
| 2 | Param tweaks → observable changes | Day 2 |
| 3 | Gradio demo page | Day 1 MVP |
| 4 | Report (Notion template) | Day 3 morning |

---

## Day 1 (Mar 8, Today) — MVP: Criteria 1 + 3

**Goal:** `yfinance → TimesFM/Chronos zero-shot → Gradio + Plotly`

### Task 1.1 — Data Loader
- `yfinance.download()` for S&P 500 tickers (AAPL, MSFT, GOOGL, NVDA, TSLA)
- Return: daily closing prices, last 2–3 years
- Output: `pd.DataFrame` with DatetimeIndex

### Task 1.2 — HF Model Inference (Zero-shot, no fine-tuning)
- Primary: `google/timesfm-2.5-200m-pytorch`
- Fallback: `amazon/chronos-t5-small` (lighter, faster on CPU)
- Input: historical close prices as context window
- Output: point forecast + optional quantile intervals for forecast horizon

### Task 1.3 — Gradio Single-Screen App
- Input: ticker text box + submit button
- Output: Plotly figure
  - Historical prices (line)
  - Forecasted prices (dashed line)
  - Confidence interval band (if model supports it)
- Single `gr.Interface` or minimal `gr.Blocks` layout

**✅ Done = minimum viable submission. Everything below is upside.**

---

## Day 2 (Mar 9, Mon) — Param Tuning: Criterion 2

**Goal:** "Tweak params, observe changes" — visible inside the Gradio UI

### Task 2.1 — Basic Tuning (Low effort, high value for report)
Add Gradio sliders/inputs for:
- `context_length`: e.g., 32 / 64 / 128 / 256 bars
- `horizon`: e.g., 7 / 14 / 30 / 60 days
- Show side-by-side or toggle comparison on the same Plotly chart

### Task 2.2 — Advanced Tuning (Optional, medium difficulty)
- Load model in 4-bit (NF4) via `bitsandbytes`:
  ```python
  from transformers import BitsAndBytesConfig
  bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
  ```
- Record: VRAM before vs after, inference latency delta
- Screenshot both — good content for the report's "나만의 한 끗" section

---

## Day 3 (Mar 10, Tue) — Report + Bonus: Criterion 4

**Goal:** Report submitted well before 17:00, bonus if bandwidth allows

### Task 3.1 — Report (target: done by 13:00)
Fill `Summit - Notion_ReportFormat.md` template:

| Section | Content |
|---------|---------|
| 1. 모델 선정 이유 | HF Hub search → TimesFM/Chronos, zero-shot advantage over ARIMA |
| 2. 모델 실행 결과 | AAPL forecast screenshot, accuracy observation |
| 3. 나만의 한 끗 | context_length sweep results + (if done) NF4 quantization VRAM comparison |
| 4. 소감 | Brief |
| 부록 | Gradio screen capture or URL |

### Task 3.2 — Final Upload (by 17:00 for presenter slot, hard by 21:00)
- [ ] Report finalized in Notion
- [ ] Gradio demo: screen capture OR public URL (use `gr.Interface(share=True)`)
- [ ] Submit presenter application by 17:00

### Task 3.3 — Bonus Track (only if Tasks 3.1+3.2 are done early)
Connect TimesFM μ estimates → cvxpy Markowitz optimizer:

```
μ  = TimesFM predicted returns (annualized)
Σ  = historical sample covariance (pandas .cov())
w* = cvxpy QP: maximize Sharpe, subject to Σw = 1, w ≥ 0

Gradio layout:
  Left panel  → forecast chart (existing)
  Right panel → pie chart of optimal weights w*
```

Files to create if pursuing bonus:
- `src/optimizer/markowitz.py` — cvxpy solver
- Add right panel to Gradio Blocks layout

---

## File Structure (MVP target)

```
HFM_Implementation/
├── app.py                  # Gradio entry point — run this
├── src/
│   ├── data.py             # yfinance loader
│   ├── forecast.py         # TimesFM / Chronos wrapper
│   └── optimizer.py        # (bonus) cvxpy Markowitz
├── requirements.txt        # pinned deps
├── .env                    # FRED_API_KEY (if used)
├── PLAN.md                 # this file
├── CLAUDE.md               # claude code context
└── Summit - Notion_ReportFormat.md
```

---

## Model Priority Decision

| Condition | Use |
|-----------|-----|
| CUDA available | `google/timesfm-2.5-200m-pytorch` |
| CPU only / slow machine | `amazon/chronos-t5-small` |
| Both fail | `amazon/chronos-t5-tiny` (smallest) |

Device detection pattern (already in CLAUDE.md):
```python
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
```

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| `timesfm` install fails (Python version conflict) | Use `uv venv --python 3.10` or fall back to Chronos |
| Model too slow on CPU | Use `chronos-t5-tiny` + shorter context |
| Gradio `share=True` URL unavailable | Screen capture is explicitly listed as acceptable |
| Report takes longer than expected | Use template literally — it's short by design |

---

_Created: 2026-03-08 | Strategy: Fast & Clean MVP First_
_Bonus track (cvxpy optimizer) is time-permitting only — do NOT let it block the deadline._
