# /project:init — HFM_Implementation Environment Bootstrap

Run this at the start of every session to ensure the environment is ready.

## Steps

### 1. Check project structure

Verify the expected source directories exist. If not, create them:

```
src/
  data/          # data ingestion: yfinance, fredapi, Kaggle CSV loader
  forecasting/   # TimesFM / Chronos inference wrappers
  risk/          # covariance engine (sample + GARCH)
  optimizer/     # cvxpy QP solver, constraints
  ui/            # Gradio app entry point
data/
  raw/           # place Kaggle S&P 500 CSV here
  processed/     # cleaned, aligned DataFrames (Parquet)
notebooks/       # scratch / EDA
```

Create any missing directories. Do NOT create placeholder files — empty dirs only.

### 2. Check for virtual environment

Look for `.venv/` in the project root.

- If missing: run `uv venv .venv --python 3.10` (Python 3.10 for timesfm compatibility)
- If present: confirm it's activatable

### 3. Install / sync dependencies

Check if `pyproject.toml` or `requirements.txt` exists.

- If `pyproject.toml` exists: run `uv sync`
- If `requirements.txt` exists: run `uv pip install -r requirements.txt`
- If neither exists: install the core stack:

```bash
uv pip install \
  timesfm \
  chronos-forecasting \
  transformers \
  torch \
  pandas \
  numpy \
  yfinance \
  fredapi \
  cvxpy \
  scipy \
  gradio \
  plotly \
  matplotlib \
  arch \
  scikit-learn \
  python-dotenv
```

Note: `timesfm` requires Python ≤ 3.11 and has heavy torch deps — be patient.

### 4. Verify key imports

Run a quick smoke test in the venv:

```python
import torch
import pandas
import numpy
import cvxpy
import gradio
import plotly
print(f"torch: {torch.__version__}, device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
```

Report any import failures immediately. Do NOT proceed if torch or cvxpy fail.

### 5. Check .env

Look for `.env` in the project root.

- If present: confirm `FRED_API_KEY` is defined (count keys, do NOT print values)
- If missing: create `.env` from this template and tell the user to fill it in:

```
FRED_API_KEY=your_key_here
# Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html
```

Also ensure `.gitignore` includes `.env`.

### 6. Check raw data

Look for any CSV files in `data/raw/`.

- If empty: remind the user to download the S&P 500 dataset from Kaggle:
  - Dataset: "S&P 500 Stocks" (ticker OHLCV, bulk)
  - Place CSVs in `data/raw/`
- If files exist: report count and date range if detectable from filenames

### 7. Report summary

Print a clean status table:

```
✅ / ❌  Project structure
✅ / ❌  Virtual environment (.venv)
✅ / ❌  Dependencies installed
✅ / ❌  Key imports (torch, cvxpy, gradio)
✅ / ❌  .env (FRED_API_KEY)
✅ / ❌  Raw data (data/raw/)
Device: cuda / cpu
```

If anything is ❌, fix it before declaring init complete.
Only declare "READY" when all checks pass.
