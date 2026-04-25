# AI 駆動型ダイナミックポートフォリオ最適化（TSFM 版）

[English (README.md)](README.md) | [한국어 (README.ko.md)](README.ko.md) | **日本語**

時系列基盤モデル（TSFM）の **TimesFM 1.0** と **Amazon Chronos-2** で期待リターンをゼロショット予測し、**Markowitz 型 2 次計画法（`cvxpy`）** でセクター・銘柄上限付きのロングオンリー・ポートフォリオを構成します。**Gradio** の 2 タブ UI で単一資産の予測と複数資産の最適化を行えます。

評価項目、モデルカード、ノートブック指標、ギャップ分析などの**全文書**はリポジトリの表紙向けではなく **[REPORT.md](REPORT.md)** にまとめています。

## 主な機能

- **データ:** Kaggle S&P 500 一括、`yfinance`、FRED + Kaggle マクロなどの自動パイプライン → `data/sp500_macro_master.csv`
- **予測:** Chronos-2（マクロ共変量）+ TimesFM 1.0（単変量）の**重み付きアンサンブル**
- **最適化:** Sharpe 重視の QP — 予算、ロングオンリー、GICS セクター ≤30%、単一銘柄 ≤25%
- **UI:** `app.py` — 予測タブ + ポートフォリオタブ（Plotly）

## 技術スタック

`torch`, `transformers`, `timesfm`, `chronos-forecasting`, `pandas`, `numpy`, `yfinance`, `fredapi`, `cvxpy`, `gradio`, `plotly`

---

## クイックスタート

**環境:** Python 3.10+ を推奨。Chronos-2 / TimesFM の推論には **CUDA GPU** を強く推奨します。

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

続けて **環境変数** を設定し、データビルドとアプリを実行します。

マスターデータセットのビルド:

```bash
python scripts/build_dataset.py
```

ダッシュボードの起動:

```bash
python app.py
```

任意: `python scripts/preload_models.py` で Hub の重みをウォームキャッシュ、`python scripts/run_experiments.py` で実験スクリプトを実行。

---

## 環境変数（`.env`）

1. テンプレートをコピーして値を記入します。実キーをコミットしないでください（`.env` は gitignore されます）。

```bash
cp .env.example .env
```

2. **読み込みの挙動:** `src/forecast.py` が `load_dotenv()` を呼ぶため、`python app.py` はプロジェクトルートの `.env` を自動で読み込みます。`scripts/build_dataset.py` は **`os.environ` のみ**参照し、`.env` を自動では読みません。シェルで export する、[direnv](https://direnv.net/) を使う、または Bash で次のように一度実行してください。

```bash
set -a && source .env && set +a && python scripts/build_dataset.py
```

| 変数 | 用途 | メモ |
|------|------|------|
| `HF_TOKEN` | Chronos-2 / TimesFM の Hugging Face Hub 認証 | `app.py`・予測ノートブックに **必須**（`src/forecast.py`）。 |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle API で S&P 500・マクロ CSV を一括取得 | キャッシュがなければ **必須**。スクリプトがこれらから `kaggle.json` を生成可能。 |
| `FRED_API_KEY` | `fredapi` で FRED マクロ系列 | **推奨**。未設定だと FRED 列が空になり警告。`--no-fred` で明示スキップ可。 |
| `ALPHAVANTAGE_API_KEY` | — | 任意。現状コアスクリプトでは未使用。 |
| `OPENAI_API_KEY` | — | 任意。現状コアスクリプトでは未使用。 |

トークン取得先: [Hugging Face トークン](https://huggingface.co/settings/tokens)、[Kaggle API](https://www.kaggle.com/settings)、[FRED API キー](https://fred.stlouisfed.org/docs/api/api_key.html)。

---

## ノートブック

| ノートブック | 役割 |
|--------------|------|
| `notebooks/01_chronos2_basic_inference.ipynb` | Chronos-2 ゼロショット予測デモ |
| `notebooks/02_data_overview_visualization.ipynb` | 価格・マクロの EDA |
| `notebooks/03_portfolio_optimization_backtest.ipynb` | アンサンブル μ + QP + ウォークフォワードバックテスト |

セルを実行すると図・HTML の出力は `notebooks/` に保存されます。

## リポジトリ構成（概要）

```text
app.py                 # Gradio エントリポイント
src/                   # 予測・最適化モジュール
scripts/               # データビルド、実験、テスト
data/                  # 生成 CSV（一部パスは gitignore の場合あり）
notebooks/             # 分析 + 出力物
docs/                  # テンプレート・下書きなど
REPORT.md              # 全文レポート（長文）
```

## ライセンス

本プロジェクトは **[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)** (SPDX: `PolyForm-Noncommercial-1.0.0`) の下に提供されます。**ソースは公開**されますが、OSI の定める「オープンソース」ライセンスではありません。ライセンス上の **非商用** 目的や、所定の非営利・教育・公的研究機関などの利用は許容されますが、**商用利用**（本コードに基づく有償の製品・サービス等）は **別途の書面による許可** が必要です。

- **著作権** © 2026 Jaehyun Park（`LICENSE` 内の `Required Notice` を参照）。再配布する場合は、本ライセンス（または上記 URL）と必須の通知行を相手方にも渡す必要があります。
- **サードパーティ:** PyTorch、Hugging Face モデル、Python パッケージ、データセットは **各自のライセンス** が適用されます。上流（モデル・データ条件等）の遵守は、本 `LICENSE` に加えて利用者の責任です。
- 利用が商用に該当するか不明な場合、または **商用ライセンス** が必要な場合は、**`LICENSE` の案内に従い著作権者へ問い合わせ**てください。

法的に有効な全文は **[LICENSE](LICENSE)** にあります。上記は要約であり、全文に代わるものではありません。
