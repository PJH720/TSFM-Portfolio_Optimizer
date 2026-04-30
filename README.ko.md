# AI 기반 동적 포트폴리오 최적화 (TSFM 에디션)

[English (README.md)](README.md) | **한국어** | [日本語 (README.ja.md)](README.ja.md)

<p align="center">
  <img src="docs/AI%20Portfolio%20Manager.jpg" alt="AI Portfolio Manager 배너" width="880" />
</p>

<p align="center">
  <img src="docs/Chronos-TimesFM.jpg" alt="Chronos TimesFM 로고" width="360" />
</p>

<p align="center">
  <a href="REPORT.md"><img src="https://img.shields.io/badge/Full%20Report-REPORT.md-6f42c1?style=for-the-badge&logo=readme&logoColor=white" alt="전체 리포트"></a>
  <a href="#데모-미리보기"><img src="https://img.shields.io/badge/Live%20Demo-Gradio-ff4b4b?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio 데모"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Models-TimesFM%20%7C%20Chronos--2-1f6feb?style=flat-square" alt="모델 TimesFM Chronos-2">
  <img src="https://img.shields.io/badge/UI-Gradio%20%2B%20Plotly-f97316?style=flat-square" alt="UI Gradio Plotly">
  <img src="https://img.shields.io/badge/Optimization-Markowitz%20QP-0ea5e9?style=flat-square" alt="최적화 Markowitz QP">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-PolyForm%20Noncommercial-22c55e?style=flat-square" alt="라이선스"></a>
</p>

시계열 파운데이션 모델(TSFM)인 **TimesFM 1.0**과 **Amazon Chronos-2**로 기대수익률을 제로샷 예측하고, **Markowitz형 2차계획법(`cvxpy`)**으로 섹터·종목 상한이 있는 롱온리 포트폴리오를 구성합니다. **Gradio** 2탭 UI에서 단일 자산 예측과 다자산 최적화를 제공합니다.

평가 기준, 모델 카드, 노트북 지표, 갭 분석 등 **전체 서술**은 정문용이 아닌 **[REPORT.md](REPORT.md)**에 있습니다.

## 데모 미리보기

![Gradio 데모](docs/%5BHugging%20Face%5D%20Amazon%20Chronos-2%20%26%20TimesFM%202026%2003%2009.gif)

## 주요 기능

- **데이터:** Kaggle S&P 500 벌크, `yfinance`, FRED·Kaggle 거시 지표 등 자동 파이프라인 → `data/sp500_macro_master.csv`
- **예측:** Chronos-2(거시 공변량) + TimesFM 1.0(단변량) **가중 앙상블**
- **최적화:** Sharpe 지향 QP — 예산, 롱온리, GICS 섹터 ≤30%, 단일 종목 ≤25%
- **UI:** `app.py` — 예측 탭 + 포트폴리오 탭 (Plotly)

## 기술 스택

`torch`, `transformers`, `timesfm`, `chronos-forecasting`, `pandas`, `numpy`, `yfinance`, `fredapi`, `cvxpy`, `gradio`, `plotly`

---

## 빠른 시작

**환경:** Python 3.10+ 권장; Chronos-2 / TimesFM 추론에는 **CUDA GPU** 강력 권장.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

아래 **환경 변수**를 설정한 뒤 데이터 빌드와 앱을 실행합니다.

마스터 데이터셋 생성:

```bash
python scripts/build_dataset.py
```

대시보드 실행:

```bash
python app.py
```

선택: `python scripts/preload_models.py`로 Hub 가중치 프리로드; `python scripts/run_experiments.py`로 실험 스크립트 실행.

---

## 환경 변수 (`.env`)

1. 템플릿을 복사한 뒤 값을 채웁니다. 실제 키는 커밋하지 마세요 (`.env`는 gitignore).

```bash
cp .env.example .env
```

2. **로딩 방식:** `src/forecast.py`에서 `load_dotenv()`를 호출하므로 `python app.py`는 프로젝트 루트의 `.env`를 자동으로 읽습니다. `scripts/build_dataset.py`는 **`os.environ`만** 사용하며 `.env`를 자동 로드하지 않습니다. 셸에 export하거나 [`direnv`](https://direnv.net/) 등을 쓰거나, Bash에서 한 번에 실행할 수 있습니다:

```bash
set -a && source .env && set +a && python scripts/build_dataset.py
```

| 변수 | 용도 | 비고 |
|------|------|------|
| `HF_TOKEN` | Chronos-2 / TimesFM 가중치 Hugging Face Hub 인증 | `app.py`·예측 노트북에 **필수** (`src/forecast.py`). |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Kaggle API로 S&P 500·거시 CSV 일괄 다운로드 | 데이터가 없으면 **필수**. 스크립트가 이 값으로 `kaggle.json`을 만들 수 있음. |
| `FRED_API_KEY` | `fredapi`로 FRED 거시 시리즈 | **권장**; 없으면 FRED 열이 비고 경고 로그. `--no-fred`로 명시 스킵 가능. |
| `ALPHAVANTAGE_API_KEY` | — | 선택. 현재 핵심 스크립트에서는 미사용. |
| `OPENAI_API_KEY` | — | 선택. 현재 핵심 스크립트에서는 미사용. |

토큰 발급: [Hugging Face 토큰](https://huggingface.co/settings/tokens), [Kaggle API](https://www.kaggle.com/settings), [FRED API 키](https://fred.stlouisfed.org/docs/api/api_key.html).

---

## 노트북

| 노트북 | 내용 |
|--------|------|
| `notebooks/01_chronos2_basic_inference.ipynb` | Chronos-2 제로샷 예측 데모 |
| `notebooks/02_data_overview_visualization.ipynb` | 가격·거시 EDA |
| `notebooks/03_portfolio_optimization_backtest.ipynb` | 앙상블 μ + QP + 워크포워드 백테스트 |

셀 실행 시 그림·HTML 산출물은 `notebooks/`에 저장됩니다.

## 디렉터리 개요

```text
app.py                 # Gradio 진입점
src/                   # 예측·최적화 모듈
scripts/               # 데이터 빌드, 실험, 테스트
data/                  # 생성 CSV (일부 경로는 gitignore 가능)
notebooks/             # 분석 + 출력물
docs/                  # 템플릿·초안 등
REPORT.md              # 전체 보고서(장문)
```

## 라이선스

이 프로젝트는 **[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/)** (SPDX: `PolyForm-Noncommercial-1.0.0`)을 따릅니다. **소스는 공개**되지만 OSI가 정의하는 “오픈소스” 라이선스는 아닙니다. 라이선스가 정한 **비상업** 용도 및 일부 비영리·공공·교육 등 기구의 사용은 허용되나, **상업적 이용**(예: 이 코드를 기반으로 한 유료 제품·서비스)은 **별도 서면 허가**가 필요합니다.

- **저작권** © 2026 Jaehyun Park (`LICENSE`의 `Required Notice` 참고). 소프트웨어를 다시 배포할 때는 본 라이선스(또는 URL)와 필수 고지를 함께 전달해야 합니다.
- **쉽게 말해:** 개인 투자자가 본 코드를 참고해 **개인용 투자 판단/어드바이스 용도**로 활용하는 것은 취지상 허용됩니다. 다만 이를 앱·서비스·상품 형태로 **상업적 재배포/판매/유료 제공**하는 것은 금지되며, 그런 사용은 별도 서면 허가가 필요합니다.
- **서드파티:** PyTorch, Hugging Face 모델, Python 패키지, 데이터셋은 **각자의 라이선스**가 적용됩니다. 상위(모델·데이터 약관 등) 준수는 본 `LICENSE`와 별도로 이용자 책임입니다.
- 사용이 상업에 해당하는지 불명확하거나 **상업 라이선스**가 필요하면 **`LICENSE`에 안내된 대로 저작권자에게 문의**하세요.

법적 효력 있는 전문은 **[LICENSE](LICENSE)** 에 있으며, 위는 이해를 돕기 위한 요약이며 법적 구속력을 대체하지 않습니다.
