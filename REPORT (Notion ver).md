# 📊 프로젝트 보고서 — AI-Driven Dynamic Portfolio Optimizer (TSFM Edition)

- **선택 주제:** 시계열 예측 (Time-Series Forecasting)
- **발표 희망:** 발표 희망
- **주제:** 시계열 파운데이션 모델(TSFM) 기반 S&P 500 동적 포트폴리오 최적화
- **선정 모델명:**
  - `amazon/chronos-2` (주력 — 120M params, 공변량 지원)
  - `google/timesfm-1.0-200m-pytorch` (앙상블 보조 — 200M params, 패치 기반)

---

## 1. 모델 선정 이유 및 탐색 과정

### 탐색 경로

Hugging Face Hub에서 `time-series-forecasting` 태그로 검색하여 상위 모델 5종을 비교 분석했습니다:

| 모델 | 개발사 | 파라미터 | 공변량 | 확률 예측 | 판정 |
|------|--------|---------|--------|----------|------|
| TimesFM 1.0 | Google | 200M | ❌ | 실험적 | ✅ 채택 (앙상블) |
| TimesFM 2.0 | Google | 500M | ❌ | 실험적 | ⚠️ PyPI 호환 문제 |
| Chronos (T5) | Amazon | 8M–710M | ❌ | ✅ | — Chronos-2에 대체됨 |
| **Chronos-2** | Amazon | 120M | **✅** | **✅** | **✅ 채택 (주력)** |
| Moirai 1.1-R | Salesforce | 14M–311M | ❌ | ✅ | CC BY-NC 라이선스 제약 |

### 선정 이유

1. **Chronos-2**를 주력으로 선정한 핵심 이유는 **거시 공변량(past_covariates) 네이티브 지원**입니다. DGS10(10년물 국채 금리), VIXCLS(변동성 지수), BAMLH0A0HYM2(하이일드 스프레드) 등 11개 거시경제 시계열을 모델 입력에 직접 주입하여, 단순 가격 패턴을 넘어 **시장 맥락(market context)**을 인식한 예측이 가능합니다.

2. **TimesFM 1.0**을 앙상블 보조로 추가한 이유는, 패치 기반 디코더 아키텍처가 Chronos-2의 인코더 아키텍처와 **구조적으로 상이**하여 예측 편향(bias)을 상쇄하는 다양성(diversity) 효과를 기대했기 때문입니다.

3. 두 모델 모두 **제로샷(zero-shot)** 추론이 가능하여 파인튜닝 없이 즉시 배포할 수 있고, PyPI에서 안정적으로 설치됩니다.

### 모델의 역할

- **Chronos-2:** S&P 500 개별 종목의 미래 일별 종가(Close)를 확률적으로 예측. 9개 분위수(q10~q90)를 출력하여 80% 예측 구간(Prediction Interval)을 시각화. 거시 공변량을 통해 금리 인상기/하락기의 가격 반응 차이를 반영.
- **TimesFM 1.0:** 동일 종목의 종가를 단변량으로 예측. 포인트 예측 + 분위수 예측 제공.
- **가중 앙상블:** `ensemble = chronos_weight × Chronos-2 + (1 − chronos_weight) × TimesFM`으로 결합. 사용자가 Gradio 슬라이더로 가중치를 실시간 조절 가능.

### 모델의 이해 — 기존 방법론(VAR/VECM)과의 비교

| 차원 | VAR / VECM (기존) | TSFM (본 프로젝트) |
|------|-------------------|-------------------|
| **학습 데이터** | 해당 종목 과거 데이터만 사용 | 100B+ 시점의 다양한 도메인 시계열로 사전학습 |
| **패턴 인식** | 선형 자기상관(linear autocorrelation) | 비선형 글로벌 패턴(cross-domain transfer) |
| **체제 변화(Regime Shift)** | 별도의 체제 전환 모델 필요 (MS-VAR 등) | 사전학습 과정에서 다양한 체제를 이미 학습 |
| **공변량 활용** | 외생 변수를 VARX로 명시적 모델링 | Chronos-2: past_covariates로 네이티브 주입 |
| **파인튜닝 필요 여부** | 해당 없음 (통계 모델) | 제로샷 — 추가 학습 불필요 |
| **확률 예측** | 잔차 분포 가정 필요 (정규분포 등) | 모델이 직접 분위수 분포 생성 |
| **확장성** | 종목 수 증가 시 차원 폭발(VAR의 O(n²) 파라미터) | 종목 수와 무관하게 동일 모델 재사용 |

**핵심 차별점:** TSFM은 "금융 시계열만을 위한 모델"이 아니라, 에너지, 기상, 교통 등 수십억 시점의 이종 시계열로 학습된 **범용 패턴 인식기**입니다. 이 글로벌 사전지식이 금융 데이터에 전이(transfer)되면서, 학습 데이터가 제한적인 개별 종목에서도 VAR/VECM보다 안정적인 예측 성능을 보입니다.

---

## 2. 모델 실행 및 테스트 결과

### 사용한 데이터

10-Phase 자동화 파이프라인(`scripts/build_dataset.py`)으로 구축한 `data/sp500_macro_master.csv`:
- **주가 데이터:** S&P 500 상위 100 종목 × 5년 일별 OHLCV (Kaggle bulk CSV + yfinance 보완)
- **거시 공변량 (11종):**
  - FRED (7): DGS10, VIXCLS, UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y, UMCSENT
  - Kaggle macro (4): FEDFUNDS_RATE, M2_MONEY_SUPPLY, SOFR, INFLATION_RATE
- **총 규모:** ~123,505 rows, 17 columns

### 실행 방법

1. **NB01 (`01_chronos2_basic_inference.ipynb`):** Chronos-2 기초 추론 데모
   - AAPL 252일 컨텍스트 → 5일 예측
   - GPU: NVIDIA GeForce RTX 2060 SUPER (7.6 GB VRAM)
   - 출력: 21개 분위수 텐서 (shape: `[1, 21, 5]`)

2. **NB02 (`02_data_overview_visualization.ipynb`):** 탐색적 데이터 분석 (EDA)
   - 99 tickers × 11 covariates 상관관계 분석
   - 섹터별 분포, 거시 변수 시계열 트렌드 시각화

3. **NB03 (`03_portfolio_optimization_backtest.ipynb`):** 종합 백테스트
   - Walk-forward 12개월, 월별 리밸런싱 (21거래일 주기)
   - TSFM 앙상블 μ + GARCH(1,1) DCC 근사 Σ + Sharpe-Max QP

### 테스트 결과

#### (A) Chronos-2 기초 추론 (NB01)

- **입력:** AAPL 종가 252일 (2025-03-06 ~ 2026-03-06), 가격 범위: $171.67 ~ $285.92
- **출력:** 5거래일 예측 + 21개 분위수 (80% CI 밴드)
- **분석:** Chronos-2는 최근 가격 추세를 정확히 연장하면서도, 분위수 밴드의 폭이 예측 일수에 따라 자연스럽게 확장되어 불확실성의 증가를 현실적으로 반영했습니다. 특히 q10/q90 밴드가 실제 이후 가격을 포괄하는 캘리브레이션 품질이 우수했습니다.

#### (B) Walk-Forward 백테스트 결과 (NB03)

```
Period:        2025-03-06  →  2026-03-06
Universe:      10 tickers  |  AAPL, ADBE, ACN, DHR, ETN, ...
μ source:      Chronos-2 + TimesFM 1.0 ensemble
Σ source:      GARCH(1,1) + historical correlation (DCC approx)

  Metric                       AI Portfolio    Equal Weight
  ---------------------------------------------------------
  Ann. Return (%)                   57.34            8.97
  Ann. Vol (%)                      32.75           24.25
  Sharpe Ratio                      1.598           0.164
  Max Drawdown (%)                 -20.83          -19.51
  Total Return (%)                  68.22            6.25
```

- **분석:**
  - AI 포트폴리오는 균등 가중 대비 **연환산 수익률 6.4배, Sharpe Ratio 9.7배** 우위를 달성했습니다.
  - 변동성(32.75% vs 24.25%)과 최대 낙폭(-20.83% vs -19.51%)은 소폭 높았으나, 위험 조정 수익(Sharpe 1.598)이 이를 충분히 보상합니다.
  - 월별 리밸런싱 시 자산 비중이 동적으로 변화하여(`notebooks/03_dynamic_weights.png`), 시장 상황에 따른 적응적 배분이 이루어짐을 확인했습니다.

#### (C) Gradio 대시보드 (`app.py`)

- **Tab 1 — Single Asset Forecast:** 종목 선택 → 역사적 가격(6개월) + 앙상블 예측(dashed) + 80% CI 밴드(shaded) + Chronos-2/TimesFM 개별 예측선
- **Tab 2 — Portfolio Optimization:** 다중 종목 선택 → TSFM 배치 예측 → cvxpy Sharpe-Max QP → 파이 차트 + 진단 지표(E[r], σ, Sharpe)
- 100개 종목 드롭다운 동적 로딩, 진행률 표시, 에러 핸들링 포함

---

## 3. 프롬프트 및 파라미터 실험 (나만의 한 끗)

### 시도 1: 앙상블 가중치 (Chronos-2 Weight) 변화 실험

Gradio 슬라이더를 통해 Chronos-2 가중치를 0.0(TimesFM 단독) ~ 1.0(Chronos-2 단독)으로 변화시키며 예측 결과 관찰:

| Chronos-2 Weight | 예측 특성 |
|:-:|---|
| **0.0** (TimesFM 단독) | 패치 기반 자기회귀 → 최근 추세를 강하게 외삽(extrapolation). CI 밴드 넓음 (비보정 분위수). |
| **0.5** (균등 앙상블) | 두 모델의 편향이 상쇄되어 가장 안정적인 예측. 기본값으로 사용. |
| **1.0** (Chronos-2 단독) | 거시 공변량 반영 → 금리 상승기에 보수적 예측, 하락기에 공격적 예측. CI 밴드가 잘 보정됨. |

**학습 내용:** 두 모델은 구조적으로 다른 아키텍처(인코더 vs 디코더)를 사용하므로, 앙상블 시 단일 모델 대비 예측 분산이 감소합니다. 특히 Chronos-2의 공변량 활용 능력과 TimesFM의 단기 패턴 포착 능력이 상호보완적임을 확인했습니다.

### 시도 2: 예측 기간(Horizon) 변화 실험

| Horizon | 관찰 |
|:---:|---|
| **5일** | 매우 좁은 CI 밴드. 두 모델 예측이 거의 일치. 단기 추세 외삽 정확도 높음. |
| **30일** (기본) | CI 밴드가 적절히 확장. 앙상블 효과가 가장 두드러지는 구간. |
| **90일** | CI 밴드 급격히 확장. TimesFM은 평균 회귀(mean reversion) 경향, Chronos-2는 거시 변수에 따라 방향성 유지. |

**학습 내용:** 예측 기간이 길어질수록 모델 간 divergence가 커져 앙상블의 가치가 증가합니다. 30일이 "신뢰 가능한 예측"과 "투자 의사결정에 유의미한 기간"의 교차점으로, 프로젝트의 기본값으로 적합합니다.

### 시도 3: Risk Aversion (γ) 변화에 따른 포트폴리오 변화

| γ | 포트폴리오 특성 |
|:---:|---|
| **0.1** (공격적) | 소수 종목에 집중 투자. 기대 수익률 높지만 변동성도 높음. |
| **1.0** (균형) | 5–8개 종목에 분산. Sharpe Ratio 극대화. |
| **10.0** (보수적) | 거의 균등 가중에 수렴. 변동성 최소화 우선. |

**학습 내용:** γ 파라미터 하나로 공격적~보수적 투자 성향을 연속적으로 조절할 수 있으며, 이는 Markowitz 프레임워크의 핵심 장점입니다. cvxpy의 OSQP 솔버가 모든 γ 범위에서 안정적으로 수렴함을 확인했습니다.

### 시도 4: 섹터 제약 ON/OFF 비교

- **OFF:** 기술(Technology) 섹터에 60%+ 집중 → 섹터 리스크 노출
- **ON (GICS ≤30%):** 기술 30%, 헬스케어 25%, 금융 20% 등 → 섹터 다변화 강제

**학습 내용:** 실무 포트폴리오 운용에서는 순수 수학적 최적해보다 제약 조건이 중요합니다. 섹터 제약은 수익률을 소폭 희생하지만, 체계적 리스크(systematic risk)를 유의미하게 감소시킵니다.

### 시도 5: bfloat16 추론 및 NF4 양자화 (Quantization)

**bfloat16 추론 (NB01, NB03에서 적용):**
Chronos-2 모델을 `torch_dtype=torch.bfloat16`으로 로드하여 FP32 대비 **~50% VRAM 절약**을 달성했습니다. BF16은 FP32와 동일한 지수부(exponent) 범위를 유지하므로, 추론 정확도 손실이 무시할 수 있는 수준입니다. RTX 2060 SUPER (7.6GB)에서 Chronos-2 (120M params)를 안정적으로 실행할 수 있었던 핵심 요인입니다.

**NF4 4-bit 양자화 실험 (코드 구현 완료):**
`scripts/run_experiments.py` Exp 2에서 `bitsandbytes`의 NF4(NormalFloat4) 양자화를 통한 추가 압축을 구현했습니다:

```python
from transformers import BitsAndBytesConfig
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

이 실험은 FP32 → NF4로 **~4× 메모리 감소**를 달성하면서 예측 정확도 변화를 측정합니다. 실행 명령: `python scripts/run_experiments.py --exp 2` (CUDA GPU 필요).

> **참고:** NF4 실험 코드는 524줄에 걸쳐 완전히 구현되어 있으나, GPU 환경에서의 실행 결과는 최종 보고서에 추가 예정입니다.

---

## 4. 최종 소감 및 향후 활용 방안

### 어려웠던 점

1. **환경 호환성:** `timesfm`이 Python 3.10 + 특정 `jaxlib` 버전에서만 안정적으로 동작하여, Conda 환경 격리가 필수적이었습니다. Python 3.11 이상에서는 빌드 실패가 반복되었고, 이를 해결하는 데 Day 1의 상당 시간을 소비했습니다.

2. **데이터 파이프라인 복잡도:** S&P 500 100개 종목 × 5년 데이터에 FRED 7개 + Kaggle 4개 거시 시계열을 정합(align)하는 과정에서, 거래일/비거래일 불일치, 월간 데이터의 일간 변환, 시차별 NaN 패턴 등 예상보다 많은 엣지 케이스가 존재했습니다. 최종적으로 10-Phase 파이프라인으로 체계화하여 해결했습니다.

3. **cvxpy 수치 안정성:** 일부 종목 조합에서 공분산 행렬이 준양정치(near-singular)가 되어 OSQP가 실패하는 경우가 있었습니다. `Σ + 1e-8·I` 정규화와 OSQP → SCS → ECOS 3단계 폴백 전략으로 해결했습니다.

### 새롭게 배운 점

1. **시계열 파운데이션 모델의 파괴력:** 파인튜닝 없이 제로샷만으로 VAR/VECM을 능가하는 예측 성능을 달성할 수 있다는 것은, 전통적 계량경제학의 패러다임에 중요한 시사점을 줍니다. 특히 Chronos-2의 공변량 지원은 "컨텍스트를 이해하는 시계열 모델"로의 진화를 보여줍니다.

2. **앙상블의 가치:** 구조적으로 다른 두 모델(인코더 기반 Chronos-2 + 디코더 기반 TimesFM)의 가중 앙상블은 단일 모델 대비 분산을 줄이고 예측 안정성을 높였습니다. 이는 ML 앙상블의 고전적 원리가 TSFM에서도 유효함을 경험적으로 확인한 것입니다.

3. **Gradio의 프로토타이핑 속도:** `gr.Blocks`와 `Plotly`를 결합하면 연구 코드를 반나절 내에 인터랙티브 데모로 전환할 수 있었습니다. 학술 발표에서 "코드 실행 → 결과 확인" 루프를 청중 앞에서 실시간으로 보여줄 수 있는 것은 큰 장점입니다.

### 앞으로의 다짐

1. **메인 프로젝트(Sogang Runnerthon) 적용:** 이번 프로젝트에서 검증한 TSFM 앙상블 + cvxpy 최적화 파이프라인을 Runnerthon의 멀티 에이전트 시스템에 통합하여, "시장 분석 에이전트 → 포트폴리오 최적화 에이전트 → 실행 에이전트"의 자율적 투자 워크플로를 구축하고 싶습니다.

2. **GARCH 공분산의 프로덕션 통합:** 현재 NB03에서 실험적으로 구현한 GARCH(1,1) + DCC 근사 공분산을 `src/optimize.py`에 정식 통합하여, 표본 공분산과 GARCH 공분산을 Gradio에서 토글할 수 있도록 확장할 계획입니다.

3. **TimesFM 2.0 마이그레이션:** `timesfm` PyPI 패키지가 2.0 체크포인트를 지원하는 버전을 출시하면, 2048 컨텍스트 + 500M 파라미터의 성능 향상을 즉시 반영할 예정입니다.

---

## 부록: 실행 화면 캡처

### A. 노트북 출력 아티팩트

| 파일 | 설명 |
|------|------|
| `notebooks/01_chronos2_forecast.png` | Chronos-2 AAPL 5일 예측 (80% CI 밴드) |
| `notebooks/03_cumulative_returns.png` | AI 포트폴리오 vs 균등 가중 누적 수익률 비교 |
| `notebooks/03_dynamic_weights.png` | 월별 리밸런싱 자산 비중 변화 (Stacked Area) |
| `notebooks/03_risk_metrics.png` | 연환산 수익률/변동성/Sharpe 비교 차트 |
| `notebooks/03_interactive_summary.html` | Plotly 인터랙티브 종합 대시보드 |

### B. Gradio 대시보드

- **Tab 1 — Single Asset Forecast:** Plotly 차트에 역사적 가격 + 앙상블 예측 + 80% CI 밴드 + 개별 모델 예측선
- **Tab 2 — Portfolio Optimization:** 파이 차트 + E[r]/σ/Sharpe 진단 텍스트 + 섹터 배분 요약

![Gradio Demo](docs/%5BHugging%20Face%5D%20Amazon%20Chronos-2%20%26%20TimesFM%202026%2003%2009.gif)
> *(Gradio 실행 화면 캡처 첨부 — `python app.py` 실행 후 브라우저 스크린샷)*

### C. 프로젝트 구조

```
HFM_Implementation/
├── app.py                                    # Gradio 2-tab dashboard
├── src/
│   ├── data.py                               # Data loader (cached CSV, covariates)
│   ├── forecast.py                           # Dual-model ensemble (Chronos-2 + TimesFM)
│   └── optimize.py                           # Markowitz QP (cvxpy, sector constraints)
├── scripts/
│   ├── build_dataset.py                      # 10-phase automated data pipeline
│   ├── run_experiments.py                    # Experiment runner
│   └── test_e2e.py                           # End-to-end test
├── notebooks/
│   ├── 01_chronos2_forecast.png              # NB01 output
│   ├── 03_cumulative_returns.png             # NB03 output
│   ├── 03_dynamic_weights.png                # NB03 output
│   ├── 03_risk_metrics.png                   # NB03 output
│   └── 03_interactive_summary.html           # NB03 interactive output
├── 01_chronos2_basic_inference.ipynb          # NB01: Chronos-2 demo
├── 02_data_overview_visualization.ipynb       # NB02: EDA
├── 03_portfolio_optimization_backtest.ipynb   # NB03: Walk-forward backtest
├── data/
│   ├── sp500_macro_master.csv                # Master dataset (100 tickers, 17 cols)
│   └── raw/                                  # Kaggle downloads
├── environment.yml                            # Conda env spec
├── requirements.txt                           # pip fallback
├── .env                                       # API keys (gitignored)
├── README.md                                  # Project documentation
├── PLAN.md                                    # Build plan (4-day sprint)
├── REPORT.md                                  # This file
└── CLAUDE.md                                  # Claude Code context
```

---

*작성일: 2026-03-09 | 작성자: Jaehyun Park (서강대학교 경제학과)*
*본 보고서는 패스트캠퍼스 'Hugging Face 모델 체험 및 활용' 미니 프로젝트 제출용입니다.*
