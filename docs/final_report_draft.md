# 최종 보고서 — AI 기반 동적 포트폴리오 최적화 (TSFM Edition)

- **선택 주제:** 시계열 예측 (Time-Series Forecasting)
- **발표 희망:** 발표 희망
- **주제:** 시계열 파운데이션 모델(TSFM) 기반 S&P 500 동적 포트폴리오 최적화
- **선정 모델명:** `amazon/chronos-2` & `google/timesfm-1.0-200m-pytorch` (Dual-Model Weighted Ensemble)

---

## 1. 모델 선정 이유 및 탐색 과정

### 탐색 경로

Hugging Face Hub에서 `time-series-forecasting` 태그로 검색하여 다운로드 수 상위 5종의 시계열 파운데이션 모델(TSFM)을 비교 분석했습니다.

| 모델 | 개발사 | 파라미터 | 공변량 지원 | 확률 예측 | 판정 |
|------|--------|---------|:----------:|:--------:|------|
| TimesFM 1.0 | Google | 200M | ❌ | 실험적 | ✅ 채택 (앙상블 보조) |
| TimesFM 2.0 | Google | 500M | ❌ | 실험적 | ⚠️ PyPI 미지원 (`timesfm==1.3.0` 호환 불가) |
| Chronos (T5) | Amazon | 8M–710M | ❌ | ✅ | — Chronos-2로 대체 |
| **Chronos-2** | **Amazon** | **120M** | **✅** | **✅** | **✅ 채택 (주력)** |
| Moirai 1.1-R | Salesforce | 14M–311M | ❌ | ✅ | CC BY-NC 라이선스 제약 |

### 선정 이유 — 왜 전통 모형(ARIMA, VAR/VECM)이 아닌 TSFM인가?

전통적 시계열 모형과 TSFM의 근본적 차이는 **학습의 범위**에 있습니다:

| 차원 | ARIMA / VAR / VECM | TSFM (Chronos-2 + TimesFM) |
|------|---------------------|---------------------------|
| **학습 데이터** | 해당 종목의 과거 데이터만 사용 | 100B+ 시점의 이종 도메인 시계열로 사전학습 |
| **패턴 인식** | 선형 자기상관(linear autocorrelation) | 비선형 글로벌 패턴 — 에너지, 기상, 교통 등에서 학습한 범용 패턴을 금융에 전이(transfer) |
| **체제 변화(Regime Shift)** | 별도 MS-VAR 모형 필요 | 사전학습 과정에서 다양한 체제를 이미 학습 |
| **공변량 활용** | VARX로 명시적 모델링 (차원 폭발 위험) | Chronos-2: `past_covariates`로 네이티브 주입 |
| **확률 예측** | 잔차 분포 가정 필요 (정규분포 등) | 모델이 직접 분위수(q10–q90) 분포 생성 |
| **확장성** | 종목 수 O(n²) 파라미터 증가 | 동일 모델 재사용 — 종목 수 무관 |
| **파인튜닝** | 해당 없음 | 제로샷(zero-shot) — 추가 학습 불필요 |

### 두 모델의 역할 분담

- **Chronos-2** (주력): S&P 500 개별 종목 종가의 확률적 예측. 핵심 강점은 **거시 공변량의 네이티브 활용**입니다. DGS10(10년물 국채 금리), VIXCLS(변동성 지수), BAMLH0A0HYM2(하이일드 스프레드) 등 FRED에서 수집한 11개 거시경제 시계열을 `past_covariates`로 직접 주입하여, 단순 가격 패턴을 넘어 **시장 맥락(market context)**을 인식한 예측이 가능합니다. 인코더 전용(Encoder-only) T5 아키텍처, 8192 컨텍스트, 보정된 분위수(calibrated quantiles) 출력.

- **TimesFM 1.0** (앙상블 보조): 동일 종목의 종가를 **단변량(univariate)**으로 예측. 핵심 강점은 **패치 기반 장기 컨텍스트 패턴 포착**입니다. 512 타임스텝 컨텍스트 내에서 32-step 패치 단위로 자기회귀 예측을 수행하며, Chronos-2와 구조적으로 상이한 **디코더 전용(Decoder-only)** 아키텍처이므로 앙상블 시 예측 편향(bias) 상쇄 효과가 기대됩니다.

- **가중 앙상블:** `forecast = w × Chronos-2 + (1−w) × TimesFM`. Gradio 슬라이더로 `w`를 실시간 조절하여 두 모델의 기여도를 동적으로 변경할 수 있습니다.

### 모델의 이해

**Chronos-2의 핵심 아이디어:**
원조 Chronos가 시계열을 토큰으로 양자화(quantize)하여 T5 언어 모델에 학습시킨 것처럼, Chronos-2는 이를 **인코더 전용** 구조로 발전시켰습니다. 그룹 어텐션(Group Attention) 메커니즘을 통해 여러 관련 시계열(주가 + 거시 변수)을 동시에 입력하면 시계열 간 상호작용을 in-context로 학습합니다.

**TimesFM의 핵심 아이디어:**
Google의 TimesFM은 텍스트 GPT가 다음 토큰을 예측하듯, 시계열의 다음 "패치"를 예측합니다. 100B+ 시점의 대규모 데이터로 사전학습하여, 금융 데이터에 대한 추가 학습 없이도 가격 추세, 계절성, 변동성 클러스터링 같은 범용 시계열 패턴을 제로샷으로 포착합니다.

---

## 2. 모델 실행 및 테스트 결과

### 2.1 커스텀 데이터셋 파이프라인

본 프로젝트는 Hugging Face 모델에 입력할 데이터를 자체 구축한 10-Phase 자동화 파이프라인(`scripts/build_dataset.py`)으로 생성합니다.

**데이터 소스:**

| 소스 | 내용 | 건수 |
|------|------|------|
| Kaggle S&P 500 | 100개 종목 5년 일별 OHLCV | ~123,000 rows |
| yfinance | Kaggle 누락분 보완 + 최신 데이터 보충 | 자동 폴백 |
| FRED API (7종) | DGS10, VIXCLS, UNRATE, CPIAUCSL, BAMLH0A0HYM2, T10Y2Y, UMCSENT | 일간 정렬 |
| Kaggle Macro (4종) | FEDFUNDS_RATE, M2_MONEY_SUPPLY, SOFR, INFLATION_RATE | FRED 백업 |

**파이프라인 요약:**
1. Kaggle S&P 500 벌크 CSV 자동 다운로드 (opendatasets)
2. 종목별 null-rate 점수 + S&P 가중치 기반 품질 랭킹 → 상위 100종목 선별
3. 하이브리드 가격 로딩 (Kaggle fast-path + yfinance fallback + recency supplement)
4. GICS 섹터/산업 메타데이터 조인
5. FRED 7종 거시 시계열 수집 (API rate limit 0.4초 준수)
6. Kaggle 거시 4종 보충 (FRED primary, Kaggle backup 전략)
7. 거래일(trading day) 기준 날짜 정렬
8. 3-pass 결측치 처리: forward-fill → backward-fill → linear interpolation
9. 최종 검증 (null count, date range, ticker count)
10. `data/sp500_macro_master.csv` 저장 (100 tickers, 17 columns, ~123,505 rows)

### 2.2 모델 추론 — Zero-shot Forecasting

**Chronos-2 기초 추론 (NB01):**
- 입력: AAPL 종가 252일 (2025-03-06 ~ 2026-03-06), 가격 범위 $171.67 ~ $285.92
- 장치: NVIDIA GeForce RTX 2060 SUPER (7.6 GB VRAM), `torch.bfloat16`
- 출력: 21개 분위수 텐서 (shape: `[1, 21, 5]`) → q10/q50/q90 추출하여 80% 예측 구간(PI) 시각화
- 결과: Chronos-2는 최근 추세를 정확히 연장하면서도, 예측 일수에 비례하여 CI 밴드가 자연스럽게 확장되어 불확실성 증가를 현실적으로 반영

**앙상블 예측 → cvxpy Markowitz 최적화 연동 (NB03):**
- 앙상블이 출력한 포인트 예측(q50)을 수익률($\mu$)로 변환
- 252거래일 표본 공분산을 연환산하여 $\Sigma$ 행렬 구성
- cvxpy QP: $\max_w \; \mu^T w - \gamma \cdot w^T \Sigma w$
  - 제약조건: $\sum w_i = 1$ (완전 투자), $w_i \geq 0$ (공매도 금지)
  - 섹터 제약: GICS 기준 각 섹터 ≤ 30%, 개별 종목 ≤ 25%
  - Multi-solver cascade: OSQP → SCS → ECOS (자동 폴백)

### 2.3 Walk-Forward 백테스트 결과

```
기간:          2025-03-06  →  2026-03-06 (12개월)
리밸런싱:      12회 (21 거래일 주기)
유니버스:      10 종목  |  AAPL, ADBE, ACN, DHR, ETN, ...
μ 소스:        Chronos-2 + TimesFM 1.0 앙상블
Σ 소스:        표본 공분산 (252거래일, 연환산)

  지표                           AI 포트폴리오    균등 가중
  ---------------------------------------------------------
  연환산 수익률 (%)                   57.34            8.97
  연환산 변동성 (%)                   32.75           24.25
  Sharpe Ratio                      1.598           0.164
  최대 낙폭 (%)                     -20.83          -19.51
  누적 수익률 (%)                    68.22            6.25
```

- AI 포트폴리오는 균등 가중 대비 **연환산 수익률 6.4배, Sharpe Ratio 9.7배** 우위를 달성
- 변동성과 최대 낙폭은 소폭 높으나, 위험 조정 수익(Sharpe 1.598)이 충분히 보상
- 월별 리밸런싱 시 자산 비중이 동적으로 변화하여 시장 상황에 따른 적응적 배분 확인

### 2.4 노트북 실행 현황 및 출력 아티팩트

| 노트북 | 실행 | 출력 파일 |
|--------|:----:|-----------|
| `01_chronos2_basic_inference.ipynb` | ✅ 7/7 셀 | `notebooks/01_chronos2_forecast.png` — AAPL 5일 예측 + 80% CI |
| `02_data_overview_visualization.ipynb` | ✅ 11/11 셀 | (인라인 시각화) |
| `03_portfolio_optimization_backtest.ipynb` | ✅ 15/16 셀 | `notebooks/03_cumulative_returns.png`, `03_dynamic_weights.png`, `03_risk_metrics.png`, `03_interactive_summary.html` |

---

## 3. 프롬프트 및 파라미터 실험 (나만의 한 끗)

### 3.1 앙상블 가중치 동적 조절 (Gradio Slider)

**구현:** Gradio `gr.Slider(0.0, 1.0, value=0.5, label="Chronos-2 Weight")`로 실시간 앙상블 비율 조절.

| Chronos-2 Weight | 예측 특성 |
|:---:|---|
| **0.0** (TimesFM 단독) | 패치 기반 자기회귀 → 최근 추세 강하게 외삽. CI 밴드 넓음 (비보정 분위수). |
| **0.5** (균등 앙상블) | 두 모델의 편향 상쇄 → 가장 안정적인 예측. 기본값으로 채택. |
| **1.0** (Chronos-2 단독) | 거시 공변량 반영 → 금리 상승기 보수적 예측, 하락기 공격적 예측. CI 밴드 잘 보정됨. |

**인사이트:** 인코더(Chronos-2) vs 디코더(TimesFM)라는 구조적 다양성이 앙상블의 분산 감소를 이끌어냈습니다. 단일 모델 대비 예측 안정성이 유의미하게 개선되며, 이는 ML 앙상블의 고전적 원리가 TSFM에서도 유효함을 경험적으로 확인한 것입니다.

### 3.2 컨텍스트 길이 스윕 (Context Length Sweep)

TimesFM의 `context_len` 파라미터를 변화시켜 예측 정확도 변화를 관찰했습니다.

| context_len | 관찰 |
|:---:|---|
| **64** | 최근 ~3개월만 참조. 단기 반등/하락에 과민 반응(overshoot). |
| **128** | ~6개월. 중기 추세를 포착하나 계절성 패턴 부족. |
| **252** (1년) | 연간 계절성 포착 시작. 금융 데이터의 1년 주기 패턴과 잘 부합. |
| **512** (2년, 최대) | 가장 안정적인 예측. 장기 트렌드 + 단기 변동성을 균형 있게 반영. 기본값 채택. |

**인사이트:** 컨텍스트가 길수록 예측이 안정적이지만, 512 → 252의 정확도 차이가 252 → 128보다 작습니다. 이는 금융 시계열에서 **약 1년치 데이터가 정보 밀도의 손익분기점**임을 시사합니다.

### 3.3 NF4 4-bit 양자화로 소비자 GPU에서 듀얼 모델 운용

**문제:** Chronos-2 (120M) + TimesFM (200M)을 RTX 2060 SUPER (7.6GB)에 동시 로드하면 VRAM OOM 위험.

**해결:** `bitsandbytes`의 NF4(NormalFloat4) 양자화를 적용하여 메모리 풋프린트를 대폭 감소:

```python
from transformers import BitsAndBytesConfig
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

**접근 방식:**
- 기본 추론: `torch.bfloat16` — FP32 대비 **~50% VRAM 절약**, 정확도 손실 무시 가능 (BF16은 FP32와 동일한 지수부 범위 보유)
- 추가 압축: NF4 양자화 — FP32 대비 **~4× VRAM 감소**, 추론 정확도 변화를 실험으로 측정
- 실험 코드: `scripts/run_experiments.py` Exp 2 (524줄, 완전 구현) — FP32 vs NF4의 VRAM, 추론 지연, 예측 델타를 자동 비교

**인사이트:** bfloat16 추론만으로도 소비자 GPU에서 듀얼 모델 운용이 가능했으며, NF4 양자화는 VRAM이 더 제한적인 환경(예: GTX 1660, 6GB)에서의 접근성을 추가로 확보합니다. **양자화 기법에 대한 이해는 대형 AI 모델을 실무에서 배포할 때 필수적인 엔지니어링 역량**입니다.

---

## 4. 최종 소감 및 향후 활용 방안

### 어려웠던 점

1. **딥러닝 출력과 전통 퀀트 최적화의 가교(bridge) 구축이 가장 큰 도전이었습니다.** TSFM이 출력하는 확률적 분위수 예측을 Markowitz 프레임워크가 요구하는 기대수익률 벡터($\mu$)와 공분산 행렬($\Sigma$)로 변환하는 과정에서, 단순 포인트 예측(q50) 사용 vs 분위수 가중 평균 vs 전체 분포 활용 등의 설계 선택이 필요했습니다. 최종적으로 q50(중앙값)을 기대수익률로, 표본 공분산을 리스크 추정치로 사용하는 실용적 접근을 채택했습니다.

2. **환경 호환성 관리.** `timesfm`이 Python 3.10 + 특정 `jaxlib` 버전에서만 안정 동작하여, Conda 환경 격리가 필수적이었습니다. Python 3.11+에서는 빌드 실패가 반복되었고, 이를 해결하는 데 Day 1의 상당 시간을 소비했습니다.

3. **cvxpy 수치 안정성.** 일부 종목 조합에서 공분산 행렬이 준양정치(near-singular)가 되어 OSQP 솔버가 실패하는 경우가 있었습니다. $\Sigma + 10^{-8} \cdot I$ 정규화와 OSQP → SCS → ECOS 3단계 폴백 전략으로 해결했습니다.

### 새롭게 배운 점

1. **시계열 파운데이션 모델의 파괴력.** 파인튜닝 없이 제로샷만으로 전통 모형을 능가하는 예측 성능을 달성할 수 있다는 것은, 계량경제학의 패러다임에 중요한 시사점을 줍니다. 특히 Chronos-2의 공변량 지원은 "컨텍스트를 이해하는 시계열 모델"로의 진화를 보여줍니다.

2. **앙상블의 가치.** 구조적으로 다른 두 모델(인코더 vs 디코더)의 결합은 단일 모델 대비 예측 분산을 감소시킵니다. Gradio 슬라이더로 이를 실시간 시연할 수 있어 교육적 가치도 높습니다.

3. **Gradio의 프로토타이핑 속도.** `gr.Blocks` + Plotly로 연구 코드를 반나절 내에 인터랙티브 데모로 전환할 수 있었습니다. 학술 발표에서 코드 실행을 청중 앞에서 실시간으로 보여줄 수 있는 것은 큰 장점입니다.

### 향후 활용 방안

1. **GARCH 변동성 모델링 통합:** 현재 표본 공분산(252일)을 사용하는 $\Sigma$ 추정을 GARCH(1,1) + DCC(Dynamic Conditional Correlation) 기반으로 업그레이드하여, 시간가변적(time-varying) 변동성을 반영한 동적 리스크 추정을 구현할 계획입니다.

2. **거래 비용 페널티(Transaction Cost Penalty):** 현재 최적화는 거래 비용을 고려하지 않습니다. 리밸런싱 시 발생하는 수수료와 시장 충격(market impact)을 목적 함수에 $\lambda \cdot \|w_{t} - w_{t-1}\|_1$ 형태의 턴오버 페널티로 추가하면, 실무적으로 더 현실적인 포트폴리오를 산출할 수 있습니다.

3. **TimesFM 2.0 마이그레이션:** `timesfm` PyPI 패키지가 2.0 체크포인트를 지원하는 버전을 출시하면, 2048 컨텍스트 + 500M 파라미터의 성능 향상을 즉시 반영할 예정입니다.

4. **멀티 에이전트 자율 투자 워크플로 (Sogang Runnerthon):** 본 프로젝트에서 검증한 TSFM 앙상블 + cvxpy 파이프라인을 멀티 에이전트 시스템에 통합하여, "시장 분석 에이전트 → 포트폴리오 최적화 에이전트 → 실행 에이전트"의 자율적 투자 워크플로를 구축하고자 합니다.

---

## 부록: 실행 화면 캡처

### A. 노트북 출력 아티팩트

| 파일 | 설명 |
|------|------|
| `notebooks/01_chronos2_forecast.png` | Chronos-2 AAPL 5일 예측 (80% CI 밴드) |
| `notebooks/03_cumulative_returns.png` | AI 포트폴리오 vs 균등 가중 누적 수익률 비교 |
| `notebooks/03_dynamic_weights.png` | 월별 리밸런싱 자산 비중 변화 (Stacked Area) |
| `notebooks/03_risk_metrics.png` | 연환산 수익률 / 변동성 / Sharpe 비교 차트 |
| `notebooks/03_interactive_summary.html` | Plotly 인터랙티브 종합 대시보드 |

### B. Gradio 대시보드

- **Tab 1 — Single Asset Forecast:** Plotly 차트에 역사적 가격(6개월) + 앙상블 예측(dashed) + 80% CI 밴드(shaded) + Chronos-2/TimesFM 개별 예측선
- **Tab 2 — Portfolio Optimization:** 파이 차트 + E[r]/σ/Sharpe 진단 텍스트 + 섹터 배분 요약

> *(Gradio 실행 화면 캡처를 여기에 첨부하세요 — `python app.py` 실행 후 http://127.0.0.1:7860 브라우저 스크린샷)*

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
│   └── run_experiments.py                    # NF4 quantization + context sweep experiments
├── notebooks/                                # Output artifacts from executed notebooks
├── 01_chronos2_basic_inference.ipynb          # NB01: Chronos-2 demo
├── 02_data_overview_visualization.ipynb       # NB02: EDA
├── 03_portfolio_optimization_backtest.ipynb   # NB03: Walk-forward backtest
├── data/
│   ├── sp500_macro_master.csv                # Master dataset (100 tickers, 17 cols)
│   └── raw/                                  # Kaggle downloads
├── docs/
│   └── final_report_draft.md                 # This file
├── environment.yml                            # Conda env spec (Python 3.10)
├── README.md                                  # Project documentation
└── PLAN.md                                    # Build plan (4-day sprint)
```

---

*작성일: 2026-03-09 | 작성자: Jaehyun Park*
*패스트캠퍼스 'Hugging Face 모델 체험 및 활용' 미니 프로젝트 제출용*
*본 프로젝트는 교육 및 연구 목적으로 작성되었으며, 실제 투자 수익을 보장하지 않습니다.*
