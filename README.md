
# 📈 AI-Driven Dynamic Portfolio Optimizer (TSFM Edition)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Transformers-orange.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-red.svg)
![Optimization](https://img.shields.io/badge/Optimization-cvxpy-green.svg)
![Conda](https://img.shields.io/badge/Conda-hfm--data-green.svg)
![Environment](https://img.shields.io/badge/env-hfm--data_(Python_3.10)-brightgreen.svg)

본 프로젝트는 전통적인 자산 배분 모델(Markowitz Mean-Variance, VAR/VECM)의 한계를 극복하기 위해, **Hugging Face의 최신 시계열 파운데이션 모델(Time-Series Foundation Models, TSFMs)**을 도입하여 미래 기대수익률($\mu$)을 예측하고 동적으로 포트폴리오를 최적화하는 자동화된 프로젝트입니다.

## 🚀 프로젝트 개요 (Project Overview)

기존의 정적 포트폴리오 최적화는 자산의 과거 평균 수익률에 크게 의존하여 시장의 체제 변화(Regime Shift)에 취약했습니다. 본 프로젝트는 이 한계를 극복하고자 기획되었습니다.

- **As-Is (기존 방법론):** 통계적 시계열 모델(AutoARIMA, VAR/VECM) 기반의 기대수익률 예측
- **To-Be (AI 방법론):** `Google TimesFM 2.5` 및 `Amazon Chronos-2`와 같은 대규모 사전 학습된 AI 모델의 Zero-shot 추론을 통한 기대수익률($\mu$) 예측
- **목표:** AI가 예측한 기대수익률과 역사적/GARCH 기반 공분산($\Sigma$) 행렬을 결합하여, **Sharpe Ratio를 극대화**하는 최적의 자산 가중치($w^*$)를 산출하고 이를 Gradio 대시보드로 시각화합니다.

## ✨ 주요 기능 (Key Features)

1. **Zero-Shot AI Forecasting:** Hugging Face의 `TimesFM` 또는 `Chronos` 모델을 활용한 S&P 500 주식의 미래 단기 수익률 예측 (Point & Quantile Forecasts).
2. **Dynamic Portfolio Optimization:** `cvxpy` 기반의 2차 계획법(Quadratic Programming)을 사용한 실시간 효율적 프론티어(Efficient Frontier) 및 최적 가중치 도출.
3. **Interactive Dashboard:** `Gradio`와 `Plotly`를 연동하여 자산 가격 추이, 예측 밴드(Confidence Interval), 최적화된 포트폴리오 비중을 직관적으로 시각화.

## 🛠 기술 스택 (Tech Stack)

- **AI / ML:** `transformers`, `torch`, `peft` (LoRA/QLoRA), `timesfm`, `chronos-forecasting`
- **Data Processing:** `pandas`, `numpy`, `yfinance`, `fredapi`
- **Optimization:** `cvxpy`, `scipy`
- **Visualization / UI:** `gradio`, `plotly`, `matplotlib`

## 📊 아키텍처 및 파이프라인 (Architecture)

```text
[Data Ingestion] 
  👉 S&P 500 Bulk Data (Kaggle) & Macro Indicators (FRED)
      ↓
[AI Forecasting Module]
  👉 Hugging Face TSFM (Google TimesFM 2.5 / Amazon Chronos-2)
  👉 출력: 종목별 기대수익률 (예측 μ) 및 불확실성 구간
      ↓
[Risk & Optimization Engine]
  👉 공분산 행렬 (Σ) 계산 (Sample Covariance / GARCH)
  👉 cvxpy 적용: Maximize Sharpe Ratio (제약조건: 합=1, 공매도 금지)
      ↓
[Gradio Web Interface]
  👉 사용자의 Risk Tolerance 입력에 따른 포트폴리오 추천 및 Plotly 시각화
```

---

## 📓 분석 노트북 (Notebooks)

| 번호 | 파일 | 내용 | 핵심 라이브러리 |
|------|------|------|----------------|
| 01 | `notebooks/01_chronos2_basic_inference.ipynb` | Chronos-2 Zero-Shot 추론 기초 데모 — GPU bfloat16, 5일 예측, 80% CI 시각화 | `chronos`, `torch` |
| 02 | `notebooks/02_data_overview_visualization.ipynb` | S&P 500 + 거시 데이터 탐색적 분석 (EDA) — 9개 섹션, 20개 셀 | `seaborn`, `pandas` |
| 03 | `notebooks/03_portfolio_optimization_backtest.ipynb` | TSFM 앙상블 μ + GARCH Σ + Sharpe-Max QP + Walk-Forward 백테스트 | `cvxpy`, `arch`, `plotly` |

```bash
# 노트북 실행 방법
conda activate hfm-data
jupyter notebook  # 또는 jupyter lab
# 커널 선택: hfm-data (Python 3.10)
```

## 🤖 후보 모델 카드 분석 (Candidate Model Card Analysis)

본 프로젝트에서 검토한 시계열 파운데이션 모델(TSFM) 5종을 비교 분석합니다.  
각 모델의 공식 Hugging Face 모델 카드를 기반으로 작성하였습니다.

---

### 📌 비교 요약표

| 항목 | TimesFM 1.0 | TimesFM 2.0 | Chronos (T5) | **Chronos-2** ✅ | Moirai 1.1-R |
|------|:-----------:|:-----------:|:------------:|:---------------:|:------------:|
| **개발사** | Google | Google | Amazon | Amazon | Salesforce |
| **파라미터** | 200M | 500M | 8M–710M | 120M | 14M–311M |
| **아키텍처** | Decoder-only | Decoder-only | T5 (Enc-Dec) | Encoder-only | Patch-based |
| **컨텍스트 길이** | 512 | 2048 | 512 | **8192** | 5000+ |
| **최대 예측 기간** | 128 (권장) | 128 (권장) | 64 | **1024** | 제한 없음 |
| **확률적 예측** | 실험적 | 실험적 | ✅ (샘플링) | ✅ (분위수) | ✅ |
| **공변량(Covariate) 지원** | ❌ | ❌ | ❌ | ✅ (과거·미래) | ❌ |
| **다변량(Multivariate) 지원** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **논문 발표** | ICML 2024 | 2024 | TMLR 2024 | arXiv 2025 | ICML 2024 |
| **본 프로젝트 채택** | ✅ 채택 | ⚠️ 호환 문제 | — | ✅ 채택 | — |

---

### 1. 🔵 Google TimesFM 1.0 — `google/timesfm-1.0-200m-pytorch`

**개발사:** Google Research | **파라미터:** 200M | **라이선스:** Apache-2.0

#### 모델 카드 핵심 스펙

| 항목 | 내용 |
|------|------|
| 아키텍처 | Decoder-only Transformer (패치 기반) |
| 컨텍스트 길이 | 최대 512 타임스텝 |
| 예측 방식 | 포인트 예측 중심 + 실험적 분위수(비보정) |
| 학습 데이터 | 100B 시점 규모의 실세계 시계열 데이터 |
| 주요 하이퍼파라미터 | `num_layers=20`, `model_dims=1280`, `input_patch_len=32`, `output_patch_len=128` |
| 지원 빈도 | 0(일 이하), 1(주·월), 2(분기 이상) |

#### 한국어 요약

TimesFM 1.0은 Google Research가 2024년 ICML에 발표한 **디코더 전용(Decoder-only) 시계열 파운데이션 모델**입니다. 핵심 아이디어는 텍스트 언어 모델처럼 시계열을 고정 크기 "패치(patch)"로 분할하여 자기회귀(auto-regressive) 방식으로 예측한다는 것입니다. 200M 파라미터 규모로, 추가 파인튜닝 없이 제로샷(zero-shot)으로 다양한 도메인의 시계열을 예측할 수 있습니다.

**강점:** 패치 기반 설계로 long-horizon 예측이 효율적이며, 단변량 일별 주가 예측에 적합합니다.  
**한계:** 최대 컨텍스트 512 타임스텝, 공변량 미지원, 분위수 예측 미보정(실험적).

> **본 프로젝트 채택 이유:** PyPI `timesfm==1.3.0`이 안정적으로 설치되며, `google/timesfm-1.0-200m-pytorch` 체크포인트와 완전 호환. 앙상블의 한 축으로 활용.

---

### 2. 🔵 Google TimesFM 2.0 — `google/timesfm-2.0-500m-pytorch`

**개발사:** Google Research | **파라미터:** 500M | **라이선스:** Apache-2.0

#### 모델 카드 핵심 스펙

| 항목 | 내용 |
|------|------|
| 아키텍처 | Decoder-only Transformer (패치 기반, 1.0 확장) |
| 컨텍스트 길이 | 최대 2048 타임스텝 (2048 초과도 처리 가능) |
| 예측 방식 | 포인트 예측 + 10개 분위수 헤드(비보정) |
| 학습 데이터 | TimesFM 1.0 데이터 + LOTSA 추가 데이터셋 18종 |
| 주요 하이퍼파라미터 | `num_layers=50`, `model_dims=1280`, `use_positional_embedding=False` |
| 특이 사항 | `use_positional_embedding=False` — 1.0과 반드시 구분 필요 |

#### 한국어 요약

TimesFM 2.0은 1.0 대비 파라미터를 200M → 500M으로 늘리고, 컨텍스트 창을 512 → 2048으로 확장한 **업그레이드 버전**입니다. LOTSA(Large-scale Open Time Series Archive) 데이터셋(Azure VM 트레이스, 스마트 미터, 태양광·풍력 등 18종)으로 추가 사전학습하여 전반적인 벤치마크 성능이 향상되었습니다. 특히 장기 컨텍스트가 풍부한 금융 시계열(5년 = ~1260 거래일)에 더 적합합니다.

**강점:** 1.0 대비 2× 이상 긴 컨텍스트, 더 광범위한 학습 데이터, 동일한 사용 API.  
**한계:** `timesfm==1.3.0` (PyPI 현재 최신)이 2.0 체크포인트의 `ForecastConfig` API를 지원하지 않음. 2.x 버전 PyPI 출시 후 사용 가능.

> **본 프로젝트 미채택 이유:** `timesfm` PyPI 패키지가 v1.3.0까지만 배포됨 — 2.0 가중치 로드 시 API 불일치. 호환 가능한 버전 출시 시 마이그레이션 예정.

---

### 3. 🟡 Amazon Chronos (원조) — `amazon/chronos-t5-large`

**개발사:** Amazon | **파라미터:** 8M~710M (tiny~large) | **라이선스:** Apache-2.0

#### 모델 카드 핵심 스펙

| 항목 | 내용 |
|------|------|
| 아키텍처 | T5 (Encoder-Decoder) 기반 언어 모델 |
| 어휘 처리 방식 | 시계열 → 스케일링 + 양자화(quantization) → 토큰 시퀀스 |
| 예측 방식 | 다중 미래 궤적 샘플링 → 확률 분포 추출 |
| 컨텍스트 길이 | 최대 512 타임스텝 |
| 최대 예측 기간 | 64 타임스텝 |
| 학습 데이터 | 공개 시계열 코퍼스 + GP 기반 합성 데이터 |
| 모델 라인업 | tiny(8M) / mini(20M) / small(46M) / base(200M) / large(710M) |

#### 한국어 요약

Amazon Chronos는 **시계열을 언어 모델이 이해하는 '토큰'으로 변환**하여 예측하는 혁신적인 접근법을 채택했습니다. 구체적으로, 시계열 값을 스케일링·양자화하여 4096개의 어휘 토큰으로 매핑하고 T5 언어 모델로 학습합니다. 추론 시에는 다수의 미래 궤적(trajectory)을 샘플링하여 확률적 예측 분포를 생성합니다. TMLR 2024에 발표되었으며, Tiny(8M)부터 Large(710M)까지 5가지 크기를 제공합니다.

**강점:** 언어 모델 패러다임을 시계열에 적용한 선구자적 모델; 잘 보정된 확률 예측.  
**한계:** 단변량 예측만 지원, 공변량 미지원, 최대 예측 기간 64스텝(단기), Chronos-2에 성능·기능 면에서 뒤처짐.

> **본 프로젝트 미채택 이유:** Chronos-2가 동일 라이브러리(`chronos-forecasting`) 내에서 공변량 지원·더 긴 예측 기간 등 상위 호환 기능 제공.

---

### 4. 🟠 Amazon Chronos-2 — `amazon/chronos-2` ✅ **채택**

**개발사:** Amazon | **파라미터:** 120M | **라이선스:** Apache-2.0

#### 모델 카드 핵심 스펙

| 항목 | 내용 |
|------|------|
| 아키텍처 | T5 Encoder 기반 (인코더 전용) |
| 핵심 메커니즘 | 그룹 어텐션(Group Attention) — 관련 시계열 간 in-context learning |
| 컨텍스트 길이 | 최대 **8192** 타임스텝 |
| 최대 예측 기간 | 최대 **1024** 타임스텝 |
| 공변량 지원 | ✅ 과거 공변량 + ✅ 미래 공변량 (네이티브) |
| 다변량 지원 | ✅ 크로스-시리즈 학습 가능 |
| 추론 속도 | A10G GPU 기준 **300개 시계열/초 이상** |
| 벤치마크 성능 | fev-bench / GIFT-Eval / Chronos Benchmark II — 공개 모델 SOTA |
| 학습 데이터 | Chronos Datasets + GIFT-Eval Pretrain + 합성 데이터 (단변량·다변량) |

#### 한국어 요약

Chronos-2는 Amazon이 2025년에 발표한 **범용 시계열 파운데이션 모델**로, 원조 Chronos의 한계였던 단변량/단기 예측을 완전히 극복했습니다. 가장 큰 차별점은 세 가지입니다:

1. **공변량 네이티브 지원**: DGS10(10년물 국채 금리), VIXCLS(변동성 지수) 등 거시 변수를 `past_covariates`로 직접 입력하여 시장 맥락을 인식한 예측이 가능합니다.
2. **8192 타임스텝 컨텍스트**: 약 32년치 일별 데이터를 한 번에 처리할 수 있어 장기 패턴 포착에 유리합니다.
3. **그룹 어텐션 메커니즘**: 여러 종목을 동시에 입력하면 종목 간 상관관계를 학습하여 포트폴리오 예측에 시너지를 냅니다.

`Chronos2Pipeline`을 통해 `{"target": arr, "past_covariates": {"DGS10": arr, "VIXCLS": arr}}` 형식으로 거시 변수를 주입하며, 출력은 9개 분위수(q10~q90) 확률 예측입니다.

> **본 프로젝트 채택 이유:** 거시 공변량(DGS10, VIXCLS) 네이티브 지원으로 금융 예측의 컨텍스트를 풍부하게 구성; 분위수 예측으로 불확실성 구간(80% CI) 시각화 가능; TimesFM 1.0과의 앙상블로 예측 편향 감소.

---

### 5. ⚪ Salesforce Moirai 1.1-R — `Salesforce/moirai-1.1-R-large`

**개발사:** Salesforce AI Research | **파라미터:** 14M~311M (small~large) | **라이선스:** CC BY-NC 4.0

#### 모델 카드 핵심 스펙

| 항목 | 내용 |
|------|------|
| 아키텍처 | Patch-based Universal Transformer |
| 특화 영역 | 저빈도(연간·분기) 데이터에서 탁월한 성능 |
| 성능 향상 | Moirai 1.0 대비 저빈도 NMAE **~20% 개선** |
| 다변량 지원 | ✅ |
| 벤치마크 | Monash 레포지토리 40개 데이터셋 기준 |
| 라이선스 | CC BY-NC 4.0 (비상업적 연구 목적) |

#### 한국어 요약

Moirai는 Salesforce AI Research가 개발한 **패치 기반 범용 시계열 파운데이션 모델**로, ICML 2024에서 발표되었습니다. 1.1-R 버전은 1.0 대비 연간·분기 데이터에서 NMAE(정규화 평균 절대 오차)를 약 20% 개선한 업데이트 버전입니다. Monash 벤치마크 40개 데이터셋에서 검증되었으며, 다변량 시계열을 패치 단위로 처리하는 독자적인 설계를 채택합니다.

**강점:** 저빈도 장기 예측에 강점, 다변량 지원, 패치 기반 효율적 추론.  
**한계:** CC BY-NC 라이선스로 상업적 활용 제한; PyPI 공식 패키지 미제공(별도 설치 필요); 고빈도 일별 주가 예측에서는 Chronos-2 대비 상대적 약점.

> **본 프로젝트 미채택 이유:** 라이선스 제약 + 일별 고빈도 예측에서 Chronos-2 대비 우위 없음. 분기·연간 거시 지표 예측이 추가될 경우 재검토.

---

### 🎯 최종 모델 선정 근거

```
본 프로젝트의 목표: S&P 500 종목의 일별 단기 수익률 예측 (최대 90거래일)
사용 데이터: 5년치 일별 OHLCV + FRED 거시 변수 (DGS10, VIXCLS, UNRATE)
```

| 선정 기준 | TimesFM 1.0 | Chronos-2 |
|-----------|:-----------:|:---------:|
| 설치 안정성 (PyPI) | ✅ | ✅ |
| 일별 주가 예측 적합성 | ✅ | ✅ |
| 거시 공변량 활용 | ❌ | ✅ |
| 확률 예측 (80% CI) | 실험적 | ✅ 보정됨 |
| 추론 속도 (CPU/GPU) | 중간 | 빠름 |
| **채택 결정** | **앙상블 구성요소** | **주력 + 앙상블** |

두 모델을 **가중 앙상블(weighted ensemble)**으로 결합하여 단일 모델의 편향을 상쇄하고, Chronos-2의 거시 공변량 정보와 TimesFM의 패치 기반 패턴 포착 능력을 동시에 활용합니다. Gradio UI의 **Chronos-2 Weight 슬라이더**로 실시간 가중치 조정이 가능합니다.

---

## ⚙️ 시작하기 (Getting Started)

### 사전 요구 사항 (Prerequisites)

- [Miniconda](https://docs.anaconda.com/miniconda/) 또는 Anaconda
- NVIDIA GPU (권장, CUDA 12.x) — CPU 모드도 지원되나 추론 속도 저하
- Hugging Face 계정 및 Access Token
- (선택) FRED API Key, Kaggle API Key

### 1. Conda 환경 설정 (Environment Setup)

```bash
git clone https://github.com/your-username/ai-portfolio-optimizer-tsfm.git
cd ai-portfolio-optimizer-tsfm

# conda 초기화 (최초 1회)
conda init zsh   # 또는 conda init bash

# 환경 생성 (environment.yml 기반 — Python 3.10, 전체 의존성 포함)
conda env create -f environment.yml

# 환경 활성화
conda activate hfm-data

# Jupyter 커널 등록 (최초 1회)
python -m ipykernel install --user --name hfm-data --display-name "Python (hfm-data)"
```

> **⚠️ Python 버전 주의:** `timesfm` 및 `chronos-forecasting`은 **Python 3.10**에서만 안정적으로 동작합니다.
> Python 3.11+ 또는 Linuxbrew/Homebrew Python 사용 시 `jaxlib` 호환 오류가 발생할 수 있습니다.
> 반드시 `hfm-data` Conda 환경에서 실행하세요.

### 2. 환경 변수 설정 (Environment Variables)

```bash
# .env 파일 생성 (.env.example 참고)
cp .env.example .env
```

`.env` 파일에 아래 키 입력:

```env
HF_TOKEN=your_huggingface_token
MODEL_REQUIRE_TOKEN=1
MODEL_LOCAL_ONLY=0
HF_HUB_OFFLINE=0
HF_HOME=
HF_HUB_ETAG_TIMEOUT=30
HF_HUB_DOWNLOAD_TIMEOUT=60
FRED_API_KEY=your_fred_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key
```

모델 로딩 안정화용 변수:

- `MODEL_REQUIRE_TOKEN=1`  
  `HF_TOKEN`이 없으면 모델 로딩을 중단하고 명확한 오류를 반환합니다.
- `MODEL_LOCAL_ONLY=1`  
  Hugging Face 원격 조회를 하지 않고 로컬 캐시만 사용합니다.
- `HF_HUB_OFFLINE=1`  
  허깅페이스 허브 오프라인 모드를 활성화합니다.
- `HF_HUB_ETAG_TIMEOUT`, `HF_HUB_DOWNLOAD_TIMEOUT`  
  메타데이터 조회/다운로드 타임아웃(초)입니다.

### 3. 모델 사전 다운로드 (Preload Models, 권장)

```bash
conda activate hfm-data
python scripts/preload_models.py --all
```

개별 모델만 받을 수도 있습니다:

```bash
python scripts/preload_models.py --model chronos2
python scripts/preload_models.py --model timesfm1
```

### 4. 데이터셋 구축 (Dataset Build)

```bash
conda activate hfm-data
python scripts/build_dataset.py --top-n 100 --period 5y
# 출력: data/sp500_macro_master.csv (99 tickers, 17 cols, ~123,505 rows)
```

### 5. 앱 실행 (Run the App)

```bash
conda activate hfm-data
python app.py
# → http://127.0.0.1:7860
```

### 6. 노트북 실행 (Run Notebooks)

```bash
conda activate hfm-data
jupyter notebook  # 또는 jupyter lab
```

노트북 우상단에서 커널을 **`Python (hfm-data)`** 또는 **`hfm-data`** 로 선택 후 실행하세요.

커널이 보이지 않는 경우:

```bash
conda activate hfm-data
python -m ipykernel install --user --name hfm-data --display-name "Python (hfm-data)"
```

### 7. 패키지 확인 (Verify Installation)

```bash
conda activate hfm-data
python -c "import chronos, timesfm, cvxpy, gradio, torch, arch; print('All packages OK')"
# → All packages OK
```

### 모델 로딩 트러블슈팅

- `ReadTimeoutError ... huggingface.co`  
  네트워크 타임아웃입니다. `python scripts/preload_models.py --all`을 먼저 실행하고, 필요 시 `HF_HUB_ETAG_TIMEOUT`, `HF_HUB_DOWNLOAD_TIMEOUT` 값을 늘리세요.
- `401` 또는 `403`  
  인증/권한 문제입니다. `HF_TOKEN` 재발급 또는 권한 확인 후 재시도하세요.
- `not found in cache` + `MODEL_LOCAL_ONLY=1`  
  로컬 전용 모드에서 캐시가 없습니다. 온라인 상태에서 사전 다운로드 후 다시 실행하세요.

## 📅 프로젝트 진행 로드맵 (4-Day Sprint)

본 프로젝트는 패스트캠퍼스 'Hugging Face 모델 체험 및 활용' 미니 프로젝트 일정에 맞춰 진행되었습니다.

- **Day 1:** TSFM 모델 선정 (`TimesFM`, `Chronos`) 및 S&P 500 & 거시 변수 데이터셋 구축
- **Day 2:** Hugging Face 모델 로드 및 Zero-shot 기본 추론(Inference) 실행
- **Day 3:** 기존 cvxpy 최적화 엔진과 AI 예측값 결합, `Gradio` 대시보드 UI 구현
- **Day 4:** 최종 백테스트 결과 분석, 보고서 작성 및 발표 시연

## 📚 참고 문헌 (References)

- Das, A., et al. (2024). "A decoder-only foundation model for time-series forecasting" (TimesFM). *ICML*.
- Ansari, A. F., et al. (2024). "Chronos: Learning the Language of Time Series." *arXiv preprint*.
- Rahimikia, E., et al. (2025). "Re(Visiting) Time Series Foundation Models in Finance." *arXiv*.
- Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*.

## 👨‍💻 작성자 (Author)

- **이름:** Jaehyun Park
- **소속:** 소강대학교 경제학과
- **과정:** 패스트캠퍼스 — Hugging Face 모델 체험 및 활용

---
*이 프로젝트는 교육 및 연구 목적으로 작성되었으며, 실제 투자 수익을 보장하지 않습니다.*
