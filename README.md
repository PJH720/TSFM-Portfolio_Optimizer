

# 📈 AI-Driven Dynamic Portfolio Optimizer (TSFM Edition)

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Transformers-orange.svg)
![Gradio](https://img.shields.io/badge/UI-Gradio-red.svg)
![Optimization](https://img.shields.io/badge/Optimization-cvxpy-green.svg)

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

## ⚙️ 시작하기 (Getting Started)

### 1. 사전 요구 사항 (Prerequisites)
- Python 3.10 이상
- Hugging Face 계정 및 Access Token
- (선택) FRED API Key

### 2. 패키지 설치 (Installation)
```bash
git clone https://github.com/your-username/ai-portfolio-optimizer-tsfm.git
cd ai-portfolio-optimizer-tsfm

# 필수 라이브러리 설치
pip install -r requirements.txt

# (필요시) TimesFM 또는 Chronos 설치
pip install timesfm
pip install chronos-forecasting
```

### 3. 환경 변수 설정 (Environment Variables)
`.env` 파일을 생성하고 다음 키를 입력합니다.
```env
HF_TOKEN=your_huggingface_token
FRED_API_KEY=your_fred_api_key
```

### 4. 앱 실행 (Running the App)
```bash
python app.py
# Running on local URL:  http://127.0.0.1:7860
```

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
- **이름:** [본인 이름]
- **소속:** [본인 소속 또는 과정명 기입]
- **Contact:** [이메일 주소]

---
*이 프로젝트는 교육 및 연구 목적으로 작성되었으며, 실제 투자 수익을 보장하지 않습니다.*
이름: Jaehyun Park
