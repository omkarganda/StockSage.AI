# StockSage.AI – Quick-Start Guide  
*(Last updated: 2025-07-21)*

---

## 🚀 Overview
StockSage.AI is an **end-to-end quantitative research platform** that brings together
market data, macro-economic indicators, news sentiment (FinBERT **and** LLM-powered), advanced feature engineering, several forecasting model families, back-testing and a production-ready FastAPI + Streamlit front-end.

The codebase is modular ‑ each layer can be used in isolation or as part of the complete research pipeline.

---

## 🏗️ Architecture at a Glance
| Layer | Key Modules / Scripts | Purpose |
|-------|----------------------|---------|
| **Configuration** | `src/config.py` | Central config & API-keys |
| **Data Ingestion** | `src/data/` | Market, economic & sentiment collectors + caching |
| **Feature Engineering** | `src/features/` | Technical indicators, micro-structure & LLM sentiment |
| **Validation** | `src/data/validation.py` | Schema & business-logic checks |
| **Model Training** | `scripts/train_models.py` + `src/models/` | Baseline, classical & deep-learning models |
| **Evaluation** | `scripts/evaluate_models.py`, `scripts/backtest.py` | Metrics, equity curves, statistical tests |
| **Monitoring** | `scripts/continuous_validation.py`, GH-Action | Daily model & data drift check |
| **API** | `src/app/api.py` | FastAPI for inference & explainability |
| **Dashboard** | `src/app/dashboard.py` | Streamlit front-end |

---

## 📋 Prerequisites
1. **Python ≥ 3.10**  (tested on 3.10 & 3.12).
2. **Install dependencies** (versions are unpinned – rely on the lock-file of your env manager if you need reproducibility):
   ```bash
   pip install -r requirements.txt
   ```
3. **Environment variables** – copy & edit the template:
   ```bash
   cp .env.template .env  # create if the template does not yet exist
   # Add your FREE API keys
   # FRED_API_KEY=xxxx
   # NEWS_API_KEY=xxxx
   # OPENAI_API_KEY=xxxx  # optional but unlocks LLM features
   ```
4. *(Optional)* **Create a dedicated virtual-env**:
   ```bash
   python -m venv .venv && source .venv/bin/activate     # Linux/macOS
   # OR
   .venv\Scripts\Activate.ps1                             # Windows PowerShell
   ```

---

## 🚦 Smoke Test
Run the integrated pipeline test – it downloads 7 days of AAPL data, joins all data sources and runs validation.
```bash
python scripts/test_data_pipeline.py
```
If all checks pass you are ready to explore the full stack.

---

## 🛠️ Common Tasks
| Task | Command |
|------|---------|
| **Start API (localhost:8000)** | `uvicorn src.app.api:app --reload` |
| **Open docs** | visit `http://127.0.0.1:8000/docs` |
| **Launch Dashboard** | `streamlit run src/app/dashboard.py` |
| **Train full model suite** | `python scripts/train_models.py --symbols AAPL MSFT --start 2020-01-01 --end 2024-12-31` |
| **Back-test a strategy** | `python scripts/backtest.py --symbol AAPL --start 2023-01-01 --end 2023-12-31` |
| **Evaluate trained models** | `python scripts/evaluate_models.py --symbol AAPL` |
| **Run daily validation (CI safe)** | `python scripts/continuous_validation.py` |

---

## 🔍 Module Highlights
### 1&nbsp;·&nbsp;Data Collectors (`src/data/…`)
* **Market** – yfinance wrapper with retry & caching.
* **Economic** – FRED integration, yield-curve helper, recession dates.
* **Sentiment** – NewsAPI, FinBERT **and** GPT-4o sentiment scoring with daily aggregation.

### 2&nbsp;·&nbsp;Feature Engineering (`src/features/…`)
* 40+ technical indicators (MA, RSI, MACD, Bollinger, …)
* Volume & micro-structure stats
* Generative-AI **LLM sentiment**: article-level scores, daily summary & scenario generation.

### 3&nbsp;·&nbsp;Models (`src/models/…`)
* Baseline linear & tree ensembles
* Statistical forecasting (ETS, ARIMA, Prophet, etc.)
* Deep learning (LSTM-Attention, generic Transformer)
* Foundation-model adapters (TimesFM, TTM) – optional heavy dependencies

Training orchestration lives in `scripts/train_models.py` and supports quick-tests & full runs.

### 4&nbsp;·&nbsp;Back-testing (`src/backtesting/…` + `scripts/backtest.py`)
Vectorised engine supports long-only, shorting, parameter sweeps and outputs full trade-logs & metrics.

### 5&nbsp;·&nbsp;Serving Layer
* **FastAPI** – single & batch prediction, sentiment, scenario generation, explainability.
* **Streamlit** – rich dashboard with price chart, technicals, LLM sentiment overlay & AI scenarios.

---

## 🔧 Usage Snippets
```python
from src.data.download_market import MarketDataDownloader
from src.features.indicators import add_all_technical_indicators

df = MarketDataDownloader().download_stock_data("AAPL", "2023-01-01")
df = add_all_technical_indicators(df)
print(df.tail())
```

```python
from src.features.generative_sentiment import generate_market_scenarios
print(generate_market_scenarios("AAPL", horizon_days=30))
```

---

## 🐛 Troubleshooting
| Issue | Fix |
|-------|-----|
| Missing C++ build tools (Prophet / cmdstanpy) | `conda install -c conda-forge libpython m2-base` *or* use Docker image |
| `OPENAI_API_KEY` not set | LLM features silently fallback to neutral – set key to enable full functionality |
| GPU not utilised | Ensure CUDA version matches the `torch` build installed (`pip install torch --index-url …`) |

---

## 🗺️ Roadmap
- Hyper-parameter tuning with **Optuna** & Ray
- Broker integration for paper-trading
- Auto-generated PDF reports
- Docker & Helm charts for easy deployment

---

Happy researching 🎉
