# StockStage.AI ─ Project Structure

Below is a recommended, production‑ready layout for the **StockStage.AI** repository.  It separates concerns clearly—data, features, models, evaluation, and deployment—making collaboration, testing, and CI/CD straightforward.

```text
StockStage.AI/
├── data/                       # Local data storage (git‑ignored)
│   ├── raw/                    # Unmodified downloads from APIs
│   ├── processed/              # Cleaned & feature‑rich datasets
│   └── external/               # Any 3rd‑party data snapshots / benchmarks
│
├── notebooks/                  # Exploratory analysis & prototyping
│   ├── 01_exploration.ipynb
│   └── 02_modeling.ipynb
│
├── src/                        # Core Python package (pip‑installable)
│   ├── __init__.py
│   ├── config.py               # Global paths, API keys (reads .env)
│   │
│   ├── data/                   # Data ingestion & alignment
│   │   ├── download_market.py          # yfinance wrappers
│   │   ├── download_economic.py        # FRED / other macro APIs
│   │   ├── download_sentiment.py       # FinBERT-based sentiment data collection
│   │   └── merge.py                    # Combine & time‑sync datasets
│   │
│   ├── features/               # Feature engineering & sentiments
│   │   ├── indicators.py               # TA indicators, moving avgs, RSI …
│   │   └── sentiment.py                # FinBERT model inference + LangChain helpers
│   │
│   ├── models/                 # Forecasting & generative models
│   │   ├── neuralforecast_model.py     # NEW: Replace prophet_model.py  
│   │   ├── sktime_model.py            # NEW: Replace darts_model.py
│   │   ├── lightning_model.py         # Keep existing
│   │   └── openai_scenario.py         # Keep existing
│   │
│   ├── evaluation/             # Backtesting & metrics
│   │   ├── backtester.py               # Walk‑forward & cross‑val engine
│   │   └── metrics.py                  # MAPE, RMSE, Sharpe‑ratio, drawdowns
│   │
│   ├── visualization/          # Plotting utilities (Plotly)
│   │   └── plots.py
│   │
│   └── app/                    # Production interface
│       ├── dashboard.py                # Streamlit/Dash interactive UI
│       └── api.py                     # FastAPI endpoints for predictions
│
├── scripts/                    # CLI entry‑points for automation
│   ├── train.py                # Orchestrates full training pipeline
│   ├── evaluate.py             # Generates backtesting reports
│   └── run_dashboard.sh        # Convenience launcher for the web app
│
├── tests/                      # PyTest suites (unit + integration)
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── config/                     # YAML/JSON configs (hyper‑params, logging)
│   ├── model_params.yaml         # Keep existing
│   ├── logging.yaml              # Keep existing  
│   └── experiment_config.yaml    # NEW: W&B experiment configuration
│
├── .env.template               # API keys & secrets (copy to .env)
├── requirements.txt              # Add: neuralforecast, sktime, finbert, polars, wandb
├── README.md                   # Project overview & quick‑start
└── LICENSE
```

---

## File & Directory Descriptions

| Path              | Purpose                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ |
| **data/**         | Keeps the dataset hierarchy tidy. *Never commit raw data*—add `data/` to `.gitignore`.                             |
| **notebooks/**    | Jupyter/Colab notebooks for initial EDA, quick experiments, and presentations.                                     |
| **src/**          | Installable Python package (`pip install -e .`). All production code lives here, logically split into sub‑modules. |
| **scripts/**      | One‑shot or scheduled command‑line utilities—ideal for cron/Airflow jobs.                                          |
| **tests/**        | Ensures every critical function is unit‑ or integration‑tested before deployment.                                  |
| **app/**          | Real‑time serving layer: interactive dashboard and REST endpoints.                                                 |
| **config/**       | Central location for hyper‑parameters, experiment tracking settings, and logging formats.                          |
| **.env.template** | Shows expected environment variables without exposing secrets—copy/rename to `.env`.                               |

---

## Getting Started (README snippet)

```bash
# clone repo & install
$ git clone https://github.com/omkarganda/StockStage.AI.git && cd StockStage.AI
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# download default datasets & preprocess
$ python scripts/train.py --stage fetch

# full training run (incl. backtesting)
$ python scripts/train.py --stage full

# launch dashboard
$ ./scripts/run_dashboard.sh
```

Feel free to adapt naming conventions, add Docker, CI workflows, or Terraform for cloud infra as the project matures.

## **1. First, create the `.env.template` file:**

You'll need to manually create this file in your root directory:

```bash
# .env.template

# StockStage.AI Environment Variables Template
# Copy this file to .env and fill in your actual API keys

# =============================================================================
# ESSENTIAL API KEYS (Required for core functionality)
# =============================================================================

# Economic Data - Federal Reserve Economic Data (FREE)
# Get your key at: https://fred.stlouisfed.org/docs/api/
FRED_API_KEY=your_fred_api_key_here

# News Sentiment Data - NewsAPI (FREE tier: 500 requests/day)
# Get your key at: https://newsapi.org/
NEWS_API_KEY=your_news_api_key_here

# OpenAI API for GPT-powered insights and explainability
# Get your key at: https://platform.openai.com/api-keys
OPENAI_API_KEY=your_openai_api_key_here

# =============================================================================
# OPTIONAL API KEYS (For enhanced functionality)
# =============================================================================

# Alternative Market Data - Alpha Vantage (FREE tier available)
# Get your key at: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# Financial News - Finnhub (FREE tier available)  
# Get your key at: https://finnhub.io/register
FINNHUB_API_KEY=your_finnhub_key_here

# Weights & Biases for experiment tracking (FREE account available)
# Get your key at: https://wandb.ai/authorize
WANDB_API_KEY=your_wandb_api_key_here

# Hugging Face API for advanced NLP models (FREE tier available)
# Get your key at: https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Environment settings
ENVIRONMENT=development
DEBUG=true
```

## **2. Now, create the exploration notebook:**

Create `notebooks/01_exploration.ipynb` with this content:

```python
<code_block_to_apply_changes_from>
```

## **🎯 Steps to Test:**

1. **Create the files manually:**
   - Copy the `.env.template` content to a new file called `.env.template`
   - Copy the notebook content to `notebooks/01_exploration.ipynb`

2. **Get your API keys:**
   - **FRED**: https://fred.stlouisfed.org/docs/api/ (FREE)
   - **NewsAPI**: https://newsapi.org/ (FREE tier: 500 requests/day)
   - **OpenAI**: https://platform.openai.com/api-keys (Paid, but cheap for testing)

3. **Copy .env.template to .env and add your keys:**
   ```bash
   cp .env.template .env
   # Edit .env with your actual API keys
   ```

4. **Install additional packages:**
   ```bash
   pip install newsapi-python python-dotenv
   ```

5. **Run the notebook:**
   ```bash
   jupyter notebook notebooks/01_exploration.ipynb
   ```

This notebook will test all your data sources and show you exactly what's working and what needs API keys! 🚀

