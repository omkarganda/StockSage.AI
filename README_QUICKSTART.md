# StockSage.AI - Quick Start Guide

## 🚀 What We've Built

We've successfully created the core data pipeline components for StockSage.AI:

### 1. **Central Configuration** (`src/config.py`)
- ✅ Manages all API keys from environment variables
- ✅ Central settings for data collection
- ✅ Path management and cache configuration
- ✅ Automatic directory creation

### 2. **Market Data Module** (`src/data/download_market.py`)
- ✅ yfinance wrapper with robust error handling
- ✅ Intelligent caching to reduce API calls
- ✅ Parallel downloads for multiple stocks
- ✅ Real-time price fetching
- ✅ Company information retrieval

### 3. **Economic Data Module** (`src/data/download_economic.py`)
- ✅ FRED API integration
- ✅ Downloads key economic indicators (Treasury rates, VIX, etc.)
- ✅ Yield curve analysis
- ✅ Recession date tracking
- ✅ Caching for efficiency

### 4. **Sentiment Analysis Module** (`src/data/download_sentiment.py`)
- ✅ NewsAPI integration for news collection
- ✅ FinBERT sentiment analysis (when transformers installed)
- ✅ Daily sentiment aggregation
- ✅ Multi-source news deduplication
- ✅ Weighted sentiment scoring

### 5. **Test Script** (`scripts/test_data_pipeline.py`)
- ✅ Complete pipeline test for AAPL
- ✅ Collects 1 week of data from all sources
- ✅ Merges all data into unified DataFrame
- ✅ Comprehensive logging and error handling

## 📋 Prerequisites

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get API Keys (FREE):**
   - **FRED API**: https://fred.stlouisfed.org/docs/api/ (FREE)
   - **NewsAPI**: https://newsapi.org/ (FREE tier: 500 requests/day)
   - **OpenAI**: https://platform.openai.com/api-keys (Optional, for GPT features)

3. **Setup Environment:**
   ```bash
   cp .env.template .env
   # Edit .env and add your API keys
   ```

## 🏃‍♂️ Running the Test

Test the complete data pipeline:

```bash
python scripts/test_data_pipeline.py
```

This will:
1. ✅ Download 1 week of AAPL market data
2. ✅ Fetch economic indicators (if FRED key available)
3. ✅ Collect and analyze news sentiment (if NewsAPI key available)
4. ✅ Merge all data into a single CSV file
5. ✅ Save output to `data/raw/`

## 📊 Expected Output

```
StockSage.AI Data Pipeline Test
Testing 1 week of AAPL data collection
======================================================================

Checking API Keys:
  ✓ fred
  ✓ news_api
  ✓ openai
  ...

==================================================
Testing Market Data Collection
==================================================
✓ Downloaded 5 days of AAPL market data
  Date range: 2024-01-15 to 2024-01-22
  Columns: ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits', 'Ticker']
  Latest close: $195.89

✓ AAPL Company Info:
  Name: Apple Inc.
  Sector: Technology
  Market Cap: $3,045,678,000,000

✓ Real-time Price: $195.92
...
```

## 🔧 Usage Examples

### Download Market Data
```python
from src.data.download_market import MarketDataDownloader

downloader = MarketDataDownloader()
aapl_data = downloader.download_stock_data("AAPL", start_date="2024-01-01")
```

### Get Economic Indicators
```python
from src.data.download_economic import EconomicDataDownloader

eco_downloader = EconomicDataDownloader()
treasury_rate = eco_downloader.download_indicator("DGS10")  # 10-year Treasury
```

### Analyze Sentiment
```python
from src.data.download_sentiment import SentimentDataProcessor

processor = SentimentDataProcessor()
sentiment = processor.get_daily_sentiment(
    ticker="AAPL",
    company_name="Apple Inc",
    start_date="2024-01-01",
    end_date="2024-01-07"
)
```

## 🐛 Troubleshooting

1. **Import Errors**: Make sure you've installed all requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key Errors**: Check your `.env` file has the correct keys

3. **No Data**: Some APIs have rate limits or require paid tiers for historical data

4. **FinBERT Not Available**: Install transformers and torch:
   ```bash
   pip install transformers torch
   ```

## 🎯 Next Steps

1. **Add More Data Sources**: Integrate Alpha Vantage, Finnhub
2. **Build Features**: Create technical indicators module
3. **Add Models**: Implement forecasting models
4. **Create Dashboard**: Build Streamlit visualization app
5. **Setup Backtesting**: Implement evaluation framework

## 📁 Output Files

All data is saved to `data/raw/`:
- `market_data_YYYYMMDD.csv` - Stock price data
- `economic_data_YYYYMMDD.csv` - Economic indicators
- `sentiment_data_YYYYMMDD.csv` - Sentiment analysis results
- `aapl_merged_data_YYYYMMDD_HHMMSS.csv` - Combined dataset

Happy trading! 🚀📈