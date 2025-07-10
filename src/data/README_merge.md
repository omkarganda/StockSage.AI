# Data Merging & Alignment Module

## Overview

The `merge.py` module is the cornerstone of StockSage.AI's data preprocessing pipeline. It handles the complex task of combining multiple data sources with different frequencies and characteristics into a unified, analysis-ready dataset.

## Key Features

### 1. Multi-Source Data Integration
- **Market Data**: Daily OHLCV data from yfinance
- **Economic Indicators**: Monthly/quarterly data from FRED (GDP, unemployment, etc.)
- **Sentiment Scores**: Irregular time-series from news analysis

### 2. Intelligent Time Alignment
- Handles different data frequencies (daily, weekly, monthly, quarterly)
- Smart forward-filling for economic indicators
- Sentiment aggregation with multiple methods

### 3. Feature Engineering
- Technical indicators (RSI, MACD, Bollinger Bands)
- Market regime detection
- Temporal features (day of week, month effects)
- Data quality indicators

## Main Functions

### `merge_market_economic_data()`
Merges daily market data with economic indicators that may have different frequencies.

```python
merged_df = merge_market_economic_data(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    market_data=market_df,        # DataFrame with OHLCV data
    economic_data={               # Dict of economic indicators
        'GDP': gdp_df,
        'UNEMPLOYMENT': unemployment_df,
        'VIX': vix_df
    },
    alignment='inner',            # How to join data
    fill_method='ffill'          # How to handle missing values
)
```

**Key Parameters:**
- `alignment`: 'inner' (only overlapping dates), 'outer' (all dates), 'left', 'right'
- `fill_method`: 'ffill', 'bfill', 'interpolate', or None

### `align_sentiment_with_market_data()`
Aligns irregular sentiment data with regular market data.

```python
aligned_df = align_sentiment_with_market_data(
    market_df=market_data,
    sentiment_df=sentiment_scores,
    sentiment_window='1D',           # Aggregation window
    aggregation_method='weighted_mean',  # How to combine sentiments
    weight_decay=0.8                # For exponential weighting
)
```

**Aggregation Methods:**
- `'mean'`: Simple average of sentiment scores
- `'weighted_mean'`: Weight by confidence scores
- `'ewm'`: Exponentially weighted (recent news matters more)
- `'last'`: Use most recent sentiment

### `create_unified_dataset()`
The main orchestration function that creates a complete dataset.

```python
unified_df = create_unified_dataset(
    symbol='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    market_data=market_df,
    economic_data=economic_dict,
    sentiment_data=sentiment_df,
    include_technical_indicators=True,  # Add TA features
    include_market_regime=True,         # Add volatility regimes
    handle_missing='interpolate'        # Missing value strategy
)
```

## Feature Categories

The unified dataset includes:

### 1. Market Features
- OHLCV data
- Returns (simple and log)
- Price ratios

### 2. Economic Features
- Original indicator values
- Days since last update
- Forward-filled values

### 3. Sentiment Features
- Aggregated sentiment scores
- Sentiment momentum (3d, 7d)
- Sentiment volatility
- Age of sentiment data

### 4. Technical Indicators
- Moving averages (SMA, EMA)
- MACD and signal line
- RSI (Relative Strength Index)
- Bollinger Bands
- Volume indicators

### 5. Market Regime Features
- Volatility (20d, 60d)
- Volatility ratio
- Trend strength
- High volatility flag

### 6. Temporal Features
- Day of week, month, quarter
- Trading day flags (Monday, Friday)
- Month/quarter start/end flags

### 7. Lag Features
- Configurable lags for key variables
- Default: 1, 2, 3, 5, 10 day lags

## Usage Example

```python
from datetime import datetime, timedelta
import pandas as pd
from src.data.merge import create_unified_dataset

# Define time period
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# Assume we have loaded our data
market_data = load_market_data('AAPL')
economic_data = {
    'GDP': load_fred_data('GDP'),
    'UNEMPLOYMENT': load_fred_data('UNRATE'),
    'VIX': load_market_data('^VIX')
}
sentiment_data = load_sentiment_scores('AAPL')

# Create unified dataset
unified_df = create_unified_dataset(
    symbol='AAPL',
    start_date=start_date,
    end_date=end_date,
    market_data=market_data,
    economic_data=economic_data,
    sentiment_data=sentiment_data,
    include_technical_indicators=True,
    include_market_regime=True,
    handle_missing='interpolate'
)

print(f"Dataset shape: {unified_df.shape}")
print(f"Features: {unified_df.columns.tolist()}")
```

## Data Quality Handling

### Missing Value Strategies
1. **'drop'**: Remove rows with any missing values
2. **'interpolate'**: Linear interpolation (max 5 days)
3. **'forward_fill'**: Propagate last valid value (max 5 days)
4. **'mean'**: Fill with column mean

### Automatic Quality Checks
- Removes duplicate timestamps
- Handles infinite values
- Drops columns with >50% missing data
- Adds missing data indicators

## Best Practices

1. **Data Frequency**: Ensure your market data is at business day frequency
2. **Economic Data**: Monthly/quarterly data will be forward-filled automatically
3. **Sentiment Data**: More frequent updates (hourly/daily) work better
4. **Memory Usage**: For large datasets, consider chunking by date ranges
5. **Feature Selection**: Not all features may be relevant - select based on your model

## Performance Considerations

- The module uses pandas for compatibility but can be memory-intensive
- For production, consider using Polars or chunked processing
- Technical indicators add ~25-30 features
- Lag features can significantly increase dataset size

## Testing

Run the test script to verify functionality:

```bash
python scripts/test_merge.py
```

This will create sample data and demonstrate all merge functions.