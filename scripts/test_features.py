#!/usr/bin/env python3
"""
Test script for Feature Engineering Pipeline

This script demonstrates the usage of the technical indicators and sentiment features modules.
It creates sample data and shows how to apply all the feature engineering functions.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the src directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.indicators import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_volume_indicators,
    calculate_price_momentum,
    add_all_technical_indicators
)

from features.sentiment import (
    aggregate_daily_sentiment,
    calculate_sentiment_momentum,
    calculate_news_volume_indicators,
    add_all_sentiment_features
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_market_data(start_date: str = "2023-01-01", end_date: str = "2024-01-01", symbol: str = "AAPL"):
    """Create sample OHLCV market data for testing."""
    
    # Create date range (business days only)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)
    
    # Generate realistic price data using random walk with drift
    np.random.seed(42)  # For reproducible results
    
    # Starting price
    initial_price = 150.0
    
    # Generate returns with some autocorrelation and volatility clustering
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns with slight upward drift
    returns[0] = 0  # First day return is 0
    
    # Add some volatility clustering
    for i in range(1, len(returns)):
        if abs(returns[i-1]) > 0.03:  # If previous day was high volatility
            returns[i] += np.random.normal(0, 0.01)  # Increase current day volatility
    
    # Calculate cumulative prices
    price_levels = initial_price * np.cumprod(1 + returns)
    
    # Generate OHLC data
    highs = price_levels * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = price_levels * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    
    # Ensure OHLC consistency
    opens = np.roll(price_levels, 1)
    opens[0] = initial_price
    closes = price_levels
    
    # Generate volume (higher volume on larger price movements)
    base_volume = 50_000_000
    volume_multiplier = 1 + 2 * np.abs(returns)  # Higher volume with larger returns
    volumes = base_volume * volume_multiplier * np.random.lognormal(0, 0.3, n_days)
    
    # Create DataFrame
    market_df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes.astype(int),
        'Symbol': symbol
    }, index=dates)
    
    # Ensure OHLC logic (High >= max(O,C), Low <= min(O,C))
    market_df['High'] = np.maximum(market_df['High'], 
                                  np.maximum(market_df['Open'], market_df['Close']))
    market_df['Low'] = np.minimum(market_df['Low'], 
                                 np.minimum(market_df['Open'], market_df['Close']))
    
    return market_df


def create_sample_sentiment_data(start_date: str = "2023-01-01", end_date: str = "2024-01-01", symbol: str = "AAPL"):
    """Create sample sentiment data for testing."""
    
    np.random.seed(42)  # For reproducible results
    
    # Create irregular timestamps (multiple per day, but not every day)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    sentiment_records = []
    current_date = start
    
    while current_date <= end:
        # Random number of sentiment records per day (0-10)
        n_records_today = np.random.poisson(3)  # Average 3 records per day
        
        if n_records_today > 0:
            for i in range(n_records_today):
                # Random time during market hours (9 AM to 4 PM EST)
                hour = np.random.randint(9, 17)
                minute = np.random.randint(0, 60)
                timestamp = current_date.replace(hour=hour, minute=minute)
                
                # Generate sentiment score (-1 to 1)
                # Add some correlation with previous day's sentiment
                if len(sentiment_records) > 0:
                    prev_sentiment = sentiment_records[-1]['sentiment_score']
                    sentiment_score = prev_sentiment * 0.3 + np.random.normal(0, 0.4)
                else:
                    sentiment_score = np.random.normal(0, 0.4)
                
                # Clip to [-1, 1] range
                sentiment_score = np.clip(sentiment_score, -1, 1)
                
                # Generate confidence score (0.5 to 1.0)
                confidence = np.random.uniform(0.5, 1.0)
                
                # Random source
                sources = ['news', 'twitter', 'reddit', 'analyst_reports', 'earnings_calls']
                source = np.random.choice(sources, p=[0.4, 0.25, 0.15, 0.15, 0.05])
                
                sentiment_records.append({
                    'date': timestamp,
                    'sentiment_score': sentiment_score,
                    'confidence': confidence,
                    'source': source,
                    'symbol': symbol
                })
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(sentiment_records)


def test_technical_indicators():
    """Test technical indicators functionality."""
    
    logger.info("Testing Technical Indicators...")
    
    # Create sample market data
    market_df = create_sample_market_data()
    logger.info(f"Created sample market data with shape: {market_df.shape}")
    
    # Test individual indicator functions
    logger.info("Testing moving averages...")
    market_with_ma = calculate_moving_averages(market_df)
    
    logger.info("Testing RSI...")
    market_with_rsi = calculate_rsi(market_with_ma)
    
    logger.info("Testing MACD...")
    market_with_macd = calculate_macd(market_with_rsi)
    
    logger.info("Testing Bollinger Bands...")
    market_with_bb = calculate_bollinger_bands(market_with_macd)
    
    logger.info("Testing volume indicators...")
    market_with_volume = calculate_volume_indicators(market_with_bb)
    
    logger.info("Testing price momentum...")
    market_with_momentum = calculate_price_momentum(market_with_volume)
    
    # Test all indicators at once
    logger.info("Testing all technical indicators together...")
    market_all_indicators = add_all_technical_indicators(market_df)
    
    logger.info(f"Final technical indicators dataset shape: {market_all_indicators.shape}")
    logger.info(f"Number of technical features created: {len(market_all_indicators.columns) - len(market_df.columns)}")
    
    # Show sample of features
    technical_cols = [col for col in market_all_indicators.columns if col not in market_df.columns]
    logger.info(f"Sample technical features: {technical_cols[:10]}")
    
    return market_all_indicators


def test_sentiment_features():
    """Test sentiment features functionality."""
    
    logger.info("Testing Sentiment Features...")
    
    # Create sample sentiment data
    raw_sentiment_df = create_sample_sentiment_data()
    logger.info(f"Created sample raw sentiment data with shape: {raw_sentiment_df.shape}")
    
    # Test daily aggregation
    logger.info("Testing daily sentiment aggregation...")
    daily_sentiment = aggregate_daily_sentiment(raw_sentiment_df)
    logger.info(f"Daily sentiment data shape: {daily_sentiment.shape}")
    
    # Test sentiment momentum
    logger.info("Testing sentiment momentum...")
    sentiment_with_momentum = calculate_sentiment_momentum(daily_sentiment)
    
    # Test news volume indicators
    logger.info("Testing news volume indicators...")
    sentiment_with_volume = calculate_news_volume_indicators(sentiment_with_momentum)
    
    # Test all sentiment features
    logger.info("Testing all sentiment features together...")
    sentiment_all_features = add_all_sentiment_features(
        daily_sentiment, 
        raw_sentiment_df=raw_sentiment_df
    )
    
    logger.info(f"Final sentiment dataset shape: {sentiment_all_features.shape}")
    logger.info(f"Number of sentiment features created: {len(sentiment_all_features.columns)}")
    
    # Show sample of features
    sentiment_cols = list(sentiment_all_features.columns)
    logger.info(f"Sample sentiment features: {sentiment_cols[:10]}")
    
    return sentiment_all_features, daily_sentiment


def test_integration():
    """Test integration of technical and sentiment features."""
    
    logger.info("Testing Integration of Technical and Sentiment Features...")
    
    # Get both feature sets
    market_features = test_technical_indicators()
    sentiment_features, _ = test_sentiment_features()
    
    # Merge on date index
    logger.info("Merging technical and sentiment features...")
    combined_features = market_features.join(sentiment_features, how='left', rsuffix='_sentiment')
    
    logger.info(f"Combined dataset shape: {combined_features.shape}")
    logger.info(f"Total features: {len(combined_features.columns)}")
    
    # Show data quality
    missing_pct = combined_features.isnull().mean() * 100
    logger.info("Missing data percentage per feature (top 10 with missing data):")
    missing_features = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if len(missing_features) > 0:
        for feature, pct in missing_features.head(10).items():
            logger.info(f"  {feature}: {pct:.1f}%")
    else:
        logger.info("  No missing data found!")
    
    # Show sample statistics
    numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
    logger.info(f"Numeric features: {len(numeric_cols)}")
    
    # Save sample of the data
    sample_output = combined_features.head(20)
    output_file = "data/processed/feature_engineering_sample.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sample_output.to_csv(output_file)
    logger.info(f"Saved sample output to: {output_file}")
    
    return combined_features


def demonstrate_usage_examples():
    """Show practical usage examples."""
    
    logger.info("Demonstrating Practical Usage Examples...")
    
    # Example 1: Quick technical analysis
    logger.info("Example 1: Quick Technical Analysis Setup")
    market_df = create_sample_market_data()
    
    # Add just the essential indicators
    essential_indicators = calculate_moving_averages(market_df, sma_periods=[20, 50])
    essential_indicators = calculate_rsi(essential_indicators, periods=[14])
    essential_indicators = calculate_macd(essential_indicators)
    
    logger.info(f"Essential indicators shape: {essential_indicators.shape}")
    
    # Example 2: Sentiment analysis workflow
    logger.info("Example 2: Sentiment Analysis Workflow")
    raw_sentiment = create_sample_sentiment_data()
    
    # Custom aggregation
    daily_sent = aggregate_daily_sentiment(
        raw_sentiment,
        aggregation_methods=['mean', 'weighted_mean', 'count'],
        weight_by_confidence=True
    )
    
    # Add momentum for shorter timeframes
    sent_with_momentum = calculate_sentiment_momentum(
        daily_sent,
        windows=[3, 7],
        include_acceleration=False
    )
    
    logger.info(f"Custom sentiment features shape: {sent_with_momentum.shape}")
    
    # Example 3: Feature selection for modeling
    logger.info("Example 3: Feature Categories for Modeling")
    
    full_features = test_integration()
    
    # Categorize features
    price_features = [col for col in full_features.columns if any(term in col for term in 
                     ['SMA', 'EMA', 'Close', 'Open', 'High', 'Low', 'price'])]
    volume_features = [col for col in full_features.columns if 'Volume' in col or 'volume' in col]
    momentum_features = [col for col in full_features.columns if any(term in col for term in 
                        ['RSI', 'MACD', 'ROC', 'momentum', 'Momentum'])]
    volatility_features = [col for col in full_features.columns if any(term in col for term in 
                          ['BB', 'ATR', 'volatility', 'std'])]
    sentiment_features = [col for col in full_features.columns if 'sentiment' in col]
    
    logger.info(f"Feature categories:")
    logger.info(f"  Price features: {len(price_features)}")
    logger.info(f"  Volume features: {len(volume_features)}")
    logger.info(f"  Momentum features: {len(momentum_features)}")
    logger.info(f"  Volatility features: {len(volatility_features)}")
    logger.info(f"  Sentiment features: {len(sentiment_features)}")


if __name__ == "__main__":
    logger.info("Starting Feature Engineering Pipeline Tests...")
    
    try:
        # Run individual tests
        logger.info("=" * 60)
        test_technical_indicators()
        
        logger.info("=" * 60)
        test_sentiment_features()
        
        logger.info("=" * 60)
        test_integration()
        
        logger.info("=" * 60)
        demonstrate_usage_examples()
        
        logger.info("=" * 60)
        logger.info("✅ All tests completed successfully!")
        logger.info("Feature Engineering Pipeline is ready for use.")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)