#!/usr/bin/env python3
"""
Test script for the data merge module.

This script demonstrates how to use the merge functions with sample data.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.merge import (
    merge_market_economic_data,
    align_sentiment_with_market_data,
    create_unified_dataset
)


def create_sample_market_data(symbol='AAPL', days=100):
    """Create sample market data for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create business day index
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate synthetic price data
    np.random.seed(42)
    initial_price = 150.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    market_df = pd.DataFrame({
        'Open': prices * np.random.uniform(0.98, 1.02, len(dates)),
        'High': prices * np.random.uniform(1.01, 1.03, len(dates)),
        'Low': prices * np.random.uniform(0.97, 0.99, len(dates)),
        'Close': prices,
        'Volume': np.random.randint(10000000, 50000000, len(dates))
    }, index=dates)
    
    return market_df


def create_sample_economic_data(start_date, end_date):
    """Create sample economic indicator data."""
    # Monthly data - GDP growth rate
    gdp_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    gdp_data = pd.DataFrame({
        'value': np.random.normal(2.5, 0.5, len(gdp_dates))
    }, index=gdp_dates)
    
    # Weekly data - unemployment rate
    unemp_dates = pd.date_range(start=start_date, end=end_date, freq='W')
    unemp_data = pd.DataFrame({
        'value': np.random.normal(4.0, 0.3, len(unemp_dates))
    }, index=unemp_dates)
    
    # Daily data - VIX
    vix_dates = pd.date_range(start=start_date, end=end_date, freq='B')
    vix_data = pd.DataFrame({
        'value': np.random.lognormal(2.8, 0.3, len(vix_dates))
    }, index=vix_dates)
    
    return {
        'GDP_GROWTH': gdp_data,
        'UNEMPLOYMENT': unemp_data,
        'VIX': vix_data
    }


def create_sample_sentiment_data(start_date, end_date):
    """Create sample sentiment data."""
    # Generate random timestamps throughout the day
    n_articles = 200
    timestamps = []
    
    current = start_date
    while current <= end_date:
        # Random number of articles per day (0-5)
        n_daily = np.random.randint(0, 6)
        for _ in range(n_daily):
            # Random time during trading hours
            hour = np.random.randint(9, 16)
            minute = np.random.randint(0, 60)
            timestamps.append(current.replace(hour=hour, minute=minute))
        current += timedelta(days=1)
    
    # Create sentiment scores
    sentiment_df = pd.DataFrame({
        'sentiment_score': np.random.uniform(-1, 1, len(timestamps)),
        'confidence': np.random.uniform(0.5, 1.0, len(timestamps)),
        'source': np.random.choice(['reuters', 'bloomberg', 'wsj'], len(timestamps))
    }, index=pd.DatetimeIndex(timestamps))
    
    return sentiment_df.sort_index()


def main():
    """Run test of merge functionality."""
    print("Testing StockSage.AI Data Merge Module")
    print("=" * 50)
    
    # Parameters
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    # Create sample data
    print("\n1. Creating sample data...")
    market_data = create_sample_market_data(symbol, days=100)
    economic_data = create_sample_economic_data(start_date, end_date)
    sentiment_data = create_sample_sentiment_data(start_date, end_date)
    
    print(f"   - Market data shape: {market_data.shape}")
    print(f"   - Economic indicators: {list(economic_data.keys())}")
    print(f"   - Sentiment data points: {len(sentiment_data)}")
    
    # Test merge_market_economic_data
    print("\n2. Testing market + economic data merge...")
    merged_df = merge_market_economic_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        market_data=market_data,
        economic_data=economic_data
    )
    print(f"   - Merged shape: {merged_df.shape}")
    print(f"   - Columns added: {[col for col in merged_df.columns if col not in market_data.columns][:5]}...")
    
    # Test align_sentiment_with_market_data
    print("\n3. Testing sentiment alignment...")
    aligned_df = align_sentiment_with_market_data(
        market_df=merged_df,
        sentiment_df=sentiment_data,
        aggregation_method='weighted_mean'
    )
    print(f"   - Aligned shape: {aligned_df.shape}")
    sentiment_cols = [col for col in aligned_df.columns if 'sentiment' in col]
    print(f"   - Sentiment columns: {sentiment_cols}")
    
    # Test create_unified_dataset
    print("\n4. Testing unified dataset creation...")
    unified_df = create_unified_dataset(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        market_data=market_data,
        economic_data=economic_data,
        sentiment_data=sentiment_data,
        include_technical_indicators=True,
        include_market_regime=True
    )
    
    print(f"   - Final shape: {unified_df.shape}")
    print(f"   - Total features: {len(unified_df.columns)}")
    
    # Show feature categories
    print("\n5. Feature categories in unified dataset:")
    
    # Market features
    market_features = [col for col in unified_df.columns if any(x in col.lower() for x in ['open', 'high', 'low', 'close', 'volume'])]
    print(f"   - Market features: {len(market_features)}")
    
    # Economic features
    econ_features = [col for col in unified_df.columns if any(x in col.upper() for x in ['GDP', 'UNEMPLOYMENT', 'VIX'])]
    print(f"   - Economic features: {len(econ_features)}")
    
    # Sentiment features
    sentiment_features = [col for col in unified_df.columns if 'sentiment' in col]
    print(f"   - Sentiment features: {len(sentiment_features)}")
    
    # Technical indicators
    tech_features = [col for col in unified_df.columns if any(x in col for x in ['sma_', 'ema_', 'rsi', 'macd', 'bb_'])]
    print(f"   - Technical indicators: {len(tech_features)}")
    
    # Temporal features
    temporal_features = [col for col in unified_df.columns if any(x in col for x in ['day_', 'week_', 'month', 'quarter', 'year', 'is_'])]
    print(f"   - Temporal features: {len(temporal_features)}")
    
    # Sample output
    print("\n6. Sample of unified dataset:")
    print(unified_df.head())
    
    # Data quality check
    print("\n7. Data quality check:")
    missing_pct = unified_df.isnull().sum() / len(unified_df) * 100
    print(f"   - Columns with missing data: {(missing_pct > 0).sum()}")
    print(f"   - Average missing %: {missing_pct.mean():.2f}%")
    
    # Save sample
    output_path = 'unified_dataset_sample.csv'
    unified_df.tail(50).to_csv(output_path)
    print(f"\n8. Sample saved to: {output_path}")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()