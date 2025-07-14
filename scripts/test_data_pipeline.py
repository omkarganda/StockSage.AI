#!/usr/bin/env python
"""
Test script to collect 1 week of data for AAPL
Demonstrates the complete data pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime, timedelta
import logging

# Import our modules
from src.data.download_market import MarketDataDownloader
from src.data.download_economic import EconomicDataDownloader
from src.data.download_sentiment import SentimentDataProcessor
from src.config import APIConfig, RAW_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_market_data():
    """Test market data collection"""
    logger.info("=" * 50)
    logger.info("Testing Market Data Collection")
    logger.info("=" * 50)
    
    downloader = MarketDataDownloader()
    
    # Define date range (1 week)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Download AAPL data
    try:
        aapl_data = downloader.download_stock_data(
            ticker="AAPL",
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        logger.info(f"✓ Downloaded {len(aapl_data)} days of AAPL market data")
        logger.info(f"  Date range: {aapl_data.index.min()} to {aapl_data.index.max()}")
        logger.info(f"  Columns: {list(aapl_data.columns)}")
        logger.info(f"  Latest close: ${aapl_data['Close'].iloc[-1]:.2f}")
        
        # Get stock info
        aapl_info = downloader.get_stock_info("AAPL")
        logger.info(f"\n✓ AAPL Company Info:")
        logger.info(f"  Name: {aapl_info['name']}")
        logger.info(f"  Sector: {aapl_info['sector']}")
        logger.info(f"  Market Cap: ${aapl_info['market_cap']:,.0f}")
        
        # Get real-time price
        realtime = downloader.get_realtime_price("AAPL")
        logger.info(f"\n✓ Real-time Price: ${realtime['current_price']:.2f}")
        
        return aapl_data
        
    except Exception as e:
        logger.error(f"✗ Failed to download market data: {str(e)}")
        return None

def test_economic_data():
    """Test economic data collection"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Economic Data Collection")
    logger.info("=" * 50)
    
    if not APIConfig.FRED_API_KEY:
        logger.warning("✗ FRED API key not found. Skipping economic data test.")
        logger.warning("  Get your free API key at: https://fred.stlouisfed.org/docs/api/")
        return None
    
    downloader = EconomicDataDownloader()
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        # Download key indicators
        indicators_to_test = {
            'DGS10': '10-Year Treasury Rate',
            'DFF': 'Federal Funds Rate',
            'VIXCLS': 'VIX Volatility Index'
        }
        
        economic_data = {}
        for series_id, name in indicators_to_test.items():
            data = downloader.download_indicator(
                indicator=series_id,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            economic_data[series_id] = data
            logger.info(f"✓ Downloaded {name}: {len(data)} observations")
            if not data.empty:
                logger.info(f"  Latest value: {data.iloc[-1, 0]:.2f}")
        
        # Combine all economic data
        combined_economic = pd.concat(economic_data.values(), axis=1)
        combined_economic = combined_economic.fillna(method='ffill')
        
        return combined_economic
        
    except Exception as e:
        logger.error(f"✗ Failed to download economic data: {str(e)}")
        return None

def test_sentiment_data():
    """Test sentiment data collection"""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Sentiment Data Collection")
    logger.info("=" * 50)
    
    if not APIConfig.NEWS_API_KEY:
        logger.warning("✗ News API key not found. Skipping sentiment data test.")
        logger.warning("  Get your free API key at: https://newsapi.org/")
        return None
    
    processor = SentimentDataProcessor()
    
    # Define date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        # Get sentiment for Apple
        sentiment_data = processor.get_daily_sentiment(
            ticker="AAPL",
            company_name="Apple Inc",
            start_date=start_date,
            end_date=end_date
        )
        
        if not sentiment_data.empty:
            logger.info(f"✓ Collected sentiment data: {len(sentiment_data)} days")
            logger.info(f"  Average compound sentiment: {sentiment_data['sentiment_compound'].mean():.3f}")
            logger.info(f"  Total articles analyzed: {sentiment_data['article_count'].sum()}")
            
            # Show sentiment breakdown
            latest_sentiment = sentiment_data.iloc[-1]
            logger.info(f"\n  Latest sentiment scores:")
            logger.info(f"    Positive: {latest_sentiment['sentiment_positive']:.3f}")
            logger.info(f"    Neutral:  {latest_sentiment['sentiment_neutral']:.3f}")
            logger.info(f"    Negative: {latest_sentiment['sentiment_negative']:.3f}")
        else:
            logger.warning("✗ No sentiment data collected")
        
        return sentiment_data
        
    except Exception as e:
        logger.error(f"✗ Failed to collect sentiment data: {str(e)}")
        return None

def merge_all_data(market_data, economic_data, sentiment_data):
    """Merge all data sources into a single DataFrame"""
    logger.info("\n" + "=" * 50)
    logger.info("Merging All Data Sources")
    logger.info("=" * 50)
    
    all_data = []
    
    # Prepare market data
    if market_data is not None:
        market_daily = market_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        market_daily.columns = [f'AAPL_{col}' for col in market_daily.columns]
        
        # Normalize timezone - convert to timezone-naive (remove timezone info)
        if market_daily.index.tz is not None:
            market_daily.index = market_daily.index.tz_convert('UTC').tz_localize(None)
        
        all_data.append(market_daily)
        logger.info(f"✓ Market data: {len(market_daily)} rows")
    
    # Prepare economic data
    if economic_data is not None:
        # Resample to daily if needed
        economic_daily = economic_data.resample('D').ffill()
        
        # Ensure timezone-naive
        if economic_daily.index.tz is not None:
            economic_daily.index = economic_daily.index.tz_convert('UTC').tz_localize(None)
        
        all_data.append(economic_daily)
        logger.info(f"✓ Economic data: {len(economic_daily)} rows")
    
    # Prepare sentiment data
    if sentiment_data is not None:
        # Convert date index to datetime
        sentiment_daily = sentiment_data.copy()
        sentiment_daily.index = pd.to_datetime(sentiment_daily.index)
        
        # Ensure timezone-naive
        if sentiment_daily.index.tz is not None:
            sentiment_daily.index = sentiment_daily.index.tz_convert('UTC').tz_localize(None)
        
        all_data.append(sentiment_daily)
        logger.info(f"✓ Sentiment data: {len(sentiment_daily)} rows")
    
    if all_data:
        # Merge all data - now all indices are timezone-naive
        merged_data = pd.concat(all_data, axis=1, join='outer')
        merged_data = merged_data.sort_index()
        
        # Forward fill missing values
        merged_data = merged_data.fillna(method='ffill')
        
        # Save to file
        output_path = RAW_DATA_DIR / f"aapl_merged_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        merged_data.to_csv(output_path)
        logger.info(f"\n✓ Merged data saved to: {output_path}")
        logger.info(f"  Total rows: {len(merged_data)}")
        logger.info(f"  Total columns: {len(merged_data.columns)}")
        logger.info(f"  Date range: {merged_data.index.min()} to {merged_data.index.max()}")
        
        return merged_data
    else:
        logger.error("✗ No data available to merge")
        return None

def main():
    """Main test function"""
    logger.info("StockSage.AI Data Pipeline Test")
    logger.info("Testing 1 week of AAPL data collection")
    logger.info("=" * 70)
    
    # Check API keys
    logger.info("\nChecking API Keys:")
    api_status = APIConfig.get_available_apis()
    for api, available in api_status.items():
        status = "✓" if available else "✗"
        logger.info(f"  {status} {api}")
    
    # Test each data source
    market_data = test_market_data()
    economic_data = test_economic_data()
    sentiment_data = test_sentiment_data()
    
    # Merge all data
    merged_data = merge_all_data(market_data, economic_data, sentiment_data)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary:")
    logger.info(f"  Market Data: {'✓ Success' if market_data is not None else '✗ Failed'}")
    logger.info(f"  Economic Data: {'✓ Success' if economic_data is not None else '✗ Failed/Skipped'}")
    logger.info(f"  Sentiment Data: {'✓ Success' if sentiment_data is not None else '✗ Failed/Skipped'}")
    logger.info(f"  Data Merge: {'✓ Success' if merged_data is not None else '✗ Failed'}")
    
    if merged_data is not None:
        logger.info("\n✓ Data pipeline test completed successfully!")
        logger.info(f"  Output saved to: {RAW_DATA_DIR}")
        
        # Show sample of merged data
        logger.info("\nSample of merged data (last 3 rows):")
        logger.info(merged_data.tail(3).to_string())
    else:
        logger.warning("\n⚠ Data pipeline test completed with some failures.")
        logger.warning("  Check API keys and try again.")

if __name__ == "__main__":
    main()