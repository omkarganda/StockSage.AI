#!/usr/bin/env python3
"""
Enhanced Test Script for StockSage.AI Data Merge Module with Validation

This script demonstrates the integrated data merge functionality with 
comprehensive validation and quality checks, including:
- Market data validation
- Economic data validation
- Sentiment data validation  
- Unified dataset validation
- Error handling and logging
- Data quality reporting
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.merge import (
    merge_market_economic_data,
    align_sentiment_with_market_data,
    create_unified_dataset
)
from src.data.validation import (
    validate_market_data,
    validate_sentiment_data,
    validate_economic_data,
    validate_unified_data,
    DataValidator
)
from src.utils.logging import get_logger, setup_logging, log_system_info
from scripts.data_quality_dashboard import DataQualityDashboard


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
    """Run comprehensive test of merge functionality with validation."""
    # Setup enhanced logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    print("🚀 Testing StockSage.AI Data Merge Module with Validation")
    print("=" * 80)
    logger.info("Starting enhanced merge module test with validation")
    
    # Initialize quality dashboard for tracking
    dashboard = DataQualityDashboard()
    all_validation_reports = []
    
    # Parameters
    symbol = 'AAPL'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    
    # Create sample data
    print("\n📊 1. Creating sample data...")
    logger.info("Creating sample data for testing", symbol=symbol, days=100)
    
    market_data = create_sample_market_data(symbol, days=100)
    economic_data = create_sample_economic_data(start_date, end_date)
    sentiment_data = create_sample_sentiment_data(start_date, end_date)
    
    print(f"   - Market data shape: {market_data.shape}")
    print(f"   - Economic indicators: {list(economic_data.keys())}")
    print(f"   - Sentiment data points: {len(sentiment_data)}")
    
    # Validate input data
    print("\n🔍 2. Validating input data...")
    
    # Validate market data
    print("   - Validating market data...")
    market_report = validate_market_data(market_data, f"{symbol}_market_input")
    market_report.print_summary()
    all_validation_reports.append(market_report)
    
    # Validate sentiment data
    print("   - Validating sentiment data...")
    sentiment_report = validate_sentiment_data(sentiment_data, f"{symbol}_sentiment_input")
    sentiment_report.print_summary()
    all_validation_reports.append(sentiment_report)
    
    # Validate economic data
    for indicator, econ_df in economic_data.items():
        print(f"   - Validating {indicator} data...")
        econ_report = validate_economic_data(econ_df, f"{indicator}_input")
        if not econ_report.is_valid:
            print(f"     ⚠️  {indicator} has validation issues!")
        all_validation_reports.append(econ_report)
    
    # Test merge_market_economic_data with validation
    print("\n🔄 3. Testing market + economic data merge with validation...")
    
    try:
        merged_df, merge_reports = merge_market_economic_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            market_data=market_data,
            economic_data=economic_data,
            validate_inputs=True,
            validate_output=True
        )
        all_validation_reports.extend(merge_reports)
        
        print(f"   ✅ Merged shape: {merged_df.shape}")
        new_columns = [col for col in merged_df.columns if col not in market_data.columns]
        print(f"   - New columns added: {len(new_columns)}")
        if new_columns:
            print(f"   - Sample new columns: {new_columns[:5]}...")
            
    except Exception as e:
        logger.error(f"Market-economic merge failed: {e}")
        print(f"   ❌ Merge failed: {e}")
        return False
    
    # Test align_sentiment_with_market_data with validation
    print("\n💭 4. Testing sentiment alignment with validation...")
    
    try:
        aligned_df, sentiment_reports = align_sentiment_with_market_data(
            market_df=merged_df,
            sentiment_df=sentiment_data,
            aggregation_method='weighted_mean',
            validate_inputs=True
        )
        all_validation_reports.extend(sentiment_reports)
        
        print(f"   ✅ Aligned shape: {aligned_df.shape}")
        sentiment_cols = [col for col in aligned_df.columns if 'sentiment' in col]
        print(f"   - Sentiment columns added: {len(sentiment_cols)}")
        if sentiment_cols:
            print(f"   - Sentiment features: {sentiment_cols}")
            
    except Exception as e:
        logger.error(f"Sentiment alignment failed: {e}")
        print(f"   ❌ Alignment failed: {e}")
        return False
    
    # Test create_unified_dataset with comprehensive validation
    print("\n🎯 5. Testing unified dataset creation with validation...")
    
    try:
        unified_df, unified_reports = create_unified_dataset(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            market_data=market_data,
            economic_data=economic_data,
            sentiment_data=sentiment_data,
            include_technical_indicators=True,
            include_market_regime=True,
            validate_inputs=True,
            validate_output=True
        )
        all_validation_reports.extend(unified_reports)
        
        print(f"   ✅ Final unified shape: {unified_df.shape}")
        print(f"   - Total features: {len(unified_df.columns)}")
        
    except Exception as e:
        logger.error(f"Unified dataset creation failed: {e}")
        print(f"   ❌ Unified dataset creation failed: {e}")
        return False
    
    # Analyze feature categories
    print("\n📋 6. Feature categories in unified dataset:")
    
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
    
    # Final validation and quality assessment
    print("\n🔬 7. Comprehensive data quality assessment...")
    
    final_report = validate_unified_data(unified_df, f"{symbol}_final_unified")
    final_report.print_summary()
    all_validation_reports.append(final_report)
    
    # Record results in dashboard
    summary = dashboard.record_validation_results(all_validation_reports, "test_merge_pipeline")
    
    print(f"\n📊 8. Quality metrics summary:")
    print(f"   - Total validation reports: {len(all_validation_reports)}")
    print(f"   - Overall pass rate: {summary['pass_rate']:.1f}%")
    print(f"   - Average quality score: {summary['avg_quality_score']:.1f}/100")
    print(f"   - Critical issues: {summary['critical_issues']}")
    print(f"   - Error issues: {summary['error_issues']}")
    print(f"   - Warning issues: {summary['warning_issues']}")
    
    # Check for quality alerts
    alerts = dashboard.check_quality_alerts()
    if alerts:
        print(f"\n🚨 Quality alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"   - {alert['severity'].upper()}: {alert['message']}")
    else:
        print("\n✅ No quality alerts - data quality is within acceptable thresholds")
    
    # Data quality insights
    print(f"\n💡 9. Data quality insights:")
    missing_pct = unified_df.isnull().sum() / len(unified_df) * 100
    print(f"   - Columns with missing data: {(missing_pct > 0).sum()}")
    print(f"   - Average missing percentage: {missing_pct.mean():.2f}%")
    print(f"   - Dataset completeness: {100 - missing_pct.mean():.1f}%")
    
    # Performance insights
    from src.utils.logging import get_all_performance_stats
    perf_stats = get_all_performance_stats()
    
    if perf_stats:
        print(f"\n⚡ 10. Performance insights:")
        total_operations = sum(sum(stats.get('count', 0) for stats in module_stats.values()) 
                             for module_stats in perf_stats.values())
        total_time = sum(sum(stats.get('total_time', 0) for stats in module_stats.values()) 
                        for module_stats in perf_stats.values())
        print(f"   - Total operations: {total_operations}")
        print(f"   - Total processing time: {total_time:.2f} seconds")
        if total_operations > 0:
            print(f"   - Average operation time: {total_time/total_operations:.4f} seconds")
    
    # Save outputs
    print(f"\n💾 11. Saving outputs...")
    
    # Save unified dataset sample
    output_path = Path('unified_dataset_sample.csv')
    unified_df.tail(50).to_csv(output_path)
    print(f"   - Dataset sample saved to: {output_path}")
    
    # Generate quality report
    try:
        html_report = dashboard.generate_data_quality_report()
        print(f"   - Quality report generated in: {dashboard.output_dir}")
    except Exception as e:
        logger.warning(f"Could not generate quality report: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    if summary['critical_issues'] == 0 and summary['pass_rate'] >= 80:
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ Data merge pipeline with validation is working correctly")
        logger.info("Test completed successfully", 
                   pass_rate=summary['pass_rate'], 
                   quality_score=summary['avg_quality_score'])
        return True
    else:
        print("⚠️  TESTS COMPLETED WITH ISSUES")
        print("❌ Data quality issues detected - review validation reports")
        logger.warning("Test completed with quality issues", 
                      critical_issues=summary['critical_issues'],
                      pass_rate=summary['pass_rate'])
        return False


if __name__ == "__main__":
    main()