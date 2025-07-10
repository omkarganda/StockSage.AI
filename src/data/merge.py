"""
Data Merging & Alignment Module for StockSage.AI

This module provides functionality to merge and align different data sources:
- Market data (from yfinance)
- Economic indicators (from FRED)
- Sentiment scores (from news/FinBERT)

All data is aligned by timestamp with proper handling of different frequencies
and missing values.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta
import warnings
from functools import reduce
import logging

# Import validation and enhanced logging
from .validation import DataValidator, ValidationReport, validate_market_data, validate_sentiment_data, validate_economic_data, validate_unified_data
from ..utils.logging import get_logger, log_data_operation

# Configure enhanced logging
logger = get_logger(__name__)

# Create a global validator instance
validator = DataValidator(strict_mode=False)


@log_data_operation("market_economic_merge")
def merge_market_economic_data(
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    market_data: Optional[pd.DataFrame] = None,
    economic_data: Optional[Dict[str, pd.DataFrame]] = None,
    alignment: str = 'inner',
    fill_method: str = 'ffill',
    validate_inputs: bool = True,
    validate_output: bool = True
) -> Tuple[pd.DataFrame, List[ValidationReport]]:
    """
    Merge market data with economic indicators.
    
    This function handles the complexity of merging daily market data with
    economic indicators that may have different frequencies (monthly, quarterly).
    
    Parameters:
    -----------
    symbol : str
        Stock symbol (e.g., 'AAPL')
    start_date : str or datetime
        Start date for the data
    end_date : str or datetime
        End date for the data
    market_data : pd.DataFrame, optional
        Pre-loaded market data. If None, should be loaded via download_market.py
    economic_data : dict, optional
        Dict of economic indicator DataFrames. Keys are indicator names.
        If None, should be loaded via download_economic.py
    alignment : str, default='inner'
        How to align data ('inner', 'outer', 'left', 'right')
    fill_method : str, default='ffill'
        Method to fill missing values ('ffill', 'bfill', 'interpolate', None)
    validate_inputs : bool, default=True
        Whether to validate input data
    validate_output : bool, default=True
        Whether to validate merged output
    
    Returns:
    --------
    tuple[pd.DataFrame, List[ValidationReport]]
        Merged dataset with market and economic data aligned by date, and validation reports
    """
    logger.info(f"Starting merge of market and economic data for {symbol}", 
                symbol=symbol, start_date=start_date, end_date=end_date)
    
    validation_reports = []
    
    # Convert dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Validate market data if provided
    if validate_inputs and market_data is not None:
        logger.info("Validating market data input")
        market_report = validate_market_data(market_data, f"{symbol}_market_data")
        validation_reports.append(market_report)
        
        if market_report.has_critical_issues():
            logger.error("Critical issues found in market data", symbol=symbol)
            market_report.print_summary()
            raise ValueError(f"Market data for {symbol} has critical validation issues")
    
    # Validate economic data if provided
    if validate_inputs and economic_data:
        for indicator_name, econ_df in economic_data.items():
            logger.info(f"Validating {indicator_name} economic data")
            econ_report = validate_economic_data(econ_df, f"{indicator_name}_data")
            validation_reports.append(econ_report)
            
            if econ_report.has_critical_issues():
                logger.error(f"Critical issues found in {indicator_name} data", 
                           indicator=indicator_name, symbol=symbol)
                econ_report.print_summary()
    
    # If data not provided, we would load it here
    # For now, we'll work with provided data or create sample structure
    if market_data is None:
        logger.warning(f"No market data provided for {symbol}. Creating empty structure.")
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        market_data = pd.DataFrame(index=date_range)
        market_data.index.name = 'Date'
    
    # Ensure market data has datetime index
    if not isinstance(market_data.index, pd.DatetimeIndex):
        market_data.index = pd.to_datetime(market_data.index)
    
    # Start with market data as base
    merged_df = market_data.copy()
    
    # Add symbol column if not present
    if 'symbol' not in merged_df.columns:
        merged_df['symbol'] = symbol
    
    # Merge economic indicators
    if economic_data:
        for indicator_name, indicator_df in economic_data.items():
            logger.info(f"Merging {indicator_name} data...", indicator=indicator_name)
            
            try:
                # Ensure datetime index
                if not isinstance(indicator_df.index, pd.DatetimeIndex):
                    indicator_df.index = pd.to_datetime(indicator_df.index)
                
                # Rename columns to avoid conflicts
                indicator_df = indicator_df.copy()
                indicator_df.columns = [f"{indicator_name}_{col}" if col != indicator_name else col 
                                       for col in indicator_df.columns]
                
                # Handle different frequencies
                if len(indicator_df) < len(merged_df) * 0.5:
                    # Likely lower frequency (monthly/quarterly)
                    # Use forward fill to propagate values
                    indicator_df = indicator_df.reindex(merged_df.index, method='ffill')
                    
                    # Add a column to track when the value was last updated
                    for col in indicator_df.columns:
                        merged_df[f"{col}_days_since_update"] = (
                            merged_df.index - 
                            indicator_df[col].dropna().index.to_series().reindex(merged_df.index, method='ffill')
                        ).dt.days
                
                # Merge the data
                merged_df = merged_df.join(indicator_df, how=alignment)
                
                logger.info(f"Successfully merged {indicator_name}", 
                           rows_added=len(indicator_df.columns), 
                           indicator=indicator_name)
                
            except Exception as e:
                logger.error(f"Failed to merge {indicator_name} data", 
                           indicator=indicator_name, error=str(e))
                # Continue with other indicators rather than failing completely
                continue
    
    # Handle missing values based on fill_method
    if fill_method:
        logger.info(f"Applying fill method: {fill_method}")
        with logger.timer(f"fill_missing_{fill_method}"):
            if fill_method == 'ffill':
                merged_df = merged_df.ffill()
            elif fill_method == 'bfill':
                merged_df = merged_df.bfill()
            elif fill_method == 'interpolate':
                numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
                merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear')
    
    # Add derived features
    merged_df = _add_temporal_features(merged_df)
    
    # Validate output if requested
    if validate_output:
        logger.info("Validating merged output")
        output_report = validate_unified_data(merged_df, f"{symbol}_merged_data")
        validation_reports.append(output_report)
        
        if output_report.has_critical_issues():
            logger.error("Critical issues found in merged data", symbol=symbol)
            output_report.print_summary()
    
    # Log merge statistics
    logger.info(f"Merged data shape: {merged_df.shape}", 
               shape=str(merged_df.shape), symbol=symbol)
    logger.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    
    missing_stats = merged_df.isnull().sum()
    if missing_stats.sum() > 0:
        logger.warning("Missing values found after merge", 
                      missing_columns=missing_stats[missing_stats > 0].to_dict())
    else:
        logger.info("No missing values in merged data")
    
    return merged_df, validation_reports


@log_data_operation("sentiment_alignment")
def align_sentiment_with_market_data(
    market_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    sentiment_window: str = '1D',
    aggregation_method: str = 'weighted_mean',
    weight_decay: float = 0.8,
    validate_inputs: bool = True
) -> Tuple[pd.DataFrame, List[ValidationReport]]:
    """
    Align sentiment scores with market data, handling intraday sentiment updates.
    
    Sentiment data often comes at irregular intervals (when news is published),
    so we need to aggregate it appropriately to match market data frequency.
    
    Parameters:
    -----------
    market_df : pd.DataFrame
        Market data with datetime index
    sentiment_df : pd.DataFrame
        Sentiment scores with datetime index and columns like 'sentiment_score',
        'confidence', 'source', etc.
    sentiment_window : str, default='1D'
        Time window for sentiment aggregation ('1D', '4H', '1H', etc.)
    aggregation_method : str, default='weighted_mean'
        How to aggregate multiple sentiment scores:
        - 'mean': Simple average
        - 'weighted_mean': Weight by confidence scores
        - 'ewm': Exponentially weighted mean (recent news matters more)
        - 'last': Use most recent sentiment
    weight_decay : float, default=0.8
        Decay factor for exponential weighting (used if aggregation_method='ewm')
    validate_inputs : bool, default=True
        Whether to validate input data
    
    Returns:
    --------
    tuple[pd.DataFrame, List[ValidationReport]]
        Market data with aligned sentiment features, and validation reports
    """
    logger.info(f"Starting sentiment alignment for {market_df.index.name} data", 
                symbol=market_df.index.name, start_date=market_df.index.min(), end_date=market_df.index.max())
    
    validation_reports = []
    
    # Ensure datetime indices
    if not isinstance(market_df.index, pd.DatetimeIndex):
        market_df = market_df.copy()
        market_df.index = pd.to_datetime(market_df.index)
    
    if not isinstance(sentiment_df.index, pd.DatetimeIndex):
        sentiment_df = sentiment_df.copy()
        sentiment_df.index = pd.to_datetime(sentiment_df.index)
    
    # Sort by time
    sentiment_df = sentiment_df.sort_index()
    
    # Create result dataframe
    result_df = market_df.copy()
    
    # Aggregate sentiment based on method
    if aggregation_method == 'mean':
        # Simple rolling mean
        sentiment_agg = sentiment_df.resample(sentiment_window).mean()
        
    elif aggregation_method == 'weighted_mean':
        # Weight by confidence if available
        if 'confidence' in sentiment_df.columns:
            def weighted_avg(group):
                if len(group) == 0:
                    return pd.Series(dtype=float, name='sentiment_score')
                weights = group['confidence']
                return pd.Series({
                    'sentiment_score': np.average(group['sentiment_score'], weights=weights),
                    'sentiment_count': len(group),
                    'avg_confidence': weights.mean()
                })
            sentiment_agg = sentiment_df.resample(sentiment_window).apply(weighted_avg)
        else:
            # Fall back to simple mean
            sentiment_agg = sentiment_df.resample(sentiment_window).mean()
            
    elif aggregation_method == 'ewm':
        # Exponentially weighted mean - recent news matters more
        sentiment_agg = sentiment_df.resample(sentiment_window).apply(
            lambda x: x.ewm(halflife=weight_decay, times=x.index).mean().iloc[-1] if len(x) > 0 else np.nan
        )
        
    elif aggregation_method == 'last':
        # Use most recent sentiment
        sentiment_agg = sentiment_df.resample(sentiment_window).last()
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    # Forward fill to propagate sentiment to next market day
    sentiment_agg = sentiment_agg.ffill(limit=7)  # Max 7 days forward fill
    
    # Align with market data
    result_df = result_df.join(sentiment_agg, how='left', rsuffix='_sentiment')
    
    # Add sentiment age feature (how old is the sentiment data)
    if 'sentiment_score' in result_df.columns:
        # Track when sentiment was last updated
        sentiment_updated = result_df['sentiment_score'].notna()
        last_update_idx = sentiment_updated.cumsum()
        last_update_dates = result_df.index.to_series().where(sentiment_updated).ffill()
        result_df['sentiment_age_hours'] = (
            (result_df.index - last_update_dates).total_seconds() / 3600
        ).fillna(0)
    
    # Add sentiment momentum features
    if 'sentiment_score' in result_df.columns:
        result_df['sentiment_ma_3d'] = result_df['sentiment_score'].rolling(window=3).mean()
        result_df['sentiment_ma_7d'] = result_df['sentiment_score'].rolling(window=7).mean()
        result_df['sentiment_momentum'] = result_df['sentiment_score'] - result_df['sentiment_ma_7d']
        result_df['sentiment_volatility'] = result_df['sentiment_score'].rolling(window=7).std()
    
    # Validate output if requested
    if validate_inputs:
        logger.info("Validating sentiment alignment output")
        output_report = validate_sentiment_data(result_df, f"{market_df.index.name}_sentiment_data")
        validation_reports.append(output_report)
        
        if output_report.has_critical_issues():
            logger.error("Critical issues found in sentiment data", symbol=market_df.index.name)
            output_report.print_summary()
    
    # Log alignment statistics
    logger.info(f"Sentiment alignment complete. Added {len([c for c in result_df.columns if 'sentiment' in c])} sentiment features")
    if 'sentiment_score' in result_df.columns:
        logger.info(f"Sentiment coverage: {result_df['sentiment_score'].notna().sum() / len(result_df) * 100:.1f}%")
    
    return result_df, validation_reports


@log_data_operation("unified_dataset_creation")
def create_unified_dataset(
    symbol: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    market_data: Optional[pd.DataFrame] = None,
    economic_data: Optional[Dict[str, pd.DataFrame]] = None,
    sentiment_data: Optional[pd.DataFrame] = None,
    include_technical_indicators: bool = True,
    include_market_regime: bool = True,
    handle_missing: str = 'interpolate',
    validate_inputs: bool = True,
    validate_output: bool = True
) -> Tuple[pd.DataFrame, List[ValidationReport]]:
    """
    Create a unified dataset by combining all data sources.
    
    This is the main orchestration function that brings together market data,
    economic indicators, and sentiment scores into a single, analysis-ready dataset.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    start_date : str or datetime
        Start date for the dataset
    end_date : str or datetime
        End date for the dataset
    market_data : pd.DataFrame, optional
        Pre-loaded market data
    economic_data : dict, optional
        Dictionary of economic indicator DataFrames
    sentiment_data : pd.DataFrame, optional
        Sentiment scores data
    include_technical_indicators : bool, default=True
        Whether to calculate and include technical indicators
    include_market_regime : bool, default=True
        Whether to include market regime detection features
    handle_missing : str, default='interpolate'
        How to handle missing values ('drop', 'interpolate', 'forward_fill', 'mean')
    validate_inputs : bool, default=True
        Whether to validate input data sources
    validate_output : bool, default=True
        Whether to validate the final unified dataset
    
    Returns:
    --------
    tuple[pd.DataFrame, List[ValidationReport]]
        Unified dataset ready for modeling and all validation reports
    """
    logger.info(f"Creating unified dataset for {symbol} from {start_date} to {end_date}",
                symbol=symbol, start_date=start_date, end_date=end_date)
    
    all_validation_reports = []
    
    # Step 1: Merge market and economic data
    with logger.timer("market_economic_merge"):
        unified_df, merge_reports = merge_market_economic_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            market_data=market_data,
            economic_data=economic_data,
            fill_method='ffill',
            validate_inputs=validate_inputs,
            validate_output=validate_inputs
        )
        all_validation_reports.extend(merge_reports)
    
    # Step 2: Align sentiment data if provided
    if sentiment_data is not None:
        with logger.timer("sentiment_alignment"):
            unified_df, sentiment_reports = align_sentiment_with_market_data(
                market_df=unified_df,
                sentiment_df=sentiment_data,
                aggregation_method='weighted_mean',
                validate_inputs=validate_inputs
            )
            all_validation_reports.extend(sentiment_reports)
    
    # Step 3: Add technical indicators if requested
    if include_technical_indicators and 'Close' in unified_df.columns:
        with logger.timer("technical_indicators"):
            logger.info("Adding technical indicators")
            unified_df = _add_technical_indicators(unified_df)
    
    # Step 4: Add market regime features if requested
    if include_market_regime and 'Close' in unified_df.columns:
        with logger.timer("market_regime_features"):
            logger.info("Adding market regime features")
            unified_df = _add_market_regime_features(unified_df)
    
    # Step 5: Handle missing values
    with logger.timer("handle_missing_values"):
        logger.info(f"Handling missing values using method: {handle_missing}")
        unified_df = _handle_missing_values(unified_df, method=handle_missing)
    
    # Step 6: Add data quality indicators
    with logger.timer("data_quality_features"):
        logger.info("Adding data quality indicators")
        unified_df = _add_data_quality_features(unified_df)
    
    # Step 7: Feature engineering for time series
    with logger.timer("lag_features"):
        logger.info("Adding lag features")
        unified_df = _add_lag_features(unified_df)
    
    # Step 8: Final cleanup and validation
    with logger.timer("final_cleanup"):
        logger.info("Final cleanup and validation")
        unified_df = _validate_and_clean_dataset(unified_df)
    
    # Step 9: Final validation of unified dataset
    if validate_output:
        logger.info("Running final validation on unified dataset")
        final_report = validate_unified_data(unified_df, f"{symbol}_unified_dataset")
        all_validation_reports.append(final_report)
        
        if final_report.has_critical_issues():
            logger.error("Critical issues found in final unified dataset", symbol=symbol)
            final_report.print_summary()
            raise ValueError(f"Unified dataset for {symbol} has critical validation issues")
        else:
            logger.info(f"Final dataset validation passed with quality score: {final_report.quality_score:.1f}")
    
    # Log final statistics
    logger.info(f"Unified dataset created successfully!",
                symbol=symbol, shape=str(unified_df.shape))
    logger.info(f"Shape: {unified_df.shape}")
    logger.info(f"Features: {len(unified_df.columns)} columns")
    logger.info(f"Memory usage: {unified_df.memory_usage().sum() / 1024**2:.2f} MB")
    
    # Log validation summary
    critical_issues = sum(1 for report in all_validation_reports if report.has_critical_issues())
    avg_quality_score = np.mean([report.quality_score for report in all_validation_reports]) if all_validation_reports else 100.0
    
    logger.info(f"Validation summary: {len(all_validation_reports)} reports, "
                f"{critical_issues} with critical issues, "
                f"average quality score: {avg_quality_score:.1f}")
    
    return unified_df, all_validation_reports


# Helper functions

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features like day of week, month, quarter, etc."""
    df = df.copy()
    
    # Basic temporal features
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    # Trading-specific features
    df['is_monday'] = (df.index.dayofweek == 0).astype(int)
    df['is_friday'] = (df.index.dayofweek == 4).astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    
    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators."""
    df = df.copy()
    
    if 'Close' not in df.columns:
        logger.warning("No 'Close' column found. Skipping technical indicators.")
        return df
    
    # Price-based features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'sma_{window}_ratio'] = df['Close'] / df[f'sma_{window}']
    
    # Exponential moving averages
    for span in [12, 26]:
        df[f'ema_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / df['bb_width']
    
    # RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi'] = calculate_rsi(df['Close'])
    
    # Volume features if available
    if 'Volume' in df.columns:
        df['volume_sma_10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
        df['price_volume'] = df['Close'] * df['Volume']
    
    return df


def _add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features to identify market regimes (trending, ranging, volatile)."""
    df = df.copy()
    
    if 'returns' not in df.columns and 'Close' in df.columns:
        df['returns'] = df['Close'].pct_change()
    
    # Volatility regimes
    df['volatility_20d'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
    df['volatility_60d'] = df['returns'].rolling(window=60).std() * np.sqrt(252)
    df['volatility_ratio'] = df['volatility_20d'] / df['volatility_60d']
    
    # Trend strength
    if 'Close' in df.columns:
        # ADX-like trend strength
        df['trend_strength'] = abs(df['Close'].rolling(window=20).mean() - df['Close'].rolling(window=50).mean()) / df['Close'].rolling(window=20).std()
    
    # Market regime classification
    df['high_volatility'] = (df['volatility_20d'] > df['volatility_20d'].rolling(window=252).mean() + 
                             df['volatility_20d'].rolling(window=252).std()).astype(int)
    
    return df


def _add_lag_features(df: pd.DataFrame, target_cols: List[str] = None, lags: List[int] = None) -> pd.DataFrame:
    """Add lagged features for time series modeling."""
    df = df.copy()
    
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    
    if target_cols is None:
        # Default to returns and key indicators
        target_cols = [col for col in ['returns', 'Close', 'sentiment_score', 'rsi'] if col in df.columns]
    
    for col in target_cols:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df


def _add_data_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features indicating data quality and completeness."""
    df = df.copy()
    
    # Count missing values in each row
    df['missing_count'] = df.isnull().sum(axis=1)
    df['missing_pct'] = df['missing_count'] / len(df.columns)
    
    # Flag rows with interpolated or filled data
    # This is a simplified version - in production, track during fill operations
    
    return df


def _handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """Handle missing values using specified method."""
    df = df.copy()
    
    if method == 'drop':
        df = df.dropna()
    elif method == 'interpolate':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=5)
    elif method == 'forward_fill':
        df = df.fillna(method='ffill', limit=5)
    elif method == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    return df


def _validate_and_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Final validation and cleaning of the dataset."""
    df = df.copy()
    
    # Remove any duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort by date
    df = df.sort_index()
    
    # Remove any infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Remove columns with too many missing values (>50%)
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
    if cols_to_drop:
        logger.warning(f"Dropping columns with >50% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df


# Utility functions for data alignment

def resample_to_business_days(df: pd.DataFrame, fill_method: str = 'ffill') -> pd.DataFrame:
    """Resample data to business days frequency."""
    # Create business day index
    bus_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    
    # Reindex to business days
    df_resampled = df.reindex(bus_days, method=fill_method)
    
    return df_resampled


def align_multiple_dataframes(dfs: List[pd.DataFrame], join: str = 'inner') -> pd.DataFrame:
    """Align multiple dataframes with different time indices."""
    if not dfs:
        return pd.DataFrame()
    
    # Use reduce to sequentially join all dataframes
    aligned_df = reduce(lambda left, right: left.join(right, how=join), dfs)
    
    return aligned_df


# Main execution example
if __name__ == "__main__":
    # Example usage
    print("Data merge module loaded successfully!")
    print("Available functions:")
    print("- merge_market_economic_data()")
    print("- align_sentiment_with_market_data()")
    print("- create_unified_dataset()")