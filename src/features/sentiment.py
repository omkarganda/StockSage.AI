"""
Sentiment Features Module for StockSage.AI

This module provides comprehensive sentiment analysis features for financial time series data.
It processes news sentiment, social media sentiment, and other textual data sources to create
meaningful features for stock price prediction.

Key Features:
- Daily sentiment scores aggregation (multiple methods)
- Sentiment momentum (3-day, 7-day, 30-day trends)
- News volume indicators and attention metrics
- Sentiment volatility and stability measures
- Cross-correlation with market events
- Sentiment regime detection
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Tuple, Callable
import warnings
import logging
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


def aggregate_daily_sentiment(
    sentiment_df: pd.DataFrame,
    date_col: str = 'date',
    sentiment_col: str = 'sentiment_score',
    confidence_col: str = 'confidence',
    source_col: str = 'source',
    aggregation_methods: List[str] = ['mean', 'weighted_mean', 'median', 'count'],
    weight_by_confidence: bool = True,
    weight_by_recency: bool = True,
    recency_decay: float = 0.1
) -> pd.DataFrame:
    """
    Aggregate intraday sentiment scores to daily level.
    
    Parameters:
    -----------
    sentiment_df : pd.DataFrame
        Raw sentiment data with timestamp, scores, and metadata
    date_col : str, default='date'
        Column containing datetime information
    sentiment_col : str, default='sentiment_score'
        Column containing sentiment scores (-1 to 1 or 0 to 1)
    confidence_col : str, default='confidence'
        Column containing confidence scores for each sentiment
    source_col : str, default='source'
        Column indicating source of sentiment (news, twitter, reddit, etc.)
    aggregation_methods : List[str]
        Methods to use for aggregation ['mean', 'weighted_mean', 'median', 'count', 'std']
    weight_by_confidence : bool, default=True
        Whether to weight by confidence scores
    weight_by_recency : bool, default=True
        Whether to give more weight to recent news within the day
    recency_decay : float, default=0.1
        Decay factor for recency weighting (higher = more decay)
    
    Returns:
    --------
    pd.DataFrame
        Daily aggregated sentiment features with datetime index
    """
    df = sentiment_df.copy()
    
    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Create date-only column for grouping
    df['date_only'] = df[date_col].dt.date
    
    # Initialize result dictionary
    daily_features = {}
    
    # Group by date for aggregation
    for date, group in df.groupby('date_only'):
        if len(group) == 0:
            continue
            
        date_features = {}
        
        # Basic aggregations
        if 'mean' in aggregation_methods:
            date_features['sentiment_mean'] = group[sentiment_col].mean()
        
        if 'median' in aggregation_methods:
            date_features['sentiment_median'] = group[sentiment_col].median()
        
        if 'std' in aggregation_methods:
            date_features['sentiment_std'] = group[sentiment_col].std()
        
        if 'count' in aggregation_methods:
            date_features['sentiment_count'] = len(group)
            date_features['news_volume'] = len(group)
        
        # Weighted mean by confidence
        if 'weighted_mean' in aggregation_methods and confidence_col in group.columns:
            if weight_by_confidence:
                weights = group[confidence_col]
                if weight_by_recency:
                    # Add recency weights (more recent = higher weight)
                    hours_from_start = (group[date_col] - group[date_col].min()).dt.total_seconds() / 3600
                    recency_weights = np.exp(-recency_decay * (24 - hours_from_start))
                    weights = weights * recency_weights
                
                if weights.sum() > 0:
                    date_features['sentiment_weighted_mean'] = np.average(
                        group[sentiment_col], weights=weights
                    )
                    date_features['avg_confidence'] = weights.mean()
                else:
                    date_features['sentiment_weighted_mean'] = group[sentiment_col].mean()
                    date_features['avg_confidence'] = 0.5
            else:
                date_features['sentiment_weighted_mean'] = group[sentiment_col].mean()
                date_features['avg_confidence'] = group[confidence_col].mean()
        
        # Source-based aggregations
        if source_col in group.columns:
            source_counts = group[source_col].value_counts()
            date_features['source_diversity'] = len(source_counts)
            date_features['dominant_source_pct'] = source_counts.iloc[0] / len(group) if len(source_counts) > 0 else 0
            
            # Source-specific sentiments
            for source in group[source_col].unique():
                source_data = group[group[source_col] == source]
                date_features[f'sentiment_{source}_mean'] = source_data[sentiment_col].mean()
                date_features[f'sentiment_{source}_count'] = len(source_data)
        
        # Sentiment distribution features
        date_features['sentiment_min'] = group[sentiment_col].min()
        date_features['sentiment_max'] = group[sentiment_col].max()
        date_features['sentiment_range'] = date_features['sentiment_max'] - date_features['sentiment_min']
        
        # Sentiment polarity distribution
        positive_mask = group[sentiment_col] > 0.1  # Threshold for positive
        negative_mask = group[sentiment_col] < -0.1  # Threshold for negative
        neutral_mask = (~positive_mask) & (~negative_mask)
        
        date_features['positive_sentiment_count'] = positive_mask.sum()
        date_features['negative_sentiment_count'] = negative_mask.sum()
        date_features['neutral_sentiment_count'] = neutral_mask.sum()
        
        date_features['positive_sentiment_pct'] = positive_mask.mean()
        date_features['negative_sentiment_pct'] = negative_mask.mean()
        date_features['neutral_sentiment_pct'] = neutral_mask.mean()
        
        # Sentiment intensity (distance from neutral)
        date_features['sentiment_intensity'] = np.abs(group[sentiment_col]).mean()
        date_features['sentiment_extremeness'] = (np.abs(group[sentiment_col]) > 0.5).mean()
        
        # Time-based features within the day
        if len(group) > 1:
            # Intraday sentiment trend
            group_sorted = group.sort_values(date_col)
            first_half = group_sorted.iloc[:len(group_sorted)//2][sentiment_col].mean()
            second_half = group_sorted.iloc[len(group_sorted)//2:][sentiment_col].mean()
            date_features['intraday_sentiment_trend'] = second_half - first_half
            
            # Sentiment momentum within day
            time_weights = np.linspace(0, 1, len(group_sorted))
            date_features['intraday_momentum'] = np.corrcoef(
                group_sorted[sentiment_col], time_weights
            )[0, 1] if len(group_sorted) > 2 else 0
        
        daily_features[pd.to_datetime(date)] = date_features
    
    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(daily_features, orient='index')
    result_df.index.name = 'date'
    
    # Fill missing values
    result_df = result_df.fillna(0)
    
    logger.info(f"Aggregated sentiment data to {len(result_df)} daily observations")
    logger.info(f"Created {len(result_df.columns)} sentiment features")
    
    return result_df


def calculate_sentiment_momentum(
    sentiment_df: pd.DataFrame,
    sentiment_col: str = 'sentiment_mean',
    windows: List[int] = [3, 7, 14, 30],
    include_acceleration: bool = True,
    include_volatility: bool = True
) -> pd.DataFrame:
    """
    Calculate sentiment momentum features over various time windows.
    
    Parameters:
    -----------
    sentiment_df : pd.DataFrame
        Daily sentiment data with datetime index
    sentiment_col : str, default='sentiment_mean'
        Column containing daily sentiment scores
    windows : List[int]
        Time windows for momentum calculation (in days)
    include_acceleration : bool, default=True
        Whether to include sentiment acceleration features
    include_volatility : bool, default=True
        Whether to include sentiment volatility features
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with sentiment momentum features
    """
    df = sentiment_df.copy()
    
    if sentiment_col not in df.columns:
        raise ValueError(f"Column '{sentiment_col}' not found in DataFrame")
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    for window in windows:
        # Simple momentum (difference from N days ago)
        df[f'sentiment_momentum_{window}d'] = df[sentiment_col] - df[sentiment_col].shift(window)
        
        # Rate of change momentum
        df[f'sentiment_roc_{window}d'] = (
            (df[sentiment_col] - df[sentiment_col].shift(window)) / 
            np.abs(df[sentiment_col].shift(window)).replace(0, 0.001) * 100
        )
        
        # Moving average momentum
        ma_sentiment = df[sentiment_col].rolling(window=window, min_periods=1).mean()
        df[f'sentiment_ma_{window}d'] = ma_sentiment
        df[f'sentiment_ma_momentum_{window}d'] = df[sentiment_col] - ma_sentiment
        
        # Trend strength (correlation with time)
        def trend_strength(series):
            if len(series) < 3:
                return 0
            time_index = np.arange(len(series))
            correlation = np.corrcoef(series.values, time_index)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        
        df[f'sentiment_trend_{window}d'] = df[sentiment_col].rolling(
            window=window, min_periods=3
        ).apply(trend_strength)
        
        # Exponentially weighted momentum
        alpha = 2 / (window + 1)
        ema_sentiment = df[sentiment_col].ewm(alpha=alpha, adjust=False).mean()
        df[f'sentiment_ema_{window}d'] = ema_sentiment
        df[f'sentiment_ema_momentum_{window}d'] = df[sentiment_col] - ema_sentiment
        
        # Volatility features
        if include_volatility:
            df[f'sentiment_volatility_{window}d'] = df[sentiment_col].rolling(
                window=window, min_periods=1
            ).std()
            
            df[f'sentiment_volatility_norm_{window}d'] = (
                df[f'sentiment_volatility_{window}d'] / 
                (np.abs(ma_sentiment).replace(0, 0.001))
            )
        
        # Regime detection (bullish/bearish sentiment periods)
        rolling_mean = df[sentiment_col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[sentiment_col].rolling(window=window, min_periods=1).std()
        
        df[f'sentiment_regime_{window}d'] = np.where(
            df[sentiment_col] > rolling_mean + 0.5 * rolling_std, 1,  # Bullish
            np.where(df[sentiment_col] < rolling_mean - 0.5 * rolling_std, -1, 0)  # Bearish / Neutral
        )
        
        # Momentum persistence (how long has momentum been in same direction)
        momentum_direction = np.sign(df[f'sentiment_momentum_{window}d'])
        momentum_changes = momentum_direction != momentum_direction.shift(1)
        momentum_groups = momentum_changes.cumsum()
        df[f'sentiment_momentum_persistence_{window}d'] = df.groupby(momentum_groups).cumcount() + 1
    
    # Cross-window momentum comparisons
    if len(windows) >= 2:
        short_window, long_window = sorted(windows)[:2]
        
        # Fast vs slow momentum
        df['sentiment_momentum_fast_vs_slow'] = (
            df[f'sentiment_momentum_{short_window}d'] - 
            df[f'sentiment_momentum_{long_window}d']
        )
        
        # Moving average crossover signals
        df['sentiment_ma_bullish_cross'] = (
            (df[f'sentiment_ma_{short_window}d'] > df[f'sentiment_ma_{long_window}d']) &
            (df[f'sentiment_ma_{short_window}d'].shift(1) <= df[f'sentiment_ma_{long_window}d'].shift(1))
        ).astype(int)
        
        df['sentiment_ma_bearish_cross'] = (
            (df[f'sentiment_ma_{short_window}d'] < df[f'sentiment_ma_{long_window}d']) &
            (df[f'sentiment_ma_{short_window}d'].shift(1) >= df[f'sentiment_ma_{long_window}d'].shift(1))
        ).astype(int)
    
    # Acceleration features
    if include_acceleration:
        # Sentiment acceleration (second derivative)
        df['sentiment_acceleration'] = df[sentiment_col].diff().diff()
        
        # Momentum acceleration
        for window in windows:
            if f'sentiment_momentum_{window}d' in df.columns:
                df[f'sentiment_momentum_acceleration_{window}d'] = (
                    df[f'sentiment_momentum_{window}d'].diff()
                )
    
    # Relative sentiment position
    for window in windows:
        rolling_min = df[sentiment_col].rolling(window=window, min_periods=1).min()
        rolling_max = df[sentiment_col].rolling(window=window, min_periods=1).max()
        df[f'sentiment_position_{window}d'] = (
            (df[sentiment_col] - rolling_min) / (rolling_max - rolling_min).replace(0, 1)
        )
    
    logger.info(f"Added sentiment momentum features for windows: {windows}")
    return df


def calculate_news_volume_indicators(
    sentiment_df: pd.DataFrame,
    volume_col: str = 'news_volume',
    attention_cols: List[str] = ['sentiment_count', 'source_diversity'],
    baseline_windows: List[int] = [7, 30, 90],
    include_surprise_metrics: bool = True
) -> pd.DataFrame:
    """
    Calculate news volume and attention indicators.
    
    Parameters:
    -----------
    sentiment_df : pd.DataFrame
        Daily sentiment data with datetime index
    volume_col : str, default='news_volume'
        Column containing daily news volume counts
    attention_cols : List[str]
        Columns to use for attention metrics
    baseline_windows : List[int]
        Windows for baseline comparisons (in days)
    include_surprise_metrics : bool, default=True
        Whether to include surprise/shock metrics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with news volume indicators
    """
    df = sentiment_df.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    # Create volume column if not exists
    if volume_col not in df.columns:
        if 'sentiment_count' in df.columns:
            df[volume_col] = df['sentiment_count']
        else:
            logger.warning("No volume column found. Creating dummy volume.")
            df[volume_col] = 1
    
    # Fill missing values with 0
    df[volume_col] = df[volume_col].fillna(0)
    
    # Basic volume features
    for window in baseline_windows:
        # Moving averages
        df[f'news_volume_ma_{window}d'] = df[volume_col].rolling(
            window=window, min_periods=1
        ).mean()
        
        # Volume ratio vs baseline
        df[f'news_volume_ratio_{window}d'] = (
            df[volume_col] / df[f'news_volume_ma_{window}d'].replace(0, 1)
        )
        
        # Volume percentile
        df[f'news_volume_percentile_{window}d'] = df[volume_col].rolling(
            window=window, min_periods=1
        ).rank(pct=True)
        
        # Volume above average
        df[f'news_volume_above_avg_{window}d'] = (
            df[volume_col] > df[f'news_volume_ma_{window}d']
        ).astype(int)
    
    # Volume momentum
    for lag in [1, 3, 7]:
        df[f'news_volume_momentum_{lag}d'] = df[volume_col] - df[volume_col].shift(lag)
        df[f'news_volume_roc_{lag}d'] = (
            (df[volume_col] - df[volume_col].shift(lag)) / 
            df[volume_col].shift(lag).replace(0, 1) * 100
        )
    
    # Volume volatility
    for window in [7, 14, 30]:
        df[f'news_volume_volatility_{window}d'] = df[volume_col].rolling(
            window=window, min_periods=1
        ).std()
        
        # Coefficient of variation
        df[f'news_volume_cv_{window}d'] = (
            df[f'news_volume_volatility_{window}d'] / 
            df[f'news_volume_ma_{window}d'].replace(0, 1)
        )
    
    # Volume spikes and dry spells
    baseline_ma = df[f'news_volume_ma_{baseline_windows[1]}d']  # Use medium-term baseline
    baseline_std = df[volume_col].rolling(window=baseline_windows[1], min_periods=1).std()
    
    # Volume spike detection
    df['news_volume_spike'] = (
        df[volume_col] > baseline_ma + 2 * baseline_std
    ).astype(int)
    
    df['news_volume_major_spike'] = (
        df[volume_col] > baseline_ma + 3 * baseline_std
    ).astype(int)
    
    # Volume dry spell detection
    df['news_volume_dry_spell'] = (
        df[volume_col] < baseline_ma - baseline_std
    ).astype(int)
    
    # Zero news days
    df['zero_news_day'] = (df[volume_col] == 0).astype(int)
    
    # Consecutive patterns
    def consecutive_count(series):
        """Count consecutive True values."""
        return series.groupby((series != series.shift()).cumsum()).cumsum()
    
    df['consecutive_high_volume'] = consecutive_count(df['news_volume_spike'])
    df['consecutive_low_volume'] = consecutive_count(df['news_volume_dry_spell'])
    
    # Attention metrics
    for col in attention_cols:
        if col in df.columns:
            # Attention intensity relative to volume
            df[f'{col}_per_volume'] = df[col] / df[volume_col].replace(0, 1)
            
            # Attention momentum
            for window in [3, 7]:
                df[f'{col}_momentum_{window}d'] = df[col] - df[col].shift(window)
                
            # Attention concentration
            for window in [7, 14]:
                df[f'{col}_concentration_{window}d'] = (
                    df[col] / df[col].rolling(window=window, min_periods=1).mean()
                )
    
    # Weekend/weekday effects
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    
    # Surprise metrics
    if include_surprise_metrics:
        # News surprise (volume much higher than expected)
        expected_volume = df[f'news_volume_ma_{baseline_windows[0]}d']
        volume_residual = df[volume_col] - expected_volume
        volume_std = df[volume_col].rolling(window=baseline_windows[0], min_periods=1).std()
        
        df['news_surprise_score'] = volume_residual / volume_std.replace(0, 1)
        df['positive_news_surprise'] = (df['news_surprise_score'] > 2).astype(int)
        df['negative_news_surprise'] = (df['news_surprise_score'] < -1).astype(int)
        
        # Cumulative attention
        df['cumulative_attention_7d'] = df[volume_col].rolling(window=7, min_periods=1).sum()
        df['cumulative_attention_30d'] = df[volume_col].rolling(window=30, min_periods=1).sum()
        
        # Attention decay (weighted recent attention)
        decay_weights = np.exp(-0.1 * np.arange(7))[::-1]  # More weight to recent days
        df['attention_decay_score'] = df[volume_col].rolling(window=7, min_periods=1).apply(
            lambda x: np.average(x, weights=decay_weights[:len(x)])
        )
    
    # Regime indicators
    median_volume = df[volume_col].rolling(window=90, min_periods=1).median()
    df['high_attention_regime'] = (df[volume_col] > median_volume * 1.5).astype(int)
    df['low_attention_regime'] = (df[volume_col] < median_volume * 0.5).astype(int)
    
    logger.info("Added news volume and attention indicators")
    return df


def add_all_sentiment_features(
    sentiment_df: pd.DataFrame,
    raw_sentiment_df: Optional[pd.DataFrame] = None,
    sentiment_col: str = 'sentiment_mean',
    volume_col: str = 'news_volume',
    include_momentum: bool = True,
    include_volume_indicators: bool = True,
    momentum_windows: List[int] = [3, 7, 14, 30],
    volume_baselines: List[int] = [7, 30, 90]
) -> pd.DataFrame:
    """
    Add all sentiment features to the DataFrame.
    
    Parameters:
    -----------
    sentiment_df : pd.DataFrame
        Daily aggregated sentiment data
    raw_sentiment_df : pd.DataFrame, optional
        Raw sentiment data for additional processing
    sentiment_col : str, default='sentiment_mean'
        Primary sentiment column
    volume_col : str, default='news_volume'
        News volume column
    include_momentum : bool, default=True
        Whether to include momentum features
    include_volume_indicators : bool, default=True
        Whether to include volume indicators
    momentum_windows : List[int]
        Windows for momentum calculations
    volume_baselines : List[int]
        Baseline windows for volume metrics
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all sentiment features
    """
    logger.info("Adding all sentiment features...")
    
    df = sentiment_df.copy()
    
    # Re-aggregate raw sentiment if provided and not already aggregated
    if raw_sentiment_df is not None and len(raw_sentiment_df) > len(df) * 2:
        logger.info("Re-aggregating raw sentiment data...")
        aggregated = aggregate_daily_sentiment(raw_sentiment_df)
        # Merge with existing data
        df = df.combine_first(aggregated)
    
    # Add momentum features
    if include_momentum:
        df = calculate_sentiment_momentum(
            df, 
            sentiment_col=sentiment_col,
            windows=momentum_windows,
            include_acceleration=True,
            include_volatility=True
        )
    
    # Add volume indicators
    if include_volume_indicators:
        df = calculate_news_volume_indicators(
            df,
            volume_col=volume_col,
            baseline_windows=volume_baselines,
            include_surprise_metrics=True
        )
    
    # Cross-feature interactions
    if sentiment_col in df.columns and volume_col in df.columns:
        # Sentiment-weighted by volume
        df['sentiment_volume_weighted'] = df[sentiment_col] * np.log1p(df[volume_col])
        
        # Sentiment intensity when high volume
        df['sentiment_intensity_high_volume'] = np.where(
            df[volume_col] > df[volume_col].median(),
            np.abs(df[sentiment_col]),
            0
        )
        
        # Volume-adjusted sentiment
        volume_percentile = df[volume_col].rolling(window=30, min_periods=1).rank(pct=True)
        df['sentiment_volume_adjusted'] = df[sentiment_col] * volume_percentile
    
    # Quality metrics
    sentiment_cols = [col for col in df.columns if 'sentiment' in col and col != sentiment_col]
    df['sentiment_feature_count'] = len(sentiment_cols)
    df['sentiment_data_quality'] = (
        df[sentiment_cols].notna().sum(axis=1) / len(sentiment_cols)
    )
    
    # Summary statistics
    sentiment_feature_cols = [col for col in df.columns if any(term in col for term in 
                             ['sentiment', 'news_volume', 'attention'])]
    
    logger.info(f"Added {len(sentiment_feature_cols)} sentiment features")
    logger.info(f"DataFrame shape: {df.shape}")
    
    return df


def calculate_sentiment_market_alignment(
    sentiment_df: pd.DataFrame,
    market_df: pd.DataFrame,
    sentiment_col: str = 'sentiment_mean',
    price_col: str = 'Close',
    return_col: str = 'returns',
    windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate alignment metrics between sentiment and market movements.
    
    Parameters:
    -----------
    sentiment_df : pd.DataFrame
        Sentiment data with datetime index
    market_df : pd.DataFrame
        Market data with datetime index
    sentiment_col : str, default='sentiment_mean'
        Sentiment column to analyze
    price_col : str, default='Close'
        Price column for market data
    return_col : str, default='returns'
        Returns column (will be calculated if not present)
    windows : List[int]
        Rolling windows for correlation analysis
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with alignment features
    """
    # Merge sentiment and market data
    combined_df = pd.merge(
        sentiment_df[[sentiment_col]], 
        market_df[[price_col]], 
        left_index=True, right_index=True, how='inner'
    )
    
    # Calculate returns if not present
    if return_col not in combined_df.columns:
        combined_df[return_col] = combined_df[price_col].pct_change()
    
    # Rolling correlations
    for window in windows:
        combined_df[f'sentiment_return_corr_{window}d'] = (
            combined_df[sentiment_col].rolling(window=window)
            .corr(combined_df[return_col])
        )
    
    # Lead-lag relationships
    for lag in [1, 2, 3]:
        combined_df[f'sentiment_lead_corr_{lag}d'] = (
            combined_df[sentiment_col].rolling(window=20)
            .corr(combined_df[return_col].shift(-lag))
        )
        
        combined_df[f'sentiment_lag_corr_{lag}d'] = (
            combined_df[sentiment_col].rolling(window=20)
            .corr(combined_df[return_col].shift(lag))
        )
    
    # Directional agreement
    sentiment_direction = np.sign(combined_df[sentiment_col])
    return_direction = np.sign(combined_df[return_col])
    
    for window in windows:
        agreement = (sentiment_direction == return_direction).rolling(window=window).mean()
        combined_df[f'sentiment_direction_agreement_{window}d'] = agreement
    
    logger.info("Added sentiment-market alignment features")
    return combined_df