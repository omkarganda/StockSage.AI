"""
Technical Indicators Module for StockSage.AI

This module provides comprehensive technical analysis indicators for financial time series data.
All functions accept pandas DataFrames with OHLCV data and return the DataFrame with new indicator columns.

Key Features:
- Moving averages (SMA, EMA, WMA)
- Momentum indicators (RSI, Stochastic, ROC)
- Trend indicators (MACD, ADX, Aroon)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, VWAP, Volume Profile)
- Price momentum and pattern recognition features
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Tuple
import warnings
import logging

# Configure logging
logger = logging.getLogger(__name__)


def calculate_moving_averages(
    df: pd.DataFrame,
    price_col: str = 'Close',
    sma_periods: List[int] = [5, 10, 20, 50, 100, 200],
    ema_periods: List[int] = [12, 26, 50],
    wma_periods: List[int] = [10, 20]
) -> pd.DataFrame:
    """
    Calculate various moving averages.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with datetime index
    price_col : str, default='Close'
        Column to calculate moving averages for
    sma_periods : List[int]
        Periods for Simple Moving Averages
    ema_periods : List[int]
        Periods for Exponential Moving Averages
    wma_periods : List[int]
        Periods for Weighted Moving Averages
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional moving average columns
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    # Simple Moving Averages
    for period in sma_periods:
        df[f'SMA_{period}'] = df[price_col].rolling(window=period, min_periods=1).mean()
        df[f'SMA_{period}_ratio'] = df[price_col] / df[f'SMA_{period}']
        df[f'SMA_{period}_distance'] = (df[price_col] - df[f'SMA_{period}']) / df[f'SMA_{period}'] * 100
    
    # Exponential Moving Averages
    for period in ema_periods:
        df[f'EMA_{period}'] = df[price_col].ewm(span=period, adjust=False).mean()
        df[f'EMA_{period}_ratio'] = df[price_col] / df[f'EMA_{period}']
        df[f'EMA_{period}_distance'] = (df[price_col] - df[f'EMA_{period}']) / df[f'EMA_{period}'] * 100
    
    # Weighted Moving Averages
    def wma(prices, period):
        weights = np.arange(1, period + 1)
        return prices.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    for period in wma_periods:
        df[f'WMA_{period}'] = wma(df[price_col], period)
        df[f'WMA_{period}_ratio'] = df[price_col] / df[f'WMA_{period}']
    
    # Moving average convergence/divergence signals
    if 20 in sma_periods and 50 in sma_periods:
        df['MA_golden_cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        df['MA_death_cross'] = (df['SMA_20'] < df['SMA_50']).astype(int)
    
    if 12 in ema_periods and 26 in ema_periods:
        df['EMA_bull_signal'] = (df['EMA_12'] > df['EMA_26']).astype(int)
        df['EMA_bear_signal'] = (df['EMA_12'] < df['EMA_26']).astype(int)
    
    logger.info(f"Added {len([c for c in df.columns if any(ma in c for ma in ['SMA', 'EMA', 'WMA'])])} moving average features")
    return df


def calculate_rsi(
    df: pd.DataFrame,
    price_col: str = 'Close',
    periods: List[int] = [14, 21, 30],
    overbought: float = 70,
    oversold: float = 30
) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI) and related features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with datetime index
    price_col : str, default='Close'
        Column to calculate RSI for
    periods : List[int]
        RSI calculation periods
    overbought : float, default=70
        Overbought threshold
    oversold : float, default=30
        Oversold threshold
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with RSI columns
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    # Calculate price changes
    delta = df[price_col].diff()
    
    for period in periods:
        # Calculate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        df[f'RSI_{period}'] = rsi
        df[f'RSI_{period}_overbought'] = (rsi > overbought).astype(int)
        df[f'RSI_{period}_oversold'] = (rsi < oversold).astype(int)
        df[f'RSI_{period}_normalized'] = (rsi - 50) / 50  # Normalize to [-1, 1]
        
        # RSI momentum
        df[f'RSI_{period}_momentum'] = rsi.diff()
        df[f'RSI_{period}_trend'] = np.where(
            rsi.diff() > 0, 1,
            np.where(rsi.diff() < 0, -1, 0)
        )
    
    # RSI divergence signals (simplified)
    if 14 in periods:
        df['RSI_bullish_divergence'] = (
            (df[price_col] < df[price_col].shift(5)) & 
            (df['RSI_14'] > df['RSI_14'].shift(5))
        ).astype(int)
        
        df['RSI_bearish_divergence'] = (
            (df[price_col] > df[price_col].shift(5)) & 
            (df['RSI_14'] < df['RSI_14'].shift(5))
        ).astype(int)
    
    logger.info(f"Added RSI features for periods: {periods}")
    return df


def calculate_macd(
    df: pd.DataFrame,
    price_col: str = 'Close',
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) and related signals.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with datetime index
    price_col : str, default='Close'
        Column to calculate MACD for
    fast_period : int, default=12
        Fast EMA period
    slow_period : int, default=26
        Slow EMA period
    signal_period : int, default=9
        Signal line EMA period
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with MACD columns
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    # Calculate EMAs
    ema_fast = df[price_col].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    df['MACD'] = ema_fast - ema_slow
    
    # Calculate Signal line
    df['MACD_signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    
    # Calculate MACD histogram
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # MACD signals
    df['MACD_bullish_crossover'] = (
        (df['MACD'] > df['MACD_signal']) & 
        (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))
    ).astype(int)
    
    df['MACD_bearish_crossover'] = (
        (df['MACD'] < df['MACD_signal']) & 
        (df['MACD'].shift(1) >= df['MACD_signal'].shift(1))
    ).astype(int)
    
    # Zero line crossovers
    df['MACD_zero_cross_up'] = (
        (df['MACD'] > 0) & (df['MACD'].shift(1) <= 0)
    ).astype(int)
    
    df['MACD_zero_cross_down'] = (
        (df['MACD'] < 0) & (df['MACD'].shift(1) >= 0)
    ).astype(int)
    
    # MACD momentum
    df['MACD_momentum'] = df['MACD'].diff()
    df['MACD_histogram_momentum'] = df['MACD_histogram'].diff()
    
    # Normalized MACD (relative to price)
    df['MACD_normalized'] = df['MACD'] / df[price_col] * 100
    
    logger.info(f"Added MACD features with periods: {fast_period}, {slow_period}, {signal_period}")
    return df


def calculate_bollinger_bands(
    df: pd.DataFrame,
    price_col: str = 'Close',
    period: int = 20,
    std_dev: float = 2.0,
    additional_bands: List[float] = [1.0, 2.5]
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands and related features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with datetime index
    price_col : str, default='Close'
        Column to calculate Bollinger Bands for
    period : int, default=20
        Moving average period
    std_dev : float, default=2.0
        Standard deviation multiplier for main bands
    additional_bands : List[float]
        Additional standard deviation multipliers
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Bollinger Bands columns
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    # Calculate middle band (SMA)
    df[f'BB_middle_{period}'] = df[price_col].rolling(window=period, min_periods=1).mean()
    
    # Calculate standard deviation
    rolling_std = df[price_col].rolling(window=period, min_periods=1).std()
    
    # Main Bollinger Bands
    df[f'BB_upper_{period}'] = df[f'BB_middle_{period}'] + (std_dev * rolling_std)
    df[f'BB_lower_{period}'] = df[f'BB_middle_{period}'] - (std_dev * rolling_std)
    
    # Band width and position
    df[f'BB_width_{period}'] = df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']
    df[f'BB_width_normalized_{period}'] = df[f'BB_width_{period}'] / df[f'BB_middle_{period}']
    
    df[f'BB_position_{period}'] = (
        (df[price_col] - df[f'BB_lower_{period}']) / 
        (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'])
    )
    
    # Distance from bands
    df[f'BB_upper_distance_{period}'] = (df[f'BB_upper_{period}'] - df[price_col]) / df[price_col] * 100
    df[f'BB_lower_distance_{period}'] = (df[price_col] - df[f'BB_lower_{period}']) / df[price_col] * 100
    
    # Band touches and breaks
    df[f'BB_upper_touch_{period}'] = (df[price_col] >= df[f'BB_upper_{period}']).astype(int)
    df[f'BB_lower_touch_{period}'] = (df[price_col] <= df[f'BB_lower_{period}']).astype(int)
    
    # Squeeze detection (narrow bands)
    df[f'BB_squeeze_{period}'] = (
        df[f'BB_width_normalized_{period}'] < 
        df[f'BB_width_normalized_{period}'].rolling(window=50).quantile(0.2)
    ).astype(int)
    
    # Additional bands
    for std_mult in additional_bands:
        df[f'BB_upper_{period}_{std_mult}'] = df[f'BB_middle_{period}'] + (std_mult * rolling_std)
        df[f'BB_lower_{period}_{std_mult}'] = df[f'BB_middle_{period}'] - (std_mult * rolling_std)
    
    # %B indicator
    df[f'BB_percent_b_{period}'] = (
        (df[price_col] - df[f'BB_lower_{period}']) / 
        (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'])
    )
    
    logger.info(f"Added Bollinger Bands features with period: {period}")
    return df


def calculate_volume_indicators(
    df: pd.DataFrame,
    price_col: str = 'Close',
    volume_col: str = 'Volume',
    high_col: str = 'High',
    low_col: str = 'Low'
) -> pd.DataFrame:
    """
    Calculate volume-based indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with datetime index
    price_col : str, default='Close'
        Price column name
    volume_col : str, default='Volume'
        Volume column name
    high_col : str, default='High'
        High price column name
    low_col : str, default='Low'
        Low price column name
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with volume indicator columns
    """
    df = df.copy()
    
    required_cols = [price_col, volume_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}. Skipping volume indicators.")
        return df
    
    # Volume moving averages
    for period in [10, 20, 50]:
        df[f'Volume_SMA_{period}'] = df[volume_col].rolling(window=period, min_periods=1).mean()
        df[f'Volume_ratio_{period}'] = df[volume_col] / df[f'Volume_SMA_{period}']
    
    # On-Balance Volume (OBV)
    df['OBV'] = (df[volume_col] * np.where(df[price_col].diff() > 0, 1, 
                 np.where(df[price_col].diff() < 0, -1, 0))).cumsum()
    
    # OBV trend
    df['OBV_trend'] = df['OBV'].rolling(window=10).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
    )
    
    # Volume-Weighted Average Price (VWAP)
    if high_col in df.columns and low_col in df.columns:
        typical_price = (df[high_col] + df[low_col] + df[price_col]) / 3
        df['VWAP'] = (typical_price * df[volume_col]).cumsum() / df[volume_col].cumsum()
        df['VWAP_distance'] = (df[price_col] - df['VWAP']) / df['VWAP'] * 100
    
    # Price-Volume Trend (PVT)
    df['PVT'] = (df[volume_col] * df[price_col].pct_change()).cumsum()
    df['PVT_signal'] = df['PVT'].ewm(span=9).mean()
    
    # Volume oscillator
    df['Volume_oscillator'] = (
        (df['Volume_SMA_10'] - df['Volume_SMA_20']) / df['Volume_SMA_20'] * 100
    )
    
    # Accumulation/Distribution Line
    if high_col in df.columns and low_col in df.columns:
        clv = ((df[price_col] - df[low_col]) - (df[high_col] - df[price_col])) / (df[high_col] - df[low_col])
        clv = clv.fillna(0)  # Handle division by zero
        df['AD_line'] = (clv * df[volume_col]).cumsum()
    
    # Volume spikes
    df['Volume_spike'] = (df[volume_col] > df['Volume_SMA_20'] * 2).astype(int)
    df['Volume_dry_up'] = (df[volume_col] < df['Volume_SMA_20'] * 0.5).astype(int)
    
    # Volume momentum
    df['Volume_momentum'] = df[volume_col].pct_change()
    df['Volume_acceleration'] = df['Volume_momentum'].diff()
    
    logger.info("Added volume-based indicators")
    return df


def calculate_price_momentum(
    df: pd.DataFrame,
    price_col: str = 'Close',
    high_col: str = 'High',
    low_col: str = 'Low'
) -> pd.DataFrame:
    """
    Calculate price momentum and pattern recognition features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with datetime index
    price_col : str, default='Close'
        Price column name
    high_col : str, default='High'
        High price column name
    low_col : str, default='Low'
        Low price column name
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with momentum columns
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    # Rate of Change (ROC)
    for period in [1, 5, 10, 20]:
        df[f'ROC_{period}'] = (
            (df[price_col] - df[price_col].shift(period)) / df[price_col].shift(period) * 100
        )
    
    # Momentum
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df[price_col] - df[price_col].shift(period)
        df[f'Momentum_{period}_normalized'] = df[f'Momentum_{period}'] / df[price_col] * 100
    
    # Price acceleration
    df['Price_acceleration'] = df[price_col].diff().diff()
    df['Returns_acceleration'] = df[price_col].pct_change().diff()
    
    # Volatility measures
    df['Price_volatility_10'] = df[price_col].pct_change().rolling(window=10).std() * np.sqrt(252)
    df['Price_volatility_20'] = df[price_col].pct_change().rolling(window=20).std() * np.sqrt(252)
    df['Volatility_ratio'] = df['Price_volatility_10'] / df['Price_volatility_20']
    
    # Average True Range (ATR) if High/Low available
    if high_col in df.columns and low_col in df.columns:
        high_low = df[high_col] - df[low_col]
        high_close = np.abs(df[high_col] - df[price_col].shift())
        low_close = np.abs(df[low_col] - df[price_col].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = true_range.rolling(window=14, min_periods=1).mean()
        df['ATR_normalized'] = df['ATR_14'] / df[price_col] * 100
    
    # Support and resistance levels (simplified)
    df['Recent_high_20'] = df[price_col].rolling(window=20).max()
    df['Recent_low_20'] = df[price_col].rolling(window=20).min()
    df['Distance_from_high'] = (df['Recent_high_20'] - df[price_col]) / df[price_col] * 100
    df['Distance_from_low'] = (df[price_col] - df['Recent_low_20']) / df[price_col] * 100
    
    # Price position in recent range
    df['Price_position_20'] = (
        (df[price_col] - df['Recent_low_20']) / 
        (df['Recent_high_20'] - df['Recent_low_20'])
    )
    
    # Trend strength
    df['Trend_strength_10'] = (
        (df[price_col] - df[price_col].shift(10)) / 
        df[price_col].rolling(window=10).std()
    )
    
    # Price gaps
    df['Gap_up'] = (df[price_col] > df[price_col].shift(1) * 1.02).astype(int)
    df['Gap_down'] = (df[price_col] < df[price_col].shift(1) * 0.98).astype(int)
    
    # Higher highs and lower lows
    df['Higher_high'] = (
        (df[price_col] > df[price_col].shift(1)) & 
        (df[price_col].shift(1) > df[price_col].shift(2))
    ).astype(int)
    
    df['Lower_low'] = (
        (df[price_col] < df[price_col].shift(1)) & 
        (df[price_col].shift(1) < df[price_col].shift(2))
    ).astype(int)
    
    logger.info("Added price momentum and pattern features")
    return df


def add_all_technical_indicators(
    df: pd.DataFrame,
    price_col: str = 'Close',
    volume_col: str = 'Volume',
    high_col: str = 'High',
    low_col: str = 'Low',
    include_advanced: bool = True
) -> pd.DataFrame:
    """
    Add all technical indicators to the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with datetime index
    price_col : str, default='Close'
        Price column name
    volume_col : str, default='Volume'
        Volume column name
    high_col : str, default='High'
        High price column name
    low_col : str, default='Low'
        Low price column name
    include_advanced : bool, default=True
        Whether to include advanced indicators (more computationally intensive)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with all technical indicators
    """
    logger.info("Adding all technical indicators...")
    
    # Core indicators
    df = calculate_moving_averages(df, price_col)
    df = calculate_rsi(df, price_col)
    df = calculate_macd(df, price_col)
    df = calculate_bollinger_bands(df, price_col)
    
    # Volume indicators (if volume data available)
    if volume_col in df.columns:
        df = calculate_volume_indicators(df, price_col, volume_col, high_col, low_col)
    
    # Momentum indicators
    df = calculate_price_momentum(df, price_col, high_col, low_col)
    
    # Advanced indicators
    if include_advanced:
        # Stochastic Oscillator
        if high_col in df.columns and low_col in df.columns:
            df = _calculate_stochastic(df, high_col, low_col, price_col)
        
        # Williams %R
        if high_col in df.columns and low_col in df.columns:
            df = _calculate_williams_r(df, high_col, low_col, price_col)
        
        # Commodity Channel Index (CCI)
        if high_col in df.columns and low_col in df.columns:
            df = _calculate_cci(df, high_col, low_col, price_col)
    
    # Clean up NaN values that might have been created
    # Replace infinite values with NaN first
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with forward fill, then backward fill, then 0
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Summary statistics
    technical_cols = [col for col in df.columns if any(indicator in col for indicator in 
                     ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'Volume', 'OBV', 'ROC', 'Momentum'])]
    
    logger.info(f"Added {len(technical_cols)} technical indicator features")
    logger.info(f"DataFrame shape: {df.shape}")
    
    return df


def _calculate_stochastic(df: pd.DataFrame, high_col: str, low_col: str, price_col: str, 
                         k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    df = df.copy()
    
    highest_high = df[high_col].rolling(window=k_period, min_periods=1).max()
    lowest_low = df[low_col].rolling(window=k_period, min_periods=1).min()
    
    df['Stoch_K'] = (df[price_col] - lowest_low) / (highest_high - lowest_low) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period, min_periods=1).mean()
    
    df['Stoch_overbought'] = (df['Stoch_K'] > 80).astype(int)
    df['Stoch_oversold'] = (df['Stoch_K'] < 20).astype(int)
    
    return df


def _calculate_williams_r(df: pd.DataFrame, high_col: str, low_col: str, price_col: str, 
                         period: int = 14) -> pd.DataFrame:
    """Calculate Williams %R."""
    df = df.copy()
    
    highest_high = df[high_col].rolling(window=period, min_periods=1).max()
    lowest_low = df[low_col].rolling(window=period, min_periods=1).min()
    
    df['Williams_R'] = (highest_high - df[price_col]) / (highest_high - lowest_low) * -100
    df['Williams_R_overbought'] = (df['Williams_R'] > -20).astype(int)
    df['Williams_R_oversold'] = (df['Williams_R'] < -80).astype(int)
    
    return df


def _calculate_cci(df: pd.DataFrame, high_col: str, low_col: str, price_col: str, 
                  period: int = 20) -> pd.DataFrame:
    """Calculate Commodity Channel Index (CCI)."""
    df = df.copy()
    
    typical_price = (df[high_col] + df[low_col] + df[price_col]) / 3
    sma_tp = typical_price.rolling(window=period, min_periods=1).mean()
    mad = typical_price.rolling(window=period, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean()))
    )
    
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
    df['CCI_overbought'] = (df['CCI'] > 100).astype(int)
    df['CCI_oversold'] = (df['CCI'] < -100).astype(int)
    
    return df