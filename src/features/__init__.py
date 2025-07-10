"""
StockSage.AI Feature Engineering Package

This package provides comprehensive feature engineering functionality for financial data:
- Technical indicators (moving averages, RSI, MACD, Bollinger Bands, etc.)
- Sentiment features (aggregation, momentum, volume indicators)
- Volume and price momentum features
"""

from .indicators import (
    calculate_moving_averages,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_volume_indicators,
    calculate_price_momentum,
    add_all_technical_indicators
)

from .sentiment import (
    aggregate_daily_sentiment,
    calculate_sentiment_momentum,
    calculate_news_volume_indicators,
    add_all_sentiment_features
)

__all__ = [
    # Technical Indicators
    'calculate_moving_averages',
    'calculate_rsi', 
    'calculate_macd',
    'calculate_bollinger_bands',
    'calculate_volume_indicators',
    'calculate_price_momentum',
    'add_all_technical_indicators',
    
    # Sentiment Features
    'aggregate_daily_sentiment',
    'calculate_sentiment_momentum',
    'calculate_news_volume_indicators',
    'add_all_sentiment_features'
]