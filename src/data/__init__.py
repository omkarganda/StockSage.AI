"""
Data processing module for StockSage.AI

This module handles data downloading, processing, and merging from various sources:
- Market data (yfinance)
- Economic indicators (FRED)
- Sentiment data (news/FinBERT)
"""

# Import main functions for easy access
from .merge import (
    merge_market_economic_data,
    align_sentiment_with_market_data,
    create_unified_dataset,
    resample_to_business_days,
    align_multiple_dataframes
)

__all__ = [
    'merge_market_economic_data',
    'align_sentiment_with_market_data', 
    'create_unified_dataset',
    'resample_to_business_days',
    'align_multiple_dataframes'
]