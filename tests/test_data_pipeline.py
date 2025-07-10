"""
Test suite for data pipeline components

Tests the end-to-end data pipeline including:
- Market data download
- Economic data integration
- Sentiment data processing
- Feature engineering
- Data validation and quality checks
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.download_market import MarketDataDownloader, MarketDataCache
from src.data.validation import DataValidator
from src.features.indicators import (
    add_moving_averages, 
    add_rsi, 
    add_macd,
    add_bollinger_bands,
    add_volume_indicators,
    add_all_technical_indicators
)
from src.config import DataConfig


class TestMarketDataDownloader:
    """Test market data download functionality"""
    
    @pytest.fixture
    def downloader(self):
        """Create a downloader instance with mocked cache"""
        return MarketDataDownloader(use_cache=False)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 100),
            'Ticker': 'TEST'
        }, index=dates)
        return data
    
    @patch('yfinance.Ticker')
    def test_download_single_stock(self, mock_ticker, downloader, sample_data):
        """Test downloading data for a single stock"""
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Download data
        result = downloader.download_stock_data(
            'TEST',
            start_date='2023-01-01',
            end_date='2023-04-10'
        )
        
        # Assertions
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Ticker' in result.columns
        assert len(result) == 100
        
    @patch('yfinance.Ticker')
    def test_download_multiple_stocks(self, mock_ticker, downloader, sample_data):
        """Test downloading data for multiple stocks"""
        # Mock yfinance response for multiple tickers
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_data.copy()
        mock_ticker.return_value = mock_ticker_instance
        
        # Download data
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        results = downloader.download_multiple_stocks(
            tickers,
            start_date='2023-01-01',
            end_date='2023-04-10',
            max_workers=2
        )
        
        # Assertions
        assert isinstance(results, dict)
        assert len(results) == len(tickers)
        for ticker in tickers:
            assert ticker in results
            assert isinstance(results[ticker], pd.DataFrame)
    
    def test_validate_tickers(self, downloader):
        """Test ticker validation"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock valid ticker
            mock_valid = Mock()
            mock_valid.info = {'symbol': 'AAPL'}
            
            # Mock invalid ticker
            mock_invalid = Mock()
            mock_invalid.info.get.side_effect = Exception("Invalid ticker")
            
            # Configure return values
            def ticker_side_effect(symbol):
                if symbol == 'INVALID':
                    return mock_invalid
                return mock_valid
            
            mock_ticker.side_effect = ticker_side_effect
            
            # Test validation
            valid, invalid = downloader.validate_tickers(['AAPL', 'GOOGL', 'INVALID'])
            
            assert 'AAPL' in valid
            assert 'GOOGL' in valid
            assert 'INVALID' in invalid


class TestMarketDataCache:
    """Test market data caching functionality"""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance with temporary directory"""
        temp_dir = tempfile.mkdtemp()
        cache = MarketDataCache(cache_dir=Path(temp_dir))
        yield cache
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_save_and_load_cache(self, cache):
        """Test saving and loading data from cache"""
        # Create sample data
        data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 2000, 3000]
        })
        
        # Save to cache
        cache.save(data, 'TEST', 'ohlcv')
        
        # Load from cache
        loaded = cache.load('TEST', 'ohlcv')
        
        # Assertions
        assert loaded is not None
        pd.testing.assert_frame_equal(data, loaded)
    
    def test_cache_expiry(self, cache):
        """Test cache expiry logic"""
        # Create sample data
        data = pd.DataFrame({'Close': [100]})
        
        # Save to cache
        cache.save(data, 'TEST', 'ohlcv')
        
        # Mock old file time
        cache_path = cache._get_cache_path('TEST', 'ohlcv')
        old_time = datetime.now() - timedelta(hours=25)
        
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_mtime = old_time.timestamp()
            
            # Should return None due to expiry
            loaded = cache.load('TEST', 'ohlcv')
            assert loaded is None


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    @pytest.fixture
    def market_data(self):
        """Create sample market data for feature engineering"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        prices = 100 * np.exp(returns.cumsum())
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(100) * 0.001),
            'High': prices * (1 + np.abs(np.random.randn(100)) * 0.005),
            'Low': prices * (1 - np.abs(np.random.randn(100)) * 0.005),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        return data
    
    def test_moving_averages(self, market_data):
        """Test moving average calculation"""
        result = add_moving_averages(market_data)
        
        # Check if MA columns are added
        assert 'MA_10' in result.columns
        assert 'MA_20' in result.columns
        assert 'MA_50' in result.columns
        
        # Check values
        assert result['MA_10'].iloc[9] == pytest.approx(result['Close'].iloc[:10].mean())
        assert pd.isna(result['MA_50'].iloc[48])  # Should be NaN before 50 periods
        assert not pd.isna(result['MA_50'].iloc[49])  # Should have value at 50th period
    
    def test_rsi_calculation(self, market_data):
        """Test RSI calculation"""
        result = add_rsi(market_data)
        
        # Check if RSI column is added
        assert 'RSI' in result.columns
        
        # Check RSI bounds
        rsi_values = result['RSI'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
    
    def test_macd_calculation(self, market_data):
        """Test MACD calculation"""
        result = add_macd(market_data)
        
        # Check if MACD columns are added
        assert 'MACD' in result.columns
        assert 'MACD_signal' in result.columns
        assert 'MACD_histogram' in result.columns
        
        # Check that histogram equals MACD - signal
        valid_idx = ~(result['MACD'].isna() | result['MACD_signal'].isna())
        histogram_calc = result.loc[valid_idx, 'MACD'] - result.loc[valid_idx, 'MACD_signal']
        pd.testing.assert_series_equal(
            histogram_calc,
            result.loc[valid_idx, 'MACD_histogram'],
            check_names=False
        )
    
    def test_bollinger_bands(self, market_data):
        """Test Bollinger Bands calculation"""
        result = add_bollinger_bands(market_data)
        
        # Check if BB columns are added
        assert 'BB_upper' in result.columns
        assert 'BB_middle' in result.columns
        assert 'BB_lower' in result.columns
        assert 'BB_width' in result.columns
        assert 'BB_percent' in result.columns
        
        # Check logical constraints
        valid_idx = ~result['BB_upper'].isna()
        assert (result.loc[valid_idx, 'BB_upper'] >= result.loc[valid_idx, 'BB_middle']).all()
        assert (result.loc[valid_idx, 'BB_middle'] >= result.loc[valid_idx, 'BB_lower']).all()
    
    def test_volume_indicators(self, market_data):
        """Test volume indicators calculation"""
        result = add_volume_indicators(market_data)
        
        # Check if volume indicator columns are added
        assert 'OBV' in result.columns
        assert 'Volume_SMA' in result.columns
        assert 'Volume_ratio' in result.columns
        assert 'PVT' in result.columns
        
        # Check OBV calculation
        # OBV should increase when close > previous close
        for i in range(1, len(result)):
            if pd.notna(result['OBV'].iloc[i]) and pd.notna(result['OBV'].iloc[i-1]):
                if result['Close'].iloc[i] > result['Close'].iloc[i-1]:
                    assert result['OBV'].iloc[i] > result['OBV'].iloc[i-1]
    
    def test_all_technical_indicators(self, market_data):
        """Test adding all technical indicators"""
        result = add_all_technical_indicators(market_data)
        
        # Check that all indicator groups are present
        ma_cols = [col for col in result.columns if col.startswith('MA_')]
        assert len(ma_cols) >= 3
        
        assert 'RSI' in result.columns
        assert 'MACD' in result.columns
        assert 'BB_upper' in result.columns
        assert 'OBV' in result.columns
        
        # Check data integrity
        assert len(result) == len(market_data)
        assert result.index.equals(market_data.index)


class TestDataValidation:
    """Test data validation functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance"""
        return DataValidator()
    
    @pytest.fixture
    def valid_data(self):
        """Create valid sample data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        return pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
    
    def test_validate_market_data_valid(self, validator, valid_data):
        """Test validation of valid market data"""
        errors = validator.validate_market_data(valid_data)
        assert len(errors) == 0
    
    def test_validate_market_data_missing_columns(self, validator, valid_data):
        """Test validation with missing columns"""
        # Remove required column
        invalid_data = valid_data.drop(columns=['Volume'])
        errors = validator.validate_market_data(invalid_data)
        
        assert len(errors) > 0
        assert any('Volume' in error['message'] for error in errors)
    
    def test_validate_market_data_invalid_prices(self, validator, valid_data):
        """Test validation with invalid price data"""
        # Add negative prices
        invalid_data = valid_data.copy()
        invalid_data.loc[invalid_data.index[10], 'Close'] = -10
        
        errors = validator.validate_market_data(invalid_data)
        assert len(errors) > 0
        assert any('negative' in error['message'].lower() for error in errors)
    
    def test_validate_market_data_high_low_consistency(self, validator, valid_data):
        """Test validation of high/low price consistency"""
        # Make low > high
        invalid_data = valid_data.copy()
        invalid_data.loc[invalid_data.index[10], 'Low'] = 110
        invalid_data.loc[invalid_data.index[10], 'High'] = 90
        
        errors = validator.validate_market_data(invalid_data)
        assert len(errors) > 0
        assert any('high/low' in error['message'].lower() for error in errors)


class TestEndToEndPipeline:
    """Test complete data pipeline integration"""
    
    @patch('yfinance.Ticker')
    def test_complete_pipeline(self, mock_ticker):
        """Test complete data pipeline from download to feature engineering"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 100),
            'Dividends': 0,
            'Stock_Splits': 0
        }, index=dates)
        
        # Mock yfinance
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Execute pipeline
        downloader = MarketDataDownloader(use_cache=False)
        
        # Step 1: Download data
        raw_data = downloader.download_stock_data('TEST', '2023-01-01', '2023-04-10')
        assert not raw_data.empty
        
        # Step 2: Validate data
        validator = DataValidator()
        errors = validator.validate_market_data(raw_data)
        assert len(errors) == 0
        
        # Step 3: Add features
        featured_data = add_all_technical_indicators(raw_data)
        
        # Verify pipeline output
        assert 'MA_20' in featured_data.columns
        assert 'RSI' in featured_data.columns
        assert 'MACD' in featured_data.columns
        assert 'BB_upper' in featured_data.columns
        assert 'OBV' in featured_data.columns
        
        # Check data integrity
        assert len(featured_data) == len(raw_data)
        assert not featured_data.isnull().all().any()  # No columns should be all NaN


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])