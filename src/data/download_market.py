"""
Market data collection using yfinance with robust error handling and caching
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import wraps
import warnings

from ..config import (
    DataConfig, 
    RAW_DATA_DIR, 
    get_cache_path,
    logger as config_logger
)

# Suppress yfinance warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# DECORATORS
# ============================================================================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}). "
                            f"Retrying in {wait_time}s... Error: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts: {str(e)}")
            raise last_exception
        return wrapper
    return decorator

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

class MarketDataCache:
    """Handle caching of market data to reduce API calls"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or RAW_DATA_DIR / "cache" / "market"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = DataConfig.CACHE_EXPIRY_HOURS
    
    def _get_cache_path(self, ticker: str, data_type: str) -> Path:
        """Generate cache file path"""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.cache_dir / f"{ticker}_{data_type}_{date_str}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=self.cache_expiry_hours)
    
    def save(self, data: pd.DataFrame, ticker: str, data_type: str) -> None:
        """Save data to cache"""
        cache_path = self._get_cache_path(ticker, data_type)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached {data_type} data for {ticker} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {str(e)}")
    
    def load(self, ticker: str, data_type: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        cache_path = self._get_cache_path(ticker, data_type)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded {data_type} data for {ticker} from cache")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return None
    
    def clear_cache(self, older_than_days: int = 7) -> None:
        """Clear old cache files"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if datetime.fromtimestamp(cache_file.stat().st_mtime) < cutoff_date:
                try:
                    cache_file.unlink()
                    logger.info(f"Deleted old cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")

# ============================================================================
# MARKET DATA DOWNLOADER
# ============================================================================

class MarketDataDownloader:
    """Enhanced yfinance wrapper with error handling and caching"""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache and DataConfig.USE_CACHE
        self.cache = MarketDataCache() if self.use_cache else None
        
    @retry_on_failure(max_retries=3)
    def download_stock_data(
        self,
        ticker: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Download historical stock data for a single ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data collection
            end_date: End date for data collection  
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Use cache if available and interval is daily
        if self.cache and interval == "1d":
            cached_data = self.cache.load(ticker, "ohlcv")
            if cached_data is not None:
                # Normalize cached data index to timezone-naive for comparison
                if hasattr(cached_data.index, 'tz') and cached_data.index.tz is not None:
                    cached_data.index = cached_data.index.tz_convert('UTC').tz_localize(None)
                
                # Filter by date range if specified
                if start_date:
                    start_ts = pd.to_datetime(start_date)
                    if hasattr(start_ts, 'tz') and start_ts.tz is not None:
                        start_ts = start_ts.tz_convert('UTC').tz_localize(None)
                    cached_data = cached_data[cached_data.index >= start_ts]
                if end_date:
                    end_ts = pd.to_datetime(end_date)
                    if hasattr(end_ts, 'tz') and end_ts.tz is not None:
                        end_ts = end_ts.tz_convert('UTC').tz_localize(None)
                    cached_data = cached_data[cached_data.index <= end_ts]
                
                if not cached_data.empty:
                    return cached_data
        
        # Set default dates if not provided
        if not start_date:
            start_date = DataConfig.DEFAULT_START_DATE
        if not end_date:
            end_date = DataConfig.DEFAULT_END_DATE
        
        logger.info(f"Downloading {ticker} data from {start_date} to {end_date}")
        
        try:
            # Create ticker object
            stock = yf.Ticker(ticker)
            
            # Download data
            df = stock.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,  # Adjust for splits and dividends
                prepost=False,     # Don't include pre/post market data
                actions=True       # Include dividends and stock splits
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Clean column names
            df.columns = [col.replace(' ', '_') for col in df.columns]
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Save to cache if daily data
            if self.cache and interval == "1d":
                self.cache.save(df, ticker, "ohlcv")
            
            logger.info(f"Successfully downloaded {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {str(e)}")
            raise
    
    def download_multiple_stocks(
        self,
        tickers: List[str],
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        interval: str = "1d",
        max_workers: int = 5
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple stocks in parallel
        
        Args:
            tickers: List of stock symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data interval
            max_workers: Maximum number of parallel downloads
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        failed_tickers = []
        
        logger.info(f"Downloading data for {len(tickers)} stocks...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_ticker = {
                executor.submit(
                    self.download_stock_data,
                    ticker, start_date, end_date, interval
                ): ticker
                for ticker in tickers
            }
            
            # Process completed tasks
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    results[ticker] = data
                except Exception as e:
                    logger.error(f"Failed to download {ticker}: {str(e)}")
                    failed_tickers.append(ticker)
        
        if failed_tickers:
            logger.warning(f"Failed to download data for: {', '.join(failed_tickers)}")
        
        logger.info(f"Successfully downloaded data for {len(results)}/{len(tickers)} stocks")
        return results
    
    @retry_on_failure(max_retries=3)
    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get detailed information about a stock
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        # Check cache first
        if self.cache:
            cached_info = self.cache.load(ticker, "info")
            if cached_info is not None:
                return cached_info.to_dict('records')[0]
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant information
            stock_info = {
                'ticker': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', None),
                '52_week_high': info.get('fiftyTwoWeekHigh', None),
                '52_week_low': info.get('fiftyTwoWeekLow', None),
                'avg_volume': info.get('averageVolume', 0),
                'description': info.get('longBusinessSummary', 'N/A')[:500]  # Limit description length
            }
            
            # Cache the info
            if self.cache:
                info_df = pd.DataFrame([stock_info])
                self.cache.save(info_df, ticker, "info")
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Failed to get info for {ticker}: {str(e)}")
            raise
    
    def get_realtime_price(self, ticker: str) -> Dict:
        """
        Get real-time price for a stock
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Dictionary with current price information
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get current price from different sources
            current_price = None
            
            # Try to get from fast_info first
            try:
                current_price = stock.fast_info['lastPrice']
            except:
                pass
            
            # If not available, try regular info
            if current_price is None:
                info = stock.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            # Get additional real-time data
            realtime_data = {
                'ticker': ticker,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'market_state': stock.info.get('marketState', 'UNKNOWN'),
                'currency': stock.info.get('currency', 'USD')
            }
            
            # Try to get more detailed quote data
            try:
                quote = stock.info
                realtime_data.update({
                    'bid': quote.get('bid'),
                    'ask': quote.get('ask'),
                    'bid_size': quote.get('bidSize'),
                    'ask_size': quote.get('askSize'),
                    'volume': quote.get('volume'),
                    'day_high': quote.get('dayHigh'),
                    'day_low': quote.get('dayLow'),
                    'previous_close': quote.get('previousClose')
                })
            except:
                pass
            
            return realtime_data
            
        except Exception as e:
            logger.error(f"Failed to get real-time price for {ticker}: {str(e)}")
            raise
    
    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate a list of tickers
        
        Args:
            tickers: List of stock symbols to validate
            
        Returns:
            Tuple of (valid_tickers, invalid_tickers)
        """
        valid_tickers = []
        invalid_tickers = []
        
        logger.info(f"Validating {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Try to access info to verify ticker is valid
                _ = stock.info.get('symbol')
                valid_tickers.append(ticker)
            except:
                invalid_tickers.append(ticker)
                logger.warning(f"Invalid ticker: {ticker}")
        
        logger.info(f"Validation complete: {len(valid_tickers)} valid, {len(invalid_tickers)} invalid")
        return valid_tickers, invalid_tickers

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def download_default_stocks(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Download data for default list of stocks and combine into single DataFrame
    
    Args:
        start_date: Start date for data collection
        end_date: End date for data collection
        use_cache: Whether to use cached data
        
    Returns:
        Combined DataFrame with all stock data
    """
    downloader = MarketDataDownloader(use_cache=use_cache)
    
    # Download data for all default tickers
    stock_data = downloader.download_multiple_stocks(
        tickers=DataConfig.DEFAULT_TICKERS,
        start_date=start_date,
        end_date=end_date
    )
    
    # Combine all data into single DataFrame
    combined_df = pd.concat(stock_data.values(), axis=0)
    combined_df = combined_df.sort_index()
    
    # Save combined data
    output_path = RAW_DATA_DIR / f"market_data_{datetime.now().strftime('%Y%m%d')}.csv"
    combined_df.to_csv(output_path)
    logger.info(f"Saved combined market data to {output_path}")
    
    return combined_df

def get_market_summary() -> pd.DataFrame:
    """
    Get summary of major market indices
    
    Returns:
        DataFrame with current market summary
    """
    indices = {
        '^GSPC': 'S&P 500',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ',
        '^VIX': 'VIX',
        '^RUT': 'Russell 2000'
    }
    
    downloader = MarketDataDownloader(use_cache=False)
    summary_data = []
    
    for symbol, name in indices.items():
        try:
            price_data = downloader.get_realtime_price(symbol)
            summary_data.append({
                'Index': name,
                'Symbol': symbol,
                'Price': price_data['current_price'],
                'Change': price_data['current_price'] - price_data.get('previous_close', price_data['current_price']),
                'Change %': ((price_data['current_price'] / price_data.get('previous_close', price_data['current_price'])) - 1) * 100
            })
        except Exception as e:
            logger.warning(f"Failed to get data for {name}: {str(e)}")
    
    return pd.DataFrame(summary_data)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    downloader = MarketDataDownloader()
    
    # Download single stock
    aapl_data = downloader.download_stock_data("AAPL", start_date="2023-01-01")
    print(f"Downloaded {len(aapl_data)} days of AAPL data")
    print(aapl_data.head())
    
    # Get stock info
    aapl_info = downloader.get_stock_info("AAPL")
    print(f"\nAAPL Info: {aapl_info}")
    
    # Get real-time price
    aapl_price = downloader.get_realtime_price("AAPL")
    print(f"\nAAPL Current Price: ${aapl_price['current_price']}")
    
    # Download default stocks
    all_data = download_default_stocks(start_date="2023-01-01")
    print(f"\nDownloaded data for {all_data['Ticker'].nunique()} stocks")