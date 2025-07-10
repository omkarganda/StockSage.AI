"""
Economic data collection from FRED API with error handling and caching
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import requests
import pickle
import logging
import time
from functools import wraps

from ..config import (
    APIConfig,
    DataConfig,
    RAW_DATA_DIR,
    get_cache_path,
    logger as config_logger
)

# Configure module logger
logger = logging.getLogger(__name__)

# FRED API base URL
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# ============================================================================
# DECORATORS
# ============================================================================

def require_fred_api(func):
    """Decorator to check if FRED API key is available"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not APIConfig.FRED_API_KEY:
            raise ValueError(
                "FRED API key not found. Please set FRED_API_KEY in your .env file. "
                "Get your free API key at: https://fred.stlouisfed.org/docs/api/"
            )
        return func(*args, **kwargs)
    return wrapper

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

class EconomicDataCache:
    """Handle caching of economic data to reduce API calls"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or RAW_DATA_DIR / "cache" / "economic"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = DataConfig.CACHE_EXPIRY_HOURS
    
    def _get_cache_path(self, series_id: str) -> Path:
        """Generate cache file path"""
        date_str = datetime.now().strftime("%Y%m%d")
        return self.cache_dir / f"{series_id}_{date_str}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=self.cache_expiry_hours)
    
    def save(self, data: pd.DataFrame, series_id: str) -> None:
        """Save data to cache"""
        cache_path = self._get_cache_path(series_id)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached {series_id} data to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {str(e)}")
    
    def load(self, series_id: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        cache_path = self._get_cache_path(series_id)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded {series_id} data from cache")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return None

# ============================================================================
# FRED API CLIENT
# ============================================================================

class FREDClient:
    """Client for interacting with FRED API"""
    
    def __init__(self, use_cache: bool = True):
        self.api_key = APIConfig.FRED_API_KEY
        self.use_cache = use_cache and DataConfig.USE_CACHE
        self.cache = EconomicDataCache() if self.use_cache else None
        self.session = requests.Session()
        
    @require_fred_api
    @retry_on_failure(max_retries=3)
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request to FRED"""
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{FRED_BASE_URL}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {str(e)}")
            raise
    
    @require_fred_api
    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch time series data from FRED
        
        Args:
            series_id: FRED series ID (e.g., 'DGS10' for 10-year Treasury)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            frequency: Data frequency (d, w, bw, m, q, sa, a)
            
        Returns:
            DataFrame with date index and value column
        """
        # Check cache first
        if self.cache:
            cached_data = self.cache.load(series_id)
            if cached_data is not None:
                # Filter by date range if specified
                if start_date:
                    cached_data = cached_data[cached_data.index >= pd.to_datetime(start_date)]
                if end_date:
                    cached_data = cached_data[cached_data.index <= pd.to_datetime(end_date)]
                
                if not cached_data.empty:
                    return cached_data
        
        # Set default dates if not provided
        if not start_date:
            start_date = DataConfig.DEFAULT_START_DATE
        if not end_date:
            end_date = DataConfig.DEFAULT_END_DATE
        
        logger.info(f"Fetching {series_id} from FRED API ({start_date} to {end_date})")
        
        # Prepare parameters
        params = {
            'series_id': series_id,
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        if frequency:
            params['frequency'] = frequency
        
        # Make API request
        response = self._make_request('series/observations', params)
        
        # Parse response
        observations = response.get('observations', [])
        if not observations:
            raise ValueError(f"No data returned for series {series_id}")
        
        # Convert to DataFrame
        data = []
        for obs in observations:
            try:
                value = float(obs['value'])
                data.append({
                    'date': pd.to_datetime(obs['date']),
                    'value': value
                })
            except (ValueError, TypeError):
                # Skip non-numeric values (e.g., ".")
                continue
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        df.columns = [series_id]
        
        # Save to cache
        if self.cache:
            self.cache.save(df, series_id)
        
        logger.info(f"Successfully fetched {len(df)} observations for {series_id}")
        return df
    
    @require_fred_api
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get metadata about a FRED series
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series information
        """
        logger.info(f"Fetching series info for {series_id}")
        
        params = {'series_id': series_id}
        response = self._make_request('series', params)
        
        series_info = response.get('seriess', [])
        if not series_info:
            raise ValueError(f"No series info found for {series_id}")
        
        return series_info[0]
    
    def search_series(self, search_text: str, limit: int = 100) -> pd.DataFrame:
        """
        Search for FRED series by text
        
        Args:
            search_text: Text to search for
            limit: Maximum number of results
            
        Returns:
            DataFrame with search results
        """
        logger.info(f"Searching FRED for: {search_text}")
        
        params = {
            'search_text': search_text,
            'limit': limit
        }
        
        response = self._make_request('series/search', params)
        series_list = response.get('seriess', [])
        
        return pd.DataFrame(series_list)

# ============================================================================
# ECONOMIC DATA DOWNLOADER
# ============================================================================

class EconomicDataDownloader:
    """High-level interface for downloading economic data"""
    
    def __init__(self, use_cache: bool = True):
        self.client = FREDClient(use_cache=use_cache)
        self.indicators = DataConfig.ECONOMIC_INDICATORS
    
    def download_indicator(
        self,
        indicator: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download single economic indicator
        
        Args:
            indicator: Indicator code or name
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with indicator data
        """
        # Check if indicator is a known name
        series_id = indicator
        for code, name in self.indicators.items():
            if indicator.lower() == name.lower():
                series_id = code
                break
        
        return self.client.get_series(series_id, start_date, end_date)
    
    def download_all_indicators(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Download all configured economic indicators
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with all indicators (columns)
        """
        logger.info(f"Downloading {len(self.indicators)} economic indicators...")
        
        all_data = []
        failed_indicators = []
        
        for series_id, name in self.indicators.items():
            try:
                data = self.client.get_series(series_id, start_date, end_date)
                all_data.append(data)
                logger.info(f"✓ Downloaded {name} ({series_id})")
            except Exception as e:
                logger.error(f"✗ Failed to download {name} ({series_id}): {str(e)}")
                failed_indicators.append(series_id)
        
        if not all_data:
            raise ValueError("Failed to download any economic indicators")
        
        # Combine all data
        combined_df = pd.concat(all_data, axis=1)
        
        # Forward fill missing values (economic data often has gaps)
        combined_df = combined_df.fillna(method='ffill')
        
        # Save combined data
        output_path = RAW_DATA_DIR / f"economic_data_{datetime.now().strftime('%Y%m%d')}.csv"
        combined_df.to_csv(output_path)
        logger.info(f"Saved combined economic data to {output_path}")
        
        if failed_indicators:
            logger.warning(f"Failed indicators: {', '.join(failed_indicators)}")
        
        return combined_df
    
    def get_indicator_correlation(
        self,
        indicator: str,
        stock_returns: pd.Series,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        Calculate correlation between economic indicator and stock returns
        
        Args:
            indicator: Economic indicator code
            stock_returns: Series of stock returns
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with correlation statistics
        """
        # Download indicator data
        indicator_data = self.download_indicator(indicator, start_date, end_date)
        
        # Align data
        aligned_data = pd.concat([
            indicator_data,
            stock_returns.to_frame('returns')
        ], axis=1).dropna()
        
        # Calculate correlations
        correlation = aligned_data.corr().iloc[0, 1]
        
        # Calculate rolling correlation
        rolling_corr = aligned_data.iloc[:, 0].rolling(window=252).corr(
            aligned_data['returns']
        )
        
        return {
            'correlation': correlation,
            'rolling_correlation_mean': rolling_corr.mean(),
            'rolling_correlation_std': rolling_corr.std(),
            'data_points': len(aligned_data)
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def download_default_indicators(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """Download all default economic indicators"""
    downloader = EconomicDataDownloader()
    return downloader.download_all_indicators(start_date, end_date)

def get_recession_dates() -> pd.DataFrame:
    """Get NBER recession dates from FRED"""
    client = FREDClient()
    
    # USREC: 1 = Recession, 0 = Expansion
    recession_data = client.get_series('USREC', start_date='1950-01-01')
    
    # Find recession periods
    recession_periods = []
    in_recession = False
    start = None
    
    for date, row in recession_data.iterrows():
        if row['USREC'] == 1 and not in_recession:
            start = date
            in_recession = True
        elif row['USREC'] == 0 and in_recession:
            recession_periods.append({
                'start': start,
                'end': date,
                'duration_days': (date - start).days
            })
            in_recession = False
    
    return pd.DataFrame(recession_periods)

def get_yield_curve(date: Optional[str] = None) -> pd.DataFrame:
    """
    Get Treasury yield curve for a specific date
    
    Args:
        date: Date for yield curve (defaults to most recent)
        
    Returns:
        DataFrame with maturities and yields
    """
    client = FREDClient()
    
    # Treasury yield series
    yield_series = {
        'DGS1MO': '1 Month',
        'DGS3MO': '3 Month',
        'DGS6MO': '6 Month',
        'DGS1': '1 Year',
        'DGS2': '2 Year',
        'DGS3': '3 Year',
        'DGS5': '5 Year',
        'DGS7': '7 Year',
        'DGS10': '10 Year',
        'DGS20': '20 Year',
        'DGS30': '30 Year'
    }
    
    # If no date specified, use today
    if not date:
        date = datetime.now().strftime('%Y-%m-%d')
    
    yields = []
    for series_id, maturity in yield_series.items():
        try:
            data = client.get_series(series_id, start_date=date, end_date=date)
            if not data.empty:
                yields.append({
                    'maturity': maturity,
                    'maturity_years': float(maturity.split()[0]) if maturity.split()[0].replace('.', '').isdigit() else 0.083,
                    'yield': data.iloc[-1, 0]
                })
        except Exception as e:
            logger.warning(f"Failed to get {maturity} yield: {str(e)}")
    
    return pd.DataFrame(yields).sort_values('maturity_years')

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    
    # Check if API key is available
    if not APIConfig.FRED_API_KEY:
        print("Please set FRED_API_KEY in your .env file")
        print("Get your free API key at: https://fred.stlouisfed.org/docs/api/")
    else:
        # Download single indicator
        downloader = EconomicDataDownloader()
        
        # Get 10-year Treasury rate
        treasury_10y = downloader.download_indicator('DGS10', start_date='2023-01-01')
        print(f"Downloaded {len(treasury_10y)} days of 10-Year Treasury data")
        print(treasury_10y.tail())
        
        # Get all default indicators
        all_indicators = download_default_indicators(start_date='2023-01-01')
        print(f"\nDownloaded {all_indicators.shape[1]} economic indicators")
        print(all_indicators.columns.tolist())
        
        # Get yield curve
        yield_curve = get_yield_curve()
        print(f"\nCurrent Yield Curve:")
        print(yield_curve)
        
        # Get recession dates
        recessions = get_recession_dates()
        print(f"\nRecent Recessions:")
        print(recessions.tail())