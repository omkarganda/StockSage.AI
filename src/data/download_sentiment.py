"""
News sentiment collection using NewsAPI and FinBERT for financial sentiment analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
import pickle
import logging
import time
from functools import wraps
import json

# For FinBERT sentiment analysis
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    
# For NewsAPI
try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

from ..config import (
    APIConfig,
    DataConfig,
    ModelConfig,
    RAW_DATA_DIR,
    get_cache_path,
    logger as config_logger
)

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# DECORATORS
# ============================================================================

def require_news_api(func):
    """Decorator to check if News API key is available"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not APIConfig.NEWS_API_KEY:
            raise ValueError(
                "News API key not found. Please set NEWS_API_KEY in your .env file. "
                "Get your free API key at: https://newsapi.org/"
            )
        if not NEWSAPI_AVAILABLE:
            raise ImportError(
                "newsapi-python not installed. Install with: pip install newsapi-python"
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

class SentimentDataCache:
    """Handle caching of sentiment data to reduce API calls"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or RAW_DATA_DIR / "cache" / "sentiment"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = DataConfig.CACHE_EXPIRY_HOURS
    
    def _get_cache_path(self, query: str, date: str) -> Path:
        """Generate cache file path"""
        # Clean query for filename
        clean_query = query.replace(' ', '_').replace('/', '_')[:50]
        return self.cache_dir / f"{clean_query}_{date}.pkl"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid"""
        if not cache_path.exists():
            return False
        
        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age < timedelta(hours=self.cache_expiry_hours)
    
    def save(self, data: pd.DataFrame, query: str, date: str) -> None:
        """Save data to cache"""
        cache_path = self._get_cache_path(query, date)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Cached sentiment data for {query} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {str(e)}")
    
    def load(self, query: str, date: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid"""
        cache_path = self._get_cache_path(query, date)
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded sentiment data for {query} from cache")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return None

# ============================================================================
# FINBERT SENTIMENT ANALYZER
# ============================================================================

class FinBertSentimentAnalyzer:
    """Financial sentiment analysis using FinBERT"""
    
    def __init__(self, model_name: Optional[str] = None):
        if not FINBERT_AVAILABLE:
            raise ImportError(
                "transformers and torch not installed. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name or ModelConfig.SENTIMENT_MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading FinBERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Sentiment labels (may vary by model)
        self.labels = ['negative', 'neutral', 'positive']
        
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to dict
        scores = predictions[0].cpu().numpy()
        sentiment_scores = {
            label: float(score) 
            for label, score in zip(self.labels, scores)
        }
        
        # Add compound score (positive - negative)
        sentiment_scores['compound'] = (
            sentiment_scores.get('positive', 0) - 
            sentiment_scores.get('negative', 0)
        )
        
        return sentiment_scores
    
    def analyze_batch(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts in batches
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment dictionaries
        """
        batch_size = batch_size or ModelConfig.SENTIMENT_BATCH_SIZE
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to list of dicts
            for j, scores in enumerate(predictions.cpu().numpy()):
                sentiment_scores = {
                    label: float(score) 
                    for label, score in zip(self.labels, scores)
                }
                sentiment_scores['compound'] = (
                    sentiment_scores.get('positive', 0) - 
                    sentiment_scores.get('negative', 0)
                )
                results.append(sentiment_scores)
        
        return results

# ============================================================================
# NEWS COLLECTOR
# ============================================================================

class NewsCollector:
    """Collect news articles from various sources"""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache and DataConfig.USE_CACHE
        self.cache = SentimentDataCache() if self.use_cache else None
        
        # Initialize NewsAPI client if available
        if APIConfig.NEWS_API_KEY and NEWSAPI_AVAILABLE:
            self.newsapi = NewsApiClient(api_key=APIConfig.NEWS_API_KEY)
        else:
            self.newsapi = None
    
    @require_news_api
    @retry_on_failure(max_retries=3)
    def fetch_news_articles(
        self,
        query: str,
        from_date: Union[str, datetime],
        to_date: Union[str, datetime],
        sources: Optional[List[str]] = None,
        language: str = 'en',
        sort_by: str = 'relevancy'
    ) -> pd.DataFrame:
        """
        Fetch news articles from NewsAPI
        
        Args:
            query: Search query (e.g., "AAPL" or "Apple Inc")
            from_date: Start date for articles
            to_date: End date for articles
            sources: List of news sources (optional)
            language: Language code
            sort_by: Sort order (relevancy, popularity, publishedAt)
            
        Returns:
            DataFrame with news articles
        """
        # Convert dates to string format
        if isinstance(from_date, datetime):
            from_date = from_date.strftime('%Y-%m-%d')
        if isinstance(to_date, datetime):
            to_date = to_date.strftime('%Y-%m-%d')
        
        # NewsAPI free tier limitation: only allows articles from the last 30 days
        today = datetime.now()
        earliest_allowed = (today - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Adjust date range if it's too far in the past
        original_from_date = from_date
        if from_date < earliest_allowed:
            logger.warning(f"Requested from_date {from_date} is too far in the past for NewsAPI free tier. "
                         f"Adjusting to {earliest_allowed}")
            from_date = earliest_allowed
        
        # Check cache
        cache_key = f"{query}_{from_date}_{to_date}"
        if self.cache:
            cached_data = self.cache.load(query, cache_key)
            if cached_data is not None:
                return cached_data
        
        logger.info(f"Fetching news for '{query}' from {from_date} to {to_date}"
                   f"{' (adjusted from ' + original_from_date + ')' if original_from_date != from_date else ''}")
        
        # Prepare sources
        if sources:
            sources_str = ','.join(sources)
        else:
            sources_str = ','.join(DataConfig.NEWS_SOURCES) if DataConfig.NEWS_SOURCES else None
        
        # Fetch articles
        all_articles = []
        page = 1
        
        while True:
            try:
                response = self.newsapi.get_everything(
                    q=query,
                    from_param=from_date,
                    to=to_date,
                    sources=sources_str,
                    language=language,
                    sort_by=sort_by,
                    page=page,
                    page_size=100  # Max allowed
                )
                
                articles = response.get('articles', [])
                if not articles:
                    break
                
                all_articles.extend(articles)
                
                # Check if we've reached the limit
                if len(all_articles) >= DataConfig.MAX_NEWS_ARTICLES_PER_DAY:
                    break
                
                # Check if there are more pages
                total_results = response.get('totalResults', 0)
                if len(all_articles) >= total_results:
                    break
                
                page += 1
                
            except Exception as e:
                error_msg = str(e)
                if 'parameterInvalid' in error_msg and 'too far in the past' in error_msg:
                    logger.warning(f"NewsAPI date limitation reached: {error_msg}")
                    logger.info("Consider upgrading to a paid NewsAPI plan for historical data access")
                else:
                    logger.error(f"Error fetching news page {page}: {error_msg}")
                break
        
        if not all_articles:
            logger.warning(f"No articles found for query: {query}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        # Clean and process data
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df['source_name'] = df['source'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
        
        # Select relevant columns
        columns = [
            'publishedAt', 'source_name', 'author', 'title', 
            'description', 'content', 'url', 'urlToImage'
        ]
        df = df[columns].copy()
        
        # Save to cache
        if self.cache:
            self.cache.save(df, query, cache_key)
        
        logger.info(f"Fetched {len(df)} articles for {query}")
        return df
    
    def fetch_company_news(
        self,
        ticker: str,
        company_name: str,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch news for a specific company
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            days_back: Number of days to look back
            
        Returns:
            DataFrame with news articles
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Try multiple queries to get comprehensive coverage
        queries = [
            ticker,
            company_name,
            f"{ticker} stock",
            f"{company_name} earnings"
        ]
        
        all_articles = []
        seen_urls = set()
        
        for query in queries:
            try:
                articles = self.fetch_news_articles(
                    query=query,
                    from_date=start_date,
                    to_date=end_date
                )
                
                # Deduplicate by URL
                for _, article in articles.iterrows():
                    if article['url'] not in seen_urls:
                        seen_urls.add(article['url'])
                        all_articles.append(article)
                        
            except Exception as e:
                logger.warning(f"Failed to fetch news for query '{query}': {str(e)}")
        
        if all_articles:
            df = pd.DataFrame(all_articles)
            df = df.sort_values('publishedAt', ascending=False)
            return df
        else:
            return pd.DataFrame()

# ============================================================================
# SENTIMENT DATA PROCESSOR
# ============================================================================

class SentimentDataProcessor:
    """Process news articles and generate sentiment scores"""
    
    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert and FINBERT_AVAILABLE
        
        if self.use_finbert:
            try:
                self.sentiment_analyzer = FinBertSentimentAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {str(e)}")
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
        
        self.news_collector = NewsCollector()
    
    def process_articles(
        self,
        articles_df: pd.DataFrame,
        text_columns: List[str] = ['title', 'description']
    ) -> pd.DataFrame:
        """
        Process articles and add sentiment scores
        
        Args:
            articles_df: DataFrame with news articles
            text_columns: Columns to analyze for sentiment
            
        Returns:
            DataFrame with added sentiment scores
        """
        if articles_df.empty:
            return articles_df
        
        # Combine text columns
        texts = []
        for _, row in articles_df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            texts.append(' '.join(text_parts))
        
        # Analyze sentiment
        if self.sentiment_analyzer and texts:
            logger.info(f"Analyzing sentiment for {len(texts)} articles...")
            sentiment_scores = self.sentiment_analyzer.analyze_batch(texts)
            
            # Add scores to DataFrame
            for key in ['negative', 'neutral', 'positive', 'compound']:
                articles_df[f'sentiment_{key}'] = [
                    score.get(key, 0) for score in sentiment_scores
                ]
        else:
            # Use placeholder scores if FinBERT not available
            logger.warning("FinBERT not available, using neutral sentiment scores")
            articles_df['sentiment_negative'] = 0.0
            articles_df['sentiment_neutral'] = 1.0
            articles_df['sentiment_positive'] = 0.0
            articles_df['sentiment_compound'] = 0.0

        # ------------------------------------------------------------------
        # Augment with generative LLM sentiment features (optional)
        # ------------------------------------------------------------------
        try:
            from src.features.generative_sentiment import add_llm_sentiment_features

            articles_df = add_llm_sentiment_features(articles_df, text_columns=text_columns)
        except Exception as exc:  # pragma: no cover – missing openai etc.
            logger.debug("Skipping LLM sentiment augmentation – %s", exc)
 
        return articles_df
    
    def get_daily_sentiment(
        self,
        ticker: str,
        company_name: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Get aggregated daily sentiment scores for a company
        
        Args:
            ticker: Stock ticker
            company_name: Company name
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with daily sentiment scores
        """
        # Convert dates to pandas Timestamp and ensure timezone-naive
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Convert to timezone-naive if they have timezone info
        if start_date.tz is not None:
            start_date = start_date.tz_convert('UTC').tz_localize(None)
        if end_date.tz is not None:
            end_date = end_date.tz_convert('UTC').tz_localize(None)
        
        # Calculate days
        days = (end_date - start_date).days + 1
        
        # Fetch news
        articles = self.news_collector.fetch_company_news(
            ticker=ticker,
            company_name=company_name,
            days_back=days
        )
        
        if articles.empty:
            logger.warning(f"No articles found for {ticker}")
            return pd.DataFrame()
        
        # Normalize publishedAt column to timezone-naive for comparison
        if 'publishedAt' in articles.columns:
            if articles['publishedAt'].dt.tz is not None:
                articles['publishedAt'] = articles['publishedAt'].dt.tz_convert('UTC').dt.tz_localize(None)
        
        # Filter by date range
        articles = articles[
            (articles['publishedAt'] >= start_date) &
            (articles['publishedAt'] <= end_date)
        ]
        
        # Process sentiment
        articles = self.process_articles(articles)
        
        # Aggregate by day
        articles['date'] = articles['publishedAt'].dt.date
        
        daily_sentiment = articles.groupby('date').agg({
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_compound': 'mean',
            'title': 'count'  # Article count
        }).rename(columns={'title': 'article_count'})
        
        # Add weighted sentiment (compound * article_count)
        daily_sentiment['weighted_sentiment'] = (
            daily_sentiment['sentiment_compound'] * 
            np.log1p(daily_sentiment['article_count'])  # Log scale for article count
        )
        
        return daily_sentiment

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def collect_market_sentiment(
    tickers: List[str],
    company_names: Dict[str, str],
    days_back: int = 30
) -> pd.DataFrame:
    """
    Collect sentiment data for multiple stocks
    
    Args:
        tickers: List of stock tickers
        company_names: Dict mapping ticker to company name
        days_back: Number of days to look back
        
    Returns:
        Combined DataFrame with sentiment data
    """
    processor = SentimentDataProcessor()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    all_sentiment = []
    
    for ticker in tickers:
        company_name = company_names.get(ticker, ticker)
        logger.info(f"Processing sentiment for {ticker} ({company_name})")
        
        try:
            sentiment = processor.get_daily_sentiment(
                ticker=ticker,
                company_name=company_name,
                start_date=start_date,
                end_date=end_date
            )
            
            if not sentiment.empty:
                sentiment['ticker'] = ticker
                all_sentiment.append(sentiment)
                
        except Exception as e:
            logger.error(f"Failed to process {ticker}: {str(e)}")
    
    if all_sentiment:
        combined = pd.concat(all_sentiment)
        
        # Save to file
        output_path = RAW_DATA_DIR / f"sentiment_data_{datetime.now().strftime('%Y%m%d')}.csv"
        combined.to_csv(output_path)
        logger.info(f"Saved sentiment data to {output_path}")
        
        return combined
    else:
        return pd.DataFrame()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    
    # Check requirements
    if not APIConfig.NEWS_API_KEY:
        print("Please set NEWS_API_KEY in your .env file")
        print("Get your free API key at: https://newsapi.org/")
    
    if not FINBERT_AVAILABLE:
        print("FinBERT not available. Install with:")
        print("pip install transformers torch")
    
    if APIConfig.NEWS_API_KEY:
        # Test with single stock
        processor = SentimentDataProcessor()
        
        # Get sentiment for Apple
        sentiment = processor.get_daily_sentiment(
            ticker="AAPL",
            company_name="Apple Inc",
            start_date="2024-01-01",
            end_date="2024-01-07"
        )
        
        if not sentiment.empty:
            print("Apple Sentiment Summary:")
            print(sentiment)
            print(f"\nAverage compound sentiment: {sentiment['sentiment_compound'].mean():.3f}")
        
        # Test multiple stocks
        tickers = ["AAPL", "MSFT", "GOOGL"]
        company_names = {
            "AAPL": "Apple Inc",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc"
        }
        
        market_sentiment = collect_market_sentiment(
            tickers=tickers,
            company_names=company_names,
            days_back=7
        )
        
        if not market_sentiment.empty:
            print(f"\nCollected sentiment for {market_sentiment['ticker'].nunique()} stocks")
            print(f"Total data points: {len(market_sentiment)}")