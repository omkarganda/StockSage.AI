"""
Central configuration for StockSage.AI
Handles API keys, paths, and settings from environment variables
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Project root directory
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# API KEY CONFIGURATION
# ============================================================================

class APIConfig:
    """Central API key management"""
    
    # Essential APIs
    FRED_API_KEY: Optional[str] = os.getenv("FRED_API_KEY")
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Optional APIs
    ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    FINNHUB_API_KEY: Optional[str] = os.getenv("FINNHUB_API_KEY")
    WANDB_API_KEY: Optional[str] = os.getenv("WANDB_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    
    @classmethod
    def validate_essential_keys(cls) -> bool:
        """Check if essential API keys are present"""
        essential_keys = {
            "FRED": cls.FRED_API_KEY,
            "NewsAPI": cls.NEWS_API_KEY,
        }
        
        missing_keys = [name for name, key in essential_keys.items() if not key]
        
        if missing_keys:
            logger.warning(f"Missing essential API keys: {', '.join(missing_keys)}")
            logger.warning("Some features may not work. Set these in your .env file.")
            return False
        return True
    
    @classmethod
    def get_available_apis(cls) -> Dict[str, bool]:
        """Return which APIs are available"""
        return {
            "fred": bool(cls.FRED_API_KEY),
            "news_api": bool(cls.NEWS_API_KEY),
            "openai": bool(cls.OPENAI_API_KEY),
            "alpha_vantage": bool(cls.ALPHA_VANTAGE_API_KEY),
            "finnhub": bool(cls.FINNHUB_API_KEY),
            "wandb": bool(cls.WANDB_API_KEY),
            "huggingface": bool(cls.HUGGINGFACE_API_KEY),
        }

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

class DataConfig:
    """Data collection and processing settings"""
    
    # Market data settings
    DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # Economic indicators
    ECONOMIC_INDICATORS = {
        "DGS10": "10-Year Treasury Rate",
        "DEXUSEU": "USD/EUR Exchange Rate",
        "DFF": "Federal Funds Rate",
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "Consumer Price Index",
        "GDPC1": "Real GDP",
        "DCOILWTICO": "Crude Oil Prices WTI",
        "VIXCLS": "VIX Volatility Index"
    }
    
    # News sentiment settings
    NEWS_SOURCES = [
        "bloomberg",
        "reuters",
        "financial-times",
        "the-wall-street-journal",
        "cnbc",
        "fortune"
    ]
    
    NEWS_CATEGORIES = ["business", "technology", "finance"]
    MAX_NEWS_ARTICLES_PER_DAY = 100
    
    # Cache settings
    CACHE_EXPIRY_HOURS = 24
    USE_CACHE = True
    
    # Data quality settings
    MIN_DATA_POINTS = 30  # Minimum data points for modeling
    MAX_MISSING_RATIO = 0.1  # Maximum 10% missing data allowed

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

class ModelConfig:
    """Machine learning model settings"""
    
    # Time series settings
    FORECAST_HORIZON = 30  # Days to forecast
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # Neural network settings
    LSTM_UNITS = [128, 64, 32]
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    
    # Feature engineering
    TECHNICAL_INDICATORS = [
        "SMA_20", "SMA_50", "SMA_200",
        "EMA_12", "EMA_26",
        "RSI_14",
        "MACD", "MACD_signal",
        "BB_upper", "BB_middle", "BB_lower",
        "ATR_14",
        "OBV"
    ]
    
    # Sentiment analysis
    SENTIMENT_MODEL = "ProsusAI/finbert"
    SENTIMENT_BATCH_SIZE = 16
    
    # Backtesting
    INITIAL_CAPITAL = 10000
    COMMISSION_RATE = 0.001  # 0.1%
    SLIPPAGE_RATE = 0.0005  # 0.05%

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

class AppConfig:
    """Application settings"""
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # Server settings
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cache settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_config(config_dict: Dict[str, Any], filename: str) -> None:
    """Save configuration to JSON file"""
    config_path = CONFIG_DIR / filename
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Configuration saved to {config_path}")

def load_config(filename: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    config_path = CONFIG_DIR / filename
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Configuration file {config_path} not found")
        return {}

def get_cache_path(data_type: str, ticker: str = None) -> Path:
    """Get standardized cache file path"""
    cache_dir = RAW_DATA_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    if ticker:
        return cache_dir / f"{data_type}_{ticker}_{datetime.now().strftime('%Y%m%d')}.pkl"
    else:
        return cache_dir / f"{data_type}_{datetime.now().strftime('%Y%m%d')}.pkl"

# ============================================================================
# INITIALIZATION
# ============================================================================

# Validate API keys on import
APIConfig.validate_essential_keys()

# Log available APIs
available_apis = APIConfig.get_available_apis()
logger.info(f"Available APIs: {[api for api, available in available_apis.items() if available]}")

# Create config directory if it doesn't exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Create default config files if they don't exist
if not (CONFIG_DIR / "model_params.yaml").exists():
    default_model_params = {
        "lstm": {
            "units": ModelConfig.LSTM_UNITS,
            "dropout": ModelConfig.DROPOUT_RATE,
            "batch_size": ModelConfig.BATCH_SIZE
        },
        "training": {
            "epochs": ModelConfig.MAX_EPOCHS,
            "early_stopping": ModelConfig.EARLY_STOPPING_PATIENCE
        }
    }
    save_config(default_model_params, "model_params.json")

# Export commonly used configurations
__all__ = [
    'APIConfig', 'DataConfig', 'ModelConfig', 'AppConfig',
    'ROOT_DIR', 'DATA_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR',
    'save_config', 'load_config', 'get_cache_path'
]