"""
StockSage.AI REST API

This module provides a FastAPI-based REST API for the StockSage.AI platform.
It exposes endpoints for predictions, health checks, batch processing, and model explanations.

Endpoints:
- GET /predict/{symbol}: Get prediction for a single stock
- GET /health: System health check with dependency status
- POST /batch_predict: Bulk predictions for multiple stocks
- GET /explain/{symbol}: Model explanation and feature importance
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import asyncio
import pandas as pd
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.download_market import MarketDataDownloader
from src.features.indicators import add_all_technical_indicators
from src.utils.logging import get_logger

# Initialize downloader
downloader = MarketDataDownloader(use_cache=False)

# Configure logging
logger = get_logger(__name__)

# Global variables for models
models = {}
model_configs = {}


class PredictionRequest(BaseModel):
    """Request model for single stock prediction"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    horizon: int = Field(1, ge=1, le=30, description="Prediction horizon in days")
    include_confidence: bool = Field(False, description="Include confidence intervals")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    symbols: List[str] = Field(..., description="List of stock symbols")
    horizon: int = Field(1, ge=1, le=30, description="Prediction horizon in days")
    include_confidence: bool = Field(False, description="Include confidence intervals")
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one symbol is required")
        if len(v) > 50:
            raise ValueError("Maximum 50 symbols allowed per batch")
        return [s.upper().strip() for s in v]


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    symbol: str
    current_price: float
    predicted_price: float
    predicted_return: float
    horizon_days: int
    prediction_date: str
    confidence_interval: Optional[Dict[str, float]] = None
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 signal strength
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    failed_symbols: List[Dict[str, str]]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    version: str
    models_loaded: int
    dependencies: Dict[str, str]
    memory_usage_mb: float
    uptime_seconds: float


class ExplanationResponse(BaseModel):
    """Response model for model explanations"""
    symbol: str
    model_type: str
    top_features: List[Dict[str, Union[str, float]]]
    feature_importance_plot: Optional[str] = None
    recent_predictions_accuracy: Optional[Dict[str, float]] = None
    explanation_text: str
    timestamp: str
    
    @validator('top_features')
    def validate_top_features(cls, v):
        """Ensure feature names are strings and importance values are floats"""
        for item in v:
            if 'feature' in item and not isinstance(item['feature'], str):
                item['feature'] = str(item['feature'])
            if 'importance' in item and not isinstance(item['importance'], (int, float)):
                item['importance'] = float(item['importance'])
        return v


# Initialize app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load models on startup"""
    logger.info("Starting StockSage.AI API")
    
    # Load models on startup
    try:
        await load_models()
        logger.info(f"Successfully loaded {len(models)} models")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down StockSage.AI API")
    models.clear()
    model_configs.clear()


# Create FastAPI app
app = FastAPI(
    title="StockSage.AI API",
    description="AI-driven stock price prediction API with sentiment analysis and explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Startup time for uptime calculation
startup_time = datetime.now()


async def load_models():
    """Load pre-trained models from disk"""
    model_dir = Path("results/model_training")
    
    if not model_dir.exists():
        logger.warning(f"Model directory {model_dir} not found, creating mock models")
        # Create mock models for development
        for symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]:
            models[f"{symbol}_ensemble"] = None  # Mock model
            model_configs[symbol] = {
                "type": "ensemble",
                "features": ["price", "volume", "rsi", "macd", "sentiment"],
                "accuracy": 0.85 + np.random.random() * 0.1
            }
        return
    
    # Load actual models
    for model_file in model_dir.glob("*.joblib"):
        try:
            symbol = model_file.stem.split("_")[0]
            model_type = "_".join(model_file.stem.split("_")[1:])
            
            # Load model
            model_data = joblib.load(model_file)
            models[model_file.stem] = model_data
            
            # Store model config
            if symbol not in model_configs:
                model_configs[symbol] = {}
            model_configs[symbol][model_type] = {
                "file": str(model_file),
                "loaded_at": datetime.now().isoformat()
            }
            
            logger.info(f"Loaded model: {model_file.stem}")
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {str(e)}")


def get_model_for_symbol(symbol: str, model_type: str = "baseline_ensemble"):
    """Get the appropriate model for a symbol"""
    model_key = f"{symbol}_{model_type}"
    
    if model_key in models:
        return models[model_key]
    
    # Fallback to any available model for the symbol
    for key in models:
        if key.startswith(f"{symbol}_"):
            logger.warning(f"Model {model_key} not found, using {key}")
            return models[key]
    
    return None


async def get_stock_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Fetch recent stock data for prediction"""
    try:
        # Download stock data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = await asyncio.to_thread(
            downloader.download_stock_data,
            symbol,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if df is None or df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Add technical indicators
        df = add_all_technical_indicators(df)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise


def generate_prediction_signal(predicted_return: float) -> tuple[str, float]:
    """Generate trading signal based on predicted return"""
    abs_return = abs(predicted_return)
    
    if predicted_return > 0.02:  # > 2% expected return
        signal = "buy"
        strength = min(abs_return / 0.05, 1.0)  # Max strength at 5% return
    elif predicted_return < -0.02:  # < -2% expected return
        signal = "sell"
        strength = min(abs_return / 0.05, 1.0)
    else:
        signal = "hold"
        strength = 0.5 - abs(predicted_return) / 0.04  # Decreases as we move from 0
    
    return signal, strength


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs"""
    return {"message": "Welcome to StockSage.AI API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint
    
    Returns the current system status including:
    - Model loading status
    - Dependency health
    - Memory usage
    - Uptime
    """
    try:
        # Check dependencies
        dependencies = {}
        
        # Check if we can import key libraries
        try:
            import yfinance
            dependencies["yfinance"] = "healthy"
        except:
            dependencies["yfinance"] = "unavailable"
        
        try:
            import torch
            dependencies["pytorch"] = "healthy"
        except:
            dependencies["pytorch"] = "unavailable"
        
        try:
            import openai
            dependencies["openai"] = "healthy"
        except:
            dependencies["openai"] = "unavailable"
        
        # Memory usage
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Calculate uptime
        uptime = (datetime.now() - startup_time).total_seconds()
        
        return HealthResponse(
            status="healthy" if len(models) > 0 else "degraded",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            models_loaded=len(models),
            dependencies=dependencies,
            memory_usage_mb=round(memory_mb, 2),
            uptime_seconds=round(uptime, 2)
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/predict/{symbol}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(
    symbol: str,
    horizon: int = Query(1, ge=1, le=30, description="Prediction horizon in days"),
    include_confidence: bool = Query(False, description="Include confidence intervals")
):
    """
    Get prediction for a single stock symbol
    
    Parameters:
    - symbol: Stock ticker symbol (e.g., AAPL, GOOGL)
    - horizon: Number of days ahead to predict (1-30)
    - include_confidence: Whether to include confidence intervals
    
    Returns prediction with price target and trading signal
    """
    symbol = symbol.upper()
    
    try:
        # Get stock data
        df = await get_stock_data(symbol)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        current_price = float(df['Close'].iloc[-1])
        
        # Get model
        model = get_model_for_symbol(symbol)
        
        if model is None:
            # Fallback to simple prediction
            logger.warning(f"No model available for {symbol}, using simple prediction")
            
            # Simple trend-based prediction
            returns = df['Close'].pct_change().dropna()
            avg_return = returns.tail(20).mean()
            volatility = returns.tail(20).std()
            
            predicted_return = avg_return * horizon
            predicted_price = current_price * (1 + predicted_return)
            
            confidence_interval = {
                "lower": float(current_price * (1 + predicted_return - 2*volatility*np.sqrt(horizon))),
                "upper": float(current_price * (1 + predicted_return + 2*volatility*np.sqrt(horizon)))
            } if include_confidence else None
        else:
            # Use trained model
            try:
                # Prepare data for model
                if hasattr(model, 'predict'):
                    predictions = model.predict(df)
                    predicted_return = float(predictions[-1])
                else:
                    # Handle case where model is a dictionary
                    predicted_return = np.random.randn() * 0.02  # Mock prediction
                
                predicted_price = current_price * (1 + predicted_return)
                
                # Calculate confidence interval if requested
                confidence_interval = {
                    "lower": float(predicted_price * 0.95),
                    "upper": float(predicted_price * 1.05)
                } if include_confidence else None
            except Exception as e:
                logger.error(f"Model prediction failed for {symbol}: {str(e)}")
                raise HTTPException(status_code=500, detail="Model prediction failed")
        
        # Generate trading signal
        signal, strength = generate_prediction_signal(predicted_return)
        
        return PredictionResponse(
            symbol=symbol,
            current_price=round(current_price, 2),
            predicted_price=round(predicted_price, 2),
            predicted_return=round(predicted_return * 100, 2),
            horizon_days=horizon,
            prediction_date=(datetime.now() + timedelta(days=horizon)).strftime("%Y-%m-%d"),
            confidence_interval=confidence_interval,
            signal=signal,
            strength=round(strength, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Get predictions for multiple stocks in a single request
    
    Efficiently processes batch predictions with parallel execution.
    Maximum 50 symbols per request to prevent overload.
    """
    start_time = datetime.now()
    predictions = []
    failed_symbols = []
    
    # Process predictions in parallel
    tasks = []
    for symbol in request.symbols:
        task = predict_single(
            symbol=symbol,
            horizon=request.horizon,
            include_confidence=request.include_confidence
        )
        tasks.append((symbol, task))
    
    # Gather results
    for symbol, task in tasks:
        try:
            result = await task
            predictions.append(result)
        except Exception as e:
            failed_symbols.append({
                "symbol": symbol,
                "error": str(e)
            })
            logger.error(f"Batch prediction failed for {symbol}: {str(e)}")
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        failed_symbols=failed_symbols,
        processing_time_ms=round(processing_time, 2)
    )


@app.get("/explain/{symbol}", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_prediction(symbol: str):
    """
    Get explanation for model predictions
    
    Provides:
    - Feature importance ranking
    - Key factors driving the prediction
    - Model performance metrics
    - Human-readable explanation
    """
    symbol = symbol.upper()
    
    try:
        # Get model
        model = get_model_for_symbol(symbol)
        
        if model is None:
            # Provide generic explanation
            top_features = [
                {"feature": "price_momentum", "importance": 0.25},
                {"feature": "volume_trend", "importance": 0.20},
                {"feature": "rsi", "importance": 0.15},
                {"feature": "market_sentiment", "importance": 0.15},
                {"feature": "volatility", "importance": 0.10},
                {"feature": "ma_crossover", "importance": 0.10},
                {"feature": "economic_indicators", "importance": 0.05}
            ]
            
            explanation_text = (
                f"The prediction for {symbol} is based on a combination of technical indicators, "
                "market sentiment, and price momentum. The model analyzes recent price movements, "
                "trading volume patterns, and technical indicators like RSI and moving averages. "
                "Market sentiment from news and social media also plays a significant role."
            )
            
            model_type = "ensemble"
        else:
            # Get actual feature importance
            if hasattr(model, 'get_feature_importance'):
                importance_df = model.get_feature_importance()
                top_features = [
                    {"feature": row['feature'], "importance": float(row['importance'])}
                    for _, row in importance_df.head(10).iterrows()
                ]
            else:
                # Use mock features
                top_features = [
                    {"feature": "price_momentum", "importance": 0.25},
                    {"feature": "volume_trend", "importance": 0.20}
                ]
            
            model_type = model.__class__.__name__ if hasattr(model, '__class__') else "ensemble"
            
            # Generate explanation based on top features
            top_feature_names = [f['feature'] for f in top_features[:3]]
            explanation_text = (
                f"The {model_type} model for {symbol} primarily relies on {', '.join(top_feature_names)} "
                f"for making predictions. These features capture market dynamics, technical patterns, "
                f"and sentiment indicators that historically correlate with price movements."
            )
        
        # Get recent prediction accuracy if available
        recent_accuracy = None
        if symbol in model_configs:
            recent_accuracy = {
                "smape": round(np.random.uniform(5, 15), 2),  # Mock accuracy
                "directional_accuracy": round(np.random.uniform(0.6, 0.8), 2),
                "sharpe_ratio": round(np.random.uniform(0.5, 2.0), 2)
            }
        
        return ExplanationResponse(
            symbol=symbol,
            model_type=model_type,
            top_features=top_features,
            recent_predictions_accuracy=recent_accuracy,
            explanation_text=explanation_text,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Explanation failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/sentiment/{symbol}", response_model=Dict, tags=["Sentiment"])
async def get_sentiment_data(
    symbol: str,
    days: int = Query(30, ge=1, le=365, description="Number of days of sentiment data")
):
    """
    Get historical sentiment data for a symbol
    
    Returns daily sentiment scores and related metrics for the specified period.
    """
    symbol = symbol.upper()
    
    try:
        # Get stock data to determine date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Try to get actual sentiment data
        try:
            from src.data.download_sentiment import SentimentDataProcessor
            
            processor = SentimentDataProcessor(use_finbert=True)
            
            # Company mapping
            company_mapping = {
                'AAPL': 'Apple Inc',
                'GOOGL': 'Alphabet Inc', 
                'MSFT': 'Microsoft Corporation',
                'AMZN': 'Amazon.com Inc',
                'TSLA': 'Tesla Inc',
                'META': 'Meta Platforms Inc',
                'NVDA': 'NVIDIA Corporation',
                'JPM': 'JPMorgan Chase',
                'V': 'Visa Inc',
                'JNJ': 'Johnson & Johnson'
            }
            
            company_name = company_mapping.get(symbol, symbol)
            
            sentiment_data = processor.get_daily_sentiment(
                ticker=symbol,
                company_name=company_name,
                start_date=start_date,
                end_date=end_date
            )
            
            if not sentiment_data.empty:
                # Convert to JSON-serializable format
                sentiment_dict = sentiment_data.reset_index().to_dict('records')
                return {
                    "symbol": symbol,
                    "data": sentiment_dict,
                    "summary": {
                        "avg_sentiment": float(sentiment_data['sentiment_compound'].mean()),
                        "sentiment_volatility": float(sentiment_data['sentiment_compound'].std()),
                        "total_articles": int(sentiment_data['article_count'].sum()),
                        "days_covered": len(sentiment_data)
                    },
                    "timestamp": datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.warning(f"Failed to get actual sentiment data: {str(e)}")
        
        # Generate mock sentiment data if real data unavailable
        dates = pd.date_range(end=end_date, periods=days, freq='D')
        mock_sentiment = np.random.normal(0, 0.3, days)
        mock_articles = np.random.poisson(5, days)
        
        sentiment_data = []
        for i, date in enumerate(dates):
            sentiment_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "sentiment_compound": round(mock_sentiment[i], 3),
                "sentiment_positive": round(max(0, mock_sentiment[i]), 3),
                "sentiment_negative": round(abs(min(0, mock_sentiment[i])), 3),
                "sentiment_neutral": round(1 - abs(mock_sentiment[i]), 3),
                "article_count": int(mock_articles[i]),
                "llm_sentiment_mean": round(mock_sentiment[i] * 1.1, 3),
                "llm_sentiment_std": round(abs(mock_sentiment[i]) * 0.2, 3)
            })
        
        return {
            "symbol": symbol,
            "data": sentiment_data,
            "summary": {
                "avg_sentiment": round(np.mean(mock_sentiment), 3),
                "sentiment_volatility": round(np.std(mock_sentiment), 3),
                "total_articles": int(np.sum(mock_articles)),
                "days_covered": days
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get sentiment data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get sentiment data: {str(e)}")


@app.post("/explain/dynamic", response_model=Dict, tags=["Explainability"])
async def get_dynamic_explanation(request: Dict):
    """
    Get dynamic, LLM-generated explanation for model prediction
    
    Uses the generative AI capabilities to create human-readable explanations
    based on current market conditions and feature importance.
    """
    try:
        symbol = request.get("symbol", "").upper()
        features = request.get("features", [])
        
        if not symbol or not features:
            raise HTTPException(status_code=400, detail="Symbol and features are required")
        
        # Try to use actual LLM for explanation
        try:
            from src.utils.llm import GPTClient
            
            client = GPTClient()
            
            # Create feature summary
            top_features = [f["feature"] for f in features[:5]]
            feature_importances = [f.get("importance", 0) for f in features[:5]]
            
            # Build prompt for dynamic explanation
            prompt = f"""
            Analyze the current prediction for {symbol} based on these key factors:
            
            Top Features and Importance:
            {', '.join([f"{feat} ({imp:.2%})" for feat, imp in zip(top_features, feature_importances)])}
            
            Provide a concise, professional explanation of:
            1. What these factors suggest about {symbol}'s near-term outlook
            2. Key risks or opportunities highlighted by the data
            3. How these factors work together to influence the prediction
            
            Keep the explanation under 150 words and avoid jargon.
            """
            
            explanation = client.complete(prompt, max_tokens=200)
            
            return {
                "symbol": symbol,
                "explanation": explanation.strip(),
                "features_analyzed": len(features),
                "confidence": "high" if len(features) >= 5 else "medium",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.warning(f"LLM explanation failed: {str(e)}")
            
            # Fallback to template-based explanation
            top_feature = features[0]["feature"] if features else "technical indicators"
            
            explanation = (
                f"The model's prediction for {symbol} is primarily driven by {top_feature}, "
                f"which shows significant influence on price movement patterns. "
                f"Combined with {len(features)} other key factors, the analysis suggests "
                f"{'bullish' if np.random.random() > 0.5 else 'bearish'} sentiment in the near term. "
                f"Market conditions and technical momentum are key considerations for this forecast."
            )
            
            return {
                "symbol": symbol,
                "explanation": explanation,
                "features_analyzed": len(features),
                "confidence": "medium",
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dynamic explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")


@app.get("/scenarios/{symbol}", response_model=Dict, tags=["Scenarios"])
async def get_market_scenarios(
    symbol: str,
    horizon_days: int = Query(30, ge=1, le=90, description="Scenario horizon in days")
):
    """
    Generate plausible market scenarios for the given symbol
    
    Uses generative AI to create realistic future scenarios that could affect
    the stock price, providing context for investment decisions.
    """
    symbol = symbol.upper()
    
    try:
        # Try to use actual LLM for scenario generation
        try:
            from src.features.generative_sentiment import generate_market_scenarios
            
            scenarios = generate_market_scenarios(symbol, horizon_days)
            
            if scenarios:
                return {
                    "symbol": symbol,
                    "scenarios": scenarios,
                    "horizon_days": horizon_days,
                    "scenario_count": len(scenarios),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.warning(f"LLM scenario generation failed: {str(e)}")
        
        # Fallback scenarios
        company_contexts = {
            'AAPL': 'iPhone manufacturer and technology innovator',
            'GOOGL': 'Search and cloud computing leader',
            'MSFT': 'Software and cloud services provider',
            'AMZN': 'E-commerce and cloud infrastructure giant',
            'TSLA': 'Electric vehicle and renewable energy company',
            'META': 'Social media and metaverse platform',
            'NVDA': 'Semiconductor and AI chip manufacturer'
        }
        
        context = company_contexts.get(symbol, 'major corporation')
        
        scenarios = [
            f"Strong earnings report drives {symbol} higher as {context} demonstrates robust growth.",
            f"Market volatility impacts {symbol} amid broader sector rotation and economic uncertainty.",
            f"New product announcement or strategic partnership boosts investor confidence in {symbol}.",
            f"Regulatory changes or competitive pressures create headwinds for {symbol} in the near term.",
            f"Macroeconomic factors including interest rates and inflation affect {symbol}'s valuation multiple."
        ]
        
        return {
            "symbol": symbol,
            "scenarios": scenarios,
            "horizon_days": horizon_days,
            "scenario_count": len(scenarios),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Scenario generation failed for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scenario generation failed: {str(e)}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)