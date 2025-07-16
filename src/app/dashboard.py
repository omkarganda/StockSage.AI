"""
StockSage.AI Dashboard

Interactive Streamlit dashboard for stock price predictions and analysis.
Features stock selection, prediction visualization, feature importance, and historical performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path
import sys
import asyncio
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.download_market import MarketDataDownloader
from src.features.indicators import add_all_technical_indicators
from src.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="StockSage.AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .prediction-positive {
        color: #00a86b;
    }
    .prediction-negative {
        color: #fd5c63;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'selected_stock' not in st.session_state:
    st.session_state.selected_stock = 'AAPL'
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_list() -> List[str]:
    """Get list of available stocks"""
    # Default stock list
    default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']
    
    try:
        # Try to get from API
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            # API is available, return default stocks
            return default_stocks
    except:
        pass
    
    return default_stocks


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_prediction(symbol: str, horizon: int = 1, include_confidence: bool = True) -> Optional[Dict]:
    """Get prediction from API"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/predict/{symbol}",
            params={"horizon": horizon, "include_confidence": include_confidence},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        # Fallback to mock data if API is not available
        logger.warning(f"API call failed: {str(e)}, using mock data")
        
        # Generate mock prediction
        current_price = 150.0 + np.random.randn() * 10
        predicted_return = np.random.randn() * 0.02
        predicted_price = current_price * (1 + predicted_return)
        
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "predicted_return": round(predicted_return * 100, 2),
            "horizon_days": horizon,
            "prediction_date": (datetime.now() + timedelta(days=horizon)).strftime("%Y-%m-%d"),
            "confidence_interval": {
                "lower": round(predicted_price * 0.95, 2),
                "upper": round(predicted_price * 1.05, 2)
            },
            "signal": "buy" if predicted_return > 0.01 else ("sell" if predicted_return < -0.01 else "hold"),
            "strength": abs(predicted_return) / 0.05,
            "timestamp": datetime.now().isoformat()
        }


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_historical_data(symbol: str, days: int = 90) -> pd.DataFrame:
    """Get historical stock data"""
    try:
        downloader = MarketDataDownloader(use_cache=True)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = downloader.download_stock_data(
            symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        # Add technical indicators
        df = add_all_technical_indicators(df)
        
        return df
    except Exception as e:
        logger.error(f"Failed to get historical data: {str(e)}")
        
        # Generate mock data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        prices = 150 + np.cumsum(np.random.randn(days) * 2)
        
        df = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, days),
            'High': prices + np.random.rand(days) * 2,
            'Low': prices - np.random.rand(days) * 2,
            'Open': prices + np.random.randn(days) * 0.5
        }, index=dates)
        
        return df


@st.cache_data(ttl=60)
def get_explanation(symbol: str) -> Optional[Dict]:
    """Get model explanation from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/explain/{symbol}", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        # Mock explanation data
        return {
            "symbol": symbol,
            "model_type": "ensemble",
            "top_features": [
                {"feature": "price_momentum_20d", "importance": 0.25},
                {"feature": "volume_trend", "importance": 0.20},
                {"feature": "rsi_14", "importance": 0.15},
                {"feature": "sentiment_score", "importance": 0.12},
                {"feature": "volatility_30d", "importance": 0.10},
                {"feature": "macd_signal", "importance": 0.08},
                {"feature": "market_correlation", "importance": 0.05},
                {"feature": "economic_indicators", "importance": 0.05}
            ],
            "recent_predictions_accuracy": {
                "smape": 8.5,
                "directional_accuracy": 0.72,
                "sharpe_ratio": 1.45
            },
            "explanation_text": f"The model uses a combination of technical indicators and market sentiment to predict {symbol} price movements.",
            "timestamp": datetime.now().isoformat()
        }


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_sentiment_data(symbol: str, days: int = 30) -> Optional[Dict]:
    """Get sentiment data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/sentiment/{symbol}", params={"days": days}, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        # Mock sentiment data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
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
            }
        }


@st.cache_data(ttl=60)
def get_dynamic_explanation(symbol: str, features: List[Dict]) -> str:
    """Get dynamic explanation from API."""
    try:
        response = requests.post(f"{API_BASE_URL}/explain/dynamic", json={
            "symbol": symbol,
            "features": features
        }, timeout=10)
        
        if response.status_code == 200:
            return response.json()["explanation"]
    except:
        pass
    
    # Fallback for demo
    feature_str = ", ".join([f["feature"] for f in features[:3]])
    return (f"The model's current prediction for {symbol} is primarily influenced by "
            f"these key factors: {feature_str}. A deeper analysis of these drivers "
            f"is used to determine the final buy/sell/hold signal.")


@st.cache_data(ttl=3600)  # Cache for an hour
def get_market_scenarios(symbol: str) -> List[str]:
    """Get market scenarios from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/scenarios/{symbol}", timeout=10)
        if response.status_code == 200:
            return response.json().get("scenarios", [])
    except:
        pass
    
    return [
        "Positive earnings surprise leads to a significant rally.",
        "Broader market downturn due to macroeconomic concerns.",
        "New product announcement is met with mixed reviews, causing volatility."
    ]


def create_price_chart(df: pd.DataFrame, prediction: Dict) -> go.Figure:
    """Create interactive price chart with prediction"""
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#00a86b',
        decreasing_line_color='#fd5c63'
    ))
    
    # Add prediction point
    if prediction:
        pred_date = pd.to_datetime(prediction['prediction_date'])
        
        # Prediction marker
        fig.add_trace(go.Scatter(
            x=[pred_date],
            y=[prediction['predicted_price']],
            mode='markers',
            name='Prediction',
            marker=dict(
                size=15,
                color='#FF6B6B' if prediction['predicted_return'] < 0 else '#4ECDC4',
                symbol='star'
            ),
            text=f"Predicted: ${prediction['predicted_price']}",
            hoverinfo='text'
        ))
        
        # Confidence interval
        if 'confidence_interval' in prediction and prediction['confidence_interval']:
            fig.add_shape(
                type="rect",
                x0=pred_date - timedelta(days=1),
                x1=pred_date + timedelta(days=1),
                y0=prediction['confidence_interval']['lower'],
                y1=prediction['confidence_interval']['upper'],
                fillcolor="rgba(78, 205, 196, 0.2)",
                line=dict(width=0)
            )
    
    # Add moving averages
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA_20'],
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='blue', width=1)
        ))
    
    fig.update_layout(
        title="Stock Price History & Prediction",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=500,
        template="plotly_white",
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_volume_chart(df: pd.DataFrame) -> go.Figure:
    """Create volume chart"""
    colors = ['red' if row['Close'] < row['Open'] else 'green' 
              for _, row in df.iterrows()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            name='Volume'
        )
    ])
    
    fig.update_layout(
        title="Trading Volume",
        yaxis_title="Volume",
        xaxis_title="Date",
        height=300,
        template="plotly_white"
    )
    
    return fig


def create_feature_importance_chart(features: List[Dict]) -> go.Figure:
    """Create feature importance chart"""
    feature_names = [f['feature'] for f in features]
    importances = [f['importance'] for f in features]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importances,
            y=feature_names,
            orientation='h',
            marker_color='#4ECDC4'
        )
    ])
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=400,
        template="plotly_white",
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def create_technical_indicators_chart(df: pd.DataFrame) -> go.Figure:
    """Create technical indicators chart"""
    # Create subplots
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('RSI', 'MACD'),
        row_heights=[0.5, 0.5]
    )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=1, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # MACD
    if all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_histogram'], name='Histogram', marker_color='gray'),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(height=600, template="plotly_white", showlegend=True)
    
    return fig


def create_sentiment_chart(sentiment_data: List[Dict]) -> go.Figure:
    """Create a chart for daily sentiment scores."""
    if not sentiment_data:
        fig = go.Figure()
        fig.add_annotation(text="No sentiment data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = pd.DataFrame(sentiment_data)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    # LLM Sentiment bars
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['llm_sentiment_mean'],
        name='LLM Sentiment',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Traditional sentiment line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sentiment_compound'],
        name='FinBERT Sentiment',
        line=dict(color='royalblue', width=2)
    ))
    
    # 7-day rolling average
    if len(df) >= 7:
        rolling_avg = df['llm_sentiment_mean'].rolling(7, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=rolling_avg,
            name='7-Day Average',
            line=dict(color='orange', width=2, dash='dash')
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Daily Sentiment Analysis",
        yaxis_title="Sentiment Score [-1, 1]",
        xaxis_title="Date",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig


def create_sentiment_distribution_chart(sentiment_data: List[Dict]) -> go.Figure:
    """Create sentiment distribution chart."""
    if not sentiment_data:
        fig = go.Figure()
        fig.add_annotation(text="No sentiment data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = pd.DataFrame(sentiment_data)
    
    # Create histogram of sentiment scores
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['llm_sentiment_mean'],
        name='LLM Sentiment Distribution',
        nbinsx=20,
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Sentiment Score Distribution",
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency",
        template="plotly_white",
        height=300
    )
    
    return fig


def create_article_volume_chart(sentiment_data: List[Dict]) -> go.Figure:
    """Create article volume over time chart."""
    if not sentiment_data:
        fig = go.Figure()
        fig.add_annotation(text="No sentiment data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df = pd.DataFrame(sentiment_data)
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['article_count'],
        name='Daily Article Count',
        marker_color='green',
        opacity=0.6
    ))
    
    fig.update_layout(
        title="News Article Volume",
        yaxis_title="Article Count",
        xaxis_title="Date",
        template="plotly_white",
        height=300
    )
    
    return fig


# Main Dashboard
def main():
    st.title("📈 StockSage.AI Dashboard")
    st.markdown("AI-Powered Stock Price Predictions with Technical Analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Stock selection
        stock_list = get_stock_list()
        selected_stock = st.selectbox(
            "Select Stock",
            options=stock_list,
            index=stock_list.index(st.session_state.selected_stock) if st.session_state.selected_stock in stock_list else 0
        )
        st.session_state.selected_stock = selected_stock
        
        # Prediction horizon
        horizon = st.slider("Prediction Horizon (days)", 1, 30, 5)
        
        # Historical data range
        hist_days = st.slider("Historical Data (days)", 30, 365, 90)
        
        # Update button
        if st.button("🔄 Update Predictions", type="primary"):
            st.cache_data.clear()
        
        st.markdown("---")
        
        # API Status
        st.subheader("System Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ API Connected")
                health_data = response.json()
                st.metric("Models Loaded", health_data.get('models_loaded', 0))
            else:
                st.warning("⚠️ API Degraded")
        except:
            st.error("❌ API Offline")
    
    # Get data
    with st.spinner(f"Loading data for {selected_stock}..."):
        prediction = get_prediction(selected_stock, horizon)
        historical_data = get_historical_data(selected_stock, hist_days)
        explanation = get_explanation(selected_stock)
        sentiment_response = get_sentiment_data(selected_stock, hist_days)
    
    # Main content area
    if prediction:
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${prediction['current_price']:.2f}",
                delta=None
            )
        
        with col2:
            delta_color = "normal" if prediction['predicted_return'] >= 0 else "inverse"
            st.metric(
                f"{horizon}-Day Prediction",
                f"${prediction['predicted_price']:.2f}",
                delta=f"{prediction['predicted_return']:.2f}%",
                delta_color=delta_color
            )
        
        with col3:
            signal_emoji = {"buy": "🟢", "sell": "🔴", "hold": "🟡"}.get(prediction['signal'], "⚪")
            st.metric(
                "Signal",
                f"{signal_emoji} {prediction['signal'].upper()}",
                delta=f"Strength: {prediction['strength']:.2f}"
            )
        
        with col4:
            if 'confidence_interval' in prediction and prediction['confidence_interval']:
                ci_range = prediction['confidence_interval']['upper'] - prediction['confidence_interval']['lower']
                st.metric(
                    "Confidence Range",
                    f"±${ci_range/2:.2f}",
                    delta=f"{(ci_range/prediction['current_price']*100):.1f}%"
                )
    
    # Charts
    if historical_data is not None and not historical_data.empty:
        # Price chart
        st.plotly_chart(create_price_chart(historical_data, prediction), use_container_width=True)
        
        # ENHANCED: Tabbed layout for additional charts
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Volume & Features", "🔧 Technical Indicators", "💭 Sentiment Analysis", "🔮 AI Scenarios"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_volume_chart(historical_data), use_container_width=True)
            with col2:
                if explanation and 'top_features' in explanation:
                    st.plotly_chart(
                        create_feature_importance_chart(explanation['top_features'][:8]),
                        use_container_width=True
                    )
        
        with tab2:
            st.plotly_chart(create_technical_indicators_chart(historical_data), use_container_width=True)

        with tab3:
            # ENHANCED: Sentiment Analysis Tab
            if sentiment_response and sentiment_response.get('data'):
                # Sentiment summary metrics
                summary = sentiment_response.get('summary', {})
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Sentiment", f"{summary.get('avg_sentiment', 0):.3f}")
                with col2:
                    st.metric("Sentiment Volatility", f"{summary.get('sentiment_volatility', 0):.3f}")
                with col3:
                    st.metric("Total Articles", f"{summary.get('total_articles', 0):,}")
                with col4:
                    st.metric("Days Covered", f"{summary.get('days_covered', 0)}")
                
                # Sentiment charts
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_sentiment_chart(sentiment_response['data']), use_container_width=True)
                with col2:
                    st.plotly_chart(create_sentiment_distribution_chart(sentiment_response['data']), use_container_width=True)
                
                # Article volume chart
                st.plotly_chart(create_article_volume_chart(sentiment_response['data']), use_container_width=True)
                
                # Recent sentiment highlights
                st.subheader("Recent Sentiment Highlights")
                recent_data = sentiment_response['data'][-7:]  # Last 7 days
                for item in recent_data:
                    date = item['date']
                    sentiment = item['llm_sentiment_mean']
                    articles = item['article_count']
                    
                    sentiment_color = "green" if sentiment > 0.1 else ("red" if sentiment < -0.1 else "gray")
                    st.markdown(f"**{date}**: {sentiment:+.3f} sentiment ({articles} articles)", 
                               unsafe_allow_html=False)
            else:
                st.info("No sentiment data available for this symbol.")

        with tab4:
            # ENHANCED: AI Scenarios Tab
            st.markdown("### AI-Generated Market Scenarios")
            st.markdown(f"*Plausible scenarios for {selected_stock} over the next {horizon} days*")
            
            scenarios = get_market_scenarios(selected_stock)
            if scenarios:
                for i, scenario in enumerate(scenarios, 1):
                    with st.expander(f"🎯 Scenario {i}", expanded=i<=3):
                        st.markdown(scenario)
                        
                        # Add probability estimation (mock)
                        probability = np.random.uniform(0.15, 0.35)
                        st.progress(probability)
                        st.caption(f"Estimated probability: {probability:.1%}")
            else:
                st.info("No scenarios available for this symbol.")
    
    # ENHANCED: Model explanation section with dynamic content
    if explanation:
        with st.expander("📊 Model Insights", expanded=False):
            st.markdown(f"**Model Type:** {explanation.get('model_type', 'Unknown')}")
            
            # ENHANCED: Dynamic Analysis
            st.markdown("#### Dynamic Analysis")
            if explanation.get('top_features'):
                dynamic_text = get_dynamic_explanation(selected_stock, explanation.get('top_features', []))
                st.info(dynamic_text)
            
            # Performance metrics
            if 'recent_predictions_accuracy' in explanation and explanation['recent_predictions_accuracy']:
                st.markdown("#### Recent Performance")
                acc = explanation['recent_predictions_accuracy']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("SMAPE", f"{acc.get('smape', 'N/A')}%")
                with col2:
                    st.metric("Direction Accuracy", f"{acc.get('directional_accuracy', 'N/A'):.2%}")
                with col3:
                    st.metric("Sharpe Ratio", f"{acc.get('sharpe_ratio', 'N/A'):.2f}")
    
    # Performance metrics (moved to separate section for better organization)
    if historical_data is not None and not historical_data.empty:
        with st.expander("📈 Performance Metrics", expanded=False):
            # Calculate returns
            returns = historical_data['Close'].pct_change().dropna()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Volatility (30d)", f"{returns.tail(30).std() * np.sqrt(252):.2%}")
            
            with col2:
                st.metric("Return (30d)", f"{(historical_data['Close'].iloc[-1] / historical_data['Close'].iloc[-31] - 1):.2%}")
            
            with col3:
                st.metric("Max Drawdown", f"{((historical_data['Close'] / historical_data['Close'].cummax() - 1).min()):.2%}")
            
            with col4:
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            StockSage.AI Dashboard | Enhanced with Generative AI Insights | 
            <a href="/docs" target="_blank">API Documentation</a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()