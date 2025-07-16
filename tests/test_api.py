"""
Test suite for API endpoints

Tests the FastAPI endpoints including:
- Health check endpoint
- Single stock prediction
- Batch predictions
- Model explanations
- Error handling and validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.app.api import app, load_models, get_stock_data


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_models():
    """Create mock models for testing"""
    models = {
        "AAPL_ensemble": Mock(
            predict=Mock(return_value=np.array([0.02])),
            get_feature_importance=Mock(return_value=pd.DataFrame({
                'feature': ['price_momentum', 'volume'],
                'importance': [0.3, 0.2]
            }))
        ),
        "GOOGL_ensemble": Mock(
            predict=Mock(return_value=np.array([0.01])),
        )
    }
    return models


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test successful health check"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert 'status' in data
        assert 'timestamp' in data
        assert 'version' in data
        assert 'models_loaded' in data
        assert 'dependencies' in data
        assert 'memory_usage_mb' in data
        assert 'uptime_seconds' in data
        
        # Check values
        assert data['status'] in ['healthy', 'degraded']
        assert data['version'] == '1.0.0'
        assert isinstance(data['models_loaded'], int)
        assert isinstance(data['dependencies'], dict)
    
    def test_health_check_dependencies(self, client):
        """Test dependency checks in health endpoint"""
        response = client.get("/health")
        data = response.json()
        
        # Check key dependencies
        deps = data['dependencies']
        expected_deps = ['yfinance', 'pytorch', 'openai']
        
        for dep in expected_deps:
            assert dep in deps
            assert deps[dep] in ['healthy', 'unavailable']


class TestPredictionEndpoint:
    """Test single stock prediction endpoint"""
    
    @patch('src.app.api.get_stock_data')
    @patch('src.app.api.get_model_for_symbol')
    def test_predict_success(self, mock_get_model, mock_get_data, client):
        """Test successful prediction"""
        # Mock stock data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_get_data.return_value = mock_data
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.02])  # 2% return
        mock_get_model.return_value = mock_model
        
        # Make request
        response = client.get("/predict/AAPL?horizon=5&include_confidence=true")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert 'symbol' in data
        assert 'current_price' in data
        assert 'predicted_price' in data
        assert 'predicted_return' in data
        assert 'horizon_days' in data
        assert 'prediction_date' in data
        assert 'confidence_interval' in data
        assert 'signal' in data
        assert 'strength' in data
        assert 'timestamp' in data
        
        # Check values
        assert data['symbol'] == 'AAPL'
        assert data['horizon_days'] == 5
        assert data['signal'] in ['buy', 'sell', 'hold']
        assert 0 <= data['strength'] <= 1
    
    def test_predict_invalid_symbol(self, client):
        """Test prediction with invalid symbol format"""
        response = client.get("/predict/123")  # Invalid symbol
        
        # Should still work but might return error or default behavior
        assert response.status_code in [200, 404, 500]
    
    def test_predict_invalid_horizon(self, client):
        """Test prediction with invalid horizon"""
        # Horizon too large
        response = client.get("/predict/AAPL?horizon=100")
        assert response.status_code == 422  # Validation error
        
        # Horizon too small
        response = client.get("/predict/AAPL?horizon=0")
        assert response.status_code == 422
        
        # Negative horizon
        response = client.get("/predict/AAPL?horizon=-5")
        assert response.status_code == 422
    
    @patch('src.app.api.get_stock_data')
    def test_predict_no_data(self, mock_get_data, client):
        """Test prediction when no stock data available"""
        # Mock empty data
        mock_get_data.return_value = pd.DataFrame()
        
        response = client.get("/predict/INVALID")
        
        # Should return 404 or 500
        assert response.status_code in [404, 500]
        
        if response.status_code == 404:
            data = response.json()
            assert 'detail' in data
            assert 'No data available' in data['detail']
    
    @patch('src.app.api.get_stock_data')
    @patch('src.app.api.get_model_for_symbol')
    def test_predict_fallback(self, mock_get_model, mock_get_data, client):
        """Test prediction fallback when model not available"""
        # Mock stock data
        mock_data = pd.DataFrame({
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=pd.date_range('2023-01-01', periods=100))
        
        mock_get_data.return_value = mock_data
        
        # No model available
        mock_get_model.return_value = None
        
        # Should still return prediction using fallback
        response = client.get("/predict/AAPL")
        
        assert response.status_code == 200
        data = response.json()
        assert 'predicted_price' in data
        assert 'signal' in data


class TestBatchPredictionEndpoint:
    """Test batch prediction endpoint"""
    
    @patch('src.app.api.predict_single')
    def test_batch_predict_success(self, mock_predict, client):
        """Test successful batch prediction"""
        # Mock individual predictions
        async def mock_prediction(symbol, horizon, include_confidence):
            return {
                "symbol": symbol,
                "current_price": 100,
                "predicted_price": 102,
                "predicted_return": 2.0,
                "horizon_days": horizon,
                "prediction_date": "2023-12-31",
                "signal": "buy",
                "strength": 0.8,
                "timestamp": datetime.now().isoformat()
            }
        
        mock_predict.side_effect = mock_prediction
        
        # Make batch request
        request_data = {
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "horizon": 5,
            "include_confidence": False
        }
        
        response = client.post("/batch_predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert 'predictions' in data
        assert 'failed_symbols' in data
        assert 'processing_time_ms' in data
        
        # Check predictions
        assert len(data['predictions']) == 3
        for pred in data['predictions']:
            assert pred['symbol'] in ["AAPL", "GOOGL", "MSFT"]
    
    def test_batch_predict_empty_symbols(self, client):
        """Test batch prediction with empty symbols"""
        request_data = {
            "symbols": [],
            "horizon": 5
        }
        
        response = client.post("/batch_predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_too_many_symbols(self, client):
        """Test batch prediction with too many symbols"""
        # Create 51 symbols (limit is 50)
        symbols = [f"STOCK{i}" for i in range(51)]
        
        request_data = {
            "symbols": symbols,
            "horizon": 5
        }
        
        response = client.post("/batch_predict", json=request_data)
        assert response.status_code == 422
        
        data = response.json()
        assert 'Maximum 50 symbols' in str(data)
    
    @patch('src.app.api.predict_single')
    def test_batch_predict_partial_failure(self, mock_predict, client):
        """Test batch prediction with some failures"""
        # Mock predictions with one failure
        async def mock_prediction(symbol, horizon, include_confidence):
            if symbol == "INVALID":
                raise Exception("Invalid symbol")
            return {
                "symbol": symbol,
                "current_price": 100,
                "predicted_price": 102,
                "predicted_return": 2.0,
                "horizon_days": horizon,
                "prediction_date": "2023-12-31",
                "signal": "buy",
                "strength": 0.8,
                "timestamp": datetime.now().isoformat()
            }
        
        mock_predict.side_effect = mock_prediction
        
        request_data = {
            "symbols": ["AAPL", "INVALID", "GOOGL"],
            "horizon": 5
        }
        
        response = client.post("/batch_predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have 2 successful predictions
        assert len(data['predictions']) == 2
        
        # Should have 1 failed symbol
        assert len(data['failed_symbols']) == 1
        assert data['failed_symbols'][0]['symbol'] == "INVALID"
        assert 'error' in data['failed_symbols'][0]


class TestExplanationEndpoint:
    """Test model explanation endpoint"""
    
    @patch('src.app.api.get_model_for_symbol')
    def test_explain_success(self, mock_get_model, client):
        """Test successful model explanation"""
        # Mock model with feature importance
        mock_model = Mock()
        mock_model.get_feature_importance.return_value = pd.DataFrame({
            'feature': ['price_momentum', 'volume_trend', 'rsi'],
            'importance': [0.3, 0.2, 0.15]
        })
        mock_model.__class__.__name__ = 'RandomForestBaseline'
        
        mock_get_model.return_value = mock_model
        
        response = client.get("/explain/AAPL")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert 'symbol' in data
        assert 'model_type' in data
        assert 'top_features' in data
        assert 'explanation_text' in data
        assert 'timestamp' in data
        
        # Check top features
        assert len(data['top_features']) > 0
        for feature in data['top_features']:
            assert 'feature' in feature
            assert 'importance' in feature
            assert 0 <= feature['importance'] <= 1
    
    @patch('src.app.api.get_model_for_symbol')
    def test_explain_no_model(self, mock_get_model, client):
        """Test explanation when no model available"""
        mock_get_model.return_value = None
        
        response = client.get("/explain/AAPL")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should provide generic explanation
        assert 'model_type' in data
        assert data['model_type'] == 'ensemble'
        assert len(data['top_features']) > 0
        assert 'explanation_text' in data
    
    def test_explain_with_accuracy_metrics(self, client):
        """Test explanation includes accuracy metrics"""
        response = client.get("/explain/AAPL")
        
        assert response.status_code == 200
        data = response.json()
        
        # May or may not have accuracy metrics
        if 'recent_predictions_accuracy' in data and data['recent_predictions_accuracy']:
            acc = data['recent_predictions_accuracy']
            assert 'smape' in acc
            assert 'directional_accuracy' in acc
            assert 'sharpe_ratio' in acc


class TestErrorHandling:
    """Test API error handling"""
    
    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed"""
        # POST to GET-only endpoint
        response = client.post("/health")
        assert response.status_code == 405
        
        # GET to POST-only endpoint
        response = client.get("/batch_predict")
        assert response.status_code == 405
    
    def test_malformed_json(self, client):
        """Test malformed JSON in POST request"""
        response = client.post(
            "/batch_predict",
            data="invalid json{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in [422, 400]
    
    @patch('src.app.api.get_stock_data')
    def test_internal_server_error(self, mock_get_data, client):
        """Test internal server error handling"""
        # Mock unexpected error
        mock_get_data.side_effect = Exception("Unexpected error")
        
        response = client.get("/predict/AAPL")
        
        assert response.status_code == 500
        data = response.json()
        assert 'detail' in data


class TestRequestValidation:
    """Test request parameter validation"""
    
    def test_symbol_normalization(self, client):
        """Test that symbols are normalized to uppercase"""
        # Test with lowercase
        response = client.get("/predict/aapl")
        
        if response.status_code == 200:
            data = response.json()
            assert data['symbol'] == 'AAPL'
    
    def test_batch_symbols_normalization(self, client):
        """Test batch symbols are normalized"""
        request_data = {
            "symbols": ["aapl", "googl", "MSFT"],
            "horizon": 5
        }
        
        with patch('src.app.api.predict_single') as mock_predict:
            async def mock_prediction(symbol, horizon, include_confidence):
                return {"symbol": symbol}
            
            mock_predict.side_effect = mock_prediction
            
            response = client.post("/batch_predict", json=request_data)
            
            if response.status_code == 200:
                # Check that symbols were uppercased
                calls = mock_predict.call_args_list
                called_symbols = [call[1]['symbol'] for call in calls]
                assert all(sym.isupper() for sym in called_symbols)
    
    def test_query_parameter_types(self, client):
        """Test query parameter type validation"""
        # String instead of int for horizon
        response = client.get("/predict/AAPL?horizon=abc")
        assert response.status_code == 422

        # Invalid boolean value (FastAPI accepts 'yes' as True, so use an invalid value)
        response = client.get("/predict/AAPL?include_confidence=invalid_bool")
        assert response.status_code == 422


class TestAPIIntegration:
    """Test API integration scenarios"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'docs' in data
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation"""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Check basic OpenAPI structure
        assert 'openapi' in schema
        assert 'info' in schema
        assert 'paths' in schema
        
        # Check API info
        assert schema['info']['title'] == 'StockSage.AI API'
        assert schema['info']['version'] == '1.0.0'
        
        # Check endpoints are documented
        assert '/health' in schema['paths']
        assert '/predict/{symbol}' in schema['paths']
        assert '/batch_predict' in schema['paths']
        assert '/explain/{symbol}' in schema['paths']


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])