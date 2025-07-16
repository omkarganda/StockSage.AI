"""
Test suite for model components

Tests model functionality including:
- Model training and fitting
- Prediction accuracy
- Feature importance extraction
- Model serialization and loading
- Performance metrics calculation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import joblib
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.baseline import (
    BaselineModel,
    LinearRegressionBaseline,
    RandomForestBaseline,
    EnsembleBaseline,
    create_baseline_models
)


class TestBaselineModel:
    """Test base model functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Generate realistic price data
        returns = np.random.randn(200) * 0.02
        prices = 100 * np.exp(returns.cumsum())
        
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(200) * 0.001),
            'High': prices * (1 + np.abs(np.random.randn(200)) * 0.005),
            'Low': prices * (1 - np.abs(np.random.randn(200)) * 0.005),
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 200),
            'Ticker': 'TEST'
        }, index=dates)
        
        return data
    
    def test_baseline_model_initialization(self):
        """Test BaselineModel initialization"""
        model = BaselineModel(
            target_col='target',
            prediction_horizon=5,
            use_technical_indicators=True,
            random_state=42
        )
        
        assert model.target_col == 'target'
        assert model.prediction_horizon == 5
        assert model.use_technical_indicators is True
        assert model.random_state == 42
        assert model.model is None  # Not fitted yet
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation"""
        model = BaselineModel()
        prepared_data = model._prepare_features(sample_data)
        
        # Check that features are added
        assert 'target' in prepared_data.columns
        assert len(prepared_data.columns) > len(sample_data.columns)
        
        # Check lag features
        assert 'Close_lag_1' in prepared_data.columns
        assert 'Close_return_lag_1' in prepared_data.columns
        
        # Check time features
        assert 'day_of_week' in prepared_data.columns
        assert 'month' in prepared_data.columns
    
    def test_create_lag_features(self, sample_data):
        """Test lag feature creation"""
        model = BaselineModel()
        lagged_data = model._create_lag_features(sample_data)
        
        # Check lag columns exist
        lag_columns = [col for col in lagged_data.columns if 'lag' in col]
        assert len(lag_columns) > 0
        
        # Check rolling statistics
        rolling_columns = [col for col in lagged_data.columns if 'rolling' in col]
        assert len(rolling_columns) > 0
        
        # Verify lag calculation
        assert lagged_data['Close_lag_1'].iloc[1] == sample_data['Close'].iloc[0]
    
    def test_create_time_features(self, sample_data):
        """Test time feature creation"""
        model = BaselineModel()
        time_data = model._create_time_features(sample_data)
        
        # Check time features
        assert 'day_of_week' in time_data.columns
        assert 'day_of_month' in time_data.columns
        assert 'month' in time_data.columns
        assert 'quarter' in time_data.columns
        assert 'is_month_end' in time_data.columns
        
        # Verify values
        assert time_data['day_of_week'].iloc[0] in range(7)
        assert time_data['month'].iloc[0] in range(1, 13)


class TestLinearRegressionBaseline:
    """Test Linear Regression baseline model"""
    
    @pytest.fixture
    def model(self):
        """Create LinearRegressionBaseline instance"""
        return LinearRegressionBaseline(
            regularization='ridge',
            alpha=1.0,
            random_state=42
        )
    
    @pytest.fixture
    def training_data(self):
        """Create training data"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create simple trend with noise
        trend = np.linspace(100, 110, 100)
        noise = np.random.randn(100) * 2
        prices = trend + noise
        
        data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        return data
    
    def test_linear_regression_fit(self, model, training_data):
        """Test model fitting"""
        # Fit model
        model.fit(training_data)
        
        # Check model is fitted
        assert model.model is not None
        assert model.feature_names is not None
        assert model.training_metrics is not None
        
        # Check metrics
        assert 'r2' in model.training_metrics
        assert 'rmse' in model.training_metrics
        assert model.training_metrics['r2'] > -1  # Basic sanity check
    
    def test_linear_regression_predict(self, model, training_data):
        """Test model prediction"""
        # Fit model
        model.fit(training_data)
        
        # Make predictions
        predictions = model.predict(training_data.tail(20))
        
        # Check predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 20
        assert not np.any(np.isnan(predictions))
    
    def test_regularization_options(self, training_data):
        """Test different regularization options"""
        # Test Ridge
        ridge_model = LinearRegressionBaseline(regularization='ridge', alpha=1.0)
        ridge_model.fit(training_data)
        assert ridge_model.model is not None
        
        # Test Lasso
        lasso_model = LinearRegressionBaseline(regularization='lasso', alpha=0.1)
        lasso_model.fit(training_data)
        assert lasso_model.model is not None
        
        # Test ElasticNet
        elastic_model = LinearRegressionBaseline(regularization='elastic', alpha=0.1)
        elastic_model.fit(training_data)
        assert elastic_model.model is not None


class TestRandomForestBaseline:
    """Test Random Forest baseline model"""
    
    @pytest.fixture
    def model(self):
        """Create RandomForestBaseline instance"""
        return RandomForestBaseline(
            n_estimators=10,  # Small for testing
            max_depth=5,
            random_state=42,
            use_grid_search=False  # Disable for faster tests
        )
    
    @pytest.fixture
    def training_data(self):
        """Create training data with patterns"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        
        # Create data with seasonal pattern
        t = np.arange(150)
        seasonal = 10 * np.sin(2 * np.pi * t / 30)
        trend = 0.1 * t
        noise = np.random.randn(150) * 2
        prices = 100 + trend + seasonal + noise
        
        data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 150)
        }, index=dates)
        
        return data
    
    def test_random_forest_fit(self, model, training_data):
        """Test Random Forest fitting"""
        # Fit model
        model.fit(training_data)
        
        # Check model is fitted
        assert model.model is not None
        assert hasattr(model.model, 'feature_importances_')
        
        # Check feature importance
        importance_df = model.get_feature_importance()
        assert not importance_df.empty
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_random_forest_predict(self, model, training_data):
        """Test Random Forest prediction"""
        # Fit model
        model.fit(training_data)
        
        # Make predictions
        test_data = training_data.tail(30)
        predictions = model.predict(test_data)
        
        # Check predictions
        assert len(predictions) == 30
        assert not np.any(np.isnan(predictions))
        
        # Predictions should be reasonable
        assert np.all(predictions > -1)  # Returns should be reasonable
        assert np.all(predictions < 1)
    
    def test_grid_search_optimization(self, training_data):
        """Test grid search hyperparameter optimization"""
        # Create model with grid search
        model = RandomForestBaseline(
            n_estimators=10,
            use_grid_search=True,
            cv_folds=2  # Small for testing
        )
        
        # Fit should perform grid search
        model.fit(training_data)
        
        # Check that best parameters were found
        assert model.model is not None
        assert hasattr(model.model, 'best_params_') or hasattr(model.model, 'n_estimators')


class TestEnsembleBaseline:
    """Test Ensemble baseline model"""
    
    @pytest.fixture
    def model(self):
        """Create EnsembleBaseline instance"""
        return EnsembleBaseline(
            voting='soft',
            weights=None,
            random_state=42
        )
    
    @pytest.fixture
    def training_data(self):
        """Create diverse training data"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        
        # Complex pattern
        t = np.arange(200)
        trend = 0.05 * t
        seasonal = 5 * np.sin(2 * np.pi * t / 30)
        noise = np.random.randn(200) * 3
        prices = 100 + trend + seasonal + noise
        
        data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.02,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 200)
        }, index=dates)
        
        return data
    
    def test_ensemble_fit(self, model, training_data):
        """Test ensemble model fitting"""
        # Fit model
        model.fit(training_data)
        
        # Check models are fitted
        assert model.model is not None
        assert hasattr(model.model, 'estimators_')
        assert len(model.model.estimators_) >= 2  # At least 2 models
    
    def test_ensemble_predict(self, model, training_data):
        """Test ensemble prediction"""
        # Fit model
        model.fit(training_data)
        
        # Make predictions
        test_data = training_data.tail(40)
        predictions = model.predict(test_data)
        
        # Check predictions
        assert len(predictions) == 40
        assert not np.any(np.isnan(predictions))
    
    def test_individual_predictions(self, model, training_data):
        """Test getting individual model predictions"""
        # Fit model
        model.fit(training_data)
        
        # Get individual predictions
        test_data = training_data.tail(20)
        individual_preds = model.get_individual_predictions(test_data)
        
        # Check structure
        assert isinstance(individual_preds, dict)
        assert len(individual_preds) >= 2  # At least 2 models
        
        # Check each model's predictions
        for model_name, preds in individual_preds.items():
            assert len(preds) == 20
            assert not np.any(np.isnan(preds))


class TestModelSerialization:
    """Test model saving and loading"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
    
    def test_save_and_load_model(self, temp_dir):
        """Test model serialization"""
        # Create and fit model
        model = LinearRegressionBaseline(random_state=42)
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 101,
            'Low': np.random.randn(100).cumsum() + 99,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Fit model
        model.fit(data)
        original_metrics = model.training_metrics.copy()
        
        # Save model
        model_path = temp_dir / "test_model.joblib"
        model.save_model(model_path)
        assert model_path.exists()
        
        # Load model
        new_model = LinearRegressionBaseline()
        new_model.load_model(model_path)
        
        # Check loaded model
        assert new_model.model is not None
        assert new_model.feature_names == model.feature_names
        assert new_model.training_metrics == original_metrics
        
        # Test predictions are same
        predictions_original = model.predict(data.tail(30))
        predictions_loaded = new_model.predict(data.tail(30))
        np.testing.assert_array_almost_equal(predictions_original, predictions_loaded)


class TestModelFactory:
    """Test model factory function"""
    
    def test_create_baseline_models(self):
        """Test creating multiple baseline models"""
        models = create_baseline_models(
            target_col='returns',
            prediction_horizon=5,
            random_state=42
        )
        
        # Check models created
        assert isinstance(models, dict)
        assert 'linear_regression' in models
        assert 'random_forest' in models
        assert 'ensemble' in models
        
        # Check model types
        assert isinstance(models['linear_regression'], LinearRegressionBaseline)
        assert isinstance(models['random_forest'], RandomForestBaseline)
        assert isinstance(models['ensemble'], EnsembleBaseline)
        
        # Check configuration
        for model in models.values():
            assert model.target_col == 'returns'
            assert model.prediction_horizon == 5
            assert model.random_state == 42


class TestModelMetrics:
    """Test model evaluation metrics"""
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        model = BaselineModel()
        
        # Create test data
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        # Calculate metrics
        metrics = model._calculate_metrics(y_true, y_pred)
        
        # Check metrics exist
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'smape' in metrics
        
        # Check metric values make sense
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['r2'] <= 1
        assert metrics['smape'] >= 0
    
    def test_evaluate_model(self):
        """Test model evaluation on test set"""
        # Create and fit model
        model = LinearRegressionBaseline(random_state=42)
        
        # Create train/test data
        dates = pd.date_range(start='2023-01-01', periods=150, freq='D')
        prices = 100 + np.arange(150) * 0.1 + np.random.randn(150) * 2
        
        data = pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 150)
        }, index=dates)
        
        # Split data
        train_data = data.iloc[:100]
        test_data = data.iloc[100:]
        
        # Fit and evaluate
        model.fit(train_data)
        test_metrics = model.evaluate(test_data)
        
        # Check metrics
        assert isinstance(test_metrics, dict)
        assert 'rmse' in test_metrics
        assert 'r2' in test_metrics
        assert test_metrics['rmse'] >= 0


# Test configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])