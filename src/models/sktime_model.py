"""
Statistical Forecasting Models for StockSage.AI using sktime

This module provides statistical and classical time series forecasting models
using the sktime ecosystem, including ARIMA, Prophet, Exponential Smoothing,
and ensemble methods optimized for financial data.

Key Features:
- Classical statistical models (ARIMA, ETS, Prophet)
- Ensemble forecasting methods
- Automated model selection and hyperparameter tuning
- Financial time series specific preprocessing
- Regime-aware forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import joblib

# Core sktime imports
try:
    from sktime.forecasting.arima import ARIMA, AutoARIMA
    from sktime.forecasting.ets import ExponentialSmoothing
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.trend import PolynomialTrendForecaster
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.fbprophet import Prophet
    from sktime.forecasting.ensemble import EnsembleForecaster
    from sktime.forecasting.model_selection import (
        temporal_train_test_split, ForecastingGridSearchCV,
        SlidingWindowSplitter, ExpandingWindowSplitter
    )
    from sktime.forecasting.compose import (
        TransformedTargetForecaster, ForecastingPipeline
    )
    from sktime.transformations.series.detrend import Deseasonalizer, Detrender
    from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    from sktime.transformations.series.boxcox import BoxCoxTransformer
    from sktime.transformations.series.difference import Differencer
    from sktime.performance_metrics.forecasting import (
        mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    )
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False
    warnings.warn("sktime not available. Install: pip install sktime")

# Prophet specific import
try:
    from prophet import Prophet as ProphetCore
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install: pip install prophet")

# Additional statistical models
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETSStatsmodels
    from statsmodels.tsa.arima.model import ARIMA as ARIMAStatsmodels
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available for some advanced models")

# Local imports
from ..utils.logging import get_logger
from ..features.indicators import add_all_technical_indicators

# Configure logging
logger = get_logger(__name__)


class StatisticalForecastingModel:
    """
    Base class for statistical forecasting models using sktime.
    
    This class provides a unified interface for various statistical forecasting
    approaches with financial time series specific enhancements.
    """
    
    def __init__(
        self,
        model_type: str = 'auto_arima',
        forecast_horizon: int = 30,
        confidence_level: float = 0.95,
        seasonal_period: Optional[int] = None,
        use_log_transform: bool = True,
        use_differencing: bool = True,
        handle_volatility_clustering: bool = True,
        random_state: int = 42
    ):
        """
        Initialize statistical forecasting model.
        
        Parameters:
        -----------
        model_type : str, default='auto_arima'
            Type of statistical model ('auto_arima', 'prophet', 'ets', 'theta', 'ensemble')
        forecast_horizon : int, default=30
            Number of periods to forecast
        confidence_level : float, default=0.95
            Confidence level for prediction intervals
        seasonal_period : int, optional
            Seasonal period for the data (e.g., 252 for daily financial data)
        use_log_transform : bool, default=True
            Whether to apply log transformation
        use_differencing : bool, default=True
            Whether to apply differencing for stationarity
        handle_volatility_clustering : bool, default=True
            Whether to handle volatility clustering in financial data
        random_state : int, default=42
            Random state for reproducibility
        """
        if not SKTIME_AVAILABLE:
            raise ImportError("sktime is required but not available")
        
        self.model_type = model_type
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level
        self.seasonal_period = seasonal_period
        self.use_log_transform = use_log_transform
        self.use_differencing = use_differencing
        self.handle_volatility_clustering = handle_volatility_clustering
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.transformers = []
        self.is_fitted = False
        self.training_metrics = {}
        self.feature_importance = {}
        
        # Data preprocessing components
        self.log_transformer = None
        self.differencer = None
        self.deseasonalizer = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the forecasting model based on model_type."""
        logger.info(f"Initializing {self.model_type} forecasting model")
        
        try:
            if self.model_type == 'auto_arima':
                self.model = AutoARIMA(
                    start_p=0, max_p=5,
                    start_q=0, max_q=5,
                    seasonal=True if self.seasonal_period else False,
                    m=self.seasonal_period or 1,
                    suppress_warnings=True,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
            elif self.model_type == 'arima':
                self.model = ARIMA(
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, self.seasonal_period) if self.seasonal_period else None,
                    suppress_warnings=True
                )
                
            elif self.model_type == 'prophet' and PROPHET_AVAILABLE:
                self.model = Prophet(
                    seasonality_mode='multiplicative',
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    add_country_holidays={'country_name': 'US'}
                )
                
            elif self.model_type == 'ets':
                self.model = ExponentialSmoothing(
                    trend='add',
                    seasonal='add' if self.seasonal_period else None,
                    sp=self.seasonal_period or 1
                )
                
            elif self.model_type == 'theta':
                self.model = ThetaForecaster(
                    sp=self.seasonal_period or 1
                )
                
            elif self.model_type == 'ensemble':
                self.model = self._create_ensemble_model()
                
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            logger.info(f"Model {self.model_type} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_type}: {e}")
            raise
    
    def _create_ensemble_model(self) -> EnsembleForecaster:
        """Create an ensemble of multiple forecasting models."""
        # Define individual models for ensemble
        forecasters = []
        
        # ARIMA component
        if SKTIME_AVAILABLE:
            arima_model = AutoARIMA(
                start_p=0, max_p=3,
                start_q=0, max_q=3,
                seasonal=True if self.seasonal_period else False,
                m=self.seasonal_period or 1,
                suppress_warnings=True,
                random_state=self.random_state
            )
            forecasters.append(('arima', arima_model))
        
        # ETS component
        ets_model = ExponentialSmoothing(
            trend='add',
            seasonal='add' if self.seasonal_period else None,
            sp=self.seasonal_period or 1
        )
        forecasters.append(('ets', ets_model))
        
        # Theta component
        theta_model = ThetaForecaster(sp=self.seasonal_period or 1)
        forecasters.append(('theta', theta_model))
        
        # Naive baseline
        naive_model = NaiveForecaster(strategy='last', sp=self.seasonal_period or 1)
        forecasters.append(('naive', naive_model))
        
        # Create ensemble with equal weights
        ensemble = EnsembleForecaster(
            forecasters=forecasters,
            aggfunc='mean'  # Simple average ensemble
        )
        
        return ensemble
    
    def _preprocess_data(self, y: pd.Series) -> pd.Series:
        """
        Preprocess financial time series data.
        
        Parameters:
        -----------
        y : pd.Series
            Input time series
            
        Returns:
        --------
        pd.Series
            Preprocessed time series
        """
        y_processed = y.copy()
        
        # Handle missing values
        if y_processed.isnull().any():
            logger.warning("Missing values detected, forward filling")
            y_processed = y_processed.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure positive values for log transformation
        if self.use_log_transform:
            if (y_processed <= 0).any():
                logger.warning("Non-positive values detected, adding constant")
                y_processed = y_processed + abs(y_processed.min()) + 1e-8
            
            y_processed = np.log(y_processed)
            logger.info("Applied log transformation")
        
        # Store original index and frequency
        self.original_index = y_processed.index
        self.original_freq = y_processed.index.freq
        
        return y_processed
    
    def _add_exogenous_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Create exogenous features for models that support them.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataframe with OHLCV data
            
        Returns:
        --------
        pd.DataFrame or None
            Exogenous features dataframe
        """
        if self.model_type not in ['prophet', 'arima']:
            return None
        
        exog_features = pd.DataFrame(index=df.index)
        
        # Volume features
        if 'Volume' in df.columns:
            exog_features['volume_ma'] = df['Volume'].rolling(window=20).mean()
            exog_features['volume_ratio'] = df['Volume'] / exog_features['volume_ma']
        
        # Volatility features
        if 'Close' in df.columns:
            returns = df['Close'].pct_change()
            exog_features['volatility'] = returns.rolling(window=20).std()
            exog_features['volatility_ma'] = exog_features['volatility'].rolling(window=5).mean()
        
        # Technical indicators
        if 'High' in df.columns and 'Low' in df.columns:
            # RSI
            delta = df['Close'].diff()
            gains = delta.where(delta > 0, 0).rolling(window=14).mean()
            losses = (-delta).where(delta < 0, 0).rolling(window=14).mean()
            rs = gains / losses
            exog_features['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands position
            ma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            exog_features['bb_position'] = (df['Close'] - ma_20) / (2 * std_20)
        
        # Calendar effects
        exog_features['day_of_week'] = df.index.dayofweek
        exog_features['month'] = df.index.month
        exog_features['is_month_end'] = df.index.is_month_end.astype(int)
        exog_features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Remove missing values
        exog_features = exog_features.fillna(method='ffill').fillna(0)
        
        return exog_features
    
    def fit(self, df: pd.DataFrame, target_col: str = 'Close') -> 'StatisticalForecastingModel':
        """
        Fit the statistical forecasting model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with OHLCV columns
        target_col : str, default='Close'
            Target column for forecasting
            
        Returns:
        --------
        self
            Fitted model
        """
        logger.info(f"Training {self.model_type} model on {len(df)} samples")
        
        try:
            # Extract target series
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            y = df[target_col].copy()
            
            # Preprocess data
            y_processed = self._preprocess_data(y)
            
            # Get exogenous features
            X = self._add_exogenous_features(df)
            
            # Fit model
            if X is not None and self.model_type in ['prophet']:
                # Models that support exogenous variables
                self.model.fit(y_processed, X=X)
            else:
                # Univariate models
                self.model.fit(y_processed)
            
            self.is_fitted = True
            
            # Calculate in-sample metrics
            if len(y_processed) > self.forecast_horizon:
                train_predictions = self.model.predict(fh=range(1, len(y_processed) + 1))
                self.training_metrics = self._calculate_metrics(y_processed, train_predictions)
            
            logger.info(f"{self.model_type} model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training {self.model_type} model: {e}")
            raise
        
        return self
    
    def predict(
        self, 
        fh: Optional[Union[int, List[int], np.ndarray]] = None,
        return_pred_int: bool = True,
        alpha: Optional[float] = None
    ) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
        """
        Make forecasts using the fitted model.
        
        Parameters:
        -----------
        fh : int, list, or array, optional
            Forecast horizon. If None, uses self.forecast_horizon
        return_pred_int : bool, default=True
            Whether to return prediction intervals
        alpha : float, optional
            Significance level for prediction intervals
            
        Returns:
        --------
        pd.Series or tuple
            Point forecasts, optionally with prediction intervals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if fh is None:
            fh = range(1, self.forecast_horizon + 1)
        elif isinstance(fh, int):
            fh = range(1, fh + 1)
        
        if alpha is None:
            alpha = 1 - self.confidence_level
        
        try:
            # Make predictions
            if return_pred_int:
                y_pred, pred_int = self.model.predict(fh=fh, return_pred_int=return_pred_int, alpha=alpha)
            else:
                y_pred = self.model.predict(fh=fh, return_pred_int=return_pred_int)
            
            # Transform back from log space if needed
            if self.use_log_transform:
                y_pred = np.exp(y_pred)
                if return_pred_int:
                    pred_int = np.exp(pred_int)
            
            if return_pred_int:
                return y_pred, pred_int
            else:
                return y_pred
                
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_with_updates(
        self, 
        df_new: pd.DataFrame, 
        target_col: str = 'Close',
        update_params: bool = False
    ) -> pd.Series:
        """
        Make predictions with updated data (online forecasting).
        
        Parameters:
        -----------
        df_new : pd.DataFrame
            New data to update the model with
        target_col : str, default='Close'
            Target column
        update_params : bool, default=False
            Whether to update model parameters
            
        Returns:
        --------
        pd.Series
            Updated forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            y_new = df_new[target_col].copy()
            y_new_processed = self._preprocess_data(y_new)
            
            # Update model with new data
            if update_params:
                self.model.update(y_new_processed, update_params=True)
            else:
                self.model.update(y_new_processed, update_params=False)
            
            # Make new predictions
            predictions = self.predict(return_pred_int=False)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error updating predictions: {e}")
            raise
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate forecasting metrics."""
        try:
            # Align series
            common_index = y_true.index.intersection(y_pred.index)
            y_true_aligned = y_true.loc[common_index]
            y_pred_aligned = y_pred.loc[common_index]
            
            if len(y_true_aligned) == 0:
                return {}
            
            metrics = {
                'mae': mean_absolute_error(y_true_aligned, y_pred_aligned),
                'mse': mean_squared_error(y_true_aligned, y_pred_aligned),
                'rmse': np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned)),
                'mape': mean_absolute_percentage_error(y_true_aligned, y_pred_aligned, symmetric=False) * 100
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return {}
    
    def evaluate(self, df_test: pd.DataFrame, target_col: str = 'Close') -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        df_test : pd.DataFrame
            Test data
        target_col : str, default='Close'
            Target column
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if not self.is_fitted:
            return {}
        
        try:
            y_test = df_test[target_col].copy()
            
            # Make predictions for the test period
            fh = range(1, len(y_test) + 1)
            y_pred = self.predict(fh=fh, return_pred_int=False)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Add directional accuracy
            if len(y_test) > 1 and len(y_pred) > 1:
                actual_directions = np.diff(y_test.values) > 0
                predicted_directions = np.diff(y_pred.values) > 0
                direction_accuracy = np.mean(actual_directions == predicted_directions) * 100
                metrics['direction_accuracy'] = direction_accuracy
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def cross_validate(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'Close',
        cv_splits: int = 5,
        test_size: int = 30
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset
        target_col : str, default='Close'
            Target column
        cv_splits : int, default=5
            Number of CV splits
        test_size : int, default=30
            Size of test set for each split
            
        Returns:
        --------
        dict
            Cross-validation metrics
        """
        try:
            y = df[target_col].copy()
            
            # Create time series CV splitter
            cv = SlidingWindowSplitter(
                window_length=len(y) // (cv_splits + 1),
                fh=range(1, test_size + 1)
            )
            
            # Perform cross-validation
            metrics_list = []
            
            for train_idx, test_idx in cv.split(y):
                # Split data
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                df_train = df.iloc[train_idx]
                df_test = df.iloc[test_idx]
                
                # Create and fit model
                temp_model = StatisticalForecastingModel(
                    model_type=self.model_type,
                    forecast_horizon=test_size,
                    seasonal_period=self.seasonal_period,
                    use_log_transform=self.use_log_transform,
                    random_state=self.random_state
                )
                
                temp_model.fit(df_train, target_col)
                
                # Evaluate
                metrics = temp_model.evaluate(df_test, target_col)
                if metrics:
                    metrics_list.append(metrics)
            
            # Aggregate results
            cv_results = {}
            if metrics_list:
                for metric_name in metrics_list[0].keys():
                    cv_results[f'{metric_name}_mean'] = np.mean([m[metric_name] for m in metrics_list])
                    cv_results[f'{metric_name}_std'] = np.std([m[metric_name] for m in metrics_list])
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of the fitted model."""
        if not self.is_fitted:
            return {}
        
        summary = {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'forecast_horizon': self.forecast_horizon,
            'use_log_transform': self.use_log_transform,
            'seasonal_period': self.seasonal_period,
            'training_metrics': self.training_metrics
        }
        
        # Add model-specific information
        if hasattr(self.model, 'summary') and callable(getattr(self.model, 'summary')):
            try:
                summary['model_summary'] = str(self.model.summary())
            except:
                pass
        
        return summary
    
    def save_model(self, filepath: Union[str, Path]):
        """Save the fitted model."""
        model_data = {
            'model': self.model,
            'config': {
                'model_type': self.model_type,
                'forecast_horizon': self.forecast_horizon,
                'confidence_level': self.confidence_level,
                'seasonal_period': self.seasonal_period,
                'use_log_transform': self.use_log_transform,
                'use_differencing': self.use_differencing,
                'handle_volatility_clustering': self.handle_volatility_clustering,
                'random_state': self.random_state
            },
            'training_metrics': self.training_metrics,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]):
        """Load a saved model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.training_metrics = model_data.get('training_metrics', {})
        self.is_fitted = model_data.get('is_fitted', False)
        
        # Update config
        config = model_data.get('config', {})
        for key, value in config.items():
            setattr(self, key, value)
        
        logger.info(f"Model loaded from {filepath}")


class AutoMLForecaster:
    """
    Automated Machine Learning for time series forecasting.
    
    This class automatically selects the best statistical forecasting model
    from a suite of candidates using cross-validation.
    """
    
    def __init__(
        self,
        models_to_try: Optional[List[str]] = None,
        forecast_horizon: int = 30,
        cv_splits: int = 3,
        metric: str = 'mape',
        random_state: int = 42
    ):
        """
        Initialize AutoML forecaster.
        
        Parameters:
        -----------
        models_to_try : list, optional
            List of model types to try
        forecast_horizon : int, default=30
            Forecast horizon
        cv_splits : int, default=3
            Number of CV splits for model selection
        metric : str, default='mape'
            Metric to optimize
        random_state : int, default=42
            Random state
        """
        if models_to_try is None:
            models_to_try = ['auto_arima', 'ets', 'theta', 'ensemble']
            if PROPHET_AVAILABLE:
                models_to_try.append('prophet')
        
        self.models_to_try = models_to_try
        self.forecast_horizon = forecast_horizon
        self.cv_splits = cv_splits
        self.metric = metric
        self.random_state = random_state
        
        self.best_model = None
        self.cv_results = {}
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame, target_col: str = 'Close') -> 'AutoMLForecaster':
        """
        Fit the AutoML forecaster by selecting the best model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        target_col : str, default='Close'
            Target column
            
        Returns:
        --------
        self
            Fitted AutoML forecaster
        """
        logger.info(f"AutoML: Testing {len(self.models_to_try)} model types")
        
        best_score = float('inf')
        best_model_type = None
        
        # Try each model type
        for model_type in self.models_to_try:
            try:
                logger.info(f"Testing {model_type}...")
                
                # Create model
                model = StatisticalForecastingModel(
                    model_type=model_type,
                    forecast_horizon=self.forecast_horizon,
                    random_state=self.random_state
                )
                
                # Cross-validate
                cv_results = model.cross_validate(
                    df, target_col, 
                    cv_splits=self.cv_splits,
                    test_size=min(self.forecast_horizon, len(df) // (self.cv_splits * 2))
                )
                
                if cv_results and f'{self.metric}_mean' in cv_results:
                    score = cv_results[f'{self.metric}_mean']
                    self.cv_results[model_type] = cv_results
                    
                    logger.info(f"{model_type} {self.metric}: {score:.4f}")
                    
                    if score < best_score:
                        best_score = score
                        best_model_type = model_type
                
            except Exception as e:
                logger.warning(f"Failed to test {model_type}: {e}")
                continue
        
        if best_model_type is None:
            raise ValueError("No models could be fitted successfully")
        
        # Fit the best model on full data
        logger.info(f"Best model: {best_model_type} ({self.metric}: {best_score:.4f})")
        
        self.best_model = StatisticalForecastingModel(
            model_type=best_model_type,
            forecast_horizon=self.forecast_horizon,
            random_state=self.random_state
        )
        
        self.best_model.fit(df, target_col)
        self.is_fitted = True
        
        return self
    
    def predict(self, **kwargs):
        """Make predictions using the best model."""
        if not self.is_fitted:
            raise ValueError("AutoML forecaster must be fitted first")
        
        return self.best_model.predict(**kwargs)
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard sorted by performance."""
        if not self.cv_results:
            return pd.DataFrame()
        
        leaderboard_data = []
        
        for model_type, results in self.cv_results.items():
            row = {'model': model_type}
            for metric_name, value in results.items():
                row[metric_name] = value
            leaderboard_data.append(row)
        
        leaderboard = pd.DataFrame(leaderboard_data)
        
        if f'{self.metric}_mean' in leaderboard.columns:
            leaderboard = leaderboard.sort_values(f'{self.metric}_mean')
        
        return leaderboard


def create_statistical_models(
    forecast_horizon: int = 30,
    seasonal_period: Optional[int] = None,
    include_automl: bool = True
) -> Dict[str, Any]:
    """
    Create a suite of statistical forecasting models.
    
    Parameters:
    -----------
    forecast_horizon : int, default=30
        Forecast horizon
    seasonal_period : int, optional
        Seasonal period
    include_automl : bool, default=True
        Whether to include AutoML forecaster
        
    Returns:
    --------
    dict
        Dictionary of statistical models
    """
    models = {}
    
    # Individual models
    model_types = ['auto_arima', 'ets', 'theta', 'ensemble']
    if PROPHET_AVAILABLE:
        model_types.append('prophet')
    
    for model_type in model_types:
        models[model_type] = StatisticalForecastingModel(
            model_type=model_type,
            forecast_horizon=forecast_horizon,
            seasonal_period=seasonal_period
        )
    
    # AutoML forecaster
    if include_automl:
        models['automl'] = AutoMLForecaster(
            models_to_try=model_types,
            forecast_horizon=forecast_horizon
        )
    
    return models


# Example usage
if __name__ == "__main__":
    # Example usage with sample data
    import yfinance as yf
    
    # Download sample data
    data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
    
    # Create statistical models
    models = create_statistical_models(forecast_horizon=30)
    
    # Split data
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Train model
            model.fit(train_data)
            
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict()
                print(f"Forecast shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
            
            # Evaluate if we have test data
            if hasattr(model, 'evaluate') and len(test_data) > 0:
                metrics = model.evaluate(test_data)
                print(f"Test metrics: {metrics}")
            
        except Exception as e:
            print(f"Error with {name}: {e}")