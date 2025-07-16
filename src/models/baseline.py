"""
Baseline Models for StockSage.AI

This module provides simple, interpretable baseline models using traditional machine learning
approaches. These models serve as benchmarks for more complex time series and deep learning models.

Key Features:
- Linear Regression with feature engineering
- Random Forest with hyperparameter optimization
- Ensemble methods combining multiple approaches
- Built-in cross-validation and feature importance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import joblib

# Scikit-learn imports
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    VotingRegressor, VotingClassifier
)
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV,
    cross_val_score
)
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PolynomialFeatures
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, RFECV
)

# Local imports
from ..utils.logging import get_logger
from ..features.indicators import add_all_technical_indicators

# Configure logging
logger = get_logger(__name__)


class BaselineModel:
    """
    Base class for traditional ML baseline models.
    
    This class provides common functionality for preprocessing, training,
    and evaluation of baseline models.
    """
    
    def __init__(
        self,
        target_col: str = 'target',
        prediction_horizon: int = 1,
        use_technical_indicators: bool = True,
        use_feature_selection: bool = True,
        scaling_method: str = 'robust',
        random_state: int = 42
    ):
        """
        Initialize the baseline model.
        
        Parameters:
        -----------
        target_col : str, default='target'
            Name of the target column
        prediction_horizon : int, default=1
            Number of days ahead to predict
        use_technical_indicators : bool, default=True
            Whether to include technical indicators as features
        use_feature_selection : bool, default=True
            Whether to perform automatic feature selection
        scaling_method : str, default='robust'
            Scaling method ('standard', 'robust', 'minmax', 'none')
        random_state : int, default=42
            Random state for reproducibility
        """
        self.target_col = target_col
        self.prediction_horizon = prediction_horizon
        self.use_technical_indicators = use_technical_indicators
        self.use_feature_selection = use_feature_selection
        self.scaling_method = scaling_method
        self.random_state = random_state
        
        # Model components
        self.scaler = None
        self.feature_selector = None
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
        # Training metrics
        self.training_metrics = {}
        self.validation_metrics = {}
        
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training/prediction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with market data
            
        Returns:
        --------
        pd.DataFrame
            Processed dataframe with features
        """
        df = df.copy()
        
        # Add technical indicators if requested
        if self.use_technical_indicators:
            logger.info("Adding technical indicators")
            df = add_all_technical_indicators(df)
        
        # Create target variable
        if 'Close' in df.columns:
            # Price prediction target (future return)
            df[self.target_col] = df['Close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Create lag features
        df = self._create_lag_features(df)
        
        # Create time-based features
        df = self._create_time_features(df)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for key variables."""
        df = df.copy()
        
        # Key columns to create lags for
        key_cols = ['Close', 'Volume', 'High', 'Low', 'Open']
        lag_periods = [1, 2, 3, 5, 10, 20]
        
        for col in key_cols:
            if col in df.columns:
                for lag in lag_periods:
                    # Price lags
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
                    # Return lags
                    if col in ['Close', 'High', 'Low', 'Open']:
                        df[f'{col}_return_lag_{lag}'] = df[col].pct_change().shift(lag)
        
        # Rolling statistics
        if 'Close' in df.columns:
            for window in [5, 10, 20]:
                df[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window).mean()
                df[f'Close_rolling_std_{window}'] = df['Close'].rolling(window).std()
                df[f'Close_rolling_max_{window}'] = df['Close'].rolling(window).max()
                df[f'Close_rolling_min_{window}'] = df['Close'].rolling(window).min()
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Time features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform feature selection.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
            
        Returns:
        --------
        tuple
            Selected features and feature names
        """
        if not self.use_feature_selection:
            return X, list(X.columns)
        
        # Remove features with too many missing values
        missing_threshold = 0.3
        valid_features = X.columns[X.isnull().mean() < missing_threshold]
        X = X[valid_features]
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        X = X.drop(columns=constant_features)
        
        # Statistical feature selection
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        if len(X_valid) > 0:
            # Select top features based on statistical tests
            k_features = min(50, len(X.columns))
            selector = SelectKBest(score_func=f_regression, k=k_features)
            X_selected = selector.fit_transform(X_valid, y_valid)
            selected_features = X.columns[selector.get_support()].tolist()
            
            logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
            self.feature_selector = selector
            
            return X[selected_features], selected_features
        else:
            logger.warning("No valid samples for feature selection")
            return X, list(X.columns)
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/prediction.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        tuple
            Feature matrix and target series
        """
        # Prepare features
        df_processed = self._prepare_features(df)
        
        # Separate features and target
        if self.target_col in df_processed.columns:
            y = df_processed[self.target_col]
            X = df_processed.drop(columns=[self.target_col])
        else:
            logger.warning(f"Target column '{self.target_col}' not found")
            y = pd.Series(index=df_processed.index, dtype=float)
            X = df_processed
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        return X, y
    
    def fit(self, df: pd.DataFrame) -> 'BaselineModel':
        """
        Fit the baseline model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data
            
        Returns:
        --------
        self
            Fitted model
        """
        logger.info(f"Training baseline model with {len(df)} samples")
        
        # Prepare data
        X, y = self._prepare_data(df)
        
        # Remove samples with missing targets
        valid_mask = ~y.isnull()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid training samples found")
        
        # Feature selection
        X, selected_features = self._select_features(X, y)
        self.feature_names = selected_features
        
        # Handle missing values more robustly
        # First, fill with forward fill, then backward fill, then median
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(X.median())
        # Drop any remaining rows with NaN values
        nan_rows = X.isnull().any(axis=1)
        if nan_rows.any():
            logger.warning(f"Dropping {nan_rows.sum()} rows with remaining NaN values before model fitting.")
            X = X[~nan_rows]
            y = y.loc[X.index]
        
        # Scale features
        if self.scaling_method != 'none':
            self._fit_scaler(X)
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Train model
        self._fit_model(X, y)
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        self.training_metrics = self._calculate_metrics(y, y_pred)
        
        logger.info(f"Training completed. R² score: {self.training_metrics.get('r2', 'N/A'):.4f}")
        
        return self
    
    def _fit_scaler(self, X: pd.DataFrame):
        """Fit the feature scaler."""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        self.scaler.fit(X)
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit the actual model. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _fit_model")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data for prediction
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X, _ = self._prepare_data(df)
        
        # Select features
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    X[feature] = 0
            X = X[self.feature_names]
        
        # Handle missing values more robustly
        # First, fill with forward fill, then backward fill, then median
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(X.median() if len(X) > 0 else 0)
        # Drop any remaining rows with NaN values
        nan_rows = X.isnull().any(axis=1)
        if nan_rows.any():
            logger.warning(f"Dropping {nan_rows.sum()} rows with remaining NaN values before prediction.")
            X = X[~nan_rows]
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(y_true_valid) == 0:
            return {}
        
        metrics = {
            'mse': mean_squared_error(y_true_valid, y_pred_valid),
            'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
            'mae': mean_absolute_error(y_true_valid, y_pred_valid),
            'r2': r2_score(y_true_valid, y_pred_valid),
            'smape': np.mean(2 * np.abs(y_true_valid - y_pred_valid) / (np.abs(y_true_valid) + np.abs(y_pred_valid) + 1e-8)) * 100
        }
        
        return metrics
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Test data
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        X, y = self._prepare_data(df)
        valid_mask = ~y.isnull()
        y_true = y[valid_mask]
        
        if len(y_true) == 0:
            logger.warning("No valid samples for evaluation")
            return {}
        
        # Make predictions
        predictions = self.predict(df)
        y_pred = predictions[valid_mask]
        
        metrics = self._calculate_metrics(y_true, y_pred)
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available."""
        if self.model is None:
            return pd.DataFrame()
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names or range(len(importance)),
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: Union[str, Path]):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'config': {
                'target_col': self.target_col,
                'prediction_horizon': self.prediction_horizon,
                'use_technical_indicators': self.use_technical_indicators,
                'use_feature_selection': self.use_feature_selection,
                'scaling_method': self.scaling_method,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.feature_names = model_data['feature_names']
        self.training_metrics = model_data['training_metrics']
        
        # Update config
        config = model_data.get('config', {})
        for key, value in config.items():
            setattr(self, key, value)
        
        logger.info(f"Model loaded from {filepath}")


class LinearRegressionBaseline(BaselineModel):
    """
    Linear Regression baseline model with regularization options.
    
    This model uses linear regression as the base learner with options for
    Ridge, Lasso, or ElasticNet regularization.
    """
    
    def __init__(
        self,
        regularization: str = 'ridge',
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        polynomial_degree: int = 1,
        **kwargs
    ):
        """
        Initialize Linear Regression baseline.
        
        Parameters:
        -----------
        regularization : str, default='ridge'
            Type of regularization ('none', 'ridge', 'lasso', 'elastic')
        alpha : float, default=1.0
            Regularization strength
        l1_ratio : float, default=0.5
            ElasticNet mixing parameter (only used for elastic regularization)
        polynomial_degree : int, default=1
            Degree of polynomial features
        """
        super().__init__(**kwargs)
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.polynomial_degree = polynomial_degree
        self.poly_features = None
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit the linear regression model."""
        # Add polynomial features if requested
        if self.polynomial_degree > 1:
            self.poly_features = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False
            )
            X_poly = self.poly_features.fit_transform(X)
            X = pd.DataFrame(X_poly, index=X.index)
        
        # Choose regularization
        if self.regularization == 'none':
            self.model = LinearRegression()
        elif self.regularization == 'ridge':
            self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.regularization == 'lasso':
            self.model = Lasso(alpha=self.alpha, random_state=self.random_state)
        elif self.regularization == 'elastic':
            self.model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")
        
        # Fit model
        self.model.fit(X, y)
        
        logger.info(f"Fitted {self.regularization} regression with alpha={self.alpha}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions with polynomial features if used."""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare data
        X, _ = self._prepare_data(df)
        
        # Select features
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                for feature in missing_features:
                    X[feature] = 0
            X = X[self.feature_names]
        
        # Handle missing values more robustly
        # First, fill with forward fill, then backward fill, then median
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(X.median() if len(X) > 0 else 0)
        # Drop any remaining rows with NaN values
        nan_rows = X.isnull().any(axis=1)
        if nan_rows.any():
            logger.warning(f"Dropping {nan_rows.sum()} rows with remaining NaN values before prediction.")
            X = X[~nan_rows]
        
        # Scale features
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Add polynomial features if used
        if self.poly_features is not None:
            X = self.poly_features.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions


class RandomForestBaseline(BaselineModel):
    """
    Random Forest baseline model with hyperparameter optimization.
    
    This model uses Random Forest as the base learner with built-in
    hyperparameter tuning and feature importance analysis.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        use_grid_search: bool = True,
        cv_folds: int = 3,
        **kwargs
    ):
        """
        Initialize Random Forest baseline.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required in a leaf node
        max_features : str, default='sqrt'
            Number of features to consider for best split
        use_grid_search : bool, default=True
            Whether to use grid search for hyperparameter tuning
        cv_folds : int, default=3
            Number of cross-validation folds
        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.use_grid_search = use_grid_search
        self.cv_folds = cv_folds
        self.best_params = None
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit the Random Forest model with optional hyperparameter tuning."""
        
        if self.use_grid_search and len(X) > 100:  # Only use grid search for sufficient data
            logger.info("Performing hyperparameter tuning with grid search")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3]
            }
            
            # Create base model
            base_model = RandomForestRegressor(random_state=self.random_state)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            # Grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.model.fit(X, y)
            
            logger.info(f"Fitted Random Forest with {self.n_estimators} estimators")


class EnsembleBaseline(BaselineModel):
    """
    Ensemble baseline combining multiple models.
    
    This model combines Linear Regression and Random Forest using
    voting ensemble approach.
    """
    
    def __init__(
        self,
        voting: str = 'soft',
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize Ensemble baseline.
        
        Parameters:
        -----------
        voting : str, default='soft'
            Voting method ('hard' or 'soft')
        weights : list, optional
            Weights for each model in ensemble
        """
        super().__init__(**kwargs)
        self.voting = voting
        self.weights = weights
        self.individual_models = {}
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit the ensemble model."""
        
        # Create individual models
        linear_model = Ridge(alpha=1.0, random_state=self.random_state)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            random_state=self.random_state
        )
        
        # Create ensemble
        estimators = [
            ('linear', linear_model),
            ('random_forest', rf_model),
            ('gradient_boosting', gb_model)
        ]
        
        self.model = VotingRegressor(
            estimators=estimators,
            weights=self.weights,
            n_jobs=-1
        )
        
        # Fit ensemble
        self.model.fit(X, y)
        
        # Store individual models for analysis
        self.individual_models = {name: model for name, model in self.model.named_estimators_.items()}
        
        logger.info("Fitted ensemble model with Linear Regression, Random Forest, and Gradient Boosting")
    
    def get_individual_predictions(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models in the ensemble."""
        if not self.individual_models:
            return {}
        
        # Prepare data
        X, _ = self._prepare_data(df)
        
        # Select features and preprocess
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                for feature in missing_features:
                    X[feature] = 0
            X = X[self.feature_names]
        
        X = X.fillna(X.median() if len(X) > 0 else 0)
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.individual_models.items():
            predictions[name] = model.predict(X)
        
        return predictions


def create_baseline_models(
    target_col: str = 'target',
    prediction_horizon: int = 1,
    random_state: int = 42
) -> Dict[str, BaselineModel]:
    """
    Create a suite of baseline models for comparison.
    
    Parameters:
    -----------
    target_col : str, default='target'
        Target column name
    prediction_horizon : int, default=1
        Prediction horizon in days
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    dict
        Dictionary of baseline models
    """
    models = {
        'linear_regression': LinearRegressionBaseline(
            target_col=target_col,
            prediction_horizon=prediction_horizon,
            regularization='ridge',
            random_state=random_state
        ),
        'random_forest': RandomForestBaseline(
            target_col=target_col,
            prediction_horizon=prediction_horizon,
            use_grid_search=True,
            random_state=random_state
        ),
        'ensemble': EnsembleBaseline(
            target_col=target_col,
            prediction_horizon=prediction_horizon,
            random_state=random_state
        )
    }
    
    return models


# Example usage
if __name__ == "__main__":
    # This would typically be called from a training script
    import yfinance as yf
    
    # Download sample data
    data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
    
    # Create and train models
    models = create_baseline_models()
    
    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(data)
            metrics = model.training_metrics
            print(f"Training R² score: {metrics.get('r2', 'N/A'):.4f}")
            
            # Get feature importance
            importance_df = model.get_feature_importance()
            if not importance_df.empty:
                print(f"Top 5 features:")
                print(importance_df.head())
                
        except Exception as e:
            print(f"Error training {name}: {e}")