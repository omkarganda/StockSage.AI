#!/usr/bin/env python3
"""
Model Diagnostic Script for StockSage.AI

This script addresses the specific issues identified in the model evaluation:
1. Verify target-prediction alignment (plot y_true vs. y_pred for LSTM)
2. Implement rolling window evaluation 
3. Create diagnostic visualizations
4. Check for lag-based alignment issues

Usage:
    python scripts/diagnose_models.py --symbol AAPL --model dl_lstm_attention
    python scripts/diagnose_models.py --rolling-eval --n-folds 5 --fold-size 30
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
import logging

# Import our models for loading
from src.models.baseline import LinearRegressionBaseline, RandomForestBaseline, EnsembleBaseline
from src.models.neuralforecast_model import TimesFMFinancialModel, TTMFinancialModel, NeuralForecastBaseline
from src.models.sktime_model import StatisticalForecastingModel, AutoMLForecaster
from src.models.lstm_attention_model import LSTMAttentionModel
from src.models.transformer_model import TransformerModel

# Data imports
import yfinance as yf
from src.features.indicators import add_all_technical_indicators

# Plotting imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelDiagnostic:
    """
    Comprehensive model diagnostic system for StockSage.AI
    
    This class provides tools to diagnose model performance issues including:
    - Target-prediction alignment verification
    - Rolling window evaluation
    - Lag analysis and temporal consistency checks
    - Visualization of prediction patterns
    """
    
    def __init__(
        self,
        results_dir: str = "results/model_training",
        output_dir: str = "reports/diagnostics",
    ):
        """
        Initialize the model diagnostic system.
        
        Parameters:
        -----------
        results_dir : str
            Directory containing training results and saved models
        output_dir : str
            Directory to save diagnostic reports
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for loaded data
        self.loaded_models = {}
        self.diagnostic_results = {}
        
        logger.info(f"ModelDiagnostic initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_model(self, symbol: str, model_name: str) -> Optional[Any]:
        """
        Load a specific model for diagnostics.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        model_name : str
            Model name to load
            
        Returns:
        --------
        model instance or None
        """
        model_file = self.results_dir / f"{symbol}_{model_name}.joblib"
        
        if not model_file.exists():
            logger.error(f"Model file not found: {model_file}")
            return None
        
        try:
            # Load the model based on its type
            if model_name.startswith('baseline_'):
                base_name = model_name.replace('baseline_', '')
                if base_name == 'linear_regression':
                    model = LinearRegressionBaseline()
                elif base_name == 'random_forest':
                    model = RandomForestBaseline()
                elif base_name == 'ensemble':
                    model = EnsembleBaseline()
                else:
                    logger.warning(f"Unknown baseline model: {base_name}")
                    return None
            
            elif model_name.startswith('neural_'):
                base_name = model_name.replace('neural_', '')
                if base_name == 'timesfm':
                    model = TimesFMFinancialModel()
                elif base_name == 'ttm':
                    model = TTMFinancialModel()
                elif base_name == 'neuralforecast':
                    model = NeuralForecastBaseline()
                else:
                    logger.warning(f"Unknown neural model: {base_name}")
                    return None
            
            elif model_name.startswith('statistical_'):
                base_name = model_name.replace('statistical_', '')
                if base_name == 'automl':
                    model = AutoMLForecaster()
                else:
                    model = StatisticalForecastingModel()
            
            elif model_name.startswith('dl_lstm_attention'):
                model = LSTMAttentionModel()
            
            elif model_name.startswith('dl_transformer'):
                model = TransformerModel()
            
            else:
                logger.warning(f"Unknown model category: {model_name}")
                return None
            
            # Load the model
            model.load_model(model_file)
            logger.info(f"Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            return None
    
    def download_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download fresh data for diagnostics."""
        logger.info(f"Downloading data for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Clean columns
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Add technical indicators
            data = add_all_technical_indicators(data)
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Downloaded {len(data)} rows for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            raise
    
    def verify_target_prediction_alignment(
        self, 
        symbol: str, 
        model_name: str, 
        data: pd.DataFrame,
        horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Verify target-prediction alignment for a specific model.
        
        This function addresses Section E.1 of the analysis: 
        "Verify target-prediction alignment (plot y_true vs. y_pred for LSTM)"
        """
        logger.info(f"Verifying target-prediction alignment for {model_name}")
        
        model = self.load_model(symbol, model_name)
        if model is None:
            return {'error': 'Failed to load model'}
        
        try:
            # Split data for prediction context and target
            context_end = len(data) - horizon
            context_data = data.iloc[:context_end]
            
            # Make prediction
            predictions = model.predict(context_data)
            
            # Get actual values
            actual_prices = data['Close'].values
            if model_name.startswith('dl_'):
                # Deep learning models predict returns
                last_price = actual_prices[context_end - 1]
                actual_returns = (actual_prices[context_end:context_end + horizon] - last_price) / last_price
                pred_values = np.array(predictions).flatten()
            else:
                # Other models might predict prices directly
                actual_returns = actual_prices[context_end:context_end + horizon] 
                pred_values = np.array(predictions).flatten()
            
            # Ensure same length
            min_len = min(len(actual_returns), len(pred_values))
            actual_returns = actual_returns[:min_len]
            pred_values = pred_values[:min_len]
            
            # Calculate alignment metrics
            correlation = np.corrcoef(actual_returns, pred_values)[0, 1] if len(actual_returns) > 1 else 0
            if np.isnan(correlation):
                correlation = 0.0
            
            # Check lag correlation
            lag_correlations = {}
            for lag in range(-5, 6):
                if lag == 0:
                    lag_corr = correlation
                elif lag > 0 and len(actual_returns) > lag:
                    lag_corr = np.corrcoef(actual_returns[:-lag], pred_values[lag:])[0, 1]
                elif lag < 0 and len(pred_values) > abs(lag):
                    lag_corr = np.corrcoef(actual_returns[abs(lag):], pred_values[:lag])[0, 1]
                else:
                    lag_corr = 0
                
                if not np.isnan(lag_corr):
                    lag_correlations[lag] = lag_corr
            
            # Find best lag
            best_lag = max(lag_correlations.items(), key=lambda x: abs(x[1]))[0]
            best_correlation = lag_correlations[best_lag]
            
            # Create diagnostic plot
            if HAS_PLOTLY:
                self._create_alignment_plot(
                    actual_returns, pred_values, symbol, model_name, 
                    correlation, best_lag, best_correlation
                )
            
            result = {
                'correlation': float(correlation),
                'best_lag': best_lag,
                'best_correlation': float(best_correlation),
                'lag_correlations': {str(k): float(v) for k, v in lag_correlations.items()},
                'actual_std': float(np.std(actual_returns)),
                'pred_std': float(np.std(pred_values)),
                'actual_mean': float(np.mean(actual_returns)),
                'pred_mean': float(np.mean(pred_values)),
                'alignment_issue': abs(best_lag) > 0 and abs(best_correlation) > abs(correlation),
                'num_samples': min_len
            }
            
            logger.info(f"Alignment check for {model_name}:")
            logger.info(f"  Correlation: {correlation:.3f}")
            logger.info(f"  Best lag: {best_lag} (corr: {best_correlation:.3f})")
            logger.info(f"  Alignment issue: {result['alignment_issue']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in alignment verification: {e}")
            return {'error': str(e)}
    
    def _create_alignment_plot(
        self, 
        actual: np.ndarray, 
        predicted: np.ndarray, 
        symbol: str, 
        model_name: str,
        correlation: float,
        best_lag: int,
        best_correlation: float
    ):
        """Create alignment diagnostic plot."""
        if not HAS_PLOTLY:
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Actual vs Predicted (Time Series)',
                'Scatter Plot: Predicted vs Actual',
                'Correlation at Different Lags',
                'Residuals Over Time'
            )
        )
        
        # Time series plot
        time_steps = list(range(len(actual)))
        fig.add_trace(
            go.Scatter(x=time_steps, y=actual, name='Actual', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=predicted, name='Predicted', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=predicted, y=actual, mode='markers', name='Predicted vs Actual'),
            row=1, col=2
        )
        # Add diagonal line
        min_val, max_val = min(min(actual), min(predicted)), max(max(actual), max(predicted))
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction', line=dict(color='gray', dash='dot')),
            row=1, col=2
        )
        
        # Lag correlation plot
        lags = list(range(-5, 6))
        lag_corrs = []
        for lag in lags:
            if lag == 0:
                corr = correlation
            elif lag > 0 and len(actual) > lag:
                corr = np.corrcoef(actual[:-lag], predicted[lag:])[0, 1]
            elif lag < 0 and len(predicted) > abs(lag):
                corr = np.corrcoef(actual[abs(lag):], predicted[:lag])[0, 1]
            else:
                corr = 0
            lag_corrs.append(corr if not np.isnan(corr) else 0)
        
        fig.add_trace(
            go.Bar(x=lags, y=lag_corrs, name='Lag Correlation'),
            row=2, col=1
        )
        
        # Residuals
        residuals = actual - predicted
        fig.add_trace(
            go.Scatter(x=time_steps, y=residuals, mode='markers', name='Residuals'),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
        
        fig.update_layout(
            title=f"Alignment Diagnostics: {symbol} - {model_name}<br>"
                  f"Correlation: {correlation:.3f}, Best Lag: {best_lag} (corr: {best_correlation:.3f})",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plot_file = self.output_dir / f"{symbol}_{model_name}_alignment_diagnostic.html"
        fig.write_html(plot_file)
        logger.info(f"Saved alignment plot: {plot_file}")
    
    def rolling_window_evaluation(
        self,
        symbol: str,
        model_name: str,
        data: pd.DataFrame,
        n_folds: int = 5,
        fold_size: int = 30
    ) -> Dict[str, Any]:
        """
        Perform rolling window evaluation.
        
        This addresses Section E.4 of the analysis:
        "Rerun evaluation on a rolling window (e.g., 5 folds × 30 days)"
        """
        logger.info(f"Running rolling window evaluation for {model_name} (n_folds={n_folds}, fold_size={fold_size})")
        
        model = self.load_model(symbol, model_name)
        if model is None:
            return {'error': 'Failed to load model'}
        
        try:
            total_needed = n_folds * fold_size
            if len(data) < total_needed + 60:  # Need extra data for context
                logger.warning(f"Insufficient data for rolling evaluation. Need {total_needed + 60}, have {len(data)}")
                n_folds = max(1, (len(data) - 60) // fold_size)
                logger.info(f"Adjusted n_folds to {n_folds}")
            
            fold_results = []
            
            for fold in range(n_folds):
                start_idx = len(data) - (n_folds - fold) * fold_size
                end_idx = start_idx + fold_size
                
                # Context for prediction (use data before the test window)
                context_data = data.iloc[:start_idx]
                
                # Test data
                test_data = data.iloc[start_idx:end_idx]
                
                if len(context_data) < 60:  # Minimum context needed
                    continue
                
                try:
                    # Make predictions
                    predictions = model.predict(context_data)
                    pred_values = np.array(predictions).flatten()[:len(test_data)]
                    
                    # Get actual values (returns for deep learning models)
                    if model_name.startswith('dl_'):
                        last_price = context_data['Close'].iloc[-1]
                        actual_returns = (test_data['Close'].values - last_price) / last_price
                    else:
                        actual_returns = test_data['Close'].pct_change().fillna(0).values
                    
                    # Align lengths
                    min_len = min(len(actual_returns), len(pred_values))
                    actual_returns = actual_returns[:min_len]
                    pred_values = pred_values[:min_len]
                    
                    if min_len > 0:
                        # Calculate metrics
                        mse = np.mean((actual_returns - pred_values) ** 2)
                        mae = np.mean(np.abs(actual_returns - pred_values))
                        rmse = np.sqrt(mse)
                        
                        # SMAPE
                        denominator = (np.abs(actual_returns) + np.abs(pred_values)) / 2.0
                        denominator = np.where(denominator < 1e-8, 1e-8, denominator)
                        smape = np.mean(np.abs(actual_returns - pred_values) / denominator) * 100
                        
                        # Direction accuracy
                        actual_directions = actual_returns > 0
                        pred_directions = pred_values > 0
                        direction_accuracy = np.mean(actual_directions == pred_directions) * 100
                        
                        # Correlation
                        correlation = np.corrcoef(actual_returns, pred_values)[0, 1] if np.std(actual_returns) > 0 and np.std(pred_values) > 0 else 0
                        if np.isnan(correlation):
                            correlation = 0.0
                        
                        fold_results.append({
                            'fold': fold,
                            'start_date': str(test_data.index[0].date()),
                            'end_date': str(test_data.index[-1].date()),
                            'mse': float(mse),
                            'mae': float(mae),
                            'rmse': float(rmse),
                            'smape': float(smape),
                            'direction_accuracy': float(direction_accuracy),
                            'correlation': float(correlation),
                            'num_samples': min_len
                        })
                        
                except Exception as e:
                    logger.warning(f"Error in fold {fold}: {e}")
                    continue
            
            if not fold_results:
                return {'error': 'No successful folds'}
            
            # Aggregate results
            metrics = ['mse', 'mae', 'rmse', 'smape', 'direction_accuracy', 'correlation']
            aggregated = {}
            
            for metric in metrics:
                values = [result[metric] for result in fold_results]
                aggregated[f'{metric}_mean'] = float(np.mean(values))
                aggregated[f'{metric}_std'] = float(np.std(values))
                aggregated[f'{metric}_min'] = float(np.min(values))
                aggregated[f'{metric}_max'] = float(np.max(values))
            
            result = {
                'fold_results': fold_results,
                'aggregated_metrics': aggregated,
                'n_successful_folds': len(fold_results),
                'total_samples': sum(result['num_samples'] for result in fold_results)
            }
            
            logger.info(f"Rolling evaluation completed: {len(fold_results)} successful folds")
            logger.info(f"Average SMAPE: {aggregated['smape_mean']:.2f}% ± {aggregated['smape_std']:.2f}%")
            logger.info(f"Average Direction Accuracy: {aggregated['direction_accuracy_mean']:.1f}% ± {aggregated['direction_accuracy_std']:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in rolling window evaluation: {e}")
            return {'error': str(e)}
    
    def generate_diagnostic_report(
        self, 
        symbol: str, 
        model_names: List[str],
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31"
    ) -> str:
        """Generate a comprehensive diagnostic report."""
        logger.info(f"Generating diagnostic report for {symbol}")
        
        # Download data
        data = self.download_data(symbol, start_date, end_date)
        
        report_data = {
            'symbol': symbol,
            'data_info': {
                'total_samples': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}"
            },
            'alignment_results': {},
            'rolling_results': {}
        }
        
        for model_name in model_names:
            logger.info(f"Diagnosing {model_name}...")
            
            # Target-prediction alignment
            alignment_result = self.verify_target_prediction_alignment(symbol, model_name, data)
            report_data['alignment_results'][model_name] = alignment_result
            
            # Rolling window evaluation
            rolling_result = self.rolling_window_evaluation(symbol, model_name, data)
            report_data['rolling_results'][model_name] = rolling_result
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"diagnostic_report_{symbol}_{timestamp}.json"
        
        import json
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Diagnostic report saved: {report_file}")
        return str(report_file)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Diagnose StockSage.AI Models')
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Stock symbol to diagnose'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Specific model to diagnose (e.g., dl_lstm_attention)'
    )
    
    parser.add_argument(
        '--rolling-eval',
        action='store_true',
        help='Perform rolling window evaluation'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds for rolling evaluation'
    )
    
    parser.add_argument(
        '--fold-size',
        type=int,
        default=30,
        help='Size of each fold in days'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Start date for data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-12-31',
        help='End date for data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/model_training',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/diagnostics',
        help='Output directory for diagnostic reports'
    )
    
    return parser.parse_args()


def main():
    """Main diagnostic function."""
    print("StockSage.AI Model Diagnostic Script")
    print("===================================\n")
    
    args = parse_arguments()
    
    # Create diagnostic system
    diagnostic = ModelDiagnostic(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    try:
        # Determine models to diagnose
        if args.model:
            model_names = [args.model]
        else:
            # Default models that commonly have issues
            model_names = ['dl_lstm_attention', 'dl_transformer', 'statistical_exp_smoothing']
        
        # Generate comprehensive diagnostic report
        report_file = diagnostic.generate_diagnostic_report(
            symbol=args.symbol,
            model_names=model_names,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        print(f"\nDiagnostic completed successfully!")
        print(f"Report saved: {report_file}")
        print(f"Check diagnostic plots in: {diagnostic.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())