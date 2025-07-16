#!/usr/bin/env python3
"""
Model Evaluation and Comparison Script for StockSage.AI

This script loads trained models and provides comprehensive evaluation including:
- Performance comparison across models
- Prediction visualization
- Statistical significance testing
- Model ranking and recommendations

Usage:
    python scripts/evaluate_models.py --results-dir results/model_training
    python scripts/evaluate_models.py --symbol AAPL --models baseline_random_forest,statistical_auto_arima
    python scripts/evaluate_models.py --generate-report --output-dir reports/evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
import argparse
import json
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
import logging
from scipy import stats

# Import our models for loading
from src.models.baseline import LinearRegressionBaseline, RandomForestBaseline, EnsembleBaseline
from src.models.neuralforecast_model import TimesFMFinancialModel, TTMFinancialModel, NeuralForecastBaseline
from src.models.sktime_model import StatisticalForecastingModel, AutoMLForecaster
from src.models.lstm_attention_model import LSTMAttentionModel
from src.models.transformer_model import TransformerModel

# Data imports
import yfinance as yf
from src.features.indicators import add_all_technical_indicators

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('default')

if HAS_SEABORN:
    sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison system.
    
    This class loads trained models, evaluates their performance,
    and generates detailed comparison reports.
    """
    
    def __init__(
        self,
        results_dir: str = "results/model_training",
        output_dir: str = "reports/evaluation",
        generate_plots: bool = True
    ):
        """
        Initialize the model evaluator.
        
        Parameters:
        -----------
        results_dir : str, default="results/model_training"
            Directory containing training results and saved models
        output_dir : str, default="reports/evaluation"
            Directory to save evaluation reports
        generate_plots : bool, default=True
            Whether to generate visualization plots
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.generate_plots = generate_plots
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for loaded data
        self.training_results = {}
        self.loaded_models = {}
        self.evaluation_metrics = {}
        self.comparison_data = {}
        
        logger.info(f"ModelEvaluator initialized")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_training_results(self) -> Dict[str, Any]:
        """
        Load training results from JSON files.
        
        Returns:
        --------
        dict
            Loaded training results
        """
        logger.info("Loading training results...")
        
        # Find the most recent results file
        json_files = list(self.results_dir.glob("training_results_*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No training results found in {self.results_dir}")
        
        # Get the most recent file
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Loading results from: {latest_file}")
        
        with open(latest_file, 'r') as f:
            self.training_results = json.load(f)
        
        logger.info(f"Loaded results for {len(self.training_results)} symbols")
        return self.training_results
    
    def load_saved_models(self, symbol: str, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Load saved model files.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        model_names : list, optional
            Specific models to load. If None, loads all available
            
        Returns:
        --------
        dict
            Loaded model instances
        """
        logger.info(f"Loading saved models for {symbol}...")
        
        # Find model files for this symbol
        model_files = list(self.results_dir.glob(f"{symbol}_*.joblib"))
        
        if not model_files:
            logger.warning(f"No saved models found for {symbol}")
            return {}
        
        loaded_models = {}
        
        for model_file in model_files:
            # Extract model name from filename
            model_name = model_file.stem.replace(f"{symbol}_", "")
            
            # Skip if specific models requested and this isn't one of them
            if model_names and model_name not in model_names:
                continue
            
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
                        continue
                
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
                        continue
                
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
                    continue
                
                # Load the model
                model.load_model(model_file)
                loaded_models[model_name] = model
                
                logger.info(f"[SUCCESS] Loaded {model_name}")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to load {model_name}: {e}")
                continue
        
        self.loaded_models[symbol] = loaded_models
        logger.info(f"Loaded {len(loaded_models)} models for {symbol}")
        
        return loaded_models
    
    def download_fresh_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download fresh data for evaluation.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start_date : str
            Start date
        end_date : str
            End date
            
        Returns:
        --------
        pd.DataFrame
            Fresh stock data with technical indicators
        """
        logger.info(f"Downloading fresh data for {symbol}...")
        
        try:
            # Download data
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
    
    def evaluate_models_on_fresh_data(
        self, 
        symbol: str, 
        test_start_date: str, 
        test_end_date: str,
        forecast_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Evaluate all loaded models on fresh test data.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        test_start_date : str
            Start date for test data
        test_end_date : str
            End date for test data
        forecast_horizon : int, default=30
            Forecast horizon
            
        Returns:
        --------
        dict
            Evaluation results for all models
        """
        logger.info(f"Evaluating models for {symbol} on fresh data...")
        
        # Download fresh data
        fresh_data = self.download_fresh_data(symbol, test_start_date, test_end_date)
        
        if symbol not in self.loaded_models:
            logger.warning(f"No models loaded for {symbol}")
            return {}
        
        evaluation_results = {}
        
        for model_name, model in self.loaded_models[symbol].items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                # Make predictions based on model type
                if hasattr(model, 'predict'):
                    # Handle different model interfaces
                    if model_name.startswith('statistical_'):
                        # Statistical models need forecasting horizon (fh) parameter
                        # First check if model is fitted
                        if not hasattr(model, 'is_fitted') or not model.is_fitted:
                            logger.warning(f"[WARNING] {model_name} is not fitted, skipping prediction")
                            continue
                            
                        predictions = model.predict(fh=forecast_horizon)
                        # Convert to numpy array if it's a pandas Series
                        if hasattr(predictions, 'values'):
                            pred_values = predictions.values
                        else:
                            pred_values = np.array(predictions)
                    else:
                        # Other models (baseline, neural, deep learning) expect DataFrame
                        predictions = model.predict(fresh_data)
                        
                        # Handle different return types
                        if hasattr(predictions, '__len__') and len(predictions) > 0:
                            pred_values = predictions[-forecast_horizon:] if len(predictions) >= forecast_horizon else predictions
                        else:
                            pred_values = [predictions] * forecast_horizon
                    
                    # Convert pred_values to numpy array for consistent handling
                    pred_values = np.array(pred_values).flatten()
                    
                    # Compute actual future returns matching the model's target horizon
                    actual_returns = (
                        fresh_data['Close'].pct_change(forecast_horizon).shift(-forecast_horizon)
                    )
                    # Drop NaNs introduced by pct_change/shift
                    actual_returns = actual_returns.dropna()
                    actual = actual_returns.values[-forecast_horizon:] if len(actual_returns) >= forecast_horizon else actual_returns.values

                    if len(pred_values) > 0 and len(actual) > 0:
                        # Align lengths safely
                        min_len = min(len(actual), len(pred_values))
                        actual = actual[:min_len]
                        pred_values = pred_values[:min_len]

                        # Calculate metrics on returns
                        if min_len > 0:
                            mse = np.mean((actual - pred_values) ** 2)
                            mae = np.mean(np.abs(actual - pred_values))
                            rmse = np.sqrt(mse)
                            # Use SMAPE (Symmetric Mean Absolute Percentage Error) instead of MAPE
                            # SMAPE is more robust to values near zero
                            smape = np.mean(2 * np.abs(actual - pred_values) / (np.abs(actual) + np.abs(pred_values) + 1e-8)) * 100

                            # Direction accuracy (sign of returns)
                            actual_directions = actual > 0
                            pred_directions = pred_values > 0
                            direction_accuracy = np.mean(actual_directions == pred_directions) * 100 if len(actual_directions) > 0 else 0

                            # Correlation
                            correlation = np.corrcoef(actual, pred_values)[0, 1] if len(actual) > 1 and np.std(actual) > 0 and np.std(pred_values) > 0 else 0
                            # Handle NaN correlation
                            if np.isnan(correlation):
                                correlation = 0.0

                            evaluation_results[model_name] = {
                                'mse': float(mse),
                                'mae': float(mae),
                                'rmse': float(rmse),
                                'smape': float(smape),
                                'direction_accuracy': float(direction_accuracy),
                                'correlation': float(correlation),
                                'predictions': pred_values[:10].tolist() if hasattr(pred_values, 'tolist') else list(pred_values[:10]),
                                'actual': actual[:10].tolist() if hasattr(actual, 'tolist') else list(actual[:10])
                            }
                            
                            logger.info(f"[SUCCESS] {model_name}: SMAPE={smape:.2f}%, Dir_Acc={direction_accuracy:.1f}%")
                        else:
                            logger.warning(f"[WARNING] No valid predictions for {model_name}")
                    else:
                        logger.warning(f"[WARNING] Empty predictions or actuals for {model_name}")
                    
                else:
                    logger.warning(f"[WARNING] {model_name} doesn't have predict method")
                    
            except Exception as e:
                logger.error(f"[ERROR] Error evaluating {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        self.evaluation_metrics[symbol] = evaluation_results
        return evaluation_results
    
    def compare_models(self, symbol: str) -> pd.DataFrame:
        """
        Create a comparison dataframe of model performance.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        pd.DataFrame
            Model comparison dataframe
        """
        if symbol not in self.evaluation_metrics:
            logger.warning(f"No evaluation metrics found for {symbol}")
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, metrics in self.evaluation_metrics[symbol].items():
            if 'error' not in metrics:
                row = {'model': model_name, 'symbol': symbol, **metrics}
                # Remove predictions and actual arrays for cleaner comparison
                if 'predictions' in row:
                    del row['predictions']
                if 'actual' in row:
                    del row['actual']
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Sort by SMAPE (lower is better)
            comparison_df = comparison_df.sort_values('smape')
            
            # Add rank
            comparison_df['rank'] = range(1, len(comparison_df) + 1)
            
            # Reorder columns
            cols = ['rank', 'model', 'symbol', 'smape', 'rmse', 'mae', 'direction_accuracy', 'correlation']
            cols = [col for col in cols if col in comparison_df.columns]
            comparison_df = comparison_df[cols]
        
        return comparison_df
    
    def generate_performance_plots(self, symbol: str):
        """
        Generate performance visualization plots.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        """
        if not self.generate_plots or not HAS_PLOTLY:
            if not HAS_PLOTLY:
                logger.warning("Plotly not available, skipping plot generation")
            return
        
        logger.info(f"Generating performance plots for {symbol}...")
        
        # Get comparison data
        comparison_df = self.compare_models(symbol)
        
        if comparison_df.empty:
            logger.warning(f"No data to plot for {symbol}")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('SMAPE Comparison', 'RMSE Comparison', 
                          'Direction Accuracy', 'Model Correlation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
                # SMAPE plot
        fig.add_trace(
            go.Bar(x=comparison_df['model'], y=comparison_df['smape'],
                   name='SMAPE (%)', marker_color='lightblue'),
            row=1, col=1
        )
        
        # RMSE plot
        fig.add_trace(
            go.Bar(x=comparison_df['model'], y=comparison_df['rmse'], 
                   name='RMSE', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Direction Accuracy plot
        fig.add_trace(
            go.Bar(x=comparison_df['model'], y=comparison_df['direction_accuracy'], 
                   name='Direction Accuracy (%)', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Correlation plot
        fig.add_trace(
            go.Bar(x=comparison_df['model'], y=comparison_df['correlation'], 
                   name='Correlation', marker_color='orange'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Model Performance Comparison - {symbol}",
            height=800,
            showlegend=False
        )
        
        # Update x-axis labels to be vertical for readability
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(tickangle=45, row=i, col=j)
        
        # Save plot
        plot_file = self.output_dir / f"{symbol}_performance_comparison.html"
        fig.write_html(plot_file)
        logger.info(f"Saved performance plot: {plot_file}")
        
        # Create prediction vs actual plot if we have predictions
        self._create_prediction_plot(symbol)
    
    def _create_prediction_plot(self, symbol: str):
        """Create prediction vs actual comparison plot."""
        if symbol not in self.evaluation_metrics or not HAS_PLOTLY:
            return
        
        fig = go.Figure()
        
        for model_name, metrics in self.evaluation_metrics[symbol].items():
            if 'predictions' in metrics and 'actual' in metrics:
                predictions = metrics['predictions']
                actual = metrics['actual']
                
                # Add scatter plot
                fig.add_trace(go.Scatter(
                    x=list(range(len(predictions))),
                    y=predictions,
                    mode='lines+markers',
                    name=f"{model_name} (Predicted)",
                    line=dict(dash='dash')
                ))
        
        # Add actual values (assuming they're the same for all models)
        if self.evaluation_metrics[symbol]:
            first_model = list(self.evaluation_metrics[symbol].values())[0]
            if 'actual' in first_model:
                actual = first_model['actual']
                fig.add_trace(go.Scatter(
                    x=list(range(len(actual))),
                    y=actual,
                    mode='lines+markers',
                    name="Actual",
                    line=dict(color='black', width=3)
                ))
        
        fig.update_layout(
            title=f"Predictions vs Actual - {symbol}",
            xaxis_title="Time Steps",
            yaxis_title="Price",
            height=500
        )
        
        plot_file = self.output_dir / f"{symbol}_predictions_vs_actual.html"
        fig.write_html(plot_file)
        logger.info(f"Saved prediction plot: {plot_file}")
    
    def statistical_significance_test(self, symbol: str) -> Dict[str, Any]:
        """
        Perform statistical significance tests on model performance.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
            
        Returns:
        --------
        dict
            Statistical test results
        """
        logger.info(f"Performing statistical significance tests for {symbol}...")
        
        if symbol not in self.evaluation_metrics:
            return {}
        
        # Get errors for each model
        model_errors = {}
        
        for model_name, metrics in self.evaluation_metrics[symbol].items():
            if 'predictions' in metrics and 'actual' in metrics:
                predictions = np.array(metrics['predictions'])
                actual = np.array(metrics['actual'])
                errors = np.abs(actual - predictions)
                model_errors[model_name] = errors
        
        if len(model_errors) < 2:
            logger.warning("Need at least 2 models for significance testing")
            return {}
        
        # Perform pairwise t-tests
        test_results = {}
        model_names = list(model_errors.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(model_errors[model1], model_errors[model2])
                
                test_results[f"{model1}_vs_{model2}"] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'model1_mean_error': float(np.mean(model_errors[model1])),
                    'model2_mean_error': float(np.mean(model_errors[model2]))
                }
        
        return test_results
    
    def generate_comprehensive_report(self, symbols: List[str]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Parameters:
        -----------
        symbols : list
            List of symbols to include in report
            
        Returns:
        --------
        str
            Path to generated report
        """
        logger.info("Generating comprehensive evaluation report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"model_evaluation_report_{timestamp}.html"
        
        html_content = self._create_html_report(symbols)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report saved: {report_file}")
        return str(report_file)
    
    def _create_html_report(self, symbols: List[str]) -> str:
        """Create HTML report content."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>StockSage.AI Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 30px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #e9f4ff; border-radius: 5px; }
                .best { background-color: #d4edda; }
                .worst { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>StockSage.AI Model Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>
                <p>Symbols analyzed: {symbols}</p>
            </div>
        """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbols=", ".join(symbols)
        )
        
        # Add summary for each symbol
        for symbol in symbols:
            if symbol in self.evaluation_metrics:
                comparison_df = self.compare_models(symbol)
                
                if not comparison_df.empty:
                    html += f"""
                    <div class="section">
                        <h2>{symbol} Performance Summary</h2>
                        <div class="metric">
                            <strong>Best Model:</strong> {comparison_df.iloc[0]['model']} 
                            (SMAPE: {comparison_df.iloc[0]['smape']:.2f}%)
                        </div>
                        <div class="metric">
                            <strong>Models Evaluated:</strong> {len(comparison_df)}
                        </div>
                        
                        <h3>Detailed Results</h3>
                        {comparison_df.to_html(index=False, classes='table')}
                    </div>
                    """
        
        html += """
            <div class="section">
                <h2>Methodology</h2>
                <p>This evaluation used the following metrics:</p>
                <ul>
                    <li><strong>SMAPE:</strong> Symmetric Mean Absolute Percentage Error (lower is better)</li>
                    <li><strong>RMSE:</strong> Root Mean Square Error (lower is better)</li>
                    <li><strong>MAE:</strong> Mean Absolute Error (lower is better)</li>
                    <li><strong>Direction Accuracy:</strong> Percentage of correct direction predictions (higher is better)</li>
                    <li><strong>Correlation:</strong> Correlation between predicted and actual values (higher is better)</li>
                </ul>
            </div>
            
            <div class="section">
                <p><em>Generated by StockSage.AI Model Evaluation System</em></p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_results(self, symbols: List[str]):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=2, default=str)
        
        # Save comparison as CSV
        all_comparisons = []
        for symbol in symbols:
            comparison_df = self.compare_models(symbol)
            if not comparison_df.empty:
                all_comparisons.append(comparison_df)
        
        if all_comparisons:
            combined_df = pd.concat(all_comparisons, ignore_index=True)
            csv_file = self.output_dir / f"model_comparison_{timestamp}.csv"
            combined_df.to_csv(csv_file, index=False)
            logger.info(f"Saved comparison CSV: {csv_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate StockSage.AI Models')
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/model_training',
        help='Directory containing training results'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols to evaluate (if not provided, uses all from results)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        help='Comma-separated list of specific models to evaluate'
    )
    
    parser.add_argument(
        '--test-start-date',
        type=str,
        default='2024-01-01',
        help='Start date for fresh test data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--test-end-date',
        type=str,
        default='2024-12-31',
        help='End date for fresh test data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/evaluation',
        help='Output directory for reports'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate comprehensive HTML report'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    print("StockSage.AI Model Evaluation Script")
    print("===================================\n")
    
    args = parse_arguments()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots
    )
    
    try:
        # Load training results
        training_results = evaluator.load_training_results()
        
        # Determine symbols to evaluate
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
        else:
            symbols = list(training_results.keys())
        
        logger.info(f"Evaluating symbols: {symbols}")
        
        # Parse specific models if provided
        specific_models = None
        if args.models:
            specific_models = [m.strip() for m in args.models.split(',')]
        
        # Evaluate each symbol
        for symbol in symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating {symbol}")
            logger.info(f"{'='*50}")
            
            # Load models
            models = evaluator.load_saved_models(symbol, specific_models)
            
            if models:
                # Evaluate on fresh data
                evaluation_results = evaluator.evaluate_models_on_fresh_data(
                    symbol, 
                    args.test_start_date, 
                    args.test_end_date
                )
                
                # Generate plots
                if not args.no_plots:
                    evaluator.generate_performance_plots(symbol)
                
                # Statistical significance testing
                sig_tests = evaluator.statistical_significance_test(symbol)
                
                # Print summary
                comparison_df = evaluator.compare_models(symbol)
                if not comparison_df.empty:
                    print(f"\nTop 3 Models for {symbol}:")
                    for _, row in comparison_df.head(3).iterrows():
                        print(f"  {row['rank']}. {row['model']} - SMAPE: {row['smape']:.2f}%")
        
        # Save results
        evaluator.save_results(symbols)
        
        # Generate comprehensive report if requested
        if args.generate_report:
            report_file = evaluator.generate_comprehensive_report(symbols)
            print(f"\nComprehensive report generated: {report_file}")
        
        print(f"\nEvaluation completed successfully!")
        print(f"Check results in: {evaluator.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())