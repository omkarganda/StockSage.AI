#!/usr/bin/env python3
"""
Model Training Script for StockSage.AI

This script downloads stock data on the fly and trains all available models:
- Traditional ML models (Linear Regression, Random Forest, Ensemble)
- Neural forecasting models (TimesFM, TTM, NeuralForecast)
- Statistical models (ARIMA, Prophet, ETS, etc.)

Usage:
    python scripts/train_models.py --symbol AAPL --start-date 2020-01-01 --end-date 2023-12-31
    python scripts/train_models.py --symbols AAPL,GOOGL,MSFT --quick-test
    python scripts/train_models.py --config configs/training_config.yaml
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import time
from typing import Dict, List, Optional, Any, Tuple
import logging

# Import our models
from src.models.baseline import create_baseline_models
from src.models.neuralforecast_model import create_neural_forecasting_models
from src.models.sktime_model import create_statistical_models

# Import data processing
from src.data.merge import create_unified_dataset
from src.features.indicators import add_all_technical_indicators
# NEW: data cleaning & validation utilities
from src.data.cleaning import clean_market_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Comprehensive model trainer for StockSage.AI
    
    This class handles data downloading, preprocessing, model training,
    and evaluation for all available model types.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        train_split: float = 0.8,
        forecast_horizon: int = 30,
        quick_test: bool = False,
        save_models: bool = True,
        results_dir: str = "results/model_training"
    ):
        """
        Initialize the model trainer.
        
        Parameters:
        -----------
        symbols : List[str]
            List of stock symbols to train on
        start_date : str
            Start date for data (YYYY-MM-DD)
        end_date : str
            End date for data (YYYY-MM-DD)
        train_split : float, default=0.8
            Fraction of data to use for training
        forecast_horizon : int, default=30
            Number of days to forecast
        quick_test : bool, default=False
            Whether to run a quick test with limited models
        save_models : bool, default=True
            Whether to save trained models
        results_dir : str, default="results/model_training"
            Directory to save results
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.train_split = train_split
        self.forecast_horizon = forecast_horizon
        self.quick_test = quick_test
        self.save_models = save_models
        self.results_dir = Path(results_dir)
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.training_results = {}
        self.evaluation_results = {}
        self.model_instances = {}
        
        logger.info(f"Initialized ModelTrainer for symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Forecast horizon: {forecast_horizon} days")
    
    def download_data(self, symbol: str) -> pd.DataFrame:
        """
        Download and preprocess stock data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to download
            
        Returns:
        --------
        pd.DataFrame
            Preprocessed stock data with technical indicators
        """
        logger.info(f"Downloading data for {symbol}...")
        
        try:
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start_date, end=self.end_date)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Rename columns to match our expected format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Remove dividends and stock splits for now
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # First pass cleaning to remove duplicates / bad rows
            data = clean_market_data(data, symbol=symbol, validate=True)

            # Add technical indicators (after base cleaning)
            logger.info(f"Adding technical indicators for {symbol}...")
            data = add_all_technical_indicators(data)

            # A second fill to cover any NaNs introduced by indicator calc
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Downloaded {len(data)} rows for {symbol}")
            logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")
            raise
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Full dataset
            
        Returns:
        --------
        tuple
            Training and testing dataframes
        """
        split_idx = int(len(data) * self.train_split)
        
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)} rows, Test: {len(test_data)} rows")
        
        return train_data, test_data
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create all available models.
        
        Returns:
        --------
        dict
            Dictionary of all model instances
        """
        logger.info("Creating models...")
        
        models = {}
        
        # Baseline models (always include)
        try:
            baseline_models = create_baseline_models(
                prediction_horizon=self.forecast_horizon
            )
            models.update({f"baseline_{k}": v for k, v in baseline_models.items()})
            logger.info(f"Created {len(baseline_models)} baseline models")
        except Exception as e:
            logger.error(f"Error creating baseline models: {e}")
        
        # Neural forecasting models (if not quick test)
        if not self.quick_test:
            try:
                neural_models = create_neural_forecasting_models(
                    context_length=min(252, self.forecast_horizon * 4),  # About 1 year or 4x horizon
                    horizon_length=self.forecast_horizon,
                    fine_tune_foundation_models=True
                )
                models.update({f"neural_{k}": v for k, v in neural_models.items()})
                logger.info(f"Created {len(neural_models)} neural forecasting models")
            except Exception as e:
                logger.warning(f"Error creating neural models: {e}")
        
        # Statistical models
        try:
            statistical_models = create_statistical_models(
                forecast_horizon=self.forecast_horizon,
                seasonal_period=252 if not self.quick_test else None,  # Daily data, yearly seasonality
                include_automl=not self.quick_test
            )
            models.update({f"statistical_{k}": v for k, v in statistical_models.items()})
            logger.info(f"Created {len(statistical_models)} statistical models")
        except Exception as e:
            logger.error(f"Error creating statistical models: {e}")
        
        logger.info(f"Total models created: {len(models)}")
        return models
    
    def train_model(
        self, 
        model_name: str, 
        model: Any, 
        train_data: pd.DataFrame, 
        symbol: str
    ) -> Dict[str, Any]:
        """
        Train a single model and return training metrics.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : Any
            Model instance
        train_data : pd.DataFrame
            Training data
        symbol : str
            Stock symbol
            
        Returns:
        --------
        dict
            Training results and metrics
        """
        logger.info(f"Training {model_name} for {symbol}...")
        
        start_time = time.time()
        training_result = {
            'model_name': model_name,
            'symbol': symbol,
            'train_samples': len(train_data),
            'start_time': start_time,
            'status': 'failed',
            'error': None,
            'training_metrics': {},
            'training_time': 0
        }
        
        try:
            # Train the model
            model.fit(train_data)
            
            training_time = time.time() - start_time
            training_result.update({
                'status': 'success',
                'training_time': training_time
            })
            
            # Get training metrics if available
            if hasattr(model, 'training_metrics'):
                training_result['training_metrics'] = model.training_metrics
            elif hasattr(model, 'get_model_summary'):
                summary = model.get_model_summary()
                if 'training_metrics' in summary:
                    training_result['training_metrics'] = summary['training_metrics']
            
            logger.info(f"✅ {model_name} training completed in {training_time:.2f}s")
            
        except Exception as e:
            training_time = time.time() - start_time
            error_msg = str(e)
            training_result.update({
                'status': 'failed',
                'error': error_msg,
                'training_time': training_time
            })
            
            logger.error(f"❌ {model_name} training failed: {error_msg}")
        
        return training_result
    
    def evaluate_model(
        self, 
        model_name: str, 
        model: Any, 
        test_data: pd.DataFrame, 
        symbol: str
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : Any
            Trained model instance
        test_data : pd.DataFrame
            Test data
        symbol : str
            Stock symbol
            
        Returns:
        --------
        dict
            Evaluation results and metrics
        """
        logger.info(f"Evaluating {model_name} for {symbol}...")
        
        evaluation_result = {
            'model_name': model_name,
            'symbol': symbol,
            'test_samples': len(test_data),
            'status': 'failed',
            'error': None,
            'evaluation_metrics': {},
            'predictions': None
        }
        
        try:
            # Check if model was trained successfully
            if not hasattr(model, 'is_fitted') or not model.is_fitted:
                if not hasattr(model, 'model') or model.model is None:
                    raise ValueError("Model is not fitted")
            
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(test_data)
                evaluation_result['predictions'] = predictions[:10].tolist() if hasattr(predictions, 'tolist') else str(predictions)[:100]
            
            # Get evaluation metrics
            if hasattr(model, 'evaluate'):
                metrics = model.evaluate(test_data)
                evaluation_result['evaluation_metrics'] = metrics
            elif hasattr(model, 'predict'):
                # Calculate basic metrics manually
                try:
                    actual = test_data['Close'].values[:self.forecast_horizon]
                    predicted = predictions[:len(actual)] if hasattr(predictions, '__len__') else [predictions]
                    
                    if len(predicted) > 0 and len(actual) > 0:
                        mse = np.mean((actual - predicted) ** 2)
                        mae = np.mean(np.abs(actual - predicted))
                        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                        
                        evaluation_result['evaluation_metrics'] = {
                            'mse': float(mse),
                            'rmse': float(np.sqrt(mse)),
                            'mae': float(mae),
                            'mape': float(mape)
                        }
                except Exception as metric_error:
                    logger.warning(f"Could not calculate metrics for {model_name}: {metric_error}")
            
            evaluation_result['status'] = 'success'
            logger.info(f"✅ {model_name} evaluation completed")
            
        except Exception as e:
            error_msg = str(e)
            evaluation_result.update({
                'status': 'failed',
                'error': error_msg
            })
            
            logger.error(f"❌ {model_name} evaluation failed: {error_msg}")
        
        return evaluation_result
    
    def train_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Train all models for all symbols.
        
        Returns:
        --------
        dict
            Complete training and evaluation results
        """
        logger.info("Starting comprehensive model training...")
        
        all_results = {}
        
        for symbol in self.symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing symbol: {symbol}")
            logger.info(f"{'='*50}")
            
            symbol_results = {
                'symbol': symbol,
                'data_info': {},
                'training_results': {},
                'evaluation_results': {},
                'summary': {}
            }
            
            try:
                # Download and prepare data
                data = self.download_data(symbol)
                train_data, test_data = self.split_data(data)
                
                symbol_results['data_info'] = {
                    'total_samples': len(data),
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                    'features': len(data.columns),
                    'date_range': f"{data.index.min()} to {data.index.max()}"
                }
                
                # Create models
                models = self.create_models()
                
                if not models:
                    logger.warning(f"No models created for {symbol}")
                    continue
                
                # Train and evaluate each model
                successful_models = 0
                failed_models = 0
                
                for model_name, model in models.items():
                    logger.info(f"\n--- Processing {model_name} ---")
                    
                    # Train model
                    training_result = self.train_model(model_name, model, train_data, symbol)
                    symbol_results['training_results'][model_name] = training_result
                    
                    # Evaluate model if training was successful
                    if training_result['status'] == 'success':
                        evaluation_result = self.evaluate_model(model_name, model, test_data, symbol)
                        symbol_results['evaluation_results'][model_name] = evaluation_result
                        
                        if evaluation_result['status'] == 'success':
                            successful_models += 1
                            
                            # Save model if requested
                            if self.save_models:
                                try:
                                    model_path = self.results_dir / f"{symbol}_{model_name}.joblib"
                                    model.save_model(model_path)
                                except Exception as save_error:
                                    logger.warning(f"Could not save {model_name}: {save_error}")
                        else:
                            failed_models += 1
                    else:
                        failed_models += 1
                
                # Create summary
                symbol_results['summary'] = {
                    'total_models': len(models),
                    'successful_models': successful_models,
                    'failed_models': failed_models,
                    'success_rate': successful_models / len(models) * 100 if models else 0
                }
                
                logger.info(f"\n📊 {symbol} Summary:")
                logger.info(f"Total models: {len(models)}")
                logger.info(f"Successful: {successful_models}")
                logger.info(f"Failed: {failed_models}")
                logger.info(f"Success rate: {symbol_results['summary']['success_rate']:.1f}%")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                symbol_results['error'] = str(e)
            
            all_results[symbol] = symbol_results
        
        # Save results
        self.save_results(all_results)
        
        # Print final summary
        self.print_final_summary(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save training results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        results_file = self.results_dir / f"training_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Create summary CSV
        self.create_summary_csv(results, timestamp)
    
    def create_summary_csv(self, results: Dict[str, Any], timestamp: str):
        """Create a summary CSV of model performance."""
        summary_data = []
        
        for symbol, symbol_results in results.items():
            if 'evaluation_results' not in symbol_results:
                continue
                
            for model_name, eval_result in symbol_results['evaluation_results'].items():
                if eval_result['status'] == 'success' and eval_result['evaluation_metrics']:
                    row = {
                        'symbol': symbol,
                        'model': model_name,
                        'status': eval_result['status'],
                        **eval_result['evaluation_metrics']
                    }
                    
                    # Add training time if available
                    if model_name in symbol_results['training_results']:
                        row['training_time'] = symbol_results['training_results'][model_name]['training_time']
                    
                    summary_data.append(row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = self.results_dir / f"model_summary_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"Summary CSV saved to {summary_file}")
            
            # Print top performers
            if 'mape' in summary_df.columns:
                logger.info("\n🏆 Top 5 Models by MAPE:")
                top_models = summary_df.nsmallest(5, 'mape')[['symbol', 'model', 'mape', 'rmse']]
                for _, row in top_models.iterrows():
                    logger.info(f"  {row['symbol']:6} | {row['model']:20} | MAPE: {row['mape']:6.2f}% | RMSE: {row['rmse']:8.2f}")
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Print a comprehensive summary of all results."""
        logger.info(f"\n{'='*60}")
        logger.info("🎯 FINAL TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        
        total_symbols = len(results)
        total_models_attempted = 0
        total_models_successful = 0
        
        for symbol, symbol_results in results.items():
            if 'summary' in symbol_results:
                total_models_attempted += symbol_results['summary']['total_models']
                total_models_successful += symbol_results['summary']['successful_models']
        
        logger.info(f"📈 Symbols processed: {total_symbols}")
        logger.info(f"🤖 Models attempted: {total_models_attempted}")
        logger.info(f"✅ Models successful: {total_models_successful}")
        logger.info(f"📊 Overall success rate: {total_models_successful/total_models_attempted*100:.1f}%" if total_models_attempted > 0 else "📊 Overall success rate: 0%")
        
        logger.info(f"\n💾 Results saved in: {self.results_dir}")
        logger.info(f"📁 Model files saved: {self.save_models}")
        
        # Model type breakdown
        model_types = {}
        for symbol_results in results.values():
            if 'evaluation_results' in symbol_results:
                for model_name, eval_result in symbol_results['evaluation_results'].items():
                    model_type = model_name.split('_')[0]  # baseline, neural, statistical
                    if model_type not in model_types:
                        model_types[model_type] = {'success': 0, 'total': 0}
                    model_types[model_type]['total'] += 1
                    if eval_result['status'] == 'success':
                        model_types[model_type]['success'] += 1
        
        logger.info(f"\n📋 Model Type Performance:")
        for model_type, stats in model_types.items():
            success_rate = stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0
            logger.info(f"  {model_type.capitalize():12} | {stats['success']:2}/{stats['total']:2} | {success_rate:5.1f}%")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train StockSage.AI Models')
    
    parser.add_argument(
        '--symbols', 
        type=str, 
        default='AAPL',
        help='Comma-separated list of stock symbols (e.g., AAPL,GOOGL,MSFT)'
    )
    
    parser.add_argument(
        '--start-date', 
        type=str, 
        default='2020-01-01',
        help='Start date for data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str, 
        default='2023-12-31',
        help='End date for data (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--train-split', 
        type=float, 
        default=0.8,
        help='Fraction of data to use for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--forecast-horizon', 
        type=int, 
        default=30,
        help='Number of days to forecast (default: 30)'
    )
    
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run quick test with limited models'
    )
    
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Do not save trained models'
    )
    
    parser.add_argument(
        '--results-dir', 
        type=str, 
        default='results/model_training',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='Path to YAML configuration file'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training function."""
    print("🚀 StockSage.AI Model Training Script")
    print("=====================================\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Load config if provided
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Create trainer
    trainer = ModelTrainer(
        symbols=symbols,
        start_date=config.get('start_date', args.start_date),
        end_date=config.get('end_date', args.end_date),
        train_split=config.get('train_split', args.train_split),
        forecast_horizon=config.get('forecast_horizon', args.forecast_horizon),
        quick_test=config.get('quick_test', args.quick_test),
        save_models=not args.no_save,
        results_dir=config.get('results_dir', args.results_dir)
    )
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run training
    try:
        results = trainer.train_all_models()
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Check results in: {trainer.results_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())