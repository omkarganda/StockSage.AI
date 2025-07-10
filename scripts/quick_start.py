#!/usr/bin/env python3
"""
Quick Start Script for StockSage.AI

This script provides an easy way to get started with training and evaluating
all baseline models with just a few commands.

Usage:
    python scripts/quick_start.py                    # Quick demo with AAPL
    python scripts/quick_start.py --full             # Full training with multiple stocks
    python scripts/quick_start.py --symbol TSLA      # Specific stock
    python scripts/quick_start.py --demo             # Demo mode (fast, limited models)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command: str, description: str) -> bool:
    """
    Run a shell command and return success status.
    
    Parameters:
    -----------
    command : str
        Command to execute
    description : str
        Description of what the command does
        
    Returns:
    --------
    bool
        True if command succeeded, False otherwise
    """
    logger.info(f"🔄 {description}")
    logger.info(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - SUCCESS")
            return True
        else:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"   Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {description} - ERROR: {e}")
        return False


def create_config_file(symbols, start_date, end_date, quick_test=False):
    """Create a configuration file for training."""
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    config_content = f"""# StockSage.AI Training Configuration
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Data Configuration
start_date: "{start_date}"
end_date: "{end_date}"
train_split: 0.8

# Model Configuration
forecast_horizon: 30
quick_test: {str(quick_test).lower()}

# Results Configuration
results_dir: "results/model_training"
"""
    
    config_file = config_dir / "training_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    logger.info(f"📝 Created configuration file: {config_file}")
    return str(config_file)


def quick_demo():
    """Run a quick demo with AAPL."""
    print("🎯 StockSage.AI Quick Demo")
    print("==========================")
    print("This demo will:")
    print("1. Download AAPL data (2022-2024)")
    print("2. Train 3 baseline models quickly")
    print("3. Evaluate and compare results")
    print()
    
    # Define parameters
    symbol = "AAPL"
    start_date = "2022-01-01"
    end_date = "2024-01-01"  # Leave 2024 for testing
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Step 1: Quick training
    train_command = f"""python scripts/train_models.py \
        --symbols {symbol} \
        --start-date {start_date} \
        --end-date {end_date} \
        --quick-test \
        --forecast-horizon 14"""
    
    if run_command(train_command, "Quick model training"):
        
        # Step 2: Evaluation
        eval_command = f"""python scripts/evaluate_models.py \
            --symbols {symbol} \
            --test-start-date 2024-01-01 \
            --test-end-date 2024-12-31 \
            --generate-report"""
        
        run_command(eval_command, "Model evaluation and reporting")
    
    print("\n🎉 Quick demo completed!")
    print("📊 Check 'results/model_training' for training results")
    print("📋 Check 'reports/evaluation' for evaluation reports")


def full_training(symbols):
    """Run full training with multiple models."""
    print("🚀 StockSage.AI Full Training")
    print("=============================")
    print(f"Training on symbols: {', '.join(symbols)}")
    print("This will train all available models including:")
    print("- Traditional ML (Linear Regression, Random Forest, Ensemble)")
    print("- Statistical (ARIMA, Prophet, ETS, AutoML)")
    print("- Neural Networks (if dependencies available)")
    print()
    
    # Define date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')  # 3 years
    test_start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Last year for testing
    
    # Create config file
    config_file = create_config_file(symbols, start_date, end_date, quick_test=False)
    
    # Step 1: Full training
    train_command = f"""python scripts/train_models.py \
        --symbols {','.join(symbols)} \
        --config {config_file}"""
    
    if run_command(train_command, "Full model training"):
        
        # Step 2: Comprehensive evaluation
        eval_command = f"""python scripts/evaluate_models.py \
            --symbols {','.join(symbols)} \
            --test-start-date {test_start} \
            --test-end-date {end_date} \
            --generate-report"""
        
        run_command(eval_command, "Comprehensive model evaluation")
    
    print("\n🎉 Full training completed!")
    print("📊 Check 'results/model_training' for detailed results")
    print("📋 Check 'reports/evaluation' for evaluation reports")


def specific_symbol_training(symbol):
    """Train models for a specific symbol."""
    print(f"📈 StockSage.AI Training for {symbol}")
    print("=====================================")
    
    # Define date range (2 years of data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
    test_start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')  # Last 6 months for testing
    
    # Training
    train_command = f"""python scripts/train_models.py \
        --symbols {symbol} \
        --start-date {start_date} \
        --end-date {end_date} \
        --forecast-horizon 30"""
    
    if run_command(train_command, f"Training models for {symbol}"):
        
        # Evaluation
        eval_command = f"""python scripts/evaluate_models.py \
            --symbols {symbol} \
            --test-start-date {test_start} \
            --test-end-date {end_date} \
            --generate-report"""
        
        run_command(eval_command, f"Evaluating models for {symbol}")
    
    print(f"\n🎉 Training for {symbol} completed!")


def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'scikit-learn', 
        'matplotlib', 'plotly', 'joblib'
    ]
    
    optional_packages = [
        'sktime', 'neuralforecast', 'prophet', 'timesfm'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"❌ Missing required packages: {', '.join(missing_required)}")
        print("Please install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional packages: {', '.join(missing_optional)}")
        print("Some advanced models may not be available.")
        print("Install with: pip install " + " ".join(missing_optional))
    
    print("✅ Core dependencies check passed!")
    return True


def main():
    """Main function."""
    print("🚀 StockSage.AI Quick Start")
    print("===========================\n")
    
    parser = argparse.ArgumentParser(description='StockSage.AI Quick Start')
    parser.add_argument('--demo', action='store_true', 
                       help='Run quick demo with AAPL')
    parser.add_argument('--full', action='store_true', 
                       help='Run full training with major stocks')
    parser.add_argument('--symbol', type=str, 
                       help='Train models for specific symbol')
    parser.add_argument('--check', action='store_true', 
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies():
        return 1
    
    if args.check:
        return 0
    
    try:
        if args.demo:
            quick_demo()
        elif args.full:
            # Major tech stocks
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            full_training(symbols)
        elif args.symbol:
            specific_symbol_training(args.symbol.upper())
        else:
            # Default: quick demo
            print("No specific option provided. Running quick demo...")
            print("Use --help to see all options.\n")
            quick_demo()
        
        print("\n" + "="*50)
        print("🎉 All done! Here's what you can do next:")
        print("="*50)
        print("📊 View training results: results/model_training/")
        print("📋 View evaluation reports: reports/evaluation/")
        print("📈 View performance plots: reports/evaluation/*.html")
        print("\n💡 Tips:")
        print("- Open HTML files in your browser for interactive plots")
        print("- Check CSV files for model performance comparison")
        print("- Use evaluate_models.py for fresh evaluations")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())