# 🚀 StockSage.AI Model Training & Testing Guide

Welcome to the StockSage.AI model training and testing system! This guide will help you train and evaluate all baseline models quickly and easily.

## 📋 Quick Start

### Option 1: Super Easy Demo (Recommended for First Time)
```bash
# Run a quick demo with Apple stock
python scripts/quick_start.py --demo
```
This will:
- Download 2 years of AAPL data automatically
- Train 3 baseline models quickly 
- Evaluate and compare results
- Generate interactive reports

### Option 2: Specific Stock
```bash
# Train models for Tesla
python scripts/quick_start.py --symbol TSLA

# Train models for Google
python scripts/quick_start.py --symbol GOOGL
```

### Option 3: Full Training (Multiple Stocks)
```bash
# Train on major tech stocks with all models
python scripts/quick_start.py --full
```
This trains on: AAPL, GOOGL, MSFT, AMZN, TSLA

## 🛠️ Manual Training & Testing

### Step 1: Train Models
```bash
# Basic training (single stock)
python scripts/train_models.py --symbols AAPL --start-date 2020-01-01 --end-date 2023-12-31

# Multiple stocks
python scripts/train_models.py --symbols AAPL,GOOGL,MSFT --quick-test

# Full training with all models (slower but comprehensive)
python scripts/train_models.py --symbols AAPL --start-date 2020-01-01 --end-date 2023-12-31 --forecast-horizon 30
```

### Step 2: Evaluate Models
```bash
# Evaluate trained models
python scripts/evaluate_models.py --symbols AAPL --generate-report

# Evaluate specific models only
python scripts/evaluate_models.py --symbols AAPL --models baseline_random_forest,statistical_auto_arima

# Custom test period
python scripts/evaluate_models.py --symbols AAPL --test-start-date 2024-01-01 --test-end-date 2024-12-31
```

## 📊 What Models Are Trained?

### 🎯 Baseline Models (Traditional ML)
- **Linear Regression** with Ridge/Lasso/ElasticNet regularization
- **Random Forest** with automated hyperparameter tuning
- **Ensemble Model** combining multiple algorithms

### 📈 Statistical Models
- **Auto-ARIMA** with automatic parameter selection
- **Prophet** for trend and seasonality modeling  
- **ETS** (Exponential Smoothing)
- **Theta Forecasting**
- **AutoML** for automatic model selection

### 🧠 Neural Forecasting Models (if dependencies available)
- **TimesFM** (Google's 200M parameter foundation model)
- **TTM** (IBM's Tiny Time Mixers)
- **NeuralForecast** (N-BEATS, N-HiTS, PatchTST)

## 📁 Output Structure

After training, you'll find:

```
results/model_training/
├── training_results_20240115_143022.json    # Detailed training results
├── model_summary_20240115_143022.csv        # Performance summary
├── AAPL_baseline_random_forest.joblib       # Saved model files
├── AAPL_statistical_auto_arima.joblib
└── ...

reports/evaluation/
├── model_evaluation_report_20240115_150045.html  # Interactive report
├── AAPL_performance_comparison.html               # Performance plots
├── AAPL_predictions_vs_actual.html               # Prediction visualization
└── evaluation_results_20240115_150045.json       # Detailed metrics
```

## 🎛️ Advanced Configuration

### Using Configuration Files
Create `configs/my_config.yaml`:
```yaml
# Data Configuration
start_date: "2020-01-01"
end_date: "2023-12-31"
train_split: 0.8

# Model Configuration
forecast_horizon: 30
quick_test: false

# Results Configuration
results_dir: "results/my_experiment"
```

Then run:
```bash
python scripts/train_models.py --config configs/my_config.yaml --symbols AAPL,TSLA
```

### Command Line Options

#### Training Script Options
```bash
python scripts/train_models.py --help

Options:
  --symbols AAPL,GOOGL,MSFT     # Comma-separated stock symbols
  --start-date 2020-01-01       # Start date for data
  --end-date 2023-12-31         # End date for data  
  --train-split 0.8             # Training/test split ratio
  --forecast-horizon 30         # Days to forecast
  --quick-test                  # Use limited models for speed
  --no-save                     # Don't save trained models
  --results-dir results/custom  # Custom results directory
  --config path/to/config.yaml  # Use configuration file
```

#### Evaluation Script Options
```bash
python scripts/evaluate_models.py --help

Options:
  --results-dir results/model_training     # Directory with training results
  --symbols AAPL,GOOGL                     # Symbols to evaluate
  --models baseline_random_forest,auto_arima # Specific models to evaluate
  --test-start-date 2024-01-01             # Fresh test data start
  --test-end-date 2024-12-31               # Fresh test data end
  --generate-report                        # Generate HTML report
  --no-plots                               # Disable plot generation
```

## 📊 Understanding Results

### Performance Metrics
- **MAPE**: Mean Absolute Percentage Error (lower = better)
- **RMSE**: Root Mean Square Error (lower = better)  
- **MAE**: Mean Absolute Error (lower = better)
- **Direction Accuracy**: % of correct up/down predictions (higher = better)
- **Correlation**: Correlation between predicted and actual (higher = better)

### Model Rankings
Models are automatically ranked by MAPE (lower is better). Check the CSV summary files for quick comparisons.

### Interactive Reports
Open the HTML files in your browser to see:
- Performance comparison charts
- Prediction vs actual plots  
- Model rankings and statistics
- Detailed methodology

## 🔧 Troubleshooting

### Missing Dependencies
```bash
# Check what's available
python scripts/quick_start.py --check

# Install core requirements
pip install -r requirements.txt

# Install optional packages for advanced models
pip install sktime prophet neuralforecast
```

### Common Issues

1. **"No data found"**: Check internet connection and symbol spelling
2. **Model training fails**: Try `--quick-test` first to test basic functionality
3. **Out of memory**: Reduce forecast horizon or use `--quick-test`
4. **Missing models in evaluation**: Check that training completed successfully

### Performance Tips

1. **Start small**: Use `--quick-test` first
2. **Use specific dates**: Shorter date ranges train faster
3. **Parallel training**: The system automatically uses multiple cores
4. **Save models**: Enable model saving to avoid retraining

## 📈 Example Workflows

### Research Workflow
```bash
# 1. Quick test to verify everything works
python scripts/quick_start.py --demo

# 2. Train comprehensive models for research
python scripts/train_models.py --symbols AAPL,GOOGL,MSFT,AMZN,TSLA --start-date 2019-01-01 --end-date 2023-12-31

# 3. Evaluate with fresh data
python scripts/evaluate_models.py --test-start-date 2024-01-01 --test-end-date 2024-12-31 --generate-report
```

### Production Workflow  
```bash
# 1. Train models monthly with recent data
python scripts/train_models.py --symbols SPY,QQQ --start-date 2021-01-01 --end-date $(date +%Y-%m-%d)

# 2. Evaluate performance weekly
python scripts/evaluate_models.py --symbols SPY,QQQ --generate-report
```

### Comparison Workflow
```bash
# 1. Train different configurations
python scripts/train_models.py --symbols AAPL --forecast-horizon 7 --results-dir results/weekly
python scripts/train_models.py --symbols AAPL --forecast-horizon 30 --results-dir results/monthly

# 2. Compare results
python scripts/evaluate_models.py --results-dir results/weekly --generate-report --output-dir reports/weekly
python scripts/evaluate_models.py --results-dir results/monthly --generate-report --output-dir reports/monthly
```

## 🎯 Next Steps

1. **Start with the demo**: `python scripts/quick_start.py --demo`
2. **Explore results**: Open HTML reports in your browser
3. **Try different stocks**: Use `--symbol TICKER` 
4. **Experiment with parameters**: Adjust forecast horizons and date ranges
5. **Compare models**: Look at the performance rankings
6. **Deploy best models**: Use the saved model files for production

## 💡 Tips for Best Results

1. **Use at least 2-3 years of data** for training
2. **Test on recent out-of-sample data** (last 6-12 months)
3. **Compare multiple metrics**, not just MAPE
4. **Consider ensemble approaches** for robustness
5. **Validate with domain knowledge** - do the predictions make sense?
6. **Monitor model drift** - retrain periodically with fresh data

---

**Happy modeling! 🚀📈**

For more advanced usage, check the individual script files and model documentation in `src/models/`.