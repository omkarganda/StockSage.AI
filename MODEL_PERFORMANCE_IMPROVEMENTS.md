# StockSage.AI Model Performance Improvements

## Executive Summary

This document outlines the comprehensive improvements implemented to address the performance issues identified in the sMAPE-based evaluation analysis. The issues ranged from minimal training parameters to target-prediction alignment problems and statistical model transformation issues.

## Issues Identified and Root Causes

### A. Performance Problems from Analysis
- **sMAPE scores between 135-200%**: Far above usable thresholds (10-20% for price forecasts)
- **Negative correlations with high directional accuracy**: Suggesting index shift or magnitude scaling issues
- **Statistical models exploding**: RMSE values of 5-150 indicating transformation problems
- **Small evaluation windows**: Directional accuracies in multiples of 10% (10-15 test points)

### B. Root Causes Identified
1. **Minimal Training**: Deep learning models had insufficient epochs and basic early stopping
2. **Log Transform Issues**: Statistical models (ETS/Theta/Ensemble) were applying log transforms to price series that cross zero
3. **Target-Prediction Misalignment**: Potential index shifts between predictions and actual values
4. **Inadequate Evaluation**: Single-split evaluation on small test sets

## Improvements Implemented

### 1. Deep Learning Model Training Enhancements

#### A. LSTM Attention Model (`src/models/lstm_attention_model.py`)

**Previous Issues:**
- Only 50 epochs with basic early stopping
- No learning rate scheduling
- No gradient clipping
- Simple early stopping (patience=10)

**Improvements Made:**
```python
# Training parameters enhanced
epochs: int = 100  # Increased from 50
patience: int = 15  # Increased from 10
min_delta: float = 1e-6  # Minimum improvement threshold

# Added learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
)

# Added gradient clipping
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

# Enhanced training metrics tracking
self.training_metrics = {
    "train_mse": float(best_loss),
    "final_epoch": epoch,
    "train_losses": train_losses[-10:],
    "early_stopped": patience_counter >= self.patience
}
```

#### B. Transformer Model (`src/models/transformer_model.py`)

**Similar enhancements applied:**
- Increased epochs from 50 to 100
- Added learning rate scheduling
- Gradient clipping
- Better early stopping with minimum delta
- Enhanced logging and metrics tracking

### 2. Statistical Model Improvements

#### A. Log Transform Issue Fixed (`src/models/sktime_model.py`)

**Problem:** Log transforms were being applied to all statistical models by default, causing issues with price series that cross zero or have negative returns.

**Solution:**
```python
# Disable log transform for ETS, Theta, and Ensemble models
if model_type in ['ets', 'theta', 'ensemble']:
    self.use_log_transform = False
    logger.info(f"Disabled log transform for {model_type} model as recommended")
else:
    self.use_log_transform = use_log_transform
```

This addresses the statistical model explosion issues (RMSE ≈ 5 to 150) mentioned in the analysis.

### 3. Enhanced Model Evaluation System

#### A. Improved sMAPE Calculation (`scripts/evaluate_models.py`)

**Previous Issues:**
- Basic sMAPE calculation
- No alignment verification
- Limited diagnostic information

**Improvements:**
```python
# Improved SMAPE calculation
denominator = (np.abs(actual) + np.abs(pred_values)) / 2.0
denominator = np.where(denominator < 1e-8, 1e-8, denominator)
smape = np.mean(np.abs(actual - pred_values) / denominator) * 100

# Added lag-based alignment verification
correlations = []
for lag in range(-3, 4):  # Check lags from -3 to +3
    # Calculate correlation at different lags
    # Find best lag alignment

# Enhanced diagnostic metrics
evaluation_results[model_name] = {
    # ... existing metrics ...
    'mean_actual': float(mean_actual),
    'mean_predicted': float(mean_predicted),
    'prediction_bias': float(mean_predicted - mean_actual),
    'best_lag': best_lag,
    'best_lag_correlation': float(best_corr),
    'num_samples': min_len
}
```

### 4. New Diagnostic System

#### A. Model Diagnostic Script (`scripts/diagnose_models.py`)

**Purpose:** Address Section E recommendations for target-prediction alignment verification and rolling window evaluation.

**Key Features:**
1. **Target-Prediction Alignment Verification:**
   - Visual plots showing y_true vs y_pred for LSTM and other models
   - Lag correlation analysis to detect index shifts
   - Residual analysis

2. **Rolling Window Evaluation:**
   - Implements 5 folds × 30 days evaluation as recommended
   - Provides statistical significance across multiple test periods
   - Aggregated metrics with confidence intervals

3. **Diagnostic Visualizations:**
   - Time series alignment plots
   - Scatter plots for prediction accuracy
   - Lag correlation analysis
   - Residual distribution analysis

**Usage Examples:**
```bash
# Diagnose specific model
python scripts/diagnose_models.py --symbol AAPL --model dl_lstm_attention

# Run rolling window evaluation
python scripts/diagnose_models.py --rolling-eval --n-folds 5 --fold-size 30
```

### 5. Training Process Improvements

#### A. Enhanced Training Script (`scripts/train_models.py`)

**Improvements:**
- Better hyperparameter tuning integration
- Enhanced logging and progress tracking
- Improved model saving and validation
- More robust error handling

## Expected Performance Improvements

### A. Deep Learning Models
**Before:**
- LSTM: sMAPE 135.2%, Correlation -0.55
- Transformer: sMAPE 200.0%, Correlation 0.00

**Expected After Improvements:**
- **Better Training:** Increased epochs (100 vs 50) with learning rate scheduling should improve convergence
- **Gradient Clipping:** Prevents exploding gradients that could cause poor performance
- **Early Stopping:** Better early stopping with min_delta prevents overfitting

### B. Statistical Models
**Before:**
- ETS: sMAPE 194.9%, RMSE 4.939
- Theta: sMAPE 199.8%, RMSE 148.1

**Expected After Improvements:**
- **No Log Transform:** Removing blanket log transforms should dramatically reduce RMSE explosion
- **Better Frequency Handling:** Improved PeriodIndex conversion for sktime models

### C. Evaluation Quality
**Before:**
- Single test split with ~10-15 test points
- Directional accuracy in multiples of 10%

**Expected After Improvements:**
- **Rolling Window:** 5 folds × 30 days = 150 total test points
- **Better Alignment:** Lag detection prevents correlation sign flips
- **Robust Metrics:** Confidence intervals and statistical significance

## Next Steps and Recommendations

### 1. Immediate Actions (Section E Implementation)

1. **Retrain Models:**
   ```bash
   # Retrain LSTM with improved parameters
   python scripts/train_models.py --symbols AAPL --tune
   ```

2. **Run Diagnostic Analysis:**
   ```bash
   # Verify target-prediction alignment
   python scripts/diagnose_models.py --symbol AAPL --model dl_lstm_attention
   ```

3. **Rolling Window Evaluation:**
   ```bash
   # Get robust performance estimates
   python scripts/evaluate_models.py --symbol AAPL --rolling-eval
   ```

### 2. Success Criteria

As outlined in Section E, success would be achieving:
- **sMAPE < 50%** (down from 135-200%)
- **Directional accuracy ≥ 55-60%** (up from 10-66%)
- **Correlation ≥ 0.3** (up from negative correlations)

### 3. Feature Engineering Improvements

**Potential Next Steps:**
- Add macro/sentiment features as suggested in Section D.5
- Implement PCA for highly collinear technical indicators
- Add option-implied volatility and earnings date features

### 4. Ensembling Strategy

Once individual models improve:
- Simple average of top-k directionally correct models
- Dynamic weight adjustment based on recent performance
- Regime-aware ensemble switching

## Technical Implementation Details

### File Changes Made
1. `src/models/lstm_attention_model.py` - Enhanced training parameters
2. `src/models/transformer_model.py` - Enhanced training parameters  
3. `src/models/sktime_model.py` - Fixed log transform issues
4. `scripts/evaluate_models.py` - Improved evaluation metrics
5. `scripts/diagnose_models.py` - **NEW** diagnostic system

### Configuration Updates
- Deep learning epochs: 50 → 100
- Early stopping patience: 10 → 15
- Added learning rate scheduling
- Added gradient clipping
- Disabled log transforms for statistical models

## Monitoring and Validation

### A. Performance Tracking
- Track sMAPE improvements over time
- Monitor correlation stability
- Validate directional accuracy improvements

### B. Diagnostic Monitoring
- Regular alignment checks
- Rolling window performance tracking
- Lag correlation monitoring

### C. Model Health Checks
- Training convergence monitoring
- Gradient flow analysis
- Feature importance stability

## Conclusion

These comprehensive improvements address all the major issues identified in the performance analysis:

1. **Training Quality:** Enhanced deep learning training with proper scheduling and regularization
2. **Statistical Models:** Fixed log transform issues causing RMSE explosions
3. **Evaluation Robustness:** Implemented rolling window evaluation and alignment verification
4. **Diagnostic Capabilities:** Created comprehensive diagnostic tools for ongoing monitoring

The improvements should result in significantly better model performance, with sMAPE scores dropping closer to usable ranges and correlations becoming positive and meaningful.

---

**Generated on:** 2024-12-19  
**Author:** StockSage.AI Development Team  
**Status:** Ready for Implementation and Testing