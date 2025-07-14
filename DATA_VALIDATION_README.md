# StockSage.AI Data Validation & Quality Checks System

## Overview

This document describes the comprehensive data validation and quality checks system implemented for StockSage.AI. The system provides robust error handling, data quality validation, and centralized logging to ensure data reliability throughout the pipeline.

## 🚀 Quick Start

### Run the Complete Validation Demo
```bash
python scripts/test_validation.py
```

### Run Enhanced Merge Test with Validation
```bash
python scripts/test_merge.py
```

### Run Data Quality Dashboard
```bash
python scripts/data_quality_dashboard.py --action demo
```

## 📁 System Components

### Core Modules

#### 1. Data Validation (`src/data/validation.py`)
- **DataValidator**: Main validation class with configurable rules
- **ValidationReport**: Structured validation results
- **ValidationIssue**: Individual validation problems
- **Quick validation functions**: `validate_market_data()`, `validate_sentiment_data()`, etc.

#### 2. Enhanced Logging (`src/utils/logging.py`)
- **StockSageLogger**: Enhanced logger with performance monitoring
- **Structured logging**: Context-aware logging with JSON support
- **Performance tracking**: Automatic timing and statistics
- **Decorators**: `@log_function_call`, `@log_data_operation`

#### 3. Data Quality Dashboard (`scripts/data_quality_dashboard.py`)
- **Historical tracking**: Quality metrics over time
- **Trend analysis**: Quality improvement/degradation detection
- **HTML reports**: Comprehensive quality reports
- **Alert system**: Configurable quality thresholds

## ✅ Validation Features

### Data Type Validation
- **Market Data**: OHLCV validation, business logic checks
- **Sentiment Data**: Score ranges, confidence validation
- **Economic Data**: Reasonable value ranges, temporal consistency
- **Unified Datasets**: Cross-dataset consistency checks

### Quality Checks
- ✅ Missing data analysis
- ✅ Outlier detection (IQR method)
- ✅ Duplicate identification
- ✅ Data type consistency
- ✅ Business logic validation
- ✅ Time series validation
- ✅ Index validation (datetime checks)

### Severity Levels
- **INFO**: Informational notices
- **WARNING**: Issues that should be reviewed
- **ERROR**: Data problems that need fixing
- **CRITICAL**: Issues that prevent processing

## 📊 Usage Examples

### Basic Data Validation

```python
from src.data.validation import validate_market_data
import pandas as pd

# Load your market data
market_df = pd.read_csv('market_data.csv', index_col=0, parse_dates=True)

# Validate the data
report = validate_market_data(market_df, "AAPL_market_data")

# Check results
print(f"Is valid: {report.is_valid}")
print(f"Quality score: {report.quality_score}/100")

# Print detailed report
report.print_summary()
```

### Custom Validation Rules

```python
from src.data.validation import DataValidator

# Create validator with strict mode
validator = DataValidator(strict_mode=True)

# Validate with custom rules
report = validator.validate_dataset(
    df=your_data,
    dataset_type="market",  # or "sentiment", "economic", "unified"
    dataset_name="custom_dataset"
)
```

### Pipeline Integration

```python
from src.data.merge import create_unified_dataset

# Create unified dataset with validation
unified_df, validation_reports = create_unified_dataset(
    symbol="AAPL",
    start_date="2023-01-01",
    end_date="2023-12-31",
    market_data=market_data,
    economic_data=economic_data,
    sentiment_data=sentiment_data,
    validate_inputs=True,   # Validate input data
    validate_output=True    # Validate final dataset
)

# Check if any critical issues
critical_issues = sum(1 for report in validation_reports 
                     if report.has_critical_issues())
print(f"Critical issues found: {critical_issues}")
```

### Enhanced Logging

```python
from src.utils.logging import get_logger, log_data_operation

# Get logger for your module
logger = get_logger(__name__)

# Use structured logging
logger.info("Processing started", 
           symbol="AAPL", 
           dataset_size=len(data))

# Time operations automatically
with logger.timer("data_processing"):
    # Your data processing code here
    pass

# Use decorators for automatic logging
@log_data_operation("data_download")
def download_data(symbol):
    # Your function code
    pass
```

### Data Quality Dashboard

```python
from scripts.data_quality_dashboard import DataQualityDashboard

# Initialize dashboard
dashboard = DataQualityDashboard()

# Record validation results
summary = dashboard.record_validation_results(
    reports=validation_reports,
    dataset_category="daily_pipeline"
)

# Generate quality report
html_report = dashboard.generate_data_quality_report()

# Check for alerts
alerts = dashboard.check_quality_alerts()
for alert in alerts:
    print(f"ALERT: {alert['message']}")
```

## 🔧 Configuration

### Validation Rules
Validation rules are defined in `DataValidator._load_validation_rules()`:

```python
validation_rules = {
    'market': {
        'required_columns': ['Open', 'High', 'Low', 'Close'],
        'column_types': {
            'Open': 'numeric',
            'High': 'numeric',
            'Low': 'numeric',
            'Close': 'numeric',
            'Volume': 'numeric'
        },
        'value_ranges': {
            'Open': {'min': 0},
            'High': {'min': 0},
            'Low': {'min': 0},
            'Close': {'min': 0},
            'Volume': {'min': 0}
        }
    }
}
```

### Alert Thresholds
Configure quality alert thresholds:

```python
thresholds = {
    'min_pass_rate': 80.0,        # Minimum pass rate (%)
    'min_quality_score': 70.0,    # Minimum quality score
    'max_critical_issues': 0,     # Max critical issues allowed
    'max_error_issues': 5         # Max error issues allowed
}

alerts = dashboard.check_quality_alerts(thresholds)
```

## 📈 Quality Metrics

### Quality Score Calculation
The quality score (0-100) is calculated based on:
- **Base score**: 100 points
- **Deductions**: 
  - Critical issues: -25 points each
  - Error issues: -10 points each
  - Warning issues: -5 points each
  - Info issues: -1 point each
- **Missing data penalty**: -0.5 points per % of missing data

### Validation Categories
- **structure**: Basic DataFrame structure issues
- **index**: DateTime index problems
- **columns**: Missing or suspicious columns
- **data_types**: Data type inconsistencies
- **value_ranges**: Values outside expected ranges
- **missing_data**: Missing data patterns
- **duplicates**: Duplicate records
- **outliers**: Statistical outliers
- **time_series**: Time series specific issues
- **business_logic**: Domain-specific rule violations

## 🚨 Error Handling

### Validation Integration
The merge pipeline automatically validates data at key points:

1. **Input validation**: Before processing
2. **Intermediate validation**: After major transformations
3. **Output validation**: Final dataset validation

### Error Recovery
- **Critical errors**: Stop processing, require intervention
- **Errors**: Log and continue with warnings
- **Warnings**: Log for review, continue processing
- **Info**: Log for awareness

### Logging Levels
- **DEBUG**: Detailed technical information
- **INFO**: General operational information
- **WARNING**: Issues that should be reviewed
- **ERROR**: Errors that need attention
- **CRITICAL**: Critical failures

## 📊 Reports and Dashboards

### HTML Quality Reports
Generated reports include:
- Latest quality check results
- Historical quality trends
- Issue breakdown by category
- Data quality recommendations
- Performance metrics

### Quality History Tracking
The dashboard maintains JSON files with:
- Validation results over time
- Quality score trends
- Issue frequency analysis
- Performance statistics

## 🧪 Testing and Validation

### Demo Scripts
1. **`test_validation.py`**: Comprehensive validation demo
2. **`test_merge.py`**: Enhanced merge testing with validation
3. **`data_quality_dashboard.py`**: Dashboard functionality demo

### Sample Data Generation
The demo scripts include functions to create realistic sample data:
- Market data with OHLCV patterns
- Sentiment data with realistic score distributions
- Economic indicators with appropriate ranges
- Intentional data quality issues for testing

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure you're in the project root directory
cd /path/to/StockSage.AI

# Run with proper Python path
PYTHONPATH=. python scripts/test_validation.py
```

#### Missing Dependencies
```bash
# Install required packages
pip install pandas numpy
```

#### Log Directory Permissions
```bash
# Create logs directory if it doesn't exist
mkdir -p logs
chmod 755 logs
```

### Performance Issues
- Use `validate_inputs=False` during development
- Monitor performance stats via logging
- Adjust validation rules for large datasets

## 📝 Best Practices

### Development Workflow
1. **Enable validation** during development
2. **Monitor quality trends** in production
3. **Set appropriate thresholds** for your use case
4. **Review validation reports** regularly
5. **Update validation rules** as data evolves

### Production Deployment
1. **Use structured logging** (JSON format)
2. **Set up quality alerts** for critical issues
3. **Monitor performance metrics** 
4. **Archive quality reports** for compliance
5. **Implement automated responses** to quality issues

## 🚀 Future Enhancements

### Planned Features
- **Real-time monitoring**: Live quality dashboards
- **Machine learning validation**: Anomaly detection models
- **Data lineage tracking**: Track data quality through transformations
- **Custom validation rules**: User-defined validation logic
- **Integration with alerting systems**: Slack, email, PagerDuty

### Extension Points
- **Custom validators**: Implement domain-specific checks
- **Additional data types**: Extend validation for new data sources
- **Quality metrics**: Define custom quality measures
- **Reporting formats**: Add PDF, Excel export options

## 📞 Support

For questions or issues with the validation system:
1. Check the logs in `logs/` directory
2. Review validation reports in `reports/data_quality/`
3. Run the demo scripts to verify functionality
4. Check the source code documentation in `src/data/validation.py`

---

**Note**: This validation system is designed to be comprehensive yet performant. For large-scale production deployments, consider implementing sampling strategies and asynchronous validation processes.