"""
Data Validation & Quality Checks Module for StockSage.AI

This module provides comprehensive data validation and quality checking capabilities
for all data sources used in the StockSage.AI pipeline:
- Market data validation (OHLCV, price checks, volume validation)
- Economic indicators validation (range checks, consistency)
- Sentiment data validation (score ranges, confidence thresholds)
- Data completeness and consistency checks
- Time series specific validations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.logging import get_logger

# Get module logger
logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in data"""
    severity: ValidationSeverity
    category: str
    message: str
    column: Optional[str] = None
    row_count: Optional[int] = None
    percentage: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Complete validation report for a dataset"""
    dataset_name: str
    validation_timestamp: datetime
    total_rows: int
    total_columns: int
    issues: List[ValidationIssue]
    is_valid: bool
    quality_score: float  # 0-100 scale
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity"""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def has_critical_issues(self) -> bool:
        """Check if there are any critical issues"""
        return len(self.get_issues_by_severity(ValidationSeverity.CRITICAL)) > 0
    
    def print_summary(self):
        """Print a summary of the validation report"""
        logger.info(f"=== Validation Report for {self.dataset_name} ===")
        logger.info(f"Dataset: {self.total_rows} rows × {self.total_columns} columns")
        logger.info(f"Quality Score: {self.quality_score:.1f}/100")
        logger.info(f"Valid: {'✓' if self.is_valid else '✗'}")
        
        severity_counts = {}
        for severity in ValidationSeverity:
            count = len(self.get_issues_by_severity(severity))
            if count > 0:
                severity_counts[severity.value] = count
        
        if severity_counts:
            logger.info(f"Issues: {severity_counts}")
        else:
            logger.info("No issues found")
            
        # Print critical and error issues
        for issue in self.issues:
            if issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
                logger.error(f"{issue.severity.value.upper()}: {issue.message}")


class DataValidator:
    """Main data validation class"""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator
        
        Parameters:
        -----------
        strict_mode : bool
            If True, treats warnings as errors
        """
        self.strict_mode = strict_mode
        self.validation_rules = self._load_validation_rules()
    
    def validate_dataset(self, 
                        df: pd.DataFrame, 
                        dataset_type: str = "general",
                        dataset_name: str = "dataset") -> ValidationReport:
        """
        Perform comprehensive validation on a dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to validate
        dataset_type : str
            Type of dataset ('market', 'economic', 'sentiment', 'unified', 'general')
        dataset_name : str
            Name for the dataset (used in reporting)
            
        Returns:
        --------
        ValidationReport
            Complete validation report
        """
        logger.info(f"Starting validation for {dataset_name} ({dataset_type})")
        
        issues = []
        
        # Basic structure validation
        issues.extend(self._validate_basic_structure(df))
        
        # Index validation (datetime checks for time series)
        issues.extend(self._validate_index(df))
        
        # Column validation
        issues.extend(self._validate_columns(df, dataset_type))
        
        # Data type validation
        issues.extend(self._validate_data_types(df, dataset_type))
        
        # Value range validation
        issues.extend(self._validate_value_ranges(df, dataset_type))
        
        # Missing data validation
        issues.extend(self._validate_missing_data(df))
        
        # Duplicate validation
        issues.extend(self._validate_duplicates(df))
        
        # Outlier detection
        issues.extend(self._detect_outliers(df, dataset_type))
        
        # Time series specific validation
        if dataset_type in ['market', 'economic', 'unified']:
            issues.extend(self._validate_time_series(df, dataset_type))
        
        # Business logic validation
        issues.extend(self._validate_business_logic(df, dataset_type))
        
        # Calculate quality score and validity
        quality_score = self._calculate_quality_score(df, issues)
        is_valid = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        
        # In strict mode, treat warnings as errors
        if self.strict_mode:
            is_valid = is_valid and not any(issue.severity == ValidationSeverity.WARNING for issue in issues)
        
        report = ValidationReport(
            dataset_name=dataset_name,
            validation_timestamp=datetime.now(),
            total_rows=len(df),
            total_columns=len(df.columns),
            issues=issues,
            is_valid=is_valid,
            quality_score=quality_score
        )
        
        logger.info(f"Validation completed for {dataset_name}. Quality score: {quality_score:.1f}")
        
        return report
    
    def _validate_basic_structure(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate basic DataFrame structure"""
        issues = []
        
        # Check if DataFrame is empty
        if df.empty:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="structure",
                message="DataFrame is empty"
            ))
            return issues
        
        # Check minimum size requirements
        if len(df) < 10:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message=f"Dataset has only {len(df)} rows, which may be insufficient for analysis",
                row_count=len(df)
            ))
        
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="structure",
                message=f"Found {len(unnamed_cols)} unnamed columns",
                details={"unnamed_columns": unnamed_cols}
            ))
        
        return issues
    
    def _validate_index(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate DataFrame index"""
        issues = []
        
        # Check for datetime index (expected for time series data)
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="index",
                message=f"Index is not DatetimeIndex (found {type(df.index).__name__})"
            ))
        else:
            # Validate datetime index properties
            # Check for timezone info
            if df.index.tz is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="index",
                    message="DatetimeIndex has no timezone information"
                ))
            
            # Check for proper ordering
            if not df.index.is_monotonic_increasing:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="index",
                    message="DatetimeIndex is not properly sorted"
                ))
            
            # Check for reasonable date range
            min_date = df.index.min()
            max_date = df.index.max()
            
            if min_date < pd.Timestamp('1990-01-01'):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="index",
                    message=f"Start date seems too early: {min_date}"
                ))
            
            if max_date > pd.Timestamp.now() + timedelta(days=7):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="index",
                    message=f"End date is in the future: {max_date}"
                ))
        
        # Check for duplicate index values
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="index",
                message=f"Found {dup_count} duplicate index values",
                row_count=dup_count
            ))
        
        return issues
    
    def _validate_columns(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationIssue]:
        """Validate column presence and naming"""
        issues = []
        
        expected_columns = self.validation_rules.get(dataset_type, {}).get('required_columns', [])
        
        if expected_columns:
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="columns",
                    message=f"Missing required columns: {missing_cols}",
                    details={"missing_columns": missing_cols}
                ))
        
        # Check for suspicious column names
        suspicious_patterns = ['unnamed', 'untitled', 'column']
        suspicious_cols = []
        for col in df.columns:
            if any(pattern in str(col).lower() for pattern in suspicious_patterns):
                suspicious_cols.append(col)
        
        if suspicious_cols:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="columns",
                message=f"Found suspicious column names: {suspicious_cols}",
                details={"suspicious_columns": suspicious_cols}
            ))
        
        return issues
    
    def _validate_data_types(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationIssue]:
        """Validate data types for columns"""
        issues = []
        
        expected_types = self.validation_rules.get(dataset_type, {}).get('column_types', {})
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                
                # Check if numeric columns are actually numeric
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(actual_type):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="data_types",
                        message=f"Column '{col}' should be numeric but found {actual_type}",
                        column=col
                    ))
                
                # Check if string columns contain mostly non-string data
                elif expected_type == 'string' and pd.api.types.is_numeric_dtype(actual_type):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="data_types",
                        message=f"Column '{col}' expected to be string but found {actual_type}",
                        column=col
                    ))
        
        # Check for object columns that might need conversion
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            # Try to detect if object column should be numeric
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                try:
                    pd.to_numeric(non_null_values.iloc[:min(100, len(non_null_values))])
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        category="data_types",
                        message=f"Column '{col}' is object type but appears to be numeric",
                        column=col
                    ))
                except (ValueError, TypeError):
                    pass
        
        return issues
    
    def _validate_value_ranges(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationIssue]:
        """Validate value ranges for specific columns"""
        issues = []
        
        value_ranges = self.validation_rules.get(dataset_type, {}).get('value_ranges', {})
        
        for col, range_def in value_ranges.items():
            if col not in df.columns:
                continue
            
            min_val = range_def.get('min')
            max_val = range_def.get('max')
            
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            
            # Check minimum values
            if min_val is not None:
                violations = numeric_data < min_val
                if violations.any():
                    violation_count = violations.sum()
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="value_ranges",
                        message=f"Column '{col}' has {violation_count} values below minimum {min_val}",
                        column=col,
                        row_count=violation_count,
                        percentage=(violation_count / len(df)) * 100
                    ))
            
            # Check maximum values
            if max_val is not None:
                violations = numeric_data > max_val
                if violations.any():
                    violation_count = violations.sum()
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="value_ranges",
                        message=f"Column '{col}' has {violation_count} values above maximum {max_val}",
                        column=col,
                        row_count=violation_count,
                        percentage=(violation_count / len(df)) * 100
                    ))
        
        return issues
    
    def _validate_missing_data(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate missing data patterns"""
        issues = []
        
        # Check overall missing data percentage
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        if missing_percentage > 20:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="missing_data",
                message=f"High missing data percentage: {missing_percentage:.1f}%",
                percentage=missing_percentage
            ))
        
        # Check per-column missing data
        for col in df.columns:
            col_missing = df[col].isnull().sum()
            col_missing_pct = (col_missing / len(df)) * 100
            
            if col_missing_pct > 50:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="missing_data",
                    message=f"Column '{col}' has {col_missing_pct:.1f}% missing data",
                    column=col,
                    row_count=col_missing,
                    percentage=col_missing_pct
                ))
            elif col_missing_pct > 10:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="missing_data",
                    message=f"Column '{col}' has {col_missing_pct:.1f}% missing data",
                    column=col,
                    row_count=col_missing,
                    percentage=col_missing_pct
                ))
        
        # Check for rows with all missing data
        all_missing_rows = df.isnull().all(axis=1).sum()
        if all_missing_rows > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="missing_data",
                message=f"Found {all_missing_rows} rows with all missing data",
                row_count=all_missing_rows
            ))
        
        return issues
    
    def _validate_duplicates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate duplicate data"""
        issues = []
        
        # Check for completely duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="duplicates",
                message=f"Found {duplicate_rows} duplicate rows",
                row_count=duplicate_rows,
                percentage=(duplicate_rows / len(df)) * 100
            ))
        
        return issues
    
    def _detect_outliers(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationIssue]:
        """Detect statistical outliers in numeric columns"""
        issues = []
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns and df[col].notna().sum() > 10:  # Need sufficient data
                
                # Use IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:  # Avoid division by zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)) & df[col].notna()
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        outlier_percentage = (outlier_count / df[col].notna().sum()) * 100
                        
                        severity = ValidationSeverity.INFO
                        if outlier_percentage > 10:
                            severity = ValidationSeverity.WARNING
                        if outlier_percentage > 25:
                            severity = ValidationSeverity.ERROR
                        
                        issues.append(ValidationIssue(
                            severity=severity,
                            category="outliers",
                            message=f"Column '{col}' has {outlier_count} outliers ({outlier_percentage:.1f}%)",
                            column=col,
                            row_count=outlier_count,
                            percentage=outlier_percentage
                        ))
        
        return issues
    
    def _validate_time_series(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationIssue]:
        """Validate time series specific properties"""
        issues = []
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return issues
        
        # Check for gaps in time series
        if len(df) > 1:
            # Estimate expected frequency
            time_diffs = df.index.to_series().diff().dropna()
            most_common_diff = time_diffs.mode()
            
            if len(most_common_diff) > 0:
                expected_freq = most_common_diff.iloc[0]
                
                # Find gaps larger than expected
                large_gaps = time_diffs > expected_freq * 1.5  # Allow 50% tolerance
                gap_count = large_gaps.sum()
                
                if gap_count > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="time_series",
                        message=f"Found {gap_count} time gaps larger than expected frequency",
                        row_count=gap_count
                    ))
        
        # Check for weekend/holiday data (for daily market data)
        if dataset_type == 'market':
            weekend_data = df.index.dayofweek.isin([5, 6])  # Saturday=5, Sunday=6
            weekend_count = weekend_data.sum()
            
            if weekend_count > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="time_series",
                    message=f"Found {weekend_count} weekend data points",
                    row_count=weekend_count
                ))
        
        return issues
    
    def _validate_business_logic(self, df: pd.DataFrame, dataset_type: str) -> List[ValidationIssue]:
        """Validate business logic specific to dataset type"""
        issues = []
        
        if dataset_type == 'market':
            issues.extend(self._validate_market_data_logic(df))
        elif dataset_type == 'sentiment':
            issues.extend(self._validate_sentiment_data_logic(df))
        elif dataset_type == 'economic':
            issues.extend(self._validate_economic_data_logic(df))
        
        return issues
    
    def _validate_market_data_logic(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate market data specific business logic"""
        issues = []
        
        # OHLC validation
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            
            # High should be >= max(Open, Close)
            high_violations = df['High'] < df[['Open', 'Close']].max(axis=1)
            if high_violations.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"High price violations: {high_violations.sum()} cases where High < max(Open, Close)",
                    row_count=high_violations.sum()
                ))
            
            # Low should be <= min(Open, Close)
            low_violations = df['Low'] > df[['Open', 'Close']].min(axis=1)
            if low_violations.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Low price violations: {low_violations.sum()} cases where Low > min(Open, Close)",
                    row_count=low_violations.sum()
                ))
        
        # Volume validation
        if 'Volume' in df.columns:
            negative_volume = df['Volume'] < 0
            if negative_volume.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Negative volume: {negative_volume.sum()} cases",
                    column='Volume',
                    row_count=negative_volume.sum()
                ))
        
        # Price validation (should be positive)
        price_columns = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]
        for col in price_columns:
            negative_prices = df[col] <= 0
            if negative_prices.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Non-positive prices in {col}: {negative_prices.sum()} cases",
                    column=col,
                    row_count=negative_prices.sum()
                ))
        
        return issues
    
    def _validate_sentiment_data_logic(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate sentiment data specific business logic"""
        issues = []
        
        # Sentiment score should be in range [-1, 1] or [0, 1]
        if 'sentiment_score' in df.columns:
            score_col = df['sentiment_score']
            
            # Check range
            out_of_range = (score_col < -1) | (score_col > 1)
            if out_of_range.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Sentiment scores out of range [-1,1]: {out_of_range.sum()} cases",
                    column='sentiment_score',
                    row_count=out_of_range.sum()
                ))
        
        # Confidence should be in range [0, 1]
        if 'confidence' in df.columns:
            conf_col = df['confidence']
            
            out_of_range = (conf_col < 0) | (conf_col > 1)
            if out_of_range.any():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="business_logic",
                    message=f"Confidence scores out of range [0,1]: {out_of_range.sum()} cases",
                    column='confidence',
                    row_count=out_of_range.sum()
                ))
        
        return issues
    
    def _validate_economic_data_logic(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validate economic data specific business logic"""
        issues = []
        
        # Interest rates should be reasonable (0-50%)
        rate_columns = [col for col in df.columns if any(keyword in col.lower() 
                       for keyword in ['rate', 'yield', 'dff', 'dgs'])]
        
        for col in rate_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                unreasonable_rates = (df[col] < 0) | (df[col] > 50)
                if unreasonable_rates.any():
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="business_logic",
                        message=f"Unreasonable interest rates in {col}: {unreasonable_rates.sum()} cases",
                        column=col,
                        row_count=unreasonable_rates.sum()
                    ))
        
        return issues
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        """Calculate overall data quality score (0-100)"""
        base_score = 100.0
        
        # Deduct points based on issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                base_score -= 25
            elif issue.severity == ValidationSeverity.ERROR:
                base_score -= 10
            elif issue.severity == ValidationSeverity.WARNING:
                base_score -= 5
            elif issue.severity == ValidationSeverity.INFO:
                base_score -= 1
        
        # Adjust for missing data
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        base_score -= missing_percentage * 0.5
        
        return max(0.0, base_score)
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different dataset types"""
        return {
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
            },
            'sentiment': {
                'required_columns': ['sentiment_score'],
                'column_types': {
                    'sentiment_score': 'numeric',
                    'confidence': 'numeric'
                },
                'value_ranges': {
                    'sentiment_score': {'min': -1, 'max': 1},
                    'confidence': {'min': 0, 'max': 1}
                }
            },
            'economic': {
                'column_types': {
                    'DFF': 'numeric',
                    'DGS10': 'numeric',
                    'VIXCLS': 'numeric'
                },
                'value_ranges': {
                    'DFF': {'min': 0, 'max': 50},
                    'DGS10': {'min': 0, 'max': 50},
                    'VIXCLS': {'min': 0, 'max': 100}
                }
            }
        }


# Convenience functions

def validate_market_data(df: pd.DataFrame, dataset_name: str = "market_data") -> ValidationReport:
    """Quick validation for market data"""
    validator = DataValidator()
    return validator.validate_dataset(df, dataset_type="market", dataset_name=dataset_name)


def validate_sentiment_data(df: pd.DataFrame, dataset_name: str = "sentiment_data") -> ValidationReport:
    """Quick validation for sentiment data"""
    validator = DataValidator()
    return validator.validate_dataset(df, dataset_type="sentiment", dataset_name=dataset_name)


def validate_economic_data(df: pd.DataFrame, dataset_name: str = "economic_data") -> ValidationReport:
    """Quick validation for economic data"""
    validator = DataValidator()
    return validator.validate_dataset(df, dataset_type="economic", dataset_name=dataset_name)


def validate_unified_data(df: pd.DataFrame, dataset_name: str = "unified_dataset") -> ValidationReport:
    """Quick validation for unified dataset"""
    validator = DataValidator()
    return validator.validate_dataset(df, dataset_type="unified", dataset_name=dataset_name)


# Main execution
if __name__ == "__main__":
    # Example usage
    print("Data validation module loaded successfully!")
    print("Available validation functions:")
    print("- validate_market_data()")
    print("- validate_sentiment_data()")
    print("- validate_economic_data()")
    print("- validate_unified_data()")
    print("- DataValidator class for custom validation")