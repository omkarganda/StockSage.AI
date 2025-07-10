#!/usr/bin/env python3
"""
Data Quality Dashboard for StockSage.AI

This script provides a comprehensive dashboard for monitoring data quality
and validation results over time. It can be used to track data quality trends,
identify recurring issues, and generate quality reports.

Features:
- Data quality monitoring
- Historical quality tracking
- Validation report aggregation
- Quality metrics visualization
- Automated quality alerts
- Data quality recommendations
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.validation import (
    DataValidator, 
    ValidationSeverity,
    ValidationReport,
    validate_market_data,
    validate_sentiment_data,
    validate_economic_data,
    validate_unified_data
)
from src.utils.logging import get_logger, setup_logging


class DataQualityDashboard:
    """Data Quality Dashboard for monitoring and reporting"""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize the dashboard
        
        Parameters:
        -----------
        output_dir : Path, optional
            Directory to save reports and metrics
        """
        self.logger = get_logger(__name__)
        self.output_dir = output_dir or (Path(__file__).parent.parent / "reports" / "data_quality")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Historical data storage
        self.quality_history_file = self.output_dir / "quality_history.json"
        self.quality_history = self._load_quality_history()
        
        self.logger.info(f"Data Quality Dashboard initialized. Output dir: {self.output_dir}")
    
    def _load_quality_history(self) -> List[Dict]:
        """Load historical quality data"""
        if self.quality_history_file.exists():
            try:
                with open(self.quality_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load quality history: {e}")
        return []
    
    def _save_quality_history(self):
        """Save quality history to file"""
        try:
            with open(self.quality_history_file, 'w') as f:
                json.dump(self.quality_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Could not save quality history: {e}")
    
    def record_validation_results(self, 
                                 reports: List[ValidationReport],
                                 dataset_category: str = "general") -> Dict:
        """
        Record validation results for historical tracking
        
        Parameters:
        -----------
        reports : List[ValidationReport]
            List of validation reports to record
        dataset_category : str
            Category of dataset (e.g., 'daily_pipeline', 'batch_process')
            
        Returns:
        --------
        dict
            Summary of recorded results
        """
        timestamp = datetime.now().isoformat()
        
        # Aggregate metrics from all reports
        total_datasets = len(reports)
        total_issues = sum(len(report.issues) for report in reports)
        critical_issues = sum(len(report.get_issues_by_severity(ValidationSeverity.CRITICAL)) 
                             for report in reports)
        error_issues = sum(len(report.get_issues_by_severity(ValidationSeverity.ERROR)) 
                          for report in reports)
        warning_issues = sum(len(report.get_issues_by_severity(ValidationSeverity.WARNING)) 
                            for report in reports)
        
        avg_quality_score = np.mean([report.quality_score for report in reports]) if reports else 100.0
        datasets_passed = sum(1 for report in reports if report.is_valid)
        pass_rate = (datasets_passed / total_datasets * 100) if total_datasets > 0 else 100.0
        
        # Create summary record
        summary = {
            'timestamp': timestamp,
            'dataset_category': dataset_category,
            'total_datasets': total_datasets,
            'datasets_passed': datasets_passed,
            'pass_rate': pass_rate,
            'avg_quality_score': avg_quality_score,
            'total_issues': total_issues,
            'critical_issues': critical_issues,
            'error_issues': error_issues,
            'warning_issues': warning_issues,
            'dataset_details': []
        }
        
        # Add individual dataset details
        for report in reports:
            detail = {
                'dataset_name': report.dataset_name,
                'total_rows': report.total_rows,
                'total_columns': report.total_columns,
                'is_valid': report.is_valid,
                'quality_score': report.quality_score,
                'issue_counts': {
                    'critical': len(report.get_issues_by_severity(ValidationSeverity.CRITICAL)),
                    'error': len(report.get_issues_by_severity(ValidationSeverity.ERROR)),
                    'warning': len(report.get_issues_by_severity(ValidationSeverity.WARNING)),
                    'info': len(report.get_issues_by_severity(ValidationSeverity.INFO))
                },
                'issues': [
                    {
                        'severity': issue.severity.value,
                        'category': issue.category,
                        'message': issue.message,
                        'column': issue.column,
                        'row_count': issue.row_count,
                        'percentage': issue.percentage
                    }
                    for issue in report.issues
                ]
            }
            summary['dataset_details'].append(detail)
        
        # Add to history
        self.quality_history.append(summary)
        self._save_quality_history()
        
        self.logger.info(f"Recorded validation results: {total_datasets} datasets, "
                        f"{pass_rate:.1f}% pass rate, avg quality: {avg_quality_score:.1f}")
        
        return summary
    
    def generate_quality_trends_report(self, days: int = 30) -> Dict:
        """
        Generate quality trends report for the last N days
        
        Parameters:
        -----------
        days : int
            Number of days to analyze
            
        Returns:
        --------
        dict
            Quality trends report
        """
        self.logger.info(f"Generating quality trends report for last {days} days")
        
        # Filter recent history
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            record for record in self.quality_history
            if datetime.fromisoformat(record['timestamp']) >= cutoff_date
        ]
        
        if not recent_history:
            return {
                'message': 'No quality data available for the specified period',
                'days_analyzed': days,
                'records_found': 0
            }
        
        # Calculate trends
        df = pd.DataFrame(recent_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Overall trends
        trends = {
            'analysis_period': {
                'days': days,
                'start_date': df['timestamp'].min().isoformat(),
                'end_date': df['timestamp'].max().isoformat(),
                'total_records': len(df)
            },
            'overall_metrics': {
                'avg_pass_rate': df['pass_rate'].mean(),
                'avg_quality_score': df['avg_quality_score'].mean(),
                'total_datasets_processed': df['total_datasets'].sum(),
                'total_issues_found': df['total_issues'].sum()
            },
            'trends': {
                'pass_rate_trend': self._calculate_trend(df['pass_rate'].values),
                'quality_score_trend': self._calculate_trend(df['avg_quality_score'].values),
                'issue_count_trend': self._calculate_trend(df['total_issues'].values)
            },
            'issue_breakdown': {
                'critical_issues': df['critical_issues'].sum(),
                'error_issues': df['error_issues'].sum(),
                'warning_issues': df['warning_issues'].sum(),
                'avg_critical_per_run': df['critical_issues'].mean(),
                'avg_error_per_run': df['error_issues'].mean(),
                'avg_warning_per_run': df['warning_issues'].mean()
            }
        }
        
        # Category-specific trends
        if 'dataset_category' in df.columns:
            category_trends = {}
            for category in df['dataset_category'].unique():
                cat_data = df[df['dataset_category'] == category]
                category_trends[category] = {
                    'avg_pass_rate': cat_data['pass_rate'].mean(),
                    'avg_quality_score': cat_data['avg_quality_score'].mean(),
                    'total_runs': len(cat_data),
                    'pass_rate_trend': self._calculate_trend(cat_data['pass_rate'].values)
                }
            trends['category_trends'] = category_trends
        
        return trends
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def generate_data_quality_report(self, 
                                   include_trends: bool = True,
                                   include_recommendations: bool = True) -> str:
        """
        Generate comprehensive data quality report
        
        Parameters:
        -----------
        include_trends : bool
            Whether to include trend analysis
        include_recommendations : bool
            Whether to include quality recommendations
            
        Returns:
        --------
        str
            HTML report content
        """
        self.logger.info("Generating comprehensive data quality report")
        
        # Get recent data
        recent_data = self.quality_history[-10:] if self.quality_history else []
        
        # Start building HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>StockSage.AI Data Quality Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f8ff; padding: 20px; border-radius: 5px; }
                .metric-card { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .error { color: #d32f2f; }
                .warning { color: #f57c00; }
                .success { color: #388e3c; }
                .table { border-collapse: collapse; width: 100%; }
                .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .table th { background-color: #f2f2f2; }
                .trend-up { color: #4caf50; }
                .trend-down { color: #f44336; }
                .trend-stable { color: #666; }
            </style>
        </head>
        <body>
        """
        
        # Header
        html += f"""
        <div class="header">
            <h1>🔍 StockSage.AI Data Quality Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Historical Records:</strong> {len(self.quality_history)}</p>
        </div>
        """
        
        if recent_data:
            # Latest run summary
            latest = recent_data[-1]
            html += f"""
            <h2>📊 Latest Quality Check Results</h2>
            <div class="metric-card">
                <h3>Run Details</h3>
                <p><strong>Timestamp:</strong> {latest['timestamp']}</p>
                <p><strong>Category:</strong> {latest['dataset_category']}</p>
                <p><strong>Datasets Processed:</strong> {latest['total_datasets']}</p>
                <p><strong>Pass Rate:</strong> <span class="{'success' if latest['pass_rate'] >= 90 else 'warning' if latest['pass_rate'] >= 70 else 'error'}">{latest['pass_rate']:.1f}%</span></p>
                <p><strong>Average Quality Score:</strong> <span class="{'success' if latest['avg_quality_score'] >= 80 else 'warning' if latest['avg_quality_score'] >= 60 else 'error'}">{latest['avg_quality_score']:.1f}/100</span></p>
            </div>
            
            <div class="metric-card">
                <h3>Issue Summary</h3>
                <p><strong>Total Issues:</strong> {latest['total_issues']}</p>
                <p><strong>Critical:</strong> <span class="error">{latest['critical_issues']}</span></p>
                <p><strong>Errors:</strong> <span class="error">{latest['error_issues']}</span></p>
                <p><strong>Warnings:</strong> <span class="warning">{latest['warning_issues']}</span></p>
            </div>
            """
        
        # Trends analysis
        if include_trends and len(self.quality_history) >= 2:
            trends = self.generate_quality_trends_report(days=30)
            
            html += f"""
            <h2>📈 Quality Trends (Last 30 Days)</h2>
            <div class="metric-card">
                <h3>Overall Trends</h3>
                <p><strong>Average Pass Rate:</strong> {trends['overall_metrics']['avg_pass_rate']:.1f}%</p>
                <p><strong>Average Quality Score:</strong> {trends['overall_metrics']['avg_quality_score']:.1f}/100</p>
                <p><strong>Pass Rate Trend:</strong> <span class="trend-{trends['trends']['pass_rate_trend'].replace('_', '-')}">{trends['trends']['pass_rate_trend'].replace('_', ' ').title()}</span></p>
                <p><strong>Quality Score Trend:</strong> <span class="trend-{trends['trends']['quality_score_trend'].replace('_', '-')}">{trends['trends']['quality_score_trend'].replace('_', ' ').title()}</span></p>
            </div>
            """
        
        # Recent dataset details
        if recent_data:
            html += """
            <h2>📋 Recent Dataset Details</h2>
            <table class="table">
                <tr>
                    <th>Dataset</th>
                    <th>Rows</th>
                    <th>Columns</th>
                    <th>Quality Score</th>
                    <th>Status</th>
                    <th>Issues</th>
                </tr>
            """
            
            for record in recent_data[-5:]:  # Last 5 runs
                for dataset in record['dataset_details']:
                    status_class = 'success' if dataset['is_valid'] else 'error'
                    total_issues = sum(dataset['issue_counts'].values())
                    
                    html += f"""
                    <tr>
                        <td>{dataset['dataset_name']}</td>
                        <td>{dataset['total_rows']}</td>
                        <td>{dataset['total_columns']}</td>
                        <td>{dataset['quality_score']:.1f}</td>
                        <td><span class="{status_class}">{'✓ Valid' if dataset['is_valid'] else '✗ Invalid'}</span></td>
                        <td>{total_issues}</td>
                    </tr>
                    """
            
            html += "</table>"
        
        # Recommendations
        if include_recommendations:
            recommendations = self._generate_recommendations()
            if recommendations:
                html += """
                <h2>💡 Quality Improvement Recommendations</h2>
                <div class="metric-card">
                    <ul>
                """
                for rec in recommendations:
                    html += f"<li>{rec}</li>"
                
                html += """
                    </ul>
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        # Save report
        report_file = self.output_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(report_file, 'w') as f:
            f.write(html)
        
        self.logger.info(f"Quality report saved to: {report_file}")
        
        return html
    
    def _generate_recommendations(self) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        
        if not self.quality_history:
            return recommendations
        
        recent_data = self.quality_history[-10:] if len(self.quality_history) >= 10 else self.quality_history
        
        # Analyze common issues
        all_issues = []
        for record in recent_data:
            for dataset in record['dataset_details']:
                all_issues.extend(dataset['issues'])
        
        if not all_issues:
            recommendations.append("🎉 No issues found in recent data quality checks!")
            return recommendations
        
        # Issue frequency analysis
        issue_categories = {}
        for issue in all_issues:
            category = issue['category']
            issue_categories[category] = issue_categories.get(category, 0) + 1
        
        # Generate recommendations based on common issues
        if issue_categories.get('missing_data', 0) > 5:
            recommendations.append("Consider implementing more robust data collection to reduce missing values")
        
        if issue_categories.get('outliers', 0) > 3:
            recommendations.append("Implement outlier detection and handling in data preprocessing pipeline")
        
        if issue_categories.get('business_logic', 0) > 2:
            recommendations.append("Review and strengthen business logic validation rules")
        
        if issue_categories.get('data_types', 0) > 2:
            recommendations.append("Implement automatic data type validation and conversion")
        
        # Check pass rates
        avg_pass_rate = np.mean([record['pass_rate'] for record in recent_data])
        if avg_pass_rate < 80:
            recommendations.append("Overall pass rate is below 80% - consider reviewing validation thresholds")
        
        # Check quality scores
        avg_quality = np.mean([record['avg_quality_score'] for record in recent_data])
        if avg_quality < 70:
            recommendations.append("Average quality score is below 70 - implement stricter data quality controls")
        
        return recommendations
    
    def check_quality_alerts(self, thresholds: Dict = None) -> List[Dict]:
        """
        Check for quality issues that require alerts
        
        Parameters:
        -----------
        thresholds : dict, optional
            Custom alert thresholds
            
        Returns:
        --------
        list
            List of alerts
        """
        if not thresholds:
            thresholds = {
                'min_pass_rate': 80.0,
                'min_quality_score': 70.0,
                'max_critical_issues': 0,
                'max_error_issues': 5
            }
        
        alerts = []
        
        if not self.quality_history:
            return alerts
        
        latest = self.quality_history[-1]
        
        # Check pass rate
        if latest['pass_rate'] < thresholds['min_pass_rate']:
            alerts.append({
                'severity': 'warning',
                'type': 'low_pass_rate',
                'message': f"Pass rate ({latest['pass_rate']:.1f}%) below threshold ({thresholds['min_pass_rate']}%)",
                'timestamp': latest['timestamp']
            })
        
        # Check quality score
        if latest['avg_quality_score'] < thresholds['min_quality_score']:
            alerts.append({
                'severity': 'warning',
                'type': 'low_quality_score',
                'message': f"Quality score ({latest['avg_quality_score']:.1f}) below threshold ({thresholds['min_quality_score']})",
                'timestamp': latest['timestamp']
            })
        
        # Check critical issues
        if latest['critical_issues'] > thresholds['max_critical_issues']:
            alerts.append({
                'severity': 'critical',
                'type': 'critical_issues',
                'message': f"Found {latest['critical_issues']} critical issues (max allowed: {thresholds['max_critical_issues']})",
                'timestamp': latest['timestamp']
            })
        
        # Check error issues
        if latest['error_issues'] > thresholds['max_error_issues']:
            alerts.append({
                'severity': 'error',
                'type': 'error_issues',
                'message': f"Found {latest['error_issues']} error issues (max allowed: {thresholds['max_error_issues']})",
                'timestamp': latest['timestamp']
            })
        
        return alerts


def main():
    """Main CLI interface for the data quality dashboard"""
    parser = argparse.ArgumentParser(description="StockSage.AI Data Quality Dashboard")
    parser.add_argument('--action', choices=['report', 'trends', 'demo'], default='demo',
                       help='Action to perform')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days for trend analysis')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)
    
    # Initialize dashboard
    output_dir = Path(args.output_dir) if args.output_dir else None
    dashboard = DataQualityDashboard(output_dir=output_dir)
    
    if args.action == 'report':
        # Generate quality report
        html_report = dashboard.generate_data_quality_report()
        logger.info("Quality report generated successfully")
        
    elif args.action == 'trends':
        # Generate trends report
        trends = dashboard.generate_quality_trends_report(days=args.days)
        
        logger.info(f"Quality trends for last {args.days} days:")
        logger.info(f"Average pass rate: {trends.get('overall_metrics', {}).get('avg_pass_rate', 'N/A')}")
        logger.info(f"Average quality score: {trends.get('overall_metrics', {}).get('avg_quality_score', 'N/A')}")
        
    elif args.action == 'demo':
        # Run demo with sample data
        logger.info("Running data quality dashboard demo...")
        
        # Import test validation script functions
        from test_validation import (
            create_sample_market_data,
            create_sample_sentiment_data,
            create_sample_economic_data
        )
        
        # Create sample validation reports
        sample_reports = []
        
        # Market data
        market_data = create_sample_market_data("AAPL", days=30, introduce_errors=False)
        market_report = validate_market_data(market_data, "AAPL_demo_market")
        sample_reports.append(market_report)
        
        # Sentiment data  
        sentiment_data = create_sample_sentiment_data("AAPL", days=30, introduce_errors=False)
        sentiment_report = validate_sentiment_data(sentiment_data, "AAPL_demo_sentiment")
        sample_reports.append(sentiment_report)
        
        # Economic data
        econ_data = create_sample_economic_data("DFF", days=30, introduce_errors=False)
        econ_report = validate_economic_data(econ_data, "DFF_demo")
        sample_reports.append(econ_report)
        
        # Record results
        summary = dashboard.record_validation_results(sample_reports, "demo_run")
        
        # Generate report
        html_report = dashboard.generate_data_quality_report()
        
        # Check for alerts
        alerts = dashboard.check_quality_alerts()
        
        logger.info(f"Demo completed successfully!")
        logger.info(f"Recorded {len(sample_reports)} validation reports")
        logger.info(f"Quality alerts: {len(alerts)}")
        
        if alerts:
            for alert in alerts:
                logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")


if __name__ == "__main__":
    main()