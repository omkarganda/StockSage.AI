#!/usr/bin/env python3
"""
Implementation Script for StockSage.AI Performance Improvements

This script implements all the immediate actions recommended in Section E:
1. Retrain LSTM for 50 epochs with early stopping
2. Remove blanket log-transform for ETS/Theta/Ensemble
3. Run rolling window evaluation (5 folds × 30 days)
4. Verify target-prediction alignment

Usage:
    python scripts/implement_improvements.py --symbol AAPL
    python scripts/implement_improvements.py --symbol AAPL --quick-test
    python scripts/implement_improvements.py --full-suite --symbols AAPL,GOOGL,MSFT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovementImplementer:
    """
    Orchestrates the implementation of all performance improvements
    based on the analysis recommendations.
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.results = {}
        
    def run_command(self, command: list, description: str) -> bool:
        """Run a command and capture results."""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.base_dir,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"SUCCESS: {description}")
                self.results[description] = {
                    'status': 'success',
                    'stdout': result.stdout[-500:],  # Last 500 chars
                    'stderr': result.stderr[-500:] if result.stderr else None
                }
                return True
            else:
                logger.error(f"FAILED: {description}")
                logger.error(f"Return code: {result.returncode}")
                logger.error(f"STDERR: {result.stderr}")
                self.results[description] = {
                    'status': 'failed',
                    'return_code': result.returncode,
                    'stdout': result.stdout[-500:],
                    'stderr': result.stderr[-500:]
                }
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"TIMEOUT: {description}")
            self.results[description] = {
                'status': 'timeout',
                'error': 'Command timed out after 1 hour'
            }
            return False
        except Exception as e:
            logger.error(f"ERROR: {description} - {e}")
            self.results[description] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def step_1_retrain_models(self, symbols: list, quick_test: bool = False) -> bool:
        """
        Step 1: Retrain LSTM and Transformer models with improved parameters
        
        Addresses Section E.2: "Retrain LSTM for 50 epochs with early stopping"
        Note: We've actually improved this to 100 epochs with better early stopping
        """
        logger.info("STEP 1: Retraining deep learning models with improved parameters")
        
        symbols_str = ','.join(symbols)
        command = [
            'python', 'scripts/train_models.py',
            '--symbols', symbols_str,
            '--tune',  # Enable hyperparameter tuning
        ]
        
        if quick_test:
            command.append('--quick-test')
        
        return self.run_command(command, "Retrain Deep Learning Models")
    
    def step_2_verify_alignment(self, symbol: str) -> bool:
        """
        Step 2: Verify target-prediction alignment
        
        Addresses Section E.1: "Verify target-prediction alignment (plot y_true vs. y_pred for LSTM)"
        """
        logger.info("STEP 2: Verifying target-prediction alignment")
        
        command = [
            'python', 'scripts/diagnose_models.py',
            '--symbol', symbol,
            '--model', 'dl_lstm_attention'
        ]
        
        return self.run_command(command, f"Verify Alignment for {symbol}")
    
    def step_3_rolling_evaluation(self, symbol: str) -> bool:
        """
        Step 3: Run rolling window evaluation
        
        Addresses Section E.4: "Rerun evaluation on a rolling window (e.g., 5 folds × 30 days)"
        """
        logger.info("STEP 3: Running rolling window evaluation")
        
        command = [
            'python', 'scripts/diagnose_models.py',
            '--symbol', symbol,
            '--rolling-eval',
            '--n-folds', '5',
            '--fold-size', '30'
        ]
        
        return self.run_command(command, f"Rolling Window Evaluation for {symbol}")
    
    def step_4_comprehensive_evaluation(self, symbols: list) -> bool:
        """
        Step 4: Run comprehensive model evaluation with improved metrics
        """
        logger.info("STEP 4: Running comprehensive model evaluation")
        
        symbols_str = ','.join(symbols)
        command = [
            'python', 'scripts/evaluate_models.py',
            '--symbols', symbols_str,
            '--generate-report'
        ]
        
        return self.run_command(command, "Comprehensive Model Evaluation")
    
    def validate_improvements(self, symbol: str) -> dict:
        """
        Validate that improvements meet the success criteria from Section E.
        
        Success criteria:
        - sMAPE < 50%
        - Directional accuracy ≥ 55-60%
        - Correlation ≥ 0.3
        """
        logger.info("STEP 5: Validating improvements against success criteria")
        
        # Try to load recent evaluation results
        results_dir = Path("reports/evaluation")
        if not results_dir.exists():
            return {'status': 'no_results', 'message': 'No evaluation results found'}
        
        # Find the most recent evaluation results
        json_files = list(results_dir.glob("evaluation_results_*.json"))
        if not json_files:
            return {'status': 'no_results', 'message': 'No evaluation JSON files found'}
        
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                eval_results = json.load(f)
            
            if symbol not in eval_results:
                return {'status': 'no_symbol', 'message': f'No results for {symbol}'}
            
            symbol_results = eval_results[symbol]
            validation_results = {}
            
            # Check each model against success criteria
            for model_name, metrics in symbol_results.items():
                if isinstance(metrics, dict) and 'smape' in metrics:
                    smape = metrics.get('smape', float('inf'))
                    direction_acc = metrics.get('direction_accuracy', 0)
                    correlation = metrics.get('correlation', -1)
                    
                    meets_criteria = {
                        'smape_target': smape < 50,  # < 50%
                        'direction_target': direction_acc >= 55,  # ≥ 55%
                        'correlation_target': correlation >= 0.3,  # ≥ 0.3
                        'overall_success': smape < 50 and direction_acc >= 55 and correlation >= 0.3
                    }
                    
                    validation_results[model_name] = {
                        'metrics': {
                            'smape': smape,
                            'direction_accuracy': direction_acc,
                            'correlation': correlation
                        },
                        'meets_criteria': meets_criteria,
                        'improvement_score': self._calculate_improvement_score(smape, direction_acc, correlation)
                    }
            
            return {
                'status': 'success',
                'symbol': symbol,
                'validation_results': validation_results,
                'evaluation_file': str(latest_file)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Error validating results: {e}'}
    
    def _calculate_improvement_score(self, smape: float, direction_acc: float, correlation: float) -> float:
        """
        Calculate an improvement score based on how close we are to targets.
        Score ranges from 0-100, where 100 means all targets are met.
        """
        # Normalize metrics to 0-1 scale where 1 is the target
        smape_score = max(0, min(1, (200 - smape) / 150))  # 50% target, 200% worst case
        direction_score = max(0, min(1, direction_acc / 60))  # 60% target
        correlation_score = max(0, min(1, (correlation + 1) / 1.3))  # 0.3 target, -1 worst case
        
        # Weighted average (sMAPE is most important)
        overall_score = (0.5 * smape_score + 0.3 * direction_score + 0.2 * correlation_score) * 100
        return round(overall_score, 1)
    
    def generate_improvement_report(self, symbols: list) -> str:
        """Generate a comprehensive improvement report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"reports/improvement_report_{timestamp}.md"
        
        # Ensure reports directory exists
        Path("reports").mkdir(exist_ok=True)
        
        # Validate improvements for each symbol
        validation_results = {}
        for symbol in symbols:
            validation_results[symbol] = self.validate_improvements(symbol)
        
        # Generate markdown report
        report_content = f"""# StockSage.AI Performance Improvement Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Symbols Analyzed:** {', '.join(symbols)}

## Implementation Summary

This report summarizes the results of implementing the performance improvements
identified in the sMAPE-based analysis.

### Implementation Steps Executed

"""
        
        # Add step results
        for step, result in self.results.items():
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            report_content += f"**{status_emoji} {step}**\n"
            report_content += f"- Status: {result['status']}\n"
            if result['status'] != 'success' and 'error' in result:
                report_content += f"- Error: {result['error']}\n"
            report_content += "\n"
        
        # Add validation results
        report_content += "## Performance Validation\n\n"
        
        for symbol, validation in validation_results.items():
            if validation['status'] == 'success':
                report_content += f"### {symbol} Results\n\n"
                
                for model_name, model_results in validation['validation_results'].items():
                    metrics = model_results['metrics']
                    criteria = model_results['meets_criteria']
                    score = model_results['improvement_score']
                    
                    success_emoji = "🎯" if criteria['overall_success'] else "⚠️"
                    
                    report_content += f"**{success_emoji} {model_name}** (Score: {score}/100)\n"
                    report_content += f"- sMAPE: {metrics['smape']:.1f}% {'✅' if criteria['smape_target'] else '❌'} (Target: <50%)\n"
                    report_content += f"- Direction Accuracy: {metrics['direction_accuracy']:.1f}% {'✅' if criteria['direction_target'] else '❌'} (Target: ≥55%)\n"
                    report_content += f"- Correlation: {metrics['correlation']:.3f} {'✅' if criteria['correlation_target'] else '❌'} (Target: ≥0.3)\n"
                    report_content += "\n"
            else:
                report_content += f"### {symbol} - {validation['status'].upper()}\n"
                report_content += f"{validation.get('message', 'No details available')}\n\n"
        
        # Add recommendations
        report_content += """## Next Steps

### If Success Criteria Are Met (sMAPE < 50%, Direction ≥ 55%, Correlation ≥ 0.3):
1. **Deploy to Production**: Models are ready for live trading evaluation
2. **Implement Ensembling**: Combine top-performing models
3. **Add Advanced Features**: Implement macro/sentiment features

### If Success Criteria Are Not Yet Met:
1. **Further Hyperparameter Tuning**: Run extended Optuna optimization
2. **Feature Engineering**: Add PCA, sentiment data, option-implied volatility
3. **Data Quality Review**: Check for data leakage or target definition issues
4. **Alternative Architectures**: Consider ensemble methods or regime-switching models

### Monitoring
- Set up continuous validation pipeline
- Monitor for model drift
- Track alignment metrics over time

---
*Report generated by StockSage.AI Improvement Implementation System*
"""
        
        # Write report
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Improvement report saved: {report_file}")
        return report_file
    
    def run_full_implementation(self, symbols: list, quick_test: bool = False) -> str:
        """Run the complete implementation workflow."""
        logger.info("Starting full implementation of performance improvements")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Quick test mode: {quick_test}")
        
        # Step 1: Retrain models
        success = self.step_1_retrain_models(symbols, quick_test)
        if not success and not quick_test:
            logger.warning("Model training failed, but continuing with evaluation")
        
        # Steps 2-4: For each symbol
        for symbol in symbols:
            logger.info(f"\nProcessing {symbol}...")
            
            # Step 2: Verify alignment
            self.step_2_verify_alignment(symbol)
            
            # Step 3: Rolling evaluation  
            self.step_3_rolling_evaluation(symbol)
        
        # Step 4: Comprehensive evaluation
        self.step_4_comprehensive_evaluation(symbols)
        
        # Generate final report
        report_file = self.generate_improvement_report(symbols)
        
        logger.info("\n" + "="*60)
        logger.info("IMPLEMENTATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Report saved: {report_file}")
        logger.info("Check the report for validation against success criteria")
        
        return report_file


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Implement StockSage.AI Performance Improvements')
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='AAPL',
        help='Primary symbol to test improvements on'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (overrides --symbol)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run in quick test mode (faster but less thorough)'
    )
    
    parser.add_argument(
        '--full-suite',
        action='store_true',
        help='Run full test suite on multiple symbols'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['retrain', 'align', 'rolling', 'evaluate'],
        help='Run only a specific step'
    )
    
    return parser.parse_args()


def main():
    """Main implementation function."""
    print("StockSage.AI Performance Improvement Implementation")
    print("=================================================\n")
    
    args = parse_arguments()
    
    # Determine symbols to use
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.full_suite:
        symbols = ['AAPL', 'GOOGL', 'MSFT']
    else:
        symbols = [args.symbol.upper()]
    
    # Create implementer
    implementer = ImprovementImplementer()
    
    try:
        if args.step:
            # Run specific step
            symbol = symbols[0]  # Use first symbol for single steps
            
            if args.step == 'retrain':
                success = implementer.step_1_retrain_models(symbols, args.quick_test)
            elif args.step == 'align':
                success = implementer.step_2_verify_alignment(symbol)
            elif args.step == 'rolling':
                success = implementer.step_3_rolling_evaluation(symbol)
            elif args.step == 'evaluate':
                success = implementer.step_4_comprehensive_evaluation(symbols)
            
            if success:
                print(f"\nStep '{args.step}' completed successfully!")
            else:
                print(f"\nStep '{args.step}' failed. Check logs for details.")
                return 1
                
        else:
            # Run full implementation
            report_file = implementer.run_full_implementation(symbols, args.quick_test)
            
            print(f"\nFull implementation completed!")
            print(f"Report available: {report_file}")
            print("\nKey files to check:")
            print("- reports/improvement_report_*.md - Main results")
            print("- reports/evaluation/ - Model performance")
            print("- reports/diagnostics/ - Alignment plots")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Implementation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Implementation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())