"""
Centralized Logging System for StockSage.AI

This module provides comprehensive logging capabilities including:
- Structured logging with context
- Performance monitoring and timing
- Error tracking and alerts
- Data pipeline logging
- Model training/inference logging
- API request logging
"""

import logging
import logging.handlers
import os
import sys
import json
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Union, Callable
from functools import wraps
from pathlib import Path
import threading
from contextlib import contextmanager

# Configuration from main config
try:
    from ..config import AppConfig, ROOT_DIR
except ImportError:
    # Fallback if config is not available
    ROOT_DIR = Path(__file__).parent.parent.parent
    class AppConfig:
        LOG_LEVEL = "INFO"
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ENVIRONMENT = "development"
        DEBUG = False


class StockSageLogger:
    """Enhanced logger with structured logging and performance monitoring"""
    
    def __init__(self, name: str, level: str = None):
        """
        Initialize logger with enhanced capabilities
        
        Parameters:
        -----------
        name : str
            Logger name (usually module name)
        level : str, optional
            Log level override
        """
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set level
        level = level or AppConfig.LOG_LEVEL
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Threading lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Performance tracking
        self._timers = {}
        self._operation_stats = {}
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler for all logs
        log_dir = ROOT_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "stocksage.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "stocksage_errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        
        # Formatters
        if AppConfig.ENVIRONMENT == "production":
            # JSON formatter for production
            formatter = JSONFormatter()
        else:
            # Human readable for development
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with optional context"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with optional context"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Internal method to log with context"""
        
        # Extract context info
        context = {}
        
        # Add standard context
        context['timestamp'] = datetime.now(timezone.utc).isoformat()
        context['module'] = self.name
        
        # Add custom context
        for key, value in kwargs.items():
            if key not in ['exc_info']:
                context[key] = value
        
        # Format message with context
        if context and AppConfig.ENVIRONMENT == "production":
            # For production, log as JSON
            log_data = {
                'message': message,
                'context': context,
                'level': logging.getLevelName(level)
            }
            formatted_message = json.dumps(log_data)
        else:
            # For development, append context to message
            if context:
                context_str = " | ".join([f"{k}={v}" for k, v in context.items() if k != 'timestamp'])
                formatted_message = f"{message} | {context_str}"
            else:
                formatted_message = message
        
        # Log the message
        self.logger.log(level, formatted_message, exc_info=kwargs.get('exc_info', False))
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._record_timing(operation_name, duration)
            self.info(f"Operation '{operation_name}' completed", 
                     operation=operation_name, duration_seconds=round(duration, 4))
    
    def _record_timing(self, operation: str, duration: float):
        """Record timing statistics"""
        with self._lock:
            if operation not in self._operation_stats:
                self._operation_stats[operation] = {
                    'count': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0
                }
            
            stats = self._operation_stats[operation]
            stats['count'] += 1
            stats['total_time'] += duration
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            stats = {}
            for operation, data in self._operation_stats.items():
                if data['count'] > 0:
                    stats[operation] = {
                        'count': data['count'],
                        'total_time': round(data['total_time'], 4),
                        'avg_time': round(data['total_time'] / data['count'], 4),
                        'min_time': round(data['min_time'], 4),
                        'max_time': round(data['max_time'], 4)
                    }
            return stats
    
    def log_data_pipeline_event(self, event_type: str, dataset: str, **kwargs):
        """Log data pipeline specific events"""
        self.info(f"Data pipeline event: {event_type}", 
                 event_type=event_type, 
                 dataset=dataset, 
                 pipeline_stage="data",
                 **kwargs)
    
    def log_model_event(self, event_type: str, model_name: str, **kwargs):
        """Log model training/inference events"""
        self.info(f"Model event: {event_type}", 
                 event_type=event_type, 
                 model_name=model_name, 
                 pipeline_stage="model",
                 **kwargs)
    
    def log_api_request(self, method: str, endpoint: str, status_code: int, duration: float, **kwargs):
        """Log API request"""
        self.info(f"API request: {method} {endpoint}", 
                 method=method, 
                 endpoint=endpoint, 
                 status_code=status_code, 
                 duration_seconds=round(duration, 4),
                 request_type="api",
                 **kwargs)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry)


# Decorator for automatic function logging
def log_function_call(logger: StockSageLogger = None, level: str = "info"):
    """Decorator to automatically log function calls"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            func_logger = logger or get_logger(func.__module__)
            
            # Log function entry
            func_logger._log_with_context(
                getattr(logging, level.upper()),
                f"Entering function '{func.__name__}'",
                function=func.__name__,
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                with func_logger.timer(f"function_{func.__name__}"):
                    result = func(*args, **kwargs)
                
                # Log successful completion
                func_logger._log_with_context(
                    getattr(logging, level.upper()),
                    f"Function '{func.__name__}' completed successfully",
                    function=func.__name__
                )
                
                return result
                
            except Exception as e:
                # Log exception
                func_logger.exception(
                    f"Function '{func.__name__}' failed",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


# Decorator for data validation logging
def log_data_operation(operation_type: str):
    """Decorator for logging data operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            with logger.timer(f"data_operation_{operation_type}"):
                logger.log_data_pipeline_event(
                    event_type=f"{operation_type}_start",
                    dataset=kwargs.get('dataset_name', 'unknown'),
                    function=func.__name__
                )
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log success with result info
                    result_info = {}
                    if hasattr(result, 'shape'):
                        result_info['shape'] = str(result.shape)
                    elif hasattr(result, '__len__'):
                        result_info['length'] = len(result)
                    
                    logger.log_data_pipeline_event(
                        event_type=f"{operation_type}_success",
                        dataset=kwargs.get('dataset_name', 'unknown'),
                        function=func.__name__,
                        **result_info
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.log_data_pipeline_event(
                        event_type=f"{operation_type}_error",
                        dataset=kwargs.get('dataset_name', 'unknown'),
                        function=func.__name__,
                        error_type=type(e).__name__,
                        error_message=str(e)
                    )
                    raise
        
        return wrapper
    return decorator


# Global logger registry
_loggers: Dict[str, StockSageLogger] = {}
_lock = threading.Lock()


def get_logger(name: str) -> StockSageLogger:
    """Get or create a logger instance"""
    with _lock:
        if name not in _loggers:
            _loggers[name] = StockSageLogger(name)
        return _loggers[name]


def setup_logging(level: str = None, log_dir: Path = None):
    """Setup global logging configuration"""
    level = level or AppConfig.LOG_LEVEL
    log_dir = log_dir or (ROOT_DIR / "logs")
    
    # Create log directory
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup new handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "stocksage_root.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def log_system_info():
    """Log system information on startup"""
    logger = get_logger("system")
    
    logger.info("StockSage.AI logging system initialized")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log level: {AppConfig.LOG_LEVEL}")
    logger.info(f"Environment: {AppConfig.ENVIRONMENT}")


def get_all_performance_stats() -> Dict[str, Any]:
    """Get performance statistics from all loggers"""
    all_stats = {}
    with _lock:
        for name, logger in _loggers.items():
            stats = logger.get_performance_stats()
            if stats:
                all_stats[name] = stats
    return all_stats


# Context manager for request tracking
@contextmanager
def request_context(request_id: str, user_id: str = None, operation: str = None):
    """Context manager for tracking requests across multiple operations"""
    logger = get_logger("request_tracking")
    
    context = {
        'request_id': request_id,
        'user_id': user_id,
        'operation': operation
    }
    
    logger.info("Request started", **context)
    start_time = time.time()
    
    try:
        yield context
        duration = time.time() - start_time
        logger.info("Request completed successfully", 
                   duration_seconds=round(duration, 4), **context)
    except Exception as e:
        duration = time.time() - start_time
        logger.error("Request failed", 
                    duration_seconds=round(duration, 4),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    **context)
        raise


# Alert functions for critical issues
def send_alert(message: str, severity: str = "warning", **kwargs):
    """Send alert for critical issues (placeholder for future implementation)"""
    logger = get_logger("alerts")
    
    alert_data = {
        'alert_message': message,
        'severity': severity,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    # Log the alert
    if severity == "critical":
        logger.critical(f"ALERT: {message}", **alert_data)
    elif severity == "error":
        logger.error(f"ALERT: {message}", **alert_data)
    else:
        logger.warning(f"ALERT: {message}", **alert_data)
    
    # TODO: Implement actual alerting mechanism
    # (email, Slack, PagerDuty, etc.)


# Initialize logging on module import
if __name__ != "__main__":
    setup_logging()
    log_system_info()


if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    
    logger.info("Testing logging system")
    
    with logger.timer("test_operation"):
        time.sleep(0.1)
    
    logger.log_data_pipeline_event("data_load", "test_dataset", rows=1000)
    logger.log_model_event("training_start", "lstm_model", epochs=100)
    
    try:
        raise ValueError("Test error")
    except ValueError:
        logger.exception("Caught test exception")
    
    print("Performance stats:", logger.get_performance_stats())