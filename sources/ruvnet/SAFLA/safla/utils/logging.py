"""
SAFLA Logging Utilities

This module provides centralized logging configuration and utilities for the SAFLA system.
"""

import logging
import sys
from typing import Optional
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rich_logging: bool = True,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging configuration for SAFLA.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        rich_logging: Whether to use rich formatting for console output
        format_string: Custom format string for log messages
    """
    # Install rich traceback handler for better error formatting
    if rich_logging:
        install(show_locals=True)
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if rich_logging:
        console = Console()
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            enable_link_path=True,
            markup=True
        )
        console_handler.setLevel(numeric_level)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class SAFLALogger:
    """
    Enhanced logger class with SAFLA-specific functionality.
    """
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._component_name = name.split('.')[-1] if '.' in name else name
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with component context."""
        self.logger.debug(f"[{self._component_name}] {message}", **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with component context."""
        self.logger.info(f"[{self._component_name}] {message}", **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with component context."""
        self.logger.warning(f"[{self._component_name}] {message}", **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with component context."""
        self.logger.error(f"[{self._component_name}] {message}", **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with component context."""
        self.logger.critical(f"[{self._component_name}] {message}", **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback and component context."""
        self.logger.exception(f"[{self._component_name}] {message}", **kwargs)
    
    def memory_operation(self, operation: str, details: str) -> None:
        """Log memory-related operations."""
        self.info(f"Memory {operation}: {details}")
    
    def safety_event(self, event_type: str, details: str, severity: str = "INFO") -> None:
        """Log safety-related events."""
        log_method = getattr(self, severity.lower(), self.info)
        log_method(f"Safety {event_type}: {details}")
    
    def mcp_operation(self, operation: str, server: str, details: str) -> None:
        """Log MCP-related operations."""
        self.info(f"MCP {operation} [{server}]: {details}")
    
    def performance_metric(self, metric_name: str, value: float, unit: str = "") -> None:
        """Log performance metrics."""
        unit_str = f" {unit}" if unit else ""
        self.info(f"Performance {metric_name}: {value}{unit_str}")
    
    def delta_evaluation(self, delta_type: str, value: float, threshold: float) -> None:
        """Log delta evaluation results."""
        status = "ABOVE" if value >= threshold else "BELOW"
        self.info(f"Delta {delta_type}: {value:.4f} ({status} threshold {threshold:.4f})")


# Performance logging utilities
class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: SAFLALogger):
        self.logger = logger
        self._timers = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        import time
        self._timers[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and log the duration."""
        import time
        if operation not in self._timers:
            self.logger.warning(f"Timer not found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self._timers[operation]
        del self._timers[operation]
        
        self.logger.performance_metric(f"{operation}_duration", duration, "seconds")
        return duration
    
    def log_memory_usage(self) -> None:
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.performance_metric("memory_rss", memory_info.rss / 1024 / 1024, "MB")
            self.logger.performance_metric("memory_vms", memory_info.vms / 1024 / 1024, "MB")
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
    
    def log_cpu_usage(self) -> None:
        """Log current CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            self.logger.performance_metric("cpu_usage", cpu_percent, "%")
        except ImportError:
            self.logger.warning("psutil not available for CPU monitoring")


# Context manager for operation logging
class LoggedOperation:
    """Context manager for logging operations with timing."""
    
    def __init__(self, logger: SAFLALogger, operation: str, log_performance: bool = True):
        self.logger = logger
        self.operation = operation
        self.log_performance = log_performance
        self.perf_logger = PerformanceLogger(logger) if log_performance else None
    
    def __enter__(self):
        self.logger.info(f"Starting operation: {self.operation}")
        if self.perf_logger:
            self.perf_logger.start_timer(self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.info(f"Completed operation: {self.operation}")
            if self.perf_logger:
                duration = self.perf_logger.end_timer(self.operation)
                self.logger.info(f"Operation {self.operation} took {duration:.3f} seconds")
        else:
            self.logger.error(f"Failed operation: {self.operation} - {exc_val}")
            if self.perf_logger and self.operation in self.perf_logger._timers:
                del self.perf_logger._timers[self.operation]


# Factory function for creating SAFLA loggers
def create_safla_logger(name: str) -> SAFLALogger:
    """Create a SAFLA logger instance."""
    return SAFLALogger(name)