"""
SAFLA Benchmarking Suite
========================

This package provides a comprehensive benchmarking framework for the SAFLA CLI,
including performance measurement, result storage, and optimization tracking.

The benchmarking suite follows a modular architecture with:
- Core benchmark framework
- SQLite database for result storage
- CLI integration for easy execution
- Extensible benchmark definitions
- Performance optimization tracking
"""

__version__ = "1.0.0"
__author__ = "SAFLA Development Team"

from .core import Benchmark, BenchmarkRunner, BenchmarkResult, BenchmarkSuite, BenchmarkMetrics
from .database import BenchmarkDatabase
from .cli_benchmarks import CLIBenchmarkSuite
from .utils import BenchmarkConfig, PerformanceMetrics, BenchmarkAnalyzer

__all__ = [
    "Benchmark",
    "BenchmarkRunner",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkMetrics",
    "BenchmarkDatabase",
    "CLIBenchmarkSuite",
    "BenchmarkConfig",
    "PerformanceMetrics",
    "BenchmarkAnalyzer"
]