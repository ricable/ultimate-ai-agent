"""
Flow2 Performance Benchmark Suite
=================================

Comprehensive benchmarking tools for MLX, LlamaCpp, and HuggingFace frameworks.
"""

from .comprehensive_framework_benchmark import (
    FrameworkBenchmark,
    BenchmarkResult,
    run_comprehensive_framework_benchmark
)

# Import framework-specific benchmarks
try:
    from .mlx import *
except ImportError:
    pass

try:
    from .huggingface import *
except ImportError:
    pass

try:
    from .llama_cpp import *
except ImportError:
    pass

try:
    from .models_8b import *
except ImportError:
    pass

__all__ = [
    'FrameworkBenchmark',
    'BenchmarkResult', 
    'run_comprehensive_framework_benchmark'
]