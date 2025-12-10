"""
MLX Framework Benchmarks
========================

Comprehensive benchmarking suite for MLX framework on Apple Silicon.
"""

try:
    from .comprehensive_benchmark import (
        MLXBenchmarkSuite,
        run_comprehensive_mlx_benchmark
    )
    MLX_COMPREHENSIVE_AVAILABLE = True
except ImportError:
    MLX_COMPREHENSIVE_AVAILABLE = False

try:
    from .flash_attention_benchmark import (
        MLXFlashAttentionBenchmark,
        run_flash_attention_benchmark
    )
    MLX_FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    MLX_FLASH_ATTENTION_AVAILABLE = False

__all__ = []

if MLX_COMPREHENSIVE_AVAILABLE:
    __all__.extend([
        'MLXBenchmarkSuite',
        'run_comprehensive_mlx_benchmark'
    ])

if MLX_FLASH_ATTENTION_AVAILABLE:
    __all__.extend([
        'MLXFlashAttentionBenchmark', 
        'run_flash_attention_benchmark'
    ])