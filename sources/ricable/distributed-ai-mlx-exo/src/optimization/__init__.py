"""
Performance Optimization Package for MLX Distributed System
Contains network, memory, compute, and profiling optimizations
"""

from .network_optimizer import (
    NetworkOptimizer,
    CompressionEngine,
    ThunderboltOptimizer,
    TCPOptimizer,
    AsyncIOManager,
    NetworkMetrics,
    NodeConnection
)

from .memory_optimizer import (
    MemoryOptimizer,
    QuantizationEngine,
    ActivationCache,
    MemoryPoolManager,
    GCOptimizer,
    QuantizationLevel,
    MemoryStats
)

from .compute_optimizer import (
    ComputeOptimizer,
    AppleSiliconOptimizer,
    GPUKernelOptimizer,
    BatchProcessor,
    PipelineOptimizer,
    ComputeStats,
    OptimizationLevel,
    ComputeDevice
)

from .profiler import (
    PerformanceProfiler,
    MetricsCollector,
    BottleneckDetector,
    AutoOptimizer,
    PerformanceMetric,
    BottleneckDetection,
    PerformanceAlert,
    BottleneckType,
    AlertSeverity
)

__all__ = [
    # Network optimization
    'NetworkOptimizer',
    'CompressionEngine', 
    'ThunderboltOptimizer',
    'TCPOptimizer',
    'AsyncIOManager',
    'NetworkMetrics',
    'NodeConnection',
    
    # Memory optimization
    'MemoryOptimizer',
    'QuantizationEngine',
    'ActivationCache',
    'MemoryPoolManager',
    'GCOptimizer',
    'QuantizationLevel',
    'MemoryStats',
    
    # Compute optimization
    'ComputeOptimizer',
    'AppleSiliconOptimizer',
    'GPUKernelOptimizer',
    'BatchProcessor',
    'PipelineOptimizer',
    'ComputeStats',
    'OptimizationLevel',
    'ComputeDevice',
    
    # Performance profiling
    'PerformanceProfiler',
    'MetricsCollector',
    'BottleneckDetector',
    'AutoOptimizer',
    'PerformanceMetric',
    'BottleneckDetection',
    'PerformanceAlert',
    'BottleneckType',
    'AlertSeverity'
]

# Version info
__version__ = "1.0.0"
__author__ = "MLX Distributed System Team"
__description__ = "Performance optimization suite for MLX distributed AI/ML systems"