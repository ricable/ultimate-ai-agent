"""
Profiling and benchmarking utilities for matrix operations.
"""

import numpy as np
import time
import psutil
import os
import json
from typing import Dict, List, Callable, Any, Tuple
from dataclasses import dataclass, asdict
import cProfile
import pstats
import io
from memory_profiler import profile as memory_profile
import tracemalloc


@dataclass
class BenchmarkResult:
    """Store benchmark results for an operation."""
    operation_name: str
    input_size: str
    execution_time: float
    gflops: float
    memory_peak_mb: float
    memory_used_mb: float
    cache_misses: int = 0
    cpu_percent: float = 0.0
    speedup: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PerformanceProfiler:
    """Comprehensive performance profiling for matrix operations."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_times: Dict[str, float] = {}
        
    def time_operation(self, func: Callable, *args, warmup: int = 3, 
                      iterations: int = 10, **kwargs) -> Tuple[float, Any]:
        """
        Time an operation with warmup runs.
        Returns: (average_time, result)
        """
        # Warmup runs
        for _ in range(warmup):
            result = func(*args, **kwargs)
        
        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        return avg_time, result
    
    def profile_memory(self, func: Callable, *args, **kwargs) -> Tuple[float, float, Any]:
        """
        Profile memory usage of a function.
        Returns: (peak_memory_mb, used_memory_mb, result)
        """
        # Start memory tracking
        tracemalloc.start()
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        used_mb = current / 1024 / 1024
        
        return peak_mb, used_mb, result
    
    def profile_cpu(self, func: Callable, *args, **kwargs) -> Tuple[float, str, Any]:
        """
        Profile CPU usage and generate detailed stats.
        Returns: (cpu_percent, profile_stats, result)
        """
        # Start CPU monitoring
        process = psutil.Process(os.getpid())
        process.cpu_percent()  # Initialize
        
        # Profile the function
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Get CPU usage
        cpu_percent = process.cpu_percent()
        
        # Generate stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        profile_stats = s.getvalue()
        
        return cpu_percent, profile_stats, result
    
    def calculate_gflops(self, flops: int, time_seconds: float) -> float:
        """Calculate GFLOPS from operations and time."""
        return (flops / 1e9) / time_seconds if time_seconds > 0 else 0
    
    def benchmark_matrix_multiply(self, func: Callable, A: np.ndarray, 
                                 B: np.ndarray, name: str) -> BenchmarkResult:
        """Benchmark matrix multiplication operation."""
        m, k = A.shape
        k2, n = B.shape
        flops = 2 * m * k * n
        
        # Time the operation
        exec_time, result = self.time_operation(func, A, B)
        
        # Profile memory
        peak_mb, used_mb, _ = self.profile_memory(func, A, B)
        
        # Profile CPU
        cpu_percent, _, _ = self.profile_cpu(func, A, B)
        
        # Calculate performance metrics
        gflops = self.calculate_gflops(flops, exec_time)
        
        # Calculate speedup if baseline exists
        baseline_key = f"matmul_{m}x{k}x{n}"
        if baseline_key in self.baseline_times:
            speedup = self.baseline_times[baseline_key] / exec_time
        else:
            speedup = 1.0
            self.baseline_times[baseline_key] = exec_time
        
        result = BenchmarkResult(
            operation_name=name,
            input_size=f"{m}x{k}x{n}",
            execution_time=exec_time,
            gflops=gflops,
            memory_peak_mb=peak_mb,
            memory_used_mb=used_mb,
            cpu_percent=cpu_percent,
            speedup=speedup
        )
        
        self.results.append(result)
        return result
    
    def benchmark_convolution(self, func: Callable, image: np.ndarray, 
                             kernel: np.ndarray, name: str) -> BenchmarkResult:
        """Benchmark convolution operation."""
        img_h, img_w = image.shape
        ker_h, ker_w = kernel.shape
        out_h = img_h - ker_h + 1
        out_w = img_w - ker_w + 1
        flops = 2 * out_h * out_w * ker_h * ker_w
        
        # Time the operation
        exec_time, result = self.time_operation(func, image, kernel)
        
        # Profile memory
        peak_mb, used_mb, _ = self.profile_memory(func, image, kernel)
        
        # Profile CPU
        cpu_percent, _, _ = self.profile_cpu(func, image, kernel)
        
        # Calculate performance metrics
        gflops = self.calculate_gflops(flops, exec_time)
        
        # Calculate speedup if baseline exists
        baseline_key = f"conv_{img_h}x{img_w}_{ker_h}x{ker_w}"
        if baseline_key in self.baseline_times:
            speedup = self.baseline_times[baseline_key] / exec_time
        else:
            speedup = 1.0
            self.baseline_times[baseline_key] = exec_time
        
        result = BenchmarkResult(
            operation_name=name,
            input_size=f"img:{img_h}x{img_w},ker:{ker_h}x{ker_w}",
            execution_time=exec_time,
            gflops=gflops,
            memory_peak_mb=peak_mb,
            memory_used_mb=used_mb,
            cpu_percent=cpu_percent,
            speedup=speedup
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file."""
        results_dict = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "python_version": os.sys.version
            },
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Group by operation type
        by_operation = {}
        for result in self.results:
            op_type = result.operation_name.split('_')[0]
            if op_type not in by_operation:
                by_operation[op_type] = []
            by_operation[op_type].append(result)
        
        for op_type, results in by_operation.items():
            print(f"\n{op_type.upper()} Operations:")
            print("-" * 70)
            print(f"{'Method':<25} {'Size':<20} {'Time(s)':<10} {'GFLOPS':<10} {'Speedup':<10}")
            print("-" * 70)
            
            for r in sorted(results, key=lambda x: x.speedup, reverse=True):
                print(f"{r.operation_name:<25} {r.input_size:<20} "
                      f"{r.execution_time:<10.4f} {r.gflops:<10.2f} "
                      f"{r.speedup:<10.2f}x")


class CacheProfiler:
    """Profile cache performance using perf counters (Linux only)."""
    
    @staticmethod
    def profile_cache_misses(func: Callable, *args, **kwargs) -> Dict[str, int]:
        """
        Profile cache misses using perf (requires Linux and perf tools).
        This is a simplified version - real implementation would use perf_event_open.
        """
        # Simplified cache miss estimation based on memory access patterns
        # In production, use perf_event_open or Intel VTune
        return {
            "L1_cache_misses": 0,
            "L2_cache_misses": 0,
            "L3_cache_misses": 0,
            "TLB_misses": 0
        }


def analyze_memory_access_pattern(func_name: str, matrix_size: int) -> Dict[str, Any]:
    """Analyze theoretical memory access patterns for different algorithms."""
    patterns = {
        "naive": {
            "description": "Poor spatial locality, random access pattern",
            "cache_efficiency": "Low",
            "memory_touches": matrix_size ** 3,
            "reuse_distance": "High"
        },
        "blocked": {
            "description": "Good spatial and temporal locality within blocks",
            "cache_efficiency": "High",
            "memory_touches": matrix_size ** 3 / 8,  # Approximate
            "reuse_distance": "Low"
        },
        "vectorized": {
            "description": "Sequential access, SIMD-friendly",
            "cache_efficiency": "Very High",
            "memory_touches": matrix_size ** 2,
            "reuse_distance": "Minimal"
        }
    }
    
    return patterns.get(func_name, patterns["naive"])


if __name__ == "__main__":
    # Test the profiler
    print("Testing Performance Profiler")
    profiler = PerformanceProfiler()
    
    # Simple test function
    def test_func(n):
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        return np.dot(A, B)
    
    # Profile the test function
    exec_time, result = profiler.time_operation(test_func, 100)
    print(f"Execution time: {exec_time:.4f} seconds")
    
    peak_mb, used_mb, _ = profiler.profile_memory(test_func, 100)
    print(f"Peak memory: {peak_mb:.2f} MB, Used: {used_mb:.2f} MB")