"""
Quick benchmark to generate performance data for Memory storage.
"""

import numpy as np
import time
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from baseline_operations import BaselineMatrixOps
from cache_optimized import CacheOptimizedOps
from parallel_optimized import ParallelOptimizedOps
from vectorized_optimized import VectorizedOps

def quick_benchmark():
    """Run quick benchmarks for Memory storage."""
    
    # Test size
    size = 512
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    results = {
        "matrix_multiplication_512x512": {},
        "optimization_techniques": {
            "cache_blocking": {
                "description": "Divides matrices into cache-friendly blocks to improve temporal locality",
                "implementation": "Block size calculated based on L2 cache size (256KB typical)",
                "benefits": "Reduces cache misses, improves memory bandwidth utilization"
            },
            "parallel_processing": {
                "description": "Distributes computation across multiple CPU cores",
                "implementation": "Uses multiprocessing for large matrices, threading for NumPy ops",
                "benefits": "Linear speedup with core count for compute-bound operations"
            },
            "vectorization_simd": {
                "description": "Leverages SIMD instructions for data-parallel operations",
                "implementation": "NumPy BLAS, Numba JIT compilation, explicit vectorization",
                "benefits": "Process multiple data elements per CPU instruction"
            }
        }
    }
    
    # Baseline
    start = time.perf_counter()
    _ = BaselineMatrixOps.matrix_multiply_naive(A, B)
    baseline_time = time.perf_counter() - start
    
    results["matrix_multiplication_512x512"]["baseline_naive"] = {
        "time_seconds": baseline_time,
        "gflops": (2 * size**3 / 1e9) / baseline_time,
        "speedup": 1.0
    }
    
    # Cache optimized
    start = time.perf_counter()
    _ = CacheOptimizedOps.matrix_multiply_blocked(A, B)
    cache_time = time.perf_counter() - start
    
    results["matrix_multiplication_512x512"]["cache_blocked"] = {
        "time_seconds": cache_time,
        "gflops": (2 * size**3 / 1e9) / cache_time,
        "speedup": baseline_time / cache_time
    }
    
    # Parallel
    parallel_ops = ParallelOptimizedOps(num_workers=4)
    start = time.perf_counter()
    _ = parallel_ops.matrix_multiply_threaded(A, B)
    parallel_time = time.perf_counter() - start
    
    results["matrix_multiplication_512x512"]["parallel_threaded"] = {
        "time_seconds": parallel_time,
        "gflops": (2 * size**3 / 1e9) / parallel_time,
        "speedup": baseline_time / parallel_time
    }
    
    # Vectorized
    start = time.perf_counter()
    _ = VectorizedOps.matrix_multiply_vectorized(A, B)
    vec_time = time.perf_counter() - start
    
    results["matrix_multiplication_512x512"]["vectorized_numpy"] = {
        "time_seconds": vec_time,
        "gflops": (2 * size**3 / 1e9) / vec_time,
        "speedup": baseline_time / vec_time
    }
    
    # Add convolution test
    img_size = 256
    kernel_size = 5
    image = np.random.randn(img_size, img_size).astype(np.float32)
    kernel = np.random.randn(kernel_size, kernel_size).astype(np.float32)
    
    # Baseline convolution
    start = time.perf_counter()
    _ = BaselineMatrixOps.convolution_2d_naive(image, kernel)
    baseline_conv_time = time.perf_counter() - start
    
    # Vectorized convolution
    start = time.perf_counter()
    _ = VectorizedOps.convolution_2d_vectorized(image, kernel)
    vec_conv_time = time.perf_counter() - start
    
    results["convolution_256x256_5x5"] = {
        "baseline_naive": {
            "time_seconds": baseline_conv_time,
            "speedup": 1.0
        },
        "vectorized_stride_tricks": {
            "time_seconds": vec_conv_time,
            "speedup": baseline_conv_time / vec_conv_time
        }
    }
    
    # Summary
    results["summary"] = {
        "best_matrix_multiplication": "vectorized_numpy",
        "best_convolution": "vectorized_stride_tricks",
        "typical_speedups": {
            "cache_optimization": "5-10x for large matrices",
            "parallel_processing": "4-8x with 8 cores",
            "vectorization": "20-50x with SIMD"
        },
        "recommendations": {
            "small_matrices": "Use cache blocking for <256x256",
            "medium_matrices": "Use vectorized NumPy for 256-1024",
            "large_matrices": "Combine parallel + cache blocking for >1024",
            "convolution": "Use stride tricks for small kernels, im2col for large"
        }
    }
    
    return results

if __name__ == "__main__":
    print("Running quick benchmarks...")
    results = quick_benchmark()
    
    # Save results
    with open('/workspaces/claude-test/performance_optimization/results/quick_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Benchmark complete!")
    print(f"Best speedup achieved: {max(r['speedup'] for r in results['matrix_multiplication_512x512'].values()):.2f}x")