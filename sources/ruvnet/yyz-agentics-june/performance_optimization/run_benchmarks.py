"""
Comprehensive benchmarking script for all matrix operation optimizations.
"""

import numpy as np
import time
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all optimization modules
from baseline_operations import BaselineMatrixOps, generate_test_data, generate_conv_test_data
from profiling_utils import PerformanceProfiler, BenchmarkResult
from cache_optimized import CacheOptimizedOps
from parallel_optimized import ParallelOptimizedOps
from vectorized_optimized import VectorizedOps

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ComprehensiveBenchmark:
    """Run comprehensive benchmarks on all optimization techniques."""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.results = {}
        self.parallel_ops = ParallelOptimizedOps()
        
        # Pre-compile Numba functions
        self._warmup_numba()
    
    def _warmup_numba(self):
        """Warm up Numba JIT compilation."""
        print("Warming up Numba JIT compiler...")
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)
        _ = VectorizedOps.matrix_multiply_numba(A, B)
        
        img = np.random.randn(64, 64).astype(np.float32)
        ker = np.random.randn(5, 5).astype(np.float32)
        _ = VectorizedOps.convolution_2d_numba(img, ker)
    
    def benchmark_matrix_multiplication(self, sizes: List[int]):
        """Benchmark all matrix multiplication implementations."""
        print("\n" + "="*80)
        print("MATRIX MULTIPLICATION BENCHMARKS")
        print("="*80)
        
        results = {}
        
        for size in sizes:
            print(f"\nBenchmarking {size}x{size} matrices...")
            A, B = generate_test_data(size)
            
            # Baseline
            print("  - Baseline naive...")
            baseline_result = self.profiler.benchmark_matrix_multiply(
                BaselineMatrixOps.matrix_multiply_naive, A, B, 
                f"baseline_naive_{size}"
            )
            
            # Cache optimized
            print("  - Cache optimized (blocked)...")
            cache_result = self.profiler.benchmark_matrix_multiply(
                CacheOptimizedOps.matrix_multiply_blocked, A, B,
                f"cache_blocked_{size}"
            )
            
            # Parallel
            print("  - Parallel (multiprocess)...")
            mp_result = self.profiler.benchmark_matrix_multiply(
                self.parallel_ops.matrix_multiply_multiprocess, A, B,
                f"parallel_multiprocess_{size}"
            )
            
            print("  - Parallel (threaded)...")
            thread_result = self.profiler.benchmark_matrix_multiply(
                self.parallel_ops.matrix_multiply_threaded, A, B,
                f"parallel_threaded_{size}"
            )
            
            # Vectorized
            print("  - Vectorized (NumPy)...")
            numpy_result = self.profiler.benchmark_matrix_multiply(
                VectorizedOps.matrix_multiply_vectorized, A, B,
                f"vectorized_numpy_{size}"
            )
            
            print("  - Vectorized (Numba JIT)...")
            numba_result = self.profiler.benchmark_matrix_multiply(
                VectorizedOps.matrix_multiply_numba, A, B,
                f"vectorized_numba_{size}"
            )
            
            # Store results
            results[size] = {
                'baseline': baseline_result,
                'cache_blocked': cache_result,
                'parallel_mp': mp_result,
                'parallel_thread': thread_result,
                'vectorized_numpy': numpy_result,
                'vectorized_numba': numba_result
            }
        
        self.results['matrix_multiplication'] = results
        return results
    
    def benchmark_convolution(self, img_sizes: List[int], kernel_sizes: List[int]):
        """Benchmark all convolution implementations."""
        print("\n" + "="*80)
        print("2D CONVOLUTION BENCHMARKS")
        print("="*80)
        
        results = {}
        
        for img_size in img_sizes:
            for ker_size in kernel_sizes:
                key = f"{img_size}x{ker_size}"
                print(f"\nBenchmarking {img_size}x{img_size} image, {ker_size}x{ker_size} kernel...")
                image, kernel = generate_conv_test_data(img_size, ker_size)
                
                # Baseline
                print("  - Baseline naive...")
                baseline_result = self.profiler.benchmark_convolution(
                    BaselineMatrixOps.convolution_2d_naive, image, kernel,
                    f"baseline_conv_{key}"
                )
                
                # Cache optimized
                print("  - Cache optimized (tiled)...")
                cache_result = self.profiler.benchmark_convolution(
                    CacheOptimizedOps.convolution_2d_tiled, image, kernel,
                    f"cache_tiled_{key}"
                )
                
                print("  - Cache optimized (im2col)...")
                im2col_result = self.profiler.benchmark_convolution(
                    CacheOptimizedOps.convolution_2d_im2col, image, kernel,
                    f"cache_im2col_{key}"
                )
                
                # Parallel
                print("  - Parallel convolution...")
                parallel_result = self.profiler.benchmark_convolution(
                    self.parallel_ops.convolution_2d_parallel, image, kernel,
                    f"parallel_conv_{key}"
                )
                
                # Vectorized
                print("  - Vectorized (stride tricks)...")
                vec_result = self.profiler.benchmark_convolution(
                    VectorizedOps.convolution_2d_vectorized, image, kernel,
                    f"vectorized_stride_{key}"
                )
                
                print("  - Vectorized (Numba)...")
                numba_result = self.profiler.benchmark_convolution(
                    VectorizedOps.convolution_2d_numba, image, kernel,
                    f"vectorized_numba_{key}"
                )
                
                # Store results
                results[key] = {
                    'baseline': baseline_result,
                    'cache_tiled': cache_result,
                    'cache_im2col': im2col_result,
                    'parallel': parallel_result,
                    'vectorized_stride': vec_result,
                    'vectorized_numba': numba_result
                }
        
        self.results['convolution'] = results
        return results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'cpu_count': os.cpu_count(),
                'numpy_version': np.__version__,
            },
            'optimization_techniques': {
                'cache_optimization': {
                    'description': 'Blocking, tiling, and memory access pattern optimization',
                    'techniques': ['Loop blocking', 'Cache-aware tiling', 'Im2col transformation']
                },
                'parallel_processing': {
                    'description': 'Multi-core parallelization',
                    'techniques': ['Multiprocessing', 'Threading', 'Shared memory']
                },
                'vectorization': {
                    'description': 'SIMD and vectorized operations',
                    'techniques': ['NumPy BLAS', 'Numba JIT', 'Stride tricks', 'Einstein summation']
                }
            },
            'results_summary': self._summarize_results(),
            'detailed_results': self.results
        }
        
        return report
    
    def _summarize_results(self) -> Dict:
        """Summarize benchmark results with key insights."""
        summary = {}
        
        # Matrix multiplication summary
        if 'matrix_multiplication' in self.results:
            mm_summary = {}
            for size, implementations in self.results['matrix_multiplication'].items():
                baseline_time = implementations['baseline'].execution_time
                
                speedups = {
                    name: baseline_time / impl.execution_time
                    for name, impl in implementations.items()
                }
                
                best_impl = max(speedups.items(), key=lambda x: x[1])
                
                mm_summary[f"{size}x{size}"] = {
                    'best_implementation': best_impl[0],
                    'best_speedup': f"{best_impl[1]:.2f}x",
                    'baseline_gflops': f"{implementations['baseline'].gflops:.2f}",
                    'best_gflops': f"{implementations[best_impl[0]].gflops:.2f}",
                    'all_speedups': {k: f"{v:.2f}x" for k, v in speedups.items()}
                }
            
            summary['matrix_multiplication'] = mm_summary
        
        # Convolution summary
        if 'convolution' in self.results:
            conv_summary = {}
            for key, implementations in self.results['convolution'].items():
                baseline_time = implementations['baseline'].execution_time
                
                speedups = {
                    name: baseline_time / impl.execution_time
                    for name, impl in implementations.items()
                }
                
                best_impl = max(speedups.items(), key=lambda x: x[1])
                
                conv_summary[key] = {
                    'best_implementation': best_impl[0],
                    'best_speedup': f"{best_impl[1]:.2f}x",
                    'baseline_gflops': f"{implementations['baseline'].gflops:.2f}",
                    'best_gflops': f"{implementations[best_impl[0]].gflops:.2f}",
                    'all_speedups': {k: f"{v:.2f}x" for k, v in speedups.items()}
                }
            
            summary['convolution'] = conv_summary
        
        return summary
    
    def plot_results(self, output_dir: str):
        """Generate performance plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Matrix multiplication speedup plot
        if 'matrix_multiplication' in self.results:
            self._plot_matrix_multiplication_speedups(output_dir)
        
        # Convolution speedup plot
        if 'convolution' in self.results:
            self._plot_convolution_speedups(output_dir)
    
    def _plot_matrix_multiplication_speedups(self, output_dir: str):
        """Plot matrix multiplication speedups."""
        sizes = []
        implementations = {}
        
        for size, impls in self.results['matrix_multiplication'].items():
            sizes.append(size)
            baseline_time = impls['baseline'].execution_time
            
            for name, impl in impls.items():
                if name not in implementations:
                    implementations[name] = []
                speedup = baseline_time / impl.execution_time
                implementations[name].append(speedup)
        
        plt.figure(figsize=(10, 6))
        
        for name, speedups in implementations.items():
            plt.plot(sizes, speedups, marker='o', label=name)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Speedup vs Baseline')
        plt.title('Matrix Multiplication Optimization Speedups')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'matrix_multiplication_speedups.png'))
        plt.close()
    
    def _plot_convolution_speedups(self, output_dir: str):
        """Plot convolution speedups."""
        labels = []
        implementations = {}
        
        for key, impls in self.results['convolution'].items():
            labels.append(key)
            baseline_time = impls['baseline'].execution_time
            
            for name, impl in impls.items():
                if name not in implementations:
                    implementations[name] = []
                speedup = baseline_time / impl.execution_time
                implementations[name].append(speedup)
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(labels))
        width = 0.15
        
        for i, (name, speedups) in enumerate(implementations.items()):
            plt.bar(x + i * width, speedups, width, label=name)
        
        plt.xlabel('Image Size x Kernel Size')
        plt.ylabel('Speedup vs Baseline')
        plt.title('2D Convolution Optimization Speedups')
        plt.xticks(x + width * 2.5, labels)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convolution_speedups.png'))
        plt.close()


def main():
    """Run comprehensive benchmarks."""
    print("Performance Optimization Benchmark Suite")
    print("=" * 80)
    print(f"Started at: {datetime.now()}")
    print(f"NumPy version: {np.__version__}")
    print(f"CPU count: {os.cpu_count()}")
    print("=" * 80)
    
    # Create benchmark instance
    benchmark = ComprehensiveBenchmark()
    
    # Run matrix multiplication benchmarks
    matrix_sizes = [128, 256, 512, 1024]
    benchmark.benchmark_matrix_multiplication(matrix_sizes)
    
    # Run convolution benchmarks
    img_sizes = [256, 512]
    kernel_sizes = [3, 5, 7]
    benchmark.benchmark_convolution(img_sizes, kernel_sizes)
    
    # Generate report
    report = benchmark.generate_performance_report()
    
    # Save detailed results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(results_dir, f'benchmark_report_{timestamp}.json')
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Generate plots
    benchmark.plot_results(results_dir)
    print(f"Performance plots saved to: {results_dir}")
    
    # Print summary
    benchmark.profiler.print_summary()
    
    # Print optimization recommendations
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    summary = report['results_summary']
    
    print("\nMatrix Multiplication:")
    for size, info in summary.get('matrix_multiplication', {}).items():
        print(f"  {size}: Use {info['best_implementation']} for {info['best_speedup']} speedup")
    
    print("\n2D Convolution:")
    for config, info in summary.get('convolution', {}).items():
        print(f"  {config}: Use {info['best_implementation']} for {info['best_speedup']} speedup")
    
    print("\nGeneral Recommendations:")
    print("  1. For small matrices (<256x256): Cache optimization provides best results")
    print("  2. For medium matrices (256-512): Vectorized NumPy/BLAS is optimal")
    print("  3. For large matrices (>512): Parallel processing shows significant gains")
    print("  4. For convolution: Vectorized stride tricks excel for small kernels")
    print("  5. Always align data to cache boundaries for optimal performance")
    
    return report


if __name__ == "__main__":
    report = main()