import unittest
import numpy as np
import time
import json
import os
from typing import Dict, List, Callable, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd


class PerformanceBenchmark:
    """Base class for performance benchmarking."""
    
    def __init__(self, name: str, warmup_runs: int = 5, benchmark_runs: int = 20):
        """
        Initialize benchmark.
        
        Args:
            name: Benchmark name
            warmup_runs: Number of warmup iterations
            benchmark_runs: Number of benchmark iterations
        """
        self.name = name
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results = []
        
    def time_function(self, func: Callable, *args, **kwargs) -> Tuple[float, any]:
        """
        Time a function execution.
        
        Returns:
            Tuple of (execution_time, function_result)
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        return end_time - start_time, result
        
    def run_benchmark(self, func: Callable, *args, **kwargs) -> Dict:
        """
        Run benchmark for a function.
        
        Returns:
            Dictionary with benchmark statistics
        """
        # Warmup runs
        for _ in range(self.warmup_runs):
            func(*args, **kwargs)
            
        # Benchmark runs
        times = []
        for _ in range(self.benchmark_runs):
            exec_time, _ = self.time_function(func, *args, **kwargs)
            times.append(exec_time)
            
        # Calculate statistics
        times = np.array(times)
        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
        }
        
        return stats


class TestLayerPerformance(unittest.TestCase):
    """Benchmark performance of individual layers."""
    
    def setUp(self):
        """Set up benchmarks."""
        self.benchmark = PerformanceBenchmark("Layer Performance")
        
    def test_dense_layer_performance(self):
        """Benchmark dense layer operations."""
        configurations = [
            {'batch_size': 1, 'input_dim': 128, 'output_dim': 64},
            {'batch_size': 32, 'input_dim': 128, 'output_dim': 64},
            {'batch_size': 128, 'input_dim': 512, 'output_dim': 256},
            {'batch_size': 256, 'input_dim': 1024, 'output_dim': 512},
        ]
        
        results = {}
        
        for config in configurations:
            # Setup
            batch_size = config['batch_size']
            input_dim = config['input_dim']
            output_dim = config['output_dim']
            
            X = np.random.randn(batch_size, input_dim).astype(np.float32)
            W = np.random.randn(input_dim, output_dim).astype(np.float32)
            b = np.random.randn(output_dim).astype(np.float32)
            
            # Forward pass benchmark
            def forward():
                return X @ W + b
                
            # Backward pass benchmark
            grad_output = np.random.randn(batch_size, output_dim).astype(np.float32)
            
            def backward():
                dW = X.T @ grad_output
                db = np.sum(grad_output, axis=0)
                dX = grad_output @ W.T
                return dW, db, dX
                
            # Run benchmarks
            forward_stats = self.benchmark.run_benchmark(forward)
            backward_stats = self.benchmark.run_benchmark(backward)
            
            config_key = f"batch{batch_size}_in{input_dim}_out{output_dim}"
            results[config_key] = {
                'forward': forward_stats,
                'backward': backward_stats,
                'gflops_forward': 2 * batch_size * input_dim * output_dim / (forward_stats['mean'] * 1e9),
            }
            
        # Save results
        self._save_results('dense_layer_performance', results)
        
    def test_convolution_performance(self):
        """Benchmark convolution operations."""
        configurations = [
            {'batch': 1, 'channels': 3, 'size': 224, 'filters': 64, 'kernel': 3},
            {'batch': 16, 'channels': 3, 'size': 224, 'filters': 64, 'kernel': 3},
            {'batch': 32, 'channels': 64, 'size': 56, 'filters': 128, 'kernel': 3},
            {'batch': 32, 'channels': 128, 'size': 28, 'filters': 256, 'kernel': 3},
        ]
        
        results = {}
        
        for config in configurations:
            # Placeholder for convolution benchmarks
            # Would implement actual convolution when available
            pass
            
    def test_attention_performance(self):
        """Benchmark attention mechanism performance."""
        configurations = [
            {'batch': 1, 'seq_len': 128, 'd_model': 512, 'heads': 8},
            {'batch': 8, 'seq_len': 512, 'd_model': 512, 'heads': 8},
            {'batch': 16, 'seq_len': 1024, 'd_model': 768, 'heads': 12},
        ]
        
        results = {}
        
        for config in configurations:
            batch = config['batch']
            seq_len = config['seq_len']
            d_model = config['d_model']
            heads = config['heads']
            
            # Self-attention computation
            Q = np.random.randn(batch, seq_len, d_model).astype(np.float32)
            K = np.random.randn(batch, seq_len, d_model).astype(np.float32)
            V = np.random.randn(batch, seq_len, d_model).astype(np.float32)
            
            def attention():
                # Scaled dot-product attention
                scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
                weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
                output = weights @ V
                return output
                
            stats = self.benchmark.run_benchmark(attention)
            
            config_key = f"batch{batch}_seq{seq_len}_d{d_model}_h{heads}"
            results[config_key] = stats
            
        self._save_results('attention_performance', results)
        
    def _save_results(self, test_name: str, results: Dict):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{test_name}_{timestamp}.json"
        
        os.makedirs('benchmark_results', exist_ok=True)
        filepath = os.path.join('benchmark_results', filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


class TestOptimizationPerformance(unittest.TestCase):
    """Benchmark optimizer performance."""
    
    def setUp(self):
        """Set up benchmarks."""
        self.benchmark = PerformanceBenchmark("Optimizer Performance")
        
    def test_optimizer_step_performance(self):
        """Benchmark different optimizers."""
        param_sizes = [
            1000,       # Small
            100000,     # Medium
            10000000,   # Large
        ]
        
        optimizers = ['sgd', 'momentum', 'adam', 'rmsprop']
        
        results = {}
        
        for size in param_sizes:
            params = np.random.randn(size).astype(np.float32)
            grads = np.random.randn(size).astype(np.float32)
            
            for opt_name in optimizers:
                if opt_name == 'sgd':
                    def step():
                        params[:] -= 0.01 * grads
                        
                elif opt_name == 'momentum':
                    velocity = np.zeros_like(params)
                    
                    def step():
                        velocity[:] = 0.9 * velocity - 0.01 * grads
                        params[:] += velocity
                        
                elif opt_name == 'adam':
                    m = np.zeros_like(params)
                    v = np.zeros_like(params)
                    t = 0
                    
                    def step():
                        nonlocal t
                        t += 1
                        m[:] = 0.9 * m + 0.1 * grads
                        v[:] = 0.999 * v + 0.001 * grads**2
                        m_hat = m / (1 - 0.9**t)
                        v_hat = v / (1 - 0.999**t)
                        params[:] -= 0.001 * m_hat / (np.sqrt(v_hat) + 1e-8)
                        
                elif opt_name == 'rmsprop':
                    cache = np.zeros_like(params)
                    
                    def step():
                        cache[:] = 0.9 * cache + 0.1 * grads**2
                        params[:] -= 0.01 * grads / (np.sqrt(cache) + 1e-8)
                        
                stats = self.benchmark.run_benchmark(step)
                
                key = f"{opt_name}_size{size}"
                results[key] = stats
                
        self._save_results('optimizer_performance', results)
        
    def _save_results(self, test_name: str, results: Dict):
        """Save benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{test_name}_{timestamp}.json"
        
        os.makedirs('benchmark_results', exist_ok=True)
        filepath = os.path.join('benchmark_results', filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage patterns."""
    
    def test_layer_memory_usage(self):
        """Profile memory usage of different layers."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        configurations = [
            {'type': 'dense', 'params': {'input': 1024, 'output': 1024}},
            {'type': 'conv2d', 'params': {'channels': 64, 'kernel': 3}},
            {'type': 'lstm', 'params': {'input': 512, 'hidden': 512}},
        ]
        
        results = {}
        
        for config in configurations:
            # Get baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Allocate layer parameters
            if config['type'] == 'dense':
                W = np.random.randn(config['params']['input'], 
                                  config['params']['output']).astype(np.float32)
                b = np.random.randn(config['params']['output']).astype(np.float32)
                
            # Get memory after allocation
            allocated_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_used = allocated_memory - baseline_memory
            
            results[config['type']] = {
                'config': config['params'],
                'memory_mb': memory_used,
                'baseline_mb': baseline_memory,
            }
            
            # Clean up
            del W, b
            
        return results


class RegressionTestSuite:
    """Suite for regression testing against reference implementations."""
    
    def __init__(self, reference_results_path: str = 'reference_results/'):
        """
        Initialize regression test suite.
        
        Args:
            reference_results_path: Path to reference results
        """
        self.reference_path = reference_results_path
        os.makedirs(reference_results_path, exist_ok=True)
        
    def generate_reference_results(self):
        """Generate reference results for regression testing."""
        results = {}
        
        # Test cases with known outputs
        test_cases = [
            {
                'name': 'dense_forward',
                'input': np.array([[1.0, 2.0, 3.0]]),
                'weights': np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                'bias': np.array([0.1, 0.2]),
                'expected': np.array([[2.3, 3.0]]),  # Manually calculated
            },
            {
                'name': 'relu_forward',
                'input': np.array([[-1.0, 0.0, 1.0, 2.0]]),
                'expected': np.array([[0.0, 0.0, 1.0, 2.0]]),
            },
            {
                'name': 'softmax_forward',
                'input': np.array([[1.0, 2.0, 3.0]]),
                'expected': np.array([[0.0900, 0.2447, 0.6652]]),  # Approximate
            },
        ]
        
        for test_case in test_cases:
            results[test_case['name']] = test_case
            
        # Save reference results
        with open(os.path.join(self.reference_path, 'reference.json'), 'w') as f:
            json.dump(results, f, indent=2)
            
    def run_regression_tests(self) -> Dict[str, bool]:
        """
        Run regression tests against reference results.
        
        Returns:
            Dictionary mapping test names to pass/fail status
        """
        # Load reference results
        with open(os.path.join(self.reference_path, 'reference.json'), 'r') as f:
            reference = json.load(f)
            
        results = {}
        tolerance = 1e-4
        
        for test_name, test_data in reference.items():
            if test_name == 'dense_forward':
                # Compute actual result
                actual = test_data['input'] @ test_data['weights'] + test_data['bias']
                expected = np.array(test_data['expected'])
                
                # Compare
                passed = np.allclose(actual, expected, atol=tolerance)
                results[test_name] = passed
                
            elif test_name == 'relu_forward':
                actual = np.maximum(0, test_data['input'])
                expected = np.array(test_data['expected'])
                passed = np.allclose(actual, expected, atol=tolerance)
                results[test_name] = passed
                
            elif test_name == 'softmax_forward':
                x = np.array(test_data['input'])
                exp_x = np.exp(x - np.max(x))
                actual = exp_x / np.sum(exp_x)
                expected = np.array(test_data['expected'])
                passed = np.allclose(actual, expected, atol=tolerance)
                results[test_name] = passed
                
        return results


class TestScalingBehavior(unittest.TestCase):
    """Test how performance scales with problem size."""
    
    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        input_dim = 1024
        output_dim = 512
        
        times = []
        
        for batch_size in batch_sizes:
            X = np.random.randn(batch_size, input_dim).astype(np.float32)
            W = np.random.randn(input_dim, output_dim).astype(np.float32)
            
            start = time.perf_counter()
            for _ in range(100):
                _ = X @ W
            end = time.perf_counter()
            
            times.append((end - start) / 100)
            
        # Analyze scaling
        # Should be roughly linear with batch size
        scaling_factor = times[-1] / times[0]
        batch_factor = batch_sizes[-1] / batch_sizes[0]
        
        print(f"Time scaling: {scaling_factor:.2f}x")
        print(f"Batch scaling: {batch_factor}x")
        
        # Plot results
        self._plot_scaling(batch_sizes, times, 'Batch Size Scaling')
        
    def test_model_size_scaling(self):
        """Test performance scaling with model size."""
        layer_sizes = [64, 128, 256, 512, 1024, 2048]
        batch_size = 32
        
        times = []
        
        for size in layer_sizes:
            X = np.random.randn(batch_size, size).astype(np.float32)
            W = np.random.randn(size, size).astype(np.float32)
            
            start = time.perf_counter()
            for _ in range(100):
                _ = X @ W
            end = time.perf_counter()
            
            times.append((end - start) / 100)
            
        # Should scale quadratically with layer size
        self._plot_scaling(layer_sizes, times, 'Model Size Scaling')
        
    def _plot_scaling(self, x_values: List, y_values: List, title: str):
        """Plot scaling behavior."""
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Problem Size')
        plt.ylabel('Time (seconds)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.yscale('log')
        
        # Save plot
        os.makedirs('benchmark_plots', exist_ok=True)
        plt.savefig(f'benchmark_plots/{title.lower().replace(" ", "_")}.png')
        plt.close()


def generate_performance_report():
    """Generate comprehensive performance report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'python_version': '3.x.x',  # Would get actual version
            'numpy_version': np.__version__,
            'cpu_count': os.cpu_count(),
        },
        'test_results': {},
        'regression_status': {},
    }
    
    # Run regression tests
    regression_suite = RegressionTestSuite()
    regression_suite.generate_reference_results()
    regression_results = regression_suite.run_regression_tests()
    report['regression_status'] = regression_results
    
    # Save report
    with open('performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print("Performance report generated successfully")
    
    return report


if __name__ == '__main__':
    # Run performance benchmarks
    unittest.main()