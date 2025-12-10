#!/usr/bin/env python3
"""
Baseline Attention Benchmark - No Flash Attention Optimizations
Provides comparison baseline for measuring Flash Attention improvements
"""

import mlx.core as mx
import mlx.nn as nn
import time
import math
import numpy as np
from typing import Optional, Dict, Any
import json

class StandardMLXMultiHeadAttention(nn.Module):
    """
    Standard MLX MultiHeadAttention - No optimizations
    Baseline implementation for comparison
    """
    
    def __init__(self, dims: int, num_heads: int, bias: bool = False):
        super().__init__()
        
        if dims % num_heads != 0:
            raise ValueError(f"dims ({dims}) must be divisible by num_heads ({num_heads})")
        
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Create projection layers
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)
        
        print(f"ℹ️ Standard MLX attention (dims={dims}, heads={num_heads})")
    
    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape
        
        # Apply projections
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        # Reshape for multi-head attention
        queries = queries.reshape(B, L, self.num_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_heads, self.head_dim)
        values = values.reshape(B, L, self.num_heads, self.head_dim)
        
        # Manual attention computation - No optimizations
        # Transpose keys for matrix multiplication: [B, L, H, D] -> [B, H, D, L]
        keys_t = mx.transpose(keys, [0, 2, 3, 1])
        
        # Reshape for batch matrix multiplication
        # queries: [B, L, H, D] -> [B*H, L, D]
        # keys_t: [B, H, D, L] -> [B*H, D, L]
        # values: [B, L, H, D] -> [B*H, L, D]
        queries_flat = queries.reshape(B * self.num_heads, L, self.head_dim)
        keys_flat = keys_t.reshape(B * self.num_heads, self.head_dim, L)
        values_flat = values.reshape(B * self.num_heads, L, self.head_dim)
        
        # Compute attention scores [B*H, L, L]
        scores = mx.matmul(queries_flat, keys_flat) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax computation
        attention_weights = mx.softmax(scores, axis=-1)
        
        # Apply attention to values [B*H, L, D]
        output_flat = mx.matmul(attention_weights, values_flat)
        
        # Reshape back to [B, L, H, D]
        output = output_flat.reshape(B, L, self.num_heads, self.head_dim)
        
        # Reshape and apply output projection
        output = output.reshape(B, L, D)
        return self.out_proj(output)

class BaselineBenchmark:
    """
    Baseline attention benchmarking without any optimizations
    """
    
    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context"""
        try:
            import platform
            import psutil
            
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 1),
                'mlx_device': str(mx.default_device())
            }
        except ImportError:
            return {'platform': 'Unknown', 'processor': 'Unknown'}
    
    def benchmark_baseline_attention(self, 
                                   batch_sizes=[1, 2, 4], 
                                   seq_lengths=[64, 128, 256],
                                   head_dims=[32, 64, 128],
                                   num_heads=8,
                                   num_runs=5):
        """
        Benchmark baseline MLX attention without optimizations
        """
        
        print("📊 BASELINE MLX ATTENTION BENCHMARK")
        print("=" * 70)
        print(f"System: {self.system_info.get('processor', 'Unknown')}")
        print(f"Device: {self.system_info.get('mlx_device', 'Unknown')}")
        print(f"Memory: {self.system_info.get('memory_gb', 'Unknown')}GB")
        print("=" * 70)
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                for head_dim in head_dims:
                    dims = num_heads * head_dim
                    
                    print(f"\n📊 Testing: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}")
                    
                    # Create test data
                    x = mx.random.normal((batch_size, seq_len, dims))
                    
                    # Standard baseline attention
                    baseline_attn = StandardMLXMultiHeadAttention(dims, num_heads)
                    
                    # Benchmark baseline attention
                    baseline_times = []
                    for _ in range(num_runs):
                        start_time = time.time()
                        baseline_output = baseline_attn(x)
                        mx.eval(baseline_output)  # Force evaluation
                        baseline_times.append(time.time() - start_time)
                    
                    baseline_time = np.mean(baseline_times)
                    baseline_std = np.std(baseline_times)
                    
                    # Calculate throughput (tokens/sec)
                    total_tokens = batch_size * seq_len
                    baseline_throughput = total_tokens / baseline_time
                    
                    print(f"  📈 Baseline: {baseline_time*1000:.2f}±{baseline_std*1000:.1f}ms ({baseline_throughput:.0f} tok/s)")
                    
                    # Store results
                    key = (batch_size, seq_len, head_dim)
                    self.results[key] = {
                        'baseline_time': baseline_time,
                        'baseline_std': baseline_std,
                        'baseline_throughput': baseline_throughput
                    }
        
        return self.results
    
    def print_summary(self):
        """Print baseline benchmark summary"""
        
        if not self.results:
            print("❌ No benchmark results available")
            return
        
        print("\n" + "=" * 70)
        print("📊 BASELINE BENCHMARK SUMMARY")
        print("=" * 70)
        
        throughputs = [r['baseline_throughput'] for r in self.results.values()]
        times = [r['baseline_time'] for r in self.results.values()]
        
        avg_throughput = np.mean(throughputs)
        max_throughput = max(throughputs)
        avg_time = np.mean(times)
        min_time = min(times)
        
        print(f"🏃 Average throughput: {avg_throughput:.0f} tokens/sec")
        print(f"🏃 Maximum throughput: {max_throughput:.0f} tokens/sec")
        print(f"⏱️ Average time: {avg_time*1000:.1f}ms")
        print(f"⏱️ Minimum time: {min_time*1000:.1f}ms")
        print(f"📊 Total configurations tested: {len(self.results)}")
        
        # Best performing configuration
        best_config = max(self.results.items(), key=lambda x: x[1]['baseline_throughput'])
        batch_size, seq_len, head_dim = best_config[0]
        metrics = best_config[1]
        
        print(f"\n🏆 Best performance configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Head dimension: {head_dim}")
        print(f"  Throughput: {metrics['baseline_throughput']:.0f} tokens/sec")
        print(f"  Time: {metrics['baseline_time']*1000:.1f}ms")
    
    def save_results(self, filename: str):
        """Save benchmark results to JSON file"""
        
        # Convert tuple keys to strings for JSON serialization
        json_results = {
            f"{k[0]}_{k[1]}_{k[2]}": v for k, v in self.results.items()
        }
        
        full_results = {
            'system_info': self.system_info,
            'benchmark_results': json_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")

def run_baseline_demo():
    """
    Run baseline attention benchmark for comparison
    """
    
    print("📊 BASELINE ATTENTION BENCHMARK")
    print("=" * 60)
    print("Standard MLX attention without any optimizations")
    print("Provides comparison baseline for Flash Attention")
    print("=" * 60)
    
    # Test basic functionality
    print("\n1️⃣ Testing baseline attention functionality...")
    
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    dims = num_heads * head_dim
    
    # Create test input
    x = mx.random.normal((batch_size, seq_len, dims))
    
    # Test baseline attention layer
    baseline_attn = StandardMLXMultiHeadAttention(dims, num_heads)
    
    try:
        output = baseline_attn(x)
        print(f"✅ Baseline attention output shape: {output.shape}")
        print(f"✅ Output statistics: mean={float(mx.mean(output)):.4f}, std={float(mx.std(output)):.4f}")
    except Exception as e:
        print(f"❌ Baseline attention test failed: {e}")
        return
    
    # Run comprehensive benchmark
    print("\n2️⃣ Running baseline performance benchmark...")
    
    benchmark = BaselineBenchmark()
    
    # Same configurations as Flash Attention tests for comparison
    results = benchmark.benchmark_baseline_attention(
        batch_sizes=[1, 2, 4],
        seq_lengths=[64, 128, 256],
        head_dims=[32, 64, 128],
        num_heads=8,
        num_runs=5
    )
    
    benchmark.print_summary()
    
    # Save results
    benchmark.save_results("baseline_attention_benchmark.json")
    
    print("\n🏁 Baseline benchmark complete!")
    print("\nUse these results to compare against Flash Attention optimizations:")
    print("1. Compare throughput improvements")
    print("2. Measure speedup ratios")
    print("3. Analyze performance patterns")
    print("4. Validate optimization effectiveness")

if __name__ == "__main__":
    run_baseline_demo()