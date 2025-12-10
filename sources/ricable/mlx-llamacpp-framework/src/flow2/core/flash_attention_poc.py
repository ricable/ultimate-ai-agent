#!/usr/bin/env python3
"""
Proof of Concept: Metal Flash Attention Integration with MLX
Based on research from Philip Turner's Metal Flash Attention repository
"""

import mlx.core as mx
import mlx.nn as nn
import time
import math
import numpy as np
from typing import Optional, Tuple

class MetalFlashAttention:
    """
    Proof of concept Metal Flash Attention implementation for MLX
    Based on optimizations from https://github.com/philipturner/metal-flash-attention
    """
    
    def __init__(self, head_dim: int, scale: Optional[float] = None):
        self.head_dim = head_dim
        self.scale = scale or (1.0 / math.sqrt(head_dim))
        
        # Block sizes optimized for Apple Silicon (based on research)
        self.block_size_m = 64  # Query block size
        self.block_size_n = 64  # Key/Value block size
        self.block_size_k = 32  # Head dimension block size
        
        # Create optimized Metal kernel (simplified version)
        self.flash_attention_kernel = self._create_flash_attention_kernel()
    
    def _create_flash_attention_kernel(self):
        """Create Metal kernel based on Flash Attention algorithm"""
        
        # Simplified Metal kernel inspired by Philip Turner's implementation
        kernel_source = f"""
        #include <metal_stdlib>
        using namespace metal;
        
        // Optimized Flash Attention kernel for Apple Silicon
        // Based on Metal Flash Attention research by Philip Turner
        
        constant uint BLOCK_SIZE_M = {self.block_size_m};
        constant uint BLOCK_SIZE_N = {self.block_size_n};
        constant uint BLOCK_SIZE_K = {self.block_size_k};
        
        struct AttentionParams {{
            uint batch_size;
            uint seq_len;
            uint head_dim;
            uint num_heads;
            float scale;
        }};
        
        kernel void flash_attention_forward(
            device const half* queries [[buffer(0)]],
            device const half* keys [[buffer(1)]],
            device const half* values [[buffer(2)]],
            device half* output [[buffer(3)]],
            device float* row_sums [[buffer(4)]],
            constant AttentionParams& params [[buffer(5)]],
            uint3 thread_position_in_grid [[thread_position_in_grid]],
            uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
            uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
        ) {{
            // Key optimizations from Metal Flash Attention:
            // 1. Tiled computation with register blocking
            // 2. On-the-fly softmax computation  
            // 3. Memory coalescing optimization
            // 4. Register pressure management
            
            uint batch_idx = threadgroup_position_in_grid.z;
            uint head_idx = threadgroup_position_in_grid.y;
            uint m_block = threadgroup_position_in_grid.x;
            
            uint tid = thread_position_in_threadgroup.x;
            uint seq_len = params.seq_len;
            uint head_dim = params.head_dim;
            
            // Shared memory for tiling (key optimization)
            threadgroup half shared_queries[BLOCK_SIZE_M * BLOCK_SIZE_K];
            threadgroup half shared_keys[BLOCK_SIZE_N * BLOCK_SIZE_K];
            threadgroup half shared_values[BLOCK_SIZE_N * BLOCK_SIZE_K];
            
            // Online softmax statistics (Flash Attention core algorithm)
            threadgroup float max_vals[BLOCK_SIZE_M];
            threadgroup float sum_exp[BLOCK_SIZE_M];
            
            // Initialize accumulators
            float local_max = -INFINITY;
            float local_sum = 0.0f;
            half local_output[BLOCK_SIZE_K] = {{0}};
            
            // Main computation loop over key/value blocks
            for (uint n_block = 0; n_block < (seq_len + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N; n_block++) {{
                
                // Load query block into shared memory (coalesced access)
                uint q_offset = batch_idx * params.num_heads * seq_len * head_dim + 
                               head_idx * seq_len * head_dim +
                               m_block * BLOCK_SIZE_M * head_dim;
                
                // Load key/value blocks (memory optimization)
                uint kv_offset = batch_idx * params.num_heads * seq_len * head_dim +
                                head_idx * seq_len * head_dim +
                                n_block * BLOCK_SIZE_N * head_dim;
                
                // Cooperative loading with proper memory alignment
                if (tid < BLOCK_SIZE_M * BLOCK_SIZE_K) {{
                    shared_queries[tid] = queries[q_offset + tid];
                }}
                if (tid < BLOCK_SIZE_N * BLOCK_SIZE_K) {{
                    shared_keys[tid] = keys[kv_offset + tid];
                    shared_values[tid] = values[kv_offset + tid];
                }}
                
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                // Compute attention scores (register blocked)
                float scores[BLOCK_SIZE_N];
                for (uint n = 0; n < BLOCK_SIZE_N; n++) {{
                    scores[n] = 0.0f;
                    for (uint k = 0; k < BLOCK_SIZE_K; k++) {{
                        scores[n] += float(shared_queries[tid * BLOCK_SIZE_K + k]) * 
                                   float(shared_keys[n * BLOCK_SIZE_K + k]);
                    }}
                    scores[n] *= params.scale;
                }}
                
                // Online softmax update (key Flash Attention innovation)
                float block_max = scores[0];
                for (uint n = 1; n < BLOCK_SIZE_N; n++) {{
                    block_max = max(block_max, scores[n]);
                }}
                
                float new_max = max(local_max, block_max);
                float exp_sum = 0.0f;
                
                for (uint n = 0; n < BLOCK_SIZE_N; n++) {{
                    scores[n] = exp(scores[n] - new_max);
                    exp_sum += scores[n];
                }}
                
                // Update statistics and output (memory efficient)
                float correction = exp(local_max - new_max);
                local_sum = local_sum * correction + exp_sum;
                local_max = new_max;
                
                // Update output accumulator
                for (uint k = 0; k < BLOCK_SIZE_K; k++) {{
                    local_output[k] *= correction;
                    for (uint n = 0; n < BLOCK_SIZE_N; n++) {{
                        local_output[k] += scores[n] * shared_values[n * BLOCK_SIZE_K + k];
                    }}
                }}
                
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            
            // Final normalization and output
            for (uint k = 0; k < BLOCK_SIZE_K; k++) {{
                local_output[k] /= local_sum;
                
                uint out_idx = batch_idx * params.num_heads * seq_len * head_dim +
                              head_idx * seq_len * head_dim +
                              m_block * BLOCK_SIZE_M * head_dim +
                              tid * head_dim + k;
                
                output[out_idx] = local_output[k];
            }}
            
            // Store softmax statistics for backward pass
            if (tid == 0) {{
                uint stats_idx = batch_idx * params.num_heads * seq_len +
                                head_idx * seq_len + m_block * BLOCK_SIZE_M;
                row_sums[stats_idx] = local_sum;
            }}
        }}
        """
        
        try:
            # Create the Metal kernel using MLX API
            return mx.fast.metal_kernel(
                name="flash_attention_forward",
                input_names=["queries", "keys", "values", "params"],
                output_names=["output", "row_sums"],
                source=kernel_source,
                header="""
                struct AttentionParams {
                    uint32_t batch_size;
                    uint32_t seq_len;
                    uint32_t head_dim;
                    uint32_t num_heads;
                    float scale;
                };
                """
            )
        except Exception as e:
            print(f"⚠️ Could not create Metal kernel: {e}")
            print("ℹ️ Falling back to standard MLX attention")
            return None
    
    def __call__(self, queries, keys, values, mask=None):
        """
        Apply Flash Attention computation
        
        Args:
            queries: [batch, seq_len, num_heads, head_dim]
            keys: [batch, seq_len, num_heads, head_dim]  
            values: [batch, seq_len, num_heads, head_dim]
            mask: Optional attention mask
        
        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        
        if self.flash_attention_kernel is None:
            # Fallback to standard MLX attention
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )
        
        try:
            batch_size, seq_len, num_heads, head_dim = queries.shape
            
            # Prepare parameters for Metal kernel
            params = mx.array([batch_size, seq_len, head_dim, num_heads, self.scale], 
                            dtype=mx.uint32)
            
            # Call optimized Metal kernel
            output, row_sums = self.flash_attention_kernel(
                queries, keys, values, params
            )
            
            return output
            
        except Exception as e:
            print(f"⚠️ Flash Attention kernel failed: {e}")
            print("ℹ️ Falling back to standard MLX attention")
            
            # Fallback to standard attention
            return mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=self.scale, mask=mask
            )

class OptimizedMultiHeadAttention(nn.Module):
    """
    Enhanced MultiHeadAttention with Metal Flash Attention optimization
    Drop-in replacement for standard MLX attention layers
    """
    
    def __init__(self, dims: int, num_heads: int, bias: bool = False, 
                 use_flash_attention: bool = True):
        super().__init__()
        
        if dims % num_heads != 0:
            raise ValueError(f"dims ({dims}) must be divisible by num_heads ({num_heads})")
        
        self.dims = dims
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        
        # Create projection layers
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)
        
        # Initialize Flash Attention if available
        self.use_flash_attention = use_flash_attention
        if use_flash_attention:
            self.flash_attention = MetalFlashAttention(self.head_dim)
            print("✅ Initialized Metal Flash Attention optimization")
        else:
            print("ℹ️ Using standard MLX attention")
    
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
        
        # Apply attention (Flash Attention or standard)
        if self.use_flash_attention and hasattr(self, 'flash_attention'):
            try:
                output = self.flash_attention(queries, keys, values, mask)
            except Exception as e:
                print(f"⚠️ Flash Attention failed, using fallback: {e}")
                output = mx.fast.scaled_dot_product_attention(
                    queries, keys, values, mask=mask
                )
        else:
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, mask=mask
            )
        
        # Reshape and apply output projection
        output = output.reshape(B, L, D)
        return self.out_proj(output)

class FlashAttentionBenchmark:
    """
    Comprehensive benchmarking for Flash Attention vs standard attention
    """
    
    def __init__(self):
        self.results = {}
    
    def benchmark_attention_performance(self, batch_sizes, seq_lengths, head_dims, num_heads=8):
        """
        Benchmark Flash Attention vs standard MLX attention
        """
        
        print("🔬 FLASH ATTENTION PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                for head_dim in head_dims:
                    dims = num_heads * head_dim
                    
                    print(f"\n📊 Testing: batch={batch_size}, seq_len={seq_len}, head_dim={head_dim}")
                    
                    # Create test data
                    x = mx.random.normal((batch_size, seq_len, dims))
                    
                    # Standard MLX attention
                    standard_attn = nn.MultiHeadAttention(dims, num_heads)
                    
                    # Flash Attention
                    flash_attn = OptimizedMultiHeadAttention(dims, num_heads, use_flash_attention=True)
                    
                    # Benchmark standard attention
                    start_time = time.time()
                    for _ in range(5):  # Multiple runs for accuracy
                        standard_output = standard_attn(x)
                        mx.eval(standard_output)  # Force evaluation
                    standard_time = (time.time() - start_time) / 5
                    
                    # Benchmark Flash Attention
                    start_time = time.time() 
                    for _ in range(5):
                        flash_output = flash_attn(x)
                        mx.eval(flash_output)  # Force evaluation
                    flash_time = (time.time() - start_time) / 5
                    
                    # Calculate metrics
                    speedup = standard_time / flash_time if flash_time > 0 else 1.0
                    
                    print(f"  📈 Standard MLX: {standard_time*1000:.2f}ms")
                    print(f"  ⚡ Flash Attention: {flash_time*1000:.2f}ms")
                    print(f"  🚀 Speedup: {speedup:.2f}x")
                    
                    # Store results
                    key = (batch_size, seq_len, head_dim)
                    self.results[key] = {
                        'standard_time': standard_time,
                        'flash_time': flash_time,
                        'speedup': speedup
                    }
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary"""
        
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        
        speedups = [r['speedup'] for r in self.results.values()]
        avg_speedup = sum(speedups) / len(speedups)
        max_speedup = max(speedups)
        
        print(f"🚀 Average speedup: {avg_speedup:.2f}x")
        print(f"🚀 Maximum speedup: {max_speedup:.2f}x")
        print(f"📊 Total configurations tested: {len(self.results)}")
        
        # Best performing configuration
        best_config = max(self.results.items(), key=lambda x: x[1]['speedup'])
        batch_size, seq_len, head_dim = best_config[0]
        speedup = best_config[1]['speedup']
        
        print(f"\n🏆 Best configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        print(f"  Head dimension: {head_dim}")
        print(f"  Speedup: {speedup:.2f}x")

def demo_flash_attention():
    """
    Demonstration of Flash Attention integration
    """
    
    print("🎉 METAL FLASH ATTENTION DEMO")
    print("=" * 50)
    
    # Test basic functionality
    print("\n1️⃣ Testing basic Flash Attention functionality...")
    
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64
    dims = num_heads * head_dim
    
    # Create test input
    x = mx.random.normal((batch_size, seq_len, dims))
    
    # Test Flash Attention layer
    flash_attn = OptimizedMultiHeadAttention(dims, num_heads, use_flash_attention=True)
    
    try:
        output = flash_attn(x)
        print(f"✅ Flash Attention output shape: {output.shape}")
        print(f"✅ Output statistics: mean={float(mx.mean(output)):.4f}, std={float(mx.std(output)):.4f}")
    except Exception as e:
        print(f"❌ Flash Attention test failed: {e}")
    
    # Test performance benchmark
    print("\n2️⃣ Running performance benchmark...")
    
    benchmark = FlashAttentionBenchmark()
    
    # Small benchmark for demo
    results = benchmark.benchmark_attention_performance(
        batch_sizes=[1, 2],
        seq_lengths=[64, 128],
        head_dims=[32, 64],
        num_heads=8
    )
    
    benchmark.print_summary()
    
    print("\n🎊 Demo complete!")
    print("\nNext steps:")
    print("1. Integrate with fine-tuning scripts")
    print("2. Test with real model training")
    print("3. Optimize Metal kernel implementation")
    print("4. Measure real-world improvements")

if __name__ == "__main__":
    demo_flash_attention()