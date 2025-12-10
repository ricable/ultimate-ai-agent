# Performance Optimization for LLMs on Apple Silicon

This guide provides advanced techniques for optimizing the performance of large language models on Apple Silicon hardware, helping you get the most out of your Mac.

## Table of Contents
- [Understanding Performance Factors](#understanding-performance-factors)
- [Metal Optimization](#metal-optimization)
- [Memory Optimization](#memory-optimization)
- [Quantization Strategies](#quantization-strategies)
- [Batch Processing](#batch-processing)
- [Context Length Management](#context-length-management)
- [Framework-Specific Optimizations](#framework-specific-optimizations)
- [Cooling and Thermal Management](#cooling-and-thermal-management)
- [Performance Benchmarking](#performance-benchmarking)
- [Advanced Techniques](#advanced-techniques)

## Understanding Performance Factors

LLM performance on Apple Silicon is influenced by several key factors:

### Key Performance Metrics

1. **Tokens per Second (TPS)**: The primary measure of generation speed
2. **Time to First Token (TTFT)**: How quickly the model starts generating
3. **Memory Usage**: Affects which models and tasks are possible
4. **Power Efficiency**: Important for battery life on laptops
5. **Sustained Performance**: Performance over longer workloads

### Hardware Bottlenecks

| Component | Impact on LLM Performance |
|-----------|---------------------------|
| **RAM** | Limits model size and precision |
| **Neural Engine** | Accelerates matrix operations |
| **GPU Cores** | Handles parallel computation |
| **CPU Cores** | Manages tokenization and coordination |
| **Memory Bandwidth** | Affects data transfer speed |
| **Thermal Design** | Determines sustained performance |

## Metal Optimization

Apple's Metal API provides GPU acceleration for LLMs, and proper configuration is critical for performance.

### Enabling Metal Acceleration

#### For llama.cpp:

```bash
# Build with Metal support
cmake .. -DLLAMA_METAL=ON

# Run with Metal enabled
./main -m models/llama-2-7b-q4_k.gguf --metal -p "Your prompt"

# Enable advanced Metal optimizations
./main -m models/llama-2-7b-q4_k.gguf --metal --metal-mmq
```

#### For MLX:

```python
import mlx.core as mx

# Set GPU as default device
mx.set_default_device(mx.gpu)

# Load and run model (Metal is used automatically)
from mlx_lm import load, generate
model, tokenizer = load("llama-2-7b", quantization="int4")
```

### Metal Performance Tuning

```python
# Fine-tuning Metal performance in MLX
import mlx.core as mx

# Control memory allocation
mx.set_allocation_limit(0.8)  # Use up to 80% of available memory

# Control automatic buffer reuse
mx.enable_buffer_reuse()  # Reuse memory buffers

# Optimize compilation cache
mx.enable_compile_cache(cache_dir="./mlx_cache")
```

### Metal Profiling

Use Apple's Metal profiling tools to identify bottlenecks:

1. **Instruments App**: 
   - Launch from Xcode > Open Developer Tool > Instruments
   - Select "Metal System Trace" template
   - Record during model inference

2. **Metal Performance Shader Insights**:
   ```bash
   # Run with MPS insights
   MPS_INSIGHTS=1 ./main -m models/llama-2-7b-q4_k.gguf --metal
   ```

## Memory Optimization

Efficient memory usage is crucial for running larger models or longer contexts.

### Memory-Efficient Inference

```python
# MLX memory-efficient inference
from mlx_lm import load, generate
import mlx.core as mx

# Set available memory
mx.set_allocation_limit(0.9)  # Use up to 90% of available memory

# Load with appropriate quantization for your hardware
model, tokenizer = load(
    "llama-2-7b", 
    quantization="int4",  # Most memory-efficient
)

# Generate with memory-efficient settings
tokens = generate(
    model,
    tokenizer,
    "Your prompt",
    max_tokens=512,  # Control output length
    stream=True  # Stream tokens to reduce peak memory
)
```

### Memory Monitoring

```bash
# Monitor memory usage with Activity Monitor
# Or use this Python script

import psutil
import os
import time
import mlx.core as mx

def monitor_memory():
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {before:.2f} MB")
    
    # Load model
    from mlx_lm import load
    model, tokenizer = load("llama-2-7b", quantization="int4")
    
    after_load = process.memory_info().rss / 1024 / 1024  # MB
    print(f"After model load: {after_load:.2f} MB")
    print(f"Model memory: {after_load - before:.2f} MB")
    
    # Check MLX memory stats
    mx_memory = mx.get_available_memory() / 1024 / 1024 / 1024  # GB
    print(f"MLX available memory: {mx_memory:.2f} GB")

# Run memory monitoring
monitor_memory()
```

### Memory Cleanup

```python
# Force memory cleanup in MLX
import gc
import mlx.core as mx

def cleanup_memory():
    # Delete model reference
    del model
    
    # Collect Python garbage
    gc.collect()
    
    # Clear MLX caches
    mx.clear_compilation_cache()
    mx.clear_buffer_cache()
    
    # Print available memory
    print(f"Available memory: {mx.get_available_memory() / 1024**3:.2f} GB")
```

## Quantization Strategies

Quantization reduces model size and memory usage, often with minimal quality loss.

### Optimal Quantization Selection

| Hardware | 7B Model | 13B Model | 33B Model | 70B Model |
|----------|----------|-----------|-----------|-----------|
| 8GB RAM | INT4/Q4_K | Not recommended | Not possible | Not possible |
| 16GB RAM | INT8/Q8_0 | INT4/Q4_K | Not recommended | Not possible |
| 32GB RAM | FP16 | INT8/Q8_0 | INT4/Q4_K | Not possible |
| 64GB RAM | FP16 | FP16 | INT8/Q8_0 | INT4/Q4_K |
| 128GB RAM | FP16 | FP16 | FP16 | INT8/Q8_0 |

### Quality-Optimized Quantization (llama.cpp)

```bash
# For highest quality with good compression
python quantize.py --model models/llama-2-7b.gguf --outfile models/llama-2-7b-q6_K.gguf --type q6_K

# For balanced quality and size
python quantize.py --model models/llama-2-7b.gguf --outfile models/llama-2-7b-q5_K.gguf --type q5_K

# For maximum compression with acceptable quality
python quantize.py --model models/llama-2-7b.gguf --outfile models/llama-2-7b-q3_K.gguf --type q3_K
```

### Advanced Quantization (MLX)

```python
from mlx_lm.utils import load_model
from mlx_lm.quantize import quantize_model
import mlx.core as mx

# Load full precision model
model, tokenizer = load_model("llama-2-7b")

# Quantize with custom parameters
model = quantize_model(
    model,
    nbits=4,        # Bit precision (4 or 8)
    group_size=64,  # Quantization group size
    scheme="sym",   # Symmetrical quantization
    quant_all=True  # Quantize all layers
)

# Save quantized model
mx.save("llama-2-7b-custom-int4.npz", model.parameters())
```

### Quantization Comparison

Run this script to compare quantization methods:

```python
import time
import mlx.core as mx
from mlx_lm import load, generate

# Test different quantization levels
def benchmark_quantization(model_name, prompt, max_tokens=100):
    results = {}
    
    # Test different quantization levels
    for quant in [None, "int8", "int4"]:
        # Load model
        print(f"Loading with quantization: {quant if quant else 'none (fp16)'}")
        start_time = time.time()
        model, tokenizer = load(model_name, quantization=quant)
        load_time = time.time() - start_time
        
        # Measure memory
        before_gen = mx.get_available_memory() / 1024**3  # GB
        
        # Generate text
        start_time = time.time()
        tokens = generate(model, tokenizer, prompt, max_tokens=max_tokens)
        generation_time = time.time() - start_time
        
        # Calculate tokens per second
        tokens_per_second = max_tokens / generation_time
        
        # Record results
        results[quant if quant else "fp16"] = {
            "load_time": load_time,
            "tokens_per_second": tokens_per_second,
            "memory_used": before_gen - (mx.get_available_memory() / 1024**3)
        }
        
        # Cleanup
        del model, tokenizer
        mx.clear_buffer_cache()
        mx.clear_compilation_cache()
    
    return results

# Run benchmark
results = benchmark_quantization(
    "llama-2-7b", 
    "Explain the benefits of quantization in machine learning models",
    max_tokens=100
)

# Print results
print("\nQuantization Benchmark Results:")
print("-------------------------------")
for quant, metrics in results.items():
    print(f"Quantization: {quant}")
    print(f"  Load Time: {metrics['load_time']:.2f} seconds")
    print(f"  Speed: {metrics['tokens_per_second']:.2f} tokens/second")
    print(f"  Memory Used: {metrics['memory_used']:.2f} GB")
    print("-------------------------------")
```

## Batch Processing

Batch processing improves throughput for multiple inputs.

### Batch Inference with llama.cpp

```bash
# Create batch input file
cat > batch_prompts.txt << EOL
Tell me about Apple Silicon
Explain quantum computing
What is machine learning?
EOL

# Run batch inference
./main -m models/llama-2-7b-q4_k.gguf --metal -f batch_prompts.txt -n 100 --batch-size 512
```

### Batch Inference with MLX

```python
import mlx.core as mx
from mlx_lm import load

# Load model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Prepare batch of prompts
prompts = [
    "Tell me about Apple Silicon",
    "Explain quantum computing",
    "What is machine learning?"
]

# Tokenize all prompts
batch_tokens = [tokenizer.encode(prompt) for prompt in prompts]

# Pad to same length
max_len = max(len(tokens) for tokens in batch_tokens)
batch_tokens = [tokens + [tokenizer.pad_id] * (max_len - len(tokens)) for tokens in batch_tokens]

# Convert to MLX array
batch_inputs = mx.array(batch_tokens)

# Generate for all prompts in batch
batch_outputs = model.generate(
    batch_inputs,
    max_tokens=100,
    temp=0.7,
    top_p=0.95
)

# Decode outputs
for i, output in enumerate(batch_outputs):
    decoded = tokenizer.decode(output.tolist())
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Response: {decoded}")
    print("---")
```

### Optimal Batch Sizes

| Hardware | Recommended Batch Size | Notes |
|----------|------------------------|-------|
| MacBook Air (8GB) | 1-2 | Limited by memory |
| MacBook Pro (16GB) | 4-8 | Good balance |
| Mac Studio (32GB+) | 8-32 | Higher throughput |

## Context Length Management

Managing context length is crucial for both performance and memory usage.

### Optimal Context Settings

```bash
# llama.cpp context management
./main -m models/llama-2-7b-q4_k.gguf --metal -c 2048 -p "Your prompt"
```

```python
# MLX context management
from mlx_lm import load, generate

# Set max tokens during model loading
model, tokenizer = load("llama-2-7b", quantization="int4", max_tokens=2048)

# Generate with specific context length
tokens = generate(model, tokenizer, "Your prompt", max_tokens=512)
```

### Context Length vs. Memory Usage

| Model Size | Memory per 1K Context Tokens |
|------------|------------------------------|
| 7B | ~150MB |
| 13B | ~300MB |
| 33B | ~700MB |
| 70B | ~1.5GB |

### Dynamic Context Management

```python
# Adaptive context length based on available memory
import mlx.core as mx
from mlx_lm import load, generate

def adaptive_context(model_name, prompt, quantization="int4"):
    # Get available memory
    available_gb = mx.get_available_memory() / 1024**3
    
    # Determine appropriate context length based on model size and memory
    if "7b" in model_name.lower():
        mem_per_1k = 0.15  # ~150MB per 1K tokens for 7B models
    elif "13b" in model_name.lower():
        mem_per_1k = 0.3   # ~300MB per 1K tokens for 13B models
    elif "33b" in model_name.lower():
        mem_per_1k = 0.7   # ~700MB per 1K tokens for 33B models
    else:
        mem_per_1k = 1.5   # ~1.5GB per 1K tokens for 70B models
    
    # Reserve 20% memory for overhead
    usable_gb = available_gb * 0.8
    
    # Calculate maximum safe context length (in tokens)
    max_context = int((usable_gb / mem_per_1k) * 1000)
    
    # Cap at model's maximum context
    max_context = min(max_context, 8192)
    
    print(f"Using adaptive context length: {max_context} tokens")
    
    # Load model with calculated context
    model, tokenizer = load(model_name, quantization=quantization, max_tokens=max_context)
    
    # Generate with adaptive context
    tokens = generate(model, tokenizer, prompt, max_tokens=min(2048, max_context // 2))
    return tokenizer.decode(tokens)

# Example usage
response = adaptive_context("llama-2-7b", "Write a comprehensive essay about artificial intelligence")
print(response)
```

## Framework-Specific Optimizations

### llama.cpp Optimization

```bash
# Optimal settings for M1/M2 MacBook Air
./main \
  -m models/llama-2-7b-q4_k.gguf \
  --metal \
  -t 4 \
  -c 2048 \
  -b 512 \
  --temp 0.7 \
  -p "Your prompt"

# Optimal settings for M1/M2 Pro MacBook Pro
./main \
  -m models/llama-2-7b-q4_k.gguf \
  --metal \
  --metal-mmq \
  -t 8 \
  -c 4096 \
  -b 1024 \
  --temp 0.7 \
  -p "Your prompt"

# Optimal settings for M1/M2 Max/Ultra Mac Studio
./main \
  -m models/llama-2-7b-q4_k.gguf \
  --metal \
  --metal-mmq \
  -t 16 \
  -c 8192 \
  -b 2048 \
  --temp 0.7 \
  -p "Your prompt"
```

### MLX Optimization

```python
import mlx.core as mx
from mlx_lm import load, generate

# Set GPU as default device
mx.set_default_device(mx.gpu)

# Enable compile cache for faster repeated runs
mx.enable_compile_cache(cache_dir="./mlx_cache")

# Set memory utilization
mx.set_allocation_limit(0.9)  # Use up to 90% of available memory

# Enable buffer reuse for memory efficiency
mx.enable_buffer_reuse()

# Load model with optimal settings
model, tokenizer = load(
    "llama-2-7b",
    quantization="int4",
    max_tokens=2048
)

# Generate with optimal parameters
tokens = generate(
    model,
    tokenizer,
    "Your prompt",
    max_tokens=512,
    temp=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
    batch_size=32  # Adjust based on available memory
)
```

### Framework Performance Comparison

```python
# Compare MLX vs llama.cpp performance
import time
import subprocess
import json

def benchmark_frameworks(prompt, max_tokens=100):
    results = {}
    
    # Benchmark MLX
    print("Benchmarking MLX...")
    
    start_time = time.time()
    from mlx_lm import load, generate
    model, tokenizer = load("llama-2-7b", quantization="int4")
    load_time = time.time() - start_time
    
    start_time = time.time()
    tokens = generate(model, tokenizer, prompt, max_tokens=max_tokens)
    generation_time = time.time() - start_time
    
    mlx_result = {
        "load_time": load_time,
        "generation_time": generation_time,
        "tokens_per_second": max_tokens / generation_time
    }
    results["MLX"] = mlx_result
    
    # Benchmark llama.cpp
    print("Benchmarking llama.cpp...")
    
    # Measure load time (approximate)
    start_time = time.time()
    # Just run a minimal generation to measure load time
    subprocess.run(
        ["./main", "-m", "models/llama-2-7b-q4_k.gguf", "--metal", "-p", "test", "-n", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    load_time = time.time() - start_time
    
    # Measure generation time
    start_time = time.time()
    subprocess.run(
        ["./main", "-m", "models/llama-2-7b-q4_k.gguf", "--metal", "-p", prompt, "-n", str(max_tokens)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    generation_time = time.time() - start_time
    
    llama_result = {
        "load_time": load_time,
        "generation_time": generation_time,
        "tokens_per_second": max_tokens / generation_time
    }
    results["llama.cpp"] = llama_result
    
    return results

# Run benchmark
results = benchmark_frameworks(
    "Explain the differences between MLX and llama.cpp frameworks for running LLMs on Apple Silicon",
    max_tokens=100
)

# Print results
print("\nFramework Benchmark Results:")
print("---------------------------")
for framework, metrics in results.items():
    print(f"Framework: {framework}")
    print(f"  Load Time: {metrics['load_time']:.2f} seconds")
    print(f"  Generation Time: {metrics['generation_time']:.2f} seconds")
    print(f"  Speed: {metrics['tokens_per_second']:.2f} tokens/second")
    print("---------------------------")
```

## Cooling and Thermal Management

Thermal management is crucial for sustained performance, especially on laptops.

### Monitoring Temperature

```bash
# Install temperature monitoring tool
brew install osx-cpu-temp

# Monitor in real-time
watch -n 1 osx-cpu-temp
```

### Thermal Optimization Strategies

1. **MacBook Air (Passive Cooling)**:
   - Use in well-ventilated area
   - Consider a cooling pad
   - Run intensive tasks in shorter bursts
   - Reduce ambient temperature

2. **MacBook Pro (Active Cooling)**:
   - Ensure vents are unobstructed
   - Clean fans periodically
   - Use in well-ventilated area
   - Consider advanced cooling pads

3. **Mac Studio/Mac Pro**:
   - Ensure adequate airflow around device
   - Keep dust-free
   - Monitor temperature during sustained workloads

### Thermal Throttling Detection

```python
# Monitor performance over time to detect thermal throttling
import time
import mlx.core as mx
from mlx_lm import load, generate

def detect_throttling(model_name, prompt, duration_minutes=10):
    # Load model
    model, tokenizer = load(model_name, quantization="int4")
    
    # Run inference in a loop and track performance
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    iterations = 0
    total_tokens = 0
    performance_log = []
    
    print(f"Running thermal test for {duration_minutes} minutes...")
    
    while time.time() < end_time:
        iteration_start = time.time()
        
        # Generate text
        tokens = generate(model, tokenizer, prompt, max_tokens=100)
        
        iteration_time = time.time() - iteration_start
        tokens_per_second = 100 / iteration_time
        
        # Log performance
        performance_log.append({
            "iteration": iterations,
            "elapsed_minutes": (time.time() - start_time) / 60,
            "tokens_per_second": tokens_per_second
        })
        
        iterations += 1
        total_tokens += 100
        
        # Print current performance
        print(f"Iteration {iterations}: {tokens_per_second:.2f} tokens/second")
        
        # Short cooldown to prevent overheating
        time.sleep(1)
    
    # Analyze for throttling
    baseline_tps = performance_log[0]["tokens_per_second"]
    final_tps = performance_log[-1]["tokens_per_second"]
    max_tps = max(log["tokens_per_second"] for log in performance_log)
    min_tps = min(log["tokens_per_second"] for log in performance_log)
    
    print("\nThermal Test Results:")
    print(f"Total iterations: {iterations}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Initial performance: {baseline_tps:.2f} tokens/second")
    print(f"Final performance: {final_tps:.2f} tokens/second")
    print(f"Maximum performance: {max_tps:.2f} tokens/second")
    print(f"Minimum performance: {min_tps:.2f} tokens/second")
    print(f"Performance drop: {((baseline_tps - min_tps) / baseline_tps) * 100:.2f}%")
    
    if min_tps < baseline_tps * 0.8:
        print("Significant thermal throttling detected!")
    elif min_tps < baseline_tps * 0.9:
        print("Moderate thermal throttling detected.")
    else:
        print("No significant thermal throttling detected.")
    
    return performance_log

# Run throttling detection
performance_data = detect_throttling("llama-2-7b", "Explain machine learning concepts", duration_minutes=5)
```

## Performance Benchmarking

Benchmarking helps you understand your system's capabilities and optimize settings.

### Basic Benchmark Script

```python
import time
import mlx.core as mx
from mlx_lm import load, generate

def benchmark_generation(model_name, prompt, max_tokens=100, quantization="int4", trials=3):
    # Load model
    start_time = time.time()
    model, tokenizer = load(model_name, quantization=quantization)
    load_time = time.time() - start_time
    
    print(f"Model loading time: {load_time:.2f} seconds")
    
    # Run multiple trials
    generation_times = []
    for i in range(trials):
        print(f"Running trial {i+1}/{trials}...")
        
        # Generate text
        start_time = time.time()
        tokens = generate(model, tokenizer, prompt, max_tokens=max_tokens)
        generation_time = time.time() - start_time
        
        tokens_per_second = max_tokens / generation_time
        generation_times.append(tokens_per_second)
        
        print(f"  Trial {i+1}: {tokens_per_second:.2f} tokens/second")
    
    # Calculate statistics
    avg_tps = sum(generation_times) / len(generation_times)
    max_tps = max(generation_times)
    min_tps = min(generation_times)
    
    print("\nBenchmark Results:")
    print(f"Average speed: {avg_tps:.2f} tokens/second")
    print(f"Maximum speed: {max_tps:.2f} tokens/second")
    print(f"Minimum speed: {min_tps:.2f} tokens/second")
    print(f"Variance: {max_tps - min_tps:.2f} tokens/second")
    
    return {
        "load_time": load_time,
        "avg_tokens_per_second": avg_tps,
        "max_tokens_per_second": max_tps,
        "min_tokens_per_second": min_tps
    }

# Run benchmark
benchmark_generation(
    "llama-2-7b",
    "Explain the impact of optimization techniques on large language model performance",
    max_tokens=200,
    trials=3
)
```

### Comprehensive Benchmark Suite

```python
import time
import json
import mlx.core as mx
from mlx_lm import load, generate

def comprehensive_benchmark(models, quantizations, token_counts, prompts):
    results = {}
    
    for model in models:
        model_results = {}
        for quant in quantizations:
            quant_results = {}
            
            print(f"\nBenchmarking {model} with {quant if quant else 'fp16'} quantization")
            try:
                # Load model
                start_time = time.time()
                loaded_model, tokenizer = load(model, quantization=quant)
                load_time = time.time() - start_time
                
                quant_results["load_time"] = load_time
                quant_results["token_results"] = {}
                
                # Test different token counts
                for token_count in token_counts:
                    token_results = []
                    
                    # Test with different prompts
                    for prompt in prompts:
                        print(f"  Generating {token_count} tokens...")
                        
                        # Warmup run
                        generate(loaded_model, tokenizer, prompt, max_tokens=10)
                        
                        # Timed run
                        start_time = time.time()
                        tokens = generate(loaded_model, tokenizer, prompt, max_tokens=token_count)
                        generation_time = time.time() - start_time
                        
                        tokens_per_second = token_count / generation_time
                        
                        token_results.append({
                            "prompt": prompt,
                            "tokens_per_second": tokens_per_second,
                            "generation_time": generation_time
                        })
                    
                    # Calculate average for this token count
                    avg_tps = sum(r["tokens_per_second"] for r in token_results) / len(token_results)
                    quant_results["token_results"][token_count] = {
                        "detailed": token_results,
                        "average_tokens_per_second": avg_tps
                    }
                
                # Memory usage test
                baseline_mem = mx.get_available_memory() / 1024**3  # GB
                generate(loaded_model, tokenizer, prompts[0], max_tokens=100)
                after_mem = mx.get_available_memory() / 1024**3  # GB
                quant_results["memory_usage_gb"] = baseline_mem - after_mem
                
                # Cleanup
                del loaded_model
                mx.clear_buffer_cache()
                mx.clear_compilation_cache()
                
            except Exception as e:
                quant_results["error"] = str(e)
            
            model_results[quant if quant else "fp16"] = quant_results
        
        results[model] = model_results
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nBenchmark Summary:")
    for model in models:
        print(f"\nModel: {model}")
        for quant in quantizations:
            quant_name = quant if quant else "fp16"
            if "error" in results[model][quant_name]:
                print(f"  {quant_name}: ERROR - {results[model][quant_name]['error']}")
                continue
                
            print(f"  {quant_name}:")
            print(f"    Load time: {results[model][quant_name]['load_time']:.2f} seconds")
            print(f"    Memory usage: {results[model][quant_name]['memory_usage_gb']:.2f} GB")
            
            for token_count, token_result in results[model][quant_name]["token_results"].items():
                print(f"    {token_count} tokens: {token_result['average_tokens_per_second']:.2f} tokens/second")
    
    return results

# Run comprehensive benchmark
benchmark_results = comprehensive_benchmark(
    models=["llama-2-7b"],
    quantizations=[None, "int8", "int4"],
    token_counts=[100, 512, 1024],
    prompts=[
        "Explain the concept of artificial intelligence",
        "What are the benefits of quantum computing?",
        "Describe the process of photosynthesis in plants"
    ]
)
```

## Advanced Techniques

### KV Cache Optimization

```python
# MLX KV cache optimization
from mlx_lm.models import Transformer
import mlx.core as mx

class OptimizedTransformer(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache = None
    
    def forward(self, inputs, cache=None):
        # Use cached key/value pairs for autoregressive generation
        if cache is not None and self.kv_cache is not None:
            # Use cached values for previously processed tokens
            # Only process the new tokens
            new_tokens = inputs[:, -1:]
            result = super().forward(new_tokens, cache)
            
            # Update cache
            self.kv_cache = self._update_cache(self.kv_cache, result.cache)
            return result._replace(cache=self.kv_cache)
        else:
            # Process all tokens and set initial cache
            result = super().forward(inputs, cache)
            self.kv_cache = result.cache
            return result
    
    def _update_cache(self, old_cache, new_cache):
        # Combine old cache with new entries
        updated_cache = []
        for old_layer, new_layer in zip(old_cache, new_cache):
            updated_k = mx.concatenate([old_layer[0], new_layer[0]], axis=1)
            updated_v = mx.concatenate([old_layer[1], new_layer[1]], axis=1)
            updated_cache.append((updated_k, updated_v))
        return updated_cache
    
    def clear_cache(self):
        self.kv_cache = None
```

### Mixed Precision Operations

```python
# Mixed precision inference
import mlx.core as mx
from mlx_lm import load

# Load model
model, tokenizer = load("llama-2-7b")

# Convert specific operations to fp16
def mixed_precision_conversion(model):
    for name, param in model.parameters().items():
        # Keep attention weights in fp16 for faster matrix multiplication
        if any(layer in name for layer in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            if param.dtype == mx.float32:
                model.update_parameter(name, param.astype(mx.float16))
    return model

# Apply mixed precision
model = mixed_precision_conversion(model)
```

### Energy Efficiency Mode

```python
# Energy-efficient inference mode
import mlx.core as mx
from mlx_lm import load, generate

def energy_efficient_generate(model_name, prompt, max_tokens=512):
    # Load with memory-efficient settings
    model, tokenizer = load(model_name, quantization="int4")
    
    # Set lower performance, higher efficiency device settings
    mx.set_allocation_limit(0.7)  # Use less memory
    
    # Generate with energy-efficient settings
    # Smaller batch size, more sequential processing
    tokens = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        batch_size=1,    # Process tokens one by one
        temp=0.7,
        stream=True      # Stream tokens to reduce peak memory
    )
    
    return tokenizer.decode(tokens)
```

### Custom Compilation Cache

```python
# Custom compilation cache for faster startup
import os
import mlx.core as mx

# Set up a persistent compilation cache
cache_dir = os.path.expanduser("~/.mlx_cache")
os.makedirs(cache_dir, exist_ok=True)

# Enable compilation cache
mx.enable_compile_cache(cache_dir=cache_dir)

# Load model (will use cached computations if available)
from mlx_lm import load
model, tokenizer = load("llama-2-7b", quantization="int4")

# After running, the cache will be populated
# Future runs will be faster
```

### Performance Profiling

```python
# Performance profiling
import time
import mlx.core as mx
from mlx_lm import load

def profile_model_operations(model_name, prompt, quantization="int4"):
    # Load model
    model, tokenizer = load(model_name, quantization=quantization)
    
    # Tokenize prompt
    input_ids = mx.array([tokenizer.encode(prompt)])
    
    # Profile forward pass
    layer_times = {}
    
    # Run once to compile
    _ = model(input_ids)
    
    # Profile entire forward pass
    start_time = time.time()
    output = model(input_ids)
    total_time = time.time() - start_time
    
    print(f"Total forward pass time: {total_time*1000:.2f} ms")
    
    # Profile individual operations if possible
    try:
        # This is a simplified example - actual profiling would depend on model structure
        for name, module in model.named_modules():
            # Skip certain module types
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                # Time this specific module
                start_time = time.time()
                if isinstance(module, nn.Embedding):
                    _ = module(input_ids)
                else:
                    # For other layers, create dummy input of right shape
                    dummy_input = mx.zeros(output.shape)
                    _ = module(dummy_input)
                op_time = time.time() - start_time
                
                layer_times[name] = op_time * 1000  # Convert to ms
    except:
        print("Detailed layer profiling not supported")
    
    # Sort and print results
    sorted_times = sorted(layer_times.items(), key=lambda x: x[1], reverse=True)
    print("\nOperation Profile (Top 10):")
    for name, time_ms in sorted_times[:10]:
        print(f"{name}: {time_ms:.2f} ms")
    
    return {
        "total_time_ms": total_time * 1000,
        "layer_times_ms": layer_times
    }

# Run profiling
profile_results = profile_model_operations("llama-2-7b", "Explain the concept of neural networks")
```

## Further Reading

- [Quantization Guide](../advanced/quantization-guide.md) - Detailed guide to model quantization
- [Hardware Recommendations](hardware-recommendations.md) - Choosing the right Mac for your needs
- [Memory Management Guide](memory-management.md) - Advanced memory optimization techniques