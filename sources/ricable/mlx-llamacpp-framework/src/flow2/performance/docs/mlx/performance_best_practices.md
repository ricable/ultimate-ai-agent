# MLX Performance Best Practices for Apple Silicon

This guide covers best practices for maximizing MLX performance on Apple Silicon Macs.

## Table of Contents
1. [Metal GPU Acceleration](#1-metal-gpu-acceleration)
2. [Memory Optimization](#2-memory-optimization)
3. [Quantization Selection](#3-quantization-selection)
4. [Model Loading and Caching](#4-model-loading-and-caching)
5. [Generation Parameters](#5-generation-parameters)
6. [Hardware-Specific Recommendations](#6-hardware-specific-recommendations)
7. [Fine-tuning Optimization](#7-fine-tuning-optimization)
8. [Python Environment Setup](#8-python-environment-setup)
9. [Monitoring and Profiling](#9-monitoring-and-profiling)
10. [Model Conversion and Formats](#10-model-conversion-and-formats)

## 1. Metal GPU Acceleration

MLX is designed specifically for Apple Silicon and uses Metal for GPU acceleration by default.

### Key Settings

- **Default Device**: Set the default device to GPU for best performance
- **Graph Compilation**: MLX compiles computation graphs for Metal automatically
- **Performance Impact**: 2-5x speedup over CPU-only inference

### Example Code

```python
import mlx.core as mx
from mlx_lm import load, generate

# Set GPU as default device
mx.set_default_device(mx.gpu)

# Load model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Generate text
output = generate(model, tokenizer, "Your prompt here", max_tokens=512)
print(tokenizer.decode(output))
```

### Chip-Specific Recommendations

| Apple Silicon Chip | Expected Speedup vs CPU |
|--------------------|-------------------------|
| M1                 | 2-3x                    |
| M1 Pro/Max         | 3-4x                    |
| M2                 | 3-4x                    |
| M2 Pro/Max/Ultra   | 4-5x                    |
| M3                 | 4-5x                    |
| M3 Pro/Max/Ultra   | 5-6x                    |

### Common Issues

- If performance is unexpectedly low, check if the default device is set to GPU
- First-run compilation overhead is normal - subsequent runs will be faster
- Watch for Python overhead in small operations - batch operations when possible

## 2. Memory Optimization

Memory management is critical for running large models efficiently.

### Memory Usage Formula

As a rough estimate, a model's memory usage during inference can be calculated as:
- Base model memory + Context window overhead
- Example: A 7B model with INT4 quantization uses ~4GB for the model plus additional memory for the context window

### Memory-Saving Techniques

1. **Clear Cache**: Use `mx.clear_cache()` between inference runs
2. **Garbage Collection**: Call `gc.collect()` to reclaim memory
3. **Unified Memory**: MLX uses Apple's unified memory architecture efficiently
4. **Stream Generation**: Use streaming for constant memory usage regardless of output length

### Hardware Recommendations

| RAM | Maximum Recommended Model Size |
|-----|--------------------------------|
| 8GB | 7B with INT4 quantization |
| 16GB | 13B with INT4 or 7B with INT8/FP16 |
| 32GB | 33B with INT4 or 13B with INT8/FP16 |
| 64GB | 70B with INT4 or 33B with INT8/FP16 |

### Memory Management Example

```python
import gc
import mlx.core as mx
from mlx_lm import load, generate

# Clear cache before loading model
mx.clear_cache()
gc.collect()

# Load model with memory-efficient quantization
model, tokenizer = load("llama-2-7b", quantization="int4")

# Generate text
output = generate(model, tokenizer, "Your prompt", max_tokens=512)

# Clear memory after use
del model, tokenizer
gc.collect()
mx.clear_cache()
```

## 3. Quantization Selection

MLX supports various quantization methods to reduce memory requirements.

### Quantization Types

| Quantization | Size Reduction | Quality Impact | Use Case |
|--------------|----------------|----------------|----------|
| None (FP16)  | Baseline       | None           | When maximum quality is required |
| INT8         | ~50%           | Minimal        | High-quality, memory available |
| INT4         | ~75%           | Moderate       | **Recommended default** |

### Quantization Parameters

MLX's quantization supports additional parameters:
- **Group Size**: Controls granularity of quantization (default: 64)
- **Block Size**: Size of blocks for block-wise quantization

### Hardware-Based Recommendations

| Hardware | RAM | Recommended Quantization |
|----------|-----|--------------------------|
| MacBook Air (8GB) | 8GB | INT4 |
| MacBook Pro (16GB) | 16GB | INT4/INT8 |
| MacBook Pro (32GB) | 32GB | INT8/FP16 |
| Mac Studio (64GB+) | 64GB+ | FP16 |

### Quantization Example

```python
# Load with quantization
from mlx_lm import load

# Option 1: Load with built-in quantization
model, tokenizer = load("llama-2-7b", quantization="int4")

# Option 2: Custom quantization after loading
from mlx_lm.utils import load_model
from mlx_lm.quantize import quantize_model

model, tokenizer = load_model("llama-2-7b")
model = quantize_model(model, nbits=4, group_size=64)
```

## 4. Model Loading and Caching

Efficient model loading and caching can significantly improve performance.

### Loading Optimization

- **Precompile Operations**: Run a small example through the model after loading
- **Safetensors Format**: Prefer Safetensors format when available
- **Memory Mapping**: MLX uses memory mapping for efficient loading

### Caching

- **Compiled Graphs**: MLX caches compiled graphs between runs
- **Weights Cache**: Enable weights caching for faster loading

### Startup Time Optimization

The first execution includes compilation time. Strategies to manage this:
- Use a small warm-up input to compile the model before actual use
- For server applications, perform compilation during initialization

### Example: Optimized Loading

```python
import mlx.core as mx
from mlx_lm import load, generate

def load_optimized_model(model_name, quantization=None):
    # Load model
    model, tokenizer = load(model_name, quantization=quantization)
    
    # Force compilation with a small input
    input_tokens = tokenizer.encode("This is a test")
    _ = model(mx.array([input_tokens]))
    
    return model, tokenizer

# Load with optimization
model, tokenizer = load_optimized_model("llama-2-7b", quantization="int4")
```

## 5. Generation Parameters

Tuning generation parameters can significantly impact both speed and quality.

### Key Parameters

- **Temperature**: Controls randomness (0.0 for deterministic, 0.7-1.0 for creative)
- **Top-p**: Nucleus sampling parameter (typically 0.9-0.95)
- **Top-k**: Limits vocabulary to top k tokens (typically 40-50)
- **Repetition Penalty**: Prevents repetition (typically 1.1-1.2)

### Performance Impact

- **Lower Temperature**: Faster generation (especially at 0.0)
- **Streaming**: More responsive user experience
- **Max Tokens**: Set appropriately to avoid unnecessary computation

### Recommended Settings for Different Use Cases

| Use Case | Temperature | Top-p | Top-k | Repetition Penalty |
|----------|------------|-------|-------|-------------------|
| Creative Writing | 0.7-0.9 | 0.9 | 50 | 1.1 |
| Q&A/Factual | 0.1-0.3 | 0.9 | 40 | 1.2 |
| Code Generation | 0.2-0.5 | 0.95 | 50 | 1.1 |
| Chat | 0.6-0.8 | 0.9 | 40 | 1.1 |

### Optimized Generation Example

```python
# Optimized generation parameters
tokens = generate(
    model,
    tokenizer,
    "Write a short story about a robot learning to paint.",
    max_tokens=512,
    temp=0.7,      # Good balance of creativity and coherence
    top_p=0.9,     # Consider tokens with top 90% probability mass
    top_k=50,      # Consider only top 50 tokens
    repetition_penalty=1.1,  # Reduce repetition
    stream=True    # Stream tokens for better UX
)
```

## 6. Hardware-Specific Recommendations

Each Apple Silicon variant has different performance characteristics.

### Entry Level (M1/M2/M3)

- **Models**: Focus on 7B models with INT4 quantization
- **Settings**: Use default GPU settings with INT4 quantization
- **Memory Management**: Clear cache between runs
- **Use Case**: Interactive chat, code completion, simple writing assistance

### Mid-Range (Pro variants)

- **Models**: 7B models at INT8/FP16 or 13B models at INT4
- **Settings**: GPU with INT4/INT8 depending on quality needs
- **Memory Management**: Can run multiple smaller models
- **Use Case**: Document analysis, content generation, longer contexts

### High-End (Max/Ultra variants)

- **Models**: 13B models at INT8/FP16 or 33B+ models at INT4
- **Settings**: GPU with highest quality quantization suitable for task
- **Advanced Usage**: Can handle multiple models or parallel inferences
- **Use Case**: Complex reasoning, research, fine-tuning

### Example Configurations

#### MacBook Air M2 (8GB)
```python
mx.set_default_device(mx.gpu)
model, tokenizer = load("llama-2-7b", quantization="int4")
```

#### MacBook Pro M2 Pro (16GB)
```python
mx.set_default_device(mx.gpu)
model, tokenizer = load("llama-2-13b", quantization="int4")
# Or for better quality with 7B
model, tokenizer = load("llama-2-7b", quantization="int8")
```

#### Mac Studio M1 Ultra (64GB)
```python
mx.set_default_device(mx.gpu)
# Full precision for best quality
model, tokenizer = load("llama-2-70b", quantization="int4")
# Or
model, tokenizer = load("llama-2-33b", quantization="int8")
```

## 7. Fine-tuning Optimization

MLX supports efficient fine-tuning on Apple Silicon.

### Memory Requirements

| Model Size | Technique | Minimum RAM |
|------------|-----------|-------------|
| 7B | Full Fine-tuning | 32GB |
| 7B | LoRA (r=16) | 16GB |
| 7B | QLoRA (INT4+LoRA) | 8GB |
| 13B | Full Fine-tuning | 64GB |
| 13B | LoRA (r=16) | 32GB |
| 13B | QLoRA (INT4+LoRA) | 16GB |

### Optimizing Fine-tuning

- Use gradient accumulation to fit larger batch sizes
- Set smaller learning rates for stable training
- Checkpoint models regularly during fine-tuning
- Use mixed precision training when available

### Example: QLoRA Fine-tuning

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.lora import apply_lora

# Load quantized model
model, tokenizer = load("llama-2-7b", quantization="int4")

# Apply LoRA
model = apply_lora(
    model,
    r=8,          # LoRA rank
    alpha=16,     # LoRA alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Set up optimizer (with reasonable defaults for memory efficiency)
optimizer = optim.AdamW(learning_rate=1e-4)

# Training loop would follow...
```

## 8. Python Environment Setup

Proper environment setup ensures optimal MLX performance.

### Installation Best Practices

- Use a dedicated Python environment (conda or venv)
- Install the latest MLX version
- Install MLX-LM for language model support

```bash
# Create conda environment
conda create -n mlx-env python=3.10
conda activate mlx-env

# Install latest MLX
pip install mlx

# Install MLX-LM
pip install mlx-lm
```

### Dependencies

- Ensure you have the latest macOS version (Sonoma recommended)
- Update Xcode and Command Line Tools
- Keep MLX and MLX-LM versions in sync

### Potential Issues

- Python version compatibility (3.9+ recommended)
- Old macOS versions may have limited Metal support
- Mixed dependency versions can cause conflicts

## 9. Monitoring and Profiling

Monitoring performance helps identify bottlenecks.

### Key Metrics

- **Tokens per Second**: Primary performance metric
- **Memory Usage**: Monitor with Activity Monitor
- **Compilation Time**: One-time cost on first run
- **Loading Time**: Should be fast after first load

### MLX Profiling

```python
# Enable MLX memory tracing (experimental)
mx.enable_memory_trace()

# Your code here
model, tokenizer = load("llama-2-7b", quantization="int4")
output = generate(model, tokenizer, "Your prompt", max_tokens=100)

# Get memory trace
memory_trace = mx.get_memory_trace()
print(memory_trace)
```

### Monitoring Tools

- **Activity Monitor**: Built-in macOS tool for memory/CPU/GPU monitoring
- **metal_optimizer.py**: Our utility for benchmarking and optimizing settings
- **powermetrics**: Terminal command to monitor power and performance (requires sudo)

## 10. Model Conversion and Formats

MLX supports multiple model formats and conversion utilities.

### Supported Formats

- **MLX Native Format**: Most efficient for MLX
- **Hugging Face Models**: Direct loading with `mlx_lm.convert`
- **Safetensors**: Efficient format for large models

### Conversion Process

```python
# Convert from Hugging Face to MLX format
from mlx_lm.convert import convert_hf_to_mlx

# Convert a model
convert_hf_to_mlx("meta-llama/Llama-2-7b", "llama-2-7b-mlx")

# Or use the command line
# python -m mlx_lm.convert --hf-path meta-llama/Llama-2-7b --mlx-path llama-2-7b-mlx
```

### Best Practices

- Convert models once and save in MLX format
- Use the latest conversion utilities
- Quantize after conversion for best results

## Conclusion

Following these best practices will help you achieve optimal performance when running MLX models on Apple Silicon. For automated optimization, use our performance utilities:

```bash
# Memory analysis
python performance_utils/memory/mlx/memory_analyzer.py --model llama-2-7b --quant int4

# Performance benchmarking
python performance_utils/benchmark/mlx/benchmark.py --model llama-2-7b --quant int4

# Metal optimization
python performance_utils/gpu/mlx/metal_optimizer.py --model llama-2-7b --optimize-quant --optimize-params
```