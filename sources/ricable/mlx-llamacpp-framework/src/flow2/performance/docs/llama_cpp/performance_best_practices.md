# llama.cpp Performance Best Practices for Apple Silicon

This guide covers best practices for maximizing llama.cpp performance on Apple Silicon Macs.

## Table of Contents
1. [Metal GPU Acceleration](#1-metal-gpu-acceleration)
2. [Memory Optimization](#2-memory-optimization)
3. [Quantization Selection](#3-quantization-selection)
4. [Context Length Management](#4-context-length-management)
5. [Batch Size Tuning](#5-batch-size-tuning)
6. [Thread Count Optimization](#6-thread-count-optimization)
7. [Hardware-Specific Recommendations](#7-hardware-specific-recommendations)
8. [Building from Source](#8-building-from-source)
9. [Fine-tuning Optimization](#9-fine-tuning-optimization)
10. [Monitoring and Debugging](#10-monitoring-and-debugging)

## 1. Metal GPU Acceleration

Metal is Apple's graphics and compute API, which enables llama.cpp to leverage the GPU on Apple Silicon for significantly faster inference.

### Key Settings

- **Enable Metal**: Always use the `--metal` flag when running on Apple Silicon
- **Metal Matrix Multiplication**: Enable `--metal-mmq` for additional performance boost on most models
- **Performance Impact**: 2-3x speedup over CPU-only inference

### Example Command

```bash
./main \
  -m models/llama-2-7b-q4_k.gguf \
  --metal \
  --metal-mmq \
  -c 2048 \
  -b 512 \
  -t 4
```

### Chip-Specific Recommendations

| Apple Silicon Chip | Metal Flags                | Expected Speedup |
|--------------------|----------------------------|-----------------|
| M1                 | `--metal`                  | 2-2.5x          |
| M1 Pro/Max         | `--metal --metal-mmq`      | 2.5-3x          |
| M2                 | `--metal --metal-mmq`      | 2.5-3x          |
| M2 Pro/Max/Ultra   | `--metal --metal-mmq`      | 3-4x            |
| M3                 | `--metal --metal-mmq`      | 3.5-4.5x        |
| M3 Pro/Max/Ultra   | `--metal --metal-mmq`      | 4-5x            |

### Common Issues

- If you see "Metal: Not supported", ensure you're using an Apple Silicon Mac
- If performance is lower than expected, try rebuilding llama.cpp with proper Metal support
- For specific errors with Metal, check the llama.cpp build log and ensure Metal framework is properly detected

## 2. Memory Optimization

Memory management is critical for running large models on memory-constrained systems.

### Memory Usage Formula

As a rough estimate, a model's memory usage during inference can be calculated as:
- Base model memory + Context window overhead
- Example: A 7B model with Q4_K quantization uses ~4GB for the model plus additional memory for the context window

### Memory-Saving Techniques

1. **Reduce Batch Size**: Lower `-b` values reduce memory at the cost of some throughput
2. **Limit Context Length**: Only use the context length you need with `-c` parameter
3. **Offload to Disk**: For very large models, enable disk offloading with `--mlock 0 --no-mmap`
4. **Clean Cache**: Between multiple inference runs, allow time for memory cleanup

### Hardware Recommendations

| RAM | Maximum Recommended Model Size |
|-----|--------------------------------|
| 8GB | 7B with INT4/Q4_K quantization |
| 16GB | 13B with INT4/Q4_K or 7B with INT8/Q8_0 |
| 32GB | 33B with INT4/Q4_K or 13B with INT8/Q8_0 |
| 64GB | 70B with INT4/Q4_K or 33B with INT8/Q8_0 |

## 3. Quantization Selection

Quantization significantly reduces memory requirements with minimal quality impact when done properly.

### Quantization Types

| Quantization | Size Reduction | Quality Impact | Use Case |
|--------------|----------------|----------------|----------|
| F16 (No Quantization) | Baseline | None | When maximum quality is required |
| Q8_0 | ~50% | Minimal | High-quality, memory available |
| Q6_K | ~62.5% | Very low | Good balance for most uses |
| Q5_K | ~68.75% | Low | Good balance for most uses |
| Q4_K | ~75% | Moderate | **Recommended default** |
| Q3_K | ~81.25% | High | Memory-constrained systems |
| Q2_K | ~87.5% | Very high | Extremely memory-constrained |

### Hardware-Based Recommendations

| Hardware | RAM | Recommended Quantization |
|----------|-----|--------------------------|
| MacBook Air (8GB) | 8GB | Q4_K/Q3_K |
| MacBook Pro (16GB) | 16GB | Q5_K/Q4_K |
| MacBook Pro (32GB) | 32GB | Q6_K/Q5_K |
| Mac Studio (64GB+) | 64GB+ | Q8_0/Q6_K |

### Converting and Quantizing Models

```bash
# Convert from Hugging Face to GGUF and quantize in one step
python convert.py --outtype q4_k --outfile models/llama-2-7b-q4_k.gguf meta-llama/Llama-2-7b

# Or quantize existing GGUF model
python quantize.py --model models/llama-2-7b.gguf --outfile models/llama-2-7b-q4_k.gguf --type q4_k
```

## 4. Context Length Management

Context length affects both memory usage and inference speed.

### Impact on Performance

- **Memory Usage**: Longer contexts require significantly more memory
- **Inference Speed**: Longer contexts reduce tokens/second due to attention computation
- **Quality**: Longer contexts provide better responses for complex tasks

### Recommended Context Lengths

| Use Case | Recommended Context Length |
|----------|----------------------------|
| Simple Q&A | 1024-2048 |
| Chat | 2048-4096 |
| Document Analysis | 4096-8192 |
| Complex Tasks | 8192+ |

### Memory Impact

Each doubling of context length increases memory usage by approximately:
- 20-30% for 7B models
- 30-40% for 13B models
- 40-50% for 33B+ models

### Setting Context Length

```bash
# Set context length to 4096 tokens
./main -m models/llama-2-7b-q4_k.gguf -c 4096
```

## 5. Batch Size Tuning

Batch size controls how many tokens are processed in parallel during generation.

### Impact on Performance

- **Higher Batch Size**: Increased throughput (tokens/second) but more memory
- **Lower Batch Size**: Reduced memory usage but lower throughput

### Recommended Batch Sizes

| Hardware | Recommended Batch Size Range |
|----------|-----------------------------|
| M1       | 256-512 |
| M1 Pro   | 512-768 |
| M1 Max   | 512-1024 |
| M2       | 512-768 |
| M2 Pro   | 768-1024 |
| M2 Max   | 1024-1536 |
| M3       | 768-1024 |
| M3 Pro+  | 1024-2048 |

### Finding Optimal Batch Size

Use our `metal_optimizer.py` tool to automatically find the optimal batch size:

```bash
python performance_utils/gpu/llama_cpp/metal_optimizer.py --model models/llama-2-7b-q4_k.gguf
```

## 6. Thread Count Optimization

Setting the correct number of CPU threads is important even when using Metal acceleration.

### Impact on Performance

- **Too Few Threads**: Underutilization of CPU resources
- **Too Many Threads**: Overhead from thread management, potential throttling

### Recommended Thread Counts

| Hardware | CPU Cores | Recommended Threads |
|----------|-----------|---------------------|
| M1       | 8 (4+4)   | 4 |
| M1 Pro   | 8-10      | 4-6 |
| M1 Max   | 10        | 6 |
| M1 Ultra | 20        | 10-12 |
| M2       | 8 (4+4)   | 4 |
| M2 Pro   | 10-12     | 6-8 |
| M2 Max   | 12        | 8 |
| M2 Ultra | 24        | 12-16 |
| M3       | 8 (4+4)   | 4 |
| M3 Pro+  | 12-16     | 8-12 |

### Setting Thread Count

```bash
# Set thread count to 6
./main -m models/llama-2-7b-q4_k.gguf -t 6
```

## 7. Hardware-Specific Recommendations

Each Apple Silicon variant has different performance characteristics.

### Entry Level (M1/M2/M3)

- **Models**: Focus on 7B models with Q4_K/Q3_K quantization
- **Context**: Keep context length ≤ 4096 tokens
- **Settings**: `--metal -t 4 -b 512 -c 2048`
- **Memory Management**: Close other applications when running models

### Mid-Range (Pro variants)

- **Models**: 7B models at Q6_K/Q5_K or 13B models at Q4_K
- **Context**: Context length up to 8192 tokens
- **Settings**: `--metal --metal-mmq -t 6 -b 768 -c 4096`
- **Multi-tasking**: Can run model while using other applications

### High-End (Max/Ultra variants)

- **Models**: 13B models at Q6_K or 33B+ models at Q4_K
- **Context**: Context length up to 32K tokens
- **Settings**: `--metal --metal-mmq -t 8 -b 1024 -c 8192`
- **Advanced Usage**: Can handle multiple models or simultaneous inferences

### Optimized Command Examples

#### MacBook Air M2 (8GB)
```bash
./main -m models/llama-2-7b-q4_k.gguf --metal -t 4 -b 512 -c 2048 --temp 0.7 --repeat_penalty 1.1
```

#### MacBook Pro M2 Pro (16GB)
```bash
./main -m models/llama-2-13b-q4_k.gguf --metal --metal-mmq -t 6 -b 768 -c 4096 --temp 0.7 --repeat_penalty 1.1
```

#### Mac Studio M1 Ultra (64GB)
```bash
./main -m models/llama-2-70b-q4_k.gguf --metal --metal-mmq -t 12 -b 1536 -c 8192 --temp 0.7 --repeat_penalty 1.1
```

## 8. Building from Source

Building llama.cpp from source with proper optimizations can significantly improve performance.

### Optimal Build Instructions

```bash
# Clone repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with Metal support
mkdir build && cd build
cmake .. -DLLAMA_METAL=ON -DLLAMA_CUBLAS=OFF -DLLAMA_OPENBLAS=OFF
cmake --build . --config Release

# Verify Metal support
./main -h | grep Metal
```

### Critical Build Flags

- `-DLLAMA_METAL=ON`: Enable Metal acceleration (critical)
- `-DCMAKE_BUILD_TYPE=Release`: Enable optimization
- `-DLLAMA_BLAS=ON`: Enable optimized linear algebra (can improve CPU performance)
- `-DLLAMA_AVX=ON`: Enable AVX instructions for Rosetta 2 on Intel Macs

### Common Build Issues

- **Metal Not Detected**: Ensure XCode command line tools are installed
- **Build Failures**: Update to latest version of llama.cpp
- **Slow Performance**: Check if Metal is properly enabled with `./main -h | grep Metal`

## 9. Fine-tuning Optimization

Fine-tuning on Apple Silicon requires careful resource management.

### Memory Requirements for LoRA Fine-tuning

| Model Size | Technique | Minimum RAM |
|------------|-----------|-------------|
| 7B | LoRA (r=16) | 16GB |
| 7B | QLoRA (INT4+LoRA) | 8GB |
| 13B | LoRA (r=16) | 28GB |
| 13B | QLoRA (INT4+LoRA) | 12GB |

### Optimizing Fine-tuning

- Use the `--thread-count` parameter to control CPU utilization
- For models ≥13B, use QLoRA with int4 quantization
- Reduce batch size to fit in memory
- Enable gradient checkpointing when available

### Example LoRA Fine-tuning Command

```bash
./llama-finetune \
  --model-base ./models/llama-2-7b-q4_0.gguf \
  --lora-rank 8 \
  --lora-layers all \
  --data-train ./data/train.jsonl \
  --data-val ./data/val.jsonl \
  --batch-size 4 \
  --thread-count 4 \
  --lora-out ./lora-finetune.bin
```

## 10. Monitoring and Debugging

Properly monitoring performance helps identify bottlenecks.

### Key Metrics to Monitor

- **Tokens per Second**: Primary performance metric
- **Memory Usage**: Monitor with Activity Monitor
- **Temperature**: Watch for thermal throttling on laptops
- **Load Time**: If slow, check disk speed and quantization

### Debugging Common Issues

- **Low Performance**: Check if Metal is enabled and working
- **Out of Memory**: Reduce batch size, context length, or use higher quantization
- **Slow First Run**: Normal - Metal shader compilation occurs on first run
- **Thermal Throttling**: Improve cooling, reduce thread count

### Monitoring Tools

- **Activity Monitor**: Built-in macOS tool for memory/CPU monitoring
- **metal_optimizer.py**: Our utility for benchmarking and optimizing settings
- **powermetrics**: Terminal command to monitor power and performance (requires sudo)

## Conclusion

Following these best practices will help you achieve optimal performance when running llama.cpp models on Apple Silicon. For automated optimization, use our performance utilities:

```bash
# Memory analysis
python performance_utils/memory/llama_cpp/memory_analyzer.py --model models/llama-2-7b-q4_k.gguf

# Performance benchmarking
python performance_utils/benchmark/llama_cpp/benchmark.py --model models/llama-2-7b-q4_k.gguf

# Metal optimization
python performance_utils/gpu/llama_cpp/metal_optimizer.py --model models/llama-2-7b-q4_k.gguf
```