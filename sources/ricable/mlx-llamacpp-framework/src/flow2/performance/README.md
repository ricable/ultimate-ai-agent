# Performance Optimization Utilities for LLMs on Apple Silicon

A comprehensive suite of tools and utilities for optimizing the performance of Large Language Models (LLMs) on Apple Silicon hardware using llama.cpp and MLX frameworks.

## Overview

This package provides specialized utilities for:

1. **Memory Usage Optimization**: Tools to analyze and optimize memory usage for different model sizes
2. **Benchmark Utilities**: Measure inference speed under different configurations
3. **GPU Acceleration**: Optimize Metal GPU acceleration on Apple Silicon
4. **Best Practices**: Documentation on maximizing performance on different Mac hardware
5. **Example Workflows**: Optimized configurations for different use cases

## Directory Structure

```
performance_utils/
├── benchmark/              # Benchmark utilities
│   ├── llama_cpp/          # Benchmarking tools for llama.cpp
│   └── mlx/                # Benchmarking tools for MLX
├── memory/                 # Memory optimization utilities
│   ├── llama_cpp/          # Memory analysis for llama.cpp
│   └── mlx/                # Memory analysis for MLX
├── gpu/                    # Metal GPU optimization
│   ├── llama_cpp/          # GPU optimization for llama.cpp
│   └── mlx/                # GPU optimization for MLX
├── docs/                   # Documentation
│   ├── llama_cpp/          # Best practices for llama.cpp
│   └── mlx/                # Best practices for MLX
└── examples/               # Example workflows
    ├── llama_cpp/          # Example scripts for llama.cpp
    └── mlx/                # Example scripts for MLX
```

## Key Features

### Memory Analysis and Optimization

- Analyze memory usage for models of different sizes
- Recommend optimal settings based on your hardware
- Optimize context length and batch size settings for memory efficiency
- Generate memory profile reports

### Performance Benchmarking

- Measure tokens per second across different configurations
- Compare CPU vs GPU performance
- Test impact of quantization on speed and quality
- Generate benchmark reports with optimal settings

### Metal GPU Optimization

- Optimize Metal settings for different Apple Silicon chips
- Fine-tune batch size and other parameters for maximum GPU performance
- Create optimized configuration scripts automatically
- Support for latest Metal features on M1/M2/M3 chips

### Example Workflows

- Optimized chat applications
- Batch inference scripts
- Memory-efficient fine-tuning examples
- Real-world usage patterns with optimal settings

## Installation and Requirements

### Prerequisites

- Apple Silicon Mac (M1, M2, or M3 family)
- macOS Sonoma or later (recommended)
- Python 3.9 or later

### Dependencies

- For llama.cpp utilities:
  - A compiled version of llama.cpp with Metal support
  - Python with numpy, matplotlib, psutil

- For MLX utilities:
  - MLX and MLX-LM packages
  - Python with numpy, matplotlib, psutil

```bash
# Install dependencies
pip install numpy matplotlib psutil tqdm

# Install MLX (for MLX utilities)
pip install mlx mlx-lm
```

## Quick Start

### Memory Analysis

```bash
# Analyze memory usage for llama.cpp model
python memory/llama_cpp/memory_analyzer.py --model /path/to/model.gguf

# Analyze memory usage for MLX model
python memory/mlx/memory_analyzer.py --model llama-2-7b --quant int4
```

### Benchmarking

```bash
# Benchmark llama.cpp model
python benchmark/llama_cpp/benchmark.py --model /path/to/model.gguf --output results.json

# Benchmark MLX model
python benchmark/mlx/benchmark.py --model llama-2-7b --quant int4 --output results.json
```

### GPU Optimization

```bash
# Optimize Metal settings for llama.cpp
python gpu/llama_cpp/metal_optimizer.py --model /path/to/model.gguf --script run_optimized.sh

# Optimize Metal settings for MLX
python gpu/mlx/metal_optimizer.py --model llama-2-7b --optimize-quant --optimize-params --script run_optimized.py
```

### Run Optimized Examples

```bash
# Run optimized chat with llama.cpp
./examples/llama_cpp/optimized_chat.sh /path/to/model.gguf

# Run optimized chat with MLX
python examples/mlx/optimized_chat.py --model llama-2-7b --quant int4

# Run batch inference with llama.cpp
python examples/llama_cpp/batch_inference.py --model /path/to/model.gguf --input prompts.txt --output results.jsonl

# Run batch inference with MLX
python examples/mlx/batch_inference.py --model llama-2-7b --quant int4 --input prompts.txt --output results.jsonl
```

## Documentation

See the `docs/` directory for comprehensive guides:

- [llama.cpp Performance Best Practices](docs/llama_cpp/performance_best_practices.md)
- [MLX Performance Best Practices](docs/mlx/performance_best_practices.md)

## Hardware-Specific Recommendations

### Entry Level (8GB RAM)

- **Hardware**: MacBook Air M1/M2, Mac Mini M1/M2
- **Models**: 7B with INT4/Q4_K quantization
- **Context Length**: Up to 2048 tokens
- **Framework**: MLX with INT4 quantization or llama.cpp with Q4_K

### Mid-Range (16GB RAM)

- **Hardware**: MacBook Pro M1/M2 Pro, Mac Mini M2 Pro
- **Models**: 7B models with INT8/Q8_0 or 13B models with INT4/Q4_K
- **Context Length**: Up to 4096 tokens
- **Framework**: MLX or llama.cpp, depending on use case

### High-End (32GB RAM)

- **Hardware**: MacBook Pro M1/M2 Max, Mac Studio M1/M2 Max
- **Models**: 13B models with INT8/Q8_0 or 33B models with INT4/Q4_K
- **Context Length**: Up to 8192 tokens
- **Framework**: MLX for research, llama.cpp for deployment

### Workstation (64GB+ RAM)

- **Hardware**: Mac Studio M1/M2 Ultra, Mac Pro
- **Models**: 33B models with INT8/Q8_0 or 70B models with INT4/Q4_K
- **Context Length**: Up to 32K tokens
- **Framework**: Either framework works well

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The llama.cpp project
- The MLX project and Apple ML team
- The wider Apple Silicon and LLM community