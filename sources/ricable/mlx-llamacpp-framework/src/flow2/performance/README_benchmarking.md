# LLM Benchmarking Tools for Apple Silicon

A comprehensive suite of benchmarking tools for comparing and optimizing Large Language Models (LLMs) on Apple Silicon hardware. This toolkit provides rigorous, reproducible methods for evaluating and comparing llama.cpp and MLX frameworks across different models, quantization levels, and hardware configurations.

## Key Features

- **Framework Comparison**: Direct comparison between llama.cpp and MLX performance
- **Quantization Analysis**: Evaluate the impact of different quantization levels on performance and quality
- **Hardware Parameter Optimization**: Test how context length, thread count, and batch size affect performance
- **Quality Evaluation**: Assess output quality across different configurations
- **Comprehensive Workflows**: End-to-end benchmark workflows with automated reporting
- **Visualization Tools**: Generate informative charts and reports from benchmark results

## Directory Structure

```
performance_utils/
├── benchmark/                    # Benchmarking tools
│   ├── llama_cpp/                # llama.cpp-specific benchmarks
│   │   └── benchmark.py          # Core llama.cpp benchmark script
│   ├── mlx/                      # MLX-specific benchmarks
│   │   └── benchmark.py          # Core MLX benchmark script
│   ├── framework_comparison.py   # Compare llama.cpp vs MLX
│   ├── quantization_comparison.py # Compare different quantization levels
│   ├── visualize_results.py      # Generate visualizations from results
│   └── benchmark_workflow.py     # End-to-end benchmark workflows
├── memory/                       # Memory analysis tools
│   ├── llama_cpp/                # llama.cpp memory analysis
│   │   └── memory_analyzer.py    # Memory usage analyzer for llama.cpp
│   └── mlx/                      # MLX memory analysis
│       └── memory_analyzer.py    # Memory usage analyzer for MLX
├── docs/                         # Documentation
│   ├── benchmarking_guide.md     # Comprehensive benchmarking guide
│   ├── llama_cpp/                # llama.cpp-specific documentation
│   └── mlx/                      # MLX-specific documentation
└── examples/                     # Example workflows
    └── benchmark_comparison_workflow.sh # Complete benchmark workflow example
```

## Prerequisites

- **Hardware**: Apple Silicon Mac (M1, M2, or M3 family)
- **Operating System**: macOS Sonoma or later (recommended)
- **Python**: 3.9 or later
- **Dependencies**:
  ```bash
  pip install numpy matplotlib pandas seaborn psutil tqdm
  ```
- **Framework-specific dependencies**:
  - llama.cpp: Compiled with Metal support
  - MLX: `pip install mlx mlx-lm`

## Quick Start

### Framework Comparison

Compare llama.cpp and MLX performance:

```bash
python benchmark/framework_comparison.py \
  --llama-model /path/to/model.gguf \
  --mlx-model llama-2-7b \
  --output comparison_results
```

### Quantization Study

Evaluate different quantization levels:

```bash
python benchmark/quantization_comparison.py \
  --llama-base /path/to/base_model.gguf \
  --mlx-model llama-2-7b \
  --quants q4_k,q8_0,f16,int4,int8 \
  --output quantization_results
```

### End-to-End Workflow

Run a comprehensive benchmark workflow:

```bash
python benchmark/benchmark_workflow.py comprehensive \
  --llama-model /path/to/model.gguf \
  --mlx-model llama-2-7b \
  --output full_benchmark
```

### Run Example Workflow

```bash
cd examples
./benchmark_comparison_workflow.sh
```

## Benchmark Metrics

The benchmarking tools measure:

- **Inference Speed**: Tokens per second during generation
- **Memory Usage**: Peak and average memory consumption
- **Load Time**: Time required to load the model
- **Quality Metrics**: Sample responses to standardized prompts
- **Efficiency**: Performance per memory usage (tokens/s/GB)

## Visualization

Generate visualizations from benchmark results:

```bash
python benchmark/visualize_results.py \
  --input path/to/results.json \
  --output visualization_output
```

This creates:
- Performance comparison charts
- Memory usage visualizations
- Quality assessment reports
- Interactive HTML reports

## Documentation

For detailed information, see:

- [Comprehensive Benchmarking Guide](docs/benchmarking_guide.md): Complete guide to all benchmarking tools
- [llama.cpp Performance Best Practices](docs/llama_cpp/performance_best_practices.md): Optimization tips for llama.cpp
- [MLX Performance Best Practices](docs/mlx/performance_best_practices.md): Optimization tips for MLX

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

## Best Practices

For the most accurate benchmarks:

1. **Consistent Environment**: Minimize background activity during testing
2. **Multiple Runs**: Use at least 3 runs per configuration (`--runs 3`)
3. **Equivalent Models**: Compare the same model architecture across frameworks
4. **Cooling Considerations**: For extended benchmarks, ensure adequate cooling
5. **Balanced Evaluation**: Consider both performance and output quality
6. **Diverse Testing**: Test with different prompt types and lengths

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The llama.cpp project
- The MLX project and Apple ML team
- The wider Apple Silicon and LLM community