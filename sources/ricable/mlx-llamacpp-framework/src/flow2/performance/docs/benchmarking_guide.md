# Comprehensive Benchmarking Guide for LLMs on Apple Silicon

This guide provides detailed instructions for benchmarking and comparing the performance of Large Language Models (LLMs) using llama.cpp and MLX frameworks on Apple Silicon hardware.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Benchmarking Tools](#benchmarking-tools)
4. [Framework Comparison](#framework-comparison)
5. [Quantization Comparison](#quantization-comparison)
6. [Hardware Parameter Scaling](#hardware-parameter-scaling)
7. [Quality Evaluation](#quality-evaluation)
8. [Comprehensive Benchmarking](#comprehensive-benchmarking)
9. [Visualizing Results](#visualizing-results)
10. [Methodology](#methodology)
11. [Best Practices](#best-practices)
12. [Example Workflows](#example-workflows)

## Introduction

The benchmarking tools in this package enable rigorous performance testing and comparison of LLM inference across different frameworks, quantization levels, and hardware parameters. They help you:

- Compare llama.cpp and MLX frameworks head-to-head
- Evaluate the impact of different quantization levels on performance and quality
- Optimize hardware parameters like context length, thread count, and batch size
- Assess the quality impact of different optimization strategies
- Generate comprehensive reports with visualizations

## Prerequisites

Before running benchmarks, ensure you have:

- **Apple Silicon Mac** (M1, M2, or M3 family)
- **Python 3.9+** with required packages:
  ```bash
  pip install numpy matplotlib pandas seaborn psutil tqdm
  ```
- **llama.cpp** compiled with Metal support (for llama.cpp benchmarks)
- **MLX and MLX-LM** installed (for MLX benchmarks):
  ```bash
  pip install mlx mlx-lm
  ```
- **Models**:
  - llama.cpp: GGUF format models (e.g., Llama 2, Mistral, etc.)
  - MLX: Compatible models (either downloaded via mlx-lm or converted from Hugging Face)

## Benchmarking Tools

The benchmarking suite includes the following tools:

### Individual Tools

- `benchmark/llama_cpp/benchmark.py` - Benchmark llama.cpp models
- `benchmark/mlx/benchmark.py` - Benchmark MLX models
- `memory/llama_cpp/memory_analyzer.py` - Analyze memory usage for llama.cpp
- `memory/mlx/memory_analyzer.py` - Analyze memory usage for MLX

### Comparison Tools

- `benchmark/framework_comparison.py` - Compare llama.cpp and MLX performance
- `benchmark/quantization_comparison.py` - Compare different quantization levels
- `benchmark/visualize_results.py` - Generate visualizations from benchmark results

### Workflow Tool

- `benchmark/benchmark_workflow.py` - Run end-to-end benchmark workflows

## Framework Comparison

The framework comparison tool allows you to directly compare the performance of llama.cpp and MLX models with equivalent parameters.

### Basic Usage

```bash
python benchmark/framework_comparison.py \
  --llama-model /path/to/model.gguf \
  --mlx-model llama-2-7b \
  --output framework_comparison_results
```

### Additional Options

- `--ctx 2048,4096,8192` - Test multiple context lengths
- `--threads 4,8,16` - Test multiple thread counts (llama.cpp)
- `--batch 512,1024` - Test multiple batch sizes (llama.cpp)
- `--quant q4_k` - Specify quantization (will be mapped between frameworks)
- `--no-metal` - Disable Metal acceleration
- `--quality-test` - Include quality evaluation
- `--quick` - Run a faster benchmark with fewer configurations

### Output

The tool generates:
- Performance metrics (tokens/second, memory usage)
- Side-by-side comparison charts
- Quality comparison if requested
- HTML report with all results

## Quantization Comparison

This tool evaluates the impact of different quantization levels on both performance and output quality.

### Basic Usage

```bash
python benchmark/quantization_comparison.py \
  --llama-base /path/to/base_model.gguf \
  --mlx-model llama-2-7b \
  --quants q4_k,q8_0,f16,int4,int8 \
  --output quantization_results
```

### Additional Options

- `--ctx 2048` - Set context length
- `--threads 4` - Set thread count (llama.cpp)
- `--n-predict 256` - Number of tokens to generate
- `--skip-quality` - Skip quality evaluation
- `--runs 3` - Number of runs per configuration

### Output

The tool generates:
- Performance comparison across quantization levels
- Memory usage comparison
- Quality assessment (if not skipped)
- Visualizations of the trade-offs
- Recommendations for different use cases

## Hardware Parameter Scaling

This workflow tests how performance scales with different hardware parameters (context length, thread count, batch size).

### Basic Usage

```bash
python benchmark/benchmark_workflow.py hardware-scaling \
  --llama-model /path/to/model.gguf \
  --framework llama \
  --ctx 2048,4096,8192 \
  --threads 4,8,16 \
  --batch 512,1024,2048 \
  --output hardware_scaling_results
```

You can also test MLX:

```bash
python benchmark/benchmark_workflow.py hardware-scaling \
  --mlx-model llama-2-7b \
  --framework mlx \
  --quants none,int4,int8 \
  --output hardware_scaling_results
```

### Output

This workflow generates:
- Performance scaling charts for each parameter
- Heatmaps showing parameter interactions
- Optimal configuration recommendations
- Combined HTML report

## Quality Evaluation

This workflow focuses on evaluating the output quality across different configurations.

### Basic Usage

```bash
python benchmark/benchmark_workflow.py quality-evaluation \
  --llama-model /path/to/model.gguf \
  --mlx-model llama-2-7b \
  --quants q4_k,q8_0,f16,int4,int8 \
  --output quality_results
```

### Output

The quality evaluation generates:
- Sample model outputs for comparison
- Response time measurements
- Quality assessment across configurations
- HTML report with all quality results

## Comprehensive Benchmarking

For a complete evaluation, the comprehensive workflow runs all benchmark types and generates a combined report.

### Basic Usage

```bash
python benchmark/benchmark_workflow.py comprehensive \
  --llama-model /path/to/model.gguf \
  --mlx-model llama-2-7b \
  --quants q4_k,q8_0,f16,int4,int8 \
  --output comprehensive_results
```

### Output

This workflow generates:
- Complete framework comparison
- Quantization analysis
- Hardware parameter optimization
- Quality evaluation
- Comprehensive HTML report with recommendations

## Visualizing Results

You can visualize results from any benchmark using the visualization tool:

```bash
python benchmark/visualize_results.py \
  --input path/to/results.json \
  --output visualization_output \
  --type auto
```

### Visualization Types

- `--type framework` - Framework comparison visualizations
- `--type quantization` - Quantization comparison visualizations
- `--type comprehensive` - Comprehensive benchmark visualizations
- `--type single` - Single benchmark run visualizations
- `--type auto` - Auto-detect the result type
- `--type all` - Generate all possible visualizations

## Methodology

The benchmarking tools use the following methodology:

1. **Inference Speed**: Measured in tokens per second during generation
2. **Memory Usage**: Peak memory consumption during inference
3. **Quality Assessment**: Sample responses to standardized prompts
4. **Balanced Scoring**: Combined metrics that balance speed and memory
5. **Multiple Runs**: Each test is run multiple times to ensure consistency
6. **Parameter Optimization**: Testing different parameters to find optimal settings

For quality assessment, the tools use:
- Factual questions
- Coding problems
- Creative writing prompts
- Knowledge-intensive questions
- Multi-step reasoning tasks

## Best Practices

For accurate and meaningful benchmarks:

1. **Consistent Environment**: Run benchmarks on a system with minimal background activity
2. **Multiple Runs**: Use at least 3 runs per configuration (`--runs 3`)
3. **Equivalent Models**: Compare the same model architecture across frameworks
4. **Quantization Mapping**: Use equivalent quantization levels (e.g., Q4_K ≈ INT4)
5. **Hardware Considerations**: Test parameters relevant to your hardware (threads, context)
6. **Quality-Speed Balance**: Consider both performance and output quality
7. **Diverse Prompts**: Test with different prompt types and lengths
8. **Thermal Stability**: For long-running benchmarks, monitor for thermal throttling

## Example Workflows

### Basic Performance Check

Quick performance assessment for a model:

```bash
python benchmark/llama_cpp/benchmark.py \
  --model /path/to/model.gguf \
  --quick \
  --output quick_benchmark.json
```

### Framework Selection Decision

Determine which framework is best for your use case:

```bash
python benchmark/benchmark_workflow.py framework-comparison \
  --llama-model /path/to/model.gguf \
  --mlx-model llama-2-7b \
  --quant q4_k \
  --output framework_selection
```

### Finding Optimal Quantization

Determine the best quantization level for your needs:

```bash
python benchmark/benchmark_workflow.py quantization-study \
  --llama-base /path/to/base_model.gguf \
  --quants q2_k,q3_k,q4_k,q5_k,q6_k,q8_0,f16 \
  --output quantization_study
```

### Optimizing for Memory-Constrained Device

For memory-constrained systems:

```bash
python benchmark/benchmark_workflow.py hardware-scaling \
  --llama-model /path/to/model.gguf \
  --framework llama \
  --ctx 1024,2048,4096 \
  --threads 2,4,8 \
  --batch 128,256,512 \
  --output memory_optimization
```

### Quality-Critical Application

For applications where output quality is paramount:

```bash
python benchmark/benchmark_workflow.py quality-evaluation \
  --llama-model /path/to/model.gguf \
  --mlx-model llama-2-7b \
  --quants q8_0,f16,int8 \
  --n-predict 512 \
  --output quality_critical
```

### Full System Evaluation

For a complete evaluation of all aspects:

```bash
python benchmark/benchmark_workflow.py comprehensive \
  --llama-model /path/to/model.gguf \
  --mlx-model llama-2-7b \
  --ctx 2048,4096,8192 \
  --threads 4,8,16 \
  --batch 512,1024 \
  --quants q4_k,q8_0,f16,int4,int8 \
  --output full_evaluation
```