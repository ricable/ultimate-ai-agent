# Getting Started with LLMs on Mac Silicon

This guide will help you get started with running Large Language Models on your Mac Silicon machine using the tools and scripts we've created.

## Prerequisites

- A Mac with Apple Silicon (M1, M2, or M3 series)
- macOS 12.0 (Monterey) or later
- At least 8GB RAM (16GB+ recommended)
- 10GB+ free storage space

## Quick Start

### 1. Setting Up Your Environment

First, let's set up both frameworks:

```bash
# Create a working directory
mkdir -p ~/llm-mac-silicon
cd ~/llm-mac-silicon

# Clone the repository
git clone https://github.com/yourusername/llm-mac-silicon.git .

# Set up llama.cpp
./llama.cpp-setup/scripts/setup.sh

# Set up MLX
./mlx-setup/scripts/setup.sh
```

### 2. Downloading Your First Model

For a quick start, download a smaller model that runs well on most Mac hardware:

```bash
# Download a 7B model for llama.cpp
./model_utils/download-model.sh --framework llama.cpp --model mistral-7b-v0.1-q4_0

# Download the same model for MLX
./model_utils/download-model.sh --framework mlx --model mistral-7b-v0.1
```

### 3. Running Inference

Try a simple inference to make sure everything is working:

```bash
# Using llama.cpp
python inference_scripts/llama_cpp/inference.py --model models/llama.cpp/mistral-7b-v0.1-q4_0.gguf --prompt "Explain quantum computing to me" --max-tokens 256

# Using MLX
python inference_scripts/mlx/inference.py --model models/mlx/mistral-7b-v0.1 --prompt "Explain quantum computing to me" --max-tokens 256
```

### 4. Starting a Chat Session

For interactive use, try the chat application:

```bash
# Using llama.cpp
python inference_scripts/llama_cpp/inference.py --model models/llama.cpp/mistral-7b-v0.1-q4_0.gguf --chat --system-prompt "You are a helpful assistant"

# Using MLX
python inference_scripts/mlx/inference.py --model models/mlx/mistral-7b-v0.1 --chat --system-prompt "You are a helpful assistant"
```

## Choosing the Right Model Size

Based on your Mac's hardware:

| Mac Configuration | RAM | Recommended Models |
|-------------------|-----|-------------------|
| 8GB RAM | 8GB | 7B models with INT4 quantization |
| 16GB RAM | 16GB | 7B models with INT8/FP16, 13B models with INT4 |
| 32GB RAM | 32GB | 13B models with INT8/FP16, 33B models with INT4 |
| 64GB+ RAM | 64GB+ | Most models including 70B with appropriate quantization |

## Framework Selection Guide

### Choose llama.cpp when:
- You need cross-platform compatibility
- You want maximum control over quantization
- You're building a deployment-focused application
- C/C++ integration is important

### Choose MLX when:
- You're working exclusively on Mac
- Python workflow integration is important
- You want the best performance on Apple Silicon
- You need a more PyTorch/JAX-like API

## Next Steps

### Fine-tuning a Model

To personalize a model for your specific task:

```bash
# Prepare your training data
mkdir -p data/finetune
# Create a JSONL file with your training data
# Format: {"prompt": "input", "response": "desired output"}

# Fine-tune with llama.cpp (LoRA)
cd fine_tuning_utils/llamacpp
python lora_finetune.py --model ../../models/llama.cpp/mistral-7b-v0.1-q4_0.gguf --data ../../data/finetune/train.jsonl --output ../../models/lora/my-finetune.bin

# Fine-tune with MLX (QLoRA)
cd fine_tuning_utils/mlx
python qlora_finetune.py --model ../../models/mlx/mistral-7b-v0.1 --data ../../data/finetune/train.jsonl --output ../../models/mlx/my-finetune
```

### Performance Optimization

To get the most out of your hardware:

```bash
# Benchmark your model with different settings
python performance_optimization/benchmark.py --model models/llama.cpp/mistral-7b-v0.1-q4_0.gguf --framework llama.cpp

# Find optimal parameters for your hardware
python performance_optimization/optimize.py --model models/mlx/mistral-7b-v0.1 --framework mlx --hardware "M1 Pro 16GB"
```

### Comparing Frameworks

To decide which framework works best for your use case:

```bash
# Run a full benchmark comparison
python benchmarking_tools/compare_frameworks.py --llama-model models/llama.cpp/mistral-7b-v0.1-q4_0.gguf --mlx-model models/mlx/mistral-7b-v0.1 --tasks "completion,chat,embeddings"
```

## Troubleshooting

### Common Issues

- **Out of memory errors**: Try a smaller model or more aggressive quantization
- **Slow inference**: Enable Metal acceleration and optimize batch size
- **Model loading fails**: Check model format compatibility with your framework version
- **Metal acceleration not working**: Update to latest macOS and framework versions

### Getting Help

- Check the documentation in `docs/`
- Look for specific framework issues in `docs/troubleshooting.md`
- Consult the community resources listed in `docs/resources.md`

## Keeping Updated

As Apple Silicon and these frameworks evolve, check for updates:

```bash
# Update your local repository
git pull

# Update llama.cpp
./llama.cpp-setup/scripts/update.sh

# Update MLX
./mlx-setup/scripts/update.sh
```

## Contributing

We welcome contributions to improve these tools! See `CONTRIBUTING.md` for guidelines on how to submit improvements.