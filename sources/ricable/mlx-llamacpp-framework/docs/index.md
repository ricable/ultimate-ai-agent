# Running LLMs on Apple Silicon

Welcome to the comprehensive documentation for running Large Language Models on Apple Silicon. This project enables you to run, quantize, and fine-tune LLMs locally on your Mac using the llama.cpp and MLX frameworks.

## Why Run LLMs Locally?

Running LLMs on your own Apple Silicon hardware provides several advantages:

- **Privacy**: Your data never leaves your device
- **Cost**: No API usage fees or cloud compute costs
- **Speed**: Low latency without network overhead
- **Control**: Full control over model parameters and behavior
- **Offline Use**: Work without an internet connection
- **Customization**: Fine-tune models to your specific needs

## Getting Started

New to running LLMs on Apple Silicon? Start here:

- [Installation Guide](getting-started.md) - Set up your environment
- [Project Overview](project-overview.md) - Understand the project architecture
- [Framework Comparison](frameworks/framework-comparison.md) - Choose the right framework

## Core Documentation

- **Framework Guides**
  - [llama.cpp Guide](frameworks/llama-cpp-guide.md)
  - [MLX Guide](frameworks/mlx-guide.md)

- **Use Case Guides**
  - [Inference Guide](use-cases/inference-guide.md)
  - [Chat Applications](use-cases/chat-applications.md)
  - [Fine-tuning Guide](use-cases/fine-tuning-guide.md)

- **Hardware and Performance**
  - [Hardware Recommendations](hardware/hardware-recommendations.md)
  - [Performance Optimization](hardware/performance-optimization.md)

## Quick Reference

### Hardware Requirements

| Usage Level | Minimum | Recommended | Optimal |
|-------------|---------|-------------|---------|
| Basic Inference | MacBook Air M1/M2 (8GB) | MacBook Pro M1/M2 Pro (16GB) | Mac Studio M1/M2 Max/Ultra (32GB+) |
| Fine-tuning | MacBook Pro M1/M2 Pro (16GB) | MacBook Pro M1/M2 Max (32GB) | Mac Studio M1/M2 Ultra (64GB+) |

### Example: Text Generation with llama.cpp

```bash
cd llama.cpp-setup
./bin/main -m models/llama-2-7b-q4_0.gguf --metal -p "Tell me about Apple Silicon" -n 256
```

### Example: Text Generation with MLX

```python
from mlx_lm import load, generate

model, tokenizer = load("llama-2-7b", quantization="int4")
tokens = generate(model, tokenizer, "Tell me about Apple Silicon", max_tokens=256)
print(tokenizer.decode(tokens))
```

## Community Resources

- [Contributing Guide](contributing.md) - How to contribute to the project
- [GitHub Repository](https://github.com/yourusername/flow2) - Source code and issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.