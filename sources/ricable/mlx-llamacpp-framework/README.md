# MLX & LlamaCpp Framework

> Comprehensive AI model training and inference toolkit with MLX and LlamaCpp frameworks, Flash Attention optimization, and performance utilities for Apple Silicon

## 🚀 Features

- **🔥 Multi-Framework Support**: MLX (Apple Silicon) + LlamaCpp (Cross-platform)
- **⚡ Flash Attention**: Memory-efficient attention optimization  
- **🎯 Fine-tuning**: LoRA, QLoRA, and full fine-tuning capabilities
- **💬 Chat Interfaces**: CLI and Web-based chat with real-time streaming
- **📊 Performance Tools**: Comprehensive benchmarking and analysis
- **🛠️ Easy Installation**: Simple pip install with dependency management

## ⚡ Quick Start

### Installation
```bash
# Clone and install
git clone https://github.com/ricable/mlx-llamacpp-framework.git
cd mlx-llamacpp-framework
pip install -e .[all]

# Or framework-specific
pip install -e .[mlx]      # MLX for Apple Silicon
pip install -e .[llamacpp] # LlamaCpp for any platform
```

### Basic Usage
```python
import flow2

# Check what's available
print(f"MLX: {flow2.MLX_AVAILABLE}")
print(f"LlamaCpp: {flow2.LLAMACPP_AVAILABLE}")  
print(f"Flash Attention: {flow2.FLASH_ATTENTION_AVAILABLE}")

# Quick inference
if flow2.MLX_AVAILABLE:
    from flow2.frameworks.mlx import load_mlx_model, generate_completion
    model, tokenizer = load_mlx_model("models/mlx/tinyllama-1.1b-chat")
    response = generate_completion(model, tokenizer, "Hello!")
```

## 📁 Project Structure

```
src/flow2/
├── 🧠 core/           # Flash Attention & benchmarks
├── 🔧 frameworks/     # MLX & LlamaCpp implementations  
├── 💬 chat/           # Interactive interfaces
├── 📊 performance/    # Benchmarking tools
└── 🛠️ utils/          # Model management
```

## 🎯 Examples

### Fine-tuning with Flash Attention
```bash
cd examples/mlx
python run_mlx_finetune_improved.py --use-flash-attention --prepare-data
```

### Framework Comparison
```bash
cd examples/workflows  
bash benchmark_comparison_workflow.sh
```

### Interactive Chat
```bash
# MLX chat (Apple Silicon)
python src/flow2/chat/interfaces/cli/mlx_chat.py

# LlamaCpp chat (Any platform)
python src/flow2/chat/interfaces/cli/llamacpp_chat.py
```

## 🏗️ Key Components

### Multi-Framework Training
- **LoRA**: Memory-efficient fine-tuning
- **QLoRA**: Quantized LoRA for ultra-low memory
- **Full Fine-tuning**: Complete model retraining
- **Flash Attention**: Automatic memory optimization

### Performance Analysis  
- **Framework Comparison**: MLX vs LlamaCpp benchmarks
- **Quantization Analysis**: Quality vs speed trade-offs
- **Hardware Scaling**: Multi-core optimization
- **Interactive Reports**: HTML dashboards

### Chat Interfaces
- **CLI**: Terminal-based with history
- **Web**: Browser interface with streaming
- **Templates**: Customizable prompts
- **Multi-model**: Switch between frameworks

## 🔧 Hardware Support

### Apple Silicon (MLX)
- **Optimized for**: M1/M2/M3/M4 chips
- **Memory**: 16GB+ recommended for training
- **OS**: macOS 12.0+ (Monterey)
- **Acceleration**: Metal GPU acceleration

### Cross-Platform (LlamaCpp)
- **CPU**: Multi-core (8+ recommended)
- **Memory**: 16GB+ for larger models  
- **GPU**: Optional CUDA/OpenCL support
- **OS**: macOS, Linux, Windows

## 📖 Documentation

- **[Complete Guide](CLAUDE.md)**: Detailed documentation
- **[Examples](examples/)**: Ready-to-run scripts
- **[API Reference](src/flow2/)**: Framework APIs
- **[Performance](docs/)**: Benchmarking guides

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- **Apple MLX Team**: MLX framework for Apple Silicon
- **LlamaCpp Contributors**: Cross-platform inference engine  
- **Philip Turner**: Metal Flash Attention research
- **Hugging Face**: Model hosting and ecosystem

---

<div align="center">

**[📚 Documentation](CLAUDE.md)** • **[🚀 Examples](examples/)** • **[🐛 Issues](https://github.com/ricable/mlx-llamacpp-framework/issues)** • **[💬 Discussions](https://github.com/ricable/mlx-llamacpp-framework/discussions)**

</div>