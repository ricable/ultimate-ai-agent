# Flow2: Multi-Framework AI Toolkit

## Project Overview
Flow2 is a comprehensive AI model training and inference toolkit with support for MLX, LlamaCpp, and HuggingFace frameworks, featuring Flash Attention optimization, MPS acceleration for Apple Silicon, and extensive performance utilities across all major AI frameworks.

## Build Commands
- `python -m pip install -e .`: Install package in development mode
- `python -m pip install -e .[mlx]`: Install with MLX dependencies  
- `python -m pip install -e .[llamacpp]`: Install with LlamaCpp dependencies
- `python -m pip install -e .[huggingface]`: Install with HuggingFace dependencies
- `python -m pip install -e .[tgi]`: Install with Text Generation Inference support
- `python -m pip install -e .[quantization]`: Install with quantization libraries
- `python -m pip install -e .[all]`: Install with all dependencies
- `python -m pytest tests/`: Run the test suite
- `python -c "import flow2; print(flow2.__version__)"`: Verify installation

## Package Structure
```
flow2/
├── src/flow2/         # Core package source code
│   ├── core/          # Flash Attention & benchmarks
│   ├── frameworks/    # Multi-framework implementations
│   │   ├── mlx/       # MLX training, inference, quantization
│   │   ├── llamacpp/  # LlamaCpp training, inference, quantization
│   │   └── huggingface/ # HuggingFace training, inference, quantization with MPS
│   ├── chat/          # Interactive chat interfaces
│   ├── performance/   # Benchmarking & analysis tools
│   └── utils/         # Model management & utilities
├── data/              # Organized data directory
│   ├── datasets/      # Training and evaluation datasets
│   ├── results/       # Benchmark results and performance data
│   └── outputs/       # Model outputs, adapters, and fine-tuned models
├── examples/          # Framework-specific examples and workflows
├── scripts/           # Shell scripts and environment setup
├── tools/             # Development tools and utilities
├── models/            # Model storage (MLX, LlamaCpp, HuggingFace)
├── docs/              # Documentation and guides
└── tests/             # Test suite
```

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/flow2.git
cd flow2

# Set up environment (includes virtual environment and dependencies)
source scripts/activate.sh

# Install with all dependencies
pip install -e .[all]

# Or install framework-specific
pip install -e .[mlx]         # For MLX on Apple Silicon
pip install -e .[llamacpp]    # For LlamaCpp
pip install -e .[huggingface] # For HuggingFace with MPS support
```

### Environment Setup
```bash
# Initial environment setup
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Activate environment for development
source scripts/activate.sh
```

### Basic Usage
```python
import flow2

# Check available frameworks
print(f"MLX Available: {flow2.MLX_AVAILABLE}")
print(f"LlamaCpp Available: {flow2.LLAMACPP_AVAILABLE}")
print(f"HuggingFace Available: {flow2.HUGGINGFACE_AVAILABLE}")
print(f"MPS Available: {flow2.MPS_AVAILABLE}")
print(f"Flash Attention: {flow2.FLASH_ATTENTION_AVAILABLE}")

# MLX usage
if flow2.MLX_AVAILABLE:
    from flow2.frameworks.mlx import load_mlx_model, generate_completion
    
# LlamaCpp usage  
if flow2.LLAMACPP_AVAILABLE:
    from flow2.frameworks.llamacpp import create_llama_model, generate_completion

# HuggingFace usage with MPS acceleration
if flow2.HUGGINGFACE_AVAILABLE:
    from flow2.frameworks.huggingface import load_hf_model, generate_completion
```

## Examples & Workflows

### Fine-tuning with MLX
```bash
# Basic MLX fine-tuning
cd examples/mlx
python run_mlx_finetune.py

# Enhanced fine-tuning with Flash Attention
python run_mlx_finetune_improved.py --use-flash-attention --prepare-data

# Flash Attention comparison
python test_flash_attention_comparison.py
```

### Fine-tuning with HuggingFace
```bash
# Basic HuggingFace inference with MPS
cd examples/huggingface
python run_hf_inference.py --model microsoft/DialoGPT-small --device mps

# LoRA fine-tuning with Accelerate
python run_hf_lora_finetune.py --model microsoft/DialoGPT-small --dataset data/datasets/train.jsonl

# QLoRA fine-tuning with 4-bit quantization
python run_hf_lora_finetune.py --model microsoft/DialoGPT-medium --dataset data/datasets/train.jsonl --quantized

# Model quantization benchmark
python run_hf_quantization.py --model microsoft/DialoGPT-small --benchmark
```

### Benchmarking Workflows
```bash
# Comprehensive 3-framework comparison (MLX, LlamaCpp, HuggingFace)
python src/flow2/performance/benchmark/comprehensive_framework_benchmark.py

# Individual framework benchmarks
cd examples/workflows
bash benchmark_comparison_workflow.sh

# MLX comprehensive benchmark
python src/flow2/performance/benchmark/mlx/comprehensive_benchmark.py

# MLX Flash Attention benchmark
python src/flow2/performance/benchmark/mlx/flash_attention_benchmark.py

# 8B model comprehensive analysis
python src/flow2/performance/benchmark/models_8b/comprehensive_model_report.py

# MLX LoRA workflow
python examples/workflows/mlx_lora_workflow.py

# LlamaCpp LoRA workflow  
bash examples/workflows/llamacpp_lora_workflow.sh

# HuggingFace quantization benchmark
cd examples/huggingface
python run_hf_quantization.py --benchmark --method all
```

### Chat Interfaces
```bash
# MLX chat interface
python src/flow2/chat/interfaces/cli/mlx_chat.py

# LlamaCpp chat interface
python src/flow2/chat/interfaces/cli/llamacpp_chat.py

# HuggingFace chat interface with MPS support
python src/flow2/chat/interfaces/cli/hf_chat.py --model microsoft/DialoGPT-small --device mps

# HuggingFace chat with quantization
python src/flow2/chat/interfaces/cli/hf_chat.py --model microsoft/DialoGPT-medium --quantization 4bit

# Web interfaces (Flask-based)
python src/flow2/chat/interfaces/web/mlx_web.py
python src/flow2/chat/interfaces/web/llamacpp_web.py
python src/flow2/chat/interfaces/web/hf_web.py
```

## Key Features

### 🚀 Multi-Framework Support
- **MLX**: Optimized for Apple Silicon with Metal acceleration
- **LlamaCpp**: Cross-platform with CPU/GPU support
- **HuggingFace**: Complete ecosystem with MPS, Accelerate, TGI, and quantization
- **Flash Attention**: Memory-efficient attention optimization

### 🎯 Training & Fine-tuning
- **LoRA**: Low-rank adaptation fine-tuning
- **QLoRA**: Quantized LoRA for memory efficiency
- **Full Fine-tuning**: Complete model retraining
- **Flash Attention Integration**: Automatic optimization

### 📊 Performance & Benchmarking
- **Framework Comparison**: Head-to-head MLX vs LlamaCpp vs HuggingFace
- **Quantization Analysis**: BitsAndBytes, GPTQ, AWQ comparison
- **MPS Optimization**: Apple Silicon GPU acceleration benchmarks
- **Hardware Scaling**: Multi-core and memory optimization
- **Interactive Reports**: HTML dashboards with visualizations

### 💬 Chat Interfaces
- **CLI**: Terminal-based chat with all three frameworks
- **Web**: Browser-based interface with real-time streaming
- **History**: Persistent conversation management
- **Templates**: Customizable prompt templates
- **Quantization**: Real-time quantized model chat

## Framework-Specific Commands

### MLX Framework
```python
from flow2.frameworks.mlx import (
    finetune_lora,           # LoRA fine-tuning
    finetune_qlora,          # QLoRA fine-tuning  
    finetune_full,           # Full fine-tuning
    load_mlx_model,          # Model loading
    generate_completion,     # Text generation
    chat_completion,         # Chat completion
    quantize_model,          # Model quantization
    batch_quantize_models    # Batch quantization
)
```

### LlamaCpp Framework
```python
from flow2.frameworks.llamacpp import (
    finetune_lora,          # LoRA fine-tuning
    apply_lora_adapter,     # LoRA adapter application
    create_llama_model,     # Model creation
    generate_completion,    # Text generation
    chat_completion,        # Chat completion
    quantize_model,         # Model quantization
    batch_quantize_models   # Batch quantization
)
```

### HuggingFace Framework
```python
from flow2.frameworks.huggingface import (
    load_hf_model,           # Model loading with MPS support
    generate_completion,     # Text generation
    chat_completion,         # Chat completion
    streaming_completion,    # Streaming generation
    batch_generate,          # Batch processing
    # Training
    finetune_lora,          # LoRA fine-tuning
    finetune_qlora,         # QLoRA fine-tuning
    merge_lora_adapter,     # Adapter merging
    # Quantization
    quantize_model,         # Model quantization
    load_quantized_model,   # Load quantized models
    benchmark_quantization, # Quantization benchmarking
    # Utils
    setup_mps_device,       # MPS device setup
    get_model_info,         # Model information
)
```

### Performance Tools
```python
from flow2.performance.benchmark import (
    framework_comparison,    # Compare MLX vs LlamaCpp vs HuggingFace
    quantization_comparison, # Compare quantization methods
    benchmark_workflow      # Comprehensive benchmarking
)
```

## Configuration

### Model Paths
Models are stored in `models/` directory:
```
models/
├── mlx/                    # MLX format models
│   ├── tinyllama-1.1b-chat/
│   └── qwen2.5-1.5b-instruct/
├── llamacpp/              # GGUF format models
│   ├── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf
└── huggingface/           # HuggingFace models (or use HF model names)
    ├── microsoft/DialoGPT-small
    ├── microsoft/DialoGPT-medium
    └── quantized/         # Local quantized models
```

### Training Data
Training datasets in `data/datasets/`:
- `train.jsonl` - Training examples
- `valid.jsonl` - Validation examples  
- `test.jsonl` - Test examples
- `quotes_train.jsonl` - Inspirational quotes dataset
- Custom datasets supported

### Output Structure
Training outputs in `data/outputs/`:
- `adapters.safetensors` - LoRA/QLoRA adapters
- `quotes_lora_adapter/` - Example fine-tuned adapter
- `qwen_enhanced/` - Enhanced model variants
- `tinyllama_enhanced/` - Enhanced model variants
- `hf_8b_test_output/` - HuggingFace fine-tuning results

### Benchmark Results
All benchmark results in `data/results/`:
- `mlx_comprehensive/` - MLX framework benchmarks
- `mlx_flash_attention/` - Flash attention performance tests
- `quantization_test/` - Quantization comparison results
- Performance logs and CSV data

## Data Organization

### Working with Datasets
```bash
# List available datasets
ls data/datasets/

# Add new training data
cp my_dataset.jsonl data/datasets/

# View dataset format
head -n 1 data/datasets/train.jsonl
```

### Managing Outputs
```bash
# List fine-tuned adapters
ls data/outputs/*/adapters.safetensors

# Check training results
ls data/outputs/hf_8b_test_output/adapter/

# Archive old outputs
tar -czf archive_$(date +%Y%m%d).tar.gz data/outputs/old_*/
```

### Accessing Results
```bash
# Find recent benchmark results
find data/results/ -name "*.json" -mtime -7

# View performance data
ls data/results/mlx_comprehensive/

# Check logs
tail data/results/model_manager.log
```

### Development Tools
```bash
# Use development tools
python tools/setup.py develop

# Run scripts
./scripts/setup_environment.sh
source scripts/activate.sh
```

## Development Guidelines

### Code Style
- Use ES modules syntax where applicable
- Follow PEP 8 for Python code
- Add type hints for all public APIs
- Include docstrings for all functions
- Prefer async/await for I/O operations

### Testing
- Run tests before committing: `pytest tests/`
- Add tests for new functionality
- Use meaningful test names
- Test both MLX and LlamaCpp code paths

### Performance
- Profile memory usage during training
- Use Flash Attention when available
- Optimize for Apple Silicon (MLX) and multi-core (LlamaCpp)
- Include benchmarks for performance-critical features

## Hardware Requirements

### Recommended for MLX
- Apple Silicon Mac (M1/M2/M3/M4)
- 16GB+ unified memory for training
- macOS 12.0+ (Monterey)

### Recommended for LlamaCpp  
- Multi-core CPU (8+ cores recommended)
- 16GB+ RAM for larger models
- GPU support optional but beneficial

### Recommended for HuggingFace
- Apple Silicon Mac (M1/M2/M3/M4) for MPS acceleration
- NVIDIA GPU with CUDA for maximum performance
- 16GB+ RAM (32GB+ for larger models)
- Python 3.8+ with PyTorch 1.12+

## Examples

### Quick Model Inference
```python
import flow2

# MLX inference
if flow2.MLX_AVAILABLE:
    from flow2.frameworks.mlx import load_mlx_model, generate_completion
    model, tokenizer = load_mlx_model("models/mlx/tinyllama-1.1b-chat")
    response = generate_completion(model, tokenizer, "Hello, how are you?")
    print(response)

# LlamaCpp inference
if flow2.LLAMACPP_AVAILABLE:
    from flow2.frameworks.llamacpp import create_llama_model, generate_completion
    model = create_llama_model("models/llamacpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    response = generate_completion(model, "Hello, how are you?")
    print(response)

# HuggingFace inference with MPS
if flow2.HUGGINGFACE_AVAILABLE:
    from flow2.frameworks.huggingface import load_hf_model, generate_completion, GenerationParams
    model, tokenizer = load_hf_model("microsoft/DialoGPT-small")
    gen_params = GenerationParams(max_new_tokens=50, temperature=0.7)
    response = generate_completion(model, tokenizer, "Hello, how are you?", gen_params)
    print(response)
```

### Fine-tuning Example
```python
# MLX LoRA fine-tuning
from flow2.frameworks.mlx import finetune_lora

finetune_lora(
    model_path="models/mlx/tinyllama-1.1b-chat",
    data_path="data/datasets", 
    output_path="data/outputs/my_adapter",
    num_iters=100,
    learning_rate=1e-4,
    use_flash_attention=True
)

# HuggingFace LoRA fine-tuning with MPS
from flow2.frameworks.huggingface import finetune_lora, create_lora_config, TrainingConfig

lora_config = create_lora_config("microsoft/DialoGPT-small", r=16, lora_alpha=32)
training_config = TrainingConfig(output_dir="data/outputs/hf_lora_output", num_train_epochs=3)

finetune_lora(
    model_name="microsoft/DialoGPT-small",
    dataset_path="data/datasets/train.jsonl",
    output_dir="data/outputs/hf_lora_output",
    lora_config=lora_config,
    training_config=training_config
)

# HuggingFace QLoRA with 4-bit quantization
from flow2.frameworks.huggingface import finetune_qlora, QLoRAConfig

qlora_config = QLoRAConfig(lora_config=lora_config)
finetune_qlora(
    model_name="microsoft/DialoGPT-medium",
    dataset_path="data/datasets/train.jsonl",
    output_dir="data/outputs/hf_qlora_output",
    qlora_config=qlora_config,
    training_config=training_config
)
```

### Benchmarking Example
```python
# Comprehensive 3-framework benchmark
from flow2 import comprehensive_framework_benchmark

test_models = {
    "mlx": {"small": "models/mlx/tinyllama-1.1b-chat"},
    "llamacpp": {"small": "models/llamacpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
    "huggingface": {"small": "microsoft/DialoGPT-small"}
}

benchmark = comprehensive_framework_benchmark.FrameworkBenchmark()
results = benchmark.run_comprehensive_benchmark(
    test_models=test_models,
    test_prompts=["Hello, how are you?", "What is machine learning?"],
    include_quantization=True
)

# HuggingFace quantization benchmark
from flow2.frameworks.huggingface import benchmark_quantization, QuantizationMethod

quant_results = benchmark_quantization(
    model_name="microsoft/DialoGPT-small",
    methods=[QuantizationMethod.BITSANDBYTES_4BIT, QuantizationMethod.BITSANDBYTES_8BIT],
    test_prompts=["Test prompt 1", "Test prompt 2"]
)
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## License
MIT License - see LICENSE file for details

## Acknowledgments
- MLX team at Apple for the MLX framework
- LlamaCpp contributors for the inference engine
- Philip Turner for Metal Flash Attention research
- Hugging Face for model hosting and tools