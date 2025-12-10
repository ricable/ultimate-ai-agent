# Complete Guide to llama.cpp on Apple Silicon

This guide provides a comprehensive overview of using llama.cpp on Apple Silicon Macs, from installation to advanced usage.

## What is llama.cpp?

llama.cpp is an efficient C/C++ implementation of the LLaMA language model architecture, optimized for CPU and GPU inference. It's designed to run large language models on consumer hardware with impressive performance and low memory requirements.

Key features:
- High-performance inference on CPUs and GPUs
- Extensive quantization options to reduce memory usage
- Cross-platform support (macOS, Windows, Linux)
- Metal GPU acceleration for Apple Silicon
- GGUF model format with broad compatibility
- LoRA fine-tuning capabilities

## Installation

### Prerequisites

- Apple Silicon Mac (M1/M2/M3)
- macOS 12+ (Monterey or newer)
- Xcode Command Line Tools
- CMake (install via `brew install cmake`)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with Metal support
mkdir build && cd build
cmake .. -DLLAMA_METAL=ON
cmake --build . --config Release

# Test the installation
./bin/main --help
```

### Advanced Installation Options

```bash
# For maximum performance
cmake .. -DLLAMA_METAL=ON -DLLAMA_CUBLAS=OFF -DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_F16C=OFF -DLLAMA_FMA=OFF

# For additional features
cmake .. -DLLAMA_METAL=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_EXAMPLES=ON
```

### Installing Python Bindings (Optional)

```bash
pip install llama-cpp-python
```

Or with Metal support:

```bash
CMAKE_ARGS="-DLLAMA_METAL=ON" pip install llama-cpp-python
```

## Model Management

### Supported Model Formats

llama.cpp uses the GGUF (GPT-Generated Unified Format) format, which supports:
- LLaMA models (1, 2, and 3)
- Mistral models
- Most open-source models (Falcon, MPT, etc.)

### Downloading Models

You can download models from Hugging Face and convert them:

```bash
# Install conversion tools
pip install transformers huggingface_hub

# Convert from Hugging Face
python convert.py --outtype f16 --outfile models/llama-2-7b.gguf meta-llama/Llama-2-7b
```

Or download pre-converted models from sources like TheBloke on Hugging Face.

### Quantizing Models

To reduce model size and memory usage:

```bash
# Quantize to 4-bit (Q4_K)
python quantize.py --model models/llama-2-7b.gguf --outfile models/llama-2-7b-q4_k.gguf --type q4_k

# Or 2-bit for maximum compression (with quality loss)
python quantize.py --model models/llama-2-7b.gguf --outfile models/llama-2-7b-q2_k.gguf --type q2_k
```

## Basic Usage

### Command-Line Text Generation

```bash
# Basic prompt completion
./main -m models/llama-2-7b-q4_k.gguf -p "Tell me about Apple Silicon" -n 256

# With Metal acceleration
./main -m models/llama-2-7b-q4_k.gguf --metal -p "Tell me about Apple Silicon" -n 256
```

### Interactive Chat Mode

```bash
# Simple interactive mode
./main -m models/llama-2-7b-q4_k.gguf --metal --interactive

# With a chat template
./main -m models/llama-2-7b-q4_k.gguf --metal --interactive --color -i -r "User:" -f prompts/chat-with-bob.txt
```

### Key Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-m, --model` | Model path | `-m models/llama-2-7b-q4_k.gguf` |
| `-p, --prompt` | Input prompt | `-p "Tell me a story"` |
| `-n, --n-predict` | Number of tokens to predict | `-n 512` |
| `--metal` | Enable Metal GPU acceleration | `--metal` |
| `--metal-mmq` | Enable Metal matrix multiplication | `--metal-mmq` |
| `-t, --threads` | Number of CPU threads | `-t 4` |
| `-c, --ctx-size` | Context window size | `-c 2048` |
| `-b, --batch-size` | Batch size for prompt processing | `-b 512` |
| `--temp` | Temperature (randomness) | `--temp 0.7` |
| `--top-p` | Top-p sampling | `--top-p 0.9` |
| `--repeat-penalty` | Repetition penalty | `--repeat-penalty 1.1` |

## Advanced Usage

### Server Mode

Run llama.cpp as a local API server:

```bash
./server -m models/llama-2-7b-q4_k.gguf --metal -c 2048 --host 0.0.0.0 --port 8080
```

Access via HTTP:
```bash
curl -X POST http://localhost:8080/completion -d '{
  "prompt": "Tell me about Apple Silicon",
  "n_predict": 128
}'
```

### Embedding Generation

Generate vector embeddings for text:

```bash
./embedding -m models/llama-2-7b-q4_k.gguf -p "This is a sample text"
```

### Performance Optimization

For maximum performance:

```bash
./main \
  -m models/llama-2-7b-q4_k.gguf \
  --metal \           # Enable Metal acceleration
  --metal-mmq \       # Enable Metal matrix multiplication
  -t 4 \              # Number of threads (adjust for your CPU)
  -c 2048 \           # Context size
  -b 512 \            # Batch size
  --temp 0.7 \        # Temperature
  --repeat_penalty 1.1 # Repetition penalty
```

## Fine-tuning with LoRA

### Prerequisites

Ensure you have the necessary tools:
```bash
# Build the finetune tool
cmake --build . --config Release --target llama-finetune
```

### Preparing Training Data

Create a JSONL file with training examples:
```json
{"prompt": "Question?", "response": "Answer"}
{"prompt": "Another question?", "response": "Another answer"}
```

### Running Fine-tuning

```bash
./llama-finetune \
  --model-base ./models/llama-2-7b-q4_0.gguf \
  --lora-rank 8 \
  --lora-layers all \
  --data-train ./data/train.jsonl \
  --data-val ./data/val.jsonl \
  --lora-out ./lora-finetune.bin
```

### Using Fine-tuned Models

```bash
./main \
  --model ./models/llama-2-7b-q4_0.gguf \
  --lora ./lora-finetune.bin \
  --prompt "Your prompt here"
```

## Memory Usage Guide

| Quantization | 7B Model | 13B Model | 33B Model | 70B Model |
|--------------|----------|-----------|-----------|-----------|
| F16 (no quant) | ~14GB | ~26GB | ~65GB | ~140GB |
| Q8_0 | ~8GB | ~13GB | ~32GB | ~70GB |
| Q6_K | ~6GB | ~10GB | ~24GB | ~50GB |
| Q5_K | ~5GB | ~9GB | ~20GB | ~43GB |
| Q4_K | ~4GB | ~7GB | ~16GB | ~35GB |
| Q3_K | ~3.5GB | ~6GB | ~14GB | ~30GB |
| Q2_K | ~3GB | ~5GB | ~12GB | ~25GB |

*Note: Values are approximate and include context window overhead. Actual usage may vary.*

## Troubleshooting

### Common Issues

1. **Metal Support Problems**
   - Error: "Metal not supported"
   - Solution: Ensure you built with `-DLLAMA_METAL=ON`

2. **Out of Memory Errors**
   - Error: "Failed to allocate memory"
   - Solution: Use a more aggressive quantization or reduce context size

3. **Build Failures**
   - Error: Compilation errors during build
   - Solution: Ensure you have the latest Xcode Command Line Tools

### Performance Issues

1. **Slow Inference**
   - Check that Metal is actually being used (`--metal`)
   - Try enabling matrix multiplication (`--metal-mmq`)
   - Adjust batch size for your specific hardware

2. **High Memory Usage**
   - Use a more efficient quantization method
   - Reduce context window size
   - Limit number of threads

## Python Integration

Using the Python bindings:

```python
from llama_cpp import Llama

# Load the model
llm = Llama(
    model_path="models/llama-2-7b-q4_k.gguf",
    n_ctx=2048,
    n_gpu_layers=-1  # Use all layers on GPU
)

# Generate text
output = llm(
    "Tell me about Apple Silicon",
    max_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repeat_penalty=1.1
)

print(output["choices"][0]["text"])
```

## Resources

- [Official llama.cpp Repository](https://github.com/ggerganov/llama.cpp)
- [llama.cpp Discord](https://discord.gg/llama)
- [Model Sources (TheBloke)](https://huggingface.co/TheBloke)
- [Local LLaMA Subreddit](https://www.reddit.com/r/LocalLLaMA/)

## Advanced Topics

- [Integrating with LangChain](../advanced/application-integration.md)
- [Custom Chat Templates](../use-cases/chat-applications.md)
- [Comparison with Other Frameworks](framework-comparison.md)