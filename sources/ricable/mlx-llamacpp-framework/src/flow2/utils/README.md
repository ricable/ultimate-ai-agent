# LLM Model Management System

This directory contains utilities for downloading, converting, and managing LLM models for both the llama.cpp and MLX frameworks on Apple Silicon Macs.

## Overview

The model management system provides a unified interface for:

1. Downloading popular models (Llama 2, Mistral, etc.) for both frameworks
2. Converting models between different formats
3. Quantizing models to reduce memory usage
4. Verifying model integrity
5. Managing model metadata and licensing information

## Getting Started

### Prerequisites

- Python 3.8+
- For llama.cpp models: llama.cpp installed in `../llama.cpp-setup`
- For MLX models: MLX framework installed (`pip install mlx mlx-lm`)
- Hugging Face account (for accessing gated models)

### Installation

No installation is required. Just make sure the scripts are executable:

```bash
chmod +x *.py *.sh
```

### Usage

You can use the unified CLI tool `model_cli.py` or the specialized scripts for each framework.

#### Quick Start

```bash
# List available models
./model_cli.py list

# Get model recommendations based on your hardware
./model_cli.py recommend

# Download a model for llama.cpp
./model_cli.py download llama2-7b-gguf --framework llama.cpp --quant q4_k_m

# Download a model for MLX
./model_cli.py download mistral-7b-mlx --framework mlx

# Or use the convenience shell script
./download-model.sh llama2-7b-gguf --framework llama.cpp --quant q4_k_m
```

## Command Reference

### Unified CLI (`model_cli.py`)

```
python model_cli.py <command> [options]
```

Available commands:

- `list`: List available or installed models
- `info`: Get information about a model
- `license`: Get license information for a model
- `recommend`: Get model recommendations based on system specs
- `download`: Download a model
- `convert`: Convert a model from Hugging Face
- `quantize`: Quantize a model
- `verify`: Verify model integrity
- `run`: Show examples of how to run a model
- `update`: Update model registry with new models

### Specialized CLIs

#### For llama.cpp models (`download_llama_model.py`)

```
python download_llama_model.py <model_name> [options]
```

Options:
- `--list`: List available models
- `--recommend`: Show recommended models
- `--quant <type>`: Quantization type (q4_k_m, q8_0, etc.)
- `--output-dir <dir>`: Directory to save the model
- `--hf <id>`: Download from Hugging Face
- `--verify`: Verify model integrity

#### For MLX models (`download_mlx_model.py`)

```
python download_mlx_model.py <model_name> [options]
```

Options:
- `--list`: List available models
- `--recommend`: Show recommended models
- `--quant <type>`: Quantization type (int4, int8)
- `--output-dir <dir>`: Directory to save the model
- `--hf <id>`: Download from Hugging Face
- `--verify`: Verify model structure

## Examples

### Download a model for llama.cpp

```bash
# Download Llama 2 7B with 4-bit quantization
./model_cli.py download llama2-7b-gguf --framework llama.cpp --quant q4_k_m

# Download Mistral 7B Instruct with 8-bit quantization
./model_cli.py download mistral-7b-instruct-gguf --framework llama.cpp --quant q8_0
```

### Download a model for MLX

```bash
# Download Llama 2 7B
./model_cli.py download llama2-7b-mlx --framework mlx

# Download and quantize Mistral 7B to INT4
./model_cli.py download mistral-7b-mlx --framework mlx --quant int4
```

### Convert models from Hugging Face

```bash
# Convert a model to GGUF format for llama.cpp
./model_cli.py convert --hf meta-llama/Llama-2-7b --framework llama.cpp --output models/llama-2-7b.gguf --quant q4_k

# Convert a model to MLX format
./model_cli.py convert --hf mistralai/Mistral-7B-v0.1 --framework mlx --output models/mistral-7b
```

### Quantize existing models

```bash
# Quantize a GGUF model for llama.cpp
./model_cli.py quantize --input models/llama-2-7b.f16.gguf --output models/llama-2-7b.q4_k.gguf --framework llama.cpp --quant q4_k

# Quantize an MLX model
./model_cli.py quantize --input models/mistral-7b --output models/mistral-7b_int4 --framework mlx --quant int4
```

## Model Sources and Licensing

The model management system includes metadata about model sources, licensing, and usage restrictions. Use the `info` and `license` commands to get detailed information:

```bash
# Get information about a model
./model_cli.py info llama2-7b-gguf

# Get license information
./model_cli.py license llama2-7b-gguf
```

### Supported Models

#### llama.cpp Models (GGUF format)

- Llama 2 (7B, 13B)
- Mistral (7B, Instruct)
- Phi-2

#### MLX Models

- Llama 2 (7B, 13B)
- Mistral (7B, Instruct)
- Phi-2
- Gemma (2B, 7B)

## Advanced Usage

### Verifying Model Integrity

```bash
# Verify a GGUF model
./model_cli.py verify --model-path models/llama-2-7b.q4_k.gguf --framework llama.cpp

# Verify an MLX model
./model_cli.py verify --model-path models/mistral-7b --framework mlx
```

### Getting Run Examples

```bash
# Get examples of how to run a llama.cpp model
./model_cli.py run --model-path models/llama-2-7b.q4_k.gguf --framework llama.cpp

# Get examples of how to run an MLX model
./model_cli.py run --model-path models/mistral-7b --framework mlx
```

## Hardware Recommendations

The `recommend` command provides model recommendations based on your system's RAM:

```bash
./model_cli.py recommend
```

General guidelines:

| RAM | llama.cpp | MLX |
|-----|-----------|-----|
| 8GB | 7B models with q4_k_m | 7B models with int4, Phi-2, Gemma 2B |
| 16GB | 7B models with q8_0, 13B with q4_k_m | 7B models with int8, 13B with int4 |
| 32GB+ | 7B/13B models with higher precision | 7B/13B models with fp16 or int8 |

## Architecture

The model management system consists of:

1. `model_manager.py`: Core library with all the functionality
2. `model_cli.py`: Unified command-line interface
3. `download_llama_model.py`: Specialized CLI for llama.cpp models
4. `download_mlx_model.py`: Specialized CLI for MLX models
5. `download-model.sh`: Convenience shell script

The system uses a model registry (stored in `model_registry.json`) to track available models, their sources, and metadata.

## Troubleshooting

### Common Issues

1. **Permission denied when running scripts**
   - Make sure the scripts are executable: `chmod +x *.py *.sh`

2. **Module not found errors**
   - Make sure you're running the scripts from the `model_utils` directory
   - Install any missing dependencies: `pip install requests tqdm`

3. **Model download fails**
   - For gated models, make sure you're logged in to Hugging Face: `huggingface-cli login`
   - Check your internet connection
   - For large models, make sure you have enough disk space

4. **Quantization fails**
   - Make sure you have the correct frameworks installed
   - Check that you're using a valid quantization type for the framework

### Getting Help

For more detailed help on any command, use the `--help` option:

```bash
./model_cli.py --help
./model_cli.py download --help
```

## Credits

This model management system was created for running large language models locally on Apple Silicon hardware using llama.cpp and MLX frameworks.

## License

This software is provided for educational and research purposes only. See individual model licenses for usage restrictions.