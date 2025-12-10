# Getting Started with LLMs on Apple Silicon

This guide will help you set up your Apple Silicon Mac to run Large Language Models locally using llama.cpp and MLX frameworks.

## Prerequisites

- An Apple Silicon Mac (M1, M2, or M3 series)
- macOS 12 (Monterey) or newer
- At least 8GB of RAM (16GB+ recommended)
- Command-line familiarity
- Xcode Command Line Tools
  ```bash
  xcode-select --install
  ```
- Homebrew (recommended)
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

## Installation Options

You can choose between two installation approaches:

### Option 1: Quick Start with Automated Scripts

For the fastest setup experience:

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-mac.git
cd llm-mac

# Run the automated setup script
./setup.sh
```

This will:
1. Install necessary dependencies
2. Set up both llama.cpp and MLX frameworks
3. Download a small test model
4. Run a verification test

### Option 2: Manual Installation

If you prefer to understand each step:

#### Setting up llama.cpp

```bash
# Install prerequisites
brew install cmake

# Clone and build llama.cpp
cd llama.cpp-setup
./scripts/setup.sh

# Verify the installation
./scripts/verify-installation.sh
```

#### Setting up MLX

```bash
# Create and activate a Python virtual environment (recommended)
python -m venv mlx-env
source mlx-env/bin/activate

# Install MLX
cd mlx-setup
./scripts/setup.sh

# Test the installation
python scripts/test-installation.py
```

## Downloading Your First Model

After installation, you'll need to download a model to work with:

```bash
# Using our model utilities
cd model_utils
python model_cli.py download llama-2-7b
```

This will download the Llama 2 7B model and convert it to the appropriate format for both frameworks.

## Running Your First Inference

### With llama.cpp

```bash
cd llama.cpp-setup
./bin/main -m ../models/llama-2-7b-q4_0.gguf --metal -p "Hello, my name is" -n 50
```

### With MLX

```bash
cd mlx-setup
python -c "
from mlx_lm import load, generate
model, tokenizer = load('../models/llama-2-7b', quantization='int4')
output = generate(model, tokenizer, 'Hello, my name is', max_tokens=50)
print(tokenizer.decode(output))
"
```

## Next Steps

Now that you have your environment set up, you can:

1. [Learn about different frameworks](frameworks/framework-comparison.md)
2. [Explore inference options](use-cases/inference-guide.md)
3. [Set up a chat application](use-cases/chat-applications.md)
4. [Try fine-tuning a model](use-cases/fine-tuning-guide.md)

## Troubleshooting

If you encounter issues during installation:

- **llama.cpp build failures**: See [llama.cpp troubleshooting](llama.cpp-setup/docs/troubleshooting.md)
- **MLX installation issues**: See [MLX troubleshooting](mlx-setup/docs/troubleshooting.md)
- **Metal acceleration problems**: Ensure you have the latest macOS updates installed
- **Download failures**: Check [model downloading guide](model_utils/README.md)

For other issues, please check our FAQ or open an issue on GitHub.