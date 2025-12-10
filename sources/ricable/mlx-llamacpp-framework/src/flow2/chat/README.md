# Chat Interfaces for Local LLMs on Apple Silicon

This repository contains command-line and web-based chat interfaces for running local LLMs on Apple Silicon hardware using llama.cpp and MLX frameworks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Command-Line Chat Interfaces](#command-line-chat-interfaces)
- [Web Chat Interfaces](#web-chat-interfaces)
- [Example Workflows](#example-workflows)
- [Improving Chat Quality](#improving-chat-quality)
- [Contributing](#contributing)

## Overview

This project provides interactive chat applications for both llama.cpp and MLX frameworks optimized for Apple Silicon. It includes both command-line and web-based interfaces, with features for managing chat history, context, and generation parameters.

These interfaces are designed to work with the existing llama.cpp and MLX setups in the parent project, making it easy to start chatting with your local models.

## Features

- **Command-line interfaces** for both llama.cpp and MLX
- **Web interfaces** with real-time streaming responses
- **Chat history management** with persistence
- **Context handling** to optimize token usage
- **Multiple prompt templates** (ChatML, Llama-2, Alpaca)
- **Parameter adjustment** during chat sessions
- **Example workflows** demonstrating different use cases
- **Documentation** on improving chat quality

## Directory Structure

```
chat_interfaces/
├── common/                 # Shared utilities
│   └── chat_history.py     # Chat history and context management
├── llama_cpp/              # llama.cpp interfaces
│   ├── cli/                # Command-line interface
│   │   └── chat_cli.py     # CLI implementation
│   └── web/                # Web interface
│       └── web_app.py      # Flask web app
├── mlx/                    # MLX interfaces
│   ├── cli/                # Command-line interface
│   │   └── chat_cli.py     # CLI implementation
│   └── web/                # Web interface
│       └── web_app.py      # Flask web app
├── examples/               # Example workflow scripts
│   ├── code_assistant_workflow.py     # Code generation/explanation
│   └── data_analysis_workflow.py      # Data analysis assistance
└── docs/                   # Documentation
    └── improving_chat_quality.md      # Prompt engineering guide
```

## Installation

These chat interfaces are designed to work with the existing llama.cpp and MLX setups in the parent project. Ensure those are properly installed before using these interfaces.

### Prerequisites

For llama.cpp interfaces:
- Properly built llama.cpp (see `/llama.cpp-setup/README.md`)
- One or more GGUF model files in `/llama.cpp-setup/models/`

For MLX interfaces:
- MLX and MLX-LM Python packages installed
- One or more MLX-compatible models in `/mlx-setup/models/`

### Additional Python Dependencies

```bash
pip install flask flask-socketio
```

## Command-Line Chat Interfaces

### llama.cpp CLI

```bash
# Basic usage
python chat_interfaces/llama_cpp/cli/chat_cli.py

# Specify a model
python chat_interfaces/llama_cpp/cli/chat_cli.py --model /path/to/model.gguf

# Additional options
python chat_interfaces/llama_cpp/cli/chat_cli.py --model /path/to/model.gguf \
  --max-tokens 1024 \
  --temperature 0.7 \
  --system-message "You are a helpful assistant." \
  --prompt-template chatml \
  --history-file chat_history.json
```

### MLX CLI

```bash
# Basic usage
python chat_interfaces/mlx/cli/chat_cli.py

# Specify a model
python chat_interfaces/mlx/cli/chat_cli.py --model /path/to/model/directory

# Additional options
python chat_interfaces/mlx/cli/chat_cli.py --model /path/to/model/directory \
  --max-tokens 1024 \
  --temperature 0.7 \
  --system-message "You are a helpful assistant." \
  --history-file chat_history.json
```

### CLI Commands

Both CLIs support the following commands during chat:

- `/help` - Show help information
- `/clear` - Clear the conversation history
- `/params` - Show current parameters
- `/temp N` - Set temperature to N (e.g., `/temp 0.8`)
- `/quit` - Exit the program

## Web Chat Interfaces

### llama.cpp Web UI

```bash
# Basic usage
python chat_interfaces/llama_cpp/web/web_app.py

# Specify port and host
python chat_interfaces/llama_cpp/web/web_app.py --port 5000 --host 0.0.0.0

# Debug mode
python chat_interfaces/llama_cpp/web/web_app.py --debug
```

### MLX Web UI

```bash
# Basic usage
python chat_interfaces/mlx/web/web_app.py

# Specify port and host
python chat_interfaces/mlx/web/web_app.py --port 5001 --host 0.0.0.0

# Debug mode
python chat_interfaces/mlx/web/web_app.py --debug
```

Once running, open your browser and navigate to:
- llama.cpp Web UI: http://localhost:5000
- MLX Web UI: http://localhost:5001

## Example Workflows

The project includes example workflows demonstrating practical applications of local LLMs:

### Code Assistant Workflow

```bash
# Run all code assistant examples
python chat_interfaces/examples/code_assistant_workflow.py

# Run specific example
python chat_interfaces/examples/code_assistant_workflow.py --example generation

# Use MLX framework
python chat_interfaces/examples/code_assistant_workflow.py --framework mlx
```

This workflow demonstrates:
- Code generation with detailed comments
- Explaining complex code concepts
- Debugging and fixing buggy code

### Data Analysis Workflow

```bash
# Run all data analysis examples
python chat_interfaces/examples/data_analysis_workflow.py

# Run specific example
python chat_interfaces/examples/data_analysis_workflow.py --example viz

# Use MLX framework
python chat_interfaces/examples/data_analysis_workflow.py --framework mlx
```

This workflow demonstrates:
- Exploratory data analysis guidance
- Statistical test selection and interpretation
- Data visualization recommendations

## Improving Chat Quality

For guidance on improving chat quality through prompt engineering and parameter tuning, see [Improving Chat Quality](docs/improving_chat_quality.md).

Key topics covered:
- System message optimization
- Conversation templates
- Chain-of-thought techniques
- Few-shot learning
- Parameter tuning
- Context management
- Framework-specific optimizations

## Contributing

Contributions are welcome! Here are some ways to improve this project:

1. Add support for more prompt templates
2. Improve error handling and recovery
3. Add more example workflows for different use cases
4. Enhance the web UI with additional features
5. Create specialized interfaces for specific tasks (coding, data analysis, etc.)
6. Optimize performance for different model sizes and quantization levels

Please ensure your code follows the existing style and includes proper documentation.