# SPARC CLI Usage Guide

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Quick Install

```bash
./install.sh
```

### Manual Install

```bash
pip install -e .
```

## Basic Usage

### Command Structure

```bash
sparc -m "Your task description" [options]
```

### Common Options

- `-m, --message`: Task or query to execute (required)
- `--research-only`: Only perform research without implementation
- `--provider`: LLM provider (anthropic|openai|openrouter|openai-compatible)
- `--model`: Model name (required for non-Anthropic providers)
- `--cowboy-mode`: Skip interactive approval for shell commands
- `--hil, -H`: Enable human-in-the-loop mode
- `--chat`: Enable interactive chat mode

### Basic Examples

```bash
# Simple research task
sparc -m "Analyze the security of my Flask application" --research-only

# Code implementation with human oversight
sparc -m "Add input validation to user registration" --hil

# Interactive chat mode
sparc -m "Help me refactor this module" --chat
```

### Getting Started

1. Install SPARC CLI using the installation instructions above
2. Set up your environment variables for your chosen LLM provider
3. Run a simple research task to verify installation
4. Try an implementation task with human oversight
5. Explore more advanced features as needed

For more advanced usage and features, see the [Advanced Features Guide](advanced.md).
