# SPARC CLI

Version: 0.86.2

SPARC CLI is a powerful command-line interface that implements the SPARC Framework methodology for AI-assisted software development. Combining autonomous research capabilities with guided implementation, it provides a comprehensive toolkit for analyzing codebases, planning changes, and executing development tasks with advanced AI assistance.

Made by rUv, cause he could.

[GitHub Repository](https://github.com/ruvnet/sparc)

## Key Benefits

- **Framework Integration**: Seamlessly implements SPARC Framework's methodology for systematic software development
- **Autonomous Capabilities**: Provides both research analysis and guided implementation with AI assistance
- **Safety Controls**: Features human-in-the-loop controls and review mechanisms for AI actions
- **Provider Flexibility**: Supports multiple LLM providers (Anthropic, OpenAI, OpenRouter) for diverse needs
- **Development Workflow**: Enhances productivity through AI-assisted analysis, planning, and implementation
- **Quantum-Coherent Analysis**: Optimizes code structure through advanced complexity analysis
- **Consciousness Integration**: Leverages pseudo consciousness for enhanced decision-making capabilities
- **Symbolic Reasoning**: Enables sophisticated pattern recognition and code optimization

## Core Features

- Research and implementation capabilities
- Multiple LLM provider support (Anthropic, OpenAI, OpenRouter)
- Interactive chat mode
- Human-in-the-loop interaction
- Expert knowledge queries
- Shell command execution with "cowboy mode"
- Rich console output formatting
- Quantum consciousness calculation capabilities
- Integrated symbolic reasoning
- Quantum-coherent complexity features
- Advanced pattern recognition and complexity management
- Self-aware coding processes and intelligent optimization
- Combined classical and quantum-inspired reasoning
- **PolarisOne Integration**:
  - Adaptive Token Weighting (ATW) for identifying key concepts
  - Focused response generation based on weighted tokens
  - Improved context understanding and relevance
  - Token-aware memory storage and retrieval
  - Efficient context pruning and re-expansion
  - Hierarchical token weighting for better memory organization

## Tool System

SPARC CLI provides a set of built-in tools that work together to enable AI-assisted development:

### Core Tools
- **File Tools**: read_file, write_file, file_str_replace for file operations
- **Directory Tools**: list_directory, fuzzy_find for navigating codebases
- **Shell Tool**: Executes system commands with safety controls
- **Memory Tool**: Manages context and information across operations
- **Expert Tool**: Provides specialized knowledge and analysis
- **Research Tool**: Analyzes codebases and documentation
- **Scrape Tool**: Web scraping with HTML to markdown conversion

## Autonomous Capabilities

- Autonomous Research: Analyze codebases and provide insights without making changes
- Autonomous Implementation: Plan and execute code changes with AI guidance
- Human-in-the-loop Controls: Review and approve AI actions during execution
- Expert Knowledge Integration: Access specialized knowledge for complex analysis
- Shell Command Automation: Execute system commands autonomously in cowboy mode

## Installation

Requires Python 3.8 or higher.

### PyPI Install
```bash
pip install sparc
```

### Quick Install
```bash
./install.sh
```

### Manual Install
```bash
pip install -e .
```

### Environment Setup

Create a `.env` file in your project root with the following variables:

```bash
# Required: At least one of these LLM provider API keys
ANTHROPIC_API_KEY=your_anthropic_key    # Required for Claude models
OPENAI_API_KEY=your_openai_key          # Required for GPT models
OPENROUTER_API_KEY=your_openrouter_key  # Required for OpenRouter

# Optional: Expert knowledge configuration
EXPERT_PROVIDER=openai                   # Default provider for expert queries (anthropic|openai|openrouter)
EXPERT_MODEL=o1-preview                  # Model to use for expert knowledge queries

# Optional: Default provider settings
DEFAULT_PROVIDER=anthropic               # Default LLM provider (anthropic|openai|openrouter)
DEFAULT_MODEL=claude-3-opus-20240229     # Default model name

# Optional: Development settings
DEBUG=false                              # Enable debug logging
COWBOY_MODE=false                        # Skip command approval prompts
```

Note: At least one provider API key must be configured for SPARC to function.

## Usage

Basic command structure:
```bash
sparc -m "Your task description" [options]
```

### Options

- `-m, --message`: The task or query to execute (required)
- `--research-only`: Only perform research without implementation
- `--provider`: LLM provider to use (anthropic|openai|openrouter|openai-compatible)
- `--model`: Model name to use (required for non-Anthropic providers)
- `--cowboy-mode`: Skip interactive approval for shell commands
- `--expert-provider`: Provider for expert knowledge queries
- `--expert-model`: Model for expert queries
- `--hil, -H`: Enable human-in-the-loop mode
- `--chat`: Enable interactive chat mode

### Examples

```bash
# Autonomous research analysis
sparc -m "Analyze the security of my Flask application" --research-only

# Autonomous implementation with human oversight
sparc -m "Add input validation to user registration" --hil

# Expert knowledge consultation
sparc -m "Review authentication implementation" --expert-model o1-preview

# Autonomous shell command execution
sparc -m "Run test suite and analyze failures" --cowboy-mode

# Interactive chat mode
sparc --chat
```

## Advanced Features

### PolarisOne Integration
- Token weighting for identifying key concepts in code and queries
- Memory management for maintaining context
- Focused response generation based on weighted tokens
- Hierarchical token organization for better context management

### Expert Knowledge System
- Specialized domain expertise through configurable expert models
- Deep technical analysis capabilities
- Integration with external knowledge bases

### Web Scraping Capabilities
- HTML to markdown conversion
- JavaScript-enabled site support through Playwright
- Fallback to HTTPX for basic scraping
- Automated cleanup and formatting

## Contributing

Visit our [GitHub repository](https://github.com/ruvnet/sparc) to:
- Report issues
- Submit pull requests
- View documentation
- Join the community

## Acknowledgments

- **RA.Aid**: For inspiration and contributions to research assistant capabilities (https://github.com/ai-christianson/RA.Aid)
- **Playwright**: For robust web automation and scraping capabilities
- **Langchain**: For powerful language model tools and chain-of-thought implementations
- **OpenAI**: For the GPT models that enhance our capabilities
- **Anthropic**: For Claude's advanced reasoning capabilities
- **SPARC Framework Community**: For continuous feedback and contributions

Made with ❤️ by rUv
