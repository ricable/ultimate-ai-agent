# Polyglot Development Environment

> **🚀 AI-Optimized Multi-Language Development with Intelligent Automation**

A sophisticated polyglot development environment supporting Python, TypeScript, Rust, Go, and Nushell with DevBox isolation, intelligent automation, and seamless AI integration featuring **112 MCP tools**, **Enhanced AI Hooks**, **Docker MCP integration**, **AG-UI protocol**, and **Claude-Flow orchestration**.

[![Nushell](https://img.shields.io/badge/Nushell-0.105.1-blue)](https://www.nushell.sh/)
[![DevBox](https://img.shields.io/badge/DevBox-Latest-green)](https://www.jetify.com/devbox)
[![MCP](https://img.shields.io/badge/MCP-Protocol-orange)](https://github.com/modelcontextprotocol)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Features

### 🔧 **Multi-Language Development**
- **Python**: uv + FastAPI + async/await + SQLAlchemy
- **TypeScript**: Strict mode + ES modules + Jest + Result patterns  
- **Rust**: Async Tokio + ownership patterns + serde + thiserror
- **Go**: Context patterns + small interfaces + explicit errors
- **Nushell**: Structured data + type hints + cross-environment orchestration

### 🤖 **AI-Powered Automation**
- **Model Context Protocol (MCP)**: **112 tools** across 15 categories for comprehensive Claude integration
- **Enhanced AI Hooks**: 4 production-ready hooks with context engineering auto-triggers and intelligent error resolution
- **Claude-Flow Integration**: AI agent orchestration with hive-mind coordination and automated task spawning
- **AG-UI Protocol**: 5 agentic environment templates with CopilotKit integration and generative UI
- **Docker MCP Toolkit**: 34+ containerized tools with HTTP/SSE transport and secure execution
- **Context Engineering Framework**: Enhanced PRP generation with dynamic templates and dojo integration
- **Advanced Analytics**: ML-based performance monitoring, predictive insights, and business intelligence

### 🐳 **Container Development**
- **Centralized DevPod Management**: Single script manages all environments including agentic variants ✅
- **DevPod Integration**: 1-10 parallel containerized workspaces per environment + 5 agentic variants
- **Auto .claude/ Installation**: Zero-configuration AI hooks deployment to all containers
- **VS Code Integration**: Auto-launch with language-specific extensions and Claude-Flow integration
- **Resource Management**: AI-powered smart lifecycle with optimization and automated cleanup
- **Container Security**: Multi-layer isolation with host/container boundary validation

### 🛡️ **Quality & Security**
- **Cross-Language Validation**: Parallel testing across all environments with intelligent quality gates
- **Enhanced Security**: Multi-layer scanning with dependency tracking, container isolation, and host boundary validation
- **AI-Powered Error Resolution**: Intelligent error analysis with learning and automated suggestions
- **Performance Monitoring**: ML-based analytics with predictive insights and anomaly detection
- **Zero-Drift Configuration**: Single source of truth with automated synchronization and validation

## 🚀 Quick Start


### 2. Install Dependencies

```bash
# Install DevBox (environment isolation)
curl -fsSL https://get.jetify.com/devbox | bash

# Install direnv (auto environment activation)
# macOS
brew install direnv

# Linux
sudo apt install direnv  # Ubuntu/Debian
sudo dnf install direnv  # Fedora
sudo pacman -S direnv    # Arch

# Add to shell (choose your shell)
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc    # Bash
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc      # Zsh
echo 'direnv hook fish | source' >> ~/.config/fish/config.fish  # Fish
```

### 3. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ricable/polyglot-devenv.git
cd polyglot-devenv

# Quick setup with Nushell automation
nu scripts/setup-all.nu

# Or manual setup
make install
```

### 4. Verify Installation

```bash
# Test Nushell
nu --version

# Test DevBox
devbox version

# Test the unified environment
nu scripts/validate-all.nu quick

# Test MCP server
cd mcp && npm run build && npm run start
```

## 📁 Project Structure

```
polyglot-devenv/
├── dev-env/                    # 🏠 Unified Development Environment
│   ├── python/                 # 🐍 Python (uv + FastAPI + async)
│   ├── typescript/             # 📘 TypeScript (strict + ES modules)
│   ├── rust/                   # 🦀 Rust (Tokio + ownership patterns)
│   ├── go/                     # 🐹 Go (context + interfaces)
│   └── nushell/                # 🐚 Nushell (automation + orchestration)
│       ├── scripts/            # 📜 25+ automation scripts
│       ├── config/             # ⚙️ Configuration files
│       └── common.nu           # 🔧 Shared utilities
├── mcp/                        # 🤖 Model Context Protocol Server (Production ✅)
│   ├── polyglot-server.ts      # 📡 Main MCP server (32 core tools)
│   ├── modules/                # 🔧 Modular tool implementation (7 modules, 80 tools)
│   │   ├── claude-flow.ts      # 🤖 AI agent orchestration (10 tools)
│   │   ├── enhanced-hooks.ts   # 🚀 Intelligent automation (8 tools)
│   │   ├── docker-mcp.ts       # 🐳 Containerized execution (16 tools)
│   │   ├── host-container.ts   # 🏗️ Security boundaries (8 tools)
│   │   ├── nushell-automation.ts # 🐚 Cross-language orchestration (23 tools)
│   │   ├── config-management.ts # ⚙️ Zero-drift configuration (7 tools)
│   │   └── advanced-analytics.ts # 📈 ML-powered insights (8 tools)
│   ├── polyglot-utils.ts       # 🛠️ Shared utilities & DevPod integration
│   ├── polyglot-types.ts       # 📝 TypeScript types and interfaces
│   └── dist/                   # 📦 Compiled JavaScript (112 total tools)
├── scripts/                    # 🔄 Cross-language validation
│   └── validate-all.nu         # ✅ Parallel validation script
├── host-tooling/               # 🖥️ Host machine scripts (host/container separation)
│   ├── devpod-management/       # 🐳 CENTRALIZED DevPod management ✅
│   ├── installation/            # ⚙️ Host dependency installation
│   ├── monitoring/              # 📊 Infrastructure access
│   └── shell-integration/       # 🐚 Host shell integration
├── devpod-automation/          # 🐳 Container development (Enhanced ✅)
│   ├── templates/              # 📄 DevPod environment templates
│   │   ├── .claude-core/       # 🤖 AI automation template (auto-installed)
│   │   ├── python/             # 🐍 Standard Python devcontainer
│   │   ├── typescript/         # 📘 Standard TypeScript devcontainer
│   │   ├── rust/               # 🦀 Standard Rust devcontainer
│   │   ├── go/                 # 🐹 Standard Go devcontainer
│   │   ├── nushell/            # 🐚 Standard Nushell devcontainer
│   │   ├── agentic-python/     # 🤖 AG-UI Python (FastAPI + agents + CopilotKit)
│   │   ├── agentic-typescript/ # 🤖 AG-UI TypeScript (Next.js + CopilotKit + agents)
│   │   ├── agentic-rust/       # 🤖 AG-UI Rust (Tokio + async agents + protocol)
│   │   ├── agentic-go/         # 🤖 AG-UI Go (HTTP server + agent middleware)
│   │   └── agentic-nushell/    # 🤖 AG-UI Nushell (pipeline-based agents)
│   ├── agents/                 # 🤖 Agent configuration storage by environment
│   └── scripts/                # 📜 DevPod provisioning and management scripts
├── context-engineering/        # 📝 Context Engineering Framework (REORGANIZED ✅)
│   ├── workspace/              # 🏗️ Local development & PRP generation
│   │   ├── features/           # 📋 Feature definitions (input)
│   │   ├── templates/          # 📄 PRP templates by environment
│   │   ├── generators/         # ⚙️ PRP generation tools
│   │   └── docs/              # 📚 Workspace usage documentation
│   ├── devpod/                # 🐳 Containerized execution environment
│   │   ├── environments/      # 🌍 Environment-specific configs (python/, typescript/, rust/, go/, nushell/)
│   │   ├── execution/         # 🚀 Execution engines & reports
│   │   ├── monitoring/        # 📊 Performance & security tracking
│   │   └── configs/           # ⚙️ DevPod-specific configurations
│   ├── shared/                # 🔄 Resources used by both workspace & devpod
│   │   ├── examples/          # 📖 Reference examples (including dojo/)
│   │   ├── utils/            # 🛠️ Common utilities (Nushell tools)
│   │   ├── schemas/          # ✅ Validation schemas
│   │   └── docs/             # 📚 Shared documentation
│   └── archive/               # 🗄️ Historical PRPs and reports
├── .claude/                    # 🧠 Claude Code integration (Enhanced ✅)
│   ├── commands/               # ⚡ Slash commands with context engineering
│   ├── hooks/                  # 🪝 Enhanced AI hooks (4 production-ready)
│   │   ├── context-engineering-auto-triggers.py     # 🚀 Auto PRP generation
│   │   ├── intelligent-error-resolution.py          # 🚀 AI-powered error analysis
│   │   ├── smart-environment-orchestration.py       # 🚀 Auto DevPod management
│   │   └── cross-environment-dependency-tracking.py # 🚀 Security & compatibility
│   ├── settings.json           # ⚙️ Enhanced hooks configuration
│   └── docker-mcp/             # 🐳 Docker MCP integration scripts
├── Makefile                    # 🔨 Automation commands
├── .mcp.json                   # 🔗 MCP server configuration
└── README.md                   # 📖 This file
```

## 🛠️ Getting Started

### Automatic Setup (Recommended)

```bash
# Complete automated setup
make setup

# Or step by step
make install-deps    # Install all dependencies
make setup-envs      # Setup all environments
make validate        # Validate installation
make start-mcp       # Start MCP server
```

### Manual Setup

1. **Setup Individual Environments:**
```bash
# Python environment
cd dev-env/python
devbox shell
devbox run install

# TypeScript environment  
cd ../typescript
devbox shell
devbox run install

# Continue for rust, go, nushell...
```

2. **Test Cross-Language Validation:**
```bash
# Quick validation
nu scripts/validate-all.nu quick

# Full parallel validation
nu scripts/validate-all.nu --parallel

# Environment-specific validation
nu scripts/validate-all.nu --environment python
```

3. **Setup MCP Server:**
```bash
cd mcp
npm install
npm run build
npm run start
```

## 🤖 MCP Server Integration

The project includes a comprehensive MCP server for sophisticated AI integration:

### Available Tools (112 Total) 🌟

| Category | Tools | Count | Description |
|----------|-------|-------|-------------|
| **🤖 Claude-Flow** | AI orchestration, hive-mind coordination | **10** | AI agent orchestration, terminal management |
| **🚀 Enhanced AI Hooks** | Context engineering, error resolution | **8** | Intelligent automation, smart environment orchestration |
| **🐳 Docker MCP** | Containerized execution, HTTP/SSE transport | **16** | Secure tool execution, comprehensive security scanning |
| **🏗️ Host/Container** | Security boundaries, credential isolation | **8** | Infrastructure access control, container isolation |
| **🐚 Nushell Automation** | Cross-language orchestration, data processing | **23** | Performance monitoring, testing frameworks |
| **⚙️ Configuration** | Zero-drift management, automated sync | **7** | Template management, backup & recovery |
| **📈 Advanced Analytics** | ML-based insights, predictive analytics | **8** | Performance optimization, business intelligence |
| **🤖 AG-UI Protocol** | Agentic environments, generative UI | **9** | Agent lifecycle, CopilotKit integration |
| **🌍 Core Foundation** | Environment, DevBox, DevPod, Security, PRP | **23** | Essential development tools, validation |

### Quick Start Commands

**Essential MCP Operations:**
```bash
# Environment & DevPod Management
mcp tool environment_detect '{}' 
mcp tool devpod_provision '{"environment": "dev-env/python", "count": 1}'

# AI Agent Orchestration 
mcp tool claude_flow_init '{"environment": "dev-env/python"}'
mcp tool agui_provision '{"environment": "agentic-python"}'

# Enhanced Automation
mcp tool enhanced_hook_env_orchestration '{"action": "switch", "target_environment": "dev-env/typescript"}'

# Security & Performance
mcp tool docker_mcp_gateway_start '{"port": 8080}'
mcp tool performance_analytics '{"action": "analyze", "time_range": "week"}'
```

### Configuration

The MCP server is pre-configured in `.mcp.json`:

```json
{
  "mcpServers": {
    "polyglot-devenv": {
      "command": "node",
      "args": ["dist/index.js"],
      "cwd": "mcp"
    }
  }
}
```

### Advanced Usage Examples

**Multi-Tool AI Workflows:**
```bash
# Complete AI-Powered Development Workflow
mcp tool claude_flow_init '{"environment": "dev-env/python"}'
mcp tool docker_mcp_gateway_start '{"port": 8080, "background": true}'
mcp tool agui_provision '{"environment": "agentic-python", "features": ["agentic_chat"]}'
mcp tool claude_flow_spawn '{"environment": "dev-env/python", "task": "Create FastAPI microservice"}'

# Cross-Environment Polyglot Development
mcp tool enhanced_hook_env_orchestration '{"action": "switch", "target_environment": "dev-env/typescript"}'
mcp tool agui_agent_create '{"name": "FrontendAgent", "type": "generative_ui", "environment": "agentic-typescript"}'
mcp tool claude_flow_hive_mind '{"environment": "dev-env", "command": "coordinate"}'

# Security & Performance Monitoring
mcp tool docker_mcp_security_scan '{"target": "all", "detailed": true}'
mcp tool enhanced_hook_dependency_tracking '{"action": "scan", "security_check": true}'
mcp tool performance_analytics '{"action": "optimize", "export_format": "dashboard"}'
```

### Natural Language Integration

Use natural language with Claude Code to interact with all 112 tools:
- *"Initialize Claude-Flow in Python environment and spawn an AI agent for FastAPI development"*
- *"Provision 2 agentic TypeScript environments with CopilotKit integration"*
- *"Run comprehensive security scan across all containers and environments"*
- *"Generate performance analytics dashboard for the last week"*
- *"Switch to Rust environment and optimize DevPod resource allocation"*

**📖 Complete Tool Reference**: See [`mcp/CLAUDE.md`](mcp/CLAUDE.md) for detailed documentation of all 112 tools, usage examples, and advanced workflows.

## 🔄 Development Workflows

### Environment-Specific Development

```bash
# Python development
cd dev-env/python && devbox shell
devbox run test      # Run tests
devbox run lint      # Run linting
devbox run format    # Format code

# TypeScript development  
cd dev-env/typescript && devbox shell
devbox run test      # Jest tests
devbox run lint      # ESLint
devbox run format    # Prettier

# Similar patterns for Rust, Go, Nushell
```

### Container Development with DevPod (Centralized Management ✅)

```bash
# From any environment directory (unified interface)
cd dev-env/python && devbox run devpod:provision    # Create Python workspace
cd dev-env/typescript && devbox run devpod:status   # Check TypeScript workspaces
cd dev-env/rust && devbox run devpod:help           # Get Rust DevPod help

# Direct centralized management
nu host-tooling/devpod-management/manage-devpod.nu provision python
nu host-tooling/devpod-management/manage-devpod.nu status typescript
nu host-tooling/devpod-management/manage-devpod.nu help rust

# Legacy commands (still supported)
make devpod-python     # Single workspace via makefile
/devpod-python 2       # Multiple workspaces via slash commands
```

### Cross-Language Operations

```bash
# Validate all environments
make validate

# Clean all environments
make clean

# Performance analysis
make perf-report

# Security scan
make security-scan
```

## 📝 Context Engineering Framework

### Architecture Overview

The Context Engineering system provides clear separation between development and execution:

- **Workspace** (`context-engineering/workspace/`): Local PRP generation, template development, feature definitions
- **DevPod** (`context-engineering/devpod/`): Containerized execution, environment-specific configs, monitoring  
- **Shared** (`context-engineering/shared/`): Common utilities, examples (dojo/), documentation
- **Archive** (`context-engineering/archive/`): Historical tracking, performance analysis

### Workflow Examples

```bash
# Generate PRP in workspace
cd context-engineering/workspace
/generate-prp features/user-api.md --env dev-env/python

# Execute in DevPod container
/devpod-python
/execute-prp context-engineering/devpod/environments/python/PRPs/user-api-python.md --validate

# Personal productivity shortcuts (add to CLAUDE.local.md)
alias prp-gen="cd context-engineering/workspace && /generate-prp"
alias prp-exec-py="/devpod-python && /execute-prp"
```

### Enterprise Features

```bash
# Enhanced generation with dynamic templates
/generate-prp features/api.md --env python-env --include-dojo --verbose

# Enhanced execution with auto-rollback
python .claude/commands/execute-prp-v2.py context-engineering/devpod/environments/python/PRPs/api-python.md --validate --monitor
```

## 🧠 Intelligent Automation

### Auto-Formatting Hooks
Files are automatically formatted on save:
- **Python**: `ruff format`
- **TypeScript**: `prettier`
- **Rust**: `rustfmt`
- **Go**: `goimports`
- **Nushell**: `nu format`

### Auto-Testing
Tests run automatically when test files are modified:
- **Python**: `pytest` for `test_*.py`, `*_test.py`
- **TypeScript**: `jest` for `*.test.ts`, `*.spec.js`
- **Rust**: `cargo test` for `*_test.rs`
- **Go**: `go test` for `*_test.go`
- **Nushell**: `nu test` for `test_*.nu`

### Quality Gates
Pre-commit validation ensures code quality:
- Linting across all environments
- Secret scanning
- Cross-environment validation
- Performance regression detection

## 📊 Performance & Monitoring

### Real-Time Analytics
```bash
# Performance dashboard
nu dev-env/nushell/scripts/performance-analytics.nu dashboard

# Resource monitoring
nu dev-env/nushell/scripts/resource-monitor.nu watch

# Generate reports
nu dev-env/nushell/scripts/performance-analytics.nu report --days 7
```

### Optimization Recommendations
```bash
# Get optimization suggestions
nu dev-env/nushell/scripts/performance-analytics.nu optimize

# Resource cleanup
nu dev-env/nushell/scripts/resource-monitor.nu cleanup
```

## 🛡️ Security Features

### Automated Security Scanning
```bash
# Scan all environments
nu dev-env/nushell/scripts/security-scanner.nu scan-all

# Scan specific files
nu dev-env/nushell/scripts/security-scanner.nu scan-file src/main.py

# Vulnerability analysis
nu dev-env/nushell/scripts/security-scanner.nu vulnerabilities
```

### Secret Detection
- Pre-commit hooks scan for secrets in `.env`, `.config`, `.json`, `.yaml` files
- Integration with git-secrets
- Automatic blocking of commits containing secrets

## 🐛 Troubleshooting

### Common Issues

**Nushell Not Found:**
```bash
# Verify installation
which nu
nu --version

# Add to PATH if needed (macOS/Linux)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**DevBox Issues:**
```bash
# Check DevBox installation
devbox version

# Reinstall if needed
curl -fsSL https://get.jetify.com/devbox | bash
```

**Environment Not Loading:**
```bash
# Check direnv
direnv status

# Reload environment
direnv reload

# Manual activation
cd dev-env/python && devbox shell
```

**MCP Server Issues:**
```bash
# Rebuild MCP server with all 112 tools
cd mcp && npm run build

# Test server with comprehensive test suite
npm run test

# Test server startup
npm run start

# Check logs
tail -f ~/.claude/notifications.log
```

### Debug Mode

Enable verbose logging:
```bash
# Set debug environment
export MCP_LOG_LEVEL=debug
export NU_LOG_LEVEL=debug

# Run with debug
nu scripts/validate-all.nu --verbose
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Follow the established patterns**:
   - Use the unified `dev-env/` structure
   - Add tests for new functionality
   - Update documentation
   - Follow language-specific style guides
4. **Validate your changes**: `make validate`
5. **Commit and push**: `git commit -m 'feat: add amazing feature'`
6. **Create a Pull Request**

### Development Guidelines

- **Python**: Use `uv` exclusively, type hints mandatory, 88 char line length
- **TypeScript**: Strict mode, never `any`, prefer `unknown`, Result patterns
- **Rust**: Embrace ownership, avoid clones, use `Result<T, E>` + `?` operator
- **Go**: Simple explicit code, always check errors, small interfaces
- **Nushell**: `def "namespace command"` pattern, type hints, structured data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Success Metrics

### ✅ Tested & Verified Features
- **MCP Server Integration**: **112 tools** across 15 categories with comprehensive testing ✅
- **Enhanced AI Hooks**: 4 production-ready hooks with intelligent automation ✅
- **Claude-Flow Integration**: AI agent orchestration with hive-mind coordination ✅
- **AG-UI Protocol**: 5 agentic environment templates with CopilotKit integration ✅
- **Docker MCP Toolkit**: 34+ containerized tools with HTTP/SSE transport ✅
- **DevPod Multi-Environment**: 8 workspaces + 5 agentic variants across all languages ✅
- **Container Auto-Installation**: Zero-configuration AI hooks deployment ✅
- **Cross-Language Validation**: Parallel execution with intelligent quality gates ✅
- **Advanced Analytics**: ML-based performance monitoring and predictive insights ✅
- **Zero-Drift Configuration**: Single source of truth with automated synchronization ✅

### 🚀 Getting Started Commands

```bash
# Quick start
git clone https://github.com/ricable/polyglot-devenv.git
cd polyglot-devenv
make setup

# Verify everything works
make validate

# Start developing
cd dev-env/python && devbox shell
```

---

**🎉 Welcome to the future of polyglot development!** This environment combines the power of isolated development environments, intelligent automation, and seamless AI integration with **112 MCP tools**, **Enhanced AI Hooks**, **Claude-Flow orchestration**, and **AG-UI protocol** to create the most sophisticated development experience available.

## 📚 Documentation Navigation

**🏠 Getting Started** (You are here):
- This README - Project overview, quick start, essential features
- Project structure, workflows, and success metrics

**📋 Core Documentation**:
- **[`CLAUDE.md`](CLAUDE.md)** - Complete project standards, architecture, workflows, and setup instructions
- **[`CLAUDE.local.md`](CLAUDE.local.md)** - Personal productivity, aliases, IDE config, and troubleshooting

**🔧 Technical Deep Dive**:
- **[`mcp/CLAUDE.md`](mcp/CLAUDE.md)** - Complete MCP tool reference (112 tools), development guidelines, advanced features
- **[Context Engineering](context-engineering/README.md)** - Workspace/DevPod architecture and PRP workflows
- **[DevPod Automation](devpod-automation/README.md)** - Container development and agentic environments

**🚀 Advanced Features**:
- **Enhanced AI Hooks** - 4 production-ready hooks with intelligent automation  
- **Claude-Flow Integration** - AI agent orchestration with hive-mind coordination
- **AG-UI Protocol** - 5 agentic environment templates with CopilotKit integration
- **Docker MCP Toolkit** - 34+ containerized tools with HTTP/SSE transport
- **Advanced Analytics** - ML-based performance monitoring and business intelligence