# Context Engineering DevPod

The **devpod** directory contains all resources for containerized PRP execution in isolated environments.

## Structure

- **environments/** - Environment-specific configurations for each language
  - `python/` - Python DevPod configuration and execution (maps to `/devpod-python`)
  - `typescript/` - TypeScript DevPod configuration (maps to `/devpod-typescript`) 
  - `rust/` - Rust DevPod configuration (maps to `/devpod-rust`)
  - `go/` - Go DevPod configuration (maps to `/devpod-go`)
  - `nushell/` - Nushell DevPod configuration (maps to `devbox run devpod:provision`)
- **execution/** - Execution engines, reports, and monitoring
- **monitoring/** - Performance and security tracking for DevPod executions
- **configs/** - DevPod-specific configurations and settings

## Workflow

1. **Receive PRPs**: PRPs generated in workspace are executed here
2. **Provision Environments**: DevPod containers are provisioned for execution
3. **Execute PRPs**: PRPs run in isolated, containerized environments
4. **Monitor & Report**: Execution is monitored and results are tracked

## DevPod Integration

Each environment directory maps to DevPod commands:

```bash
# Python DevPod execution
/devpod-python    # Creates: polyglot-python-devpod-{timestamp}-{N}
/execute-prp context-engineering/devpod/environments/python/PRPs/feature-python.md

# TypeScript DevPod execution  
/devpod-typescript    # Creates: polyglot-typescript-devpod-{timestamp}-{N}
/execute-prp context-engineering/devpod/environments/typescript/PRPs/feature-typescript.md

# Multi-workspace provisioning
/devpod-python 3     # Creates 3 Python workspaces for parallel development
```

## Personal Workflow Integration

```bash
# Personal aliases (add to CLAUDE.local.md)
alias prp-exec-py="/devpod-python && /execute-prp"
alias prp-exec-ts="/devpod-typescript && /execute-prp"
alias prp-exec-rust="/devpod-rust && /execute-prp"
alias prp-exec-go="/devpod-go && /execute-prp"
```

## Environment Features

### Python Environment
- **Container**: Python 3.12.11 + uv 0.7.19
- **Features**: FastAPI, async/await, SQLAlchemy, pytest-asyncio
- **VS Code**: Python, Pylance, autopep8 extensions

### TypeScript Environment  
- **Container**: Node.js 20.19.3 + TypeScript 5.8.3
- **Features**: Strict mode, ES modules, Jest, ESLint/Prettier
- **VS Code**: ESLint for JavaScript/TypeScript extensions

### Rust Environment
- **Container**: Latest stable Rust + Cargo
- **Features**: Tokio async, serde, thiserror, cargo testing
- **VS Code**: rust-analyzer, CodeLLDB, TOML extensions

### Go Environment
- **Container**: Go 1.22+ with standard toolchain
- **Features**: Context patterns, interfaces, table-driven tests
- **VS Code**: Go extension, ESLint

### Nushell Environment
- **Container**: Nushell 0.105.1 with devbox
- **Features**: Structured data, automation, cross-environment orchestration
- **VS Code**: nushell-vscode-extension

## Monitoring & Performance

Execution monitoring includes:
- **Performance Tracking**: Build times, test execution, resource usage
- **Security Scanning**: Vulnerability detection, secret scanning
- **Quality Gates**: Linting, type checking, test coverage
- **Resource Management**: Container lifecycle, cleanup, optimization

## Archive Integration

Completed executions and their results are archived in `../archive/` for historical tracking and analysis.