# Context Engineering Framework

This project implements an enhanced context engineering system for generating and executing Product Requirements Prompts (PRPs) across multiple development environments with clear separation between development (workspace) and deployment (devpod) concerns.

## Architecture Overview

The system is organized with clear separation between development (workspace) and deployment (devpod) concerns:

```
context-engineering/
├── workspace/              # Local development & PRP generation
│   ├── features/          # Feature definitions (input)
│   ├── templates/         # PRP templates by environment
│   ├── generators/        # PRP generation tools
│   └── docs/             # Workspace usage docs
├── devpod/               # Containerized execution environment
│   ├── environments/     # Environment-specific configs
│   │   ├── python/       # Maps to /devpod-python
│   │   ├── typescript/   # Maps to /devpod-typescript
│   │   ├── rust/         # Maps to /devpod-rust
│   │   ├── go/           # Maps to /devpod-go
│   │   └── nushell/      # Maps to devbox run devpod:provision
│   ├── execution/        # Execution engines & reports
│   ├── monitoring/       # Performance & security tracking
│   └── configs/          # DevPod-specific configurations
├── shared/               # Resources used by both workspace & devpod
│   ├── examples/         # Reference examples (including dojo/)
│   ├── utils/           # Common utilities
│   ├── schemas/         # Validation schemas
│   └── docs/            # Shared documentation
└── archive/             # Historical PRPs and reports
```

## 🚀 Quick Start

### Workspace Development (PRP Generation)

```bash
# Navigate to workspace
cd context-engineering/workspace

# Create a feature definition
code features/user-api.md

# Generate a PRP for Python environment
/generate-prp features/user-api.md --env dev-env/python

# Review generated PRP
code workspace/generated-PRPs/user-api-python.md
```

### DevPod Execution (Containerized)

```bash
# Provision Python DevPod environment
/devpod-python

# Execute PRP in isolated container
/execute-prp context-engineering/devpod/environments/python/PRPs/user-api-python.md --validate

# Monitor execution
nu context-engineering/devpod/monitoring/python-monitor.nu
```

## Development Workflows

### 1. Workspace-Only Development

For local development and PRP generation:

```bash
# Generate PRP locally
cd context-engineering/workspace
/generate-prp features/chat-interface.md --env dev-env/typescript

# Review and refine templates
code templates/typescript_prp.md
```

### 2. Full DevPod Workflow

For complete isolation and production-like execution:

```bash
# Step 1: Generate in workspace
cd context-engineering/workspace
/generate-prp features/api-backend.md --env dev-env/python

# Step 2: Execute in DevPod
/devpod-python
/execute-prp context-engineering/devpod/environments/python/PRPs/api-backend-python.md

# Step 3: Monitor and review
nu context-engineering/devpod/monitoring/execution-report.nu --latest
```

### 3. Multi-Environment Development

For features spanning multiple languages:

```bash
# Generate backend PRP
/generate-prp features/api-backend.md --env dev-env/python

# Generate frontend PRP  
/generate-prp features/web-frontend.md --env dev-env/typescript

# Generate monitoring PRP
/generate-prp features/monitoring.md --env dev-env/nushell

# Execute in order with dependency management
/devpod-python && /execute-prp context-engineering/devpod/environments/python/PRPs/api-backend-python.md
/devpod-typescript && /execute-prp context-engineering/devpod/environments/typescript/PRPs/web-frontend-typescript.md
nu context-engineering/devpod/environments/nushell/execute-monitoring.nu
```

## Integration with Polyglot Environment

### DevPod Command Mapping

Each DevPod environment maps to containerized execution:

| Command | Environment | Container Pattern | Max Workspaces |
|---------|-------------|------------------|----------------|
| `/devpod-python` | `dev-env/python` | `polyglot-python-devpod-{timestamp}-{N}` | 5 |
| `/devpod-typescript` | `dev-env/typescript` | `polyglot-typescript-devpod-{timestamp}-{N}` | 5 |
| `/devpod-rust` | `dev-env/rust` | `polyglot-rust-devpod-{timestamp}-{N}` | 5 |
| `/devpod-go` | `dev-env/go` | `polyglot-go-devpod-{timestamp}-{N}` | 5 |
| `devbox run devpod:provision` | `dev-env/nushell` | Nushell automation | N/A |

### Personal Workflow Integration

Add to your `CLAUDE.local.md`:

```bash
# Context Engineering Shortcuts
alias prp-gen="cd context-engineering/workspace && /generate-prp"
alias prp-features="code context-engineering/workspace/features"
alias prp-templates="code context-engineering/workspace/templates"

# DevPod Execution Shortcuts
alias prp-exec-py="/devpod-python && /execute-prp"
alias prp-exec-ts="/devpod-typescript && /execute-prp"
alias prp-exec-rust="/devpod-rust && /execute-prp"
alias prp-exec-go="/devpod-go && /execute-prp"

# Complete Workflow
personal-prp-workflow() {
    local feature_name=$1
    local environment=${2:-python}
    
    echo "🚀 Starting PRP workflow for $feature_name in $environment environment"
    
    # Generate PRP in workspace
    cd context-engineering/workspace
    /generate-prp features/$feature_name.md --env dev-env/$environment
    
    # Execute in DevPod environment
    case $environment in
        python) /devpod-python && /execute-prp context-engineering/devpod/environments/python/PRPs/$feature_name-python.md ;;
        typescript) /devpod-typescript && /execute-prp context-engineering/devpod/environments/typescript/PRPs/$feature_name-typescript.md ;;
        rust) /devpod-rust && /execute-prp context-engineering/devpod/environments/rust/PRPs/$feature_name-rust.md ;;
        go) /devpod-go && /execute-prp context-engineering/devpod/environments/go/PRPs/$feature_name-go.md ;;
    esac
}
```

## Environment-Specific Features

### Python Environment
- **Framework**: FastAPI with async/await patterns
- **Database**: SQLAlchemy async with PostgreSQL
- **Testing**: pytest-asyncio with comprehensive coverage
- **Package Management**: uv exclusively (no pip/poetry)
- **Container**: Python 3.12.11 + uv 0.7.19

### TypeScript Environment
- **Framework**: Modern Node.js with ES modules
- **Type Safety**: Strict TypeScript mode, no `any` types
- **Testing**: Jest with comprehensive test patterns
- **Quality**: ESLint + Prettier with strict rules
- **Container**: Node.js 20.19.3 + TypeScript 5.8.3

### Rust Environment
- **Runtime**: Tokio for async operations
- **Memory Safety**: Leverage ownership system effectively
- **Error Handling**: Result<T, E> with custom error types
- **Testing**: Cargo test with comprehensive scenarios
- **Container**: Latest stable Rust + Cargo

### Go Environment
- **Patterns**: Clean, simple Go with context patterns
- **Error Handling**: Explicit error checking with context
- **Testing**: Table-driven tests
- **Interfaces**: Small, focused interface design
- **Container**: Go 1.22+ with standard toolchain

### Nushell Environment
- **Data Processing**: Structured data with built-in types
- **Automation**: Cross-environment orchestration
- **Type Safety**: Type hints for all parameters
- **Integration**: DevBox automation and monitoring
- **Container**: Nushell 0.105.1 with devbox

## Performance & Monitoring

### Built-in Analytics

The system includes comprehensive performance tracking:

```bash
# Performance dashboard
nu context-engineering/devpod/monitoring/dashboard.nu

# Environment-specific metrics
nu dev-env/nushell/scripts/performance-analytics.nu report --env python

# Cross-environment validation
nu scripts/validate-all.nu --parallel

# Resource optimization
nu dev-env/nushell/scripts/resource-monitor.nu optimize
```

### Quality Gates

Each environment enforces quality standards:
- **Linting**: Environment-specific linters (ruff, eslint, clippy, golangci-lint)
- **Type Checking**: mypy (Python), TypeScript strict mode, Rust ownership
- **Testing**: Minimum 80% coverage with framework-specific patterns
- **Security**: Secret scanning, dependency vulnerability checks
- **Performance**: Build time tracking, resource usage monitoring

## Advanced Features

### CopilotKit Integration

The `shared/examples/dojo/` directory provides complete Next.js patterns:
- **Agentic Chat**: Interactive AI conversations
- **Generative UI**: Dynamic interface generation
- **Human in the Loop**: Human oversight patterns
- **Shared State**: Cross-component state management
- **Tool-based UI**: Tool-driven interface generation

### Multi-Agent Systems

Support for complex multi-agent workflows:
- **Agent Coordination**: Multiple AI agents working together
- **Task Distribution**: Parallel execution across agents
- **State Synchronization**: Shared state management
- **Communication Patterns**: Inter-agent messaging

### Version Control & Archiving

- **Historical Tracking**: All PRPs and executions archived
- **Performance Analysis**: Track improvements over time
- **Pattern Recognition**: Identify successful implementations
- **Regression Testing**: Use archived PRPs for validation

## Getting Started

### 1. Understand the Architecture

Review the README files in each directory:
- `workspace/README.md` - PRP generation and development
- `devpod/README.md` - Containerized execution
- `shared/README.md` - Common resources and utilities
- `archive/README.md` - Historical tracking and analysis

### 2. Set Up Personal Workflows

Add context engineering integration to your `CLAUDE.local.md`:

```bash
# Copy examples from CLAUDE.local.md.template
cp CLAUDE.local.md.template CLAUDE.local.md

# Add context engineering sections
code CLAUDE.local.md
```

### 3. Create Your First Feature

```bash
# Create feature definition
cd context-engineering/workspace
code features/my-first-feature.md

# Generate PRP
/generate-prp features/my-first-feature.md --env dev-env/python

# Execute in DevPod
/devpod-python
/execute-prp context-engineering/devpod/environments/python/PRPs/my-first-feature-python.md
```

### 4. Monitor and Iterate

```bash
# Check execution results
nu context-engineering/devpod/monitoring/execution-report.nu

# Analyze performance
nu dev-env/nushell/scripts/performance-analytics.nu dashboard

# Review archived results
ls context-engineering/archive/PRPs/
```

## Best Practices

### 1. Clear Separation of Concerns
- **Generate** in workspace for development and iteration
- **Execute** in devpod for isolation and production-like conditions
- **Archive** results for historical analysis and learning

### 2. Environment-Specific Optimization
- Use language-appropriate patterns and frameworks
- Leverage environment-specific tooling and best practices
- Follow established conventions for each language

### 3. Comprehensive Testing
- Include test requirements in feature definitions
- Use environment-specific testing frameworks
- Maintain high coverage standards

### 4. Continuous Monitoring
- Track performance across all environments
- Monitor resource usage and optimization opportunities
- Use analytics for continuous improvement

## Resources

- **Integration Guide**: `workspace/docs/integration-guide.md`
- **Architecture Docs**: `shared/docs/` directory
- **Performance Analytics**: `dev-env/nushell/scripts/performance-analytics.nu`
- **Cross-Environment Validation**: `scripts/validate-all.nu`
- **DevPod Documentation**: `devpod/README.md` and environment-specific docs