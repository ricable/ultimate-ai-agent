# Edge-Native AI SaaS Platform

Decentralized Edge-Native AI Architecture using Kairos, SpinKube, WasmEdge, and Agentic Protocols with **mise** for reproducible environments.

## Overview

This platform provides a comprehensive framework for deploying distributed AI agents across edge infrastructure using:

- **mise** - Reproducible environment manager for all agent categories
- **Kairos** - Immutable OS for edge nodes with K3s
- **SpinKube/WasmEdge** - WebAssembly runtimes for lightweight agents
- **kagenti-operator** - Kubernetes operator for AI agent lifecycle
- **claude-flow** - SPARC methodology and agent coordination

## Quick Start

### Prerequisites

```bash
# Install mise (https://mise.jdx.dev)
curl https://mise.run | sh

# Add to shell (bash/zsh)
echo 'eval "$(mise activate bash)"' >> ~/.bashrc
# or for zsh
echo 'eval "$(mise activate zsh)"' >> ~/.zshrc
```

### Setup Environment

```bash
# Clone repository
git clone https://github.com/ricable/ruvector-slef-learning.git
cd ruvector-slef-learning

# Trust mise configuration
mise trust

# Install all tools and dependencies
mise run setup

# Or install by category
./scripts/mise/install-tools.sh wasm    # WASM agents
./scripts/mise/install-tools.sh python  # Python/ML agents
./scripts/mise/install-tools.sh rust    # Rust agents
./scripts/mise/install-tools.sh infra   # Kubernetes tools
```

### Development

```bash
# Start local development environment
mise run dev

# Build all agents
mise run build

# Run tests
mise run test

# Lint code
mise run lint
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    mise Environment Manager                      │
├─────────────────────────────────────────────────────────────────┤
│  agents/wasm/     │  agents/python/  │  agents/rust/  │ agents/ │
│  mise.toml        │  mise.toml       │  mise.toml     │ infra/  │
│  ├─ node@22       │  ├─ python@3.12  │  ├─ rust       │ mise.   │
│  ├─ bun           │  ├─ uv           │  ├─ wasm-pack  │ toml    │
│  ├─ wasm-pack     │  ├─ ruff         │  ├─ cross      │         │
│  └─ spin          │  └─ pytest       │  └─ cargo-*    │         │
├─────────────────────────────────────────────────────────────────┤
│                    Deployment Targets                            │
├──────────────────┬──────────────────┬───────────────────────────┤
│  Docker (Mac)    │  Kairos + K3s    │  kagenti-operator         │
│  Local Dev       │  Edge Nodes      │  Kubernetes               │
└──────────────────┴──────────────────┴───────────────────────────┘
```

## Agent Categories

### WASM Agents (`agents/wasm/`)
WebAssembly-based agents using WasmEdge or Spin runtimes.

```bash
# Build WASM agent
mise run build-wasm

# Run locally with WasmEdge
mise run run-wasmedge

# Deploy to SpinKube
mise run spinapp-deploy myagent ghcr.io/ruvnet/wasm-agent:latest
```

### Python Agents (`agents/python/`)
FastAPI-based agents for ML/AI workloads.

```bash
# Setup Python environment
cd agents/python && mise run setup

# Run FastAPI locally
mise run run

# Build Docker image
mise run docker-build
```

### Rust Agents (`agents/rust/`)
High-performance WASM agents compiled from Rust.

```bash
# Build for WASM
cd agents/rust && mise run build-wasm

# Cross-compile for multiple targets
mise run cross-build
```

### Infrastructure (`agents/infra/`)
Kubernetes and infrastructure management tools.

```bash
# Apply Kubernetes manifests
mise run k8s-apply

# Install kagenti-operator
mise run kagenti-install

# Deploy Platform CR
mise run kagenti-platform config/kagenti/platform.yaml
```

## mise Tasks Reference

### Core Tasks
| Task | Description |
|------|-------------|
| `mise run setup` | Initialize environment with all dependencies |
| `mise run dev` | Start local Docker development environment |
| `mise run build` | Build all agent artifacts |
| `mise run test` | Run all tests |
| `mise run lint` | Lint and format code |

### Agent Lifecycle
| Task | Description |
|------|-------------|
| `mise run agent-create <name> <type>` | Create new agent from template |
| `mise run agent-list` | List all available agents |
| `mise run agent-deploy <path>` | Deploy agent to cluster |

### Kubernetes
| Task | Description |
|------|-------------|
| `mise run k8s-apply` | Apply Kubernetes manifests |
| `mise run k8s-status` | Check cluster status |
| `mise run k8s-delete` | Remove Kubernetes resources |

### Secrets Management
| Task | Description |
|------|-------------|
| `mise run secrets-init` | Initialize age encryption |
| `mise run secrets-set <name>` | Set encrypted secret |

### Shims (for CI/CD, Docker)
| Task | Description |
|------|-------------|
| `mise run shims-setup` | Setup shims for non-interactive environments |
| `mise run shims-verify` | Verify shims are working |

### kagenti-operator
| Task | Description |
|------|-------------|
| `mise run kagenti-install` | Install operator to cluster |
| `mise run kagenti-platform <file>` | Deploy Platform CR |
| `mise run kagenti-component <file>` | Deploy Component CR |
| `mise run kagenti-status` | Check kagenti resources |

### SpinKube/WASM
| Task | Description |
|------|-------------|
| `mise run spinkube-install` | Install SpinKube operator |
| `mise run spinapp-deploy <name> <image>` | Deploy SpinApp |
| `mise run shim-install` | Install containerd WASM shims |

## Deployment Options

### Local Development (Docker on Mac)

```bash
# Start full development environment
docker compose -f config/docker/docker-compose.mise.yaml up -d

# Or use mise task
mise run dev
```

### Kairos + K3s Cluster

```bash
# Prepare Kairos node configuration
mise run kairos-prepare

# Join new node to cluster
K3S_URL=https://master:6443 K3S_TOKEN=xxx mise run k3s-agent
```

### kagenti-operator on Kubernetes

```bash
# Install operator
mise run kagenti-install

# Deploy platform with all agents
kubectl apply -f config/kagenti/platform.yaml

# Deploy individual components
kubectl apply -f config/kagenti/components/
```

## Project Structure

```
├── mise.toml                    # Root mise configuration
├── agents/
│   ├── wasm/mise.toml          # WASM agent environment
│   ├── python/mise.toml        # Python agent environment
│   ├── rust/mise.toml          # Rust agent environment
│   └── infra/mise.toml         # Infrastructure tools
├── apps/
│   ├── wasmedge-js/            # WasmEdge JavaScript agent
│   ├── spin-js/                # Spin JavaScript agent
│   └── fastapi/                # Python FastAPI agent
├── config/
│   ├── docker/                 # Docker configurations
│   ├── kairos/                 # Kairos cloud-config
│   ├── kagenti/                # kagenti Platform/Component CRs
│   └── k8s/                    # Kubernetes manifests
├── infrastructure/
│   ├── k3s/                    # K3s cluster config
│   ├── kairos/                 # Kairos OS config
│   ├── spinkube/               # SpinKube operator
│   └── wasmedge/               # WasmEdge runtime
├── scripts/
│   └── mise/                   # mise setup scripts
└── gateway/
    ├── mcp-gateway/            # MCP protocol gateway
    └── litellm/                # LiteLLM proxy
```

## Secrets Management

This project uses **age** encryption for secrets via mise:

```bash
# Initialize age key (one-time)
mise run secrets-init

# Set encrypted secret
mise run secrets-set ANTHROPIC_API_KEY

# Secrets are stored encrypted in mise.local.toml
# Never commit age-key.txt or secrets.toml!
```

## Shims for Non-Interactive Environments

For Docker, CI/CD, and Kubernetes, use mise shims:

```bash
# Setup shims
./scripts/mise/setup-shims.sh

# In Dockerfile or CI
export PATH="$HOME/.local/share/mise/shims:$PATH"
node --version  # Uses mise-managed node
```

## Integration with claude-flow

```bash
# Initialize claude-flow
mise run flow-init

# Run SPARC development workflow
mise run flow-sparc architect "design agent system"

# Run TDD workflow
mise run flow-tdd "implement vector search"
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Use mise for consistent environment (`mise install`)
4. Run tests (`mise run test`)
5. Commit changes
6. Push and create PR

## License

MIT License - see LICENSE file for details.

## Resources

- [mise Documentation](https://mise.jdx.dev/)
- [kagenti-operator](https://github.com/kagenti/kagenti-operator)
- [SpinKube](https://www.spinkube.dev/)
- [WasmEdge](https://wasmedge.org/)
- [Kairos](https://kairos.io/)
- [claude-flow](https://github.com/ruvnet/claude-flow)
