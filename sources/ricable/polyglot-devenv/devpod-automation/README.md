# DevPod Automation for Polyglot Development Environment

This directory contains automated DevPod provisioning and management tools for the polyglot development environment.

## Overview

The DevPod automation system provides containerized development environments that complement the existing Devbox-based setup. Each language environment can be deployed as an isolated DevPod workspace with full toolchain and IDE support.

## Quick Start

```bash
# Complete DevPod setup and configuration (first time only)
nu scripts/docker-setup.nu --install --configure --optimize

# Option 1: Use simple bash scripts (recommended)
bash scripts/provision-python.sh      # Provision Python workspace
bash scripts/provision-typescript.sh  # Provision TypeScript workspace
bash scripts/provision-rust.sh        # Provision Rust workspace
bash scripts/provision-go.sh          # Provision Go workspace
bash scripts/provision-nushell.sh     # Provision Nushell workspace

# Option 2: Use from within environments
cd ../python-env && devbox run devpod:provision
cd ../typescript-env && devbox run devpod:provision
cd ../rust-env && devbox run devpod:provision
cd ../go-env && devbox run devpod:provision
cd ../nushell-env && devbox run devpod:provision

# Option 3: Use management script
bash scripts/provision-all.sh status      # Check all workspace status
bash scripts/provision-all.sh provision   # Interactive provisioning
bash scripts/provision-all.sh list        # List all available scripts
```

## Directory Structure

```
devpod-automation/
├── scripts/                    # Automation scripts
│   ├── provision-python.sh    # Python workspace provisioning
│   ├── provision-typescript.sh # TypeScript workspace provisioning
│   ├── provision-rust.sh      # Rust workspace provisioning
│   ├── provision-go.sh        # Go workspace provisioning
│   ├── provision-nushell.sh   # Nushell workspace provisioning
│   ├── provision-all.sh       # Management and overview script
│   ├── docker-setup.nu        # Docker provider setup
│   ├── devpod-provision.nu    # Advanced provisioning orchestrator
│   ├── devpod-generate.nu     # DevContainer generation
│   ├── devpod-manage.nu       # Workspace management
│   └── devpod-sync.nu         # Environment synchronization
├── templates/               # DevContainer templates
│   ├── base/               # Base template
│   ├── python/             # Python-specific template
│   ├── typescript/         # TypeScript-specific template
│   ├── rust/               # Rust-specific template
│   ├── go/                 # Go-specific template
│   ├── nushell/            # Nushell-specific template
│   └── full-stack/         # Combined template
├── config/                 # Configuration files
│   ├── devpod-provider.yaml # Docker provider configuration
│   └── workspace-defaults.json # Default workspace settings
└── README.md               # This file
```

## Features

- **Automated Provisioning**: One-command setup for any language environment
- **DevBox Integration**: Generates devcontainer.json from existing devbox.json
- **Multiple Workspaces**: Separate containers per language or combined full-stack
- **IDE Support**: VS Code, JetBrains, SSH, and terminal access
- **Performance Optimization**: Registry caching, prebuilds, resource monitoring
- **Intelligence Integration**: Works with existing monitoring and analytics
- **Lifecycle Management**: Create, start, stop, delete, and recreate workspaces

## Available Workspaces

| Workspace | Description | Base Image |
|-----------|-------------|------------|
| `polyglot-python-devpod` | Python development environment | Python 3.12 + uv |
| `polyglot-typescript-devpod` | TypeScript/Node.js environment | Node.js 20 + TypeScript |
| `polyglot-rust-devpod` | Rust development environment | Rust toolchain |
| `polyglot-go-devpod` | Go development environment | Go 1.22 |
| `polyglot-nushell-devpod` | Nushell scripting environment | Nushell + automation tools |
| `polyglot-full-devpod` | Full polyglot environment | All languages combined |

## Integration with Existing Workflow

The DevPod automation integrates seamlessly with the existing Devbox workflow:

- **Devbox First**: Continue using Devbox for local development
- **DevPod Enhancement**: Use DevPod for containerized, isolated environments
- **Automatic Sync**: Changes in devbox.json automatically update devcontainer.json
- **Shared Intelligence**: DevPod workspaces integrate with existing monitoring systems
- **Consistent Tooling**: Same development tools and scripts across environments

## Prerequisites

- Docker installed and running
- DevPod CLI installed (automated setup available)
- Devbox environments already configured
- Nushell for automation scripts

## Configuration

See `config/` directory for:
- Docker provider settings
- Workspace default configurations
- Performance optimization settings
- Registry caching setup