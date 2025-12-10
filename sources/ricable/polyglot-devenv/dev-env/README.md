# Development Environments (Container-Only)

**PURPOSE**: Isolated development environments that run INSIDE DevPod containers, NOT on the host machine.

## Container Execution Principle

All scripts and tools in this directory are designed to run exclusively inside containerized development environments:

- **Isolated Dependencies**: Each language has its own container with specific tool versions
- **Reproducible Environments**: Identical setup across all developers and CI/CD
- **Host Protection**: Host system remains clean and unmodified
- **Security Boundary**: Development work isolated from host credentials and system

## Directory Structure

```
dev-env/
├── python/                 # Python development container
│   ├── devbox.json         # Python tool configuration
│   ├── pyproject.toml      # Python project configuration
│   └── src/                # Python source code
├── typescript/             # TypeScript/Node.js container  
│   ├── devbox.json         # Node.js tool configuration
│   ├── package.json        # NPM project configuration
│   └── src/                # TypeScript source code
├── rust/                   # Rust development container
│   ├── devbox.json         # Rust tool configuration  
│   ├── Cargo.toml          # Rust project configuration
│   └── src/                # Rust source code
├── go/                     # Go development container
│   ├── devbox.json         # Go tool configuration
│   ├── go.mod              # Go module configuration
│   └── cmd/                # Go source code
└── nushell/                # Nushell scripting container
    ├── devbox.json         # Nushell tool configuration
    ├── common.nu           # Shared Nushell utilities
    ├── config/             # Nushell configuration
    └── scripts/            # Container automation scripts
```

## Container vs Host Responsibilities

### Container Responsibilities (This Directory)
- **Language Runtimes**: Python, Node.js, Rust, Go, Nushell interpreters
- **Development Tools**: Linters, formatters, test runners, debuggers
- **Code Processing**: Source code analysis, building, testing
- **Package Management**: pip/uv, npm, cargo, go mod
- **Application Serving**: Development servers and applications
- **Code Quality**: Formatting, linting, type checking, testing

### Host Responsibilities (host-tooling/)
- **Container Management**: Creating, starting, stopping containers
- **System Installation**: Docker, DevPod, host dependencies
- **Infrastructure Access**: Kubernetes, GitHub, external APIs
- **Credential Management**: API keys, SSH keys, cloud credentials
- **Host Shell Integration**: Aliases, environment variables

## Container Scripts

### Nushell Automation Scripts (`nushell/scripts/`)

**Container-only scripts** (run inside Nushell container):
- `format.nu` - Format source code inside container
- `test.nu` - Run tests with container tools
- `check.nu` - Validate code syntax and style
- `setup.nu` - Initialize container development environment
- `validate.nu` - Cross-language validation
- `performance-analytics.nu` - Container performance monitoring
- `resource-monitor.nu` - Container resource usage
- `security-scanner.nu` - Code security analysis
- `test-intelligence.nu` - Automated test analysis

## Entry Points

### Access Container Environments

Use HOST commands to enter containers:

```bash
# From host (uses host-tooling aliases)
enter-python      # SSH into Python container
enter-typescript  # SSH into TypeScript container  
enter-rust        # SSH into Rust container
enter-go          # SSH into Go container

# Or use DevPod directly
devpod ssh <workspace-name>
```

### Run Container Commands

```bash
# Inside Python container
devbox run format   # Format Python code
devbox run test     # Run Python tests
devbox run lint     # Lint Python code

# Inside TypeScript container  
devbox run format   # Format TypeScript code
devbox run test     # Run Jest tests
devbox run lint     # Run ESLint

# Inside Rust container
devbox run format   # Format Rust code
devbox run test     # Run cargo tests
devbox run lint     # Run clippy

# Inside Go container
devbox run format   # Format Go code
devbox run test     # Run go tests
devbox run lint     # Run golangci-lint
```

## Isolation Benefits

1. **Version Consistency**: Exact tool versions specified in devbox.json
2. **Dependency Isolation**: Container dependencies don't affect host
3. **Reproducibility**: Identical environments across developers
4. **Security**: Code processing isolated from host system
5. **Cleanup**: Container removal completely cleans environment

## Integration with Host

Containers integrate with host through DevPod:
- **File Synchronization**: Source code mounted from host
- **Port Forwarding**: Container services accessible on host
- **SSH Access**: Host can shell into containers
- **Command Execution**: Host can run commands in containers

## Migration Notes

Scripts that should NOT be moved to host-tooling/:
- All `dev-env/*/devbox.json` configuration files
- All source code in `src/`, `cmd/` directories
- Container-specific scripts in `nushell/scripts/` except infrastructure tools
- Language-specific configuration files (pyproject.toml, package.json, etc.)

These remain in containers because they require containerized language runtimes and tools.