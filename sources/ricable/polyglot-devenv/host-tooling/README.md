# Host Tooling

**PURPOSE**: Scripts and configurations that run on the developer's local machine (host), NOT inside DevPod containers.

## Clear Separation Principle

This directory enforces a clear "air gap" between host and container responsibilities:

- **Host Tooling** (this directory): Local machine setup, DevPod management, infrastructure access
- **Container Code** (`dev-env/`, `context-engineering/`): Development tools that run inside isolated containers

## Directory Structure

```
host-tooling/
├── installation/           # Host dependency installation
│   ├── docker-setup.nu     # Docker & DevPod installation/configuration
│   └── shell-integration/  # Host shell configuration
├── devpod-management/      # DevPod lifecycle management (host-side)
│   ├── devpod-generate.nu  # Container generation
│   ├── devpod-manage.nu    # Container lifecycle
│   ├── devpod-provision.nu # Container provisioning
│   └── devpod-sync.nu      # Host-container synchronization
├── monitoring/             # Infrastructure & external services
│   ├── kubernetes.nu       # Kubernetes cluster management
│   └── github.nu           # GitHub integration (requires host credentials)
└── shell-integration/      # Host shell enhancements
    └── aliases.sh          # Host-specific aliases
```

## Host vs Container Responsibilities

### Host Responsibilities (This Directory)
- **Installation**: Installing Docker, DevPod, and system dependencies
- **DevPod Management**: Creating, starting, stopping, and destroying containers
- **Infrastructure Access**: Kubernetes clusters, cloud services, external APIs
- **Credential Management**: Host-stored secrets, SSH keys, API tokens
- **Shell Integration**: Host shell aliases, environment variables
- **System Monitoring**: Host resource usage, container health checks

### Container Responsibilities (dev-env/, context-engineering/)
- **Development Tools**: Language runtimes, linters, formatters, test runners
- **Code Processing**: Source code formatting, testing, building
- **Application Logic**: Actual development work and code execution
- **Package Management**: Language-specific package installation
- **Local Development**: Code editing, debugging, local server running

## Security Benefits

1. **Credential Isolation**: Host credentials never enter containers
2. **Dependency Isolation**: Container tools don't pollute host system
3. **Access Control**: Infrastructure access limited to host environment
4. **Backup Safety**: Host-only tools for system backup and recovery

## Usage Guidelines

### When to Put Scripts Here (Host)
- Installs system dependencies (Docker, DevPod)
- Manages container lifecycles
- Accesses external infrastructure (K8s, GitHub)
- Requires host credentials or system access
- Configures host shell or environment

### When to Put Scripts in Containers
- Formats, lints, or tests source code
- Runs language-specific tools
- Processes application code
- Manages project dependencies
- Serves development applications

## Migration Impact

Scripts moved to this directory from their previous locations:

**From `devpod-automation/scripts/`**:
- `docker-setup.nu` → `host-tooling/installation/`
- `devpod-*.nu` → `host-tooling/devpod-management/`

**From `dev-env/nushell/scripts/`**:
- `kubernetes.nu` → `host-tooling/monitoring/`
- `github.nu` → `host-tooling/monitoring/`

## Integration

Host tooling integrates with containers through:
- **Volume Mounts**: Source code synchronized to containers
- **Port Forwarding**: Container services accessible on host
- **Command Execution**: Host can execute commands in containers
- **File Watching**: Host monitors files, triggers container actions

This separation ensures reproducible development environments while maintaining clear security boundaries.