# Polyglot Development Environment MCP Server

This MCP server provides comprehensive tools for managing and working with polyglot development environments that use DevBox for package management and DevPod for containerized development.

## Features

### Environment Management
- **Environment Detection**: Automatically detect Python, TypeScript, Rust, Go, and Nushell environments
- **Environment Validation**: Validate configuration and health of development environments
- **Environment Information**: Get detailed information about specific environments

### DevBox Integration
- **DevBox Shell**: Access isolated development environments
- **Script Execution**: Run DevBox scripts with progress tracking
- **Package Management**: Add packages to environments
- **Status Monitoring**: Check DevBox environment status

### DevPod Workspace Management
- **Multi-Workspace Provisioning**: Create 1-10 workspaces per environment
- **Workspace Listing**: List all DevPod workspaces with filtering
- **Status Monitoring**: Get detailed workspace status information
- **VS Code Integration**: Automatic VS Code setup with language extensions

### Cross-Language Quality Assurance
- **Polyglot Checks**: Comprehensive quality checks across all environments
- **Parallel Validation**: Fast cross-environment validation
- **Environment Cleanup**: Clean up build artifacts and caches

### Performance Analytics
- **Command Measurement**: Measure and record performance of operations
- **Performance Reports**: Generate performance analysis reports
- **Duration Tracking**: Track execution times across environments

### Security Scanning
- **Secret Detection**: Scan for exposed secrets in configuration files
- **Vulnerability Scanning**: Check for security vulnerabilities (planned)
- **Security Reports**: Generate security analysis reports

### Hook Management
- **Hook Status**: Monitor Claude Code hooks configuration
- **Hook Triggering**: Manually trigger hooks for testing (planned)
- **Hook History**: View hook execution history (planned)

### Context Engineering (PRP)
- **PRP Generation**: Generate Project Request Proposals from feature files
- **PRP Execution**: Execute PRPs with validation and monitoring
- **Environment-Specific PRPs**: Target specific development environments

## Tool Categories

### Environment Tools
- `environment_detect`: Detect all polyglot environments
- `environment_info`: Get detailed environment information
- `environment_validate`: Validate environment health

### DevBox Tools
- `devbox_shell`: Enter DevBox environment shells
- `devbox_run`: Execute DevBox scripts
- `devbox_status`: Check DevBox status
- `devbox_add_package`: Add packages to environments

### DevPod Tools
- `devpod_provision`: Provision multiple workspaces
- `devpod_list`: List all workspaces
- `devpod_status`: Get workspace status

### Cross-Language Tools
- `polyglot_check`: Comprehensive quality checks
- `polyglot_validate`: Cross-environment validation
- `polyglot_clean`: Environment cleanup

### Performance Tools
- `performance_measure`: Measure command performance
- `performance_report`: Generate performance reports

### Security Tools
- `security_scan`: Run security scans

### Hook Tools
- `hook_status`: Check hooks configuration
- `hook_trigger`: Trigger hooks manually

### PRP Tools
- `prp_generate`: Generate PRPs from features
- `prp_execute`: Execute PRP files

## Supported Environments

### Python Environment (`python-env`)
- **Packages**: Python 3.12, uv, ruff, mypy, pytest
- **Features**: Type checking, linting, testing, package management
- **DevPod**: Python development container with VS Code extensions

### TypeScript Environment (`typescript-env`)
- **Packages**: Node.js 20, TypeScript, ESLint, Prettier, Jest
- **Features**: Type checking, linting, formatting, testing
- **DevPod**: Node.js development container with TypeScript extensions

### Rust Environment (`rust-env`)
- **Packages**: Rust compiler, Cargo, Clippy, rustfmt
- **Features**: Compilation, linting, formatting, testing
- **DevPod**: Rust development container with rust-analyzer

### Go Environment (`go-env`)
- **Packages**: Go compiler, golangci-lint, goimports
- **Features**: Compilation, linting, formatting, testing
- **DevPod**: Go development container with Go extensions

### Nushell Environment (`nushell-env`)
- **Packages**: Nushell shell, automation scripts
- **Features**: Script validation, automation, orchestration
- **DevPod**: Nushell scripting environment

## Integration Points

### Claude Code Hooks
- **Auto-formatting**: Triggered on file edits
- **Auto-testing**: Triggered on test file modifications
- **Pre-commit validation**: Triggered before Git commits
- **Security scanning**: Triggered on configuration file changes

### Performance Analytics
- **Command tracking**: Records execution times and results
- **Resource monitoring**: Tracks memory and CPU usage
- **Optimization recommendations**: Suggests performance improvements

### Security Integration
- **Secret detection**: Scans for exposed credentials
- **Configuration validation**: Checks security settings
- **Vulnerability tracking**: Monitors dependency security

### DevPod Automation
- **Workspace provisioning**: Creates isolated development environments
- **Resource management**: Manages container lifecycle
- **VS Code integration**: Automatic IDE setup and extensions

## Usage Examples

### Basic Environment Management
```
# Detect all environments
environment_detect

# Get detailed info about Python environment
environment_info {"environment": "python-env"}

# Validate all environments
environment_validate
```

### DevBox Operations
```
# Run tests in Python environment
devbox_run {"environment": "python-env", "script": "test"}

# Add a package to TypeScript environment
devbox_add_package {"environment": "typescript-env", "package": "lodash"}
```

### DevPod Workspace Management
```
# Provision 3 Python workspaces
devpod_provision {"environment": "python-env", "count": 3}

# List all workspaces
devpod_list

# Check workspace status
devpod_status {"workspace": "polyglot-python-devpod-20241207-123456-1"}
```

### Quality Assurance
```
# Comprehensive check across all environments
polyglot_check {"include_security": true, "include_performance": true}

# Parallel validation
polyglot_validate {"parallel": true}

# Clean up all environments
polyglot_clean
```

### Performance and Security
```
# Measure build performance
performance_measure {"command": "npm run build", "environment": "typescript-env"}

# Security scan
security_scan {"scan_type": "all"}
```

### Context Engineering
```
# Generate PRP from feature file
prp_generate {"feature_file": "features/user-api.md", "environment": "python-env"}

# Execute PRP with validation
prp_execute {"prp_file": "context-engineering/PRPs/user-api-python.md", "validate": true}
```

## Technical Implementation

### Architecture
- **TypeScript-based**: Full type safety and modern JavaScript features
- **MCP SDK Integration**: Complete Model Context Protocol compliance
- **Cross-platform**: Works on macOS, Linux, and Windows
- **Async Operations**: Non-blocking operations with progress tracking

### Error Handling
- **Graceful Degradation**: Continues operation when individual tools fail
- **Detailed Error Messages**: Comprehensive error reporting with context
- **Timeout Management**: Prevents hanging operations
- **Recovery Suggestions**: Provides actionable error resolution steps

### Performance Optimization
- **Parallel Execution**: Runs operations concurrently when possible
- **Progress Tracking**: Real-time progress updates for long operations
- **Resource Management**: Efficient memory and CPU usage
- **Caching**: Intelligent caching of expensive operations

### Security Features
- **Input Validation**: Comprehensive input sanitization
- **Path Validation**: Prevents directory traversal attacks
- **Command Injection Prevention**: Safe command execution
- **Secret Detection**: Automatic credential scanning

## Installation and Setup

### Prerequisites
- Node.js 20+ and npm
- DevBox CLI tool
- DevPod CLI tool (for workspace management)
- Git and standard development tools

### Installation
```bash
# Navigate to the MCP server directory
cd mcp

# Install dependencies
npm install

# Build the server
npm run build

# Test the server
npm run start
```

### Claude Code Integration
```bash
# Add the server to Claude Code
claude mcp add polyglot-devenv nu mcp/dist/index.js stdio

# Test the integration
claude mcp test polyglot-devenv
```

## Future Enhancements

### Planned Features
- **Enhanced Security Scanning**: Integration with vulnerability databases
- **Advanced Performance Analytics**: Machine learning-based optimization
- **Custom Hook Development**: User-defined hook creation
- **Multi-Repository Support**: Cross-repository polyglot development
- **Cloud Integration**: Remote workspace management
- **AI-Powered Debugging**: Intelligent error analysis and resolution

### Extensibility
- **Plugin Architecture**: Support for custom tool plugins
- **Custom Environments**: Support for additional programming languages
- **Third-party Integrations**: GitHub, GitLab, and other platform integrations
- **Custom Validators**: User-defined validation rules and checks