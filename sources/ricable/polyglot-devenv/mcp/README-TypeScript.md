# Polyglot Development Environment MCP Server (TypeScript)

A comprehensive Model Context Protocol (MCP) server for polyglot development environments supporting Python, TypeScript, Rust, Go, and Nushell with DevBox and DevPod integration.

## 🚀 Features

### 🌍 Multi-Language Support
- **Python**: Python 3.12, uv package manager, ruff, mypy, pytest
- **TypeScript**: Node.js 20, TypeScript, ESLint, Prettier, Jest
- **Rust**: Rust compiler, Cargo, Clippy, rustfmt
- **Go**: Go compiler, golangci-lint, goimports
- **Nushell**: Nushell shell with automation scripts

### 🛠️ DevBox Integration
- Environment shell access
- Script execution with progress tracking
- Package management
- Status monitoring and validation

### 🐳 DevPod Workspace Management
- Multi-workspace provisioning (1-10 per environment)
- VS Code integration with language extensions
- Container lifecycle management
- Workspace status monitoring

### 🔍 Quality Assurance
- Cross-environment validation
- Parallel linting and testing
- Performance analytics
- Security scanning

### 🪝 Hook Integration
- Claude Code hooks monitoring
- Real-time automation tracking
- Hook execution history

### 📝 Context Engineering
- PRP (Project Request Proposal) generation
- PRP execution with validation
- Environment-specific templates

## 📦 Installation

### Prerequisites

```bash
# Required tools
npm install -g @modelcontextprotocol/inspector
brew install devbox  # or your package manager
brew install devpod  # for containerized development

# Optional but recommended
brew install git-secrets  # for enhanced security scanning
```

### Build from Source

```bash
# Navigate to the MCP server directory
cd mcp

# Install dependencies
npm install

# Build the server
npm run build

# Test the server (optional)
npm run start
```

## 🔧 Claude Code Integration

### Method 1: Manual Installation

```bash
# Add the MCP server to Claude Code
claude mcp add polyglot-devenv node /path/to/mcp/dist/index.js stdio

# Verify the installation
claude mcp list

# Test the integration
claude mcp test polyglot-devenv
```

### Method 2: Development Mode

For active development, you can run the server directly:

```bash
# Start the server in development mode
cd mcp
npm run dev

# In another terminal, connect Claude Code
claude mcp add polyglot-devenv node ./dist/index.js stdio
```

### Configuration Verification

After installation, verify the server is working:

```bash
# Check server status
claude mcp test polyglot-devenv

# List available tools
claude mcp tools polyglot-devenv
```

## 🎯 Quick Start

### 1. Environment Detection

```typescript
// Detect all polyglot environments
environment_detect

// Expected output:
// 🔍 Polyglot Development Environments Detected
// ✅ 🐍 python-env
// ✅ 📘 typescript-env  
// ✅ 🦀 rust-env
// ✅ 🐹 go-env
// ✅ 🐚 nushell-env
```

### 2. DevBox Operations

```typescript
// Run tests in Python environment
devbox_run {
  "environment": "python-env",
  "script": "test"
}

// Add a package to TypeScript environment  
devbox_add_package {
  "environment": "typescript-env",
  "package": "lodash"
}
```

### 3. DevPod Workspace Management

```typescript
// Provision 3 Python development workspaces
devpod_provision {
  "environment": "python-env",
  "count": 3
}

// List all workspaces
devpod_list
```

### 4. Quality Assurance

```typescript
// Comprehensive quality check
polyglot_check {
  "include_security": true,
  "include_performance": true
}

// Cross-environment validation
polyglot_validate {
  "parallel": true
}
```

## 🛠️ Available Tools

### Environment Management
| Tool | Description | Example |
|------|-------------|---------|
| `environment_detect` | Detect all environments | `environment_detect {}` |
| `environment_info` | Get environment details | `{"environment": "python-env"}` |
| `environment_validate` | Validate environment health | `{"environment": "python-env"}` |

### DevBox Operations
| Tool | Description | Example |
|------|-------------|---------|
| `devbox_shell` | Shell access guidance | `{"environment": "python-env"}` |
| `devbox_run` | Execute scripts | `{"environment": "python-env", "script": "test"}` |
| `devbox_status` | Environment status | `{"environment": "python-env"}` |
| `devbox_add_package` | Add packages | `{"environment": "python-env", "package": "requests"}` |

### DevPod Management
| Tool | Description | Example |
|------|-------------|---------|
| `devpod_provision` | Create workspaces | `{"environment": "python-env", "count": 2}` |
| `devpod_list` | List workspaces | `devpod_list {}` |
| `devpod_status` | Workspace status | `{"workspace": "polyglot-python-devpod-123"}` |

### Quality Assurance
| Tool | Description | Example |
|------|-------------|---------|
| `polyglot_check` | Quality checks | `{"include_security": true}` |
| `polyglot_validate` | Cross-validation | `{"parallel": true}` |
| `polyglot_clean` | Environment cleanup | `{"environment": "python-env"}` |

### Performance & Security
| Tool | Description | Example |
|------|-------------|---------|
| `performance_measure` | Measure commands | `{"command": "npm test", "environment": "typescript-env"}` |
| `performance_report` | Performance reports | `{"days": 7}` |
| `security_scan` | Security scanning | `{"scan_type": "all"}` |

### Hook Management
| Tool | Description | Example |
|------|-------------|---------|
| `hook_status` | Hook configuration | `hook_status {}` |
| `hook_trigger` | Manual triggers | `{"hook_type": "PostToolUse"}` |

### Context Engineering
| Tool | Description | Example |
|------|-------------|---------|
| `prp_generate` | Generate PRPs | `{"feature_file": "features/api.md", "environment": "python-env"}` |
| `prp_execute` | Execute PRPs | `{"prp_file": "PRPs/api-python.md", "validate": true}` |

## 🔧 Configuration

### Environment Structure

The server expects your polyglot project to have this structure:

```
polyglot-project/
├── python-env/           # Python development environment
│   ├── devbox.json      # DevBox configuration
│   ├── pyproject.toml   # Python project config
│   └── src/             # Source code
├── typescript-env/       # TypeScript development environment
│   ├── devbox.json      # DevBox configuration
│   ├── package.json     # Node.js project config
│   └── src/             # Source code
├── rust-env/            # Rust development environment
│   ├── devbox.json      # DevBox configuration
│   ├── Cargo.toml       # Rust project config
│   └── src/             # Source code
├── go-env/              # Go development environment
│   ├── devbox.json      # DevBox configuration
│   ├── go.mod           # Go module config
│   └── cmd/             # Go applications
├── nushell-env/         # Nushell scripting environment
│   ├── devbox.json      # DevBox configuration
│   ├── scripts/         # Nushell scripts
│   └── config/          # Nushell configuration
└── .claude/             # Claude Code configuration
    ├── settings.json    # Hook configuration
    └── commands/        # Custom commands
```

### DevBox Configuration

Each environment should have a `devbox.json` with appropriate packages and scripts:

#### Python Environment (`python-env/devbox.json`)
```json
{
  "packages": ["python@3.12", "uv", "ruff", "mypy"],
  "shell": {
    "scripts": {
      "install": "uv sync --dev",
      "test": "uv run pytest --cov=src",
      "lint": "uv run ruff check . --fix",
      "format": "uv run ruff format .",
      "type-check": "uv run mypy ."
    }
  }
}
```

#### TypeScript Environment (`typescript-env/devbox.json`)
```json
{
  "packages": ["nodejs@20", "typescript@latest"],
  "shell": {
    "scripts": {
      "install": "npm install",
      "test": "npm run test",
      "lint": "npm run lint",
      "format": "npm run format",
      "build": "npm run build"
    }
  }
}
```

### Claude Code Hooks Integration

The server integrates with Claude Code hooks for real-time automation:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|MultiEdit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "Auto-formatting based on file type"
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command", 
            "command": "Pre-commit validation"
          }
        ]
      }
    ]
  }
}
```

## 🔍 Troubleshooting

### Common Issues

#### Server Won't Start
```bash
# Check Node.js version (requires 20+)
node --version

# Rebuild the server
cd mcp
npm run clean
npm install
npm run build
```

#### Environment Not Detected
```bash
# Verify devbox.json exists
ls -la */devbox.json

# Check workspace structure
environment_detect {}
```

#### DevPod Issues
```bash
# Verify DevPod installation
devpod version

# Check Docker is running
docker ps

# Test workspace creation
devpod_provision {"environment": "python-env", "count": 1}
```

#### Permission Errors
```bash
# Make server executable
chmod +x dist/index.js

# Check file permissions
ls -la dist/
```

### Debug Mode

Enable debug logging:

```bash
# Set environment variable
export DEBUG=mcp:*

# Run server with debug output
node dist/index.js
```

### Logs and Diagnostics

```bash
# Check Claude Code logs
tail -f ~/.claude/logs/mcp.log

# Test specific tools
claude mcp test polyglot-devenv --tool environment_detect

# Validate configuration
environment_validate {}
```

## 🔄 Development

### Contributing

1. **Fork and Clone**
   ```bash
   git clone <your-fork>
   cd mcp
   ```

2. **Install Dependencies**
   ```bash
   npm install
   ```

3. **Development Workflow**
   ```bash
   # Start development mode
   npm run watch
   
   # In another terminal, test changes
   npm run start
   ```

4. **Testing**
   ```bash
   # Lint code
   npm run lint
   
   # Type check
   npm run type-check
   
   # Format code
   npm run format
   ```

### Adding New Tools

1. **Define Schema** in `polyglot-server.ts`:
   ```typescript
   const NewToolSchema = z.object({
     parameter: z.string().describe("Description"),
   });
   ```

2. **Add to Tool Registry**:
   ```typescript
   {
     name: ToolName.NEW_TOOL,
     description: "Tool description",
     inputSchema: zodToJsonSchema(NewToolSchema) as ToolInput,
   }
   ```

3. **Implement Handler**:
   ```typescript
   async function handleNewTool(args: z.infer<typeof NewToolSchema>) {
     // Implementation
     return {
       content: [{ type: "text", text: "Result" }],
     };
   }
   ```

### Architecture

```
polyglot-server.ts     # Main server implementation
├── Tool Schemas      # Zod schemas for input validation  
├── Tool Registry     # Available tools definition
├── Tool Handlers     # Implementation functions
└── Helper Functions  # Utilities and icons

polyglot-utils.ts      # Utility functions
├── Command Execution # Safe command execution
├── Environment Detection # Auto-detect environments
├── DevBox Integration # DevBox command wrappers
├── DevPod Management # DevPod workspace operations
└── Validation Helpers # Environment validation

polyglot-types.ts      # TypeScript definitions
├── Environment Types # Environment information
├── Command Results   # Command execution results
├── Validation Types  # Validation result structures
└── Security Types    # Security scan results
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/polyglot-devenv/typescript-env/issues)
- **Documentation**: See `polyglot-instructions.md` for detailed tool documentation
- **Claude Code Integration**: [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)

## 🎉 Acknowledgments

Built with:
- [Model Context Protocol SDK](https://github.com/modelcontextprotocol/sdk)
- [DevBox](https://www.jetify.com/devbox) for environment management
- [DevPod](https://devpod.sh/) for containerized development
- [Claude Code](https://claude.ai/code) for AI-assisted development

---

*Polyglot Development Environment MCP Server - Bringing AI-powered development tools to your multi-language projects* 🚀