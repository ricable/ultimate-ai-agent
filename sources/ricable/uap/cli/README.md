# UAP CLI

The UAP CLI is a command-line tool for managing and deploying the Unified Agentic Platform.

## Installation

```bash
pip install uap-cli
```

Or install with the SDK:

```bash
pip install uap-sdk[cli]
```

## Quick Start

### Initialize a new project

```bash
uap project create my-agent-project
cd my-agent-project
```

### Start UAP services

```bash
uap deploy start
```

### Chat with an agent

```bash
uap agent chat default --message "Hello!"
```

### Check system status

```bash
uap monitor status
```

## Commands

### Project Management

```bash
# Create new project
uap project create my-project --template advanced

# Initialize existing project  
uap project init --template basic

# Add components
uap project add agent my-agent
uap project add plugin my-plugin
```

### Authentication

```bash
# Login to UAP
uap auth login

# Check authentication status
uap auth status

# Logout
uap auth logout
```

### Agent Management

```bash
# List available agents
uap agent list

# Create new agent
uap agent create my-agent --framework simple

# Chat with agent
uap agent chat my-agent --message "Hello"

# Get agent status
uap agent status my-agent

# Test agent with multiple messages
uap agent test my-agent --count 5
```

### Deployment

```bash
# Start local deployment
uap deploy start

# Start only backend
uap deploy start --backend-only

# Start only frontend
uap deploy start --frontend-only

# Deploy to cloud
uap deploy cloud --provider aws

# Check deployment status
uap deploy status

# Stop deployment
uap deploy stop
```

### Configuration

```bash
# Show current configuration
uap config show

# Set configuration value
uap config set backend_url http://localhost:8080

# Get configuration value
uap config get backend_url

# Create default config file
uap config create --file my-config.json --format json

# Profile management
uap config profile list
uap config profile create production
uap config profile switch production
uap config profile delete staging
```

### Plugin Management

```bash
# List available plugins
uap plugin list

# Install plugin
uap plugin install /path/to/plugin.py --enable

# Enable plugin
uap plugin enable my-plugin

# Disable plugin  
uap plugin disable my-plugin

# Get plugin info
uap plugin info my-plugin
```

### Monitoring

```bash
# System status
uap monitor status

# Health check
uap monitor health

# View metrics
uap monitor metrics --format json

# View logs
uap monitor logs --lines 100 --follow
```

## Configuration

### Global Options

All commands support these global options:

```bash
uap --config /path/to/config.json <command>
uap --profile production <command>
uap --verbose <command>
uap --quiet <command>
uap --format json <command>
```

### Output Formats

The CLI supports multiple output formats:

- `table` (default) - Human-readable table format
- `json` - JSON output for programmatic use
- `yaml` - YAML output

```bash
uap agent list --format json
uap monitor status --format yaml
```

### Configuration Files

The CLI looks for configuration files in this order:

1. File specified with `--config`
2. `./uap.json` (current directory)
3. `~/.uap/config.json` (user config)
4. Environment variables

Example configuration file:

```json
{
  "backend_url": "http://localhost:8000",
  "websocket_url": "ws://localhost:8000",
  "http_timeout": 30,
  "log_level": "INFO"
}
```

### Environment Variables

Configure UAP CLI using environment variables:

```bash
export UAP_BACKEND_URL=http://localhost:8000
export UAP_WEBSOCKET_URL=ws://localhost:8000
export UAP_LOG_LEVEL=DEBUG
```

## Project Templates

### Basic Template

Simple project structure with one agent:

```bash
uap project create my-app --template basic
```

Creates:
- `main.py` - Main application
- `config/uap.json` - Configuration
- `requirements.txt` - Dependencies

### Advanced Template

Multi-agent project with custom frameworks:

```bash
uap project create my-app --template advanced
```

Creates:
- `agents/` - Custom agent implementations
- `plugins/` - Custom plugins
- `config/` - Environment-specific configs
- `tests/` - Test files

### Custom Agent Template

Template for building custom agent frameworks:

```bash
uap project create my-app --template custom-agent
```

Creates:
- Custom agent framework
- Example implementation
- Testing utilities

## Examples

### Deploy and test an agent

```bash
# Start services
uap deploy start

# Wait for services to start
sleep 5

# Test the system
uap monitor health

# Chat with different frameworks
uap agent chat copilot-agent --message "Help me write Python code"
uap agent chat agno-agent --message "Analyze this document"
uap agent chat mastra-agent --message "Create a workflow"

# Stop services
uap deploy stop
```

### Create and deploy a custom agent

```bash
# Create project
uap project create weather-agent --template custom-agent

# Navigate to project
cd weather-agent

# Start local development
uap deploy start

# Test the agent
uap agent chat weather-agent --message "What's the weather like?"
```

### Manage configuration profiles

```bash
# Create development profile
uap config profile create development
uap config set backend_url http://localhost:8000
uap config set log_level DEBUG

# Create production profile  
uap config profile create production
uap config set backend_url https://api.example.com
uap config set log_level INFO

# Switch between profiles
uap config profile switch development
uap deploy start

uap config profile switch production
uap deploy cloud --provider aws
```

## Troubleshooting

### Common Issues

**"Backend connection failed"**
```bash
# Check if backend is running
uap deploy status

# Check configuration
uap config show

# Verify connectivity
curl http://localhost:8000/health
```

**"Authentication required"**
```bash
# Login first
uap auth login

# Check authentication status
uap auth status
```

**"Agent not found"**
```bash
# List available agents
uap agent list

# Check system status
uap monitor status
```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
uap --verbose agent chat my-agent --message "test"
```

### Getting Help

```bash
# General help
uap --help

# Command-specific help
uap agent --help
uap deploy start --help
```

## Development

### Building from Source

```bash
git clone https://github.com/uap/uap-cli
cd uap-cli
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/
```

### Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## License

UAP CLI is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Support

- [GitHub Issues](https://github.com/uap/uap-cli/issues)
- [Documentation](https://docs.uap.ai/cli)
- [Discussions](https://github.com/uap/discussions)