# UAP Developer Documentation

Welcome to the UAP (Unified Agentic Platform) Developer Documentation. This guide will help you build custom agents, integrations, and extensions using the UAP SDK.

## Table of Contents

- [Getting Started](#getting-started)
- [Quick Start Guide](#quick-start-guide)
- [SDK Reference](#sdk-reference)
- [CLI Tools](#cli-tools)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Python 3.11 or higher
- UAP Backend running (see deployment guide)
- Basic understanding of async/await in Python

### Installation

#### Install UAP SDK

```bash
pip install uap-sdk
```

#### Or install from source

```bash
git clone <uap-repository>
cd uap/sdk
pip install -e .
```

#### Install UAP CLI

```bash
pip install uap-cli
```

Or install both:

```bash
pip install uap-sdk[cli]
```

## Quick Start Guide

### 1. Create Your First Agent

```python
import asyncio
from uap_sdk import UAPAgent, CustomAgentBuilder, Configuration

async def main():
    # Create configuration
    config = Configuration({
        "backend_url": "http://localhost:8000",
        "websocket_url": "ws://localhost:8000"
    })
    
    # Create a simple agent
    agent = (CustomAgentBuilder("my-first-agent")
             .with_simple_framework()
             .with_config(config)
             .build())
    
    # Start the agent
    await agent.start()
    
    # Interact with the agent
    response = await agent.process_message("Hello, world!")
    print(f"Agent: {response['content']}")
    
    # Stop the agent
    await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Connect to UAP Backend

```python
import asyncio
from uap_sdk import UAPClient, Configuration

async def main():
    # Create client
    config = Configuration()
    client = UAPClient(config)
    
    # Authenticate
    await client.login("username", "password")
    
    # Chat with backend agents
    response = await client.chat(
        agent_id="default",
        message="What can you do?",
        framework="auto"
    )
    
    print(f"Response: {response['content']}")
    
    # Cleanup
    await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Use CLI Tools

```bash
# Initialize a new project
uap project create my-agent-project

# Start UAP services
uap deploy start

# Chat with an agent
uap agent chat default --message "Hello!"

# Check system status
uap monitor status
```

## SDK Reference

### Core Components

#### UAPAgent
The main agent class for building custom agents.

```python
from uap_sdk import UAPAgent, AgentFramework

class MyAgent(UAPAgent):
    # Your custom agent implementation
    pass
```

#### AgentFramework
Abstract base class for creating custom agent frameworks.

```python
from uap_sdk import AgentFramework
from typing import Dict, Any

class MyFramework(AgentFramework):
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Your message processing logic
        return {"content": "Response"}
    
    def get_status(self) -> Dict[str, Any]:
        # Return framework status
        return {"status": "active"}
    
    async def initialize(self) -> None:
        # Initialize framework resources
        pass
```

#### UAPClient
Client for connecting to UAP backend services.

```python
from uap_sdk import UAPClient

client = UAPClient(config)
await client.login("user", "pass")
response = await client.chat("agent-id", "message")
```

#### Configuration
Configuration management for SDK components.

```python
from uap_sdk import Configuration

# From file
config = Configuration(config_file="config.json")

# From dictionary
config = Configuration({
    "backend_url": "http://localhost:8000",
    "websocket_url": "ws://localhost:8000"
})

# From environment variables
config = Configuration.from_env()
```

### Plugin System

#### Creating Plugins

```python
from uap_sdk.plugin import AgentPlugin

class MyPlugin(AgentPlugin):
    PLUGIN_NAME = "my-plugin"
    PLUGIN_VERSION = "1.0.0"
    
    async def process_message(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "my-keyword" in message.lower():
            return {"content": "Handled by my plugin!"}
        return None
    
    def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        return "my-keyword" in message.lower()
```

#### Using Plugins

```python
from uap_sdk import PluginManager

plugin_manager = PluginManager(config)
await plugin_manager.discover_plugins()
await plugin_manager.enable_plugin("my-plugin")
```

## CLI Tools

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

# Test agent
uap agent test my-agent --count 5
```

### Deployment

```bash
# Start local deployment
uap deploy start

# Start only backend
uap deploy start --backend-only

# Deploy to cloud
uap deploy cloud --provider aws

# Check deployment status
uap deploy status
```

### Configuration

```bash
# Show current configuration
uap config show

# Set configuration value
uap config set backend_url http://localhost:8080

# Create default config file
uap config create --file my-config.json

# Manage profiles
uap config profile create production
uap config profile switch production
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
uap monitor logs --lines 100
```

## Examples

### Agent Examples

- [Simple Agent](../examples/agents/simple_agent_example.py) - Basic agent usage
- [Custom Agent](../examples/agents/custom_agent_example.py) - Custom framework development
- [Middleware](../examples/agents/middleware_example.py) - Request/response middleware

### Plugin Examples

- [Agent Plugin](../examples/plugins/example_agent_plugin.py) - Message processing plugins
- [Tool Plugin](../examples/plugins/example_tool_plugin.py) - Function/tool plugins

### Workflow Examples

- [Basic Workflow](../examples/workflows/basic_workflow.py) - Multi-agent workflows
- [Client Integration](../examples/workflows/client_integration_example.py) - Backend integration

## Best Practices

### Agent Development

1. **Use Async/Await**: All agent operations are asynchronous
2. **Handle Errors Gracefully**: Wrap operations in try/catch blocks
3. **Validate Input**: Always validate user input before processing
4. **Keep State Minimal**: Agents should be stateless when possible
5. **Use Metadata**: Include relevant metadata in responses

```python
# Good
async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Validate input
        if not message or len(message) > 1000:
            return {"content": "Invalid message", "metadata": {"error": True}}
        
        # Process message
        result = await self.some_processing(message)
        
        return {
            "content": result,
            "metadata": {
                "processing_time": time.time(),
                "message_length": len(message)
            }
        }
    except Exception as e:
        return {
            "content": f"Error: {str(e)}",
            "metadata": {"error": True, "error_type": type(e).__name__}
        }
```

### Configuration Management

1. **Use Environment Variables**: For sensitive configuration
2. **Validate Configuration**: Always validate configuration on startup
3. **Use Profiles**: For different environments (dev, staging, prod)

```python
# Good
config = Configuration()
config.validate()  # Throws error if invalid

# Use environment variables for secrets
config.set("api_key", os.getenv("API_KEY"))
```

### Plugin Development

1. **Single Responsibility**: Each plugin should have one clear purpose
2. **Fail Fast**: Return None quickly if plugin can't handle message
3. **Clean Up Resources**: Always implement cleanup method

```python
class MyPlugin(AgentPlugin):
    def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        # Fast check
        return "my-keyword" in message.lower()
    
    async def process_message(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.should_handle_message(message, context):
            return None  # Don't handle
        
        # Process message
        return {"content": "Handled!"}
    
    async def cleanup(self) -> None:
        # Clean up resources
        await self.close_connections()
```

### Error Handling

1. **Use Specific Exceptions**: Use UAP-specific exceptions when possible
2. **Log Errors**: Always log errors for debugging
3. **Provide User-Friendly Messages**: Don't expose internal errors to users

```python
from uap_sdk.exceptions import UAPException, UAPConnectionError

try:
    response = await client.chat("agent", "message")
except UAPConnectionError:
    # Handle connection issues
    logger.error("Backend connection failed")
    return {"error": "Service temporarily unavailable"}
except UAPException as e:
    # Handle UAP-specific errors
    logger.error(f"UAP error: {e.message}")
    return {"error": "Something went wrong"}
```

## Troubleshooting

### Common Issues

#### "Backend connection failed"
- Ensure UAP backend is running: `uap deploy status`
- Check configuration: `uap config show`
- Verify URL is correct: `http://localhost:8000`

#### "Authentication required"
- Login first: `uap auth login`
- Check token: `uap auth status`
- Verify credentials are correct

#### "Agent not found"
- List available agents: `uap agent list`
- Check agent is running: `uap monitor status`
- Verify agent ID is correct

#### "Plugin not loading"
- Check plugin directory: `~/.uap/plugins/`
- Verify plugin syntax: `python -m py_compile plugin.py`
- Check plugin dependencies

### Debug Mode

Enable verbose logging:

```bash
uap --verbose agent chat my-agent --message "test"
```

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from uap_sdk import UAPClient
# Client will now log debug information
```

### Getting Help

1. Check the [API Reference](api/)
2. Browse [Tutorials](tutorials/)
3. Review [Examples](../../examples/)
4. Check [GitHub Issues](https://github.com/uap/issues)

## Next Steps

- [API Reference](api/README.md) - Detailed API documentation
- [Tutorials](tutorials/README.md) - Step-by-step guides
- [Advanced Topics](guides/README.md) - In-depth guides

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

UAP SDK is licensed under the MIT License. See [LICENSE](../../LICENSE) for details.