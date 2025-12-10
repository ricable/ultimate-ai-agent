# UAP SDK Examples

This directory contains comprehensive examples demonstrating various aspects of the UAP SDK.

## Directory Structure

```
examples/
├── agents/                 # Agent development examples
├── plugins/               # Plugin development examples
├── workflows/             # Workflow and integration examples
└── README.md             # This file
```

## Agent Examples

### [simple_agent_example.py](agents/simple_agent_example.py)
Demonstrates basic agent creation and interaction using the built-in SimpleAgent framework.

**What you'll learn:**
- Creating agents with CustomAgentBuilder
- Basic agent lifecycle (start, process, stop)
- Simple agent response patterns

**Run:**
```bash
python examples/agents/simple_agent_example.py
```

### [custom_agent_example.py](agents/custom_agent_example.py)
Shows how to create custom agent frameworks with specialized capabilities.

**Features:**
- WeatherAgent with city-specific weather data
- MathAgent with calculation capabilities
- Custom response logic and metadata

**Run:**
```bash
python examples/agents/custom_agent_example.py
```

### [middleware_example.py](agents/middleware_example.py)
Demonstrates sophisticated middleware for request/response processing.

**Middleware types:**
- Logging middleware
- Profanity filtering
- Language translation
- Sentiment analysis
- Response enhancement

**Run:**
```bash
python examples/agents/middleware_example.py
```

## Plugin Examples

### [example_agent_plugin.py](plugins/example_agent_plugin.py)
Complete examples of agent plugins that handle specific message types.

**Plugins included:**
- CalculatorPlugin - Mathematical operations
- TimePlugin - Date and time information
- QuotePlugin - Inspirational quotes

**Run:**
```bash
python examples/plugins/example_agent_plugin.py
```

### [example_tool_plugin.py](plugins/example_tool_plugin.py)
Advanced tool plugins that provide functions for agents to use.

**Tool types:**
- FileSystemToolPlugin - File operations
- HTTPToolPlugin - Web requests
- UtilityToolPlugin - Encoding, hashing, utilities

**Run:**
```bash
python examples/plugins/example_tool_plugin.py
```

## Workflow Examples

### [basic_workflow.py](workflows/basic_workflow.py)
Multi-agent workflow orchestration with conditional logic and parallel processing.

**Features:**
- Customer support workflow
- Data processing pipeline
- Parallel agent execution
- Conditional routing

**Run:**
```bash
python examples/workflows/basic_workflow.py
```

### [client_integration_example.py](workflows/client_integration_example.py)
Complete integration with UAP backend services.

**Demonstrates:**
- Backend authentication
- Framework-specific communication
- WebSocket real-time communication
- Document processing
- Error handling patterns

**Run (requires backend):**
```bash
# Start UAP backend first
uap deploy start

# Then run the example
python examples/workflows/client_integration_example.py
```

## Running Examples

### Prerequisites

1. Install UAP SDK:
```bash
pip install uap-sdk
```

2. For backend integration examples, start UAP:
```bash
uap deploy start
```

### Individual Examples

Run any example directly:
```bash
python examples/agents/simple_agent_example.py
python examples/plugins/example_agent_plugin.py
python examples/workflows/basic_workflow.py
```

### All Examples

Run all examples with a script:
```bash
#!/bin/bash
echo "Running UAP SDK Examples..."

echo "1. Simple Agent Example"
python examples/agents/simple_agent_example.py

echo "2. Custom Agent Example"  
python examples/agents/custom_agent_example.py

echo "3. Middleware Example"
python examples/agents/middleware_example.py

echo "4. Agent Plugin Example"
python examples/plugins/example_agent_plugin.py

echo "5. Tool Plugin Example"
python examples/plugins/example_tool_plugin.py

echo "6. Basic Workflow Example"
python examples/workflows/basic_workflow.py

echo "All examples completed!"
```

## Example Use Cases

### Building a Customer Service Bot

1. Start with [simple_agent_example.py](agents/simple_agent_example.py)
2. Add custom logic from [custom_agent_example.py](agents/custom_agent_example.py)
3. Implement sentiment analysis from [middleware_example.py](agents/middleware_example.py)
4. Create workflow from [basic_workflow.py](workflows/basic_workflow.py)

### Creating a Document Analysis System

1. Use Agno framework from [client_integration_example.py](workflows/client_integration_example.py)
2. Add file operations from [example_tool_plugin.py](plugins/example_tool_plugin.py)
3. Create custom analysis from [custom_agent_example.py](agents/custom_agent_example.py)

### Building a Multi-Agent Workflow

1. Study workflow orchestration in [basic_workflow.py](workflows/basic_workflow.py)
2. Implement custom frameworks from [custom_agent_example.py](agents/custom_agent_example.py)
3. Add specialized plugins from [example_agent_plugin.py](plugins/example_agent_plugin.py)

## Common Patterns

### Agent Creation Pattern

```python
from uap_sdk import CustomAgentBuilder, Configuration

# Standard pattern for creating agents
agent = (CustomAgentBuilder("my-agent")
         .with_simple_framework()  # or custom framework
         .with_config(config)
         .add_middleware(middleware_func)
         .with_metadata("version", "1.0")
         .build())

await agent.start()
# Use agent...
await agent.stop()
```

### Plugin Registration Pattern

```python
from uap_sdk import PluginManager

# Standard pattern for plugins
plugin_manager = PluginManager(config)
await plugin_manager.discover_plugins()
await plugin_manager.enable_plugin("my-plugin")

# Use plugins with agents
plugins = plugin_manager.get_enabled_plugins("agent")
```

### Client Integration Pattern

```python
from uap_sdk import UAPClient, Configuration

# Standard pattern for backend integration
config = Configuration()
client = UAPClient(config)

await client.login("username", "password")
response = await client.chat("agent-id", "message")
await client.cleanup()
```

### Error Handling Pattern

```python
from uap_sdk.exceptions import UAPException, UAPConnectionError, UAPAuthError

try:
    response = await client.chat("agent", "message")
except UAPAuthError:
    # Handle authentication issues
    await client.login("user", "pass")
except UAPConnectionError:
    # Handle connection issues
    print("Backend unavailable")
except UAPException as e:
    # Handle general UAP errors
    print(f"UAP error: {e.message}")
```

## Next Steps

After exploring the examples:

1. **Read the [Developer Guide](../docs/developer/README.md)** for comprehensive documentation
2. **Follow the [Tutorials](../docs/developer/tutorials/README.md)** for step-by-step learning
3. **Check the [API Reference](../docs/developer/api/README.md)** for detailed API documentation
4. **Review [Advanced Guides](../docs/developer/guides/README.md)** for production deployment

## Contributing Examples

We welcome new examples! To contribute:

1. Follow the existing example structure
2. Include comprehensive comments
3. Add error handling
4. Test thoroughly
5. Update this README

See our [Contributing Guide](../CONTRIBUTING.md) for details.

## Support

If you have questions about the examples:

1. Check the [Documentation](../docs/developer/README.md)
2. Browse [GitHub Discussions](https://github.com/uap/discussions)
3. Report issues in [GitHub Issues](https://github.com/uap/issues)