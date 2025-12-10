# UAP SDK

The UAP SDK is a Python library for building powerful AI agents and integrations with the Unified Agentic Platform.

## Quick Start

### Installation

```bash
pip install uap-sdk
```

### Basic Usage

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
    agent = (CustomAgentBuilder("my-agent")
             .with_simple_framework()
             .with_config(config)
             .build())
    
    # Start and interact with the agent
    await agent.start()
    response = await agent.process_message("Hello, world!")
    print(f"Agent: {response['content']}")
    await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Connecting to UAP Backend

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
    await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Features

- **Agent Development**: Build custom AI agents with different frameworks
- **Backend Integration**: Connect to UAP backend services
- **Plugin System**: Extend functionality with plugins
- **Middleware Support**: Process requests and responses
- **WebSocket Communication**: Real-time bidirectional communication
- **Configuration Management**: Flexible configuration with profiles
- **Type Safety**: Full type hints and validation

## Core Components

### UAPAgent
Main agent class for building custom agents with different frameworks.

### UAPClient
Client for connecting to UAP backend services via HTTP and WebSocket.

### AgentFramework
Abstract base class for creating custom agent frameworks.

### Configuration
Configuration management with file, environment, and runtime support.

### Plugin System
Extensible plugin architecture for adding custom functionality.

## Examples

### Custom Agent Framework

```python
from uap_sdk import AgentFramework
from typing import Dict, Any

class MyCustomAgent(AgentFramework):
    def __init__(self, config=None):
        super().__init__("my-custom", config)
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Your custom processing logic
        return {
            "content": f"Custom response to: {message}",
            "metadata": {"framework": "my-custom"}
        }
    
    def get_status(self) -> Dict[str, Any]:
        return {"status": "active", "framework": self.framework_name}
    
    async def initialize(self) -> None:
        self.is_initialized = True
        self.status = "active"
```

### Agent Plugin

```python
from uap_sdk.plugin import AgentPlugin
from typing import Dict, Any, Optional

class MyPlugin(AgentPlugin):
    PLUGIN_NAME = "my-plugin"
    PLUGIN_VERSION = "1.0.0"
    
    def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        return "my-keyword" in message.lower()
    
    async def process_message(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.should_handle_message(message, context):
            return {"content": "Handled by my plugin!"}
        return None
```

### Middleware

```python
async def logging_middleware(message: str, context: Dict[str, Any]):
    print(f"Processing: {message}")
    context['logged'] = True
    return message, context

# Add to agent
agent = (CustomAgentBuilder("my-agent")
         .with_simple_framework()
         .add_middleware(logging_middleware)
         .build())
```

## Installation Options

### Basic Installation
```bash
pip install uap-sdk
```

### With CLI Tools
```bash
pip install uap-sdk[cli]
```

### Full Installation (with all extras)
```bash
pip install uap-sdk[full]
```

### Development Installation
```bash
pip install uap-sdk[dev]
```

## Requirements

- Python 3.11+
- httpx>=0.24.0
- websockets>=11.0.0
- pyyaml>=6.0.0
- aiofiles>=23.0.0
- pydantic>=2.0.0

## Documentation

- [Developer Guide](../docs/developer/README.md)
- [API Reference](../docs/developer/api/README.md)
- [Tutorials](../docs/developer/tutorials/README.md)
- [Examples](../examples/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## License

UAP SDK is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Support

- [GitHub Issues](https://github.com/uap/uap-sdk/issues)
- [Documentation](https://docs.uap.ai/sdk)
- [Discussions](https://github.com/uap/discussions)