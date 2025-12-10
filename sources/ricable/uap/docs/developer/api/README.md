# UAP SDK API Reference

This document provides detailed API reference for all UAP SDK components.

## Core Classes

### UAPAgent

The main agent class for building custom agents.

#### Constructor

```python
UAPAgent(agent_id: str, framework: AgentFramework, config: Configuration = None, client: UAPClient = None)
```

**Parameters:**
- `agent_id`: Unique identifier for the agent
- `framework`: Agent framework implementation
- `config`: Configuration object (optional)
- `client`: UAP client for backend communication (optional)

#### Methods

##### `async start() -> None`

Start the agent and initialize all components.

**Raises:**
- `UAPException`: If agent startup fails

**Example:**
```python
agent = UAPAgent("my-agent", framework, config)
await agent.start()
```

##### `async stop() -> None`

Stop the agent and clean up resources.

**Example:**
```python
await agent.stop()
```

##### `async process_message(message: str, context: Dict = None) -> Dict[str, Any]`

Process a message through the agent's framework.

**Parameters:**
- `message`: Input message to process
- `context`: Optional context dictionary

**Returns:**
- Dictionary with `content` and `metadata` keys

**Raises:**
- `UAPException`: If agent is not running

**Example:**
```python
response = await agent.process_message("Hello!")
print(response["content"])
```

##### `get_status() -> Dict[str, Any]`

Get the current status of the agent.

**Returns:**
- Dictionary containing agent status information

**Example:**
```python
status = agent.get_status()
print(f"Agent running: {status['is_running']}")
```

##### `add_middleware(middleware: Callable) -> None`

Add middleware to process messages before framework processing.

**Parameters:**
- `middleware`: Async function that processes messages

**Example:**
```python
async def logging_middleware(message, context):
    print(f"Processing: {message}")
    return message, context

agent.add_middleware(logging_middleware)
```

##### `set_metadata(key: str, value: Any) -> None`

Set agent metadata.

**Parameters:**
- `key`: Metadata key
- `value`: Metadata value

##### `get_metadata(key: str, default: Any = None) -> Any`

Get agent metadata.

**Parameters:**
- `key`: Metadata key
- `default`: Default value if key not found

**Returns:**
- Metadata value or default

---

### AgentFramework

Abstract base class for creating custom agent frameworks.

#### Constructor

```python
AgentFramework(framework_name: str, config: Configuration = None)
```

**Parameters:**
- `framework_name`: Name of the framework
- `config`: Configuration object (optional)

#### Abstract Methods

##### `async process_message(message: str, context: Dict[str, Any]) -> Dict[str, Any]`

Process a user message and return a response.

**Parameters:**
- `message`: The user's input message
- `context`: Additional context for processing

**Returns:**
- Dictionary containing 'content' and 'metadata' keys

**Must be implemented by subclasses.**

##### `get_status() -> Dict[str, Any]`

Get the current status of the framework.

**Returns:**
- Dictionary containing status information

**Must be implemented by subclasses.**

##### `async initialize() -> None`

Initialize the framework resources.

**Must be implemented by subclasses.**

#### Optional Methods

##### `async cleanup() -> None`

Clean up framework resources. Override if needed.

##### `get_capabilities() -> List[str]`

Get the capabilities of this framework. Override if needed.

**Returns:**
- List of capability strings

---

### UAPClient

Client for connecting to UAP backend services.

#### Constructor

```python
UAPClient(config: Configuration = None)
```

**Parameters:**
- `config`: Configuration object (optional)

#### Methods

##### `async login(username: str, password: str) -> Dict[str, Any]`

Login to UAP backend.

**Parameters:**
- `username`: Username
- `password`: Password

**Returns:**
- Dictionary with authentication information

**Raises:**
- `UAPAuthError`: If authentication fails
- `UAPConnectionError`: If connection fails

**Example:**
```python
result = await client.login("admin", "password")
print(f"Logged in: {result['message']}")
```

##### `async chat(agent_id: str, message: str, framework: str = "auto", context: Dict = None, use_websocket: bool = False) -> Dict[str, Any]`

Chat with an agent using HTTP or WebSocket.

**Parameters:**
- `agent_id`: Target agent identifier
- `message`: Message to send
- `framework`: Framework routing ("auto", "copilot", "agno", "mastra")
- `context`: Optional context dictionary
- `use_websocket`: Use WebSocket instead of HTTP

**Returns:**
- Dictionary with agent response

**Raises:**
- `UAPException`: If chat fails
- `UAPAuthError`: If authentication required

**Example:**
```python
response = await client.chat("my-agent", "Hello!", framework="auto")
print(response["content"])
```

##### `async get_status() -> Dict[str, Any]`

Get system status.

**Returns:**
- Dictionary with system status information

**Example:**
```python
status = await client.get_status()
print(f"System status: {status['status']}")
```

##### `async upload_document(file_path: str, process_immediately: bool = True) -> Dict[str, Any]`

Upload a document for processing.

**Parameters:**
- `file_path`: Path to the file to upload
- `process_immediately`: Process the document immediately

**Returns:**
- Dictionary with upload result

**Raises:**
- `UAPException`: If upload fails

**Example:**
```python
result = await client.upload_document("document.pdf")
print(f"Document uploaded: {result['document_id']}")
```

##### `async connect_websocket(agent_id: str) -> None`

Connect to WebSocket for real-time communication.

**Parameters:**
- `agent_id`: Target agent identifier

**Raises:**
- `UAPConnectionError`: If connection fails

**Example:**
```python
await client.connect_websocket("my-agent")
```

##### `async cleanup() -> None`

Clean up all connections.

**Example:**
```python
await client.cleanup()
```

---

### Configuration

Configuration management for UAP SDK.

#### Constructor

```python
Configuration(config_file: Union[str, Path] = None, config_dict: Dict[str, Any] = None)
```

**Parameters:**
- `config_file`: Path to configuration file (JSON or YAML)
- `config_dict`: Configuration dictionary

#### Methods

##### `get(key: str, default: Any = None) -> Any`

Get a configuration value.

**Parameters:**
- `key`: Configuration key
- `default`: Default value if key not found

**Returns:**
- Configuration value or default

**Example:**
```python
backend_url = config.get("backend_url", "http://localhost:8000")
```

##### `set(key: str, value: Any) -> None`

Set a configuration value.

**Parameters:**
- `key`: Configuration key
- `value`: Configuration value

**Example:**
```python
config.set("backend_url", "https://api.example.com")
```

##### `update(config_dict: Dict[str, Any]) -> None`

Update configuration with a dictionary.

**Parameters:**
- `config_dict`: Dictionary of configuration values

**Example:**
```python
config.update({
    "backend_url": "https://api.example.com",
    "timeout": 30
})
```

##### `validate() -> bool`

Validate the configuration.

**Returns:**
- True if configuration is valid

**Raises:**
- `ValueError`: If configuration is invalid

**Example:**
```python
config.validate()  # Raises ValueError if invalid
```

##### `save_to_file(config_file: Union[str, Path] = None, format: str = "json") -> None`

Save configuration to a file.

**Parameters:**
- `config_file`: Output file path (optional)
- `format`: File format ("json" or "yaml")

**Raises:**
- `ValueError`: If save fails

**Example:**
```python
config.save_to_file("my-config.json")
```

##### `to_dict() -> Dict[str, Any]`

Get configuration as a dictionary.

**Returns:**
- Configuration dictionary

**Example:**
```python
config_dict = config.to_dict()
print(json.dumps(config_dict, indent=2))
```

#### Class Methods

##### `from_env() -> Configuration`

Create configuration from environment variables only.

**Returns:**
- Configuration object

**Example:**
```python
config = Configuration.from_env()
```

##### `create_default_config_file(config_file: Union[str, Path], format: str = "json") -> None`

Create a default configuration file.

**Parameters:**
- `config_file`: Output file path
- `format`: File format ("json" or "yaml")

**Example:**
```python
Configuration.create_default_config_file("config.json")
```

---

### CustomAgentBuilder

Builder class for creating custom agents with fluent interface.

#### Constructor

```python
CustomAgentBuilder(agent_id: str)
```

**Parameters:**
- `agent_id`: Agent identifier

#### Methods

##### `with_framework(framework: AgentFramework) -> CustomAgentBuilder`

Set the agent framework.

**Parameters:**
- `framework`: Agent framework instance

**Returns:**
- Builder instance for chaining

##### `with_simple_framework() -> CustomAgentBuilder`

Use the built-in simple framework.

**Returns:**
- Builder instance for chaining

##### `with_config(config: Configuration) -> CustomAgentBuilder`

Set the agent configuration.

**Parameters:**
- `config`: Configuration object

**Returns:**
- Builder instance for chaining

##### `add_middleware(middleware: Callable) -> CustomAgentBuilder`

Add middleware to the agent.

**Parameters:**
- `middleware`: Middleware function

**Returns:**
- Builder instance for chaining

##### `on_message(event_type: str, handler: Callable) -> CustomAgentBuilder`

Add a message handler.

**Parameters:**
- `event_type`: Event type to handle
- `handler`: Handler function

**Returns:**
- Builder instance for chaining

##### `with_metadata(key: str, value: Any) -> CustomAgentBuilder`

Add metadata to the agent.

**Parameters:**
- `key`: Metadata key
- `value`: Metadata value

**Returns:**
- Builder instance for chaining

##### `build() -> UAPAgent`

Build the custom agent.

**Returns:**
- Configured UAPAgent instance

**Raises:**
- `UAPException`: If framework is not set

**Example:**
```python
agent = (CustomAgentBuilder("my-agent")
         .with_simple_framework()
         .with_config(config)
         .add_middleware(my_middleware)
         .with_metadata("version", "1.0")
         .build())
```

---

## Plugin System

### UAPPlugin

Abstract base class for UAP plugins.

#### Constructor

```python
UAPPlugin(name: str, version: str = "1.0.0")
```

**Parameters:**
- `name`: Plugin name
- `version`: Plugin version

#### Abstract Methods

##### `async initialize(config: Configuration) -> None`

Initialize the plugin with configuration.

**Parameters:**
- `config`: Configuration object

##### `async cleanup() -> None`

Clean up plugin resources.

#### Methods

##### `get_info() -> Dict[str, Any]`

Get plugin information.

**Returns:**
- Dictionary with plugin information

##### `async enable(config: Configuration) -> None`

Enable the plugin.

**Parameters:**
- `config`: Configuration object

##### `async disable() -> None`

Disable the plugin.

---

### AgentPlugin

Plugin for extending agent functionality.

#### Abstract Methods

##### `async process_message(agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]`

Process a message through the plugin.

**Parameters:**
- `agent_id`: Agent identifier
- `message`: Message to process
- `context`: Message context

**Returns:**
- Response dictionary or None if not handled

##### `should_handle_message(message: str, context: Dict[str, Any]) -> bool`

Determine if this plugin should handle the message.

**Parameters:**
- `message`: Input message
- `context`: Message context

**Returns:**
- True if plugin should handle the message

---

### ToolPlugin

Plugin for adding tools/functions to agents.

#### Abstract Methods

##### `get_tools() -> Dict[str, Callable]`

Get tools provided by this plugin.

**Returns:**
- Dictionary mapping tool names to functions

##### `async execute_tool(tool_name: str, **kwargs) -> Any`

Execute a tool function.

**Parameters:**
- `tool_name`: Name of the tool to execute
- `**kwargs`: Tool arguments

**Returns:**
- Tool execution result

---

### PluginManager

Main plugin manager for UAP SDK.

#### Constructor

```python
PluginManager(config: Configuration = None)
```

**Parameters:**
- `config`: Configuration object (optional)

#### Methods

##### `discover_plugins() -> Dict[str, Dict[str, Any]]`

Discover all available plugins.

**Returns:**
- Dictionary of discovered plugins

##### `async enable_plugin(plugin_name: str) -> bool`

Enable a plugin.

**Parameters:**
- `plugin_name`: Name of the plugin to enable

**Returns:**
- True if successful

##### `async disable_plugin(plugin_name: str) -> bool`

Disable a plugin.

**Parameters:**
- `plugin_name`: Name of the plugin to disable

**Returns:**
- True if successful

##### `get_enabled_plugins(plugin_type: str = None) -> List[UAPPlugin]`

Get enabled plugins, optionally filtered by type.

**Parameters:**
- `plugin_type`: Plugin type filter (optional)

**Returns:**
- List of enabled plugins

---

## Exceptions

### UAPException

Base exception for UAP SDK operations.

#### Constructor

```python
UAPException(message: str, error_code: str = None, details: dict = None)
```

**Parameters:**
- `message`: Error message
- `error_code`: Error code (optional)
- `details`: Error details dictionary (optional)

### UAPConnectionError

Exception raised when connection to UAP fails.

### UAPAuthError

Exception raised when authentication fails.

### UAPConfigError

Exception raised when configuration is invalid.

### UAPPluginError

Exception raised when plugin operations fail.

### UAPTimeoutError

Exception raised when operations timeout.

### UAPValidationError

Exception raised when data validation fails.

---

## Constants

### Default Configuration

```python
DEFAULT_CONFIG = {
    "backend_url": "http://localhost:8000",
    "websocket_url": "ws://localhost:8000", 
    "http_timeout": 30,
    "websocket_timeout": 30,
    "use_websocket": False,
    "log_level": "INFO",
    "max_retries": 3,
    "retry_delay": 1.0,
    "max_conversation_history": 50,
    "auto_reconnect": True
}
```

### Environment Variables

- `UAP_BACKEND_URL`: Backend URL
- `UAP_WEBSOCKET_URL`: WebSocket URL
- `UAP_HTTP_TIMEOUT`: HTTP timeout in seconds
- `UAP_WEBSOCKET_TIMEOUT`: WebSocket timeout in seconds
- `UAP_USE_WEBSOCKET`: Use WebSocket by default
- `UAP_LOG_LEVEL`: Logging level
- `UAP_MAX_RETRIES`: Maximum retry attempts
- `UAP_RETRY_DELAY`: Delay between retries in seconds
- `UAP_MAX_CONVERSATION_HISTORY`: Maximum conversation history length
- `UAP_AUTO_RECONNECT`: Auto-reconnect on connection loss
- `UAP_ACCESS_TOKEN`: Access token for authentication
- `UAP_REFRESH_TOKEN`: Refresh token for authentication

---

## Type Hints

Common type hints used throughout the SDK:

```python
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path

# Common types
ConfigDict = Dict[str, Any]
MessageContext = Dict[str, Any]
AgentResponse = Dict[str, Any]
PluginResult = Optional[Dict[str, Any]]
MiddlewareFunction = Callable[[str, Dict[str, Any]], Tuple[str, Dict[str, Any]]]
EventHandler = Callable[[Dict[str, Any]], None]
```