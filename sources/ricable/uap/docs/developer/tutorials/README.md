# UAP SDK Tutorials

Welcome to the UAP SDK tutorials! These step-by-step guides will help you learn how to build powerful AI agents and integrations.

## Tutorial Series

### Beginner Tutorials

1. [Your First Agent](#tutorial-1-your-first-agent) - Create a simple agent
2. [Connecting to UAP Backend](#tutorial-2-connecting-to-uap-backend) - Backend integration
3. [Agent Middleware](#tutorial-3-agent-middleware) - Request/response processing
4. [Configuration Management](#tutorial-4-configuration-management) - Config best practices

### Intermediate Tutorials

5. [Custom Agent Frameworks](#tutorial-5-custom-agent-frameworks) - Build custom frameworks
6. [Plugin Development](#tutorial-6-plugin-development) - Create and use plugins
7. [Multi-Agent Workflows](#tutorial-7-multi-agent-workflows) - Orchestrate multiple agents
8. [Real-time Communication](#tutorial-8-real-time-communication) - WebSocket integration

### Advanced Tutorials

9. [Production Deployment](#tutorial-9-production-deployment) - Deploy agents to production
10. [Performance Optimization](#tutorial-10-performance-optimization) - Optimize agent performance
11. [Custom Tools and Functions](#tutorial-11-custom-tools-and-functions) - Advanced tool development
12. [Testing and Debugging](#tutorial-12-testing-and-debugging) - Testing strategies

---

## Tutorial 1: Your First Agent

In this tutorial, you'll create your first UAP agent from scratch.

### Prerequisites

- Python 3.11+
- UAP SDK installed (`pip install uap-sdk`)

### Step 1: Create the Project

```bash
mkdir my-first-agent
cd my-first-agent
```

### Step 2: Create a Simple Agent

Create `simple_agent.py`:

```python
import asyncio
from uap_sdk import UAPAgent, CustomAgentBuilder, Configuration

async def main():
    print("Creating your first UAP agent...")
    
    # Step 1: Create configuration
    config = Configuration({
        "backend_url": "http://localhost:8000",
        "log_level": "INFO"
    })
    
    # Step 2: Build the agent
    agent = (CustomAgentBuilder("my-first-agent")
             .with_simple_framework()
             .with_config(config)
             .build())
    
    # Step 3: Start the agent
    await agent.start()
    print("Agent started successfully!")
    
    # Step 4: Test the agent
    test_messages = [
        "Hello, agent!",
        "How are you today?",
        "What can you do for me?",
        "Goodbye!"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = await agent.process_message(message)
        print(f"Agent: {response['content']}")
    
    # Step 5: Stop the agent
    await agent.stop()
    print("\nAgent stopped. Tutorial complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Run Your Agent

```bash
python simple_agent.py
```

### Expected Output

```
Creating your first UAP agent...
Agent started successfully!

User: Hello, agent!
Agent: Hello! I'm a simple UAP agent. How can I help you?

User: How are you today?
Agent: I'm a simple agent. I can respond to basic messages. You said: 'How are you today?'

User: What can you do for me?
Agent: I'm a simple agent. I can respond to basic messages. You said: 'What can you do for me?'

User: Goodbye!
Agent: Goodbye! It was nice talking with you.

Agent stopped. Tutorial complete!
```

### What You Learned

- How to create a basic UAP agent
- How to use the `CustomAgentBuilder`
- How to start, interact with, and stop an agent
- Basic agent response patterns

### Next Steps

Try modifying the agent's responses by creating a custom framework (Tutorial 5).

---

## Tutorial 2: Connecting to UAP Backend

Learn how to connect your agents to the UAP backend for advanced capabilities.

### Prerequisites

- Completed Tutorial 1
- UAP backend running (`uap deploy start`)

### Step 1: Install Dependencies

```bash
pip install httpx websockets
```

### Step 2: Create Backend Client

Create `backend_client.py`:

```python
import asyncio
from uap_sdk import UAPClient, Configuration
from uap_sdk.exceptions import UAPConnectionError, UAPAuthError

async def main():
    print("Connecting to UAP Backend...")
    
    # Step 1: Create configuration
    config = Configuration({
        "backend_url": "http://localhost:8000",
        "websocket_url": "ws://localhost:8000"
    })
    
    # Step 2: Create client
    client = UAPClient(config)
    
    try:
        # Step 3: Check system status
        print("Checking system status...")
        status = await client.get_status()
        print(f"Backend status: {status.get('status', 'unknown')}")
        
        # Step 4: Authenticate (optional)
        try:
            print("Attempting authentication...")
            auth_result = await client.login("admin", "admin123!")
            print(f"Authentication successful: {auth_result.get('message', 'Logged in')}")
        except UAPAuthError as e:
            print(f"Authentication failed: {e.message}")
            print("Continuing without authentication...")
        
        # Step 5: Test different frameworks
        frameworks = ["copilot", "agno", "mastra", "auto"]
        
        for framework in frameworks:
            print(f"\n--- Testing {framework.upper()} Framework ---")
            
            try:
                response = await client.chat(
                    agent_id=f"{framework}-agent",
                    message=f"Hello from the {framework} framework!",
                    framework=framework
                )
                
                content = response.get('content', '')
                if len(content) > 100:
                    content = content[:100] + "..."
                
                print(f"Response: {content}")
                
                # Show metadata
                metadata = response.get('metadata', {})
                if metadata:
                    print(f"Framework: {metadata.get('framework', 'unknown')}")
                    print(f"Response time: {metadata.get('response_time', 'N/A')}")
                
            except Exception as e:
                print(f"Error with {framework}: {e}")
    
    except UAPConnectionError as e:
        print(f"Connection failed: {e.message}")
        print("Make sure UAP backend is running:")
        print("  uap deploy start")
        print("  or: devbox run dev")
    
    finally:
        # Step 6: Cleanup
        await client.cleanup()
        print("\nClient cleaned up. Tutorial complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Run the Client

```bash
python backend_client.py
```

### Expected Output (with backend running)

```
Connecting to UAP Backend...
Checking system status...
Backend status: healthy
Attempting authentication...
Authentication successful: Login successful

--- Testing COPILOT Framework ---
Response: I'm CopilotKit, an AI coding assistant. I can help you with programming tasks, code review...
Framework: copilot
Response time: 150

--- Testing AGNO Framework ---
Response: I'm Agno, specialized in document processing and analysis. I can help you extract...
Framework: agno
Response time: 120

--- Testing MASTRA Framework ---
Response: I'm Mastra, your workflow automation assistant. I can help you create and manage...
Framework: mastra
Response time: 135

--- Testing AUTO Framework ---
Response: Hello! I'm the UAP auto-routing system. I'll route your message to the best framework...
Framework: auto
Response time: 95

Client cleaned up. Tutorial complete!
```

### What You Learned

- How to connect to the UAP backend
- How to authenticate with the system
- How to use different agent frameworks
- How to handle connection errors gracefully

### Next Steps

Learn about WebSocket communication in Tutorial 8.

---

## Tutorial 3: Agent Middleware

Learn how to add middleware to process messages before and after agent processing.

### Prerequisites

- Completed Tutorial 1

### Step 1: Create Middleware Functions

Create `middleware_agent.py`:

```python
import asyncio
import time
from typing import Dict, Any, Tuple
from datetime import datetime
from uap_sdk import UAPAgent, CustomAgentBuilder, Configuration

# Step 1: Define middleware functions
async def logging_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Log all messages with timestamps."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Processing message: {message}")
    
    # Add logging info to context
    context['logged_at'] = timestamp
    context['original_message'] = message
    
    return message, context

async def profanity_filter_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Filter inappropriate content."""
    # Simple profanity filter (in production, use a proper library)
    blocked_words = ['spam', 'badword', 'inappropriate']
    
    filtered_message = message
    for word in blocked_words:
        if word.lower() in message.lower():
            filtered_message = filtered_message.replace(word, "***")
            context['content_filtered'] = True
            print(f"Content filtered: '{word}' -> '***'")
    
    return filtered_message, context

async def sentiment_analysis_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Add sentiment analysis to context."""
    # Simple sentiment analysis
    positive_words = ['happy', 'good', 'great', 'excellent', 'love', 'like']
    negative_words = ['sad', 'bad', 'terrible', 'hate', 'dislike', 'awful']
    
    message_lower = message.lower()
    positive_count = sum(1 for word in positive_words if word in message_lower)
    negative_count = sum(1 for word in negative_words if word in message_lower)
    
    if positive_count > negative_count:
        sentiment = 'positive'
    elif negative_count > positive_count:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    context['sentiment'] = {
        'classification': sentiment,
        'positive_words': positive_count,
        'negative_words': negative_count
    }
    
    print(f"Sentiment detected: {sentiment}")
    return message, context

async def timing_middleware(message: str, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Add timing information."""
    context['processing_start'] = time.time()
    return message, context

async def main():
    print("=== Middleware Tutorial ===\n")
    
    # Step 2: Create configuration
    config = Configuration()
    
    # Step 3: Build agent with middleware
    agent = (CustomAgentBuilder("middleware-agent")
             .with_simple_framework()
             .with_config(config)
             .add_middleware(timing_middleware)      # First: timing
             .add_middleware(logging_middleware)     # Second: logging
             .add_middleware(profanity_filter_middleware)  # Third: filtering
             .add_middleware(sentiment_analysis_middleware)  # Fourth: sentiment
             .build())
    
    await agent.start()
    print("Agent with middleware started!\n")
    
    # Step 4: Test with different types of messages
    test_messages = [
        "Hello! I'm having a great day!",
        "This is terrible and I hate it",
        "Can you help me with something?",
        "I love this badword system!",  # Will trigger filter
        "What's the weather like today?"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"--- Test {i} ---")
        print(f"Input: {message}")
        
        start_time = time.time()
        response = await agent.process_message(message)
        end_time = time.time()
        
        print(f"Output: {response['content']}")
        
        # Show processing information
        metadata = response.get('metadata', {})
        print(f"Processing time: {(end_time - start_time) * 1000:.2f}ms")
        
        # Show middleware effects
        if 'sentiment' in metadata:
            sentiment_info = metadata['sentiment']
            print(f"Sentiment: {sentiment_info['classification']}")
        
        if metadata.get('content_filtered'):
            print("Content was filtered")
        
        print()
    
    await agent.stop()
    print("Middleware tutorial complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 2: Run the Middleware Agent

```bash
python middleware_agent.py
```

### Expected Output

```
=== Middleware Tutorial ===

Agent with middleware started!

--- Test 1 ---
[14:30:15] Processing message: Hello! I'm having a great day!
Sentiment detected: positive
Input: Hello! I'm having a great day!
Output: Hello! I'm a simple UAP agent. How can I help you?
Processing time: 2.45ms
Sentiment: positive

--- Test 2 ---
[14:30:15] Processing message: This is terrible and I hate it
Sentiment detected: negative
Input: This is terrible and I hate it
Output: I'm a simple agent. I can respond to basic messages. You said: 'This is terrible and I hate it'
Processing time: 1.98ms
Sentiment: negative

--- Test 3 ---
[14:30:15] Processing message: Can you help me with something?
Sentiment detected: neutral
Input: Can you help me with something?
Output: I'm a simple agent. I can respond to basic messages. You said: 'Can you help me with something?'
Processing time: 1.85ms
Sentiment: neutral

--- Test 4 ---
[14:30:15] Processing message: I love this badword system!
Content filtered: 'badword' -> '***'
Sentiment detected: positive
Input: I love this badword system!
Output: I'm a simple agent. I can respond to basic messages. You said: 'I love this *** system!'
Processing time: 2.12ms
Sentiment: positive
Content was filtered

--- Test 5 ---
[14:30:15] Processing message: What's the weather like today?
Sentiment detected: neutral
Input: What's the weather like today?
Output: I'm a simple agent. I can respond to basic messages. You said: 'What's the weather like today?'
Processing time: 1.78ms
Sentiment: neutral

Middleware tutorial complete!
```

### What You Learned

- How to create middleware functions
- How to chain multiple middleware components
- How to pass context between middleware
- How to modify messages in the processing pipeline
- How to add metadata and analytics

### Next Steps

Learn about configuration management in Tutorial 4.

---

## Tutorial 4: Configuration Management

Learn best practices for managing configuration in your UAP applications.

### Prerequisites

- Basic understanding of JSON and YAML

### Step 1: Create Configuration Files

Create `config/development.json`:

```json
{
  "backend_url": "http://localhost:8000",
  "websocket_url": "ws://localhost:8000",
  "http_timeout": 30,
  "log_level": "DEBUG",
  "agent_settings": {
    "max_conversation_history": 10,
    "default_framework": "auto"
  },
  "features": {
    "websocket_enabled": true,
    "analytics_enabled": false
  }
}
```

Create `config/production.json`:

```json
{
  "backend_url": "https://api.yourdomain.com",
  "websocket_url": "wss://api.yourdomain.com",
  "http_timeout": 60,
  "log_level": "INFO",
  "agent_settings": {
    "max_conversation_history": 50,
    "default_framework": "auto"
  },
  "features": {
    "websocket_enabled": true,
    "analytics_enabled": true
  }
}
```

Create `config/config.yaml` (alternative format):

```yaml
# UAP Configuration
backend_url: "http://localhost:8000"
websocket_url: "ws://localhost:8000"
http_timeout: 30
log_level: "INFO"

agent_settings:
  max_conversation_history: 25
  default_framework: "auto"
  retry_attempts: 3

features:
  websocket_enabled: true
  analytics_enabled: false
  debug_mode: false

plugins:
  enabled:
    - "calculator"
    - "weather"
  disabled:
    - "experimental"
```

### Step 2: Create Configuration Manager

Create `config_tutorial.py`:

```python
import asyncio
import os
from pathlib import Path
from uap_sdk import Configuration, UAPAgent, CustomAgentBuilder
from uap_sdk.config import ConfigurationProfile

class ConfigurationTutorial:
    """Tutorial for configuration management."""
    
    def __init__(self):
        self.configs = {}
    
    def demo_basic_configuration(self):
        """Demonstrate basic configuration usage."""
        print("=== Basic Configuration ===")
        
        # Method 1: From dictionary
        config_dict = {
            "backend_url": "http://localhost:8000",
            "log_level": "INFO",
            "timeout": 30
        }
        config1 = Configuration(config_dict=config_dict)
        print(f"Config from dict: {config1.get('backend_url')}")
        
        # Method 2: From file
        config2 = Configuration(config_file="config/development.json")
        print(f"Config from file: {config2.get('backend_url')}")
        
        # Method 3: From environment variables
        os.environ["UAP_BACKEND_URL"] = "http://env-backend:8000"
        config3 = Configuration.from_env()
        print(f"Config from env: {config3.get('backend_url')}")
        
        self.configs.update({
            "dict": config1,
            "file": config2,
            "env": config3
        })
        print()
    
    def demo_configuration_validation(self):
        """Demonstrate configuration validation."""
        print("=== Configuration Validation ===")
        
        # Valid configuration
        try:
            valid_config = Configuration({
                "backend_url": "http://localhost:8000",
                "websocket_url": "ws://localhost:8000",
                "http_timeout": 30
            })
            valid_config.validate()
            print("✓ Valid configuration passed validation")
        except ValueError as e:
            print(f"✗ Validation failed: {e}")
        
        # Invalid configuration
        try:
            invalid_config = Configuration({
                "backend_url": "invalid-url",  # Invalid URL format
                "websocket_url": "ws://localhost:8000",
                "http_timeout": -5  # Invalid timeout
            })
            invalid_config.validate()
            print("✓ Invalid configuration somehow passed")
        except ValueError as e:
            print(f"✓ Validation correctly caught error: {e}")
        
        print()
    
    def demo_configuration_profiles(self):
        """Demonstrate configuration profiles."""
        print("=== Configuration Profiles ===")
        
        # Create profile manager
        profile_manager = ConfigurationProfile()
        
        # Create different profiles
        dev_config = Configuration(config_file="config/development.json")
        prod_config = Configuration(config_file="config/production.json")
        
        # Save profiles
        profile_manager.create_profile("development", dev_config)
        profile_manager.create_profile("production", prod_config)
        
        print("Created profiles:")
        for profile in profile_manager.list_profiles():
            print(f"  - {profile}")
        
        # Switch between profiles
        profile_manager.set_current_profile("development")
        current_config = profile_manager.get_current_config()
        print(f"Current profile config: {current_config.get('log_level')}")
        
        profile_manager.set_current_profile("production")
        current_config = profile_manager.get_current_config()
        print(f"Production profile config: {current_config.get('log_level')}")
        
        print()
    
    def demo_environment_specific_config(self):
        """Demonstrate environment-specific configuration."""
        print("=== Environment-Specific Configuration ===")
        
        # Simulate different environments
        environments = {
            "development": {
                "UAP_BACKEND_URL": "http://localhost:8000",
                "UAP_LOG_LEVEL": "DEBUG",
                "UAP_USE_WEBSOCKET": "true"
            },
            "staging": {
                "UAP_BACKEND_URL": "https://staging-api.example.com",
                "UAP_LOG_LEVEL": "INFO",
                "UAP_USE_WEBSOCKET": "true"
            },
            "production": {
                "UAP_BACKEND_URL": "https://api.example.com",
                "UAP_LOG_LEVEL": "WARNING",
                "UAP_USE_WEBSOCKET": "true"
            }
        }
        
        for env_name, env_vars in environments.items():
            print(f"--- {env_name.upper()} Environment ---")
            
            # Set environment variables
            for key, value in env_vars.items():
                os.environ[key] = value
            
            # Load configuration from environment
            config = Configuration.from_env()
            
            print(f"Backend URL: {config.get('backend_url')}")
            print(f"Log Level: {config.get('log_level')}")
            print(f"WebSocket: {config.get('use_websocket')}")
            print()
    
    def demo_runtime_configuration(self):
        """Demonstrate runtime configuration changes."""
        print("=== Runtime Configuration Changes ===")
        
        config = Configuration({
            "backend_url": "http://localhost:8000",
            "timeout": 30,
            "retries": 3
        })
        
        print(f"Initial timeout: {config.get('timeout')}")
        
        # Update configuration at runtime
        config.set("timeout", 60)
        print(f"Updated timeout: {config.get('timeout')}")
        
        # Bulk update
        config.update({
            "retries": 5,
            "new_feature": True
        })
        print(f"Updated retries: {config.get('retries')}")
        print(f"New feature: {config.get('new_feature')}")
        
        # Save updated configuration
        config.save_to_file("config/runtime-updated.json")
        print("Configuration saved to file")
        
        print()
    
    async def demo_agent_with_configuration(self):
        """Demonstrate using configuration with agents."""
        print("=== Agent Configuration Integration ===")
        
        # Load configuration with agent-specific settings
        config = Configuration(config_file="config/config.yaml")
        
        # Extract agent settings
        agent_settings = config.get("agent_settings", {})
        max_history = agent_settings.get("max_conversation_history", 10)
        default_framework = agent_settings.get("default_framework", "simple")
        
        print(f"Agent configuration:")
        print(f"  Max conversation history: {max_history}")
        print(f"  Default framework: {default_framework}")
        
        # Create agent with configuration
        agent = (CustomAgentBuilder("config-demo-agent")
                .with_simple_framework()
                .with_config(config)
                .with_metadata("max_history", max_history)
                .build())
        
        await agent.start()
        
        # Test agent
        for i in range(3):
            response = await agent.process_message(f"Test message {i+1}")
            print(f"Response {i+1}: {response['content'][:50]}...")
        
        # Show agent status with configuration info
        status = agent.get_status()
        print(f"Agent metadata: {status.get('metadata', {})}")
        
        await agent.stop()
        print()
    
    def demo_configuration_best_practices(self):
        """Demonstrate configuration best practices."""
        print("=== Configuration Best Practices ===")
        
        # 1. Use defaults
        config = Configuration()
        
        # 2. Validate early
        try:
            config.validate()
            print("✓ Configuration validated successfully")
        except ValueError as e:
            print(f"✗ Configuration validation failed: {e}")
        
        # 3. Use type-safe getters
        backend_url = config.get("backend_url", "http://localhost:8000")
        timeout = config.get("http_timeout", 30)
        use_ws = config.get("use_websocket", False)
        
        print(f"Backend URL (string): {backend_url}")
        print(f"Timeout (int): {timeout}")
        print(f"Use WebSocket (bool): {use_ws}")
        
        # 4. Separate sensitive config
        # Don't put API keys in config files!
        api_key = os.getenv("API_KEY", "your-api-key-here")
        config.set("api_key", api_key)
        
        # 5. Use hierarchical configuration
        nested_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "uap_db"
            },
            "redis": {
                "host": "localhost", 
                "port": 6379,
                "db": 0
            }
        }
        config.update(nested_config)
        
        db_config = config.get("database", {})
        print(f"Database host: {db_config.get('host')}")
        
        print()

async def main():
    """Run the configuration tutorial."""
    print("=== UAP Configuration Management Tutorial ===\n")
    
    tutorial = ConfigurationTutorial()
    
    tutorial.demo_basic_configuration()
    tutorial.demo_configuration_validation()
    tutorial.demo_configuration_profiles()
    tutorial.demo_environment_specific_config()
    tutorial.demo_runtime_configuration()
    await tutorial.demo_agent_with_configuration()
    tutorial.demo_configuration_best_practices()
    
    print("Configuration tutorial complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Run the Tutorial

```bash
mkdir -p config
# Copy the config files from Step 1
python config_tutorial.py
```

### What You Learned

- How to load configuration from files, dictionaries, and environment variables
- How to validate configuration
- How to use configuration profiles for different environments
- How to update configuration at runtime
- How to integrate configuration with agents
- Configuration best practices and security considerations

### Next Steps

Learn about custom agent frameworks in Tutorial 5.

---

## More Tutorials Coming Soon!

The remaining tutorials are being developed:

- Tutorial 5: Custom Agent Frameworks
- Tutorial 6: Plugin Development  
- Tutorial 7: Multi-Agent Workflows
- Tutorial 8: Real-time Communication
- Tutorial 9: Production Deployment
- Tutorial 10: Performance Optimization
- Tutorial 11: Custom Tools and Functions
- Tutorial 12: Testing and Debugging

Check back soon for the complete tutorial series!

---

## Getting Help

If you need help with any tutorial:

1. Check the [API Reference](../api/README.md)
2. Review the [Examples](../../../examples/)
3. Ask questions in our [GitHub Discussions](https://github.com/uap/discussions)
4. Report issues in [GitHub Issues](https://github.com/uap/issues)

Happy coding with UAP! 🚀