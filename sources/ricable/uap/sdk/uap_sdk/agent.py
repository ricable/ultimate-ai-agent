# UAP SDK Agent Module
"""
Base classes and framework interfaces for building custom UAP agents.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
import uuid

from .config import Configuration
from .client import UAPClient
from .exceptions import UAPException

logger = logging.getLogger(__name__)


class AgentFramework(ABC):
    """Abstract base class for UAP agent frameworks."""
    
    def __init__(self, framework_name: str, config: Optional[Configuration] = None):
        self.framework_name = framework_name
        self.config = config or Configuration()
        self.is_initialized = False
        self.status = "initializing"
        self.error_count = 0
        self.last_error = None
        
    @abstractmethod
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a user message and return a response.
        
        Args:
            message: The user's input message
            context: Additional context for processing
            
        Returns:
            Dict containing 'content' and 'metadata' keys
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the framework.
        
        Returns:
            Dict containing status information
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the framework resources."""
        pass
    
    async def cleanup(self) -> None:
        """Clean up framework resources. Override if needed."""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Get the capabilities of this framework. Override if needed."""
        return ["general"]


class UAPAgent:
    """Main agent class for building custom UAP agents."""
    
    def __init__(self, 
                 agent_id: str,
                 framework: AgentFramework,
                 config: Optional[Configuration] = None,
                 client: Optional[UAPClient] = None):
        self.agent_id = agent_id
        self.framework = framework
        self.config = config or Configuration()
        self.client = client or UAPClient(self.config)
        
        # Agent state
        self.is_running = False
        self.conversation_history: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.middleware: List[Callable] = []
        
        # Statistics
        self.message_count = 0
        self.error_count = 0
        self.start_time = None
        
        logger.info(f"Agent {agent_id} created with framework {framework.framework_name}")
    
    async def start(self) -> None:
        """Start the agent and initialize all components."""
        try:
            # Initialize framework
            await self.framework.initialize()
            
            # Set up WebSocket handlers if using real-time communication
            if self.config.get("use_websocket", False):
                await self._setup_websocket_handlers()
            
            self.is_running = True
            self.start_time = datetime.utcnow()
            
            logger.info(f"Agent {self.agent_id} started successfully")
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to start agent {self.agent_id}: {e}")
            raise UAPException(f"Agent startup failed: {str(e)}")
    
    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        self.is_running = False
        
        # Clean up framework
        await self.framework.cleanup()
        
        # Clean up client connections
        await self.client.cleanup()
        
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def process_message(self, message: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a message through the agent's framework."""
        if not self.is_running:
            raise UAPException("Agent is not running")
        
        # Apply middleware
        processed_message = message
        processed_context = context or {}
        
        for middleware in self.middleware:
            try:
                if asyncio.iscoroutinefunction(middleware):
                    processed_message, processed_context = await middleware(processed_message, processed_context)
                else:
                    processed_message, processed_context = middleware(processed_message, processed_context)
            except Exception as e:
                logger.error(f"Middleware error: {e}")
                continue
        
        # Add agent context
        agent_context = {
            "agent_id": self.agent_id,
            "framework": self.framework.framework_name,
            "conversation_history": self.conversation_history[-5:],  # Last 5 messages
            "message_count": self.message_count,
            **processed_context
        }
        
        try:
            # Process with framework
            response = await self.framework.process_message(processed_message, agent_context)
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": processed_message,
                "timestamp": datetime.utcnow().isoformat()
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": response.get("content", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": response.get("metadata", {})
            })
            
            # Keep history manageable
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
            
            self.message_count += 1
            
            return response
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Agent {self.agent_id} processing error: {e}")
            
            error_response = {
                "content": f"I encountered an error processing your message: {str(e)}",
                "metadata": {
                    "error": True,
                    "error_message": str(e),
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            return error_response
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to process messages before framework processing."""
        self.middleware.append(middleware)
    
    def on_message(self, event_type: str, handler: Callable) -> None:
        """Register a message handler for WebSocket events."""
        self.message_handlers[event_type] = handler
    
    async def send_message(self, message: str, use_websocket: bool = False) -> Dict[str, Any]:
        """Send a message using the agent's client."""
        return await self.client.chat(
            self.agent_id, 
            message, 
            framework=self.framework.framework_name,
            use_websocket=use_websocket
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        framework_status = self.framework.get_status()
        
        uptime_seconds = 0
        if self.start_time:
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "agent_id": self.agent_id,
            "is_running": self.is_running,
            "framework": framework_status,
            "statistics": {
                "message_count": self.message_count,
                "error_count": self.error_count,
                "uptime_seconds": uptime_seconds,
                "conversation_length": len(self.conversation_history)
            },
            "capabilities": self.framework.get_capabilities(),
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set agent metadata."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get agent metadata."""
        return self.metadata.get(key, default)
    
    async def _setup_websocket_handlers(self) -> None:
        """Set up WebSocket message handlers."""
        # Connect to WebSocket
        await self.client.connect_websocket(self.agent_id)
        
        # Set up default handlers
        async def handle_user_message(event: Dict[str, Any]) -> None:
            message = event.get("content", "")
            context = event.get("metadata", {})
            
            try:
                response = await self.process_message(message, context)
                # In a real implementation, you'd send the response back
                # through the WebSocket connection
                logger.info(f"Processed WebSocket message: {response.get('content', '')[:100]}...")
            except Exception as e:
                logger.error(f"WebSocket message processing error: {e}")
        
        # Register default handlers
        self.client.websocket.on_message("user_message", handle_user_message)
        
        # Register custom handlers
        for event_type, handler in self.message_handlers.items():
            self.client.websocket.on_message(event_type, handler)


class SimpleAgent(AgentFramework):
    """A simple agent framework implementation for basic use cases."""
    
    def __init__(self, config: Optional[Configuration] = None):
        super().__init__("simple", config)
        self.responses = {
            "greeting": "Hello! I'm a simple UAP agent. How can I help you?",
            "goodbye": "Goodbye! It was nice talking with you.",
            "default": "I'm a simple agent. I can respond to basic messages."
        }
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message with simple pattern matching."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey"]):
            response = self.responses["greeting"]
        elif any(word in message_lower for word in ["bye", "goodbye", "exit"]):
            response = self.responses["goodbye"]
        else:
            response = f"{self.responses['default']} You said: '{message}'"
        
        return {
            "content": response,
            "metadata": {
                "source": self.framework_name,
                "simple_agent": True,
                "message_length": len(message),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get simple agent status."""
        return {
            "status": self.status,
            "framework": self.framework_name,
            "initialized": self.is_initialized,
            "response_patterns": len(self.responses),
            "capabilities": self.get_capabilities()
        }
    
    async def initialize(self) -> None:
        """Initialize the simple agent."""
        self.is_initialized = True
        self.status = "active"
        logger.info("Simple agent initialized")
    
    def get_capabilities(self) -> List[str]:
        """Get simple agent capabilities."""
        return ["greeting", "goodbye", "echo", "simple_responses"]


class CustomAgentBuilder:
    """Builder class for creating custom agents with fluent interface."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._framework = None
        self._config = Configuration()
        self._middleware = []
        self._handlers = {}
        self._metadata = {}
    
    def with_framework(self, framework: AgentFramework) -> 'CustomAgentBuilder':
        """Set the agent framework."""
        self._framework = framework
        return self
    
    def with_config(self, config: Configuration) -> 'CustomAgentBuilder':
        """Set the agent configuration."""
        self._config = config
        return self
    
    def with_simple_framework(self) -> 'CustomAgentBuilder':
        """Use the built-in simple framework."""
        self._framework = SimpleAgent(self._config)
        return self
    
    def add_middleware(self, middleware: Callable) -> 'CustomAgentBuilder':
        """Add middleware to the agent."""
        self._middleware.append(middleware)
        return self
    
    def on_message(self, event_type: str, handler: Callable) -> 'CustomAgentBuilder':
        """Add a message handler."""
        self._handlers[event_type] = handler
        return self
    
    def with_metadata(self, key: str, value: Any) -> 'CustomAgentBuilder':
        """Add metadata to the agent."""
        self._metadata[key] = value
        return self
    
    def build(self) -> UAPAgent:
        """Build the custom agent."""
        if not self._framework:
            raise UAPException("Framework is required to build agent")
        
        agent = UAPAgent(self.agent_id, self._framework, self._config)
        
        # Add middleware
        for middleware in self._middleware:
            agent.add_middleware(middleware)
        
        # Add message handlers
        for event_type, handler in self._handlers.items():
            agent.on_message(event_type, handler)
        
        # Set metadata
        for key, value in self._metadata.items():
            agent.set_metadata(key, value)
        
        return agent