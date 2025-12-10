"""
FastMCP Server for SAFLA

This module provides a high-level server interface for creating FastMCP
services that integrate with SAFLA's architecture, including automatic
tool registration, resource management, and health monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass
from contextlib import asynccontextmanager
import inspect

try:
    from fastmcp import FastMCP
    from fastmcp.server import Server
    from fastmcp.exceptions import ToolError, ResourceError, McpError
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

from .fastmcp_adapter import FastMCPConfig, FastMCPEndpoint
from ..exceptions import SAFLAError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a FastMCP tool."""
    name: str
    description: str
    handler: Callable
    parameters: Optional[Dict[str, Any]] = None
    return_type: Optional[Type] = None
    tags: Optional[List[str]] = None


@dataclass
class ResourceDefinition:
    """Definition of a FastMCP resource."""
    uri: str
    name: str
    description: str
    handler: Callable
    mime_type: Optional[str] = None
    tags: Optional[List[str]] = None


class FastMCPServer:
    """
    High-level FastMCP server that provides a simplified interface for
    creating FastMCP services integrated with SAFLA.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        version: str = "1.0.0",
        config: Optional[FastMCPConfig] = None
    ):
        """
        Initialize the FastMCP server.
        
        Args:
            name: Server name
            description: Server description
            version: Server version
            config: FastMCP configuration
        """
        if not FASTMCP_AVAILABLE:
            raise SAFLAError(
                "FastMCP is not available. Install with: pip install fastmcp"
            )
        
        self.name = name
        self.description = description or f"SAFLA FastMCP Server: {name}"
        self.version = version
        self.config = config or FastMCPConfig()
        
        # Initialize FastMCP server
        self.server = FastMCP(name=name)
        
        # Tool and resource registries
        self.tools: Dict[str, ToolDefinition] = {}
        self.resources: Dict[str, ResourceDefinition] = {}
        
        # Server state
        self._running = False
        self._health_check_handler: Optional[Callable] = None
        
        # Setup default health check
        self.register_tool(
            name="health_check",
            description="Check server health status",
            handler=self._default_health_check
        )
        
        logger.info(f"Initialized FastMCP server: {name}")
    
    def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        return_type: Optional[Type] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a tool with the FastMCP server.
        
        Args:
            name: Tool name
            description: Tool description
            handler: Tool handler function
            parameters: Tool parameters schema
            return_type: Expected return type
            tags: Tool tags for categorization
        """
        # Validate handler
        if not callable(handler):
            raise SAFLAError(f"Tool handler must be callable: {name}")
        
        # Auto-detect parameters if not provided
        if parameters is None:
            parameters = self._extract_parameters(handler)
        
        # Create tool definition
        tool_def = ToolDefinition(
            name=name,
            description=description,
            handler=handler,
            parameters=parameters,
            return_type=return_type,
            tags=tags or []
        )
        
        # Register with FastMCP
        @self.server.tool(name=name, description=description)
        async def tool_wrapper(*args, **kwargs):
            try:
                # Call the handler
                if inspect.iscoroutinefunction(handler):
                    result = await handler(*args, **kwargs)
                else:
                    result = handler(*args, **kwargs)
                
                logger.debug(f"Tool executed successfully: {name}")
                return result
                
            except Exception as e:
                logger.error(f"Tool execution failed: {name} - {e}")
                raise ToolError(f"Tool '{name}' failed: {str(e)}")
        
        # Store definition
        self.tools[name] = tool_def
        logger.info(f"Registered tool: {name}")
    
    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        handler: Callable,
        mime_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a resource with the FastMCP server.
        
        Args:
            uri: Resource URI
            name: Resource name
            description: Resource description
            handler: Resource handler function
            mime_type: Resource MIME type
            tags: Resource tags for categorization
        """
        # Validate handler
        if not callable(handler):
            raise SAFLAError(f"Resource handler must be callable: {uri}")
        
        # Create resource definition
        resource_def = ResourceDefinition(
            uri=uri,
            name=name,
            description=description,
            handler=handler,
            mime_type=mime_type,
            tags=tags or []
        )
        
        # Register with FastMCP
        @self.server.resource(uri=uri)
        async def resource_wrapper():
            try:
                # Call the handler
                if inspect.iscoroutinefunction(handler):
                    result = await handler()
                else:
                    result = handler()
                
                logger.debug(f"Resource accessed successfully: {uri}")
                return result
                
            except Exception as e:
                logger.error(f"Resource access failed: {uri} - {e}")
                raise ResourceError(f"Resource '{uri}' failed: {str(e)}")
        
        # Store definition
        self.resources[uri] = resource_def
        logger.info(f"Registered resource: {uri}")
    
    def set_health_check_handler(self, handler: Callable) -> None:
        """
        Set a custom health check handler.
        
        Args:
            handler: Health check handler function
        """
        self._health_check_handler = handler
        logger.info("Set custom health check handler")
    
    async def _default_health_check(self) -> Dict[str, Any]:
        """
        Default health check implementation.
        
        Returns:
            Health status information
        """
        if self._health_check_handler:
            try:
                if inspect.iscoroutinefunction(self._health_check_handler):
                    custom_status = await self._health_check_handler()
                else:
                    custom_status = self._health_check_handler()
                
                if isinstance(custom_status, dict):
                    return {
                        "status": "healthy",
                        "server": self.name,
                        "version": self.version,
                        "tools": len(self.tools),
                        "resources": len(self.resources),
                        **custom_status
                    }
            except Exception as e:
                logger.warning(f"Custom health check failed: {e}")
        
        return {
            "status": "healthy",
            "server": self.name,
            "version": self.version,
            "tools": len(self.tools),
            "resources": len(self.resources),
            "uptime": "running" if self._running else "stopped"
        }
    
    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        return_type: Optional[Type] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Decorator for registering tools.
        
        Args:
            name: Tool name (uses function name if not provided)
            description: Tool description
            parameters: Tool parameters schema
            return_type: Expected return type
            tags: Tool tags
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or f"Tool: {tool_name}"
            
            self.register_tool(
                name=tool_name,
                description=tool_description,
                handler=func,
                parameters=parameters,
                return_type=return_type,
                tags=tags
            )
            
            return func
        
        return decorator
    
    def resource(
        self,
        uri: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        mime_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Decorator for registering resources.
        
        Args:
            uri: Resource URI
            name: Resource name
            description: Resource description
            mime_type: Resource MIME type
            tags: Resource tags
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            resource_uri = uri or f"resource://{func.__name__}"
            resource_name = name or func.__name__
            resource_description = description or func.__doc__ or f"Resource: {resource_name}"
            
            self.register_resource(
                uri=resource_uri,
                name=resource_name,
                description=resource_description,
                handler=func,
                mime_type=mime_type,
                tags=tags
            )
            
            return func
        
        return decorator
    
    async def start(self, host: str = "localhost", port: int = 8000) -> None:
        """
        Start the FastMCP server.
        
        Args:
            host: Server host
            port: Server port
        """
        try:
            self._running = True
            logger.info(f"Starting FastMCP server {self.name} on {host}:{port}")
            
            # Start the server (implementation depends on FastMCP version)
            await self.server.run(host=host, port=port)
            
        except Exception as e:
            self._running = False
            logger.error(f"Failed to start FastMCP server: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the FastMCP server."""
        try:
            self._running = False
            logger.info(f"Stopping FastMCP server: {self.name}")
            
            # Stop the server (implementation depends on FastMCP version)
            if hasattr(self.server, 'stop'):
                await self.server.stop()
            
        except Exception as e:
            logger.error(f"Error stopping FastMCP server: {e}")
            raise
    
    def is_running(self) -> bool:
        """
        Check if the server is running.
        
        Returns:
            True if server is running, False otherwise
        """
        return self._running
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information or None if not found
        """
        tool_def = self.tools.get(tool_name)
        if not tool_def:
            return None
        
        return {
            "name": tool_def.name,
            "description": tool_def.description,
            "parameters": tool_def.parameters,
            "return_type": tool_def.return_type.__name__ if tool_def.return_type else None,
            "tags": tool_def.tags
        }
    
    def get_resource_info(self, resource_uri: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered resource.
        
        Args:
            resource_uri: URI of the resource
            
        Returns:
            Resource information or None if not found
        """
        resource_def = self.resources.get(resource_uri)
        if not resource_def:
            return None
        
        return {
            "uri": resource_def.uri,
            "name": resource_def.name,
            "description": resource_def.description,
            "mime_type": resource_def.mime_type,
            "tags": resource_def.tags
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools.
        
        Returns:
            List of tool information
        """
        return [self.get_tool_info(name) for name in self.tools.keys()]
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """
        List all registered resources.
        
        Returns:
            List of resource information
        """
        return [self.get_resource_info(uri) for uri in self.resources.keys()]
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get server information.
        
        Returns:
            Server information
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "running": self._running,
            "tools": len(self.tools),
            "resources": len(self.resources),
            "tool_names": list(self.tools.keys()),
            "resource_uris": list(self.resources.keys())
        }
    
    @staticmethod
    def _extract_parameters(func: Callable) -> Dict[str, Any]:
        """
        Extract parameter schema from function signature.
        
        Args:
            func: Function to analyze
            
        Returns:
            Parameter schema
        """
        try:
            sig = inspect.signature(func)
            parameters = {}
            
            for param_name, param in sig.parameters.items():
                param_info = {"name": param_name}
                
                # Add type information if available
                if param.annotation != inspect.Parameter.empty:
                    param_info["type"] = param.annotation.__name__
                
                # Add default value if available
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                    param_info["required"] = False
                else:
                    param_info["required"] = True
                
                parameters[param_name] = param_info
            
            return parameters
            
        except Exception as e:
            logger.warning(f"Failed to extract parameters from {func.__name__}: {e}")
            return {}
    
    @asynccontextmanager
    async def lifespan(self):
        """
        Context manager for server lifespan management.
        
        Yields:
            Server instance
        """
        try:
            # Server startup logic here if needed
            logger.info(f"FastMCP server {self.name} starting up")
            yield self
        finally:
            # Server shutdown logic here if needed
            logger.info(f"FastMCP server {self.name} shutting down")
            if self._running:
                await self.stop()


class FastMCPServerBuilder:
    """Builder class for creating FastMCP servers with fluent interface."""
    
    def __init__(self, name: str):
        """
        Initialize the server builder.
        
        Args:
            name: Server name
        """
        self.name = name
        self.description: Optional[str] = None
        self.version: str = "1.0.0"
        self.config: Optional[FastMCPConfig] = None
        self._tools: List[Dict[str, Any]] = []
        self._resources: List[Dict[str, Any]] = []
        self._health_check_handler: Optional[Callable] = None
    
    def with_description(self, description: str) -> 'FastMCPServerBuilder':
        """
        Set server description.
        
        Args:
            description: Server description
            
        Returns:
            Builder instance for chaining
        """
        self.description = description
        return self
    
    def with_version(self, version: str) -> 'FastMCPServerBuilder':
        """
        Set server version.
        
        Args:
            version: Server version
            
        Returns:
            Builder instance for chaining
        """
        self.version = version
        return self
    
    def with_config(self, config: FastMCPConfig) -> 'FastMCPServerBuilder':
        """
        Set server configuration.
        
        Args:
            config: FastMCP configuration
            
        Returns:
            Builder instance for chaining
        """
        self.config = config
        return self
    
    def add_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[Dict[str, Any]] = None,
        return_type: Optional[Type] = None,
        tags: Optional[List[str]] = None
    ) -> 'FastMCPServerBuilder':
        """
        Add a tool to the server.
        
        Args:
            name: Tool name
            description: Tool description
            handler: Tool handler function
            parameters: Tool parameters schema
            return_type: Expected return type
            tags: Tool tags
            
        Returns:
            Builder instance for chaining
        """
        self._tools.append({
            "name": name,
            "description": description,
            "handler": handler,
            "parameters": parameters,
            "return_type": return_type,
            "tags": tags
        })
        return self
    
    def add_resource(
        self,
        uri: str,
        name: str,
        description: str,
        handler: Callable,
        mime_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> 'FastMCPServerBuilder':
        """
        Add a resource to the server.
        
        Args:
            uri: Resource URI
            name: Resource name
            description: Resource description
            handler: Resource handler function
            mime_type: Resource MIME type
            tags: Resource tags
            
        Returns:
            Builder instance for chaining
        """
        self._resources.append({
            "uri": uri,
            "name": name,
            "description": description,
            "handler": handler,
            "mime_type": mime_type,
            "tags": tags
        })
        return self
    
    def with_health_check(self, handler: Callable) -> 'FastMCPServerBuilder':
        """
        Set custom health check handler.
        
        Args:
            handler: Health check handler function
            
        Returns:
            Builder instance for chaining
        """
        self._health_check_handler = handler
        return self
    
    def build(self) -> FastMCPServer:
        """
        Build the FastMCP server.
        
        Returns:
            Configured FastMCP server instance
        """
        # Create server
        server = FastMCPServer(
            name=self.name,
            description=self.description,
            version=self.version,
            config=self.config
        )
        
        # Register tools
        for tool_config in self._tools:
            server.register_tool(**tool_config)
        
        # Register resources
        for resource_config in self._resources:
            server.register_resource(**resource_config)
        
        # Set health check handler
        if self._health_check_handler:
            server.set_health_check_handler(self._health_check_handler)
        
        return server