"""
FastMCP Adapter for SAFLA

This module provides a comprehensive adapter for integrating FastMCP services
with the SAFLA system, including authentication, session management, error handling,
and integration with the existing MCP orchestration infrastructure.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, SecretStr, AnyUrl, field_validator

try:
    from fastmcp import FastMCP, Client
    from fastmcp.client import SSETransport, StreamableHttpTransport
    from fastmcp.exceptions import ClientError, ToolError, ResourceError, McpError
    # Import handlers from their correct locations
    try:
        from fastmcp.client.roots import RootsHandler
    except ImportError:
        RootsHandler = Callable[[], None]
    
    try:
        from fastmcp.client.logging import LogHandler
    except ImportError:
        LogHandler = Callable[[str], None]
    
    try:
        from fastmcp.client.progress import ProgressHandler
    except ImportError:
        ProgressHandler = Callable[[float, str], None]
    
    try:
        from fastmcp.client.sampling import SamplingHandler
    except ImportError:
        SamplingHandler = Callable[[], None]
    
    MessageHandler = Callable[[Dict[str, Any]], None]
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    # Define fallback types when FastMCP is not available
    class Client:
        """Fallback Client class when FastMCP is not available."""
        pass
    
    class FastMCP:
        """Fallback FastMCP class when FastMCP is not available."""
        pass
    
    class SSETransport:
        """Fallback SSETransport class when FastMCP is not available."""
        pass
    
    class StreamableHttpTransport:
        """Fallback StreamableHttpTransport class when FastMCP is not available."""
        pass
    
    ProgressHandler = Callable[[float, str], None]
    RootsHandler = Callable[[], None]
    LogHandler = Callable[[str], None]
    MessageHandler = Callable[[Dict[str, Any]], None]
    SamplingHandler = Callable[[], None]
    ClientError = Exception
    ConnectionError = Exception
    ToolError = Exception
    ResourceError = Exception
    McpError = Exception

from ..core.mcp_orchestration import MCPServer, MCPServerStatus, MCPOrchestrator
from ..exceptions import SAFLAError

logger = logging.getLogger(__name__)


class FastMCPTransportType(Enum):
    """Enumeration of FastMCP transport types."""
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"
    IN_MEMORY = "in_memory"


class FastMCPAuthType(Enum):
    """Enumeration of FastMCP authentication types."""
    NONE = "none"
    BEARER_TOKEN = "bearer_token"
    API_KEY = "api_key"
    CUSTOM_HEADERS = "custom_headers"


@dataclass
class FastMCPEndpoint:
    """Configuration for a FastMCP endpoint."""
    name: str
    url: str
    transport_type: FastMCPTransportType
    auth_type: FastMCPAuthType = FastMCPAuthType.NONE
    auth_token: Optional[str] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    max_retries: int = 3
    health_check_interval: float = 60.0
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FastMCPConfig(BaseModel):
    """Pydantic configuration for FastMCP integration."""
    
    enabled: bool = Field(
        default=False,
        description="Enable FastMCP integration"
    )
    
    default_timeout: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Default timeout for FastMCP operations in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed operations"
    )
    
    connection_pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Connection pool size for FastMCP clients"
    )
    
    health_check_interval: float = Field(
        default=60.0,
        ge=10.0,
        le=600.0,
        description="Health check interval in seconds"
    )
    
    enable_logging: bool = Field(
        default=True,
        description="Enable FastMCP operation logging"
    )
    
    enable_progress_tracking: bool = Field(
        default=True,
        description="Enable progress tracking for long-running operations"
    )
    
    enable_sampling: bool = Field(
        default=False,
        description="Enable LLM sampling through FastMCP"
    )
    
    mask_error_details: bool = Field(
        default=True,
        description="Mask internal error details for security"
    )
    
    endpoints: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="FastMCP endpoint configurations"
    )
    
    # Authentication settings
    default_auth_type: str = Field(
        default="none",
        description="Default authentication type"
    )
    
    bearer_token: Optional[SecretStr] = Field(
        default=None,
        description="Default Bearer token for authentication"
    )
    
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Default API key for authentication"
    )
    
    custom_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Default custom headers for requests"
    )


class FastMCPAdapter:
    """
    FastMCP adapter that integrates with SAFLA's MCP orchestration system.
    
    This adapter provides:
    - Connection management to FastMCP services
    - Authentication and session management
    - Error handling and retries
    - Integration with SAFLA's MCP orchestrator
    - Health monitoring and failover
    """
    
    def __init__(self, config: FastMCPConfig, orchestrator: Optional[MCPOrchestrator] = None):
        """
        Initialize the FastMCP adapter.
        
        Args:
            config: FastMCP configuration
            orchestrator: Optional MCP orchestrator for integration
        """
        if not FASTMCP_AVAILABLE:
            raise SAFLAError(
                "FastMCP is not available. Install with: pip install fastmcp"
            )
        
        self.config = config
        self.orchestrator = orchestrator
        self.endpoints: Dict[str, FastMCPEndpoint] = {}
        self.clients: Dict[str, Client] = {}
        self.servers: Dict[str, FastMCP] = {}
        self.is_running = False
        
        # Event handlers
        self.log_handlers: Dict[str, LogHandler] = {}
        self.progress_handlers: Dict[str, ProgressHandler] = {}
        self.sampling_handlers: Dict[str, SamplingHandler] = {}
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._connection_pools: Dict[str, List[Client]] = {}
        
        logger.info("FastMCP adapter initialized")
    
    async def start(self) -> None:
        """Start the FastMCP adapter."""
        if self.is_running:
            logger.warning("FastMCP adapter is already running")
            return
        
        if not self.config.enabled:
            logger.info("FastMCP integration is disabled")
            return
        
        logger.info("Starting FastMCP adapter")
        
        try:
            # Load endpoint configurations
            await self._load_endpoints()
            
            # Initialize connections
            await self._initialize_connections()
            
            # Start health monitoring
            if self.config.health_check_interval > 0:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            # Register with MCP orchestrator if available
            if self.orchestrator:
                await self._register_with_orchestrator()
            
            self.is_running = True
            logger.info("FastMCP adapter started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start FastMCP adapter: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the FastMCP adapter."""
        if not self.is_running:
            logger.warning("FastMCP adapter is not running")
            return
        
        logger.info("Shutting down FastMCP adapter")
        
        self.is_running = False
        
        # Cancel health monitoring
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        for client in self.clients.values():
            try:
                if hasattr(client, 'close'):
                    await client.close()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
        
        # Close connection pools
        for pool in self._connection_pools.values():
            for client in pool:
                try:
                    if hasattr(client, 'close'):
                        await client.close()
                except Exception as e:
                    logger.warning(f"Error closing pooled client: {e}")
        
        self.clients.clear()
        self._connection_pools.clear()
        
        logger.info("FastMCP adapter shut down successfully")
    
    async def add_endpoint(self, endpoint: FastMCPEndpoint) -> bool:
        """
        Add a new FastMCP endpoint.
        
        Args:
            endpoint: FastMCP endpoint configuration
            
        Returns:
            True if endpoint added successfully, False otherwise
        """
        if endpoint.name in self.endpoints:
            logger.warning(f"Endpoint {endpoint.name} already exists")
            return False
        
        try:
            # Create client for the endpoint
            client = await self._create_client(endpoint)
            
            # Test connection
            async with client:
                await client.ping()
            
            # Store endpoint and client
            self.endpoints[endpoint.name] = endpoint
            self.clients[endpoint.name] = client
            
            # Register with orchestrator if available
            if self.orchestrator:
                await self._register_endpoint_with_orchestrator(endpoint)
            
            logger.info(f"Added FastMCP endpoint: {endpoint.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add endpoint {endpoint.name}: {e}")
            return False
    
    async def remove_endpoint(self, endpoint_name: str) -> bool:
        """
        Remove a FastMCP endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to remove
            
        Returns:
            True if endpoint removed successfully, False otherwise
        """
        if endpoint_name not in self.endpoints:
            logger.warning(f"Endpoint {endpoint_name} not found")
            return False
        
        try:
            # Close client connection
            client = self.clients.get(endpoint_name)
            if client and hasattr(client, 'close'):
                await client.close()
            
            # Remove from collections
            del self.endpoints[endpoint_name]
            if endpoint_name in self.clients:
                del self.clients[endpoint_name]
            
            # Unregister from orchestrator if available
            if self.orchestrator:
                await self._unregister_endpoint_from_orchestrator(endpoint_name)
            
            logger.info(f"Removed FastMCP endpoint: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove endpoint {endpoint_name}: {e}")
            return False
    
    async def call_tool(
        self,
        endpoint_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
        progress_handler: Optional[ProgressHandler] = None
    ) -> Any:
        """
        Call a tool on a FastMCP endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            tool_name: Name of the tool to call
            arguments: Tool arguments
            timeout: Optional timeout override
            progress_handler: Optional progress handler
            
        Returns:
            Tool execution result
            
        Raises:
            SAFLAError: If endpoint not found or tool call fails
        """
        if endpoint_name not in self.clients:
            raise SAFLAError(f"FastMCP endpoint not found: {endpoint_name}")
        
        client = self.clients[endpoint_name]
        endpoint = self.endpoints[endpoint_name]
        
        # Use endpoint timeout if not specified
        if timeout is None:
            timeout = endpoint.timeout
        
        try:
            async with client:
                result = await client.call_tool(
                    tool_name,
                    arguments,
                    timeout=timeout,
                    progress_handler=progress_handler
                )
                
                logger.debug(f"Tool call successful: {endpoint_name}.{tool_name}")
                return result
                
        except ClientError as e:
            logger.error(f"Tool call failed: {endpoint_name}.{tool_name} - {e}")
            raise SAFLAError(f"FastMCP tool call failed: {e}")
        except ConnectionError as e:
            logger.error(f"Connection error: {endpoint_name} - {e}")
            # Attempt recovery
            await self._handle_connection_error(endpoint_name, e)
            raise SAFLAError(f"FastMCP connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in tool call: {e}")
            raise SAFLAError(f"FastMCP tool call error: {e}")
    
    async def read_resource(
        self,
        endpoint_name: str,
        resource_uri: str,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Read a resource from a FastMCP endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            resource_uri: URI of the resource to read
            timeout: Optional timeout override
            
        Returns:
            Resource content
            
        Raises:
            SAFLAError: If endpoint not found or resource read fails
        """
        if endpoint_name not in self.clients:
            raise SAFLAError(f"FastMCP endpoint not found: {endpoint_name}")
        
        client = self.clients[endpoint_name]
        endpoint = self.endpoints[endpoint_name]
        
        # Use endpoint timeout if not specified
        if timeout is None:
            timeout = endpoint.timeout
        
        try:
            async with client:
                result = await client.read_resource(resource_uri, timeout=timeout)
                
                logger.debug(f"Resource read successful: {endpoint_name}:{resource_uri}")
                return result
                
        except ClientError as e:
            logger.error(f"Resource read failed: {endpoint_name}:{resource_uri} - {e}")
            raise SAFLAError(f"FastMCP resource read failed: {e}")
        except ConnectionError as e:
            logger.error(f"Connection error: {endpoint_name} - {e}")
            # Attempt recovery
            await self._handle_connection_error(endpoint_name, e)
            raise SAFLAError(f"FastMCP connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in resource read: {e}")
            raise SAFLAError(f"FastMCP resource read error: {e}")
    
    async def list_tools(self, endpoint_name: str) -> List[Dict[str, Any]]:
        """
        List available tools on a FastMCP endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            List of available tools
            
        Raises:
            SAFLAError: If endpoint not found or listing fails
        """
        if endpoint_name not in self.clients:
            raise SAFLAError(f"FastMCP endpoint not found: {endpoint_name}")
        
        client = self.clients[endpoint_name]
        
        try:
            async with client:
                tools = await client.list_tools()
                return [tool.dict() if hasattr(tool, 'dict') else tool for tool in tools]
                
        except Exception as e:
            logger.error(f"Failed to list tools for {endpoint_name}: {e}")
            raise SAFLAError(f"FastMCP list tools error: {e}")
    
    async def list_resources(self, endpoint_name: str) -> List[Dict[str, Any]]:
        """
        List available resources on a FastMCP endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            List of available resources
            
        Raises:
            SAFLAError: If endpoint not found or listing fails
        """
        if endpoint_name not in self.clients:
            raise SAFLAError(f"FastMCP endpoint not found: {endpoint_name}")
        
        client = self.clients[endpoint_name]
        
        try:
            async with client:
                resources = await client.list_resources()
                return [resource.dict() if hasattr(resource, 'dict') else resource for resource in resources]
                
        except Exception as e:
            logger.error(f"Failed to list resources for {endpoint_name}: {e}")
            raise SAFLAError(f"FastMCP list resources error: {e}")
    
    async def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Get the status of a FastMCP endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            Endpoint status information
        """
        if endpoint_name not in self.endpoints:
            return {"status": "not_found", "error": "Endpoint not found"}
        
        endpoint = self.endpoints[endpoint_name]
        client = self.clients.get(endpoint_name)
        
        status = {
            "name": endpoint_name,
            "url": endpoint.url,
            "transport_type": endpoint.transport_type.value,
            "auth_type": endpoint.auth_type.value,
            "timeout": endpoint.timeout,
            "capabilities": endpoint.capabilities,
            "metadata": endpoint.metadata
        }
        
        if client:
            try:
                async with client:
                    await client.ping()
                    status["status"] = "healthy"
                    status["connected"] = True
            except Exception as e:
                status["status"] = "unhealthy"
                status["connected"] = False
                status["error"] = str(e)
        else:
            status["status"] = "no_client"
            status["connected"] = False
        
        return status
    
    async def _load_endpoints(self) -> None:
        """Load endpoint configurations from config."""
        for name, endpoint_config in self.config.endpoints.items():
            try:
                # Parse transport type
                transport_type = FastMCPTransportType(
                    endpoint_config.get("transport_type", "streamable_http")
                )
                
                # Parse auth type
                auth_type = FastMCPAuthType(
                    endpoint_config.get("auth_type", self.config.default_auth_type)
                )
                
                # Create endpoint
                endpoint = FastMCPEndpoint(
                    name=name,
                    url=endpoint_config["url"],
                    transport_type=transport_type,
                    auth_type=auth_type,
                    auth_token=endpoint_config.get("auth_token"),
                    custom_headers=endpoint_config.get("custom_headers", {}),
                    timeout=endpoint_config.get("timeout", self.config.default_timeout),
                    max_retries=endpoint_config.get("max_retries", self.config.max_retries),
                    health_check_interval=endpoint_config.get(
                        "health_check_interval", self.config.health_check_interval
                    ),
                    capabilities=endpoint_config.get("capabilities", []),
                    metadata=endpoint_config.get("metadata", {})
                )
                
                self.endpoints[name] = endpoint
                logger.info(f"Loaded FastMCP endpoint configuration: {name}")
                
            except Exception as e:
                logger.error(f"Failed to load endpoint configuration {name}: {e}")
    
    async def _initialize_connections(self) -> None:
        """Initialize connections to all configured endpoints."""
        for endpoint in self.endpoints.values():
            try:
                client = await self._create_client(endpoint)
                self.clients[endpoint.name] = client
                
                # Initialize connection pool
                pool = []
                for _ in range(self.config.connection_pool_size):
                    pool_client = await self._create_client(endpoint)
                    pool.append(pool_client)
                self._connection_pools[endpoint.name] = pool
                
                logger.info(f"Initialized FastMCP client: {endpoint.name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize client for {endpoint.name}: {e}")
    
    async def _create_client(self, endpoint: FastMCPEndpoint) -> Client:
        """
        Create a FastMCP client for an endpoint.
        
        Args:
            endpoint: FastMCP endpoint configuration
            
        Returns:
            Configured FastMCP client
        """
        # Prepare headers
        headers = {}
        
        # Add authentication headers
        if endpoint.auth_type == FastMCPAuthType.BEARER_TOKEN:
            if endpoint.auth_token:
                headers["Authorization"] = f"Bearer {endpoint.auth_token}"
            elif self.config.bearer_token:
                headers["Authorization"] = f"Bearer {self.config.bearer_token.get_secret_value()}"
        
        elif endpoint.auth_type == FastMCPAuthType.API_KEY:
            if endpoint.auth_token:
                headers["X-API-Key"] = endpoint.auth_token
            elif self.config.api_key:
                headers["X-API-Key"] = self.config.api_key.get_secret_value()
        
        elif endpoint.auth_type == FastMCPAuthType.CUSTOM_HEADERS:
            headers.update(endpoint.custom_headers)
            headers.update(self.config.custom_headers)
        
        # Create transport
        if endpoint.transport_type == FastMCPTransportType.SSE:
            transport = SSETransport(url=endpoint.url, headers=headers)
        elif endpoint.transport_type == FastMCPTransportType.STREAMABLE_HTTP:
            transport = StreamableHttpTransport(url=endpoint.url, headers=headers)
        else:
            raise SAFLAError(f"Unsupported transport type: {endpoint.transport_type}")
        
        # Create client with handlers
        client_kwargs = {
            "timeout": endpoint.timeout
        }
        
        # Add handlers if enabled
        if self.config.enable_logging:
            client_kwargs["log_handler"] = self._create_log_handler(endpoint.name)
        
        if self.config.enable_progress_tracking:
            client_kwargs["progress_handler"] = self._create_progress_handler(endpoint.name)
        
        if self.config.enable_sampling:
            client_kwargs["sampling_handler"] = self._create_sampling_handler(endpoint.name)
        
        return Client(transport, **client_kwargs)
    
    def _create_log_handler(self, endpoint_name: str) -> LogHandler:
        """Create a log handler for an endpoint."""
        async def log_handler(level: str, message: str, data: Optional[Dict[str, Any]] = None):
            log_level = getattr(logging, level.upper(), logging.INFO)
            logger.log(log_level, f"[{endpoint_name}] {message}", extra=data or {})
        
        self.log_handlers[endpoint_name] = log_handler
        return log_handler
    
    def _create_progress_handler(self, endpoint_name: str) -> ProgressHandler:
        """Create a progress handler for an endpoint."""
        async def progress_handler(progress: float, message: Optional[str] = None):
            logger.info(f"[{endpoint_name}] Progress: {progress:.1%} - {message or 'Processing...'}")
        
        self.progress_handlers[endpoint_name] = progress_handler
        return progress_handler
    
    def _create_sampling_handler(self, endpoint_name: str) -> SamplingHandler:
        """Create a sampling handler for an endpoint."""
        async def sampling_handler(messages, params, context):
            # This would integrate with SAFLA's LLM capabilities
            # For now, return a simple response
            logger.info(f"[{endpoint_name}] Sampling request received")
            return "Sampling response from SAFLA"
        
        self.sampling_handlers[endpoint_name] = sampling_handler
        return sampling_handler
    
    async def _health_check_loop(self) -> None:
        """Background task for health checking endpoints."""
        while self.is_running:
            try:
                for endpoint_name in list(self.endpoints.keys()):
                    await self._check_endpoint_health(endpoint_name)
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _check_endpoint_health(self, endpoint_name: str) -> bool:
        """
        Check the health of a specific endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to check
            
        Returns:
            True if endpoint is healthy, False otherwise
        """
        if endpoint_name not in self.clients:
            return False
        
        client = self.clients[endpoint_name]
        
        try:
            async with client:
                await client.ping()
            
            logger.debug(f"Health check passed: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for {endpoint_name}: {e}")
            
            # Attempt to recover the connection
            await self._handle_connection_error(endpoint_name, e)
            return False
    
    async def _handle_connection_error(self, endpoint_name: str, error: Exception) -> None:
        """
        Handle connection errors for an endpoint.
        
        Args:
            endpoint_name: Name of the endpoint with connection error
            error: The connection error that occurred
        """
        logger.warning(f"Handling connection error for {endpoint_name}: {error}")
        
        endpoint = self.endpoints.get(endpoint_name)
        if not endpoint:
            return
        
        # Try to recreate the client
        try:
            new_client = await self._create_client(endpoint)
            
            # Test the new client
            async with new_client:
                await new_client.ping()
            
            # Replace the old client
            old_client = self.clients.get(endpoint_name)
            if old_client and hasattr(old_client, 'close'):
                try:
                    await old_client.close()
                except Exception:
                    pass
            
            self.clients[endpoint_name] = new_client
            logger.info(f"Successfully recovered connection for {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to recover connection for {endpoint_name}: {e}")
            
            # Report to orchestrator if available
            if self.orchestrator:
                await self._report_endpoint_error(endpoint_name, error)
    
    async def _register_with_orchestrator(self) -> None:
        """Register FastMCP endpoints with the MCP orchestrator."""
        if not self.orchestrator:
            return
        
        for endpoint in self.endpoints.values():
            await self._register_endpoint_with_orchestrator(endpoint)
    
    async def _register_endpoint_with_orchestrator(self, endpoint: FastMCPEndpoint) -> None:
        """Register a single endpoint with the MCP orchestrator."""
        if not self.orchestrator:
            return
        
        # Create MCP server representation
        mcp_server = MCPServer(
            server_id=f"fastmcp_{endpoint.name}",
            name=f"FastMCP: {endpoint.name}",
            endpoint=endpoint.url,
            capabilities=endpoint.capabilities + ["fastmcp"],
            status=MCPServerStatus.ACTIVE,
            metadata={
                "type": "fastmcp",
                "transport_type": endpoint.transport_type.value,
                "auth_type": endpoint.auth_type.value,
                **endpoint.metadata
            }
        )
        
        # Register with orchestrator
        success = self.orchestrator.server_manager.register_server(mcp_server)
        if success:
            logger.info(f"Registered FastMCP endpoint with orchestrator: {endpoint.name}")
        else:
            logger.warning(f"Failed to register FastMCP endpoint with orchestrator: {endpoint.name}")
    
    async def _unregister_endpoint_from_orchestrator(self, endpoint_name: str) -> None:
        """Unregister an endpoint from the MCP orchestrator."""
        if not self.orchestrator:
            return
        
        server_id = f"fastmcp_{endpoint_name}"
        success = self.orchestrator.server_manager.unregister_server(server_id)
        if success:
            logger.info(f"Unregistered FastMCP endpoint from orchestrator: {endpoint_name}")
        else:
            logger.warning(f"Failed to unregister FastMCP endpoint from orchestrator: {endpoint_name}")
    
    async def _report_endpoint_error(self, endpoint_name: str, error: Exception) -> None:
        """Report an endpoint error to the MCP orchestrator."""
        if not self.orchestrator:
            return
        
        error_context = {
            "server_id": f"fastmcp_{endpoint_name}",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time(),
            "component": "fastmcp_adapter"
        }
        
        # Use orchestrator's error handler
        await self.orchestrator.error_handler.handle_error(error_context)