"""
FastMCP Client for SAFLA

This module provides a high-level client interface for interacting with FastMCP
services, with built-in error handling, retries, and integration with SAFLA's
configuration and logging systems.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from contextlib import asynccontextmanager

try:
    from fastmcp import Client
    from fastmcp.client import SSETransport, StreamableHttpTransport
    from fastmcp.exceptions import ClientError, McpError
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

from .fastmcp_adapter import FastMCPAdapter, FastMCPConfig, FastMCPEndpoint
from ..exceptions import SAFLAError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FastMCPClient:
    """
    High-level FastMCP client that provides a simplified interface for
    interacting with FastMCP services through the SAFLA system.
    """
    
    def __init__(self, adapter: FastMCPAdapter):
        """
        Initialize the FastMCP client.
        
        Args:
            adapter: FastMCP adapter instance
        """
        if not FASTMCP_AVAILABLE:
            raise SAFLAError(
                "FastMCP is not available. Install with: pip install fastmcp"
            )
        
        self.adapter = adapter
        self._default_endpoint: Optional[str] = None
    
    def set_default_endpoint(self, endpoint_name: str) -> None:
        """
        Set the default endpoint for operations.
        
        Args:
            endpoint_name: Name of the endpoint to use as default
            
        Raises:
            SAFLAError: If endpoint not found
        """
        if endpoint_name not in self.adapter.endpoints:
            raise SAFLAError(f"Endpoint not found: {endpoint_name}")
        
        self._default_endpoint = endpoint_name
        logger.info(f"Set default FastMCP endpoint: {endpoint_name}")
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Any:
        """
        Call a tool on a FastMCP endpoint.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments (optional)
            endpoint: Endpoint name (uses default if not specified)
            timeout: Operation timeout (uses endpoint default if not specified)
            progress_callback: Optional progress callback function
            
        Returns:
            Tool execution result
            
        Raises:
            SAFLAError: If operation fails
        """
        endpoint_name = endpoint or self._default_endpoint
        if not endpoint_name:
            raise SAFLAError("No endpoint specified and no default endpoint set")
        
        if arguments is None:
            arguments = {}
        
        # Create progress handler if callback provided
        progress_handler = None
        if progress_callback:
            async def progress_handler(progress: float, message: Optional[str] = None):
                try:
                    progress_callback(progress, message or "Processing...")
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
        
        try:
            result = await self.adapter.call_tool(
                endpoint_name=endpoint_name,
                tool_name=tool_name,
                arguments=arguments,
                timeout=timeout,
                progress_handler=progress_handler
            )
            
            logger.info(f"Tool call successful: {tool_name} on {endpoint_name}")
            return result
            
        except Exception as e:
            logger.error(f"Tool call failed: {tool_name} on {endpoint_name} - {e}")
            raise
    
    async def read_resource(
        self,
        resource_uri: str,
        endpoint: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Read a resource from a FastMCP endpoint.
        
        Args:
            resource_uri: URI of the resource to read
            endpoint: Endpoint name (uses default if not specified)
            timeout: Operation timeout (uses endpoint default if not specified)
            
        Returns:
            Resource content
            
        Raises:
            SAFLAError: If operation fails
        """
        endpoint_name = endpoint or self._default_endpoint
        if not endpoint_name:
            raise SAFLAError("No endpoint specified and no default endpoint set")
        
        try:
            result = await self.adapter.read_resource(
                endpoint_name=endpoint_name,
                resource_uri=resource_uri,
                timeout=timeout
            )
            
            logger.info(f"Resource read successful: {resource_uri} from {endpoint_name}")
            return result
            
        except Exception as e:
            logger.error(f"Resource read failed: {resource_uri} from {endpoint_name} - {e}")
            raise
    
    async def list_tools(self, endpoint: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available tools on a FastMCP endpoint.
        
        Args:
            endpoint: Endpoint name (uses default if not specified)
            
        Returns:
            List of available tools
            
        Raises:
            SAFLAError: If operation fails
        """
        endpoint_name = endpoint or self._default_endpoint
        if not endpoint_name:
            raise SAFLAError("No endpoint specified and no default endpoint set")
        
        try:
            tools = await self.adapter.list_tools(endpoint_name)
            logger.debug(f"Listed {len(tools)} tools from {endpoint_name}")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to list tools from {endpoint_name}: {e}")
            raise
    
    async def list_resources(self, endpoint: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available resources on a FastMCP endpoint.
        
        Args:
            endpoint: Endpoint name (uses default if not specified)
            
        Returns:
            List of available resources
            
        Raises:
            SAFLAError: If operation fails
        """
        endpoint_name = endpoint or self._default_endpoint
        if not endpoint_name:
            raise SAFLAError("No endpoint specified and no default endpoint set")
        
        try:
            resources = await self.adapter.list_resources(endpoint_name)
            logger.debug(f"Listed {len(resources)} resources from {endpoint_name}")
            return resources
            
        except Exception as e:
            logger.error(f"Failed to list resources from {endpoint_name}: {e}")
            raise
    
    async def get_endpoint_info(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a FastMCP endpoint.
        
        Args:
            endpoint: Endpoint name (uses default if not specified)
            
        Returns:
            Endpoint information
            
        Raises:
            SAFLAError: If endpoint not found
        """
        endpoint_name = endpoint or self._default_endpoint
        if not endpoint_name:
            raise SAFLAError("No endpoint specified and no default endpoint set")
        
        try:
            status = await self.adapter.get_endpoint_status(endpoint_name)
            
            # Add additional information
            info = {
                **status,
                "is_default": endpoint_name == self._default_endpoint
            }
            
            # Get capabilities if endpoint is healthy
            if status.get("status") == "healthy":
                try:
                    tools = await self.list_tools(endpoint_name)
                    resources = await self.list_resources(endpoint_name)
                    info["tool_count"] = len(tools)
                    info["resource_count"] = len(resources)
                except Exception:
                    # Don't fail if we can't get capabilities
                    pass
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get endpoint info for {endpoint_name}: {e}")
            raise
    
    async def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all configured FastMCP endpoints.
        
        Returns:
            List of endpoint information
        """
        endpoints = []
        
        for endpoint_name in self.adapter.endpoints.keys():
            try:
                info = await self.get_endpoint_info(endpoint_name)
                endpoints.append(info)
            except Exception as e:
                # Include failed endpoints with error info
                endpoints.append({
                    "name": endpoint_name,
                    "status": "error",
                    "error": str(e),
                    "is_default": endpoint_name == self._default_endpoint
                })
        
        return endpoints
    
    async def ping_endpoint(self, endpoint: Optional[str] = None) -> bool:
        """
        Ping a FastMCP endpoint to check connectivity.
        
        Args:
            endpoint: Endpoint name (uses default if not specified)
            
        Returns:
            True if endpoint is reachable, False otherwise
        """
        endpoint_name = endpoint or self._default_endpoint
        if not endpoint_name:
            raise SAFLAError("No endpoint specified and no default endpoint set")
        
        try:
            status = await self.adapter.get_endpoint_status(endpoint_name)
            return status.get("status") == "healthy"
            
        except Exception as e:
            logger.warning(f"Ping failed for {endpoint_name}: {e}")
            return False
    
    async def execute_workflow(
        self,
        workflow: List[Dict[str, Any]],
        endpoint: Optional[str] = None,
        stop_on_error: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a workflow of FastMCP operations.
        
        Args:
            workflow: List of operations to execute
            endpoint: Endpoint name (uses default if not specified)
            stop_on_error: Whether to stop execution on first error
            progress_callback: Optional progress callback (step, total, message)
            
        Returns:
            List of operation results
            
        Raises:
            SAFLAError: If workflow execution fails and stop_on_error is True
        """
        endpoint_name = endpoint or self._default_endpoint
        if not endpoint_name:
            raise SAFLAError("No endpoint specified and no default endpoint set")
        
        results = []
        total_steps = len(workflow)
        
        for i, operation in enumerate(workflow):
            if progress_callback:
                try:
                    progress_callback(i, total_steps, f"Executing step {i+1}")
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
            
            try:
                op_type = operation.get("type")
                
                if op_type == "call_tool":
                    result = await self.call_tool(
                        tool_name=operation["tool_name"],
                        arguments=operation.get("arguments", {}),
                        endpoint=endpoint_name,
                        timeout=operation.get("timeout")
                    )
                
                elif op_type == "read_resource":
                    result = await self.read_resource(
                        resource_uri=operation["resource_uri"],
                        endpoint=endpoint_name,
                        timeout=operation.get("timeout")
                    )
                
                else:
                    raise SAFLAError(f"Unknown operation type: {op_type}")
                
                results.append({
                    "step": i,
                    "operation": operation,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                error_result = {
                    "step": i,
                    "operation": operation,
                    "success": False,
                    "error": str(e)
                }
                results.append(error_result)
                
                if stop_on_error:
                    logger.error(f"Workflow stopped at step {i}: {e}")
                    break
                else:
                    logger.warning(f"Workflow step {i} failed, continuing: {e}")
        
        if progress_callback:
            try:
                progress_callback(total_steps, total_steps, "Workflow completed")
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
        
        return results
    
    @asynccontextmanager
    async def batch_context(self, endpoint: Optional[str] = None):
        """
        Context manager for batch operations on a FastMCP endpoint.
        
        Args:
            endpoint: Endpoint name (uses default if not specified)
            
        Yields:
            BatchOperations instance for executing multiple operations efficiently
        """
        endpoint_name = endpoint or self._default_endpoint
        if not endpoint_name:
            raise SAFLAError("No endpoint specified and no default endpoint set")
        
        batch_ops = BatchOperations(self, endpoint_name)
        try:
            yield batch_ops
        finally:
            # Cleanup if needed
            pass


class BatchOperations:
    """Helper class for batch operations within a FastMCP client context."""
    
    def __init__(self, client: FastMCPClient, endpoint_name: str):
        """
        Initialize batch operations.
        
        Args:
            client: FastMCP client instance
            endpoint_name: Name of the endpoint for batch operations
        """
        self.client = client
        self.endpoint_name = endpoint_name
        self.operations: List[Dict[str, Any]] = []
    
    def add_tool_call(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> 'BatchOperations':
        """
        Add a tool call to the batch.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            timeout: Operation timeout
            
        Returns:
            Self for method chaining
        """
        self.operations.append({
            "type": "call_tool",
            "tool_name": tool_name,
            "arguments": arguments or {},
            "timeout": timeout
        })
        return self
    
    def add_resource_read(
        self,
        resource_uri: str,
        timeout: Optional[float] = None
    ) -> 'BatchOperations':
        """
        Add a resource read to the batch.
        
        Args:
            resource_uri: URI of the resource to read
            timeout: Operation timeout
            
        Returns:
            Self for method chaining
        """
        self.operations.append({
            "type": "read_resource",
            "resource_uri": resource_uri,
            "timeout": timeout
        })
        return self
    
    async def execute(
        self,
        stop_on_error: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute all batched operations.
        
        Args:
            stop_on_error: Whether to stop on first error
            progress_callback: Optional progress callback
            
        Returns:
            List of operation results
        """
        return await self.client.execute_workflow(
            workflow=self.operations,
            endpoint=self.endpoint_name,
            stop_on_error=stop_on_error,
            progress_callback=progress_callback
        )
    
    def clear(self) -> 'BatchOperations':
        """
        Clear all batched operations.
        
        Returns:
            Self for method chaining
        """
        self.operations.clear()
        return self
    
    def count(self) -> int:
        """
        Get the number of batched operations.
        
        Returns:
            Number of operations in the batch
        """
        return len(self.operations)