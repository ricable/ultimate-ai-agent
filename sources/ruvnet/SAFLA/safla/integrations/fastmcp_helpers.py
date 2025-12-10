"""
FastMCP Helper Functions for SAFLA

This module provides utility functions for common FastMCP operations,
including endpoint discovery, protocol handling, and data format conversion.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from urllib.parse import urlparse, urljoin
from dataclasses import asdict

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from .fastmcp_adapter import FastMCPAdapter, FastMCPConfig, FastMCPEndpoint
from .fastmcp_client import FastMCPClient
from .fastmcp_server import FastMCPServer
from ..exceptions import SAFLAError
from ..utils.logging import get_logger

logger = get_logger(__name__)


async def discover_fastmcp_endpoints(
    base_urls: List[str],
    timeout: float = 10.0,
    verify_ssl: bool = True
) -> List[Dict[str, Any]]:
    """
    Discover FastMCP endpoints from a list of base URLs.
    
    Args:
        base_urls: List of base URLs to check for FastMCP endpoints
        timeout: Request timeout in seconds
        verify_ssl: Whether to verify SSL certificates
        
    Returns:
        List of discovered endpoint information
        
    Raises:
        SAFLAError: If discovery fails
    """
    if not AIOHTTP_AVAILABLE:
        raise SAFLAError("aiohttp is required for endpoint discovery")
    
    discovered = []
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        connector=aiohttp.TCPConnector(verify_ssl=verify_ssl)
    ) as session:
        
        for base_url in base_urls:
            try:
                # Try common FastMCP discovery paths
                discovery_paths = [
                    "/.well-known/mcp",
                    "/mcp/info",
                    "/api/mcp",
                    "/fastmcp/info"
                ]
                
                for path in discovery_paths:
                    discovery_url = urljoin(base_url, path)
                    
                    try:
                        async with session.get(discovery_url) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                # Validate FastMCP endpoint info
                                if _is_valid_fastmcp_info(data):
                                    endpoint_info = {
                                        "base_url": base_url,
                                        "discovery_url": discovery_url,
                                        "info": data,
                                        "name": data.get("name", f"endpoint_{len(discovered)}"),
                                        "version": data.get("version", "unknown"),
                                        "capabilities": data.get("capabilities", [])
                                    }
                                    discovered.append(endpoint_info)
                                    logger.info(f"Discovered FastMCP endpoint: {base_url}")
                                    break
                    
                    except (aiohttp.ClientError, json.JSONDecodeError):
                        continue
                
            except Exception as e:
                logger.warning(f"Failed to discover endpoint at {base_url}: {e}")
                continue
    
    logger.info(f"Discovered {len(discovered)} FastMCP endpoints")
    return discovered


def _is_valid_fastmcp_info(data: Dict[str, Any]) -> bool:
    """
    Validate if the response contains valid FastMCP endpoint information.
    
    Args:
        data: Response data to validate
        
    Returns:
        True if valid FastMCP info, False otherwise
    """
    required_fields = ["name", "protocol"]
    return (
        isinstance(data, dict) and
        all(field in data for field in required_fields) and
        data.get("protocol") in ["mcp", "fastmcp"]
    )


async def create_fastmcp_client_from_config(
    config_path: str,
    endpoint_name: Optional[str] = None
) -> FastMCPClient:
    """
    Create a FastMCP client from a configuration file.
    
    Args:
        config_path: Path to the configuration file
        endpoint_name: Name of the endpoint to set as default
        
    Returns:
        Configured FastMCP client
        
    Raises:
        SAFLAError: If configuration is invalid or client creation fails
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create FastMCP config
        config = FastMCPConfig(**config_data)
        
        # Create adapter
        adapter = FastMCPAdapter(config)
        await adapter.start()
        
        # Create client
        client = FastMCPClient(adapter)
        
        # Set default endpoint if specified
        if endpoint_name:
            client.set_default_endpoint(endpoint_name)
        elif config.endpoints:
            # Use first endpoint as default
            first_endpoint = next(iter(config.endpoints.keys()))
            client.set_default_endpoint(first_endpoint)
        
        logger.info(f"Created FastMCP client from config: {config_path}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create FastMCP client from config: {e}")
        raise SAFLAError(f"Failed to create client: {str(e)}")


def create_fastmcp_server_from_modules(
    name: str,
    modules: List[Any],
    description: Optional[str] = None,
    version: str = "1.0.0"
) -> FastMCPServer:
    """
    Create a FastMCP server by automatically registering tools and resources
    from Python modules.
    
    Args:
        name: Server name
        modules: List of modules to scan for tools and resources
        description: Server description
        version: Server version
        
    Returns:
        Configured FastMCP server
        
    Raises:
        SAFLAError: If server creation fails
    """
    try:
        server = FastMCPServer(name=name, description=description, version=version)
        
        for module in modules:
            _register_module_tools_and_resources(server, module)
        
        logger.info(f"Created FastMCP server '{name}' with {len(server.tools)} tools and {len(server.resources)} resources")
        return server
        
    except Exception as e:
        logger.error(f"Failed to create FastMCP server from modules: {e}")
        raise SAFLAError(f"Failed to create server: {str(e)}")


def _register_module_tools_and_resources(server: FastMCPServer, module: Any) -> None:
    """
    Register tools and resources from a module.
    
    Args:
        server: FastMCP server instance
        module: Module to scan
    """
    import inspect
    
    for name, obj in inspect.getmembers(module):
        # Skip private members
        if name.startswith('_'):
            continue
        
        # Check for tool markers
        if hasattr(obj, '_fastmcp_tool'):
            tool_config = obj._fastmcp_tool
            server.register_tool(
                name=tool_config.get('name', name),
                description=tool_config.get('description', obj.__doc__ or f"Tool: {name}"),
                handler=obj,
                parameters=tool_config.get('parameters'),
                return_type=tool_config.get('return_type'),
                tags=tool_config.get('tags')
            )
        
        # Check for resource markers
        elif hasattr(obj, '_fastmcp_resource'):
            resource_config = obj._fastmcp_resource
            server.register_resource(
                uri=resource_config.get('uri', f"resource://{name}"),
                name=resource_config.get('name', name),
                description=resource_config.get('description', obj.__doc__ or f"Resource: {name}"),
                handler=obj,
                mime_type=resource_config.get('mime_type'),
                tags=resource_config.get('tags')
            )
        
        # Auto-register callable functions as tools
        elif callable(obj) and not inspect.isclass(obj):
            # Only register if it has a docstring or specific naming pattern
            if obj.__doc__ or name.startswith(('tool_', 'fastmcp_')):
                server.register_tool(
                    name=name,
                    description=obj.__doc__ or f"Auto-registered tool: {name}",
                    handler=obj
                )


def fastmcp_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    return_type: Optional[type] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator to mark functions as FastMCP tools for auto-registration.
    
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
        func._fastmcp_tool = {
            'name': name,
            'description': description,
            'parameters': parameters,
            'return_type': return_type,
            'tags': tags
        }
        return func
    
    return decorator


def fastmcp_resource(
    uri: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    mime_type: Optional[str] = None,
    tags: Optional[List[str]] = None
):
    """
    Decorator to mark functions as FastMCP resources for auto-registration.
    
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
        func._fastmcp_resource = {
            'uri': uri,
            'name': name,
            'description': description,
            'mime_type': mime_type,
            'tags': tags
        }
        return func
    
    return decorator


async def check_fastmcp_endpoint(
    endpoint_url: str,
    auth_token: Optional[str] = None,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Test connectivity and capabilities of a FastMCP endpoint.
    
    Args:
        endpoint_url: URL of the FastMCP endpoint
        auth_token: Optional authentication token
        timeout: Request timeout in seconds
        
    Returns:
        Test results including connectivity, capabilities, and performance
        
    Raises:
        SAFLAError: If testing fails
    """
    if not AIOHTTP_AVAILABLE:
        raise SAFLAError("aiohttp is required for endpoint testing")
    
    results = {
        "endpoint_url": endpoint_url,
        "timestamp": asyncio.get_event_loop().time(),
        "connectivity": False,
        "capabilities": {},
        "performance": {},
        "errors": []
    }
    
    try:
        # Create temporary endpoint configuration
        endpoint = FastMCPEndpoint(
            url=endpoint_url,
            auth_type="bearer" if auth_token else "none",
            auth_token=auth_token
        )
        
        config = FastMCPConfig(
            endpoints={"test": endpoint},
            default_timeout=timeout
        )
        
        # Create temporary adapter
        adapter = FastMCPAdapter(config)
        
        # Test connectivity
        start_time = asyncio.get_event_loop().time()
        await adapter.start()
        
        try:
            # Test basic connectivity
            status = await adapter.get_endpoint_status("test")
            results["connectivity"] = status.get("status") == "healthy"
            
            if results["connectivity"]:
                # Test capabilities
                try:
                    tools = await adapter.list_tools("test")
                    resources = await adapter.list_resources("test")
                    
                    results["capabilities"] = {
                        "tools": len(tools),
                        "resources": len(resources),
                        "tool_names": [tool.get("name") for tool in tools],
                        "resource_uris": [res.get("uri") for res in resources]
                    }
                    
                except Exception as e:
                    results["errors"].append(f"Capabilities test failed: {str(e)}")
                
                # Measure performance
                end_time = asyncio.get_event_loop().time()
                results["performance"] = {
                    "connection_time": end_time - start_time,
                    "response_time": end_time - start_time
                }
        
        finally:
            await adapter.shutdown()
    
    except Exception as e:
        results["errors"].append(f"Connectivity test failed: {str(e)}")
    
    logger.info(f"FastMCP endpoint test completed: {endpoint_url}")
    return results


def convert_safla_config_to_fastmcp(
    safla_config: Dict[str, Any]
) -> FastMCPConfig:
    """
    Convert SAFLA configuration format to FastMCP configuration.
    Also supports converting MCP server configurations to FastMCP format.
    
    Args:
        safla_config: SAFLA configuration dictionary or MCP server configuration
        
    Returns:
        FastMCP configuration object
        
    Raises:
        SAFLAError: If conversion fails
    """
    try:
        endpoints = {}
        
        # Check if this is an MCP server configuration format
        if "servers" in safla_config:
            # Convert MCP server format to FastMCP endpoints
            for name, server_config in safla_config["servers"].items():
                endpoint_config = {}
                
                # Handle stdio servers
                if "command" in server_config:
                    endpoint_config["transport_type"] = "stdio"
                    endpoint_config["auth_type"] = "none"
                    endpoint_config["command"] = server_config["command"]
                    if "args" in server_config:
                        endpoint_config["args"] = server_config["args"]
                    if "env" in server_config:
                        endpoint_config["env"] = server_config["env"]
                
                # Handle HTTP/SSE servers
                elif "url" in server_config:
                    endpoint_config["url"] = server_config["url"]
                    endpoint_config["transport_type"] = server_config.get("transport", "sse")
                    
                    # Handle authentication
                    if "auth" in server_config:
                        auth_config = server_config["auth"]
                        auth_type = auth_config.get("type", "none")
                        endpoint_config["auth_type"] = auth_type
                        
                        if auth_type == "bearer" and "token" in auth_config:
                            endpoint_config["auth_token"] = auth_config["token"]
                        elif auth_type == "api_key" and "key" in auth_config:
                            endpoint_config["api_key"] = auth_config["key"]
                    else:
                        endpoint_config["auth_type"] = "none"
                
                endpoints[name] = endpoint_config
        
        else:
            # Extract FastMCP-specific configuration from SAFLA format
            fastmcp_section = safla_config.get("fastmcp", {})
            
            # Convert endpoints - FastMCPConfig expects Dict[str, Dict[str, Any]]
            for name, endpoint_config in fastmcp_section.get("endpoints", {}).items():
                # Convert bearer_token to auth_token if present
                converted_config = dict(endpoint_config)
                if "bearer_token" in converted_config:
                    converted_config["auth_token"] = converted_config.pop("bearer_token")
                endpoints[name] = converted_config
        
        # Create FastMCP config with only valid fields
        fastmcp_section = safla_config.get("fastmcp", {})
        config = FastMCPConfig(
            endpoints=endpoints,
            default_timeout=fastmcp_section.get("default_timeout", 30.0),
            max_retries=fastmcp_section.get("max_retries", 3),
            connection_pool_size=fastmcp_section.get("connection_pool_size", 10),
            health_check_interval=fastmcp_section.get("health_check_interval", 60.0)
        )
        
        logger.info("Converted SAFLA config to FastMCP config")
        return config
        
    except Exception as e:
        logger.error(f"Failed to convert SAFLA config to FastMCP: {e}")
        raise SAFLAError(f"Config conversion failed: {str(e)}")


def convert_fastmcp_config_to_safla(
    fastmcp_config: FastMCPConfig
) -> Dict[str, Any]:
    """
    Convert FastMCP configuration to SAFLA configuration format.
    
    Args:
        fastmcp_config: FastMCP configuration object
        
    Returns:
        SAFLA configuration dictionary
        
    Raises:
        SAFLAError: If conversion fails
    """
    try:
        # Convert endpoints
        endpoints = {}
        for name, endpoint in fastmcp_config.endpoints.items():
            endpoints[name] = asdict(endpoint)
        
        # Create SAFLA config section
        safla_config = {
            "fastmcp": {
                "endpoints": endpoints,
                "default_timeout": fastmcp_config.default_timeout,
                "max_retries": fastmcp_config.max_retries,
                "retry_delay": fastmcp_config.retry_delay,
                "connection_pool_size": fastmcp_config.connection_pool_size,
                "enable_health_checks": fastmcp_config.enable_health_checks,
                "health_check_interval": fastmcp_config.health_check_interval
            }
        }
        
        logger.info("Converted FastMCP config to SAFLA config")
        return safla_config
        
    except Exception as e:
        logger.error(f"Failed to convert FastMCP config to SAFLA: {e}")
        raise SAFLAError(f"Config conversion failed: {str(e)}")


async def batch_call_tools(
    client: FastMCPClient,
    tool_calls: List[Dict[str, Any]],
    endpoint: Optional[str] = None,
    max_concurrent: int = 5,
    stop_on_error: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> List[Dict[str, Any]]:
    """
    Execute multiple tool calls concurrently with controlled concurrency.
    
    Args:
        client: FastMCP client instance
        tool_calls: List of tool call configurations
        endpoint: Endpoint name (uses client default if not specified)
        max_concurrent: Maximum number of concurrent calls
        stop_on_error: Whether to stop on first error
        progress_callback: Optional progress callback (completed, total)
        
    Returns:
        List of tool call results
        
    Raises:
        SAFLAError: If batch execution fails
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    completed = 0
    total = len(tool_calls)
    
    async def execute_tool_call(call_config: Dict[str, Any], index: int) -> Dict[str, Any]:
        nonlocal completed
        
        async with semaphore:
            try:
                result = await client.call_tool(
                    tool_name=call_config["tool_name"],
                    arguments=call_config.get("arguments", {}),
                    endpoint=endpoint,
                    timeout=call_config.get("timeout")
                )
                
                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, total)
                    except Exception:
                        pass
                
                return {
                    "index": index,
                    "success": True,
                    "result": result,
                    "tool_name": call_config["tool_name"]
                }
                
            except Exception as e:
                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, total)
                    except Exception:
                        pass
                
                error_result = {
                    "index": index,
                    "success": False,
                    "error": str(e),
                    "tool_name": call_config["tool_name"]
                }
                
                if stop_on_error:
                    raise SAFLAError(f"Tool call failed: {str(e)}")
                
                return error_result
    
    try:
        # Execute all tool calls concurrently
        tasks = [
            execute_tool_call(call_config, i)
            for i, call_config in enumerate(tool_calls)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=not stop_on_error)
        
        # Handle exceptions if stop_on_error is False
        if not stop_on_error:
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "index": i,
                        "success": False,
                        "error": str(result),
                        "tool_name": tool_calls[i]["tool_name"]
                    })
                else:
                    processed_results.append(result)
            results = processed_results
        
        logger.info(f"Batch tool execution completed: {len(results)} calls")
        return results
        
    except Exception as e:
        logger.error(f"Batch tool execution failed: {e}")
        raise SAFLAError(f"Batch execution failed: {str(e)}")


def create_fastmcp_proxy_server(
    name: str,
    target_endpoints: List[str],
    load_balancing: str = "round_robin",
    health_check_enabled: bool = True
) -> FastMCPServer:
    """
    Create a FastMCP proxy server that forwards requests to multiple target endpoints.
    
    Args:
        name: Proxy server name
        target_endpoints: List of target endpoint URLs
        load_balancing: Load balancing strategy ("round_robin", "random", "least_connections")
        health_check_enabled: Whether to enable health checks for targets
        
    Returns:
        Configured FastMCP proxy server
        
    Raises:
        SAFLAError: If proxy server creation fails
    """
    try:
        server = FastMCPServer(name=name, description=f"FastMCP Proxy Server: {name}")
        
        # Proxy state
        proxy_state = {
            "targets": target_endpoints,
            "current_index": 0,
            "load_balancing": load_balancing,
            "health_check_enabled": health_check_enabled,
            "healthy_targets": set(target_endpoints)
        }
        
        @server.tool(
            name="proxy_call_tool",
            description="Proxy tool call to target endpoints"
        )
        async def proxy_call_tool(tool_name: str, arguments: Dict[str, Any] = None):
            target = _select_target(proxy_state)
            if not target:
                raise Exception("No healthy targets available")
            
            # Forward the call (implementation would depend on actual FastMCP client)
            # This is a simplified example
            return {"proxied_to": target, "tool_name": tool_name, "arguments": arguments}
        
        @server.resource(
            uri="proxy://status",
            name="proxy_status",
            description="Proxy server status"
        )
        async def proxy_status():
            return {
                "targets": proxy_state["targets"],
                "healthy_targets": list(proxy_state["healthy_targets"]),
                "load_balancing": proxy_state["load_balancing"],
                "health_check_enabled": proxy_state["health_check_enabled"]
            }
        
        logger.info(f"Created FastMCP proxy server: {name}")
        return server
        
    except Exception as e:
        logger.error(f"Failed to create FastMCP proxy server: {e}")
        raise SAFLAError(f"Proxy server creation failed: {str(e)}")


def _select_target(proxy_state: Dict[str, Any]) -> Optional[str]:
    """
    Select a target endpoint based on load balancing strategy.
    
    Args:
        proxy_state: Proxy state dictionary
        
    Returns:
        Selected target URL or None if no healthy targets
    """
    healthy_targets = list(proxy_state["healthy_targets"])
    if not healthy_targets:
        return None
    
    strategy = proxy_state["load_balancing"]
    
    if strategy == "round_robin":
        target = healthy_targets[proxy_state["current_index"] % len(healthy_targets)]
        proxy_state["current_index"] += 1
        return target
    
    elif strategy == "random":
        import random
        return random.choice(healthy_targets)
    
    else:  # Default to first available
        return healthy_targets[0]