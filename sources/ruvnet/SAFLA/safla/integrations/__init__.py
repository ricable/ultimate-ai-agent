"""
SAFLA Integrations Package

This package contains integrations with external services and protocols,
including FastMCP, API clients, and other third-party systems.
"""

from .fastmcp_adapter import FastMCPAdapter, FastMCPConfig, FastMCPEndpoint
from .fastmcp_client import FastMCPClient, BatchOperations
from .fastmcp_server import FastMCPServer, FastMCPServerBuilder, ToolDefinition, ResourceDefinition
from .fastmcp_helpers import (
    discover_fastmcp_endpoints,
    create_fastmcp_client_from_config,
    create_fastmcp_server_from_modules,
    check_fastmcp_endpoint,
    convert_safla_config_to_fastmcp,
    convert_fastmcp_config_to_safla,
    batch_call_tools,
    create_fastmcp_proxy_server,
    fastmcp_tool,
    fastmcp_resource
)

__all__ = [
    "FastMCPAdapter",
    "FastMCPConfig",
    "FastMCPEndpoint",
    "FastMCPClient",
    "BatchOperations",
    "FastMCPServer",
    "FastMCPServerBuilder",
    "ToolDefinition",
    "ResourceDefinition",
    "discover_fastmcp_endpoints",
    "create_fastmcp_client_from_config",
    "create_fastmcp_server_from_modules",
    "check_fastmcp_endpoint",
    "convert_safla_config_to_fastmcp",
    "convert_fastmcp_config_to_safla",
    "batch_call_tools",
    "create_fastmcp_proxy_server",
    "fastmcp_tool",
    "fastmcp_resource"
]