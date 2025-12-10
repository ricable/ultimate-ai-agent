"""
SAFLA Modular MCP Server Package.

This package provides a modular implementation of the Model Context Protocol (MCP)
server for SAFLA, with separated concerns for better maintainability.
"""

from .server import ModularMCPServer
from .handlers.base import BaseHandler, ToolDefinition
from .resources.base import BaseResource, ResourceDefinition
from .state.manager import StateManager

__all__ = [
    "ModularMCPServer",
    "BaseHandler",
    "ToolDefinition",
    "BaseResource",
    "ResourceDefinition",
    "StateManager"
]