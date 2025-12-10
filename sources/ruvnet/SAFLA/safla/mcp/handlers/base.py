"""
Base handler class for SAFLA MCP handlers.

This module provides the foundation for all MCP tool handlers,
implementing common functionality and establishing the handler interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
from dataclasses import dataclass

from safla.exceptions import ValidationError


logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler_method: str


class BaseHandler(ABC):
    """
    Base class for all MCP handlers.
    
    Provides common functionality for tool registration, validation,
    and error handling. All domain-specific handlers should inherit
    from this class.
    """
    
    def __init__(self, config: Any, state_manager: Optional[Any] = None):
        """
        Initialize the handler.
        
        Args:
            config: SAFLA configuration object
            state_manager: Optional state manager for shared state
        """
        self.config = config
        self.state_manager = state_manager
        self._tools: Dict[str, ToolDefinition] = {}
        self._initialize_tools()
    
    @abstractmethod
    def _initialize_tools(self) -> None:
        """Initialize and register tools for this handler."""
        pass
    
    def register_tool(self, tool: ToolDefinition) -> None:
        """
        Register a tool with the handler.
        
        Args:
            tool: Tool definition to register
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools registered with this handler.
        
        Returns:
            List of tool definitions in MCP format
        """
        tools = []
        for tool in self._tools.values():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })
        return tools
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if this handler has a specific tool."""
        return tool_name in self._tools
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool call.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ValidationError: If tool not found or validation fails
        """
        if tool_name not in self._tools:
            raise ValidationError(f"Tool not found: {tool_name}")
        
        tool = self._tools[tool_name]
        
        # Validate arguments
        self._validate_arguments(tool, arguments)
        
        # Get handler method
        handler_method = getattr(self, tool.handler_method, None)
        if not handler_method:
            raise ValidationError(f"Handler method not found: {tool.handler_method}")
        
        try:
            # Execute the tool
            result = await handler_method(**arguments)
            return self._format_success_response(result)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return self._format_error_response(str(e))
    
    def _validate_arguments(self, tool: ToolDefinition, arguments: Dict[str, Any]) -> None:
        """
        Validate tool arguments against schema.
        
        Args:
            tool: Tool definition
            arguments: Arguments to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Extract required properties from schema
        schema = tool.input_schema
        required_props = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required arguments
        for prop in required_props:
            if prop not in arguments:
                raise ValidationError(f"Missing required argument: {prop}")
        
        # Validate types (basic validation)
        for prop, value in arguments.items():
            if prop in properties:
                expected_type = properties[prop].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    raise ValidationError(
                        f"Invalid type for {prop}: expected {expected_type}"
                    )
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        return True
    
    def _format_success_response(self, result: Any) -> Dict[str, Any]:
        """Format a successful tool response."""
        return {
            "success": True,
            "result": result
        }
    
    def _format_error_response(self, error: str) -> Dict[str, Any]:
        """Format an error response."""
        return {
            "success": False,
            "error": error
        }
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get state from the state manager.
        
        Args:
            key: State key
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        if self.state_manager:
            return self.state_manager.get(key, default)
        return default
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set state in the state manager.
        
        Args:
            key: State key
            value: State value
        """
        if self.state_manager:
            self.state_manager.set(key, value)