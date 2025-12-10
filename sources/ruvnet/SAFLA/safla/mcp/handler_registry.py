"""Handler registry for modular MCP server architecture"""

import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class HandlerInfo:
    """Information about a registered handler"""
    name: str
    handler: Callable
    description: str
    category: str
    requires_auth: bool = True


class HandlerRegistry:
    """Registry for MCP request handlers"""
    
    def __init__(self):
        """Initialize handler registry"""
        self._handlers: Dict[str, HandlerInfo] = {}
        self._categories: Dict[str, List[str]] = {}
        logger.info("Handler registry initialized")
    
    def register(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        category: str = "general",
        requires_auth: bool = True
    ) -> None:
        """
        Register a handler.
        
        Args:
            name: Handler/tool name
            handler: Async handler function
            description: Handler description
            category: Handler category
            requires_auth: Whether handler requires authentication
        """
        if name in self._handlers:
            logger.warning(f"Overwriting existing handler: {name}")
        
        self._handlers[name] = HandlerInfo(
            name=name,
            handler=handler,
            description=description,
            category=category,
            requires_auth=requires_auth
        )
        
        # Update category index
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        
        logger.debug(f"Registered handler: {name} (category: {category})")
    
    def register_batch(self, handlers: List[Dict[str, Any]]) -> None:
        """Register multiple handlers at once"""
        for handler_info in handlers:
            self.register(**handler_info)
    
    async def dispatch(
        self,
        name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Dispatch request to appropriate handler.
        
        Args:
            name: Handler/tool name
            args: Handler arguments
            context: Request context (auth info, etc.)
            
        Returns:
            Handler result
            
        Raises:
            KeyError: If handler not found
        """
        if name not in self._handlers:
            raise KeyError(f"Handler not found: {name}")
        
        handler_info = self._handlers[name]
        
        # Check authentication if required
        if handler_info.requires_auth and context:
            if not context.get('authenticated', False):
                raise PermissionError(f"Authentication required for handler: {name}")
        
        # Execute handler
        handler = handler_info.handler
        
        # Support both sync and async handlers
        if asyncio.iscoroutinefunction(handler):
            result = await handler(args, context)
        else:
            result = handler(args, context)
        
        return result
    
    def get_handler(self, name: str) -> Optional[HandlerInfo]:
        """Get handler info by name"""
        return self._handlers.get(name)
    
    def list_handlers(self, category: Optional[str] = None) -> List[HandlerInfo]:
        """List all registered handlers, optionally filtered by category"""
        if category:
            handler_names = self._categories.get(category, [])
            return [self._handlers[name] for name in handler_names]
        return list(self._handlers.values())
    
    def list_categories(self) -> List[str]:
        """List all handler categories"""
        return list(self._categories.keys())
    
    def get_tools_definition(self) -> List[Dict[str, Any]]:
        """Get MCP tools definition for all handlers"""
        tools = []
        
        for handler_info in self._handlers.values():
            # Extract tool definition from handler
            tool_def = {
                "name": handler_info.name,
                "description": handler_info.description,
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Try to get schema from handler if available
            handler = handler_info.handler
            if hasattr(handler, '__mcp_schema__'):
                tool_def["inputSchema"] = handler.__mcp_schema__
            elif hasattr(handler, 'get_schema'):
                tool_def["inputSchema"] = handler.get_schema()
            
            tools.append(tool_def)
        
        return tools
    
    def clear(self) -> None:
        """Clear all registered handlers"""
        self._handlers.clear()
        self._categories.clear()
        logger.info("Handler registry cleared")


# Global registry instance
_global_registry = HandlerRegistry()


def get_registry() -> HandlerRegistry:
    """Get the global handler registry"""
    return _global_registry


def register_handler(
    name: str,
    description: str = "",
    category: str = "general",
    requires_auth: bool = True,
    input_schema: Optional[Dict[str, Any]] = None
):
    """
    Decorator to register a handler function.
    
    Example:
        @register_handler("my_tool", description="Does something", category="utils")
        async def my_tool_handler(args: Dict[str, Any], context: Dict[str, Any]):
            return {"result": "success"}
    """
    def decorator(func: Callable):
        # Attach schema if provided
        if input_schema:
            func.__mcp_schema__ = input_schema
        
        # Register the handler
        get_registry().register(
            name=name,
            handler=func,
            description=description,
            category=category,
            requires_auth=requires_auth
        )
        
        return func
    
    return decorator