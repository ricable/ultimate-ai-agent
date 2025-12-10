# UAP SDK Plugin System
"""
Plugin system for extending UAP capabilities.
Allows loading and managing plugins for custom functionality.
"""

import asyncio
import importlib
import importlib.util
import inspect
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Type, Union
import sys
import pkgutil

from .config import Configuration
from .exceptions import UAPException

logger = logging.getLogger(__name__)


class UAPPlugin(ABC):
    """Abstract base class for UAP plugins."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.is_enabled = False
        self.dependencies: List[str] = []
        self.config: Optional[Configuration] = None
        
    @abstractmethod
    async def initialize(self, config: Configuration) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.is_enabled,
            "dependencies": self.dependencies,
            "class": self.__class__.__name__,
            "module": self.__class__.__module__
        }
    
    async def enable(self, config: Configuration) -> None:
        """Enable the plugin."""
        if not self.is_enabled:
            self.config = config
            await self.initialize(config)
            self.is_enabled = True
            logger.info(f"Plugin {self.name} enabled")
    
    async def disable(self) -> None:
        """Disable the plugin."""
        if self.is_enabled:
            await self.cleanup()
            self.is_enabled = False
            self.config = None
            logger.info(f"Plugin {self.name} disabled")


class AgentPlugin(UAPPlugin):
    """Plugin for extending agent functionality."""
    
    @abstractmethod
    async def process_message(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a message through the plugin.
        
        Returns None if the plugin doesn't handle this message,
        or a response dict if it does.
        """
        pass
    
    def should_handle_message(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if this plugin should handle the message."""
        return False


class MiddlewarePlugin(UAPPlugin):
    """Plugin for message middleware."""
    
    @abstractmethod
    async def process_request(self, message: str, context: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Process incoming request."""
        pass
    
    @abstractmethod
    async def process_response(self, response: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing response."""
        pass


class ToolPlugin(UAPPlugin):
    """Plugin for adding tools/functions to agents."""
    
    @abstractmethod
    def get_tools(self) -> Dict[str, Callable]:
        """Get tools provided by this plugin."""
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool function."""
        pass


class PluginRegistry:
    """Registry for managing plugin types and discovery."""
    
    def __init__(self):
        self.plugin_types: Dict[str, Type[UAPPlugin]] = {
            "agent": AgentPlugin,
            "middleware": MiddlewarePlugin,
            "tool": ToolPlugin
        }
        self.discovered_plugins: Dict[str, Dict[str, Any]] = {}
    
    def register_plugin_type(self, name: str, plugin_class: Type[UAPPlugin]) -> None:
        """Register a new plugin type."""
        self.plugin_types[name] = plugin_class
    
    def discover_plugins(self, search_paths: List[Union[str, Path]]) -> Dict[str, Dict[str, Any]]:
        """Discover plugins in the given search paths."""
        plugins = {}
        
        for search_path in search_paths:
            path = Path(search_path)
            if path.is_file() and path.suffix == '.py':
                # Single plugin file
                plugin_info = self._load_plugin_from_file(path)
                if plugin_info:
                    plugins[plugin_info['name']] = plugin_info
            elif path.is_dir():
                # Plugin directory
                plugins.update(self._discover_plugins_in_directory(path))
        
        self.discovered_plugins.update(plugins)
        return plugins
    
    def _load_plugin_from_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load plugin information from a Python file."""
        try:
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if not spec or not spec.loader:
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, UAPPlugin) and 
                    obj != UAPPlugin and
                    not inspect.isabstract(obj)):
                    plugin_classes.append(obj)
            
            if plugin_classes:
                # Use the first plugin class found
                plugin_class = plugin_classes[0]
                return {
                    'name': getattr(plugin_class, 'PLUGIN_NAME', plugin_class.__name__),
                    'class': plugin_class,
                    'module': module,
                    'file_path': str(file_path),
                    'version': getattr(plugin_class, 'PLUGIN_VERSION', '1.0.0'),
                    'description': getattr(plugin_class, 'PLUGIN_DESCRIPTION', ''),
                    'type': self._get_plugin_type(plugin_class)
                }
        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return None
    
    def _discover_plugins_in_directory(self, directory: Path) -> Dict[str, Dict[str, Any]]:
        """Discover plugins in a directory."""
        plugins = {}
        
        # Look for Python files
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            plugin_info = self._load_plugin_from_file(py_file)
            if plugin_info:
                plugins[plugin_info['name']] = plugin_info
        
        # Look for plugin packages
        for pkg_dir in directory.iterdir():
            if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                try:
                    # Add to path temporarily
                    if str(directory) not in sys.path:
                        sys.path.insert(0, str(directory))
                    
                    module = importlib.import_module(pkg_dir.name)
                    
                    # Look for plugin classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, UAPPlugin) and 
                            obj != UAPPlugin and
                            not inspect.isabstract(obj)):
                            
                            plugin_info = {
                                'name': getattr(obj, 'PLUGIN_NAME', obj.__name__),
                                'class': obj,
                                'module': module,
                                'file_path': str(pkg_dir),
                                'version': getattr(obj, 'PLUGIN_VERSION', '1.0.0'),
                                'description': getattr(obj, 'PLUGIN_DESCRIPTION', ''),
                                'type': self._get_plugin_type(obj)
                            }
                            plugins[plugin_info['name']] = plugin_info
                            
                except Exception as e:
                    logger.error(f"Failed to load plugin package {pkg_dir}: {e}")
                finally:
                    # Remove from path
                    if str(directory) in sys.path:
                        sys.path.remove(str(directory))
        
        return plugins
    
    def _get_plugin_type(self, plugin_class: Type[UAPPlugin]) -> str:
        """Determine the type of a plugin class."""
        for type_name, type_class in self.plugin_types.items():
            if issubclass(plugin_class, type_class):
                return type_name
        return "unknown"


class PluginManager:
    """Main plugin manager for UAP SDK."""
    
    def __init__(self, config: Optional[Configuration] = None):
        self.config = config or Configuration()
        self.registry = PluginRegistry()
        self.loaded_plugins: Dict[str, UAPPlugin] = {}
        self.enabled_plugins: Dict[str, UAPPlugin] = {}
        self.plugin_order: List[str] = []
        
        # Plugin directories
        self.plugin_directories = [
            Path.cwd() / "plugins",
            Path.home() / ".uap" / "plugins",
            Path(__file__).parent / "plugins"
        ]
    
    def add_plugin_directory(self, directory: Union[str, Path]) -> None:
        """Add a directory to search for plugins."""
        path = Path(directory)
        if path not in self.plugin_directories:
            self.plugin_directories.append(path)
    
    def discover_plugins(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available plugins."""
        return self.registry.discover_plugins(self.plugin_directories)
    
    def load_plugin(self, plugin_name: str) -> Optional[UAPPlugin]:
        """Load a plugin by name."""
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name]
        
        # Discover plugins if not already done
        if plugin_name not in self.registry.discovered_plugins:
            self.discover_plugins()
        
        plugin_info = self.registry.discovered_plugins.get(plugin_name)
        if not plugin_info:
            logger.error(f"Plugin '{plugin_name}' not found")
            return None
        
        try:
            plugin_class = plugin_info['class']
            plugin = plugin_class(plugin_name, plugin_info.get('version', '1.0.0'))
            self.loaded_plugins[plugin_name] = plugin
            
            logger.info(f"Plugin '{plugin_name}' loaded successfully")
            return plugin
            
        except Exception as e:
            logger.error(f"Failed to load plugin '{plugin_name}': {e}")
            return None
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        plugin = self.load_plugin(plugin_name)
        if not plugin:
            return False
        
        try:
            await plugin.enable(self.config)
            self.enabled_plugins[plugin_name] = plugin
            
            if plugin_name not in self.plugin_order:
                self.plugin_order.append(plugin_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable plugin '{plugin_name}': {e}")
            return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name not in self.enabled_plugins:
            return False
        
        try:
            plugin = self.enabled_plugins[plugin_name]
            await plugin.disable()
            del self.enabled_plugins[plugin_name]
            
            if plugin_name in self.plugin_order:
                self.plugin_order.remove(plugin_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable plugin '{plugin_name}': {e}")
            return False
    
    def get_enabled_plugins(self, plugin_type: Optional[str] = None) -> List[UAPPlugin]:
        """Get enabled plugins, optionally filtered by type."""
        plugins = list(self.enabled_plugins.values())
        
        if plugin_type:
            type_class = self.registry.plugin_types.get(plugin_type)
            if type_class:
                plugins = [p for p in plugins if isinstance(p, type_class)]
        
        # Sort by plugin order
        ordered_plugins = []
        for plugin_name in self.plugin_order:
            if plugin_name in self.enabled_plugins:
                plugin = self.enabled_plugins[plugin_name]
                if not plugin_type or isinstance(plugin, self.registry.plugin_types.get(plugin_type, UAPPlugin)):
                    ordered_plugins.append(plugin)
        
        return ordered_plugins
    
    async def process_message_through_plugins(self, agent_id: str, message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a message through agent plugins."""
        agent_plugins = self.get_enabled_plugins("agent")
        
        for plugin in agent_plugins:
            if isinstance(plugin, AgentPlugin) and plugin.should_handle_message(message, context):
                try:
                    result = await plugin.process_message(agent_id, message, context)
                    if result:
                        return result
                except Exception as e:
                    logger.error(f"Plugin '{plugin.name}' error: {e}")
                    continue
        
        return None
    
    async def apply_middleware_plugins(self, message: str, context: Dict[str, Any], stage: str = "request") -> tuple:
        """Apply middleware plugins to a message."""
        middleware_plugins = self.get_enabled_plugins("middleware")
        
        if stage == "request":
            for plugin in middleware_plugins:
                if isinstance(plugin, MiddlewarePlugin):
                    try:
                        message, context = await plugin.process_request(message, context)
                    except Exception as e:
                        logger.error(f"Middleware plugin '{plugin.name}' error: {e}")
                        continue
            return message, context
        
        elif stage == "response" and isinstance(message, dict):
            response = message
            for plugin in reversed(middleware_plugins):  # Reverse order for response
                if isinstance(plugin, MiddlewarePlugin):
                    try:
                        response = await plugin.process_response(response, context)
                    except Exception as e:
                        logger.error(f"Middleware plugin '{plugin.name}' error: {e}")
                        continue
            return response, context
        
        return message, context
    
    def get_tools_from_plugins(self) -> Dict[str, Callable]:
        """Get all tools from tool plugins."""
        tools = {}
        tool_plugins = self.get_enabled_plugins("tool")
        
        for plugin in tool_plugins:
            if isinstance(plugin, ToolPlugin):
                try:
                    plugin_tools = plugin.get_tools()
                    for tool_name, tool_func in plugin_tools.items():
                        # Prefix with plugin name to avoid conflicts
                        prefixed_name = f"{plugin.name}_{tool_name}"
                        tools[prefixed_name] = tool_func
                except Exception as e:
                    logger.error(f"Failed to get tools from plugin '{plugin.name}': {e}")
        
        return tools
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool from a plugin."""
        if "_" not in tool_name:
            raise UAPException(f"Invalid tool name format: {tool_name}")
        
        plugin_name, actual_tool_name = tool_name.split("_", 1)
        
        if plugin_name not in self.enabled_plugins:
            raise UAPException(f"Plugin '{plugin_name}' not enabled")
        
        plugin = self.enabled_plugins[plugin_name]
        if not isinstance(plugin, ToolPlugin):
            raise UAPException(f"Plugin '{plugin_name}' is not a tool plugin")
        
        return await plugin.execute_tool(actual_tool_name, **kwargs)
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get information about all plugins."""
        return {
            "discovered": {name: info for name, info in self.registry.discovered_plugins.items()},
            "loaded": {name: plugin.get_info() for name, plugin in self.loaded_plugins.items()},
            "enabled": {name: plugin.get_info() for name, plugin in self.enabled_plugins.items()},
            "plugin_order": self.plugin_order.copy()
        }
    
    async def cleanup(self) -> None:
        """Clean up all enabled plugins."""
        for plugin_name in list(self.enabled_plugins.keys()):
            await self.disable_plugin(plugin_name)