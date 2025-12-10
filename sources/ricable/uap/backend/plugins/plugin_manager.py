# File: backend/plugins/plugin_manager.py
"""
Plugin Manager - Dynamic loading and lifecycle management for plugins.
"""

import asyncio
import importlib.util
import sys
import os
import json
import hashlib
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from .plugin_base import (
    PluginBase, PluginManifest, PluginConfig, PluginContext,
    PluginResponse, PluginEvent, PluginStatus, PluginError
)
from .sandbox import PluginSandbox
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel


class PluginManager:
    """
    Central manager for all UAP plugins.
    
    Handles plugin discovery, loading, lifecycle management, and execution.
    """
    
    def __init__(self, plugins_directory: str = "plugins", auth_service=None):
        self.plugins_directory = Path(plugins_directory)
        self.auth_service = auth_service
        
        # Plugin storage
        self.manifests: Dict[str, PluginManifest] = {}
        self.plugins: Dict[str, PluginBase] = {}
        self.configs: Dict[str, PluginConfig] = {}
        self.installed_plugins: Dict[str, Path] = {}
        
        # Runtime management
        self.sandbox = PluginSandbox()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._is_initialized = False
        self._plugin_modules: Dict[str, Any] = {}
        
        # Ensure plugins directory exists
        self.plugins_directory.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.plugins_directory / "installed").mkdir(exist_ok=True)
        (self.plugins_directory / "marketplace").mkdir(exist_ok=True)
        (self.plugins_directory / "cache").mkdir(exist_ok=True)
    
    async def initialize(self) -> bool:
        """
        Initialize the plugin manager and discover installed plugins.
        
        Returns:
            True if initialization successful
        """
        try:
            uap_logger.log_event(
                LogLevel.INFO,
                "Initializing Plugin Manager",
                EventType.SYSTEM,
                {"plugins_directory": str(self.plugins_directory)},
                "plugin_manager"
            )
            
            # Initialize sandbox
            await self.sandbox.initialize()
            
            # Discover and load installed plugins
            await self._discover_plugins()
            
            # Initialize all active plugins
            initialization_tasks = []
            for plugin_id, plugin in self.plugins.items():
                config = self.configs.get(plugin_id)
                if config and config.enabled:
                    task = asyncio.create_task(
                        self._safe_initialize_plugin(plugin_id, plugin)
                    )
                    initialization_tasks.append(task)
            
            if initialization_tasks:
                results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
                
                # Log results
                successful = sum(1 for r in results if r is True)
                failed = len(results) - successful
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Plugin initialization complete: {successful} successful, {failed} failed",
                    EventType.SYSTEM,
                    {
                        "successful": successful,
                        "failed": failed,
                        "total": len(results)
                    },
                    "plugin_manager"
                )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to initialize Plugin Manager: {str(e)}",
                EventType.ERROR,
                {"error": str(e)},
                "plugin_manager"
            )
            return False
    
    async def _discover_plugins(self):
        """Discover and load plugin manifests from the plugins directory."""
        installed_dir = self.plugins_directory / "installed"
        
        for plugin_path in installed_dir.iterdir():
            if plugin_path.is_dir():
                manifest_path = plugin_path / "manifest.json"
                config_path = plugin_path / "config.json"
                
                if manifest_path.exists():
                    try:
                        # Load manifest
                        with open(manifest_path, 'r') as f:
                            manifest_data = json.load(f)
                        manifest = PluginManifest(**manifest_data)
                        
                        # Load or create config
                        if config_path.exists():
                            with open(config_path, 'r') as f:
                                config_data = json.load(f)
                            config = PluginConfig(**config_data)
                        else:
                            config = PluginConfig(
                                plugin_id=manifest.plugin_id,
                                name=manifest.name,
                                config=manifest.default_config.copy()
                            )
                            # Save default config
                            await self._save_plugin_config(config)
                        
                        # Store plugin info
                        self.manifests[manifest.plugin_id] = manifest
                        self.configs[manifest.plugin_id] = config
                        self.installed_plugins[manifest.plugin_id] = plugin_path
                        
                        # Load plugin module if enabled
                        if config.enabled:
                            await self._load_plugin_module(manifest, plugin_path)
                        
                        uap_logger.log_event(
                            LogLevel.INFO,
                            f"Discovered plugin: {manifest.name} v{manifest.version}",
                            EventType.PLUGIN,
                            {
                                "plugin_id": manifest.plugin_id,
                                "plugin_name": manifest.name,
                                "version": manifest.version,
                                "enabled": config.enabled
                            },
                            "plugin_manager"
                        )
                        
                    except Exception as e:
                        uap_logger.log_event(
                            LogLevel.ERROR,
                            f"Failed to load plugin from {plugin_path}: {str(e)}",
                            EventType.ERROR,
                            {"plugin_path": str(plugin_path), "error": str(e)},
                            "plugin_manager"
                        )
    
    async def _load_plugin_module(self, manifest: PluginManifest, plugin_path: Path):
        """Load plugin module dynamically."""
        try:
            # Add plugin path to sys.path temporarily
            sys.path.insert(0, str(plugin_path))
            
            try:
                # Import the main module
                module_path = plugin_path / f"{manifest.main_module}.py"
                if not module_path.exists():
                    raise PluginError(f"Main module {manifest.main_module}.py not found", manifest.plugin_id)
                
                spec = importlib.util.spec_from_file_location(
                    f"plugin_{manifest.plugin_id}_{manifest.main_module}",
                    module_path
                )
                
                if spec is None or spec.loader is None:
                    raise PluginError(f"Failed to load module spec for {manifest.main_module}", manifest.plugin_id)
                
                module = importlib.util.module_from_spec(spec)
                self._plugin_modules[manifest.plugin_id] = module
                spec.loader.exec_module(module)
                
                # Get the plugin class
                if not hasattr(module, manifest.main_class):
                    raise PluginError(f"Main class {manifest.main_class} not found in module", manifest.plugin_id)
                
                plugin_class = getattr(module, manifest.main_class)
                
                # Validate plugin class
                if not issubclass(plugin_class, PluginBase):
                    raise PluginError(f"Plugin class {manifest.main_class} must inherit from PluginBase", manifest.plugin_id)
                
                # Create plugin instance
                config = self.configs[manifest.plugin_id]
                plugin_instance = plugin_class(manifest, config)
                self.plugins[manifest.plugin_id] = plugin_instance
                
            finally:
                # Remove plugin path from sys.path
                if str(plugin_path) in sys.path:
                    sys.path.remove(str(plugin_path))
                
        except Exception as e:
            raise PluginError(f"Failed to load plugin module: {str(e)}", manifest.plugin_id)
    
    async def _safe_initialize_plugin(self, plugin_id: str, plugin: PluginBase) -> bool:
        """Safely initialize a single plugin with error handling."""
        try:
            context = PluginContext(
                plugin_id=plugin_id,
                instance_id=plugin.config.instance_id,
                permissions=plugin.config.permissions,
                resource_limits=plugin.config.resource_limits
            )
            
            # Initialize within sandbox
            response = await self.sandbox.execute_safe(
                plugin.initialize,
                context,
                resource_limits=plugin.config.resource_limits
            )
            
            if response.success:
                plugin.status = PluginStatus.ACTIVE
                plugin._startup_time = datetime.now(timezone.utc)
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Plugin {plugin_id} initialized successfully",
                    EventType.PLUGIN,
                    {"plugin_id": plugin_id, "status": "initialized"},
                    "plugin_manager"
                )
                return True
            else:
                plugin.status = PluginStatus.ERROR
                plugin._last_error = response.error
                
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Plugin {plugin_id} initialization failed: {response.error}",
                    EventType.ERROR,
                    {"plugin_id": plugin_id, "error": response.error},
                    "plugin_manager"
                )
                return False
                
        except Exception as e:
            plugin.status = PluginStatus.ERROR
            plugin._last_error = str(e)
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Exception during {plugin_id} initialization: {str(e)}",
                EventType.ERROR,
                {"plugin_id": plugin_id, "exception": str(e)},
                "plugin_manager"
            )
            return False
    
    async def install_plugin(self, plugin_package: bytes, manifest_override: Dict[str, Any] = None) -> PluginResponse:
        """Install a plugin from a package file."""
        try:
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                package_path = temp_path / "plugin.zip"
                
                # Write package to temporary file
                with open(package_path, 'wb') as f:
                    f.write(plugin_package)
                
                # Extract package
                extract_path = temp_path / "extracted"
                with zipfile.ZipFile(package_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                
                # Load and validate manifest
                manifest_path = extract_path / "manifest.json"
                if not manifest_path.exists():
                    raise PluginError("Plugin package missing manifest.json")
                
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                
                # Apply manifest overrides
                if manifest_override:
                    manifest_data.update(manifest_override)
                
                manifest = PluginManifest(**manifest_data)
                
                # Validate plugin
                await self._validate_plugin_package(extract_path, manifest)
                
                # Check if plugin already exists
                if manifest.plugin_id in self.manifests:
                    # Check version for update
                    existing_version = self.manifests[manifest.plugin_id].version
                    if manifest.version <= existing_version:
                        raise PluginError(f"Plugin version {manifest.version} is not newer than installed version {existing_version}")
                
                # Install plugin
                install_path = self.plugins_directory / "installed" / manifest.plugin_id
                if install_path.exists():
                    shutil.rmtree(install_path)
                
                shutil.copytree(extract_path, install_path)
                
                # Create default config
                config = PluginConfig(
                    plugin_id=manifest.plugin_id,
                    name=manifest.name,
                    config=manifest.default_config.copy()
                )
                
                # Save plugin data
                await self._save_plugin_config(config)
                
                # Store in memory
                self.manifests[manifest.plugin_id] = manifest
                self.configs[manifest.plugin_id] = config
                self.installed_plugins[manifest.plugin_id] = install_path
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Plugin installed: {manifest.name} v{manifest.version}",
                    EventType.PLUGIN,
                    {
                        "plugin_id": manifest.plugin_id,
                        "plugin_name": manifest.name,
                        "version": manifest.version
                    },
                    "plugin_manager"
                )
                
                return PluginResponse(
                    success=True,
                    data={
                        "plugin_id": manifest.plugin_id,
                        "name": manifest.name,
                        "version": manifest.version,
                        "requires_configuration": len(manifest.config_schema) > 0
                    }
                )
                
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Plugin installation failed: {str(e)}",
                error_code="install_failed"
            )
    
    async def _validate_plugin_package(self, package_path: Path, manifest: PluginManifest):
        """Validate plugin package structure and security."""
        # Check required files
        main_module_path = package_path / f"{manifest.main_module}.py"
        if not main_module_path.exists():
            raise PluginError(f"Main module {manifest.main_module}.py not found")
        
        # Validate manifest fields
        required_fields = ['name', 'version', 'description', 'author', 'main_module', 'main_class']
        for field in required_fields:
            if not getattr(manifest, field):
                raise PluginError(f"Required manifest field '{field}' is missing or empty")
        
        # Security validation
        await self._validate_plugin_security(package_path, manifest)
    
    async def _validate_plugin_security(self, package_path: Path, manifest: PluginManifest):
        """Validate plugin security and permissions."""
        # Check for suspicious file patterns
        suspicious_patterns = ['.so', '.dll', '.exe', '.bat', '.sh']
        for file_path in package_path.rglob('*'):
            if file_path.is_file():
                if any(file_path.name.endswith(pattern) for pattern in suspicious_patterns):
                    uap_logger.log_event(
                        LogLevel.WARNING,
                        f"Suspicious file found in plugin: {file_path.name}",
                        EventType.SECURITY,
                        {"plugin_id": manifest.plugin_id, "file": str(file_path)},
                        "plugin_manager"
                    )
        
        # Validate permissions
        dangerous_permissions = ['file_system_write', 'network_access', 'system_commands']
        for permission in manifest.permissions:
            if permission in dangerous_permissions:
                uap_logger.log_event(
                    LogLevel.WARNING,
                    f"Plugin requests dangerous permission: {permission}",
                    EventType.SECURITY,
                    {"plugin_id": manifest.plugin_id, "permission": permission},
                    "plugin_manager"
                )
    
    async def uninstall_plugin(self, plugin_id: str) -> PluginResponse:
        """Uninstall a plugin."""
        try:
            if plugin_id not in self.manifests:
                return PluginResponse(
                    success=False,
                    error=f"Plugin {plugin_id} not found",
                    error_code="plugin_not_found"
                )
            
            # Stop plugin if running
            if plugin_id in self.plugins:
                await self.stop_plugin(plugin_id)
            
            # Remove plugin files
            install_path = self.installed_plugins.get(plugin_id)
            if install_path and install_path.exists():
                shutil.rmtree(install_path)
            
            # Remove from memory
            manifest = self.manifests.pop(plugin_id, None)
            self.configs.pop(plugin_id, None)
            self.plugins.pop(plugin_id, None)
            self.installed_plugins.pop(plugin_id, None)
            self._plugin_modules.pop(plugin_id, None)
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Plugin uninstalled: {manifest.name if manifest else plugin_id}",
                EventType.PLUGIN,
                {"plugin_id": plugin_id},
                "plugin_manager"
            )
            
            return PluginResponse(
                success=True,
                data={"plugin_id": plugin_id, "status": "uninstalled"}
            )
            
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Plugin uninstallation failed: {str(e)}",
                error_code="uninstall_failed"
            )
    
    async def enable_plugin(self, plugin_id: str) -> PluginResponse:
        """Enable and start a plugin."""
        try:
            if plugin_id not in self.configs:
                return PluginResponse(
                    success=False,
                    error=f"Plugin {plugin_id} not found",
                    error_code="plugin_not_found"
                )
            
            config = self.configs[plugin_id]
            config.enabled = True
            config.updated_at = datetime.now(timezone.utc)
            
            # Save config
            await self._save_plugin_config(config)
            
            # Load and initialize plugin if not already loaded
            if plugin_id not in self.plugins:
                manifest = self.manifests[plugin_id]
                plugin_path = self.installed_plugins[plugin_id]
                await self._load_plugin_module(manifest, plugin_path)
            
            # Initialize plugin
            if plugin_id in self.plugins:
                plugin = self.plugins[plugin_id]
                await self._safe_initialize_plugin(plugin_id, plugin)
            
            return PluginResponse(
                success=True,
                data={"plugin_id": plugin_id, "status": "enabled"}
            )
            
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Failed to enable plugin: {str(e)}",
                error_code="enable_failed"
            )
    
    async def disable_plugin(self, plugin_id: str) -> PluginResponse:
        """Disable and stop a plugin."""
        try:
            if plugin_id not in self.configs:
                return PluginResponse(
                    success=False,
                    error=f"Plugin {plugin_id} not found",
                    error_code="plugin_not_found"
                )
            
            # Stop plugin
            await self.stop_plugin(plugin_id)
            
            # Update config
            config = self.configs[plugin_id]
            config.enabled = False
            config.updated_at = datetime.now(timezone.utc)
            
            # Save config
            await self._save_plugin_config(config)
            
            return PluginResponse(
                success=True,
                data={"plugin_id": plugin_id, "status": "disabled"}
            )
            
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Failed to disable plugin: {str(e)}",
                error_code="disable_failed"
            )
    
    async def stop_plugin(self, plugin_id: str) -> PluginResponse:
        """Stop a running plugin."""
        try:
            if plugin_id not in self.plugins:
                return PluginResponse(
                    success=True,
                    data={"plugin_id": plugin_id, "status": "not_running"}
                )
            
            plugin = self.plugins[plugin_id]
            
            # Cleanup plugin
            response = await self.sandbox.execute_safe(
                plugin.cleanup,
                resource_limits=plugin.config.resource_limits
            )
            
            plugin.status = PluginStatus.INACTIVE
            plugin._startup_time = None
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Plugin stopped: {plugin.name}",
                EventType.PLUGIN,
                {"plugin_id": plugin_id, "status": "stopped"},
                "plugin_manager"
            )
            
            return PluginResponse(
                success=True,
                data={"plugin_id": plugin_id, "status": "stopped"}
            )
            
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Failed to stop plugin: {str(e)}",
                error_code="stop_failed"
            )
    
    async def execute_plugin_action(self, plugin_id: str, action: str, params: Dict[str, Any], 
                                  context: PluginContext) -> PluginResponse:
        """Execute an action on a specific plugin."""
        try:
            if plugin_id not in self.plugins:
                return PluginResponse(
                    success=False,
                    error=f"Plugin {plugin_id} not found or not active",
                    error_code="plugin_not_active"
                )
            
            plugin = self.plugins[plugin_id]
            
            # Validate permissions
            if not await plugin.validate_permissions(plugin.config.permissions, context):
                return PluginResponse(
                    success=False,
                    error="Insufficient permissions",
                    error_code="permission_denied"
                )
            
            # Execute action within sandbox
            response = await self.sandbox.execute_safe(
                plugin.execute,
                action,
                params,
                context,
                resource_limits=plugin.config.resource_limits
            )
            
            # Update metrics
            plugin._update_metrics(f"action_{action}_count", 
                                 plugin._metrics.get(f"action_{action}_count", 0) + 1)
            
            return response
            
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Plugin action execution failed: {str(e)}",
                error_code="execution_failed"
            )
    
    async def send_event_to_plugin(self, plugin_id: str, event: PluginEvent) -> PluginResponse:
        """Send an event to a specific plugin."""
        try:
            if plugin_id not in self.plugins:
                return PluginResponse(
                    success=False,
                    error=f"Plugin {plugin_id} not found or not active",
                    error_code="plugin_not_active"
                )
            
            plugin = self.plugins[plugin_id]
            
            # Send event within sandbox
            response = await self.sandbox.execute_safe(
                plugin.handle_event,
                event,
                resource_limits=plugin.config.resource_limits
            )
            
            return response
            
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Event sending failed: {str(e)}",
                error_code="event_failed"
            )
    
    async def broadcast_event(self, event: PluginEvent, target_plugins: List[str] = None) -> Dict[str, PluginResponse]:
        """Broadcast an event to multiple plugins."""
        results = {}
        
        target_list = target_plugins or list(self.plugins.keys())
        
        for plugin_id in target_list:
            if plugin_id in self.plugins:
                try:
                    response = await self.send_event_to_plugin(plugin_id, event)
                    results[plugin_id] = response
                except Exception as e:
                    results[plugin_id] = PluginResponse(
                        success=False,
                        error=str(e),
                        error_code="broadcast_failed"
                    )
        
        return results
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginBase]:
        """Get plugin instance by ID."""
        return self.plugins.get(plugin_id)
    
    def get_plugin_manifest(self, plugin_id: str) -> Optional[PluginManifest]:
        """Get plugin manifest by ID."""
        return self.manifests.get(plugin_id)
    
    def get_plugin_config(self, plugin_id: str) -> Optional[PluginConfig]:
        """Get plugin configuration by ID."""
        return self.configs.get(plugin_id)
    
    def list_plugins(self, status_filter: PluginStatus = None) -> List[Dict[str, Any]]:
        """List all plugins with their status."""
        plugins_list = []
        
        for plugin_id, manifest in self.manifests.items():
            config = self.configs.get(plugin_id)
            plugin = self.plugins.get(plugin_id)
            
            plugin_info = {
                "plugin_id": plugin_id,
                "name": manifest.name,
                "display_name": manifest.display_name,
                "version": manifest.version,
                "description": manifest.description,
                "author": manifest.author,
                "plugin_type": manifest.plugin_type,
                "category": manifest.category,
                "enabled": config.enabled if config else False,
                "status": plugin.status if plugin else PluginStatus.INACTIVE,
                "created_at": manifest.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat() if config else None
            }
            
            if status_filter is None or plugin_info["status"] == status_filter:
                plugins_list.append(plugin_info)
        
        return plugins_list
    
    async def update_plugin_config(self, plugin_id: str, config_updates: Dict[str, Any]) -> PluginResponse:
        """Update plugin configuration."""
        try:
            if plugin_id not in self.configs:
                return PluginResponse(
                    success=False,
                    error=f"Plugin {plugin_id} not found",
                    error_code="plugin_not_found"
                )
            
            config = self.configs[plugin_id]
            
            # Update configuration
            config.config.update(config_updates)
            config.updated_at = datetime.now(timezone.utc)
            
            # Save configuration
            await self._save_plugin_config(config)
            
            # If plugin is running, notify of config change
            if plugin_id in self.plugins:
                event = PluginEvent(
                    plugin_id=plugin_id,
                    event_type="config_updated",
                    source="plugin_manager",
                    data={"config_updates": config_updates}
                )
                await self.send_event_to_plugin(plugin_id, event)
            
            return PluginResponse(
                success=True,
                data={"plugin_id": plugin_id, "config": config.config}
            )
            
        except Exception as e:
            return PluginResponse(
                success=False,
                error=f"Failed to update plugin config: {str(e)}",
                error_code="config_update_failed"
            )
    
    async def _save_plugin_config(self, config: PluginConfig):
        """Save plugin configuration to disk."""
        install_path = self.installed_plugins.get(config.plugin_id)
        if install_path:
            config_path = install_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config.dict(), f, indent=2, default=str)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall plugin system status."""
        total_plugins = len(self.manifests)
        active_plugins = sum(1 for p in self.plugins.values() if p.status == PluginStatus.ACTIVE)
        error_plugins = sum(1 for p in self.plugins.values() if p.status == PluginStatus.ERROR)
        enabled_plugins = sum(1 for c in self.configs.values() if c.enabled)
        
        # Plugin types distribution
        plugin_types = {}
        for manifest in self.manifests.values():
            plugin_type = manifest.plugin_type
            plugin_types[plugin_type] = plugin_types.get(plugin_type, 0) + 1
        
        return {
            "initialized": self._is_initialized,
            "plugins": {
                "total": total_plugins,
                "enabled": enabled_plugins,
                "active": active_plugins,
                "error": error_plugins,
                "types": plugin_types
            },
            "plugins_directory": str(self.plugins_directory),
            "sandbox_active": self.sandbox.is_active,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def cleanup(self):
        """Clean up plugin manager resources."""
        try:
            # Stop all plugins
            cleanup_tasks = []
            for plugin_id in list(self.plugins.keys()):
                task = asyncio.create_task(self.stop_plugin(plugin_id))
                cleanup_tasks.append(task)
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Cleanup sandbox
            await self.sandbox.cleanup()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Plugin Manager cleanup complete",
                EventType.SYSTEM,
                {"manager": "plugins"},
                "plugin_manager"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Error during plugin manager cleanup: {str(e)}",
                EventType.ERROR,
                {"error": str(e)},
                "plugin_manager"
            )