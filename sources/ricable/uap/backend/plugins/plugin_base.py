# File: backend/plugins/plugin_base.py
"""
Base classes and interfaces for UAP plugins.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import importlib.util
import sys


class PluginStatus(str, Enum):
    """Plugin status enumeration"""
    INACTIVE = "inactive"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
    UPDATING = "updating"


class PluginType(str, Enum):
    """Types of plugins"""
    INTEGRATION = "integration"  # Third-party service integrations
    PROCESSOR = "processor"     # Data/document processors
    AI_AGENT = "ai_agent"       # Custom AI agents
    WORKFLOW = "workflow"       # Workflow extensions
    ANALYTICS = "analytics"     # Analytics and reporting
    SECURITY = "security"       # Security and compliance
    UTILITY = "utility"         # General utilities
    CUSTOM = "custom"           # Custom functionality


class PluginError(Exception):
    """Base exception for plugin errors"""
    def __init__(self, message: str, plugin_id: str = None, error_code: str = None):
        super().__init__(message)
        self.plugin_id = plugin_id
        self.error_code = error_code


class PluginManifest(BaseModel):
    """Plugin manifest with metadata and configuration"""
    plugin_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    display_name: str
    version: str
    description: str
    author: str
    author_email: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    
    # Plugin classification
    plugin_type: PluginType
    category: str
    tags: List[str] = Field(default_factory=list)
    
    # Runtime requirements
    python_version: str = ">=3.11"
    dependencies: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    
    # Entry points
    main_module: str
    main_class: str
    
    # Configuration
    config_schema: Dict[str, Any] = Field(default_factory=dict)
    default_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Marketplace info
    logo_url: Optional[str] = None
    screenshots: List[str] = Field(default_factory=list)
    documentation_url: Optional[str] = None
    support_url: Optional[str] = None
    pricing_model: Optional[str] = "free"
    popularity_score: int = 0
    
    # Compatibility
    uap_version: str = ">=1.0.0"
    supported_platforms: List[str] = Field(default_factory=lambda: ["linux", "darwin", "win32"])
    
    # Security
    signature: Optional[str] = None
    checksum: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Installation
    install_size: Optional[int] = None
    download_url: Optional[str] = None


class PluginConfig(BaseModel):
    """Runtime configuration for a plugin instance"""
    plugin_id: str
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    enabled: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    permissions: List[str] = Field(default_factory=list)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PluginContext(BaseModel):
    """Execution context for plugin operations"""
    plugin_id: str
    instance_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PluginResponse(BaseModel):
    """Standard response format for plugin operations"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PluginEvent(BaseModel):
    """Event structure for plugin communications"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    plugin_id: str
    event_type: str
    source: str
    target: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PluginBase(ABC):
    """
    Abstract base class for all UAP plugins.
    
    This class defines the interface that all plugins must implement.
    """
    
    def __init__(self, manifest: PluginManifest, config: PluginConfig):
        self.manifest = manifest
        self.config = config
        self.status = PluginStatus.INACTIVE
        self._last_error: Optional[str] = None
        self._startup_time: Optional[datetime] = None
        self._metrics: Dict[str, Any] = {}
        
    @property
    def plugin_id(self) -> str:
        return self.manifest.plugin_id
    
    @property
    def name(self) -> str:
        return self.manifest.name
    
    @property
    def display_name(self) -> str:
        return self.manifest.display_name
    
    @property
    def version(self) -> str:
        return self.manifest.version
    
    @property
    def plugin_type(self) -> PluginType:
        return self.manifest.plugin_type
    
    @property
    def is_active(self) -> bool:
        return self.status == PluginStatus.ACTIVE
    
    @abstractmethod
    async def initialize(self, context: PluginContext) -> PluginResponse:
        """
        Initialize the plugin with the given context.
        
        Args:
            context: Plugin execution context
            
        Returns:
            PluginResponse indicating success/failure
        """
        pass
    
    @abstractmethod
    async def execute(self, action: str, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """
        Execute a plugin action with the given parameters.
        
        Args:
            action: Action to execute
            params: Action parameters
            context: Execution context
            
        Returns:
            PluginResponse with execution results
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> PluginResponse:
        """
        Clean up plugin resources.
        
        Returns:
            PluginResponse with cleanup status
        """
        pass
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current plugin status and health information.
        
        Returns:
            Dictionary with status information
        """
        return {
            "plugin_id": self.plugin_id,
            "name": self.name,
            "display_name": self.display_name,
            "version": self.version,
            "type": self.plugin_type,
            "status": self.status,
            "enabled": self.config.enabled,
            "last_error": self._last_error,
            "startup_time": self._startup_time.isoformat() if self._startup_time else None,
            "uptime_seconds": (datetime.now(timezone.utc) - self._startup_time).total_seconds() if self._startup_time else 0,
            "metrics": self._metrics,
            "config": {
                "instance_id": self.config.instance_id,
                "permissions": self.config.permissions,
                "resource_limits": self.config.resource_limits
            }
        }
    
    async def validate_permissions(self, required_permissions: List[str], context: PluginContext) -> bool:
        """
        Validate that the plugin has required permissions.
        
        Args:
            required_permissions: List of required permissions
            context: Execution context
            
        Returns:
            True if all permissions are granted
        """
        return all(perm in context.permissions for perm in required_permissions)
    
    async def handle_event(self, event: PluginEvent) -> PluginResponse:
        """
        Handle incoming events from other plugins or the system.
        
        Args:
            event: Plugin event to handle
            
        Returns:
            PluginResponse with handling results
        """
        # Default implementation - can be overridden by plugins
        return PluginResponse(
            success=True,
            data={"status": "ignored", "event_type": event.event_type}
        )
    
    async def get_actions(self) -> List[Dict[str, Any]]:
        """
        Get list of available actions for this plugin.
        
        Returns:
            List of action definitions
        """
        # Default implementation - should be overridden by plugins
        return []
    
    async def get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for this plugin.
        
        Returns:
            JSON schema for plugin configuration
        """
        return self.manifest.config_schema
    
    def _format_error_response(self, error: Exception, error_code: str = None) -> PluginResponse:
        """
        Format error response consistently.
        
        Args:
            error: Exception that occurred
            error_code: Optional error code
            
        Returns:
            Formatted PluginResponse
        """
        self._last_error = str(error)
        return PluginResponse(
            success=False,
            error=str(error),
            error_code=error_code or "plugin_error",
            metadata={
                "plugin_id": self.plugin_id,
                "plugin_name": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
    
    def _update_metrics(self, metric_name: str, value: Any):
        """
        Update plugin metrics.
        
        Args:
            metric_name: Name of metric to update
            value: Metric value
        """
        self._metrics[metric_name] = value
        self._metrics["last_updated"] = datetime.now(timezone.utc).isoformat()


class IntegrationPlugin(PluginBase):
    """
    Base class for integration plugins that connect to third-party services.
    """
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """
        Authenticate with the third-party service.
        
        Args:
            credentials: Authentication credentials
            context: Execution context
            
        Returns:
            PluginResponse with authentication status
        """
        pass
    
    @abstractmethod
    async def test_connection(self, context: PluginContext) -> PluginResponse:
        """
        Test connection to the third-party service.
        
        Args:
            context: Execution context
            
        Returns:
            PluginResponse with connection status
        """
        pass
    
    @abstractmethod
    async def send_data(self, data: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """
        Send data to the third-party service.
        
        Args:
            data: Data to send
            context: Execution context
            
        Returns:
            PluginResponse with send status
        """
        pass
    
    @abstractmethod
    async def receive_webhook(self, event: PluginEvent) -> PluginResponse:
        """
        Process incoming webhook from the third-party service.
        
        Args:
            event: Webhook event data
            
        Returns:
            PluginResponse with processing status
        """
        pass


class ProcessorPlugin(PluginBase):
    """
    Base class for processor plugins that transform or analyze data.
    """
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """
        Process input data and return results.
        
        Args:
            input_data: Data to process
            context: Execution context
            
        Returns:
            PluginResponse with processed data
        """
        pass
    
    @abstractmethod
    async def get_supported_formats(self) -> List[str]:
        """
        Get list of supported input formats.
        
        Returns:
            List of supported formats
        """
        pass


class AIAgentPlugin(PluginBase):
    """
    Base class for AI agent plugins that provide intelligent capabilities.
    """
    
    @abstractmethod
    async def process_message(self, message: str, context: PluginContext) -> PluginResponse:
        """
        Process a message and generate a response.
        
        Args:
            message: Input message
            context: Execution context
            
        Returns:
            PluginResponse with AI response
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[Dict[str, Any]]:
        """
        Get list of AI capabilities.
        
        Returns:
            List of capability definitions
        """
        pass


class WorkflowPlugin(PluginBase):
    """
    Base class for workflow plugins that extend workflow capabilities.
    """
    
    @abstractmethod
    async def execute_step(self, step_config: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """
        Execute a workflow step.
        
        Args:
            step_config: Step configuration
            context: Execution context
            
        Returns:
            PluginResponse with step results
        """
        pass
    
    @abstractmethod
    async def get_step_types(self) -> List[Dict[str, Any]]:
        """
        Get list of available workflow step types.
        
        Returns:
            List of step type definitions
        """
        pass