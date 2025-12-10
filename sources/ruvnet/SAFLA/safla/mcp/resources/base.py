"""
Base resource class for SAFLA MCP resources.

This module provides the foundation for all MCP resource handlers,
implementing common functionality for resource management.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from safla.exceptions import ValidationError


logger = logging.getLogger(__name__)


@dataclass
class ResourceDefinition:
    """Definition of an MCP resource."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


class BaseResource(ABC):
    """
    Base class for all MCP resource handlers.
    
    Provides common functionality for resource registration, validation,
    and data retrieval. All domain-specific resources should inherit
    from this class.
    """
    
    def __init__(self, config: Any, state_manager: Optional[Any] = None):
        """
        Initialize the resource handler.
        
        Args:
            config: SAFLA configuration object
            state_manager: Optional state manager for shared state
        """
        self.config = config
        self.state_manager = state_manager
        self._resources: Dict[str, ResourceDefinition] = {}
        self._initialize_resources()
    
    @abstractmethod
    def _initialize_resources(self) -> None:
        """Initialize and register resources for this handler."""
        pass
    
    def register_resource(self, resource: ResourceDefinition) -> None:
        """
        Register a resource with the handler.
        
        Args:
            resource: Resource definition to register
        """
        self._resources[resource.uri] = resource
        logger.debug(f"Registered resource: {resource.uri}")
    
    def get_resources(self) -> List[Dict[str, Any]]:
        """
        Get all resources registered with this handler.
        
        Returns:
            List of resource definitions in MCP format
        """
        resources = []
        for resource in self._resources.values():
            resources.append({
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mime_type
            })
        return resources
    
    def has_resource(self, uri: str) -> bool:
        """Check if this handler has a specific resource."""
        return uri in self._resources
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource data with metadata
            
        Raises:
            ValidationError: If resource not found
        """
        if uri not in self._resources:
            raise ValidationError(f"Resource not found: {uri}")
        
        resource = self._resources[uri]
        
        try:
            # Get resource data
            data = await self._get_resource_data(uri)
            
            return {
                "uri": uri,
                "mimeType": resource.mime_type,
                "text": self._serialize_data(data),
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "name": resource.name,
                    "description": resource.description
                }
            }
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {str(e)}")
            raise ValidationError(f"Failed to read resource: {str(e)}")
    
    @abstractmethod
    async def _get_resource_data(self, uri: str) -> Any:
        """
        Get the actual data for a resource.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource data
        """
        pass
    
    def _serialize_data(self, data: Any) -> str:
        """
        Serialize resource data to string.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized string
        """
        import json
        
        if isinstance(data, str):
            return data
        
        try:
            return json.dumps(data, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to serialize data: {str(e)}")
            return str(data)
    
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
    
    def _format_error(self, error: str) -> Dict[str, Any]:
        """Format an error response."""
        return {
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }