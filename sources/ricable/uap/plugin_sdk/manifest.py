# File: plugin_sdk/manifest.py
"""
Plugin manifest builder and validation utilities.
"""

import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pathlib import Path

from .base import PluginManifest, PluginType


class ManifestBuilder:
    """
    Builder class for creating plugin manifests.
    
    Provides a fluent interface for constructing plugin manifests
    with validation and best practices.
    """
    
    def __init__(self):
        self._data = {
            "plugin_id": str(uuid.uuid4()),
            "python_version": ">=3.11",
            "uap_version": ">=1.0.0",
            "pricing_model": "free",
            "popularity_score": 0,
            "supported_platforms": ["linux", "darwin", "win32"],
            "dependencies": [],
            "permissions": [],
            "tags": [],
            "config_schema": {},
            "default_config": {},
            "screenshots": [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
    
    def set_basic_info(self, name: str, display_name: str, version: str, 
                      description: str, author: str) -> 'ManifestBuilder':
        """Set basic plugin information."""
        self._data.update({
            "name": name,
            "display_name": display_name,
            "version": version,
            "description": description,
            "author": author
        })
        return self
    
    def set_type_and_category(self, plugin_type: PluginType, category: str) -> 'ManifestBuilder':
        """Set plugin type and category."""
        self._data.update({
            "plugin_type": plugin_type,
            "category": category
        })
        return self
    
    def set_entry_point(self, main_module: str, main_class: str) -> 'ManifestBuilder':
        """Set plugin entry point."""
        self._data.update({
            "main_module": main_module,
            "main_class": main_class
        })
        return self
    
    def set_author_info(self, author_email: str = None, homepage: str = None, 
                       repository: str = None) -> 'ManifestBuilder':
        """Set additional author information."""
        if author_email:
            self._data["author_email"] = author_email
        if homepage:
            self._data["homepage"] = homepage
        if repository:
            self._data["repository"] = repository
        return self
    
    def set_license(self, license_name: str) -> 'ManifestBuilder':
        """Set plugin license."""
        self._data["license"] = license_name
        return self
    
    def add_dependencies(self, *dependencies: str) -> 'ManifestBuilder':
        """Add Python dependencies."""
        self._data["dependencies"].extend(dependencies)
        return self
    
    def add_permissions(self, *permissions: str) -> 'ManifestBuilder':
        """Add required permissions."""
        self._data["permissions"].extend(permissions)
        return self
    
    def add_tags(self, *tags: str) -> 'ManifestBuilder':
        """Add plugin tags."""
        self._data["tags"].extend(tags)
        return self
    
    def set_config_schema(self, schema: Dict[str, Any]) -> 'ManifestBuilder':
        """Set configuration schema."""
        self._data["config_schema"] = schema
        return self
    
    def set_default_config(self, config: Dict[str, Any]) -> 'ManifestBuilder':
        """Set default configuration."""
        self._data["default_config"] = config
        return self
    
    def set_marketplace_info(self, logo_url: str = None, documentation_url: str = None,
                           support_url: str = None, pricing_model: str = None) -> 'ManifestBuilder':
        """Set marketplace display information."""
        if logo_url:
            self._data["logo_url"] = logo_url
        if documentation_url:
            self._data["documentation_url"] = documentation_url
        if support_url:
            self._data["support_url"] = support_url
        if pricing_model:
            self._data["pricing_model"] = pricing_model
        return self
    
    def add_screenshots(self, *screenshot_urls: str) -> 'ManifestBuilder':
        """Add screenshot URLs."""
        self._data["screenshots"].extend(screenshot_urls)
        return self
    
    def set_requirements(self, python_version: str = None, uap_version: str = None,
                        platforms: List[str] = None) -> 'ManifestBuilder':
        """Set version and platform requirements."""
        if python_version:
            self._data["python_version"] = python_version
        if uap_version:
            self._data["uap_version"] = uap_version
        if platforms:
            self._data["supported_platforms"] = platforms
        return self
    
    def build(self) -> PluginManifest:
        """Build and validate the manifest."""
        # Validate required fields
        required_fields = [
            "name", "display_name", "version", "description", 
            "author", "plugin_type", "category", "main_module", "main_class"
        ]
        
        missing_fields = [field for field in required_fields if field not in self._data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Create and return manifest
        return PluginManifest(**self._data)
    
    def save_to_file(self, file_path: Path) -> PluginManifest:
        """Build manifest and save to file."""
        manifest = self.build()
        
        with open(file_path, 'w') as f:
            json.dump(manifest.dict(), f, indent=2, default=str)
        
        return manifest
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'ManifestBuilder':
        """Load manifest from file and create builder."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        builder = cls()
        builder._data.update(data)
        return builder
    
    @classmethod
    def create_integration_plugin(cls, name: str, display_name: str, version: str,
                                description: str, author: str, service_name: str) -> 'ManifestBuilder':
        """Create a builder for an integration plugin."""
        return (cls()
                .set_basic_info(name, display_name, version, description, author)
                .set_type_and_category(PluginType.INTEGRATION, "communication")
                .set_entry_point(f"{name}_integration", f"{service_name}Integration")
                .add_tags("integration", service_name.lower())
                .add_permissions("network_access", "store_credentials")
                .set_config_schema({
                    "type": "object",
                    "properties": {
                        "api_key": {
                            "type": "string",
                            "description": f"{service_name} API key",
                            "secret": True
                        },
                        "base_url": {
                            "type": "string", 
                            "description": f"{service_name} API base URL",
                            "default": "https://api.example.com"
                        }
                    },
                    "required": ["api_key"]
                }))
    
    @classmethod
    def create_processor_plugin(cls, name: str, display_name: str, version: str,
                              description: str, author: str, input_formats: List[str]) -> 'ManifestBuilder':
        """Create a builder for a processor plugin."""
        return (cls()
                .set_basic_info(name, display_name, version, description, author)
                .set_type_and_category(PluginType.PROCESSOR, "data_processing")
                .set_entry_point(f"{name}_processor", f"{name.title()}Processor")
                .add_tags("processor", "data", *input_formats)
                .add_permissions("read_files")
                .set_config_schema({
                    "type": "object",
                    "properties": {
                        "input_formats": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": input_formats
                        },
                        "output_format": {
                            "type": "string",
                            "default": "json"
                        }
                    }
                }))
    
    @classmethod
    def create_ai_agent_plugin(cls, name: str, display_name: str, version: str,
                             description: str, author: str, model_type: str) -> 'ManifestBuilder':
        """Create a builder for an AI agent plugin."""
        return (cls()
                .set_basic_info(name, display_name, version, description, author)
                .set_type_and_category(PluginType.AI_AGENT, "artificial_intelligence")
                .set_entry_point(f"{name}_agent", f"{name.title()}Agent")
                .add_tags("ai", "agent", model_type.lower())
                .add_permissions("ai_inference", "network_access")
                .set_config_schema({
                    "type": "object",
                    "properties": {
                        "model_name": {
                            "type": "string",
                            "description": "AI model to use",
                            "default": model_type
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens per response",
                            "default": 2048,
                            "minimum": 1,
                            "maximum": 8192
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Response creativity (0.0-1.0)",
                            "default": 0.7,
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    }
                }))


class ManifestValidator:
    """
    Utility class for validating plugin manifests.
    """
    
    @staticmethod
    def validate_manifest(manifest: PluginManifest) -> List[str]:
        """Validate a plugin manifest and return list of issues."""
        issues = []
        
        # Check version format
        if not ManifestValidator._is_valid_version(manifest.version):
            issues.append(f"Invalid version format: {manifest.version}")
        
        # Check plugin name format
        if not manifest.name.replace('_', '').replace('-', '').isalnum():
            issues.append(f"Plugin name contains invalid characters: {manifest.name}")
        
        # Check description length
        if len(manifest.description) < 10:
            issues.append("Description should be at least 10 characters")
        if len(manifest.description) > 500:
            issues.append("Description should be less than 500 characters")
        
        # Check required permissions
        dangerous_permissions = ['system_commands', 'file_system_write', 'network_unrestricted']
        for permission in manifest.permissions:
            if permission in dangerous_permissions:
                issues.append(f"Potentially dangerous permission requested: {permission}")
        
        # Check config schema
        if manifest.config_schema and not isinstance(manifest.config_schema, dict):
            issues.append("Config schema must be a valid JSON schema object")
        
        return issues
    
    @staticmethod
    def _is_valid_version(version: str) -> bool:
        """Check if version follows semantic versioning."""
        import re
        pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?(?:\+[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*)?$'
        return bool(re.match(pattern, version))
    
    @staticmethod
    def suggest_improvements(manifest: PluginManifest) -> List[str]:
        """Suggest improvements for a plugin manifest."""
        suggestions = []
        
        if not manifest.logo_url:
            suggestions.append("Consider adding a logo_url for better marketplace presentation")
        
        if not manifest.documentation_url:
            suggestions.append("Consider adding documentation_url to help users")
        
        if not manifest.repository:
            suggestions.append("Consider adding repository URL for open source projects")
        
        if not manifest.tags:
            suggestions.append("Consider adding tags to improve discoverability")
        
        if len(manifest.tags) > 10:
            suggestions.append("Consider reducing the number of tags (10 or fewer is recommended)")
        
        if not manifest.license:
            suggestions.append("Consider specifying a license for your plugin")
        
        return suggestions