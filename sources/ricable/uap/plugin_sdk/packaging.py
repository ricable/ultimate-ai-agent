# File: plugin_sdk/packaging.py
"""
Plugin packaging utilities for creating distributable plugin packages.
"""

import zipfile
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from .base import PluginManifest
from .manifest import ManifestValidator


class PluginPackager:
    """
    Utility for packaging plugins into distributable ZIP files.
    """
    
    def __init__(self, plugin_directory: Path):
        self.plugin_directory = Path(plugin_directory)
        self.manifest_path = self.plugin_directory / "manifest.json"
        
        if not self.plugin_directory.exists():
            raise ValueError(f"Plugin directory does not exist: {plugin_directory}")
        
        if not self.manifest_path.exists():
            raise ValueError(f"Manifest file not found: {self.manifest_path}")
    
    def validate_plugin(self) -> Dict[str, Any]:
        """Validate the plugin structure and manifest."""
        issues = []
        warnings = []
        
        # Load and validate manifest
        try:
            with open(self.manifest_path, 'r') as f:
                manifest_data = json.load(f)
            manifest = PluginManifest(**manifest_data)
        except Exception as e:
            issues.append(f"Invalid manifest: {str(e)}")
            return {"valid": False, "issues": issues, "warnings": warnings}
        
        # Validate manifest content
        manifest_issues = ManifestValidator.validate_manifest(manifest)
        issues.extend(manifest_issues)
        
        # Get improvement suggestions
        suggestions = ManifestValidator.suggest_improvements(manifest)
        warnings.extend(suggestions)
        
        # Check main module exists
        main_module_path = self.plugin_directory / f"{manifest.main_module}.py"
        if not main_module_path.exists():
            issues.append(f"Main module not found: {manifest.main_module}.py")
        
        # Check for required files
        required_files = ["manifest.json", f"{manifest.main_module}.py"]
        for required_file in required_files:
            file_path = self.plugin_directory / required_file
            if not file_path.exists():
                issues.append(f"Required file missing: {required_file}")
        
        # Check for common optional files
        optional_files = ["README.md", "LICENSE", "requirements.txt"]
        for optional_file in optional_files:
            file_path = self.plugin_directory / optional_file
            if not file_path.exists():
                warnings.append(f"Consider adding {optional_file} for better documentation")
        
        # Validate Python syntax
        if main_module_path.exists():
            try:
                with open(main_module_path, 'r') as f:
                    code = f.read()
                compile(code, str(main_module_path), 'exec')
            except SyntaxError as e:
                issues.append(f"Syntax error in {manifest.main_module}.py: {str(e)}")
        
        # Check file sizes
        total_size = 0
        large_files = []
        for file_path in self.plugin_directory.rglob('*'):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                if size > 10 * 1024 * 1024:  # 10MB
                    large_files.append((str(file_path.relative_to(self.plugin_directory)), size))
        
        if total_size > 100 * 1024 * 1024:  # 100MB
            warnings.append(f"Plugin package is large ({total_size // (1024*1024)}MB). Consider optimizing.")
        
        if large_files:
            warnings.append(f"Large files detected: {large_files}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "manifest": manifest.dict(),
            "total_size_bytes": total_size,
            "file_count": len(list(self.plugin_directory.rglob('*')))
        }
    
    def create_package(self, output_path: Path, include_patterns: List[str] = None,
                      exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Create a plugin package ZIP file."""
        # Validate plugin first
        validation_result = self.validate_plugin()
        if not validation_result["valid"]:
            raise ValueError(f"Plugin validation failed: {validation_result['issues']}")
        
        # Default patterns
        if include_patterns is None:
            include_patterns = ["*.py", "*.json", "*.md", "*.txt", "*.yml", "*.yaml"]
        
        if exclude_patterns is None:
            exclude_patterns = [
                "__pycache__", "*.pyc", "*.pyo", "*.pyd", ".git", ".gitignore",
                ".pytest_cache", "*.log", "*.tmp", ".DS_Store"
            ]
        
        # Create ZIP package
        package_info = {
            "package_path": str(output_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files_included": [],
            "total_size_bytes": 0
        }
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in self.plugin_directory.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.plugin_directory)
                    
                    # Check include/exclude patterns
                    if self._should_include_file(relative_path, include_patterns, exclude_patterns):
                        zipf.write(file_path, relative_path)
                        file_size = file_path.stat().st_size
                        package_info["files_included"].append({
                            "path": str(relative_path),
                            "size_bytes": file_size
                        })
                        package_info["total_size_bytes"] += file_size
        
        # Calculate package checksum
        package_checksum = self._calculate_file_checksum(output_path)
        package_info["checksum_sha256"] = package_checksum
        
        # Update manifest with package info
        manifest_data = validation_result["manifest"]
        manifest_data["install_size"] = package_info["total_size_bytes"]
        manifest_data["checksum"] = package_checksum
        manifest_data["updated_at"] = package_info["created_at"]
        
        return {
            "success": True,
            "package_info": package_info,
            "manifest": manifest_data,
            "validation": validation_result
        }
    
    def _should_include_file(self, file_path: Path, include_patterns: List[str], 
                           exclude_patterns: List[str]) -> bool:
        """Check if file should be included based on patterns."""
        import fnmatch
        
        file_str = str(file_path)
        
        # Check exclude patterns first
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_str, pattern) or pattern in file_str:
                return False
        
        # Check include patterns
        for pattern in include_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                return True
        
        return False
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def extract_package(package_path: Path, extract_to: Path) -> Dict[str, Any]:
        """Extract a plugin package to a directory."""
        if not package_path.exists():
            raise ValueError(f"Package file does not exist: {package_path}")
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        extracted_files = []
        manifest_data = None
        
        with zipfile.ZipFile(package_path, 'r') as zipf:
            # Validate package structure
            file_list = zipf.namelist()
            if "manifest.json" not in file_list:
                raise ValueError("Package missing manifest.json")
            
            # Extract all files
            for file_name in file_list:
                zipf.extract(file_name, extract_to)
                extracted_files.append(file_name)
            
            # Read manifest
            with zipf.open("manifest.json") as manifest_file:
                manifest_data = json.load(manifest_file)
        
        # Validate extracted manifest
        try:
            manifest = PluginManifest(**manifest_data)
        except Exception as e:
            raise ValueError(f"Invalid manifest in package: {str(e)}")
        
        return {
            "success": True,
            "extracted_files": extracted_files,
            "manifest": manifest_data,
            "extract_path": str(extract_to)
        }
    
    @classmethod
    def create_from_template(cls, template_name: str, plugin_name: str, 
                           output_directory: Path, **template_args) -> 'PluginPackager':
        """Create a new plugin from a template."""
        templates = {
            "integration": cls._create_integration_template,
            "processor": cls._create_processor_template,
            "ai_agent": cls._create_ai_agent_template,
            "workflow": cls._create_workflow_template
        }
        
        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")
        
        plugin_directory = output_directory / plugin_name
        plugin_directory.mkdir(parents=True, exist_ok=True)
        
        # Create template files
        templates[template_name](plugin_directory, plugin_name, **template_args)
        
        return cls(plugin_directory)
    
    @staticmethod
    def _create_integration_template(plugin_directory: Path, plugin_name: str, 
                                   service_name: str = "ExampleService", **kwargs):
        """Create integration plugin template."""
        from .manifest import ManifestBuilder
        
        # Create manifest
        manifest = (ManifestBuilder
                   .create_integration_plugin(
                       name=plugin_name,
                       display_name=kwargs.get("display_name", f"{service_name} Integration"),
                       version="1.0.0",
                       description=kwargs.get("description", f"Integration with {service_name}"),
                       author=kwargs.get("author", "Plugin Developer"),
                       service_name=service_name
                   )
                   .build())
        
        # Save manifest
        with open(plugin_directory / "manifest.json", 'w') as f:
            json.dump(manifest.dict(), f, indent=2, default=str)
        
        # Create main module
        main_module_content = f'''# {service_name} Integration Plugin
"""
{service_name} integration for UAP.
"""

import asyncio
from typing import Dict, Any

from uap_sdk import IntegrationPlugin, PluginContext, PluginResponse


class {service_name}Integration(IntegrationPlugin):
    """
    Integration plugin for {service_name}.
    """
    
    async def initialize(self, context: PluginContext) -> PluginResponse:
        """Initialize the {service_name} integration."""
        try:
            # Initialize your integration here
            self._api_key = self.config.config.get("api_key")
            self._base_url = self.config.config.get("base_url", "https://api.example.com")
            
            if not self._api_key:
                return PluginResponse(
                    success=False,
                    error="API key not configured",
                    error_code="missing_api_key"
                )
            
            return PluginResponse(
                success=True,
                data={{"status": "initialized", "service": "{service_name}"}}
            )
            
        except Exception as e:
            return self._format_error_response(e, "initialization_failed")
    
    async def execute(self, action: str, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Execute an action."""
        try:
            if action == "send_message":
                return await self._send_message(params, context)
            elif action == "get_status":
                return await self._get_status(params, context)
            else:
                return PluginResponse(
                    success=False,
                    error=f"Unknown action: {{action}}",
                    error_code="unknown_action"
                )
                
        except Exception as e:
            return self._format_error_response(e, "execution_failed")
    
    async def cleanup(self) -> PluginResponse:
        """Clean up resources."""
        try:
            # Cleanup your integration here
            return PluginResponse(
                success=True,
                data={{"status": "cleaned_up"}}
            )
            
        except Exception as e:
            return self._format_error_response(e, "cleanup_failed")
    
    async def authenticate(self, credentials: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Authenticate with {service_name}."""
        try:
            # Implement authentication logic
            api_key = credentials.get("api_key")
            if not api_key:
                return PluginResponse(
                    success=False,
                    error="API key required",
                    error_code="missing_credentials"
                )
            
            # Test the API key
            # ... implement actual authentication test ...
            
            return PluginResponse(
                success=True,
                data={{"status": "authenticated", "service": "{service_name}"}}
            )
            
        except Exception as e:
            return self._format_error_response(e, "authentication_failed")
    
    async def test_connection(self, context: PluginContext) -> PluginResponse:
        """Test connection to {service_name}."""
        try:
            # Implement connection test
            return PluginResponse(
                success=True,
                data={{"status": "connected", "service": "{service_name}"}}
            )
            
        except Exception as e:
            return self._format_error_response(e, "connection_test_failed")
    
    async def send_data(self, data: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Send data to {service_name}."""
        try:
            # Implement data sending logic
            return PluginResponse(
                success=True,
                data={{"status": "sent", "service": "{service_name}"}}
            )
            
        except Exception as e:
            return self._format_error_response(e, "send_data_failed")
    
    async def receive_webhook(self, event) -> PluginResponse:
        """Process webhook from {service_name}."""
        try:
            # Implement webhook processing
            return PluginResponse(
                success=True,
                data={{"status": "processed", "event_type": event.event_type}}
            )
            
        except Exception as e:
            return self._format_error_response(e, "webhook_processing_failed")
    
    async def _send_message(self, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Send a message via {service_name}."""
        message = params.get("message")
        channel = params.get("channel")
        
        if not message:
            return PluginResponse(
                success=False,
                error="Message content required",
                error_code="missing_message"
            )
        
        # Implement message sending logic
        return PluginResponse(
            success=True,
            data={{
                "message_sent": True,
                "message": message,
                "channel": channel,
                "service": "{service_name}"
            }}
        )
    
    async def _get_status(self, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Get integration status."""
        return PluginResponse(
            success=True,
            data={{
                "service": "{service_name}",
                "status": "active",
                "plugin_id": self.plugin_id,
                "version": self.version
            }}
        )
'''
        
        with open(plugin_directory / f"{plugin_name}_integration.py", 'w') as f:
            f.write(main_module_content)
        
        # Create README
        readme_content = f'''# {service_name} Integration Plugin

{kwargs.get("description", f"Integration with {service_name} for UAP.")}

## Installation

1. Upload the plugin package to UAP
2. Configure your {service_name} API credentials
3. Enable the plugin

## Configuration

- `api_key`: Your {service_name} API key
- `base_url`: {service_name} API base URL (optional)

## Actions

- `send_message`: Send a message
- `get_status`: Get integration status

## Author

{kwargs.get("author", "Plugin Developer")}
'''
        
        with open(plugin_directory / "README.md", 'w') as f:
            f.write(readme_content)
    
    @staticmethod
    def _create_processor_template(plugin_directory: Path, plugin_name: str, **kwargs):
        """Create processor plugin template."""
        # Similar implementation for processor plugins
        pass
    
    @staticmethod
    def _create_ai_agent_template(plugin_directory: Path, plugin_name: str, **kwargs):
        """Create AI agent plugin template."""
        # Similar implementation for AI agent plugins
        pass
    
    @staticmethod
    def _create_workflow_template(plugin_directory: Path, plugin_name: str, **kwargs):
        """Create workflow plugin template."""
        # Similar implementation for workflow plugins
        pass