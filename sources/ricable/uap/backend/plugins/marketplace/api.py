# File: backend/plugins/marketplace/api.py
"""
Marketplace API - REST endpoints for plugin marketplace operations.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..plugin_manager import PluginManager
from ..plugin_base import PluginManifest, PluginConfig, PluginType
from ...integrations.registry import IntegrationRegistry
from ...integrations.manager import IntegrationManager
from ...services.auth import get_current_user
from ...monitoring.logs.logger import uap_logger, EventType, LogLevel


class InstallPluginRequest(BaseModel):
    """Request model for plugin installation"""
    template_name: Optional[str] = None
    plugin_url: Optional[str] = None
    config_overrides: Dict[str, Any] = {}
    custom_name: Optional[str] = None


class ConfigurePluginRequest(BaseModel):
    """Request model for plugin configuration"""
    integration_id: str
    credentials: Dict[str, Any]
    config_updates: Dict[str, Any] = {}


class MarketplaceAPI:
    """
    API endpoints for the plugin marketplace.
    """
    
    def __init__(self, plugin_manager: PluginManager, integration_manager: IntegrationManager):
        self.plugin_manager = plugin_manager
        self.integration_manager = integration_manager
        self.integration_registry = IntegrationRegistry()
        self.router = APIRouter(prefix="/api/marketplace", tags=["marketplace"])
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register all marketplace API routes."""
        
        @self.router.get("/")
        async def get_marketplace_overview(current_user=Depends(get_current_user)):
            """Get marketplace overview with statistics."""
            try:
                # Get integration templates
                templates = self.integration_registry.list_templates()
                
                # Get installed integrations
                installed = self.integration_manager.list_integrations()
                
                # Get plugins
                plugins = self.plugin_manager.list_plugins()
                
                # Calculate stats
                stats = {
                    "total_integrations": len(templates),
                    "installed_integrations": len(installed),
                    "active_integrations": len([i for i in installed if i["is_authenticated"]]),
                    "categories": len(self.integration_registry.get_categories()),
                    "popular_integrations": [t.name for t in self.integration_registry.get_popular_integrations(5)],
                    "total_plugins": len(plugins),
                    "active_plugins": len([p for p in plugins if p["status"] == "active"])
                }
                
                return {
                    "stats": stats,
                    "featured_integrations": templates[:6],
                    "recent_updates": sorted(templates, key=lambda t: t.popularity_score, reverse=True)[:5]
                }
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Failed to get marketplace overview: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e), "user_id": current_user.get("user_id")},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail="Failed to load marketplace overview")
        
        @self.router.get("/categories")
        async def get_categories(current_user=Depends(get_current_user)):
            """Get all available categories with counts."""
            try:
                return self.integration_registry.get_categories()
            except Exception as e:
                raise HTTPException(status_code=500, detail="Failed to load categories")
        
        @self.router.get("/integrations")
        async def get_integrations(
            category: Optional[str] = Query(None),
            search: Optional[str] = Query(None),
            sort: str = Query("popularity"),
            limit: int = Query(50, le=100),
            current_user=Depends(get_current_user)
        ):
            """Get available integrations with filtering and search."""
            try:
                templates = self.integration_registry.list_templates()
                
                # Apply filters
                if category:
                    templates = [t for t in templates if t.category == category]
                
                if search:
                    search_results = self.integration_registry.search_templates(search)
                    # Combine with category filter if applied
                    if category:
                        template_names = {t.name for t in templates}
                        search_results = [t for t in search_results if t.name in template_names]
                    templates = search_results
                
                # Apply sorting
                if sort == "name":
                    templates.sort(key=lambda t: t.display_name.lower())
                elif sort == "category":
                    templates.sort(key=lambda t: (t.category, t.display_name.lower()))
                else:  # popularity (default)
                    templates.sort(key=lambda t: t.popularity_score, reverse=True)
                
                # Apply limit
                templates = templates[:limit]
                
                # Convert to dict format for JSON response
                result = []
                for template in templates:
                    result.append({
                        "name": template.name,
                        "display_name": template.display_name,
                        "description": template.description,
                        "category": template.category,
                        "integration_type": template.integration_type,
                        "auth_method": template.auth_method,
                        "logo_url": template.logo_url,
                        "documentation_url": template.documentation_url,
                        "pricing_model": template.pricing_model,
                        "popularity_score": template.popularity_score,
                        "required_credentials": template.required_credentials,
                        "supported_features": template.supported_features,
                        "webhook_events": template.webhook_events,
                        "rate_limits": template.rate_limits
                    })
                
                return result
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Failed to get integrations: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e), "user_id": current_user.get("user_id")},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail="Failed to load integrations")
        
        @self.router.get("/installed")
        async def get_installed_integrations(current_user=Depends(get_current_user)):
            """Get user's installed integrations."""
            try:
                user_id = current_user.get("user_id")
                if not user_id:
                    return []
                
                # Get installed integrations from integration manager
                installed = self.integration_manager.get_user_integrations(user_id)
                
                # Also get installed plugins
                plugins = self.plugin_manager.list_plugins()
                
                # Combine results
                result = []
                
                # Add integrations
                for integration in installed:
                    result.append({
                        "integration_id": integration["integration_id"],
                        "name": integration["name"],
                        "display_name": integration["display_name"],
                        "type": "integration",
                        "status": integration["status"],
                        "is_authenticated": integration.get("authenticated_at") is not None,
                        "created_at": integration.get("authenticated_at"),
                        "last_error": None
                    })
                
                # Add plugins
                for plugin in plugins:
                    if plugin["enabled"]:
                        result.append({
                            "integration_id": plugin["plugin_id"],
                            "name": plugin["name"],
                            "display_name": plugin["display_name"],
                            "type": "plugin",
                            "status": plugin["status"],
                            "is_authenticated": plugin["status"] == "active",
                            "created_at": plugin["created_at"],
                            "last_error": None
                        })
                
                return result
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Failed to get installed integrations: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e), "user_id": current_user.get("user_id")},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail="Failed to load installed integrations")
        
        @self.router.post("/install")
        async def install_integration(
            request: InstallPluginRequest,
            current_user=Depends(get_current_user)
        ):
            """Install an integration or plugin."""
            try:
                user_id = current_user.get("user_id")
                
                if request.template_name:
                    # Install from template
                    template = self.integration_registry.get_template(request.template_name)
                    if not template:
                        raise HTTPException(status_code=404, detail="Integration template not found")
                    
                    # Create integration config
                    config = self.integration_registry.create_config_from_template(
                        request.template_name,
                        **request.config_overrides
                    )
                    
                    if request.custom_name:
                        config.display_name = request.custom_name
                    
                    # Get integration class
                    integration_class = self.integration_registry.get_integration_class(request.template_name)
                    if not integration_class:
                        raise HTTPException(status_code=400, detail="Integration implementation not available")
                    
                    # Register with integration manager
                    self.integration_manager.register_integration(integration_class, config)
                    
                    uap_logger.log_event(
                        LogLevel.INFO,
                        f"Integration installed: {config.display_name}",
                        EventType.INTEGRATION,
                        {
                            "integration_id": config.integration_id,
                            "user_id": user_id,
                            "template_name": request.template_name
                        },
                        "marketplace_api"
                    )
                    
                    return {
                        "success": True,
                        "integration_id": config.integration_id,
                        "name": config.name,
                        "display_name": config.display_name,
                        "requires_authentication": len(template.required_credentials) > 0,
                        "required_credentials": template.required_credentials
                    }
                
                elif request.plugin_url:
                    # Install from URL (future feature)
                    raise HTTPException(status_code=501, detail="Plugin installation from URL not yet implemented")
                
                else:
                    raise HTTPException(status_code=400, detail="Either template_name or plugin_url must be provided")
                
            except HTTPException:
                raise
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Failed to install integration: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e), "user_id": current_user.get("user_id")},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail="Installation failed")
        
        @self.router.post("/configure")
        async def configure_integration(
            request: ConfigurePluginRequest,
            current_user=Depends(get_current_user)
        ):
            """Configure an installed integration with credentials."""
            try:
                user_id = current_user.get("user_id")
                
                # Authenticate the integration
                response = await self.integration_manager.authenticate_integration(
                    request.integration_id,
                    user_id,
                    request.credentials
                )
                
                if response.success:
                    uap_logger.log_event(
                        LogLevel.INFO,
                        f"Integration configured: {request.integration_id}",
                        EventType.INTEGRATION,
                        {
                            "integration_id": request.integration_id,
                            "user_id": user_id
                        },
                        "marketplace_api"
                    )
                    
                    return {
                        "success": True,
                        "integration_id": request.integration_id,
                        "status": "configured",
                        "data": response.data
                    }
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Configuration failed: {response.error}"
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Failed to configure integration: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e), "user_id": current_user.get("user_id")},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail="Configuration failed")
        
        @self.router.delete("/uninstall/{integration_id}")
        async def uninstall_integration(
            integration_id: str,
            current_user=Depends(get_current_user)
        ):
            """Uninstall an integration or plugin."""
            try:
                user_id = current_user.get("user_id")
                
                # Try plugin manager first
                if integration_id in self.plugin_manager.manifests:
                    response = await self.plugin_manager.uninstall_plugin(integration_id)
                    if response.success:
                        uap_logger.log_event(
                            LogLevel.INFO,
                            f"Plugin uninstalled: {integration_id}",
                            EventType.PLUGIN,
                            {"plugin_id": integration_id, "user_id": user_id},
                            "marketplace_api"
                        )
                        return {"success": True, "message": "Plugin uninstalled successfully"}
                    else:
                        raise HTTPException(status_code=400, detail=response.error)
                
                # Try integration manager
                integration = self.integration_manager.get_integration(integration_id)
                if integration:
                    # Remove from user credentials (simplified - in production would be more sophisticated)
                    if user_id in self.integration_manager.user_credentials:
                        self.integration_manager.user_credentials[user_id].pop(integration_id, None)
                    
                    uap_logger.log_event(
                        LogLevel.INFO,
                        f"Integration uninstalled: {integration_id}",
                        EventType.INTEGRATION,
                        {"integration_id": integration_id, "user_id": user_id},
                        "marketplace_api"
                    )
                    return {"success": True, "message": "Integration uninstalled successfully"}
                
                raise HTTPException(status_code=404, detail="Integration or plugin not found")
                
            except HTTPException:
                raise
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Failed to uninstall integration: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e), "user_id": current_user.get("user_id")},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail="Uninstallation failed")
        
        @self.router.get("/plugins")
        async def get_plugins(
            plugin_type: Optional[PluginType] = Query(None),
            status: Optional[str] = Query(None),
            current_user=Depends(get_current_user)
        ):
            """Get available plugins with filtering."""
            try:
                plugins = self.plugin_manager.list_plugins()
                
                # Apply filters
                if plugin_type:
                    plugins = [p for p in plugins if p["plugin_type"] == plugin_type]
                
                if status:
                    plugins = [p for p in plugins if p["status"] == status]
                
                return plugins
                
            except Exception as e:
                raise HTTPException(status_code=500, detail="Failed to load plugins")
        
        @self.router.post("/plugins/{plugin_id}/enable")
        async def enable_plugin(
            plugin_id: str,
            current_user=Depends(get_current_user)
        ):
            """Enable a plugin."""
            try:
                response = await self.plugin_manager.enable_plugin(plugin_id)
                
                if response.success:
                    return {"success": True, "message": "Plugin enabled successfully"}
                else:
                    raise HTTPException(status_code=400, detail=response.error)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail="Failed to enable plugin")
        
        @self.router.post("/plugins/{plugin_id}/disable")
        async def disable_plugin(
            plugin_id: str,
            current_user=Depends(get_current_user)
        ):
            """Disable a plugin."""
            try:
                response = await self.plugin_manager.disable_plugin(plugin_id)
                
                if response.success:
                    return {"success": True, "message": "Plugin disabled successfully"}
                else:
                    raise HTTPException(status_code=400, detail=response.error)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail="Failed to disable plugin")
        
        @self.router.post("/plugins/upload")
        async def upload_plugin(
            file: UploadFile = File(...),
            current_user=Depends(get_current_user)
        ):
            """Upload and install a plugin package."""
            try:
                if not file.filename.endswith('.zip'):
                    raise HTTPException(status_code=400, detail="Plugin package must be a ZIP file")
                
                # Read plugin package
                package_data = await file.read()
                
                # Install plugin
                response = await self.plugin_manager.install_plugin(package_data)
                
                if response.success:
                    uap_logger.log_event(
                        LogLevel.INFO,
                        f"Plugin uploaded and installed: {response.data.get('name')}",
                        EventType.PLUGIN,
                        {
                            "plugin_id": response.data.get("plugin_id"),
                            "user_id": current_user.get("user_id"),
                            "filename": file.filename
                        },
                        "marketplace_api"
                    )
                    return response.data
                else:
                    raise HTTPException(status_code=400, detail=response.error)
                
            except HTTPException:
                raise
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Failed to upload plugin: {str(e)}",
                    EventType.ERROR,
                    {"error": str(e), "user_id": current_user.get("user_id")},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail="Plugin upload failed")
        
        @self.router.get("/status")
        async def get_marketplace_status(current_user=Depends(get_current_user)):
            """Get marketplace system status."""
            try:
                integration_status = self.integration_manager.get_system_status()
                plugin_status = self.plugin_manager.get_system_status()
                
                return {
                    "marketplace_active": True,
                    "integrations": integration_status,
                    "plugins": plugin_status,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail="Failed to get marketplace status")


# Helper function to create the marketplace API router
def create_marketplace_router(plugin_manager: PluginManager, integration_manager: IntegrationManager) -> APIRouter:
    """Create and configure the marketplace API router."""
    marketplace_api = MarketplaceAPI(plugin_manager, integration_manager)
    return marketplace_api.router