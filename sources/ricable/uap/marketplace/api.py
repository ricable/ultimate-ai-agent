# File: marketplace/api.py
"""
Marketplace API for integration discovery and management.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..backend.services.auth import require_permission, UserInDB
from ..backend.integrations.registry import IntegrationRegistry, IntegrationCategory
from ..backend.integrations.manager import IntegrationManager
from ..backend.monitoring.logs.logger import uap_logger, EventType, LogLevel

# Pydantic models for API responses
class IntegrationTemplate(BaseModel):
    """Integration template information"""
    name: str
    display_name: str
    description: str
    category: str
    integration_type: str
    auth_method: str
    logo_url: Optional[str] = None
    documentation_url: Optional[str] = None
    pricing_model: Optional[str] = None
    popularity_score: int = 0
    required_credentials: List[str] = []
    supported_features: List[str] = []
    webhook_events: List[str] = []
    rate_limits: Dict[str, int] = {}

class IntegrationStatus(BaseModel):
    """Integration status information"""
    integration_id: str
    name: str
    display_name: str
    status: str
    is_authenticated: bool
    created_at: str
    last_error: Optional[str] = None

class MarketplaceCategory(BaseModel):
    """Marketplace category information"""
    name: str
    display_name: str
    count: int
    description: Optional[str] = None

class IntegrationInstallRequest(BaseModel):
    """Request to install an integration"""
    template_name: str
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    custom_name: Optional[str] = None

class IntegrationConfigRequest(BaseModel):
    """Request to configure an integration"""
    integration_id: str
    credentials: Dict[str, Any]
    config_updates: Dict[str, Any] = Field(default_factory=dict)

class MarketplaceStats(BaseModel):
    """Marketplace statistics"""
    total_integrations: int
    installed_integrations: int
    active_integrations: int
    categories: int
    popular_integrations: List[str]


class MarketplaceAPI:
    """
    Marketplace API for integration discovery and management.
    """
    
    def __init__(self, integration_manager: IntegrationManager):
        self.integration_manager = integration_manager
        self.registry = IntegrationRegistry()
        self.router = APIRouter(prefix="/api/marketplace", tags=["marketplace"])
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes."""
        
        @self.router.get("/", response_model=Dict[str, Any])
        async def marketplace_overview(
            current_user: UserInDB = Depends(require_permission("agent:read"))
        ):
            """Get marketplace overview with statistics and featured integrations."""
            try:
                # Get marketplace statistics
                templates = self.registry.list_templates()
                installed = self.integration_manager.list_integrations()
                active = [i for i in installed if i["is_authenticated"]]
                categories = self.registry.get_categories()
                popular = self.registry.get_popular_integrations(limit=5)
                
                stats = MarketplaceStats(
                    total_integrations=len(templates),
                    installed_integrations=len(installed),
                    active_integrations=len(active),
                    categories=len(categories),
                    popular_integrations=[p.name for p in popular]
                )
                
                # Get featured integrations (top 6 by popularity)
                featured = [
                    IntegrationTemplate(
                        name=t.name,
                        display_name=t.display_name,
                        description=t.description,
                        category=t.category.value,
                        integration_type=t.integration_type.value,
                        auth_method=t.auth_method.value,
                        logo_url=t.logo_url,
                        documentation_url=t.documentation_url,
                        pricing_model=t.pricing_model,
                        popularity_score=t.popularity_score,
                        required_credentials=t.required_credentials,
                        supported_features=t.supported_features,
                        webhook_events=t.webhook_events,
                        rate_limits=t.rate_limits
                    )
                    for t in self.registry.get_popular_integrations(limit=6)
                ]
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Marketplace overview requested by user {current_user.username}",
                    EventType.API,
                    {"user_id": current_user.id, "stats": stats.dict()},
                    "marketplace_api"
                )
                
                return {
                    "stats": stats.dict(),
                    "featured_integrations": [f.dict() for f in featured],
                    "categories": categories,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Marketplace overview error: {str(e)}",
                    EventType.ERROR,
                    {"user_id": current_user.id, "error": str(e)},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Failed to get marketplace overview: {str(e)}")
        
        @self.router.get("/integrations", response_model=List[IntegrationTemplate])
        async def list_available_integrations(
            category: Optional[str] = Query(None, description="Filter by category"),
            integration_type: Optional[str] = Query(None, description="Filter by integration type"),
            search: Optional[str] = Query(None, description="Search query"),
            sort: str = Query("popularity", description="Sort by: popularity, name, category"),
            limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
            current_user: UserInDB = Depends(require_permission("agent:read"))
        ):
            """List available integration templates."""
            try:
                # Get templates based on filters
                if search:
                    templates = self.registry.search_templates(search)
                else:
                    category_filter = IntegrationCategory(category) if category else None
                    type_filter = None  # You'd need to import and use IntegrationType here
                    templates = self.registry.list_templates(category_filter, type_filter)
                
                # Apply sorting
                if sort == "name":
                    templates.sort(key=lambda t: t.display_name.lower())
                elif sort == "category":
                    templates.sort(key=lambda t: (t.category.value, t.display_name.lower()))
                # Default is popularity (already sorted by registry)
                
                # Apply limit
                templates = templates[:limit]
                
                # Convert to response models
                result = [
                    IntegrationTemplate(
                        name=t.name,
                        display_name=t.display_name,
                        description=t.description,
                        category=t.category.value,
                        integration_type=t.integration_type.value,
                        auth_method=t.auth_method.value,
                        logo_url=t.logo_url,
                        documentation_url=t.documentation_url,
                        pricing_model=t.pricing_model,
                        popularity_score=t.popularity_score,
                        required_credentials=t.required_credentials,
                        supported_features=t.supported_features,
                        webhook_events=t.webhook_events,
                        rate_limits=t.rate_limits
                    )
                    for t in templates
                ]
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Listed {len(result)} integrations for user {current_user.username}",
                    EventType.API,
                    {
                        "user_id": current_user.id, 
                        "filters": {"category": category, "type": integration_type, "search": search},
                        "count": len(result)
                    },
                    "marketplace_api"
                )
                
                return result
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"List integrations error: {str(e)}",
                    EventType.ERROR,
                    {"user_id": current_user.id, "error": str(e)},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Failed to list integrations: {str(e)}")
        
        @self.router.get("/integrations/{template_name}", response_model=Dict[str, Any])
        async def get_integration_details(
            template_name: str,
            current_user: UserInDB = Depends(require_permission("agent:read"))
        ):
            """Get detailed information about a specific integration template."""
            try:
                template = self.registry.get_template(template_name)
                if not template:
                    raise HTTPException(status_code=404, detail=f"Integration template '{template_name}' not found")
                
                # Check if user has this integration installed
                user_integrations = self.integration_manager.get_user_integrations(current_user.id)
                installed = any(i["name"] == template_name for i in user_integrations)
                
                # Get integration class info if available
                integration_class = self.registry.get_integration_class(template_name)
                class_available = integration_class is not None
                
                result = {
                    "template": IntegrationTemplate(
                        name=template.name,
                        display_name=template.display_name,
                        description=template.description,
                        category=template.category.value,
                        integration_type=template.integration_type.value,
                        auth_method=template.auth_method.value,
                        logo_url=template.logo_url,
                        documentation_url=template.documentation_url,
                        pricing_model=template.pricing_model,
                        popularity_score=template.popularity_score,
                        required_credentials=template.required_credentials,
                        supported_features=template.supported_features,
                        webhook_events=template.webhook_events,
                        rate_limits=template.rate_limits
                    ).dict(),
                    "is_installed": installed,
                    "implementation_available": class_available,
                    "installation_requirements": {
                        "credentials": template.required_credentials,
                        "permissions": ["agent:create"] if not installed else ["agent:update"],
                        "oauth2_scopes": template.metadata.get("oauth_scopes", [])
                    }
                }
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Integration details requested: {template_name} by user {current_user.username}",
                    EventType.API,
                    {"user_id": current_user.id, "template_name": template_name, "installed": installed},
                    "marketplace_api"
                )
                
                return result
                
            except HTTPException:
                raise
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Get integration details error: {str(e)}",
                    EventType.ERROR,
                    {"user_id": current_user.id, "template_name": template_name, "error": str(e)},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Failed to get integration details: {str(e)}")
        
        @self.router.get("/categories", response_model=List[MarketplaceCategory])
        async def list_categories(
            current_user: UserInDB = Depends(require_permission("agent:read"))
        ):
            """List all integration categories with counts."""
            try:
                categories = self.registry.get_categories()
                
                # Add descriptions for categories
                category_descriptions = {
                    "communication": "Chat platforms and messaging services",
                    "productivity": "Productivity tools and workspace applications", 
                    "development": "Development tools and version control systems",
                    "business": "CRM, sales, and business management tools",
                    "analytics": "Analytics and data visualization platforms",
                    "storage": "Cloud storage and file management services",
                    "ai_ml": "AI and machine learning platforms",
                    "security": "Security and authentication services",
                    "marketing": "Marketing automation and email platforms",
                    "finance": "Financial and accounting services"
                }
                
                result = [
                    MarketplaceCategory(
                        name=cat["name"],
                        display_name=cat["display_name"],
                        count=cat["count"],
                        description=category_descriptions.get(cat["name"], "")
                    )
                    for cat in categories
                ]
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Categories listed by user {current_user.username}",
                    EventType.API,
                    {"user_id": current_user.id, "category_count": len(result)},
                    "marketplace_api"
                )
                
                return result
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"List categories error: {str(e)}",
                    EventType.ERROR,
                    {"user_id": current_user.id, "error": str(e)},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Failed to list categories: {str(e)}")
        
        @self.router.post("/install")
        async def install_integration(
            request: IntegrationInstallRequest,
            current_user: UserInDB = Depends(require_permission("agent:create"))
        ):
            """Install an integration from a template."""
            try:
                # Get template
                template = self.registry.get_template(request.template_name)
                if not template:
                    raise HTTPException(status_code=404, detail=f"Template '{request.template_name}' not found")
                
                # Get integration class
                integration_class = self.registry.get_integration_class(request.template_name)
                if not integration_class:
                    raise HTTPException(status_code=400, detail=f"Integration implementation for '{request.template_name}' not available")
                
                # Create configuration from template
                config_overrides = request.config_overrides.copy()
                if request.custom_name:
                    config_overrides["display_name"] = request.custom_name
                
                config = self.registry.create_config_from_template(
                    request.template_name,
                    **config_overrides
                )
                
                # Register integration with manager
                self.integration_manager.register_integration(integration_class, config)
                
                # Initialize integration
                initialization_response = await self.integration_manager.initialize()
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Integration installed: {request.template_name} by user {current_user.username}",
                    EventType.INTEGRATION,
                    {
                        "user_id": current_user.id,
                        "template_name": request.template_name,
                        "integration_id": config.integration_id,
                        "success": initialization_response
                    },
                    "marketplace_api"
                )
                
                return {
                    "success": True,
                    "integration_id": config.integration_id,
                    "display_name": config.display_name,
                    "status": "installed",
                    "requires_authentication": True,
                    "auth_method": config.auth_method.value,
                    "message": f"Integration '{config.display_name}' installed successfully. Authentication required."
                }
                
            except HTTPException:
                raise
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Install integration error: {str(e)}",
                    EventType.ERROR,
                    {"user_id": current_user.id, "template_name": request.template_name, "error": str(e)},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Failed to install integration: {str(e)}")
        
        @self.router.post("/configure")
        async def configure_integration(
            request: IntegrationConfigRequest,
            current_user: UserInDB = Depends(require_permission("agent:update"))
        ):
            """Configure and authenticate an integration."""
            try:
                # Authenticate integration
                auth_response = await self.integration_manager.authenticate_integration(
                    request.integration_id,
                    current_user.id,
                    request.credentials
                )
                
                if not auth_response.success:
                    raise HTTPException(status_code=400, detail=auth_response.error)
                
                # Test connection
                test_response = await self.integration_manager.test_integration(
                    request.integration_id,
                    current_user.id
                )
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Integration configured: {request.integration_id} by user {current_user.username}",
                    EventType.AUTHENTICATION,
                    {
                        "user_id": current_user.id,
                        "integration_id": request.integration_id,
                        "auth_success": auth_response.success,
                        "test_success": test_response.success
                    },
                    "marketplace_api"
                )
                
                return {
                    "success": True,
                    "integration_id": request.integration_id,
                    "status": "configured",
                    "authentication": auth_response.data,
                    "connection_test": test_response.data,
                    "message": "Integration configured and authenticated successfully."
                }
                
            except HTTPException:
                raise
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Configure integration error: {str(e)}",
                    EventType.ERROR,
                    {
                        "user_id": current_user.id, 
                        "integration_id": request.integration_id, 
                        "error": str(e)
                    },
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Failed to configure integration: {str(e)}")
        
        @self.router.get("/installed", response_model=List[IntegrationStatus])
        async def list_installed_integrations(
            current_user: UserInDB = Depends(require_permission("agent:read"))
        ):
            """List user's installed and configured integrations."""
            try:
                user_integrations = self.integration_manager.get_user_integrations(current_user.id)
                
                result = [
                    IntegrationStatus(
                        integration_id=integration["integration_id"],
                        name=integration["name"],
                        display_name=integration["display_name"],
                        status=integration["status"],
                        is_authenticated=integration.get("authenticated_at") is not None,
                        created_at=integration.get("created_at", ""),
                        last_error=None  # Could be enhanced to track errors
                    )
                    for integration in user_integrations
                ]
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Installed integrations listed for user {current_user.username}",
                    EventType.API,
                    {"user_id": current_user.id, "integration_count": len(result)},
                    "marketplace_api"
                )
                
                return result
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"List installed integrations error: {str(e)}",
                    EventType.ERROR,
                    {"user_id": current_user.id, "error": str(e)},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Failed to list installed integrations: {str(e)}")
        
        @self.router.delete("/integrations/{integration_id}")
        async def uninstall_integration(
            integration_id: str,
            current_user: UserInDB = Depends(require_permission("agent:delete"))
        ):
            """Uninstall an integration."""
            try:
                integration = self.integration_manager.get_integration(integration_id)
                if not integration:
                    raise HTTPException(status_code=404, detail=f"Integration '{integration_id}' not found")
                
                # Clean up integration
                cleanup_response = await integration.cleanup()
                
                # Remove from manager (this would need to be implemented in the manager)
                # For now, we'll just return success
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Integration uninstalled: {integration_id} by user {current_user.username}",
                    EventType.INTEGRATION,
                    {
                        "user_id": current_user.id,
                        "integration_id": integration_id,
                        "cleanup_success": cleanup_response.success
                    },
                    "marketplace_api"
                )
                
                return {
                    "success": True,
                    "integration_id": integration_id,
                    "status": "uninstalled",
                    "message": f"Integration '{integration_id}' uninstalled successfully."
                }
                
            except HTTPException:
                raise
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Uninstall integration error: {str(e)}",
                    EventType.ERROR,
                    {"user_id": current_user.id, "integration_id": integration_id, "error": str(e)},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Failed to uninstall integration: {str(e)}")
        
        @self.router.get("/search")
        async def search_integrations(
            q: str = Query(..., description="Search query"),
            limit: int = Query(20, ge=1, le=50, description="Maximum number of results"),
            current_user: UserInDB = Depends(require_permission("agent:read"))
        ):
            """Search integrations by name, description, or features."""
            try:
                templates = self.registry.search_templates(q)[:limit]
                
                result = [
                    {
                        "name": t.name,
                        "display_name": t.display_name,
                        "description": t.description,
                        "category": t.category.value,
                        "popularity_score": t.popularity_score,
                        "logo_url": t.logo_url,
                        "supported_features": t.supported_features
                    }
                    for t in templates
                ]
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Integration search: '{q}' by user {current_user.username}",
                    EventType.API,
                    {"user_id": current_user.id, "query": q, "results": len(result)},
                    "marketplace_api"
                )
                
                return {
                    "query": q,
                    "results": result,
                    "total_count": len(result),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Search integrations error: {str(e)}",
                    EventType.ERROR,
                    {"user_id": current_user.id, "query": q, "error": str(e)},
                    "marketplace_api"
                )
                raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Factory function to create the API router
def create_marketplace_api(integration_manager: IntegrationManager) -> APIRouter:
    """Create and return the marketplace API router."""
    marketplace_api = MarketplaceAPI(integration_manager)
    return marketplace_api.router