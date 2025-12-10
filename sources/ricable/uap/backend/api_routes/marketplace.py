# File: backend/api_routes/marketplace.py
"""
API routes for the integration marketplace.

This module provides REST endpoints for plugin and integration management.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from typing import Optional, List

from ..plugins.plugin_manager import PluginManager
from ..plugins.marketplace.api import create_marketplace_router
from ..integrations.manager import IntegrationManager
from ..services.auth import get_current_user
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel


def create_marketplace_routes(plugin_manager: PluginManager, integration_manager: IntegrationManager) -> APIRouter:
    """
    Create marketplace API routes.
    
    Args:
        plugin_manager: Plugin manager instance
        integration_manager: Integration manager instance
        
    Returns:
        Configured APIRouter
    """
    # Get the marketplace router from the marketplace API
    marketplace_router = create_marketplace_router(plugin_manager, integration_manager)
    
    # Add additional routes specific to the main application
    @marketplace_router.get("/health")
    async def marketplace_health():
        """Health check for marketplace system."""
        return {
            "status": "healthy",
            "plugin_manager_active": plugin_manager._is_initialized,
            "integration_manager_active": integration_manager._is_initialized,
            "timestamp": "2024-12-28T12:00:00Z"
        }
    
    @marketplace_router.post("/plugins/{plugin_id}/test")
    async def test_plugin(
        plugin_id: str,
        background_tasks: BackgroundTasks,
        current_user=Depends(get_current_user)
    ):
        """Test a plugin's functionality."""
        try:
            plugin = plugin_manager.get_plugin(plugin_id)
            if not plugin:
                raise HTTPException(status_code=404, detail="Plugin not found")
            
            # Run a basic test
            from ..plugins.plugin_base import PluginContext
            
            context = PluginContext(
                plugin_id=plugin_id,
                instance_id=plugin.config.instance_id,
                user_id=current_user.get("user_id"),
                permissions=plugin.config.permissions
            )
            
            # Execute a test action
            response = await plugin_manager.execute_plugin_action(
                plugin_id, "get_status", {}, context
            )
            
            return {
                "test_successful": response.success,
                "response": response.data if response.success else response.error,
                "plugin_status": plugin.status
            }
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Plugin test failed: {str(e)}",
                EventType.ERROR,
                {"plugin_id": plugin_id, "error": str(e)},
                "marketplace_api"
            )
            raise HTTPException(status_code=500, detail="Plugin test failed")
    
    return marketplace_router