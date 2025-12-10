# File: backend/integrations/manager.py
"""
Integration Manager - Orchestrates all third-party integrations.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Type
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from .base import (
    IntegrationBase, IntegrationConfig, IntegrationCredentials, 
    IntegrationResponse, IntegrationEvent, IntegrationStatus, IntegrationType
)
from .oauth_provider import OAuth2Provider
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel


class IntegrationManager:
    """
    Central manager for all third-party integrations.
    
    Handles integration lifecycle, authentication, and message routing.
    """
    
    def __init__(self, auth_service, base_url: str = "http://localhost:8000"):
        self.auth_service = auth_service
        self.base_url = base_url
        self.integrations: Dict[str, IntegrationBase] = {}
        self.integration_configs: Dict[str, IntegrationConfig] = {}
        self.user_credentials: Dict[str, Dict[str, IntegrationCredentials]] = {}  # user_id -> integration_id -> credentials
        self.oauth2_provider = OAuth2Provider(auth_service, base_url)
        self.webhook_endpoints: Dict[str, str] = {}  # integration_id -> webhook_url
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._is_initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize the integration manager and all registered integrations.
        
        Returns:
            True if initialization successful
        """
        try:
            uap_logger.log_event(
                LogLevel.INFO,
                "Initializing Integration Manager",
                EventType.SYSTEM,
                {"manager": "integrations"},
                "integration_manager"
            )
            
            # Initialize all registered integrations
            initialization_tasks = []
            for integration_id, integration in self.integrations.items():
                task = asyncio.create_task(
                    self._safe_initialize_integration(integration_id, integration)
                )
                initialization_tasks.append(task)
            
            if initialization_tasks:
                results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
                
                # Log results
                successful = sum(1 for r in results if r is True)
                failed = len(results) - successful
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Integration initialization complete: {successful} successful, {failed} failed",
                    EventType.SYSTEM,
                    {
                        "successful": successful,
                        "failed": failed,
                        "total": len(results)
                    },
                    "integration_manager"
                )
            
            self._is_initialized = True
            return True
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to initialize Integration Manager: {str(e)}",
                EventType.ERROR,
                {"error": str(e)},
                "integration_manager"
            )
            return False
    
    async def _safe_initialize_integration(self, integration_id: str, integration: IntegrationBase) -> bool:
        """
        Safely initialize a single integration with error handling.
        
        Args:
            integration_id: Integration identifier
            integration: Integration instance
            
        Returns:
            True if successful
        """
        try:
            response = await integration.initialize()
            if response.success:
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Integration {integration_id} initialized successfully",
                    EventType.INTEGRATION,
                    {"integration_id": integration_id, "status": "initialized"},
                    "integration_manager"
                )
                return True
            else:
                uap_logger.log_event(
                    LogLevel.ERROR,
                    f"Integration {integration_id} initialization failed: {response.error}",
                    EventType.ERROR,
                    {"integration_id": integration_id, "error": response.error},
                    "integration_manager"
                )
                return False
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Exception during {integration_id} initialization: {str(e)}",
                EventType.ERROR,
                {"integration_id": integration_id, "exception": str(e)},
                "integration_manager"
            )
            return False
    
    def register_integration(self, integration_class: Type[IntegrationBase], config: IntegrationConfig):
        """
        Register a new integration class with configuration.
        
        Args:
            integration_class: Integration class to register
            config: Integration configuration
        """
        integration_id = config.integration_id
        
        # Store configuration
        self.integration_configs[integration_id] = config
        
        # Create integration instance
        integration = integration_class(config)
        self.integrations[integration_id] = integration
        
        # Set up webhook endpoint if needed
        if config.webhook_config:
            webhook_path = f"/webhooks/{integration_id}"
            self.webhook_endpoints[integration_id] = f"{self.base_url}{webhook_path}"
        
        uap_logger.log_event(
            LogLevel.INFO,
            f"Registered integration: {integration_id} ({config.display_name})",
            EventType.INTEGRATION,
            {
                "integration_id": integration_id,
                "integration_type": config.integration_type,
                "auth_method": config.auth_method
            },
            "integration_manager"
        )
    
    def get_integration(self, integration_id: str) -> Optional[IntegrationBase]:
        """Get integration by ID."""
        return self.integrations.get(integration_id)
    
    def list_integrations(self, integration_type: IntegrationType = None) -> List[Dict[str, Any]]:
        """
        List all registered integrations with their status.
        
        Args:
            integration_type: Optional filter by integration type
            
        Returns:
            List of integration information
        """
        integrations = []
        for integration_id, integration in self.integrations.items():
            config = self.integration_configs[integration_id]
            
            if integration_type and config.integration_type != integration_type:
                continue
            
            integrations.append({
                "integration_id": integration_id,
                "name": config.name,
                "display_name": config.display_name,
                "description": config.description,
                "type": config.integration_type,
                "status": integration.status,
                "auth_method": config.auth_method,
                "is_authenticated": integration.is_authenticated,
                "webhook_url": self.webhook_endpoints.get(integration_id),
                "created_at": config.created_at.isoformat()
            })
        
        return integrations
    
    async def authenticate_integration(self, integration_id: str, user_id: str, 
                                     credentials: Dict[str, Any]) -> IntegrationResponse:
        """
        Authenticate a user with a specific integration.
        
        Args:
            integration_id: Integration to authenticate with
            user_id: User performing authentication
            credentials: Authentication credentials
            
        Returns:
            Authentication response
        """
        integration = self.get_integration(integration_id)
        if not integration:
            return IntegrationResponse(
                success=False,
                error=f"Integration {integration_id} not found",
                error_code="integration_not_found"
            )
        
        try:
            # Authenticate with the integration
            response = await integration.authenticate(credentials)
            
            if response.success:
                # Store credentials securely
                integration_creds = IntegrationCredentials(
                    integration_id=integration_id,
                    user_id=user_id,
                    credentials=credentials  # Should be encrypted in production
                )
                
                if user_id not in self.user_credentials:
                    self.user_credentials[user_id] = {}
                self.user_credentials[user_id][integration_id] = integration_creds
                
                # Set credentials on integration
                integration.set_credentials(integration_creds)
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"User {user_id} authenticated with integration {integration_id}",
                    EventType.AUTHENTICATION,
                    {
                        "user_id": user_id,
                        "integration_id": integration_id,
                        "success": True
                    },
                    "integration_manager"
                )
            
            return response
            
        except Exception as e:
            error_response = IntegrationResponse(
                success=False,
                error=f"Authentication failed: {str(e)}",
                error_code="authentication_error"
            )
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Authentication failed for user {user_id} with integration {integration_id}: {str(e)}",
                EventType.ERROR,
                {
                    "user_id": user_id,
                    "integration_id": integration_id,
                    "error": str(e)
                },
                "integration_manager"
            )
            
            return error_response
    
    async def send_message(self, integration_id: str, user_id: str, message: str, 
                          channel: str = None, **kwargs) -> IntegrationResponse:
        """
        Send a message through a specific integration.
        
        Args:
            integration_id: Integration to use
            user_id: User sending the message
            message: Message content
            channel: Target channel/room
            **kwargs: Additional platform-specific parameters
            
        Returns:
            Send response
        """
        integration = self.get_integration(integration_id)
        if not integration:
            return IntegrationResponse(
                success=False,
                error=f"Integration {integration_id} not found",
                error_code="integration_not_found"
            )
        
        # Check if user is authenticated with this integration
        if not self._is_user_authenticated(user_id, integration_id):
            return IntegrationResponse(
                success=False,
                error="User not authenticated with this integration",
                error_code="not_authenticated"
            )
        
        try:
            response = await integration.send_message(message, channel, **kwargs)
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Message sent via {integration_id} for user {user_id}",
                EventType.INTEGRATION,
                {
                    "user_id": user_id,
                    "integration_id": integration_id,
                    "channel": channel,
                    "success": response.success
                },
                "integration_manager"
            )
            
            return response
            
        except Exception as e:
            error_response = IntegrationResponse(
                success=False,
                error=f"Failed to send message: {str(e)}",
                error_code="send_failed"
            )
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to send message via {integration_id} for user {user_id}: {str(e)}",
                EventType.ERROR,
                {
                    "user_id": user_id,
                    "integration_id": integration_id,
                    "error": str(e)
                },
                "integration_manager"
            )
            
            return error_response
    
    async def handle_webhook(self, integration_id: str, payload: Dict[str, Any], 
                           headers: Dict[str, str]) -> IntegrationResponse:
        """
        Handle incoming webhook from a third-party integration.
        
        Args:
            integration_id: Integration that sent the webhook
            payload: Webhook payload
            headers: HTTP headers
            
        Returns:
            Processing response
        """
        integration = self.get_integration(integration_id)
        if not integration:
            return IntegrationResponse(
                success=False,
                error=f"Integration {integration_id} not found",
                error_code="integration_not_found"
            )
        
        try:
            # Create integration event
            event = IntegrationEvent(
                integration_id=integration_id,
                event_type=payload.get("type", "webhook"),
                source=integration.name,
                data=payload,
                metadata={"headers": headers}
            )
            
            # Process webhook
            response = await integration.receive_webhook(event)
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Webhook processed for integration {integration_id}",
                EventType.WEBHOOK,
                {
                    "integration_id": integration_id,
                    "event_type": event.event_type,
                    "success": response.success
                },
                "integration_manager"
            )
            
            return response
            
        except Exception as e:
            error_response = IntegrationResponse(
                success=False,
                error=f"Webhook processing failed: {str(e)}",
                error_code="webhook_failed"
            )
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Webhook processing failed for integration {integration_id}: {str(e)}",
                EventType.ERROR,
                {
                    "integration_id": integration_id,
                    "error": str(e)
                },
                "integration_manager"
            )
            
            return error_response
    
    async def test_integration(self, integration_id: str, user_id: str) -> IntegrationResponse:
        """
        Test connection to a specific integration.
        
        Args:
            integration_id: Integration to test
            user_id: User requesting the test
            
        Returns:
            Test response
        """
        integration = self.get_integration(integration_id)
        if not integration:
            return IntegrationResponse(
                success=False,
                error=f"Integration {integration_id} not found",
                error_code="integration_not_found"
            )
        
        try:
            response = await integration.test_connection()
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Integration test for {integration_id} by user {user_id}",
                EventType.INTEGRATION,
                {
                    "user_id": user_id,
                    "integration_id": integration_id,
                    "test_success": response.success
                },
                "integration_manager"
            )
            
            return response
            
        except Exception as e:
            error_response = IntegrationResponse(
                success=False,
                error=f"Connection test failed: {str(e)}",
                error_code="test_failed"
            )
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Integration test failed for {integration_id} by user {user_id}: {str(e)}",
                EventType.ERROR,
                {
                    "user_id": user_id,
                    "integration_id": integration_id,
                    "error": str(e)
                },
                "integration_manager"
            )
            
            return error_response
    
    def get_user_integrations(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all integrations authenticated by a specific user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of user's authenticated integrations
        """
        user_integrations = []
        user_creds = self.user_credentials.get(user_id, {})
        
        for integration_id, credentials in user_creds.items():
            integration = self.get_integration(integration_id)
            if integration:
                config = self.integration_configs[integration_id]
                user_integrations.append({
                    "integration_id": integration_id,
                    "name": config.name,
                    "display_name": config.display_name,
                    "type": config.integration_type,
                    "status": integration.status,
                    "authenticated_at": credentials.created_at.isoformat(),
                    "expires_at": credentials.expires_at.isoformat() if credentials.expires_at else None
                })
        
        return user_integrations
    
    def _is_user_authenticated(self, user_id: str, integration_id: str) -> bool:
        """Check if user is authenticated with a specific integration."""
        return (user_id in self.user_credentials and 
                integration_id in self.user_credentials[user_id])
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall integration system status.
        
        Returns:
            System status information
        """
        total_integrations = len(self.integrations)
        active_integrations = sum(1 for i in self.integrations.values() 
                                if i.status == IntegrationStatus.ACTIVE)
        error_integrations = sum(1 for i in self.integrations.values() 
                               if i.status == IntegrationStatus.ERROR)
        
        # Count authenticated users
        total_authenticated_users = len(self.user_credentials)
        total_user_integrations = sum(len(creds) for creds in self.user_credentials.values())
        
        # OAuth2 statistics
        oauth2_clients = len(self.oauth2_provider.clients)
        active_tokens = len(self.oauth2_provider.access_tokens)
        
        return {
            "initialized": self._is_initialized,
            "integrations": {
                "total": total_integrations,
                "active": active_integrations,
                "error": error_integrations,
                "types": list(set(config.integration_type for config in self.integration_configs.values()))
            },
            "authentication": {
                "authenticated_users": total_authenticated_users,
                "total_user_integrations": total_user_integrations,
                "oauth2_clients": oauth2_clients,
                "active_tokens": active_tokens
            },
            "webhooks": {
                "registered_endpoints": len(self.webhook_endpoints)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def cleanup(self):
        """Clean up integration manager resources."""
        try:
            # Clean up all integrations
            cleanup_tasks = []
            for integration in self.integrations.values():
                task = asyncio.create_task(integration.cleanup())
                cleanup_tasks.append(task)
            
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            # Clean up OAuth2 tokens
            self.oauth2_provider.cleanup_expired_tokens()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Integration Manager cleanup complete",
                EventType.SYSTEM,
                {"manager": "integrations"},
                "integration_manager"
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Error during integration manager cleanup: {str(e)}",
                EventType.ERROR,
                {"error": str(e)},
                "integration_manager"
            )