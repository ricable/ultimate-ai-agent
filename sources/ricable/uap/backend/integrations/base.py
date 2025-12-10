# File: backend/integrations/base.py
"""
Base classes and interfaces for third-party integrations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid
import asyncio


class IntegrationStatus(str, Enum):
    """Integration status enumeration"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"
    AUTHENTICATING = "authenticating"
    CONFIGURING = "configuring"


class IntegrationType(str, Enum):
    """Types of integrations"""
    CHAT = "chat"  # Slack, Teams, Discord
    PRODUCTIVITY = "productivity"  # Notion, Airtable, Trello
    DEVELOPMENT = "development"  # GitHub, GitLab, Jira
    CRM = "crm"  # Salesforce, HubSpot
    EMAIL = "email"  # Gmail, Outlook
    STORAGE = "storage"  # Dropbox, Google Drive
    ANALYTICS = "analytics"  # Google Analytics, Mixpanel
    CUSTOM = "custom"  # Custom integrations


class IntegrationError(Exception):
    """Base exception for integration errors"""
    def __init__(self, message: str, integration_id: str = None, error_code: str = None):
        super().__init__(message)
        self.integration_id = integration_id
        self.error_code = error_code


class AuthMethod(str, Enum):
    """Authentication methods for integrations"""
    OAUTH2 = "oauth2"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    WEBHOOK_SIGNATURE = "webhook_signature"


class IntegrationConfig(BaseModel):
    """Configuration for an integration"""
    integration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    display_name: str
    description: str
    integration_type: IntegrationType
    auth_method: AuthMethod
    base_url: str
    api_version: Optional[str] = None
    auth_config: Dict[str, Any] = Field(default_factory=dict)
    webhook_config: Dict[str, Any] = Field(default_factory=dict)
    rate_limits: Dict[str, int] = Field(default_factory=dict)
    required_permissions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IntegrationCredentials(BaseModel):
    """Secure storage for integration credentials"""
    integration_id: str
    user_id: str
    credentials: Dict[str, Any]  # Encrypted in production
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IntegrationEvent(BaseModel):
    """Event structure for integration communications"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    integration_id: str
    event_type: str
    source: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntegrationResponse(BaseModel):
    """Standard response format for integration operations"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IntegrationBase(ABC):
    """
    Abstract base class for all third-party integrations.
    
    This class defines the interface that all integrations must implement.
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.status = IntegrationStatus.INACTIVE
        self._credentials: Optional[IntegrationCredentials] = None
        self._last_error: Optional[str] = None
        self._connection_pool = None
        
    @property
    def integration_id(self) -> str:
        return self.config.integration_id
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def display_name(self) -> str:
        return self.config.display_name
    
    @property
    def integration_type(self) -> IntegrationType:
        return self.config.integration_type
    
    @property
    def is_authenticated(self) -> bool:
        return self._credentials is not None and self.status == IntegrationStatus.ACTIVE
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """
        Authenticate with the third-party service.
        
        Args:
            credentials: Authentication credentials (API keys, tokens, etc.)
            
        Returns:
            IntegrationResponse indicating success/failure
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> IntegrationResponse:
        """
        Test the connection to the third-party service.
        
        Returns:
            IntegrationResponse with connection status
        """
        pass
    
    @abstractmethod
    async def send_message(self, message: str, channel: str = None, **kwargs) -> IntegrationResponse:
        """
        Send a message through the integration.
        
        Args:
            message: Message content to send
            channel: Target channel/room/conversation
            **kwargs: Additional platform-specific parameters
            
        Returns:
            IntegrationResponse with send status
        """
        pass
    
    @abstractmethod
    async def receive_webhook(self, event: IntegrationEvent) -> IntegrationResponse:
        """
        Process incoming webhook from the third-party service.
        
        Args:
            event: Webhook event data
            
        Returns:
            IntegrationResponse with processing status
        """
        pass
    
    @abstractmethod
    async def get_user_info(self, user_id: str = None) -> IntegrationResponse:
        """
        Get user information from the third-party service.
        
        Args:
            user_id: Optional specific user ID, defaults to authenticated user
            
        Returns:
            IntegrationResponse with user data
        """
        pass
    
    @abstractmethod
    async def refresh_credentials(self) -> IntegrationResponse:
        """
        Refresh authentication credentials if supported.
        
        Returns:
            IntegrationResponse with refresh status
        """
        pass
    
    async def initialize(self) -> IntegrationResponse:
        """
        Initialize the integration (common setup tasks).
        
        Returns:
            IntegrationResponse with initialization status
        """
        try:
            self.status = IntegrationStatus.CONFIGURING
            
            # Validate configuration
            if not self.config.base_url:
                raise IntegrationError("Base URL is required", self.integration_id)
            
            # Initialize connection pool if needed
            await self._initialize_connection_pool()
            
            self.status = IntegrationStatus.INACTIVE
            return IntegrationResponse(
                success=True,
                data={"status": self.status, "integration_id": self.integration_id}
            )
            
        except Exception as e:
            self.status = IntegrationStatus.ERROR
            self._last_error = str(e)
            return IntegrationResponse(
                success=False,
                error=str(e),
                error_code="initialization_failed"
            )
    
    async def cleanup(self) -> IntegrationResponse:
        """
        Clean up integration resources.
        
        Returns:
            IntegrationResponse with cleanup status
        """
        try:
            # Close connection pool
            if self._connection_pool:
                await self._close_connection_pool()
            
            self.status = IntegrationStatus.INACTIVE
            self._credentials = None
            
            return IntegrationResponse(
                success=True,
                data={"status": "cleaned_up"}
            )
            
        except Exception as e:
            return IntegrationResponse(
                success=False,
                error=str(e),
                error_code="cleanup_failed"
            )
    
    def set_credentials(self, credentials: IntegrationCredentials):
        """Set authentication credentials for the integration."""
        self._credentials = credentials
        if credentials:
            self.status = IntegrationStatus.ACTIVE
        else:
            self.status = IntegrationStatus.INACTIVE
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current integration status and health information.
        
        Returns:
            Dictionary with status information
        """
        return {
            "integration_id": self.integration_id,
            "name": self.name,
            "display_name": self.display_name,
            "type": self.integration_type,
            "status": self.status,
            "authenticated": self.is_authenticated,
            "last_error": self._last_error,
            "config": {
                "auth_method": self.config.auth_method,
                "base_url": self.config.base_url,
                "api_version": self.config.api_version
            },
            "credentials_valid": self._credentials is not None,
            "credentials_expires": self._credentials.expires_at.isoformat() if self._credentials and self._credentials.expires_at else None
        }
    
    async def _initialize_connection_pool(self):
        """Initialize HTTP connection pool for the integration."""
        # Implementation would depend on HTTP client library (aiohttp, httpx, etc.)
        pass
    
    async def _close_connection_pool(self):
        """Close HTTP connection pool."""
        # Implementation would depend on HTTP client library
        pass
    
    def _validate_webhook_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """
        Validate webhook signature for security.
        
        Args:
            payload: Raw webhook payload
            signature: Provided signature
            secret: Webhook secret
            
        Returns:
            True if signature is valid
        """
        import hmac
        import hashlib
        
        expected_signature = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    async def _handle_rate_limit(self, remaining: int = None, reset_time: datetime = None):
        """
        Handle rate limiting for API calls.
        
        Args:
            remaining: Number of requests remaining
            reset_time: When rate limit resets
        """
        if remaining is not None and remaining <= 0:
            if reset_time:
                sleep_time = (reset_time - datetime.now(timezone.utc)).total_seconds()
                if sleep_time > 0:
                    await asyncio.sleep(min(sleep_time, 60))  # Max 1 minute wait
    
    def _format_error_response(self, error: Exception, error_code: str = None) -> IntegrationResponse:
        """
        Format error response consistently.
        
        Args:
            error: Exception that occurred
            error_code: Optional error code
            
        Returns:
            Formatted IntegrationResponse
        """
        self._last_error = str(error)
        return IntegrationResponse(
            success=False,
            error=str(error),
            error_code=error_code or "integration_error",
            metadata={
                "integration_id": self.integration_id,
                "integration_name": self.name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


class WebhookIntegration(IntegrationBase):
    """
    Base class for webhook-based integrations.
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.webhook_url = config.webhook_config.get("url")
        self.webhook_secret = config.webhook_config.get("secret")
    
    @abstractmethod
    async def verify_webhook_signature(self, payload: bytes, headers: Dict[str, str]) -> bool:
        """
        Verify webhook signature using platform-specific method.
        
        Args:
            payload: Raw webhook payload
            headers: HTTP headers from webhook request
            
        Returns:
            True if signature is valid
        """
        pass
    
    @abstractmethod
    async def parse_webhook_event(self, payload: Dict[str, Any], headers: Dict[str, str]) -> IntegrationEvent:
        """
        Parse incoming webhook payload into standardized event format.
        
        Args:
            payload: Webhook payload
            headers: HTTP headers
            
        Returns:
            Standardized IntegrationEvent
        """
        pass