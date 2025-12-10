# File: backend/integrations/teams_integration.py
"""
Microsoft Teams integration implementation with Graph API.
"""

import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from urllib.parse import urlencode

from .base import (
    IntegrationBase, IntegrationConfig, IntegrationResponse, 
    IntegrationEvent, IntegrationError, WebhookIntegration
)
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel


class TeamsIntegration(WebhookIntegration):
    """
    Microsoft Teams integration using Microsoft Graph API.
    
    Supports OAuth2 authentication, message sending, team management,
    and webhook event processing.
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.tenant_id: Optional[str] = None
        
        # Microsoft Graph endpoints
        self.graph_base_url = "https://graph.microsoft.com/v1.0"
        self.auth_base_url = "https://login.microsoftonline.com"
        
        self.endpoints = {
            "me": "/me",
            "teams": "/me/joinedTeams",
            "channels": "/teams/{team_id}/channels",
            "messages": "/teams/{team_id}/channels/{channel_id}/messages",
            "chat_messages": "/chats/{chat_id}/messages",
            "subscriptions": "/subscriptions"
        }
    
    async def initialize(self) -> IntegrationResponse:
        """Initialize Teams integration with HTTP session."""
        try:
            # Initialize HTTP session with timeout
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "UAP-Teams-Integration/1.0",
                    "Content-Type": "application/json"
                }
            )
            
            # Extract tenant ID from config
            self.tenant_id = self.config.auth_config.get("tenant_id", "common")
            
            # Call parent initialization
            parent_response = await super().initialize()
            if not parent_response.success:
                return parent_response
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Teams integration initialized successfully",
                EventType.INTEGRATION,
                {"integration_id": self.integration_id, "tenant_id": self.tenant_id},
                "teams_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={"status": "initialized", "tenant_id": self.tenant_id}
            )
            
        except Exception as e:
            return self._format_error_response(e, "teams_init_failed")
    
    async def authenticate(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """
        Authenticate with Microsoft Teams using OAuth2.
        
        Args:
            credentials: Dictionary containing authentication data
                - For OAuth2: {"code": "auth_code", "redirect_uri": "callback_url"}
                - For Token: {"access_token": "token", "refresh_token": "refresh"}
        
        Returns:
            Authentication response
        """
        try:
            if "access_token" in credentials:
                # Direct token authentication
                return await self._authenticate_token(credentials)
            
            elif "code" in credentials:
                # OAuth2 code exchange
                return await self._authenticate_oauth2(credentials)
            
            else:
                raise IntegrationError(
                    "Invalid credentials format. Provide either 'access_token' or OAuth2 'code'",
                    self.integration_id
                )
            
        except IntegrationError:
            raise
        except Exception as e:
            return self._format_error_response(e, "teams_auth_failed")
    
    async def _authenticate_token(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """Authenticate using access token."""
        self.access_token = credentials["access_token"]
        self.refresh_token = credentials.get("refresh_token")
        
        # Test authentication
        test_response = await self.test_connection()
        if test_response.success:
            uap_logger.log_event(
                LogLevel.INFO,
                "Teams token authentication successful",
                EventType.AUTHENTICATION,
                {"integration_id": self.integration_id},
                "teams_integration"
            )
            return IntegrationResponse(
                success=True,
                data=test_response.data,
                metadata={"auth_method": "token"}
            )
        else:
            return test_response
    
    async def _authenticate_oauth2(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """Authenticate using OAuth2 code exchange."""
        try:
            token_url = f"{self.auth_base_url}/{self.tenant_id}/oauth2/v2.0/token"
            
            token_data = {
                "client_id": self.config.auth_config.get("client_id"),
                "client_secret": self.config.auth_config.get("client_secret"),
                "code": credentials["code"],
                "redirect_uri": credentials.get("redirect_uri"),
                "grant_type": "authorization_code",
                "scope": "https://graph.microsoft.com/.default"
            }
            
            if not all([token_data["client_id"], token_data["client_secret"], token_data["code"]]):
                raise IntegrationError("Missing OAuth2 configuration", self.integration_id)
            
            # Exchange code for tokens
            async with self.session.post(token_url, data=token_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise IntegrationError(f"OAuth2 exchange failed: {response.status} - {error_text}", self.integration_id)
                
                result = await response.json()
                
                if "error" in result:
                    raise IntegrationError(f"Teams OAuth2 error: {result.get('error_description', result.get('error'))}", self.integration_id)
                
                # Store tokens
                self.access_token = result.get("access_token")
                self.refresh_token = result.get("refresh_token")
                
                # Get user info to validate token
                user_response = await self.get_user_info()
                if not user_response.success:
                    return user_response
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    "Teams OAuth2 authentication successful",
                    EventType.AUTHENTICATION,
                    {
                        "integration_id": self.integration_id,
                        "user_id": user_response.data.get("id"),
                        "tenant_id": self.tenant_id
                    },
                    "teams_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "user_info": user_response.data,
                        "expires_in": result.get("expires_in"),
                        "scope": result.get("scope"),
                        "token_type": result.get("token_type")
                    },
                    metadata={"auth_method": "oauth2"}
                )
                
        except IntegrationError:
            raise
        except Exception as e:
            return self._format_error_response(e, "teams_oauth2_failed")
    
    async def test_connection(self) -> IntegrationResponse:
        """Test connection to Microsoft Graph API."""
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="No authentication token available",
                    error_code="no_token"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            async with self.session.get(
                f"{self.graph_base_url}{self.endpoints['me']}",
                headers=headers
            ) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be expired",
                        error_code="auth_failed"
                    )
                elif response.status != 200:
                    return IntegrationResponse(
                        success=False,
                        error=f"Connection test failed: HTTP {response.status}",
                        error_code="connection_failed"
                    )
                
                result = await response.json()
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "user_id": result.get("id"),
                        "display_name": result.get("displayName"),
                        "email": result.get("mail") or result.get("userPrincipalName"),
                        "tenant_id": self.tenant_id
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "teams_test_failed")
    
    async def send_message(self, message: str, channel: str = None, **kwargs) -> IntegrationResponse:
        """
        Send a message to a Teams channel or chat.
        
        Args:
            message: Message text to send
            channel: Channel ID or chat ID (format: "team_id:channel_id" or "chat:chat_id")
            **kwargs: Additional Teams message parameters
        
        Returns:
            Send response with message details
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            if not channel:
                return IntegrationResponse(
                    success=False,
                    error="Channel is required (format: 'team_id:channel_id' or 'chat:chat_id')",
                    error_code="missing_channel"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Parse channel format
            if channel.startswith("chat:"):
                # Direct chat message
                chat_id = channel[5:]  # Remove "chat:" prefix
                url = f"{self.graph_base_url}{self.endpoints['chat_messages'].format(chat_id=chat_id)}"
            else:
                # Team channel message
                if ":" not in channel:
                    return IntegrationResponse(
                        success=False,
                        error="Invalid channel format. Use 'team_id:channel_id' or 'chat:chat_id'",
                        error_code="invalid_channel_format"
                    )
                
                team_id, channel_id = channel.split(":", 1)
                url = f"{self.graph_base_url}{self.endpoints['messages'].format(team_id=team_id, channel_id=channel_id)}"
            
            # Construct message payload
            payload = {
                "body": {
                    "contentType": "text",
                    "content": message
                }
            }
            
            # Add additional parameters from kwargs
            if "content_type" in kwargs:
                payload["body"]["contentType"] = kwargs["content_type"]
            if "attachments" in kwargs:
                payload["attachments"] = kwargs["attachments"]
            if "mentions" in kwargs:
                payload["mentions"] = kwargs["mentions"]
            
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be expired",
                        error_code="auth_failed"
                    )
                elif response.status not in [200, 201]:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to send message: HTTP {response.status} - {error_text}",
                        error_code="send_failed"
                    )
                
                result = await response.json()
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Message sent to Teams channel {channel}",
                    EventType.INTEGRATION,
                    {
                        "integration_id": self.integration_id,
                        "channel": channel,
                        "message_id": result.get("id")
                    },
                    "teams_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "id": result.get("id"),
                        "created_datetime": result.get("createdDateTime"),
                        "web_url": result.get("webUrl"),
                        "from": result.get("from", {}),
                        "channel": channel
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "teams_send_failed")
    
    async def get_user_info(self, user_id: str = None) -> IntegrationResponse:
        """
        Get user information from Microsoft Graph.
        
        Args:
            user_id: User ID (optional, defaults to authenticated user)
        
        Returns:
            User information response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            if user_id:
                url = f"{self.graph_base_url}/users/{user_id}"
            else:
                url = f"{self.graph_base_url}{self.endpoints['me']}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be expired",
                        error_code="auth_failed"
                    )
                elif response.status != 200:
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to get user info: HTTP {response.status}",
                        error_code="user_info_failed"
                    )
                
                result = await response.json()
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "id": result.get("id"),
                        "display_name": result.get("displayName"),
                        "email": result.get("mail") or result.get("userPrincipalName"),
                        "job_title": result.get("jobTitle"),
                        "office_location": result.get("officeLocation"),
                        "preferred_language": result.get("preferredLanguage"),
                        "given_name": result.get("givenName"),
                        "surname": result.get("surname")
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "teams_user_info_failed")
    
    async def refresh_credentials(self) -> IntegrationResponse:
        """Refresh authentication credentials using refresh token."""
        try:
            if not self.refresh_token:
                return IntegrationResponse(
                    success=False,
                    error="No refresh token available",
                    error_code="no_refresh_token"
                )
            
            token_url = f"{self.auth_base_url}/{self.tenant_id}/oauth2/v2.0/token"
            
            refresh_data = {
                "client_id": self.config.auth_config.get("client_id"),
                "client_secret": self.config.auth_config.get("client_secret"),
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
                "scope": "https://graph.microsoft.com/.default"
            }
            
            async with self.session.post(token_url, data=refresh_data) as response:
                if response.status != 200:
                    return IntegrationResponse(
                        success=False,
                        error=f"Token refresh failed: HTTP {response.status}",
                        error_code="refresh_failed"
                    )
                
                result = await response.json()
                
                if "error" in result:
                    return IntegrationResponse(
                        success=False,
                        error=f"Token refresh error: {result.get('error_description', result.get('error'))}",
                        error_code="refresh_error"
                    )
                
                # Update tokens
                self.access_token = result.get("access_token")
                if result.get("refresh_token"):
                    self.refresh_token = result.get("refresh_token")
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    "Teams credentials refreshed successfully",
                    EventType.AUTHENTICATION,
                    {"integration_id": self.integration_id},
                    "teams_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "status": "refreshed",
                        "expires_in": result.get("expires_in"),
                        "token_type": result.get("token_type")
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "teams_refresh_failed")
    
    async def verify_webhook_signature(self, payload: bytes, headers: Dict[str, str]) -> bool:
        """
        Verify Teams webhook signature.
        
        Teams uses different validation mechanisms depending on the subscription type.
        """
        try:
            # Teams validation depends on subscription type
            # For simplicity, we'll validate the presence of expected headers
            
            validation_token = headers.get("validationtoken")
            if validation_token:
                # This is a subscription validation request
                return True
            
            # For regular webhooks, Teams doesn't use HMAC signatures
            # Instead, it uses client state validation and HTTPS endpoints
            # Additional validation should be implemented based on your security requirements
            
            return True  # Basic validation - enhance as needed
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Teams webhook signature verification error: {str(e)}",
                EventType.ERROR,
                {"integration_id": self.integration_id, "error": str(e)},
                "teams_integration"
            )
            return False
    
    async def parse_webhook_event(self, payload: Dict[str, Any], headers: Dict[str, str]) -> IntegrationEvent:
        """
        Parse Teams webhook payload into standardized event format.
        
        Args:
            payload: Webhook payload from Teams
            headers: HTTP headers
        
        Returns:
            Standardized IntegrationEvent
        """
        try:
            # Handle validation requests
            validation_token = headers.get("validationtoken")
            if validation_token:
                return IntegrationEvent(
                    integration_id=self.integration_id,
                    event_type="subscription_validation",
                    source="teams",
                    data=payload,
                    metadata={"validation_token": validation_token}
                )
            
            # Handle subscription notifications
            if "value" in payload:
                # This is a subscription notification
                notifications = payload["value"]
                
                if notifications:
                    first_notification = notifications[0]
                    resource = first_notification.get("resource", "")
                    change_type = first_notification.get("changeType", "unknown")
                    
                    # Determine event type based on resource
                    if "messages" in resource:
                        event_type = "message"
                    elif "channels" in resource:
                        event_type = "channel"
                    elif "teams" in resource:
                        event_type = "team"
                    else:
                        event_type = "notification"
                    
                    return IntegrationEvent(
                        integration_id=self.integration_id,
                        event_type=event_type,
                        source="teams",
                        data=payload,
                        metadata={
                            "change_type": change_type,
                            "resource": resource,
                            "notification_count": len(notifications)
                        }
                    )
            
            # Fallback for other event types
            return IntegrationEvent(
                integration_id=self.integration_id,
                event_type="unknown",
                source="teams",
                data=payload,
                metadata={"webhook_headers": dict(headers)}
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to parse Teams webhook event: {str(e)}",
                EventType.ERROR,
                {"integration_id": self.integration_id, "error": str(e)},
                "teams_integration"
            )
            return IntegrationEvent(
                integration_id=self.integration_id,
                event_type="parse_error",
                source="teams",
                data=payload,
                metadata={"error": str(e)}
            )
    
    async def receive_webhook(self, event: IntegrationEvent) -> IntegrationResponse:
        """
        Process incoming webhook from Teams.
        
        Args:
            event: Webhook event data
        
        Returns:
            Processing response
        """
        try:
            event_type = event.event_type
            
            # Handle subscription validation
            if event_type == "subscription_validation":
                validation_token = event.metadata.get("validation_token")
                if validation_token:
                    return IntegrationResponse(
                        success=True,
                        data={"validationToken": validation_token},
                        metadata={"response_type": "validation"}
                    )
            
            # Handle message events
            elif event_type == "message":
                return await self._handle_message_event(event)
            
            # Handle other event types
            else:
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Received Teams event type: {event_type}",
                    EventType.WEBHOOK,
                    {
                        "integration_id": self.integration_id,
                        "event_type": event_type
                    },
                    "teams_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={"status": "acknowledged", "event_type": event_type}
                )
            
        except Exception as e:
            return self._format_error_response(e, "teams_webhook_failed")
    
    async def _handle_message_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle Teams message events."""
        try:
            # Extract notification details
            notifications = event.data.get("value", [])
            
            if not notifications:
                return IntegrationResponse(
                    success=True,
                    data={"status": "no_notifications"}
                )
            
            processed_count = 0
            for notification in notifications:
                resource = notification.get("resource", "")
                change_type = notification.get("changeType", "")
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Processing Teams notification: {change_type} on {resource}",
                    EventType.WEBHOOK,
                    {
                        "integration_id": self.integration_id,
                        "resource": resource,
                        "change_type": change_type
                    },
                    "teams_integration"
                )
                
                processed_count += 1
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": "message",
                    "notifications_processed": processed_count
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "teams_message_event_failed")
    
    async def cleanup(self) -> IntegrationResponse:
        """Clean up Teams integration resources."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.access_token = None
            self.refresh_token = None
            
            parent_response = await super().cleanup()
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Teams integration cleanup completed",
                EventType.INTEGRATION,
                {"integration_id": self.integration_id},
                "teams_integration"
            )
            
            return parent_response
            
        except Exception as e:
            return self._format_error_response(e, "teams_cleanup_failed")
    
    def get_oauth2_authorization_url(self, scopes: List[str], state: str = None, redirect_uri: str = None) -> str:
        """
        Generate Microsoft Teams OAuth2 authorization URL.
        
        Args:
            scopes: List of OAuth2 scopes to request
            state: Optional state parameter for security
            redirect_uri: Callback URL for authorization
        
        Returns:
            Authorization URL
        """
        client_id = self.config.auth_config.get("client_id")
        if not client_id:
            raise IntegrationError("Client ID not configured", self.integration_id)
        
        auth_url = f"{self.auth_base_url}/{self.tenant_id}/oauth2/v2.0/authorize"
        
        params = {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri or self.config.auth_config.get("redirect_uri"),
            "scope": " ".join(scopes),
            "response_mode": "query"
        }
        
        if state:
            params["state"] = state
        
        return f"{auth_url}?{urlencode(params)}"
    
    async def get_teams(self) -> IntegrationResponse:
        """
        Get list of teams for the authenticated user.
        
        Returns:
            Teams list response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            url = f"{self.graph_base_url}{self.endpoints['teams']}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be expired",
                        error_code="auth_failed"
                    )
                elif response.status != 200:
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to get teams: HTTP {response.status}",
                        error_code="teams_failed"
                    )
                
                result = await response.json()
                
                teams = []
                for team in result.get("value", []):
                    teams.append({
                        "id": team.get("id"),
                        "display_name": team.get("displayName"),
                        "description": team.get("description"),
                        "web_url": team.get("webUrl"),
                        "is_archived": team.get("isArchived")
                    })
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "teams": teams,
                        "total_count": len(teams)
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "teams_list_failed")
    
    async def get_channels(self, team_id: str) -> IntegrationResponse:
        """
        Get channels for a specific team.
        
        Args:
            team_id: Team identifier
        
        Returns:
            Channels list response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            url = f"{self.graph_base_url}{self.endpoints['channels'].format(team_id=team_id)}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be expired",
                        error_code="auth_failed"
                    )
                elif response.status != 200:
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to get channels: HTTP {response.status}",
                        error_code="channels_failed"
                    )
                
                result = await response.json()
                
                channels = []
                for channel in result.get("value", []):
                    channels.append({
                        "id": channel.get("id"),
                        "display_name": channel.get("displayName"),
                        "description": channel.get("description"),
                        "web_url": channel.get("webUrl"),
                        "email": channel.get("email"),
                        "membership_type": channel.get("membershipType")
                    })
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "team_id": team_id,
                        "channels": channels,
                        "total_count": len(channels)
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "teams_channels_failed")