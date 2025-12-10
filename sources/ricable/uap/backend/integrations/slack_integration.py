# File: backend/integrations/slack_integration.py
"""
Slack integration implementation with real API functionality.
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


class SlackIntegration(WebhookIntegration):
    """
    Slack integration with full API support.
    
    Supports OAuth2 authentication, message sending, channel management,
    and webhook event processing.
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.bot_token: Optional[str] = None
        self.user_token: Optional[str] = None
        
        # Slack API endpoints
        self.endpoints = {
            "auth_test": "/auth.test",
            "chat_post_message": "/chat.postMessage",
            "conversations_list": "/conversations.list",
            "conversations_info": "/conversations.info",
            "users_info": "/users.info",
            "files_upload": "/files.upload",
            "oauth_access": "/oauth.v2.access"
        }
    
    async def initialize(self) -> IntegrationResponse:
        """Initialize Slack integration with HTTP session."""
        try:
            # Initialize HTTP session with timeout
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "UAP-Slack-Integration/1.0",
                    "Content-Type": "application/json"
                }
            )
            
            # Call parent initialization
            parent_response = await super().initialize()
            if not parent_response.success:
                return parent_response
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Slack integration initialized successfully",
                EventType.INTEGRATION,
                {"integration_id": self.integration_id},
                "slack_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={"status": "initialized", "endpoints": len(self.endpoints)}
            )
            
        except Exception as e:
            return self._format_error_response(e, "slack_init_failed")
    
    async def authenticate(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """
        Authenticate with Slack using OAuth2 or bot token.
        
        Args:
            credentials: Dictionary containing authentication data
                - For OAuth2: {"code": "auth_code", "redirect_uri": "callback_url"}
                - For Bot Token: {"bot_token": "xoxb-..."}
        
        Returns:
            Authentication response
        """
        try:
            if "bot_token" in credentials:
                # Direct bot token authentication
                return await self._authenticate_bot_token(credentials["bot_token"])
            
            elif "code" in credentials:
                # OAuth2 code exchange
                return await self._authenticate_oauth2(credentials)
            
            else:
                raise IntegrationError(
                    "Invalid credentials format. Provide either 'bot_token' or OAuth2 'code'",
                    self.integration_id
                )
            
        except IntegrationError:
            raise
        except Exception as e:
            return self._format_error_response(e, "slack_auth_failed")
    
    async def _authenticate_bot_token(self, bot_token: str) -> IntegrationResponse:
        """Authenticate using bot token."""
        self.bot_token = bot_token
        
        # Test authentication
        test_response = await self.test_connection()
        if test_response.success:
            uap_logger.log_event(
                LogLevel.INFO,
                "Slack bot token authentication successful",
                EventType.AUTHENTICATION,
                {"integration_id": self.integration_id},
                "slack_integration"
            )
            return IntegrationResponse(
                success=True,
                data=test_response.data,
                metadata={"auth_method": "bot_token"}
            )
        else:
            return test_response
    
    async def _authenticate_oauth2(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """Authenticate using OAuth2 code exchange."""
        try:
            auth_data = {
                "client_id": self.config.auth_config.get("client_id"),
                "client_secret": self.config.auth_config.get("client_secret"),
                "code": credentials["code"],
                "redirect_uri": credentials.get("redirect_uri")
            }
            
            if not all([auth_data["client_id"], auth_data["client_secret"], auth_data["code"]]):
                raise IntegrationError("Missing OAuth2 configuration", self.integration_id)
            
            # Exchange code for tokens
            async with self.session.post(
                f"{self.config.base_url}{self.endpoints['oauth_access']}",
                data=auth_data
            ) as response:
                if response.status != 200:
                    raise IntegrationError(f"OAuth2 exchange failed: {response.status}", self.integration_id)
                
                result = await response.json()
                
                if not result.get("ok"):
                    raise IntegrationError(f"Slack OAuth2 error: {result.get('error')}", self.integration_id)
                
                # Store tokens
                self.bot_token = result.get("access_token")
                self.user_token = result.get("authed_user", {}).get("access_token")
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    "Slack OAuth2 authentication successful",
                    EventType.AUTHENTICATION,
                    {
                        "integration_id": self.integration_id,
                        "team_id": result.get("team", {}).get("id"),
                        "bot_user_id": result.get("bot_user_id")
                    },
                    "slack_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "team_name": result.get("team", {}).get("name"),
                        "bot_user_id": result.get("bot_user_id"),
                        "scope": result.get("scope"),
                        "token_type": result.get("token_type")
                    },
                    metadata={"auth_method": "oauth2"}
                )
                
        except IntegrationError:
            raise
        except Exception as e:
            return self._format_error_response(e, "slack_oauth2_failed")
    
    async def test_connection(self) -> IntegrationResponse:
        """Test connection to Slack API."""
        try:
            if not self.bot_token:
                return IntegrationResponse(
                    success=False,
                    error="No authentication token available",
                    error_code="no_token"
                )
            
            headers = {"Authorization": f"Bearer {self.bot_token}"}
            
            async with self.session.get(
                f"{self.config.base_url}{self.endpoints['auth_test']}",
                headers=headers
            ) as response:
                if response.status != 200:
                    return IntegrationResponse(
                        success=False,
                        error=f"Connection test failed: HTTP {response.status}",
                        error_code="connection_failed"
                    )
                
                result = await response.json()
                
                if not result.get("ok"):
                    return IntegrationResponse(
                        success=False,
                        error=f"Slack API error: {result.get('error')}",
                        error_code="api_error"
                    )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "team": result.get("team"),
                        "user": result.get("user"),
                        "bot_id": result.get("bot_id"),
                        "url": result.get("url")
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "slack_test_failed")
    
    async def send_message(self, message: str, channel: str = None, **kwargs) -> IntegrationResponse:
        """
        Send a message to a Slack channel.
        
        Args:
            message: Message text to send
            channel: Channel ID or name (e.g., "#general", "C1234567890")
            **kwargs: Additional Slack message parameters (blocks, attachments, etc.)
        
        Returns:
            Send response with message details
        """
        try:
            if not self.bot_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            if not channel:
                return IntegrationResponse(
                    success=False,
                    error="Channel is required",
                    error_code="missing_channel"
                )
            
            headers = {"Authorization": f"Bearer {self.bot_token}"}
            
            payload = {
                "channel": channel,
                "text": message,
                **kwargs  # Allow additional Slack parameters
            }
            
            async with self.session.post(
                f"{self.config.base_url}{self.endpoints['chat_post_message']}",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                
                if not result.get("ok"):
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to send message: {result.get('error')}",
                        error_code="send_failed"
                    )
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Message sent to Slack channel {channel}",
                    EventType.INTEGRATION,
                    {
                        "integration_id": self.integration_id,
                        "channel": channel,
                        "message_ts": result.get("ts")
                    },
                    "slack_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "channel": result.get("channel"),
                        "timestamp": result.get("ts"),
                        "message": result.get("message", {}),
                        "permalink": f"https://{result.get('message', {}).get('team')}.slack.com/archives/{channel}/p{result.get('ts', '').replace('.', '')}"
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "slack_send_failed")
    
    async def get_user_info(self, user_id: str = None) -> IntegrationResponse:
        """
        Get user information from Slack.
        
        Args:
            user_id: Slack user ID (optional, defaults to authenticated user)
        
        Returns:
            User information response
        """
        try:
            if not self.bot_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.bot_token}"}
            
            if user_id:
                url = f"{self.config.base_url}{self.endpoints['users_info']}?user={user_id}"
            else:
                # Get info about the bot user
                auth_response = await self.test_connection()
                if not auth_response.success:
                    return auth_response
                user_id = auth_response.data.get("user")
                url = f"{self.config.base_url}{self.endpoints['users_info']}?user={user_id}"
            
            async with self.session.get(url, headers=headers) as response:
                result = await response.json()
                
                if not result.get("ok"):
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to get user info: {result.get('error')}",
                        error_code="user_info_failed"
                    )
                
                user = result.get("user", {})
                return IntegrationResponse(
                    success=True,
                    data={
                        "id": user.get("id"),
                        "name": user.get("name"),
                        "real_name": user.get("real_name"),
                        "display_name": user.get("profile", {}).get("display_name"),
                        "email": user.get("profile", {}).get("email"),
                        "is_bot": user.get("is_bot"),
                        "team_id": user.get("team_id")
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "slack_user_info_failed")
    
    async def refresh_credentials(self) -> IntegrationResponse:
        """
        Refresh authentication credentials.
        
        Note: Slack bot tokens don't expire, but user tokens may need refresh.
        """
        try:
            # Test current credentials
            test_response = await self.test_connection()
            if test_response.success:
                return IntegrationResponse(
                    success=True,
                    data={"status": "credentials_valid", "refresh_needed": False}
                )
            else:
                return IntegrationResponse(
                    success=False,
                    error="Credentials need manual refresh",
                    error_code="manual_refresh_required",
                    data={"refresh_needed": True}
                )
                
        except Exception as e:
            return self._format_error_response(e, "slack_refresh_failed")
    
    async def verify_webhook_signature(self, payload: bytes, headers: Dict[str, str]) -> bool:
        """
        Verify Slack webhook signature.
        
        Args:
            payload: Raw webhook payload
            headers: HTTP headers from webhook request
        
        Returns:
            True if signature is valid
        """
        try:
            import hmac
            import hashlib
            
            timestamp = headers.get("X-Slack-Request-Timestamp")
            signature = headers.get("X-Slack-Signature")
            
            if not timestamp or not signature:
                return False
            
            # Check timestamp (prevent replay attacks)
            try:
                request_time = int(timestamp)
                if abs(datetime.now().timestamp() - request_time) > 300:  # 5 minutes
                    return False
            except ValueError:
                return False
            
            # Verify signature
            if not self.webhook_secret:
                uap_logger.log_event(
                    LogLevel.WARNING,
                    "Slack webhook secret not configured",
                    EventType.SECURITY,
                    {"integration_id": self.integration_id},
                    "slack_integration"
                )
                return False
            
            sig_basestring = f"v0:{timestamp}:{payload.decode('utf-8')}"
            expected_signature = "v0=" + hmac.new(
                self.webhook_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Slack webhook signature verification error: {str(e)}",
                EventType.ERROR,
                {"integration_id": self.integration_id, "error": str(e)},
                "slack_integration"
            )
            return False
    
    async def parse_webhook_event(self, payload: Dict[str, Any], headers: Dict[str, str]) -> IntegrationEvent:
        """
        Parse Slack webhook payload into standardized event format.
        
        Args:
            payload: Webhook payload from Slack
            headers: HTTP headers
        
        Returns:
            Standardized IntegrationEvent
        """
        try:
            # Handle URL verification challenge
            if payload.get("type") == "url_verification":
                return IntegrationEvent(
                    integration_id=self.integration_id,
                    event_type="url_verification",
                    source="slack",
                    data=payload,
                    metadata={"challenge": payload.get("challenge")}
                )
            
            # Handle regular events
            event_data = payload.get("event", {})
            event_type = event_data.get("type", "unknown")
            
            # Extract common event information
            parsed_data = {
                "team_id": payload.get("team_id"),
                "api_app_id": payload.get("api_app_id"),
                "event": event_data,
                "event_id": payload.get("event_id"),
                "event_time": payload.get("event_time")
            }
            
            # Add event-specific metadata
            metadata = {
                "channel": event_data.get("channel"),
                "user": event_data.get("user"),
                "timestamp": event_data.get("ts"),
                "webhook_headers": dict(headers)
            }
            
            return IntegrationEvent(
                integration_id=self.integration_id,
                event_type=event_type,
                source="slack",
                data=parsed_data,
                metadata=metadata
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to parse Slack webhook event: {str(e)}",
                EventType.ERROR,
                {"integration_id": self.integration_id, "error": str(e)},
                "slack_integration"
            )
            # Return a fallback event
            return IntegrationEvent(
                integration_id=self.integration_id,
                event_type="parse_error",
                source="slack",
                data=payload,
                metadata={"error": str(e)}
            )
    
    async def receive_webhook(self, event: IntegrationEvent) -> IntegrationResponse:
        """
        Process incoming webhook from Slack.
        
        Args:
            event: Webhook event data
        
        Returns:
            Processing response
        """
        try:
            event_type = event.event_type
            
            # Handle URL verification
            if event_type == "url_verification":
                challenge = event.metadata.get("challenge")
                if challenge:
                    return IntegrationResponse(
                        success=True,
                        data={"challenge": challenge},
                        metadata={"response_type": "url_verification"}
                    )
            
            # Handle message events
            elif event_type == "message":
                return await self._handle_message_event(event)
            
            # Handle other event types
            elif event_type in ["channel_created", "member_joined_channel", "team_join"]:
                return await self._handle_team_event(event)
            
            else:
                # Log unhandled event type
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Unhandled Slack event type: {event_type}",
                    EventType.WEBHOOK,
                    {
                        "integration_id": self.integration_id,
                        "event_type": event_type
                    },
                    "slack_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={"status": "acknowledged", "event_type": event_type}
                )
            
        except Exception as e:
            return self._format_error_response(e, "slack_webhook_failed")
    
    async def _handle_message_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle Slack message events."""
        try:
            event_data = event.data.get("event", {})
            
            # Extract message details
            channel = event_data.get("channel")
            user = event_data.get("user")
            text = event_data.get("text", "")
            timestamp = event_data.get("ts")
            
            # Skip bot messages to avoid loops
            if event_data.get("bot_id") or event_data.get("subtype") == "bot_message":
                return IntegrationResponse(
                    success=True,
                    data={"status": "ignored", "reason": "bot_message"}
                )
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Received Slack message from user {user} in channel {channel}",
                EventType.WEBHOOK,
                {
                    "integration_id": self.integration_id,
                    "channel": channel,
                    "user": user,
                    "timestamp": timestamp
                },
                "slack_integration"
            )
            
            # Here you could implement custom message processing logic
            # For example, trigger an agent response or update a database
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": "message",
                    "channel": channel,
                    "user": user,
                    "message_length": len(text)
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "slack_message_event_failed")
    
    async def _handle_team_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle Slack team-related events."""
        try:
            event_type = event.event_type
            event_data = event.data.get("event", {})
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Received Slack team event: {event_type}",
                EventType.WEBHOOK,
                {
                    "integration_id": self.integration_id,
                    "event_type": event_type,
                    "team_id": event.data.get("team_id")
                },
                "slack_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": event_type,
                    "team_id": event.data.get("team_id")
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "slack_team_event_failed")
    
    async def cleanup(self) -> IntegrationResponse:
        """Clean up Slack integration resources."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.bot_token = None
            self.user_token = None
            
            parent_response = await super().cleanup()
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Slack integration cleanup completed",
                EventType.INTEGRATION,
                {"integration_id": self.integration_id},
                "slack_integration"
            )
            
            return parent_response
            
        except Exception as e:
            return self._format_error_response(e, "slack_cleanup_failed")
    
    def get_oauth2_authorization_url(self, scopes: List[str], state: str = None, redirect_uri: str = None) -> str:
        """
        Generate Slack OAuth2 authorization URL.
        
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
        
        params = {
            "client_id": client_id,
            "scope": ",".join(scopes),
            "redirect_uri": redirect_uri or self.config.auth_config.get("redirect_uri")
        }
        
        if state:
            params["state"] = state
        
        return f"https://slack.com/oauth/v2/authorize?{urlencode(params)}"
    
    async def get_channels(self, limit: int = 100, cursor: str = None) -> IntegrationResponse:
        """
        Get list of channels from Slack.
        
        Args:
            limit: Maximum number of channels to return
            cursor: Pagination cursor
        
        Returns:
            Channels list response
        """
        try:
            if not self.bot_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.bot_token}"}
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            
            url = f"{self.config.base_url}{self.endpoints['conversations_list']}"
            
            async with self.session.get(url, headers=headers, params=params) as response:
                result = await response.json()
                
                if not result.get("ok"):
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to get channels: {result.get('error')}",
                        error_code="channels_failed"
                    )
                
                channels = []
                for channel in result.get("channels", []):
                    channels.append({
                        "id": channel.get("id"),
                        "name": channel.get("name"),
                        "is_private": channel.get("is_private"),
                        "is_member": channel.get("is_member"),
                        "topic": channel.get("topic", {}).get("value"),
                        "purpose": channel.get("purpose", {}).get("value")
                    })
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "channels": channels,
                        "next_cursor": result.get("response_metadata", {}).get("next_cursor"),
                        "total_count": len(channels)
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "slack_channels_failed")