# Discord Integration Plugin
"""
Discord integration for UAP.

This plugin provides Discord bot functionality including:
- Sending messages to channels
- Managing server members
- Handling webhook events
- Bot command processing
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Import from the UAP plugin SDK
try:
    from uap_sdk import (
        IntegrationPlugin, PluginContext, PluginResponse, PluginEvent,
        action, permission_required, rate_limit, log_execution
    )
except ImportError:
    # Fallback for development - import from backend directly
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent / "backend"))
    
    from plugins.plugin_base import (
        IntegrationPlugin, PluginContext, PluginResponse, PluginEvent
    )
    
    # Mock decorators for development
    def action(name=None, **kwargs):
        def decorator(func):
            func._is_action = True
            func._action_name = name or func.__name__
            return func
        return decorator
    
    def permission_required(*perms):
        def decorator(func):
            return func
        return decorator
    
    def rate_limit(**kwargs):
        def decorator(func):
            return func
        return decorator
    
    def log_execution(**kwargs):
        def decorator(func):
            return func
        return decorator


class DiscordIntegration(IntegrationPlugin):
    """
    Discord integration plugin for UAP.
    
    Provides Discord bot functionality with message sending,
    server management, and webhook processing capabilities.
    """
    
    def __init__(self, manifest, config):
        super().__init__(manifest, config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.bot_token: Optional[str] = None
        self.base_url = "https://discord.com/api/v10"
        
        # Discord API endpoints
        self.endpoints = {
            "gateway": "/gateway/bot",
            "channels": "/channels",
            "guilds": "/guilds",
            "users": "/users/@me",
            "messages": "/channels/{channel_id}/messages"
        }
    
    @log_execution(include_params=True)
    async def initialize(self, context: PluginContext) -> PluginResponse:
        """Initialize the Discord integration."""
        try:
            self.bot_token = self.config.config.get("bot_token")
            if not self.bot_token:
                return PluginResponse(
                    success=False,
                    error="Discord bot token not configured",
                    error_code="missing_bot_token"
                )
            
            # Initialize HTTP session
            headers = {
                "Authorization": f"Bot {self.bot_token}",
                "Content-Type": "application/json",
                "User-Agent": "UAP-Discord-Integration/1.0"
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
            
            # Test bot token validity
            test_response = await self.test_connection(context)
            if not test_response.success:
                return test_response
            
            return PluginResponse(
                success=True,
                data={
                    "status": "initialized",
                    "service": "Discord",
                    "bot_user": test_response.data.get("bot_user")
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "discord_initialization_failed")
    
    @action(name="send_message", description="Send a message to a Discord channel")
    @permission_required("network_access")
    @rate_limit(calls_per_minute=30)
    async def execute(self, action: str, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Execute an action."""
        try:
            if action == "send_message":
                return await self._send_message(params, context)
            elif action == "get_channels":
                return await self._get_channels(params, context)
            elif action == "get_guild_info":
                return await self._get_guild_info(params, context)
            elif action == "get_status":
                return await self._get_status(params, context)
            else:
                return PluginResponse(
                    success=False,
                    error=f"Unknown action: {action}",
                    error_code="unknown_action"
                )
                
        except Exception as e:
            return self._format_error_response(e, "discord_execution_failed")
    
    async def cleanup(self) -> PluginResponse:
        """Clean up Discord integration resources."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.bot_token = None
            
            return PluginResponse(
                success=True,
                data={"status": "cleaned_up"}
            )
            
        except Exception as e:
            return self._format_error_response(e, "discord_cleanup_failed")
    
    async def authenticate(self, credentials: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Authenticate with Discord."""
        try:
            bot_token = credentials.get("bot_token")
            if not bot_token:
                return PluginResponse(
                    success=False,
                    error="Bot token required",
                    error_code="missing_credentials"
                )
            
            # Store token temporarily for testing
            old_token = self.bot_token
            self.bot_token = bot_token
            
            # Test the bot token
            if self.session:
                await self.session.close()
            
            headers = {
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
                "User-Agent": "UAP-Discord-Integration/1.0"
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
            
            # Test connection
            test_response = await self.test_connection(context)
            
            if test_response.success:
                return PluginResponse(
                    success=True,
                    data={
                        "status": "authenticated",
                        "service": "Discord",
                        "bot_user": test_response.data.get("bot_user")
                    }
                )
            else:
                # Restore old token on failure
                self.bot_token = old_token
                return test_response
            
        except Exception as e:
            return self._format_error_response(e, "discord_authentication_failed")
    
    async def test_connection(self, context: PluginContext) -> PluginResponse:
        """Test connection to Discord API."""
        try:
            if not self.session:
                return PluginResponse(
                    success=False,
                    error="Discord session not initialized",
                    error_code="session_not_initialized"
                )
            
            async with self.session.get(f"{self.base_url}{self.endpoints['users']}") as response:
                if response.status == 401:
                    return PluginResponse(
                        success=False,
                        error="Invalid Discord bot token",
                        error_code="invalid_token"
                    )
                elif response.status != 200:
                    return PluginResponse(
                        success=False,
                        error=f"Discord API error: HTTP {response.status}",
                        error_code="api_error"
                    )
                
                bot_user = await response.json()
                
                return PluginResponse(
                    success=True,
                    data={
                        "status": "connected",
                        "service": "Discord",
                        "bot_user": {
                            "id": bot_user.get("id"),
                            "username": bot_user.get("username"),
                            "discriminator": bot_user.get("discriminator")
                        }
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "discord_connection_test_failed")
    
    async def send_data(self, data: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Send data to Discord (wrapper for send_message)."""
        return await self._send_message(data, context)
    
    async def receive_webhook(self, event: PluginEvent) -> PluginResponse:
        """Process webhook from Discord."""
        try:
            event_type = event.event_type
            event_data = event.data
            
            # Handle different Discord webhook events
            if event_type == "MESSAGE_CREATE":
                return await self._handle_message_event(event_data)
            elif event_type == "GUILD_MEMBER_ADD":
                return await self._handle_member_join_event(event_data)
            elif event_type == "INTERACTION_CREATE":
                return await self._handle_interaction_event(event_data)
            else:
                return PluginResponse(
                    success=True,
                    data={
                        "status": "acknowledged",
                        "event_type": event_type,
                        "processed": False
                    }
                )
            
        except Exception as e:
            return self._format_error_response(e, "discord_webhook_processing_failed")
    
    async def _send_message(self, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Send a message to a Discord channel."""
        try:
            message = params.get("message") or params.get("content")
            channel_id = params.get("channel_id") or params.get("channel")
            
            if not message:
                return PluginResponse(
                    success=False,
                    error="Message content required",
                    error_code="missing_message"
                )
            
            # Use default channel if none specified
            if not channel_id:
                channel_id = self.config.config.get("default_channel")
                if not channel_id:
                    return PluginResponse(
                        success=False,
                        error="Channel ID required (no default channel configured)",
                        error_code="missing_channel"
                    )
            
            # Prepare message payload
            payload = {"content": str(message)}
            
            # Add optional parameters
            if "embeds" in params:
                payload["embeds"] = params["embeds"]
            if "files" in params:
                payload["files"] = params["files"]
            if "tts" in params:
                payload["tts"] = bool(params["tts"])
            
            # Send message
            url = f"{self.base_url}{self.endpoints['messages'].format(channel_id=channel_id)}"
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 403:
                    return PluginResponse(
                        success=False,
                        error="Bot lacks permission to send messages to this channel",
                        error_code="insufficient_permissions"
                    )
                elif response.status == 404:
                    return PluginResponse(
                        success=False,
                        error="Channel not found",
                        error_code="channel_not_found"
                    )
                elif response.status not in [200, 201]:
                    error_text = await response.text()
                    return PluginResponse(
                        success=False,
                        error=f"Failed to send message: HTTP {response.status} - {error_text}",
                        error_code="send_failed"
                    )
                
                result = await response.json()
                
                return PluginResponse(
                    success=True,
                    data={
                        "message_sent": True,
                        "message_id": result.get("id"),
                        "channel_id": result.get("channel_id"),
                        "content": result.get("content"),
                        "timestamp": result.get("timestamp"),
                        "service": "Discord"
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "discord_send_message_failed")
    
    async def _get_channels(self, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Get channels from a Discord guild."""
        try:
            guild_id = params.get("guild_id")
            if not guild_id:
                return PluginResponse(
                    success=False,
                    error="Guild ID required",
                    error_code="missing_guild_id"
                )
            
            url = f"{self.base_url}/guilds/{guild_id}/channels"
            
            async with self.session.get(url) as response:
                if response.status == 403:
                    return PluginResponse(
                        success=False,
                        error="Bot lacks permission to access guild channels",
                        error_code="insufficient_permissions"
                    )
                elif response.status == 404:
                    return PluginResponse(
                        success=False,
                        error="Guild not found",
                        error_code="guild_not_found"
                    )
                elif response.status != 200:
                    return PluginResponse(
                        success=False,
                        error=f"Failed to get channels: HTTP {response.status}",
                        error_code="get_channels_failed"
                    )
                
                channels = await response.json()
                
                # Filter and format channels
                formatted_channels = []
                for channel in channels:
                    formatted_channels.append({
                        "id": channel.get("id"),
                        "name": channel.get("name"),
                        "type": channel.get("type"),
                        "position": channel.get("position"),
                        "topic": channel.get("topic")
                    })
                
                return PluginResponse(
                    success=True,
                    data={
                        "channels": formatted_channels,
                        "guild_id": guild_id,
                        "count": len(formatted_channels)
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "discord_get_channels_failed")
    
    async def _get_guild_info(self, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Get information about a Discord guild."""
        try:
            guild_id = params.get("guild_id")
            if not guild_id:
                return PluginResponse(
                    success=False,
                    error="Guild ID required",
                    error_code="missing_guild_id"
                )
            
            url = f"{self.base_url}/guilds/{guild_id}"
            
            async with self.session.get(url) as response:
                if response.status == 403:
                    return PluginResponse(
                        success=False,
                        error="Bot lacks permission to access guild information",
                        error_code="insufficient_permissions"
                    )
                elif response.status == 404:
                    return PluginResponse(
                        success=False,
                        error="Guild not found",
                        error_code="guild_not_found"
                    )
                elif response.status != 200:
                    return PluginResponse(
                        success=False,
                        error=f"Failed to get guild info: HTTP {response.status}",
                        error_code="get_guild_info_failed"
                    )
                
                guild = await response.json()
                
                return PluginResponse(
                    success=True,
                    data={
                        "guild": {
                            "id": guild.get("id"),
                            "name": guild.get("name"),
                            "description": guild.get("description"),
                            "member_count": guild.get("approximate_member_count"),
                            "owner_id": guild.get("owner_id"),
                            "verification_level": guild.get("verification_level"),
                            "created_at": guild.get("created_at")
                        }
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "discord_get_guild_info_failed")
    
    async def _get_status(self, params: Dict[str, Any], context: PluginContext) -> PluginResponse:
        """Get Discord integration status."""
        return PluginResponse(
            success=True,
            data={
                "service": "Discord",
                "status": "active" if self.session else "inactive",
                "plugin_id": self.plugin_id,
                "version": self.version,
                "bot_token_configured": bool(self.bot_token),
                "session_active": bool(self.session)
            }
        )
    
    async def _handle_message_event(self, event_data: Dict[str, Any]) -> PluginResponse:
        """Handle Discord message events."""
        message = event_data.get("d", {})
        
        # Skip bot messages to avoid loops
        if message.get("author", {}).get("bot"):
            return PluginResponse(
                success=True,
                data={"status": "ignored", "reason": "bot_message"}
            )
        
        # Process commands if they start with the configured prefix
        content = message.get("content", "")
        prefix = self.config.config.get("command_prefix", "!")
        
        if content.startswith(prefix):
            command = content[len(prefix):].strip().split()[0]
            
            # Handle basic commands
            if command == "ping":
                # Send pong response
                await self._send_message({
                    "message": "Pong! UAP Discord bot is online.",
                    "channel_id": message.get("channel_id")
                }, None)
            elif command == "help":
                # Send help message
                await self._send_message({
                    "message": f"Available commands:\n{prefix}ping - Test bot response\n{prefix}help - Show this help",
                    "channel_id": message.get("channel_id")
                }, None)
        
        return PluginResponse(
            success=True,
            data={
                "status": "processed",
                "event_type": "message",
                "message_id": message.get("id"),
                "channel_id": message.get("channel_id"),
                "author": message.get("author", {}).get("username")
            }
        )
    
    async def _handle_member_join_event(self, event_data: Dict[str, Any]) -> PluginResponse:
        """Handle Discord member join events."""
        member = event_data.get("d", {})
        
        return PluginResponse(
            success=True,
            data={
                "status": "processed",
                "event_type": "member_join",
                "user_id": member.get("user", {}).get("id"),
                "username": member.get("user", {}).get("username"),
                "guild_id": member.get("guild_id")
            }
        )
    
    async def _handle_interaction_event(self, event_data: Dict[str, Any]) -> PluginResponse:
        """Handle Discord interaction events (slash commands, buttons, etc.)."""
        interaction = event_data.get("d", {})
        
        return PluginResponse(
            success=True,
            data={
                "status": "processed",
                "event_type": "interaction",
                "interaction_id": interaction.get("id"),
                "interaction_type": interaction.get("type"),
                "user_id": interaction.get("user", {}).get("id")
            }
        )
    
    async def get_actions(self) -> List[Dict[str, Any]]:
        """Get list of available actions for this plugin."""
        return [
            {
                "name": "send_message",
                "description": "Send a message to a Discord channel",
                "parameters": {
                    "message": {"type": "string", "required": True, "description": "Message content"},
                    "channel_id": {"type": "string", "required": False, "description": "Channel ID (uses default if not specified)"},
                    "embeds": {"type": "array", "required": False, "description": "Message embeds"},
                    "tts": {"type": "boolean", "required": False, "description": "Text-to-speech"}
                },
                "permissions": ["network_access"]
            },
            {
                "name": "get_channels",
                "description": "Get channels from a Discord guild",
                "parameters": {
                    "guild_id": {"type": "string", "required": True, "description": "Guild ID"}
                },
                "permissions": ["network_access"]
            },
            {
                "name": "get_guild_info",
                "description": "Get information about a Discord guild",
                "parameters": {
                    "guild_id": {"type": "string", "required": True, "description": "Guild ID"}
                },
                "permissions": ["network_access"]
            },
            {
                "name": "get_status",
                "description": "Get Discord integration status",
                "parameters": {},
                "permissions": []
            }
        ]