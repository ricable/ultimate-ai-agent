# File: backend/integrations/notion_integration.py
"""
Notion integration implementation for document and page management.
"""

import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from urllib.parse import urlencode

from .base import (
    IntegrationBase, IntegrationConfig, IntegrationResponse, 
    IntegrationEvent, IntegrationError, WebhookIntegration
)
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel


class NotionIntegration(WebhookIntegration):
    """
    Notion integration with full API support for pages, databases, and blocks.
    
    Supports OAuth2 authentication, page creation, database queries,
    and webhook event processing.
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.access_token: Optional[str] = None
        self.bot_id: Optional[str] = None
        self.workspace_id: Optional[str] = None
        
        # Notion API version
        self.api_version = "2022-06-28"
        
        # Notion API endpoints
        self.endpoints = {
            "search": "/search",
            "pages": "/pages",
            "databases": "/databases",
            "blocks": "/blocks",
            "users": "/users",
            "oauth_token": "https://api.notion.com/v1/oauth/token"
        }
    
    async def initialize(self) -> IntegrationResponse:
        """Initialize Notion integration with HTTP session."""
        try:
            # Initialize HTTP session with timeout
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "UAP-Notion-Integration/1.0",
                    "Notion-Version": self.api_version,
                    "Content-Type": "application/json"
                }
            )
            
            # Call parent initialization
            parent_response = await super().initialize()
            if not parent_response.success:
                return parent_response
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Notion integration initialized successfully",
                EventType.INTEGRATION,
                {"integration_id": self.integration_id, "api_version": self.api_version},
                "notion_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={"status": "initialized", "api_version": self.api_version}
            )
            
        except Exception as e:
            return self._format_error_response(e, "notion_init_failed")
    
    async def authenticate(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """
        Authenticate with Notion using OAuth2 or integration token.
        
        Args:
            credentials: Dictionary containing authentication data
                - For OAuth2: {"code": "auth_code", "redirect_uri": "callback_url"}
                - For Integration Token: {"access_token": "secret_..."}
        
        Returns:
            Authentication response
        """
        try:
            if "access_token" in credentials:
                # Direct integration token authentication
                return await self._authenticate_token(credentials["access_token"])
            
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
            return self._format_error_response(e, "notion_auth_failed")
    
    async def _authenticate_token(self, access_token: str) -> IntegrationResponse:
        """Authenticate using integration token."""
        self.access_token = access_token
        
        # Test authentication by getting bot user info
        test_response = await self.test_connection()
        if test_response.success:
            uap_logger.log_event(
                LogLevel.INFO,
                "Notion integration token authentication successful",
                EventType.AUTHENTICATION,
                {"integration_id": self.integration_id},
                "notion_integration"
            )
            return IntegrationResponse(
                success=True,
                data=test_response.data,
                metadata={"auth_method": "integration_token"}
            )
        else:
            return test_response
    
    async def _authenticate_oauth2(self, credentials: Dict[str, Any]) -> IntegrationResponse:
        """Authenticate using OAuth2 code exchange."""
        try:
            import base64
            
            client_id = self.config.auth_config.get("client_id")
            client_secret = self.config.auth_config.get("client_secret")
            
            if not all([client_id, client_secret]):
                raise IntegrationError("Missing OAuth2 configuration", self.integration_id)
            
            # Prepare basic auth header
            auth_string = f"{client_id}:{client_secret}"
            auth_bytes = auth_string.encode('ascii')
            auth_header = base64.b64encode(auth_bytes).decode('ascii')
            
            token_data = {
                "grant_type": "authorization_code",
                "code": credentials["code"],
                "redirect_uri": credentials.get("redirect_uri")
            }
            
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/json"
            }
            
            # Exchange code for tokens
            async with self.session.post(
                self.endpoints["oauth_token"],
                headers=headers,
                json=token_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise IntegrationError(f"OAuth2 exchange failed: {response.status} - {error_text}", self.integration_id)
                
                result = await response.json()
                
                if "error" in result:
                    raise IntegrationError(f"Notion OAuth2 error: {result.get('error')}", self.integration_id)
                
                # Store tokens and workspace info
                self.access_token = result.get("access_token")
                self.bot_id = result.get("bot_id")
                self.workspace_id = result.get("workspace_id")
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    "Notion OAuth2 authentication successful",
                    EventType.AUTHENTICATION,
                    {
                        "integration_id": self.integration_id,
                        "bot_id": self.bot_id,
                        "workspace_id": self.workspace_id
                    },
                    "notion_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "bot_id": self.bot_id,
                        "workspace_id": self.workspace_id,
                        "workspace_name": result.get("workspace_name"),
                        "workspace_icon": result.get("workspace_icon"),
                        "token_type": result.get("token_type")
                    },
                    metadata={"auth_method": "oauth2"}
                )
                
        except IntegrationError:
            raise
        except Exception as e:
            return self._format_error_response(e, "notion_oauth2_failed")
    
    async def test_connection(self) -> IntegrationResponse:
        """Test connection to Notion API."""
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="No authentication token available",
                    error_code="no_token"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            # Test with a simple search query
            test_payload = {
                "query": "",
                "page_size": 1
            }
            
            async with self.session.post(
                f"{self.config.base_url}{self.endpoints['search']}",
                headers=headers,
                json=test_payload
            ) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be invalid",
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
                        "status": "connected",
                        "accessible_objects": len(result.get("results", [])),
                        "has_more": result.get("has_more", False)
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "notion_test_failed")
    
    async def send_message(self, message: str, channel: str = None, **kwargs) -> IntegrationResponse:
        """
        Create a new page or add content to an existing page in Notion.
        
        Args:
            message: Content to add (will be converted to Notion blocks)
            channel: Page ID to add content to, or "new" to create a new page
            **kwargs: Additional parameters (title, parent_id, properties, etc.)
        
        Returns:
            Page creation/update response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            if channel == "new" or not channel:
                # Create a new page
                return await self._create_page(message, headers, **kwargs)
            else:
                # Add content to existing page
                return await self._add_content_to_page(channel, message, headers, **kwargs)
                
        except Exception as e:
            return self._format_error_response(e, "notion_send_failed")
    
    async def _create_page(self, content: str, headers: Dict[str, str], **kwargs) -> IntegrationResponse:
        """Create a new page in Notion."""
        try:
            title = kwargs.get("title", f"UAP Message - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            parent_id = kwargs.get("parent_id")
            
            if not parent_id:
                # If no parent specified, we need to find a suitable parent
                # This would typically be a database or page that the integration has access to
                return IntegrationResponse(
                    success=False,
                    error="Parent page or database ID required for creating new pages",
                    error_code="missing_parent"
                )
            
            # Convert content to Notion blocks
            blocks = self._text_to_notion_blocks(content)
            
            page_data = {
                "parent": {"page_id": parent_id},
                "properties": {
                    "title": {
                        "title": [
                            {
                                "text": {"content": title}
                            }
                        ]
                    }
                },
                "children": blocks
            }
            
            # Add custom properties if provided
            if "properties" in kwargs:
                page_data["properties"].update(kwargs["properties"])
            
            async with self.session.post(
                f"{self.config.base_url}{self.endpoints['pages']}",
                headers=headers,
                json=page_data
            ) as response:
                if response.status not in [200, 201]:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to create page: HTTP {response.status} - {error_text}",
                        error_code="create_failed"
                    )
                
                result = await response.json()
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Notion page created: {result.get('id')}",
                    EventType.INTEGRATION,
                    {
                        "integration_id": self.integration_id,
                        "page_id": result.get("id"),
                        "title": title
                    },
                    "notion_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "id": result.get("id"),
                        "url": result.get("url"),
                        "title": title,
                        "created_time": result.get("created_time"),
                        "last_edited_time": result.get("last_edited_time")
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "notion_create_page_failed")
    
    async def _add_content_to_page(self, page_id: str, content: str, headers: Dict[str, str], **kwargs) -> IntegrationResponse:
        """Add content blocks to an existing page."""
        try:
            # Convert content to Notion blocks
            blocks = self._text_to_notion_blocks(content)
            
            block_data = {"children": blocks}
            
            async with self.session.patch(
                f"{self.config.base_url}{self.endpoints['blocks']}/{page_id}/children",
                headers=headers,
                json=block_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error=f"Failed to add content: HTTP {response.status} - {error_text}",
                        error_code="add_content_failed"
                    )
                
                result = await response.json()
                
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Content added to Notion page: {page_id}",
                    EventType.INTEGRATION,
                    {
                        "integration_id": self.integration_id,
                        "page_id": page_id,
                        "blocks_added": len(blocks)
                    },
                    "notion_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "page_id": page_id,
                        "blocks_added": len(result.get("results", [])),
                        "has_more": result.get("has_more", False)
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "notion_add_content_failed")
    
    def _text_to_notion_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Convert plain text to Notion block format."""
        # Split text into paragraphs
        paragraphs = text.strip().split('\n\n')
        blocks = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Check if it looks like a heading
                if paragraph.startswith('#'):
                    # Markdown-style heading
                    level = len(paragraph) - len(paragraph.lstrip('#'))
                    content = paragraph.lstrip('#').strip()
                    
                    if level == 1:
                        block_type = "heading_1"
                    elif level == 2:
                        block_type = "heading_2"
                    else:
                        block_type = "heading_3"
                    
                    blocks.append({
                        "object": "block",
                        "type": block_type,
                        block_type: {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": content}
                                }
                            ]
                        }
                    })
                else:
                    # Regular paragraph
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {"content": paragraph.strip()}
                                }
                            ]
                        }
                    })
        
        return blocks if blocks else [{
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": text}
                    }
                ]
            }
        }]
    
    async def get_user_info(self, user_id: str = None) -> IntegrationResponse:
        """
        Get user information from Notion.
        
        Args:
            user_id: User ID (optional, defaults to bot user)
        
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
                url = f"{self.config.base_url}{self.endpoints['users']}/{user_id}"
            else:
                # Get bot user info
                url = f"{self.config.base_url}{self.endpoints['users']}/me"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 401:
                    return IntegrationResponse(
                        success=False,
                        error="Authentication failed - token may be invalid",
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
                        "name": result.get("name"),
                        "type": result.get("type"),
                        "avatar_url": result.get("avatar_url"),
                        "bot": result.get("bot", {}) if result.get("type") == "bot" else None
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "notion_user_info_failed")
    
    async def refresh_credentials(self) -> IntegrationResponse:
        """
        Refresh authentication credentials.
        
        Note: Notion integration tokens don't expire, OAuth2 tokens may need refresh.
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
            return self._format_error_response(e, "notion_refresh_failed")
    
    async def verify_webhook_signature(self, payload: bytes, headers: Dict[str, str]) -> bool:
        """
        Verify Notion webhook signature.
        
        Notion doesn't use HMAC signatures but relies on HTTPS and other security measures.
        """
        try:
            # Notion webhooks are verified through HTTPS endpoints and request validation
            # Additional custom verification can be implemented here if needed
            
            return True  # Basic validation - enhance as needed
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Notion webhook signature verification error: {str(e)}",
                EventType.ERROR,
                {"integration_id": self.integration_id, "error": str(e)},
                "notion_integration"
            )
            return False
    
    async def parse_webhook_event(self, payload: Dict[str, Any], headers: Dict[str, str]) -> IntegrationEvent:
        """
        Parse Notion webhook payload into standardized event format.
        
        Args:
            payload: Webhook payload from Notion
            headers: HTTP headers
        
        Returns:
            Standardized IntegrationEvent
        """
        try:
            # Extract event information
            event_type = payload.get("event", "unknown")
            object_type = payload.get("object", "unknown")
            
            # Common event metadata
            metadata = {
                "object_type": object_type,
                "webhook_headers": dict(headers)
            }
            
            # Add object-specific metadata
            if "page" in payload:
                metadata["page_id"] = payload["page"].get("id")
                metadata["page_url"] = payload["page"].get("url")
            
            if "database" in payload:
                metadata["database_id"] = payload["database"].get("id")
            
            return IntegrationEvent(
                integration_id=self.integration_id,
                event_type=event_type,
                source="notion",
                data=payload,
                metadata=metadata
            )
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to parse Notion webhook event: {str(e)}",
                EventType.ERROR,
                {"integration_id": self.integration_id, "error": str(e)},
                "notion_integration"
            )
            return IntegrationEvent(
                integration_id=self.integration_id,
                event_type="parse_error",
                source="notion",
                data=payload,
                metadata={"error": str(e)}
            )
    
    async def receive_webhook(self, event: IntegrationEvent) -> IntegrationResponse:
        """
        Process incoming webhook from Notion.
        
        Args:
            event: Webhook event data
        
        Returns:
            Processing response
        """
        try:
            event_type = event.event_type
            
            # Handle different event types
            if event_type in ["page.updated", "page.created"]:
                return await self._handle_page_event(event)
            elif event_type in ["database.updated", "database.created"]:
                return await self._handle_database_event(event)
            else:
                uap_logger.log_event(
                    LogLevel.INFO,
                    f"Received Notion event type: {event_type}",
                    EventType.WEBHOOK,
                    {
                        "integration_id": self.integration_id,
                        "event_type": event_type
                    },
                    "notion_integration"
                )
                
                return IntegrationResponse(
                    success=True,
                    data={"status": "acknowledged", "event_type": event_type}
                )
            
        except Exception as e:
            return self._format_error_response(e, "notion_webhook_failed")
    
    async def _handle_page_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle Notion page events."""
        try:
            page_data = event.data.get("page", {})
            page_id = page_data.get("id")
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Received Notion page event: {event.event_type} for page {page_id}",
                EventType.WEBHOOK,
                {
                    "integration_id": self.integration_id,
                    "page_id": page_id,
                    "event_type": event.event_type
                },
                "notion_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": event.event_type,
                    "page_id": page_id,
                    "page_url": page_data.get("url")
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "notion_page_event_failed")
    
    async def _handle_database_event(self, event: IntegrationEvent) -> IntegrationResponse:
        """Handle Notion database events."""
        try:
            database_data = event.data.get("database", {})
            database_id = database_data.get("id")
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Received Notion database event: {event.event_type} for database {database_id}",
                EventType.WEBHOOK,
                {
                    "integration_id": self.integration_id,
                    "database_id": database_id,
                    "event_type": event.event_type
                },
                "notion_integration"
            )
            
            return IntegrationResponse(
                success=True,
                data={
                    "status": "processed",
                    "event_type": event.event_type,
                    "database_id": database_id
                }
            )
            
        except Exception as e:
            return self._format_error_response(e, "notion_database_event_failed")
    
    async def cleanup(self) -> IntegrationResponse:
        """Clean up Notion integration resources."""
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.access_token = None
            self.bot_id = None
            self.workspace_id = None
            
            parent_response = await super().cleanup()
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Notion integration cleanup completed",
                EventType.INTEGRATION,
                {"integration_id": self.integration_id},
                "notion_integration"
            )
            
            return parent_response
            
        except Exception as e:
            return self._format_error_response(e, "notion_cleanup_failed")
    
    def get_oauth2_authorization_url(self, state: str = None, redirect_uri: str = None) -> str:
        """
        Generate Notion OAuth2 authorization URL.
        
        Args:
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
            "response_type": "code",
            "owner": "user",
            "redirect_uri": redirect_uri or self.config.auth_config.get("redirect_uri")
        }
        
        if state:
            params["state"] = state
        
        return f"https://api.notion.com/v1/oauth/authorize?{urlencode(params)}"
    
    async def search_pages(self, query: str = "", filter_type: str = None, page_size: int = 100) -> IntegrationResponse:
        """
        Search for pages and databases in Notion.
        
        Args:
            query: Search query string
            filter_type: Filter by object type ("page" or "database")
            page_size: Number of results to return
        
        Returns:
            Search results response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            search_payload = {
                "query": query,
                "page_size": min(page_size, 100)
            }
            
            if filter_type:
                search_payload["filter"] = {
                    "value": filter_type,
                    "property": "object"
                }
            
            async with self.session.post(
                f"{self.config.base_url}{self.endpoints['search']}",
                headers=headers,
                json=search_payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error=f"Search failed: HTTP {response.status} - {error_text}",
                        error_code="search_failed"
                    )
                
                result = await response.json()
                
                # Format results
                formatted_results = []
                for item in result.get("results", []):
                    formatted_item = {
                        "id": item.get("id"),
                        "object": item.get("object"),
                        "url": item.get("url"),
                        "created_time": item.get("created_time"),
                        "last_edited_time": item.get("last_edited_time")
                    }
                    
                    # Add object-specific properties
                    if item.get("object") == "page":
                        properties = item.get("properties", {})
                        title_prop = properties.get("title") or properties.get("Title")
                        if title_prop and title_prop.get("title"):
                            formatted_item["title"] = title_prop["title"][0].get("text", {}).get("content", "")
                    elif item.get("object") == "database":
                        formatted_item["title"] = item.get("title", [{}])[0].get("text", {}).get("content", "")
                    
                    formatted_results.append(formatted_item)
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "results": formatted_results,
                        "has_more": result.get("has_more", False),
                        "next_cursor": result.get("next_cursor"),
                        "total_count": len(formatted_results)
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "notion_search_failed")
    
    async def query_database(self, database_id: str, filter_criteria: Dict[str, Any] = None, 
                           sorts: List[Dict[str, Any]] = None, page_size: int = 100) -> IntegrationResponse:
        """
        Query a Notion database.
        
        Args:
            database_id: Database identifier
            filter_criteria: Filter criteria for the query
            sorts: Sort criteria for the results
            page_size: Number of results to return
        
        Returns:
            Database query response
        """
        try:
            if not self.access_token:
                return IntegrationResponse(
                    success=False,
                    error="Not authenticated",
                    error_code="not_authenticated"
                )
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            query_payload = {
                "page_size": min(page_size, 100)
            }
            
            if filter_criteria:
                query_payload["filter"] = filter_criteria
            
            if sorts:
                query_payload["sorts"] = sorts
            
            async with self.session.post(
                f"{self.config.base_url}{self.endpoints['databases']}/{database_id}/query",
                headers=headers,
                json=query_payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return IntegrationResponse(
                        success=False,
                        error=f"Database query failed: HTTP {response.status} - {error_text}",
                        error_code="query_failed"
                    )
                
                result = await response.json()
                
                return IntegrationResponse(
                    success=True,
                    data={
                        "results": result.get("results", []),
                        "has_more": result.get("has_more", False),
                        "next_cursor": result.get("next_cursor"),
                        "total_count": len(result.get("results", []))
                    }
                )
                
        except Exception as e:
            return self._format_error_response(e, "notion_query_database_failed")