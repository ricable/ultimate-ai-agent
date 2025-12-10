# UAP SDK Client Module
"""
Client classes for connecting to UAP backend services.
Provides WebSocket and HTTP clients for agent communication.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, List, Union, AsyncGenerator, TYPE_CHECKING
from types import TracebackType
from datetime import datetime, timedelta
import httpx
import websockets
from contextlib import asynccontextmanager
from .config import Configuration
from .exceptions import UAPConnectionError, UAPAuthError, UAPException

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class AuthClient:
    """Authentication client for UAP services."""
    
    def __init__(self, config: Configuration) -> None:
        self.config = config
        self.base_url: str = config.get("backend_url", "http://localhost:8000")
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._refresh_threshold: timedelta = timedelta(minutes=5)  # Refresh if token expires within 5 minutes
        
    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login to UAP and store authentication tokens."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/auth/login",
                    json={"username": username, "password": password}
                )
                response.raise_for_status()
                
                data = response.json()
                tokens = data.get("tokens", {})
                
                self._access_token = tokens.get("access_token")
                self._refresh_token = tokens.get("refresh_token")
                
                # Store in config for persistence
                self.config.set("access_token", self._access_token)
                self.config.set("refresh_token", self._refresh_token)
                
                logger.info("Successfully authenticated with UAP")
                return data
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise UAPAuthError("Invalid username or password")
                else:
                    raise UAPAuthError(f"Authentication failed: {e.response.text}")
            except Exception as e:
                raise UAPConnectionError(f"Failed to connect to UAP: {str(e)}")
    
    async def register(self, username: str, password: str, email: str, roles: Optional[List[str]] = None) -> Dict[str, Any]:
        """Register a new user with UAP."""
        user_data = {
            "username": username,
            "password": password,
            "email": email,
            "roles": roles or ["user"]
        }
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/auth/register",
                    json=user_data
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                raise UAPAuthError(f"Registration failed: {e.response.text}")
            except Exception as e:
                raise UAPConnectionError(f"Failed to connect to UAP: {str(e)}")
    
    async def refresh_access_token(self) -> str:
        """Refresh the access token using the refresh token."""
        if not self._refresh_token:
            raise UAPAuthError("No refresh token available")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/auth/refresh",
                    json={"refresh_token": self._refresh_token}
                )
                response.raise_for_status()
                
                data = response.json()
                self._access_token = data.get("access_token")
                self.config.set("access_token", self._access_token)
                
                return self._access_token
                
            except httpx.HTTPStatusError as e:
                raise UAPAuthError(f"Token refresh failed: {e.response.text}")
            except Exception as e:
                raise UAPConnectionError(f"Failed to refresh token: {str(e)}")
    
    def get_access_token(self) -> Optional[str]:
        """Get the current access token."""
        if not self._access_token:
            self._access_token = self.config.get("access_token")
        return self._access_token
    
    async def logout(self) -> None:
        """Logout and invalidate tokens."""
        if self._refresh_token:
            async with httpx.AsyncClient() as client:
                try:
                    await client.post(
                        f"{self.base_url}/api/auth/logout",
                        json={"refresh_token": self._refresh_token},
                        headers={"Authorization": f"Bearer {self._access_token}"}
                    )
                except Exception:
                    pass  # Ignore logout errors
        
        self._access_token = None
        self._refresh_token = None
        self.config.remove("access_token")
        self.config.remove("refresh_token")
    
    def is_token_expired(self) -> bool:
        """Check if the current token is expired or will expire soon."""
        if not self._token_expiry:
            return True
        return datetime.utcnow() + self._refresh_threshold >= self._token_expiry
    
    async def ensure_valid_token(self) -> str:
        """Ensure we have a valid access token, refreshing if necessary."""
        if not self._access_token or self.is_token_expired():
            if self._refresh_token:
                return await self.refresh_access_token()
            else:
                raise UAPAuthError("No valid token available. Please login first.")
        return self._access_token


class HTTPClient:
    """HTTP client for UAP REST API communication."""
    
    def __init__(self, config: Configuration, auth_client: AuthClient) -> None:
        self.config = config
        self.auth_client = auth_client
        self.base_url: str = config.get("backend_url", "http://localhost:8000")
        self.timeout: int = config.get("http_timeout", 30)
        self._client: Optional[httpx.AsyncClient] = None
        self._retry_config = {
            'max_retries': config.get('max_retries', 3),
            'retry_delay': config.get('retry_delay', 1.0),
            'backoff_factor': config.get('backoff_factor', 2.0)
        }
        
    async def _get_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        headers = {"Content-Type": "application/json"}
        
        try:
            token = await self.auth_client.ensure_valid_token()
            headers["Authorization"] = f"Bearer {token}"
        except UAPAuthError:
            # Continue without auth token for public endpoints
            pass
            
        return headers
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with retry configuration."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
                follow_redirects=True
            )
        return self._client
    
    async def _make_request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with automatic retry logic."""
        client = await self._get_client()
        last_exception = None
        
        for attempt in range(self._retry_config['max_retries'] + 1):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403, 404, 422):  # Don't retry client errors
                    raise
                last_exception = e
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e
            
            if attempt < self._retry_config['max_retries']:
                delay = self._retry_config['retry_delay'] * (self._retry_config['backoff_factor'] ** attempt)
                await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        if isinstance(last_exception, httpx.HTTPStatusError):
            raise UAPException(f"HTTP request failed after {self._retry_config['max_retries']} retries: {last_exception.response.text}")
        else:
            raise UAPConnectionError(f"Connection failed after {self._retry_config['max_retries']} retries: {str(last_exception)}")
    
    async def chat_with_agent(self, agent_id: str, message: str, framework: str = "auto", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a chat message to an agent via HTTP."""
        headers = await self._get_headers()
        
        payload = {
            "message": message,
            "framework": framework,
            "context": context or {}
        }
        
        try:
            response = await self._make_request_with_retry(
                "POST",
                f"{self.base_url}/api/agents/{agent_id}/chat",
                json=payload,
                headers=headers
            )
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise UAPAuthError("Authentication required or token expired")
            else:
                raise UAPException(f"Agent communication failed: {e.response.text}")
        except Exception as e:
            raise UAPConnectionError(f"Failed to communicate with agent: {str(e)}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get the system status."""
        headers = await self._get_headers()
        
        try:
            response = await self._make_request_with_retry(
                "GET",
                f"{self.base_url}/api/status",
                headers=headers
            )
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise UAPAuthError("Authentication required")
            else:
                raise UAPException(f"Status request failed: {e.response.text}")
        except Exception as e:
            raise UAPConnectionError(f"Failed to get system status: {str(e)}")
    
    async def upload_document(self, file_path: str, process_immediately: bool = True) -> Dict[str, Any]:
        """Upload a document for processing."""
        headers = await self._get_headers()
        del headers["Content-Type"]  # Let httpx set multipart content type
        
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {"process_immediately": str(process_immediately).lower()}
                
                response = await self._make_request_with_retry(
                    "POST",
                    f"{self.base_url}/api/documents/upload",
                    files=files,
                    data=data,
                    headers=headers
                )
                return response.json()
        except httpx.HTTPStatusError as e:
            raise UAPException(f"Document upload failed: {e.response.text}")
        except Exception as e:
            raise UAPConnectionError(f"Failed to upload document: {str(e)}")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class WebSocketClient:
    """WebSocket client for real-time UAP communication using AG-UI protocol."""
    
    def __init__(self, config: Configuration, auth_client: AuthClient) -> None:
        self.config = config
        self.auth_client = auth_client
        self.ws_url: str = config.get("websocket_url", "ws://localhost:8000")
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.message_handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self.is_connected: bool = False
        self._listen_task: Optional[asyncio.Task] = None
        self._reconnect_enabled: bool = config.get("auto_reconnect", True)
        self._reconnect_delay: float = config.get("reconnect_delay", 5.0)
        self._max_reconnect_attempts: int = config.get("max_reconnect_attempts", 10)
        self._current_agent_id: Optional[str] = None
        
    async def connect(self, agent_id: str) -> None:
        """Connect to UAP WebSocket endpoint."""
        self._current_agent_id = agent_id
        token = self.auth_client.get_access_token()
        url = f"{self.ws_url}/ws/agents/{agent_id}"
        
        if token:
            url += f"?token={token}"
        
        try:
            # Add connection headers for better debugging
            extra_headers = {
                "User-Agent": "UAP-SDK/1.0.0",
                "X-Agent-ID": agent_id
            }
            
            self.websocket = await websockets.connect(
                url,
                extra_headers=extra_headers,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            self.is_connected = True
            self._listen_task = asyncio.create_task(self._listen_for_messages())
            logger.info(f"Connected to UAP WebSocket for agent {agent_id}")
            
        except Exception as e:
            raise UAPConnectionError(f"Failed to connect to WebSocket: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self.is_connected = False
        
        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        logger.info("Disconnected from UAP WebSocket")
    
    def on_message(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a message handler for specific event types."""
        self.message_handlers[event_type] = handler
    
    async def send_message(self, message: str, metadata: Optional[Dict] = None) -> None:
        """Send a user message via AG-UI protocol."""
        if not self.is_connected or not self.websocket:
            raise UAPConnectionError("WebSocket not connected")
        
        event = {
            "type": "user_message",
            "content": message,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.websocket.send(json.dumps(event))
    
    async def _listen_for_messages(self) -> None:
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    event = json.loads(message)
                    event_type = event.get("type")
                    
                    if event_type in self.message_handlers:
                        handler = self.message_handlers[event_type]
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    
                    # Default handler for all messages
                    if "default" in self.message_handlers:
                        handler = self.message_handlers["default"]
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                            
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            logger.info("WebSocket connection closed")
            if self._reconnect_enabled and self._current_agent_id:
                asyncio.create_task(self._handle_reconnect())
        except Exception as e:
            self.is_connected = False
            logger.error(f"WebSocket listen error: {e}")
            if self._reconnect_enabled and self._current_agent_id:
                asyncio.create_task(self._handle_reconnect())
    
    async def _handle_reconnect(self) -> None:
        """Handle automatic reconnection with exponential backoff."""
        for attempt in range(self._max_reconnect_attempts):
            try:
                await asyncio.sleep(self._reconnect_delay * (2 ** attempt))
                logger.info(f"Attempting WebSocket reconnection (attempt {attempt + 1}/{self._max_reconnect_attempts})")
                await self.connect(self._current_agent_id)
                logger.info("WebSocket reconnection successful")
                break
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        else:
            logger.error("All reconnection attempts failed")


class UAPClient:
    """Main UAP client that combines all communication methods."""
    
    def __init__(self, config: Optional[Configuration] = None) -> None:
        self.config = config or Configuration()
        self.auth = AuthClient(self.config)
        self.http = HTTPClient(self.config, self.auth)
        self.websocket = WebSocketClient(self.config, self.auth)
        self._closed = False
        
    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login to UAP."""
        return await self.auth.login(username, password)
    
    async def connect_websocket(self, agent_id: str) -> None:
        """Connect to WebSocket for real-time communication."""
        await self.websocket.connect(agent_id)
    
    async def chat(self, agent_id: str, message: str, framework: str = "auto", context: Optional[Dict[str, Any]] = None, use_websocket: bool = False) -> Dict[str, Any]:
        """Chat with an agent using HTTP or WebSocket."""
        if use_websocket:
            if not self.websocket.is_connected:
                await self.connect_websocket(agent_id)
            await self.websocket.send_message(message, {"framework": framework, **context or {}})
            return {"status": "message_sent", "websocket": True}
        else:
            return await self.http.chat_with_agent(agent_id, message, framework, context)
    
    async def upload_document(self, file_path: str, process_immediately: bool = True) -> Dict[str, Any]:
        """Upload a document for processing."""
        return await self.http.upload_document(file_path, process_immediately)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return await self.http.get_system_status()
    
    async def cleanup(self) -> None:
        """Clean up all connections."""
        if self._closed:
            return
        
        self._closed = True
        
        if self.websocket.is_connected:
            await self.websocket.disconnect()
        
        await self.http.close()
        await self.auth.logout()
    
    async def __aenter__(self) -> 'Self':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        """Async context manager exit."""
        await self.cleanup()
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator['UAPClient', None]:
        """Create a managed session with automatic cleanup."""
        try:
            yield self
        finally:
            await self.cleanup()
    
    def is_healthy(self) -> bool:
        """Check if the client is in a healthy state."""
        return not self._closed and self.auth.get_access_token() is not None