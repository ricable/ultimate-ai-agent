"""Authentication handler for MCP server."""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ...auth import JWTManager, AuthenticationError
from ...utils.state_manager import StateManager

logger = logging.getLogger(__name__)


class AuthHandler:
    """Handles authentication-related MCP requests."""
    
    def __init__(self, jwt_manager: JWTManager, state_manager: StateManager):
        """Initialize auth handler with JWT manager and state manager."""
        self.jwt_manager = jwt_manager
        self.state_manager = state_manager
        
        # Register auth methods
        self.methods = {
            "auth/login": self.handle_login,
            "auth/logout": self.handle_logout,
            "auth/refresh": self.handle_refresh,
            "auth/validate": self.handle_validate
        }
    
    async def handle_login(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle user login request.
        
        Args:
            params: Login parameters (username, password, etc.)
            
        Returns:
            Login response with tokens
        """
        username = params.get("username")
        password = params.get("password")
        
        if not username or not password:
            raise AuthenticationError("Username and password required")
        
        # TODO: Implement actual user authentication
        # For now, we'll use a simple demo authentication
        user_id, roles, permissions = await self._authenticate_user(username, password)
        
        # Generate tokens
        access_token, access_exp = self.jwt_manager.generate_token(
            user_id=user_id,
            roles=roles,
            permissions=permissions,
            metadata={"username": username}
        )
        
        refresh_token, refresh_exp = self.jwt_manager.generate_refresh_token(user_id)
        
        # Store session in state manager
        session_id = f"session_{user_id}"
        await self.state_manager.set(session_id, {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "permissions": permissions,
            "login_time": datetime.utcnow().isoformat(),
            "access_token_exp": access_exp.isoformat(),
            "refresh_token_exp": refresh_exp.isoformat()
        })
        
        logger.info(f"User logged in: {username} (ID: {user_id})")
        
        return {
            "success": True,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": int((access_exp - datetime.utcnow()).total_seconds()),
            "token_type": "Bearer",
            "user": {
                "id": user_id,
                "username": username,
                "roles": roles,
                "permissions": permissions
            }
        }
    
    async def handle_logout(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle user logout request.
        
        Args:
            params: Logout parameters (token)
            
        Returns:
            Logout response
        """
        token = params.get("token")
        
        if not token:
            raise AuthenticationError("Token required for logout")
        
        try:
            # Validate and revoke token
            payload = self.jwt_manager.validate_token(token)
            self.jwt_manager.revoke_token(token)
            
            # Remove session from state manager
            session_id = f"session_{payload.sub}"
            await self.state_manager.delete(session_id)
            
            logger.info(f"User logged out: {payload.sub}")
            
            return {
                "success": True,
                "message": "Successfully logged out"
            }
            
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def handle_refresh(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle token refresh request.
        
        Args:
            params: Refresh parameters (refresh_token)
            
        Returns:
            New access token
        """
        refresh_token = params.get("refresh_token")
        
        if not refresh_token:
            raise AuthenticationError("Refresh token required")
        
        try:
            # Generate new access token
            access_token, access_exp = self.jwt_manager.refresh_access_token(refresh_token)
            
            # Get user info from refresh token
            payload = self.jwt_manager.validate_token(refresh_token)
            
            # Update session in state manager
            session_id = f"session_{payload.sub}"
            session = await self.state_manager.get(session_id)
            if session:
                session["access_token_exp"] = access_exp.isoformat()
                await self.state_manager.set(session_id, session)
            
            logger.info(f"Token refreshed for user: {payload.sub}")
            
            return {
                "success": True,
                "access_token": access_token,
                "expires_in": int((access_exp - datetime.utcnow()).total_seconds()),
                "token_type": "Bearer"
            }
            
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            raise AuthenticationError(f"Failed to refresh token: {str(e)}")
    
    async def handle_validate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle token validation request.
        
        Args:
            params: Validation parameters (token)
            
        Returns:
            Validation result
        """
        token = params.get("token")
        
        if not token:
            return {
                "valid": False,
                "error": "No token provided"
            }
        
        try:
            # Validate token
            payload = self.jwt_manager.validate_token(token)
            
            # Check if session exists
            session_id = f"session_{payload.sub}"
            session = await self.state_manager.get(session_id)
            
            return {
                "valid": True,
                "user_id": payload.sub,
                "roles": payload.roles,
                "permissions": payload.permissions,
                "expires_at": datetime.fromtimestamp(payload.exp).isoformat(),
                "session_active": session is not None
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def _authenticate_user(
        self,
        username: str,
        password: str
    ) -> tuple[str, list[str], list[str]]:
        """
        Authenticate user credentials.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (user_id, roles, permissions)
            
        Raises:
            AuthenticationError: If authentication fails
        """
        # TODO: Implement actual user authentication
        # This is a demo implementation
        
        # Demo users
        demo_users = {
            "admin": {
                "password": "admin123",
                "user_id": "user_001",
                "roles": ["admin"],
                "permissions": ["*"]
            },
            "developer": {
                "password": "dev123",
                "user_id": "user_002",
                "roles": ["developer"],
                "permissions": [
                    "tools:read", "tools:execute",
                    "resources:read", "resources:write"
                ]
            },
            "reader": {
                "password": "read123",
                "user_id": "user_003",
                "roles": ["reader"],
                "permissions": ["tools:read", "resources:read"]
            }
        }
        
        user = demo_users.get(username)
        if not user or user["password"] != password:
            raise AuthenticationError("Invalid username or password")
        
        return user["user_id"], user["roles"], user["permissions"]