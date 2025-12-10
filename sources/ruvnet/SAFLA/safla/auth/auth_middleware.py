"""Authentication middleware for MCP server requests."""

import logging
from typing import Dict, Any, Optional, Callable, List
from functools import wraps

from .jwt_manager import JWTManager, TokenPayload
from .exceptions import (
    AuthenticationError,
    MissingTokenError,
    InsufficientPermissionsError
)

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """Middleware for authenticating MCP server requests."""
    
    # Methods that don't require authentication
    PUBLIC_METHODS = [
        "initialize",
        "auth/login",
        "auth/refresh",
        "health/check"
    ]
    
    # Method-level permission requirements
    METHOD_PERMISSIONS = {
        "tools/list": ["tools:read"],
        "tools/call": ["tools:execute"],
        "resources/list": ["resources:read"],
        "resources/read": ["resources:read"],
        "resources/write": ["resources:write"],
        "resources/delete": ["resources:delete"]
    }
    
    def __init__(self, jwt_manager: JWTManager):
        """Initialize middleware with JWT manager."""
        self.jwt_manager = jwt_manager
        self._user_context: Optional[TokenPayload] = None
    
    async def authenticate_request(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> TokenPayload:
        """
        Authenticate an incoming request.
        
        Args:
            request: MCP request dictionary
            context: Optional request context
            
        Returns:
            Authenticated user token payload
            
        Raises:
            AuthenticationError: If authentication fails
        """
        method = request.get("method", "")
        
        # Check if method requires authentication
        if self._is_public_method(method):
            logger.debug(f"Public method accessed: {method}")
            return None
        
        # Extract token from request
        token = self._extract_token(request, context)
        if not token:
            raise MissingTokenError()
        
        # Validate token
        try:
            payload = self.jwt_manager.validate_token(token)
            
            # Check method permissions
            self._check_permissions(method, payload)
            
            # Store user context
            self._user_context = payload
            
            logger.info(f"Authenticated request from user: {payload.sub} for method: {method}")
            return payload
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise AuthenticationError(f"Authentication failed: {str(e)}")
    
    def require_auth(self, permissions: Optional[List[str]] = None):
        """
        Decorator for methods that require authentication.
        
        Args:
            permissions: Required permissions for the method
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(self, request: Dict[str, Any], *args, **kwargs):
                # Authenticate request
                payload = await self.authenticate_request(request)
                
                # Check additional permissions if specified
                if permissions and payload:
                    for permission in permissions:
                        if permission not in payload.permissions:
                            raise InsufficientPermissionsError(
                                f"Missing required permission: {permission}"
                            )
                
                # Add user context to kwargs
                kwargs["user_context"] = payload
                
                return await func(self, request, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_current_user(self) -> Optional[TokenPayload]:
        """Get the currently authenticated user context."""
        return self._user_context
    
    def _is_public_method(self, method: str) -> bool:
        """Check if a method is public (doesn't require auth)."""
        return method in self.PUBLIC_METHODS
    
    def _extract_token(
        self,
        request: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Extract authentication token from request.
        
        Args:
            request: MCP request dictionary
            context: Optional request context
            
        Returns:
            Token string or None
        """
        # Check params for token (MCP style)
        params = request.get("params", {})
        if "auth_token" in params:
            return params["auth_token"]
        
        # Check headers in params
        headers = params.get("headers", {})
        auth_header = headers.get("Authorization") or headers.get("authorization")
        if auth_header:
            try:
                return self.jwt_manager.extract_token_from_header(auth_header)
            except Exception:
                pass
        
        # Check context for token
        if context:
            if "auth_token" in context:
                return context["auth_token"]
            
            # Check context headers
            ctx_headers = context.get("headers", {})
            auth_header = ctx_headers.get("Authorization") or ctx_headers.get("authorization")
            if auth_header:
                try:
                    return self.jwt_manager.extract_token_from_header(auth_header)
                except Exception:
                    pass
        
        return None
    
    def _check_permissions(self, method: str, payload: TokenPayload) -> None:
        """
        Check if user has required permissions for method.
        
        Args:
            method: MCP method name
            payload: User token payload
            
        Raises:
            InsufficientPermissionsError: If user lacks permissions
        """
        # Check method-specific permissions
        required_permissions = self.METHOD_PERMISSIONS.get(method, [])
        
        for permission in required_permissions:
            if permission not in payload.permissions:
                # Check if user has a role that grants this permission
                if not self._has_permission_through_role(permission, payload.roles):
                    raise InsufficientPermissionsError(
                        f"Method '{method}' requires permission: {permission}"
                    )
    
    def _has_permission_through_role(self, permission: str, roles: List[str]) -> bool:
        """
        Check if permission is granted through user roles.
        
        Args:
            permission: Required permission
            roles: User roles
            
        Returns:
            True if permission is granted through roles
        """
        # Role-based permission mapping
        role_permissions = {
            "admin": ["*"],  # Admin has all permissions
            "developer": [
                "tools:read", "tools:execute",
                "resources:read", "resources:write"
            ],
            "reader": ["tools:read", "resources:read"]
        }
        
        for role in roles:
            if role in role_permissions:
                role_perms = role_permissions[role]
                if "*" in role_perms or permission in role_perms:
                    return True
        
        return False