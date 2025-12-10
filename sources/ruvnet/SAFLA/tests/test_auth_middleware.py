"""
Comprehensive test suite for authentication middleware.

Tests authentication flow, token validation, permissions checking,
and decorator functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from safla.auth.auth_middleware import AuthMiddleware
from safla.auth.jwt_manager import JWTManager, TokenPayload
from safla.auth.exceptions import (
    AuthenticationError,
    MissingTokenError,
    InsufficientPermissionsError,
    InvalidTokenError
)


class TestAuthMiddleware:
    """Test cases for AuthMiddleware."""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create a mock JWT manager."""
        manager = Mock(spec=JWTManager)
        return manager
    
    @pytest.fixture
    def auth_middleware(self, jwt_manager):
        """Create an AuthMiddleware instance."""
        return AuthMiddleware(jwt_manager)
    
    @pytest.fixture
    def valid_token_payload(self):
        """Create a valid token payload."""
        return TokenPayload(
            sub="user123",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            roles=["developer"],
            permissions=["tools:read", "tools:execute", "resources:read"],
            jti="token123"
        )
    
    @pytest.fixture
    def admin_token_payload(self):
        """Create an admin token payload."""
        return TokenPayload(
            sub="admin123",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            roles=["admin"],
            permissions=["*"],
            jti="admintoken123"
        )
    
    def test_public_method_detection(self, auth_middleware):
        """Test detection of public methods."""
        assert auth_middleware._is_public_method("initialize")
        assert auth_middleware._is_public_method("auth/login")
        assert auth_middleware._is_public_method("auth/refresh")
        assert auth_middleware._is_public_method("health/check")
        assert not auth_middleware._is_public_method("tools/list")
        assert not auth_middleware._is_public_method("resources/read")
    
    @pytest.mark.asyncio
    async def test_authenticate_public_method(self, auth_middleware):
        """Test authentication bypass for public methods."""
        request = {
            "method": "initialize",
            "params": {}
        }
        
        result = await auth_middleware.authenticate_request(request)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_authenticate_missing_token(self, auth_middleware):
        """Test authentication with missing token."""
        request = {
            "method": "tools/list",
            "params": {}
        }
        
        with pytest.raises(MissingTokenError):
            await auth_middleware.authenticate_request(request)
    
    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self, auth_middleware, jwt_manager, valid_token_payload):
        """Test authentication with valid token."""
        request = {
            "method": "tools/list",
            "params": {
                "auth_token": "valid_token_string"
            }
        }
        
        jwt_manager.validate_token.return_value = valid_token_payload
        
        result = await auth_middleware.authenticate_request(request)
        
        assert result == valid_token_payload
        assert auth_middleware._user_context == valid_token_payload
        jwt_manager.validate_token.assert_called_once_with("valid_token_string")
    
    @pytest.mark.asyncio
    async def test_authenticate_invalid_token(self, auth_middleware, jwt_manager):
        """Test authentication with invalid token."""
        request = {
            "method": "tools/list",
            "params": {
                "auth_token": "invalid_token"
            }
        }
        
        jwt_manager.validate_token.side_effect = InvalidTokenError("Invalid token")
        
        with pytest.raises(AuthenticationError):
            await auth_middleware.authenticate_request(request)
    
    def test_extract_token_from_params(self, auth_middleware):
        """Test token extraction from request params."""
        request = {
            "params": {
                "auth_token": "token_from_params"
            }
        }
        
        token = auth_middleware._extract_token(request)
        assert token == "token_from_params"
    
    def test_extract_token_from_headers(self, auth_middleware, jwt_manager):
        """Test token extraction from headers."""
        request = {
            "params": {
                "headers": {
                    "Authorization": "Bearer token_from_header"
                }
            }
        }
        
        jwt_manager.extract_token_from_header.return_value = "token_from_header"
        
        token = auth_middleware._extract_token(request)
        assert token == "token_from_header"
        jwt_manager.extract_token_from_header.assert_called_once_with("Bearer token_from_header")
    
    def test_extract_token_from_context(self, auth_middleware):
        """Test token extraction from context."""
        request = {"params": {}}
        context = {"auth_token": "token_from_context"}
        
        token = auth_middleware._extract_token(request, context)
        assert token == "token_from_context"
    
    def test_extract_token_priority(self, auth_middleware):
        """Test token extraction priority (params > headers > context)."""
        request = {
            "params": {
                "auth_token": "token_from_params",
                "headers": {
                    "Authorization": "Bearer token_from_header"
                }
            }
        }
        context = {"auth_token": "token_from_context"}
        
        # Params should take priority
        token = auth_middleware._extract_token(request, context)
        assert token == "token_from_params"
    
    @pytest.mark.asyncio
    async def test_check_permissions_success(self, auth_middleware, valid_token_payload):
        """Test successful permission check."""
        auth_middleware._check_permissions("tools/list", valid_token_payload)
        # Should not raise any exception
    
    @pytest.mark.asyncio
    async def test_check_permissions_insufficient(self, auth_middleware, valid_token_payload):
        """Test insufficient permissions."""
        with pytest.raises(InsufficientPermissionsError) as exc_info:
            auth_middleware._check_permissions("resources/delete", valid_token_payload)
        
        assert "requires permission: resources:delete" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_check_permissions_admin_bypass(self, auth_middleware, admin_token_payload):
        """Test admin bypass for all permissions."""
        # Admin should be able to access any method
        auth_middleware._check_permissions("resources/delete", admin_token_payload)
        auth_middleware._check_permissions("tools/execute", admin_token_payload)
        # Should not raise any exception
    
    def test_has_permission_through_role(self, auth_middleware):
        """Test role-based permission checking."""
        # Developer role
        assert auth_middleware._has_permission_through_role("tools:read", ["developer"])
        assert auth_middleware._has_permission_through_role("resources:write", ["developer"])
        assert not auth_middleware._has_permission_through_role("resources:delete", ["developer"])
        
        # Reader role
        assert auth_middleware._has_permission_through_role("tools:read", ["reader"])
        assert not auth_middleware._has_permission_through_role("tools:execute", ["reader"])
        
        # Admin role
        assert auth_middleware._has_permission_through_role("any:permission", ["admin"])
        assert auth_middleware._has_permission_through_role("resources:delete", ["admin"])
        
        # Multiple roles
        assert auth_middleware._has_permission_through_role("resources:write", ["reader", "developer"])
    
    @pytest.mark.asyncio
    async def test_require_auth_decorator_success(self, auth_middleware, jwt_manager, valid_token_payload):
        """Test require_auth decorator with successful authentication."""
        jwt_manager.validate_token.return_value = valid_token_payload
        
        # Create a mock handler class
        class MockHandler:
            @auth_middleware.require_auth(permissions=["tools:read"])
            async def protected_method(self, request, user_context=None):
                return {"user": user_context.sub}
        
        handler = MockHandler()
        request = {
            "method": "tools/list",
            "params": {"auth_token": "valid_token"}
        }
        
        result = await handler.protected_method(request)
        
        assert result["user"] == "user123"
    
    @pytest.mark.asyncio
    async def test_require_auth_decorator_missing_permission(self, auth_middleware, jwt_manager, valid_token_payload):
        """Test require_auth decorator with missing permissions."""
        jwt_manager.validate_token.return_value = valid_token_payload
        
        class MockHandler:
            @auth_middleware.require_auth(permissions=["admin:write"])
            async def admin_method(self, request, user_context=None):
                return {"success": True}
        
        handler = MockHandler()
        request = {
            "method": "admin/action",
            "params": {"auth_token": "valid_token"}
        }
        
        with pytest.raises(InsufficientPermissionsError):
            await handler.admin_method(request)
    
    @pytest.mark.asyncio
    async def test_require_auth_decorator_no_token(self, auth_middleware):
        """Test require_auth decorator without token."""
        class MockHandler:
            @auth_middleware.require_auth()
            async def protected_method(self, request, user_context=None):
                return {"success": True}
        
        handler = MockHandler()
        request = {
            "method": "protected/action",
            "params": {}
        }
        
        with pytest.raises(MissingTokenError):
            await handler.protected_method(request)
    
    def test_get_current_user(self, auth_middleware, valid_token_payload):
        """Test getting current user context."""
        assert auth_middleware.get_current_user() is None
        
        auth_middleware._user_context = valid_token_payload
        assert auth_middleware.get_current_user() == valid_token_payload
    
    @pytest.mark.asyncio
    async def test_authenticate_with_different_header_case(self, auth_middleware, jwt_manager, valid_token_payload):
        """Test authentication with different header case."""
        request = {
            "method": "tools/list",
            "params": {
                "headers": {
                    "authorization": "Bearer token"  # lowercase
                }
            }
        }
        
        jwt_manager.extract_token_from_header.return_value = "token"
        jwt_manager.validate_token.return_value = valid_token_payload
        
        result = await auth_middleware.authenticate_request(request)
        assert result == valid_token_payload
    
    @pytest.mark.asyncio
    async def test_authenticate_generic_exception_handling(self, auth_middleware, jwt_manager):
        """Test generic exception handling during authentication."""
        request = {
            "method": "tools/list",
            "params": {"auth_token": "token"}
        }
        
        jwt_manager.validate_token.side_effect = Exception("Unexpected error")
        
        with pytest.raises(AuthenticationError) as exc_info:
            await auth_middleware.authenticate_request(request)
        
        assert "Authentication failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_concurrent_user_contexts(self, auth_middleware, jwt_manager):
        """Test handling of concurrent user contexts."""
        user1_payload = TokenPayload(
            sub="user1",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            roles=["developer"],
            permissions=["tools:read"],
            jti="token1"
        )
        
        user2_payload = TokenPayload(
            sub="user2",
            exp=datetime.utcnow() + timedelta(hours=1),
            iat=datetime.utcnow(),
            roles=["admin"],
            permissions=["*"],
            jti="token2"
        )
        
        # Authenticate first user
        jwt_manager.validate_token.return_value = user1_payload
        await auth_middleware.authenticate_request({
            "method": "tools/list",
            "params": {"auth_token": "token1"}
        })
        assert auth_middleware.get_current_user().sub == "user1"
        
        # Authenticate second user
        jwt_manager.validate_token.return_value = user2_payload
        await auth_middleware.authenticate_request({
            "method": "tools/list",
            "params": {"auth_token": "token2"}
        })
        assert auth_middleware.get_current_user().sub == "user2"