"""Tests for JWT authentication in SAFLA MCP Server."""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from safla.auth import JWTManager, AuthMiddleware, AuthenticationError
from safla.auth.jwt_manager import JWTConfig, TokenPayload
from safla.auth.exceptions import TokenExpiredError, InvalidTokenError, MissingTokenError
from safla.utils.state_manager import StateManager
from safla.mcp.handlers.auth_handler import AuthHandler


class TestJWTManager:
    """Test JWT Manager functionality."""
    
    @pytest.fixture
    def jwt_config(self):
        """Create test JWT configuration."""
        return JWTConfig(
            secret_key="test-secret-key-for-testing-only",
            algorithm="HS256",
            expiration_minutes=60,
            refresh_expiration_days=7
        )
    
    @pytest.fixture
    def jwt_manager(self, jwt_config):
        """Create JWT manager instance."""
        return JWTManager(jwt_config)
    
    def test_generate_token(self, jwt_manager):
        """Test token generation."""
        user_id = "test_user_123"
        roles = ["developer"]
        permissions = ["tools:read", "tools:execute"]
        
        token, expiration = jwt_manager.generate_token(
            user_id=user_id,
            roles=roles,
            permissions=permissions
        )
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert isinstance(expiration, datetime)
        assert expiration > datetime.utcnow()
    
    def test_validate_token(self, jwt_manager):
        """Test token validation."""
        user_id = "test_user_123"
        roles = ["admin"]
        permissions = ["*"]
        
        token, _ = jwt_manager.generate_token(
            user_id=user_id,
            roles=roles,
            permissions=permissions
        )
        
        payload = jwt_manager.validate_token(token)
        
        assert payload.sub == user_id
        assert payload.roles == roles
        assert payload.permissions == permissions
    
    def test_expired_token(self, jwt_manager):
        """Test expired token validation."""
        user_id = "test_user_123"
        
        # Generate token with very short expiration
        token, _ = jwt_manager.generate_token(
            user_id=user_id,
            custom_expiration=timedelta(seconds=-1)  # Already expired
        )
        
        with pytest.raises(TokenExpiredError):
            jwt_manager.validate_token(token)
    
    def test_invalid_token(self, jwt_manager):
        """Test invalid token validation."""
        invalid_token = "invalid.token.here"
        
        with pytest.raises(InvalidTokenError):
            jwt_manager.validate_token(invalid_token)
    
    def test_refresh_token(self, jwt_manager):
        """Test refresh token generation and usage."""
        user_id = "test_user_123"
        
        # Generate refresh token
        refresh_token, refresh_exp = jwt_manager.generate_refresh_token(user_id)
        
        # Validate it's a refresh token
        payload = jwt_manager.validate_token(refresh_token)
        assert payload.metadata.get("token_type") == "refresh"
        
        # Use refresh token to get new access token
        new_token, new_exp = jwt_manager.refresh_access_token(refresh_token)
        
        # Validate new token
        new_payload = jwt_manager.validate_token(new_token)
        assert new_payload.sub == user_id
        assert new_payload.metadata.get("token_type") != "refresh"


class TestAuthMiddleware:
    """Test authentication middleware."""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager for testing."""
        config = JWTConfig(secret_key="test-secret-key")
        return JWTManager(config)
    
    @pytest.fixture
    def auth_middleware(self, jwt_manager):
        """Create auth middleware instance."""
        return AuthMiddleware(jwt_manager)
    
    @pytest.mark.asyncio
    async def test_public_method_no_auth(self, auth_middleware):
        """Test that public methods don't require authentication."""
        request = {
            "method": "initialize",
            "params": {},
            "id": 1
        }
        
        result = await auth_middleware.authenticate_request(request)
        assert result is None  # No authentication needed
    
    @pytest.mark.asyncio
    async def test_protected_method_requires_auth(self, auth_middleware):
        """Test that protected methods require authentication."""
        request = {
            "method": "tools/call",
            "params": {},
            "id": 1
        }
        
        with pytest.raises(MissingTokenError):
            await auth_middleware.authenticate_request(request)
    
    @pytest.mark.asyncio
    async def test_valid_token_authentication(self, auth_middleware, jwt_manager):
        """Test authentication with valid token."""
        # Generate valid token
        token, _ = jwt_manager.generate_token(
            user_id="test_user",
            permissions=["tools:execute"]
        )
        
        request = {
            "method": "tools/call",
            "params": {
                "auth_token": token
            },
            "id": 1
        }
        
        payload = await auth_middleware.authenticate_request(request)
        assert payload.sub == "test_user"
        assert "tools:execute" in payload.permissions
    
    @pytest.mark.asyncio
    async def test_insufficient_permissions(self, auth_middleware, jwt_manager):
        """Test authentication with insufficient permissions."""
        # Generate token without required permissions
        token, _ = jwt_manager.generate_token(
            user_id="test_user",
            permissions=["tools:read"]  # Missing tools:execute
        )
        
        request = {
            "method": "tools/call",
            "params": {
                "auth_token": token
            },
            "id": 1
        }
        
        with pytest.raises(InsufficientPermissionsError):
            await auth_middleware.authenticate_request(request)


class TestAuthHandler:
    """Test authentication handler."""
    
    @pytest.fixture
    def jwt_manager(self):
        """Create JWT manager for testing."""
        config = JWTConfig(secret_key="test-secret-key")
        return JWTManager(config)
    
    @pytest.fixture
    def state_manager(self):
        """Create mock state manager."""
        return Mock(spec=StateManager)
    
    @pytest.fixture
    def auth_handler(self, jwt_manager, state_manager):
        """Create auth handler instance."""
        return AuthHandler(jwt_manager, state_manager)
    
    @pytest.mark.asyncio
    async def test_login_success(self, auth_handler, state_manager):
        """Test successful login."""
        params = {
            "username": "admin",
            "password": "admin123"
        }
        
        result = await auth_handler.handle_login(params)
        
        assert result["success"] is True
        assert "access_token" in result
        assert "refresh_token" in result
        assert result["token_type"] == "Bearer"
        assert result["user"]["username"] == "admin"
        assert result["user"]["roles"] == ["admin"]
        
        # Verify session was stored
        state_manager.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, auth_handler):
        """Test login with invalid credentials."""
        params = {
            "username": "admin",
            "password": "wrong_password"
        }
        
        with pytest.raises(AuthenticationError, match="Invalid username or password"):
            await auth_handler.handle_login(params)
    
    @pytest.mark.asyncio
    async def test_logout(self, auth_handler, jwt_manager, state_manager):
        """Test logout functionality."""
        # Generate a valid token
        token, _ = jwt_manager.generate_token(user_id="test_user")
        
        params = {"token": token}
        result = await auth_handler.handle_logout(params)
        
        assert result["success"] is True
        assert "Successfully logged out" in result["message"]
        
        # Verify session was deleted
        state_manager.delete.assert_called_once_with("session_test_user")
    
    @pytest.mark.asyncio
    async def test_token_refresh(self, auth_handler, jwt_manager):
        """Test token refresh functionality."""
        # Generate refresh token
        refresh_token, _ = jwt_manager.generate_refresh_token("test_user")
        
        params = {"refresh_token": refresh_token}
        result = await auth_handler.handle_refresh(params)
        
        assert result["success"] is True
        assert "access_token" in result
        assert result["token_type"] == "Bearer"
    
    @pytest.mark.asyncio
    async def test_token_validation(self, auth_handler, jwt_manager):
        """Test token validation endpoint."""
        # Generate valid token
        token, _ = jwt_manager.generate_token(
            user_id="test_user",
            roles=["developer"],
            permissions=["tools:read"]
        )
        
        params = {"token": token}
        result = await auth_handler.handle_validate(params)
        
        assert result["valid"] is True
        assert result["user_id"] == "test_user"
        assert result["roles"] == ["developer"]
        assert result["permissions"] == ["tools:read"]