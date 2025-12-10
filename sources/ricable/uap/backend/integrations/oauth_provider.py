# File: backend/integrations/oauth_provider.py
"""
OAuth2 provider for secure third-party access to UAP APIs.
"""

import secrets
import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlencode, parse_qs
from pydantic import BaseModel, Field
from enum import Enum
import jwt
import uuid


class OAuth2Error(Exception):
    """OAuth2 specific error"""
    def __init__(self, error: str, description: str = None, error_uri: str = None):
        self.error = error
        self.description = description
        self.error_uri = error_uri
        super().__init__(f"OAuth2 Error: {error} - {description}")


class OAuth2ResponseType(str, Enum):
    """OAuth2 response types"""
    CODE = "code"
    TOKEN = "token"


class OAuth2GrantType(str, Enum):
    """OAuth2 grant types"""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"


class OAuth2ClientType(str, Enum):
    """OAuth2 client types"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"


class OAuth2Client(BaseModel):
    """OAuth2 client configuration"""
    client_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    client_name: str
    client_type: OAuth2ClientType
    redirect_uris: List[str]
    scopes: List[str]
    grant_types: List[OAuth2GrantType]
    response_types: List[OAuth2ResponseType]
    owner_id: str  # UAP user who registered the client
    description: Optional[str] = None
    logo_uri: Optional[str] = None
    homepage_uri: Optional[str] = None
    terms_uri: Optional[str] = None
    privacy_uri: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OAuth2AuthorizationCode(BaseModel):
    """OAuth2 authorization code"""
    code: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    client_id: str
    user_id: str
    redirect_uri: str
    scopes: List[str]
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=10))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OAuth2AccessToken(BaseModel):
    """OAuth2 access token"""
    access_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    token_type: str = "bearer"
    refresh_token: Optional[str] = Field(default_factory=lambda: secrets.token_urlsafe(32))
    client_id: str
    user_id: str
    scopes: List[str]
    expires_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1))
    refresh_expires_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OAuth2Scope(BaseModel):
    """OAuth2 scope definition"""
    name: str
    description: str
    permissions: List[str]  # UAP permissions required for this scope


# Default OAuth2 scopes for UAP
DEFAULT_OAUTH2_SCOPES = {
    "read": OAuth2Scope(
        name="read",
        description="Read access to user data and agent interactions",
        permissions=["agent:read", "user:read"]
    ),
    "write": OAuth2Scope(
        name="write", 
        description="Write access to create and modify data",
        permissions=["agent:create", "agent:update"]
    ),
    "agents": OAuth2Scope(
        name="agents",
        description="Full access to agent management",
        permissions=["agent:create", "agent:read", "agent:update", "agent:delete"]
    ),
    "documents": OAuth2Scope(
        name="documents",
        description="Access to document processing services",
        permissions=["agent:create", "agent:read"]  # Documents use agent:create for processing
    ),
    "webhooks": OAuth2Scope(
        name="webhooks",
        description="Access to webhook management",
        permissions=["system:manage"]
    ),
    "admin": OAuth2Scope(
        name="admin",
        description="Administrative access to UAP",
        permissions=["system:admin", "user:create", "user:update", "user:delete"]
    )
}


class OAuth2Provider:
    """
    OAuth2 provider implementation for UAP.
    
    Provides secure third-party access to UAP APIs using OAuth2 authorization code flow.
    """
    
    def __init__(self, auth_service, base_url: str = "http://localhost:8000"):
        self.auth_service = auth_service
        self.base_url = base_url
        self.clients: Dict[str, OAuth2Client] = {}
        self.authorization_codes: Dict[str, OAuth2AuthorizationCode] = {}
        self.access_tokens: Dict[str, OAuth2AccessToken] = {}
        self.scopes = DEFAULT_OAUTH2_SCOPES.copy()
        
        # OAuth2 endpoints
        self.authorization_endpoint = f"{base_url}/oauth2/authorize"
        self.token_endpoint = f"{base_url}/oauth2/token"
        self.introspection_endpoint = f"{base_url}/oauth2/introspect"
        self.revocation_endpoint = f"{base_url}/oauth2/revoke"
    
    def register_client(self, client_data: Dict[str, Any], owner_id: str) -> OAuth2Client:
        """
        Register a new OAuth2 client application.
        
        Args:
            client_data: Client registration data
            owner_id: UAP user ID who owns this client
            
        Returns:
            Registered OAuth2Client
        """
        # Validate redirect URIs
        redirect_uris = client_data.get("redirect_uris", [])
        if not redirect_uris:
            raise OAuth2Error("invalid_client_metadata", "redirect_uris is required")
        
        # Validate scopes
        requested_scopes = client_data.get("scopes", ["read"])
        invalid_scopes = [s for s in requested_scopes if s not in self.scopes]
        if invalid_scopes:
            raise OAuth2Error("invalid_scope", f"Invalid scopes: {invalid_scopes}")
        
        client = OAuth2Client(
            client_name=client_data["client_name"],
            client_type=OAuth2ClientType(client_data.get("client_type", "confidential")),
            redirect_uris=redirect_uris,
            scopes=requested_scopes,
            grant_types=[OAuth2GrantType.AUTHORIZATION_CODE, OAuth2GrantType.REFRESH_TOKEN],
            response_types=[OAuth2ResponseType.CODE],
            owner_id=owner_id,
            description=client_data.get("description"),
            logo_uri=client_data.get("logo_uri"),
            homepage_uri=client_data.get("homepage_uri")
        )
        
        self.clients[client.client_id] = client
        return client
    
    def get_client(self, client_id: str) -> Optional[OAuth2Client]:
        """Get OAuth2 client by ID."""
        return self.clients.get(client_id)
    
    def authenticate_client(self, client_id: str, client_secret: str = None) -> OAuth2Client:
        """
        Authenticate OAuth2 client.
        
        Args:
            client_id: Client identifier
            client_secret: Client secret (required for confidential clients)
            
        Returns:
            Authenticated OAuth2Client
            
        Raises:
            OAuth2Error: If authentication fails
        """
        client = self.get_client(client_id)
        if not client:
            raise OAuth2Error("invalid_client", "Client not found")
        
        if not client.is_active:
            raise OAuth2Error("invalid_client", "Client is disabled")
        
        if client.client_type == OAuth2ClientType.CONFIDENTIAL:
            if not client_secret:
                raise OAuth2Error("invalid_client", "Client secret required")
            if client_secret != client.client_secret:
                raise OAuth2Error("invalid_client", "Invalid client secret")
        
        return client
    
    def create_authorization_url(self, client_id: str, redirect_uri: str, scopes: List[str],
                               state: str = None, code_challenge: str = None,
                               code_challenge_method: str = None) -> str:
        """
        Create OAuth2 authorization URL.
        
        Args:
            client_id: Client identifier
            redirect_uri: Redirect URI for authorization response
            scopes: Requested scopes
            state: Optional state parameter
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE code challenge method
            
        Returns:
            Authorization URL
        """
        client = self.get_client(client_id)
        if not client:
            raise OAuth2Error("invalid_client", "Client not found")
        
        if redirect_uri not in client.redirect_uris:
            raise OAuth2Error("invalid_request", "Invalid redirect_uri")
        
        # Validate scopes
        invalid_scopes = [s for s in scopes if s not in self.scopes]
        if invalid_scopes:
            raise OAuth2Error("invalid_scope", f"Invalid scopes: {invalid_scopes}")
        
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes)
        }
        
        if state:
            params["state"] = state
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = code_challenge_method or "S256"
        
        return f"{self.authorization_endpoint}?{urlencode(params)}"
    
    def generate_authorization_code(self, client_id: str, user_id: str, redirect_uri: str,
                                  scopes: List[str], code_challenge: str = None,
                                  code_challenge_method: str = None) -> OAuth2AuthorizationCode:
        """
        Generate authorization code after user consent.
        
        Args:
            client_id: Client identifier
            user_id: User who authorized the request
            redirect_uri: Redirect URI
            scopes: Authorized scopes
            code_challenge: PKCE code challenge
            code_challenge_method: PKCE code challenge method
            
        Returns:
            Generated OAuth2AuthorizationCode
        """
        auth_code = OAuth2AuthorizationCode(
            client_id=client_id,
            user_id=user_id,
            redirect_uri=redirect_uri,
            scopes=scopes,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method
        )
        
        self.authorization_codes[auth_code.code] = auth_code
        return auth_code
    
    def exchange_code_for_tokens(self, code: str, client_id: str, client_secret: str = None,
                               redirect_uri: str = None, code_verifier: str = None) -> OAuth2AccessToken:
        """
        Exchange authorization code for access token.
        
        Args:
            code: Authorization code
            client_id: Client identifier
            client_secret: Client secret
            redirect_uri: Redirect URI used in authorization
            code_verifier: PKCE code verifier
            
        Returns:
            OAuth2AccessToken
        """
        # Authenticate client
        client = self.authenticate_client(client_id, client_secret)
        
        # Validate authorization code
        auth_code = self.authorization_codes.get(code)
        if not auth_code:
            raise OAuth2Error("invalid_grant", "Invalid authorization code")
        
        # Check expiration
        if datetime.now(timezone.utc) > auth_code.expires_at:
            del self.authorization_codes[code]
            raise OAuth2Error("invalid_grant", "Authorization code expired")
        
        # Validate client
        if auth_code.client_id != client_id:
            raise OAuth2Error("invalid_grant", "Code issued to different client")
        
        # Validate redirect URI
        if redirect_uri and auth_code.redirect_uri != redirect_uri:
            raise OAuth2Error("invalid_grant", "Invalid redirect_uri")
        
        # Validate PKCE if used
        if auth_code.code_challenge:
            if not code_verifier:
                raise OAuth2Error("invalid_request", "code_verifier required")
            
            if auth_code.code_challenge_method == "S256":
                challenge = base64.urlsafe_b64encode(
                    hashlib.sha256(code_verifier.encode()).digest()
                ).decode().rstrip("=")
            else:
                challenge = code_verifier
            
            if challenge != auth_code.code_challenge:
                raise OAuth2Error("invalid_grant", "Invalid code_verifier")
        
        # Create access token
        access_token = OAuth2AccessToken(
            client_id=client_id,
            user_id=auth_code.user_id,
            scopes=auth_code.scopes
        )
        
        self.access_tokens[access_token.access_token] = access_token
        
        # Clean up authorization code
        del self.authorization_codes[code]
        
        return access_token
    
    def refresh_access_token(self, refresh_token: str, client_id: str,
                           client_secret: str = None, scopes: List[str] = None) -> OAuth2AccessToken:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Refresh token
            client_id: Client identifier
            client_secret: Client secret
            scopes: Optional new scopes (must be subset of original)
            
        Returns:
            New OAuth2AccessToken
        """
        # Authenticate client
        client = self.authenticate_client(client_id, client_secret)
        
        # Find token by refresh token
        old_token = None
        for token in self.access_tokens.values():
            if token.refresh_token == refresh_token:
                old_token = token
                break
        
        if not old_token:
            raise OAuth2Error("invalid_grant", "Invalid refresh token")
        
        # Check expiration
        if old_token.refresh_expires_at and datetime.now(timezone.utc) > old_token.refresh_expires_at:
            # Clean up expired token
            self.access_tokens.pop(old_token.access_token, None)
            raise OAuth2Error("invalid_grant", "Refresh token expired")
        
        # Validate client
        if old_token.client_id != client_id:
            raise OAuth2Error("invalid_grant", "Token issued to different client")
        
        # Validate scopes
        token_scopes = old_token.scopes
        if scopes:
            invalid_scopes = [s for s in scopes if s not in old_token.scopes]
            if invalid_scopes:
                raise OAuth2Error("invalid_scope", f"Requested scopes exceed original: {invalid_scopes}")
            token_scopes = scopes
        
        # Create new access token
        new_token = OAuth2AccessToken(
            client_id=client_id,
            user_id=old_token.user_id,
            scopes=token_scopes
        )
        
        self.access_tokens[new_token.access_token] = new_token
        
        # Revoke old token
        self.access_tokens.pop(old_token.access_token, None)
        
        return new_token
    
    def introspect_token(self, token: str, client_id: str = None) -> Dict[str, Any]:
        """
        Introspect access token to get information.
        
        Args:
            token: Access token to introspect
            client_id: Optional client ID for validation
            
        Returns:
            Token introspection response
        """
        access_token = self.access_tokens.get(token)
        if not access_token:
            return {"active": False}
        
        # Check expiration
        if datetime.now(timezone.utc) > access_token.expires_at:
            self.access_tokens.pop(token, None)
            return {"active": False}
        
        # Optional client validation
        if client_id and access_token.client_id != client_id:
            return {"active": False}
        
        return {
            "active": True,
            "client_id": access_token.client_id,
            "username": access_token.user_id,  # Could look up actual username
            "scope": " ".join(access_token.scopes),
            "exp": int(access_token.expires_at.timestamp()),
            "iat": int(access_token.created_at.timestamp()),
            "token_type": "bearer"
        }
    
    def revoke_token(self, token: str, token_type_hint: str = None, client_id: str = None):
        """
        Revoke access or refresh token.
        
        Args:
            token: Token to revoke
            token_type_hint: Hint about token type ("access_token" or "refresh_token")
            client_id: Optional client ID for validation
        """
        # Look for access token
        access_token = self.access_tokens.get(token)
        if access_token:
            if client_id and access_token.client_id != client_id:
                return  # Silently ignore invalid client
            self.access_tokens.pop(token, None)
            return
        
        # Look for token by refresh token
        for access_token in list(self.access_tokens.values()):
            if access_token.refresh_token == token:
                if client_id and access_token.client_id != client_id:
                    return  # Silently ignore invalid client
                self.access_tokens.pop(access_token.access_token, None)
                return
    
    def validate_token_permissions(self, token: str, required_permission: str) -> bool:
        """
        Validate that an access token has required permission.
        
        Args:
            token: Access token
            required_permission: Required UAP permission
            
        Returns:
            True if token has permission
        """
        access_token = self.access_tokens.get(token)
        if not access_token:
            return False
        
        # Check expiration
        if datetime.now(timezone.utc) > access_token.expires_at:
            self.access_tokens.pop(token, None)
            return False
        
        # Check scopes for permission
        for scope_name in access_token.scopes:
            scope = self.scopes.get(scope_name)
            if scope and required_permission in scope.permissions:
                return True
        
        return False
    
    def get_user_from_token(self, token: str) -> Optional[str]:
        """
        Get user ID from access token.
        
        Args:
            token: Access token
            
        Returns:
            User ID or None if invalid
        """
        access_token = self.access_tokens.get(token)
        if not access_token:
            return None
        
        # Check expiration
        if datetime.now(timezone.utc) > access_token.expires_at:
            self.access_tokens.pop(token, None)
            return None
        
        return access_token.user_id
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens and authorization codes."""
        now = datetime.now(timezone.utc)
        
        # Clean up expired authorization codes
        expired_codes = [
            code for code, auth_code in self.authorization_codes.items()
            if now > auth_code.expires_at
        ]
        for code in expired_codes:
            del self.authorization_codes[code]
        
        # Clean up expired access tokens
        expired_tokens = [
            token for token, access_token in self.access_tokens.items()
            if now > access_token.expires_at
        ]
        for token in expired_tokens:
            del self.access_tokens[token]
    
    def get_client_statistics(self, client_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for a client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            Usage statistics
        """
        client = self.get_client(client_id)
        if not client:
            return {}
        
        # Count active tokens
        active_tokens = [
            token for token in self.access_tokens.values()
            if token.client_id == client_id and datetime.now(timezone.utc) < token.expires_at
        ]
        
        return {
            "client_id": client_id,
            "client_name": client.client_name,
            "active_tokens": len(active_tokens),
            "total_scopes": len(client.scopes),
            "created_at": client.created_at.isoformat(),
            "is_active": client.is_active
        }