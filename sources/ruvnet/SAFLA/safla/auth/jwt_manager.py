"""JWT Manager for token generation and validation."""

import os
import jwt
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from .exceptions import TokenExpiredError, InvalidTokenError

logger = logging.getLogger(__name__)


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    
    sub: str  # Subject (user ID)
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    jti: Optional[str] = None  # JWT ID for token tracking
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JWTConfig(BaseModel):
    """JWT configuration settings."""
    
    secret_key: str = Field(default_factory=lambda: os.getenv("JWT_SECRET_KEY", ""))
    algorithm: str = Field(default="HS256")
    expiration_minutes: int = Field(
        default_factory=lambda: int(os.getenv("JWT_EXPIRATION_TIME", "60"))
    )
    refresh_expiration_days: int = Field(default=7)
    issuer: str = Field(default="safla-mcp-server")
    audience: str = Field(default="safla-client")


class JWTManager:
    """Manages JWT token operations."""
    
    def __init__(self, config: Optional[JWTConfig] = None):
        """Initialize JWT manager with configuration."""
        self.config = config or JWTConfig()
        
        if not self.config.secret_key:
            raise ValueError("JWT_SECRET_KEY must be set in environment variables")
        
        logger.info(f"JWT Manager initialized with algorithm: {self.config.algorithm}")
    
    def generate_token(
        self,
        user_id: str,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        custom_expiration: Optional[timedelta] = None
    ) -> Tuple[str, datetime]:
        """
        Generate a JWT token for a user.
        
        Args:
            user_id: Unique identifier for the user
            roles: User roles for authorization
            permissions: Specific permissions granted
            metadata: Additional user metadata
            custom_expiration: Custom token expiration time
            
        Returns:
            Tuple of (token, expiration_datetime)
        """
        now = datetime.now(timezone.utc)
        
        if custom_expiration:
            expiration = now + custom_expiration
        else:
            expiration = now + timedelta(minutes=self.config.expiration_minutes)
        
        payload = TokenPayload(
            sub=user_id,
            exp=int(expiration.timestamp()),
            iat=int(now.timestamp()),
            jti=self._generate_jti(),
            roles=roles or [],
            permissions=permissions or [],
            metadata=metadata or {}
        )
        
        token = jwt.encode(
            payload.model_dump(),
            self.config.secret_key,
            algorithm=self.config.algorithm
        )
        
        logger.info(f"Generated token for user: {user_id}")
        return token, expiration
    
    def generate_refresh_token(self, user_id: str) -> Tuple[str, datetime]:
        """
        Generate a refresh token with longer expiration.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Tuple of (refresh_token, expiration_datetime)
        """
        expiration = timedelta(days=self.config.refresh_expiration_days)
        return self.generate_token(
            user_id=user_id,
            metadata={"token_type": "refresh"},
            custom_expiration=expiration
        )
    
    def validate_token(self, token: str) -> TokenPayload:
        """
        Validate and decode a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            TokenExpiredError: If token has expired
            InvalidTokenError: If token is invalid
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": True}
            )
            
            # Convert to TokenPayload model
            token_payload = TokenPayload(**payload)
            
            logger.debug(f"Token validated for user: {token_payload.sub}")
            return token_payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: expired")
            raise TokenExpiredError()
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: {str(e)}")
            raise InvalidTokenError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during token validation: {str(e)}")
            raise InvalidTokenError(f"Token validation error: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> Tuple[str, datetime]:
        """
        Generate a new access token using a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Tuple of (new_access_token, expiration_datetime)
            
        Raises:
            InvalidTokenError: If refresh token is invalid or not a refresh type
        """
        payload = self.validate_token(refresh_token)
        
        # Verify this is a refresh token
        if payload.metadata.get("token_type") != "refresh":
            raise InvalidTokenError("Token is not a refresh token")
        
        # Generate new access token with original user data
        return self.generate_token(
            user_id=payload.sub,
            roles=payload.roles,
            permissions=payload.permissions,
            metadata={k: v for k, v in payload.metadata.items() if k != "token_type"}
        )
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token (requires external storage implementation).
        
        Args:
            token: Token to revoke
            
        Returns:
            True if successfully revoked
        """
        # This would typically add the token JTI to a blacklist in Redis/DB
        # For now, we'll just validate it exists
        try:
            payload = self.validate_token(token)
            logger.info(f"Token revoked for user: {payload.sub}")
            # TODO: Implement actual revocation with external storage
            return True
        except Exception:
            return False
    
    def _generate_jti(self) -> str:
        """Generate a unique JWT ID."""
        import uuid
        return str(uuid.uuid4())
    
    def extract_token_from_header(self, auth_header: str) -> str:
        """
        Extract token from Authorization header.
        
        Args:
            auth_header: Authorization header value
            
        Returns:
            Extracted token
            
        Raises:
            InvalidTokenError: If header format is invalid
        """
        if not auth_header:
            raise InvalidTokenError("No authorization header provided")
        
        parts = auth_header.split()
        
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise InvalidTokenError("Invalid authorization header format")
        
        return parts[1]