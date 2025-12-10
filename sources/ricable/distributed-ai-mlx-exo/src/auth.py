"""
Authentication and Authorization System for Distributed API Gateway
Handles API key validation, user permissions, and security middleware
"""

import asyncio
import logging
import time
import secrets
import hashlib
import hmac
import jwt
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

try:
    from fastapi import Request, HTTPException, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available - auth module will have limited functionality")

logger = logging.getLogger(__name__)

class Permission(Enum):
    """Available permissions"""
    INFERENCE = "inference"
    MODEL_LIST = "model_list"
    MODEL_LOAD = "model_load"
    MODEL_UNLOAD = "model_unload"
    METRICS_READ = "metrics_read"
    CLUSTER_STATUS = "cluster_status"
    ADMIN = "admin"

class UserRole(Enum):
    """User roles with associated permissions"""
    GUEST = "guest"
    USER = "user"
    POWER_USER = "power_user"
    ADMIN = "admin"
    SYSTEM = "system"

# Role permissions mapping
ROLE_PERMISSIONS = {
    UserRole.GUEST: {Permission.MODEL_LIST},
    UserRole.USER: {Permission.INFERENCE, Permission.MODEL_LIST, Permission.CLUSTER_STATUS},
    UserRole.POWER_USER: {Permission.INFERENCE, Permission.MODEL_LIST, Permission.MODEL_LOAD, 
                          Permission.CLUSTER_STATUS, Permission.METRICS_READ},
    UserRole.ADMIN: {Permission.INFERENCE, Permission.MODEL_LIST, Permission.MODEL_LOAD, 
                     Permission.MODEL_UNLOAD, Permission.CLUSTER_STATUS, Permission.METRICS_READ, Permission.ADMIN},
    UserRole.SYSTEM: set(Permission)  # All permissions
}

@dataclass
class ApiKey:
    """API key information"""
    key_id: str
    key_hash: str
    user_id: str
    role: UserRole
    permissions: Set[Permission] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    last_used_at: Optional[float] = None
    expires_at: Optional[float] = None
    is_active: bool = True
    usage_count: int = 0
    rate_limit: Optional[int] = None  # requests per minute
    description: str = ""
    
    def __post_init__(self):
        if not self.permissions:
            self.permissions = ROLE_PERMISSIONS.get(self.role, set())
    
    @property
    def is_expired(self) -> bool:
        """Check if key is expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if key is valid for use"""
        return self.is_active and not self.is_expired
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if key has specific permission"""
        return self.is_valid and permission in self.permissions

@dataclass
class AuthRequest:
    """Authentication request context"""
    api_key: Optional[ApiKey]
    user_id: Optional[str]
    permissions: Set[Permission]
    request_id: str
    client_ip: str
    user_agent: str
    timestamp: float = field(default_factory=time.time)

class AuthenticationManager:
    """
    Manages API keys, user authentication, and authorization
    """
    
    def __init__(self, jwt_secret: Optional[str] = None):
        self.api_keys: Dict[str, ApiKey] = {}
        self.key_usage: Dict[str, List[float]] = {}  # For rate limiting
        self.blocked_ips: Set[str] = set()
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        
        # Configuration
        self.max_key_age_days = 365
        self.cleanup_interval = 3600  # 1 hour
        self.rate_limit_window = 60  # 1 minute
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        
        # Tracking
        self.failed_attempts: Dict[str, List[float]] = {}
        self.lockout_until: Dict[str, float] = {}
        
        logger.info("Authentication manager initialized")
    
    def generate_api_key(
        self,
        user_id: str,
        role: UserRole,
        description: str = "",
        expires_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        custom_permissions: Optional[Set[Permission]] = None
    ) -> Tuple[str, ApiKey]:
        """
        Generate a new API key
        """
        # Generate secure key
        key_id = f"sk-{secrets.token_urlsafe(8)}"
        raw_key = f"{key_id}.{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Set expiration
        expires_at = None
        if expires_days:
            expires_at = time.time() + (expires_days * 24 * 3600)
        
        # Create API key object
        api_key = ApiKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            role=role,
            permissions=custom_permissions or ROLE_PERMISSIONS.get(role, set()),
            expires_at=expires_at,
            rate_limit=rate_limit,
            description=description
        )
        
        # Store key
        self.api_keys[key_hash] = api_key
        self.key_usage[key_hash] = []
        
        logger.info(f"Generated API key {key_id} for user {user_id} with role {role.value}")
        return raw_key, api_key
    
    def authenticate_key(self, raw_key: str) -> Optional[ApiKey]:
        """
        Authenticate an API key
        """
        if not raw_key or not raw_key.startswith('sk-'):
            return None
        
        # Hash the key
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Find key
        api_key = self.api_keys.get(key_hash)
        if not api_key or not api_key.is_valid:
            return None
        
        # Update usage
        api_key.last_used_at = time.time()
        api_key.usage_count += 1
        
        # Track for rate limiting
        if key_hash not in self.key_usage:
            self.key_usage[key_hash] = []
        self.key_usage[key_hash].append(time.time())
        
        # Clean old usage records
        cutoff = time.time() - self.rate_limit_window
        self.key_usage[key_hash] = [t for t in self.key_usage[key_hash] if t > cutoff]
        
        return api_key
    
    def check_rate_limit(self, api_key: ApiKey) -> bool:
        """
        Check if API key is within rate limits
        """
        if not api_key.rate_limit:
            return True
        
        key_hash = api_key.key_hash
        recent_requests = len(self.key_usage.get(key_hash, []))
        
        return recent_requests <= api_key.rate_limit
    
    def check_ip_blocked(self, client_ip: str) -> bool:
        """
        Check if IP is blocked due to failed attempts
        """
        if client_ip in self.blocked_ips:
            return True
        
        # Check lockout
        if client_ip in self.lockout_until:
            if time.time() < self.lockout_until[client_ip]:
                return True
            else:
                del self.lockout_until[client_ip]
        
        return False
    
    def record_failed_attempt(self, client_ip: str) -> None:
        """
        Record a failed authentication attempt
        """
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
        
        self.failed_attempts[client_ip].append(time.time())
        
        # Clean old attempts
        cutoff = time.time() - self.lockout_duration
        self.failed_attempts[client_ip] = [t for t in self.failed_attempts[client_ip] if t > cutoff]
        
        # Check if should lock out
        if len(self.failed_attempts[client_ip]) >= self.max_failed_attempts:
            self.lockout_until[client_ip] = time.time() + self.lockout_duration
            logger.warning(f"IP {client_ip} locked out due to {self.max_failed_attempts} failed attempts")
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key
        """
        for api_key in self.api_keys.values():
            if api_key.key_id == key_id:
                api_key.is_active = False
                logger.info(f"Revoked API key {key_id}")
                return True
        return False
    
    def list_keys(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List API keys (filtered by user if specified)
        """
        keys = []
        for api_key in self.api_keys.values():
            if user_id and api_key.user_id != user_id:
                continue
            
            keys.append({
                'key_id': api_key.key_id,
                'user_id': api_key.user_id,
                'role': api_key.role.value,
                'permissions': [p.value for p in api_key.permissions],
                'created_at': api_key.created_at,
                'last_used_at': api_key.last_used_at,
                'expires_at': api_key.expires_at,
                'is_active': api_key.is_active,
                'usage_count': api_key.usage_count,
                'description': api_key.description
            })
        
        return keys
    
    def get_auth_stats(self) -> Dict[str, Any]:
        """
        Get authentication statistics
        """
        total_keys = len(self.api_keys)
        active_keys = sum(1 for k in self.api_keys.values() if k.is_active)
        expired_keys = sum(1 for k in self.api_keys.values() if k.is_expired)
        
        recent_usage = 0
        for usage_times in self.key_usage.values():
            recent_usage += len([t for t in usage_times if time.time() - t < 3600])  # Last hour
        
        return {
            'total_keys': total_keys,
            'active_keys': active_keys,
            'expired_keys': expired_keys,
            'blocked_ips': len(self.blocked_ips),
            'locked_out_ips': len(self.lockout_until),
            'recent_usage': recent_usage,
            'failed_attempts': len(self.failed_attempts)
        }
    
    def cleanup_expired(self) -> None:
        """
        Clean up expired keys and old tracking data
        """
        current_time = time.time()
        
        # Remove expired keys
        expired_keys = [
            key_hash for key_hash, api_key in self.api_keys.items()
            if api_key.is_expired and current_time - api_key.created_at > (self.max_key_age_days * 24 * 3600)
        ]
        
        for key_hash in expired_keys:
            del self.api_keys[key_hash]
            if key_hash in self.key_usage:
                del self.key_usage[key_hash]
        
        # Clean old usage data
        cutoff = current_time - self.rate_limit_window
        for key_hash in list(self.key_usage.keys()):
            self.key_usage[key_hash] = [t for t in self.key_usage[key_hash] if t > cutoff]
        
        # Clean old failed attempts
        lockout_cutoff = current_time - self.lockout_duration
        for ip in list(self.failed_attempts.keys()):
            self.failed_attempts[ip] = [t for t in self.failed_attempts[ip] if t > lockout_cutoff]
            if not self.failed_attempts[ip]:
                del self.failed_attempts[ip]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired keys")

class FastAPIAuthenticator:
    """
    FastAPI integration for authentication
    """
    
    def __init__(self, auth_manager: AuthenticationManager):
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI not available")
        
        self.auth_manager = auth_manager
        self.security = HTTPBearer(auto_error=False)
    
    async def authenticate(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(lambda self=None: self.security if FASTAPI_AVAILABLE else None),
        request: Optional[Request] = None
    ) -> AuthRequest:
        """
        FastAPI dependency for authentication
        """
        client_ip = "unknown"
        user_agent = "unknown"
        
        if request:
            client_ip = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")
        
        # Check if IP is blocked
        if self.auth_manager.check_ip_blocked(client_ip):
            self.auth_manager.record_failed_attempt(client_ip)
            raise HTTPException(
                status_code=429,
                detail="IP temporarily blocked due to failed authentication attempts"
            )
        
        # Extract API key
        api_key = None
        if credentials and credentials.credentials:
            api_key = self.auth_manager.authenticate_key(credentials.credentials)
        
        if not api_key:
            self.auth_manager.record_failed_attempt(client_ip)
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )
        
        # Check rate limit
        if not self.auth_manager.check_rate_limit(api_key):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        return AuthRequest(
            api_key=api_key,
            user_id=api_key.user_id,
            permissions=api_key.permissions,
            request_id=f"req-{int(time.time())}-{secrets.token_hex(4)}",
            client_ip=client_ip,
            user_agent=user_agent
        )
    
    def require_permission(self, permission: Permission):
        """
        Create a dependency that requires a specific permission
        """
        async def permission_check(auth: AuthRequest = Depends(self.authenticate)) -> AuthRequest:
            if not auth.api_key or not auth.api_key.has_permission(permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission.value}' required"
                )
            return auth
        
        return permission_check
    
    def require_role(self, min_role: UserRole):
        """
        Create a dependency that requires a minimum role
        """
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.USER: 1,
            UserRole.POWER_USER: 2,
            UserRole.ADMIN: 3,
            UserRole.SYSTEM: 4
        }
        
        async def role_check(auth: AuthRequest = Depends(self.authenticate)) -> AuthRequest:
            if not auth.api_key:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            user_role_level = role_hierarchy.get(auth.api_key.role, 0)
            required_level = role_hierarchy.get(min_role, 0)
            
            if user_role_level < required_level:
                raise HTTPException(
                    status_code=403,
                    detail=f"Role '{min_role.value}' or higher required"
                )
            
            return auth
        
        return role_check

# Factory functions
def create_auth_manager(jwt_secret: Optional[str] = None) -> AuthenticationManager:
    """Create an authentication manager"""
    return AuthenticationManager(jwt_secret)

def create_fastapi_authenticator(auth_manager: AuthenticationManager) -> FastAPIAuthenticator:
    """Create FastAPI authenticator"""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")
    return FastAPIAuthenticator(auth_manager)

# Example usage
if __name__ == "__main__":
    # Create auth manager
    auth = create_auth_manager()
    
    # Generate some sample keys
    admin_key, admin_key_obj = auth.generate_api_key(
        user_id="admin",
        role=UserRole.ADMIN,
        description="Administrator key"
    )
    
    user_key, user_key_obj = auth.generate_api_key(
        user_id="user1",
        role=UserRole.USER,
        description="Regular user key",
        rate_limit=100  # 100 requests per minute
    )
    
    print(f"Admin key: {admin_key}")
    print(f"User key: {user_key}")
    
    # Test authentication
    authenticated_admin = auth.authenticate_key(admin_key)
    print(f"Admin authenticated: {authenticated_admin is not None}")
    print(f"Admin has inference permission: {authenticated_admin.has_permission(Permission.INFERENCE) if authenticated_admin else False}")
    
    # Show stats
    stats = auth.get_auth_stats()
    print(f"Auth stats: {json.dumps(stats, indent=2)}")