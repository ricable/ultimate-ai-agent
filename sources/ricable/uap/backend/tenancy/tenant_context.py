# File: backend/tenancy/tenant_context.py
"""
Tenant context management for multi-tenant request processing
"""

from typing import Optional, List, Dict, Any
from fastapi import HTTPException, Depends, status
from contextlib import contextmanager
import threading
from dataclasses import dataclass

from .models import Tenant, TenantUser
from ..services.auth import UserInDB

# Thread-local storage for tenant context
_thread_local = threading.local()

@dataclass
class TenantContext:
    """Tenant context for the current request"""
    tenant_id: str
    organization_id: str
    user_id: str
    username: str
    tenant_roles: List[str]
    tenant_permissions: List[str]
    is_tenant_admin: bool
    tenant_limits: Dict[str, Any]
    isolation_level: str

class TenantContextManager:
    """Manager for tenant context operations"""
    
    @staticmethod
    def set_context(context: TenantContext):
        """Set tenant context for current thread"""
        _thread_local.tenant_context = context
    
    @staticmethod
    def get_context() -> Optional[TenantContext]:
        """Get tenant context for current thread"""
        return getattr(_thread_local, 'tenant_context', None)
    
    @staticmethod
    def clear_context():
        """Clear tenant context for current thread"""
        if hasattr(_thread_local, 'tenant_context'):
            delattr(_thread_local, 'tenant_context')
    
    @staticmethod
    @contextmanager
    def with_context(context: TenantContext):
        """Context manager for tenant context"""
        previous_context = TenantContextManager.get_context()
        try:
            TenantContextManager.set_context(context)
            yield
        finally:
            if previous_context:
                TenantContextManager.set_context(previous_context)
            else:
                TenantContextManager.clear_context()

def get_current_tenant() -> Optional[TenantContext]:
    """Get current tenant context (convenience function)"""
    return TenantContextManager.get_context()

def require_tenant_context() -> TenantContext:
    """Require tenant context to be present"""
    context = get_current_tenant()
    if not context:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tenant context not set"
        )
    return context

def require_tenant_admin() -> TenantContext:
    """Require current user to be tenant admin"""
    context = require_tenant_context()
    if not context.is_tenant_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant admin access required"
        )
    return context

def require_tenant_permission(permission: str) -> TenantContext:
    """Require specific tenant permission"""
    context = require_tenant_context()
    if permission not in context.tenant_permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tenant permission required: {permission}"
        )
    return context

class TenantResolver:
    """Resolves tenant information from various sources"""
    
    def __init__(self, organization_manager):
        self.organization_manager = organization_manager
    
    async def resolve_from_subdomain(self, subdomain: str) -> Optional[Tenant]:
        """Resolve tenant from subdomain"""
        for tenant in self.organization_manager.tenants.values():
            if tenant.subdomain == subdomain:
                return tenant
        return None
    
    async def resolve_from_domain(self, domain: str) -> Optional[Tenant]:
        """Resolve tenant from custom domain"""
        # First check organizations with custom domains
        for org in self.organization_manager.organizations.values():
            if org.custom_domain == domain:
                # Return default tenant for organization
                for tenant in self.organization_manager.tenants.values():
                    if tenant.organization_id == org.id and tenant.slug == "main":
                        return tenant
        
        # Check tenant custom domains
        for tenant in self.organization_manager.tenants.values():
            org = self.organization_manager.organizations.get(tenant.organization_id)
            if org and org.custom_domain == domain:
                return tenant
        
        return None
    
    async def resolve_from_header(self, tenant_header: str) -> Optional[Tenant]:
        """Resolve tenant from X-Tenant-ID header"""
        return await self.organization_manager.get_tenant(tenant_header)
    
    async def resolve_from_user_default(self, user: UserInDB) -> Optional[Tenant]:
        """Get user's default tenant (first tenant they belong to)"""
        user_tenants = await self.organization_manager.get_user_tenants(user.id)
        return user_tenants[0] if user_tenants else None
    
    async def get_tenant_user_info(self, tenant_id: str, user_id: str) -> Optional[TenantUser]:
        """Get tenant-specific user information"""
        tenant_users = self.organization_manager.tenant_users.get(tenant_id, [])
        for tenant_user in tenant_users:
            if tenant_user.user_id == user_id:
                return tenant_user
        return None

class MultiTenantAuth:
    """Multi-tenant authentication and authorization"""
    
    def __init__(self, organization_manager):
        self.organization_manager = organization_manager
        self.tenant_resolver = TenantResolver(organization_manager)
    
    async def create_tenant_context(
        self, 
        user: UserInDB, 
        tenant_id: Optional[str] = None,
        subdomain: Optional[str] = None,
        domain: Optional[str] = None
    ) -> TenantContext:
        """Create tenant context from user and tenant information"""
        
        # Resolve tenant
        tenant = None
        
        if tenant_id:
            tenant = await self.organization_manager.get_tenant(tenant_id)
        elif subdomain:
            tenant = await self.tenant_resolver.resolve_from_subdomain(subdomain)
        elif domain:
            tenant = await self.tenant_resolver.resolve_from_domain(domain)
        else:
            # Use user's default tenant
            tenant = await self.tenant_resolver.resolve_from_user_default(user)
        
        if not tenant:
            # Use default tenant if no specific tenant found
            tenant = self.organization_manager.tenants.get("default-tenant")
            if not tenant:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No accessible tenant found for user"
                )
        
        # Get tenant-specific user information
        tenant_user = await self.tenant_resolver.get_tenant_user_info(tenant.id, user.id)
        
        if not tenant_user:
            # Create default tenant user if user is in default tenant
            if tenant.id == "default-tenant":
                tenant_user = TenantUser(
                    tenant_id=tenant.id,
                    organization_id=tenant.organization_id,
                    user_id=user.id,
                    tenant_roles=user.roles,
                    is_tenant_admin="admin" in user.roles
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User is not authorized for this tenant"
                )
        
        # Check if user is active in tenant
        if not tenant_user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User access is disabled for this tenant"
            )
        
        # Combine global and tenant-specific permissions
        all_permissions = list(set(tenant_user.tenant_permissions))
        
        return TenantContext(
            tenant_id=tenant.id,
            organization_id=tenant.organization_id,
            user_id=user.id,
            username=user.username,
            tenant_roles=tenant_user.tenant_roles,
            tenant_permissions=all_permissions,
            is_tenant_admin=tenant_user.is_tenant_admin,
            tenant_limits=tenant.limits,
            isolation_level=tenant.isolation_level
        )

# Dependency functions for FastAPI

def get_tenant_id_from_header(x_tenant_id: Optional[str] = None) -> Optional[str]:
    """Extract tenant ID from header"""
    return x_tenant_id

def get_subdomain_from_host(host: Optional[str] = None) -> Optional[str]:
    """Extract subdomain from host header"""
    if not host:
        return None
    
    parts = host.split('.')
    if len(parts) > 2:  # subdomain.domain.com
        return parts[0]
    return None

async def create_tenant_dependency(organization_manager):
    """Create tenant dependency factory"""
    
    multi_tenant_auth = MultiTenantAuth(organization_manager)
    
    async def get_tenant_context(
        user: UserInDB,
        x_tenant_id: Optional[str] = None,
        host: Optional[str] = None
    ) -> TenantContext:
        """FastAPI dependency to get tenant context"""
        
        subdomain = get_subdomain_from_host(host)
        
        context = await multi_tenant_auth.create_tenant_context(
            user=user,
            tenant_id=x_tenant_id,
            subdomain=subdomain,
            domain=host
        )
        
        # Set context in thread-local storage
        TenantContextManager.set_context(context)
        
        return context
    
    return get_tenant_context

# Utility decorators

def with_tenant_context(func):
    """Decorator to ensure tenant context is available"""
    async def wrapper(*args, **kwargs):
        context = get_current_tenant()
        if not context:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tenant context not available"
            )
        return await func(*args, **kwargs)
    return wrapper

def tenant_isolated(func):
    """Decorator to ensure function operates within tenant boundaries"""
    async def wrapper(*args, **kwargs):
        context = require_tenant_context()
        # Add tenant context to kwargs if not present
        if 'tenant_context' not in kwargs:
            kwargs['tenant_context'] = context
        return await func(*args, **kwargs)
    return wrapper