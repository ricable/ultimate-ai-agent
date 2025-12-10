# UAP Enterprise Multi-Tenancy Module
"""
Enterprise multi-tenancy system with organization isolation.
Provides secure data segregation and tenant management capabilities.
"""

from .models import *
from .organization_manager import OrganizationManager
from .tenant_context import TenantContext, get_current_tenant
from .isolation import DataIsolationManager
from .middleware import TenancyMiddleware

__version__ = "1.0.0"
__all__ = [
    "Organization",
    "Tenant", 
    "TenantUser",
    "OrganizationManager",
    "TenantContext",
    "get_current_tenant",
    "DataIsolationManager",
    "TenancyMiddleware"
]