# File: backend/tenancy/middleware.py
"""
Multi-tenancy middleware for automatic tenant resolution
"""

from typing import Optional, Dict, Any, Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import logging
import time
import re

from .tenant_context import TenantContextManager, TenantContext, MultiTenantAuth
from .organization_manager import organization_manager
from ..services.auth import auth_service

logger = logging.getLogger(__name__)

class TenancyMiddleware:
    """Middleware for automatic tenant detection and context management"""
    
    def __init__(self, app, enable_tenant_routing: bool = True):
        self.app = app
        self.enable_tenant_routing = enable_tenant_routing
        self.multi_tenant_auth = MultiTenantAuth(organization_manager)
        
        # Patterns for tenant-aware endpoints
        self.tenant_aware_patterns = [
            r'^/api/tenants/',
            r'^/api/organizations/',
            r'^/api/billing/',
            r'^/api/admin/',
            r'^/api/documents/',
            r'^/api/agents/',
            r'^/api/workflows/'
        ]
        
        # Patterns that bypass tenant resolution
        self.bypass_patterns = [
            r'^/api/auth/',
            r'^/api/health',
            r'^/metrics',
            r'^/docs',
            r'^/openapi.json',
            r'^/favicon.ico',
            r'^/static/'
        ]
    
    async def __call__(self, request: Request, call_next: Callable):
        """Process request with tenant awareness"""
        start_time = time.time()
        
        try:
            # Check if endpoint should bypass tenant resolution
            if self._should_bypass_tenant_resolution(request.url.path):
                response = await call_next(request)
                return self._add_timing_header(response, start_time)
            
            # Extract tenant information from request
            tenant_info = await self._extract_tenant_info(request)
            
            # Resolve tenant context if user is authenticated
            tenant_context = None
            if self._is_tenant_aware_endpoint(request.url.path):
                tenant_context = await self._resolve_tenant_context(request, tenant_info)
                
                if tenant_context:
                    # Set tenant context for request
                    TenantContextManager.set_context(tenant_context)
                    
                    # Add tenant info to request state
                    request.state.tenant_context = tenant_context
            
            # Process request
            response = await call_next(request)
            
            # Add tenant information to response headers
            if tenant_context:
                response.headers["X-Tenant-ID"] = tenant_context.tenant_id
                response.headers["X-Organization-ID"] = tenant_context.organization_id
            
            return self._add_timing_header(response, start_time)
            
        except HTTPException as e:
            logger.warning(f"Tenant resolution failed: {e.detail}")
            raise
        except Exception as e:
            logger.error(f"Tenancy middleware error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error in tenant resolution"}
            )
        finally:
            # Clean up tenant context
            TenantContextManager.clear_context()
    
    def _should_bypass_tenant_resolution(self, path: str) -> bool:
        """Check if path should bypass tenant resolution"""
        return any(re.match(pattern, path) for pattern in self.bypass_patterns)
    
    def _is_tenant_aware_endpoint(self, path: str) -> bool:
        """Check if endpoint is tenant-aware"""
        return any(re.match(pattern, path) for pattern in self.tenant_aware_patterns)
    
    async def _extract_tenant_info(self, request: Request) -> Dict[str, Optional[str]]:
        """Extract tenant information from request"""
        tenant_info = {
            "tenant_id": None,
            "subdomain": None,
            "domain": None,
            "organization_slug": None
        }
        
        # From X-Tenant-ID header
        tenant_info["tenant_id"] = request.headers.get("X-Tenant-ID")
        
        # From Host header (subdomain or custom domain)
        host = request.headers.get("Host", "")
        if host:
            # Remove port number
            host = host.split(":")[0]
            tenant_info["domain"] = host
            
            # Extract subdomain (e.g., tenant1.uap.com -> tenant1)
            parts = host.split(".")
            if len(parts) > 2:  # subdomain.domain.com
                tenant_info["subdomain"] = parts[0]
        
        # From path parameter (e.g., /api/org/{org_slug}/...)
        path_parts = request.url.path.split("/")
        if len(path_parts) > 3 and path_parts[2] == "org":
            tenant_info["organization_slug"] = path_parts[3]
        
        return tenant_info
    
    async def _resolve_tenant_context(
        self, 
        request: Request, 
        tenant_info: Dict[str, Optional[str]]
    ) -> Optional[TenantContext]:
        """Resolve tenant context from request and user"""
        
        # Get user from JWT token if present
        user = await self._get_user_from_request(request)
        if not user:
            return None
        
        try:
            # Create tenant context
            context = await self.multi_tenant_auth.create_tenant_context(
                user=user,
                tenant_id=tenant_info["tenant_id"],
                subdomain=tenant_info["subdomain"],
                domain=tenant_info["domain"]
            )
            
            logger.debug(f"Resolved tenant context: {context.tenant_id} for user {user.username}")
            return context
            
        except HTTPException as e:
            # If tenant resolution fails, try default tenant
            if e.status_code == status.HTTP_404_NOT_FOUND:
                try:
                    context = await self.multi_tenant_auth.create_tenant_context(user=user)
                    logger.debug(f"Using default tenant for user {user.username}")
                    return context
                except:
                    pass
            
            logger.warning(f"Failed to resolve tenant context: {e.detail}")
            return None
    
    async def _get_user_from_request(self, request: Request) -> Optional[Any]:
        """Extract user from JWT token in request"""
        try:
            # Get Authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            # Extract token
            token = auth_header.replace("Bearer ", "")
            
            # Validate token and get user
            user = auth_service.get_current_user_from_token(token)
            return user
            
        except Exception as e:
            logger.debug(f"Failed to extract user from request: {str(e)}")
            return None
    
    def _add_timing_header(self, response: Response, start_time: float) -> Response:
        """Add request timing header"""
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

class WhiteLabelMiddleware:
    """Middleware for white-label customization"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable):
        """Apply white-label customizations"""
        
        # Get tenant context
        tenant_context = getattr(request.state, 'tenant_context', None)
        
        if tenant_context:
            # Get organization for white-label settings
            org = await organization_manager.get_organization(tenant_context.organization_id)
            
            if org and org.white_label_enabled:
                # Add white-label headers
                response = await call_next(request)
                
                if org.branding:
                    response.headers["X-Brand-Name"] = org.branding.get("name", org.name)
                    if "primary_color" in org.branding:
                        response.headers["X-Brand-Color"] = org.branding["primary_color"]
                
                return response
        
        return await call_next(request)

class TenantRateLimitMiddleware:
    """Middleware for tenant-specific rate limiting"""
    
    def __init__(self, app):
        self.app = app
        self.request_counts: Dict[str, Dict[str, int]] = {}  # tenant_id -> {endpoint: count}
        self.last_reset: Dict[str, float] = {}  # tenant_id -> timestamp
        self.reset_interval = 3600  # 1 hour in seconds
    
    async def __call__(self, request: Request, call_next: Callable):
        """Apply tenant-specific rate limiting"""
        
        # Get tenant context
        tenant_context = getattr(request.state, 'tenant_context', None)
        
        if tenant_context:
            # Check rate limits
            if await self._is_rate_limited(tenant_context, request):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "detail": "Rate limit exceeded for tenant",
                        "tenant_id": tenant_context.tenant_id
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Update request count
            await self._update_request_count(tenant_context, request)
            
            return response
        
        return await call_next(request)
    
    async def _is_rate_limited(self, tenant_context: TenantContext, request: Request) -> bool:
        """Check if tenant has exceeded rate limits"""
        tenant_id = tenant_context.tenant_id
        endpoint = request.url.path
        
        # Get tenant limits
        rate_limit = tenant_context.tenant_limits.get("api_rate_limit", 1000)  # requests per hour
        
        # Reset counters if needed
        current_time = time.time()
        if tenant_id not in self.last_reset:
            self.last_reset[tenant_id] = current_time
            self.request_counts[tenant_id] = {}
        elif current_time - self.last_reset[tenant_id] > self.reset_interval:
            self.last_reset[tenant_id] = current_time
            self.request_counts[tenant_id] = {}
        
        # Check current count
        current_count = sum(self.request_counts[tenant_id].values())
        return current_count >= rate_limit
    
    async def _update_request_count(self, tenant_context: TenantContext, request: Request):
        """Update request count for tenant"""
        tenant_id = tenant_context.tenant_id
        endpoint = request.url.path
        
        if tenant_id not in self.request_counts:
            self.request_counts[tenant_id] = {}
        
        if endpoint not in self.request_counts[tenant_id]:
            self.request_counts[tenant_id][endpoint] = 0
        
        self.request_counts[tenant_id][endpoint] += 1

class TenantSecurityMiddleware:
    """Middleware for tenant-specific security policies"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable):
        """Apply tenant security policies"""
        
        # Get tenant context
        tenant_context = getattr(request.state, 'tenant_context', None)
        
        if tenant_context:
            # Get tenant settings
            settings = organization_manager.tenant_settings.get(tenant_context.tenant_id)
            
            if settings:
                # IP whitelist check
                if settings.ip_whitelist and request.client:
                    client_ip = request.client.host
                    if client_ip not in settings.ip_whitelist:
                        return JSONResponse(
                            status_code=status.HTTP_403_FORBIDDEN,
                            content={"detail": "IP address not in whitelist"}
                        )
                
                # Session timeout check (would need session management)
                # MFA requirements (would need MFA validation)
        
        return await call_next(request)