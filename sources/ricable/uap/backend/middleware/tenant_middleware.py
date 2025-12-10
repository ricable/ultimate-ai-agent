# File: backend/middleware/tenant_middleware.py
"""
Enhanced Tenant Middleware for Advanced Multi-tenant Isolation

Provides:
- Advanced request isolation and routing
- Resource quota enforcement
- Performance monitoring per tenant
- Security policy enforcement
- Custom routing rules
"""

import asyncio
import time
import logging
import re
from typing import Optional, Dict, Any, Callable, List
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import ipaddress

from ..tenancy.middleware import TenancyMiddleware
from ..tenancy.tenant_context import TenantContextManager, TenantContext
from ..services.tenant_manager import enhanced_tenant_manager, ResourceType
from ..services.enhanced_orchestrator import RoutingRule
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class EnhancedTenantMiddleware:
    """Enhanced tenant middleware with resource isolation and monitoring"""
    
    def __init__(self, app):
        self.app = app
        self.base_middleware = TenancyMiddleware(app)
        
        # Request tracking for rate limiting
        self.request_tracking: Dict[str, List[float]] = {}
        
        # Performance metrics
        self.tenant_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Security policies cache
        self.policy_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("Enhanced tenant middleware initialized")
    
    async def __call__(self, request: Request, call_next: Callable):
        """Process request with enhanced tenant features"""
        start_time = time.time()
        tenant_context = None
        
        try:
            # First, run base tenant middleware logic
            tenant_context = await self._resolve_tenant_context(request)
            
            if tenant_context:
                # Enhanced security checks
                security_check = await self._check_security_policies(request, tenant_context)
                if not security_check["allowed"]:
                    return self._create_security_response(security_check["reason"])
                
                # Resource quota checks
                quota_check = await self._check_resource_quotas(request, tenant_context)
                if not quota_check["allowed"]:
                    return self._create_quota_response(quota_check)
                
                # Apply custom routing rules
                await self._apply_custom_routing(request, tenant_context)
                
                # Set enhanced tenant context
                TenantContextManager.set_context(tenant_context)
                request.state.tenant_context = tenant_context
                request.state.enhanced_tenant = True
            
            # Process request
            response = await call_next(request)
            
            # Record metrics and usage
            if tenant_context:
                await self._record_request_metrics(request, response, tenant_context, start_time)
                await self._record_resource_usage(request, response, tenant_context)
            
            # Add tenant headers
            if tenant_context:
                response.headers["X-Tenant-ID"] = tenant_context.tenant_id
                response.headers["X-Enhanced-Tenant"] = "true"
            
            return response
            
        except HTTPException as e:
            if tenant_context:
                await self._record_error_metrics(tenant_context, e.status_code)
            raise
        except Exception as e:
            if tenant_context:
                await self._record_error_metrics(tenant_context, 500)
            logger.error(f"Enhanced tenant middleware error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error in enhanced tenant processing"}
            )
        finally:
            # Clean up context
            TenantContextManager.clear_context()
    
    async def _resolve_tenant_context(self, request: Request) -> Optional[TenantContext]:
        """Resolve tenant context using base middleware logic"""
        # Extract tenant information similar to base middleware
        tenant_info = await self._extract_tenant_info(request)
        
        # Get user from request
        user = await self._get_user_from_request(request)
        if not user:
            return None
        
        # Create enhanced tenant context
        # This would integrate with the existing tenant resolution
        # For now, create a basic context
        return TenantContext(
            tenant_id=tenant_info.get("tenant_id", "default"),
            organization_id="default",
            user_id=user.id if hasattr(user, 'id') else str(user),
            roles=getattr(user, 'roles', ["user"]),
            permissions=[],
            tenant_limits={}
        )
    
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
            host = host.split(":")[0]  # Remove port
            tenant_info["domain"] = host
            
            # Extract subdomain
            parts = host.split(".")
            if len(parts) > 2:
                tenant_info["subdomain"] = parts[0]
        
        # From path parameter
        path_parts = request.url.path.split("/")
        if len(path_parts) > 3 and path_parts[2] == "org":
            tenant_info["organization_slug"] = path_parts[3]
        
        return tenant_info
    
    async def _get_user_from_request(self, request: Request) -> Optional[Any]:
        """Extract user from JWT token in request"""
        try:
            from ..services.auth import auth_service
            
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header.replace("Bearer ", "")
            user = auth_service.get_current_user_from_token(token)
            return user
            
        except Exception as e:
            logger.debug(f"Failed to extract user from request: {str(e)}")
            return None
    
    async def _check_security_policies(
        self, 
        request: Request, 
        tenant_context: TenantContext
    ) -> Dict[str, Any]:
        """Check tenant security policies"""
        tenant_id = tenant_context.tenant_id
        
        # Get cached security policy or fetch fresh
        policy = await self._get_security_policy(tenant_id)
        if not policy:
            return {"allowed": True}
        
        # IP whitelist check
        if policy.get("ip_whitelist"):
            client_ip = self._get_client_ip(request)
            if client_ip and not self._is_ip_whitelisted(client_ip, policy["ip_whitelist"]):
                await self._log_security_event(
                    tenant_id, "ip_whitelist_violation", 
                    {"client_ip": client_ip, "path": request.url.path}
                )
                return {
                    "allowed": False, 
                    "reason": "IP address not in whitelist",
                    "code": "IP_NOT_WHITELISTED"
                }
        
        # Domain restrictions
        if policy.get("allowed_domains"):
            host = request.headers.get("Host", "").split(":")[0]
            if host and host not in policy["allowed_domains"]:
                return {
                    "allowed": False, 
                    "reason": "Domain not allowed",
                    "code": "DOMAIN_NOT_ALLOWED"
                }
        
        # Blocked domains
        if policy.get("blocked_domains"):
            host = request.headers.get("Host", "").split(":")[0]
            if host and host in policy["blocked_domains"]:
                return {
                    "allowed": False, 
                    "reason": "Domain blocked",
                    "code": "DOMAIN_BLOCKED"
                }
        
        # MFA requirement check (would integrate with auth system)
        if policy.get("require_mfa"):
            # This would check if user has completed MFA
            # For now, just log the requirement
            logger.debug(f"MFA required for tenant {tenant_id}")
        
        # Session limit check
        max_sessions = policy.get("max_concurrent_sessions", 10)
        current_sessions = await self._get_active_sessions(tenant_id)
        if current_sessions >= max_sessions:
            return {
                "allowed": False, 
                "reason": "Maximum concurrent sessions exceeded",
                "code": "SESSION_LIMIT_EXCEEDED"
            }
        
        return {"allowed": True}
    
    async def _check_resource_quotas(
        self, 
        request: Request, 
        tenant_context: TenantContext
    ) -> Dict[str, Any]:
        """Check resource quotas for the request"""
        tenant_id = tenant_context.tenant_id
        
        # Estimate request resource usage
        resource_usage = await self._estimate_request_resources(request)
        
        # Check each resource type
        for resource_type, amount in resource_usage.items():
            if amount > 0:
                allowed = await enhanced_tenant_manager.record_resource_usage(
                    tenant_id, resource_type, amount
                )
                
                if not allowed:
                    return {
                        "allowed": False,
                        "resource_type": resource_type.value,
                        "requested_amount": amount,
                        "reason": f"{resource_type.value} quota exceeded"
                    }
        
        return {"allowed": True}
    
    async def _apply_custom_routing(
        self, 
        request: Request, 
        tenant_context: TenantContext
    ):
        """Apply tenant-specific routing rules"""
        tenant_id = tenant_context.tenant_id
        
        # Get tenant's custom routing rules
        isolation_policy = enhanced_tenant_manager.get_tenant_isolation_policy(tenant_id)
        if not isolation_policy or not isolation_policy.custom_routing_rules:
            return
        
        # Apply custom routing rules to request
        for rule_data in isolation_policy.custom_routing_rules:
            try:
                rule = RoutingRule(**rule_data)
                
                # Check if rule applies to current request
                if self._rule_matches_request(rule, request):
                    # Modify request for custom routing
                    request.state.custom_framework = rule.framework
                    request.state.routing_rule = rule.name
                    
                    logger.debug(f"Applied custom routing rule '{rule.name}' for tenant {tenant_id}")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to apply routing rule for tenant {tenant_id}: {e}")
    
    def _rule_matches_request(self, rule: RoutingRule, request: Request) -> bool:
        """Check if routing rule matches the current request"""
        # Check pattern against request path and query
        request_content = f"{request.url.path} {request.url.query}"
        
        if re.search(rule.pattern, request_content, re.IGNORECASE):
            # Check additional conditions
            for condition, expected_value in rule.conditions.items():
                if condition == "method" and request.method != expected_value:
                    return False
                elif condition == "header":
                    header_name = expected_value.get("name")
                    header_value = expected_value.get("value")
                    if request.headers.get(header_name) != header_value:
                        return False
            
            return True
        
        return False
    
    async def _record_request_metrics(
        self, 
        request: Request, 
        response: Response, 
        tenant_context: TenantContext,
        start_time: float
    ):
        """Record request metrics for tenant"""
        tenant_id = tenant_context.tenant_id
        processing_time = time.time() - start_time
        
        # Initialize metrics if needed
        if tenant_id not in self.tenant_metrics:
            self.tenant_metrics[tenant_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_response_time": 0.0,
                "response_times": [],
                "last_activity": None
            }
        
        metrics = self.tenant_metrics[tenant_id]
        metrics["total_requests"] += 1
        metrics["total_response_time"] += processing_time
        metrics["response_times"].append(processing_time)
        metrics["last_activity"] = datetime.now(timezone.utc)
        
        # Keep only recent response times for average calculation
        if len(metrics["response_times"]) > 100:
            metrics["response_times"] = metrics["response_times"][-50:]
        
        # Track success/failure
        if 200 <= response.status_code < 400:
            metrics["successful_requests"] += 1
        else:
            metrics["failed_requests"] += 1
        
        # Log performance metrics
        uap_logger.log_event(
            LogLevel.INFO,
            f"Request processed for tenant {tenant_id}",
            EventType.AGENT,
            {
                "tenant_id": tenant_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "response_time_ms": processing_time * 1000,
                "enhanced_tenant": True
            },
            "tenant_request"
        )
    
    async def _record_resource_usage(
        self, 
        request: Request, 
        response: Response, 
        tenant_context: TenantContext
    ):
        """Record resource usage for the request"""
        tenant_id = tenant_context.tenant_id
        
        # Calculate actual resource usage
        try:
            # Request size (approximate)
            request_size = len(str(request.url)) + sum(
                len(k) + len(v) for k, v in request.headers.items()
            )
            
            # Response size (approximate)
            response_size = sum(
                len(k) + len(v) for k, v in response.headers.items()
            )
            
            # Record bandwidth usage
            total_bandwidth = (request_size + response_size) / (1024 * 1024)  # MB
            await enhanced_tenant_manager.record_resource_usage(
                tenant_id, ResourceType.BANDWIDTH, total_bandwidth
            )
            
            # Record API request
            await enhanced_tenant_manager.record_resource_usage(
                tenant_id, ResourceType.API_REQUESTS, 1
            )
            
        except Exception as e:
            logger.warning(f"Failed to record resource usage for tenant {tenant_id}: {e}")
    
    async def _record_error_metrics(self, tenant_context: TenantContext, status_code: int):
        """Record error metrics for tenant"""
        tenant_id = tenant_context.tenant_id
        
        if tenant_id in self.tenant_metrics:
            self.tenant_metrics[tenant_id]["failed_requests"] += 1
        
        # Log error
        uap_logger.log_event(
            LogLevel.ERROR,
            f"Request error for tenant {tenant_id}",
            EventType.ERROR,
            {
                "tenant_id": tenant_id,
                "status_code": status_code,
                "enhanced_tenant": True
            },
            "tenant_error"
        )
    
    async def _estimate_request_resources(self, request: Request) -> Dict[ResourceType, float]:
        """Estimate resource usage for the request"""
        resources = {}
        
        # API request (always 1)
        resources[ResourceType.API_REQUESTS] = 1
        
        # WebSocket connection
        if request.url.path.startswith("/ws/"):
            resources[ResourceType.WEBSOCKET_CONNECTIONS] = 1
        
        # Document processing
        if ("/documents/" in request.url.path or 
            request.method == "POST" and "file" in str(request.headers.get("content-type", ""))):
            resources[ResourceType.DOCUMENT_PROCESSING] = 1
        
        # Compute time (estimated based on endpoint)
        if any(keyword in request.url.path for keyword in ["/agents/", "/chat", "/analyze"]):
            resources[ResourceType.COMPUTE_TIME] = 1.0  # 1 second estimate
        
        return resources
    
    async def _get_security_policy(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get cached security policy for tenant"""
        current_time = time.time()
        
        # Check cache
        if (tenant_id in self.policy_cache and 
            current_time - self.policy_cache[tenant_id]["timestamp"] < self.cache_ttl):
            return self.policy_cache[tenant_id]["policy"]
        
        # Fetch fresh policy
        try:
            isolation_policy = enhanced_tenant_manager.get_tenant_isolation_policy(tenant_id)
            if isolation_policy:
                policy = {
                    "ip_whitelist": isolation_policy.ip_whitelist,
                    "allowed_domains": isolation_policy.allowed_domains,
                    "blocked_domains": isolation_policy.blocked_domains,
                    "require_mfa": isolation_policy.require_mfa,
                    "max_concurrent_sessions": isolation_policy.max_concurrent_sessions
                }
                
                # Cache policy
                self.policy_cache[tenant_id] = {
                    "policy": policy,
                    "timestamp": current_time
                }
                
                return policy
        except Exception as e:
            logger.warning(f"Failed to get security policy for tenant {tenant_id}: {e}")
        
        return None
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Get client IP address from request"""
        # Check X-Forwarded-For header first
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        
        # Check X-Real-IP header
        x_real_ip = request.headers.get("X-Real-IP")
        if x_real_ip:
            return x_real_ip
        
        # Use client IP from connection
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return None
    
    def _is_ip_whitelisted(self, client_ip: str, whitelist: List[str]) -> bool:
        """Check if IP is in whitelist (supports CIDR notation)"""
        try:
            client_addr = ipaddress.ip_address(client_ip)
            
            for allowed in whitelist:
                try:
                    # Try as network (CIDR)
                    if "/" in allowed:
                        network = ipaddress.ip_network(allowed, strict=False)
                        if client_addr in network:
                            return True
                    # Try as single IP
                    else:
                        allowed_addr = ipaddress.ip_address(allowed)
                        if client_addr == allowed_addr:
                            return True
                except ValueError:
                    # Invalid IP/network format
                    continue
        except ValueError:
            # Invalid client IP
            return False
        
        return False
    
    async def _get_active_sessions(self, tenant_id: str) -> int:
        """Get count of active sessions for tenant"""
        # This would integrate with session management system
        # For now, return a placeholder value
        return 0
    
    async def _log_security_event(
        self, 
        tenant_id: str, 
        event_type: str, 
        details: Dict[str, Any]
    ):
        """Log security event for tenant"""
        uap_logger.log_event(
            LogLevel.WARNING,
            f"Security event for tenant {tenant_id}: {event_type}",
            EventType.SECURITY,
            {
                "tenant_id": tenant_id,
                "event_type": event_type,
                **details
            },
            "tenant_security"
        )
    
    def _create_security_response(self, reason: str) -> JSONResponse:
        """Create security violation response"""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "detail": "Access denied",
                "reason": reason,
                "type": "security_violation"
            }
        )
    
    def _create_quota_response(self, quota_info: Dict[str, Any]) -> JSONResponse:
        """Create quota exceeded response"""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "detail": "Resource quota exceeded",
                "resource_type": quota_info.get("resource_type"),
                "reason": quota_info.get("reason"),
                "type": "quota_exceeded"
            }
        )
    
    def get_tenant_metrics(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for specific tenant"""
        return self.tenant_metrics.get(tenant_id)
    
    def get_all_tenant_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all tenants"""
        return self.tenant_metrics.copy()
    
    def clear_tenant_metrics(self, tenant_id: str) -> bool:
        """Clear metrics for specific tenant"""
        if tenant_id in self.tenant_metrics:
            del self.tenant_metrics[tenant_id]
            return True
        return False
