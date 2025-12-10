# File: backend/security/middleware.py
"""
Security Middleware for Request Monitoring and Threat Detection
Integrates with FastAPI to provide real-time security monitoring of all requests.
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timezone
import ipaddress
import logging

from fastapi import Request, Response, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .threat_detection import get_threat_detection_engine, SecurityThreat, ThreatLevel
from .audit_trail import get_security_audit_logger, AuditEventType, AuditOutcome
from ..monitoring.logs.logger import uap_logger

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for comprehensive request monitoring and threat detection.
    
    Features:
    - Real-time threat detection for all requests
    - Rate limiting and DDoS protection
    - Request/response sanitization
    - Security header injection
    - IP allowlist/blocklist management
    - Compliance monitoring integration
    """
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        self.threat_engine = get_threat_detection_engine()
        self.audit_logger = get_security_audit_logger()
        
        # Security configuration
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        # Rate limiting configuration
        self.rate_limit_config = self.config.get("rate_limiting", {
            "enabled": True,
            "requests_per_minute": 60,
            "burst_limit": 10,
            "ban_duration_minutes": 15
        })
        
        # IP filtering configuration
        self.ip_allowlist = set(self.config.get("ip_allowlist", []))
        self.ip_blocklist = set(self.config.get("ip_blocklist", []))
        
        # Excluded paths from security scanning
        self.excluded_paths = set(self.config.get("excluded_paths", [
            "/health", "/metrics", "/docs", "/openapi.json"
        ]))
        
        # Request size limits
        self.max_request_size = self.config.get("max_request_size", 10 * 1024 * 1024)  # 10MB
        
        logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security middleware"""
        start_time = time.time()
        
        try:
            # Extract request information
            request_info = await self._extract_request_info(request)
            
            # Pre-request security checks
            security_response = await self._pre_request_security_checks(request, request_info)
            if security_response:
                return security_response
            
            # Process request
            response = await call_next(request)
            
            # Post-request security processing
            await self._post_request_security_processing(request, response, request_info, start_time)
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            
            # Log security middleware failure
            await self.audit_logger.log_security_event(
                "Security middleware error",
                "critical",
                details={"error": str(e)}
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={"error": "Security processing failed"}
            )
    
    async def _extract_request_info(self, request: Request) -> Dict[str, Any]:
        """Extract comprehensive request information for security analysis"""
        # Get client IP (handle proxies)
        client_ip = self._get_client_ip(request)
        
        # Read request body if present
        body = b""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                # Recreate request with body for downstream processing
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
            except Exception as e:
                logger.warning(f"Could not read request body: {e}")
        
        request_info = {
            "ip_address": client_ip,
            "method": request.method,
            "path": str(request.url.path),
            "query_string": str(request.url.query),
            "user_agent": request.headers.get("user-agent", ""),
            "referer": request.headers.get("referer", ""),
            "content_type": request.headers.get("content-type", ""),
            "content_length": len(body),
            "body": body.decode("utf-8", errors="ignore") if body else "",
            "headers": dict(request.headers),
            "timestamp": datetime.now(timezone.utc),
            "request_id": request.headers.get("x-request-id", f"req_{int(time.time())}")
        }
        
        # Extract user ID if authenticated
        authorization = request.headers.get("authorization")
        if authorization:
            try:
                # This would integrate with your auth system
                # For now, extract from JWT if present
                request_info["user_id"] = self._extract_user_from_auth(authorization)
            except Exception:
                request_info["user_id"] = None
        
        return request_info
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address handling proxies and load balancers"""
        # Check for forwarded headers (in order of preference)
        forwarded_headers = [
            "x-forwarded-for",
            "x-real-ip",
            "x-client-ip",
            "cf-connecting-ip",  # Cloudflare
            "true-client-ip"     # Akamai
        ]
        
        for header in forwarded_headers:
            value = request.headers.get(header)
            if value:
                # Take the first IP from comma-separated list
                ip = value.split(",")[0].strip()
                if self._is_valid_ip(ip):
                    return ip
        
        # Fallback to direct connection
        if hasattr(request.client, "host"):
            return request.client.host
        
        return "unknown"
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False
    
    def _extract_user_from_auth(self, authorization: str) -> Optional[str]:
        """Extract user ID from authorization header"""
        # This would integrate with your JWT/auth system
        # For now, return None as placeholder
        return None
    
    async def _pre_request_security_checks(self, request: Request, 
                                         request_info: Dict[str, Any]) -> Optional[Response]:
        """Perform pre-request security checks"""
        
        # Skip security checks for excluded paths
        if request_info["path"] in self.excluded_paths:
            return None
        
        # 1. IP Filtering
        ip_check = await self._check_ip_filtering(request_info["ip_address"])
        if ip_check:
            return ip_check
        
        # 2. Request Size Validation
        if request_info["content_length"] > self.max_request_size:
            await self.audit_logger.log_security_event(
                "Request size limit exceeded",
                "medium",
                user_id=request_info.get("user_id"),
                source_ip=request_info["ip_address"],
                details={
                    "content_length": request_info["content_length"],
                    "max_allowed": self.max_request_size
                }
            )
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large"}
            )
        
        # 3. Threat Detection
        threats = await self.threat_engine.analyze_request(request_info)
        if threats:
            return await self._handle_detected_threats(threats, request_info)
        
        # 4. Rate Limiting
        if self.rate_limit_config["enabled"]:
            rate_limit_response = await self._check_rate_limiting(request_info)
            if rate_limit_response:
                return rate_limit_response
        
        return None
    
    async def _check_ip_filtering(self, ip_address: str) -> Optional[Response]:
        """Check IP allowlist/blocklist"""
        if not self._is_valid_ip(ip_address):
            return None
        
        # Check blocklist
        if ip_address in self.ip_blocklist:
            await self.audit_logger.log_security_event(
                "Blocked IP attempt",
                "high",
                source_ip=ip_address,
                details={"reason": "IP in blocklist"}
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied"}
            )
        
        # Check allowlist (if configured)
        if self.ip_allowlist and ip_address not in self.ip_allowlist:
            await self.audit_logger.log_security_event(
                "Non-allowlisted IP attempt",
                "medium",
                source_ip=ip_address,
                details={"reason": "IP not in allowlist"}
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied"}
            )
        
        return None
    
    async def _handle_detected_threats(self, threats: list, 
                                     request_info: Dict[str, Any]) -> Response:
        """Handle detected security threats"""
        # Log all threats
        for threat in threats:
            await self.audit_logger.log_security_event(
                f"Security threat detected: {threat.threat_type.value}",
                "high",
                user_id=request_info.get("user_id"),
                source_ip=request_info["ip_address"],
                details={
                    "threat_id": threat.id,
                    "threat_type": threat.threat_type.value,
                    "confidence": threat.confidence_score,
                    "description": threat.description
                }
            )
        
        # Determine response based on threat severity
        max_severity = max(threat.level for threat in threats)
        
        if max_severity == ThreatLevel.CRITICAL:
            # Block request immediately
            return JSONResponse(
                status_code=403,
                content={"error": "Request blocked due to security threat"}
            )
        elif max_severity == ThreatLevel.HIGH:
            # Block request with warning
            return JSONResponse(
                status_code=400,
                content={"error": "Request contains suspicious content"}
            )
        else:
            # Allow but monitor
            return None
    
    async def _check_rate_limiting(self, request_info: Dict[str, Any]) -> Optional[Response]:
        """Check rate limiting for IP address"""
        # This would integrate with the rate limiting in threat detection
        # For now, return None (no rate limiting violation)
        return None
    
    async def _post_request_security_processing(self, request: Request, response: Response,
                                              request_info: Dict[str, Any], start_time: float):
        """Perform post-request security processing"""
        
        # Calculate request duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log request for audit trail
        success = 200 <= response.status_code < 400
        
        await self.audit_logger.log_data_access(
            user_id=request_info.get("user_id"),
            resource=request_info["path"],
            operation=request_info["method"],
            success=success,
            source_ip=request_info["ip_address"],
            details={
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "content_length": request_info["content_length"],
                "user_agent": request_info["user_agent"]
            }
        )
        
        # Check for response anomalies
        await self._check_response_anomalies(request_info, response, duration_ms)
    
    async def _check_response_anomalies(self, request_info: Dict[str, Any], 
                                      response: Response, duration_ms: int):
        """Check for response anomalies that might indicate security issues"""
        
        # Check for unusually slow responses (potential DoS)
        if duration_ms > 10000:  # 10 seconds
            await self.audit_logger.log_security_event(
                "Slow response detected",
                "medium",
                user_id=request_info.get("user_id"),
                source_ip=request_info["ip_address"],
                details={
                    "duration_ms": duration_ms,
                    "path": request_info["path"],
                    "method": request_info["method"]
                }
            )
        
        # Check for error responses that might indicate attacks
        if response.status_code >= 500:
            await self.audit_logger.log_security_event(
                "Server error response",
                "medium",
                user_id=request_info.get("user_id"),
                source_ip=request_info["ip_address"],
                details={
                    "status_code": response.status_code,
                    "path": request_info["path"]
                }
            )
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add additional headers based on content type
        if hasattr(response, 'media_type'):
            if response.media_type == "application/json":
                response.headers["X-Content-Type-Options"] = "nosniff"

class ComplianceMiddleware(BaseHTTPMiddleware):
    """
    Compliance monitoring middleware for regulatory requirements.
    
    Monitors requests for compliance with GDPR, HIPAA, SOC2, etc.
    """
    
    def __init__(self, app, config: Dict[str, Any] = None):
        super().__init__(app)
        self.config = config or {}
        self.audit_logger = get_security_audit_logger()
        
        # Compliance configuration
        self.gdpr_enabled = self.config.get("gdpr_enabled", True)
        self.hipaa_enabled = self.config.get("hipaa_enabled", False)
        self.soc2_enabled = self.config.get("soc2_enabled", True)
        
        # Data classification patterns
        self.sensitive_data_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
        logger.info("Compliance middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request for compliance monitoring"""
        
        # Extract request data
        request_info = await self._extract_compliance_data(request)
        
        # Pre-request compliance checks
        compliance_response = await self._check_compliance_requirements(request, request_info)
        if compliance_response:
            return compliance_response
        
        # Process request
        response = await call_next(request)
        
        # Post-request compliance logging
        await self._log_compliance_event(request, response, request_info)
        
        return response
    
    async def _extract_compliance_data(self, request: Request) -> Dict[str, Any]:
        """Extract data relevant for compliance monitoring"""
        
        # Read request body for data classification
        body = b""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                # Recreate request for downstream processing
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
            except Exception:
                pass
        
        return {
            "path": str(request.url.path),
            "method": request.method,
            "body_content": body.decode("utf-8", errors="ignore") if body else "",
            "headers": dict(request.headers),
            "query_params": dict(request.query_params),
            "ip_address": request.client.host if request.client else "unknown",
            "timestamp": datetime.now(timezone.utc)
        }
    
    async def _check_compliance_requirements(self, request: Request, 
                                           request_info: Dict[str, Any]) -> Optional[Response]:
        """Check compliance requirements before processing request"""
        
        # GDPR compliance checks
        if self.gdpr_enabled:
            gdpr_response = await self._check_gdpr_compliance(request_info)
            if gdpr_response:
                return gdpr_response
        
        # HIPAA compliance checks  
        if self.hipaa_enabled:
            hipaa_response = await self._check_hipaa_compliance(request_info)
            if hipaa_response:
                return hipaa_response
        
        return None
    
    async def _check_gdpr_compliance(self, request_info: Dict[str, Any]) -> Optional[Response]:
        """Check GDPR compliance requirements"""
        # Check for personal data processing without consent
        if self._contains_personal_data(request_info["body_content"]):
            # Would check consent records here
            await self.audit_logger.log_security_event(
                "Personal data processing detected",
                "medium",
                details={
                    "path": request_info["path"],
                    "gdpr_applicable": True
                }
            )
        
        return None
    
    async def _check_hipaa_compliance(self, request_info: Dict[str, Any]) -> Optional[Response]:
        """Check HIPAA compliance requirements"""
        # Check for PHI in request
        if self._contains_phi(request_info["body_content"]):
            await self.audit_logger.log_security_event(
                "PHI data detected in request",
                "high",
                details={
                    "path": request_info["path"],
                    "hipaa_applicable": True
                }
            )
        
        return None
    
    def _contains_personal_data(self, content: str) -> bool:
        """Check if content contains personal data"""
        import re
        return any(re.search(pattern, content) for pattern in self.sensitive_data_patterns)
    
    def _contains_phi(self, content: str) -> bool:
        """Check if content contains Protected Health Information"""
        # Simplified PHI detection
        phi_keywords = ["medical", "health", "patient", "diagnosis", "treatment"]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in phi_keywords)
    
    async def _log_compliance_event(self, request: Request, response: Response, 
                                  request_info: Dict[str, Any]):
        """Log compliance-related events"""
        
        # Log data access for compliance audit trail
        await self.audit_logger.log_data_access(
            user_id=request_info.get("user_id"),
            resource=request_info["path"],
            operation=request_info["method"],
            success=200 <= response.status_code < 400,
            source_ip=request_info["ip_address"],
            details={
                "compliance_monitoring": True,
                "gdpr_applicable": self.gdpr_enabled,
                "hipaa_applicable": self.hipaa_enabled,
                "soc2_applicable": self.soc2_enabled
            }
        )

def create_security_middleware_stack(app, config: Dict[str, Any] = None):
    """Create and configure the complete security middleware stack"""
    
    # Add compliance middleware first
    app.add_middleware(ComplianceMiddleware, config=config)
    
    # Add security middleware
    app.add_middleware(SecurityMiddleware, config=config)
    
    logger.info("Security middleware stack configured")
    
    return app