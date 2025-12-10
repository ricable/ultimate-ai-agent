# File: backend/services/tenant_manager.py
"""
Enhanced Tenant Manager with Advanced Resource Isolation and Quotas

Extends the existing organization manager with:
- Advanced resource quotas and monitoring
- Real-time usage tracking
- Resource isolation policies
- Performance optimization per tenant
- Custom routing rules per tenant
- Billing integration
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from enum import Enum
import uuid
import json

# Import existing tenant models and manager
from ..tenancy.organization_manager import organization_manager, OrganizationManager
from ..tenancy.models import (
    Tenant, TenantUser, TenantSettings, Organization,
    TenantCreate, TenantUpdate, TenantStatus
)
from ..services.enhanced_orchestrator import RoutingRule
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class ResourceType(str, Enum):
    """Types of resources for quota management"""
    API_REQUESTS = "api_requests"
    WEBSOCKET_CONNECTIONS = "websocket_connections"
    DOCUMENT_PROCESSING = "document_processing"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    COMPUTE_TIME = "compute_time"
    AI_INFERENCE = "ai_inference"
    CUSTOM = "custom"

class QuotaPeriod(str, Enum):
    """Quota reset periods"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"

@dataclass
class ResourceQuota:
    """Resource quota definition"""
    resource_type: ResourceType
    limit: float
    period: QuotaPeriod
    soft_limit: Optional[float] = None  # Warning threshold
    burst_limit: Optional[float] = None  # Temporary burst allowance
    rollover_allowed: bool = False  # Allow unused quota to rollover
    priority: int = 0  # Higher priority quotas checked first
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_type": self.resource_type.value,
            "limit": self.limit,
            "period": self.period.value,
            "soft_limit": self.soft_limit,
            "burst_limit": self.burst_limit,
            "rollover_allowed": self.rollover_allowed,
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResourceQuota':
        return cls(
            resource_type=ResourceType(data["resource_type"]),
            limit=data["limit"],
            period=QuotaPeriod(data["period"]),
            soft_limit=data.get("soft_limit"),
            burst_limit=data.get("burst_limit"),
            rollover_allowed=data.get("rollover_allowed", False),
            priority=data.get("priority", 0),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata", {})
        )

@dataclass
class ResourceUsage:
    """Real-time resource usage tracking"""
    tenant_id: str
    resource_type: ResourceType
    current_usage: float = 0.0
    peak_usage: float = 0.0
    usage_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    violation_count: int = 0
    last_violation: Optional[datetime] = None
    
    def record_usage(self, amount: float, timestamp: Optional[datetime] = None):
        """Record resource usage"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self.current_usage += amount
        self.peak_usage = max(self.peak_usage, self.current_usage)
        
        self.usage_history.append({
            "timestamp": timestamp,
            "amount": amount,
            "cumulative": self.current_usage
        })
    
    def reset_usage(self, period: QuotaPeriod):
        """Reset usage for new period"""
        self.current_usage = 0.0
        self.last_reset = datetime.now(timezone.utc)
        
        # Keep history for analysis
        if len(self.usage_history) > 100:
            # Remove old entries but keep some for trend analysis
            for _ in range(len(self.usage_history) - 100):
                self.usage_history.popleft()
    
    def get_usage_rate(self, period_minutes: int = 60) -> float:
        """Get usage rate over specified period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=period_minutes)
        
        recent_usage = [
            entry for entry in self.usage_history
            if entry["timestamp"] >= cutoff_time
        ]
        
        if not recent_usage:
            return 0.0
        
        total_amount = sum(entry["amount"] for entry in recent_usage)
        return total_amount / period_minutes  # Usage per minute

@dataclass
class TenantIsolationPolicy:
    """Tenant isolation policy configuration"""
    tenant_id: str
    
    # Network isolation
    dedicated_ip_pool: List[str] = field(default_factory=list)
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_domains: Set[str] = field(default_factory=set)
    
    # Compute isolation
    dedicated_cpu_cores: Optional[int] = None
    memory_limit_gb: Optional[float] = None
    dedicated_gpu: bool = False
    
    # Storage isolation
    dedicated_storage_pool: Optional[str] = None
    encryption_key: Optional[str] = None
    backup_isolation: bool = True
    
    # Framework isolation
    allowed_frameworks: Set[str] = field(default_factory=lambda: {"copilot", "agno", "mastra", "mlx", "metacognition"})
    framework_priorities: Dict[str, int] = field(default_factory=dict)
    custom_routing_rules: List[RoutingRule] = field(default_factory=list)
    
    # Security policies
    require_mfa: bool = False
    ip_whitelist: List[str] = field(default_factory=list)
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 10
    
    # Data policies
    data_residency: Optional[str] = None  # Geographic restriction
    audit_level: str = "standard"  # standard, detailed, comprehensive
    retention_days: int = 365
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "dedicated_ip_pool": self.dedicated_ip_pool,
            "allowed_domains": list(self.allowed_domains),
            "blocked_domains": list(self.blocked_domains),
            "dedicated_cpu_cores": self.dedicated_cpu_cores,
            "memory_limit_gb": self.memory_limit_gb,
            "dedicated_gpu": self.dedicated_gpu,
            "dedicated_storage_pool": self.dedicated_storage_pool,
            "encryption_key": self.encryption_key,
            "backup_isolation": self.backup_isolation,
            "allowed_frameworks": list(self.allowed_frameworks),
            "framework_priorities": self.framework_priorities,
            "custom_routing_rules": [rule.__dict__ for rule in self.custom_routing_rules],
            "require_mfa": self.require_mfa,
            "ip_whitelist": self.ip_whitelist,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "data_residency": self.data_residency,
            "audit_level": self.audit_level,
            "retention_days": self.retention_days
        }

class EnhancedTenantManager:
    """Enhanced tenant manager with advanced resource management"""
    
    def __init__(self, base_manager: OrganizationManager = None):
        self.base_manager = base_manager or organization_manager
        
        # Resource quota management
        self.tenant_quotas: Dict[str, List[ResourceQuota]] = defaultdict(list)
        self.resource_usage: Dict[str, Dict[ResourceType, ResourceUsage]] = defaultdict(dict)
        
        # Isolation policies
        self.isolation_policies: Dict[str, TenantIsolationPolicy] = {}
        
        # Performance tracking
        self.tenant_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Real-time monitoring
        self.usage_monitors: Dict[str, asyncio.Task] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        # Quota violation handlers
        self.violation_handlers: Dict[ResourceType, List[callable]] = defaultdict(list)
        
        # Initialize default quotas for enterprise features
        self._initialize_default_quotas()
        
        logger.info("Enhanced tenant manager initialized")
    
    def _initialize_default_quotas(self):
        """Initialize default quota configurations"""
        # Default quotas for different tenant tiers
        self.default_quota_configs = {
            "basic": [
                ResourceQuota(ResourceType.API_REQUESTS, 1000, QuotaPeriod.HOUR, soft_limit=800),
                ResourceQuota(ResourceType.WEBSOCKET_CONNECTIONS, 5, QuotaPeriod.HOUR),
                ResourceQuota(ResourceType.DOCUMENT_PROCESSING, 100, QuotaPeriod.DAY, soft_limit=80),
                ResourceQuota(ResourceType.STORAGE, 1.0, QuotaPeriod.MONTH),  # 1GB
                ResourceQuota(ResourceType.BANDWIDTH, 10.0, QuotaPeriod.MONTH),  # 10GB
                ResourceQuota(ResourceType.COMPUTE_TIME, 60, QuotaPeriod.DAY),  # 1 hour
            ],
            "pro": [
                ResourceQuota(ResourceType.API_REQUESTS, 10000, QuotaPeriod.HOUR, soft_limit=8000),
                ResourceQuota(ResourceType.WEBSOCKET_CONNECTIONS, 25, QuotaPeriod.HOUR),
                ResourceQuota(ResourceType.DOCUMENT_PROCESSING, 1000, QuotaPeriod.DAY, soft_limit=800),
                ResourceQuota(ResourceType.STORAGE, 10.0, QuotaPeriod.MONTH),  # 10GB
                ResourceQuota(ResourceType.BANDWIDTH, 100.0, QuotaPeriod.MONTH),  # 100GB
                ResourceQuota(ResourceType.COMPUTE_TIME, 600, QuotaPeriod.DAY),  # 10 hours
            ],
            "enterprise": [
                ResourceQuota(ResourceType.API_REQUESTS, 100000, QuotaPeriod.HOUR, soft_limit=80000),
                ResourceQuota(ResourceType.WEBSOCKET_CONNECTIONS, 100, QuotaPeriod.HOUR),
                ResourceQuota(ResourceType.DOCUMENT_PROCESSING, 10000, QuotaPeriod.DAY, soft_limit=8000),
                ResourceQuota(ResourceType.STORAGE, 100.0, QuotaPeriod.MONTH),  # 100GB
                ResourceQuota(ResourceType.BANDWIDTH, 1000.0, QuotaPeriod.MONTH),  # 1TB
                ResourceQuota(ResourceType.COMPUTE_TIME, 3600, QuotaPeriod.DAY),  # 60 hours
            ]
        }
    
    async def create_enhanced_tenant(
        self, 
        org_id: str, 
        tenant_data: TenantCreate, 
        creator_user,
        quotas: Optional[List[ResourceQuota]] = None,
        isolation_policy: Optional[TenantIsolationPolicy] = None
    ) -> Tenant:
        """Create tenant with enhanced features"""
        # Create base tenant
        tenant = await self.base_manager.create_tenant(org_id, tenant_data, creator_user)
        
        # Apply enhanced configurations
        await self._setup_tenant_quotas(tenant.id, tenant.tier, quotas)
        await self._setup_tenant_isolation(tenant.id, isolation_policy)
        await self._initialize_tenant_monitoring(tenant.id)
        
        logger.info(f"Created enhanced tenant {tenant.id} with tier {tenant.tier}")
        return tenant
    
    async def _setup_tenant_quotas(
        self, 
        tenant_id: str, 
        tier: str, 
        custom_quotas: Optional[List[ResourceQuota]] = None
    ):
        """Setup resource quotas for tenant"""
        if custom_quotas:
            self.tenant_quotas[tenant_id] = custom_quotas
        else:
            # Use default quotas for tier
            default_quotas = self.default_quota_configs.get(tier, self.default_quota_configs["basic"])
            self.tenant_quotas[tenant_id] = [quota for quota in default_quotas]
        
        # Initialize usage tracking for each quota
        for quota in self.tenant_quotas[tenant_id]:
            if quota.resource_type not in self.resource_usage[tenant_id]:
                self.resource_usage[tenant_id][quota.resource_type] = ResourceUsage(
                    tenant_id=tenant_id,
                    resource_type=quota.resource_type
                )
        
        logger.info(f"Setup {len(self.tenant_quotas[tenant_id])} quotas for tenant {tenant_id}")
    
    async def _setup_tenant_isolation(
        self, 
        tenant_id: str, 
        isolation_policy: Optional[TenantIsolationPolicy] = None
    ):
        """Setup tenant isolation policy"""
        if isolation_policy:
            self.isolation_policies[tenant_id] = isolation_policy
        else:
            # Create default isolation policy
            self.isolation_policies[tenant_id] = TenantIsolationPolicy(tenant_id=tenant_id)
        
        logger.info(f"Setup isolation policy for tenant {tenant_id}")
    
    async def _initialize_tenant_monitoring(self, tenant_id: str):
        """Initialize real-time monitoring for tenant"""
        # Start usage monitoring task
        monitor_task = asyncio.create_task(
            self._monitor_tenant_usage(tenant_id)
        )
        self.usage_monitors[tenant_id] = monitor_task
        
        # Initialize performance tracking
        self.tenant_performance[tenant_id] = {
            "request_count": 0,
            "average_response_time": 0.0,
            "error_count": 0,
            "last_activity": datetime.now(timezone.utc),
            "resource_efficiency": 1.0
        }
        
        logger.info(f"Initialized monitoring for tenant {tenant_id}")
    
    async def _monitor_tenant_usage(self, tenant_id: str):
        """Background task to monitor tenant resource usage"""
        while tenant_id in self.tenant_quotas:
            try:
                await self._check_quota_violations(tenant_id)
                await self._update_usage_metrics(tenant_id)
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Tenant monitoring error for {tenant_id}: {e}")
                await asyncio.sleep(300)  # Longer wait on error
    
    async def record_resource_usage(
        self, 
        tenant_id: str, 
        resource_type: ResourceType, 
        amount: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record resource usage and check quotas"""
        if tenant_id not in self.resource_usage:
            await self._initialize_tenant_monitoring(tenant_id)
        
        if resource_type not in self.resource_usage[tenant_id]:
            self.resource_usage[tenant_id][resource_type] = ResourceUsage(
                tenant_id=tenant_id,
                resource_type=resource_type
            )
        
        usage = self.resource_usage[tenant_id][resource_type]
        
        # Check if usage would exceed quota
        quota_check = await self._check_resource_quota(
            tenant_id, resource_type, amount
        )
        
        if not quota_check["allowed"]:
            await self._handle_quota_violation(
                tenant_id, resource_type, amount, quota_check["quota"]
            )
            return False
        
        # Record usage
        usage.record_usage(amount)
        
        # Log usage for audit
        uap_logger.log_event(
            LogLevel.INFO,
            f"Resource usage recorded: {resource_type.value} = {amount}",
            EventType.AGENT,
            {
                "tenant_id": tenant_id,
                "resource_type": resource_type.value,
                "amount": amount,
                "current_usage": usage.current_usage,
                "metadata": metadata or {}
            },
            "tenant_resource"
        )
        
        return True
    
    async def _check_resource_quota(
        self, 
        tenant_id: str, 
        resource_type: ResourceType, 
        additional_amount: float
    ) -> Dict[str, Any]:
        """Check if resource usage would exceed quota"""
        quotas = self.tenant_quotas.get(tenant_id, [])
        applicable_quotas = [
            quota for quota in quotas 
            if quota.resource_type == resource_type and quota.enabled
        ]
        
        if not applicable_quotas:
            return {"allowed": True, "quota": None}
        
        # Check each applicable quota
        for quota in sorted(applicable_quotas, key=lambda q: q.priority, reverse=True):
            usage = self.resource_usage[tenant_id].get(resource_type)
            if not usage:
                return {"allowed": True, "quota": quota}
            
            # Check if we need to reset usage for the period
            await self._maybe_reset_usage(usage, quota)
            
            projected_usage = usage.current_usage + additional_amount
            
            # Check burst limit first (if available)
            if quota.burst_limit and projected_usage > quota.burst_limit:
                return {
                    "allowed": False, 
                    "quota": quota,
                    "reason": "burst_limit_exceeded",
                    "current": usage.current_usage,
                    "requested": additional_amount,
                    "limit": quota.burst_limit
                }
            
            # Check regular limit
            if projected_usage > quota.limit:
                return {
                    "allowed": False, 
                    "quota": quota,
                    "reason": "limit_exceeded",
                    "current": usage.current_usage,
                    "requested": additional_amount,
                    "limit": quota.limit
                }
            
            # Check soft limit (warning)
            if quota.soft_limit and projected_usage > quota.soft_limit:
                await self._handle_soft_limit_warning(
                    tenant_id, resource_type, projected_usage, quota
                )
        
        return {"allowed": True, "quota": applicable_quotas[0] if applicable_quotas else None}
    
    async def _maybe_reset_usage(self, usage: ResourceUsage, quota: ResourceQuota):
        """Reset usage if period has expired"""
        now = datetime.now(timezone.utc)
        
        # Calculate period duration
        period_duration = {
            QuotaPeriod.MINUTE: timedelta(minutes=1),
            QuotaPeriod.HOUR: timedelta(hours=1),
            QuotaPeriod.DAY: timedelta(days=1),
            QuotaPeriod.WEEK: timedelta(weeks=1),
            QuotaPeriod.MONTH: timedelta(days=30),  # Approximate
            QuotaPeriod.YEAR: timedelta(days=365)   # Approximate
        }[quota.period]
        
        if now - usage.last_reset >= period_duration:
            # Save current usage for rollover if enabled
            if quota.rollover_allowed:
                unused_quota = max(0, quota.limit - usage.current_usage)
                # Could implement rollover logic here
            
            usage.reset_usage(quota.period)
            logger.debug(f"Reset usage for {usage.resource_type.value} (tenant: {usage.tenant_id})")
    
    async def _handle_quota_violation(
        self, 
        tenant_id: str, 
        resource_type: ResourceType, 
        requested_amount: float,
        quota: ResourceQuota
    ):
        """Handle quota violation"""
        usage = self.resource_usage[tenant_id][resource_type]
        usage.violation_count += 1
        usage.last_violation = datetime.now(timezone.utc)
        
        # Log violation
        uap_logger.log_event(
            LogLevel.WARNING,
            f"Quota violation: {resource_type.value} quota exceeded",
            EventType.SECURITY,
            {
                "tenant_id": tenant_id,
                "resource_type": resource_type.value,
                "current_usage": usage.current_usage,
                "requested_amount": requested_amount,
                "quota_limit": quota.limit,
                "violation_count": usage.violation_count
            },
            "quota_violation"
        )
        
        # Call registered violation handlers
        for handler in self.violation_handlers[resource_type]:
            try:
                await handler(tenant_id, resource_type, usage, quota)
            except Exception as e:
                logger.error(f"Violation handler error: {e}")
    
    async def _handle_soft_limit_warning(
        self, 
        tenant_id: str, 
        resource_type: ResourceType, 
        current_usage: float,
        quota: ResourceQuota
    ):
        """Handle soft limit warning"""
        # Log warning
        uap_logger.log_event(
            LogLevel.WARNING,
            f"Soft limit warning: {resource_type.value} approaching quota",
            EventType.AGENT,
            {
                "tenant_id": tenant_id,
                "resource_type": resource_type.value,
                "current_usage": current_usage,
                "soft_limit": quota.soft_limit,
                "hard_limit": quota.limit,
                "usage_percentage": (current_usage / quota.limit) * 100
            },
            "quota_warning"
        )
    
    async def _check_quota_violations(self, tenant_id: str):
        """Check for quota violations across all resources"""
        for resource_type, usage in self.resource_usage.get(tenant_id, {}).items():
            quotas = [
                q for q in self.tenant_quotas.get(tenant_id, []) 
                if q.resource_type == resource_type and q.enabled
            ]
            
            for quota in quotas:
                await self._maybe_reset_usage(usage, quota)
                
                if usage.current_usage > quota.limit:
                    await self._handle_quota_violation(
                        tenant_id, resource_type, 0, quota
                    )
    
    async def _update_usage_metrics(self, tenant_id: str):
        """Update usage metrics for monitoring"""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resource_usage": {},
            "quota_status": {}
        }
        
        for resource_type, usage in self.resource_usage.get(tenant_id, {}).items():
            metrics["resource_usage"][resource_type.value] = {
                "current_usage": usage.current_usage,
                "peak_usage": usage.peak_usage,
                "usage_rate": usage.get_usage_rate(),
                "violation_count": usage.violation_count
            }
            
            # Calculate quota utilization
            quotas = [
                q for q in self.tenant_quotas.get(tenant_id, []) 
                if q.resource_type == resource_type
            ]
            
            for quota in quotas:
                utilization = (usage.current_usage / quota.limit) * 100 if quota.limit > 0 else 0
                metrics["quota_status"][f"{resource_type.value}_{quota.period.value}"] = {
                    "utilization_percent": utilization,
                    "remaining": max(0, quota.limit - usage.current_usage),
                    "soft_limit_reached": quota.soft_limit and usage.current_usage >= quota.soft_limit,
                    "limit_reached": usage.current_usage >= quota.limit
                }
        
        # Store metrics for monitoring dashboard
        if tenant_id not in self.tenant_performance:
            self.tenant_performance[tenant_id] = {}
        
        self.tenant_performance[tenant_id]["usage_metrics"] = metrics
    
    def get_tenant_quotas(self, tenant_id: str) -> List[ResourceQuota]:
        """Get all quotas for tenant"""
        return self.tenant_quotas.get(tenant_id, [])
    
    def add_tenant_quota(
        self, 
        tenant_id: str, 
        quota: ResourceQuota
    ) -> bool:
        """Add quota for tenant"""
        if tenant_id not in self.tenant_quotas:
            self.tenant_quotas[tenant_id] = []
        
        self.tenant_quotas[tenant_id].append(quota)
        
        # Initialize usage tracking if needed
        if tenant_id not in self.resource_usage:
            self.resource_usage[tenant_id] = {}
        
        if quota.resource_type not in self.resource_usage[tenant_id]:
            self.resource_usage[tenant_id][quota.resource_type] = ResourceUsage(
                tenant_id=tenant_id,
                resource_type=quota.resource_type
            )
        
        logger.info(f"Added quota for tenant {tenant_id}: {quota.resource_type.value}")
        return True
    
    def remove_tenant_quota(
        self, 
        tenant_id: str, 
        resource_type: ResourceType, 
        period: QuotaPeriod
    ) -> bool:
        """Remove specific quota for tenant"""
        if tenant_id not in self.tenant_quotas:
            return False
        
        original_count = len(self.tenant_quotas[tenant_id])
        self.tenant_quotas[tenant_id] = [
            q for q in self.tenant_quotas[tenant_id]
            if not (q.resource_type == resource_type and q.period == period)
        ]
        
        removed = len(self.tenant_quotas[tenant_id]) < original_count
        if removed:
            logger.info(f"Removed quota for tenant {tenant_id}: {resource_type.value} ({period.value})")
        
        return removed
    
    def update_tenant_quota(
        self, 
        tenant_id: str, 
        resource_type: ResourceType, 
        period: QuotaPeriod,
        new_limit: float
    ) -> bool:
        """Update quota limit for tenant"""
        quotas = self.tenant_quotas.get(tenant_id, [])
        
        for quota in quotas:
            if quota.resource_type == resource_type and quota.period == period:
                old_limit = quota.limit
                quota.limit = new_limit
                logger.info(
                    f"Updated quota for tenant {tenant_id}: {resource_type.value} "
                    f"({period.value}) from {old_limit} to {new_limit}"
                )
                return True
        
        return False
    
    def get_tenant_isolation_policy(self, tenant_id: str) -> Optional[TenantIsolationPolicy]:
        """Get isolation policy for tenant"""
        return self.isolation_policies.get(tenant_id)
    
    def update_tenant_isolation_policy(
        self, 
        tenant_id: str, 
        policy: TenantIsolationPolicy
    ):
        """Update isolation policy for tenant"""
        self.isolation_policies[tenant_id] = policy
        logger.info(f"Updated isolation policy for tenant {tenant_id}")
    
    def register_violation_handler(
        self, 
        resource_type: ResourceType, 
        handler: callable
    ):
        """Register handler for quota violations"""
        self.violation_handlers[resource_type].append(handler)
        logger.info(f"Registered violation handler for {resource_type.value}")
    
    async def get_tenant_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive usage summary for tenant"""
        summary = {
            "tenant_id": tenant_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quotas": [],
            "usage": {},
            "violations": {},
            "performance": self.tenant_performance.get(tenant_id, {}),
            "isolation_policy": None
        }
        
        # Quota information
        for quota in self.tenant_quotas.get(tenant_id, []):
            usage = self.resource_usage.get(tenant_id, {}).get(quota.resource_type)
            
            quota_info = quota.to_dict()
            if usage:
                quota_info.update({
                    "current_usage": usage.current_usage,
                    "peak_usage": usage.peak_usage,
                    "utilization_percent": (usage.current_usage / quota.limit) * 100 if quota.limit > 0 else 0,
                    "usage_rate": usage.get_usage_rate(),
                    "violation_count": usage.violation_count,
                    "last_violation": usage.last_violation.isoformat() if usage.last_violation else None
                })
            
            summary["quotas"].append(quota_info)
        
        # Resource usage
        for resource_type, usage in self.resource_usage.get(tenant_id, {}).items():
            summary["usage"][resource_type.value] = {
                "current_usage": usage.current_usage,
                "peak_usage": usage.peak_usage,
                "usage_rate": usage.get_usage_rate(),
                "last_reset": usage.last_reset.isoformat()
            }
            
            if usage.violation_count > 0:
                summary["violations"][resource_type.value] = {
                    "count": usage.violation_count,
                    "last_violation": usage.last_violation.isoformat() if usage.last_violation else None
                }
        
        # Isolation policy
        isolation_policy = self.isolation_policies.get(tenant_id)
        if isolation_policy:
            summary["isolation_policy"] = isolation_policy.to_dict()
        
        return summary
    
    async def cleanup_tenant(self, tenant_id: str):
        """Clean up tenant resources"""
        # Stop monitoring
        if tenant_id in self.usage_monitors:
            self.usage_monitors[tenant_id].cancel()
            try:
                await self.usage_monitors[tenant_id]
            except asyncio.CancelledError:
                pass
            del self.usage_monitors[tenant_id]
        
        # Clean up data
        if tenant_id in self.tenant_quotas:
            del self.tenant_quotas[tenant_id]
        
        if tenant_id in self.resource_usage:
            del self.resource_usage[tenant_id]
        
        if tenant_id in self.isolation_policies:
            del self.isolation_policies[tenant_id]
        
        if tenant_id in self.tenant_performance:
            del self.tenant_performance[tenant_id]
        
        logger.info(f"Cleaned up enhanced tenant resources for {tenant_id}")
    
    async def cleanup(self):
        """Clean up enhanced tenant manager"""
        # Stop all monitoring tasks
        for task in self.usage_monitors.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.usage_monitors:
            try:
                await asyncio.gather(*self.usage_monitors.values(), return_exceptions=True)
            except Exception as e:
                logger.error(f"Error during task cleanup: {e}")
        
        # Clear data
        self.tenant_quotas.clear()
        self.resource_usage.clear()
        self.isolation_policies.clear()
        self.tenant_performance.clear()
        self.usage_monitors.clear()
        self.violation_handlers.clear()
        
        logger.info("Enhanced tenant manager cleanup complete")

# Global enhanced tenant manager instance
enhanced_tenant_manager = EnhancedTenantManager()
