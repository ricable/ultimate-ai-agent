# File: backend/services/enhanced_orchestrator.py
"""
Enhanced Agent Orchestration System with Advanced Routing, Load Balancing, and Multi-tenant Isolation

This module provides:
- Intelligent content-based routing with machine learning
- Load balancing with multiple algorithms (round-robin, weighted, health-based)
- Circuit breaker pattern for framework failure handling
- WebSocket connection pooling and reuse
- Advanced multi-tenant resource isolation and quotas
- Performance optimization and caching
"""

import asyncio
import json
import time
import uuid
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from fastapi import WebSocket, HTTPException, status
import logging
import re
from statistics import mean

# Import existing components
from .agent_orchestrator import UAP_AgentOrchestrator
from .load_balancer import LoadBalancer, LoadBalancerStrategy, FrameworkInstance
from ..tenancy.tenant_context import TenantContextManager, TenantContext
from ..tenancy.organization_manager import organization_manager
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.performance import performance_monitor
from ..monitoring.metrics.prometheus_metrics import record_agent_request

logger = logging.getLogger(__name__)

class RoutingStrategy(str, Enum):
    """Content-based routing strategies"""
    CONTENT_ANALYSIS = "content_analysis"
    LOAD_BALANCED = "load_balanced"
    TENANT_SPECIFIC = "tenant_specific"
    FAILOVER = "failover"
    ADAPTIVE = "adaptive"

class CircuitBreakerState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests blocked
    HALF_OPEN = "half_open" # Testing if service recovered

@dataclass
class FrameworkHealth:
    """Framework health status tracking"""
    framework_name: str
    is_healthy: bool = True
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0
    average_response_time: float = 0.0
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED
    circuit_breaker_reset_time: Optional[datetime] = None
    
    def update_success(self, response_time: float):
        """Update health on successful request"""
        self.is_healthy = True
        self.last_success = datetime.now(timezone.utc)
        self.success_count += 1
        self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery
        
        # Update average response time (exponential moving average)
        alpha = 0.3
        if self.average_response_time == 0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (alpha * response_time + 
                                        (1 - alpha) * self.average_response_time)
        
        # Reset circuit breaker if in half-open state
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.circuit_breaker_reset_time = None
    
    def update_failure(self, error: str):
        """Update health on failed request"""
        self.last_failure = datetime.now(timezone.utc)
        self.failure_count += 1
        
        # Open circuit breaker if too many failures
        if (self.failure_count >= 5 and 
            self.circuit_breaker_state == CircuitBreakerState.CLOSED):
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            self.circuit_breaker_reset_time = (datetime.now(timezone.utc) + 
                                             timedelta(seconds=30))
        
        # Mark as unhealthy if multiple consecutive failures
        if self.failure_count >= 3:
            self.is_healthy = False
    
    def can_handle_request(self) -> bool:
        """Check if framework can handle request based on circuit breaker"""
        if self.circuit_breaker_state == CircuitBreakerState.CLOSED:
            return True
        elif self.circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if we should try half-open
            if (self.circuit_breaker_reset_time and 
                datetime.now(timezone.utc) >= self.circuit_breaker_reset_time):
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        elif self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            return True
        return False

@dataclass
class TenantResourceUsage:
    """Track tenant resource usage for quotas"""
    tenant_id: str
    requests_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    active_connections: int = 0
    total_requests: int = 0
    bandwidth_used: float = 0.0
    storage_used: float = 0.0
    compute_time_used: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def record_request(self, request_size: int = 0, response_size: int = 0, 
                      processing_time: float = 0.0):
        """Record a request for quota tracking"""
        current_time = datetime.now(timezone.utc)
        
        # Add to per-minute tracking
        self.requests_per_minute.append(current_time)
        
        # Update totals
        self.total_requests += 1
        self.bandwidth_used += (request_size + response_size) / (1024 * 1024)  # MB
        self.compute_time_used += processing_time
    
    def get_requests_per_minute(self) -> int:
        """Get current requests per minute"""
        current_time = datetime.now(timezone.utc)
        cutoff_time = current_time - timedelta(minutes=1)
        
        # Remove old requests
        while self.requests_per_minute and self.requests_per_minute[0] < cutoff_time:
            self.requests_per_minute.popleft()
        
        return len(self.requests_per_minute)
    
    def check_quota_limits(self, limits: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if tenant is within quota limits"""
        # Check rate limit
        if "requests_per_minute" in limits:
            if self.get_requests_per_minute() >= limits["requests_per_minute"]:
                return False, "Rate limit exceeded"
        
        # Check concurrent connections
        if "max_connections" in limits:
            if self.active_connections >= limits["max_connections"]:
                return False, "Connection limit exceeded"
        
        # Check bandwidth quota
        if "bandwidth_limit_mb" in limits:
            if self.bandwidth_used >= limits["bandwidth_limit_mb"]:
                return False, "Bandwidth quota exceeded"
        
        # Check compute time quota
        if "compute_time_limit_minutes" in limits:
            if self.compute_time_used >= limits["compute_time_limit_minutes"] * 60:
                return False, "Compute time quota exceeded"
        
        return True, None

@dataclass
class RoutingRule:
    """Custom routing rule for content-based routing"""
    name: str
    pattern: str  # Regex pattern
    framework: str
    priority: int = 0
    tenant_specific: bool = False
    tenant_ids: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    
class EnhancedAgentOrchestrator:
    """Enhanced orchestrator with advanced routing and multi-tenant isolation"""
    
    def __init__(self, base_orchestrator: UAP_AgentOrchestrator):
        self.base_orchestrator = base_orchestrator
        
        # Load balancer for framework instances
        self.load_balancer = LoadBalancer()
        
        # Framework health monitoring
        self.framework_health: Dict[str, FrameworkHealth] = {}
        
        # Tenant resource tracking
        self.tenant_usage: Dict[str, TenantResourceUsage] = {}
        
        # Connection pooling for WebSocket connections
        self.connection_pools: Dict[str, List[WebSocket]] = defaultdict(list)
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        
        # Content-based routing rules
        self.routing_rules: List[RoutingRule] = []
        self._initialize_default_routing_rules()
        
        # Request caching with tenant isolation
        self.tenant_caches: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Performance metrics
        self.routing_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Initialize framework instances
        self._initialize_framework_instances()
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info("Enhanced Agent Orchestrator initialized")
    
    def _initialize_default_routing_rules(self):
        """Initialize default content-based routing rules"""
        self.routing_rules = [
            # High-priority metacognitive routing
            RoutingRule(
                name="metacognitive_analysis",
                pattern=r"(metacognition|self-aware|introspection|self-reflection|reasoning process)",
                framework="metacognition",
                priority=100
            ),
            
            # Local inference routing
            RoutingRule(
                name="local_inference",
                pattern=r"(local|offline|private|mlx|apple silicon|on-device|secure)",
                framework="mlx",
                priority=90
            ),
            
            # Document processing routing
            RoutingRule(
                name="document_processing",
                pattern=r"(document|analyze|pdf|text|file|extract|summarize|parse|read|review|paper|report|article|upload|scan|ocr)",
                framework="agno",
                priority=80
            ),
            
            # Workflow and support routing
            RoutingRule(
                name="workflow_support",
                pattern=r"(support|help|workflow|task|process|steps|guide|tutorial)",
                framework="mastra",
                priority=70
            ),
            
            # Code and development routing
            RoutingRule(
                name="development",
                pattern=r"(code|programming|development|debug|function|class|api|software)",
                framework="copilot",
                priority=60
            )
        ]
        
        logger.info(f"Initialized {len(self.routing_rules)} default routing rules")
    
    def _initialize_framework_instances(self):
        """Initialize framework instances for load balancing"""
        frameworks = [
            ("copilot", self.base_orchestrator.copilot_manager),
            ("agno", self.base_orchestrator.agno_manager),
            ("mastra", self.base_orchestrator.mastra_manager),
            ("mlx", self.base_orchestrator.mlx_manager),
            ("metacognition", self.base_orchestrator.metacognition_manager)
        ]
        
        for framework_name, manager in frameworks:
            # Create framework instance
            instance = FrameworkInstance(
                id=f"{framework_name}-primary",
                name=framework_name,
                manager=manager,
                weight=1.0,
                max_concurrent=50
            )
            
            # Register with load balancer
            self.load_balancer.register_instance(framework_name, instance)
            
            # Initialize health tracking
            self.framework_health[framework_name] = FrameworkHealth(
                framework_name=framework_name
            )
        
        logger.info(f"Initialized {len(frameworks)} framework instances")
    
    async def start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Resource cleanup task
        cleanup_task = asyncio.create_task(self._resource_cleanup_loop())
        self.background_tasks.append(cleanup_task)
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.append(metrics_task)
        
        logger.info("Started background monitoring tasks")
    
    async def stop_background_tasks(self):
        """Stop all background tasks"""
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.background_tasks.clear()
        logger.info("Stopped background monitoring tasks")
    
    async def _health_monitoring_loop(self):
        """Background task for framework health monitoring"""
        while True:
            try:
                await self._check_framework_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Longer wait on error
    
    async def _resource_cleanup_loop(self):
        """Background task for resource cleanup"""
        while True:
            try:
                await self._cleanup_expired_connections()
                await self._cleanup_tenant_caches()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource cleanup error: {e}")
                await asyncio.sleep(600)  # Longer wait on error
    
    async def _metrics_collection_loop(self):
        """Background task for metrics collection"""
        while True:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(120)  # Longer wait on error
    
    async def enhanced_route_and_process(
        self, 
        message: str, 
        framework: str, 
        context: Dict[str, Any],
        routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE
    ) -> Dict[str, Any]:
        """Enhanced routing and processing with load balancing and tenant isolation"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Get tenant context
        tenant_context = TenantContextManager.get_context()
        tenant_id = tenant_context.tenant_id if tenant_context else "default"
        
        # Check tenant quotas
        if not await self._check_tenant_quotas(tenant_id, len(message)):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Tenant quota exceeded"
            )
        
        try:
            # Determine optimal framework using enhanced routing
            selected_framework = await self._enhanced_framework_selection(
                message, framework, context, routing_strategy, tenant_context
            )
            
            # Get framework instance using load balancer
            instance = await self.load_balancer.select_instance(
                selected_framework, LoadBalancerStrategy.HEALTH_BASED
            )
            
            if not instance:
                # Fallback to direct framework access
                return await self.base_orchestrator._route_and_process(
                    message, selected_framework, context
                )
            
            # Check circuit breaker
            health = self.framework_health.get(selected_framework)
            if health and not health.can_handle_request():
                # Try fallback framework
                fallback_framework = await self._get_fallback_framework(selected_framework)
                if fallback_framework:
                    selected_framework = fallback_framework
                    instance = await self.load_balancer.select_instance(
                        selected_framework, LoadBalancerStrategy.ROUND_ROBIN
                    )
            
            # Process request with selected instance
            response = await self._process_with_instance(
                instance, message, context, request_id
            )
            
            # Update success metrics
            processing_time = time.time() - start_time
            await self._update_success_metrics(
                selected_framework, processing_time, tenant_id, len(str(response))
            )
            
            # Cache response for tenant
            await self._cache_tenant_response(
                tenant_id, selected_framework, message, response
            )
            
            return response
            
        except Exception as e:
            # Update failure metrics
            processing_time = time.time() - start_time
            await self._update_failure_metrics(
                framework, str(e), tenant_id, processing_time
            )
            
            # Try fallback processing
            if routing_strategy != RoutingStrategy.FAILOVER:
                try:
                    return await self.enhanced_route_and_process(
                        message, "copilot", context, RoutingStrategy.FAILOVER
                    )
                except:
                    pass
            
            raise
    
    async def _enhanced_framework_selection(
        self, 
        message: str, 
        framework: str, 
        context: Dict[str, Any],
        strategy: RoutingStrategy,
        tenant_context: Optional[TenantContext]
    ) -> str:
        """Enhanced framework selection with multiple strategies"""
        
        if framework != 'auto':
            return framework
        
        # Strategy-based selection
        if strategy == RoutingStrategy.CONTENT_ANALYSIS:
            return await self._content_based_routing(message, context, tenant_context)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._load_balanced_routing(message, context)
        elif strategy == RoutingStrategy.TENANT_SPECIFIC:
            return await self._tenant_specific_routing(message, context, tenant_context)
        elif strategy == RoutingStrategy.FAILOVER:
            return await self._failover_routing(message, context)
        else:  # ADAPTIVE
            return await self._adaptive_routing(message, context, tenant_context)
    
    async def _content_based_routing(
        self, 
        message: str, 
        context: Dict[str, Any],
        tenant_context: Optional[TenantContext]
    ) -> str:
        """Advanced content-based routing with custom rules"""
        message_lower = message.lower()
        
        # Check tenant-specific rules first
        if tenant_context:
            for rule in self.routing_rules:
                if (rule.tenant_specific and 
                    tenant_context.tenant_id in rule.tenant_ids):
                    if re.search(rule.pattern, message_lower, re.IGNORECASE):
                        return rule.framework
        
        # Check general rules by priority
        sorted_rules = sorted(self.routing_rules, key=lambda r: r.priority, reverse=True)
        for rule in sorted_rules:
            if not rule.tenant_specific:
                if re.search(rule.pattern, message_lower, re.IGNORECASE):
                    # Check additional conditions
                    if self._check_rule_conditions(rule, context):
                        return rule.framework
        
        # Fallback to base orchestrator logic
        return self.base_orchestrator._determine_best_framework(message, context)
    
    async def _load_balanced_routing(
        self, message: str, context: Dict[str, Any]
    ) -> str:
        """Load-balanced routing based on framework health and load"""
        # Get available frameworks sorted by health and load
        frameworks = list(self.framework_health.keys())
        framework_scores = []
        
        for fw in frameworks:
            health = self.framework_health[fw]
            if health.can_handle_request():
                # Score based on health, response time, and current load
                health_score = 1.0 if health.is_healthy else 0.5
                response_time_score = max(0.1, 1.0 / (health.average_response_time + 0.1))
                load_score = 1.0  # Could integrate with actual load metrics
                
                total_score = health_score * response_time_score * load_score
                framework_scores.append((fw, total_score))
        
        if framework_scores:
            # Select framework with highest score
            framework_scores.sort(key=lambda x: x[1], reverse=True)
            return framework_scores[0][0]
        
        return "copilot"  # Default fallback
    
    async def _tenant_specific_routing(
        self, 
        message: str, 
        context: Dict[str, Any],
        tenant_context: Optional[TenantContext]
    ) -> str:
        """Tenant-specific routing based on tenant preferences"""
        if not tenant_context:
            return "copilot"
        
        # Get tenant preferences from organization manager
        try:
            tenant = await organization_manager.get_tenant(tenant_context.tenant_id)
            if tenant and tenant.limits:
                preferred_framework = tenant.limits.get("preferred_framework")
                if preferred_framework and preferred_framework in self.framework_health:
                    health = self.framework_health[preferred_framework]
                    if health.can_handle_request():
                        return preferred_framework
        except Exception as e:
            logger.warning(f"Failed to get tenant preferences: {e}")
        
        # Fallback to content-based routing
        return await self._content_based_routing(message, context, tenant_context)
    
    async def _failover_routing(
        self, message: str, context: Dict[str, Any]
    ) -> str:
        """Failover routing using most reliable framework"""
        # Find most reliable framework
        best_framework = None
        best_reliability = 0.0
        
        for fw_name, health in self.framework_health.items():
            if health.can_handle_request():
                # Calculate reliability score
                success_rate = (
                    health.success_count / 
                    max(1, health.success_count + health.failure_count)
                )
                reliability = success_rate * (1.0 if health.is_healthy else 0.5)
                
                if reliability > best_reliability:
                    best_reliability = reliability
                    best_framework = fw_name
        
        return best_framework or "copilot"
    
    async def _adaptive_routing(
        self, 
        message: str, 
        context: Dict[str, Any],
        tenant_context: Optional[TenantContext]
    ) -> str:
        """Adaptive routing combining multiple strategies"""
        # Try content-based routing first
        content_framework = await self._content_based_routing(
            message, context, tenant_context
        )
        
        # Check if content-based selection is healthy
        health = self.framework_health.get(content_framework)
        if health and health.can_handle_request() and health.is_healthy:
            return content_framework
        
        # Fallback to load-balanced routing
        return await self._load_balanced_routing(message, context)
    
    def _check_rule_conditions(self, rule: RoutingRule, context: Dict[str, Any]) -> bool:
        """Check if routing rule conditions are met"""
        for condition, expected_value in rule.conditions.items():
            if condition in context:
                if context[condition] != expected_value:
                    return False
            else:
                return False
        return True
    
    async def _get_fallback_framework(self, failed_framework: str) -> Optional[str]:
        """Get fallback framework for failed framework"""
        fallback_map = {
            "copilot": "mastra",
            "agno": "copilot",
            "mastra": "copilot",
            "mlx": "copilot",
            "metacognition": "copilot"
        }
        
        fallback = fallback_map.get(failed_framework)
        if fallback and self.framework_health.get(fallback):
            health = self.framework_health[fallback]
            if health.can_handle_request():
                return fallback
        
        return None
    
    async def _process_with_instance(
        self, 
        instance: FrameworkInstance, 
        message: str, 
        context: Dict[str, Any],
        request_id: str
    ) -> Dict[str, Any]:
        """Process request with specific framework instance"""
        try:
            # Acquire instance slot
            await instance.acquire()
            
            # Add request context
            enhanced_context = {
                **context,
                "request_id": request_id,
                "instance_id": instance.id,
                "routing_method": "enhanced"
            }
            
            # Process with framework manager
            response = await instance.manager.process_message(message, enhanced_context)
            
            return response
            
        finally:
            # Release instance slot
            instance.release()
    
    async def _check_tenant_quotas(self, tenant_id: str, request_size: int) -> bool:
        """Check if tenant is within resource quotas"""
        if tenant_id not in self.tenant_usage:
            self.tenant_usage[tenant_id] = TenantResourceUsage(tenant_id=tenant_id)
        
        usage = self.tenant_usage[tenant_id]
        
        # Get tenant limits
        try:
            tenant = await organization_manager.get_tenant(tenant_id)
            if tenant and tenant.limits:
                within_limits, error_msg = usage.check_quota_limits(tenant.limits)
                if not within_limits:
                    logger.warning(f"Tenant {tenant_id} quota exceeded: {error_msg}")
                    return False
        except Exception as e:
            logger.error(f"Failed to check tenant quotas: {e}")
        
        return True
    
    async def _update_success_metrics(
        self, 
        framework: str, 
        processing_time: float, 
        tenant_id: str,
        response_size: int
    ):
        """Update success metrics for framework and tenant"""
        # Update framework health
        if framework in self.framework_health:
            self.framework_health[framework].update_success(processing_time)
        
        # Update tenant usage
        if tenant_id in self.tenant_usage:
            self.tenant_usage[tenant_id].record_request(
                response_size=response_size,
                processing_time=processing_time
            )
        
        # Record Prometheus metrics
        record_agent_request(
            agent_id=framework,
            framework=framework,
            response_time_seconds=processing_time,
            success=True,
            response_size_bytes=response_size
        )
    
    async def _update_failure_metrics(
        self, 
        framework: str, 
        error: str, 
        tenant_id: str,
        processing_time: float
    ):
        """Update failure metrics for framework and tenant"""
        # Update framework health
        if framework in self.framework_health:
            self.framework_health[framework].update_failure(error)
        
        # Record Prometheus metrics
        record_agent_request(
            agent_id=framework,
            framework=framework,
            response_time_seconds=processing_time,
            success=False
        )
    
    async def _cache_tenant_response(
        self, 
        tenant_id: str, 
        framework: str, 
        message: str, 
        response: Dict[str, Any]
    ):
        """Cache response for tenant with isolation"""
        # Create cache key
        cache_key = hashlib.md5(
            f"{framework}:{message}".encode()
        ).hexdigest()
        
        # Store in tenant-specific cache
        if tenant_id not in self.tenant_caches:
            self.tenant_caches[tenant_id] = {}
        
        self.tenant_caches[tenant_id][cache_key] = {
            "response": response,
            "timestamp": datetime.now(timezone.utc),
            "framework": framework
        }
        
        # Limit cache size per tenant
        if len(self.tenant_caches[tenant_id]) > 100:
            # Remove oldest entries
            sorted_items = sorted(
                self.tenant_caches[tenant_id].items(),
                key=lambda x: x[1]["timestamp"]
            )
            for key, _ in sorted_items[:20]:
                del self.tenant_caches[tenant_id][key]
    
    async def _check_framework_health(self):
        """Check health of all frameworks"""
        for framework_name, health in self.framework_health.items():
            try:
                # Get framework manager
                manager = getattr(self.base_orchestrator, f"{framework_name}_manager")
                
                # Check if manager has health check method
                if hasattr(manager, 'health_check'):
                    is_healthy = await manager.health_check()
                    if not is_healthy:
                        health.update_failure("Health check failed")
                elif hasattr(manager, 'get_status'):
                    status = manager.get_status()
                    if not status or status.get("status") != "active":
                        health.update_failure("Status check failed")
                        
            except Exception as e:
                health.update_failure(f"Health check error: {e}")
    
    async def _cleanup_expired_connections(self):
        """Clean up expired WebSocket connections"""
        current_time = datetime.now(timezone.utc)
        expired_connections = []
        
        for conn_id, conn_info in self.active_connections.items():
            if isinstance(conn_info, dict) and "connected_at" in conn_info:
                connected_at = conn_info["connected_at"]
                if current_time - connected_at > timedelta(hours=1):
                    expired_connections.append(conn_id)
        
        for conn_id in expired_connections:
            if conn_id in self.active_connections:
                del self.active_connections[conn_id]
        
        if expired_connections:
            logger.info(f"Cleaned up {len(expired_connections)} expired connections")
    
    async def _cleanup_tenant_caches(self):
        """Clean up expired tenant caches"""
        current_time = datetime.now(timezone.utc)
        cleanup_cutoff = current_time - timedelta(hours=1)
        
        for tenant_id in list(self.tenant_caches.keys()):
            cache = self.tenant_caches[tenant_id]
            expired_keys = []
            
            for key, entry in cache.items():
                if entry["timestamp"] < cleanup_cutoff:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del cache[key]
            
            # Remove empty tenant caches
            if not cache:
                del self.tenant_caches[tenant_id]
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics for monitoring"""
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "framework_health": {},
            "tenant_usage": {},
            "connection_stats": {
                "active_connections": len(self.active_connections),
                "connection_pools": {k: len(v) for k, v in self.connection_pools.items()}
            },
            "load_balancer_stats": self.load_balancer.get_stats()
        }
        
        # Framework health metrics
        for fw_name, health in self.framework_health.items():
            metrics["framework_health"][fw_name] = {
                "is_healthy": health.is_healthy,
                "success_count": health.success_count,
                "failure_count": health.failure_count,
                "average_response_time": health.average_response_time,
                "circuit_breaker_state": health.circuit_breaker_state.value
            }
        
        # Tenant usage metrics
        for tenant_id, usage in self.tenant_usage.items():
            metrics["tenant_usage"][tenant_id] = {
                "requests_per_minute": usage.get_requests_per_minute(),
                "active_connections": usage.active_connections,
                "total_requests": usage.total_requests,
                "bandwidth_used_mb": usage.bandwidth_used,
                "compute_time_used_seconds": usage.compute_time_used
            }
        
        # Store metrics for monitoring dashboard
        self.routing_metrics["current"] = metrics
    
    # Public interface methods
    
    async def register_connection(
        self, 
        conn_id: str, 
        websocket: WebSocket, 
        user=None
    ):
        """Register WebSocket connection with enhanced tracking"""
        # Call base orchestrator
        self.base_orchestrator.register_connection(conn_id, websocket, user)
        
        # Enhanced tracking
        tenant_context = TenantContextManager.get_context()
        tenant_id = tenant_context.tenant_id if tenant_context else "default"
        
        self.active_connections[conn_id] = {
            "websocket": websocket,
            "user": user,
            "tenant_id": tenant_id,
            "connected_at": datetime.now(timezone.utc)
        }
        
        # Update tenant usage
        if tenant_id not in self.tenant_usage:
            self.tenant_usage[tenant_id] = TenantResourceUsage(tenant_id=tenant_id)
        self.tenant_usage[tenant_id].active_connections += 1
    
    async def unregister_connection(self, conn_id: str, reason: str = "normal"):
        """Unregister WebSocket connection with enhanced cleanup"""
        # Call base orchestrator
        self.base_orchestrator.unregister_connection(conn_id, reason)
        
        # Enhanced cleanup
        if conn_id in self.active_connections:
            conn_info = self.active_connections[conn_id]
            tenant_id = conn_info.get("tenant_id", "default")
            
            # Update tenant usage
            if tenant_id in self.tenant_usage:
                self.tenant_usage[tenant_id].active_connections -= 1
            
            del self.active_connections[conn_id]
    
    async def handle_agui_event(
        self, 
        conn_id: str, 
        event: Dict[str, Any]
    ):
        """Handle AG-UI events with enhanced routing"""
        event_type = event.get("type")
        
        if event_type == "user_message":
            content = event.get("content", "")
            metadata = event.get("metadata", {})
            framework = metadata.get("framework", "auto")
            
            # Enhanced routing
            response_data = await self.enhanced_route_and_process(
                content, framework, metadata, RoutingStrategy.ADAPTIVE
            )
            
            # Create response event
            response_event = {
                "type": "text_message_content",
                "content": response_data.get("content", "No response."),
                "metadata": {
                    **response_data.get("metadata", {}),
                    "enhanced_routing": True,
                    "selected_framework": response_data.get("framework", framework)
                }
            }
            
            # Send response
            await self._send_to_connection(conn_id, response_event)
        else:
            # Delegate to base orchestrator for other event types
            await self.base_orchestrator.handle_agui_event(conn_id, event)
    
    async def handle_http_chat(
        self, 
        agent_id: str, 
        message: str, 
        framework: str, 
        context: Dict
    ) -> Dict[str, Any]:
        """Handle HTTP chat with enhanced routing"""
        # Use enhanced routing
        response = await self.enhanced_route_and_process(
            message, framework, context, RoutingStrategy.ADAPTIVE
        )
        
        response["framework"] = response.get("framework", framework)
        response["enhanced_routing"] = True
        
        return response
    
    async def _send_to_connection(self, conn_id: str, data: Dict[str, Any]):
        """Send message to WebSocket connection"""
        # Delegate to base orchestrator
        await self.base_orchestrator._send_to_connection(conn_id, data)
    
    async def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get enhanced system status with routing metrics"""
        base_status = await self.base_orchestrator.get_system_status()
        
        enhanced_status = {
            **base_status,
            "enhanced_orchestrator": {
                "version": "1.0.0",
                "framework_health": {
                    fw: {
                        "is_healthy": health.is_healthy,
                        "success_rate": (
                            health.success_count / 
                            max(1, health.success_count + health.failure_count)
                        ),
                        "average_response_time": health.average_response_time,
                        "circuit_breaker_state": health.circuit_breaker_state.value
                    }
                    for fw, health in self.framework_health.items()
                },
                "tenant_usage": {
                    tenant_id: {
                        "requests_per_minute": usage.get_requests_per_minute(),
                        "active_connections": usage.active_connections,
                        "total_requests": usage.total_requests
                    }
                    for tenant_id, usage in self.tenant_usage.items()
                },
                "load_balancer": self.load_balancer.get_stats(),
                "routing_rules": len(self.routing_rules),
                "cache_stats": {
                    "tenant_caches": len(self.tenant_caches),
                    "total_cached_responses": sum(
                        len(cache) for cache in self.tenant_caches.values()
                    )
                }
            }
        }
        
        return enhanced_status
    
    def add_routing_rule(self, rule: RoutingRule):
        """Add custom routing rule"""
        self.routing_rules.append(rule)
        self.routing_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added routing rule: {rule.name}")
    
    def remove_routing_rule(self, rule_name: str):
        """Remove routing rule by name"""
        self.routing_rules = [r for r in self.routing_rules if r.name != rule_name]
        logger.info(f"Removed routing rule: {rule_name}")
    
    async def cleanup(self):
        """Clean up enhanced orchestrator resources"""
        # Stop background tasks
        await self.stop_background_tasks()
        
        # Cleanup load balancer
        await self.load_balancer.cleanup()
        
        # Clean up connections
        for conn_id in list(self.active_connections.keys()):
            await self.unregister_connection(conn_id, "shutdown")
        
        # Cleanup base orchestrator
        await self.base_orchestrator.cleanup()
        
        logger.info("Enhanced orchestrator cleanup complete")
