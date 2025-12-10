# File: backend/services/load_balancer.py
"""
Load Balancer for Agent Framework Instances

Provides multiple load balancing algorithms:
- Round Robin: Distribute requests evenly across instances
- Weighted Round Robin: Distribute based on instance weights
- Health-based: Route to healthiest instances first
- Least Connections: Route to instance with fewest active connections
- Response Time: Route to fastest responding instance
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import random
from statistics import mean

logger = logging.getLogger(__name__)

class LoadBalancerStrategy(str, Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    HEALTH_BASED = "health_based"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    RANDOM = "random"
    ADAPTIVE = "adaptive"

@dataclass
class FrameworkInstance:
    """Represents a framework instance for load balancing"""
    id: str
    name: str  # Framework name (copilot, agno, mastra, etc.)
    manager: Any  # Framework manager instance
    weight: float = 1.0
    max_concurrent: int = 10
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    last_used: Optional[datetime] = None
    is_healthy: bool = True
    health_check_failures: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if not self.response_times:
            return 0.0
        # Keep only recent response times (last 100 requests)
        recent_times = self.response_times[-100:]
        return mean(recent_times)
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)"""
        if self.max_concurrent == 0:
            return 1.0
        return self.current_connections / self.max_concurrent
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)"""
        if not self.is_healthy:
            return 0.0
        
        # Combine success rate, load factor, and response time
        success_score = self.success_rate
        load_score = 1.0 - self.load_factor
        
        # Response time score (lower is better)
        avg_response = self.average_response_time
        response_score = 1.0 / (1.0 + avg_response) if avg_response > 0 else 1.0
        
        # Weighted combination
        return (success_score * 0.4 + load_score * 0.3 + response_score * 0.3)
    
    async def acquire(self) -> bool:
        """Acquire a connection slot"""
        if self.current_connections >= self.max_concurrent:
            return False
        
        self.current_connections += 1
        self.last_used = datetime.now(timezone.utc)
        return True
    
    def release(self):
        """Release a connection slot"""
        if self.current_connections > 0:
            self.current_connections -= 1
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics"""
        self.total_requests += 1
        self.response_times.append(response_time)
        
        if not success:
            self.failed_requests += 1
        
        # Keep response times list manageable
        if len(self.response_times) > 200:
            self.response_times = self.response_times[-100:]
    
    def update_health(self, is_healthy: bool):
        """Update health status"""
        if is_healthy:
            self.is_healthy = True
            self.health_check_failures = 0
        else:
            self.health_check_failures += 1
            # Mark unhealthy after 3 consecutive failures
            if self.health_check_failures >= 3:
                self.is_healthy = False

class LoadBalancer:
    """Load balancer for framework instances"""
    
    def __init__(self):
        # Framework instances grouped by framework name
        self.instances: Dict[str, List[FrameworkInstance]] = defaultdict(list)
        
        # Round-robin counters for each framework
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        
        # Weighted round-robin state
        self.weighted_counters: Dict[str, Dict[str, int]] = defaultdict(dict)
        
        # Statistics tracking
        self.stats = {
            "total_requests": 0,
            "total_failures": 0,
            "framework_requests": defaultdict(int),
            "strategy_usage": defaultdict(int),
            "last_reset": datetime.now(timezone.utc)
        }
        
        # Health checking
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        
        logger.info("Load balancer initialized")
    
    def register_instance(self, framework: str, instance: FrameworkInstance):
        """Register a framework instance"""
        self.instances[framework].append(instance)
        logger.info(f"Registered instance {instance.id} for framework {framework}")
    
    def unregister_instance(self, framework: str, instance_id: str) -> bool:
        """Unregister a framework instance"""
        if framework in self.instances:
            original_count = len(self.instances[framework])
            self.instances[framework] = [
                inst for inst in self.instances[framework] 
                if inst.id != instance_id
            ]
            
            if len(self.instances[framework]) < original_count:
                logger.info(f"Unregistered instance {instance_id} from framework {framework}")
                return True
        
        return False
    
    async def select_instance(
        self, 
        framework: str, 
        strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN
    ) -> Optional[FrameworkInstance]:
        """Select best instance using specified strategy"""
        if framework not in self.instances or not self.instances[framework]:
            return None
        
        instances = self.instances[framework]
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        
        if not healthy_instances:
            # If no healthy instances, try any available instance
            available_instances = [
                inst for inst in instances 
                if inst.current_connections < inst.max_concurrent
            ]
            if not available_instances:
                return None
            healthy_instances = available_instances
        
        # Apply strategy
        selected = None
        if strategy == LoadBalancerStrategy.ROUND_ROBIN:
            selected = self._round_robin_select(framework, healthy_instances)
        elif strategy == LoadBalancerStrategy.WEIGHTED_ROUND_ROBIN:
            selected = self._weighted_round_robin_select(framework, healthy_instances)
        elif strategy == LoadBalancerStrategy.HEALTH_BASED:
            selected = self._health_based_select(healthy_instances)
        elif strategy == LoadBalancerStrategy.LEAST_CONNECTIONS:
            selected = self._least_connections_select(healthy_instances)
        elif strategy == LoadBalancerStrategy.RESPONSE_TIME:
            selected = self._response_time_select(healthy_instances)
        elif strategy == LoadBalancerStrategy.RANDOM:
            selected = self._random_select(healthy_instances)
        elif strategy == LoadBalancerStrategy.ADAPTIVE:
            selected = await self._adaptive_select(framework, healthy_instances)
        else:
            selected = self._round_robin_select(framework, healthy_instances)
        
        # Update statistics
        if selected:
            self.stats["total_requests"] += 1
            self.stats["framework_requests"][framework] += 1
            self.stats["strategy_usage"][strategy.value] += 1
        
        return selected
    
    def _round_robin_select(
        self, framework: str, instances: List[FrameworkInstance]
    ) -> Optional[FrameworkInstance]:
        """Round-robin selection"""
        if not instances:
            return None
        
        # Find available instances
        available = [
            inst for inst in instances 
            if inst.current_connections < inst.max_concurrent
        ]
        
        if not available:
            return None
        
        # Round-robin selection
        counter = self.round_robin_counters[framework]
        selected = available[counter % len(available)]
        self.round_robin_counters[framework] = (counter + 1) % len(available)
        
        return selected
    
    def _weighted_round_robin_select(
        self, framework: str, instances: List[FrameworkInstance]
    ) -> Optional[FrameworkInstance]:
        """Weighted round-robin selection"""
        if not instances:
            return None
        
        # Find available instances
        available = [
            inst for inst in instances 
            if inst.current_connections < inst.max_concurrent
        ]
        
        if not available:
            return None
        
        # Initialize weights if needed
        if framework not in self.weighted_counters:
            self.weighted_counters[framework] = {
                inst.id: int(inst.weight * 10) for inst in available
            }
        
        # Find instance with highest remaining weight
        weights = self.weighted_counters[framework]
        best_instance = None
        best_weight = -1
        
        for inst in available:
            current_weight = weights.get(inst.id, int(inst.weight * 10))
            if current_weight > best_weight:
                best_weight = current_weight
                best_instance = inst
        
        if best_instance:
            # Decrease weight
            weights[best_instance.id] = max(0, weights[best_instance.id] - 1)
            
            # Reset weights if all are zero
            if all(w == 0 for w in weights.values()):
                for inst in available:
                    weights[inst.id] = int(inst.weight * 10)
        
        return best_instance
    
    def _health_based_select(
        self, instances: List[FrameworkInstance]
    ) -> Optional[FrameworkInstance]:
        """Health-based selection (best health score)"""
        if not instances:
            return None
        
        # Find available instances
        available = [
            inst for inst in instances 
            if inst.current_connections < inst.max_concurrent
        ]
        
        if not available:
            return None
        
        # Select instance with highest health score
        return max(available, key=lambda inst: inst.health_score)
    
    def _least_connections_select(
        self, instances: List[FrameworkInstance]
    ) -> Optional[FrameworkInstance]:
        """Least connections selection"""
        if not instances:
            return None
        
        # Find available instances
        available = [
            inst for inst in instances 
            if inst.current_connections < inst.max_concurrent
        ]
        
        if not available:
            return None
        
        # Select instance with fewest connections
        return min(available, key=lambda inst: inst.current_connections)
    
    def _response_time_select(
        self, instances: List[FrameworkInstance]
    ) -> Optional[FrameworkInstance]:
        """Response time based selection (fastest)"""
        if not instances:
            return None
        
        # Find available instances
        available = [
            inst for inst in instances 
            if inst.current_connections < inst.max_concurrent
        ]
        
        if not available:
            return None
        
        # Select instance with best response time
        return min(available, key=lambda inst: inst.average_response_time or float('inf'))
    
    def _random_select(
        self, instances: List[FrameworkInstance]
    ) -> Optional[FrameworkInstance]:
        """Random selection"""
        if not instances:
            return None
        
        # Find available instances
        available = [
            inst for inst in instances 
            if inst.current_connections < inst.max_concurrent
        ]
        
        if not available:
            return None
        
        return random.choice(available)
    
    async def _adaptive_select(
        self, framework: str, instances: List[FrameworkInstance]
    ) -> Optional[FrameworkInstance]:
        """Adaptive selection based on current conditions"""
        if not instances:
            return None
        
        # Find available instances
        available = [
            inst for inst in instances 
            if inst.current_connections < inst.max_concurrent
        ]
        
        if not available:
            return None
        
        # Choose strategy based on current conditions
        total_load = sum(inst.load_factor for inst in available)
        avg_load = total_load / len(available)
        
        # High load - use least connections
        if avg_load > 0.7:
            return self._least_connections_select(available)
        
        # Check response time variance
        response_times = [inst.average_response_time for inst in available if inst.average_response_time > 0]
        if response_times and (max(response_times) - min(response_times)) > 1.0:
            # High variance in response times - use response time based
            return self._response_time_select(available)
        
        # Check health score variance
        health_scores = [inst.health_score for inst in available]
        if (max(health_scores) - min(health_scores)) > 0.3:
            # Significant health differences - use health based
            return self._health_based_select(available)
        
        # Default to weighted round robin for balanced load
        return self._weighted_round_robin_select(framework, available)
    
    async def health_check_instances(self):
        """Perform health checks on all instances"""
        for framework, instances in self.instances.items():
            for instance in instances:
                try:
                    # Check if manager has health check method
                    if hasattr(instance.manager, 'health_check'):
                        is_healthy = await instance.manager.health_check()
                        instance.update_health(is_healthy)
                    elif hasattr(instance.manager, 'get_status'):
                        status = instance.manager.get_status()
                        is_healthy = status and status.get("status") == "active"
                        instance.update_health(is_healthy)
                    else:
                        # Assume healthy if no health check method
                        instance.update_health(True)
                        
                except Exception as e:
                    logger.warning(f"Health check failed for {instance.id}: {e}")
                    instance.update_health(False)
    
    async def start_health_checking(self):
        """Start periodic health checking"""
        if self.health_check_task:
            return
        
        async def health_check_loop():
            while True:
                try:
                    await self.health_check_instances()
                    await asyncio.sleep(self.health_check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check loop error: {e}")
                    await asyncio.sleep(self.health_check_interval)
        
        self.health_check_task = asyncio.create_task(health_check_loop())
        logger.info("Started health checking task")
    
    async def stop_health_checking(self):
        """Stop periodic health checking"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
            logger.info("Stopped health checking task")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "total_requests": self.stats["total_requests"],
            "total_failures": self.stats["total_failures"],
            "framework_requests": dict(self.stats["framework_requests"]),
            "strategy_usage": dict(self.stats["strategy_usage"]),
            "last_reset": self.stats["last_reset"].isoformat(),
            "instances": {
                framework: [
                    {
                        "id": inst.id,
                        "is_healthy": inst.is_healthy,
                        "current_connections": inst.current_connections,
                        "max_concurrent": inst.max_concurrent,
                        "load_factor": inst.load_factor,
                        "success_rate": inst.success_rate,
                        "average_response_time": inst.average_response_time,
                        "health_score": inst.health_score,
                        "total_requests": inst.total_requests,
                        "failed_requests": inst.failed_requests
                    }
                    for inst in instances
                ]
                for framework, instances in self.instances.items()
            }
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_requests": 0,
            "total_failures": 0,
            "framework_requests": defaultdict(int),
            "strategy_usage": defaultdict(int),
            "last_reset": datetime.now(timezone.utc)
        }
        logger.info("Reset load balancer statistics")
    
    def get_instance_by_id(self, instance_id: str) -> Optional[FrameworkInstance]:
        """Get instance by ID"""
        for instances in self.instances.values():
            for instance in instances:
                if instance.id == instance_id:
                    return instance
        return None
    
    def get_framework_instances(self, framework: str) -> List[FrameworkInstance]:
        """Get all instances for a framework"""
        return self.instances.get(framework, [])
    
    def update_instance_weight(self, instance_id: str, weight: float) -> bool:
        """Update instance weight"""
        instance = self.get_instance_by_id(instance_id)
        if instance:
            instance.weight = weight
            logger.info(f"Updated weight for instance {instance_id} to {weight}")
            return True
        return False
    
    def update_instance_capacity(self, instance_id: str, max_concurrent: int) -> bool:
        """Update instance capacity"""
        instance = self.get_instance_by_id(instance_id)
        if instance:
            instance.max_concurrent = max_concurrent
            logger.info(f"Updated capacity for instance {instance_id} to {max_concurrent}")
            return True
        return False
    
    async def cleanup(self):
        """Clean up load balancer resources"""
        await self.stop_health_checking()
        self.instances.clear()
        self.round_robin_counters.clear()
        self.weighted_counters.clear()
        logger.info("Load balancer cleanup complete")

class LoadBalancerMiddleware:
    """Middleware for automatic load balancer integration"""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
    
    async def __call__(self, request, call_next):
        """Middleware processing"""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record success metrics
            processing_time = time.time() - start_time
            # Could integrate with request routing info here
            
            return response
            
        except Exception as e:
            # Record failure metrics
            processing_time = time.time() - start_time
            self.load_balancer.stats["total_failures"] += 1
            raise

# Utility functions for load balancer configuration

def create_default_load_balancer() -> LoadBalancer:
    """Create load balancer with default configuration"""
    lb = LoadBalancer()
    return lb

def create_framework_instance(
    framework_name: str,
    manager: Any,
    weight: float = 1.0,
    max_concurrent: int = 10,
    instance_suffix: str = "primary"
) -> FrameworkInstance:
    """Create a framework instance with standard configuration"""
    return FrameworkInstance(
        id=f"{framework_name}-{instance_suffix}",
        name=framework_name,
        manager=manager,
        weight=weight,
        max_concurrent=max_concurrent
    )

# Load balancer factory for different deployment scenarios

class LoadBalancerFactory:
    """Factory for creating load balancers for different scenarios"""
    
    @staticmethod
    def create_development_balancer() -> LoadBalancer:
        """Create load balancer for development environment"""
        lb = LoadBalancer()
        lb.health_check_interval = 60  # Less frequent health checks
        return lb
    
    @staticmethod
    def create_production_balancer() -> LoadBalancer:
        """Create load balancer for production environment"""
        lb = LoadBalancer()
        lb.health_check_interval = 15  # More frequent health checks
        return lb
    
    @staticmethod
    def create_high_availability_balancer() -> LoadBalancer:
        """Create load balancer for high availability setup"""
        lb = LoadBalancer()
        lb.health_check_interval = 5  # Very frequent health checks
        return lb
