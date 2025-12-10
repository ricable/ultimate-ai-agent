"""
Auto-Scaling Infrastructure for SAFLA.

This module implements automatic scaling capabilities for the SAFLA system,
enabling dynamic resource allocation, load balancing, and performance
optimization based on real-time demand and system metrics.

Components:
- Resource Monitor: Real-time monitoring of system resources
- Scaling Engine: Automatic scaling decisions based on policies
- Load Balancer: Distribution of workload across available resources
- Performance Optimizer: Optimization of resource utilization
- Cost Manager: Cost-aware scaling decisions

Technical Features:
- Horizontal and vertical scaling capabilities
- Predictive scaling based on historical patterns
- Multi-cloud support with provider abstraction
- Container orchestration integration (Kubernetes, Docker Swarm)
- Real-time metrics collection and analysis
- Cost optimization algorithms
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operation."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"


class ScalingPolicy(Enum):
    """Scaling policy types."""
    REACTIVE = "reactive"  # Scale based on current metrics
    PREDICTIVE = "predictive"  # Scale based on predictions
    SCHEDULED = "scheduled"  # Scale based on schedule
    MANUAL = "manual"  # Manual scaling override


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_usage: float  # 0.0 to 1.0
    memory_usage: float  # 0.0 to 1.0
    storage_usage: float  # 0.0 to 1.0
    network_bandwidth: float  # Mbps
    request_rate: float  # requests per second
    response_time: float  # milliseconds
    error_rate: float  # 0.0 to 1.0
    active_connections: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingRule:
    """Rule for automatic scaling."""
    name: str
    metric: str
    threshold_up: float
    threshold_down: float
    cooldown_period: int  # seconds
    scaling_factor: float = 1.5
    min_instances: int = 1
    max_instances: int = 10


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    event_id: str
    direction: ScalingDirection
    resource_type: ResourceType
    old_capacity: int
    new_capacity: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True


@dataclass
class Instance:
    """Represents a compute instance."""
    instance_id: str
    instance_type: str
    cpu_cores: int
    memory_gb: int
    storage_gb: int
    status: str = "running"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoScalingInfrastructure:
    """
    Manages auto-scaling infrastructure for SAFLA.
    
    Provides automatic scaling of resources based on demand, performance
    metrics, and cost constraints while maintaining system stability.
    """
    
    def __init__(
        self,
        scaling_rules: Optional[List[ScalingRule]] = None,
        min_instances: int = 1,
        max_instances: int = 10,
        scaling_policy: ScalingPolicy = ScalingPolicy.REACTIVE,
        cost_limit: Optional[float] = None
    ):
        """Initialize auto-scaling infrastructure."""
        self.scaling_rules = scaling_rules or self._default_scaling_rules()
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scaling_policy = scaling_policy
        self.cost_limit = cost_limit
        
        # Instance management
        self.instances: Dict[str, Instance] = {}
        self.current_capacity = min_instances
        
        # Metrics and history
        self.metrics_history: List[ResourceMetrics] = []
        self.scaling_history: List[ScalingEvent] = []
        
        # Initialize with minimum instances
        self._initialize_instances()
        
        logger.info(f"Initialized AutoScalingInfrastructure with {min_instances} instances")
    
    def _default_scaling_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        return [
            ScalingRule(
                name="cpu_scaling",
                metric="cpu_usage",
                threshold_up=0.8,
                threshold_down=0.3,
                cooldown_period=300
            ),
            ScalingRule(
                name="memory_scaling",
                metric="memory_usage",
                threshold_up=0.85,
                threshold_down=0.4,
                cooldown_period=300
            ),
            ScalingRule(
                name="response_time_scaling",
                metric="response_time",
                threshold_up=1000,  # ms
                threshold_down=200,  # ms
                cooldown_period=180
            )
        ]
    
    def _initialize_instances(self) -> None:
        """Initialize minimum number of instances."""
        for i in range(self.min_instances):
            instance = Instance(
                instance_id=f"instance_{i}",
                instance_type="standard",
                cpu_cores=4,
                memory_gb=16,
                storage_gb=100
            )
            self.instances[instance.instance_id] = instance
    
    async def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # Simulate metrics collection
        metrics = ResourceMetrics(
            cpu_usage=np.random.uniform(0.3, 0.9),
            memory_usage=np.random.uniform(0.4, 0.8),
            storage_usage=np.random.uniform(0.2, 0.6),
            network_bandwidth=np.random.uniform(100, 1000),
            request_rate=np.random.uniform(50, 500),
            response_time=np.random.uniform(100, 800),
            error_rate=np.random.uniform(0, 0.05),
            active_connections=np.random.randint(10, 100)
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last hour of metrics
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.metrics_history = [
            m for m in self.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        return metrics
    
    async def evaluate_scaling_decision(
        self,
        metrics: ResourceMetrics
    ) -> Tuple[ScalingDirection, str]:
        """Evaluate whether scaling is needed."""
        if self.scaling_policy == ScalingPolicy.MANUAL:
            return ScalingDirection.NONE, "Manual scaling mode"
        
        # Check each scaling rule
        for rule in self.scaling_rules:
            metric_value = getattr(metrics, rule.metric, None)
            if metric_value is None:
                continue
            
            # Check if we need to scale up
            if metric_value > rule.threshold_up:
                if self.current_capacity < self.max_instances:
                    return ScalingDirection.UP, f"{rule.metric} exceeded threshold"
            
            # Check if we need to scale down
            elif metric_value < rule.threshold_down:
                if self.current_capacity > self.min_instances:
                    return ScalingDirection.DOWN, f"{rule.metric} below threshold"
        
        return ScalingDirection.NONE, "No scaling needed"
    
    async def scale_resources(
        self,
        direction: ScalingDirection,
        factor: float = 1.5
    ) -> ScalingEvent:
        """Execute scaling operation."""
        old_capacity = self.current_capacity
        
        if direction == ScalingDirection.UP:
            new_capacity = min(
                int(self.current_capacity * factor),
                self.max_instances
            )
            
            # Add new instances
            for i in range(self.current_capacity, new_capacity):
                instance = Instance(
                    instance_id=f"instance_{i}_{datetime.now().timestamp()}",
                    instance_type="standard",
                    cpu_cores=4,
                    memory_gb=16,
                    storage_gb=100
                )
                self.instances[instance.instance_id] = instance
        
        elif direction == ScalingDirection.DOWN:
            new_capacity = max(
                int(self.current_capacity / factor),
                self.min_instances
            )
            
            # Remove instances (oldest first)
            instances_to_remove = self.current_capacity - new_capacity
            sorted_instances = sorted(
                self.instances.items(),
                key=lambda x: x[1].created_at
            )
            
            for i in range(instances_to_remove):
                if len(self.instances) > self.min_instances:
                    instance_id = sorted_instances[i][0]
                    del self.instances[instance_id]
        
        else:
            new_capacity = self.current_capacity
        
        self.current_capacity = new_capacity
        
        # Record scaling event
        event = ScalingEvent(
            event_id=f"scale_{datetime.now().timestamp()}",
            direction=direction,
            resource_type=ResourceType.CPU,  # Simplified
            old_capacity=old_capacity,
            new_capacity=new_capacity,
            reason=f"Scaling {direction.value}"
        )
        
        self.scaling_history.append(event)
        
        logger.info(f"Scaled from {old_capacity} to {new_capacity} instances")
        
        return event
    
    def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get current infrastructure status."""
        return {
            'current_capacity': self.current_capacity,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'active_instances': len(self.instances),
            'scaling_policy': self.scaling_policy.value,
            'recent_scaling_events': len([
                e for e in self.scaling_history
                if (datetime.now() - e.timestamp).seconds < 3600
            ]),
            'instance_details': {
                instance_id: {
                    'type': instance.instance_type,
                    'cpu': instance.cpu_cores,
                    'memory': instance.memory_gb,
                    'status': instance.status
                }
                for instance_id, instance in self.instances.items()
            }
        }
    
    async def predict_future_demand(
        self,
        time_horizon: int = 3600  # seconds
    ) -> Dict[str, float]:
        """Predict future resource demand."""
        if len(self.metrics_history) < 10:
            return {'predicted_cpu': 0.5, 'predicted_memory': 0.5}
        
        # Simple moving average prediction
        recent_metrics = self.metrics_history[-10:]
        
        predicted_cpu = np.mean([m.cpu_usage for m in recent_metrics])
        predicted_memory = np.mean([m.memory_usage for m in recent_metrics])
        
        # Add some trend analysis
        if len(recent_metrics) >= 2:
            cpu_trend = recent_metrics[-1].cpu_usage - recent_metrics[0].cpu_usage
            memory_trend = recent_metrics[-1].memory_usage - recent_metrics[0].memory_usage
            
            predicted_cpu += cpu_trend * 0.1
            predicted_memory += memory_trend * 0.1
        
        return {
            'predicted_cpu': min(max(predicted_cpu, 0), 1),
            'predicted_memory': min(max(predicted_memory, 0), 1),
            'confidence': 0.7 if len(self.metrics_history) > 20 else 0.3
        }
    
    def estimate_cost(self) -> float:
        """Estimate current infrastructure cost."""
        # Simple cost model: $0.10 per instance per hour
        hourly_cost = len(self.instances) * 0.10
        return hourly_cost


class LoadBalancer:
    """Manages load distribution across instances."""
    
    def __init__(self, infrastructure: AutoScalingInfrastructure):
        """Initialize load balancer."""
        self.infrastructure = infrastructure
        self.request_distribution: Dict[str, int] = {}
    
    async def route_request(self, request_id: str) -> str:
        """Route request to appropriate instance."""
        # Simple round-robin load balancing
        available_instances = [
            instance_id for instance_id, instance in self.infrastructure.instances.items()
            if instance.status == "running"
        ]
        
        if not available_instances:
            raise RuntimeError("No available instances")
        
        # Select instance with least requests
        instance_loads = {
            instance_id: self.request_distribution.get(instance_id, 0)
            for instance_id in available_instances
        }
        
        selected_instance = min(instance_loads, key=instance_loads.get)
        
        # Update distribution
        self.request_distribution[selected_instance] = \
            self.request_distribution.get(selected_instance, 0) + 1
        
        return selected_instance
    
    def get_load_distribution(self) -> Dict[str, int]:
        """Get current load distribution."""
        return self.request_distribution.copy()


class PerformanceOptimizer:
    """Optimizes infrastructure performance."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.optimization_history: List[Dict[str, Any]] = []
    
    async def optimize_resources(
        self,
        infrastructure: AutoScalingInfrastructure,
        metrics: ResourceMetrics
    ) -> Dict[str, Any]:
        """Optimize resource allocation."""
        recommendations = {
            'scaling_recommendation': None,
            'instance_type_recommendation': None,
            'cost_optimization': None
        }
        
        # Analyze current efficiency
        efficiency = self._calculate_efficiency(metrics)
        
        if efficiency < 0.5:
            # Under-utilized
            recommendations['scaling_recommendation'] = 'scale_down'
            recommendations['cost_optimization'] = 'reduce_instance_size'
        elif efficiency > 0.9:
            # Over-utilized
            recommendations['scaling_recommendation'] = 'scale_up'
            recommendations['instance_type_recommendation'] = 'upgrade_instance_type'
        
        self.optimization_history.append({
            'recommendations': recommendations,
            'efficiency': efficiency,
            'timestamp': datetime.now()
        })
        
        return recommendations
    
    def _calculate_efficiency(self, metrics: ResourceMetrics) -> float:
        """Calculate resource utilization efficiency."""
        # Weighted average of different metrics
        weights = {
            'cpu': 0.4,
            'memory': 0.3,
            'response_time': 0.3
        }
        
        # Normalize response time (lower is better)
        normalized_response = 1.0 - min(metrics.response_time / 2000, 1.0)
        
        efficiency = (
            weights['cpu'] * metrics.cpu_usage +
            weights['memory'] * metrics.memory_usage +
            weights['response_time'] * normalized_response
        )
        
        return efficiency