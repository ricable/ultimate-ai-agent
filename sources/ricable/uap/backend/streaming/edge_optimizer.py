# backend/streaming/edge_optimizer.py
"""
Edge-Cloud Hybrid Inference Optimizer
Optimizes workload distribution between edge and cloud for minimal latency and cost.
"""

import asyncio
import time
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
import math

# Import existing components
from ..distributed.ray_manager import ray_cluster_manager, submit_distributed_task
from ..edge.edge_manager import EdgeManager
from .stream_processor import StreamEvent, EventType, ProcessingPriority
from ..monitoring.logs.logger import get_logger
from ..monitoring.metrics.prometheus_metrics import metrics_collector

logger = get_logger(__name__)

class ExecutionLocation(Enum):
    """Execution location options"""
    EDGE = "edge"
    CLOUD = "cloud"
    HYBRID = "hybrid"
    AUTO = "auto"

class OptimizationObjective(Enum):
    """Optimization objectives"""
    LATENCY = "latency"  # Minimize latency
    COST = "cost"  # Minimize cost
    THROUGHPUT = "throughput"  # Maximize throughput
    ENERGY = "energy"  # Minimize energy consumption
    BALANCED = "balanced"  # Balance all factors

class WorkloadType(Enum):
    """Types of workloads"""
    COMPUTE_INTENSIVE = "compute_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    REAL_TIME = "real_time"
    BATCH = "batch"
    INTERACTIVE = "interactive"

@dataclass
class ExecutionEnvironment:
    """Execution environment characteristics"""
    location: ExecutionLocation
    cpu_cores: int
    memory_gb: float
    gpu_available: bool = False
    network_bandwidth_mbps: float = 100.0
    latency_to_cloud_ms: float = 50.0
    cost_per_hour: float = 0.0
    energy_efficiency: float = 1.0  # Relative efficiency
    availability: float = 1.0  # 0.0 to 1.0
    current_load: float = 0.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class WorkloadProfile:
    """Workload characteristics and requirements"""
    workload_id: str
    workload_type: WorkloadType
    cpu_requirement: float  # CPU cores needed
    memory_requirement_gb: float  # Memory needed in GB
    gpu_requirement: bool = False
    latency_requirement_ms: float = 1000.0  # Maximum acceptable latency
    throughput_requirement: float = 0.0  # Minimum required throughput
    data_size_mb: float = 1.0  # Input data size
    estimated_duration_ms: float = 100.0  # Estimated execution time
    deadline_ms: Optional[float] = None  # Hard deadline
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'workload_type': self.workload_type.value,
            'priority': self.priority.value
        }

@dataclass
class OptimizationResult:
    """Result of workload placement optimization"""
    workload_id: str
    recommended_location: ExecutionLocation
    confidence: float  # 0.0 to 1.0
    expected_latency_ms: float
    expected_cost: float
    expected_energy: float
    utilization_impact: float
    reasoning: str
    alternatives: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.alternatives is None:
            self.alternatives = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'recommended_location': self.recommended_location.value
        }

@dataclass
class ExecutionMetrics:
    """Metrics for executed workloads"""
    workload_id: str
    execution_location: ExecutionLocation
    actual_latency_ms: float
    actual_cost: float
    actual_energy: float
    success: bool
    error_message: Optional[str] = None
    started_at: float = 0.0
    completed_at: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'execution_location': self.execution_location.value
        }

class EdgeCloudOptimizer:
    """Edge-cloud hybrid inference optimizer with machine learning-based placement decisions"""
    
    def __init__(self, 
                 optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED,
                 learning_enabled: bool = True,
                 adaptation_rate: float = 0.1):
        
        self.optimization_objective = optimization_objective
        self.learning_enabled = learning_enabled
        self.adaptation_rate = adaptation_rate
        
        # Environment tracking
        self.environments: Dict[ExecutionLocation, ExecutionEnvironment] = {}
        self.workload_history: deque = deque(maxlen=1000)  # Historical workload data
        self.execution_history: deque = deque(maxlen=1000)  # Historical execution metrics
        
        # Performance models
        self.latency_models: Dict[ExecutionLocation, Dict[str, Any]] = defaultdict(dict)
        self.cost_models: Dict[ExecutionLocation, Dict[str, Any]] = defaultdict(dict)
        self.throughput_models: Dict[ExecutionLocation, Dict[str, Any]] = defaultdict(dict)
        
        # Current state tracking
        self.current_loads: Dict[ExecutionLocation, float] = defaultdict(float)
        self.pending_workloads: Dict[ExecutionLocation, int] = defaultdict(int)
        
        # Optimization weights (can be learned over time)
        self.objective_weights = {
            OptimizationObjective.LATENCY: {'latency': 1.0, 'cost': 0.0, 'energy': 0.0, 'throughput': 0.0},
            OptimizationObjective.COST: {'latency': 0.0, 'cost': 1.0, 'energy': 0.0, 'throughput': 0.0},
            OptimizationObjective.THROUGHPUT: {'latency': 0.0, 'cost': 0.0, 'energy': 0.0, 'throughput': 1.0},
            OptimizationObjective.ENERGY: {'latency': 0.0, 'cost': 0.0, 'energy': 1.0, 'throughput': 0.0},
            OptimizationObjective.BALANCED: {'latency': 0.4, 'cost': 0.2, 'energy': 0.2, 'throughput': 0.2}
        }
        
        # Metrics
        self.optimization_metrics = {
            'optimizations_performed': 0,
            'edge_placements': 0,
            'cloud_placements': 0,
            'hybrid_placements': 0,
            'average_optimization_time_ms': 0.0,
            'prediction_accuracy': 0.0,
            'total_cost_saved': 0.0,
            'total_latency_saved_ms': 0.0
        }
        
        # Initialize default environments
        self._initialize_default_environments()
        
        logger.info(f"EdgeCloudOptimizer initialized with objective: {optimization_objective.value}")
    
    def _initialize_default_environments(self) -> None:
        """Initialize default edge and cloud environments"""
        # Edge environment (typical edge device)
        self.environments[ExecutionLocation.EDGE] = ExecutionEnvironment(
            location=ExecutionLocation.EDGE,
            cpu_cores=4,
            memory_gb=8.0,
            gpu_available=False,
            network_bandwidth_mbps=50.0,
            latency_to_cloud_ms=0.0,  # Local processing
            cost_per_hour=0.05,  # Low cost for edge
            energy_efficiency=1.5,  # More energy efficient
            availability=0.95,
            current_load=0.0
        )
        
        # Cloud environment (typical cloud instance)
        self.environments[ExecutionLocation.CLOUD] = ExecutionEnvironment(
            location=ExecutionLocation.CLOUD,
            cpu_cores=16,
            memory_gb=64.0,
            gpu_available=True,
            network_bandwidth_mbps=1000.0,
            latency_to_cloud_ms=50.0,  # Network latency
            cost_per_hour=2.0,  # Higher cost for cloud
            energy_efficiency=1.0,  # Baseline efficiency
            availability=0.999,
            current_load=0.0
        )
    
    async def optimize_placement(self, workload: WorkloadProfile) -> OptimizationResult:
        """Optimize workload placement using multi-factor analysis"""
        start_time = time.perf_counter()
        
        try:
            # Evaluate each available location
            location_scores = {}
            location_details = {}
            
            for location, environment in self.environments.items():
                if environment.availability < 0.1:  # Skip unavailable environments
                    continue
                
                score, details = await self._evaluate_location(
                    workload, environment
                )
                location_scores[location] = score
                location_details[location] = details
            
            if not location_scores:
                raise Exception("No available execution environments")
            
            # Select best location
            best_location = max(location_scores.keys(), key=lambda loc: location_scores[loc])
            best_score = location_scores[best_location]
            best_details = location_details[best_location]
            
            # Calculate confidence based on score difference
            scores = list(location_scores.values())
            confidence = self._calculate_confidence(scores)
            
            # Generate alternatives
            alternatives = []
            for location, score in sorted(location_scores.items(), 
                                        key=lambda x: x[1], reverse=True)[1:3]:
                alternatives.append({
                    'location': location.value,
                    'score': score,
                    'details': location_details[location]
                })
            
            # Create optimization result
            result = OptimizationResult(
                workload_id=workload.workload_id,
                recommended_location=best_location,
                confidence=confidence,
                expected_latency_ms=best_details['expected_latency_ms'],
                expected_cost=best_details['expected_cost'],
                expected_energy=best_details['expected_energy'],
                utilization_impact=best_details['utilization_impact'],
                reasoning=self._generate_reasoning(workload, best_location, best_details),
                alternatives=alternatives,
                metadata={
                    'optimization_objective': self.optimization_objective.value,
                    'scores': {loc.value: score for loc, score in location_scores.items()},
                    'optimization_time_ms': (time.perf_counter() - start_time) * 1000
                }
            )
            
            # Update metrics
            self.optimization_metrics['optimizations_performed'] += 1
            optimization_time_ms = (time.perf_counter() - start_time) * 1000
            self.optimization_metrics['average_optimization_time_ms'] = (
                (self.optimization_metrics['average_optimization_time_ms'] * 
                 (self.optimization_metrics['optimizations_performed'] - 1) + optimization_time_ms) /
                self.optimization_metrics['optimizations_performed']
            )
            
            # Update placement counters
            if best_location == ExecutionLocation.EDGE:
                self.optimization_metrics['edge_placements'] += 1
            elif best_location == ExecutionLocation.CLOUD:
                self.optimization_metrics['cloud_placements'] += 1
            else:
                self.optimization_metrics['hybrid_placements'] += 1
            
            # Store workload for learning
            self.workload_history.append((workload, result))
            
            # Update Prometheus metrics
            metrics_collector.edge_optimization_decisions.labels(
                location=best_location.value,
                workload_type=workload.workload_type.value
            ).inc()
            
            metrics_collector.edge_optimization_latency.observe(optimization_time_ms / 1000.0)
            
            logger.info(
                f"Optimized placement for {workload.workload_id}: {best_location.value} "
                f"(confidence: {confidence:.3f}, latency: {best_details['expected_latency_ms']:.1f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing placement for {workload.workload_id}: {e}")
            
            # Return fallback result
            return OptimizationResult(
                workload_id=workload.workload_id,
                recommended_location=ExecutionLocation.CLOUD,  # Safe fallback
                confidence=0.5,
                expected_latency_ms=1000.0,
                expected_cost=1.0,
                expected_energy=1.0,
                utilization_impact=0.1,
                reasoning=f"Fallback due to optimization error: {str(e)}"
            )
    
    async def _evaluate_location(self, 
                               workload: WorkloadProfile, 
                               environment: ExecutionEnvironment) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a specific execution location for the workload"""
        
        # Check basic feasibility
        if not self._is_feasible(workload, environment):
            return 0.0, {'feasible': False, 'reason': 'Resource constraints not met'}
        
        # Predict performance metrics
        expected_latency_ms = await self._predict_latency(workload, environment)
        expected_cost = await self._predict_cost(workload, environment)
        expected_energy = await self._predict_energy(workload, environment)
        expected_throughput = await self._predict_throughput(workload, environment)
        
        # Calculate utilization impact
        utilization_impact = self._calculate_utilization_impact(workload, environment)
        
        # Normalize metrics (0.0 to 1.0, higher is better)
        normalized_latency = self._normalize_latency(expected_latency_ms, workload.latency_requirement_ms)
        normalized_cost = self._normalize_cost(expected_cost)
        normalized_energy = self._normalize_energy(expected_energy)
        normalized_throughput = self._normalize_throughput(expected_throughput, workload.throughput_requirement)
        
        # Calculate weighted score based on optimization objective
        weights = self.objective_weights[self.optimization_objective]
        score = (
            weights['latency'] * normalized_latency +
            weights['cost'] * normalized_cost +
            weights['energy'] * normalized_energy +
            weights['throughput'] * normalized_throughput
        )
        
        # Apply availability penalty
        score *= environment.availability
        
        # Apply load penalty
        load_penalty = 1.0 - (environment.current_load * 0.3)  # Max 30% penalty
        score *= load_penalty
        
        details = {
            'feasible': True,
            'expected_latency_ms': expected_latency_ms,
            'expected_cost': expected_cost,
            'expected_energy': expected_energy,
            'expected_throughput': expected_throughput,
            'utilization_impact': utilization_impact,
            'normalized_metrics': {
                'latency': normalized_latency,
                'cost': normalized_cost,
                'energy': normalized_energy,
                'throughput': normalized_throughput
            },
            'score_components': {
                'base_score': score / (environment.availability * load_penalty),
                'availability_factor': environment.availability,
                'load_penalty': load_penalty
            }
        }
        
        return score, details
    
    def _is_feasible(self, workload: WorkloadProfile, environment: ExecutionEnvironment) -> bool:
        """Check if workload can be executed in the environment"""
        # Check CPU requirement
        if workload.cpu_requirement > environment.cpu_cores:
            return False
        
        # Check memory requirement
        if workload.memory_requirement_gb > environment.memory_gb:
            return False
        
        # Check GPU requirement
        if workload.gpu_requirement and not environment.gpu_available:
            return False
        
        # Check if environment is available
        if environment.availability < 0.1:
            return False
        
        return True
    
    async def _predict_latency(self, workload: WorkloadProfile, environment: ExecutionEnvironment) -> float:
        """Predict execution latency for workload in environment"""
        # Base execution time prediction
        base_latency = workload.estimated_duration_ms
        
        # Adjust for CPU performance (assuming linear relationship)
        cpu_factor = environment.cpu_cores / max(workload.cpu_requirement, 1.0)
        execution_latency = base_latency / cpu_factor
        
        # Add network latency for cloud execution
        network_latency = 0.0
        if environment.location == ExecutionLocation.CLOUD:
            # Data transfer latency
            transfer_time_ms = (workload.data_size_mb / environment.network_bandwidth_mbps) * 8 * 1000
            network_latency = environment.latency_to_cloud_ms + transfer_time_ms
        
        # Add queueing delay based on current load
        queue_delay = environment.current_load * base_latency * 0.5
        
        total_latency = execution_latency + network_latency + queue_delay
        
        # Use historical data to refine prediction if available
        if self.learning_enabled:
            total_latency = await self._refine_latency_prediction(
                total_latency, workload, environment
            )
        
        return max(total_latency, 1.0)  # Minimum 1ms
    
    async def _predict_cost(self, workload: WorkloadProfile, environment: ExecutionEnvironment) -> float:
        """Predict execution cost for workload in environment"""
        # Base cost calculation
        execution_hours = workload.estimated_duration_ms / (1000 * 3600)
        base_cost = execution_hours * environment.cost_per_hour
        
        # Add data transfer cost for cloud execution
        transfer_cost = 0.0
        if environment.location == ExecutionLocation.CLOUD:
            # Assume $0.09 per GB for data transfer
            transfer_cost = (workload.data_size_mb / 1024) * 0.09
        
        total_cost = base_cost + transfer_cost
        
        return max(total_cost, 0.001)  # Minimum cost
    
    async def _predict_energy(self, workload: WorkloadProfile, environment: ExecutionEnvironment) -> float:
        """Predict energy consumption for workload in environment"""
        # Base energy calculation (simplified)
        base_energy = workload.estimated_duration_ms * workload.cpu_requirement * 0.1  # Arbitrary units
        
        # Adjust for environment efficiency
        total_energy = base_energy / environment.energy_efficiency
        
        # Add network energy for cloud execution
        if environment.location == ExecutionLocation.CLOUD:
            network_energy = workload.data_size_mb * 0.01  # Simplified network energy model
            total_energy += network_energy
        
        return max(total_energy, 0.1)  # Minimum energy
    
    async def _predict_throughput(self, workload: WorkloadProfile, environment: ExecutionEnvironment) -> float:
        """Predict throughput for workload in environment"""
        # Simple throughput calculation (requests per second)
        if workload.estimated_duration_ms > 0:
            base_throughput = 1000.0 / workload.estimated_duration_ms
        else:
            base_throughput = 1.0
        
        # Adjust for parallel processing capability
        parallel_factor = min(environment.cpu_cores / max(workload.cpu_requirement, 1.0), 4.0)
        total_throughput = base_throughput * parallel_factor
        
        # Apply load penalty
        load_factor = 1.0 - environment.current_load * 0.5
        total_throughput *= load_factor
        
        return max(total_throughput, 0.1)  # Minimum throughput
    
    def _calculate_utilization_impact(self, 
                                    workload: WorkloadProfile, 
                                    environment: ExecutionEnvironment) -> float:
        """Calculate impact on environment utilization"""
        cpu_impact = workload.cpu_requirement / environment.cpu_cores
        memory_impact = workload.memory_requirement_gb / environment.memory_gb
        
        return max(cpu_impact, memory_impact)
    
    def _normalize_latency(self, latency_ms: float, requirement_ms: float) -> float:
        """Normalize latency (higher is better)"""
        if latency_ms <= requirement_ms:
            return 1.0
        else:
            # Exponential penalty for exceeding requirement
            penalty = math.exp(-(latency_ms / requirement_ms - 1.0))
            return max(penalty, 0.1)
    
    def _normalize_cost(self, cost: float) -> float:
        """Normalize cost (higher is better, i.e., lower cost)"""
        # Use inverse relationship, assuming max cost of $10
        max_cost = 10.0
        return max((max_cost - cost) / max_cost, 0.1)
    
    def _normalize_energy(self, energy: float) -> float:
        """Normalize energy (higher is better, i.e., lower energy)"""
        # Use inverse relationship, assuming max energy of 1000 units
        max_energy = 1000.0
        return max((max_energy - energy) / max_energy, 0.1)
    
    def _normalize_throughput(self, throughput: float, requirement: float) -> float:
        """Normalize throughput (higher is better)"""
        if requirement > 0 and throughput >= requirement:
            return 1.0
        elif requirement > 0:
            return min(throughput / requirement, 1.0)
        else:
            # No specific requirement, use logarithmic scaling
            return min(math.log10(throughput + 1) / 2.0, 1.0)
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate confidence based on score distribution"""
        if len(scores) <= 1:
            return 1.0
        
        scores_sorted = sorted(scores, reverse=True)
        best_score = scores_sorted[0]
        second_best = scores_sorted[1] if len(scores_sorted) > 1 else 0.0
        
        if best_score == 0:
            return 0.5
        
        # Confidence based on relative difference
        confidence = min((best_score - second_best) / best_score * 2.0, 1.0)
        return max(confidence, 0.1)
    
    def _generate_reasoning(self, 
                          workload: WorkloadProfile, 
                          location: ExecutionLocation, 
                          details: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for placement decision"""
        reasons = []
        
        # Primary factors
        if self.optimization_objective == OptimizationObjective.LATENCY:
            reasons.append(f"Optimizing for latency: {details['expected_latency_ms']:.1f}ms expected")
        elif self.optimization_objective == OptimizationObjective.COST:
            reasons.append(f"Optimizing for cost: ${details['expected_cost']:.4f} expected")
        elif self.optimization_objective == OptimizationObjective.ENERGY:
            reasons.append(f"Optimizing for energy: {details['expected_energy']:.1f} units expected")
        
        # Location-specific reasoning
        if location == ExecutionLocation.EDGE:
            reasons.append("Edge execution chosen for low latency and reduced data transfer")
        elif location == ExecutionLocation.CLOUD:
            reasons.append("Cloud execution chosen for high compute capacity and availability")
        
        # Workload-specific reasoning
        if workload.workload_type == WorkloadType.REAL_TIME:
            reasons.append("Real-time workload requires minimal latency")
        elif workload.workload_type == WorkloadType.COMPUTE_INTENSIVE:
            reasons.append("Compute-intensive workload benefits from high-performance resources")
        
        return "; ".join(reasons)
    
    async def _refine_latency_prediction(self, 
                                       initial_prediction: float, 
                                       workload: WorkloadProfile, 
                                       environment: ExecutionEnvironment) -> float:
        """Refine latency prediction using historical data"""
        # Find similar historical workloads
        similar_executions = []
        for _, execution_metrics in self.execution_history:
            if (execution_metrics.execution_location == environment.location and
                abs(execution_metrics.workload_id.split('-')[0] == workload.workload_type.value)):
                similar_executions.append(execution_metrics.actual_latency_ms)
        
        if similar_executions:
            historical_avg = statistics.mean(similar_executions)
            # Weighted average of prediction and historical data
            refined_prediction = (
                (1 - self.adaptation_rate) * initial_prediction +
                self.adaptation_rate * historical_avg
            )
            return refined_prediction
        
        return initial_prediction
    
    async def record_execution_metrics(self, metrics: ExecutionMetrics) -> None:
        """Record actual execution metrics for learning"""
        self.execution_history.append((time.time(), metrics))
        
        # Update prediction accuracy if we have corresponding prediction
        for workload, optimization_result in reversed(self.workload_history):
            if workload.workload_id == metrics.workload_id:
                # Calculate prediction error
                latency_error = abs(optimization_result.expected_latency_ms - metrics.actual_latency_ms)
                cost_error = abs(optimization_result.expected_cost - metrics.actual_cost)
                
                # Update accuracy metrics
                accuracy = 1.0 - min(latency_error / optimization_result.expected_latency_ms, 1.0)
                self.optimization_metrics['prediction_accuracy'] = (
                    (self.optimization_metrics['prediction_accuracy'] * 
                     (self.optimization_metrics['optimizations_performed'] - 1) + accuracy) /
                    self.optimization_metrics['optimizations_performed']
                )
                
                # Calculate savings
                if metrics.success:
                    # Compare with alternative placement (simplified)
                    alternative_location = (
                        ExecutionLocation.CLOUD if optimization_result.recommended_location == ExecutionLocation.EDGE
                        else ExecutionLocation.EDGE
                    )
                    
                    # Estimate alternative metrics (simplified)
                    if alternative_location in self.environments:
                        alt_env = self.environments[alternative_location]
                        alt_cost = await self._predict_cost(workload, alt_env)
                        alt_latency = await self._predict_latency(workload, alt_env)
                        
                        cost_saved = max(alt_cost - metrics.actual_cost, 0.0)
                        latency_saved = max(alt_latency - metrics.actual_latency_ms, 0.0)
                        
                        self.optimization_metrics['total_cost_saved'] += cost_saved
                        self.optimization_metrics['total_latency_saved_ms'] += latency_saved
                
                break
        
        logger.debug(f"Recorded execution metrics for {metrics.workload_id}: "
                    f"latency={metrics.actual_latency_ms:.1f}ms, "
                    f"cost=${metrics.actual_cost:.4f}, "
                    f"success={metrics.success}")
    
    def update_environment(self, location: ExecutionLocation, updates: Dict[str, Any]) -> None:
        """Update environment characteristics"""
        if location in self.environments:
            env = self.environments[location]
            for key, value in updates.items():
                if hasattr(env, key):
                    setattr(env, key, value)
            
            logger.debug(f"Updated {location.value} environment: {updates}")
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics"""
        return {
            **self.optimization_metrics,
            'environments': {
                loc.value: env.to_dict() for loc, env in self.environments.items()
            },
            'current_loads': {
                loc.value: load for loc, load in self.current_loads.items()
            },
            'pending_workloads': {
                loc.value: count for loc, count in self.pending_workloads.items()
            },
            'history_size': {
                'workloads': len(self.workload_history),
                'executions': len(self.execution_history)
            }
        }
    
    async def optimize_stream_event(self, event: StreamEvent) -> OptimizationResult:
        """Optimize placement for a stream event"""
        # Convert stream event to workload profile
        workload = self._event_to_workload(event)
        
        # Optimize placement
        result = await self.optimize_placement(workload)
        
        # Link result to original event
        result.metadata['original_event_id'] = event.event_id
        result.metadata['event_type'] = event.event_type.value
        result.metadata['event_source'] = event.source
        
        return result
    
    def _event_to_workload(self, event: StreamEvent) -> WorkloadProfile:
        """Convert stream event to workload profile"""
        # Extract workload characteristics from event
        data_size = len(json.dumps(event.data)) / 1024 / 1024 if event.data else 0.001  # MB
        
        # Determine workload type based on event type and data
        workload_type = WorkloadType.INTERACTIVE
        if event.event_type == EventType.DATA:
            workload_type = WorkloadType.IO_INTENSIVE
        elif event.event_type == EventType.AGGREGATION:
            workload_type = WorkloadType.COMPUTE_INTENSIVE
        elif event.priority in [ProcessingPriority.CRITICAL, ProcessingPriority.HIGH]:
            workload_type = WorkloadType.REAL_TIME
        
        # Estimate requirements based on event characteristics
        cpu_requirement = 1.0  # Default
        memory_requirement = max(data_size * 2, 0.1)  # At least 100MB
        estimated_duration = 100.0  # Default 100ms
        latency_requirement = 1000.0  # Default 1s
        
        # Adjust based on priority
        if event.priority == ProcessingPriority.CRITICAL:
            latency_requirement = 100.0  # 100ms for critical
        elif event.priority == ProcessingPriority.HIGH:
            latency_requirement = 500.0  # 500ms for high
        
        return WorkloadProfile(
            workload_id=f"{event.event_id}-stream",
            workload_type=workload_type,
            cpu_requirement=cpu_requirement,
            memory_requirement_gb=memory_requirement,
            gpu_requirement=False,
            latency_requirement_ms=latency_requirement,
            data_size_mb=data_size,
            estimated_duration_ms=estimated_duration,
            priority=event.priority
        )

# Global optimizer instance
edge_optimizer = EdgeCloudOptimizer()

# Convenience functions
async def optimize_workload_placement(workload_id: str, 
                                     workload_type: str,
                                     cpu_requirement: float = 1.0,
                                     memory_requirement_gb: float = 1.0,
                                     latency_requirement_ms: float = 1000.0) -> OptimizationResult:
    """Optimize placement for a workload"""
    workload = WorkloadProfile(
        workload_id=workload_id,
        workload_type=WorkloadType(workload_type),
        cpu_requirement=cpu_requirement,
        memory_requirement_gb=memory_requirement_gb,
        latency_requirement_ms=latency_requirement_ms
    )
    
    return await edge_optimizer.optimize_placement(workload)

async def optimize_stream_placement(event: StreamEvent) -> OptimizationResult:
    """Optimize placement for a stream event"""
    return await edge_optimizer.optimize_stream_event(event)

async def record_workload_execution(workload_id: str,
                                  execution_location: str,
                                  actual_latency_ms: float,
                                  actual_cost: float,
                                  success: bool = True,
                                  error_message: Optional[str] = None) -> None:
    """Record actual execution metrics"""
    metrics = ExecutionMetrics(
        workload_id=workload_id,
        execution_location=ExecutionLocation(execution_location),
        actual_latency_ms=actual_latency_ms,
        actual_cost=actual_cost,
        actual_energy=actual_latency_ms * 0.1,  # Simplified energy calculation
        success=success,
        error_message=error_message,
        started_at=time.time() - actual_latency_ms / 1000,
        completed_at=time.time()
    )
    
    await edge_optimizer.record_execution_metrics(metrics)

def update_execution_environment(location: str, **updates) -> None:
    """Update execution environment characteristics"""
    edge_optimizer.update_environment(ExecutionLocation(location), updates)

def get_optimization_status() -> Dict[str, Any]:
    """Get current optimization status and metrics"""
    return edge_optimizer.get_optimization_metrics()
