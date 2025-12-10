"""
Quantum-Classical Hybrid Resource Allocator
Intelligent resource allocation between quantum and classical computing resources.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor

# Import distributed processing
try:
    from ..distributed.ray_manager import ray_cluster_manager, submit_distributed_task
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Import quantum components
from .quantum_advantage import QuantumAdvantageDetector, ProblemType, AdvantageType
from .quantum_simulator import QuantumCircuitSimulator
from .quantum_ml import QuantumMLPipeline

# Import monitoring
from ..monitoring.metrics.performance import performance_monitor
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of computing resources"""
    QUANTUM_SIMULATOR = "quantum_simulator"
    CLASSICAL_CPU = "classical_cpu"
    CLASSICAL_GPU = "classical_gpu"
    DISTRIBUTED_RAY = "distributed_ray"
    HYBRID = "hybrid"

class WorkloadPriority(Enum):
    """Workload priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ResourceCapacity:
    """Resource capacity configuration"""
    quantum_qubits: int
    classical_cpu_cores: int
    classical_memory_gb: float
    distributed_workers: int
    max_concurrent_quantum: int
    max_concurrent_classical: int

@dataclass
class WorkloadRequest:
    """Request for computational resources"""
    workload_id: str
    problem_type: ProblemType
    estimated_complexity: int
    priority: WorkloadPriority
    quantum_requirements: Optional[Dict[str, Any]] = None
    classical_requirements: Optional[Dict[str, Any]] = None
    deadline: Optional[datetime] = None
    user_preference: Optional[ResourceType] = None

@dataclass
class AllocationDecision:
    """Resource allocation decision"""
    workload_id: str
    allocated_resource: ResourceType
    quantum_resources: Dict[str, Any]
    classical_resources: Dict[str, Any]
    estimated_execution_time: float
    expected_accuracy: float
    cost_factor: float
    reasoning: str
    confidence: float

@dataclass
class ResourceUtilization:
    """Current resource utilization"""
    quantum_utilization: float  # 0.0 to 1.0
    classical_utilization: float
    distributed_utilization: float
    queue_length: int
    pending_quantum: int
    pending_classical: int
    active_workloads: int

class QuantumClassicalResourceAllocator:
    """Intelligent allocator for quantum-classical hybrid computing resources"""
    
    def __init__(self, capacity: ResourceCapacity):
        self.capacity = capacity
        self.quantum_detector = QuantumAdvantageDetector(max_qubits=capacity.quantum_qubits)
        self.quantum_simulator = QuantumCircuitSimulator(max_qubits=capacity.quantum_qubits)
        self.ml_pipeline = QuantumMLPipeline()
        
        # Resource tracking
        self.active_workloads: Dict[str, WorkloadRequest] = {}
        self.resource_history: List[Dict[str, Any]] = []
        self.allocation_history: List[AllocationDecision] = []
        
        # Performance metrics
        self.allocation_stats = {
            'total_allocations': 0,
            'quantum_allocations': 0,
            'classical_allocations': 0,
            'hybrid_allocations': 0,
            'successful_predictions': 0,
            'average_accuracy': 0.0,
            'total_cost_savings': 0.0
        }
        
        # Resource pools
        self.quantum_pool = asyncio.Semaphore(capacity.max_concurrent_quantum)
        self.classical_pool = asyncio.Semaphore(capacity.max_concurrent_classical)
        
        # Executor for classical tasks
        self.classical_executor = ThreadPoolExecutor(max_workers=capacity.classical_cpu_cores)
        
        logger.info(f"Quantum-Classical Resource Allocator initialized with {capacity.quantum_qubits} qubits")
    
    async def allocate_resources(self, request: WorkloadRequest) -> AllocationDecision:
        """Allocate optimal resources for a workload request"""
        start_time = datetime.utcnow()
        
        try:
            # Analyze problem characteristics
            problem_analysis = await self._analyze_problem(request)
            
            # Get current resource utilization
            utilization = await self._get_resource_utilization()
            
            # Predict quantum advantage
            advantage_prediction = await self._predict_quantum_advantage(
                request.problem_type, request.estimated_complexity
            )
            
            # Consider resource availability and constraints
            allocation_options = await self._generate_allocation_options(
                request, problem_analysis, utilization, advantage_prediction
            )
            
            # Select optimal allocation
            optimal_allocation = await self._select_optimal_allocation(
                request, allocation_options
            )
            
            # Update tracking
            self.allocation_history.append(optimal_allocation)
            self.allocation_stats['total_allocations'] += 1
            
            if optimal_allocation.allocated_resource == ResourceType.QUANTUM_SIMULATOR:
                self.allocation_stats['quantum_allocations'] += 1
            elif optimal_allocation.allocated_resource == ResourceType.HYBRID:
                self.allocation_stats['hybrid_allocations'] += 1
            else:
                self.allocation_stats['classical_allocations'] += 1
            
            # Log allocation decision
            allocation_time = (datetime.utcnow() - start_time).total_seconds()
            
            uap_logger.log_event(
                LogLevel.INFO,
                f"Resource allocated: {optimal_allocation.allocated_resource.value}",
                EventType.AGENT,
                {
                    "workload_id": request.workload_id,
                    "problem_type": request.problem_type.value,
                    "allocated_resource": optimal_allocation.allocated_resource.value,
                    "allocation_time": allocation_time,
                    "confidence": optimal_allocation.confidence,
                    "expected_accuracy": optimal_allocation.expected_accuracy
                },
                "quantum_resource_allocator"
            )
            
            return optimal_allocation
            
        except Exception as e:
            logger.error(f"Resource allocation failed for {request.workload_id}: {e}")
            
            # Fallback to classical resources
            fallback_allocation = AllocationDecision(
                workload_id=request.workload_id,
                allocated_resource=ResourceType.CLASSICAL_CPU,
                quantum_resources={},
                classical_resources={
                    'cpu_cores': min(4, self.capacity.classical_cpu_cores),
                    'memory_gb': 4.0
                },
                estimated_execution_time=60.0,  # Conservative estimate
                expected_accuracy=0.8,
                cost_factor=1.0,
                reasoning=f"Fallback allocation due to error: {str(e)}",
                confidence=0.3
            )
            
            return fallback_allocation
    
    async def _analyze_problem(self, request: WorkloadRequest) -> Dict[str, Any]:
        """Analyze problem characteristics for allocation decision"""
        analysis = {
            'complexity_score': self._calculate_complexity_score(request),
            'quantum_suitability': self._assess_quantum_suitability(request),
            'parallelization_potential': self._assess_parallelization(request),
            'memory_requirements': self._estimate_memory_requirements(request),
            'time_sensitivity': self._assess_time_sensitivity(request)
        }
        
        return analysis
    
    def _calculate_complexity_score(self, request: WorkloadRequest) -> float:
        """Calculate computational complexity score"""
        base_complexity = request.estimated_complexity
        
        # Adjust based on problem type
        if request.problem_type == ProblemType.OPTIMIZATION:
            # Exponential complexity for combinatorial optimization
            complexity = base_complexity ** 1.5
        elif request.problem_type == ProblemType.MACHINE_LEARNING:
            # Polynomial complexity for ML
            complexity = base_complexity ** 1.2
        elif request.problem_type == ProblemType.SIMULATION:
            # High complexity for quantum simulation
            complexity = base_complexity ** 2.0
        else:
            complexity = base_complexity
        
        # Normalize to 0-1 scale
        return min(1.0, complexity / 10000.0)
    
    def _assess_quantum_suitability(self, request: WorkloadRequest) -> float:
        """Assess how suitable the problem is for quantum computing"""
        suitability_scores = {
            ProblemType.OPTIMIZATION: 0.9,  # Quantum advantage expected
            ProblemType.SIMULATION: 0.95,   # Natural fit for quantum
            ProblemType.MACHINE_LEARNING: 0.7,  # Potential advantage
            ProblemType.SEARCH: 0.8,        # Grover's algorithm
            ProblemType.SAMPLING: 0.85,     # Good for quantum sampling
            ProblemType.CRYPTOGRAPHY: 0.6   # Mixed results
        }
        
        base_score = suitability_scores.get(request.problem_type, 0.5)
        
        # Adjust based on problem size
        if request.estimated_complexity < 10:
            base_score *= 0.6  # Small problems may not show advantage
        elif request.estimated_complexity > 100:
            base_score *= 1.2  # Large problems more likely to benefit
        
        return min(1.0, base_score)
    
    def _assess_parallelization(self, request: WorkloadRequest) -> float:
        """Assess parallelization potential for distributed processing"""
        if request.problem_type in [ProblemType.MACHINE_LEARNING, ProblemType.OPTIMIZATION]:
            return 0.8  # High parallelization potential
        elif request.problem_type == ProblemType.SIMULATION:
            return 0.6  # Moderate parallelization
        else:
            return 0.4  # Lower parallelization potential
    
    def _estimate_memory_requirements(self, request: WorkloadRequest) -> float:
        """Estimate memory requirements in GB"""
        if request.quantum_requirements:
            qubits = request.quantum_requirements.get('qubits', 4)
            # Exponential memory for quantum simulation
            quantum_memory = (2 ** qubits) * 16 / (1024 ** 3)  # Complex numbers
            return max(1.0, quantum_memory)
        
        # Classical memory estimate
        complexity = request.estimated_complexity
        return max(1.0, complexity * 0.001)  # Linear scaling
    
    def _assess_time_sensitivity(self, request: WorkloadRequest) -> float:
        """Assess time sensitivity of the request"""
        if request.deadline:
            time_to_deadline = (request.deadline - datetime.utcnow()).total_seconds()
            if time_to_deadline < 300:  # 5 minutes
                return 0.9  # Very time sensitive
            elif time_to_deadline < 3600:  # 1 hour
                return 0.7  # Moderately time sensitive
            else:
                return 0.3  # Not time sensitive
        
        # Priority-based time sensitivity
        priority_sensitivity = {
            WorkloadPriority.CRITICAL: 0.9,
            WorkloadPriority.HIGH: 0.7,
            WorkloadPriority.MEDIUM: 0.5,
            WorkloadPriority.LOW: 0.3
        }
        
        return priority_sensitivity.get(request.priority, 0.5)
    
    async def _get_resource_utilization(self) -> ResourceUtilization:
        """Get current resource utilization"""
        # Get Ray cluster status if available
        ray_status = {}
        if RAY_AVAILABLE:
            try:
                ray_status = await ray_cluster_manager.get_cluster_status()
            except Exception as e:
                logger.warning(f"Could not get Ray status: {e}")
        
        # Calculate utilization metrics
        quantum_util = len([w for w in self.active_workloads.values() 
                          if w.quantum_requirements]) / self.capacity.max_concurrent_quantum
        
        classical_util = len([w for w in self.active_workloads.values() 
                            if not w.quantum_requirements]) / self.capacity.max_concurrent_classical
        
        distributed_util = ray_status.get('task_statistics', {}).get('running_tasks', 0) / 100
        
        return ResourceUtilization(
            quantum_utilization=min(1.0, quantum_util),
            classical_utilization=min(1.0, classical_util),
            distributed_utilization=min(1.0, distributed_util),
            queue_length=ray_status.get('task_statistics', {}).get('queue_length', 0),
            pending_quantum=len([w for w in self.active_workloads.values() 
                               if w.quantum_requirements and w.workload_id not in self.allocation_history]),
            pending_classical=len([w for w in self.active_workloads.values() 
                                 if not w.quantum_requirements and w.workload_id not in self.allocation_history]),
            active_workloads=len(self.active_workloads)
        )
    
    async def _predict_quantum_advantage(self, problem_type: ProblemType, 
                                       complexity: int) -> Dict[str, Any]:
        """Predict quantum advantage for the problem"""
        try:
            prediction = await self.quantum_detector.predict_advantage_for_problem(
                problem_type, complexity
            )
            return prediction
        except Exception as e:
            logger.warning(f"Quantum advantage prediction failed: {e}")
            return {
                'prediction': 'Unknown',
                'confidence': 0.5,
                'reasoning': 'Prediction failed, using heuristics',
                'recommendation': 'Consider both quantum and classical approaches'
            }
    
    async def _generate_allocation_options(self, request: WorkloadRequest,
                                         problem_analysis: Dict[str, Any],
                                         utilization: ResourceUtilization,
                                         advantage_prediction: Dict[str, Any]) -> List[AllocationDecision]:
        """Generate possible allocation options"""
        options = []
        
        # Option 1: Pure Quantum
        if (problem_analysis['quantum_suitability'] > 0.6 and 
            utilization.quantum_utilization < 0.8 and
            request.estimated_complexity <= self.capacity.quantum_qubits):
            
            quantum_option = AllocationDecision(
                workload_id=request.workload_id,
                allocated_resource=ResourceType.QUANTUM_SIMULATOR,
                quantum_resources={
                    'qubits': min(request.estimated_complexity, self.capacity.quantum_qubits),
                    'shots': 1000,
                    'error_mitigation': True,
                    'optimization_level': 2
                },
                classical_resources={'cpu_cores': 1, 'memory_gb': 2.0},
                estimated_execution_time=self._estimate_quantum_time(request, problem_analysis),
                expected_accuracy=0.95 * advantage_prediction.get('confidence', 0.7),
                cost_factor=2.0,  # Quantum resources are expensive
                reasoning="High quantum suitability with available quantum resources",
                confidence=advantage_prediction.get('confidence', 0.7)
            )
            options.append(quantum_option)
        
        # Option 2: Pure Classical
        if utilization.classical_utilization < 0.9:
            classical_cores = min(8, self.capacity.classical_cpu_cores)
            classical_memory = problem_analysis['memory_requirements']
            
            classical_option = AllocationDecision(
                workload_id=request.workload_id,
                allocated_resource=ResourceType.CLASSICAL_CPU,
                quantum_resources={},
                classical_resources={
                    'cpu_cores': classical_cores,
                    'memory_gb': classical_memory
                },
                estimated_execution_time=self._estimate_classical_time(request, problem_analysis),
                expected_accuracy=0.85,
                cost_factor=1.0,  # Baseline cost
                reasoning="Reliable classical approach with good availability",
                confidence=0.8
            )
            options.append(classical_option)
        
        # Option 3: Distributed Classical (Ray)
        if (RAY_AVAILABLE and 
            problem_analysis['parallelization_potential'] > 0.6 and
            utilization.distributed_utilization < 0.8):
            
            distributed_option = AllocationDecision(
                workload_id=request.workload_id,
                allocated_resource=ResourceType.DISTRIBUTED_RAY,
                quantum_resources={},
                classical_resources={
                    'distributed_workers': min(self.capacity.distributed_workers, 10),
                    'memory_per_worker': 4.0,
                    'parallelization_factor': problem_analysis['parallelization_potential']
                },
                estimated_execution_time=self._estimate_distributed_time(request, problem_analysis),
                expected_accuracy=0.9,
                cost_factor=1.5,  # Higher cost for distributed resources
                reasoning="High parallelization potential with distributed processing",
                confidence=0.75
            )
            options.append(distributed_option)
        
        # Option 4: Hybrid Quantum-Classical
        if (problem_analysis['quantum_suitability'] > 0.4 and 
            problem_analysis['complexity_score'] > 0.3 and
            utilization.quantum_utilization < 0.9):
            
            hybrid_option = AllocationDecision(
                workload_id=request.workload_id,
                allocated_resource=ResourceType.HYBRID,
                quantum_resources={
                    'qubits': min(request.estimated_complexity // 2, self.capacity.quantum_qubits),
                    'shots': 500,
                    'error_mitigation': True
                },
                classical_resources={
                    'cpu_cores': 4,
                    'memory_gb': problem_analysis['memory_requirements'] / 2
                },
                estimated_execution_time=self._estimate_hybrid_time(request, problem_analysis),
                expected_accuracy=0.88,
                cost_factor=1.7,
                reasoning="Hybrid approach balances quantum potential with classical reliability",
                confidence=0.7
            )
            options.append(hybrid_option)
        
        return options
    
    async def _select_optimal_allocation(self, request: WorkloadRequest,
                                       options: List[AllocationDecision]) -> AllocationDecision:
        """Select the optimal allocation from available options"""
        if not options:
            # Fallback option
            return AllocationDecision(
                workload_id=request.workload_id,
                allocated_resource=ResourceType.CLASSICAL_CPU,
                quantum_resources={},
                classical_resources={'cpu_cores': 2, 'memory_gb': 2.0},
                estimated_execution_time=120.0,
                expected_accuracy=0.7,
                cost_factor=1.0,
                reasoning="No optimal options available, using fallback",
                confidence=0.5
            )
        
        # Score each option based on multiple criteria
        scored_options = []
        for option in options:
            score = self._calculate_allocation_score(request, option)
            scored_options.append((score, option))
        
        # Sort by score and return the best option
        scored_options.sort(key=lambda x: x[0], reverse=True)
        best_score, best_option = scored_options[0]
        
        # Update reasoning with score
        best_option.reasoning += f" (Score: {best_score:.2f})"
        
        return best_option
    
    def _calculate_allocation_score(self, request: WorkloadRequest, 
                                  option: AllocationDecision) -> float:
        """Calculate allocation score based on multiple criteria"""
        # Base score from expected accuracy
        accuracy_score = option.expected_accuracy * 100
        
        # Time score (lower time is better)
        time_score = max(0, 100 - option.estimated_execution_time)
        
        # Cost score (lower cost is better)
        cost_score = max(0, 100 - option.cost_factor * 50)
        
        # Confidence score
        confidence_score = option.confidence * 100
        
        # Priority weighting
        priority_weights = {
            WorkloadPriority.CRITICAL: {'accuracy': 0.4, 'time': 0.4, 'cost': 0.1, 'confidence': 0.1},
            WorkloadPriority.HIGH: {'accuracy': 0.35, 'time': 0.35, 'cost': 0.15, 'confidence': 0.15},
            WorkloadPriority.MEDIUM: {'accuracy': 0.3, 'time': 0.3, 'cost': 0.2, 'confidence': 0.2},
            WorkloadPriority.LOW: {'accuracy': 0.25, 'time': 0.25, 'cost': 0.3, 'confidence': 0.2}
        }
        
        weights = priority_weights.get(request.priority, priority_weights[WorkloadPriority.MEDIUM])
        
        # Calculate weighted score
        total_score = (
            accuracy_score * weights['accuracy'] +
            time_score * weights['time'] +
            cost_score * weights['cost'] +
            confidence_score * weights['confidence']
        )
        
        # User preference bonus
        if request.user_preference and request.user_preference == option.allocated_resource:
            total_score *= 1.1  # 10% bonus for user preference
        
        return total_score
    
    def _estimate_quantum_time(self, request: WorkloadRequest, 
                             analysis: Dict[str, Any]) -> float:
        """Estimate quantum execution time"""
        base_time = request.estimated_complexity * 0.1  # Base time per complexity unit
        
        # Adjust for quantum overhead
        quantum_overhead = 1.5
        
        # Adjust for problem type
        problem_factors = {
            ProblemType.OPTIMIZATION: 2.0,
            ProblemType.MACHINE_LEARNING: 1.5,
            ProblemType.SIMULATION: 3.0,
            ProblemType.SEARCH: 1.2,
            ProblemType.SAMPLING: 1.0
        }
        
        factor = problem_factors.get(request.problem_type, 1.5)
        
        return base_time * quantum_overhead * factor
    
    def _estimate_classical_time(self, request: WorkloadRequest,
                               analysis: Dict[str, Any]) -> float:
        """Estimate classical execution time"""
        base_time = request.estimated_complexity * 0.2  # Slower than quantum for some problems
        
        # Adjust for problem type
        problem_factors = {
            ProblemType.OPTIMIZATION: 5.0,  # Exponential scaling
            ProblemType.MACHINE_LEARNING: 1.0,  # Good classical performance
            ProblemType.SIMULATION: 10.0,  # Poor classical scaling
            ProblemType.SEARCH: 2.0,  # Quadratic vs quantum square root
            ProblemType.SAMPLING: 1.5
        }
        
        factor = problem_factors.get(request.problem_type, 2.0)
        
        return base_time * factor
    
    def _estimate_distributed_time(self, request: WorkloadRequest,
                                 analysis: Dict[str, Any]) -> float:
        """Estimate distributed execution time"""
        classical_time = self._estimate_classical_time(request, analysis)
        
        # Apply parallelization speedup
        parallelization_factor = analysis['parallelization_potential']
        speedup = 1 + (parallelization_factor * 3)  # Max 4x speedup
        
        # Add distribution overhead
        distribution_overhead = 1.2
        
        return (classical_time / speedup) * distribution_overhead
    
    def _estimate_hybrid_time(self, request: WorkloadRequest,
                            analysis: Dict[str, Any]) -> float:
        """Estimate hybrid execution time"""
        quantum_time = self._estimate_quantum_time(request, analysis) * 0.6  # Reduced quantum portion
        classical_time = self._estimate_classical_time(request, analysis) * 0.4  # Reduced classical portion
        
        # Add coordination overhead
        coordination_overhead = 1.3
        
        return (quantum_time + classical_time) * coordination_overhead
    
    async def deallocate_resources(self, workload_id: str) -> Dict[str, Any]:
        """Deallocate resources for a completed workload"""
        if workload_id in self.active_workloads:
            workload = self.active_workloads.pop(workload_id)
            
            # Find corresponding allocation
            allocation = next(
                (a for a in self.allocation_history if a.workload_id == workload_id),
                None
            )
            
            # Update performance statistics
            if allocation:
                self.allocation_stats['successful_predictions'] += 1
                
                # Calculate actual vs predicted performance (simplified)
                actual_accuracy = 0.9  # Would be measured from actual results
                accuracy_diff = abs(actual_accuracy - allocation.expected_accuracy)
                
                # Update average accuracy
                total_predictions = self.allocation_stats['successful_predictions']
                current_avg = self.allocation_stats['average_accuracy']
                self.allocation_stats['average_accuracy'] = (
                    (current_avg * (total_predictions - 1) + (1 - accuracy_diff)) / total_predictions
                )
            
            logger.info(f"Resources deallocated for workload {workload_id}")
            
            return {
                'status': 'deallocated',
                'workload_id': workload_id,
                'resource_type': allocation.allocated_resource.value if allocation else 'unknown'
            }
        
        return {'status': 'not_found', 'workload_id': workload_id}
    
    async def get_allocation_report(self) -> Dict[str, Any]:
        """Get comprehensive allocation performance report"""
        utilization = await self._get_resource_utilization()
        
        # Calculate allocation efficiency
        recent_allocations = self.allocation_history[-100:]  # Last 100 allocations
        
        if recent_allocations:
            avg_confidence = np.mean([a.confidence for a in recent_allocations])
            avg_accuracy = np.mean([a.expected_accuracy for a in recent_allocations])
            avg_cost = np.mean([a.cost_factor for a in recent_allocations])
            
            resource_distribution = {}
            for allocation in recent_allocations:
                resource = allocation.allocated_resource.value
                resource_distribution[resource] = resource_distribution.get(resource, 0) + 1
        else:
            avg_confidence = 0.0
            avg_accuracy = 0.0
            avg_cost = 0.0
            resource_distribution = {}
        
        return {
            'allocation_statistics': self.allocation_stats,
            'current_utilization': {
                'quantum': utilization.quantum_utilization,
                'classical': utilization.classical_utilization,
                'distributed': utilization.distributed_utilization,
                'active_workloads': utilization.active_workloads
            },
            'recent_performance': {
                'average_confidence': float(avg_confidence),
                'average_expected_accuracy': float(avg_accuracy),
                'average_cost_factor': float(avg_cost),
                'resource_distribution': resource_distribution
            },
            'capacity_configuration': {
                'quantum_qubits': self.capacity.quantum_qubits,
                'classical_cpu_cores': self.capacity.classical_cpu_cores,
                'max_concurrent_quantum': self.capacity.max_concurrent_quantum,
                'max_concurrent_classical': self.capacity.max_concurrent_classical
            },
            'recommendations': self._generate_allocation_recommendations(utilization),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_allocation_recommendations(self, utilization: ResourceUtilization) -> List[str]:
        """Generate recommendations for resource allocation optimization"""
        recommendations = []
        
        if utilization.quantum_utilization > 0.9:
            recommendations.append("Consider adding more quantum resources or implementing queue management")
        
        if utilization.classical_utilization > 0.9:
            recommendations.append("Classical resources are highly utilized - consider scaling up")
        
        if utilization.distributed_utilization < 0.3:
            recommendations.append("Distributed resources are underutilized - consider promoting parallelizable workloads")
        
        if self.allocation_stats['average_accuracy'] < 0.7:
            recommendations.append("Allocation prediction accuracy is low - consider model retraining")
        
        quantum_ratio = self.allocation_stats['quantum_allocations'] / max(1, self.allocation_stats['total_allocations'])
        if quantum_ratio < 0.1:
            recommendations.append("Low quantum usage - verify quantum advantage detection is working properly")
        elif quantum_ratio > 0.8:
            recommendations.append("High quantum usage - ensure classical alternatives are being considered")
        
        if not recommendations:
            recommendations.append("Resource allocation is performing optimally")
        
        return recommendations

# Global resource allocator instance
resource_allocator: Optional[QuantumClassicalResourceAllocator] = None

def initialize_resource_allocator(capacity: ResourceCapacity) -> QuantumClassicalResourceAllocator:
    """Initialize the global resource allocator"""
    global resource_allocator
    resource_allocator = QuantumClassicalResourceAllocator(capacity)
    return resource_allocator

def get_resource_allocator() -> Optional[QuantumClassicalResourceAllocator]:
    """Get the global resource allocator instance"""
    return resource_allocator
