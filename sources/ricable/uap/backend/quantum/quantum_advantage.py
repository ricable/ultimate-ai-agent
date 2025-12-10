"""
Quantum Advantage Detection and Resource Allocation
Detects when quantum algorithms provide advantage over classical counterparts.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import json

from .quantum_simulator import QuantumCircuitSimulator, QuantumGate, GateType
from .hybrid_algorithms import VariationalQuantumClassifier, QuantumApproximateOptimizationAlgorithm

logger = logging.getLogger(__name__)

class AdvantageType(Enum):
    """Types of quantum advantage"""
    COMPUTATIONAL = "computational"  # Faster computation
    STATISTICAL = "statistical"     # Better accuracy/quality
    MEMORY = "memory"               # Lower memory usage
    ENERGY = "energy"               # Lower energy consumption
    NONE = "none"                   # No advantage detected

class ProblemType(Enum):
    """Types of computational problems"""
    OPTIMIZATION = "optimization"
    MACHINE_LEARNING = "machine_learning"
    SIMULATION = "simulation"
    SEARCH = "search"
    SAMPLING = "sampling"
    CRYPTOGRAPHY = "cryptography"

@dataclass
class PerformanceMetric:
    """Performance metric for algorithm comparison"""
    algorithm_name: str
    algorithm_type: str  # "quantum" or "classical"
    execution_time: float
    accuracy: Optional[float]
    memory_usage: float
    energy_consumption: Optional[float]
    solution_quality: Optional[float]
    convergence_iterations: Optional[int]
    error_rate: Optional[float]
    scalability_factor: Optional[float]

@dataclass
class AdvantageAnalysis:
    """Analysis of quantum advantage for a specific problem"""
    problem_type: ProblemType
    problem_size: int
    quantum_metrics: PerformanceMetric
    classical_metrics: PerformanceMetric
    advantage_type: AdvantageType
    advantage_factor: float  # Quantum improvement ratio
    confidence_score: float  # 0-1, confidence in the advantage
    resource_recommendation: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ResourceAllocation:
    """Recommended resource allocation for quantum vs classical"""
    use_quantum: bool
    quantum_resources: Dict[str, Any]
    classical_resources: Dict[str, Any]
    hybrid_approach: bool
    reasoning: str
    expected_performance: Dict[str, float]
    cost_analysis: Dict[str, float]

class ClassicalBaselines:
    """Classical algorithm implementations for comparison"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def classical_optimization(self, problem_matrix: np.ndarray, 
                                   max_iterations: int = 1000) -> Dict[str, Any]:
        """Classical optimization using simulated annealing"""
        start_time = time.time()
        
        # Simulated annealing for Max-Cut
        n = problem_matrix.shape[0]
        current_solution = np.random.randint(0, 2, n)
        current_cost = self._calculate_max_cut_cost(current_solution, problem_matrix)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        temperature = 1.0
        cooling_rate = 0.999
        
        cost_history = []
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor = current_solution.copy()
            flip_idx = np.random.randint(0, n)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            
            neighbor_cost = self._calculate_max_cut_cost(neighbor, problem_matrix)
            cost_history.append(neighbor_cost)
            
            # Accept or reject
            if neighbor_cost > current_cost or np.random.random() < np.exp((neighbor_cost - current_cost) / temperature):
                current_solution = neighbor
                current_cost = neighbor_cost
                
                if current_cost > best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            
            temperature *= cooling_rate
        
        execution_time = time.time() - start_time
        
        return {
            'solution': best_solution.tolist(),
            'cost': best_cost,
            'execution_time': execution_time,
            'cost_history': cost_history,
            'algorithm': 'simulated_annealing',
            'iterations': max_iterations
        }
    
    def _calculate_max_cut_cost(self, solution: np.ndarray, adjacency_matrix: np.ndarray) -> float:
        """Calculate Max-Cut cost for classical solution"""
        cost = 0.0
        n = len(solution)
        
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i, j] != 0 and solution[i] != solution[j]:
                    cost += adjacency_matrix[i, j]
        
        return cost
    
    async def classical_classification(self, X: np.ndarray, y: np.ndarray, 
                                     test_X: np.ndarray, test_y: np.ndarray) -> Dict[str, Any]:
        """Classical SVM classification for comparison"""
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        
        start_time = time.time()
        
        # Train SVM
        classifier = SVC(kernel='rbf', probability=True)
        classifier.fit(X, y)
        
        # Make predictions
        predictions = classifier.predict(test_X)
        probabilities = classifier.predict_proba(test_X)
        
        execution_time = time.time() - start_time
        accuracy = accuracy_score(test_y, predictions)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'accuracy': accuracy,
            'execution_time': execution_time,
            'algorithm': 'svm_rbf',
            'support_vectors': len(classifier.support_)
        }
    
    async def classical_sampling(self, distribution_params: Dict[str, Any], 
                               num_samples: int = 1000) -> Dict[str, Any]:
        """Classical sampling from complex distributions"""
        start_time = time.time()
        
        # Monte Carlo sampling
        samples = []
        for _ in range(num_samples):
            # Simple Gaussian sampling as example
            sample = np.random.normal(0, 1, distribution_params.get('dimension', 2))
            samples.append(sample)
        
        execution_time = time.time() - start_time
        
        return {
            'samples': samples,
            'execution_time': execution_time,
            'algorithm': 'monte_carlo',
            'num_samples': num_samples
        }

class QuantumAdvantageDetector:
    """Detects and analyzes quantum advantage across different problem types"""
    
    def __init__(self, max_qubits: int = 12):
        self.max_qubits = max_qubits
        self.quantum_simulator = QuantumCircuitSimulator(max_qubits)
        self.classical_baselines = ClassicalBaselines()
        self.advantage_history: List[AdvantageAnalysis] = []
        
        # Thresholds for advantage detection
        self.advantage_thresholds = {
            AdvantageType.COMPUTATIONAL: 1.5,  # 50% speedup
            AdvantageType.STATISTICAL: 1.1,   # 10% better accuracy
            AdvantageType.MEMORY: 1.2,        # 20% less memory
            AdvantageType.ENERGY: 1.3         # 30% less energy
        }
    
    async def analyze_optimization_advantage(self, adjacency_matrix: np.ndarray, 
                                           max_iterations: int = 100) -> AdvantageAnalysis:
        """Analyze quantum advantage for optimization problems"""
        problem_size = adjacency_matrix.shape[0]
        
        if problem_size > self.max_qubits:
            logger.warning(f"Problem size {problem_size} exceeds max qubits {self.max_qubits}")
            problem_size = self.max_qubits
            adjacency_matrix = adjacency_matrix[:problem_size, :problem_size]
        
        # Run quantum algorithm (QAOA)
        qaoa = QuantumApproximateOptimizationAlgorithm(problem_size, num_layers=2)
        quantum_start = time.time()
        quantum_result = await qaoa.solve_max_cut(adjacency_matrix, max_iterations)
        quantum_time = time.time() - quantum_start
        
        # Run classical algorithm
        classical_result = await self.classical_baselines.classical_optimization(
            adjacency_matrix, max_iterations
        )
        
        # Create performance metrics
        quantum_metrics = PerformanceMetric(
            algorithm_name="QAOA",
            algorithm_type="quantum",
            execution_time=quantum_time,
            accuracy=None,
            memory_usage=self._estimate_quantum_memory(problem_size),
            energy_consumption=self._estimate_quantum_energy(problem_size, quantum_time),
            solution_quality=quantum_result['cost'],
            convergence_iterations=len(quantum_result['cost_history']),
            error_rate=0.01,  # Estimated quantum error rate
            scalability_factor=self._estimate_quantum_scalability(problem_size)
        )
        
        classical_metrics = PerformanceMetric(
            algorithm_name="Simulated Annealing",
            algorithm_type="classical",
            execution_time=classical_result['execution_time'],
            accuracy=None,
            memory_usage=self._estimate_classical_memory(problem_size),
            energy_consumption=self._estimate_classical_energy(problem_size, classical_result['execution_time']),
            solution_quality=classical_result['cost'],
            convergence_iterations=max_iterations,
            error_rate=0.0,
            scalability_factor=self._estimate_classical_scalability(problem_size)
        )
        
        # Determine advantage
        advantage_analysis = self._analyze_advantage(
            ProblemType.OPTIMIZATION, problem_size, quantum_metrics, classical_metrics
        )
        
        self.advantage_history.append(advantage_analysis)
        return advantage_analysis
    
    async def analyze_ml_advantage(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray) -> AdvantageAnalysis:
        """Analyze quantum advantage for machine learning problems"""
        num_features = X_train.shape[1]
        num_qubits = min(num_features + 2, self.max_qubits)
        
        # Run quantum algorithm (VQC)
        vqc = VariationalQuantumClassifier(num_qubits=num_qubits, max_iterations=50)
        quantum_start = time.time()
        await vqc.fit(X_train, y_train)
        quantum_predictions = await vqc.predict(X_test)
        quantum_time = time.time() - quantum_start
        
        from sklearn.metrics import accuracy_score
        quantum_accuracy = accuracy_score(y_test, quantum_predictions)
        
        # Run classical algorithm
        classical_result = await self.classical_baselines.classical_classification(
            X_train, y_train, X_test, y_test
        )
        
        # Create performance metrics
        quantum_metrics = PerformanceMetric(
            algorithm_name="VQC",
            algorithm_type="quantum",
            execution_time=quantum_time,
            accuracy=quantum_accuracy,
            memory_usage=self._estimate_quantum_memory(num_qubits),
            energy_consumption=self._estimate_quantum_energy(num_qubits, quantum_time),
            solution_quality=quantum_accuracy,
            convergence_iterations=50,
            error_rate=0.02,
            scalability_factor=self._estimate_quantum_scalability(num_qubits)
        )
        
        classical_metrics = PerformanceMetric(
            algorithm_name="SVM",
            algorithm_type="classical",
            execution_time=classical_result['execution_time'],
            accuracy=classical_result['accuracy'],
            memory_usage=self._estimate_classical_memory(len(X_train)),
            energy_consumption=self._estimate_classical_energy(len(X_train), classical_result['execution_time']),
            solution_quality=classical_result['accuracy'],
            convergence_iterations=None,
            error_rate=0.0,
            scalability_factor=self._estimate_classical_scalability(len(X_train))
        )
        
        # Analyze advantage
        advantage_analysis = self._analyze_advantage(
            ProblemType.MACHINE_LEARNING, len(X_train), quantum_metrics, classical_metrics
        )
        
        self.advantage_history.append(advantage_analysis)
        return advantage_analysis
    
    async def analyze_sampling_advantage(self, distribution_params: Dict[str, Any], 
                                       num_samples: int = 1000) -> AdvantageAnalysis:
        """Analyze quantum advantage for sampling problems"""
        num_qubits = min(distribution_params.get('dimension', 4), self.max_qubits)
        
        # Quantum sampling (simplified - using random quantum states)
        quantum_start = time.time()
        quantum_samples = []
        
        for _ in range(num_samples // 100):  # Batch sampling for efficiency
            # Create random quantum circuit
            gates = []
            for i in range(num_qubits):
                gates.append(QuantumGate(GateType.HADAMARD, [i]))
                gates.append(QuantumGate(GateType.ROTATION_Y, [i], [np.random.uniform(0, 2*np.pi)]))
            
            # Add entangling gates
            for i in range(num_qubits - 1):
                gates.append(QuantumGate(GateType.CNOT, [i, i + 1]))
            
            result = await self.quantum_simulator.simulate_circuit(gates, num_qubits, shots=100)
            
            # Extract samples from measurement results
            for measurement_set in [result.measurements[i:i+num_qubits] for i in range(0, len(result.measurements), num_qubits)][:100]:
                sample = [m.outcome for m in measurement_set]
                quantum_samples.append(sample)
        
        quantum_time = time.time() - quantum_start
        
        # Classical sampling
        classical_result = await self.classical_baselines.classical_sampling(
            distribution_params, num_samples
        )
        
        # Analyze sample quality (simplified)
        quantum_quality = self._calculate_sample_quality(quantum_samples)
        classical_quality = self._calculate_sample_quality(classical_result['samples'])
        
        # Create performance metrics
        quantum_metrics = PerformanceMetric(
            algorithm_name="Quantum Sampling",
            algorithm_type="quantum",
            execution_time=quantum_time,
            accuracy=None,
            memory_usage=self._estimate_quantum_memory(num_qubits),
            energy_consumption=self._estimate_quantum_energy(num_qubits, quantum_time),
            solution_quality=quantum_quality,
            convergence_iterations=None,
            error_rate=0.03,
            scalability_factor=self._estimate_quantum_scalability(num_qubits)
        )
        
        classical_metrics = PerformanceMetric(
            algorithm_name="Monte Carlo",
            algorithm_type="classical",
            execution_time=classical_result['execution_time'],
            accuracy=None,
            memory_usage=self._estimate_classical_memory(num_samples),
            energy_consumption=self._estimate_classical_energy(num_samples, classical_result['execution_time']),
            solution_quality=classical_quality,
            convergence_iterations=None,
            error_rate=0.0,
            scalability_factor=self._estimate_classical_scalability(num_samples)
        )
        
        # Analyze advantage
        advantage_analysis = self._analyze_advantage(
            ProblemType.SAMPLING, num_samples, quantum_metrics, classical_metrics
        )
        
        self.advantage_history.append(advantage_analysis)
        return advantage_analysis
    
    def _analyze_advantage(self, problem_type: ProblemType, problem_size: int,
                          quantum_metrics: PerformanceMetric, 
                          classical_metrics: PerformanceMetric) -> AdvantageAnalysis:
        """Analyze quantum advantage based on performance metrics"""
        
        # Calculate advantage factors
        time_advantage = classical_metrics.execution_time / quantum_metrics.execution_time if quantum_metrics.execution_time > 0 else 1.0
        memory_advantage = classical_metrics.memory_usage / quantum_metrics.memory_usage if quantum_metrics.memory_usage > 0 else 1.0
        
        # Quality advantage (higher is better)
        if quantum_metrics.solution_quality and classical_metrics.solution_quality:
            if quantum_metrics.accuracy is not None:  # For ML problems
                quality_advantage = quantum_metrics.solution_quality / classical_metrics.solution_quality
            else:  # For optimization problems
                quality_advantage = quantum_metrics.solution_quality / classical_metrics.solution_quality
        else:
            quality_advantage = 1.0
        
        # Energy advantage
        energy_advantage = 1.0
        if (quantum_metrics.energy_consumption and classical_metrics.energy_consumption and 
            quantum_metrics.energy_consumption > 0):
            energy_advantage = classical_metrics.energy_consumption / quantum_metrics.energy_consumption
        
        # Determine primary advantage type
        advantage_type = AdvantageType.NONE
        advantage_factor = 1.0
        confidence_score = 0.0
        
        if time_advantage >= self.advantage_thresholds[AdvantageType.COMPUTATIONAL]:
            advantage_type = AdvantageType.COMPUTATIONAL
            advantage_factor = time_advantage
            confidence_score = min(0.9, (time_advantage - 1.0) / 2.0)
        
        elif quality_advantage >= self.advantage_thresholds[AdvantageType.STATISTICAL]:
            advantage_type = AdvantageType.STATISTICAL
            advantage_factor = quality_advantage
            confidence_score = min(0.9, (quality_advantage - 1.0) / 0.5)
        
        elif memory_advantage >= self.advantage_thresholds[AdvantageType.MEMORY]:
            advantage_type = AdvantageType.MEMORY
            advantage_factor = memory_advantage
            confidence_score = min(0.9, (memory_advantage - 1.0) / 1.0)
        
        elif energy_advantage >= self.advantage_thresholds[AdvantageType.ENERGY]:
            advantage_type = AdvantageType.ENERGY
            advantage_factor = energy_advantage
            confidence_score = min(0.9, (energy_advantage - 1.0) / 1.5)
        
        # Adjust confidence based on problem size and quantum error rates
        if quantum_metrics.error_rate:
            confidence_score *= (1.0 - quantum_metrics.error_rate)
        
        # Scale confidence with problem size (larger problems more reliable)
        size_factor = min(1.0, problem_size / 10.0)
        confidence_score *= (0.5 + 0.5 * size_factor)
        
        # Resource recommendation
        resource_recommendation = self._generate_resource_recommendation(
            advantage_type, advantage_factor, confidence_score, problem_size
        )
        
        return AdvantageAnalysis(
            problem_type=problem_type,
            problem_size=problem_size,
            quantum_metrics=quantum_metrics,
            classical_metrics=classical_metrics,
            advantage_type=advantage_type,
            advantage_factor=advantage_factor,
            confidence_score=confidence_score,
            resource_recommendation=resource_recommendation,
            timestamp=datetime.utcnow(),
            metadata={
                'time_advantage': time_advantage,
                'quality_advantage': quality_advantage,
                'memory_advantage': memory_advantage,
                'energy_advantage': energy_advantage,
                'quantum_error_rate': quantum_metrics.error_rate,
                'classical_error_rate': classical_metrics.error_rate
            }
        )
    
    def _generate_resource_recommendation(self, advantage_type: AdvantageType, 
                                        advantage_factor: float, confidence_score: float,
                                        problem_size: int) -> Dict[str, Any]:
        """Generate resource allocation recommendations"""
        
        if advantage_type == AdvantageType.NONE or confidence_score < 0.3:
            return {
                'allocation': ResourceAllocation(
                    use_quantum=False,
                    quantum_resources={},
                    classical_resources={'cpu_cores': min(8, problem_size), 'memory_gb': 4},
                    hybrid_approach=False,
                    reasoning="No significant quantum advantage detected",
                    expected_performance={'speedup': 1.0, 'accuracy': 0.0},
                    cost_analysis={'quantum_cost': 0.0, 'classical_cost': 1.0}
                )
            }
        
        elif confidence_score > 0.7 and advantage_factor > 2.0:
            return {
                'allocation': ResourceAllocation(
                    use_quantum=True,
                    quantum_resources={
                        'qubits': min(problem_size, self.max_qubits),
                        'circuit_depth': problem_size * 5,
                        'shots': 10000,
                        'error_mitigation': True
                    },
                    classical_resources={'cpu_cores': 2, 'memory_gb': 2},
                    hybrid_approach=False,
                    reasoning=f"Strong quantum {advantage_type.value} advantage detected",
                    expected_performance={
                        'speedup': advantage_factor,
                        'accuracy': confidence_score
                    },
                    cost_analysis={
                        'quantum_cost': 3.0,
                        'classical_cost': 1.0,
                        'cost_effectiveness': advantage_factor / 3.0
                    }
                )
            }
        
        else:
            return {
                'allocation': ResourceAllocation(
                    use_quantum=True,
                    quantum_resources={
                        'qubits': min(problem_size, self.max_qubits // 2),
                        'circuit_depth': problem_size * 3,
                        'shots': 5000,
                        'error_mitigation': True
                    },
                    classical_resources={'cpu_cores': 4, 'memory_gb': 4},
                    hybrid_approach=True,
                    reasoning=f"Moderate quantum {advantage_type.value} advantage - hybrid approach recommended",
                    expected_performance={
                        'speedup': advantage_factor * 0.8,
                        'accuracy': confidence_score
                    },
                    cost_analysis={
                        'quantum_cost': 2.0,
                        'classical_cost': 1.5,
                        'cost_effectiveness': (advantage_factor * 0.8) / 2.0
                    }
                )
            }
    
    def _estimate_quantum_memory(self, num_qubits: int) -> float:
        """Estimate quantum memory usage (in MB)"""
        # Exponential scaling for state vector simulation
        return (2 ** num_qubits) * 16 / (1024 * 1024)  # Complex numbers, 16 bytes each
    
    def _estimate_classical_memory(self, problem_size: int) -> float:
        """Estimate classical memory usage (in MB)"""
        # Linear scaling for most classical algorithms
        return problem_size * 0.001  # 1KB per problem unit
    
    def _estimate_quantum_energy(self, num_qubits: int, execution_time: float) -> float:
        """Estimate quantum energy consumption (in Joules)"""
        # Approximate energy model for quantum devices
        base_power = 1000  # 1kW for quantum device
        qubit_power = 10   # 10W per qubit
        total_power = base_power + (qubit_power * num_qubits)
        return total_power * execution_time
    
    def _estimate_classical_energy(self, problem_size: int, execution_time: float) -> float:
        """Estimate classical energy consumption (in Joules)"""
        # Approximate energy model for classical computing
        base_power = 100  # 100W for CPU
        scale_power = max(1, problem_size / 1000) * 50  # Scale with problem size
        total_power = base_power + scale_power
        return total_power * execution_time
    
    def _estimate_quantum_scalability(self, num_qubits: int) -> float:
        """Estimate quantum algorithm scalability factor"""
        # Exponential advantage potential, but limited by current hardware
        theoretical_advantage = 2 ** (num_qubits / 4)  # Exponential potential
        hardware_limitation = 1.0 / (1.0 + num_qubits * 0.1)  # Current hardware limits
        return theoretical_advantage * hardware_limitation
    
    def _estimate_classical_scalability(self, problem_size: int) -> float:
        """Estimate classical algorithm scalability factor"""
        # Polynomial scaling for most classical algorithms
        return problem_size ** 1.5
    
    def _calculate_sample_quality(self, samples: List[List]) -> float:
        """Calculate quality metric for sampling results"""
        if not samples:
            return 0.0
        
        # Simple quality metric: entropy of samples
        flat_samples = [item for sublist in samples for item in (sublist if isinstance(sublist, list) else [sublist])]
        unique_samples = len(set(map(str, flat_samples)))
        total_samples = len(flat_samples)
        
        if total_samples == 0:
            return 0.0
        
        # Normalized entropy-like measure
        quality = unique_samples / total_samples
        return min(1.0, quality * 2.0)  # Scale to [0, 1]
    
    async def get_advantage_summary(self) -> Dict[str, Any]:
        """Get summary of quantum advantage analyses"""
        if not self.advantage_history:
            return {
                'total_analyses': 0,
                'quantum_advantages_found': 0,
                'advantage_distribution': {},
                'average_confidence': 0.0,
                'recommendations': 'No analyses performed yet'
            }
        
        # Analyze history
        total_analyses = len(self.advantage_history)
        advantages_found = len([a for a in self.advantage_history if a.advantage_type != AdvantageType.NONE])
        
        advantage_distribution = {}
        for analysis in self.advantage_history:
            adv_type = analysis.advantage_type.value
            advantage_distribution[adv_type] = advantage_distribution.get(adv_type, 0) + 1
        
        average_confidence = np.mean([a.confidence_score for a in self.advantage_history])
        average_advantage_factor = np.mean([a.advantage_factor for a in self.advantage_history if a.advantage_type != AdvantageType.NONE])
        
        # Recent trend analysis
        recent_analyses = self.advantage_history[-10:] if len(self.advantage_history) >= 10 else self.advantage_history
        recent_advantages = len([a for a in recent_analyses if a.advantage_type != AdvantageType.NONE])
        recent_trend = "Increasing" if recent_advantages > advantages_found / 2 else "Stable" if recent_advantages == advantages_found / 2 else "Decreasing"
        
        return {
            'total_analyses': total_analyses,
            'quantum_advantages_found': advantages_found,
            'advantage_rate': advantages_found / total_analyses,
            'advantage_distribution': advantage_distribution,
            'average_confidence': float(average_confidence),
            'average_advantage_factor': float(average_advantage_factor) if average_advantage_factor else 1.0,
            'recent_trend': recent_trend,
            'problem_types_analyzed': list(set([a.problem_type.value for a in self.advantage_history])),
            'recommendations': {
                'quantum_ready_problems': [a.problem_type.value for a in self.advantage_history 
                                         if a.advantage_type != AdvantageType.NONE and a.confidence_score > 0.6],
                'hybrid_recommended_problems': [a.problem_type.value for a in self.advantage_history 
                                              if a.confidence_score > 0.3 and a.confidence_score <= 0.6],
                'classical_recommended_problems': [a.problem_type.value for a in self.advantage_history 
                                                 if a.advantage_type == AdvantageType.NONE or a.confidence_score <= 0.3]
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def predict_advantage_for_problem(self, problem_type: ProblemType, 
                                          problem_size: int) -> Dict[str, Any]:
        """Predict quantum advantage for a given problem without running algorithms"""
        
        # Find similar analyses from history
        similar_analyses = [
            a for a in self.advantage_history 
            if a.problem_type == problem_type and 
            abs(a.problem_size - problem_size) < problem_size * 0.5
        ]
        
        if not similar_analyses:
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'reasoning': 'No similar problems analyzed yet',
                'recommendation': 'Run actual analysis to determine advantage'
            }
        
        # Calculate weighted prediction based on similarity
        total_weight = 0.0
        weighted_advantage_score = 0.0
        
        for analysis in similar_analyses:
            size_similarity = 1.0 / (1.0 + abs(analysis.problem_size - problem_size) / problem_size)
            weight = size_similarity * analysis.confidence_score
            
            advantage_score = 1.0 if analysis.advantage_type != AdvantageType.NONE else 0.0
            weighted_advantage_score += weight * advantage_score
            total_weight += weight
        
        if total_weight == 0:
            prediction_confidence = 0.0
            predicted_advantage = False
        else:
            prediction_confidence = weighted_advantage_score / total_weight
            predicted_advantage = prediction_confidence > 0.5
        
        # Adjust based on problem size trends
        if problem_size > 20:  # Large problems may benefit more from quantum
            prediction_confidence *= 1.2
        elif problem_size < 5:  # Small problems unlikely to show advantage
            prediction_confidence *= 0.8
        
        prediction_confidence = min(1.0, prediction_confidence)
        
        return {
            'prediction': 'Quantum Advantage' if predicted_advantage else 'Classical Preferred',
            'confidence': float(prediction_confidence),
            'reasoning': f"Based on {len(similar_analyses)} similar analyses",
            'recommendation': (
                'Consider quantum approach' if predicted_advantage 
                else 'Classical approach recommended'
            ),
            'expected_advantage_type': (
                max([a.advantage_type.value for a in similar_analyses], 
                    key=lambda x: sum(1 for a in similar_analyses if a.advantage_type.value == x))
                if similar_analyses else 'none'
            ),
            'similar_analyses_count': len(similar_analyses)
        }
