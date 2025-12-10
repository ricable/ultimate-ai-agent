"""
Quantum Circuit Optimization Algorithms
Provides advanced optimization techniques for quantum circuits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
from collections import defaultdict
import copy

# Import from the main quantum module
import sys
sys.path.append('..')
from backend.quantum.quantum_simulator import QuantumGate, GateType

class OptimizationGoal(Enum):
    """Optimization objectives"""
    MINIMIZE_GATES = "minimize_gates"
    MINIMIZE_DEPTH = "minimize_depth"
    MINIMIZE_CNOTS = "minimize_cnots"
    MAXIMIZE_FIDELITY = "maximize_fidelity"
    MINIMIZE_ERROR = "minimize_error"
    MINIMIZE_TIME = "minimize_time"
    PARETO_OPTIMAL = "pareto_optimal"

class OptimizationMethod(Enum):
    """Optimization algorithms"""
    GREEDY = "greedy"
    GENETIC = "genetic"
    SIMULATED_ANNEALING = "simulated_annealing"
    PARTICLE_SWARM = "particle_swarm"
    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN = "bayesian"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

@dataclass
class OptimizationResult:
    """Result of circuit optimization"""
    original_circuit: List[QuantumGate]
    optimized_circuit: List[QuantumGate]
    optimization_method: OptimizationMethod
    optimization_goal: OptimizationGoal
    metrics: Dict[str, float]
    improvement_percentage: float
    optimization_time: float
    iterations: int
    convergence_reached: bool
    metadata: Dict[str, Any]

@dataclass
class CircuitMetrics:
    """Comprehensive circuit metrics"""
    gate_count: int
    depth: int
    cnot_count: int
    single_qubit_count: int
    estimated_fidelity: float
    estimated_error_rate: float
    estimated_execution_time: float
    parallelism_factor: float
    complexity_score: float

class QuantumCircuitOptimizer:
    """Advanced quantum circuit optimization engine"""
    
    def __init__(self):
        self.optimization_history: List[OptimizationResult] = []
        self.gate_rules = self._initialize_optimization_rules()
        self.hardware_constraints = {
            'max_depth': 1000,
            'max_gates': 10000,
            'connectivity': 'all_to_all',  # or 'linear', 'grid', etc.
            'gate_errors': {
                GateType.HADAMARD: 0.001,
                GateType.PAULI_X: 0.001,
                GateType.PAULI_Y: 0.001,
                GateType.PAULI_Z: 0.0005,
                GateType.CNOT: 0.01,
                GateType.ROTATION_X: 0.002,
                GateType.ROTATION_Y: 0.002,
                GateType.ROTATION_Z: 0.001
            },
            'gate_times': {
                GateType.HADAMARD: 0.1,
                GateType.PAULI_X: 0.05,
                GateType.PAULI_Y: 0.05,
                GateType.PAULI_Z: 0.02,
                GateType.CNOT: 0.5,
                GateType.ROTATION_X: 0.15,
                GateType.ROTATION_Y: 0.15,
                GateType.ROTATION_Z: 0.08
            }
        }
    
    def _initialize_optimization_rules(self) -> Dict[str, List[Callable]]:
        """Initialize gate optimization rules"""
        return {
            'identity_removal': [self._remove_identity_gates],
            'gate_cancellation': [self._cancel_inverse_gates, self._cancel_double_gates],
            'gate_merging': [self._merge_rotation_gates, self._merge_adjacent_gates],
            'gate_commutation': [self._commute_gates, self._move_gates_through_cnot],
            'decomposition': [self._decompose_multi_qubit_gates, self._decompose_high_angle_rotations],
            'synthesis': [self._synthesize_gate_sequences, self._optimize_cnot_patterns]
        }
    
    async def optimize_circuit(self, circuit: List[QuantumGate], 
                             goal: OptimizationGoal = OptimizationGoal.MINIMIZE_GATES,
                             method: OptimizationMethod = OptimizationMethod.GREEDY,
                             max_iterations: int = 100) -> OptimizationResult:
        """Optimize quantum circuit using specified method and goal"""
        
        start_time = datetime.utcnow()
        original_metrics = self._calculate_metrics(circuit)
        
        if method == OptimizationMethod.GREEDY:
            result = await self._greedy_optimization(circuit, goal, max_iterations)
        elif method == OptimizationMethod.GENETIC:
            result = await self._genetic_optimization(circuit, goal, max_iterations)
        elif method == OptimizationMethod.SIMULATED_ANNEALING:
            result = await self._simulated_annealing_optimization(circuit, goal, max_iterations)
        elif method == OptimizationMethod.PARTICLE_SWARM:
            result = await self._particle_swarm_optimization(circuit, goal, max_iterations)
        else:
            # Default to greedy
            result = await self._greedy_optimization(circuit, goal, max_iterations)
        
        optimization_time = (datetime.utcnow() - start_time).total_seconds()
        optimized_metrics = self._calculate_metrics(result['circuit'])
        
        # Calculate improvement
        improvement = self._calculate_improvement(original_metrics, optimized_metrics, goal)
        
        optimization_result = OptimizationResult(
            original_circuit=circuit,
            optimized_circuit=result['circuit'],
            optimization_method=method,
            optimization_goal=goal,
            metrics={
                'original_gates': original_metrics.gate_count,
                'optimized_gates': optimized_metrics.gate_count,
                'original_depth': original_metrics.depth,
                'optimized_depth': optimized_metrics.depth,
                'original_cnots': original_metrics.cnot_count,
                'optimized_cnots': optimized_metrics.cnot_count,
                'original_fidelity': original_metrics.estimated_fidelity,
                'optimized_fidelity': optimized_metrics.estimated_fidelity
            },
            improvement_percentage=improvement,
            optimization_time=optimization_time,
            iterations=result['iterations'],
            convergence_reached=result['converged'],
            metadata={
                'method_specific': result.get('metadata', {}),
                'hardware_constraints': self.hardware_constraints
            }
        )
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    async def _greedy_optimization(self, circuit: List[QuantumGate], 
                                 goal: OptimizationGoal, max_iterations: int) -> Dict[str, Any]:
        """Greedy optimization algorithm"""
        current_circuit = circuit.copy()
        best_score = self._evaluate_circuit(current_circuit, goal)
        iterations = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Apply all optimization rules
            for rule_category, rules in self.gate_rules.items():
                for rule in rules:
                    new_circuit = rule(current_circuit)
                    new_score = self._evaluate_circuit(new_circuit, goal)
                    
                    if new_score > best_score:
                        current_circuit = new_circuit
                        best_score = new_score
                        improved = True
            
            iterations += 1
            if not improved:
                break
        
        return {
            'circuit': current_circuit,
            'iterations': iterations,
            'converged': not improved,
            'final_score': best_score
        }
    
    async def _genetic_optimization(self, circuit: List[QuantumGate], 
                                  goal: OptimizationGoal, max_iterations: int) -> Dict[str, Any]:
        """Genetic algorithm optimization"""
        population_size = 20
        mutation_rate = 0.1
        crossover_rate = 0.7
        
        # Initialize population
        population = [circuit.copy() for _ in range(population_size)]
        
        # Add some random variations
        for i in range(1, population_size):
            population[i] = self._mutate_circuit(population[i], mutation_rate * 2)
        
        best_circuit = circuit.copy()
        best_score = self._evaluate_circuit(best_circuit, goal)
        
        for generation in range(max_iterations):
            # Evaluate population
            scores = [self._evaluate_circuit(circ, goal) for circ in population]
            
            # Find best
            best_idx = np.argmax(scores)
            if scores[best_idx] > best_score:
                best_circuit = population[best_idx].copy()
                best_score = scores[best_idx]
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                parent1 = self._tournament_selection(population, scores)
                parent2 = self._tournament_selection(population, scores)
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child = self._crossover_circuits(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child = self._mutate_circuit(child, mutation_rate)
                
                new_population.append(child)
            
            population = new_population
        
        return {
            'circuit': best_circuit,
            'iterations': max_iterations,
            'converged': False,  # GA doesn't have clear convergence
            'final_score': best_score,
            'metadata': {'population_size': population_size, 'mutation_rate': mutation_rate}
        }
    
    async def _simulated_annealing_optimization(self, circuit: List[QuantumGate], 
                                              goal: OptimizationGoal, max_iterations: int) -> Dict[str, Any]:
        """Simulated annealing optimization"""
        current_circuit = circuit.copy()
        best_circuit = circuit.copy()
        
        current_score = self._evaluate_circuit(current_circuit, goal)
        best_score = current_score
        
        # Annealing parameters
        initial_temp = 1.0
        final_temp = 0.01
        cooling_rate = (final_temp / initial_temp) ** (1.0 / max_iterations)
        
        temperature = initial_temp
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor_circuit = self._generate_neighbor(current_circuit)
            neighbor_score = self._evaluate_circuit(neighbor_circuit, goal)
            
            # Accept or reject
            delta = neighbor_score - current_score
            
            if delta > 0 or np.random.random() < np.exp(delta / temperature):
                current_circuit = neighbor_circuit
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_circuit = current_circuit.copy()
                    best_score = current_score
            
            # Cool down
            temperature *= cooling_rate
        
        return {
            'circuit': best_circuit,
            'iterations': max_iterations,
            'converged': temperature <= final_temp,
            'final_score': best_score,
            'metadata': {'final_temperature': temperature}
        }
    
    async def _particle_swarm_optimization(self, circuit: List[QuantumGate], 
                                         goal: OptimizationGoal, max_iterations: int) -> Dict[str, Any]:
        """Particle swarm optimization (adapted for discrete circuits)"""
        swarm_size = 15
        w = 0.9  # Inertia weight
        c1 = 2.0  # Cognitive parameter
        c2 = 2.0  # Social parameter
        
        # Initialize swarm
        particles = [circuit.copy() for _ in range(swarm_size)]
        velocities = [[] for _ in range(swarm_size)]  # Discrete "velocities"
        personal_best = particles.copy()
        personal_best_scores = [self._evaluate_circuit(p, goal) for p in particles]
        
        global_best = max(particles, key=lambda x: self._evaluate_circuit(x, goal))
        global_best_score = self._evaluate_circuit(global_best, goal)
        
        for iteration in range(max_iterations):
            for i in range(swarm_size):
                # Update particle (discrete version)
                new_particle = self._update_particle(
                    particles[i], personal_best[i], global_best, w, c1, c2
                )
                
                score = self._evaluate_circuit(new_particle, goal)
                
                # Update personal best
                if score > personal_best_scores[i]:
                    personal_best[i] = new_particle.copy()
                    personal_best_scores[i] = score
                
                # Update global best
                if score > global_best_score:
                    global_best = new_particle.copy()
                    global_best_score = score
                
                particles[i] = new_particle
        
        return {
            'circuit': global_best,
            'iterations': max_iterations,
            'converged': False,
            'final_score': global_best_score,
            'metadata': {'swarm_size': swarm_size}
        }
    
    def _calculate_metrics(self, circuit: List[QuantumGate]) -> CircuitMetrics:
        """Calculate comprehensive circuit metrics"""
        gate_count = len(circuit)
        cnot_count = sum(1 for gate in circuit if gate.gate_type == GateType.CNOT)
        single_qubit_count = sum(1 for gate in circuit if len(gate.qubits) == 1)
        
        # Calculate depth
        depth = self._calculate_circuit_depth(circuit)
        
        # Estimate fidelity based on error rates
        estimated_error = sum(
            self.hardware_constraints['gate_errors'].get(gate.gate_type, 0.01)
            for gate in circuit
        )
        estimated_fidelity = max(0.0, 1.0 - estimated_error)
        
        # Estimate execution time
        estimated_time = sum(
            self.hardware_constraints['gate_times'].get(gate.gate_type, 0.1)
            for gate in circuit
        )
        
        # Calculate parallelism factor
        parallelism_factor = gate_count / depth if depth > 0 else 1.0
        
        # Complexity score (weighted combination)
        complexity_score = (
            gate_count * 1.0 +
            cnot_count * 3.0 +  # CNOTs are more expensive
            depth * 2.0
        )
        
        return CircuitMetrics(
            gate_count=gate_count,
            depth=depth,
            cnot_count=cnot_count,
            single_qubit_count=single_qubit_count,
            estimated_fidelity=estimated_fidelity,
            estimated_error_rate=estimated_error,
            estimated_execution_time=estimated_time,
            parallelism_factor=parallelism_factor,
            complexity_score=complexity_score
        )
    
    def _calculate_circuit_depth(self, circuit: List[QuantumGate]) -> int:
        """Calculate circuit depth (critical path)"""
        if not circuit:
            return 0
        
        # Find maximum qubit index
        max_qubit = max(max(gate.qubits) for gate in circuit if gate.qubits)
        qubit_times = [0] * (max_qubit + 1)
        
        for gate in circuit:
            max_time = max(qubit_times[q] for q in gate.qubits)
            for qubit in gate.qubits:
                qubit_times[qubit] = max_time + 1
        
        return max(qubit_times)
    
    def _evaluate_circuit(self, circuit: List[QuantumGate], goal: OptimizationGoal) -> float:
        """Evaluate circuit based on optimization goal"""
        metrics = self._calculate_metrics(circuit)
        
        if goal == OptimizationGoal.MINIMIZE_GATES:
            return 1.0 / (1.0 + metrics.gate_count)
        elif goal == OptimizationGoal.MINIMIZE_DEPTH:
            return 1.0 / (1.0 + metrics.depth)
        elif goal == OptimizationGoal.MINIMIZE_CNOTS:
            return 1.0 / (1.0 + metrics.cnot_count)
        elif goal == OptimizationGoal.MAXIMIZE_FIDELITY:
            return metrics.estimated_fidelity
        elif goal == OptimizationGoal.MINIMIZE_ERROR:
            return 1.0 / (1.0 + metrics.estimated_error_rate)
        elif goal == OptimizationGoal.MINIMIZE_TIME:
            return 1.0 / (1.0 + metrics.estimated_execution_time)
        elif goal == OptimizationGoal.PARETO_OPTIMAL:
            # Multi-objective optimization (simplified)
            return (
                0.3 * (1.0 / (1.0 + metrics.gate_count)) +
                0.3 * (1.0 / (1.0 + metrics.depth)) +
                0.4 * metrics.estimated_fidelity
            )
        else:
            return 0.0
    
    def _calculate_improvement(self, original: CircuitMetrics, optimized: CircuitMetrics, 
                             goal: OptimizationGoal) -> float:
        """Calculate improvement percentage"""
        if goal == OptimizationGoal.MINIMIZE_GATES:
            if original.gate_count == 0:
                return 0.0
            return ((original.gate_count - optimized.gate_count) / original.gate_count) * 100
        elif goal == OptimizationGoal.MINIMIZE_DEPTH:
            if original.depth == 0:
                return 0.0
            return ((original.depth - optimized.depth) / original.depth) * 100
        elif goal == OptimizationGoal.MINIMIZE_CNOTS:
            if original.cnot_count == 0:
                return 0.0
            return ((original.cnot_count - optimized.cnot_count) / original.cnot_count) * 100
        elif goal == OptimizationGoal.MAXIMIZE_FIDELITY:
            return ((optimized.estimated_fidelity - original.estimated_fidelity) / 
                   max(original.estimated_fidelity, 1e-6)) * 100
        else:
            # General improvement based on complexity score
            if original.complexity_score == 0:
                return 0.0
            return ((original.complexity_score - optimized.complexity_score) / 
                   original.complexity_score) * 100
    
    # Optimization rule implementations
    def _remove_identity_gates(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Remove gates that act as identity"""
        optimized = []
        
        for gate in circuit:
            # Check for identity rotations
            if gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
                angle = gate.parameters[0] if gate.parameters else 0.0
                if abs(angle % (2 * np.pi)) < 1e-10:
                    continue  # Skip identity rotation
            
            optimized.append(gate)
        
        return optimized
    
    def _cancel_inverse_gates(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Cancel adjacent inverse gates"""
        optimized = []
        i = 0
        
        while i < len(circuit):
            if i + 1 < len(circuit) and self._are_inverse_gates(circuit[i], circuit[i + 1]):
                i += 2  # Skip both gates
            else:
                optimized.append(circuit[i])
                i += 1
        
        return optimized
    
    def _cancel_double_gates(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Cancel double applications of self-inverse gates"""
        optimized = []
        i = 0
        
        while i < len(circuit):
            if (i + 1 < len(circuit) and 
                circuit[i].gate_type == circuit[i + 1].gate_type and
                circuit[i].qubits == circuit[i + 1].qubits and
                self._is_self_inverse(circuit[i])):
                i += 2  # Skip both gates
            else:
                optimized.append(circuit[i])
                i += 1
        
        return optimized
    
    def _merge_rotation_gates(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Merge consecutive rotation gates on same qubit and axis"""
        optimized = []
        i = 0
        
        while i < len(circuit):
            current_gate = circuit[i]
            
            if current_gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
                # Look for consecutive rotations
                merged_angle = current_gate.parameters[0] if current_gate.parameters else 0.0
                j = i + 1
                
                while (j < len(circuit) and 
                       circuit[j].gate_type == current_gate.gate_type and
                       circuit[j].qubits == current_gate.qubits):
                    next_angle = circuit[j].parameters[0] if circuit[j].parameters else 0.0
                    merged_angle += next_angle
                    j += 1
                
                # Create merged gate if angle is non-zero
                final_angle = merged_angle % (2 * np.pi)
                if abs(final_angle) > 1e-10 and abs(final_angle - 2*np.pi) > 1e-10:
                    merged_gate = QuantumGate(
                        gate_type=current_gate.gate_type,
                        qubits=current_gate.qubits,
                        parameters=[final_angle]
                    )
                    optimized.append(merged_gate)
                
                i = j
            else:
                optimized.append(current_gate)
                i += 1
        
        return optimized
    
    def _merge_adjacent_gates(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Merge adjacent gates that can be combined"""
        # Simplified - just return original for now
        # In practice, would implement matrix multiplication for compatible gates
        return circuit
    
    def _commute_gates(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Commute gates to enable further optimizations"""
        optimized = circuit.copy()
        
        # Look for commuting gates that can be moved
        for i in range(len(optimized) - 1):
            if self._gates_commute(optimized[i], optimized[i + 1]):
                # Check if moving helps with later optimizations
                if self._would_benefit_from_swap(optimized, i):
                    optimized[i], optimized[i + 1] = optimized[i + 1], optimized[i]
        
        return optimized
    
    def _move_gates_through_cnot(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Move single-qubit gates through CNOT gates when possible"""
        optimized = circuit.copy()
        
        for i in range(len(optimized) - 1):
            current = optimized[i]
            next_gate = optimized[i + 1]
            
            # Check if single-qubit gate can commute through CNOT
            if (next_gate.gate_type == GateType.CNOT and len(current.qubits) == 1):
                control, target = next_gate.qubits[0], next_gate.qubits[1]
                
                # Z rotations commute through CNOT on control
                if (current.gate_type == GateType.ROTATION_Z and 
                    current.qubits[0] == control):
                    # Can move Z rotation before CNOT
                    optimized[i], optimized[i + 1] = optimized[i + 1], optimized[i]
        
        return optimized
    
    def _decompose_multi_qubit_gates(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Decompose multi-qubit gates into elementary gates"""
        optimized = []
        
        for gate in circuit:
            if gate.gate_type == GateType.TOFFOLI:
                # Decompose Toffoli into CNOTs and single-qubit gates
                control1, control2, target = gate.qubits
                decomposed = [
                    QuantumGate(GateType.HADAMARD, [target]),
                    QuantumGate(GateType.CNOT, [control2, target]),
                    QuantumGate(GateType.ROTATION_Z, [target], [-np.pi/4]),
                    QuantumGate(GateType.CNOT, [control1, target]),
                    QuantumGate(GateType.ROTATION_Z, [target], [np.pi/4]),
                    QuantumGate(GateType.CNOT, [control2, target]),
                    QuantumGate(GateType.ROTATION_Z, [target], [-np.pi/4]),
                    QuantumGate(GateType.CNOT, [control1, target]),
                    QuantumGate(GateType.ROTATION_Z, [control2], [np.pi/4]),
                    QuantumGate(GateType.ROTATION_Z, [target], [np.pi/4]),
                    QuantumGate(GateType.CNOT, [control1, control2]),
                    QuantumGate(GateType.HADAMARD, [target]),
                    QuantumGate(GateType.ROTATION_Z, [control1], [np.pi/4]),
                    QuantumGate(GateType.ROTATION_Z, [control2], [-np.pi/4]),
                    QuantumGate(GateType.CNOT, [control1, control2])
                ]
                optimized.extend(decomposed)
            else:
                optimized.append(gate)
        
        return optimized
    
    def _decompose_high_angle_rotations(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Decompose high-angle rotations into smaller rotations"""
        optimized = []
        max_angle = np.pi / 2  # Maximum angle before decomposition
        
        for gate in circuit:
            if (gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z] and
                gate.parameters and abs(gate.parameters[0]) > max_angle):
                
                angle = gate.parameters[0]
                num_parts = int(np.ceil(abs(angle) / max_angle))
                part_angle = angle / num_parts
                
                # Create multiple smaller rotations
                for _ in range(num_parts):
                    optimized.append(QuantumGate(
                        gate.gate_type, gate.qubits, [part_angle]
                    ))
            else:
                optimized.append(gate)
        
        return optimized
    
    def _synthesize_gate_sequences(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Synthesize optimal gate sequences for common patterns"""
        # Simplified - just return original
        # In practice, would identify patterns and replace with optimal implementations
        return circuit
    
    def _optimize_cnot_patterns(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Optimize CNOT gate patterns and connectivity"""
        optimized = []
        
        # Look for CNOT chains that can be optimized
        i = 0
        while i < len(circuit):
            if circuit[i].gate_type == GateType.CNOT:
                # Collect consecutive CNOTs
                cnot_chain = [circuit[i]]
                j = i + 1
                
                while j < len(circuit) and circuit[j].gate_type == GateType.CNOT:
                    cnot_chain.append(circuit[j])
                    j += 1
                
                # Optimize the CNOT chain
                optimized_chain = self._optimize_cnot_chain(cnot_chain)
                optimized.extend(optimized_chain)
                
                i = j
            else:
                optimized.append(circuit[i])
                i += 1
        
        return optimized
    
    def _optimize_cnot_chain(self, cnot_chain: List[QuantumGate]) -> List[QuantumGate]:
        """Optimize a chain of CNOT gates"""
        if len(cnot_chain) <= 1:
            return cnot_chain
        
        # Remove duplicate CNOTs (CNOT is self-inverse)
        optimized = []
        cnot_applied = set()
        
        for cnot in cnot_chain:
            cnot_key = tuple(cnot.qubits)
            if cnot_key in cnot_applied:
                cnot_applied.remove(cnot_key)  # Cancel out
            else:
                cnot_applied.add(cnot_key)
        
        # Convert back to gates
        for control, target in cnot_applied:
            optimized.append(QuantumGate(GateType.CNOT, [control, target]))
        
        return optimized
    
    # Helper methods for optimization algorithms
    def _are_inverse_gates(self, gate1: QuantumGate, gate2: QuantumGate) -> bool:
        """Check if two gates are inverses"""
        if gate1.gate_type != gate2.gate_type or gate1.qubits != gate2.qubits:
            return False
        
        # For rotation gates, check if angles sum to 0 (mod 2π)
        if gate1.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
            angle1 = gate1.parameters[0] if gate1.parameters else 0.0
            angle2 = gate2.parameters[0] if gate2.parameters else 0.0
            return abs((angle1 + angle2) % (2 * np.pi)) < 1e-10
        
        # Self-inverse gates
        return gate1.gate_type in [GateType.PAULI_X, GateType.PAULI_Y, GateType.PAULI_Z, 
                                 GateType.HADAMARD, GateType.CNOT]
    
    def _is_self_inverse(self, gate: QuantumGate) -> bool:
        """Check if gate is self-inverse"""
        return gate.gate_type in [GateType.PAULI_X, GateType.PAULI_Y, GateType.PAULI_Z, 
                                GateType.HADAMARD, GateType.CNOT]
    
    def _gates_commute(self, gate1: QuantumGate, gate2: QuantumGate) -> bool:
        """Check if two gates commute"""
        # Gates on different qubits always commute
        if not set(gate1.qubits).intersection(set(gate2.qubits)):
            return True
        
        # Same-type gates on same qubits commute
        if (gate1.gate_type == gate2.gate_type and 
            gate1.qubits == gate2.qubits):
            return True
        
        # Z rotations commute with each other
        if (gate1.gate_type == GateType.ROTATION_Z and 
            gate2.gate_type == GateType.ROTATION_Z and
            gate1.qubits == gate2.qubits):
            return True
        
        # More commutation rules could be added
        return False
    
    def _would_benefit_from_swap(self, circuit: List[QuantumGate], index: int) -> bool:
        """Check if swapping gates at index would benefit optimization"""
        # Simplified heuristic
        return np.random.random() < 0.3
    
    def _mutate_circuit(self, circuit: List[QuantumGate], mutation_rate: float) -> List[QuantumGate]:
        """Mutate circuit for genetic algorithm"""
        mutated = circuit.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                # Random mutation
                mutation_type = np.random.choice(['remove', 'modify', 'swap'])
                
                if mutation_type == 'remove' and len(mutated) > 1:
                    mutated.pop(i)
                elif mutation_type == 'modify':
                    mutated[i] = self._mutate_gate(mutated[i])
                elif mutation_type == 'swap' and i + 1 < len(mutated):
                    mutated[i], mutated[i + 1] = mutated[i + 1], mutated[i]
        
        return mutated
    
    def _mutate_gate(self, gate: QuantumGate) -> QuantumGate:
        """Mutate a single gate"""
        if gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
            # Modify rotation angle
            angle = gate.parameters[0] if gate.parameters else 0.0
            new_angle = angle + np.random.normal(0, 0.1)  # Small random change
            return QuantumGate(gate.gate_type, gate.qubits, [new_angle])
        
        return gate  # No mutation for non-parameterized gates
    
    def _tournament_selection(self, population: List[List[QuantumGate]], 
                            scores: List[float], tournament_size: int = 3) -> List[QuantumGate]:
        """Tournament selection for genetic algorithm"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return population[winner_idx].copy()
    
    def _crossover_circuits(self, parent1: List[QuantumGate], 
                          parent2: List[QuantumGate]) -> List[QuantumGate]:
        """Crossover two circuits"""
        if not parent1 or not parent2:
            return parent1 if parent1 else parent2
        
        # Single-point crossover
        crossover_point1 = np.random.randint(0, len(parent1))
        crossover_point2 = np.random.randint(0, len(parent2))
        
        child = parent1[:crossover_point1] + parent2[crossover_point2:]
        return child
    
    def _generate_neighbor(self, circuit: List[QuantumGate]) -> List[QuantumGate]:
        """Generate neighbor solution for simulated annealing"""
        if not circuit:
            return circuit
        
        neighbor = circuit.copy()
        
        # Random operation
        operation = np.random.choice(['remove', 'add', 'modify', 'swap'])
        
        if operation == 'remove' and len(neighbor) > 1:
            idx = np.random.randint(0, len(neighbor))
            neighbor.pop(idx)
        
        elif operation == 'modify':
            idx = np.random.randint(0, len(neighbor))
            neighbor[idx] = self._mutate_gate(neighbor[idx])
        
        elif operation == 'swap' and len(neighbor) > 1:
            idx1 = np.random.randint(0, len(neighbor))
            idx2 = np.random.randint(0, len(neighbor))
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        
        return neighbor
    
    def _update_particle(self, particle: List[QuantumGate], personal_best: List[QuantumGate],
                        global_best: List[QuantumGate], w: float, c1: float, c2: float) -> List[QuantumGate]:
        """Update particle for PSO (discrete version)"""
        # Simplified discrete PSO update
        new_particle = particle.copy()
        
        # Move towards personal best
        if np.random.random() < c1 * 0.1:
            if personal_best and len(personal_best) < len(new_particle):
                # Remove random gate
                if new_particle:
                    idx = np.random.randint(0, len(new_particle))
                    new_particle.pop(idx)
        
        # Move towards global best
        if np.random.random() < c2 * 0.1:
            if global_best and len(global_best) < len(new_particle):
                # Remove random gate
                if new_particle:
                    idx = np.random.randint(0, len(new_particle))
                    new_particle.pop(idx)
        
        return new_particle
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        if not self.optimization_history:
            return {"status": "No optimizations performed yet"}
        
        total_optimizations = len(self.optimization_history)
        
        # Calculate average improvements by method
        method_stats = defaultdict(list)
        for result in self.optimization_history:
            method_stats[result.optimization_method.value].append(result.improvement_percentage)
        
        method_averages = {
            method: {
                "count": len(improvements),
                "average_improvement": np.mean(improvements),
                "best_improvement": max(improvements),
                "success_rate": len([i for i in improvements if i > 0]) / len(improvements)
            }
            for method, improvements in method_stats.items()
        }
        
        # Best optimization
        best_optimization = max(self.optimization_history, key=lambda x: x.improvement_percentage)
        
        # Recent performance
        recent_optimizations = self.optimization_history[-10:] if total_optimizations >= 10 else self.optimization_history
        recent_average = np.mean([r.improvement_percentage for r in recent_optimizations])
        
        return {
            "summary": {
                "total_optimizations": total_optimizations,
                "average_improvement": np.mean([r.improvement_percentage for r in self.optimization_history]),
                "best_improvement": best_optimization.improvement_percentage,
                "recent_average": recent_average
            },
            "method_performance": method_averages,
            "best_result": {
                "method": best_optimization.optimization_method.value,
                "goal": best_optimization.optimization_goal.value,
                "improvement": best_optimization.improvement_percentage,
                "metrics": best_optimization.metrics
            },
            "recommendations": self._generate_optimization_recommendations(method_averages),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_optimization_recommendations(self, method_stats: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate optimization recommendations based on performance"""
        recommendations = []
        
        if not method_stats:
            recommendations.append("Run more optimizations to generate recommendations")
            return recommendations
        
        # Find best performing method
        best_method = max(method_stats.keys(), 
                         key=lambda m: method_stats[m]["average_improvement"])
        
        recommendations.append(f"Best performing method: {best_method}")
        
        # Check success rates
        high_success_methods = [m for m, stats in method_stats.items() 
                              if stats["success_rate"] > 0.8]
        
        if high_success_methods:
            recommendations.append(f"Reliable methods: {', '.join(high_success_methods)}")
        
        # Method-specific recommendations
        if "genetic" in method_stats and method_stats["genetic"]["average_improvement"] > 10:
            recommendations.append("Genetic algorithm shows promise for complex circuits")
        
        if "greedy" in method_stats and method_stats["greedy"]["success_rate"] > 0.9:
            recommendations.append("Greedy optimization is reliable for quick improvements")
        
        return recommendations

# Convenience functions
def quick_optimize(circuit: List[QuantumGate], goal: str = "gates") -> List[QuantumGate]:
    """Quick circuit optimization"""
    optimizer = QuantumCircuitOptimizer()
    
    goal_map = {
        "gates": OptimizationGoal.MINIMIZE_GATES,
        "depth": OptimizationGoal.MINIMIZE_DEPTH,
        "cnots": OptimizationGoal.MINIMIZE_CNOTS,
        "fidelity": OptimizationGoal.MAXIMIZE_FIDELITY
    }
    
    optimization_goal = goal_map.get(goal, OptimizationGoal.MINIMIZE_GATES)
    
    import asyncio
    result = asyncio.run(optimizer.optimize_circuit(circuit, optimization_goal))
    return result.optimized_circuit

def benchmark_optimizers(circuit: List[QuantumGate]) -> Dict[str, float]:
    """Benchmark different optimization methods"""
    optimizer = QuantumCircuitOptimizer()
    results = {}
    
    methods = [OptimizationMethod.GREEDY, OptimizationMethod.GENETIC, 
              OptimizationMethod.SIMULATED_ANNEALING]
    
    import asyncio
    
    async def run_benchmarks():
        benchmark_results = {}
        for method in methods:
            result = await optimizer.optimize_circuit(
                circuit, OptimizationGoal.MINIMIZE_GATES, method, max_iterations=20
            )
            benchmark_results[method.value] = result.improvement_percentage
        return benchmark_results
    
    return asyncio.run(run_benchmarks())
