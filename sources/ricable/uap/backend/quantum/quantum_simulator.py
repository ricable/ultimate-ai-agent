"""
Quantum Circuit Simulator
Implements quantum circuit simulation with state vector and density matrix methods.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class GateType(Enum):
    """Quantum gate types"""
    PAULI_X = "X"
    PAULI_Y = "Y" 
    PAULI_Z = "Z"
    HADAMARD = "H"
    PHASE = "S"
    T_GATE = "T"
    CNOT = "CNOT"
    TOFFOLI = "TOFFOLI"
    ROTATION_X = "RX"
    ROTATION_Y = "RY"
    ROTATION_Z = "RZ"
    CONTROLLED_Z = "CZ"
    SWAP = "SWAP"
    CUSTOM = "CUSTOM"

@dataclass
class QuantumGate:
    """Represents a quantum gate in the circuit"""
    gate_type: GateType
    qubits: List[int]
    parameters: Optional[List[float]] = None
    name: Optional[str] = None
    matrix: Optional[np.ndarray] = None

@dataclass
class MeasurementResult:
    """Result of quantum measurement"""
    qubit: int
    outcome: int  # 0 or 1
    probability: float
    timestamp: datetime

@dataclass
class CircuitResult:
    """Result of quantum circuit execution"""
    final_state: np.ndarray
    measurements: List[MeasurementResult]
    fidelity: float
    execution_time: float
    gate_count: int
    depth: int
    metadata: Dict[str, Any]

class QuantumState:
    """Represents a quantum state with operations"""
    
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits
        # Initialize to |0...0⟩ state
        self.state_vector = np.zeros(self.dimension, dtype=complex)
        self.state_vector[0] = 1.0
        self.density_matrix = None
        self.is_mixed = False
        
    def to_density_matrix(self) -> np.ndarray:
        """Convert state vector to density matrix"""
        if self.is_mixed and self.density_matrix is not None:
            return self.density_matrix
        return np.outer(self.state_vector, np.conj(self.state_vector))
    
    def get_probability(self, state_index: int) -> float:
        """Get probability of measuring specific state"""
        if self.is_mixed:
            return np.real(self.density_matrix[state_index, state_index])
        return np.abs(self.state_vector[state_index]) ** 2
    
    def measure_qubit(self, qubit: int) -> Tuple[int, float]:
        """Measure a specific qubit and collapse state"""
        # Calculate probabilities for 0 and 1 outcomes
        prob_0 = 0.0
        prob_1 = 0.0
        
        for i in range(self.dimension):
            if (i >> qubit) & 1 == 0:
                prob_0 += self.get_probability(i)
            else:
                prob_1 += self.get_probability(i)
        
        # Random measurement outcome
        outcome = 1 if np.random.random() < prob_1 else 0
        
        # Collapse the state
        self._collapse_state(qubit, outcome)
        
        return outcome, prob_1 if outcome == 1 else prob_0
    
    def _collapse_state(self, qubit: int, outcome: int):
        """Collapse state after measurement"""
        new_state = np.zeros_like(self.state_vector)
        normalization = 0.0
        
        for i in range(self.dimension):
            if (i >> qubit) & 1 == outcome:
                new_state[i] = self.state_vector[i]
                normalization += np.abs(self.state_vector[i]) ** 2
        
        if normalization > 0:
            new_state /= np.sqrt(normalization)
        
        self.state_vector = new_state

class QuantumCircuitSimulator:
    """High-performance quantum circuit simulator"""
    
    def __init__(self, max_qubits: int = 20):
        self.max_qubits = max_qubits
        self.gate_matrices = self._initialize_gate_matrices()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _initialize_gate_matrices(self) -> Dict[GateType, np.ndarray]:
        """Initialize standard quantum gate matrices"""
        sqrt2 = 1.0 / np.sqrt(2)
        
        return {
            GateType.PAULI_X: np.array([[0, 1], [1, 0]], dtype=complex),
            GateType.PAULI_Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
            GateType.PAULI_Z: np.array([[1, 0], [0, -1]], dtype=complex),
            GateType.HADAMARD: np.array([[sqrt2, sqrt2], [sqrt2, -sqrt2]], dtype=complex),
            GateType.PHASE: np.array([[1, 0], [0, 1j]], dtype=complex),
            GateType.T_GATE: np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        }
    
    def create_rotation_gate(self, axis: str, angle: float) -> np.ndarray:
        """Create rotation gate matrix"""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        
        if axis.upper() == 'X':
            return np.array([[cos_half, -1j * sin_half], 
                           [-1j * sin_half, cos_half]], dtype=complex)
        elif axis.upper() == 'Y':
            return np.array([[cos_half, -sin_half], 
                           [sin_half, cos_half]], dtype=complex)
        elif axis.upper() == 'Z':
            return np.array([[np.exp(-1j * angle / 2), 0], 
                           [0, np.exp(1j * angle / 2)]], dtype=complex)
        else:
            raise ValueError(f"Unknown rotation axis: {axis}")
    
    def create_controlled_gate(self, base_gate: np.ndarray, num_qubits: int, 
                             control_qubits: List[int], target_qubit: int) -> np.ndarray:
        """Create controlled version of a gate"""
        dim = 2 ** num_qubits
        controlled_gate = np.eye(dim, dtype=complex)
        
        # Apply gate only when all control qubits are |1⟩
        for i in range(dim):
            # Check if all control qubits are 1
            control_active = all((i >> control) & 1 for control in control_qubits)
            
            if control_active:
                # Apply the base gate to the target qubit
                target_0 = i & ~(1 << target_qubit)  # Set target bit to 0
                target_1 = i | (1 << target_qubit)   # Set target bit to 1
                
                for j in range(dim):
                    if j == target_0:
                        controlled_gate[i, j] = base_gate[1, 0] if (i >> target_qubit) & 1 else base_gate[0, 0]
                    elif j == target_1:
                        controlled_gate[i, j] = base_gate[1, 1] if (i >> target_qubit) & 1 else base_gate[0, 1]
        
        return controlled_gate
    
    async def simulate_circuit(self, gates: List[QuantumGate], 
                             num_qubits: int, shots: int = 1000) -> CircuitResult:
        """Simulate quantum circuit execution"""
        if num_qubits > self.max_qubits:
            raise ValueError(f"Circuit has {num_qubits} qubits, maximum is {self.max_qubits}")
        
        start_time = datetime.utcnow()
        
        # Initialize quantum state
        state = QuantumState(num_qubits)
        
        # Apply gates sequentially
        for gate in gates:
            await self._apply_gate(state, gate)
        
        # Perform measurements
        measurements = []
        for shot in range(shots):
            measurement_results = await self._perform_measurements(state, num_qubits)
            measurements.extend(measurement_results)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Calculate circuit metrics
        circuit_depth = self._calculate_depth(gates, num_qubits)
        fidelity = self._calculate_fidelity(state)
        
        return CircuitResult(
            final_state=state.state_vector,
            measurements=measurements,
            fidelity=fidelity,
            execution_time=execution_time,
            gate_count=len(gates),
            depth=circuit_depth,
            metadata={
                "shots": shots,
                "num_qubits": num_qubits,
                "timestamp": start_time.isoformat(),
                "simulator": "quantum_circuit_simulator"
            }
        )
    
    async def _apply_gate(self, state: QuantumState, gate: QuantumGate):
        """Apply a quantum gate to the state"""
        if gate.gate_type in self.gate_matrices:
            gate_matrix = self.gate_matrices[gate.gate_type]
        elif gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
            axis = gate.gate_type.value[1]  # Extract X, Y, or Z from RX, RY, RZ
            angle = gate.parameters[0] if gate.parameters else 0.0
            gate_matrix = self.create_rotation_gate(axis, angle)
        elif gate.gate_type == GateType.CUSTOM and gate.matrix is not None:
            gate_matrix = gate.matrix
        else:
            raise ValueError(f"Unsupported gate type: {gate.gate_type}")
        
        # Apply single-qubit gates
        if len(gate.qubits) == 1:
            await self._apply_single_qubit_gate(state, gate_matrix, gate.qubits[0])
        # Apply two-qubit gates
        elif len(gate.qubits) == 2:
            await self._apply_two_qubit_gate(state, gate_matrix, gate.qubits)
        else:
            # Multi-qubit controlled gates
            control_qubits = gate.qubits[:-1]
            target_qubit = gate.qubits[-1]
            controlled_matrix = self.create_controlled_gate(
                gate_matrix, state.num_qubits, control_qubits, target_qubit
            )
            state.state_vector = controlled_matrix @ state.state_vector
    
    async def _apply_single_qubit_gate(self, state: QuantumState, 
                                     gate_matrix: np.ndarray, qubit: int):
        """Apply single-qubit gate using tensor product"""
        # Create full gate matrix for all qubits
        full_matrix = np.eye(1, dtype=complex)
        
        for i in range(state.num_qubits):
            if i == qubit:
                full_matrix = np.kron(full_matrix, gate_matrix)
            else:
                full_matrix = np.kron(full_matrix, np.eye(2, dtype=complex))
        
        # Apply gate to state
        state.state_vector = full_matrix @ state.state_vector
    
    async def _apply_two_qubit_gate(self, state: QuantumState, 
                                  gate_matrix: np.ndarray, qubits: List[int]):
        """Apply two-qubit gate (like CNOT)"""
        control, target = qubits[0], qubits[1]
        
        # CNOT gate implementation
        if gate_matrix.shape == (2, 2) and np.allclose(gate_matrix, self.gate_matrices[GateType.PAULI_X]):
            # This is a CNOT with X gate - create proper CNOT matrix
            cnot_matrix = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0], 
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]], dtype=complex)
            
            # Apply CNOT to the specific qubits
            new_state = np.zeros_like(state.state_vector)
            
            for i in range(state.dimension):
                control_bit = (i >> control) & 1
                target_bit = (i >> target) & 1
                
                if control_bit == 1:
                    # Flip target bit
                    new_index = i ^ (1 << target)
                    new_state[new_index] = state.state_vector[i]
                else:
                    # Keep state unchanged
                    new_state[i] = state.state_vector[i]
            
            state.state_vector = new_state
    
    async def _perform_measurements(self, state: QuantumState, 
                                  num_qubits: int) -> List[MeasurementResult]:
        """Perform measurements on all qubits"""
        measurements = []
        
        for qubit in range(num_qubits):
            outcome, probability = state.measure_qubit(qubit)
            measurements.append(MeasurementResult(
                qubit=qubit,
                outcome=outcome,
                probability=probability,
                timestamp=datetime.utcnow()
            ))
        
        return measurements
    
    def _calculate_depth(self, gates: List[QuantumGate], num_qubits: int) -> int:
        """Calculate circuit depth (critical path length)"""
        qubit_times = [0] * num_qubits
        
        for gate in gates:
            max_time = max(qubit_times[q] for q in gate.qubits)
            for qubit in gate.qubits:
                qubit_times[qubit] = max_time + 1
        
        return max(qubit_times)
    
    def _calculate_fidelity(self, state: QuantumState) -> float:
        """Calculate state fidelity (simplified)"""
        # For pure states, fidelity with ideal state
        norm = np.linalg.norm(state.state_vector)
        return float(norm ** 2)
    
    async def optimize_circuit(self, gates: List[QuantumGate]) -> List[QuantumGate]:
        """Optimize quantum circuit by reducing gate count"""
        optimized_gates = gates.copy()
        
        # Simple optimization: remove identity operations
        optimized_gates = [gate for gate in optimized_gates 
                          if not self._is_identity_gate(gate)]
        
        # Cancel adjacent inverse gates
        optimized_gates = await self._cancel_inverse_gates(optimized_gates)
        
        # Merge rotation gates
        optimized_gates = await self._merge_rotation_gates(optimized_gates)
        
        return optimized_gates
    
    def _is_identity_gate(self, gate: QuantumGate) -> bool:
        """Check if gate is effectively identity"""
        if gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
            angle = gate.parameters[0] if gate.parameters else 0.0
            return abs(angle) < 1e-10
        return False
    
    async def _cancel_inverse_gates(self, gates: List[QuantumGate]) -> List[QuantumGate]:
        """Cancel adjacent inverse gates"""
        optimized = []
        i = 0
        
        while i < len(gates):
            if i + 1 < len(gates) and self._are_inverse_gates(gates[i], gates[i + 1]):
                i += 2  # Skip both gates
            else:
                optimized.append(gates[i])
                i += 1
        
        return optimized
    
    def _are_inverse_gates(self, gate1: QuantumGate, gate2: QuantumGate) -> bool:
        """Check if two gates are inverses of each other"""
        # Simple check for same gate type and qubits
        if (gate1.gate_type == gate2.gate_type and 
            gate1.qubits == gate2.qubits):
            
            # For rotation gates, check if angles sum to 0 (mod 2π)
            if gate1.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
                angle1 = gate1.parameters[0] if gate1.parameters else 0.0
                angle2 = gate2.parameters[0] if gate2.parameters else 0.0
                return abs((angle1 + angle2) % (2 * np.pi)) < 1e-10
            
            # Self-inverse gates
            if gate1.gate_type in [GateType.PAULI_X, GateType.PAULI_Y, GateType.PAULI_Z, GateType.HADAMARD]:
                return True
        
        return False
    
    async def _merge_rotation_gates(self, gates: List[QuantumGate]) -> List[QuantumGate]:
        """Merge consecutive rotation gates on same qubit and axis"""
        optimized = []
        i = 0
        
        while i < len(gates):
            current_gate = gates[i]
            
            if current_gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z]:
                # Look for consecutive rotation gates on same qubit and axis
                merged_angle = current_gate.parameters[0] if current_gate.parameters else 0.0
                j = i + 1
                
                while (j < len(gates) and 
                       gates[j].gate_type == current_gate.gate_type and
                       gates[j].qubits == current_gate.qubits):
                    next_angle = gates[j].parameters[0] if gates[j].parameters else 0.0
                    merged_angle += next_angle
                    j += 1
                
                # Create merged gate
                merged_gate = QuantumGate(
                    gate_type=current_gate.gate_type,
                    qubits=current_gate.qubits,
                    parameters=[merged_angle % (2 * np.pi)]
                )
                optimized.append(merged_gate)
                i = j
            else:
                optimized.append(current_gate)
                i += 1
        
        return optimized
    
    async def calculate_expectation_value(self, state: QuantumState, 
                                        observable: np.ndarray) -> complex:
        """Calculate expectation value of observable"""
        if state.is_mixed:
            # Tr(ρ * O)
            return np.trace(state.density_matrix @ observable)
        else:
            # ⟨ψ|O|ψ⟩
            return np.conj(state.state_vector) @ observable @ state.state_vector
    
    def get_circuit_statistics(self, gates: List[QuantumGate]) -> Dict[str, Any]:
        """Get detailed statistics about the circuit"""
        gate_counts = {}
        total_parameters = 0
        
        for gate in gates:
            gate_type = gate.gate_type.value
            gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
            if gate.parameters:
                total_parameters += len(gate.parameters)
        
        return {
            "total_gates": len(gates),
            "gate_counts": gate_counts,
            "parameterized_gates": total_parameters,
            "two_qubit_gates": len([g for g in gates if len(g.qubits) == 2]),
            "multi_qubit_gates": len([g for g in gates if len(g.qubits) > 2])
        }
