"""
Quantum Gates Library
Comprehensive library of quantum gates and their matrix representations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import cmath
import json

# Import from the main quantum module
import sys
sys.path.append('..')
from backend.quantum.quantum_simulator import GateType

class GateFamily(Enum):
    """Categories of quantum gates"""
    PAULI = "pauli"
    ROTATION = "rotation"
    PHASE = "phase"
    HADAMARD = "hadamard"
    CONTROLLED = "controlled"
    SWAP = "swap"
    TOFFOLI = "toffoli"
    CUSTOM = "custom"

@dataclass
class GateProperties:
    """Properties and metadata for quantum gates"""
    name: str
    family: GateFamily
    num_qubits: int
    is_unitary: bool
    is_hermitian: bool
    is_involutory: bool  # Self-inverse
    commutes_with_pauli_x: bool
    commutes_with_pauli_y: bool
    commutes_with_pauli_z: bool
    eigenvalues: Optional[List[complex]]
    period: Optional[int]  # For periodic gates
    description: str

class QuantumGateLibrary:
    """Comprehensive quantum gate library with matrix representations"""
    
    def __init__(self):
        self._initialize_gates()
        self._initialize_properties()
    
    def _initialize_gates(self):
        """Initialize standard quantum gate matrices"""
        
        # Pauli gates
        self.gates = {
            GateType.PAULI_X: np.array([[0, 1], [1, 0]], dtype=complex),
            GateType.PAULI_Y: np.array([[0, -1j], [1j, 0]], dtype=complex),
            GateType.PAULI_Z: np.array([[1, 0], [0, -1]], dtype=complex),
        }
        
        # Hadamard gate
        sqrt2 = 1.0 / np.sqrt(2)
        self.gates[GateType.HADAMARD] = np.array([
            [sqrt2, sqrt2],
            [sqrt2, -sqrt2]
        ], dtype=complex)
        
        # Phase gates
        self.gates[GateType.PHASE] = np.array([[1, 0], [0, 1j]], dtype=complex)
        self.gates[GateType.T_GATE] = np.array([
            [1, 0],
            [0, np.exp(1j * np.pi / 4)]
        ], dtype=complex)
        
        # Two-qubit gates
        self.gates[GateType.CNOT] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        
        self.gates[GateType.CONTROLLED_Z] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        
        self.gates[GateType.SWAP] = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        
        # Three-qubit Toffoli gate
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        toffoli[7, 7] = 0
        self.gates[GateType.TOFFOLI] = toffoli
    
    def _initialize_properties(self):
        """Initialize gate properties and metadata"""
        self.properties = {
            GateType.PAULI_X: GateProperties(
                name="Pauli-X",
                family=GateFamily.PAULI,
                num_qubits=1,
                is_unitary=True,
                is_hermitian=True,
                is_involutory=True,
                commutes_with_pauli_x=True,
                commutes_with_pauli_y=False,
                commutes_with_pauli_z=False,
                eigenvalues=[1, -1],
                period=2,
                description="Bit-flip gate, rotates qubit around X-axis by π"
            ),
            
            GateType.PAULI_Y: GateProperties(
                name="Pauli-Y",
                family=GateFamily.PAULI,
                num_qubits=1,
                is_unitary=True,
                is_hermitian=True,
                is_involutory=True,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=True,
                commutes_with_pauli_z=False,
                eigenvalues=[1j, -1j],
                period=2,
                description="Bit and phase flip gate, rotates around Y-axis by π"
            ),
            
            GateType.PAULI_Z: GateProperties(
                name="Pauli-Z",
                family=GateFamily.PAULI,
                num_qubits=1,
                is_unitary=True,
                is_hermitian=True,
                is_involutory=True,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=False,
                commutes_with_pauli_z=True,
                eigenvalues=[1, -1],
                period=2,
                description="Phase-flip gate, rotates around Z-axis by π"
            ),
            
            GateType.HADAMARD: GateProperties(
                name="Hadamard",
                family=GateFamily.HADAMARD,
                num_qubits=1,
                is_unitary=True,
                is_hermitian=True,
                is_involutory=True,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=True,
                commutes_with_pauli_z=False,
                eigenvalues=[1, -1],
                period=2,
                description="Creates superposition, rotates around X+Z axis by π"
            ),
            
            GateType.PHASE: GateProperties(
                name="S (Phase)",
                family=GateFamily.PHASE,
                num_qubits=1,
                is_unitary=True,
                is_hermitian=False,
                is_involutory=False,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=False,
                commutes_with_pauli_z=True,
                eigenvalues=[1, 1j],
                period=4,
                description="Quarter-turn phase gate, rotates around Z-axis by π/2"
            ),
            
            GateType.T_GATE: GateProperties(
                name="T",
                family=GateFamily.PHASE,
                num_qubits=1,
                is_unitary=True,
                is_hermitian=False,
                is_involutory=False,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=False,
                commutes_with_pauli_z=True,
                eigenvalues=[1, np.exp(1j * np.pi / 4)],
                period=8,
                description="Eighth-turn phase gate, rotates around Z-axis by π/4"
            ),
            
            GateType.CNOT: GateProperties(
                name="CNOT",
                family=GateFamily.CONTROLLED,
                num_qubits=2,
                is_unitary=True,
                is_hermitian=True,
                is_involutory=True,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=False,
                commutes_with_pauli_z=False,
                eigenvalues=[1, 1, 1, -1],
                period=2,
                description="Controlled-X gate, flips target if control is |1⟩"
            ),
            
            GateType.CONTROLLED_Z: GateProperties(
                name="CZ",
                family=GateFamily.CONTROLLED,
                num_qubits=2,
                is_unitary=True,
                is_hermitian=True,
                is_involutory=True,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=False,
                commutes_with_pauli_z=True,
                eigenvalues=[1, 1, 1, -1],
                period=2,
                description="Controlled-Z gate, phase flip if both qubits are |1⟩"
            ),
            
            GateType.SWAP: GateProperties(
                name="SWAP",
                family=GateFamily.SWAP,
                num_qubits=2,
                is_unitary=True,
                is_hermitian=True,
                is_involutory=True,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=False,
                commutes_with_pauli_z=False,
                eigenvalues=[1, 1, 1, -1],
                period=2,
                description="Exchanges the states of two qubits"
            ),
            
            GateType.TOFFOLI: GateProperties(
                name="Toffoli (CCX)",
                family=GateFamily.TOFFOLI,
                num_qubits=3,
                is_unitary=True,
                is_hermitian=True,
                is_involutory=True,
                commutes_with_pauli_x=False,
                commutes_with_pauli_y=False,
                commutes_with_pauli_z=False,
                eigenvalues=[1, 1, 1, 1, 1, 1, 1, -1],
                period=2,
                description="Controlled-controlled-X gate, universal for classical computation"
            )
        }
    
    def get_matrix(self, gate_type: GateType, parameters: Optional[List[float]] = None) -> np.ndarray:
        """Get matrix representation of a gate"""
        if gate_type in self.gates:
            return self.gates[gate_type].copy()
        
        # Handle parameterized gates
        if gate_type == GateType.ROTATION_X:
            angle = parameters[0] if parameters else 0.0
            return self.rotation_x(angle)
        elif gate_type == GateType.ROTATION_Y:
            angle = parameters[0] if parameters else 0.0
            return self.rotation_y(angle)
        elif gate_type == GateType.ROTATION_Z:
            angle = parameters[0] if parameters else 0.0
            return self.rotation_z(angle)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
    
    def get_properties(self, gate_type: GateType) -> GateProperties:
        """Get properties of a gate"""
        return self.properties.get(gate_type)
    
    # Parameterized rotation gates
    def rotation_x(self, angle: float) -> np.ndarray:
        """X-rotation gate matrix"""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        return np.array([
            [cos_half, -1j * sin_half],
            [-1j * sin_half, cos_half]
        ], dtype=complex)
    
    def rotation_y(self, angle: float) -> np.ndarray:
        """Y-rotation gate matrix"""
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        return np.array([
            [cos_half, -sin_half],
            [sin_half, cos_half]
        ], dtype=complex)
    
    def rotation_z(self, angle: float) -> np.ndarray:
        """Z-rotation gate matrix"""
        return np.array([
            [np.exp(-1j * angle / 2), 0],
            [0, np.exp(1j * angle / 2)]
        ], dtype=complex)
    
    def phase_gate(self, angle: float) -> np.ndarray:
        """General phase gate matrix"""
        return np.array([
            [1, 0],
            [0, np.exp(1j * angle)]
        ], dtype=complex)
    
    def u_gate(self, theta: float, phi: float, lambda_: float) -> np.ndarray:
        """Universal single-qubit gate (U3)"""
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        
        return np.array([
            [cos_half, -np.exp(1j * lambda_) * sin_half],
            [np.exp(1j * phi) * sin_half, np.exp(1j * (phi + lambda_)) * cos_half]
        ], dtype=complex)
    
    # Advanced gates
    def sqrt_x(self) -> np.ndarray:
        """Square root of X gate"""
        return 0.5 * np.array([
            [1 + 1j, 1 - 1j],
            [1 - 1j, 1 + 1j]
        ], dtype=complex)
    
    def sqrt_y(self) -> np.ndarray:
        """Square root of Y gate"""
        return 0.5 * np.array([
            [1 + 1j, -1 - 1j],
            [1 + 1j, 1 + 1j]
        ], dtype=complex)
    
    def sqrt_z(self) -> np.ndarray:
        """Square root of Z gate (same as S gate)"""
        return self.gates[GateType.PHASE]
    
    def iswap(self) -> np.ndarray:
        """iSWAP gate"""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    def sqrt_iswap(self) -> np.ndarray:
        """Square root of iSWAP gate"""
        sqrt2_inv = 1.0 / np.sqrt(2)
        return np.array([
            [1, 0, 0, 0],
            [0, sqrt2_inv, 1j * sqrt2_inv, 0],
            [0, 1j * sqrt2_inv, sqrt2_inv, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    def fredkin(self) -> np.ndarray:
        """Fredkin (CSWAP) gate"""
        fredkin = np.eye(8, dtype=complex)
        fredkin[5, 5] = 0
        fredkin[5, 6] = 1
        fredkin[6, 5] = 1
        fredkin[6, 6] = 0
        return fredkin
    
    # Multi-controlled gates
    def controlled_gate(self, base_gate: np.ndarray, num_controls: int = 1) -> np.ndarray:
        """Create controlled version of a gate"""
        base_size = base_gate.shape[0]
        total_qubits = int(np.log2(base_size)) + num_controls
        total_size = 2 ** total_qubits
        
        controlled = np.eye(total_size, dtype=complex)
        
        # Apply base gate only when all control qubits are |1⟩
        control_mask = (2 ** num_controls - 1) << int(np.log2(base_size))
        
        for i in range(total_size):
            if (i & control_mask) == control_mask:
                target_bits = i & (base_size - 1)
                for j in range(base_size):
                    for k in range(base_size):
                        if base_gate[j, k] != 0:
                            row = (i & ~(base_size - 1)) | j
                            col = (i & ~(base_size - 1)) | k
                            controlled[row, col] = base_gate[j, k]
        
        return controlled
    
    # Gate analysis methods
    def is_unitary(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if matrix is unitary"""
        n = matrix.shape[0]
        product = matrix @ matrix.conj().T
        identity = np.eye(n)
        return np.allclose(product, identity, atol=tolerance)
    
    def is_hermitian(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """Check if matrix is Hermitian"""
        return np.allclose(matrix, matrix.conj().T, atol=tolerance)
    
    def gate_fidelity(self, gate1: np.ndarray, gate2: np.ndarray) -> float:
        """Calculate fidelity between two gates"""
        if gate1.shape != gate2.shape:
            raise ValueError("Gates must have the same dimensions")
        
        # Average fidelity for unitary gates
        n = gate1.shape[0]
        trace = np.trace(gate1.conj().T @ gate2)
        return (np.abs(trace) ** 2 + n) / (n * (n + 1))
    
    def decompose_single_qubit(self, matrix: np.ndarray) -> Tuple[float, float, float]:
        """Decompose single-qubit unitary into Euler angles (ZYZ)"""
        if matrix.shape != (2, 2):
            raise ValueError("Matrix must be 2x2 for single-qubit decomposition")
        
        # ZYZ decomposition: U = Rz(α) Ry(β) Rz(γ)
        # Extract angles from matrix elements
        alpha = np.angle(matrix[1, 1] / matrix[0, 0])
        beta = 2 * np.arctan2(np.abs(matrix[1, 0]), np.abs(matrix[0, 0]))
        gamma = np.angle(matrix[1, 0] / matrix[0, 1]) + alpha
        
        return alpha, beta, gamma
    
    def random_unitary(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate random unitary matrix"""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random complex matrix
        A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        
        # QR decomposition to get unitary matrix
        Q, R = np.linalg.qr(A)
        
        # Adjust phases
        phases = np.diag(R) / np.abs(np.diag(R))
        U = Q @ np.diag(phases)
        
        return U
    
    def pauli_decomposition(self, matrix: np.ndarray) -> Dict[str, complex]:
        """Decompose 2x2 matrix in Pauli basis"""
        if matrix.shape != (2, 2):
            raise ValueError("Matrix must be 2x2 for Pauli decomposition")
        
        # Pauli matrices
        I = np.eye(2, dtype=complex)
        X = self.gates[GateType.PAULI_X]
        Y = self.gates[GateType.PAULI_Y]
        Z = self.gates[GateType.PAULI_Z]
        
        # Coefficients
        coeff_I = np.trace(matrix @ I) / 2
        coeff_X = np.trace(matrix @ X) / 2
        coeff_Y = np.trace(matrix @ Y) / 2
        coeff_Z = np.trace(matrix @ Z) / 2
        
        return {
            'I': coeff_I,
            'X': coeff_X,
            'Y': coeff_Y,
            'Z': coeff_Z
        }
    
    def commutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculate commutator [A, B] = AB - BA"""
        return A @ B - B @ A
    
    def anticommutator(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Calculate anticommutator {A, B} = AB + BA"""
        return A @ B + B @ A
    
    def gate_power(self, gate: np.ndarray, power: float) -> np.ndarray:
        """Calculate gate raised to a power"""
        # Diagonalize gate
        eigenvals, eigenvecs = np.linalg.eig(gate)
        
        # Raise eigenvalues to power
        powered_eigenvals = eigenvals ** power
        
        # Reconstruct matrix
        return eigenvecs @ np.diag(powered_eigenvals) @ eigenvecs.conj().T
    
    def get_gate_info(self, gate_type: GateType) -> Dict[str, Any]:
        """Get comprehensive information about a gate"""
        matrix = self.get_matrix(gate_type)
        properties = self.get_properties(gate_type)
        
        if properties is None:
            return {"error": f"No information available for {gate_type}"}
        
        # Calculate derived properties
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        trace = np.trace(matrix)
        determinant = np.linalg.det(matrix)
        
        return {
            "name": properties.name,
            "family": properties.family.value,
            "description": properties.description,
            "matrix": matrix.tolist(),
            "properties": {
                "num_qubits": properties.num_qubits,
                "is_unitary": properties.is_unitary,
                "is_hermitian": properties.is_hermitian,
                "is_involutory": properties.is_involutory,
                "period": properties.period
            },
            "eigenvalues": eigenvals.tolist(),
            "eigenvectors": eigenvecs.tolist(),
            "trace": complex(trace),
            "determinant": complex(determinant),
            "commutation": {
                "pauli_x": properties.commutes_with_pauli_x,
                "pauli_y": properties.commutes_with_pauli_y,
                "pauli_z": properties.commutes_with_pauli_z
            }
        }
    
    def list_gates_by_family(self, family: GateFamily) -> List[GateType]:
        """List all gates in a specific family"""
        return [gate_type for gate_type, props in self.properties.items() 
                if props.family == family]
    
    def suggest_decomposition(self, target_matrix: np.ndarray) -> List[str]:
        """Suggest decomposition of target matrix using standard gates"""
        suggestions = []
        
        if target_matrix.shape == (2, 2):
            # Single-qubit gate
            if self.is_hermitian(target_matrix):
                suggestions.append("Hermitian gate - can be implemented as exp(-iHt)")
            
            # Check if it's close to standard gates
            for gate_type in [GateType.PAULI_X, GateType.PAULI_Y, GateType.PAULI_Z, GateType.HADAMARD]:
                standard_gate = self.get_matrix(gate_type)
                if np.allclose(target_matrix, standard_gate, atol=1e-6):
                    props = self.get_properties(gate_type)
                    suggestions.append(f"Exact match: {props.name}")
                    return suggestions
            
            # Try Euler angle decomposition
            try:
                alpha, beta, gamma = self.decompose_single_qubit(target_matrix)
                suggestions.append(f"ZYZ decomposition: Rz({alpha:.3f}) Ry({beta:.3f}) Rz({gamma:.3f})")
            except:
                suggestions.append("Complex single-qubit gate - consider universal gate set")
        
        elif target_matrix.shape == (4, 4):
            # Two-qubit gate
            if np.allclose(target_matrix, self.get_matrix(GateType.CNOT), atol=1e-6):
                suggestions.append("Exact match: CNOT gate")
            elif np.allclose(target_matrix, self.get_matrix(GateType.CONTROLLED_Z), atol=1e-6):
                suggestions.append("Exact match: Controlled-Z gate")
            elif np.allclose(target_matrix, self.get_matrix(GateType.SWAP), atol=1e-6):
                suggestions.append("Exact match: SWAP gate")
            else:
                suggestions.append("Two-qubit gate - consider KAK decomposition")
                suggestions.append("Can be decomposed using ~3 CNOTs and single-qubit gates")
        
        else:
            # Multi-qubit gate
            n_qubits = int(np.log2(target_matrix.shape[0]))
            suggestions.append(f"{n_qubits}-qubit gate - exponentially complex")
            suggestions.append("Consider if gate has special structure (e.g., tensor products)")
            suggestions.append("May require circuit synthesis algorithms")
        
        return suggestions if suggestions else ["No standard decomposition suggestions available"]

# Global gate library instance
gate_library = QuantumGateLibrary()

# Convenience functions
def get_gate_matrix(gate_type: GateType, parameters: Optional[List[float]] = None) -> np.ndarray:
    """Get matrix for a gate type"""
    return gate_library.get_matrix(gate_type, parameters)

def get_gate_info(gate_type: GateType) -> Dict[str, Any]:
    """Get comprehensive gate information"""
    return gate_library.get_gate_info(gate_type)

def rotation_gates(angles: List[float]) -> List[np.ndarray]:
    """Generate rotation gates for given angles"""
    return [
        gate_library.rotation_x(angles[0]),
        gate_library.rotation_y(angles[1]),
        gate_library.rotation_z(angles[2])
    ]

def pauli_group() -> List[np.ndarray]:
    """Get all Pauli group elements"""
    I = np.eye(2, dtype=complex)
    X = gate_library.get_matrix(GateType.PAULI_X)
    Y = gate_library.get_matrix(GateType.PAULI_Y)
    Z = gate_library.get_matrix(GateType.PAULI_Z)
    
    # Include phases
    return [
        I, -I, 1j*I, -1j*I,
        X, -X, 1j*X, -1j*X,
        Y, -Y, 1j*Y, -1j*Y,
        Z, -Z, 1j*Z, -1j*Z
    ]

def clifford_group_generators() -> List[np.ndarray]:
    """Get generators of single-qubit Clifford group"""
    return [
        gate_library.get_matrix(GateType.HADAMARD),
        gate_library.get_matrix(GateType.PHASE)
    ]
