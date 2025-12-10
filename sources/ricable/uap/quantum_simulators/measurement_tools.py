"""
Quantum Measurement Tools
Provides tools for quantum state measurement and analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
from collections import Counter

# Import from the main quantum module
import sys
sys.path.append('..')
from backend.quantum.quantum_simulator import QuantumState, MeasurementResult

class MeasurementBasis(Enum):
    """Different measurement bases"""
    COMPUTATIONAL = "computational"  # Z basis |0⟩, |1⟩
    HADAMARD = "hadamard"            # X basis |+⟩, |-⟩
    CIRCULAR = "circular"            # Y basis |i⟩, |-i⟩
    PAULI_X = "pauli_x"
    PAULI_Y = "pauli_y"
    PAULI_Z = "pauli_z"
    CUSTOM = "custom"

class ObservableType(Enum):
    """Types of quantum observables"""
    PAULI_SINGLE = "pauli_single"
    PAULI_STRING = "pauli_string"
    TENSOR_PRODUCT = "tensor_product"
    HERMITIAN = "hermitian"
    PROJECTOR = "projector"

@dataclass
class Observable:
    """Quantum observable definition"""
    matrix: np.ndarray
    name: str
    obs_type: ObservableType
    qubits: List[int]
    eigenvalues: Optional[List[float]] = None
    eigenvectors: Optional[List[np.ndarray]] = None
    description: str = ""

@dataclass
class MeasurementStatistics:
    """Statistics from quantum measurements"""
    mean: float
    variance: float
    standard_deviation: float
    min_value: float
    max_value: float
    histogram: Dict[str, int]
    confidence_intervals: Dict[float, Tuple[float, float]]  # confidence level -> (lower, upper)
    sample_size: int

@dataclass
class EntanglementMeasure:
    """Entanglement quantification results"""
    concurrence: Optional[float]
    entanglement_entropy: Optional[float]
    negativity: Optional[float]
    schmidt_coefficients: Optional[List[float]]
    entanglement_of_formation: Optional[float]
    is_separable: bool
    is_entangled: bool

class QuantumMeasurementTools:
    """Comprehensive quantum measurement and analysis toolkit"""
    
    def __init__(self):
        self._initialize_bases()
        self._initialize_observables()
    
    def _initialize_bases(self):
        """Initialize measurement bases"""
        # Computational basis (Z)
        self.bases = {
            MeasurementBasis.COMPUTATIONAL: {
                'vectors': [np.array([1, 0]), np.array([0, 1])],
                'labels': ['0', '1'],
                'matrix': np.array([[1, 0], [0, -1]])
            },
            
            # Hadamard basis (X)
            MeasurementBasis.HADAMARD: {
                'vectors': [np.array([1, 1])/np.sqrt(2), np.array([1, -1])/np.sqrt(2)],
                'labels': ['+', '-'],
                'matrix': np.array([[0, 1], [1, 0]])
            },
            
            # Circular basis (Y)
            MeasurementBasis.CIRCULAR: {
                'vectors': [np.array([1, 1j])/np.sqrt(2), np.array([1, -1j])/np.sqrt(2)],
                'labels': ['i', '-i'],
                'matrix': np.array([[0, -1j], [1j, 0]])
            }
        }
    
    def _initialize_observables(self):
        """Initialize common quantum observables"""
        # Pauli matrices
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        self.standard_observables = {
            'X': Observable(pauli_x, "Pauli-X", ObservableType.PAULI_SINGLE, [0], 
                          eigenvalues=[1, -1], description="X Pauli operator"),
            'Y': Observable(pauli_y, "Pauli-Y", ObservableType.PAULI_SINGLE, [0], 
                          eigenvalues=[1, -1], description="Y Pauli operator"),
            'Z': Observable(pauli_z, "Pauli-Z", ObservableType.PAULI_SINGLE, [0], 
                          eigenvalues=[1, -1], description="Z Pauli operator"),
        }
    
    async def measure_state(self, state: QuantumState, basis: MeasurementBasis = MeasurementBasis.COMPUTATIONAL,
                          qubits: Optional[List[int]] = None, shots: int = 1000) -> List[MeasurementResult]:
        """Perform measurements on quantum state"""
        if qubits is None:
            qubits = list(range(state.num_qubits))
        
        measurements = []
        
        for shot in range(shots):
            shot_measurements = []
            current_state = state
            
            for qubit in qubits:
                outcome, probability = await self._measure_qubit_in_basis(
                    current_state, qubit, basis
                )
                
                measurement = MeasurementResult(
                    qubit=qubit,
                    outcome=outcome,
                    probability=probability,
                    timestamp=datetime.utcnow()
                )
                shot_measurements.append(measurement)
                
                # Update state after measurement (collapse)
                current_state = self._collapse_state_after_measurement(
                    current_state, qubit, outcome, basis
                )
            
            measurements.extend(shot_measurements)
        
        return measurements
    
    async def _measure_qubit_in_basis(self, state: QuantumState, qubit: int, 
                                    basis: MeasurementBasis) -> Tuple[int, float]:
        """Measure single qubit in specified basis"""
        if basis == MeasurementBasis.COMPUTATIONAL:
            return state.measure_qubit(qubit)
        
        # For other bases, need to transform state
        if basis == MeasurementBasis.HADAMARD:
            # Apply H† to transform to computational basis
            transformed_state = self._apply_basis_rotation(state, qubit, 'H_dagger')
            return transformed_state.measure_qubit(qubit)
        
        elif basis == MeasurementBasis.CIRCULAR:
            # Apply transformation for Y measurement
            transformed_state = self._apply_basis_rotation(state, qubit, 'Y_to_Z')
            return transformed_state.measure_qubit(qubit)
        
        else:
            # Default to computational basis
            return state.measure_qubit(qubit)
    
    def _apply_basis_rotation(self, state: QuantumState, qubit: int, rotation: str) -> QuantumState:
        """Apply basis rotation for measurement"""
        # Simplified - in practice would apply proper rotations
        return state
    
    def _collapse_state_after_measurement(self, state: QuantumState, qubit: int, 
                                        outcome: int, basis: MeasurementBasis) -> QuantumState:
        """Collapse state after measurement"""
        # Simplified - return original state for now
        return state
    
    async def expectation_value(self, state: QuantumState, observable: Observable) -> complex:
        """Calculate expectation value of observable"""
        if state.is_mixed:
            # For mixed states: Tr(ρ O)
            return np.trace(state.density_matrix @ observable.matrix)
        else:
            # For pure states: ⟨ψ|O|ψ⟩
            return np.conj(state.state_vector) @ observable.matrix @ state.state_vector
    
    async def variance(self, state: QuantumState, observable: Observable) -> float:
        """Calculate variance of observable"""
        exp_val = await self.expectation_value(state, observable)
        exp_val_squared = await self.expectation_value(state, 
            Observable(observable.matrix @ observable.matrix, 
                     f"{observable.name}^2", observable.obs_type, observable.qubits))
        
        variance = np.real(exp_val_squared - exp_val * np.conj(exp_val))
        return max(0.0, variance)  # Ensure non-negative
    
    async def uncertainty(self, state: QuantumState, observable: Observable) -> float:
        """Calculate uncertainty (standard deviation) of observable"""
        var = await self.variance(state, observable)
        return np.sqrt(var)
    
    async def heisenberg_uncertainty(self, state: QuantumState, 
                                   obs1: Observable, obs2: Observable) -> Tuple[float, float, float]:
        """Calculate Heisenberg uncertainty relation"""
        # Calculate uncertainties
        sigma1 = await self.uncertainty(state, obs1)
        sigma2 = await self.uncertainty(state, obs2)
        
        # Calculate commutator expectation value
        commutator = obs1.matrix @ obs2.matrix - obs2.matrix @ obs1.matrix
        comm_obs = Observable(commutator, f"[{obs1.name}, {obs2.name}]", 
                            ObservableType.HERMITIAN, obs1.qubits + obs2.qubits)
        comm_exp = await self.expectation_value(state, comm_obs)
        
        # Uncertainty product
        uncertainty_product = sigma1 * sigma2
        
        # Lower bound
        lower_bound = 0.5 * np.abs(np.imag(comm_exp))
        
        return uncertainty_product, lower_bound, uncertainty_product / lower_bound if lower_bound > 0 else float('inf')
    
    def analyze_measurement_statistics(self, measurements: List[MeasurementResult], 
                                     observable_eigenvalues: Optional[List[float]] = None) -> MeasurementStatistics:
        """Analyze measurement results and calculate statistics"""
        if not measurements:
            raise ValueError("No measurements provided")
        
        # Extract outcomes
        if observable_eigenvalues:
            # Map binary outcomes to eigenvalues
            values = [observable_eigenvalues[m.outcome] for m in measurements]
        else:
            # Use raw outcomes
            values = [m.outcome for m in measurements]
        
        # Calculate statistics
        values_array = np.array(values)
        mean = np.mean(values_array)
        variance = np.var(values_array)
        std_dev = np.std(values_array)
        min_val = np.min(values_array)
        max_val = np.max(values_array)
        
        # Create histogram
        outcome_counts = Counter([str(m.outcome) for m in measurements])
        
        # Calculate confidence intervals (assuming normal distribution)
        confidence_intervals = {}
        n = len(values)
        sem = std_dev / np.sqrt(n)  # Standard error of mean
        
        for confidence in [0.68, 0.95, 0.99]:  # 1σ, 2σ, 3σ
            z_score = {0.68: 1.0, 0.95: 1.96, 0.99: 2.58}[confidence]
            margin = z_score * sem
            confidence_intervals[confidence] = (mean - margin, mean + margin)
        
        return MeasurementStatistics(
            mean=float(mean),
            variance=float(variance),
            standard_deviation=float(std_dev),
            min_value=float(min_val),
            max_value=float(max_val),
            histogram=dict(outcome_counts),
            confidence_intervals=confidence_intervals,
            sample_size=len(measurements)
        )
    
    def create_pauli_observable(self, pauli_string: str, qubits: List[int]) -> Observable:
        """Create observable from Pauli string"""
        if len(pauli_string) != len(qubits):
            raise ValueError("Pauli string length must match number of qubits")
        
        # Pauli matrices
        pauli_matrices = {
            'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
        
        # Build tensor product
        result = np.array([[1]], dtype=complex)
        for pauli_char in pauli_string:
            if pauli_char not in pauli_matrices:
                raise ValueError(f"Invalid Pauli character: {pauli_char}")
            result = np.kron(result, pauli_matrices[pauli_char])
        
        return Observable(
            matrix=result,
            name=f"Pauli({pauli_string})",
            obs_type=ObservableType.PAULI_STRING,
            qubits=qubits,
            eigenvalues=None,  # Would need to calculate
            description=f"Pauli string observable: {pauli_string}"
        )
    
    def measure_entanglement(self, state: QuantumState, subsystem_qubits: List[int]) -> EntanglementMeasure:
        """Measure entanglement between subsystems"""
        if state.num_qubits < 2:
            return EntanglementMeasure(
                concurrence=0.0, entanglement_entropy=0.0, negativity=0.0,
                schmidt_coefficients=[1.0], entanglement_of_formation=0.0,
                is_separable=True, is_entangled=False
            )
        
        # Get density matrix
        if state.is_mixed:
            rho = state.density_matrix
        else:
            rho = np.outer(state.state_vector, np.conj(state.state_vector))
        
        # For simplicity, measure entanglement for 2-qubit case
        if state.num_qubits == 2 and len(subsystem_qubits) == 1:
            return self._measure_2qubit_entanglement(rho)
        
        # For general case, calculate entanglement entropy
        entanglement_entropy = self._calculate_entanglement_entropy(rho, subsystem_qubits, state.num_qubits)
        
        # Schmidt decomposition for bipartite case
        schmidt_coeffs = self._schmidt_decomposition(state.state_vector, subsystem_qubits, state.num_qubits)
        
        is_entangled = entanglement_entropy > 1e-10
        
        return EntanglementMeasure(
            concurrence=None,  # Not implemented for general case
            entanglement_entropy=entanglement_entropy,
            negativity=None,
            schmidt_coefficients=schmidt_coeffs,
            entanglement_of_formation=None,
            is_separable=not is_entangled,
            is_entangled=is_entangled
        )
    
    def _measure_2qubit_entanglement(self, rho: np.ndarray) -> EntanglementMeasure:
        """Measure entanglement for 2-qubit systems"""
        # Calculate concurrence
        # Flip matrix
        sigma_y = np.array([[0, -1j], [1j, 0]])
        flip = np.kron(sigma_y, sigma_y)
        
        # Spin-flipped density matrix
        rho_tilde = flip @ np.conj(rho) @ flip
        
        # Product matrix
        R = rho @ rho_tilde
        
        # Eigenvalues in decreasing order
        eigenvals = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
        sqrt_eigenvals = np.sqrt(np.maximum(eigenvals, 0))
        
        # Concurrence
        concurrence = max(0, sqrt_eigenvals[0] - sqrt_eigenvals[1] - sqrt_eigenvals[2] - sqrt_eigenvals[3])
        
        # Entanglement of formation
        if concurrence > 0:
            h = lambda x: -x * np.log2(x) - (1-x) * np.log2(1-x) if 0 < x < 1 else 0
            eof = h((1 + np.sqrt(1 - concurrence**2)) / 2)
        else:
            eof = 0
        
        # Entanglement entropy (von Neumann entropy of reduced state)
        rho_A = self._partial_trace(rho, [1])  # Trace out second qubit
        eigenvals_A = np.real(np.linalg.eigvals(rho_A))
        eigenvals_A = eigenvals_A[eigenvals_A > 1e-12]  # Remove numerical zeros
        entanglement_entropy = -np.sum(eigenvals_A * np.log2(eigenvals_A))
        
        is_entangled = concurrence > 1e-10
        
        return EntanglementMeasure(
            concurrence=float(concurrence),
            entanglement_entropy=float(entanglement_entropy),
            negativity=None,  # Could implement
            schmidt_coefficients=None,
            entanglement_of_formation=float(eof),
            is_separable=not is_entangled,
            is_entangled=is_entangled
        )
    
    def _calculate_entanglement_entropy(self, rho: np.ndarray, subsystem_qubits: List[int], 
                                      total_qubits: int) -> float:
        """Calculate entanglement entropy"""
        # Partial trace to get reduced density matrix
        complement_qubits = [i for i in range(total_qubits) if i not in subsystem_qubits]
        rho_reduced = self._partial_trace(rho, complement_qubits)
        
        # Calculate von Neumann entropy
        eigenvals = np.real(np.linalg.eigvals(rho_reduced))
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
        
        if len(eigenvals) == 0:
            return 0.0
        
        entropy = -np.sum(eigenvals * np.log2(eigenvals))
        return float(entropy)
    
    def _partial_trace(self, rho: np.ndarray, trace_qubits: List[int]) -> np.ndarray:
        """Compute partial trace over specified qubits"""
        # Simplified implementation for demonstration
        # In practice, would use more efficient tensor manipulation
        
        total_qubits = int(np.log2(rho.shape[0]))
        
        if not trace_qubits:
            return rho
        
        # For 2-qubit case (simplified)
        if total_qubits == 2 and len(trace_qubits) == 1:
            if trace_qubits[0] == 0:
                # Trace out first qubit
                return np.array([
                    [rho[0, 0] + rho[2, 2], rho[0, 1] + rho[2, 3]],
                    [rho[1, 0] + rho[3, 2], rho[1, 1] + rho[3, 3]]
                ])
            else:
                # Trace out second qubit
                return np.array([
                    [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
                    [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]]
                ])
        
        # General case would require more complex implementation
        return rho
    
    def _schmidt_decomposition(self, state_vector: np.ndarray, subsystem_qubits: List[int], 
                             total_qubits: int) -> List[float]:
        """Perform Schmidt decomposition"""
        # Reshape state vector into matrix
        subsys_dim = 2 ** len(subsystem_qubits)
        complement_dim = 2 ** (total_qubits - len(subsystem_qubits))
        
        # Simplified for demonstration
        if total_qubits == 2:
            state_matrix = state_vector.reshape(2, 2)
            U, s, Vh = np.linalg.svd(state_matrix)
            return s.tolist()
        
        # For general case, return uniform coefficients
        return [1.0 / np.sqrt(min(subsys_dim, complement_dim))] * min(subsys_dim, complement_dim)
    
    def quantum_state_tomography(self, measurements: Dict[str, List[MeasurementResult]], 
                               num_qubits: int) -> np.ndarray:
        """Reconstruct density matrix from measurement data"""
        # Simplified quantum state tomography
        # In practice, would use maximum likelihood estimation or linear inversion
        
        dim = 2 ** num_qubits
        rho = np.zeros((dim, dim), dtype=complex)
        
        # For single qubit case
        if num_qubits == 1:
            # Assume measurements in X, Y, Z bases
            pauli_expectations = {}
            
            for basis_name, results in measurements.items():
                if results:
                    # Calculate expectation value
                    outcomes = [r.outcome for r in results]
                    # Map 0,1 to +1,-1
                    mapped_outcomes = [1 if o == 0 else -1 for o in outcomes]
                    expectation = np.mean(mapped_outcomes)
                    pauli_expectations[basis_name] = expectation
            
            # Reconstruct density matrix
            I = np.eye(2)
            X = np.array([[0, 1], [1, 0]])
            Y = np.array([[0, -1j], [1j, 0]])
            Z = np.array([[1, 0], [0, -1]])
            
            exp_x = pauli_expectations.get('X', 0)
            exp_y = pauli_expectations.get('Y', 0)
            exp_z = pauli_expectations.get('Z', 0)
            
            rho = 0.5 * (I + exp_x * X + exp_y * Y + exp_z * Z)
        
        else:
            # For multi-qubit case, return maximally mixed state
            rho = np.eye(dim) / dim
        
        return rho
    
    def fidelity(self, state1: QuantumState, state2: QuantumState) -> float:
        """Calculate fidelity between two quantum states"""
        if state1.is_mixed or state2.is_mixed:
            # Mixed state fidelity
            rho1 = state1.density_matrix if state1.is_mixed else state1.to_density_matrix()
            rho2 = state2.density_matrix if state2.is_mixed else state2.to_density_matrix()
            
            # Fidelity: F = Tr(sqrt(sqrt(rho1) * rho2 * sqrt(rho1)))
            sqrt_rho1 = self._matrix_sqrt(rho1)
            inner = sqrt_rho1 @ rho2 @ sqrt_rho1
            sqrt_inner = self._matrix_sqrt(inner)
            fidelity = np.real(np.trace(sqrt_inner))
        else:
            # Pure state fidelity
            overlap = np.abs(np.vdot(state1.state_vector, state2.state_vector))
            fidelity = overlap ** 2
        
        return float(np.clip(fidelity, 0, 1))
    
    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix square root"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        # Ensure non-negative eigenvalues
        eigenvals = np.maximum(eigenvals, 0)
        sqrt_eigenvals = np.sqrt(eigenvals)
        return eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.conj().T
    
    def trace_distance(self, state1: QuantumState, state2: QuantumState) -> float:
        """Calculate trace distance between two quantum states"""
        rho1 = state1.density_matrix if state1.is_mixed else state1.to_density_matrix()
        rho2 = state2.density_matrix if state2.is_mixed else state2.to_density_matrix()
        
        diff = rho1 - rho2
        # Trace distance = 0.5 * Tr(|rho1 - rho2|)
        eigenvals = np.linalg.eigvals(diff @ diff.conj().T)
        trace_norm = np.sum(np.sqrt(np.real(eigenvals)))
        
        return float(0.5 * trace_norm)
    
    def purity(self, state: QuantumState) -> float:
        """Calculate purity of quantum state"""
        if state.is_mixed:
            rho = state.density_matrix
        else:
            rho = state.to_density_matrix()
        
        purity = np.real(np.trace(rho @ rho))
        return float(purity)
    
    def linear_entropy(self, state: QuantumState) -> float:
        """Calculate linear entropy"""
        p = self.purity(state)
        dimension = 2 ** state.num_qubits
        return float((dimension / (dimension - 1)) * (1 - p))
    
    def generate_measurement_report(self, measurements: List[MeasurementResult], 
                                  state: Optional[QuantumState] = None) -> Dict[str, Any]:
        """Generate comprehensive measurement analysis report"""
        if not measurements:
            return {"error": "No measurements provided"}
        
        # Basic statistics
        stats = self.analyze_measurement_statistics(measurements, [-1, 1])  # Assume Pauli eigenvalues
        
        # Group measurements by qubit
        qubit_measurements = {}
        for m in measurements:
            if m.qubit not in qubit_measurements:
                qubit_measurements[m.qubit] = []
            qubit_measurements[m.qubit].append(m)
        
        qubit_stats = {}
        for qubit, qubit_meas in qubit_measurements.items():
            qubit_stats[qubit] = self.analyze_measurement_statistics(qubit_meas, [0, 1])
        
        report = {
            "measurement_summary": {
                "total_measurements": len(measurements),
                "unique_qubits": len(qubit_measurements),
                "measurement_time_span": {
                    "start": min(m.timestamp for m in measurements).isoformat(),
                    "end": max(m.timestamp for m in measurements).isoformat()
                }
            },
            "overall_statistics": {
                "mean": stats.mean,
                "variance": stats.variance,
                "standard_deviation": stats.standard_deviation,
                "histogram": stats.histogram,
                "confidence_intervals": stats.confidence_intervals
            },
            "per_qubit_statistics": {
                str(qubit): {
                    "measurement_count": qstats.sample_size,
                    "mean": qstats.mean,
                    "variance": qstats.variance,
                    "histogram": qstats.histogram
                }
                for qubit, qstats in qubit_stats.items()
            }
        }
        
        # Add state analysis if provided
        if state is not None:
            report["state_analysis"] = {
                "purity": self.purity(state),
                "linear_entropy": self.linear_entropy(state),
                "is_mixed": state.is_mixed,
                "num_qubits": state.num_qubits
            }
            
            # Entanglement analysis for multi-qubit states
            if state.num_qubits > 1:
                entanglement = self.measure_entanglement(state, [0])  # Measure with respect to first qubit
                report["entanglement_analysis"] = {
                    "is_entangled": entanglement.is_entangled,
                    "entanglement_entropy": entanglement.entanglement_entropy,
                    "concurrence": entanglement.concurrence,
                    "schmidt_coefficients": entanglement.schmidt_coefficients
                }
        
        report["timestamp"] = datetime.utcnow().isoformat()
        return report

# Convenience functions
def quick_measurement_analysis(measurements: List[MeasurementResult]) -> Dict[str, float]:
    """Quick statistical analysis of measurements"""
    tools = QuantumMeasurementTools()
    stats = tools.analyze_measurement_statistics(measurements)
    
    return {
        "mean": stats.mean,
        "std_dev": stats.standard_deviation,
        "sample_size": stats.sample_size,
        "outcome_0_prob": stats.histogram.get('0', 0) / stats.sample_size,
        "outcome_1_prob": stats.histogram.get('1', 0) / stats.sample_size
    }

def calculate_bell_state_fidelity(measurements: List[MeasurementResult]) -> float:
    """Calculate fidelity with ideal Bell state from measurements"""
    # Simplified calculation based on correlation measurements
    if len(measurements) < 2:
        return 0.0
    
    # Group measurements by pairs
    pairs = [(measurements[i], measurements[i+1]) for i in range(0, len(measurements)-1, 2)]
    
    # Count correlations
    same_outcomes = sum(1 for m1, m2 in pairs if m1.outcome == m2.outcome)
    total_pairs = len(pairs)
    
    if total_pairs == 0:
        return 0.0
    
    # For Bell state, expect 100% correlation in Z basis
    correlation = same_outcomes / total_pairs
    
    # Rough fidelity estimate
    return correlation
