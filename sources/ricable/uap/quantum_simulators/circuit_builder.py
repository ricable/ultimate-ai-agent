"""
Quantum Circuit Builder
Provides tools for constructing and manipulating quantum circuits.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# Import from the main quantum module
import sys
sys.path.append('..')
from backend.quantum.quantum_simulator import QuantumGate, GateType

class CircuitTemplate(Enum):
    """Predefined quantum circuit templates"""
    BELL_STATE = "bell_state"
    GHZ_STATE = "ghz_state"
    QRNG = "quantum_random_number_generator"
    TELEPORTATION = "quantum_teleportation"
    SUPERDENSE_CODING = "superdense_coding"
    GROVER_SEARCH = "grover_search"
    QFT = "quantum_fourier_transform"
    PHASE_ESTIMATION = "phase_estimation"
    SHOR_FACTORING = "shor_factoring"
    VARIATIONAL_ANSATZ = "variational_ansatz"

@dataclass
class CircuitMetrics:
    """Metrics for quantum circuit analysis"""
    total_gates: int
    depth: int
    qubit_count: int
    single_qubit_gates: int
    two_qubit_gates: int
    multi_qubit_gates: int
    rotation_gates: int
    entangling_gates: int
    parallelizable_layers: int
    estimated_fidelity: float
    complexity_score: float

class QuantumCircuitBuilder:
    """High-level quantum circuit builder with templates and optimization"""
    
    def __init__(self):
        self.gates: List[QuantumGate] = []
        self.num_qubits = 0
        self.metadata = {}
        self.named_registers = {}
        
    def reset(self) -> 'QuantumCircuitBuilder':
        """Reset the circuit builder"""
        self.gates = []
        self.num_qubits = 0
        self.metadata = {}
        self.named_registers = {}
        return self
    
    def add_qubit_register(self, name: str, size: int) -> 'QuantumCircuitBuilder':
        """Add a named qubit register"""
        start_idx = self.num_qubits
        self.named_registers[name] = list(range(start_idx, start_idx + size))
        self.num_qubits += size
        return self
    
    def get_register(self, name: str) -> List[int]:
        """Get qubit indices for a named register"""
        return self.named_registers.get(name, [])
    
    # Single-qubit gates
    def h(self, qubit: Union[int, List[int]]) -> 'QuantumCircuitBuilder':
        """Hadamard gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.HADAMARD, [q]))
        return self
    
    def x(self, qubit: Union[int, List[int]]) -> 'QuantumCircuitBuilder':
        """Pauli-X gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.PAULI_X, [q]))
        return self
    
    def y(self, qubit: Union[int, List[int]]) -> 'QuantumCircuitBuilder':
        """Pauli-Y gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.PAULI_Y, [q]))
        return self
    
    def z(self, qubit: Union[int, List[int]]) -> 'QuantumCircuitBuilder':
        """Pauli-Z gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.PAULI_Z, [q]))
        return self
    
    def s(self, qubit: Union[int, List[int]]) -> 'QuantumCircuitBuilder':
        """S (Phase) gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.PHASE, [q]))
        return self
    
    def t(self, qubit: Union[int, List[int]]) -> 'QuantumCircuitBuilder':
        """T gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.T_GATE, [q]))
        return self
    
    def rx(self, qubit: Union[int, List[int]], angle: float) -> 'QuantumCircuitBuilder':
        """X-rotation gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.ROTATION_X, [q], [angle]))
        return self
    
    def ry(self, qubit: Union[int, List[int]], angle: float) -> 'QuantumCircuitBuilder':
        """Y-rotation gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.ROTATION_Y, [q], [angle]))
        return self
    
    def rz(self, qubit: Union[int, List[int]], angle: float) -> 'QuantumCircuitBuilder':
        """Z-rotation gate"""
        qubits = [qubit] if isinstance(qubit, int) else qubit
        for q in qubits:
            self.gates.append(QuantumGate(GateType.ROTATION_Z, [q], [angle]))
        return self
    
    # Two-qubit gates
    def cnot(self, control: int, target: int) -> 'QuantumCircuitBuilder':
        """CNOT gate"""
        self.gates.append(QuantumGate(GateType.CNOT, [control, target]))
        return self
    
    def cx(self, control: int, target: int) -> 'QuantumCircuitBuilder':
        """CNOT gate (alias)"""
        return self.cnot(control, target)
    
    def cz(self, control: int, target: int) -> 'QuantumCircuitBuilder':
        """Controlled-Z gate"""
        self.gates.append(QuantumGate(GateType.CONTROLLED_Z, [control, target]))
        return self
    
    def swap(self, qubit1: int, qubit2: int) -> 'QuantumCircuitBuilder':
        """SWAP gate"""
        self.gates.append(QuantumGate(GateType.SWAP, [qubit1, qubit2]))
        return self
    
    # Multi-qubit gates
    def toffoli(self, control1: int, control2: int, target: int) -> 'QuantumCircuitBuilder':
        """Toffoli (CCX) gate"""
        self.gates.append(QuantumGate(GateType.TOFFOLI, [control1, control2, target]))
        return self
    
    def ccx(self, control1: int, control2: int, target: int) -> 'QuantumCircuitBuilder':
        """Toffoli gate (alias)"""
        return self.toffoli(control1, control2, target)
    
    # Circuit template methods
    def bell_state(self, qubits: List[int]) -> 'QuantumCircuitBuilder':
        """Create Bell state between two qubits"""
        if len(qubits) != 2:
            raise ValueError("Bell state requires exactly 2 qubits")
        
        self.h(qubits[0])
        self.cnot(qubits[0], qubits[1])
        return self
    
    def ghz_state(self, qubits: List[int]) -> 'QuantumCircuitBuilder':
        """Create GHZ state across multiple qubits"""
        if len(qubits) < 2:
            raise ValueError("GHZ state requires at least 2 qubits")
        
        self.h(qubits[0])
        for i in range(1, len(qubits)):
            self.cnot(qubits[0], qubits[i])
        return self
    
    def quantum_fourier_transform(self, qubits: List[int]) -> 'QuantumCircuitBuilder':
        """Apply Quantum Fourier Transform"""
        n = len(qubits)
        
        for i in range(n):
            self.h(qubits[i])
            for j in range(i + 1, n):
                angle = np.pi / (2 ** (j - i))
                # Controlled rotation (simplified)
                self.cnot(qubits[j], qubits[i])
                self.rz(qubits[i], angle)
                self.cnot(qubits[j], qubits[i])
        
        # Reverse qubit order
        for i in range(n // 2):
            self.swap(qubits[i], qubits[n - 1 - i])
        
        return self
    
    def grover_diffusion(self, qubits: List[int]) -> 'QuantumCircuitBuilder':
        """Apply Grover diffusion operator"""
        # H^⊗n
        self.h(qubits)
        
        # X^⊗n
        self.x(qubits)
        
        # Multi-controlled Z (simplified for demonstration)
        if len(qubits) > 1:
            # Use Toffoli chains for multi-controlled gates
            for i in range(len(qubits) - 1):
                if i == len(qubits) - 2:
                    self.cz(qubits[i], qubits[i + 1])
                else:
                    self.cnot(qubits[i], qubits[i + 1])
        
        # X^⊗n
        self.x(qubits)
        
        # H^⊗n
        self.h(qubits)
        
        return self
    
    def variational_layer(self, qubits: List[int], parameters: List[float]) -> 'QuantumCircuitBuilder':
        """Add a variational layer with parameterized gates"""
        if len(parameters) < len(qubits) * 3:
            raise ValueError(f"Need at least {len(qubits) * 3} parameters for variational layer")
        
        param_idx = 0
        
        # Parameterized single-qubit rotations
        for qubit in qubits:
            self.rx(qubit, parameters[param_idx])
            param_idx += 1
            self.ry(qubit, parameters[param_idx])
            param_idx += 1
            self.rz(qubit, parameters[param_idx])
            param_idx += 1
        
        # Entangling gates
        for i in range(len(qubits) - 1):
            self.cnot(qubits[i], qubits[i + 1])
        
        return self
    
    def random_circuit(self, qubits: List[int], depth: int, seed: Optional[int] = None) -> 'QuantumCircuitBuilder':
        """Generate random quantum circuit"""
        if seed is not None:
            np.random.seed(seed)
        
        single_qubit_gates = [self.h, self.x, self.y, self.z, self.s, self.t]
        
        for layer in range(depth):
            # Random single-qubit gates
            for qubit in qubits:
                if np.random.random() < 0.7:  # 70% chance to apply gate
                    gate_func = np.random.choice(single_qubit_gates)
                    gate_func(qubit)
            
            # Random two-qubit gates
            available_qubits = qubits.copy()
            np.random.shuffle(available_qubits)
            
            for i in range(0, len(available_qubits) - 1, 2):
                if np.random.random() < 0.5:  # 50% chance to apply CNOT
                    self.cnot(available_qubits[i], available_qubits[i + 1])
        
        return self
    
    def from_template(self, template: CircuitTemplate, qubits: List[int], **kwargs) -> 'QuantumCircuitBuilder':
        """Build circuit from predefined template"""
        if template == CircuitTemplate.BELL_STATE:
            return self.bell_state(qubits[:2])
        
        elif template == CircuitTemplate.GHZ_STATE:
            return self.ghz_state(qubits)
        
        elif template == CircuitTemplate.QFT:
            return self.quantum_fourier_transform(qubits)
        
        elif template == CircuitTemplate.QRNG:
            # Quantum random number generator
            self.h(qubits)
            return self
        
        elif template == CircuitTemplate.GROVER_SEARCH:
            iterations = kwargs.get('iterations', 1)
            # Initialize superposition
            self.h(qubits)
            
            for _ in range(iterations):
                # Oracle (simplified - marks last state)
                self.z(qubits[-1])
                # Diffusion
                self.grover_diffusion(qubits)
            
            return self
        
        elif template == CircuitTemplate.VARIATIONAL_ANSATZ:
            layers = kwargs.get('layers', 2)
            parameters = kwargs.get('parameters', np.random.uniform(0, 2*np.pi, len(qubits) * 3 * layers))
            
            for layer in range(layers):
                start_idx = layer * len(qubits) * 3
                layer_params = parameters[start_idx:start_idx + len(qubits) * 3]
                self.variational_layer(qubits, layer_params)
            
            return self
        
        else:
            raise ValueError(f"Template {template} not implemented")
    
    def barrier(self) -> 'QuantumCircuitBuilder':
        """Add a barrier (for visualization/organization)"""
        # Barriers don't affect simulation but useful for organization
        self.metadata.setdefault('barriers', []).append(len(self.gates))
        return self
    
    def measure_all(self) -> 'QuantumCircuitBuilder':
        """Mark all qubits for measurement"""
        self.metadata['measure_all'] = True
        return self
    
    def get_circuit(self) -> List[QuantumGate]:
        """Get the constructed circuit"""
        return self.gates.copy()
    
    def get_metrics(self) -> CircuitMetrics:
        """Calculate circuit metrics"""
        single_qubit = sum(1 for gate in self.gates if len(gate.qubits) == 1)
        two_qubit = sum(1 for gate in self.gates if len(gate.qubits) == 2)
        multi_qubit = sum(1 for gate in self.gates if len(gate.qubits) > 2)
        
        rotation_gates = sum(1 for gate in self.gates 
                           if gate.gate_type in [GateType.ROTATION_X, GateType.ROTATION_Y, GateType.ROTATION_Z])
        
        entangling_gates = sum(1 for gate in self.gates 
                             if gate.gate_type in [GateType.CNOT, GateType.CONTROLLED_Z, GateType.SWAP, GateType.TOFFOLI])
        
        depth = self._calculate_depth()
        parallelizable_layers = self._count_parallelizable_layers()
        estimated_fidelity = self._estimate_fidelity()
        complexity_score = self._calculate_complexity_score()
        
        return CircuitMetrics(
            total_gates=len(self.gates),
            depth=depth,
            qubit_count=self.num_qubits,
            single_qubit_gates=single_qubit,
            two_qubit_gates=two_qubit,
            multi_qubit_gates=multi_qubit,
            rotation_gates=rotation_gates,
            entangling_gates=entangling_gates,
            parallelizable_layers=parallelizable_layers,
            estimated_fidelity=estimated_fidelity,
            complexity_score=complexity_score
        )
    
    def _calculate_depth(self) -> int:
        """Calculate circuit depth"""
        if not self.gates:
            return 0
        
        qubit_times = [0] * self.num_qubits
        
        for gate in self.gates:
            max_time = max(qubit_times[q] for q in gate.qubits if q < len(qubit_times))
            for qubit in gate.qubits:
                if qubit < len(qubit_times):
                    qubit_times[qubit] = max_time + 1
        
        return max(qubit_times) if qubit_times else 0
    
    def _count_parallelizable_layers(self) -> int:
        """Count parallelizable gate layers"""
        if not self.gates:
            return 0
        
        layers = []
        current_layer = set()
        
        for gate in self.gates:
            gate_qubits = set(gate.qubits)
            
            # Check if gate conflicts with current layer
            if any(qubit in current_layer for qubit in gate_qubits):
                # Start new layer
                layers.append(current_layer)
                current_layer = gate_qubits
            else:
                # Add to current layer
                current_layer.update(gate_qubits)
        
        if current_layer:
            layers.append(current_layer)
        
        return len(layers)
    
    def _estimate_fidelity(self) -> float:
        """Estimate circuit fidelity based on gate count and types"""
        if not self.gates:
            return 1.0
        
        # Simple fidelity model
        single_qubit_error = 0.001  # 0.1% error per single-qubit gate
        two_qubit_error = 0.01     # 1% error per two-qubit gate
        
        total_error = 0.0
        for gate in self.gates:
            if len(gate.qubits) == 1:
                total_error += single_qubit_error
            elif len(gate.qubits) == 2:
                total_error += two_qubit_error
            else:
                total_error += two_qubit_error * len(gate.qubits)  # Approximate for multi-qubit
        
        return max(0.0, 1.0 - total_error)
    
    def _calculate_complexity_score(self) -> float:
        """Calculate circuit complexity score"""
        if not self.gates:
            return 0.0
        
        # Weighted complexity based on gate types
        complexity = 0.0
        
        for gate in self.gates:
            if len(gate.qubits) == 1:
                complexity += 1.0
            elif len(gate.qubits) == 2:
                complexity += 3.0
            else:
                complexity += 2.0 ** len(gate.qubits)  # Exponential for multi-qubit
        
        # Normalize by circuit depth
        depth = self._calculate_depth()
        return complexity / max(1, depth)
    
    def visualize_text(self) -> str:
        """Create text-based circuit visualization"""
        if not self.gates:
            return "Empty circuit"
        
        lines = []
        lines.append(f"Quantum Circuit ({self.num_qubits} qubits, {len(self.gates)} gates)")
        lines.append("=" * 50)
        
        for i, gate in enumerate(self.gates):
            gate_str = f"{i:3d}: {gate.gate_type.value}"
            
            if len(gate.qubits) == 1:
                gate_str += f" q[{gate.qubits[0]}]"
            elif len(gate.qubits) == 2:
                gate_str += f" q[{gate.qubits[0]}], q[{gate.qubits[1]}]"
            else:
                qubit_str = ", ".join(f"q[{q}]" for q in gate.qubits)
                gate_str += f" {qubit_str}"
            
            if gate.parameters:
                param_str = ", ".join(f"{p:.3f}" for p in gate.parameters)
                gate_str += f" ({param_str})"
            
            lines.append(gate_str)
        
        return "\n".join(lines)
    
    def to_qasm(self) -> str:
        """Export circuit to OpenQASM format (simplified)"""
        lines = []
        lines.append("OPENQASM 2.0;")
        lines.append('include "qelib1.inc";')
        lines.append(f"qreg q[{self.num_qubits}];")
        lines.append(f"creg c[{self.num_qubits}];")
        lines.append("")
        
        for gate in self.gates:
            if gate.gate_type == GateType.HADAMARD:
                lines.append(f"h q[{gate.qubits[0]}];")
            elif gate.gate_type == GateType.PAULI_X:
                lines.append(f"x q[{gate.qubits[0]}];")
            elif gate.gate_type == GateType.PAULI_Y:
                lines.append(f"y q[{gate.qubits[0]}];")
            elif gate.gate_type == GateType.PAULI_Z:
                lines.append(f"z q[{gate.qubits[0]}];")
            elif gate.gate_type == GateType.CNOT:
                lines.append(f"cx q[{gate.qubits[0]}],q[{gate.qubits[1]}];")
            elif gate.gate_type == GateType.ROTATION_X:
                angle = gate.parameters[0] if gate.parameters else 0
                lines.append(f"rx({angle}) q[{gate.qubits[0]}];")
            elif gate.gate_type == GateType.ROTATION_Y:
                angle = gate.parameters[0] if gate.parameters else 0
                lines.append(f"ry({angle}) q[{gate.qubits[0]}];")
            elif gate.gate_type == GateType.ROTATION_Z:
                angle = gate.parameters[0] if gate.parameters else 0
                lines.append(f"rz({angle}) q[{gate.qubits[0]}];")
        
        if self.metadata.get('measure_all'):
            lines.append(f"measure q -> c;")
        
        return "\n".join(lines)
    
    def save_circuit(self, filename: str) -> None:
        """Save circuit to JSON file"""
        circuit_data = {
            "metadata": {
                "created": datetime.utcnow().isoformat(),
                "num_qubits": self.num_qubits,
                "num_gates": len(self.gates),
                "named_registers": self.named_registers,
                **self.metadata
            },
            "gates": [
                {
                    "type": gate.gate_type.value,
                    "qubits": gate.qubits,
                    "parameters": gate.parameters,
                    "name": gate.name
                }
                for gate in self.gates
            ],
            "metrics": {
                **self.get_metrics().__dict__
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(circuit_data, f, indent=2)
    
    @classmethod
    def load_circuit(cls, filename: str) -> 'QuantumCircuitBuilder':
        """Load circuit from JSON file"""
        with open(filename, 'r') as f:
            circuit_data = json.load(f)
        
        builder = cls()
        builder.num_qubits = circuit_data["metadata"]["num_qubits"]
        builder.named_registers = circuit_data["metadata"].get("named_registers", {})
        builder.metadata = {k: v for k, v in circuit_data["metadata"].items() 
                          if k not in ["num_qubits", "named_registers", "created", "num_gates"]}
        
        for gate_data in circuit_data["gates"]:
            gate_type = GateType(gate_data["type"])
            qubits = gate_data["qubits"]
            parameters = gate_data.get("parameters")
            name = gate_data.get("name")
            
            gate = QuantumGate(gate_type, qubits, parameters, name)
            builder.gates.append(gate)
        
        return builder

# Convenience functions for quick circuit creation
def bell_pair() -> QuantumCircuitBuilder:
    """Create Bell pair circuit"""
    return QuantumCircuitBuilder().add_qubit_register("main", 2).bell_state([0, 1])

def ghz_state(n_qubits: int) -> QuantumCircuitBuilder:
    """Create GHZ state circuit"""
    qubits = list(range(n_qubits))
    return QuantumCircuitBuilder().add_qubit_register("main", n_qubits).ghz_state(qubits)

def random_circuit(n_qubits: int, depth: int, seed: Optional[int] = None) -> QuantumCircuitBuilder:
    """Create random quantum circuit"""
    qubits = list(range(n_qubits))
    return QuantumCircuitBuilder().add_qubit_register("main", n_qubits).random_circuit(qubits, depth, seed)

def qft_circuit(n_qubits: int) -> QuantumCircuitBuilder:
    """Create Quantum Fourier Transform circuit"""
    qubits = list(range(n_qubits))
    return QuantumCircuitBuilder().add_qubit_register("main", n_qubits).quantum_fourier_transform(qubits)
