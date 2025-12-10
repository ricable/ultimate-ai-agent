# QuantumDAG Integration Architecture

## Overview

This document describes the integration between `agentic-jujutsu`'s AgentCoordination module (Rust) and `@qudag/napi-core` QuantumDAG (JavaScript/N-API).

## Architecture

### Layered Design

```
┌─────────────────────────────────────┐
│   JavaScript Application Layer      │
│  (Uses QuantumBridge for coord)     │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│      QuantumBridge (JavaScript)     │
│  - Creates QuantumDAG instances     │
│  - Manages vertex/edge operations   │
│  - Bridges to Rust coordination     │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│  AgentCoordination (Rust N-API)     │
│  - Agent registration               │
│  - Operation tracking               │
│  - Conflict detection               │
│  - DAG vertex/tip caching           │
└─────────────────────────────────────┘
```

### Component Responsibilities

#### 1. AgentCoordination (Rust)
- **Location**: `src/agent_coordination.rs`
- **Responsibilities**:
  - Agent registration and lifecycle
  - Operation message storage
  - Conflict rule management
  - DAG vertex ID caching
  - DAG tips caching
  - Statistics aggregation

**Key Methods**:
```rust
// Enable quantum features
pub fn enable_quantum(&mut self)
pub fn is_quantum_enabled(&self) -> bool

// DAG integration points (called from JS bridge)
pub async fn register_dag_vertex(&self, operation_id: String, vertex_id: String)
pub async fn update_dag_tips(&self, tips: Vec<String>)
pub async fn get_dag_vertex(&self, operation_id: &str) -> Option<String>
```

#### 2. QuantumBridge (JavaScript)
- **Location**: `src/quantum_bridge.js`
- **Responsibilities**:
  - QuantumDAG initialization
  - Vertex creation for operations
  - Conflict detection using DAG structure
  - Quantum-resistant proof verification
  - Tip tracking and updates

**Key Methods**:
```javascript
class QuantumBridge {
  // Initialize quantum-resistant DAG
  async initialize()

  // Register operation as DAG vertex
  async registerOperation(operationId, operation, affectedFiles)

  // Check conflicts using DAG topology
  async checkConflicts(operationId, operationType, affectedFiles)

  // Quantum verification
  async verifyQuantumProof(vertexId)

  // Statistics
  async getStats()
}
```

## Integration Flow

### 1. Initialization

```javascript
const { JJWrapper } = require('agentic-jujutsu');
const { createQuantumBridge } = require('agentic-jujutsu/quantum_bridge');

// Create coordination instance
const wrapper = new JJWrapper('/path/to/repo');
const coordination = wrapper.getCoordination();

// Create quantum bridge
const bridge = createQuantumBridge(coordination);
await bridge.initialize();
```

### 2. Operation Registration

When an agent performs an operation:

```javascript
// 1. Register in Rust coordination
const opId = await coordination.registerOperation(
  'agent-1',
  operation,
  ['file1.js', 'file2.js']
);

// 2. Register in QuantumDAG via bridge
const vertexId = await bridge.registerOperation(
  opId,
  {
    operationType: 'edit',
    agentId: 'agent-1',
    metadata: { ... }
  },
  ['file1.js', 'file2.js']
);

// 3. Bridge automatically updates Rust coordination with vertex mapping
```

### 3. Conflict Detection

Before an operation:

```javascript
// Check conflicts using both systems
const conflicts = await bridge.checkConflicts(
  'new-op-id',
  'edit',
  ['file1.js']
);

// Conflicts include:
// - quantumVerified: true/false
// - dagDistance: number
// - isAncestor: true/false
// - severity: 1-3
// - resolutionStrategy: 'auto_merge' | 'manual_resolution' | 'sequential_execution'
```

## Conflict Detection Algorithm

### DAG-Based Severity Calculation

The bridge uses DAG topology to determine conflict severity:

```javascript
calculateSeverity(opA, opB, dagDistance) {
  if (dagDistance < 5) return 3;     // Severe - very recent
  if (dagDistance < 20) return 2;    // Moderate - recent
  return 1;                           // Minor - distant
}
```

### Resolution Strategy Selection

Based on DAG relationships:

```javascript
getResolutionStrategy(opA, opB, isAncestor, distance) {
  if (isAncestor) return 'auto_merge';           // Safe to merge
  if (distance < 5) return 'manual_resolution';  // Needs review
  return 'sequential_execution';                 // Coordinate order
}
```

## Quantum-Resistant Features

### 1. Hash Algorithm
Uses SHA3-256 (quantum-resistant) for all vertex hashing.

### 2. Proof Generation
Each vertex includes a quantum-resistant cryptographic proof.

### 3. Verification
```javascript
const isValid = await bridge.verifyQuantumProof(vertexId);
```

## Performance Characteristics

### Memory Usage
- **Rust coordination**: O(n) where n = number of operations
- **QuantumDAG**: O(v + e) where v = vertices, e = edges
- **Bridge cache**: O(n) for vertex ID mapping

### Time Complexity
- **Operation registration**: O(1) in coordination + O(log v) in DAG
- **Conflict detection**: O(k) where k = recent operations (typically 100)
- **Tip tracking**: O(1) cached, O(t) to update where t = tip count

## Testing

### Unit Tests (Rust)
```bash
cargo test agent_coordination
```

### Integration Tests (JavaScript)
```bash
npm test tests/integration/quantum-bridge.test.js
```

### Manual Testing
```javascript
const { createQuantumBridge } = require('./src/quantum_bridge');

// Test with mock coordination
const mockCoord = {
  enableQuantum: () => {},
  registerDagVertex: async (opId, vertexId) => console.log(opId, vertexId),
  updateDagTips: async (tips) => console.log(tips),
};

const bridge = createQuantumBridge(mockCoord);
await bridge.initialize();
const stats = await bridge.getStats();
console.log(stats);
```

## Future Enhancements

### 1. Distributed Coordination
- Sync DAG across multiple instances
- CRDT-based merge for distributed agents

### 2. Machine Learning Integration
- Learn optimal resolution strategies from history
- Predict conflicts before they occur

### 3. Performance Optimization
- LRU cache for frequently accessed vertices
- Batch vertex registration
- Lazy tip updates

### 4. Advanced Conflict Resolution
- Semantic diff analysis
- Automatic merge strategies
- Rollback coordination

## API Reference

### QuantumBridge

#### `initialize(): Promise<void>`
Initialize QuantumDAG with quantum-resistant configuration.

#### `registerOperation(operationId, operation, affectedFiles): Promise<string>`
Register an agent operation in the DAG.

**Parameters**:
- `operationId`: Unique operation identifier
- `operation`: Operation details (type, agent, metadata)
- `affectedFiles`: Array of affected file paths

**Returns**: Vertex ID in the DAG

#### `checkConflicts(operationId, operationType, affectedFiles): Promise<Conflict[]>`
Check for conflicts using DAG structure.

**Returns**: Array of conflicts with quantum verification

#### `getStats(): Promise<DagStats>`
Get DAG statistics.

**Returns**:
```javascript
{
  initialized: boolean,
  vertices: number,
  tips: number,
  quantumEnabled: boolean,
  totalEdges: number,
  maxDepth: number
}
```

#### `verifyQuantumProof(vertexId): Promise<boolean>`
Verify quantum-resistant proof for a vertex.

#### `getCoordinationTips(): Promise<string[]>`
Get current DAG tips.

#### `cleanup(): Promise<void>`
Clean up resources.

## Dependencies

- `@qudag/napi-core`: ^0.1.0 - QuantumDAG implementation
- `agentic-jujutsu`: Self (Rust N-API module)

## License

Same as agentic-jujutsu package.

## Contributors

See main package README for contributor information.
