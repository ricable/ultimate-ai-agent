# QuantumDAG Coordination Usage Guide

## Quick Start

### Installation

```bash
npm install agentic-jujutsu @qudag/napi-core
```

### Basic Setup

```javascript
const { JJWrapper } = require('agentic-jujutsu');
const { createQuantumBridge } = require('agentic-jujutsu/quantum_bridge');

// Initialize
const wrapper = new JJWrapper('/path/to/repo');
const coordination = wrapper.getCoordination();
const bridge = createQuantumBridge(coordination);
await bridge.initialize();
```

## Use Cases

### 1. Simple Agent Coordination

```javascript
// Register agents
await coordination.registerAgent('agent-1', 'coder');
await coordination.registerAgent('agent-2', 'reviewer');

// Agent 1 performs operation
const op1 = {
  operationType: 'edit',
  agentId: 'agent-1',
  metadata: { purpose: 'feature' },
};

const vertexId = await bridge.registerOperation(
  'op-1',
  op1,
  ['src/file.js']
);

// Agent 2 checks for conflicts
const conflicts = await bridge.checkConflicts(
  'op-2',
  'review',
  ['src/file.js']
);

if (conflicts.length > 0) {
  console.log('Conflicts detected:', conflicts);
}
```

### 2. Multi-Agent Parallel Coordination

```javascript
// Register multiple agents
const agents = ['coder', 'reviewer', 'tester', 'architect'];
for (const type of agents) {
  await coordination.registerAgent(`${type}-1`, type);
}

// Parallel operations
const operations = [
  { id: 'op-1', type: 'edit', files: ['src/main.js'], agent: 'coder-1' },
  { id: 'op-2', type: 'review', files: ['src/main.js'], agent: 'reviewer-1' },
  { id: 'op-3', type: 'test', files: ['test/main.test.js'], agent: 'tester-1' },
  { id: 'op-4', type: 'refactor', files: ['src/utils.js'], agent: 'architect-1' },
];

// Check all operations for conflicts
for (const op of operations) {
  const conflicts = await bridge.checkConflicts(
    op.id,
    op.type,
    op.files
  );

  if (conflicts.length === 0) {
    // Safe to proceed
    await bridge.registerOperation(
      op.id,
      {
        operationType: op.type,
        agentId: op.agent,
        metadata: {},
      },
      op.files
    );
  } else {
    // Handle conflicts
    console.log(`Operation ${op.id} has conflicts:`, conflicts);
  }
}
```

### 3. Conflict Resolution Workflow

```javascript
async function coordinateOperation(opId, opType, files, agentId) {
  // Check for conflicts
  const conflicts = await bridge.checkConflicts(opId, opType, files);

  if (conflicts.length === 0) {
    // No conflicts - proceed immediately
    return await bridge.registerOperation(
      opId,
      { operationType: opType, agentId, metadata: {} },
      files
    );
  }

  // Analyze conflicts
  for (const conflict of conflicts) {
    switch (conflict.resolutionStrategy) {
      case 'auto_merge':
        // Safe to auto-merge
        console.log('Auto-merging conflict:', conflict.description);
        break;

      case 'sequential_execution':
        // Wait for previous operation to complete
        console.log('Waiting for previous operation...');
        await waitForOperation(conflict.operationB);
        break;

      case 'manual_resolution':
        // Requires human intervention
        console.log('Manual resolution required:', conflict.description);
        await notifyHuman(conflict);
        return null;
    }
  }

  // Register after resolution
  return await bridge.registerOperation(
    opId,
    { operationType: opType, agentId, metadata: {} },
    files
  );
}
```

### 4. Quantum Verification

```javascript
// Register operation
const vertexId = await bridge.registerOperation(
  'secure-op',
  {
    operationType: 'deploy',
    agentId: 'deploy-agent',
    metadata: { environment: 'production' },
  },
  ['deploy/config.yaml']
);

// Verify quantum-resistant proof
const isValid = await bridge.verifyQuantumProof(vertexId);

if (!isValid) {
  console.error('Quantum proof verification failed!');
  // Handle security issue
  await rollbackOperation(vertexId);
}
```

### 5. Real-time Monitoring

```javascript
// Monitor coordination in real-time
setInterval(async () => {
  const stats = await bridge.getStats();
  const coordStats = await coordination.getStats();

  console.log('Coordination Status:');
  console.log(`  Active Agents: ${coordStats.activeAgents}`);
  console.log(`  DAG Vertices: ${stats.vertices}`);
  console.log(`  Current Tips: ${stats.tips}`);
  console.log(`  Operations: ${coordStats.totalOperations}`);
}, 5000);
```

### 6. Agent Reputation Tracking

```javascript
// Get agent statistics
const agentStats = await coordination.getAgentStats('coder-1');

console.log(`Agent: ${agentStats.agentId}`);
console.log(`Type: ${agentStats.agentType}`);
console.log(`Operations: ${agentStats.operationsCount}`);
console.log(`Reputation: ${agentStats.reputation}`);
console.log(`Last Seen: ${agentStats.lastSeen}`);

// Update reputation based on operation success
async function updateReputation(agentId, success) {
  const stats = await coordination.getAgentStats(agentId);
  if (stats) {
    const newReputation = success
      ? Math.min(1.0, stats.reputation + 0.05)
      : Math.max(0.0, stats.reputation - 0.1);

    // Reputation updates would be handled in Rust
    console.log(`Updated ${agentId} reputation: ${newReputation}`);
  }
}
```

### 7. Batch Operations

```javascript
// Batch register multiple operations
const operations = [
  { id: 'op-1', type: 'edit', files: ['file1.js'], agent: 'agent-1' },
  { id: 'op-2', type: 'edit', files: ['file2.js'], agent: 'agent-2' },
  { id: 'op-3', type: 'edit', files: ['file3.js'], agent: 'agent-3' },
];

const vertexIds = await Promise.all(
  operations.map(op =>
    bridge.registerOperation(
      op.id,
      {
        operationType: op.type,
        agentId: op.agent,
        metadata: {},
      },
      op.files
    )
  )
);

console.log('Registered operations:', vertexIds);
```

### 8. Advanced Conflict Analysis

```javascript
async function analyzeConflicts(opId, opType, files) {
  const conflicts = await bridge.checkConflicts(opId, opType, files);

  // Group conflicts by severity
  const severe = conflicts.filter(c => c.severity === 3);
  const moderate = conflicts.filter(c => c.severity === 2);
  const minor = conflicts.filter(c => c.severity === 1);

  console.log(`Severe conflicts: ${severe.length}`);
  console.log(`Moderate conflicts: ${moderate.length}`);
  console.log(`Minor conflicts: ${minor.length}`);

  // Analyze DAG structure
  for (const conflict of conflicts) {
    console.log(`
Conflict Analysis:
  Operation A: ${conflict.operationA}
  Operation B: ${conflict.operationB}
  DAG Distance: ${conflict.dagDistance}
  Is Ancestor: ${conflict.isAncestor}
  Quantum Verified: ${conflict.quantumVerified}
  Strategy: ${conflict.resolutionStrategy}
    `);
  }

  return { severe, moderate, minor };
}
```

## Best Practices

### 1. Always Initialize Before Use

```javascript
await bridge.initialize();
// Now safe to use bridge methods
```

### 2. Check Conflicts Before Operations

```javascript
// Always check before registering
const conflicts = await bridge.checkConflicts(opId, opType, files);
if (conflicts.length === 0) {
  await bridge.registerOperation(opId, operation, files);
}
```

### 3. Handle Cleanup Properly

```javascript
try {
  // Use bridge
} finally {
  await bridge.cleanup();
}
```

### 4. Monitor Statistics Regularly

```javascript
// Check coordination health
const stats = await bridge.getStats();
if (stats.vertices > 10000) {
  console.warn('High vertex count, consider cleanup');
}
```

### 5. Verify Critical Operations

```javascript
// For production deployments
if (operation.critical) {
  const isValid = await bridge.verifyQuantumProof(vertexId);
  if (!isValid) {
    throw new Error('Quantum proof verification failed');
  }
}
```

## Error Handling

```javascript
try {
  await bridge.registerOperation(opId, operation, files);
} catch (error) {
  if (error.message.includes('not initialized')) {
    await bridge.initialize();
    // Retry
  } else if (error.message.includes('conflict')) {
    // Handle conflict
  } else {
    console.error('Unexpected error:', error);
  }
}
```

## Performance Tips

1. **Batch Operations**: Register multiple operations in parallel using `Promise.all()`
2. **Cache Tips**: Tips are cached automatically, avoid excessive `getCoordinationTips()` calls
3. **Limit Conflict Checks**: Only check recent operations (last 100 is default)
4. **Cleanup Regularly**: Call `cleanup()` when done to free resources

## Integration with CI/CD

```javascript
// In CI/CD pipeline
const coordination = setupCoordination();
const bridge = createQuantumBridge(coordination);
await bridge.initialize();

// Register CI agent
await coordination.registerAgent('ci-bot', 'automation');

// Check for conflicts before deployment
const conflicts = await bridge.checkConflicts(
  'deploy-op',
  'deploy',
  ['dist/*']
);

if (conflicts.some(c => c.severity === 3)) {
  console.error('Deployment blocked by severe conflicts');
  process.exit(1);
}

// Proceed with deployment
const vertexId = await bridge.registerOperation(
  'deploy-op',
  {
    operationType: 'deploy',
    agentId: 'ci-bot',
    metadata: { commit: process.env.COMMIT_SHA },
  },
  ['dist/*']
);

// Verify
if (!await bridge.verifyQuantumProof(vertexId)) {
  console.error('Deployment verification failed');
  process.exit(1);
}
```

## See Also

- [Architecture Documentation](../QUANTUM_DAG_INTEGRATION.md)
- [API Reference](../QUANTUM_DAG_INTEGRATION.md#api-reference)
- [Integration Tests](../../tests/integration/quantum-bridge.test.js)
- [Full Example](../../examples/quantum-coordination-example.js)
