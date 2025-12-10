/**
 * Integration tests for Quantum Bridge
 *
 * Tests the integration between AgentCoordination (Rust) and QuantumDAG (@qudag/napi-core)
 */

const { describe, it, before, after } = require('node:test');
const assert = require('node:assert');
const { createQuantumBridge } = require('../../src/quantum_bridge');

describe('QuantumBridge Integration', () => {
  let bridge;
  let mockCoordination;

  before(async () => {
    // Mock AgentCoordination interface
    mockCoordination = {
      enableQuantum: () => {},
      registerDagVertex: async (opId, vertexId) => {
        console.log(`Mock: Registered ${opId} -> ${vertexId}`);
      },
      updateDagTips: async (tips) => {
        console.log(`Mock: Updated tips: ${tips.join(', ')}`);
      },
    };

    bridge = createQuantumBridge(mockCoordination);
  });

  after(async () => {
    if (bridge) {
      await bridge.cleanup();
    }
  });

  it('should initialize QuantumDAG', async () => {
    await bridge.initialize();

    const stats = await bridge.getStats();
    assert.strictEqual(stats.initialized, true);
    assert.strictEqual(stats.quantumEnabled, true);
    assert.strictEqual(stats.vertices, 0);
  });

  it('should register operations in DAG', async () => {
    await bridge.initialize();

    const operation = {
      operationType: 'edit',
      agentId: 'agent-1',
      metadata: { test: 'data' },
    };

    const vertexId = await bridge.registerOperation(
      'op-1',
      operation,
      ['file1.js', 'file2.js']
    );

    assert.ok(vertexId);
    assert.strictEqual(typeof vertexId, 'string');

    const stats = await bridge.getStats();
    assert.strictEqual(stats.vertices, 1);
  });

  it('should detect conflicts using DAG structure', async () => {
    await bridge.initialize();

    // Register first operation
    const op1 = {
      operationType: 'edit',
      agentId: 'agent-1',
      metadata: {},
    };

    await bridge.registerOperation('op-1', op1, ['file1.js']);

    // Register second operation with same file
    const op2 = {
      operationType: 'edit',
      agentId: 'agent-2',
      metadata: {},
    };

    await bridge.registerOperation('op-2', op2, ['file1.js']);

    // Check for conflicts
    const conflicts = await bridge.checkConflicts(
      'op-3',
      'edit',
      ['file1.js']
    );

    assert.ok(Array.isArray(conflicts));
    assert.ok(conflicts.length > 0);
    assert.strictEqual(conflicts[0].quantumVerified, true);
    assert.ok(conflicts[0].conflictingFiles.includes('file1.js'));
  });

  it('should track coordination tips', async () => {
    await bridge.initialize();

    const op = {
      operationType: 'commit',
      agentId: 'agent-1',
      metadata: {},
    };

    await bridge.registerOperation('op-tip-1', op, ['file.js']);

    const tips = await bridge.getCoordinationTips();
    assert.ok(Array.isArray(tips));
    assert.ok(tips.length > 0);
  });

  it('should calculate conflict severity based on DAG distance', async () => {
    await bridge.initialize();

    // Create a chain of operations
    for (let i = 0; i < 10; i++) {
      const op = {
        operationType: 'edit',
        agentId: `agent-${i}`,
        metadata: {},
      };

      await bridge.registerOperation(`op-chain-${i}`, op, [`file-${i}.js`]);
    }

    // Register operation that conflicts with early operation
    const conflictOp = {
      operationType: 'edit',
      agentId: 'agent-conflict',
      metadata: {},
    };

    await bridge.registerOperation('op-chain-10', conflictOp, ['file-0.js']);

    const conflicts = await bridge.checkConflicts(
      'op-chain-11',
      'edit',
      ['file-0.js']
    );

    assert.ok(conflicts.length > 0);
    // Should have lower severity due to DAG distance
    assert.ok(conflicts[0].severity <= 2);
  });

  it('should provide quantum-resistant verification', async () => {
    await bridge.initialize();

    const op = {
      operationType: 'commit',
      agentId: 'agent-1',
      metadata: {},
    };

    const vertexId = await bridge.registerOperation('op-verify', op, ['file.js']);

    const isValid = await bridge.verifyQuantumProof(vertexId);
    assert.strictEqual(isValid, true);
  });

  it('should suggest resolution strategies based on DAG structure', async () => {
    await bridge.initialize();

    // Create parent operation
    const parentOp = {
      operationType: 'edit',
      agentId: 'agent-parent',
      metadata: {},
    };

    await bridge.registerOperation('op-parent', parentOp, ['shared.js']);

    // Create child operation
    const childOp = {
      operationType: 'edit',
      agentId: 'agent-child',
      metadata: {},
    };

    await bridge.registerOperation('op-child', childOp, ['shared.js']);

    // Check conflicts
    const conflicts = await bridge.checkConflicts(
      'op-new',
      'edit',
      ['shared.js']
    );

    assert.ok(conflicts.length > 0);
    assert.ok(conflicts[0].resolutionStrategy);
    assert.ok(['auto_merge', 'manual_resolution', 'sequential_execution'].includes(
      conflicts[0].resolutionStrategy
    ));
  });

  it('should handle multiple agents coordinating', async () => {
    await bridge.initialize();

    const agents = ['coder', 'reviewer', 'tester', 'architect'];
    const operations = [];

    // Register operations from multiple agents
    for (let i = 0; i < agents.length; i++) {
      const op = {
        operationType: 'process',
        agentId: agents[i],
        metadata: { step: i },
      };

      const vertexId = await bridge.registerOperation(
        `multi-op-${i}`,
        op,
        [`shared-file.js`]
      );

      operations.push({ id: `multi-op-${i}`, vertexId });
    }

    const stats = await bridge.getStats();
    assert.strictEqual(stats.vertices, agents.length);

    // Check for conflicts
    const conflicts = await bridge.checkConflicts(
      'multi-op-new',
      'process',
      ['shared-file.js']
    );

    assert.ok(conflicts.length > 0);
    assert.ok(conflicts.every(c => c.quantumVerified));
  });

  it('should clean up resources properly', async () => {
    await bridge.initialize();

    const op = {
      operationType: 'test',
      agentId: 'agent-cleanup',
      metadata: {},
    };

    await bridge.registerOperation('op-cleanup', op, ['file.js']);

    await bridge.cleanup();

    const stats = await bridge.getStats();
    assert.strictEqual(stats.initialized, false);
  });
});

console.log('Quantum Bridge integration tests ready to run with: npm test');
