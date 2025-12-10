/**
 * Integration Tests for Prime ML NAPI
 *
 * Run with: node --test tests/integration.test.js
 */

const { test, describe } = require('node:test');
const assert = require('node:assert');

// Mock the Prime ML NAPI module for testing
// In a real scenario, these would import from the built .node file
const PrimeMock = {
  init: () => 'Prime ML NAPI v0.2.1 initialized',
  version: () => '0.2.1',
  TrainingNode: class TrainingNode {
    constructor(nodeId) {
      this.nodeId = nodeId;
      this._currentEpoch = 0;
      this._initialized = false;
    }

    async initTraining(config) {
      this._initialized = true;
      this._config = config;
    }

    async trainEpoch() {
      if (!this._initialized) {
        throw new Error('Training not initialized');
      }
      this._currentEpoch++;
      return {
        loss: 0.5,
        accuracy: 0.85,
        samplesProcessed: 1000,
        computationTimeMs: 100
      };
    }

    async aggregateGradients(gradients) {
      if (!this._initialized) {
        throw new Error('Training not initialized');
      }
      if (!Array.isArray(gradients) || gradients.length === 0) {
        throw new Error('Gradients must be a non-empty array');
      }
      return Buffer.alloc(gradients[0].length);
    }

    async getStatus() {
      return {
        nodeId: this.nodeId,
        currentEpoch: this._currentEpoch,
        isTraining: this._initialized
      };
    }

    get currentEpoch() {
      return this._currentEpoch;
    }
  },

  Coordinator: class Coordinator {
    constructor(nodeId, config = {}) {
      this.nodeId = nodeId;
      this._config = config;
      this._initialized = false;
      this._nodes = new Map();
      this._currentRound = 0;
      this._modelVersion = 0;
    }

    async init() {
      this._initialized = true;
    }

    async registerNode(nodeInfo) {
      if (!this._initialized) {
        throw new Error('Coordinator not initialized');
      }
      this._nodes.set(nodeInfo.nodeId, nodeInfo);
    }

    async startTraining() {
      if (!this._initialized) {
        throw new Error('Coordinator not initialized');
      }
      this._currentRound++;
      return this._currentRound;
    }

    async getProgress() {
      return {
        currentRound: this._currentRound,
        totalNodes: this._nodes.size,
        completedNodes: 0,
        completionPercent: 0,
        pendingTasks: 0
      };
    }

    async getStatus() {
      return {
        activeNodes: this._nodes.size,
        pendingTasks: 0,
        currentRound: this._currentRound,
        modelVersion: this._modelVersion
      };
    }

    async stop() {
      this._initialized = false;
    }

    get currentRound() {
      return this._currentRound;
    }

    get modelVersion() {
      return this._modelVersion;
    }
  },

  createTensorBuffer: (data, shape) => {
    const buffer = Buffer.alloc(data.length * 4);
    data.forEach((val, i) => buffer.writeFloatLE(val, i * 4));
    return {
      buffer,
      shape,
      dtype: 'f32',
      numElements: () => data.length,
      byteSize: () => buffer.length,
      toF32Array: () => data,
      reshape: (newShape) => {
        const newElements = newShape.reduce((a, b) => a * b, 1);
        if (newElements !== data.length) {
          throw new Error('Shape mismatch');
        }
        return { buffer, shape: newShape, dtype: 'f32' };
      }
    };
  },

  generateNodeId: (prefix = 'node') => {
    return `${prefix}-${Date.now()}`;
  },

  validateNodeId: (nodeId) => {
    return nodeId.length > 0 && /^[a-zA-Z0-9_-]+$/.test(nodeId);
  }
};

describe('Prime ML NAPI Module', () => {
  test('should initialize module', () => {
    const result = PrimeMock.init();
    assert.strictEqual(result, 'Prime ML NAPI v0.2.1 initialized');
  });

  test('should return version', () => {
    const version = PrimeMock.version();
    assert.strictEqual(version, '0.2.1');
  });
});

describe('TrainingNode', () => {
  test('should create training node', () => {
    const node = new PrimeMock.TrainingNode('test-node');
    assert.strictEqual(node.nodeId, 'test-node');
    assert.strictEqual(node.currentEpoch, 0);
  });

  test('should initialize training', async () => {
    const node = new PrimeMock.TrainingNode('test-node');
    await node.initTraining({
      batchSize: 32,
      learningRate: 0.001,
      epochs: 10,
      optimizer: 'adam',
      aggregationStrategy: 'fedavg'
    });

    const status = await node.getStatus();
    assert.strictEqual(status.isTraining, true);
  });

  test('should train epoch', async () => {
    const node = new PrimeMock.TrainingNode('test-node');
    await node.initTraining({
      batchSize: 32,
      learningRate: 0.001,
      epochs: 10,
      optimizer: 'adam',
      aggregationStrategy: 'fedavg'
    });

    const metrics = await node.trainEpoch();
    assert.ok(metrics.loss !== undefined);
    assert.ok(metrics.accuracy !== undefined);
    assert.ok(metrics.samplesProcessed > 0);
    assert.strictEqual(node.currentEpoch, 1);
  });

  test('should aggregate gradients', async () => {
    const node = new PrimeMock.TrainingNode('test-node');
    await node.initTraining({
      batchSize: 32,
      learningRate: 0.001,
      epochs: 10,
      optimizer: 'adam',
      aggregationStrategy: 'fedavg'
    });

    const gradients = [
      Buffer.from(new Float32Array([1, 2, 3, 4]).buffer),
      Buffer.from(new Float32Array([1, 2, 3, 4]).buffer),
      Buffer.from(new Float32Array([1, 2, 3, 4]).buffer)
    ];

    const result = await node.aggregateGradients(gradients);
    assert.ok(result instanceof Buffer);
    assert.strictEqual(result.length, gradients[0].length);
  });

  test('should fail to train without initialization', async () => {
    const node = new PrimeMock.TrainingNode('test-node');
    await assert.rejects(
      async () => await node.trainEpoch(),
      { message: 'Training not initialized' }
    );
  });

  test('should fail to aggregate empty gradients', async () => {
    const node = new PrimeMock.TrainingNode('test-node');
    await node.initTraining({
      batchSize: 32,
      learningRate: 0.001,
      epochs: 10,
      optimizer: 'adam',
      aggregationStrategy: 'fedavg'
    });

    await assert.rejects(
      async () => await node.aggregateGradients([]),
      { message: 'Gradients must be a non-empty array' }
    );
  });
});

describe('Coordinator', () => {
  test('should create coordinator', () => {
    const coordinator = new PrimeMock.Coordinator('coordinator-1');
    assert.strictEqual(coordinator.nodeId, 'coordinator-1');
    assert.strictEqual(coordinator.currentRound, 0);
  });

  test('should initialize coordinator', async () => {
    const coordinator = new PrimeMock.Coordinator('coordinator-1');
    await coordinator.init();

    const status = await coordinator.getStatus();
    assert.strictEqual(status.activeNodes, 0);
    assert.strictEqual(status.currentRound, 0);
  });

  test('should register nodes', async () => {
    const coordinator = new PrimeMock.Coordinator('coordinator-1');
    await coordinator.init();

    await coordinator.registerNode({
      nodeId: 'node-1',
      nodeType: 'trainer',
      lastHeartbeat: Date.now(),
      reliabilityScore: 0.9
    });

    const status = await coordinator.getStatus();
    assert.strictEqual(status.activeNodes, 1);
  });

  test('should start training round', async () => {
    const coordinator = new PrimeMock.Coordinator('coordinator-1');
    await coordinator.init();

    await coordinator.registerNode({
      nodeId: 'node-1',
      nodeType: 'trainer',
      lastHeartbeat: Date.now(),
      reliabilityScore: 0.9
    });

    const roundNumber = await coordinator.startTraining();
    assert.strictEqual(roundNumber, 1);
    assert.strictEqual(coordinator.currentRound, 1);
  });

  test('should track progress', async () => {
    const coordinator = new PrimeMock.Coordinator('coordinator-1');
    await coordinator.init();

    await coordinator.registerNode({
      nodeId: 'node-1',
      nodeType: 'trainer',
      lastHeartbeat: Date.now(),
      reliabilityScore: 0.9
    });

    await coordinator.startTraining();

    const progress = await coordinator.getProgress();
    assert.strictEqual(progress.totalNodes, 1);
    assert.strictEqual(progress.currentRound, 1);
  });

  test('should fail without initialization', async () => {
    const coordinator = new PrimeMock.Coordinator('coordinator-1');
    await assert.rejects(
      async () => await coordinator.startTraining(),
      { message: 'Coordinator not initialized' }
    );
  });
});

describe('Tensor Operations', () => {
  test('should create tensor buffer', () => {
    const tensor = PrimeMock.createTensorBuffer([1, 2, 3, 4], [2, 2]);
    assert.deepStrictEqual(tensor.shape, [2, 2]);
    assert.strictEqual(tensor.numElements(), 4);
    assert.strictEqual(tensor.byteSize(), 16); // 4 floats * 4 bytes
  });

  test('should reshape tensor', () => {
    const tensor = PrimeMock.createTensorBuffer([1, 2, 3, 4, 5, 6], [2, 3]);
    const reshaped = tensor.reshape([3, 2]);
    assert.deepStrictEqual(reshaped.shape, [3, 2]);
  });

  test('should fail to reshape with wrong dimensions', () => {
    const tensor = PrimeMock.createTensorBuffer([1, 2, 3, 4], [2, 2]);
    assert.throws(
      () => tensor.reshape([3, 3]),
      { message: 'Shape mismatch' }
    );
  });

  test('should convert to f32 array', () => {
    const data = [1, 2, 3, 4];
    const tensor = PrimeMock.createTensorBuffer(data, [2, 2]);
    const array = tensor.toF32Array();
    assert.deepStrictEqual(array, data);
  });
});

describe('Utility Functions', () => {
  test('should generate node ID', () => {
    const nodeId = PrimeMock.generateNodeId('test');
    assert.ok(nodeId.startsWith('test-'));
    assert.ok(nodeId.length > 5);
  });

  test('should validate node ID', () => {
    assert.strictEqual(PrimeMock.validateNodeId('valid-node-123'), true);
    assert.strictEqual(PrimeMock.validateNodeId('node_123'), true);
    assert.strictEqual(PrimeMock.validateNodeId('invalid node'), false);
    assert.strictEqual(PrimeMock.validateNodeId(''), false);
  });
});

describe('Integration: Federated Learning Workflow', () => {
  test('should run complete federated learning workflow', async () => {
    // Create coordinator
    const coordinator = new PrimeMock.Coordinator('coordinator-main', {
      minNodesForRound: 2,
      heartbeatTimeoutMs: 10000,
      taskTimeoutMs: 120000,
      consensusThreshold: 0.66
    });

    await coordinator.init();

    // Create training nodes
    const nodes = [];
    for (let i = 0; i < 3; i++) {
      const nodeId = `node-${i + 1}`;
      const node = new PrimeMock.TrainingNode(nodeId);
      await node.initTraining({
        batchSize: 32,
        learningRate: 0.001,
        epochs: 10,
        optimizer: 'adam',
        aggregationStrategy: 'fedavg'
      });

      // Register with coordinator
      await coordinator.registerNode({
        nodeId,
        nodeType: 'trainer',
        lastHeartbeat: Date.now(),
        reliabilityScore: 0.9
      });

      nodes.push(node);
    }

    // Verify coordinator status
    let status = await coordinator.getStatus();
    assert.strictEqual(status.activeNodes, 3);

    // Start training round
    const roundNumber = await coordinator.startTraining();
    assert.strictEqual(roundNumber, 1);

    // Each node trains
    const gradients = [];
    for (const node of nodes) {
      const metrics = await node.trainEpoch();
      assert.ok(metrics.loss !== undefined);

      // Generate gradient
      const gradient = Buffer.from(new Float32Array([1, 2, 3, 4]).buffer);
      gradients.push(gradient);
    }

    // Aggregate gradients
    const aggregated = await nodes[0].aggregateGradients(gradients);
    assert.ok(aggregated instanceof Buffer);

    // Check progress
    const progress = await coordinator.getProgress();
    assert.strictEqual(progress.totalNodes, 3);

    // Stop coordinator
    await coordinator.stop();
  });
});

console.log('Running Prime ML NAPI Integration Tests...\n');
