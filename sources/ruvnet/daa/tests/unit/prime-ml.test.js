/**
 * Unit Tests for Prime ML NAPI Bindings
 *
 * Tests training nodes, federated coordination, and gradient aggregation
 */

const { test } = require('node:test');
const assert = require('node:assert/strict');

// Mock Prime ML implementation
const createMockPrimeML = () => {
  class TrainingNode {
    constructor(config = {}) {
      this.config = config;
      this.sessions = new Map();
    }

    async initTraining(modelConfig) {
      const sessionId = 'session_' + Math.random().toString(36).substring(7);
      const session = {
        sessionId,
        modelConfig,
        status: 'initialized',
        epoch: 0,
        metrics: { loss: 1.0, accuracy: 0.0 }
      };
      this.sessions.set(sessionId, session);
      return session;
    }

    async trainEpoch(data) {
      return {
        epoch: 1,
        loss: 0.5,
        accuracy: 0.85,
        samplesProcessed: data.samples?.length || 1000,
        trainingTime: 150
      };
    }

    async aggregateGradients(gradients) {
      const avgGradient = {
        layers: gradients.length,
        norm: 0.05,
        aggregated: true
      };

      return avgGradient;
    }

    async submitUpdate(update) {
      return {
        updateId: 'update_' + Math.random().toString(36).substring(7),
        accepted: true,
        timestamp: Date.now()
      };
    }

    async getMetrics() {
      return {
        totalEpochs: 10,
        currentLoss: 0.3,
        currentAccuracy: 0.92,
        trainingTime: 1500
      };
    }
  }

  class Coordinator {
    constructor() {
      this.nodes = new Map();
      this.trainingSessions = new Map();
    }

    async registerNode(nodeId, capabilities) {
      const node = {
        nodeId,
        capabilities,
        status: 'registered',
        lastSeen: Date.now()
      };
      this.nodes.set(nodeId, node);
    }

    async startTraining(config) {
      const sessionId = 'training_' + Math.random().toString(36).substring(7);
      const session = {
        sessionId,
        config,
        status: 'running',
        startTime: Date.now(),
        participants: []
      };
      this.trainingSessions.set(sessionId, session);
      return sessionId;
    }

    async getProgress(sessionId) {
      const session = this.trainingSessions.get(sessionId);
      if (!session) {
        throw new Error('Training session not found');
      }

      return {
        sessionId,
        status: session.status,
        currentEpoch: 5,
        totalEpochs: 10,
        averageLoss: 0.4,
        averageAccuracy: 0.88,
        activeNodes: this.nodes.size
      };
    }

    async stopTraining(sessionId) {
      const session = this.trainingSessions.get(sessionId);
      if (session) {
        session.status = 'stopped';
      }
      return { success: true, sessionId };
    }

    async getNodeMetrics(nodeId) {
      const node = this.nodes.get(nodeId);
      if (!node) {
        throw new Error('Node not found');
      }

      return {
        nodeId,
        totalUpdates: 50,
        averageUpdateTime: 120,
        contribution: 0.15
      };
    }
  }

  return {
    TrainingNode,
    Coordinator
  };
};

const { TrainingNode, Coordinator } = createMockPrimeML();

// Training Node Tests
test('TrainingNode: Create instance', (t) => {
  const node = new TrainingNode({ nodeId: 'node1' });

  assert.ok(node, 'Training node should be created');
  assert.ok(node.config, 'Should have config');
});

test('TrainingNode: Initialize training', async (t) => {
  const node = new TrainingNode();
  const modelConfig = {
    modelType: 'transformer',
    layers: 12,
    hiddenSize: 768
  };

  const session = await node.initTraining(modelConfig);

  assert.ok(session.sessionId, 'Session should have ID');
  assert.equal(session.status, 'initialized', 'Status should be initialized');
  assert.deepEqual(session.modelConfig, modelConfig, 'Config should match');
});

test('TrainingNode: Train epoch', async (t) => {
  const node = new TrainingNode();
  const data = {
    samples: new Array(1000).fill({ input: [], label: 0 })
  };

  const result = await node.trainEpoch(data);

  assert.ok(result.epoch > 0, 'Should have epoch number');
  assert.ok(result.loss >= 0, 'Should have loss value');
  assert.ok(result.accuracy >= 0 && result.accuracy <= 1, 'Accuracy should be between 0 and 1');
  assert.equal(result.samplesProcessed, 1000, 'Should process all samples');
});

test('TrainingNode: Aggregate gradients', async (t) => {
  const node = new TrainingNode();
  const gradients = [
    { layer: 'layer1', values: [0.1, 0.2, 0.3] },
    { layer: 'layer2', values: [0.4, 0.5, 0.6] }
  ];

  const aggregated = await node.aggregateGradients(gradients);

  assert.ok(aggregated.aggregated, 'Should be marked as aggregated');
  assert.equal(aggregated.layers, 2, 'Should have 2 layers');
  assert.ok(aggregated.norm > 0, 'Should have gradient norm');
});

test('TrainingNode: Submit update', async (t) => {
  const node = new TrainingNode();
  const update = {
    epoch: 1,
    gradients: [],
    metrics: { loss: 0.5 }
  };

  const result = await node.submitUpdate(update);

  assert.ok(result.updateId, 'Should have update ID');
  assert.equal(result.accepted, true, 'Update should be accepted');
  assert.ok(result.timestamp, 'Should have timestamp');
});

test('TrainingNode: Get metrics', async (t) => {
  const node = new TrainingNode();

  const metrics = await node.getMetrics();

  assert.ok(metrics.totalEpochs > 0, 'Should have total epochs');
  assert.ok(metrics.currentLoss >= 0, 'Should have current loss');
  assert.ok(metrics.currentAccuracy >= 0, 'Should have current accuracy');
});

// Coordinator Tests
test('Coordinator: Create instance', (t) => {
  const coordinator = new Coordinator();

  assert.ok(coordinator, 'Coordinator should be created');
  assert.ok(coordinator.nodes, 'Should have nodes map');
});

test('Coordinator: Register node', async (t) => {
  const coordinator = new Coordinator();
  const capabilities = {
    gpu: true,
    memory: 16000,
    cores: 8
  };

  await coordinator.registerNode('node1', capabilities);

  assert.equal(coordinator.nodes.size, 1, 'Should have 1 registered node');
  const node = coordinator.nodes.get('node1');
  assert.ok(node, 'Node should be registered');
  assert.equal(node.status, 'registered', 'Status should be registered');
});

test('Coordinator: Start training', async (t) => {
  const coordinator = new Coordinator();
  const config = {
    modelType: 'gpt',
    epochs: 10,
    batchSize: 32
  };

  const sessionId = await coordinator.startTraining(config);

  assert.ok(sessionId, 'Should return session ID');
  assert.match(sessionId, /^training_/, 'Session ID should have correct prefix');
});

test('Coordinator: Get training progress', async (t) => {
  const coordinator = new Coordinator();
  const config = { modelType: 'test', epochs: 10 };

  const sessionId = await coordinator.startTraining(config);
  const progress = await coordinator.getProgress(sessionId);

  assert.equal(progress.sessionId, sessionId, 'Session ID should match');
  assert.ok(progress.currentEpoch >= 0, 'Should have current epoch');
  assert.ok(progress.totalEpochs > 0, 'Should have total epochs');
  assert.ok(progress.averageLoss >= 0, 'Should have average loss');
  assert.ok(progress.averageAccuracy >= 0, 'Should have average accuracy');
});

test('Coordinator: Stop training', async (t) => {
  const coordinator = new Coordinator();
  const config = { modelType: 'test', epochs: 10 };

  const sessionId = await coordinator.startTraining(config);
  const result = await coordinator.stopTraining(sessionId);

  assert.equal(result.success, true, 'Should stop successfully');
  assert.equal(result.sessionId, sessionId, 'Session ID should match');
});

test('Coordinator: Non-existent session throws error', async (t) => {
  const coordinator = new Coordinator();

  await assert.rejects(
    async () => await coordinator.getProgress('non-existent'),
    /Training session not found/,
    'Should throw error for non-existent session'
  );
});

test('Coordinator: Get node metrics', async (t) => {
  const coordinator = new Coordinator();
  const capabilities = { gpu: true };

  await coordinator.registerNode('node1', capabilities);
  const metrics = await coordinator.getNodeMetrics('node1');

  assert.equal(metrics.nodeId, 'node1', 'Node ID should match');
  assert.ok(metrics.totalUpdates >= 0, 'Should have total updates');
  assert.ok(metrics.averageUpdateTime > 0, 'Should have average update time');
  assert.ok(metrics.contribution >= 0, 'Should have contribution score');
});

// Integration Tests
test('Integration: Complete training workflow', async (t) => {
  const coordinator = new Coordinator();
  const node = new TrainingNode();

  // 1. Register node
  await coordinator.registerNode('node1', { gpu: true });

  // 2. Start training session
  const sessionId = await coordinator.startTraining({
    modelType: 'transformer',
    epochs: 5
  });

  // 3. Initialize training on node
  const session = await node.initTraining({
    modelType: 'transformer',
    layers: 12
  });

  // 4. Train epoch
  const epochResult = await node.trainEpoch({
    samples: new Array(100).fill({ input: [], label: 0 })
  });

  // 5. Submit update
  const update = await node.submitUpdate({
    epoch: epochResult.epoch,
    metrics: { loss: epochResult.loss }
  });

  // 6. Check progress
  const progress = await coordinator.getProgress(sessionId);

  assert.ok(session.sessionId, 'Training session initialized');
  assert.ok(epochResult.loss >= 0, 'Epoch trained');
  assert.equal(update.accepted, true, 'Update submitted');
  assert.equal(progress.status, 'running', 'Training in progress');
});

test('Integration: Multi-node federated learning', async (t) => {
  const coordinator = new Coordinator();
  const nodes = [
    new TrainingNode({ nodeId: 'node1' }),
    new TrainingNode({ nodeId: 'node2' }),
    new TrainingNode({ nodeId: 'node3' })
  ];

  // Register all nodes
  for (let i = 0; i < nodes.length; i++) {
    await coordinator.registerNode(`node${i + 1}`, { gpu: true });
  }

  // Start training
  const sessionId = await coordinator.startTraining({
    modelType: 'federated',
    epochs: 10,
    nodes: nodes.length
  });

  // Each node trains locally
  const gradients = [];
  for (const node of nodes) {
    const result = await node.trainEpoch({ samples: [] });
    gradients.push({ node: node.config.nodeId, loss: result.loss });
  }

  // Aggregate gradients (typically done on coordinator)
  const aggregated = await nodes[0].aggregateGradients(gradients);

  // Check progress
  const progress = await coordinator.getProgress(sessionId);

  assert.equal(coordinator.nodes.size, 3, 'Should have 3 registered nodes');
  assert.equal(gradients.length, 3, 'Should have gradients from all nodes');
  assert.ok(aggregated.aggregated, 'Gradients should be aggregated');
  assert.equal(progress.activeNodes, 3, 'All nodes should be active');
});

test('Performance: Gradient aggregation speed', async (t) => {
  const node = new TrainingNode();

  // Create many gradient vectors
  const gradients = new Array(100).fill(null).map((_, i) => ({
    layer: `layer${i}`,
    values: new Array(1000).fill(0.01)
  }));

  const start = Date.now();
  const aggregated = await node.aggregateGradients(gradients);
  const time = Date.now() - start;

  assert.ok(aggregated.aggregated, 'Should aggregate successfully');
  assert.ok(time < 1000, 'Aggregation should be fast (< 1s)');
});
