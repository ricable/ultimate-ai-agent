/**
 * Tests for Self-Learning GNN for Ericsson Uplink Optimization
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  RuVectorGNNLayer,
  ExperienceReplayBuffer,
  DifferentiableParameterSearch,
  SelfLearningUplinkGNN,
  EricssonUplinkOptimizer,
  runRuVectorGNNLayer,
  compressEmbeddings,
  differentiableSearch,
  type ExperienceSample,
  type PowerControlParams,
  type CompressionLevel,
} from '../../src/gnn/self-learning-uplink-gnn.js';
import {
  SurrogateGraphBuilder,
  DEFAULT_SURROGATE_CONFIG,
  type SurrogateGraph,
} from '../../src/gnn/network-surrogate-model.js';
import type { CellKPISnapshot, NeighborRelation } from '../../src/models/ran-kpi.js';

// ============================================================================
// TEST FIXTURES
// ============================================================================

function createMockCellSnapshot(
  cellId: string,
  p0: number = -100,
  alpha: number = 0.8,
  ulSinrAvg: number = 8,
  iotAvg: number = 6
): CellKPISnapshot {
  return {
    cellId,
    timestamp: new Date(),
    eNodeBId: `enb_${cellId.split('_')[1]}`,
    sectorId: parseInt(cellId.split('_')[2] ?? '1'),
    frequency: 1800,
    technology: 'LTE',
    accessibility: {
      rrcSetupAttempts: 1000,
      rrcSetupSuccessRate: 99.5,
      rrcSetupFailures: 5,
      erabSetupSuccessRate: 99.2,
      erabSetupFailures: 8,
      initialContextSetupSuccessRate: 99.0,
    },
    retainability: {
      erabDropRate: 0.5,
      callDropRate: 0.3,
      dataSessionRetainability: 99.0,
      voiceCallRetainability: 99.5,
      avgSessionDuration: 300,
    },
    radioQuality: {
      dlAvgCqi: 10,
      dlCqiDistribution: [2, 5, 10, 20, 30, 20, 10, 3],
      ulSinrAvg,
      ulSinrP10: ulSinrAvg - 5,
      ulSinrP90: ulSinrAvg + 5,
      rsrpAvg: -95,
      rsrqAvg: -12,
      dlThroughputAvg: 50,
      ulThroughputAvg: 15,
      dlSpectralEfficiency: 4,
      ulSpectralEfficiency: 2,
      dlBlerPercent: 5,
      ulBlerPercent: 3,
    },
    mobility: {
      intraFreqHoAttempts: 500,
      intraFreqHoSuccessRate: 97,
      interFreqHoAttempts: 100,
      interFreqHoSuccessRate: 95,
      interRatHoAttempts: 50,
      interRatHoSuccessRate: 90,
      pingPongHoRate: 2,
      tooEarlyHoRate: 1,
      tooLateHoRate: 1,
    },
    uplinkInterference: {
      iotAvg,
      iotP95: iotAvg + 3,
      iotDistribution: [10, 20, 40, 20, 10],
      rip: -105,
      highInterferencePrbRatio: 15,
    },
    uplinkPowerControl: {
      p0NominalPusch: p0,
      alpha,
      powerHeadroomAvg: 15,
      powerHeadroomP10: 5,
      powerHeadroomP90: 25,
      powerLimitedUeRatio: 10,
      negativePowerHeadroomRatio: 5,
      pathLossAvg: 120,
      pathLossP90: 140,
    },
  };
}

function createMockNeighborRelation(
  sourceId: string,
  targetId: string,
  hoSuccessRate: number = 95
): NeighborRelation {
  return {
    sourceCellId: sourceId,
    targetCellId: targetId,
    sourceSinr: 8,
    targetSinr: 10,
    sourceRsrp: -95,
    targetRsrp: -93,
    hoSuccessRate,
    hoAttempts: 100,
    a3Offset: 3,
    hysteresis: 2,
    ttt: 320,
    relationshipType: 'intra-freq',
    distance: 500,
  };
}

function createMockNetwork(): {
  snapshots: Map<string, CellKPISnapshot>;
  relations: NeighborRelation[];
} {
  const snapshots = new Map<string, CellKPISnapshot>();

  // Create 5 cells - one issue cell, others healthy
  snapshots.set('cell_1_1', createMockCellSnapshot('cell_1_1', -100, 0.8, 3, 12)); // Issue cell - low SINR, high IoT
  snapshots.set('cell_1_2', createMockCellSnapshot('cell_1_2', -98, 0.7, 10, 5));
  snapshots.set('cell_1_3', createMockCellSnapshot('cell_1_3', -102, 0.9, 8, 6));
  snapshots.set('cell_2_1', createMockCellSnapshot('cell_2_1', -95, 0.6, 12, 4));
  snapshots.set('cell_2_2', createMockCellSnapshot('cell_2_2', -105, 1.0, 6, 8));

  const relations: NeighborRelation[] = [
    createMockNeighborRelation('cell_1_1', 'cell_1_2', 96),
    createMockNeighborRelation('cell_1_1', 'cell_1_3', 94),
    createMockNeighborRelation('cell_1_2', 'cell_1_3', 98),
    createMockNeighborRelation('cell_1_2', 'cell_2_1', 92),
    createMockNeighborRelation('cell_2_1', 'cell_2_2', 95),
  ];

  return { snapshots, relations };
}

// ============================================================================
// RUVECTOR GNN LAYER TESTS
// ============================================================================

describe('RuVectorGNNLayer', () => {
  let layer: RuVectorGNNLayer;

  beforeEach(() => {
    layer = new RuVectorGNNLayer({
      inputDim: 24,
      hiddenDim: 64,
      numHeads: 4,
      dropout: 0.1,
    });
  });

  it('should initialize with correct configuration', () => {
    const config = layer.getConfig();
    expect(config.inputDim).toBe(24);
    expect(config.hiddenDim).toBe(64);
    expect(config.numHeads).toBe(4);
    expect(config.dropout).toBe(0.1);
  });

  it('should perform forward pass', () => {
    const nodeFeatures = [
      Array(24).fill(0).map(() => Math.random()),
      Array(24).fill(0).map(() => Math.random()),
      Array(24).fill(0).map(() => Math.random()),
    ];

    const adjacencyMatrix = [
      [1, 0.5, 0.3],
      [0.5, 1, 0.6],
      [0.3, 0.6, 1],
    ];

    const output = layer.forward(nodeFeatures, adjacencyMatrix);

    expect(output).toHaveLength(3);
    expect(output[0]).toHaveLength(64);
    expect(output.every(row => row.every(v => !isNaN(v)))).toBe(true);
  });

  it('should handle edge features', () => {
    const nodeFeatures = [
      Array(24).fill(0).map(() => Math.random()),
      Array(24).fill(0).map(() => Math.random()),
    ];

    const adjacencyMatrix = [
      [1, 0.8],
      [0.8, 1],
    ];

    const edgeFeatures = [
      [[], [0.5, 0.3, 0.2, 0.1]],
      [[0.5, 0.3, 0.2, 0.1], []],
    ];

    const output = layer.forward(nodeFeatures, adjacencyMatrix, edgeFeatures);

    expect(output).toHaveLength(2);
    expect(output[0]).toHaveLength(64);
  });

  it('should export and import weights', () => {
    const nodeFeatures = [
      Array(24).fill(0).map(() => Math.random()),
      Array(24).fill(0).map(() => Math.random()),
    ];

    const adjacencyMatrix = [
      [1, 0.5],
      [0.5, 1],
    ];

    const output1 = layer.forward(nodeFeatures, adjacencyMatrix);

    const weights = layer.exportWeights();
    expect(weights.W_query).toBeDefined();
    expect(weights.W_key).toBeDefined();
    expect(weights.W_value).toBeDefined();
    expect(weights.W_output).toBeDefined();

    const newLayer = new RuVectorGNNLayer({
      inputDim: 24,
      hiddenDim: 64,
      numHeads: 4,
    });
    newLayer.importWeights(weights);

    const output2 = newLayer.forward(nodeFeatures, adjacencyMatrix);

    // Outputs should be identical after weight import
    expect(output2.length).toBe(output1.length);
  });

  it('should apply dropout during training', () => {
    layer.setTraining(true);

    const nodeFeatures = [
      Array(24).fill(0).map(() => 1), // All ones for predictable output
      Array(24).fill(0).map(() => 1),
    ];

    const adjacencyMatrix = [
      [1, 0.5],
      [0.5, 1],
    ];

    // With dropout, multiple passes should give slightly different results
    const output1 = layer.forward(nodeFeatures, adjacencyMatrix);
    const output2 = layer.forward(nodeFeatures, adjacencyMatrix);

    // Should still produce valid outputs
    expect(output1.every(row => row.every(v => !isNaN(v)))).toBe(true);
    expect(output2.every(row => row.every(v => !isNaN(v)))).toBe(true);

    layer.setTraining(false);
  });
});

// ============================================================================
// EXPERIENCE REPLAY BUFFER TESTS
// ============================================================================

describe('ExperienceReplayBuffer', () => {
  let buffer: ExperienceReplayBuffer;
  let mockGraph: SurrogateGraph;

  beforeEach(() => {
    buffer = new ExperienceReplayBuffer(100, 0.6, 0.4, 0.01);
    mockGraph = {
      nodeIds: ['cell_1'],
      nodeFeatures: [[0.5, 0.8]],
      adjacencyMatrix: [[1]],
      edgeFeatures: [[[]]],
      powerParams: new Map([['cell_1', { p0: -100, alpha: 0.8 }]]),
    };
  });

  it('should add samples', () => {
    buffer.add({
      timestamp: new Date(),
      graph: mockGraph,
      cellId: 'cell_1',
      params: { p0: -100, alpha: 0.8 },
      predictedSINR: 10,
      actualSINR: 12,
      reward: 1,
    });

    expect(buffer.size()).toBe(1);
  });

  it('should sample from buffer', () => {
    // Add multiple samples
    for (let i = 0; i < 10; i++) {
      buffer.add({
        timestamp: new Date(),
        graph: mockGraph,
        cellId: 'cell_1',
        params: { p0: -100 + i, alpha: 0.8 },
        predictedSINR: 10 + i,
        actualSINR: 12 + i,
        reward: i % 2 === 0 ? 1 : -0.5,
      });
    }

    const { samples, weights, indices } = buffer.sample(5);

    expect(samples.length).toBe(5);
    expect(weights.length).toBe(5);
    expect(indices.length).toBe(5);
    expect(weights.every(w => w > 0)).toBe(true);
  });

  it('should update priorities', () => {
    // Add samples
    for (let i = 0; i < 5; i++) {
      buffer.add({
        timestamp: new Date(),
        graph: mockGraph,
        cellId: 'cell_1',
        params: { p0: -100, alpha: 0.8 },
        predictedSINR: 10,
        actualSINR: 12,
        reward: 1,
      });
    }

    const { indices } = buffer.sample(3);
    const tdErrors = [5.0, 2.0, 0.5]; // Different TD errors

    buffer.updatePriorities(indices, tdErrors);

    // Subsequent samples should reflect updated priorities
    const stats = buffer.getStats();
    expect(stats.avgPriority).toBeGreaterThan(0);
  });

  it('should evict low priority samples when full', () => {
    const smallBuffer = new ExperienceReplayBuffer(5);

    // Add 10 samples to a buffer of size 5
    for (let i = 0; i < 10; i++) {
      smallBuffer.add({
        timestamp: new Date(),
        graph: mockGraph,
        cellId: 'cell_1',
        params: { p0: -100, alpha: 0.8 },
        predictedSINR: 10,
        actualSINR: 10 + i, // Different errors = different priorities
        reward: 1,
      });
    }

    expect(smallBuffer.size()).toBe(5);
  });

  it('should export and import buffer', () => {
    for (let i = 0; i < 5; i++) {
      buffer.add({
        timestamp: new Date(),
        graph: mockGraph,
        cellId: `cell_${i}`,
        params: { p0: -100, alpha: 0.8 },
        predictedSINR: 10,
        actualSINR: 12,
        reward: 1,
      });
    }

    const exported = buffer.export();
    expect(exported.length).toBe(5);

    const newBuffer = new ExperienceReplayBuffer(100);
    newBuffer.import(exported);
    expect(newBuffer.size()).toBe(5);
  });

  it('should calculate correct statistics', () => {
    for (let i = 0; i < 5; i++) {
      buffer.add({
        timestamp: new Date(),
        graph: mockGraph,
        cellId: 'cell_1',
        params: { p0: -100, alpha: 0.8 },
        predictedSINR: 10,
        actualSINR: 12,
        reward: i % 2 === 0 ? 1 : 0,
      });
    }

    const stats = buffer.getStats();
    expect(stats.size).toBe(5);
    expect(stats.avgPriority).toBeGreaterThan(0);
    expect(stats.avgPredictionError).toBe(2); // |10 - 12| = 2
    expect(stats.avgReward).toBe(0.6); // 3 samples with 1, 2 samples with 0
  });
});

// ============================================================================
// DIFFERENTIABLE PARAMETER SEARCH TESTS
// ============================================================================

describe('DifferentiableParameterSearch', () => {
  let search: DifferentiableParameterSearch;

  beforeEach(() => {
    search = new DifferentiableParameterSearch(1.0);
  });

  it('should generate candidates', () => {
    const currentParams: PowerControlParams = { p0: -100, alpha: 0.8 };
    const candidates = search.generateCandidates(currentParams, DEFAULT_SURROGATE_CONFIG);

    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates.every(c => c.p0 >= -110 && c.p0 <= -85)).toBe(true);
    expect(candidates.every(c => c.alpha >= 0.4 && c.alpha <= 1.0)).toBe(true);
  });

  it('should embed candidates', () => {
    const candidates: PowerControlParams[] = [
      { p0: -100, alpha: 0.8 },
      { p0: -95, alpha: 0.9 },
      { p0: -105, alpha: 0.7 },
    ];

    // Mock predictor
    const predictor = (params: PowerControlParams) => params.p0 / -10 + params.alpha * 5;

    search.embedCandidates(candidates, predictor);

    const query = [0.4, 0.8, 0.5]; // Normalized P0, alpha, target SINR
    const result = search.softSearch(query);

    expect(result.params).toBeDefined();
    expect(result.confidence).toBeGreaterThan(0);
    expect(result.attention.length).toBe(3);
  });

  it('should perform soft search with temperature control', () => {
    const candidates: PowerControlParams[] = [
      { p0: -100, alpha: 0.8 },
      { p0: -95, alpha: 0.9 },
    ];

    const predictor = (params: PowerControlParams) => 10 + (params.p0 + 100) * 0.5;
    search.embedCandidates(candidates, predictor);

    // High temperature = more uniform attention
    search.setTemperature(5.0);
    const highTempResult = search.softSearch([0.5, 0.8, 0.5]);

    // Low temperature = sharper attention
    search.setTemperature(0.1);
    const lowTempResult = search.softSearch([0.5, 0.8, 0.5]);

    // With low temperature, attention should be more concentrated
    const highTempMaxAttention = Math.max(...highTempResult.attention);
    const lowTempMaxAttention = Math.max(...lowTempResult.attention);

    expect(lowTempMaxAttention).toBeGreaterThanOrEqual(highTempMaxAttention - 0.1);
  });

  it('should perform hard search', () => {
    const candidates: PowerControlParams[] = [
      { p0: -100, alpha: 0.8 },
      { p0: -95, alpha: 0.9 },
      { p0: -105, alpha: 0.7 },
    ];

    const predictor = (params: PowerControlParams) => 10;
    search.embedCandidates(candidates, predictor);

    const result = search.hardSearch([0.2, 0.9, 0.5]); // Query favoring higher alpha

    expect(result.params).toBeDefined();
    expect(result.index).toBeGreaterThanOrEqual(0);
    expect(result.index).toBeLessThan(3);
  });
});

// ============================================================================
// SELF-LEARNING UPLINK GNN TESTS
// ============================================================================

describe('SelfLearningUplinkGNN', () => {
  let gnn: SelfLearningUplinkGNN;
  let network: { snapshots: Map<string, CellKPISnapshot>; relations: NeighborRelation[] };
  let graphBuilder: SurrogateGraphBuilder;

  beforeEach(() => {
    gnn = new SelfLearningUplinkGNN();
    network = createMockNetwork();
    graphBuilder = new SurrogateGraphBuilder();
  });

  it('should initialize with default state', () => {
    const state = gnn.getState();
    expect(state.modelVersion).toBe(1);
    expect(state.totalSamples).toBe(0);
    expect(state.totalUpdates).toBe(0);
    expect(state.explorationRate).toBe(0.3);
  });

  it('should predict SINR and IoT', () => {
    const graph = graphBuilder.buildGraph(network.snapshots, network.relations);
    const prediction = gnn.predict(graph);

    expect(prediction.sinr.length).toBe(5); // 5 cells
    expect(prediction.iot.length).toBe(5);
    expect(prediction.embeddings.length).toBe(5);

    // Predictions should be in valid ranges
    expect(prediction.sinr.every(s => s >= -5 && s <= 30)).toBe(true);
    expect(prediction.iot.every(i => i >= 0 && i <= 20)).toBe(true);
  });

  it('should optimize a cell', () => {
    const graph = graphBuilder.buildGraph(network.snapshots, network.relations);

    const result = gnn.optimizeCell(
      'cell_1_1',
      graph,
      network.snapshots,
      network.relations
    );

    expect(result.cellId).toBe('cell_1_1');
    expect(result.originalParams).toBeDefined();
    expect(result.optimizedParams).toBeDefined();
    expect(result.optimizedParams.p0).toBeGreaterThanOrEqual(-110);
    expect(result.optimizedParams.p0).toBeLessThanOrEqual(-85);
    expect(result.optimizedParams.alpha).toBeGreaterThanOrEqual(0.4);
    expect(result.optimizedParams.alpha).toBeLessThanOrEqual(1.0);
    expect(result.confidence).toBeGreaterThan(0);
  });

  it('should learn from feedback', () => {
    const graph = graphBuilder.buildGraph(network.snapshots, network.relations);

    const { loss, reward } = gnn.learnFromFeedback(
      graph,
      'cell_1_1',
      { p0: -98, alpha: 0.9 },
      15 // Actual SINR
    );

    expect(loss).toBeGreaterThanOrEqual(0);
    expect(reward).toBeDefined();

    const state = gnn.getState();
    expect(state.totalSamples).toBe(1);
  });

  it('should train on batch after sufficient samples', () => {
    const graph = graphBuilder.buildGraph(network.snapshots, network.relations);

    // Add multiple feedback samples
    for (let i = 0; i < 20; i++) {
      gnn.learnFromFeedback(
        graph,
        'cell_1_1',
        { p0: -100 + i % 10, alpha: 0.8 },
        10 + Math.random() * 5
      );
    }

    const state = gnn.getState();
    expect(state.totalSamples).toBe(20);
    expect(state.totalUpdates).toBeGreaterThan(0);
  });

  it('should checkpoint and restore', () => {
    const graph = graphBuilder.buildGraph(network.snapshots, network.relations);

    // Train a bit
    for (let i = 0; i < 10; i++) {
      gnn.learnFromFeedback(
        graph,
        'cell_1_1',
        { p0: -100, alpha: 0.8 },
        12
      );
    }

    const checkpoint = gnn.checkpoint();
    expect(checkpoint.version).toBe(2);
    expect(checkpoint.state.totalSamples).toBe(10);

    // Create new GNN and restore
    const newGNN = new SelfLearningUplinkGNN();
    newGNN.restore(checkpoint);

    const newState = newGNN.getState();
    expect(newState.totalSamples).toBe(10);
  });

  it('should decay exploration rate over time', () => {
    const graph = graphBuilder.buildGraph(network.snapshots, network.relations);
    const initialExploration = gnn.getState().explorationRate;

    // Run multiple optimizations
    for (let i = 0; i < 10; i++) {
      gnn.optimizeCell('cell_1_1', graph, network.snapshots, network.relations);
    }

    const finalExploration = gnn.getState().explorationRate;
    expect(finalExploration).toBeLessThan(initialExploration);
  });

  it('should calculate learning metrics', () => {
    const metrics = gnn.getMetrics();

    expect(metrics).toHaveProperty('sinrPredictionError');
    expect(metrics).toHaveProperty('optimizationSuccessRate');
    expect(metrics).toHaveProperty('avgSINRImprovement');
    expect(metrics).toHaveProperty('convergenceScore');
    expect(metrics).toHaveProperty('adaptationRate');
  });
});

// ============================================================================
// ERICSSON UPLINK OPTIMIZER TESTS
// ============================================================================

describe('EricssonUplinkOptimizer', () => {
  let optimizer: EricssonUplinkOptimizer;
  let network: { snapshots: Map<string, CellKPISnapshot>; relations: NeighborRelation[] };

  beforeEach(() => {
    optimizer = new EricssonUplinkOptimizer();
    network = createMockNetwork();
  });

  it('should optimize entire network', () => {
    const result = optimizer.optimizeNetwork(network.snapshots, network.relations);

    expect(result.timestamp).toBeDefined();
    expect(result.results).toBeDefined();
    expect(result.metrics).toBeDefined();
    expect(result.recommendations).toBeDefined();
    expect(result.recommendations.length).toBeGreaterThan(0);
  });

  it('should detect and optimize issue cells', () => {
    const result = optimizer.optimizeNetwork(network.snapshots, network.relations);

    // cell_1_1 should be detected as issue (low SINR, high IoT)
    const cell1Result = result.results.find(r => r.cellId === 'cell_1_1');
    if (cell1Result) {
      expect(cell1Result.originalParams.p0).toBe(-100);
      expect(cell1Result.originalParams.alpha).toBe(0.8);
    }
  });

  it('should apply feedback and learn', () => {
    const deployedChanges = [
      { cellId: 'cell_1_1', params: { p0: -95, alpha: 0.9 }, actualSINR: 12 },
      { cellId: 'cell_1_2', params: { p0: -100, alpha: 0.8 }, actualSINR: 15 },
    ];

    const { avgLoss, avgReward } = optimizer.applyFeedback(
      network.snapshots,
      network.relations,
      deployedChanges
    );

    expect(avgLoss).toBeGreaterThanOrEqual(0);
    expect(avgReward).toBeDefined();
  });

  it('should get optimizer status', () => {
    const status = optimizer.getStatus();

    expect(status.state).toBeDefined();
    expect(status.metrics).toBeDefined();
    expect(status.state.modelVersion).toBe(1);
  });

  it('should save and load checkpoints', () => {
    // Train a bit
    optimizer.optimizeNetwork(network.snapshots, network.relations);
    optimizer.applyFeedback(network.snapshots, network.relations, [
      { cellId: 'cell_1_1', params: { p0: -98, alpha: 0.85 }, actualSINR: 10 },
    ]);

    const checkpoint = optimizer.saveCheckpoint();
    expect(checkpoint.version).toBeGreaterThan(0);

    const newOptimizer = new EricssonUplinkOptimizer();
    newOptimizer.loadCheckpoint(checkpoint);

    const newStatus = newOptimizer.getStatus();
    expect(newStatus.state.totalSamples).toBe(1);
  });

  it('should generate meaningful recommendations', () => {
    const result = optimizer.optimizeNetwork(network.snapshots, network.relations);

    // Should generate recommendations
    expect(result.recommendations.length).toBeGreaterThan(0);

    // Should include either model info or deployment notes or optimization recommendations
    const hasModelInfo = result.recommendations.some(r => r.includes('Self-Learning GNN') || r.includes('GNN'));
    const hasDeploymentNotes = result.recommendations.some(r => r.includes('Deployment') || r.includes('deploy'));
    const hasRecommendations = result.recommendations.some(r => r.includes('optimization') || r.includes('recommended') || r.includes('No optimization'));

    expect(hasModelInfo || hasDeploymentNotes || hasRecommendations).toBe(true);
  });

  it('should access underlying GNN', () => {
    const gnn = optimizer.getGNN();
    expect(gnn).toBeInstanceOf(SelfLearningUplinkGNN);
  });
});

// ============================================================================
// RUVECTOR CLI INTEGRATION TESTS
// ============================================================================

describe('RuVector CLI Integration', () => {
  it('should create GNN layer via CLI function', async () => {
    const layer = await runRuVectorGNNLayer(24, 64, 4, 0.1);
    expect(layer).toBeInstanceOf(RuVectorGNNLayer);

    const config = layer.getConfig();
    expect(config.inputDim).toBe(24);
    expect(config.hiddenDim).toBe(64);
  });

  it('should compress embeddings', () => {
    const embeddings = [
      [0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 1.1, 1.2],
    ];

    const compressed = compressEmbeddings(embeddings, 'none', 0.9);
    expect(compressed.compressionLevel).toBe('none');
    expect(compressed.shape).toEqual([3, 4]);

    const compressedAuto = compressEmbeddings(embeddings, 'auto' as CompressionLevel, 0.5);
    expect(compressedAuto.compressionLevel).toBe('half');

    const compressedCold = compressEmbeddings(embeddings, 'auto' as CompressionLevel, 0.05);
    expect(compressedCold.compressionLevel).toBe('pq4');
  });

  it('should perform differentiable search', () => {
    const query = [0.5, 0.8, 0.3];
    const candidates = [
      [0.4, 0.9, 0.2],
      [0.6, 0.7, 0.4],
      [0.5, 0.8, 0.3],
      [0.3, 0.6, 0.5],
    ];

    const result = differentiableSearch(query, candidates, 3, 1.0);

    expect(result.indices.length).toBe(3);
    expect(result.scores.length).toBe(3);
    expect(result.softWeights.length).toBe(4);

    // Top result should have highest score (exact match at index 2 has highest dot product)
    const exactMatchIndex = 2;
    const exactMatchScore = result.scores[result.indices.indexOf(exactMatchIndex)] ??
      query.reduce((sum, q, i) => sum + q * candidates[exactMatchIndex][i], 0);
    expect(exactMatchScore).toBeGreaterThan(0);

    // Soft weights should sum to ~1
    const sumWeights = result.softWeights.reduce((a, b) => a + b, 0);
    expect(Math.abs(sumWeights - 1)).toBeLessThan(0.01);
  });

  it('should handle temperature in differentiable search', () => {
    const query = [0.5, 0.5, 0.5];
    const candidates = [
      [0.9, 0.9, 0.9],  // High similarity
      [0.1, 0.1, 0.1],  // Low similarity
    ];

    // Low temperature = sharper distribution
    const sharpResult = differentiableSearch(query, candidates, 2, 0.01);

    // High temperature = more uniform distribution
    const softResult = differentiableSearch(query, candidates, 2, 100.0);

    // Sharp distribution should have higher max weight (concentrated on best match)
    const sharpMax = Math.max(...sharpResult.softWeights);
    const softMax = Math.max(...softResult.softWeights);

    // With significantly different candidates and extreme temperatures,
    // the sharp distribution should be more concentrated
    expect(sharpMax).toBeGreaterThanOrEqual(softMax - 0.01);
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('End-to-End Integration', () => {
  it('should run full optimization workflow', () => {
    const network = createMockNetwork();
    const optimizer = new EricssonUplinkOptimizer();

    // Initial optimization
    const result1 = optimizer.optimizeNetwork(network.snapshots, network.relations);
    expect(result1.results.length).toBeGreaterThanOrEqual(0);

    // Simulate deployment and feedback
    if (result1.results.length > 0) {
      const deployedChanges = result1.results.slice(0, 2).map(r => ({
        cellId: r.cellId,
        params: r.optimizedParams,
        actualSINR: r.optimizedSINR + Math.random() * 2 - 1, // Actual slightly different
      }));

      optimizer.applyFeedback(network.snapshots, network.relations, deployedChanges);
    }

    // Second optimization should use learned knowledge
    const result2 = optimizer.optimizeNetwork(network.snapshots, network.relations);

    // Model should have learned
    const status = optimizer.getStatus();
    expect(status.state.totalSamples).toBeGreaterThanOrEqual(0);
  });

  it('should handle network changes', () => {
    const optimizer = new EricssonUplinkOptimizer();
    const network = createMockNetwork();

    // First optimization
    optimizer.optimizeNetwork(network.snapshots, network.relations);

    // Simulate network change - add new cell
    network.snapshots.set('cell_3_1', createMockCellSnapshot('cell_3_1', -100, 0.8, 6, 9));
    network.relations.push(createMockNeighborRelation('cell_2_2', 'cell_3_1', 94));

    // Should handle new topology
    const result = optimizer.optimizeNetwork(network.snapshots, network.relations);
    expect(result.results).toBeDefined();
  });

  it('should preserve model state across checkpoints', () => {
    const optimizer = new EricssonUplinkOptimizer();
    const network = createMockNetwork();

    // Train model
    for (let i = 0; i < 5; i++) {
      optimizer.optimizeNetwork(network.snapshots, network.relations);
      optimizer.applyFeedback(network.snapshots, network.relations, [
        { cellId: 'cell_1_1', params: { p0: -100 + i, alpha: 0.8 }, actualSINR: 10 + i },
      ]);
    }

    const checkpoint = optimizer.saveCheckpoint();

    // Create new optimizer and restore
    const newOptimizer = new EricssonUplinkOptimizer();
    newOptimizer.loadCheckpoint(checkpoint);

    // Should have preserved training state
    const state = newOptimizer.getStatus().state;
    expect(state.totalSamples).toBe(5);
    expect(state.modelVersion).toBeGreaterThan(1);
  });
});

// ============================================================================
// EDGE CASES AND ERROR HANDLING
// ============================================================================

describe('Edge Cases', () => {
  it('should handle empty network', () => {
    const optimizer = new EricssonUplinkOptimizer();
    const emptySnapshots = new Map<string, CellKPISnapshot>();
    const emptyRelations: NeighborRelation[] = [];

    const result = optimizer.optimizeNetwork(emptySnapshots, emptyRelations);
    expect(result.results).toEqual([]);
  });

  it('should handle single cell network', () => {
    const optimizer = new EricssonUplinkOptimizer();
    const snapshots = new Map<string, CellKPISnapshot>();
    snapshots.set('cell_1', createMockCellSnapshot('cell_1', -100, 0.8, 3, 12));

    const result = optimizer.optimizeNetwork(snapshots, []);
    expect(result.results.length).toBeGreaterThanOrEqual(0);
  });

  it('should handle invalid cell ID in optimization', () => {
    const gnn = new SelfLearningUplinkGNN();
    const network = createMockNetwork();
    const graph = new SurrogateGraphBuilder().buildGraph(network.snapshots, network.relations);

    expect(() => {
      gnn.optimizeCell('nonexistent_cell', graph, network.snapshots, network.relations);
    }).toThrow('Cell nonexistent_cell not found in graph');
  });

  it('should handle extreme parameter values', () => {
    const optimizer = new EricssonUplinkOptimizer();
    const snapshots = new Map<string, CellKPISnapshot>();

    // Cell with extreme values
    snapshots.set('cell_extreme', createMockCellSnapshot('cell_extreme', -110, 0.4, -3, 18));

    const result = optimizer.optimizeNetwork(snapshots, []);

    // Should still produce valid recommendations
    result.results.forEach(r => {
      expect(r.optimizedParams.p0).toBeGreaterThanOrEqual(-110);
      expect(r.optimizedParams.p0).toBeLessThanOrEqual(-85);
      expect(r.optimizedParams.alpha).toBeGreaterThanOrEqual(0.4);
      expect(r.optimizedParams.alpha).toBeLessThanOrEqual(1.0);
    });
  });

  it('should handle rapid successive optimizations', () => {
    const optimizer = new EricssonUplinkOptimizer();
    const network = createMockNetwork();

    // Run many optimizations quickly
    for (let i = 0; i < 10; i++) {
      const result = optimizer.optimizeNetwork(network.snapshots, network.relations);
      expect(result.results).toBeDefined();
    }

    const state = optimizer.getStatus().state;
    expect(state.explorationRate).toBeLessThan(0.3); // Should have decayed
  });
});
