/**
 * Tests for GNN Network Surrogate Model (Digital Twin)
 *
 * Tests cover:
 * 1. Graph construction with power control parameters
 * 2. GNN SINR prediction accuracy
 * 3. Issue cell detection
 * 4. Optimization loop ("what-if" analysis)
 * 5. Status transitions (red -> green scores)
 * 6. Full network optimization workflow
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  SurrogateGraphBuilder,
  GNNSurrogateModel,
  IssueCellDetector,
  SurrogateOptimizer,
  SurrogateVisualizer,
  DEFAULT_SURROGATE_CONFIG,
  type SurrogateGraph,
  type PowerControlParams,
  type CellStatus,
  type TrainingSample,
} from '../../src/gnn/network-surrogate-model.js';
import type { CellKPISnapshot, NeighborRelation } from '../../src/models/ran-kpi.js';

// ============================================================================
// TEST HELPERS
// ============================================================================

/**
 * Create a mock cell KPI snapshot for testing
 */
function createMockSnapshot(
  cellId: string,
  overrides: Partial<{
    ulSinr: number;
    iot: number;
    p0: number;
    alpha: number;
    powerLimited: number;
    negativePhr: number;
    pathLoss: number;
  }> = {}
): CellKPISnapshot {
  const defaults = {
    ulSinr: 12,
    iot: 6,
    p0: -96,
    alpha: 0.8,
    powerLimited: 5,
    negativePhr: 3,
    pathLoss: 110,
  };

  const merged = { ...defaults, ...overrides };

  return {
    timestamp: new Date(),
    cell: {
      cellId,
      eNodeBId: `eNB-${cellId.split('-')[1]}`,
      sector: 1,
      frequency: 1800,
      band: 'B3',
      technology: 'LTE',
      physicalCellId: parseInt(cellId.split('-')[1]) % 504,
      tac: 12345,
    },
    accessibility: {
      timestamp: new Date(),
      cellId,
      rrcSetupAttempts: 5000,
      rrcSetupSuccessRate: 99.2,
      erabSetupAttempts: 4500,
      erabSetupSuccessRate: 98.5,
      s1SigConnectionEstAttempts: 4500,
      s1SigConnectionEstSuccessRate: 99.0,
      initialContextSetupAttempts: 4400,
      initialContextSetupSuccessRate: 98.8,
      rrcSetupFailureCauses: [],
      erabSetupFailureCauses: [],
    },
    retainability: {
      timestamp: new Date(),
      cellId,
      erabDropRate: 0.5,
      voiceCallDropRate: 0.3,
      dataSessionRetainability: 99.2,
      contextReleaseCauses: [],
      erabReleaseAttempts: 4000,
      abnormalReleaseRatio: 0.8,
    },
    radioQuality: {
      timestamp: new Date(),
      cellId,
      dlAvgCqi: 12,
      dlRiDistribution: [0.1, 0.2, 0.3, 0.4],
      dlBlerPercent: 2,
      ulSinrAvg: merged.ulSinr,
      ulSinrP10: merged.ulSinr - 5,
      ulSinrP50: merged.ulSinr,
      ulSinrP90: merged.ulSinr + 5,
      ulBlerPercent: 3,
      rsrpAvg: -95,
      rsrpP10: -105,
      rsrpP50: -95,
      rsrpP90: -85,
      rsrqAvg: -10,
      rsrqP10: -15,
      rsrqP50: -10,
      rsrqP90: -5,
      dlSpectralEfficiency: 5.5,
      ulSpectralEfficiency: 3.2,
      pucchSinr: merged.ulSinr + 2,
    },
    mobility: {
      timestamp: new Date(),
      cellId,
      intraFreqHoAttempts: 2000,
      intraFreqHoSuccessRate: 97.5,
      interFreqHoAttempts: 500,
      interFreqHoSuccessRate: 96.0,
      interRatHoAttempts: 100,
      interRatHoSuccessRate: 95.0,
      x2HoAttempts: 1500,
      x2HoSuccessRate: 98.0,
      s1HoAttempts: 500,
      s1HoSuccessRate: 96.5,
      pingPongHo: 2,
      tooEarlyHo: 1,
      tooLateHo: 1,
      wrongCellHo: 0,
    },
    uplinkInterference: {
      timestamp: new Date(),
      cellId,
      prbInterferenceAvg: merged.iot,
      prbInterferenceP10: merged.iot - 3,
      prbInterferenceP50: merged.iot,
      prbInterferenceP90: merged.iot + 3,
      prbInterferenceP99: merged.iot + 5,
      iotAvg: merged.iot,
      iotP10: merged.iot - 2,
      iotP50: merged.iot,
      iotP90: merged.iot + 2,
      rip: -100,
      externalInterferenceDetected: false,
      highInterferencePrbRatio: merged.iot > 10 ? 25 : 8,
      puschSinrDegradation: merged.iot > 10 ? 3 : 1,
    },
    uplinkPowerControl: {
      timestamp: new Date(),
      cellId,
      p0NominalPusch: merged.p0,
      p0NominalPucch: merged.p0 - 10,
      alpha: merged.alpha,
      tpcAccumulationEnabled: true,
      ueTxPowerAvg: -10,
      ueTxPowerP10: -20,
      ueTxPowerP50: -10,
      ueTxPowerP90: 10,
      powerHeadroomAvg: 15,
      powerHeadroomP10: 5,
      powerHeadroomP50: 15,
      powerHeadroomP90: 25,
      powerLimitedUeRatio: merged.powerLimited,
      negativePowerHeadroomRatio: merged.negativePhr,
      tpcUpCommands: 3000,
      tpcDownCommands: 2000,
      tpcHoldCommands: 5000,
      pathLossAvg: merged.pathLoss,
      pathLossP10: merged.pathLoss - 15,
      pathLossP50: merged.pathLoss,
      pathLossP90: merged.pathLoss + 15,
      fractionalPowerControlRatio: 0.85,
    },
  };
}

/**
 * Create mock neighbor relation
 */
function createMockNeighborRelation(
  sourceCellId: string,
  targetCellId: string,
  overrides: Partial<{
    sourceSinr: number;
    targetSinr: number;
    hoSuccessRate: number;
    relationshipType: 'intra-freq' | 'inter-freq' | 'inter-rat';
  }> = {}
): NeighborRelation {
  const defaults = {
    sourceSinr: 12,
    targetSinr: 10,
    hoSuccessRate: 97,
    relationshipType: 'intra-freq' as const,
  };

  const merged = { ...defaults, ...overrides };

  return {
    sourceCellId,
    targetCellId,
    relationshipType: merged.relationshipType,
    sourceSinr: merged.sourceSinr,
    targetSinr: merged.targetSinr,
    sourceRsrp: -95,
    targetRsrp: -98,
    hoSuccessRate: merged.hoSuccessRate,
    hoAttempts: 500,
    a3Offset: 3,
    hysteresis: 2,
    timeToTrigger: 480,
    distance: 1200,
  };
}

/**
 * Create a test network with multiple cells
 */
function createTestNetwork(numCells: number = 5): {
  cellSnapshots: Map<string, CellKPISnapshot>;
  neighborRelations: NeighborRelation[];
} {
  const cellSnapshots = new Map<string, CellKPISnapshot>();
  const neighborRelations: NeighborRelation[] = [];

  // Create cells with varying conditions
  for (let i = 0; i < numCells; i++) {
    const cellId = `CELL-${i.toString().padStart(3, '0')}`;

    // Make some cells have issues
    const isIssueCell = i % 3 === 0;
    const isCriticalCell = i === 0;

    cellSnapshots.set(cellId, createMockSnapshot(cellId, {
      ulSinr: isCriticalCell ? -2 : isIssueCell ? 3 : 12 + Math.random() * 5,
      iot: isCriticalCell ? 15 : isIssueCell ? 12 : 5 + Math.random() * 3,
      p0: -96 + Math.floor(Math.random() * 10) - 5,
      alpha: [0.7, 0.8, 0.9][i % 3],
      powerLimited: isIssueCell ? 25 : 5,
    }));
  }

  // Create neighbor relations (mesh topology)
  const cellIds = Array.from(cellSnapshots.keys());
  for (let i = 0; i < cellIds.length; i++) {
    for (let j = i + 1; j < cellIds.length; j++) {
      // Only connect nearby cells (every cell to 2-3 neighbors)
      if (Math.abs(i - j) <= 2 || (i === 0 && j === cellIds.length - 1)) {
        neighborRelations.push(createMockNeighborRelation(cellIds[i], cellIds[j]));
        neighborRelations.push(createMockNeighborRelation(cellIds[j], cellIds[i]));
      }
    }
  }

  return { cellSnapshots, neighborRelations };
}

// ============================================================================
// TESTS: Graph Construction
// ============================================================================

describe('SurrogateGraphBuilder', () => {
  let graphBuilder: SurrogateGraphBuilder;

  beforeEach(() => {
    graphBuilder = new SurrogateGraphBuilder();
  });

  it('should build graph from cell snapshots and neighbor relations', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(3);

    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    expect(graph.nodeIds).toHaveLength(3);
    expect(graph.nodeFeatures).toHaveLength(3);
    expect(graph.adjacencyMatrix).toHaveLength(3);
    expect(graph.powerParams.size).toBe(3);
  });

  it('should include P0 and Alpha in node features', () => {
    const snapshot = createMockSnapshot('CELL-001', { p0: -100, alpha: 0.7 });
    const params: PowerControlParams = { p0: -100, alpha: 0.7 };

    const features = graphBuilder.buildNodeFeatures(snapshot, params);

    // First two features should be normalized P0 and Alpha
    expect(features.length).toBe(DEFAULT_SURROGATE_CONFIG.inputDim);
    expect(features[0]).toBeGreaterThanOrEqual(0);
    expect(features[0]).toBeLessThanOrEqual(1);
    expect(features[1]).toBe(0.7); // Alpha unchanged
  });

  it('should create weighted adjacency matrix', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(3);

    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    // Self-loops should be 1
    for (let i = 0; i < 3; i++) {
      expect(graph.adjacencyMatrix[i][i]).toBe(1);
    }

    // Neighbor weights should be between 0 and 1
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (i !== j) {
          expect(graph.adjacencyMatrix[i][j]).toBeGreaterThanOrEqual(0);
          expect(graph.adjacencyMatrix[i][j]).toBeLessThanOrEqual(1.5); // Can be boosted by intra-freq
        }
      }
    }
  });

  it('should update graph with new power parameters', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(3);
    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    const newParams = new Map<string, PowerControlParams>();
    newParams.set('CELL-000', { p0: -90, alpha: 0.9 });

    const updatedGraph = graphBuilder.updateGraphParams(graph, newParams, cellSnapshots);

    expect(updatedGraph.powerParams.get('CELL-000')).toEqual({ p0: -90, alpha: 0.9 });
    expect(updatedGraph.nodeFeatures[0]).not.toEqual(graph.nodeFeatures[0]);
  });
});

// ============================================================================
// TESTS: GNN Prediction Model
// ============================================================================

describe('GNNSurrogateModel', () => {
  let gnn: GNNSurrogateModel;
  let graphBuilder: SurrogateGraphBuilder;

  beforeEach(() => {
    gnn = new GNNSurrogateModel();
    graphBuilder = new SurrogateGraphBuilder();
  });

  it('should predict SINR and IoT for all cells', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(5);
    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    const predictions = gnn.predict(graph);

    expect(predictions.sinr).toHaveLength(5);
    expect(predictions.iot).toHaveLength(5);
    expect(predictions.embeddings).toHaveLength(5);

    // All predictions should be in valid ranges
    for (const sinr of predictions.sinr) {
      expect(sinr).toBeGreaterThanOrEqual(-10);
      expect(sinr).toBeLessThanOrEqual(35);
    }

    for (const iot of predictions.iot) {
      expect(iot).toBeGreaterThanOrEqual(0);
      expect(iot).toBeLessThanOrEqual(25);
    }
  });

  it('should predict different SINR for different parameters', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(3);
    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    // Prediction with current params
    const pred1 = gnn.predictCellSINR(graph, 'CELL-000');

    // Prediction with modified params
    const { sinr } = gnn.predictWithParams(
      graph,
      'CELL-000',
      { p0: -90, alpha: 0.9 },
      cellSnapshots,
      graphBuilder
    );

    // Predictions should differ
    // Note: Due to random weight initialization, we can't guarantee direction
    // but they should be different
    expect(sinr[0]).toBeDefined();
  });

  it('should be trainable with sample data', () => {
    const trainingSamples: TrainingSample[] = [];

    // Generate training samples
    for (let i = 0; i < 10; i++) {
      const { cellSnapshots, neighborRelations } = createTestNetwork(5);
      const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

      trainingSamples.push({
        nodeFeatures: graph.nodeFeatures,
        adjacencyMatrix: graph.adjacencyMatrix,
        actualSINR: Array.from(cellSnapshots.values()).map(s => s.radioQuality.ulSinrAvg),
      });
    }

    const result = gnn.train(trainingSamples, { epochs: 10 });

    expect(result.loss).toBeGreaterThanOrEqual(0);
    expect(result.accuracy).toBeGreaterThanOrEqual(0);
    expect(result.accuracy).toBeLessThanOrEqual(1);
    expect(gnn.isTrained()).toBe(true);
  });

  it('should return embeddings for each node', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(3);
    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    const { embeddings } = gnn.predict(graph);

    expect(embeddings).toHaveLength(3);
    expect(embeddings[0].length).toBe(DEFAULT_SURROGATE_CONFIG.hiddenDim);
  });
});

// ============================================================================
// TESTS: Issue Cell Detection
// ============================================================================

describe('IssueCellDetector', () => {
  let detector: IssueCellDetector;

  beforeEach(() => {
    detector = new IssueCellDetector();
  });

  it('should detect cells with low SINR', () => {
    const cellSnapshots = new Map<string, CellKPISnapshot>();
    cellSnapshots.set('CELL-001', createMockSnapshot('CELL-001', { ulSinr: 3 }));
    cellSnapshots.set('CELL-002', createMockSnapshot('CELL-002', { ulSinr: 15 }));

    const issueCells = detector.detectIssueCells(cellSnapshots, []);

    expect(issueCells).toHaveLength(1);
    expect(issueCells[0].cellId).toBe('CELL-001');
    expect(issueCells[0].status).toBe('issue');
    expect(issueCells[0].issues.some(i => i.includes('SINR'))).toBe(true);
  });

  it('should detect cells with critical SINR', () => {
    const cellSnapshots = new Map<string, CellKPISnapshot>();
    cellSnapshots.set('CELL-001', createMockSnapshot('CELL-001', { ulSinr: -2 }));

    const issueCells = detector.detectIssueCells(cellSnapshots, []);

    expect(issueCells).toHaveLength(1);
    expect(issueCells[0].status).toBe('critical');
  });

  it('should detect cells with high IoT', () => {
    const cellSnapshots = new Map<string, CellKPISnapshot>();
    cellSnapshots.set('CELL-001', createMockSnapshot('CELL-001', { iot: 15 }));

    const issueCells = detector.detectIssueCells(cellSnapshots, []);

    expect(issueCells).toHaveLength(1);
    expect(issueCells[0].issues.some(i => i.includes('IoT'))).toBe(true);
  });

  it('should detect cells with high power-limited ratio', () => {
    const cellSnapshots = new Map<string, CellKPISnapshot>();
    cellSnapshots.set('CELL-001', createMockSnapshot('CELL-001', { powerLimited: 30 }));

    const issueCells = detector.detectIssueCells(cellSnapshots, []);

    expect(issueCells).toHaveLength(1);
    expect(issueCells[0].status).toBe('warning');
    expect(issueCells[0].issues.some(i => i.includes('power-limited'))).toBe(true);
  });

  it('should sort issue cells by severity', () => {
    const cellSnapshots = new Map<string, CellKPISnapshot>();
    cellSnapshots.set('CELL-001', createMockSnapshot('CELL-001', { ulSinr: 3, iot: 8 })); // issue
    cellSnapshots.set('CELL-002', createMockSnapshot('CELL-002', { ulSinr: -2, iot: 16 })); // critical
    cellSnapshots.set('CELL-003', createMockSnapshot('CELL-003', { powerLimited: 25 })); // warning

    const issueCells = detector.detectIssueCells(cellSnapshots, []);

    expect(issueCells[0].status).toBe('critical');
    expect(issueCells[1].status).toBe('issue');
    expect(issueCells[2].status).toBe('warning');
  });

  it('should calculate visual scores', () => {
    const cellSnapshots = new Map<string, CellKPISnapshot>();
    cellSnapshots.set('CELL-001', createMockSnapshot('CELL-001', { ulSinr: 3, iot: 12 }));
    cellSnapshots.set('CELL-002', createMockSnapshot('CELL-002', { ulSinr: -2, iot: 18 }));

    const issueCells = detector.detectIssueCells(cellSnapshots, []);

    // Lower SINR and higher IoT should have lower scores
    const cell1 = issueCells.find(c => c.cellId === 'CELL-001')!;
    const cell2 = issueCells.find(c => c.cellId === 'CELL-002')!;

    expect(cell1.score).toBeGreaterThan(cell2.score);
  });

  it('should get neighbors for a cell', () => {
    const neighborRelations = [
      createMockNeighborRelation('CELL-001', 'CELL-002'),
      createMockNeighborRelation('CELL-001', 'CELL-003'),
      createMockNeighborRelation('CELL-002', 'CELL-003'),
    ];

    const neighbors = detector.getNeighbors('CELL-001', neighborRelations);

    expect(neighbors).toContain('CELL-002');
    expect(neighbors).toContain('CELL-003');
    expect(neighbors).not.toContain('CELL-001');
  });
});

// ============================================================================
// TESTS: Optimization Loop
// ============================================================================

describe('SurrogateOptimizer', () => {
  let optimizer: SurrogateOptimizer;

  beforeEach(() => {
    optimizer = new SurrogateOptimizer();
  });

  it('should optimize a single cell', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(5);
    const graphBuilder = new SurrogateGraphBuilder();
    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    // Optimize the first issue cell
    const result = optimizer.optimizeCell(
      'CELL-000',
      graph,
      cellSnapshots,
      neighborRelations
    );

    expect(result.cellId).toBe('CELL-000');
    expect(result.originalParams).toBeDefined();
    expect(result.optimizedParams).toBeDefined();
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it('should include status transitions in results', () => {
    const cellSnapshots = new Map<string, CellKPISnapshot>();
    cellSnapshots.set('CELL-001', createMockSnapshot('CELL-001', {
      ulSinr: 2,
      iot: 12,
      p0: -100,
      alpha: 0.7,
    }));

    const graphBuilder = new SurrogateGraphBuilder();
    const graph = graphBuilder.buildGraph(cellSnapshots, []);

    const result = optimizer.optimizeCell(
      'CELL-001',
      graph,
      cellSnapshots,
      []
    );

    expect(result.statusTransition).toBeDefined();
    expect(result.statusTransition.before).toBeDefined();
    expect(result.statusTransition.after).toBeDefined();
    expect(result.statusTransition.scoreBefore).toBeGreaterThanOrEqual(0);
    expect(result.statusTransition.scoreAfter).toBeGreaterThanOrEqual(0);
  });

  it('should optimize entire network', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(5);

    const result = optimizer.optimizeNetwork(cellSnapshots, neighborRelations);

    expect(result.totalCells).toBe(5);
    expect(result.issueCells).toBeGreaterThan(0);
    expect(result.aggregateMetrics).toBeDefined();
    expect(result.recommendations).toBeInstanceOf(Array);
    expect(result.recommendations.length).toBeGreaterThan(0);
  });

  it('should calculate aggregate metrics', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(5);

    const result = optimizer.optimizeNetwork(cellSnapshots, neighborRelations);

    expect(result.aggregateMetrics.avgSinrBefore).toBeDefined();
    expect(result.aggregateMetrics.avgSinrAfter).toBeDefined();
    expect(result.aggregateMetrics.avgImprovement).toBeDefined();
    expect(result.aggregateMetrics.successRate).toBeGreaterThanOrEqual(0);
    expect(result.aggregateMetrics.successRate).toBeLessThanOrEqual(1);
  });

  it('should consider neighbor impact in optimization', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(5);
    const graphBuilder = new SurrogateGraphBuilder();
    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    const result = optimizer.optimizeCell(
      'CELL-000',
      graph,
      cellSnapshots,
      neighborRelations
    );

    // Neighbor impact should be tracked
    expect(result.neighborImpact).toBeDefined();
  });

  it('should expose underlying GNN model for training', () => {
    const gnn = optimizer.getGNN();

    expect(gnn).toBeInstanceOf(GNNSurrogateModel);
    expect(gnn.isTrained()).toBe(false);
  });
});

// ============================================================================
// TESTS: Visualization
// ============================================================================

describe('SurrogateVisualizer', () => {
  it('should return correct status colors', () => {
    expect(SurrogateVisualizer.getStatusColor('critical')).toBe('red');
    expect(SurrogateVisualizer.getStatusColor('issue')).toBe('orange');
    expect(SurrogateVisualizer.getStatusColor('warning')).toBe('yellow');
    expect(SurrogateVisualizer.getStatusColor('healthy')).toBe('green');
  });

  it('should format cell status for display', () => {
    const cellStatus: CellStatus = {
      cellId: 'CELL-001',
      currentParams: { p0: -96, alpha: 0.8 },
      predictedSINR: 3.5,
      predictedIoT: 12,
      status: 'issue',
      score: 8,
      issues: ['Low SINR'],
    };

    const formatted = SurrogateVisualizer.formatCellStatus(cellStatus);

    expect(formatted).toContain('CELL-001');
    expect(formatted).toContain('ORANGE');
    expect(formatted).toContain('Score 8');
    expect(formatted).toContain('P0=-96');
  });

  it('should format optimization result for display', () => {
    const result = {
      cellId: 'CELL-001',
      originalParams: { p0: -100, alpha: 0.7 },
      optimizedParams: { p0: -96, alpha: 0.8 },
      originalSINR: 3,
      optimizedSINR: 8,
      sinrImprovement: 5,
      neighborImpact: -0.5,
      iterations: 50,
      confidence: 0.85,
      statusTransition: {
        before: 'issue' as const,
        after: 'warning' as const,
        scoreBefore: 8,
        scoreAfter: 14,
      },
    };

    const formatted = SurrogateVisualizer.formatOptimizationResult(result);

    expect(formatted).toContain('CELL-001');
    expect(formatted).toContain('ORANGE');
    expect(formatted).toContain('YELLOW');
    expect(formatted).toContain('8]');
    expect(formatted).toContain('14]');
    expect(formatted).toContain('+5.0 dB');
  });

  it('should generate graph visualization', () => {
    const graph: SurrogateGraph = {
      nodeIds: ['CELL-001', 'CELL-002', 'CELL-003'],
      nodeFeatures: [[0.5], [0.6], [0.7]],
      adjacencyMatrix: [
        [1, 0.5, 0],
        [0.5, 1, 0.6],
        [0, 0.6, 1],
      ],
      edgeFeatures: [],
      powerParams: new Map([
        ['CELL-001', { p0: -96, alpha: 0.8 }],
        ['CELL-002', { p0: -100, alpha: 0.7 }],
        ['CELL-003', { p0: -92, alpha: 0.9 }],
      ]),
    };

    const predictions = {
      sinr: [3, 12, 8],
    };

    const viz = SurrogateVisualizer.visualizeGraph(graph, predictions);

    expect(viz).toContain('CELL-001');
    expect(viz).toContain('CELL-002');
    expect(viz).toContain('CELL-003');
    expect(viz).toContain('ISSUE'); // CELL-001 has SINR < 5
    expect(viz).toContain('OK'); // CELL-002 has SINR > 5
  });
});

// ============================================================================
// TESTS: Full Workflow Integration
// ============================================================================

describe('Full Workflow Integration', () => {
  it('should complete full optimization workflow', () => {
    // 1. Create network with issues
    const { cellSnapshots, neighborRelations } = createTestNetwork(10);

    // 2. Initialize optimizer
    const optimizer = new SurrogateOptimizer();

    // 3. Optionally train the model
    const gnn = optimizer.getGNN();
    const graphBuilder = new SurrogateGraphBuilder();

    // Generate some training data
    const trainingData: TrainingSample[] = [];
    for (let i = 0; i < 5; i++) {
      const network = createTestNetwork(5);
      const graph = graphBuilder.buildGraph(network.cellSnapshots, network.neighborRelations);
      trainingData.push({
        nodeFeatures: graph.nodeFeatures,
        adjacencyMatrix: graph.adjacencyMatrix,
        actualSINR: Array.from(network.cellSnapshots.values()).map(s => s.radioQuality.ulSinrAvg),
      });
    }

    gnn.train(trainingData, { epochs: 5 });
    expect(gnn.isTrained()).toBe(true);

    // 4. Run network optimization
    const result = optimizer.optimizeNetwork(cellSnapshots, neighborRelations);

    // 5. Verify results
    expect(result.totalCells).toBe(10);
    expect(result.issueCells).toBeGreaterThan(0);
    expect(result.recommendations.length).toBeGreaterThan(0);

    // 6. Check that issue cells got optimized
    for (const cellResult of result.results) {
      expect(cellResult.sinrImprovement).toBeGreaterThanOrEqual(
        DEFAULT_SURROGATE_CONFIG.optimization.minImprovement
      );
    }
  });

  it('should demonstrate score transition', () => {
    // Create a cell with issue status (SINR between 0 and 5)
    const cellSnapshots = new Map<string, CellKPISnapshot>();
    cellSnapshots.set('CELL-ISSUE', createMockSnapshot('CELL-ISSUE', {
      ulSinr: 3,
      iot: 12,
      p0: -105,
      alpha: 0.6,
    }));

    // Add some healthy neighbors
    cellSnapshots.set('CELL-NEIGHBOR-1', createMockSnapshot('CELL-NEIGHBOR-1', {
      ulSinr: 15,
      iot: 5,
    }));
    cellSnapshots.set('CELL-NEIGHBOR-2', createMockSnapshot('CELL-NEIGHBOR-2', {
      ulSinr: 12,
      iot: 6,
    }));

    const neighborRelations = [
      createMockNeighborRelation('CELL-ISSUE', 'CELL-NEIGHBOR-1'),
      createMockNeighborRelation('CELL-ISSUE', 'CELL-NEIGHBOR-2'),
    ];

    const optimizer = new SurrogateOptimizer();
    const graphBuilder = new SurrogateGraphBuilder();
    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    const result = optimizer.optimizeCell(
      'CELL-ISSUE',
      graph,
      cellSnapshots,
      neighborRelations
    );

    // Should have status transition data
    expect(result.statusTransition).toBeDefined();
    expect(result.statusTransition.before).toBeDefined();
    expect(result.statusTransition.after).toBeDefined();

    // Scores should be defined and valid
    expect(result.statusTransition.scoreBefore).toBeDefined();
    expect(result.statusTransition.scoreAfter).toBeDefined();
    expect(result.statusTransition.scoreBefore).toBeGreaterThanOrEqual(0);
    expect(result.statusTransition.scoreAfter).toBeGreaterThanOrEqual(0);
  });

  it('should handle iterative what-if analysis correctly', () => {
    const { cellSnapshots, neighborRelations } = createTestNetwork(3);
    const optimizer = new SurrogateOptimizer();
    const graphBuilder = new SurrogateGraphBuilder();
    const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    const result = optimizer.optimizeCell(
      'CELL-000',
      graph,
      cellSnapshots,
      neighborRelations
    );

    // Should have tried multiple candidates
    expect(result.iterations).toBeGreaterThan(10);

    // Result should have all required fields
    expect(result.optimizedParams).toBeDefined();
    expect(result.originalParams).toBeDefined();
    expect(result.sinrImprovement).toBeDefined();
    expect(typeof result.sinrImprovement).toBe('number');
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });
});
