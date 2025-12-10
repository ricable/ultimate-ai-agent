/**
 * Tests for GNN-Based Interference Optimization System
 *
 * Tests cover:
 * - Single-layer GNN SINR prediction (Digital Twin)
 * - Issue cell detection
 * - Genetic Algorithm optimization
 * - Full optimization loop workflow
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  SINRPredictionGNN,
  IssueCellDetector,
  GeneticOptimizer,
  InterferenceOptimizationLoop,
  DEFAULT_OPTIMIZER_CONFIG,
  type InterferenceOptimizerConfig,
  type IssueCell,
} from '../../src/gnn/interference-optimizer.js';
import type {
  CellKPISnapshot,
  NeighborRelation,
} from '../../src/models/ran-kpi.js';

// ============================================================================
// TEST FIXTURES
// ============================================================================

/**
 * Generate a mock cell KPI snapshot for testing
 */
function createMockSnapshot(
  cellId: string,
  overrides: {
    sinr?: number;
    iot?: number;
    p0?: number;
    alpha?: number;
    powerLimited?: number;
  } = {}
): CellKPISnapshot {
  return {
    timestamp: new Date(),
    cell: {
      cellId,
      enodebId: `enb-${cellId}`,
      sectorId: 0,
      frequency: 2100,
      band: 'B1',
      technology: 'LTE',
      pci: parseInt(cellId.replace('cell-', ''), 10) || 0,
      tac: 1000,
      latitude: 37.7749 + Math.random() * 0.1,
      longitude: -122.4194 + Math.random() * 0.1,
    },
    accessibility: {
      rrcSetupAttempts: 10000,
      rrcSetupSuccess: 9800,
      rrcSetupFailure: 200,
      rrcSetupSuccessRate: 98,
      erabSetupAttempts: 9500,
      erabSetupSuccess: 9400,
      erabSetupFailure: 100,
      erabSetupSuccessRate: 98.9,
      s1SigConnEstabAttempts: 10000,
      s1SigConnEstabSuccess: 9900,
      s1SigConnEstabSuccessRate: 99,
      initialContextSetupAttempts: 9500,
      initialContextSetupSuccess: 9400,
      initialContextSetupSuccessRate: 98.9,
    },
    retainability: {
      erabNormalRelease: 9000,
      erabAbnormalRelease: 100,
      erabDropRate: 1.1,
      voiceCallAttempts: 5000,
      voiceCallDrops: 50,
      voiceCallDropRate: 1.0,
      dataSessionAttempts: 8000,
      dataSessionDrops: 80,
      dataSessionRetainability: 99.0,
    },
    radioQuality: {
      dlAvgCqi: 12,
      dlRi1Ratio: 40,
      dlRi2Ratio: 60,
      dlBlerPercent: 5,
      ulSinrAvg: overrides.sinr ?? 15,
      ulSinrP10: (overrides.sinr ?? 15) - 5,
      ulSinrP50: overrides.sinr ?? 15,
      ulSinrP90: (overrides.sinr ?? 15) + 5,
      ulBlerPercent: 3,
      rsrpAvg: -95,
      rsrpP10: -110,
      rsrpP50: -95,
      rsrpP90: -80,
      rsrqAvg: -10,
      rsrqP10: -15,
      rsrqP50: -10,
      rsrqP90: -5,
      dlSpectralEfficiency: 5.5,
      ulSpectralEfficiency: 3.2,
    },
    mobility: {
      intraFreqHoAttempts: 2000,
      intraFreqHoSuccess: 1950,
      intraFreqHoFailure: 50,
      intraFreqHoSuccessRate: 97.5,
      interFreqHoAttempts: 500,
      interFreqHoSuccess: 480,
      interFreqHoFailure: 20,
      interFreqHoSuccessRate: 96,
      interRatHoAttempts: 100,
      interRatHoSuccess: 95,
      interRatHoFailure: 5,
      interRatHoSuccessRate: 95,
      x2HoAttempts: 1800,
      x2HoSuccess: 1760,
      x2HoSuccessRate: 97.8,
      s1HoAttempts: 200,
      s1HoSuccess: 190,
      s1HoSuccessRate: 95,
      tooEarlyHo: 10,
      tooLateHo: 15,
      wrongCellHo: 5,
      pingPongHo: 20,
      incomingHoTotal: 1900,
      outgoingHoTotal: 2100,
    },
    uplinkInterference: {
      prbUlInterferenceAvg: -100,
      prbUlInterferenceP10: -105,
      prbUlInterferenceP50: -100,
      prbUlInterferenceP90: -95,
      prbUlInterferenceP99: -90,
      iotAvg: overrides.iot ?? 6,
      iotP95: (overrides.iot ?? 6) + 2,
      rip: -102,
      externalInterferenceDetected: false,
      externalInterferenceLevel: 'none',
      puschSinrDegradation: 1,
      highInterferencePrbRatio: 10,
    },
    uplinkPowerControl: {
      p0NominalPusch: overrides.p0 ?? -96,
      p0NominalPucch: -106,
      alpha: overrides.alpha ?? 0.8,
      ueTxPowerAvg: 10,
      ueTxPowerP10: 0,
      ueTxPowerP50: 10,
      ueTxPowerP90: 18,
      ueTxPowerMax: 23,
      powerHeadroomAvg: 13,
      powerHeadroomP10: 5,
      powerHeadroomP50: 13,
      powerHeadroomP90: 20,
      negativePowerHeadroomRatio: overrides.powerLimited ? overrides.powerLimited * 0.5 : 5,
      pathLossAvg: 120,
      pathLossP10: 100,
      pathLossP50: 120,
      pathLossP90: 140,
      tpcUpCommands: 5000,
      tpcDownCommands: 4500,
      tpcAccumulatedOffset: 2,
      powerLimitedUeRatio: overrides.powerLimited ?? 5,
    },
  };
}

/**
 * Generate mock neighbor relations
 */
function createMockNeighborRelations(cellIds: string[]): NeighborRelation[] {
  const relations: NeighborRelation[] = [];

  for (let i = 0; i < cellIds.length; i++) {
    for (let j = i + 1; j < cellIds.length; j++) {
      // Create bidirectional relations
      relations.push({
        sourceCellId: cellIds[i],
        targetCellId: cellIds[j],
        relationshipType: 'intra-freq',
        sourcePci: i,
        sourceFrequency: 2100,
        sourceRsrp: -95 + Math.random() * 10,
        sourceSinr: 10 + Math.random() * 10,
        targetPci: j,
        targetFrequency: 2100,
        targetRsrp: -95 + Math.random() * 10,
        targetSinr: 10 + Math.random() * 10,
        hoAttempts: 100,
        hoSuccess: 95,
        hoFailure: 5,
        hoSuccessRate: 95,
        a3Offset: 3,
        hysteresis: 2,
        timeToTrigger: 320,
        neighborQuality: 'good',
        distance: 500 + Math.random() * 1000,
      });

      relations.push({
        sourceCellId: cellIds[j],
        targetCellId: cellIds[i],
        relationshipType: 'intra-freq',
        sourcePci: j,
        sourceFrequency: 2100,
        sourceRsrp: -95 + Math.random() * 10,
        sourceSinr: 10 + Math.random() * 10,
        targetPci: i,
        targetFrequency: 2100,
        targetRsrp: -95 + Math.random() * 10,
        targetSinr: 10 + Math.random() * 10,
        hoAttempts: 100,
        hoSuccess: 95,
        hoFailure: 5,
        hoSuccessRate: 95,
        a3Offset: 3,
        hysteresis: 2,
        timeToTrigger: 320,
        neighborQuality: 'good',
        distance: 500 + Math.random() * 1000,
      });
    }
  }

  return relations;
}

// ============================================================================
// SINR PREDICTION GNN TESTS
// ============================================================================

describe('SINRPredictionGNN', () => {
  let gnn: SINRPredictionGNN;

  beforeEach(() => {
    gnn = new SINRPredictionGNN();
  });

  describe('constructor', () => {
    it('should initialize with default configuration', () => {
      expect(gnn).toBeDefined();
    });

    it('should accept custom configuration', () => {
      const customGnn = new SINRPredictionGNN({
        inputDim: 25,
        hiddenDim: 128,
        numHeads: 8,
      });
      expect(customGnn).toBeDefined();
    });
  });

  describe('buildNodeFeatures', () => {
    it('should build feature vector with correct length', () => {
      const snapshot = createMockSnapshot('cell-1');
      const features = gnn.buildNodeFeatures(snapshot, -96, 0.8);

      expect(features).toHaveLength(DEFAULT_OPTIMIZER_CONFIG.inputDim);
    });

    it('should normalize P0 parameter correctly', () => {
      const snapshot = createMockSnapshot('cell-1');

      // Min P0
      const featuresMin = gnn.buildNodeFeatures(snapshot, -110, 0.8);
      expect(featuresMin[0]).toBeCloseTo(0, 1);

      // Max P0
      const featuresMax = gnn.buildNodeFeatures(snapshot, -85, 0.8);
      expect(featuresMax[0]).toBeCloseTo(1, 1);
    });

    it('should include alpha parameter', () => {
      const snapshot = createMockSnapshot('cell-1');

      const features04 = gnn.buildNodeFeatures(snapshot, -96, 0.4);
      const features10 = gnn.buildNodeFeatures(snapshot, -96, 1.0);

      expect(features04[1]).toBe(0.4);
      expect(features10[1]).toBe(1.0);
    });
  });

  describe('predictSINR', () => {
    it('should predict SINR for all nodes', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      const cellIds = ['cell-1', 'cell-2', 'cell-3'];

      cellIds.forEach(id => {
        snapshots.set(id, createMockSnapshot(id));
      });

      const nodeFeatures = cellIds.map(id => {
        const snap = snapshots.get(id)!;
        return gnn.buildNodeFeatures(snap, snap.uplinkPowerControl.p0NominalPusch, snap.uplinkPowerControl.alpha);
      });

      const adjacencyMatrix = [
        [1, 0.8, 0.6],
        [0.8, 1, 0.7],
        [0.6, 0.7, 1],
      ];

      const sinrs = gnn.predictSINR(nodeFeatures, adjacencyMatrix);

      expect(sinrs).toHaveLength(3);
      // Note: With randomly initialized weights, predictions may be outside typical SINR range
      // After training, predictions should be within [-5, 30] dB range
      sinrs.forEach(sinr => {
        expect(typeof sinr).toBe('number');
        expect(isFinite(sinr)).toBe(true);
      });
    });

    it('should produce different predictions for different inputs', () => {
      const snapshot1 = createMockSnapshot('cell-1', { sinr: 5, iot: 12 });
      const snapshot2 = createMockSnapshot('cell-2', { sinr: 20, iot: 3 });

      const features1 = [gnn.buildNodeFeatures(snapshot1, -96, 0.8)];
      const features2 = [gnn.buildNodeFeatures(snapshot2, -96, 0.8)];

      const adjacency = [[1]];

      const sinr1 = gnn.predictSINR(features1, adjacency);
      const sinr2 = gnn.predictSINR(features2, adjacency);

      // Different inputs should produce different outputs
      expect(sinr1[0]).not.toBe(sinr2[0]);
    });
  });

  describe('predictCellSINR', () => {
    it('should predict SINR for a specific cell', () => {
      const snapshot = createMockSnapshot('cell-1');
      const features = [gnn.buildNodeFeatures(snapshot, -96, 0.8)];
      const adjacency = [[1]];

      const sinr = gnn.predictCellSINR(0, features, adjacency);

      expect(typeof sinr).toBe('number');
      expect(isFinite(sinr)).toBe(true);
      // Note: After training, should be within [-5, 30] dB range
    });
  });

  describe('train', () => {
    it('should train on sample data', () => {
      const trainingData = [
        {
          nodeFeatures: [[0.5, 0.8, 0.6, 0.5, 0.5, 0.8, 0.3, 0.5, 0.1, 0.6, 0.1, 0.5, 0.1, 0.95, 0.5, 0.95, 0.3, 0.05, 0.5, 0.5]],
          adjacencyMatrix: [[1]],
          actualSINR: [15],
        },
      ];

      const result = gnn.train(trainingData, 0.001, 10);

      expect(result).toHaveProperty('loss');
      expect(result).toHaveProperty('accuracy');
      expect(result.accuracy).toBeGreaterThanOrEqual(0);
      expect(result.accuracy).toBeLessThanOrEqual(1);
    });
  });
});

// ============================================================================
// ISSUE CELL DETECTOR TESTS
// ============================================================================

describe('IssueCellDetector', () => {
  let detector: IssueCellDetector;

  beforeEach(() => {
    detector = new IssueCellDetector();
  });

  describe('detectIssueCells', () => {
    it('should detect cells with low SINR', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 2, iot: 5 })); // Low SINR issue
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 15, iot: 5 })); // Healthy

      const relations = createMockNeighborRelations(['cell-1', 'cell-2']);
      const issues = detector.detectIssueCells(snapshots, relations);

      expect(issues.length).toBe(1);
      expect(issues[0].cellId).toBe('cell-1');
      expect(issues[0].issues.some(i => i.includes('Low SINR'))).toBe(true);
    });

    it('should detect cells with high IoT', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 15, iot: 15 })); // High IoT
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 15, iot: 5 })); // Healthy

      const relations = createMockNeighborRelations(['cell-1', 'cell-2']);
      const issues = detector.detectIssueCells(snapshots, relations);

      expect(issues.length).toBe(1);
      expect(issues[0].cellId).toBe('cell-1');
      expect(issues[0].issues.some(i => i.includes('High IoT'))).toBe(true);
    });

    it('should assign correct severity levels', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-critical', createMockSnapshot('cell-critical', { sinr: -2, iot: 18 })); // Critical
      snapshots.set('cell-high', createMockSnapshot('cell-high', { sinr: 2, iot: 13 })); // High
      snapshots.set('cell-medium', createMockSnapshot('cell-medium', { sinr: 4, iot: 8 })); // Medium

      const relations = createMockNeighborRelations(['cell-critical', 'cell-high', 'cell-medium']);
      const issues = detector.detectIssueCells(snapshots, relations);

      // Should be sorted by severity
      expect(issues[0].severity).toBe('critical');
      expect(issues[1].severity).toBe('high');
    });

    it('should detect power-limited UE issues', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 6, iot: 5, powerLimited: 25 }));

      const relations = createMockNeighborRelations(['cell-1']);
      const issues = detector.detectIssueCells(snapshots, relations);

      expect(issues.length).toBe(1);
      expect(issues[0].issues.some(i => i.includes('power-limited'))).toBe(true);
    });

    it('should identify neighbor cells for issue cells', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 2 }));
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 15 }));
      snapshots.set('cell-3', createMockSnapshot('cell-3', { sinr: 15 }));

      const relations = createMockNeighborRelations(['cell-1', 'cell-2', 'cell-3']);
      const issues = detector.detectIssueCells(snapshots, relations);

      expect(issues[0].neighborCellIds).toContain('cell-2');
      expect(issues[0].neighborCellIds).toContain('cell-3');
    });

    it('should return empty array for healthy network', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 15, iot: 5 }));
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 18, iot: 4 }));

      const relations = createMockNeighborRelations(['cell-1', 'cell-2']);
      const issues = detector.detectIssueCells(snapshots, relations);

      expect(issues.length).toBe(0);
    });
  });
});

// ============================================================================
// GENETIC OPTIMIZER TESTS
// ============================================================================

describe('GeneticOptimizer', () => {
  let gnn: SINRPredictionGNN;
  let optimizer: GeneticOptimizer;

  beforeEach(() => {
    gnn = new SINRPredictionGNN();
    optimizer = new GeneticOptimizer(gnn, {
      populationSize: 20,  // Smaller for faster tests
      numGenerations: 10,
    });
  });

  describe('optimize', () => {
    it('should return optimization result for issue cell', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 2, iot: 12 }));
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 15, iot: 5 }));

      const relations = createMockNeighborRelations(['cell-1', 'cell-2']);

      const cellIndexMap = new Map<string, number>();
      cellIndexMap.set('cell-1', 0);
      cellIndexMap.set('cell-2', 1);

      const issueCell: IssueCell = {
        cellId: 'cell-1',
        cellIndex: 0,
        currentSINR: 2,
        currentIoT: 12,
        currentP0: -96,
        currentAlpha: 0.8,
        severity: 'high',
        issues: ['Low SINR: 2 dB', 'High IoT: 12 dB'],
        neighborCellIds: ['cell-2'],
      };

      const nodeFeatures = Array.from(snapshots.entries()).map(([id, snap]) =>
        gnn.buildNodeFeatures(snap, snap.uplinkPowerControl.p0NominalPusch, snap.uplinkPowerControl.alpha)
      );

      const adjacencyMatrix = [
        [1, 0.8],
        [0.8, 1],
      ];

      const result = optimizer.optimize(
        issueCell,
        nodeFeatures,
        adjacencyMatrix,
        snapshots,
        cellIndexMap
      );

      expect(result).toHaveProperty('cellId', 'cell-1');
      expect(result).toHaveProperty('currentP0', -96);
      expect(result).toHaveProperty('currentAlpha', 0.8);
      expect(result).toHaveProperty('optimizedP0');
      expect(result).toHaveProperty('optimizedAlpha');
      expect(result).toHaveProperty('predictedSINRImprovement');
      expect(result).toHaveProperty('predictedNeighborImpact');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('rationale');
    });

    it('should produce valid P0 values within range', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 1, iot: 15 }));

      const issueCell: IssueCell = {
        cellId: 'cell-1',
        cellIndex: 0,
        currentSINR: 1,
        currentIoT: 15,
        currentP0: -96,
        currentAlpha: 0.8,
        severity: 'critical',
        issues: ['Low SINR', 'High IoT'],
        neighborCellIds: [],
      };

      const nodeFeatures = [gnn.buildNodeFeatures(snapshots.get('cell-1')!, -96, 0.8)];
      const adjacencyMatrix = [[1]];

      const result = optimizer.optimize(
        issueCell,
        nodeFeatures,
        adjacencyMatrix,
        snapshots,
        new Map([['cell-1', 0]])
      );

      expect(result.optimizedP0).toBeGreaterThanOrEqual(-110);
      expect(result.optimizedP0).toBeLessThanOrEqual(-85);
    });

    it('should produce valid alpha values', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 1, iot: 15 }));

      const issueCell: IssueCell = {
        cellId: 'cell-1',
        cellIndex: 0,
        currentSINR: 1,
        currentIoT: 15,
        currentP0: -96,
        currentAlpha: 0.8,
        severity: 'critical',
        issues: ['Low SINR'],
        neighborCellIds: [],
      };

      const nodeFeatures = [gnn.buildNodeFeatures(snapshots.get('cell-1')!, -96, 0.8)];
      const adjacencyMatrix = [[1]];

      const result = optimizer.optimize(
        issueCell,
        nodeFeatures,
        adjacencyMatrix,
        snapshots,
        new Map([['cell-1', 0]])
      );

      const validAlphaValues = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
      expect(validAlphaValues).toContain(result.optimizedAlpha);
    });

    it('should include meaningful rationale', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 2, iot: 14 }));

      const issueCell: IssueCell = {
        cellId: 'cell-1',
        cellIndex: 0,
        currentSINR: 2,
        currentIoT: 14,
        currentP0: -96,
        currentAlpha: 0.8,
        severity: 'high',
        issues: ['Low SINR: 2 dB', 'High IoT: 14 dB'],
        neighborCellIds: [],
      };

      const nodeFeatures = [gnn.buildNodeFeatures(snapshots.get('cell-1')!, -96, 0.8)];
      const adjacencyMatrix = [[1]];

      const result = optimizer.optimize(
        issueCell,
        nodeFeatures,
        adjacencyMatrix,
        snapshots,
        new Map([['cell-1', 0]])
      );

      expect(result.rationale.length).toBeGreaterThan(0);
      expect(result.rationale).toContain('cell-1');
    });
  });
});

// ============================================================================
// INTERFERENCE OPTIMIZATION LOOP TESTS
// ============================================================================

describe('InterferenceOptimizationLoop', () => {
  let loop: InterferenceOptimizationLoop;

  beforeEach(() => {
    loop = new InterferenceOptimizationLoop({
      populationSize: 15,
      numGenerations: 5,
    });
  });

  describe('constructor', () => {
    it('should initialize with default configuration', () => {
      expect(loop).toBeDefined();
    });

    it('should provide access to underlying GNN', () => {
      const gnn = loop.getGNN();
      expect(gnn).toBeInstanceOf(SINRPredictionGNN);
    });
  });

  describe('optimize', () => {
    it('should run full optimization workflow', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 2, iot: 12 }));
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 15, iot: 5 }));
      snapshots.set('cell-3', createMockSnapshot('cell-3', { sinr: 18, iot: 4 }));

      const relations = createMockNeighborRelations(['cell-1', 'cell-2', 'cell-3']);

      const result = loop.optimize(snapshots, relations);

      expect(result).toHaveProperty('timestamp');
      expect(result).toHaveProperty('totalCellsAnalyzed', 3);
      expect(result).toHaveProperty('issueCellsDetected');
      expect(result).toHaveProperty('cellsOptimized');
      expect(result).toHaveProperty('optimizationResults');
      expect(result).toHaveProperty('aggregateMetrics');
      expect(result).toHaveProperty('deploymentRecommendations');
    });

    it('should detect issue cells correctly', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('issue-cell', createMockSnapshot('issue-cell', { sinr: 1, iot: 16 }));
      snapshots.set('healthy-cell', createMockSnapshot('healthy-cell', { sinr: 20, iot: 3 }));

      const relations = createMockNeighborRelations(['issue-cell', 'healthy-cell']);

      const result = loop.optimize(snapshots, relations);

      expect(result.issueCellsDetected).toBe(1);
    });

    it('should return results for healthy network', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 18, iot: 4 }));
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 20, iot: 3 }));

      const relations = createMockNeighborRelations(['cell-1', 'cell-2']);

      const result = loop.optimize(snapshots, relations);

      expect(result.issueCellsDetected).toBe(0);
      expect(result.cellsOptimized).toBe(0);
      expect(result.deploymentRecommendations).toBeDefined();
    });

    it('should calculate aggregate metrics', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 2, iot: 13 }));
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 3, iot: 12 }));
      snapshots.set('cell-3', createMockSnapshot('cell-3', { sinr: 15, iot: 5 }));

      const relations = createMockNeighborRelations(['cell-1', 'cell-2', 'cell-3']);

      const result = loop.optimize(snapshots, relations);

      expect(result.aggregateMetrics).toHaveProperty('avgSINRImprovement');
      expect(result.aggregateMetrics).toHaveProperty('avgNeighborImpact');
      expect(result.aggregateMetrics).toHaveProperty('successRate');
    });

    it('should generate deployment recommendations', () => {
      const snapshots = new Map<string, CellKPISnapshot>();
      snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 1, iot: 15 }));
      snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 15, iot: 5 }));

      const relations = createMockNeighborRelations(['cell-1', 'cell-2']);

      const result = loop.optimize(snapshots, relations);

      expect(Array.isArray(result.deploymentRecommendations)).toBe(true);
      expect(result.deploymentRecommendations.length).toBeGreaterThan(0);
    });
  });
});

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

describe('Integration: Full Optimization Pipeline', () => {
  it('should handle a realistic multi-cell network scenario', () => {
    // Create a small network with varying health
    const snapshots = new Map<string, CellKPISnapshot>();
    const cellIds = ['cell-A', 'cell-B', 'cell-C', 'cell-D', 'cell-E'];

    // Cell A: Critical issue (very low SINR)
    snapshots.set('cell-A', createMockSnapshot('cell-A', { sinr: -1, iot: 16, p0: -100, alpha: 0.6 }));

    // Cell B: High issue (low SINR, high IoT)
    snapshots.set('cell-B', createMockSnapshot('cell-B', { sinr: 3, iot: 13, p0: -98, alpha: 0.7 }));

    // Cell C: Medium issue (borderline SINR)
    snapshots.set('cell-C', createMockSnapshot('cell-C', { sinr: 5, iot: 9, p0: -96, alpha: 0.8 }));

    // Cell D: Healthy
    snapshots.set('cell-D', createMockSnapshot('cell-D', { sinr: 15, iot: 5, p0: -96, alpha: 0.8 }));

    // Cell E: Very healthy
    snapshots.set('cell-E', createMockSnapshot('cell-E', { sinr: 22, iot: 3, p0: -94, alpha: 0.9 }));

    const relations = createMockNeighborRelations(cellIds);

    const loop = new InterferenceOptimizationLoop({
      populationSize: 30,
      numGenerations: 15,
    });

    const result = loop.optimize(snapshots, relations);

    // Verify results structure
    expect(result.totalCellsAnalyzed).toBe(5);
    expect(result.issueCellsDetected).toBeGreaterThanOrEqual(2); // At least cells A and B

    // Critical cells should be prioritized
    if (result.optimizationResults.length > 0) {
      const cellAResult = result.optimizationResults.find(r => r.cellId === 'cell-A');
      if (cellAResult) {
        // Optimization should suggest changes for critical cell
        expect(cellAResult.optimizedP0).toBeDefined();
        expect(cellAResult.optimizedAlpha).toBeDefined();
      }
    }

    // Aggregate metrics should be reasonable
    expect(result.aggregateMetrics.successRate).toBeGreaterThanOrEqual(0);
    expect(result.aggregateMetrics.successRate).toBeLessThanOrEqual(1);
  });

  it('should respect interference constraints for neighbors', () => {
    // Create a scenario where optimizing one cell could harm neighbors
    const snapshots = new Map<string, CellKPISnapshot>();

    // Issue cell that needs higher power
    snapshots.set('issue', createMockSnapshot('issue', { sinr: 1, iot: 8, p0: -100, alpha: 0.6 }));

    // Healthy neighbors that could be affected
    snapshots.set('neighbor-1', createMockSnapshot('neighbor-1', { sinr: 12, iot: 6, p0: -96, alpha: 0.8 }));
    snapshots.set('neighbor-2', createMockSnapshot('neighbor-2', { sinr: 14, iot: 5, p0: -96, alpha: 0.8 }));

    const relations = createMockNeighborRelations(['issue', 'neighbor-1', 'neighbor-2']);

    const loop = new InterferenceOptimizationLoop({
      populationSize: 30,
      numGenerations: 20,
      maxInterferenceIncrease: 2,  // Strict constraint
    });

    const result = loop.optimize(snapshots, relations);

    // Check that neighbor impact is within tolerance
    for (const optResult of result.optimizationResults) {
      // Negative impact means neighbors got worse
      // Should be within configured tolerance
      expect(optResult.predictedNeighborImpact).toBeGreaterThanOrEqual(-3);
    }
  });
});

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

describe('Edge Cases', () => {
  it('should handle single cell network', () => {
    const snapshots = new Map<string, CellKPISnapshot>();
    snapshots.set('only-cell', createMockSnapshot('only-cell', { sinr: 2, iot: 12 }));

    const loop = new InterferenceOptimizationLoop({
      populationSize: 20,
      numGenerations: 10,
    });

    const result = loop.optimize(snapshots, []);

    expect(result.totalCellsAnalyzed).toBe(1);
    expect(result.issueCellsDetected).toBe(1);
  });

  it('should handle network with no neighbor relations', () => {
    const snapshots = new Map<string, CellKPISnapshot>();
    snapshots.set('cell-1', createMockSnapshot('cell-1', { sinr: 2, iot: 12 }));
    snapshots.set('cell-2', createMockSnapshot('cell-2', { sinr: 3, iot: 11 }));

    const loop = new InterferenceOptimizationLoop({
      populationSize: 20,
      numGenerations: 10,
    });

    const result = loop.optimize(snapshots, []);

    expect(result.totalCellsAnalyzed).toBe(2);
    // Should still detect and attempt to optimize issue cells
    expect(result.issueCellsDetected).toBe(2);
  });

  it('should handle cells with extreme values', () => {
    const snapshots = new Map<string, CellKPISnapshot>();
    snapshots.set('extreme', createMockSnapshot('extreme', {
      sinr: -5,  // Minimum SINR
      iot: 20,   // Very high interference
      p0: -110,  // Min P0
      alpha: 0.4, // Low alpha
    }));

    const loop = new InterferenceOptimizationLoop({
      populationSize: 30,
      numGenerations: 15,
    });

    const result = loop.optimize(snapshots, []);

    expect(result.issueCellsDetected).toBe(1);
    if (result.optimizationResults.length > 0) {
      const optResult = result.optimizationResults[0];
      // Should suggest valid parameters even for extreme case
      expect(optResult.optimizedP0).toBeGreaterThanOrEqual(-110);
      expect(optResult.optimizedP0).toBeLessThanOrEqual(-85);
    }
  });
});
