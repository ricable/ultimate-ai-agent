
import { describe, test, expect, beforeAll } from 'vitest';
import {
  RuvectorGNN,
  RuvLLMClient,
  GraphAttentionNetwork,
  AgentDBReflexion,
  cmParametersValidator,
  recommendationValidator,
  SelfLearningRANSystem
} from '../src/ml/index.js';

describe('ML Module Integration Tests', () => {
  let gnn: RuvectorGNN;

  beforeAll(async () => {
    // Setup shared resources if needed
  });

  test('RuvVector GNN: initialization and cell operations', async () => {
    gnn = new RuvectorGNN('./test-ruvector.db');
    await gnn.initialize();

    // Add cells
    await gnn.addCell('CELL_001', {
      pmUlSinrMean: 10.5,
      pmDlSinrMean: 12.3,
      pmCssr: 0.97,
      pmCallDropRate: 0.01
    }, {
      cluster: 'cluster_A',
      site: 'site_1',
      sector: 0,
      lastOptimization: new Date(),
      performanceClass: 'good'
    });

    await gnn.addCell('CELL_002', {
      pmUlSinrMean: 10.2,
      pmDlSinrMean: 12.1,
      pmCssr: 0.96,
      pmCallDropRate: 0.012
    }, {
      cluster: 'cluster_A',
      site: 'site_1',
      sector: 1,
      lastOptimization: new Date(),
      performanceClass: 'good'
    });

    // Find similar cells
    const similar = await gnn.findSimilarCells('CELL_001', 1);

    expect(similar.length).toBeGreaterThan(0);
    expect(similar[0].cellId).toBe('CELL_002');

    const stats = gnn.getStats();
    expect(stats.cellCount).toBe(2);
  });

  test('RuvLLM Client: query and recommendations', async () => {
    // Assuming gnn is initialized from previous test or we create new one
    // Ideally tests should be isolated, so let's reuse or create new if needed
    // For this rewrite, I'll create new to be safe and isolated
    const localGnn = new RuvectorGNN('./test-ruvector-llm.db');
    await localGnn.initialize();
    await localGnn.addCell('CELL_003', {
      pmUlSinrMean: 8.0,
      pmCssr: 0.95
    }, {
      cluster: 'cluster_B',
      site: 'site_2',
      sector: 0,
      lastOptimization: new Date(),
      performanceClass: 'fair'
    });

    const ruvllm = new RuvLLMClient(localGnn);

    const insight = await ruvllm.queryRAN('What cells have similar SINR patterns to CELL_003?');

    expect(insight.query).toContain('CELL_003');
    expect(insight.answer.length).toBeGreaterThan(0);
    expect(insight.confidence).toBeGreaterThanOrEqual(0);
    expect(insight.confidence).toBeLessThanOrEqual(1);

    // Mock recommendation if needed, or rely on internal logic
    const recommendation = await ruvllm.recommendOptimization('CELL_003');
    expect(recommendation.cellId).toBe('CELL_003');
    expect(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']).toContain(recommendation.priority);
  });

  test('Graph Attention Network: attention computation', async () => {
    const gat = new GraphAttentionNetwork({
      numHeads: 4,
      hiddenDim: 32,
      nodeFeatureDim: 64,
      edgeFeatureDim: 16
    });

    // Add cells
    const cells = [
      {
        id: 'CELL_A',
        features: new Float32Array(64),
        pm: { pmUlSinrMean: 10, pmCssr: 0.96 },
        cm: { p0NominalPUSCH: -103, alpha: 0.8 },
        metadata: {
          sector: 0,
          azimuth: 0,
          beamwidth: 65,
          height: 30,
          latitude: 59.3293,
          longitude: 18.0686
        }
      },
      {
        id: 'CELL_B',
        features: new Float32Array(64),
        pm: { pmUlSinrMean: 9, pmCssr: 0.95 },
        cm: { p0NominalPUSCH: -103, alpha: 0.8 },
        metadata: {
          sector: 1,
          azimuth: 120,
          beamwidth: 65,
          height: 30,
          latitude: 59.3300,
          longitude: 18.0700
        }
      }
    ];

    for (const cell of cells) {
      gat.addCell(cell);
    }

    // Add interference edge
    gat.addInterferenceEdge({
      source: 'CELL_A',
      target: 'CELL_B',
      features: new Float32Array(16),
      metadata: {
        rsrp: -85,
        pathLoss: 95,
        couplingLoss: 100,
        distance: 500,
        azimuthDiff: 120,
        interferenceRank: 1
      }
    });

    // Compute attention
    const attention = gat.computeAttention('CELL_A');

    expect(attention.cellId).toBe('CELL_A');
    expect(attention.topNeighbors.length).toBeGreaterThan(0);

    const stats = gat.getStats();
    expect(stats.nodeCount).toBe(2);
    expect(stats.edgeCount).toBeGreaterThanOrEqual(2);
  });

  test('AgentDB Reflexion Memory', async () => {
    const agentdb = new AgentDBReflexion({
      dbPath: './test-titan.db',
      maxMemorySize: 1000
    });

    await agentdb.initialize();

    // Store optimization episode
    const episode = {
      id: 'ep_001',
      cellId: 'CELL_X',
      timestamp: new Date(),
      pmBefore: { pmUlSinrMean: 8, pmCssr: 0.95 },
      cmBefore: { p0NominalPUSCH: -103 },
      fmAlarmsBefore: [],
      action: { electricalTilt: 5.0 },
      actionType: 'tilt' as const,
      pmAfter: { pmUlSinrMean: 10, pmCssr: 0.97 },
      cmAfter: { p0NominalPUSCH: -103, electricalTilt: 5.0 },
      fmAlarmsAfter: [],
      reward: 0.65,
      sinrGain: 2.0,
      cssrGain: 0.02,
      dropRateChange: -0.005,
      outcome: 'SUCCESS' as const,
      stateEmbedding: new Float32Array(768).fill(0.1),
      actionEmbedding: new Float32Array(768).fill(0.2)
    };

    await agentdb.storeOptimization(episode);

    // Query for transfer learning
    const results = await agentdb.queryForTransferLearning(
      { pmUlSinrMean: 8.5, pmCssr: 0.94 },
      new Float32Array(768).fill(0.1),
      1,
      { outcome: 'success' }
    );

    expect(results.length).toBeGreaterThan(0);

    // Get statistics
    const stats = await agentdb.getReflexionStats();
    expect(stats.totalEpisodes).toBe(1);
    expect(stats.successfulEpisodes).toBe(1);
  });

  test('PydanticAI Validation', async () => {
    // Valid CM parameters
    const validCM = {
      p0NominalPUSCH: -103,
      alpha: 0.8,
      electricalTilt: 5.0,
      txPower: 40
    };

    const result1 = cmParametersValidator.validate(validCM, { coerce: true });
    expect(result1.valid).toBe(true);
    expect(result1.data).toBeDefined();

    // Invalid CM parameters
    const invalidCM = {
      p0NominalPUSCH: -250,  // Out of range
      alpha: 0.75,           // Invalid value
      electricalTilt: 20     // Out of range
    };

    const result2 = cmParametersValidator.validate(invalidCM);
    expect(result2.valid).toBe(false);
    expect(result2.errors.length).toBeGreaterThan(0);

    // Coercion test
    const coerceCM = {
      alpha: 0.75  // Should coerce to 0.7
    };

    const result3 = cmParametersValidator.validate(coerceCM, { coerce: true });
    expect(result3.valid).toBe(true);
    expect(result3.data?.alpha).toBe(0.7);
  });

  test('Complete System Integration', async () => {
    const system = new SelfLearningRANSystem();
    await system.initialize();

    const stats = await system.getStatistics();

    expect(stats.ruvectorGNN).toBeDefined();
    expect(stats.graphAttentionNetwork).toBeDefined();
    expect(stats.reflexionMemory).toBeDefined();

    const insight = await system.query('Show me system statistics');
    expect(insight.answer).toBeDefined();
  });
});
