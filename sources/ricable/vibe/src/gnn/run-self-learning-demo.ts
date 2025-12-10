#!/usr/bin/env tsx
/**
 * Self-Learning Uplink GNN Demo Runner
 *
 * Demonstrates the Self-Learning GNN for Ericsson Uplink Optimization
 */

import {
  SelfLearningUplinkGNN,
  EricssonUplinkOptimizer,
  RuVectorGNNLayer,
  ExperienceReplayBuffer,
  DifferentiableParameterSearch,
  runRuVectorGNNLayer,
  compressEmbeddings,
  differentiableSearch,
} from './self-learning-uplink-gnn.js';

import type { CellKPISnapshot, NeighborRelation } from '../models/ran-kpi.js';

// ============================================================================
// SAMPLE DATA GENERATOR
// ============================================================================

function generateSampleCellSnapshot(
  cellId: string,
  overrides: Partial<{ sinr: number; iot: number; p0: number; alpha: number }> = {}
): CellKPISnapshot {
  const sinr = overrides.sinr ?? 5 + Math.random() * 15;
  const iot = overrides.iot ?? 3 + Math.random() * 8;
  const p0 = overrides.p0 ?? -100 + Math.floor(Math.random() * 15);
  const alpha = overrides.alpha ?? [0.6, 0.7, 0.8, 0.9, 1.0][Math.floor(Math.random() * 5)];

  return {
    timestamp: new Date(),
    cell: {
      cellId,
      enodebId: `eNB_${cellId.split('_')[1] ?? '001'}`,
      sectorId: parseInt(cellId.split('_')[2] ?? '0', 10) % 3,
      frequency: 2600,
      band: 'B7',
      technology: 'LTE',
      pci: Math.floor(Math.random() * 504),
      tac: 12345,
    },
    accessibility: {
      rrcSetupAttempts: 1000 + Math.floor(Math.random() * 5000),
      rrcSetupSuccess: 950 + Math.floor(Math.random() * 50),
      rrcSetupFailure: Math.floor(Math.random() * 50),
      rrcSetupSuccessRate: 95 + Math.random() * 5,
      erabSetupAttempts: 800 + Math.floor(Math.random() * 4000),
      erabSetupSuccess: 780 + Math.floor(Math.random() * 200),
      erabSetupFailure: Math.floor(Math.random() * 20),
      erabSetupSuccessRate: 97 + Math.random() * 3,
      s1SigConnEstabAttempts: 1000,
      s1SigConnEstabSuccess: 990,
      s1SigConnEstabSuccessRate: 99,
      initialContextSetupAttempts: 900,
      initialContextSetupSuccess: 880,
      initialContextSetupSuccessRate: 97.8,
    },
    retainability: {
      erabNormalRelease: 5000,
      erabAbnormalRelease: 50,
      erabDropRate: 1,
      voiceCallAttempts: 1000,
      voiceCallDrops: 10,
      voiceCallDropRate: 1,
      dataSessionAttempts: 8000,
      dataSessionDrops: 80,
      dataSessionRetainability: 99,
    },
    radioQuality: {
      dlAvgCqi: 10 + Math.random() * 4,
      dlRi1Ratio: 40 + Math.random() * 20,
      dlRi2Ratio: 40 + Math.random() * 20,
      dlBlerPercent: 2 + Math.random() * 3,
      ulSinrAvg: sinr,
      ulSinrP10: sinr - 5,
      ulSinrP50: sinr,
      ulSinrP90: sinr + 5,
      ulBlerPercent: 3 + Math.random() * 4,
      rsrpAvg: -95 + Math.random() * 15,
      rsrpP10: -105,
      rsrpP50: -95,
      rsrpP90: -85,
      rsrqAvg: -10 + Math.random() * 5,
      rsrqP10: -15,
      rsrqP50: -10,
      rsrqP90: -5,
      dlSpectralEfficiency: 3 + Math.random() * 2,
      ulSpectralEfficiency: 1.5 + Math.random() * 1,
    },
    mobility: {
      intraFreqHoAttempts: 500 + Math.floor(Math.random() * 2000),
      intraFreqHoSuccess: 480 + Math.floor(Math.random() * 20),
      intraFreqHoFailure: Math.floor(Math.random() * 20),
      intraFreqHoSuccessRate: 96 + Math.random() * 4,
      interFreqHoAttempts: 100,
      interFreqHoSuccess: 95,
      interFreqHoFailure: 5,
      interFreqHoSuccessRate: 95,
      interRatHoAttempts: 50,
      interRatHoSuccess: 45,
      interRatHoFailure: 5,
      interRatHoSuccessRate: 90,
      x2HoAttempts: 400,
      x2HoSuccess: 390,
      x2HoSuccessRate: 97.5,
      s1HoAttempts: 100,
      s1HoSuccess: 95,
      s1HoSuccessRate: 95,
      tooEarlyHo: 5,
      tooLateHo: 3,
      wrongCellHo: 2,
      pingPongHo: 8,
      incomingHoTotal: 600,
      outgoingHoTotal: 580,
    },
    uplinkInterference: {
      prbUlInterferenceAvg: -105 + iot,
      prbUlInterferenceP10: -110 + iot,
      prbUlInterferenceP50: -105 + iot,
      prbUlInterferenceP90: -100 + iot,
      prbUlInterferenceP99: -95 + iot,
      iotAvg: iot,
      iotP95: iot + 2,
      rip: -100 + iot,
      externalInterferenceDetected: iot > 10,
      externalInterferenceLevel: iot > 15 ? 'high' : iot > 10 ? 'medium' : iot > 6 ? 'low' : 'none',
      puschSinrDegradation: Math.max(0, 8 - sinr),
      highInterferencePrbRatio: Math.min(100, iot * 5),
    },
    uplinkPowerControl: {
      p0NominalPusch: p0,
      p0NominalPucch: p0 - 5,
      alpha,
      ueTxPowerAvg: 10 + Math.random() * 10,
      ueTxPowerP10: 5,
      ueTxPowerP50: 15,
      ueTxPowerP90: 20,
      ueTxPowerMax: 23,
      powerHeadroomAvg: 10 + Math.random() * 10,
      powerHeadroomP10: 5,
      powerHeadroomP50: 12,
      powerHeadroomP90: 18,
      negativePowerHeadroomRatio: Math.max(0, 15 - sinr),
      pathLossAvg: 110 + Math.random() * 20,
      pathLossP10: 100,
      pathLossP50: 115,
      pathLossP90: 130,
      tpcUpCommands: 1000,
      tpcDownCommands: 800,
      tpcAccumulatedOffset: 2,
      powerLimitedUeRatio: Math.max(0, 25 - sinr),
    },
  };
}

function generateSampleNeighborRelation(
  sourceCellId: string,
  targetCellId: string
): NeighborRelation {
  return {
    sourceCellId,
    targetCellId,
    relationshipType: 'intra-freq',
    sourcePci: Math.floor(Math.random() * 504),
    sourceFrequency: 2600,
    sourceRsrp: -90 + Math.random() * 10,
    sourceSinr: 8 + Math.random() * 10,
    targetPci: Math.floor(Math.random() * 504),
    targetFrequency: 2600,
    targetRsrp: -95 + Math.random() * 15,
    targetSinr: 5 + Math.random() * 12,
    hoAttempts: 100 + Math.floor(Math.random() * 400),
    hoSuccess: 95 + Math.floor(Math.random() * 5),
    hoFailure: Math.floor(Math.random() * 5),
    hoSuccessRate: 95 + Math.random() * 5,
    a3Offset: 3,
    hysteresis: 2,
    timeToTrigger: 480,
    neighborQuality: 'good',
    distance: 500 + Math.random() * 2000,
  };
}

// ============================================================================
// DEMO FUNCTIONS
// ============================================================================

async function demoRuVectorGNNLayer() {
  console.log('\n' + '='.repeat(70));
  console.log('Demo 1: RuVector GNN Layer');
  console.log('='.repeat(70));

  const layer = await runRuVectorGNNLayer(24, 64, 4, 0.1);
  console.log('✓ Created GNN layer with multi-head attention');

  // Create sample node features (5 nodes, 24 features each)
  const nodeFeatures = Array(5).fill(null).map(() =>
    Array(24).fill(0).map(() => Math.random())
  );

  // Create sample adjacency matrix
  const adjacencyMatrix = [
    [1, 0.8, 0.3, 0, 0],
    [0.8, 1, 0.5, 0.2, 0],
    [0.3, 0.5, 1, 0.7, 0.4],
    [0, 0.2, 0.7, 1, 0.6],
    [0, 0, 0.4, 0.6, 1],
  ];

  console.log('✓ Generated sample node features (5 nodes × 24 features)');

  // Forward pass
  const output = layer.forward(nodeFeatures, adjacencyMatrix);
  console.log(`✓ Forward pass completed: output shape [${output.length}][${output[0].length}]`);

  // Check compression
  const config = layer.getConfig();
  console.log(`✓ Layer config: inputDim=${config.inputDim}, hiddenDim=${config.hiddenDim}, heads=${config.numHeads}`);
}

function demoExperienceReplay() {
  console.log('\n' + '='.repeat(70));
  console.log('Demo 2: Experience Replay Buffer');
  console.log('='.repeat(70));

  const buffer = new ExperienceReplayBuffer(1000, 0.6, 0.4, 0.001);

  // Generate synthetic graph for samples
  const cellSnapshots = new Map<string, CellKPISnapshot>();
  cellSnapshots.set('CELL_001', generateSampleCellSnapshot('CELL_001'));

  // Add 100 experience samples
  for (let i = 0; i < 100; i++) {
    buffer.add({
      timestamp: new Date(),
      graph: {
        nodeIds: ['CELL_001'],
        nodeFeatures: [[...Array(24).fill(0).map(() => Math.random())]],
        adjacencyMatrix: [[1]],
        edgeFeatures: [[[]]],
        powerParams: new Map([['CELL_001', { p0: -100, alpha: 0.8 }]]),
      },
      cellId: 'CELL_001',
      params: { p0: -100 + Math.floor(Math.random() * 15), alpha: 0.8 },
      predictedSINR: 5 + Math.random() * 10,
      actualSINR: 5 + Math.random() * 10,
      reward: Math.random() > 0.5 ? 1 : 0.5,
    });
  }

  console.log(`✓ Added 100 experience samples to buffer`);

  const stats = buffer.getStats();
  console.log(`✓ Buffer stats: size=${stats.size}, avgPriority=${stats.avgPriority.toFixed(3)}, avgReward=${stats.avgReward.toFixed(3)}`);

  // Sample a batch
  const { samples, weights } = buffer.sample(16);
  console.log(`✓ Sampled batch of ${samples.length} with importance weights`);
}

function demoDifferentiableSearch() {
  console.log('\n' + '='.repeat(70));
  console.log('Demo 3: Differentiable Parameter Search');
  console.log('='.repeat(70));

  const search = new DifferentiableParameterSearch(1.0);

  // Generate candidates
  const currentParams = { p0: -100, alpha: 0.8 };
  const candidates = search.generateCandidates(currentParams, {
    inputDim: 24,
    hiddenDim: 64,
    numHeads: 4,
    p0Range: { min: -110, max: -85, step: 1 },
    alphaValues: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    thresholds: { sinrLow: 5, sinrCritical: 0, iotHigh: 10, powerLimitedHigh: 20 },
    optimization: { maxIterations: 100, convergenceThreshold: 0.01, neighborImpactWeight: 0.3, minImprovement: 0.5 },
    training: { learningRate: 0.001, batchSize: 32, epochs: 100 },
  });

  console.log(`✓ Generated ${candidates.length} candidate parameter combinations`);

  // Embed candidates with a simple predictor
  search.embedCandidates(candidates, (params) => {
    // Simple predictor: lower P0 = higher SINR (simplified model)
    return 10 + (params.p0 + 110) * 0.5 + (1 - params.alpha) * 3;
  });

  console.log('✓ Embedded candidates with SINR predictions');

  // Soft search
  const queryEmbedding = [0.4, 0.8, 0.5]; // [normalized P0, alpha, target SINR]
  const softResult = search.softSearch(queryEmbedding);
  console.log(`✓ Soft search result: P0=${softResult.params.p0}, Alpha=${softResult.params.alpha}, confidence=${softResult.confidence.toFixed(3)}`);

  // Hard search
  const hardResult = search.hardSearch(queryEmbedding);
  console.log(`✓ Hard search result: P0=${hardResult.params.p0}, Alpha=${hardResult.params.alpha}, score=${hardResult.score.toFixed(3)}`);
}

async function demoSelfLearningGNN() {
  console.log('\n' + '='.repeat(70));
  console.log('Demo 4: Self-Learning Uplink GNN');
  console.log('='.repeat(70));

  const gnn = new SelfLearningUplinkGNN();

  // Create sample network with issue cells (deterministic values)
  const cellSnapshots = new Map<string, CellKPISnapshot>();
  const neighborRelations: NeighborRelation[] = [];

  // Add cells - some with issues (using fixed values for consistent demo)
  cellSnapshots.set('CELL_001', generateSampleCellSnapshot('CELL_001', { sinr: 2, iot: 12, p0: -98, alpha: 0.7 }));  // Issue cell
  cellSnapshots.set('CELL_002', generateSampleCellSnapshot('CELL_002', { sinr: 15, iot: 5, p0: -100, alpha: 0.8 })); // Healthy
  cellSnapshots.set('CELL_003', generateSampleCellSnapshot('CELL_003', { sinr: 3, iot: 14, p0: -95, alpha: 0.6 }));  // Issue cell
  cellSnapshots.set('CELL_004', generateSampleCellSnapshot('CELL_004', { sinr: 12, iot: 6, p0: -102, alpha: 0.9 })); // Healthy
  cellSnapshots.set('CELL_005', generateSampleCellSnapshot('CELL_005', { sinr: -1, iot: 16, p0: -90, alpha: 0.5 })); // Critical cell

  // Add neighbor relations
  neighborRelations.push(generateSampleNeighborRelation('CELL_001', 'CELL_002'));
  neighborRelations.push(generateSampleNeighborRelation('CELL_001', 'CELL_003'));
  neighborRelations.push(generateSampleNeighborRelation('CELL_002', 'CELL_003'));
  neighborRelations.push(generateSampleNeighborRelation('CELL_002', 'CELL_004'));
  neighborRelations.push(generateSampleNeighborRelation('CELL_003', 'CELL_004'));
  neighborRelations.push(generateSampleNeighborRelation('CELL_003', 'CELL_005'));
  neighborRelations.push(generateSampleNeighborRelation('CELL_004', 'CELL_005'));

  console.log(`✓ Created network with ${cellSnapshots.size} cells and ${neighborRelations.length} neighbor relations`);

  // Build graph (import at top of file)
  const { SurrogateGraphBuilder } = await import('./network-surrogate-model.js');
  const graphBuilder = new SurrogateGraphBuilder();
  const graph = graphBuilder.buildGraph(cellSnapshots, neighborRelations);

  console.log(`✓ Built graph with ${graph.nodeIds.length} nodes`);

  // Get predictions
  const predictions = gnn.predict(graph);
  console.log('\n  Cell SINR Predictions:');
  for (let i = 0; i < graph.nodeIds.length; i++) {
    const cellId = graph.nodeIds[i];
    const params = graph.powerParams.get(cellId)!;
    const status = predictions.sinr[i] < 0 ? '🔴 CRITICAL' : predictions.sinr[i] < 5 ? '🟠 ISSUE' : '🟢 OK';
    console.log(`    ${cellId}: SINR=${predictions.sinr[i].toFixed(1)} dB, IoT=${predictions.iot[i].toFixed(1)} dB, P0=${params.p0}, α=${params.alpha} ${status}`);
  }

  // Optimize issue cells and collect results
  console.log('\n  Optimizing issue cells...');
  const optimizationResults: Array<{cellId: string; params: {p0: number; alpha: number}; predictedSINR: number}> = [];

  for (const cellId of ['CELL_001', 'CELL_003', 'CELL_005']) {
    try {
      const result = gnn.optimizeCell(cellId, graph, cellSnapshots, neighborRelations);
      console.log(`    ${cellId}: P0 ${result.originalParams.p0}→${result.optimizedParams.p0}, Alpha ${result.originalParams.alpha}→${result.optimizedParams.alpha}`);
      console.log(`           SINR: ${result.originalSINR.toFixed(1)}→${result.optimizedSINR.toFixed(1)} dB (+${result.sinrImprovement.toFixed(1)} dB)`);
      optimizationResults.push({
        cellId,
        params: result.optimizedParams,
        predictedSINR: result.optimizedSINR,
      });
    } catch (e) {
      console.log(`    ${cellId}: Could not optimize - ${(e as Error).message}`);
    }
  }

  // Simulate feedback learning (as if we deployed and measured actual SINR)
  console.log('\n  Simulating deployment feedback and learning...');
  for (const opt of optimizationResults) {
    // Simulate actual SINR with some noise (real-world measurement)
    const actualSINR = opt.predictedSINR + (Math.random() - 0.5) * 3;
    const { loss, reward } = gnn.learnFromFeedback(graph, opt.cellId, opt.params, actualSINR);
    console.log(`    ${opt.cellId}: Predicted=${opt.predictedSINR.toFixed(1)}, Actual=${actualSINR.toFixed(1)}, Loss=${loss.toFixed(2)}, Reward=${reward.toFixed(2)}`);
  }

  // Get learning state after feedback
  const state = gnn.getState();
  console.log(`\n✓ Model state: version=${state.modelVersion}, samples=${state.totalSamples}, updates=${state.totalUpdates}`);
}

function demoEricssonOptimizer() {
  console.log('\n' + '='.repeat(70));
  console.log('Demo 5: Ericsson Uplink Optimizer (Production-Ready)');
  console.log('        Large-Scale Network: 100 Cells, 2000 Neighbors');
  console.log('='.repeat(70));

  const optimizer = new EricssonUplinkOptimizer();

  // Create large-scale network with 100 cells
  const cellSnapshots = new Map<string, CellKPISnapshot>();
  const neighborRelations: NeighborRelation[] = [];
  const NUM_CELLS = 100;
  const TARGET_NEIGHBORS = 2000;

  console.log('\n  Building network topology...');

  // Generate 100 cells with varying conditions
  // ~30% issue cells, ~10% critical, ~60% healthy
  for (let i = 1; i <= NUM_CELLS; i++) {
    const cellId = `CELL_${String(i).padStart(3, '0')}`;
    const rand = (i * 17 + 7) % 100; // Deterministic pseudo-random

    let cellConfig: { sinr: number; iot: number; p0: number; alpha: number };

    if (rand < 10) {
      // Critical cells (10%) - very low SINR, high interference
      cellConfig = {
        sinr: -2 + (rand % 3),
        iot: 15 + (rand % 5),
        p0: -90 + (rand % 8),
        alpha: 0.4 + (rand % 3) * 0.1,
      };
    } else if (rand < 35) {
      // Issue cells (25%) - low SINR, elevated interference
      cellConfig = {
        sinr: 1 + (rand % 4),
        iot: 11 + (rand % 4),
        p0: -95 + (rand % 10),
        alpha: 0.5 + (rand % 4) * 0.1,
      };
    } else {
      // Healthy cells (65%) - good SINR, normal interference
      cellConfig = {
        sinr: 10 + (rand % 12),
        iot: 3 + (rand % 5),
        p0: -100 + (rand % 8),
        alpha: 0.7 + (rand % 4) * 0.1,
      };
    }

    cellSnapshots.set(cellId, generateSampleCellSnapshot(cellId, cellConfig));
  }

  // Generate 2000 neighbor relations using a realistic topology
  // Each cell connects to nearby cells (simulating geographic proximity)
  const cellIds = Array.from(cellSnapshots.keys());
  const neighborsPerCell = Math.ceil(TARGET_NEIGHBORS / NUM_CELLS);

  for (let i = 0; i < NUM_CELLS; i++) {
    // Connect to nearby cells (wrap-around for edge cells)
    for (let offset = 1; offset <= neighborsPerCell && neighborRelations.length < TARGET_NEIGHBORS; offset++) {
      const j = (i + offset) % NUM_CELLS;
      if (i !== j) {
        // Avoid duplicate relations
        const existing = neighborRelations.some(
          r => (r.sourceCellId === cellIds[i] && r.targetCellId === cellIds[j]) ||
               (r.sourceCellId === cellIds[j] && r.targetCellId === cellIds[i])
        );
        if (!existing) {
          neighborRelations.push(generateSampleNeighborRelation(cellIds[i], cellIds[j]));
        }
      }
    }

    // Add some longer-range connections (macro-to-macro handovers)
    if (i % 5 === 0 && neighborRelations.length < TARGET_NEIGHBORS) {
      const farCell = (i + 25) % NUM_CELLS;
      const existing = neighborRelations.some(
        r => (r.sourceCellId === cellIds[i] && r.targetCellId === cellIds[farCell]) ||
             (r.sourceCellId === cellIds[farCell] && r.targetCellId === cellIds[i])
      );
      if (!existing) {
        neighborRelations.push(generateSampleNeighborRelation(cellIds[i], cellIds[farCell]));
      }
    }
  }

  console.log(`  ✓ Created network with ${cellSnapshots.size} cells and ${neighborRelations.length} neighbor relations`);

  // Calculate baseline network SINR before optimization
  const baselineSINRs: number[] = [];
  const baselineIoTs: number[] = [];
  let criticalCount = 0;
  let issueCount = 0;

  for (const [, snapshot] of cellSnapshots) {
    const sinr = snapshot.radioQuality.ulSinrAvg;
    const iot = snapshot.uplinkInterference.iotAvg;
    baselineSINRs.push(sinr);
    baselineIoTs.push(iot);
    if (sinr < 0) criticalCount++;
    else if (sinr < 5) issueCount++;
  }

  const avgBaselineSINR = baselineSINRs.reduce((a, b) => a + b, 0) / baselineSINRs.length;
  const avgBaselineIoT = baselineIoTs.reduce((a, b) => a + b, 0) / baselineIoTs.length;

  // Store original SINR (from optimization results) for accurate comparison
  const originalSINRMap = new Map<string, number>();

  console.log(`\n  Baseline Network Status:`);
  console.log(`  ────────────────────────────────`);
  console.log(`  Average SINR:    ${avgBaselineSINR.toFixed(1)} dB`);
  console.log(`  Average IoT:     ${avgBaselineIoT.toFixed(1)} dB`);
  console.log(`  Critical cells:  ${criticalCount} (${(criticalCount/NUM_CELLS*100).toFixed(0)}%)`);
  console.log(`  Issue cells:     ${issueCount} (${(issueCount/NUM_CELLS*100).toFixed(0)}%)`);
  console.log(`  Healthy cells:   ${NUM_CELLS - criticalCount - issueCount} (${((NUM_CELLS - criticalCount - issueCount)/NUM_CELLS*100).toFixed(0)}%)`);

  // Run network optimization
  console.log('\n  Running network-wide optimization...');
  const startTime = Date.now();
  const result = optimizer.optimizeNetwork(cellSnapshots, neighborRelations);
  const optimizationTime = Date.now() - startTime;

  console.log(`  ✓ Optimization completed in ${optimizationTime}ms`);

  // Calculate improvements
  const totalSINRImprovement = result.results.reduce((sum, r) => sum + r.sinrImprovement, 0);
  const avgImprovement = result.results.length > 0 ? totalSINRImprovement / result.results.length : 0;

  // Store original SINRs from the optimization results (model-predicted baseline)
  for (const r of result.results) {
    originalSINRMap.set(r.cellId, r.originalSINR);
  }

  // Calculate projected network-wide SINR after optimization
  // Use model-predicted values for fair comparison
  const projectedSINRs = [...baselineSINRs];
  let modelBasedSINRSum = 0;
  let modelBasedOptimizedSum = 0;
  let modelBasedCount = 0;

  for (const r of result.results) {
    const idx = cellIds.indexOf(r.cellId);
    if (idx >= 0) {
      // Update projected SINR with optimization result
      projectedSINRs[idx] = baselineSINRs[idx] + r.sinrImprovement;
      // Track model-based comparison
      modelBasedSINRSum += r.originalSINR;
      modelBasedOptimizedSum += r.optimizedSINR;
      modelBasedCount++;
    }
  }
  const avgProjectedSINR = projectedSINRs.reduce((a, b) => a + b, 0) / projectedSINRs.length;
  const globalSINRImprovement = avgProjectedSINR - avgBaselineSINR;

  // Calculate actual model-based improvement for optimized cells
  const modelBasedImprovement = modelBasedCount > 0
    ? (modelBasedOptimizedSum - modelBasedSINRSum) / modelBasedCount
    : 0;

  // Count projected status after optimization
  let projectedCritical = 0;
  let projectedIssue = 0;
  for (const sinr of projectedSINRs) {
    if (sinr < 0) projectedCritical++;
    else if (sinr < 5) projectedIssue++;
  }

  console.log(`\n  ══════════════════════════════════════════════════════════════════`);
  console.log(`  OPTIMIZATION RESULTS`);
  console.log(`  ══════════════════════════════════════════════════════════════════`);
  console.log(`  Cells analyzed:     ${cellSnapshots.size}`);
  console.log(`  Issue cells found:  ${result.results.length + criticalCount}`);
  console.log(`  Cells optimized:    ${result.results.length}`);
  console.log(`  Optimization rate:  ${(result.results.length / (issueCount + criticalCount) * 100).toFixed(0)}%`);
  console.log(`  ──────────────────────────────────────────────────────────────────`);
  console.log(`  Avg cell improvement:    +${avgImprovement.toFixed(2)} dB`);
  console.log(`  Total SINR gain:         +${totalSINRImprovement.toFixed(1)} dB`);
  console.log(`  ──────────────────────────────────────────────────────────────────`);
  console.log(`  GLOBAL NETWORK SINR:`);
  console.log(`    Before: ${avgBaselineSINR.toFixed(2)} dB`);
  console.log(`    After:  ${avgProjectedSINR.toFixed(2)} dB`);
  console.log(`    Improvement: +${globalSINRImprovement.toFixed(2)} dB ⬆`);
  console.log(`  ──────────────────────────────────────────────────────────────────`);
  console.log(`  CELL STATUS IMPROVEMENT:`);
  console.log(`    Critical: ${criticalCount} → ${projectedCritical} (${criticalCount - projectedCritical > 0 ? '-' : ''}${Math.abs(criticalCount - projectedCritical)} cells)`);
  console.log(`    Issue:    ${issueCount} → ${projectedIssue} (${issueCount - projectedIssue > 0 ? '-' : ''}${Math.abs(issueCount - projectedIssue)} cells)`);
  console.log(`    Healthy:  ${NUM_CELLS - criticalCount - issueCount} → ${NUM_CELLS - projectedCritical - projectedIssue} (+${(NUM_CELLS - projectedCritical - projectedIssue) - (NUM_CELLS - criticalCount - issueCount)} cells)`);
  console.log(`  ══════════════════════════════════════════════════════════════════`);

  // Show top optimizations
  if (result.results.length > 0) {
    const sorted = [...result.results].sort((a, b) => b.sinrImprovement - a.sinrImprovement);
    console.log(`\n  Top 10 Optimization Results:`);
    console.log(`  ┌─────────────┬──────────────────┬──────────────────┬────────────┐`);
    console.log(`  │ Cell ID     │ P0 Change        │ Alpha Change     │ SINR Gain  │`);
    console.log(`  ├─────────────┼──────────────────┼──────────────────┼────────────┤`);
    for (const r of sorted.slice(0, 10)) {
      const p0Change = `${r.originalParams.p0} → ${r.optimizedParams.p0} dBm`;
      const alphaChange = `${r.originalParams.alpha.toFixed(1)} → ${r.optimizedParams.alpha.toFixed(1)}`;
      const sinrGain = `+${r.sinrImprovement.toFixed(1)} dB`;
      console.log(`  │ ${r.cellId} │ ${p0Change.padEnd(16)} │ ${alphaChange.padEnd(16)} │ ${sinrGain.padEnd(10)} │`);
    }
    console.log(`  └─────────────┴──────────────────┴──────────────────┴────────────┘`);
  }

  // Convergence analysis
  console.log(`\n  Convergence Analysis:`);
  console.log(`  ────────────────────────────────`);
  const convergenceScore = result.metrics.convergenceScore;
  const successRate = result.results.length / Math.max(1, issueCount + criticalCount);
  const avgConfidence = result.results.length > 0
    ? result.results.reduce((sum, r) => sum + r.confidence, 0) / result.results.length
    : 0;

  console.log(`  Convergence score:  ${(convergenceScore * 100).toFixed(0)}%`);
  console.log(`  Success rate:       ${(successRate * 100).toFixed(0)}%`);
  console.log(`  Avg confidence:     ${(avgConfidence * 100).toFixed(0)}%`);
  console.log(`  Iterations/cell:    ${result.results.length > 0 ? Math.round(result.results.reduce((sum, r) => sum + r.iterations, 0) / result.results.length) : 0}`);

  // Simulate multi-round learning
  console.log('\n  Simulating deployment and learning cycles...');
  let cumulativeSamples = 0;
  let cumulativeReward = 0;

  for (let round = 1; round <= 3; round++) {
    const deployedChanges = result.results.slice(0, Math.min(10, result.results.length)).map(r => ({
      cellId: r.cellId,
      params: r.optimizedParams,
      actualSINR: r.optimizedSINR + (Math.random() - 0.3) * 2, // Slightly optimistic actual
    }));

    if (deployedChanges.length > 0) {
      const feedback = optimizer.applyFeedback(cellSnapshots, neighborRelations, deployedChanges);
      cumulativeSamples += deployedChanges.length;
      cumulativeReward += feedback.avgReward * deployedChanges.length;
      console.log(`    Round ${round}: ${deployedChanges.length} cells deployed, avg reward: ${feedback.avgReward.toFixed(2)}`);
    }
  }

  // Final status
  const status = optimizer.getStatus();
  console.log(`\n  ══════════════════════════════════════════════════════════════════`);
  console.log(`  SELF-LEARNING GNN STATUS`);
  console.log(`  ══════════════════════════════════════════════════════════════════`);
  console.log(`  Model version:      ${status.state.modelVersion}`);
  console.log(`  Total samples:      ${status.state.totalSamples}`);
  console.log(`  Total updates:      ${status.state.totalUpdates}`);
  console.log(`  Avg reward:         ${(cumulativeReward / Math.max(1, cumulativeSamples)).toFixed(2)}`);
  console.log(`  Exploration rate:   ${(status.state.explorationRate * 100).toFixed(0)}%`);
  console.log(`  Learning rate:      ${status.state.learningRate.toExponential(2)}`);
  console.log(`  ══════════════════════════════════════════════════════════════════`);
}

function demoTensorCompression() {
  console.log('\n' + '='.repeat(70));
  console.log('Demo 6: RuVector Tensor Compression');
  console.log('='.repeat(70));

  // Create sample embeddings
  const embeddings = Array(100).fill(null).map(() =>
    Array(64).fill(0).map(() => Math.random() * 2 - 1)
  );

  console.log(`✓ Created embeddings: ${embeddings.length} × ${embeddings[0].length}`);

  // Test different compression levels
  const levels: Array<'none' | 'half' | 'pq8' | 'pq4' | 'binary'> = ['none', 'half', 'pq8', 'pq4', 'binary'];

  for (const level of levels) {
    const compressed = compressEmbeddings(embeddings, level, 0.5);
    const bytesPerElement = level === 'none' ? 4 : level === 'half' ? 4 : level === 'binary' ? 1 : 1;
    const compressionRatio = (4 / bytesPerElement);
    console.log(`  ${level.padEnd(6)}: ${compressed.shape.join('×')}, compression ratio: ${compressionRatio}x`);
  }
}

async function demoClusterOptimization() {
  console.log('\n' + '='.repeat(70));
  console.log('Demo 8: Cluster-Based Multi-Cell Optimization');
  console.log('        Addressing SINR vs Interference Trade-off');
  console.log('='.repeat(70));

  const optimizer = new EricssonUplinkOptimizer();

  // Create network with interference clusters
  const cellSnapshots = new Map<string, CellKPISnapshot>();
  const neighborRelations: NeighborRelation[] = [];
  const NUM_CELLS = 50;

  console.log('\n  Building network with interference clusters...');

  // Generate cells: some critical, some issue, most healthy
  for (let i = 1; i <= NUM_CELLS; i++) {
    const cellId = `CELL_${String(i).padStart(3, '0')}`;
    const rand = (i * 13 + 5) % 100;

    let cellConfig: { sinr: number; iot: number; p0: number; alpha: number };

    if (rand < 15) {
      // Critical cells (15%) - these are the 7 stuck cells we want to fix
      cellConfig = {
        sinr: -2 + (rand % 3),
        iot: 14 + (rand % 4),
        p0: -92 + (rand % 6),
        alpha: 0.5 + (rand % 3) * 0.1,
      };
    } else if (rand < 40) {
      // Issue cells (25%)
      cellConfig = {
        sinr: 2 + (rand % 3),
        iot: 10 + (rand % 4),
        p0: -96 + (rand % 8),
        alpha: 0.6 + (rand % 3) * 0.1,
      };
    } else {
      // Healthy cells (60%)
      cellConfig = {
        sinr: 12 + (rand % 8),
        iot: 4 + (rand % 4),
        p0: -100 + (rand % 6),
        alpha: 0.7 + (rand % 3) * 0.1,
      };
    }

    cellSnapshots.set(cellId, generateSampleCellSnapshot(cellId, cellConfig));
  }

  // Generate neighbor relations creating clusters
  const cellIds = Array.from(cellSnapshots.keys());
  for (let i = 0; i < NUM_CELLS; i++) {
    // Connect to 3-5 nearby cells (creating clusters)
    for (let offset = 1; offset <= 5; offset++) {
      const j = (i + offset) % NUM_CELLS;
      neighborRelations.push(generateSampleNeighborRelation(cellIds[i], cellIds[j]));
    }
  }

  console.log(`  ✓ Created network with ${cellSnapshots.size} cells and ${neighborRelations.length} neighbor relations`);

  // Count baseline status
  let criticalCount = 0;
  let issueCount = 0;
  let baselineSINRSum = 0;

  for (const [, snapshot] of cellSnapshots) {
    const sinr = snapshot.radioQuality.ulSinrAvg;
    baselineSINRSum += sinr;
    if (sinr < 0) criticalCount++;
    else if (sinr < 5) issueCount++;
  }

  const avgBaselineSINR = baselineSINRSum / cellSnapshots.size;

  console.log(`\n  Baseline Status:`);
  console.log(`  ─────────────────────────────────`);
  console.log(`  Critical cells: ${criticalCount}`);
  console.log(`  Issue cells:    ${issueCount}`);
  console.log(`  Avg SINR:       ${avgBaselineSINR.toFixed(2)} dB`);

  // Run cluster-based optimization
  console.log('\n  Running cluster-based multi-cell optimization...');
  const startTime = Date.now();
  const result = await optimizer.optimizeNetworkClustered(cellSnapshots, neighborRelations);
  const optimizationTime = Date.now() - startTime;

  console.log(`  ✓ Optimization completed in ${optimizationTime}ms`);

  // Display cluster results
  console.log(`\n  ══════════════════════════════════════════════════════════════════`);
  console.log(`  CLUSTER OPTIMIZATION RESULTS`);
  console.log(`  ══════════════════════════════════════════════════════════════════`);
  console.log(`  Clusters identified:        ${result.clusterMetrics.totalClusters}`);
  console.log(`  Critical cells in clusters: ${result.clusterMetrics.criticalCellsInClusters}`);
  console.log(`  Cells optimized:            ${result.cellResults.length}`);
  console.log(`  Best strategy:              ${result.clusterMetrics.bestStrategy}`);
  console.log(`  Avg cluster improvement:    +${result.clusterMetrics.avgClusterImprovement.toFixed(2)} dB`);
  console.log(`  ──────────────────────────────────────────────────────────────────`);

  // Count post-optimization status
  let postCritical = 0;
  let postIssue = 0;
  let totalImprovement = 0;

  for (const cr of result.clusterResults) {
    for (const cell of cr.cellResults) {
      if (cell.statusTransition.after === 'critical') postCritical++;
      else if (cell.statusTransition.after === 'issue') postIssue++;
      totalImprovement += cell.sinrImprovement;
    }
  }

  console.log(`\n  STATUS TRANSITIONS:`);
  console.log(`    Critical: ${criticalCount} → ${postCritical} (${criticalCount - postCritical > 0 ? '↓' : ''}${Math.abs(criticalCount - postCritical)} fixed)`);
  console.log(`    Issue:    ${issueCount} → ${postIssue}`);

  // Show cluster details
  if (result.clusterResults.length > 0) {
    console.log(`\n  Top Cluster Results:`);
    console.log(`  ┌───────────────────┬────────────┬────────────┬───────────────────┐`);
    console.log(`  │ Cluster           │ Cells      │ Improvement│ Strategy          │`);
    console.log(`  ├───────────────────┼────────────┼────────────┼───────────────────┤`);

    const sortedClusters = [...result.clusterResults]
      .sort((a, b) => b.clusterImprovement - a.clusterImprovement)
      .slice(0, 5);

    for (const cr of sortedClusters) {
      const clusterId = cr.clusterId.split('_')[1]?.slice(0, 8) || 'cluster';
      const cellCount = `${cr.cellResults.length} cells`;
      const improvement = `+${cr.clusterImprovement.toFixed(2)} dB`;
      const strategy = cr.strategyUsed.slice(0, 17);
      console.log(`  │ ${clusterId.padEnd(17)} │ ${cellCount.padEnd(10)} │ ${improvement.padEnd(10)} │ ${strategy.padEnd(17)} │`);
    }
    console.log(`  └───────────────────┴────────────┴────────────┴───────────────────┘`);
  }

  // Show neighbor sacrifice strategy effectiveness
  const sacrificeClusters = result.clusterResults.filter(
    cr => cr.strategyUsed === 'neighbor_sacrifice'
  );
  if (sacrificeClusters.length > 0) {
    const avgSacrificeImprovement =
      sacrificeClusters.reduce((sum, cr) => sum + cr.clusterImprovement, 0) /
      sacrificeClusters.length;
    console.log(`\n  Neighbor Sacrifice Strategy:`);
    console.log(`    Clusters using strategy: ${sacrificeClusters.length}`);
    console.log(`    Avg improvement: +${avgSacrificeImprovement.toFixed(2)} dB`);
    console.log(`    (Neighbors reduce power to help critical cells)`);
  }

  console.log(`\n  ══════════════════════════════════════════════════════════════════`);
  console.log(`  KEY INSIGHT: Cluster optimization addresses the SINR vs IoT`);
  console.log(`  trade-off by coordinating power control across interfering cells.`);
  console.log(`  ══════════════════════════════════════════════════════════════════`);

  // Print recommendations
  console.log(`\n  Recommendations:`);
  for (const rec of result.recommendations.slice(0, 6)) {
    console.log(`    ${rec}`);
  }
}

function demoDifferentiableSearchVectors() {
  console.log('\n' + '='.repeat(70));
  console.log('Demo 7: Differentiable Vector Search');
  console.log('='.repeat(70));

  // Create query and candidates
  const query = Array(64).fill(0).map(() => Math.random() * 2 - 1);
  const candidates = Array(1000).fill(null).map(() =>
    Array(64).fill(0).map(() => Math.random() * 2 - 1)
  );

  console.log(`✓ Query vector: 64 dimensions`);
  console.log(`✓ Candidate vectors: 1000 × 64`);

  // Search with different temperatures
  for (const temp of [0.1, 1.0, 5.0]) {
    const result = differentiableSearch(query, candidates, 5, temp);
    const entropy = -result.softWeights.reduce((sum, w) => sum + (w > 0 ? w * Math.log(w) : 0), 0);
    console.log(`  Temperature ${temp}: top indices [${result.indices.slice(0, 3).join(', ')}...], entropy=${entropy.toFixed(2)}`);
  }
}

// ============================================================================
// MAIN
// ============================================================================

async function main() {
  console.log('\n' + '═'.repeat(70));
  console.log('   SELF-LEARNING GNN FOR ERICSSON UPLINK OPTIMIZATION');
  console.log('   Demo Runner');
  console.log('═'.repeat(70));

  try {
    await demoRuVectorGNNLayer();
    demoExperienceReplay();
    demoDifferentiableSearch();
    await demoSelfLearningGNN();
    demoEricssonOptimizer();
    demoTensorCompression();
    demoDifferentiableSearchVectors();
    await demoClusterOptimization();

    console.log('\n' + '═'.repeat(70));
    console.log('   ALL DEMOS COMPLETED SUCCESSFULLY ✓');
    console.log('═'.repeat(70) + '\n');
  } catch (error) {
    console.error('\nError running demo:', error);
    process.exit(1);
  }
}

main();
