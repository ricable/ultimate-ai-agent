#!/usr/bin/env tsx
/**
 * RuVector Self-Learning GNN Demo for Radio Network Uplink Optimization
 *
 * Demonstrates dynamic adjustment of pZeroNominalPusch (P0) and Alpha (α)
 * parameters to predict SINR improvement on source cells WITHOUT degrading
 * SINR on neighbor cells.
 *
 * Based on: Ericsson's GNN approach presented at PyTorch Conference 2023
 *
 * Run: npx tsx src/gnn/run-ruvector-demo.ts
 */

import { RuVectorUplinkOptimizer } from './ruvector-uplink-optimizer.js';
import type {
  CellKPISnapshot,
  NeighborRelation,
  CellInfo,
  AccessibilityKPI,
  RetainabilityKPI,
  RadioQualityKPI,
  MobilityKPI,
  UplinkInterferenceKPI,
  UplinkPowerControlKPI,
} from '../models/ran-kpi.js';

// ============================================================================
// SAMPLE DATA GENERATION
// ============================================================================

/**
 * Generate a sample cell KPI snapshot
 */
function generateCellSnapshot(
  cellId: string,
  p0: number,
  alpha: number,
  sinr: number,
  iot: number,
  isIssueCell: boolean
): CellKPISnapshot {
  const timestamp = new Date();

  const cell: CellInfo = {
    cellId,
    siteName: `Site-${cellId.split('-')[1] ?? '001'}`,
    sectorId: parseInt(cellId.split('-')[2] ?? '1'),
    technology: '5G',
    frequency: 3500,
    bandwidth: 100,
    latitude: 40.7128 + Math.random() * 0.1,
    longitude: -74.0060 + Math.random() * 0.1,
    azimuth: Math.floor(Math.random() * 360),
    tilt: 5 + Math.random() * 5,
    height: 30 + Math.random() * 20,
  };

  const accessibility: AccessibilityKPI = {
    rrcSetupAttempts: 5000 + Math.floor(Math.random() * 3000),
    rrcSetupSuccessRate: 98 + Math.random() * 2,
    erabSetupAttempts: 4500 + Math.floor(Math.random() * 2500),
    erabSetupSuccessRate: 97 + Math.random() * 3,
    initialContextSetupSuccessRate: 97 + Math.random() * 3,
    s1SignalingSetupSuccessRate: 99 + Math.random(),
  };

  const retainability: RetainabilityKPI = {
    erabDropRate: isIssueCell ? 2 + Math.random() * 3 : 0.5 + Math.random(),
    erabReleaseRate: 10 + Math.random() * 5,
    rrcConnectionAbnormalRelease: isIssueCell ? 3 + Math.random() * 2 : 1 + Math.random(),
    dataSessionRetainability: isIssueCell ? 92 + Math.random() * 3 : 97 + Math.random() * 3,
  };

  const radioQuality: RadioQualityKPI = {
    dlAvgCqi: isIssueCell ? 6 + Math.random() * 3 : 10 + Math.random() * 4,
    ulSinrAvg: sinr,
    ulSinrP10: sinr - 5 - Math.random() * 3,
    ulSinrP50: sinr - 2,
    ulSinrP90: sinr + 3 + Math.random() * 3,
    rsrpAvg: isIssueCell ? -115 - Math.random() * 10 : -95 - Math.random() * 15,
    rsrqAvg: isIssueCell ? -15 - Math.random() * 5 : -8 - Math.random() * 5,
    rssiAvg: -75 - Math.random() * 15,
    dlThroughput: isIssueCell ? 50 + Math.random() * 50 : 150 + Math.random() * 100,
    ulThroughput: isIssueCell ? 10 + Math.random() * 20 : 40 + Math.random() * 30,
    dlSpectralEfficiency: isIssueCell ? 2 + Math.random() * 2 : 5 + Math.random() * 3,
    ulSpectralEfficiency: isIssueCell ? 1 + Math.random() : 3 + Math.random() * 2,
    dlBlerPercent: isIssueCell ? 8 + Math.random() * 5 : 2 + Math.random() * 3,
    ulBlerPercent: isIssueCell ? 10 + Math.random() * 5 : 3 + Math.random() * 3,
    dlPrbUtilization: 40 + Math.random() * 40,
    ulPrbUtilization: 30 + Math.random() * 40,
  };

  const mobility: MobilityKPI = {
    intraFreqHoAttempts: 1000 + Math.floor(Math.random() * 2000),
    intraFreqHoSuccessRate: 95 + Math.random() * 5,
    interFreqHoAttempts: 200 + Math.floor(Math.random() * 300),
    interFreqHoSuccessRate: 92 + Math.random() * 8,
    interRatHoAttempts: 50 + Math.floor(Math.random() * 100),
    interRatHoSuccessRate: 88 + Math.random() * 12,
    tooEarlyHo: isIssueCell ? 5 + Math.random() * 5 : 1 + Math.random() * 2,
    tooLateHo: isIssueCell ? 4 + Math.random() * 4 : 0.5 + Math.random(),
    pingPongHo: isIssueCell ? 8 + Math.random() * 7 : 2 + Math.random() * 3,
    hoInterruptionTime: 30 + Math.random() * 20,
  };

  const uplinkInterference: UplinkInterferenceKPI = {
    timestamp,
    cellId,
    iotAvg: iot,
    iotP50: iot - 1,
    iotP95: iot + 3 + Math.random() * 2,
    rip: -105 + iot / 2,
    thermalNoiseFloor: -110,
    interferenceContribution: iot * 0.8,
    prbInterferenceDistribution: [20, 30, 25, 15, 10],
    highInterferencePrbRatio: isIssueCell ? 25 + Math.random() * 20 : 5 + Math.random() * 10,
    interferenceVariance: 2 + Math.random() * 3,
  };

  const uplinkPowerControl: UplinkPowerControlKPI = {
    timestamp,
    cellId,
    p0NominalPusch: p0,
    alpha: alpha,
    tpc: 1,
    puschPowerAvg: p0 + alpha * 120 - 10,
    powerHeadroomAvg: isIssueCell ? 3 + Math.random() * 5 : 15 + Math.random() * 10,
    powerHeadroomP10: isIssueCell ? -5 + Math.random() * 3 : 5 + Math.random() * 5,
    powerHeadroomP90: 20 + Math.random() * 10,
    powerLimitedUeRatio: isIssueCell ? 15 + Math.random() * 15 : 3 + Math.random() * 5,
    negativePowerHeadroomRatio: isIssueCell ? 12 + Math.random() * 10 : 2 + Math.random() * 3,
    pathLossAvg: 110 + Math.random() * 30,
    pathLossP10: 95 + Math.random() * 10,
    pathLossP50: 115 + Math.random() * 10,
    pathLossP90: 135 + Math.random() * 10,
    closedLoopCorrectionAvg: 0.5 + Math.random(),
  };

  return {
    timestamp,
    cell,
    accessibility,
    retainability,
    radioQuality,
    mobility,
    uplinkInterference,
    uplinkPowerControl,
  };
}

/**
 * Generate neighbor relations for a network
 */
function generateNeighborRelations(
  cellIds: string[],
  cellSnapshots: Map<string, CellKPISnapshot>
): NeighborRelation[] {
  const relations: NeighborRelation[] = [];

  for (let i = 0; i < cellIds.length; i++) {
    const sourceCellId = cellIds[i];
    const sourceSnapshot = cellSnapshots.get(sourceCellId)!;

    // Each cell has 2-4 intra-freq neighbors
    const numNeighbors = 2 + Math.floor(Math.random() * 3);
    const neighborIndices = new Set<number>();

    while (neighborIndices.size < numNeighbors && neighborIndices.size < cellIds.length - 1) {
      const idx = Math.floor(Math.random() * cellIds.length);
      if (idx !== i) neighborIndices.add(idx);
    }

    for (const j of neighborIndices) {
      const targetCellId = cellIds[j];
      const targetSnapshot = cellSnapshots.get(targetCellId)!;

      relations.push({
        timestamp: new Date(),
        sourceCellId,
        targetCellId,
        relationshipType: 'intra-freq',
        measurementType: 'A3',
        sourceSinr: sourceSnapshot.radioQuality.ulSinrAvg,
        targetSinr: targetSnapshot.radioQuality.ulSinrAvg,
        sourceRsrp: sourceSnapshot.radioQuality.rsrpAvg,
        targetRsrp: targetSnapshot.radioQuality.rsrpAvg,
        a3Offset: 3,
        hysteresis: 2,
        timeToTrigger: 320,
        hoAttempts: 500 + Math.floor(Math.random() * 500),
        hoSuccessRate: 92 + Math.random() * 8,
        avgHoTime: 40 + Math.random() * 30,
        tooEarlyHo: Math.random() * 2,
        tooLateHo: Math.random() * 2,
        pingPongRatio: Math.random() * 3,
      });
    }
  }

  return relations;
}

/**
 * Generate a sample network with issue cells
 */
function generateSampleNetwork(): {
  cellSnapshots: Map<string, CellKPISnapshot>;
  neighborRelations: NeighborRelation[];
} {
  const cellSnapshots = new Map<string, CellKPISnapshot>();
  const cellIds: string[] = [];

  console.log('Generating sample network...\n');

  // Generate cells - mix of healthy and issue cells
  const numCells = 12;
  const numIssueCells = 4;

  for (let i = 1; i <= numCells; i++) {
    const cellId = `Cell-${String(i).padStart(3, '0')}`;
    cellIds.push(cellId);

    const isIssueCell = i <= numIssueCells;

    // Issue cells have suboptimal P0/Alpha and low SINR
    let p0: number, alpha: number, sinr: number, iot: number;

    if (isIssueCell) {
      // Issue cell configurations
      const issueType = i % 3;
      switch (issueType) {
        case 0: // Low SINR, high IoT - P0 too low
          p0 = -106 - Math.floor(Math.random() * 4);
          alpha = 0.6 + Math.random() * 0.2;
          sinr = -1 + Math.random() * 4;  // -1 to 3 dB (critical/issue)
          iot = 12 + Math.random() * 4;   // High interference
          break;
        case 1: // Moderate SINR issue - unbalanced config
          p0 = -95 + Math.floor(Math.random() * 5);  // Too high P0
          alpha = 1.0;  // Too aggressive
          sinr = 3 + Math.random() * 2;  // 3-5 dB (issue)
          iot = 14 + Math.random() * 3;   // Very high interference
          break;
        default: // Cell edge problem
          p0 = -102 - Math.floor(Math.random() * 3);
          alpha = 0.5;  // Too conservative
          sinr = 2 + Math.random() * 3;  // 2-5 dB (issue)
          iot = 10 + Math.random() * 3;
      }
    } else {
      // Healthy cell configurations
      p0 = -98 - Math.floor(Math.random() * 4);
      alpha = 0.7 + Math.random() * 0.2;
      sinr = 10 + Math.random() * 10;  // 10-20 dB (healthy)
      iot = 4 + Math.random() * 4;     // Low interference
    }

    const snapshot = generateCellSnapshot(cellId, p0, alpha, sinr, iot, isIssueCell);
    cellSnapshots.set(cellId, snapshot);

    const status = isIssueCell ? '⚠️ ISSUE' : '✓ OK';
    console.log(`  ${cellId}: ${status} | P0=${p0} dBm, Alpha=${alpha} | SINR=${sinr.toFixed(1)} dB, IoT=${iot.toFixed(1)} dB`);
  }

  // Generate neighbor relations
  const neighborRelations = generateNeighborRelations(cellIds, cellSnapshots);
  console.log(`\nGenerated ${neighborRelations.length} neighbor relations\n`);

  return { cellSnapshots, neighborRelations };
}

// ============================================================================
// MAIN DEMO
// ============================================================================

async function main() {
  console.log('═'.repeat(70));
  console.log('  RuVector Self-Learning GNN for Radio Network Uplink Optimization');
  console.log('═'.repeat(70));
  console.log('\nDynamic adjustment of pZeroNominalPusch (P0) and Alpha (α)');
  console.log('to predict SINR improvement WITHOUT degrading neighbor cells.\n');
  console.log('Based on: Ericsson PyTorch Conference 2023 presentation\n');
  console.log('─'.repeat(70));

  // Generate sample network
  const { cellSnapshots, neighborRelations } = generateSampleNetwork();

  // Create RuVector optimizer
  console.log('Initializing RuVector GNN optimizer...');
  const optimizer = await RuVectorUplinkOptimizer.create({
    inputDim: 24,
    hiddenDim: 64,
    numHeads: 4,
    thresholds: {
      sinrLow: 5,
      sinrCritical: 0,
      iotHigh: 10,
      powerLimitedHigh: 20,
    },
    optimization: {
      maxIterations: 100,
      convergenceThreshold: 0.01,
      neighborImpactWeight: 0.4,  // Strong penalty for neighbor degradation
      minImprovement: 0.5,
    },
  });

  // Run network optimization
  console.log('\n' + '═'.repeat(70));
  console.log('  Starting Network Optimization');
  console.log('═'.repeat(70));

  const result = await optimizer.optimizeNetwork(cellSnapshots, neighborRelations);

  // Display results
  console.log('\n' + '═'.repeat(70));
  console.log('  Optimization Results');
  console.log('═'.repeat(70));

  if (result.results.length > 0) {
    console.log('\n📊 Optimized Cells:\n');

    for (const r of result.results) {
      const neighborStatus = r.neighborImpact >= -0.5
        ? '✓ Neighbors protected'
        : `⚠ ${Math.abs(r.neighborImpact).toFixed(1)} dB neighbor impact`;

      console.log(`┌─ ${r.cellId} ─────────────────────────────────────┐`);
      console.log(`│  P0:    ${String(r.originalParams.p0).padStart(4)} → ${String(r.optimizedParams.p0).padStart(4)} dBm`);
      console.log(`│  Alpha: ${r.originalParams.alpha.toFixed(1).padStart(4)} → ${r.optimizedParams.alpha.toFixed(1).padStart(4)}`);
      console.log(`│  SINR:  ${r.originalSINR.toFixed(1).padStart(5)} → ${r.optimizedSINR.toFixed(1).padStart(5)} dB (+${r.sinrImprovement.toFixed(1)} dB)`);
      console.log(`│  Status: ${r.statusTransition.before.padEnd(8)} → ${r.statusTransition.after}`);
      console.log(`│  ${neighborStatus}`);
      console.log(`└${'─'.repeat(47)}┘`);
    }
  }

  // Summary metrics
  console.log('\n' + '─'.repeat(70));
  console.log('📈 Summary Metrics:');
  console.log('─'.repeat(70));
  console.log(`   Issue cells detected:    ${result.metrics.issueCellsDetected}`);
  console.log(`   Cells optimized:         ${result.metrics.cellsOptimized}`);
  console.log(`   Avg SINR improvement:    +${result.metrics.avgSINRImprovement.toFixed(2)} dB`);
  console.log(`   Avg neighbor impact:     ${result.metrics.avgNeighborImpact.toFixed(2)} dB`);
  console.log(`   Success rate:            ${(result.metrics.successRate * 100).toFixed(1)}%`);

  // Recommendations
  console.log('\n' + '─'.repeat(70));
  console.log('📋 Recommendations:');
  console.log('─'.repeat(70));
  for (const rec of result.recommendations) {
    console.log(`   ${rec}`);
  }

  // Key insight
  console.log('\n' + '═'.repeat(70));
  console.log('  KEY INSIGHT: P0 and Alpha Trade-offs');
  console.log('═'.repeat(70));
  console.log(`
  pZeroNominalPusch (P0):
  • Target uplink transmission power of UE
  • HIGH P0 = High UE throughput BUT higher interference on neighbors
  • LOW P0 = Less interference BUT may cause coverage issues

  Alpha (α):
  • Pathloss compensation factor (0-1)
  • HIGH α = Better cell-edge coverage BUT higher neighbor interference
  • LOW α = Less interference BUT cell-edge users may suffer

  The GNN model learns to predict the OPTIMAL balance:
  • Maximize SINR for issue cells
  • WITHOUT degrading neighbor cell SINR > 1.5 dB
  `);

  console.log('═'.repeat(70));
  console.log('  Demo Complete');
  console.log('═'.repeat(70));
}

// Run demo
main().catch(console.error);
