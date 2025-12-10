#!/usr/bin/env tsx
/**
 * Quick demo for the Network Surrogate Model (Digital Twin)
 * Demonstrates the GNN-based uplink optimization
 */

import {
  SurrogateGraphBuilder,
  GNNSurrogateModel,
  SurrogateOptimizer,
  IssueCellDetector,
  SurrogateVisualizer,
  DEFAULT_SURROGATE_CONFIG,
} from './network-surrogate-model.js';

import type { CellKPISnapshot, NeighborRelation } from '../models/ran-kpi.js';

// Generate sample cell data
function generateCell(cellId: string, sinr: number, iot: number, p0: number, alpha: number): CellKPISnapshot {
  return {
    timestamp: new Date(),
    cell: {
      cellId, enodebId: 'eNB_001', sectorId: 0, frequency: 2600,
      band: 'B7', technology: 'LTE', pci: 100, tac: 12345,
    },
    accessibility: {
      rrcSetupAttempts: 1000, rrcSetupSuccess: 980, rrcSetupFailure: 20, rrcSetupSuccessRate: 98,
      erabSetupAttempts: 900, erabSetupSuccess: 880, erabSetupFailure: 20, erabSetupSuccessRate: 97.8,
      s1SigConnEstabAttempts: 1000, s1SigConnEstabSuccess: 990, s1SigConnEstabSuccessRate: 99,
      initialContextSetupAttempts: 900, initialContextSetupSuccess: 880, initialContextSetupSuccessRate: 97.8,
    },
    retainability: {
      erabNormalRelease: 5000, erabAbnormalRelease: 50, erabDropRate: 1,
      voiceCallAttempts: 1000, voiceCallDrops: 10, voiceCallDropRate: 1,
      dataSessionAttempts: 8000, dataSessionDrops: 80, dataSessionRetainability: 99,
    },
    radioQuality: {
      dlAvgCqi: 12, dlRi1Ratio: 50, dlRi2Ratio: 50, dlBlerPercent: 3,
      ulSinrAvg: sinr, ulSinrP10: sinr - 5, ulSinrP50: sinr, ulSinrP90: sinr + 5, ulBlerPercent: 4,
      rsrpAvg: -95, rsrpP10: -105, rsrpP50: -95, rsrpP90: -85,
      rsrqAvg: -10, rsrqP10: -15, rsrqP50: -10, rsrqP90: -5,
      dlSpectralEfficiency: 4, ulSpectralEfficiency: 2,
    },
    mobility: {
      intraFreqHoAttempts: 500, intraFreqHoSuccess: 490, intraFreqHoFailure: 10, intraFreqHoSuccessRate: 98,
      interFreqHoAttempts: 100, interFreqHoSuccess: 95, interFreqHoFailure: 5, interFreqHoSuccessRate: 95,
      interRatHoAttempts: 50, interRatHoSuccess: 45, interRatHoFailure: 5, interRatHoSuccessRate: 90,
      x2HoAttempts: 400, x2HoSuccess: 390, x2HoSuccessRate: 97.5,
      s1HoAttempts: 100, s1HoSuccess: 95, s1HoSuccessRate: 95,
      tooEarlyHo: 5, tooLateHo: 3, wrongCellHo: 2, pingPongHo: 8,
      incomingHoTotal: 600, outgoingHoTotal: 580,
    },
    uplinkInterference: {
      prbUlInterferenceAvg: -100 + iot, prbUlInterferenceP10: -105 + iot,
      prbUlInterferenceP50: -100 + iot, prbUlInterferenceP90: -95 + iot, prbUlInterferenceP99: -90 + iot,
      iotAvg: iot, iotP95: iot + 2, rip: -100 + iot,
      externalInterferenceDetected: iot > 10, externalInterferenceLevel: iot > 10 ? 'medium' : 'none',
      puschSinrDegradation: Math.max(0, 8 - sinr), highInterferencePrbRatio: iot * 5,
    },
    uplinkPowerControl: {
      p0NominalPusch: p0, p0NominalPucch: p0 - 5, alpha,
      ueTxPowerAvg: 15, ueTxPowerP10: 5, ueTxPowerP50: 15, ueTxPowerP90: 20, ueTxPowerMax: 23,
      powerHeadroomAvg: 12, powerHeadroomP10: 5, powerHeadroomP50: 12, powerHeadroomP90: 18,
      negativePowerHeadroomRatio: Math.max(0, 15 - sinr),
      pathLossAvg: 115, pathLossP10: 100, pathLossP50: 115, pathLossP90: 130,
      tpcUpCommands: 1000, tpcDownCommands: 800, tpcAccumulatedOffset: 2,
      powerLimitedUeRatio: Math.max(0, 25 - sinr),
    },
  };
}

function generateNeighbor(src: string, tgt: string): NeighborRelation {
  return {
    sourceCellId: src, targetCellId: tgt, relationshipType: 'intra-freq',
    sourcePci: 100, sourceFrequency: 2600, sourceRsrp: -90, sourceSinr: 10,
    targetPci: 101, targetFrequency: 2600, targetRsrp: -95, targetSinr: 8,
    hoAttempts: 200, hoSuccess: 195, hoFailure: 5, hoSuccessRate: 97.5,
    a3Offset: 3, hysteresis: 2, timeToTrigger: 480, neighborQuality: 'good',
    distance: 1000,
  };
}

async function main() {
  console.log('='.repeat(60));
  console.log('GNN Network Surrogate Model - Digital Twin Demo');
  console.log('='.repeat(60));

  // Create a small test network: 10 cells
  const cellSnapshots = new Map<string, CellKPISnapshot>();
  const neighborRelations: NeighborRelation[] = [];

  // Mix of healthy, issue, and critical cells
  cellSnapshots.set('CELL_001', generateCell('CELL_001', 2, 12, -98, 0.7));   // Issue
  cellSnapshots.set('CELL_002', generateCell('CELL_002', 15, 5, -100, 0.8));  // Healthy
  cellSnapshots.set('CELL_003', generateCell('CELL_003', 3, 11, -95, 0.6));   // Issue
  cellSnapshots.set('CELL_004', generateCell('CELL_004', -1, 16, -90, 0.5));  // Critical
  cellSnapshots.set('CELL_005', generateCell('CELL_005', 18, 4, -100, 0.8));  // Healthy
  cellSnapshots.set('CELL_006', generateCell('CELL_006', 4, 10, -97, 0.7));   // Issue
  cellSnapshots.set('CELL_007', generateCell('CELL_007', 12, 6, -100, 0.8));  // Healthy
  cellSnapshots.set('CELL_008', generateCell('CELL_008', 1, 13, -93, 0.6));   // Issue
  cellSnapshots.set('CELL_009', generateCell('CELL_009', 20, 3, -102, 0.8));  // Healthy
  cellSnapshots.set('CELL_010', generateCell('CELL_010', -2, 15, -88, 0.4));  // Critical

  // Create neighbor relations (ring topology + cross links)
  for (let i = 1; i <= 10; i++) {
    const next = i === 10 ? 1 : i + 1;
    neighborRelations.push(generateNeighbor(`CELL_${String(i).padStart(3, '0')}`, `CELL_${String(next).padStart(3, '0')}`));
    if (i <= 5) {
      neighborRelations.push(generateNeighbor(`CELL_${String(i).padStart(3, '0')}`, `CELL_${String(i + 5).padStart(3, '0')}`));
    }
  }

  console.log(`\nNetwork: ${cellSnapshots.size} cells, ${neighborRelations.length} neighbor relations`);

  // Run optimization
  const optimizer = new SurrogateOptimizer();
  console.log('\nRunning network optimization...');
  const result = optimizer.optimizeNetwork(cellSnapshots, neighborRelations);

  // Display results
  console.log('\n' + '='.repeat(60));
  console.log('OPTIMIZATION RESULTS');
  console.log('='.repeat(60));
  console.log(`Issue cells detected: ${result.issueCells}`);
  console.log(`Cells optimized: ${result.optimizedCells}`);
  console.log(`Success rate: ${(result.aggregateMetrics.successRate * 100).toFixed(0)}%`);
  console.log(`\nSINR Metrics:`);
  console.log(`  Avg Before: ${result.aggregateMetrics.avgSinrBefore.toFixed(2)} dB`);
  console.log(`  Avg After:  ${result.aggregateMetrics.avgSinrAfter.toFixed(2)} dB`);
  console.log(`  Improvement: +${result.aggregateMetrics.avgImprovement.toFixed(2)} dB`);
  console.log(`  Neighbor Impact: ${result.aggregateMetrics.avgNeighborImpact.toFixed(2)} dB`);

  if (result.results.length > 0) {
    console.log('\nTop Optimization Results:');
    console.log('-'.repeat(60));
    for (const r of result.results.slice(0, 5)) {
      console.log(`  ${r.cellId}: P0 ${r.originalParams.p0}→${r.optimizedParams.p0}, Alpha ${r.originalParams.alpha}→${r.optimizedParams.alpha}`);
      console.log(`           SINR: ${r.originalSINR.toFixed(1)}→${r.optimizedSINR.toFixed(1)} dB (+${r.sinrImprovement.toFixed(1)} dB)`);
    }
  }

  console.log('\nRecommendations:');
  for (const rec of result.recommendations) {
    console.log(rec);
  }

  console.log('\n' + '='.repeat(60));
  console.log('DEMO COMPLETED SUCCESSFULLY ✓');
  console.log('='.repeat(60));
}

main().catch(console.error);
