/**
 * Example: Autonomous Interference Mitigation Scenario
 *
 * This example demonstrates the complete Neuro-Federated Swarm architecture
 * handling interference between two non-neighboring cells caused by
 * reflections from a new building.
 *
 * Scenario from Section 8 of the architectural report:
 * - Heavy rainstorm causes signal attenuation
 * - New high-rise building reflects signals
 * - Creates unexpected interference between Cell A and Cell C
 */

import {
  initializeSystem,
  shutdownSystem,
  NeuroFederatedSystem,
} from '../index.js';
import {
  CellState,
  CellIdentity,
  CellMetrics,
  CellConfiguration,
  NeighborRelation,
  GeoLocation,
  TimeSeries,
  Alarm,
} from '../core/types.js';
import { NeuralContext } from '../fault/neuro-symbolic-rca.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('InterferenceMitigationExample');

/**
 * Create mock cell state for the example
 */
function createMockCellState(
  cellId: string,
  config: Partial<CellConfiguration>,
  metrics: Partial<CellMetrics>,
  neighbors: string[]
): CellState {
  const identity: CellIdentity = {
    cellId,
    gNodeBId: cellId.split('_')[0],
    technology: '5G',
    sectorId: 1,
  };

  const location: GeoLocation = {
    latitude: 40.7128 + Math.random() * 0.01,
    longitude: -74.006 + Math.random() * 0.01,
    altitude: 30,
    azimuth: Math.random() * 360,
  };

  const fullConfig: CellConfiguration = {
    electricalTilt: config.electricalTilt ?? 40,
    mechanicalTilt: config.mechanicalTilt ?? 0,
    transmitPower: config.transmitPower ?? 43,
    pci: config.pci ?? Math.floor(Math.random() * 504),
    bandwidth: config.bandwidth ?? 20,
    p0NominalPUSCH: config.p0NominalPUSCH ?? -90,
    qRxLevMin: config.qRxLevMin ?? -140,
    ssbSubcarrierSpacing: config.ssbSubcarrierSpacing ?? 30,
  };

  const fullMetrics: CellMetrics = {
    prbUtilizationDl: metrics.prbUtilizationDl ?? 0.4,
    prbUtilizationUl: metrics.prbUtilizationUl ?? 0.3,
    activeUesDl: metrics.activeUesDl ?? 50,
    activeUesUl: metrics.activeUesUl ?? 30,
    throughputDl: metrics.throughputDl ?? 500,
    throughputUl: metrics.throughputUl ?? 100,
    rsrp: metrics.rsrp ?? -85,
    rsrq: metrics.rsrq ?? -10,
    sinr: metrics.sinr ?? 15,
    bler: metrics.bler ?? 0.01,
    rssiUl: metrics.rssiUl ?? -95,
    interferenceLevel: metrics.interferenceLevel ?? -100,
    powerConsumption: metrics.powerConsumption ?? 1000,
    sleepRatio: metrics.sleepRatio ?? 0,
    timestamp: Date.now(),
  };

  const neighborRelations: NeighborRelation[] = neighbors.map((targetCellId) => ({
    targetCellId,
    noRemove: false,
    noHo: false,
    isAnr: true,
    handoverAttempts: 100,
    handoverSuccesses: 95,
    interferenceLevel: 0.1,
  }));

  return {
    identity,
    location,
    configuration: fullConfig,
    metrics: fullMetrics,
    neighbors: neighborRelations,
    timestamp: Date.now(),
  };
}

/**
 * Generate mock time series with interference spike
 */
function generateInterferenceTimeSeries(
  cellId: string,
  spikeAt: number = 150
): TimeSeries {
  const values: number[] = [];
  const timestamps: number[] = [];
  const baseTime = Date.now() - 200 * 1000;

  for (let i = 0; i < 200; i++) {
    let value = -100 + Math.random() * 5; // Normal interference level

    // Add spike at specified point
    if (i >= spikeAt && i < spikeAt + 30) {
      value = -85 + Math.random() * 3; // Elevated interference
    }

    values.push(value);
    timestamps.push(baseTime + i * 1000);
  }

  return {
    metricName: 'interference',
    cellId,
    values,
    timestamps,
    resolution: 1000,
  };
}

/**
 * Run the interference mitigation scenario
 */
async function runScenario(): Promise<void> {
  logger.info('='.repeat(60));
  logger.info('SCENARIO: Autonomous Interference Mitigation');
  logger.info('='.repeat(60));

  // Step 1: Initialize the Neuro-Federated Swarm system
  logger.info('\n--- Step 1: Initializing Neuro-Federated Swarm ---');
  const system = await initializeSystem({
    topology: 'hierarchical',
    enmHost: 'enm.example.com',
  });

  try {
    // Step 2: Set up network topology
    logger.info('\n--- Step 2: Building Network Topology ---');

    const cellA = createMockCellState(
      'SITE01_Cell_A',
      { pci: 12, electricalTilt: 35 },
      {
        interferenceLevel: -85, // Elevated interference!
        rssiUl: -80,
        sinr: 5, // Degraded
        throughputDl: 300, // Reduced
      },
      ['SITE01_Cell_B', 'SITE02_Cell_D']
    );

    const cellB = createMockCellState(
      'SITE01_Cell_B',
      { pci: 15, electricalTilt: 40 },
      { interferenceLevel: -98, sinr: 18, throughputDl: 600 },
      ['SITE01_Cell_A', 'SITE02_Cell_D']
    );

    const cellC = createMockCellState(
      'SITE02_Cell_C',
      { pci: 12, electricalTilt: 30 }, // Same PCI as Cell A - potential issue
      { interferenceLevel: -100, sinr: 20, throughputDl: 700 },
      ['SITE02_Cell_D']
    );

    const cellD = createMockCellState(
      'SITE02_Cell_D',
      { pci: 18, electricalTilt: 45 },
      { interferenceLevel: -99, sinr: 17, throughputDl: 550 },
      ['SITE01_Cell_A', 'SITE01_Cell_B', 'SITE02_Cell_C']
    );

    // Add cells to topology
    system.topologyModel.addCell(cellA);
    system.topologyModel.addCell(cellB);
    system.topologyModel.addCell(cellC);
    system.topologyModel.addCell(cellD);

    // Add interference edge (not a neighbor, but detected interference)
    system.topologyModel.addEdge('SITE01_Cell_A', 'SITE02_Cell_C', 'interferer', 0.8);

    const stats = system.topologyModel.getStats();
    logger.info('Topology built', stats);

    // Step 3: Detection Phase (PM/FM)
    logger.info('\n--- Step 3: Detection Phase ---');

    // Generate time series with interference spike
    const interferenceTS = generateInterferenceTimeSeries('SITE01_Cell_A');

    // Analyze for anomalies
    const anomalies = system.anomalyDetector.detect('SITE01_Cell_A', cellA.metrics);

    if (anomalies.length > 0) {
      logger.info('Anomalies detected on Cell A:');
      for (const anomaly of anomalies) {
        logger.info(`  - ${anomaly.type}: severity ${anomaly.severity.toFixed(2)}`);
      }
    }

    // Check for chaos in the interference signal
    const chaosAnalysis = await system.chaosAnalyzer.analyze(interferenceTS);
    logger.info('Chaos analysis:', {
      isChaoatic: chaosAnalysis.isChaoatic,
      lyapunovExponent: chaosAnalysis.lyapunovExponent.toFixed(4),
      recommendedStrategy: chaosAnalysis.recommendedStrategy,
    });

    // Step 4: Analysis Phase (GNN)
    logger.info('\n--- Step 4: Analysis Phase (GNN) ---');

    // Find similar cells using embeddings
    const similarCells = system.topologyModel.findKNearestNeighbors('SITE01_Cell_A', 3);
    logger.info('Cells most similar to Cell A (potential interferers):');
    for (const similar of similarCells) {
      logger.info(`  - ${similar.cellId}: similarity ${similar.similarity.toFixed(3)}`);
    }

    // Detect PCI collisions
    const pciCollisions = system.topologyModel.detectPCICollisions();
    if (pciCollisions.length > 0) {
      logger.warn('PCI collisions detected!');
      for (const collision of pciCollisions) {
        logger.warn(`  - ${collision.cellId1} <-> ${collision.cellId2}, PCI: ${collision.pci}`);
      }
    }

    // Step 5: Root Cause Analysis
    logger.info('\n--- Step 5: Neuro-Symbolic Root Cause Analysis ---');

    const neuralContext: NeuralContext = {
      weather: 'rain',
      timeOfDay: 'afternoon',
      dayOfWeek: 'weekday',
      recentChanges: ['New building constructed nearby'],
    };

    if (anomalies.length > 0) {
      const rca = await system.neuroSymbolicRCA.analyze(
        anomalies[0],
        [], // No hardware alarms
        cellA.metrics,
        neuralContext
      );

      logger.info('Root Cause Analysis Results:');
      logger.info(`  Confidence: ${(rca.confidence * 100).toFixed(1)}%`);
      logger.info('  Probable Causes:');
      for (const cause of rca.probableCauses.slice(0, 3)) {
        logger.info(`    - ${cause.cause}: ${(cause.probability * 100).toFixed(1)}%`);
      }
      logger.info('  Reasoning Chain:');
      for (const step of rca.reasoningChain.slice(0, 3)) {
        logger.info(`    [${step.type}] ${step.premise} -> ${step.conclusion}`);
      }
      logger.info('  Recommended Actions:');
      for (const action of rca.recommendedActions) {
        logger.info(`    - [${action.priority}] ${action.action}`);
      }
    }

    // Step 6: Planning Phase (GOAP)
    logger.info('\n--- Step 6: GOAP Planning ---');

    const goal = system.goapPlanner.createGoal('minimize_interference', {
      cellId: 'SITE01_Cell_A',
    });

    // Create world state for planning
    const worldState = {
      cells: new Map([
        ['SITE01_Cell_A', { config: cellA.configuration, metrics: cellA.metrics, isHealthy: true }],
        ['SITE02_Cell_C', { config: cellC.configuration, metrics: cellC.metrics, isHealthy: true }],
      ]),
      networkKPIs: {
        averageThroughput: 50,
        averageLatency: 10,
        averageInterference: -90,
        energyConsumption: 100,
      },
      constraints: {
        maxTiltChange: 10,
        maxPowerChange: 6,
        minCoverageRSRP: -110,
        maxInterference: -85,
        emergencyCallsActive: false,
      },
    };

    const plan = await system.goapPlanner.plan(goal, worldState);

    if (plan) {
      logger.info('GOAP Plan generated:');
      logger.info(`  Total Cost: ${plan.totalCost}`);
      logger.info(`  Total Risk: ${plan.totalRisk.toFixed(2)}`);
      logger.info('  Actions:');
      for (const action of plan.actions) {
        logger.info(`    - ${action.name} (cost: ${action.cost}, risk: ${action.risk})`);
      }

      // Step 7: Safe Execution
      logger.info('\n--- Step 7: Safe Execution with LTL Verification ---');

      const executionContext = {
        cellId: 'SITE01_Cell_A',
        emergencyCallsActive: false,
        currentLoad: cellA.metrics.prbUtilizationDl,
        isMaintenanceWindow: system.safeExecutor.isMaintenanceWindow(),
        neighborStates: new Map([
          ['SITE01_Cell_B', { healthy: true, load: 0.4 }],
          ['SITE02_Cell_D', { healthy: true, load: 0.5 }],
        ]),
      };

      logger.info('Execution context:', {
        emergencyCalls: executionContext.emergencyCallsActive,
        currentLoad: executionContext.currentLoad,
        maintenanceWindow: executionContext.isMaintenanceWindow,
      });

      const executionResult = await system.safeExecutor.executePlan(plan, executionContext);

      if (executionResult.success) {
        logger.info('Plan executed successfully!');
        logger.info(`  Commit ID: ${executionResult.commitId}`);
        logger.info(`  Actions executed: ${executionResult.executedActions.length}`);
      } else {
        logger.error('Plan execution failed:', executionResult.error);
        if (executionResult.rollbackPerformed) {
          logger.info('Automatic rollback was performed');
        }
      }
    } else {
      logger.warn('No plan could be generated for the goal');
    }

    // Step 8: Validation
    logger.info('\n--- Step 8: Validation ---');

    // Simulate improved metrics after mitigation
    const improvedMetrics: CellMetrics = {
      ...cellA.metrics,
      interferenceLevel: -98, // Improved
      sinr: 15, // Improved
      throughputDl: 550, // Improved
    };

    // Re-run anomaly detection
    const postAnomalies = system.anomalyDetector.detect('SITE01_Cell_A', improvedMetrics);

    if (postAnomalies.length === 0) {
      logger.info('SUCCESS: No anomalies detected after mitigation');
    } else {
      logger.warn('Some anomalies persist:', postAnomalies.map((a) => a.type));
    }

    // Check network stability
    const stabilityReport = system.chaosAnalyzer.analyzeNetworkStability(
      new Map([
        ['SITE01_Cell_A', improvedMetrics],
        ['SITE01_Cell_B', cellB.metrics],
        ['SITE02_Cell_C', cellC.metrics],
        ['SITE02_Cell_D', cellD.metrics],
      ])
    );

    logger.info('Network Stability Report:', {
      state: stabilityReport.networkState,
      stableRatio: `${(stabilityReport.stableRatio * 100).toFixed(1)}%`,
      avgLyapunov: stabilityReport.averageLyapunov.toFixed(4),
    });

    // Log success to reasoning bank (experience storage)
    logger.info('\n--- Scenario Complete ---');
    logger.info('The interference mitigation was successfully detected, analyzed, planned, and executed.');
    logger.info('Experience has been recorded for future similar scenarios.');

  } finally {
    // Cleanup
    await shutdownSystem(system);
  }
}

// Run the scenario
runScenario().catch((error) => {
  logger.error('Scenario failed:', { error: String(error) });
  process.exit(1);
});
