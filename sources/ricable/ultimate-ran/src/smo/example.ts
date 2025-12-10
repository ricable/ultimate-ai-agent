/**
 * SMO Manager Usage Example
 *
 * Demonstrates how to use the PM/FM data pipeline for Ericsson SMO integration
 *
 * @module smo/example
 */

import { SMOManager } from './index';

/**
 * Example: Basic SMO Manager usage
 */
async function basicExample() {
  console.log('\n========================================');
  console.log('Example 1: Basic SMO Manager');
  console.log('========================================\n');

  // Create SMO Manager with default configuration
  const smo = new SMOManager({
    pm: {
      ropInterval: 60000,  // 1-minute ROP for demo (normally 10 minutes)
      cells: ['CELL-001', 'CELL-002', 'CELL-003'],
      enableStreaming: true,
      storageEnabled: true
    },
    fm: {
      pollingInterval: 10000,  // 10 seconds
      enableSSE: true,
      enableAutoHealing: true,
      correlationWindow: 300000  // 5 minutes
    },
    enableCrossCorrelation: true
  });

  // Setup event listeners
  smo.on('pm_anomaly', (anomaly) => {
    console.log(`⚠️  PM Anomaly Detected: ${anomaly.type} in ${anomaly.cellId}`);
    console.log(`   Metric: ${anomaly.metric} = ${anomaly.value.toFixed(2)} (threshold: ${anomaly.threshold})`);
  });

  smo.on('fm_alarm', (alarm) => {
    console.log(`🚨 FM Alarm: [${alarm.severity.toUpperCase()}] ${alarm.specificProblem}`);
    console.log(`   Cell: ${alarm.managedObject}`);
    console.log(`   Cause: ${alarm.probableCause}`);
  });

  smo.on('pmfm_correlation', (correlation) => {
    console.log(`🔗 PM-FM Correlation Detected!`);
    console.log(`   Likely Cause: ${correlation.likelyCause}`);
    console.log(`   Correlation Score: ${correlation.correlationScore.toFixed(2)}`);
    console.log(`   Related Alarms: ${correlation.relatedAlarms.length}`);
  });

  smo.on('self_healing', (action) => {
    console.log(`🔧 Self-Healing Action: ${action.actionType} - ${action.status}`);
    if (action.result?.success) {
      console.log(`   Result: ${action.result.details}`);
      if (action.result.pmDelta) {
        console.log(`   PM Delta:`, action.result.pmDelta);
      }
    }
  });

  // Start SMO Manager
  await smo.start();

  // Let it run for 2 minutes
  console.log('\n📊 SMO Manager running... (will stop after 2 minutes)\n');

  await new Promise(resolve => setTimeout(resolve, 120000));

  // Get final statistics
  const stats = smo.getStats();
  console.log('\n========================================');
  console.log('Final Statistics:');
  console.log('========================================');
  console.log('PM Collector:');
  console.log(`  Total Collections: ${stats.pm.totalCollections}`);
  console.log(`  Configured Cells: ${stats.pm.configuredCells}`);
  console.log(`  Avg SINR: ${stats.pm.aggregated.avgSinr.toFixed(2)} dB`);
  console.log(`  Avg CSSR: ${(stats.pm.aggregated.avgCssr * 100).toFixed(2)}%`);
  console.log(`  Avg Drop Rate: ${(stats.pm.aggregated.avgDropRate * 100).toFixed(2)}%`);
  console.log('\nFM Handler:');
  console.log(`  Total Alarms: ${stats.fm.totalAlarms}`);
  console.log(`  Active Alarms: ${stats.fm.activeAlarms}`);
  console.log(`  Critical: ${stats.fm.alarmsBySeverity.critical}`);
  console.log(`  Major: ${stats.fm.alarmsBySeverity.major}`);
  console.log(`  Minor: ${stats.fm.alarmsBySeverity.minor}`);
  console.log(`  Correlations: ${stats.fm.correlations}`);
  console.log(`  Healing Actions: ${stats.fm.healingActions}`);
  console.log('\nCross-Correlations:');
  console.log(`  PM-FM Correlations: ${stats.correlations.pmfm}`);
  console.log('========================================\n');

  // Stop SMO Manager
  smo.stop();
}

/**
 * Example: PM Collector standalone
 */
async function pmCollectorExample() {
  console.log('\n========================================');
  console.log('Example 2: PM Collector Standalone');
  console.log('========================================\n');

  const { PMCollector } = await import('./pm-collector');

  const pmCollector = new PMCollector({
    ropInterval: 30000,  // 30 seconds for demo
    cells: ['CELL-100', 'CELL-101', 'CELL-102', 'CELL-103'],
    enableStreaming: true
  });

  pmCollector.on('collection_complete', (stats) => {
    console.log(`✅ PM Collection Complete: ${stats.cellCount} cells in ${stats.duration}ms`);
  });

  pmCollector.on('anomaly_detected', (anomaly) => {
    console.log(`⚠️  Anomaly: ${anomaly.type} - ${anomaly.cellId} (${anomaly.metric}: ${anomaly.value})`);
  });

  await pmCollector.start();

  // Run for 1 minute
  await new Promise(resolve => setTimeout(resolve, 60000));

  // Get aggregated statistics
  const aggStats = pmCollector.getAggregatedStats();
  console.log('\n📊 Aggregated Statistics:');
  console.log(`   Avg SINR: ${aggStats.avgSinr.toFixed(2)} dB`);
  console.log(`   Avg CSSR: ${(aggStats.avgCssr * 100).toFixed(2)}%`);
  console.log(`   Avg Drop Rate: ${(aggStats.avgDropRate * 100).toFixed(2)}%`);
  console.log(`   Avg PRB Usage: ${aggStats.avgPrbUsage.toFixed(2)}%`);
  console.log(`   Cell Count: ${aggStats.cellCount}`);

  pmCollector.stop();
}

/**
 * Example: FM Handler standalone
 */
async function fmHandlerExample() {
  console.log('\n========================================');
  console.log('Example 3: FM Handler Standalone');
  console.log('========================================\n');

  const { FMHandler } = await import('./fm-handler');

  const fmHandler = new FMHandler({
    pollingInterval: 15000,  // 15 seconds
    enableSSE: true,
    enableAutoHealing: true,
    correlationWindow: 180000  // 3 minutes
  });

  fmHandler.on('alarm_processed', (alarm) => {
    console.log(`🚨 [${alarm.severity.toUpperCase()}] ${alarm.specificProblem}`);
  });

  fmHandler.on('correlation_detected', (correlation) => {
    console.log(`🔗 Correlation: ${correlation.correlationType}`);
    console.log(`   Root Cause: ${correlation.rootCause.specificProblem}`);
    console.log(`   Symptoms: ${correlation.symptoms.length} alarms`);
    console.log(`   Score: ${correlation.correlationScore.toFixed(2)}`);
  });

  fmHandler.on('self_healing_triggered', (action) => {
    console.log(`🔧 Self-Healing Triggered: ${action.actionType}`);
  });

  fmHandler.on('self_healing_completed', (action) => {
    console.log(`✅ Self-Healing ${action.status}: ${action.actionType}`);
  });

  await fmHandler.start();

  // Run for 1 minute
  await new Promise(resolve => setTimeout(resolve, 60000));

  // Get statistics
  const stats = fmHandler.getStats();
  console.log('\n📊 FM Handler Statistics:');
  console.log(`   Total Alarms: ${stats.totalAlarms}`);
  console.log(`   Active Alarms: ${stats.activeAlarms}`);
  console.log(`   Correlations: ${stats.correlations}`);
  console.log(`   Healing Actions: ${stats.healingActions}`);

  // Show active alarms
  const activeAlarms = fmHandler.getActiveAlarms();
  console.log(`\n🚨 Active Alarms (${activeAlarms.length}):`);
  activeAlarms.forEach(alarm => {
    console.log(`   - [${alarm.severity}] ${alarm.specificProblem} (${alarm.managedObject})`);
  });

  // Show correlations
  const correlations = fmHandler.getCorrelations();
  console.log(`\n🔗 Correlations (${correlations.length}):`);
  correlations.forEach(corr => {
    console.log(`   - ${corr.correlationType}: ${corr.rootCause.specificProblem} + ${corr.symptoms.length} symptoms`);
  });

  fmHandler.stop();
}

/**
 * Main function to run examples
 */
async function main() {
  const exampleToRun = process.argv[2] || '1';

  try {
    switch (exampleToRun) {
      case '1':
        await basicExample();
        break;
      case '2':
        await pmCollectorExample();
        break;
      case '3':
        await fmHandlerExample();
        break;
      case 'all':
        await basicExample();
        await pmCollectorExample();
        await fmHandlerExample();
        break;
      default:
        console.log('Usage: node example.js [1|2|3|all]');
        console.log('  1: Basic SMO Manager (default)');
        console.log('  2: PM Collector standalone');
        console.log('  3: FM Handler standalone');
        console.log('  all: Run all examples');
    }

    console.log('\n✅ Example completed successfully!\n');
    process.exit(0);

  } catch (error) {
    console.error('\n❌ Example failed:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}

export { basicExample, pmCollectorExample, fmHandlerExample };
