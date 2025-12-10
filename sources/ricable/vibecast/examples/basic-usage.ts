// Example: Basic usage of the Ericsson RAN Time Series Analysis Platform

import EricssonRanPlatform from '../src/index.js';
import type { RanNode, KpiMeasurement, Alarm } from '../src/types/ran-models.js';

async function main() {
  // Initialize the platform
  const platform = new EricssonRanPlatform();

  console.log('Platform Status:', JSON.stringify(platform.getStatus(), null, 2));

  // Example RAN node
  const exampleNode: RanNode = {
    nodeId: 'gNB-001',
    nodeType: 'gNB',
    location: {
      latitude: 59.3293,
      longitude: 18.0686,
      altitude: 50,
    },
    cells: [
      {
        cellId: 'CELL-001',
        sectorId: 'SECTOR-1',
        pci: 100,
        frequencyBand: 'n78',
        bandwidthMhz: 100,
        maxPowerDbm: 46,
        azimuth: 0,
        tilt: 5,
      },
      {
        cellId: 'CELL-002',
        sectorId: 'SECTOR-2',
        pci: 101,
        frequencyBand: 'n78',
        bandwidthMhz: 100,
        maxPowerDbm: 46,
        azimuth: 120,
        tilt: 5,
      },
      {
        cellId: 'CELL-003',
        sectorId: 'SECTOR-3',
        pci: 102,
        frequencyBand: 'n78',
        bandwidthMhz: 100,
        maxPowerDbm: 46,
        azimuth: 240,
        tilt: 5,
      },
    ],
    parameters: {
      maxTxPower: 46,
      ulFrequency: 3500,
      dlFrequency: 3700,
      cellBarred: false,
      administrativeState: 'unlocked',
    },
  };

  // Generate example KPI measurements (hourly data for 7 days)
  const kpiMeasurements: KpiMeasurement[] = [];
  const startTime = Math.floor(Date.now() / 1000) - 7 * 24 * 3600; // 7 days ago

  for (let hour = 0; hour < 7 * 24; hour++) {
    const timestamp = startTime + hour * 3600;

    // Simulate KPIs with some patterns
    const hourOfDay = hour % 24;
    const baseLoad = 50 + 30 * Math.sin((hourOfDay - 6) * Math.PI / 12); // Peak during day
    const randomness = (Math.random() - 0.5) * 10;

    kpiMeasurements.push(
      {
        timestamp,
        nodeId: 'gNB-001',
        cellId: 'CELL-001',
        kpiName: 'dl_throughput_mbps',
        value: Math.max(0, baseLoad + randomness),
        unit: 'Mbps',
        granularity: 'Hourly',
      },
      {
        timestamp,
        nodeId: 'gNB-001',
        cellId: 'CELL-001',
        kpiName: 'ul_throughput_mbps',
        value: Math.max(0, (baseLoad + randomness) * 0.3),
        unit: 'Mbps',
        granularity: 'Hourly',
      },
      {
        timestamp,
        nodeId: 'gNB-001',
        cellId: 'CELL-001',
        kpiName: 'prb_utilization_pct',
        value: Math.min(100, Math.max(0, baseLoad + randomness * 1.5)),
        unit: '%',
        granularity: 'Hourly',
      },
      {
        timestamp,
        nodeId: 'gNB-001',
        cellId: 'CELL-001',
        kpiName: 'active_users',
        value: Math.max(0, Math.floor(baseLoad / 2 + randomness)),
        unit: 'count',
        granularity: 'Hourly',
      }
    );
  }

  console.log(`\nGenerated ${kpiMeasurements.length} KPI measurements`);

  // Example alarms
  const exampleAlarms: Alarm[] = [
    {
      alarmId: 'ALM-001',
      timestamp: Math.floor(Date.now() / 1000) - 3600,
      severity: 'Major',
      nodeId: 'gNB-001',
      cellId: 'CELL-001',
      alarmType: 'PRB_UTILIZATION_HIGH',
      description: 'PRB utilization exceeds 80% threshold',
      additionalInfo: {
        currentUtilization: '85',
        threshold: '80',
      },
    },
    {
      alarmId: 'ALM-002',
      timestamp: Math.floor(Date.now() / 1000) - 1800,
      severity: 'Minor',
      nodeId: 'gNB-001',
      cellId: 'CELL-002',
      alarmType: 'RACH_FAILURES_HIGH',
      description: 'Random Access failures above normal',
      additionalInfo: {
        failureRate: '5.2',
        normalRate: '2.0',
      },
    },
  ];

  // Perform comprehensive network analysis
  console.log('\n=== Starting Comprehensive Network Analysis ===\n');

  try {
    const analysisResult = await platform.analyzeNetwork({
      node: exampleNode,
      kpiMeasurements,
      alarms: exampleAlarms,
      optimizationGoals: [
        'Improve throughput',
        'Reduce PRB utilization',
        'Minimize alarm occurrences',
      ],
    });

    console.log('\n=== Analysis Results ===\n');
    console.log('Success:', analysisResult.success);
    console.log('\nAgent Analysis Summary:');
    console.log('- Insights:', analysisResult.agentAnalysis.aggregatedInsights.length);
    console.log('- Recommendations:', analysisResult.agentAnalysis.recommendations.length);

    console.log('\nTop Recommendations:');
    analysisResult.agentAnalysis.recommendations.slice(0, 5).forEach((rec, idx) => {
      console.log(`  ${idx + 1}. ${rec}`);
    });

    if (analysisResult.forecast) {
      console.log('\nForecast Results:');
      console.log(`- Forecast horizon: ${analysisResult.forecast.results.length} hours`);
      console.log(`- Processing time: ${analysisResult.forecast.metadata.processingTimeMs}ms`);
    }

    console.log('\nDetailed results saved to analysis-results.json');
    // In a real application, you would save the results
    // await fs.promises.writeFile('analysis-results.json', JSON.stringify(analysisResult, null, 2));

  } catch (error) {
    console.error('Analysis failed:', error);
  }

  // Example: Query RAN documentation
  console.log('\n=== Querying RAN Documentation ===\n');

  try {
    const docsResult = await platform.queryDocumentation(
      'What are the recommended settings for PRB utilization thresholds in 5G NR?'
    );

    console.log('Documentation Query Results:');
    console.log(docsResult.aggregatedInsights.join('\n'));
  } catch (error) {
    console.error('Documentation query failed:', error);
  }

  console.log('\n=== Example Complete ===\n');
}

// Run the example
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export default main;
