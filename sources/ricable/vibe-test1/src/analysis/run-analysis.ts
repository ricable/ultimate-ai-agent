#!/usr/bin/env tsx
/**
 * Run RAN Network Analysis
 * Executes comprehensive analysis on sample or provided data
 */

import { SampleDataGenerator } from '../data/sample-generator.js';
import { RANAnalysisOrchestrator, AnalysisReportGenerator } from '../agents/orchestrator.js';

async function main() {
  console.log('='.repeat(80));
  console.log('RAN Network Analysis System');
  console.log('AI/ML Radio Access Network KPI Analysis');
  console.log('='.repeat(80));
  console.log('');

  // Generate sample data
  console.log('Generating sample network data...');
  const generator = new SampleDataGenerator({
    numCells: 30,
    technology: 'LTE',
    healthDistribution: {
      healthy: 0.6,
      degraded: 0.3,
      critical: 0.1,
    },
    includeAnomalies: true,
    anomalyRate: 0.2,
  });

  const { cellSnapshots, timeSeriesData, neighborRelations } = generator.generateDataset();
  console.log(`Generated ${cellSnapshots.size} cells with ${neighborRelations.length} neighbor relations`);
  console.log('');

  // Run analysis
  console.log('Running comprehensive analysis...');
  console.log('');

  const orchestrator = new RANAnalysisOrchestrator();
  const result = await orchestrator.analyze({
    cellSnapshots,
    timeSeriesData,
    neighborRelations,
    analysisScope: {
      detectAnomalies: true,
      classifyCells: true,
      analyzeRootCause: true,
      optimizePowerControl: true,
      generateReport: true,
    },
  });

  // Generate report
  const reportGenerator = new AnalysisReportGenerator();
  const report = reportGenerator.generateTextReport(result);

  console.log(report);

  // Also output JSON summary
  console.log('\n\nJSON Summary:');
  console.log(JSON.stringify({
    timestamp: result.analysisTimestamp.toISOString(),
    summary: result.summary,
    gnnAnomalies: result.gnnInsights.anomalousCells.length,
    sinrRecommendations: result.gnnInsights.sinrRecommendations.length,
    powerControlChanges: result.powerControlRecommendations
      ? Array.from(result.powerControlRecommendations.values()).filter(
          r => r.optimizedParams.p0 !== r.originalParams.p0 || r.optimizedParams.alpha !== r.originalParams.alpha
        ).length
      : 0,
  }, null, 2));
}

main().catch(console.error);
