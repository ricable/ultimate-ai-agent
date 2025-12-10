/**
 * RAN Network Analysis System
 * AI/ML-powered Radio Access Network KPI analysis, anomaly detection,
 * and power control optimization using Graph Neural Networks
 *
 * @module ran-network-analysis
 */

// Internal imports for local use
import { RANAnalysisOrchestrator as RANOrchestrator } from './agents/orchestrator.js';
import type { AnalysisResult, AnalysisRequest } from './agents/orchestrator.js';
import type { CellKPISnapshot, KPITimeSeries, NeighborRelation } from './models/ran-kpi.js';

// Models
export * from './models/ran-kpi.js';

// Analysis modules
export { default as timeSeriesAnalysis } from './analysis/time-series.js';
export { default as anomalyDetection } from './analysis/anomaly-detection.js';
export { default as classification } from './analysis/classifier.js';
export { default as rootCauseAnalysis } from './analysis/root-cause.js';

// GNN modules
export { default as cellGraph } from './gnn/cell-graph.js';
export { default as uplinkPowerControl } from './gnn/uplink-power-control.js';
export { default as interferenceOptimizer } from './gnn/interference-optimizer.js';
export { default as networkSurrogateModel } from './gnn/network-surrogate-model.js';

// Agent orchestration
export { default as orchestrator } from './agents/orchestrator.js';

// Re-export main classes for convenience
export {
  computeTimeSeriesStats,
  analyzeTrend,
  analyzeSeasonality,
  decomposeTimeSeries,
  forecast,
} from './analysis/time-series.js';

export {
  UnifiedAnomalyDetector,
  StatisticalAnomalyDetector,
  TrendAnomalyDetector,
  SeasonalAnomalyDetector,
  CollectiveAnomalyDetector,
  DomainAnomalyDetector,
} from './analysis/anomaly-detection.js';

export {
  CellHealthClassifier,
  AnomalyClassifier,
  IssuePatternClassifier,
} from './analysis/classifier.js';

export {
  RootCauseAnalyzer,
  MultiCellCorrelationAnalyzer,
} from './analysis/root-cause.js';

export {
  CellGraphBuilder,
  GNNLayer,
  CellGNN,
  SINRNeighborAnalyzer,
} from './gnn/cell-graph.js';

export {
  PathLossAnalyzer,
  FractionalPathLossOptimizer,
  GNNPowerControlOptimizer,
  PowerControlValidator,
} from './gnn/uplink-power-control.js';

export {
  SINRPredictionGNN,
  IssueCellDetector,
  GeneticOptimizer,
  InterferenceOptimizationLoop,
  DEFAULT_OPTIMIZER_CONFIG,
} from './gnn/interference-optimizer.js';
export type {
  InterferenceOptimizerConfig,
  IssueCell,
  NetworkOptimizationResult,
} from './gnn/interference-optimizer.js';

// Network Surrogate Model (Digital Twin)
export {
  SurrogateGraphBuilder,
  GNNSurrogateModel,
  IssueCellDetector as SurrogateIssueCellDetector,
  SurrogateOptimizer,
  SurrogateVisualizer,
  DEFAULT_SURROGATE_CONFIG,
} from './gnn/network-surrogate-model.js';
export type {
  SurrogateModelConfig,
  PowerControlParams,
  CellStatus,
  CellOptimizationResult,
  NetworkOptimizationResult as SurrogateNetworkOptimizationResult,
  TrainingSample,
  SurrogateGraph,
} from './gnn/network-surrogate-model.js';

export {
  RANAnalysisOrchestrator,
  AnalysisReportGenerator,
} from './agents/orchestrator.js';

/**
 * Quick start function for running analysis
 */
export async function analyzeNetwork(options: {
  cellSnapshots: Map<string, CellKPISnapshot>;
  timeSeriesData: Map<string, KPITimeSeries[]>;
  neighborRelations: NeighborRelation[];
}): Promise<AnalysisResult> {
  const analysisOrchestrator = new RANOrchestrator();

  return analysisOrchestrator.analyze({
    ...options,
    analysisScope: {
      detectAnomalies: true,
      classifyCells: true,
      analyzeRootCause: true,
      optimizePowerControl: true,
      generateReport: true,
    },
  });
}

// CLI entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  console.log('RAN Network Analysis System v1.0.0');
  console.log('');
  console.log('Usage:');
  console.log('  npm run dev           - Run development server');
  console.log('  npm run analyze       - Run analysis on sample data');
  console.log('  npm run gnn:train     - Train GNN model');
  console.log('  npm run agents:start  - Start agent orchestrator');
  console.log('');
  console.log('For programmatic use:');
  console.log('  import { analyzeNetwork } from "ran-network-analysis"');
}
