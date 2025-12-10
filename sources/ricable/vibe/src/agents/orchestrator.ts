/**
 * Multi-Agent Orchestrator for RAN Network Analysis
 * Uses agentic-flow for coordinating analysis agents
 */

import type {
  CellKPISnapshot,
  NeighborRelation,
  DetectedAnomaly,
  RootCauseAnalysis,
  KPITimeSeries,
} from '../models/ran-kpi.js';
import { UnifiedAnomalyDetector } from '../analysis/anomaly-detection.js';
import { CellHealthClassifier, AnomalyClassifier, IssuePatternClassifier } from '../analysis/classifier.js';
import { RootCauseAnalyzer } from '../analysis/root-cause.js';
import { CellGraphBuilder, CellGNN, SINRNeighborAnalyzer } from '../gnn/cell-graph.js';
import { GNNPowerControlOptimizer, PowerControlValidator, type PowerControlOptimizationResult } from '../gnn/uplink-power-control.js';

// ============================================================================
// AGENT TYPES
// ============================================================================

export type AgentType =
  | 'anomaly-detector'
  | 'classifier'
  | 'root-cause-analyzer'
  | 'gnn-analyzer'
  | 'power-optimizer'
  | 'report-generator';

export interface AgentTask {
  id: string;
  type: AgentType;
  priority: 'high' | 'medium' | 'low';
  input: unknown;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: unknown;
  error?: string;
  startTime?: Date;
  endTime?: Date;
}

export interface AnalysisRequest {
  cellSnapshots: Map<string, CellKPISnapshot>;
  timeSeriesData: Map<string, KPITimeSeries[]>;
  neighborRelations: NeighborRelation[];
  analysisScope: {
    detectAnomalies: boolean;
    classifyCells: boolean;
    analyzeRootCause: boolean;
    optimizePowerControl: boolean;
    generateReport: boolean;
  };
}

export interface AnalysisResult {
  anomalies: DetectedAnomaly[];
  cellHealthStatus: Map<string, ReturnType<CellHealthClassifier['classifyCell']>>;
  rootCauseAnalysis?: RootCauseAnalysis;
  powerControlRecommendations?: Map<string, PowerControlOptimizationResult>;
  gnnInsights: {
    anomalousCells: string[];
    sinrRecommendations: Array<{
      sourceCellId: string;
      targetCellId: string;
      recommendation: string;
    }>;
  };
  summary: {
    totalCells: number;
    healthyCells: number;
    degradedCells: number;
    criticalCells: number;
    anomalyCount: number;
    topIssues: string[];
  };
  analysisTimestamp: Date;
}

// ============================================================================
// AGENT IMPLEMENTATIONS
// ============================================================================

class AnomalyDetectorAgent {
  private detector: UnifiedAnomalyDetector;

  constructor() {
    this.detector = new UnifiedAnomalyDetector();
  }

  async execute(timeSeriesData: Map<string, KPITimeSeries[]>): Promise<DetectedAnomaly[]> {
    const allAnomalies: DetectedAnomaly[] = [];

    for (const [cellId, seriesList] of timeSeriesData) {
      for (const series of seriesList) {
        const anomalies = this.detector.detectAllAnomalies(series);
        allAnomalies.push(...anomalies);
      }
    }

    return allAnomalies;
  }
}

class ClassifierAgent {
  private cellHealthClassifier: CellHealthClassifier;
  private anomalyClassifier: AnomalyClassifier;
  private issuePatternClassifier: IssuePatternClassifier;

  constructor() {
    this.cellHealthClassifier = new CellHealthClassifier();
    this.anomalyClassifier = new AnomalyClassifier();
    this.issuePatternClassifier = new IssuePatternClassifier();
  }

  async classifyCells(
    cellSnapshots: Map<string, CellKPISnapshot>
  ): Promise<Map<string, ReturnType<CellHealthClassifier['classifyCell']>>> {
    const results = new Map<string, ReturnType<CellHealthClassifier['classifyCell']>>();

    for (const [cellId, snapshot] of cellSnapshots) {
      results.set(cellId, this.cellHealthClassifier.classifyCell(snapshot));
    }

    return results;
  }

  async classifyAnomalies(
    anomalies: DetectedAnomaly[],
    cellSnapshots: Map<string, CellKPISnapshot>
  ): Promise<Array<ReturnType<AnomalyClassifier['classifyAnomaly']>>> {
    return anomalies.map(anomaly => {
      const cellSnapshot = cellSnapshots.get(anomaly.cellId);
      return this.anomalyClassifier.classifyAnomaly(anomaly, anomalies, cellSnapshot);
    });
  }

  async identifyIssuePatterns(
    cellSnapshots: Map<string, CellKPISnapshot>,
    anomalies: DetectedAnomaly[]
  ): Promise<Map<string, ReturnType<IssuePatternClassifier['classifyPattern']>>> {
    const results = new Map<string, ReturnType<IssuePatternClassifier['classifyPattern']>>();

    for (const [cellId, snapshot] of cellSnapshots) {
      const cellAnomalies = anomalies.filter(a => a.cellId === cellId);
      results.set(cellId, this.issuePatternClassifier.classifyPattern(snapshot, cellAnomalies));
    }

    return results;
  }
}

class RootCauseAgent {
  private analyzer: RootCauseAnalyzer;

  constructor() {
    this.analyzer = new RootCauseAnalyzer();
  }

  async execute(
    anomalies: DetectedAnomaly[],
    cellSnapshots: Map<string, CellKPISnapshot>,
    neighborRelations: NeighborRelation[]
  ): Promise<RootCauseAnalysis | undefined> {
    if (anomalies.length === 0) {
      return undefined;
    }

    return this.analyzer.analyze({
      anomalies,
      cellSnapshots,
      neighborRelations,
    });
  }
}

class GNNAnalyzerAgent {
  private graphBuilder: CellGraphBuilder;
  private gnn: CellGNN;
  private sinrAnalyzer: SINRNeighborAnalyzer;

  constructor() {
    this.graphBuilder = new CellGraphBuilder();
    this.gnn = new CellGNN();
    this.sinrAnalyzer = new SINRNeighborAnalyzer();
  }

  async execute(
    cellSnapshots: Map<string, CellKPISnapshot>,
    neighborRelations: NeighborRelation[]
  ): Promise<{
    anomalousCells: string[];
    sinrRecommendations: Array<{
      sourceCellId: string;
      targetCellId: string;
      recommendation: string;
    }>;
    embeddings: Map<string, number[]>;
  }> {
    const graph = this.graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    // Get GNN-based anomaly detection
    const anomalousCells = this.gnn.detectAnomalousCells(graph);

    // Get SINR-based recommendations
    const sinrAnalysis = this.sinrAnalyzer.analyzeSINRRelationships(graph);
    const sinrRecommendations = sinrAnalysis.map(r => ({
      sourceCellId: r.sourceCellId,
      targetCellId: r.targetCellId,
      recommendation: r.recommendation,
    }));

    // Get embeddings for downstream use
    const embeddings = this.gnn.getEmbeddings(graph);

    return {
      anomalousCells,
      sinrRecommendations,
      embeddings,
    };
  }
}

class PowerOptimizerAgent {
  private optimizer: GNNPowerControlOptimizer;
  private validator: PowerControlValidator;

  constructor() {
    this.optimizer = new GNNPowerControlOptimizer();
    this.validator = new PowerControlValidator();
  }

  async execute(
    cellSnapshots: Map<string, CellKPISnapshot>,
    neighborRelations: NeighborRelation[]
  ): Promise<Map<string, PowerControlOptimizationResult>> {
    const recommendations = this.optimizer.optimizeNetworkPowerControl(
      cellSnapshots,
      neighborRelations
    );

    // Validate all recommendations
    for (const [cellId, recommendation] of recommendations) {
      const snapshot = cellSnapshots.get(cellId);
      if (snapshot) {
        const validation = this.validator.validateChanges(
          { p0: recommendation.currentP0, alpha: recommendation.currentAlpha },
          { p0: recommendation.recommendedP0, alpha: recommendation.recommendedAlpha },
          snapshot
        );

        if (validation.warnings.length > 0) {
          recommendation.rationale += ` Warnings: ${validation.warnings.join('; ')}`;
        }

        if (!validation.isValid) {
          recommendation.confidence *= 0.5; // Reduce confidence for risky changes
          recommendation.rationale += ` Risks: ${validation.risks.join('; ')}`;
        }
      }
    }

    return recommendations;
  }
}

// ============================================================================
// ORCHESTRATOR
// ============================================================================

export class RANAnalysisOrchestrator {
  private anomalyAgent: AnomalyDetectorAgent;
  private classifierAgent: ClassifierAgent;
  private rootCauseAgent: RootCauseAgent;
  private gnnAgent: GNNAnalyzerAgent;
  private powerAgent: PowerOptimizerAgent;

  constructor() {
    this.anomalyAgent = new AnomalyDetectorAgent();
    this.classifierAgent = new ClassifierAgent();
    this.rootCauseAgent = new RootCauseAgent();
    this.gnnAgent = new GNNAnalyzerAgent();
    this.powerAgent = new PowerOptimizerAgent();
  }

  /**
   * Execute comprehensive RAN analysis
   */
  async analyze(request: AnalysisRequest): Promise<AnalysisResult> {
    const {
      cellSnapshots,
      timeSeriesData,
      neighborRelations,
      analysisScope,
    } = request;

    console.log(`Starting RAN analysis for ${cellSnapshots.size} cells...`);

    // Phase 1: Anomaly Detection (parallel with GNN analysis)
    const [anomalies, gnnInsights] = await Promise.all([
      analysisScope.detectAnomalies
        ? this.anomalyAgent.execute(timeSeriesData)
        : Promise.resolve([]),
      this.gnnAgent.execute(cellSnapshots, neighborRelations),
    ]);

    console.log(`Detected ${anomalies.length} anomalies`);

    // Phase 2: Classification
    const cellHealthStatus = analysisScope.classifyCells
      ? await this.classifierAgent.classifyCells(cellSnapshots)
      : new Map();

    // Phase 3: Root Cause Analysis (if anomalies found)
    const rootCauseAnalysis = analysisScope.analyzeRootCause && anomalies.length > 0
      ? await this.rootCauseAgent.execute(anomalies, cellSnapshots, neighborRelations)
      : undefined;

    // Phase 4: Power Control Optimization
    const powerControlRecommendations = analysisScope.optimizePowerControl
      ? await this.powerAgent.execute(cellSnapshots, neighborRelations)
      : undefined;

    // Generate summary
    const summary = this.generateSummary(cellHealthStatus, anomalies, gnnInsights);

    return {
      anomalies,
      cellHealthStatus,
      rootCauseAnalysis,
      powerControlRecommendations,
      gnnInsights: {
        anomalousCells: gnnInsights.anomalousCells,
        sinrRecommendations: gnnInsights.sinrRecommendations,
      },
      summary,
      analysisTimestamp: new Date(),
    };
  }

  private generateSummary(
    cellHealthStatus: Map<string, ReturnType<CellHealthClassifier['classifyCell']>>,
    anomalies: DetectedAnomaly[],
    gnnInsights: Awaited<ReturnType<GNNAnalyzerAgent['execute']>>
  ): AnalysisResult['summary'] {
    let healthyCells = 0;
    let degradedCells = 0;
    let criticalCells = 0;
    const issues: string[] = [];

    for (const [cellId, health] of cellHealthStatus) {
      switch (health.overallHealth) {
        case 'healthy':
          healthyCells++;
          break;
        case 'degraded':
          degradedCells++;
          issues.push(...health.issues.map(i => `${cellId}: ${i}`));
          break;
        case 'critical':
        case 'failed':
          criticalCells++;
          issues.push(...health.issues.map(i => `${cellId}: ${i}`));
          break;
      }
    }

    // Add GNN-detected anomalies
    for (const cellId of gnnInsights.anomalousCells) {
      if (!issues.some(i => i.startsWith(cellId))) {
        issues.push(`${cellId}: GNN-detected anomalous behavior`);
      }
    }

    // Deduplicate and limit issues
    const uniqueIssues = [...new Set(issues)].slice(0, 10);

    return {
      totalCells: cellHealthStatus.size,
      healthyCells,
      degradedCells,
      criticalCells,
      anomalyCount: anomalies.length,
      topIssues: uniqueIssues,
    };
  }
}

// ============================================================================
// REPORT GENERATOR
// ============================================================================

export class AnalysisReportGenerator {
  generateTextReport(result: AnalysisResult): string {
    const lines: string[] = [];

    lines.push('=' .repeat(80));
    lines.push('RAN NETWORK ANALYSIS REPORT');
    lines.push(`Generated: ${result.analysisTimestamp.toISOString()}`);
    lines.push('='.repeat(80));

    // Summary section
    lines.push('\n## SUMMARY\n');
    lines.push(`Total Cells Analyzed: ${result.summary.totalCells}`);
    lines.push(`  - Healthy: ${result.summary.healthyCells}`);
    lines.push(`  - Degraded: ${result.summary.degradedCells}`);
    lines.push(`  - Critical: ${result.summary.criticalCells}`);
    lines.push(`Total Anomalies Detected: ${result.summary.anomalyCount}`);

    // Top issues
    if (result.summary.topIssues.length > 0) {
      lines.push('\n## TOP ISSUES\n');
      result.summary.topIssues.forEach((issue, idx) => {
        lines.push(`${idx + 1}. ${issue}`);
      });
    }

    // GNN Insights
    if (result.gnnInsights.anomalousCells.length > 0) {
      lines.push('\n## GNN-DETECTED ANOMALOUS CELLS\n');
      lines.push(result.gnnInsights.anomalousCells.join(', '));
    }

    if (result.gnnInsights.sinrRecommendations.length > 0) {
      lines.push('\n## SINR-BASED RECOMMENDATIONS\n');
      result.gnnInsights.sinrRecommendations.slice(0, 5).forEach(rec => {
        lines.push(`  ${rec.sourceCellId} -> ${rec.targetCellId}: ${rec.recommendation}`);
      });
    }

    // Root Cause Analysis
    if (result.rootCauseAnalysis) {
      lines.push('\n## ROOT CAUSE ANALYSIS\n');
      lines.push(`Primary Cause: ${result.rootCauseAnalysis.primaryCause}`);
      lines.push(`Confidence: ${(result.rootCauseAnalysis.primaryCauseConfidence * 100).toFixed(0)}%`);

      if (result.rootCauseAnalysis.recommendations.length > 0) {
        lines.push('\nRecommendations:');
        result.rootCauseAnalysis.recommendations.forEach(rec => {
          lines.push(`  [${rec.priority.toUpperCase()}] ${rec.action}`);
          lines.push(`    Expected Impact: ${rec.expectedImpact}`);
        });
      }
    }

    // Power Control Recommendations
    if (result.powerControlRecommendations && result.powerControlRecommendations.size > 0) {
      lines.push('\n## POWER CONTROL RECOMMENDATIONS\n');

      // Only show cells with recommendations different from current
      let shown = 0;
      for (const [cellId, rec] of result.powerControlRecommendations) {
        if (rec.recommendedP0 !== rec.currentP0 || rec.recommendedAlpha !== rec.currentAlpha) {
          if (shown < 10) {
            lines.push(`\n${cellId}:`);
            lines.push(`  P0: ${rec.currentP0} dBm -> ${rec.recommendedP0} dBm`);
            lines.push(`  Alpha: ${rec.currentAlpha} -> ${rec.recommendedAlpha}`);
            lines.push(`  Confidence: ${(rec.confidence * 100).toFixed(0)}%`);
            lines.push(`  Rationale: ${rec.rationale}`);
            shown++;
          }
        }
      }

      if (shown === 0) {
        lines.push('No power control changes recommended.');
      }
    }

    lines.push('\n' + '='.repeat(80));
    lines.push('END OF REPORT');
    lines.push('='.repeat(80));

    return lines.join('\n');
  }

  generateJSONReport(result: AnalysisResult): string {
    return JSON.stringify({
      timestamp: result.analysisTimestamp.toISOString(),
      summary: result.summary,
      anomalies: result.anomalies.map(a => ({
        ...a,
        timestamp: a.timestamp.toISOString(),
      })),
      gnnInsights: result.gnnInsights,
      rootCauseAnalysis: result.rootCauseAnalysis
        ? {
            ...result.rootCauseAnalysis,
            analysisTimestamp: result.rootCauseAnalysis.analysisTimestamp.toISOString(),
          }
        : null,
      powerControlRecommendations: result.powerControlRecommendations
        ? Object.fromEntries(result.powerControlRecommendations)
        : null,
    }, null, 2);
  }
}

export default {
  RANAnalysisOrchestrator,
  AnalysisReportGenerator,
};
