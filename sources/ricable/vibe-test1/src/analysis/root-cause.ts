/**
 * Root Cause Analysis Engine for RAN Network Issues
 * Implements correlation analysis, causal inference, and automated diagnostics
 */

import { v4 as uuidv4 } from 'uuid';
import * as ss from 'simple-statistics';
import type {
  DetectedAnomaly,
  RootCauseAnalysis,
  RootCauseCategory,
  CellKPISnapshot,
  NeighborRelation,
  CellGraph,
} from '../models/ran-kpi.js';
import { AnomalyClassifier, type AnomalyClassification } from './classifier.js';
import { crossCorrelation } from './time-series.js';

// ============================================================================
// ROOT CAUSE ANALYSIS ENGINE
// ============================================================================

export interface RCAContext {
  anomalies: DetectedAnomaly[];
  cellSnapshots: Map<string, CellKPISnapshot>;
  neighborRelations: NeighborRelation[];
  historicalBaselines?: Map<string, number[]>;
  cellGraph?: CellGraph;
}

export interface CorrelationResult {
  kpi1: string;
  kpi2: string;
  correlation: number;
  lag: number; // Positive lag means kpi1 leads kpi2
  significance: number;
}

export interface CausalChain {
  rootCause: RootCauseCategory;
  chain: Array<{
    step: number;
    description: string;
    evidence: string[];
    confidence: number;
  }>;
  affectedKpis: string[];
}

export class RootCauseAnalyzer {
  private anomalyClassifier: AnomalyClassifier;

  constructor() {
    this.anomalyClassifier = new AnomalyClassifier();
  }

  /**
   * Perform comprehensive root cause analysis
   */
  async analyze(context: RCAContext): Promise<RootCauseAnalysis> {
    const { anomalies, cellSnapshots, neighborRelations } = context;

    if (anomalies.length === 0) {
      throw new Error('No anomalies provided for analysis');
    }

    // Step 1: Classify individual anomalies
    const classifications = this.classifyAnomalies(anomalies, cellSnapshots);

    // Step 2: Find correlations between anomalies
    const correlations = this.findAnomalyCorrelations(anomalies);

    // Step 3: Identify affected cells and propagation pattern
    const { affectedCells, propagationPattern } = this.analyzePropagation(anomalies, neighborRelations);

    // Step 4: Build causal chains
    const causalChains = this.buildCausalChains(classifications, correlations, cellSnapshots);

    // Step 5: Determine primary cause
    const { primaryCause, confidence } = this.determinePrimaryCause(causalChains, classifications);

    // Step 6: Identify contributing factors
    const contributingFactors = this.identifyContributingFactors(
      classifications,
      correlations,
      cellSnapshots
    );

    // Step 7: Generate recommendations
    const recommendations = this.generateRecommendations(
      primaryCause,
      contributingFactors,
      cellSnapshots,
      anomalies
    );

    // Step 8: Calculate power control adjustments if applicable
    const powerControlAdjustments = this.calculatePowerControlAdjustments(
      primaryCause,
      cellSnapshots,
      anomalies
    );

    return {
      id: uuidv4(),
      anomalyIds: anomalies.map(a => a.id),
      analysisTimestamp: new Date(),
      primaryCause,
      primaryCauseConfidence: confidence,
      contributingFactors,
      affectedCells,
      propagationPattern,
      recommendations,
      ...powerControlAdjustments,
    };
  }

  /**
   * Classify all anomalies
   */
  private classifyAnomalies(
    anomalies: DetectedAnomaly[],
    cellSnapshots: Map<string, CellKPISnapshot>
  ): AnomalyClassification[] {
    return anomalies.map(anomaly => {
      const relatedAnomalies = anomalies.filter(a => a.id !== anomaly.id);
      const cellSnapshot = cellSnapshots.get(anomaly.cellId);
      return this.anomalyClassifier.classifyAnomaly(anomaly, relatedAnomalies, cellSnapshot);
    });
  }

  /**
   * Find correlations between anomalies across different KPIs
   */
  private findAnomalyCorrelations(anomalies: DetectedAnomaly[]): CorrelationResult[] {
    const correlations: CorrelationResult[] = [];

    // Group anomalies by cell
    const byCell = new Map<string, DetectedAnomaly[]>();
    for (const anomaly of anomalies) {
      const cell = byCell.get(anomaly.cellId) || [];
      cell.push(anomaly);
      byCell.set(anomaly.cellId, cell);
    }

    // Find temporal correlations within each cell
    for (const [cellId, cellAnomalies] of byCell) {
      const kpiGroups = new Map<string, DetectedAnomaly[]>();
      for (const anomaly of cellAnomalies) {
        const group = kpiGroups.get(anomaly.kpiName) || [];
        group.push(anomaly);
        kpiGroups.set(anomaly.kpiName, group);
      }

      const kpis = Array.from(kpiGroups.keys());
      for (let i = 0; i < kpis.length; i++) {
        for (let j = i + 1; j < kpis.length; j++) {
          const anomalies1 = kpiGroups.get(kpis[i])!;
          const anomalies2 = kpiGroups.get(kpis[j])!;

          const temporalCorr = this.computeTemporalCorrelation(anomalies1, anomalies2);
          if (Math.abs(temporalCorr.correlation) > 0.5) {
            correlations.push({
              kpi1: kpis[i],
              kpi2: kpis[j],
              ...temporalCorr,
            });
          }
        }
      }
    }

    return correlations;
  }

  private computeTemporalCorrelation(
    anomalies1: DetectedAnomaly[],
    anomalies2: DetectedAnomaly[]
  ): { correlation: number; lag: number; significance: number } {
    // Simple temporal correlation based on occurrence timing
    const timeWindow = 60 * 60 * 1000; // 1 hour
    let matchCount = 0;
    let totalLag = 0;

    for (const a1 of anomalies1) {
      for (const a2 of anomalies2) {
        const timeDiff = a2.timestamp.getTime() - a1.timestamp.getTime();
        if (Math.abs(timeDiff) < timeWindow) {
          matchCount++;
          totalLag += timeDiff;
        }
      }
    }

    const maxMatches = Math.min(anomalies1.length, anomalies2.length);
    const correlation = maxMatches > 0 ? matchCount / maxMatches : 0;
    const avgLag = matchCount > 0 ? totalLag / matchCount : 0;

    return {
      correlation,
      lag: avgLag / 1000 / 60, // Convert to minutes
      significance: matchCount > 2 ? 0.9 : matchCount > 0 ? 0.6 : 0.1,
    };
  }

  /**
   * Analyze how issues propagate across cells
   */
  private analyzePropagation(
    anomalies: DetectedAnomaly[],
    neighborRelations: NeighborRelation[]
  ): { affectedCells: string[]; propagationPattern: 'isolated' | 'cluster' | 'cascade' | 'regional' } {
    const affectedCells = [...new Set(anomalies.map(a => a.cellId))];

    if (affectedCells.length === 1) {
      return { affectedCells, propagationPattern: 'isolated' };
    }

    // Check if affected cells are neighbors
    let neighborCount = 0;
    for (let i = 0; i < affectedCells.length; i++) {
      for (let j = i + 1; j < affectedCells.length; j++) {
        const isNeighbor = neighborRelations.some(
          nr =>
            (nr.sourceCellId === affectedCells[i] && nr.targetCellId === affectedCells[j]) ||
            (nr.sourceCellId === affectedCells[j] && nr.targetCellId === affectedCells[i])
        );
        if (isNeighbor) neighborCount++;
      }
    }

    const maxNeighborPairs = (affectedCells.length * (affectedCells.length - 1)) / 2;
    const neighborRatio = maxNeighborPairs > 0 ? neighborCount / maxNeighborPairs : 0;

    // Sort anomalies by time to detect cascade
    const sortedAnomalies = [...anomalies].sort(
      (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
    );

    const timeSpread =
      sortedAnomalies[sortedAnomalies.length - 1].timestamp.getTime() -
      sortedAnomalies[0].timestamp.getTime();

    if (neighborRatio > 0.6 && timeSpread < 30 * 60 * 1000) {
      return { affectedCells, propagationPattern: 'cluster' };
    }

    if (timeSpread > 30 * 60 * 1000 && neighborRatio > 0.3) {
      return { affectedCells, propagationPattern: 'cascade' };
    }

    if (affectedCells.length > 10) {
      return { affectedCells, propagationPattern: 'regional' };
    }

    return { affectedCells, propagationPattern: 'cluster' };
  }

  /**
   * Build causal chains from correlated anomalies
   */
  private buildCausalChains(
    classifications: AnomalyClassification[],
    correlations: CorrelationResult[],
    cellSnapshots: Map<string, CellKPISnapshot>
  ): CausalChain[] {
    const chains: CausalChain[] = [];

    // Group classifications by category
    const byCategory = new Map<RootCauseCategory, AnomalyClassification[]>();
    for (const classification of classifications) {
      const group = byCategory.get(classification.category) || [];
      group.push(classification);
      byCategory.set(classification.category, group);
    }

    // Build chains for common root cause patterns
    for (const [category, classificationGroup] of byCategory) {
      if (classificationGroup.length >= 1) {
        const chain = this.buildChainForCategory(category, classificationGroup, correlations);
        if (chain) {
          chains.push(chain);
        }
      }
    }

    return chains;
  }

  private buildChainForCategory(
    category: RootCauseCategory,
    classifications: AnomalyClassification[],
    correlations: CorrelationResult[]
  ): CausalChain | null {
    const avgConfidence = ss.mean(classifications.map(c => c.confidence));

    switch (category) {
      case 'interference':
        return {
          rootCause: 'interference',
          chain: [
            {
              step: 1,
              description: 'External or internal interference source present',
              evidence: classifications.flatMap(c => c.evidence),
              confidence: avgConfidence,
            },
            {
              step: 2,
              description: 'UL SINR degradation across affected PRBs',
              evidence: ['Elevated IoT', 'PUSCH SINR drops'],
              confidence: avgConfidence * 0.9,
            },
            {
              step: 3,
              description: 'UE power increases to compensate',
              evidence: ['Higher TX power', 'Reduced power headroom'],
              confidence: avgConfidence * 0.85,
            },
            {
              step: 4,
              description: 'Some UEs become power-limited, causing drops',
              evidence: ['Retransmissions increase', 'Drop rate rises'],
              confidence: avgConfidence * 0.8,
            },
          ],
          affectedKpis: ['ulSinrAvg', 'iotAvg', 'ueTxPowerAvg', 'erabDropRate'],
        };

      case 'power_control_issue':
        return {
          rootCause: 'power_control_issue',
          chain: [
            {
              step: 1,
              description: 'P0 nominal PUSCH or alpha misconfigured',
              evidence: classifications.flatMap(c => c.evidence),
              confidence: avgConfidence,
            },
            {
              step: 2,
              description: 'UEs operate at suboptimal transmit power levels',
              evidence: ['Power headroom anomalies', 'TX power distribution shift'],
              confidence: avgConfidence * 0.9,
            },
            {
              step: 3,
              description: 'Either interference increases (P0 too high) or coverage degrades (P0 too low)',
              evidence: ['IoT changes', 'RSRP degradation'],
              confidence: avgConfidence * 0.85,
            },
          ],
          affectedKpis: ['p0NominalPusch', 'powerHeadroomAvg', 'ueTxPowerAvg', 'iotAvg'],
        };

      case 'neighbor_relation_issue':
        return {
          rootCause: 'neighbor_relation_issue',
          chain: [
            {
              step: 1,
              description: 'Missing or incorrect neighbor relations',
              evidence: classifications.flatMap(c => c.evidence),
              confidence: avgConfidence,
            },
            {
              step: 2,
              description: 'Handovers fail or occur to wrong cells',
              evidence: ['HO failures increase', 'Wrong cell HO detected'],
              confidence: avgConfidence * 0.9,
            },
            {
              step: 3,
              description: 'Call drops increase at cell boundaries',
              evidence: ['Drop rate increases', 'Radio link failures'],
              confidence: avgConfidence * 0.85,
            },
          ],
          affectedKpis: ['intraFreqHoSuccessRate', 'interFreqHoSuccessRate', 'erabDropRate'],
        };

      case 'coverage_issue':
        return {
          rootCause: 'coverage_issue',
          chain: [
            {
              step: 1,
              description: 'Coverage hole or overshooting detected',
              evidence: classifications.flatMap(c => c.evidence),
              confidence: avgConfidence,
            },
            {
              step: 2,
              description: 'UEs experience poor signal quality in affected areas',
              evidence: ['RSRP degradation', 'SINR drops'],
              confidence: avgConfidence * 0.9,
            },
            {
              step: 3,
              description: 'Accessibility and retainability degrade',
              evidence: ['Setup failures', 'Drops increase'],
              confidence: avgConfidence * 0.85,
            },
          ],
          affectedKpis: ['rsrpAvg', 'rsrqAvg', 'rrcSetupSuccessRate', 'erabDropRate'],
        };

      default:
        return null;
    }
  }

  /**
   * Determine the primary root cause
   */
  private determinePrimaryCause(
    chains: CausalChain[],
    classifications: AnomalyClassification[]
  ): { primaryCause: RootCauseCategory; confidence: number } {
    if (chains.length === 0) {
      // Fall back to most common classification
      const categoryCounts = new Map<RootCauseCategory, number>();
      for (const classification of classifications) {
        categoryCounts.set(
          classification.category,
          (categoryCounts.get(classification.category) || 0) + 1
        );
      }

      let maxCategory: RootCauseCategory = 'unknown';
      let maxCount = 0;
      for (const [category, count] of categoryCounts) {
        if (count > maxCount) {
          maxCount = count;
          maxCategory = category;
        }
      }

      const avgConfidence =
        classifications.length > 0
          ? ss.mean(classifications.filter(c => c.category === maxCategory).map(c => c.confidence))
          : 0.5;

      return { primaryCause: maxCategory, confidence: avgConfidence };
    }

    // Find chain with highest overall confidence
    let bestChain = chains[0];
    let bestScore = 0;

    for (const chain of chains) {
      const avgChainConfidence = ss.mean(chain.chain.map(c => c.confidence));
      const kpiCount = chain.affectedKpis.length;
      const score = avgChainConfidence * (1 + kpiCount * 0.1);

      if (score > bestScore) {
        bestScore = score;
        bestChain = chain;
      }
    }

    return {
      primaryCause: bestChain.rootCause,
      confidence: ss.mean(bestChain.chain.map(c => c.confidence)),
    };
  }

  /**
   * Identify contributing factors
   */
  private identifyContributingFactors(
    classifications: AnomalyClassification[],
    correlations: CorrelationResult[],
    cellSnapshots: Map<string, CellKPISnapshot>
  ): RootCauseAnalysis['contributingFactors'] {
    const factors: RootCauseAnalysis['contributingFactors'] = [];

    // Group by category and find secondary causes
    const categoryConfidences = new Map<RootCauseCategory, number[]>();
    for (const classification of classifications) {
      const confidences = categoryConfidences.get(classification.category) || [];
      confidences.push(classification.confidence);
      categoryConfidences.set(classification.category, confidences);
    }

    for (const [category, confidences] of categoryConfidences) {
      const avgConfidence = ss.mean(confidences);
      if (avgConfidence > 0.3 && confidences.length > 0) {
        factors.push({
          category,
          description: this.getCategoryDescription(category),
          confidence: avgConfidence,
          evidence: classifications
            .filter(c => c.category === category)
            .flatMap(c => c.evidence)
            .slice(0, 5),
        });
      }
    }

    // Sort by confidence
    return factors.sort((a, b) => b.confidence - a.confidence).slice(0, 5);
  }

  private getCategoryDescription(category: RootCauseCategory): string {
    const descriptions: Record<RootCauseCategory, string> = {
      hardware_failure: 'Hardware component failure or degradation',
      software_issue: 'Software bug or processing error',
      configuration_error: 'Parameter misconfiguration',
      capacity_exhaustion: 'Resource capacity reached limits',
      interference: 'RF interference affecting signal quality',
      coverage_issue: 'Coverage hole or overshooting',
      backhaul_issue: 'Backhaul transport problem',
      core_network_issue: 'Core network element issue',
      parameter_drift: 'Gradual parameter drift from optimal',
      neighbor_relation_issue: 'Missing or incorrect neighbor relations',
      mobility_issue: 'Handover or mobility management problem',
      power_control_issue: 'Uplink power control misconfiguration',
      external_factor: 'External environmental factor',
      unknown: 'Cause not determined',
    };
    return descriptions[category];
  }

  /**
   * Generate actionable recommendations
   */
  private generateRecommendations(
    primaryCause: RootCauseCategory,
    contributingFactors: RootCauseAnalysis['contributingFactors'],
    cellSnapshots: Map<string, CellKPISnapshot>,
    anomalies: DetectedAnomaly[]
  ): RootCauseAnalysis['recommendations'] {
    const recommendations: RootCauseAnalysis['recommendations'] = [];

    // Primary cause recommendations
    switch (primaryCause) {
      case 'interference':
        recommendations.push({
          action: 'Perform spectrum scan to identify interference source',
          priority: 'immediate',
          expectedImpact: 'Identify root cause of interference',
        });
        recommendations.push({
          action: 'Check antenna system for PIM issues',
          priority: 'high',
          expectedImpact: 'Eliminate internal interference source',
        });
        break;

      case 'power_control_issue':
        // Get average P0 from snapshots
        const p0Values: number[] = [];
        const alphaValues: number[] = [];
        for (const snapshot of cellSnapshots.values()) {
          p0Values.push(snapshot.uplinkPowerControl.p0NominalPusch);
          alphaValues.push(snapshot.uplinkPowerControl.alpha);
        }
        const avgP0 = p0Values.length > 0 ? ss.mean(p0Values) : -96;
        const avgAlpha = alphaValues.length > 0 ? ss.mean(alphaValues) : 0.8;

        recommendations.push({
          action: 'Adjust P0 nominal PUSCH parameter',
          priority: 'high',
          expectedImpact: 'Optimize UE transmit power distribution',
          parameters: {
            currentP0: avgP0,
            suggestedP0: avgP0 + 3, // Will be refined in calculatePowerControlAdjustments
          },
        });
        recommendations.push({
          action: 'Review alpha (path loss compensation) factor',
          priority: 'medium',
          expectedImpact: 'Better balance between cell-center and cell-edge UEs',
          parameters: {
            currentAlpha: avgAlpha,
          },
        });
        break;

      case 'neighbor_relation_issue':
        recommendations.push({
          action: 'Audit neighbor relation table',
          priority: 'high',
          expectedImpact: 'Fix missing or incorrect neighbors',
        });
        recommendations.push({
          action: 'Enable ANR (Automatic Neighbor Relations) if not active',
          priority: 'medium',
          expectedImpact: 'Automatic discovery of missing neighbors',
        });
        break;

      case 'coverage_issue':
        recommendations.push({
          action: 'Review antenna tilt and azimuth settings',
          priority: 'high',
          expectedImpact: 'Optimize coverage footprint',
        });
        recommendations.push({
          action: 'Perform drive test to validate coverage',
          priority: 'medium',
          expectedImpact: 'Identify exact coverage gaps',
        });
        break;

      case 'mobility_issue':
        recommendations.push({
          action: 'Review handover parameters (A3 offset, hysteresis, TTT)',
          priority: 'high',
          expectedImpact: 'Reduce HO failures and ping-pong',
        });
        recommendations.push({
          action: 'Analyze HO failure causes in detail',
          priority: 'medium',
          expectedImpact: 'Target specific failure modes',
        });
        break;

      case 'capacity_exhaustion':
        recommendations.push({
          action: 'Review maximum connected users setting',
          priority: 'immediate',
          expectedImpact: 'Allow more simultaneous connections',
        });
        recommendations.push({
          action: 'Implement load balancing to adjacent cells',
          priority: 'high',
          expectedImpact: 'Distribute load more evenly',
        });
        break;

      default:
        recommendations.push({
          action: 'Perform detailed investigation of affected cells',
          priority: 'high',
          expectedImpact: 'Identify specific root cause',
        });
    }

    // Add recommendations for contributing factors
    for (const factor of contributingFactors.slice(0, 2)) {
      if (factor.category !== primaryCause) {
        recommendations.push({
          action: `Address contributing factor: ${factor.description}`,
          priority: 'medium',
          expectedImpact: 'Resolve secondary issues',
        });
      }
    }

    return recommendations;
  }

  /**
   * Calculate specific power control parameter adjustments
   */
  private calculatePowerControlAdjustments(
    primaryCause: RootCauseCategory,
    cellSnapshots: Map<string, CellKPISnapshot>,
    anomalies: DetectedAnomaly[]
  ): { suggestedP0Adjustment?: number; suggestedAlphaAdjustment?: number } {
    if (primaryCause !== 'power_control_issue' && primaryCause !== 'interference') {
      return {};
    }

    const metrics: {
      powerLimitedRatio: number[];
      negativePhrRatio: number[];
      iot: number[];
      p0Current: number[];
      alpha: number[];
    } = {
      powerLimitedRatio: [],
      negativePhrRatio: [],
      iot: [],
      p0Current: [],
      alpha: [],
    };

    for (const snapshot of cellSnapshots.values()) {
      metrics.powerLimitedRatio.push(snapshot.uplinkPowerControl.powerLimitedUeRatio);
      metrics.negativePhrRatio.push(snapshot.uplinkPowerControl.negativePowerHeadroomRatio);
      metrics.iot.push(snapshot.uplinkInterference.iotAvg);
      metrics.p0Current.push(snapshot.uplinkPowerControl.p0NominalPusch);
      metrics.alpha.push(snapshot.uplinkPowerControl.alpha);
    }

    if (metrics.p0Current.length === 0) {
      return {};
    }

    const avgPowerLimited = ss.mean(metrics.powerLimitedRatio);
    const avgNegativePhr = ss.mean(metrics.negativePhrRatio);
    const avgIot = ss.mean(metrics.iot);
    const avgP0 = ss.mean(metrics.p0Current);
    const avgAlpha = ss.mean(metrics.alpha);

    let p0Adjustment = 0;
    let alphaAdjustment = 0;

    // If many UEs are power-limited, P0 is likely too low
    if (avgPowerLimited > 15 || avgNegativePhr > 20) {
      p0Adjustment = 3; // Increase P0 by 3 dB
      if (avgPowerLimited > 25) {
        p0Adjustment = 5;
      }
    }
    // If IoT is high, P0 might be too high (or interference source)
    else if (avgIot > 10 && primaryCause === 'power_control_issue') {
      p0Adjustment = -2; // Decrease P0 by 2 dB
      if (avgIot > 15) {
        p0Adjustment = -3;
      }
    }

    // Alpha adjustment based on cell-edge vs cell-center balance
    if (avgPowerLimited > 20 && avgNegativePhr > 15 && avgAlpha < 0.9) {
      alphaAdjustment = 0.1; // Increase alpha for better path loss compensation
    } else if (avgIot > 12 && avgAlpha > 0.8) {
      alphaAdjustment = -0.1; // Decrease alpha to reduce overall interference
    }

    return {
      suggestedP0Adjustment: p0Adjustment !== 0 ? p0Adjustment : undefined,
      suggestedAlphaAdjustment: alphaAdjustment !== 0 ? alphaAdjustment : undefined,
    };
  }
}

// ============================================================================
// MULTI-CELL CORRELATION ANALYZER
// ============================================================================

export class MultiCellCorrelationAnalyzer {
  /**
   * Find correlations between anomalies across multiple cells
   */
  findCrossCellCorrelations(
    anomalies: DetectedAnomaly[],
    neighborRelations: NeighborRelation[],
    timeWindowMinutes: number = 30
  ): Array<{
    cells: string[];
    anomalyTypes: string[];
    correlation: number;
    temporalSpread: number;
    isNeighborCluster: boolean;
  }> {
    const results: Array<{
      cells: string[];
      anomalyTypes: string[];
      correlation: number;
      temporalSpread: number;
      isNeighborCluster: boolean;
    }> = [];

    // Group anomalies by time windows
    const timeWindowMs = timeWindowMinutes * 60 * 1000;
    const windowGroups: DetectedAnomaly[][] = [];
    const sortedAnomalies = [...anomalies].sort(
      (a, b) => a.timestamp.getTime() - b.timestamp.getTime()
    );

    let currentWindow: DetectedAnomaly[] = [];
    let windowStart = sortedAnomalies.length > 0 ? sortedAnomalies[0].timestamp.getTime() : 0;

    for (const anomaly of sortedAnomalies) {
      if (anomaly.timestamp.getTime() - windowStart > timeWindowMs) {
        if (currentWindow.length > 1) {
          windowGroups.push(currentWindow);
        }
        currentWindow = [anomaly];
        windowStart = anomaly.timestamp.getTime();
      } else {
        currentWindow.push(anomaly);
      }
    }
    if (currentWindow.length > 1) {
      windowGroups.push(currentWindow);
    }

    // Analyze each window group
    for (const group of windowGroups) {
      const cells = [...new Set(group.map(a => a.cellId))];
      const anomalyTypes = [...new Set(group.map(a => a.anomalyType))];

      if (cells.length > 1) {
        // Check if cells are neighbors
        let neighborPairs = 0;
        for (let i = 0; i < cells.length; i++) {
          for (let j = i + 1; j < cells.length; j++) {
            const isNeighbor = neighborRelations.some(
              nr =>
                (nr.sourceCellId === cells[i] && nr.targetCellId === cells[j]) ||
                (nr.sourceCellId === cells[j] && nr.targetCellId === cells[i])
            );
            if (isNeighbor) neighborPairs++;
          }
        }

        const maxPairs = (cells.length * (cells.length - 1)) / 2;
        const isNeighborCluster = neighborPairs / maxPairs > 0.5;

        const timestamps = group.map(a => a.timestamp.getTime());
        const temporalSpread = Math.max(...timestamps) - Math.min(...timestamps);

        results.push({
          cells,
          anomalyTypes,
          correlation: group.length / cells.length, // Anomalies per cell
          temporalSpread: temporalSpread / 1000 / 60, // Minutes
          isNeighborCluster,
        });
      }
    }

    return results;
  }
}

export default {
  RootCauseAnalyzer,
  MultiCellCorrelationAnalyzer,
};
