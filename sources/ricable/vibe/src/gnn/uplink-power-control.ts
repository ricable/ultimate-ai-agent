/**
 * Dynamic Uplink Power Control with GNN-Based Optimization
 * Implements fractional path loss based P0 and Alpha optimization
 * using Graph Neural Networks on intra-frequency neighbor relations
 */

import * as ss from 'simple-statistics';
import type {
  CellGraph,
  CellKPISnapshot,
  NeighborRelation,
  UplinkPowerControlKPI,
  UplinkInterferenceKPI,
} from '../models/ran-kpi.js';
import { CellGNN, CellGraphBuilder, SINRNeighborAnalyzer } from './cell-graph.js';

// ============================================================================
// UPLINK POWER CONTROL CONFIGURATION
// ============================================================================

export interface PowerControlConfig {
  // P0 nominal PUSCH range (dBm)
  p0Min: number;
  p0Max: number;
  p0Step: number;

  // Alpha (path loss compensation) range
  alphaValues: number[]; // Valid values: 0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

  // Optimization targets
  targetPowerHeadroom: number; // dB
  targetIoT: number; // dB
  targetPowerLimitedRatio: number; // %

  // Constraints
  maxInterferenceTolerance: number; // dB increase acceptable
  minCoverageMargin: number; // dB

  // GNN-based optimization weight
  gnnWeight: number; // Weight for GNN predictions vs rule-based
}

export const DEFAULT_POWER_CONTROL_CONFIG: PowerControlConfig = {
  p0Min: -110,
  p0Max: -85,
  p0Step: 1,

  alphaValues: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

  targetPowerHeadroom: 10,
  targetIoT: 6,
  targetPowerLimitedRatio: 5,

  maxInterferenceTolerance: 3,
  minCoverageMargin: 5,

  gnnWeight: 0.6,
};

// ============================================================================
// PATH LOSS MODELS
// ============================================================================

export interface PathLossDistribution {
  mean: number;
  stdDev: number;
  p10: number;
  p50: number;
  p90: number;
  distribution: number[];
}

export class PathLossAnalyzer {
  /**
   * Estimate path loss distribution from UE measurements
   */
  estimatePathLossDistribution(
    powerControl: UplinkPowerControlKPI
  ): PathLossDistribution {
    return {
      mean: powerControl.pathLossAvg,
      stdDev: (powerControl.pathLossP90 - powerControl.pathLossP10) / 2.56, // Approximate std
      p10: powerControl.pathLossP10,
      p50: powerControl.pathLossP50,
      p90: powerControl.pathLossP90,
      distribution: this.estimateDistribution(
        powerControl.pathLossP10,
        powerControl.pathLossP50,
        powerControl.pathLossP90
      ),
    };
  }

  /**
   * Calculate required P0 for target SINR given path loss
   */
  calculateRequiredP0(
    targetSinr: number,
    pathLoss: number,
    interferenceLevel: number,
    alpha: number
  ): number {
    // PUSCH power = P0 + alpha * PL + 10*log10(M) + delta_TF + f(i)
    // For target SINR: P0 = target_SINR + interference - (1-alpha)*PL
    // Simplified: P0 = target_SINR + N0 + NF - (1-alpha)*PL

    const thermalNoise = -174; // dBm/Hz
    const bandwidth = 10 * Math.log10(20e6); // 20 MHz
    const noiseFigure = 5; // dB

    const effectiveNoise = thermalNoise + bandwidth + noiseFigure;
    const requiredRxPower = effectiveNoise + targetSinr + interferenceLevel;

    // P0 = Prx - (1-alpha)*PL
    const p0 = requiredRxPower - (1 - alpha) * pathLoss;

    return Math.round(p0);
  }

  private estimateDistribution(p10: number, p50: number, p90: number): number[] {
    // Create a rough distribution based on percentiles
    const bins = 20;
    const min = p10 - 10;
    const max = p90 + 10;
    const binWidth = (max - min) / bins;

    // Use normal approximation
    const mean = (p10 + p50 + p90) / 3;
    const std = (p90 - p10) / 2.56;

    const distribution: number[] = [];
    for (let i = 0; i < bins; i++) {
      const x = min + (i + 0.5) * binWidth;
      const pdf = Math.exp(-0.5 * Math.pow((x - mean) / std, 2)) / (std * Math.sqrt(2 * Math.PI));
      distribution.push(pdf * binWidth);
    }

    // Normalize
    const sum = distribution.reduce((a, b) => a + b, 0);
    return distribution.map(v => v / sum);
  }
}

// ============================================================================
// FRACTIONAL PATH LOSS OPTIMIZER
// ============================================================================

export interface PowerControlOptimizationResult {
  cellId: string;
  currentP0: number;
  currentAlpha: number;
  recommendedP0: number;
  recommendedAlpha: number;
  expectedImprovement: {
    iotReduction: number;
    powerLimitedReduction: number;
    coverageGain: number;
    sinrImprovement: number;
  };
  confidence: number;
  rationale: string;
}

export class FractionalPathLossOptimizer {
  private config: PowerControlConfig;
  private pathLossAnalyzer: PathLossAnalyzer;

  constructor(config: Partial<PowerControlConfig> = {}) {
    this.config = { ...DEFAULT_POWER_CONTROL_CONFIG, ...config };
    this.pathLossAnalyzer = new PathLossAnalyzer();
  }

  /**
   * Optimize P0 and alpha for a single cell using rule-based approach
   */
  optimizeCell(
    powerControl: UplinkPowerControlKPI,
    interference: UplinkInterferenceKPI
  ): PowerControlOptimizationResult {
    const cellId = powerControl.cellId;
    const currentP0 = powerControl.p0NominalPusch;
    const currentAlpha = powerControl.alpha;

    const pathLoss = this.pathLossAnalyzer.estimatePathLossDistribution(powerControl);

    // Analyze current state
    const issues = this.analyzeCurrentState(powerControl, interference);

    // Generate optimization candidates
    const candidates = this.generateCandidates(currentP0, currentAlpha);

    // Score each candidate
    let bestCandidate = { p0: currentP0, alpha: currentAlpha };
    let bestScore = this.scoreCandidiate(currentP0, currentAlpha, powerControl, interference, pathLoss);

    for (const candidate of candidates) {
      const score = this.scoreCandidiate(
        candidate.p0,
        candidate.alpha,
        powerControl,
        interference,
        pathLoss
      );
      if (score > bestScore) {
        bestScore = score;
        bestCandidate = candidate;
      }
    }

    // Estimate improvements
    const improvements = this.estimateImprovements(
      currentP0,
      currentAlpha,
      bestCandidate.p0,
      bestCandidate.alpha,
      powerControl,
      interference,
      pathLoss
    );

    return {
      cellId,
      currentP0,
      currentAlpha,
      recommendedP0: bestCandidate.p0,
      recommendedAlpha: bestCandidate.alpha,
      expectedImprovement: improvements,
      confidence: this.calculateConfidence(issues, improvements),
      rationale: this.generateRationale(issues, bestCandidate, currentP0, currentAlpha, improvements),
    };
  }

  private analyzeCurrentState(
    powerControl: UplinkPowerControlKPI,
    interference: UplinkInterferenceKPI
  ): string[] {
    const issues: string[] = [];

    // Check for power-limited UEs
    if (powerControl.powerLimitedUeRatio > this.config.targetPowerLimitedRatio) {
      issues.push(`High power-limited UE ratio (${powerControl.powerLimitedUeRatio}%)`);
    }

    // Check for negative power headroom
    if (powerControl.negativePowerHeadroomRatio > 10) {
      issues.push(`High negative PHR ratio (${powerControl.negativePowerHeadroomRatio}%)`);
    }

    // Check for high interference
    if (interference.iotAvg > this.config.targetIoT) {
      issues.push(`Elevated IoT (${interference.iotAvg} dB)`);
    }

    // Check power headroom distribution
    if (powerControl.powerHeadroomAvg < 5) {
      issues.push(`Low average power headroom (${powerControl.powerHeadroomAvg} dB)`);
    }

    return issues;
  }

  private generateCandidates(
    currentP0: number,
    currentAlpha: number
  ): Array<{ p0: number; alpha: number }> {
    const candidates: Array<{ p0: number; alpha: number }> = [];

    // Generate P0 variations
    for (let p0 = currentP0 - 5; p0 <= currentP0 + 5; p0 += this.config.p0Step) {
      if (p0 >= this.config.p0Min && p0 <= this.config.p0Max) {
        // For each P0, try different alpha values
        for (const alpha of this.config.alphaValues) {
          if (p0 !== currentP0 || alpha !== currentAlpha) {
            candidates.push({ p0, alpha });
          }
        }
      }
    }

    return candidates;
  }

  private scoreCandidiate(
    p0: number,
    alpha: number,
    powerControl: UplinkPowerControlKPI,
    interference: UplinkInterferenceKPI,
    pathLoss: PathLossDistribution
  ): number {
    // Estimate metrics for this configuration
    const estimatedIoT = this.estimateIoT(p0, alpha, pathLoss, interference);
    const estimatedPowerLimited = this.estimatePowerLimitedRatio(p0, alpha, pathLoss);
    const estimatedPhr = this.estimatePowerHeadroom(p0, alpha, pathLoss);

    // Score components (higher is better)
    const iotScore = Math.max(0, 10 - estimatedIoT) / 10; // Target: IoT < 10
    const powerLimitedScore = Math.max(0, 20 - estimatedPowerLimited) / 20; // Target: < 20%
    const phrScore = Math.min(1, Math.max(0, estimatedPhr) / 15); // Target: positive PHR

    // Penalize extreme values
    const p0Penalty = Math.abs(p0 - (-96)) > 15 ? 0.1 : 0;
    const alphaPenalty = alpha < 0.5 || alpha > 0.95 ? 0.1 : 0;

    return (iotScore * 0.35 + powerLimitedScore * 0.35 + phrScore * 0.3) - p0Penalty - alphaPenalty;
  }

  private estimateIoT(
    p0: number,
    alpha: number,
    pathLoss: PathLossDistribution,
    currentInterference: UplinkInterferenceKPI
  ): number {
    // Higher P0 or higher alpha increases UE TX power, increasing interference
    const p0Effect = (p0 - (-96)) * 0.3; // Each dB of P0 adds ~0.3 dB to IoT
    const alphaEffect = (alpha - 0.8) * 5; // Alpha deviation from 0.8 affects IoT

    const baseIoT = currentInterference.iotAvg - p0Effect - alphaEffect;
    const newP0Effect = (p0 - (-96)) * 0.3;
    const newAlphaEffect = (alpha - 0.8) * 5;

    return Math.max(0, baseIoT + newP0Effect + newAlphaEffect);
  }

  private estimatePowerLimitedRatio(
    p0: number,
    alpha: number,
    pathLoss: PathLossDistribution
  ): number {
    // Calculate what fraction of UEs would hit Pcmax (23 dBm)
    const pcmax = 23;

    // UE TX power = P0 + alpha * PL + adjustments
    // For path loss at P90 (cell edge):
    const edgeTxPower = p0 + alpha * pathLoss.p90;

    // Rough estimate: % of UEs that would exceed Pcmax
    const margin = pcmax - edgeTxPower;

    if (margin > 5) return 0; // Plenty of headroom
    if (margin < -5) return 50; // Many UEs power limited

    return Math.max(0, (5 - margin) * 5); // Linear interpolation
  }

  private estimatePowerHeadroom(
    p0: number,
    alpha: number,
    pathLoss: PathLossDistribution
  ): number {
    const pcmax = 23;

    // Average TX power
    const avgTxPower = p0 + alpha * pathLoss.mean;

    // Power headroom = Pcmax - TX_power
    return pcmax - avgTxPower;
  }

  private estimateImprovements(
    oldP0: number,
    oldAlpha: number,
    newP0: number,
    newAlpha: number,
    powerControl: UplinkPowerControlKPI,
    interference: UplinkInterferenceKPI,
    pathLoss: PathLossDistribution
  ): PowerControlOptimizationResult['expectedImprovement'] {
    const oldIoT = this.estimateIoT(oldP0, oldAlpha, pathLoss, interference);
    const newIoT = this.estimateIoT(newP0, newAlpha, pathLoss, interference);

    const oldPowerLimited = this.estimatePowerLimitedRatio(oldP0, oldAlpha, pathLoss);
    const newPowerLimited = this.estimatePowerLimitedRatio(newP0, newAlpha, pathLoss);

    const oldPhr = this.estimatePowerHeadroom(oldP0, oldAlpha, pathLoss);
    const newPhr = this.estimatePowerHeadroom(newP0, newAlpha, pathLoss);

    return {
      iotReduction: oldIoT - newIoT,
      powerLimitedReduction: oldPowerLimited - newPowerLimited,
      coverageGain: newPhr - oldPhr, // More headroom = better coverage margin
      sinrImprovement: (oldIoT - newIoT) * 0.8, // Rough: 0.8 dB SINR per dB IoT reduction
    };
  }

  private calculateConfidence(
    issues: string[],
    improvements: PowerControlOptimizationResult['expectedImprovement']
  ): number {
    let confidence = 0.5;

    // More issues = more confident that change is needed
    confidence += issues.length * 0.1;

    // Larger improvements = more confident
    if (Math.abs(improvements.iotReduction) > 1) confidence += 0.1;
    if (Math.abs(improvements.powerLimitedReduction) > 5) confidence += 0.1;

    return Math.min(0.95, confidence);
  }

  private generateRationale(
    issues: string[],
    recommended: { p0: number; alpha: number },
    currentP0: number,
    currentAlpha: number,
    improvements: PowerControlOptimizationResult['expectedImprovement']
  ): string {
    const parts: string[] = [];

    if (issues.length > 0) {
      parts.push(`Issues identified: ${issues.join('; ')}`);
    }

    const p0Change = recommended.p0 - currentP0;
    const alphaChange = recommended.alpha - currentAlpha;

    if (p0Change !== 0) {
      const direction = p0Change > 0 ? 'increase' : 'decrease';
      parts.push(`Recommend ${direction} P0 by ${Math.abs(p0Change)} dB`);
    }

    if (alphaChange !== 0) {
      const direction = alphaChange > 0 ? 'increase' : 'decrease';
      parts.push(`Recommend ${direction} alpha by ${Math.abs(alphaChange).toFixed(1)}`);
    }

    if (improvements.iotReduction > 0) {
      parts.push(`Expected IoT reduction: ${improvements.iotReduction.toFixed(1)} dB`);
    }

    if (improvements.powerLimitedReduction > 0) {
      parts.push(`Expected power-limited UE reduction: ${improvements.powerLimitedReduction.toFixed(0)}%`);
    }

    return parts.join('. ');
  }
}

// ============================================================================
// GNN-BASED POWER CONTROL OPTIMIZER
// ============================================================================

export class GNNPowerControlOptimizer {
  private config: PowerControlConfig;
  private gnn: CellGNN;
  private graphBuilder: CellGraphBuilder;
  private ruleBasedOptimizer: FractionalPathLossOptimizer;

  constructor(config: Partial<PowerControlConfig> = {}) {
    this.config = { ...DEFAULT_POWER_CONTROL_CONFIG, ...config };
    this.gnn = new CellGNN({
      inputDim: 20,
      hiddenDim: 64,
      outputDim: 16,
      numLayers: 3,
      numHeads: 4,
    });
    this.graphBuilder = new CellGraphBuilder();
    this.ruleBasedOptimizer = new FractionalPathLossOptimizer(config);
  }

  /**
   * Optimize power control for entire cell graph using GNN
   */
  optimizeNetworkPowerControl(
    cellSnapshots: Map<string, CellKPISnapshot>,
    neighborRelations: NeighborRelation[]
  ): Map<string, PowerControlOptimizationResult> {
    // Build cell graph
    const graph = this.graphBuilder.buildGraph(cellSnapshots, neighborRelations);

    // Get GNN-based predictions
    const gnnPredictions = this.gnn.predictPowerControlParams(graph);

    // Combine with rule-based optimization
    const results = new Map<string, PowerControlOptimizationResult>();

    for (const [cellId, snapshot] of cellSnapshots) {
      // Rule-based optimization
      const ruleBasedResult = this.ruleBasedOptimizer.optimizeCell(
        { ...snapshot.uplinkPowerControl, timestamp: snapshot.timestamp, cellId },
        { ...snapshot.uplinkInterference, timestamp: snapshot.timestamp, cellId }
      );

      // GNN prediction
      const gnnPrediction = gnnPredictions.get(cellId);

      if (gnnPrediction) {
        // Blend GNN and rule-based results
        const blendedP0 = Math.round(
          this.config.gnnWeight * gnnPrediction.p0 +
          (1 - this.config.gnnWeight) * ruleBasedResult.recommendedP0
        );

        const blendedAlpha = this.findNearestAlpha(
          this.config.gnnWeight * gnnPrediction.alpha +
          (1 - this.config.gnnWeight) * ruleBasedResult.recommendedAlpha
        );

        // Apply neighbor-aware constraints
        const constrainedResult = this.applyNeighborConstraints(
          cellId,
          blendedP0,
          blendedAlpha,
          neighborRelations,
          cellSnapshots
        );

        results.set(cellId, {
          ...ruleBasedResult,
          recommendedP0: constrainedResult.p0,
          recommendedAlpha: constrainedResult.alpha,
          rationale: ruleBasedResult.rationale +
            ` GNN-based neighbor-aware optimization applied (weight: ${this.config.gnnWeight}).`,
          confidence: Math.min(0.95, ruleBasedResult.confidence + 0.1),
        });
      } else {
        results.set(cellId, ruleBasedResult);
      }
    }

    return results;
  }

  /**
   * Apply constraints based on neighbor cell configurations
   */
  private applyNeighborConstraints(
    cellId: string,
    proposedP0: number,
    proposedAlpha: number,
    neighborRelations: NeighborRelation[],
    cellSnapshots: Map<string, CellKPISnapshot>
  ): { p0: number; alpha: number } {
    // Find intra-frequency neighbors
    const intraFreqNeighbors = neighborRelations.filter(
      nr => nr.sourceCellId === cellId && nr.relationshipType === 'intra-freq'
    );

    if (intraFreqNeighbors.length === 0) {
      return { p0: proposedP0, alpha: proposedAlpha };
    }

    // Get neighbor P0 values
    const neighborP0s: number[] = [];
    for (const nr of intraFreqNeighbors) {
      const neighborSnapshot = cellSnapshots.get(nr.targetCellId);
      if (neighborSnapshot) {
        neighborP0s.push(neighborSnapshot.uplinkPowerControl.p0NominalPusch);
      }
    }

    if (neighborP0s.length === 0) {
      return { p0: proposedP0, alpha: proposedAlpha };
    }

    // Constrain P0 to be within reasonable range of neighbors
    const avgNeighborP0 = ss.mean(neighborP0s);
    const maxDelta = 5; // dB

    let constrainedP0 = proposedP0;
    if (proposedP0 > avgNeighborP0 + maxDelta) {
      constrainedP0 = Math.round(avgNeighborP0 + maxDelta);
    } else if (proposedP0 < avgNeighborP0 - maxDelta) {
      constrainedP0 = Math.round(avgNeighborP0 - maxDelta);
    }

    return { p0: constrainedP0, alpha: proposedAlpha };
  }

  /**
   * Find nearest valid alpha value
   */
  private findNearestAlpha(target: number): number {
    let nearest = this.config.alphaValues[0];
    let minDiff = Math.abs(target - nearest);

    for (const alpha of this.config.alphaValues) {
      const diff = Math.abs(target - alpha);
      if (diff < minDiff) {
        minDiff = diff;
        nearest = alpha;
      }
    }

    return nearest;
  }
}

// ============================================================================
// POWER CONTROL VALIDATION
// ============================================================================

export class PowerControlValidator {
  /**
   * Validate proposed power control changes
   */
  validateChanges(
    currentConfig: { p0: number; alpha: number },
    proposedConfig: { p0: number; alpha: number },
    cellSnapshot: CellKPISnapshot
  ): {
    isValid: boolean;
    warnings: string[];
    risks: string[];
  } {
    const warnings: string[] = [];
    const risks: string[] = [];

    const p0Change = proposedConfig.p0 - currentConfig.p0;
    const alphaChange = proposedConfig.alpha - currentConfig.alpha;

    // Check P0 bounds
    if (proposedConfig.p0 < -110 || proposedConfig.p0 > -80) {
      risks.push(`P0 ${proposedConfig.p0} dBm is outside typical range [-110, -80] dBm`);
    }

    // Check for large changes
    if (Math.abs(p0Change) > 5) {
      warnings.push(`Large P0 change (${p0Change} dB) - consider incremental adjustment`);
    }

    if (Math.abs(alphaChange) > 0.2) {
      warnings.push(`Large alpha change (${alphaChange}) - may cause significant behavior shift`);
    }

    // Check for potential coverage impact
    if (p0Change < -3 && cellSnapshot.uplinkPowerControl.powerLimitedUeRatio > 10) {
      risks.push('Reducing P0 may worsen power-limited UE situation');
    }

    // Check for potential interference impact
    if (p0Change > 3 && cellSnapshot.uplinkInterference.iotAvg > 8) {
      risks.push('Increasing P0 may worsen interference');
    }

    // Check alpha direction
    if (alphaChange > 0 && cellSnapshot.uplinkPowerControl.powerLimitedUeRatio > 15) {
      warnings.push('Increasing alpha may help power-limited UEs but increase interference');
    }

    return {
      isValid: risks.length === 0,
      warnings,
      risks,
    };
  }

  /**
   * Calculate expected KPI impact of power control change
   */
  estimateImpact(
    currentConfig: { p0: number; alpha: number },
    proposedConfig: { p0: number; alpha: number },
    cellSnapshot: CellKPISnapshot
  ): {
    iotDelta: number;
    powerLimitedDelta: number;
    sinrDelta: number;
    coverageMarginDelta: number;
  } {
    const p0Change = proposedConfig.p0 - currentConfig.p0;
    const alphaChange = proposedConfig.alpha - currentConfig.alpha;
    const pathLoss = cellSnapshot.uplinkPowerControl.pathLossAvg;

    // Simplified impact models
    // IoT: increases with P0 and alpha (more UE TX power = more interference)
    const iotDelta = p0Change * 0.4 + alphaChange * pathLoss * 0.3;

    // Power limited: decreases with higher P0, increases with higher alpha at cell edge
    const powerLimitedDelta = -p0Change * 2 + alphaChange * 5;

    // SINR: improves with lower interference
    const sinrDelta = -iotDelta * 0.8;

    // Coverage margin: improves with higher P0 and alpha
    const coverageMarginDelta = p0Change * 0.5 + alphaChange * 3;

    return {
      iotDelta,
      powerLimitedDelta,
      sinrDelta,
      coverageMarginDelta,
    };
  }
}

export default {
  PathLossAnalyzer,
  FractionalPathLossOptimizer,
  GNNPowerControlOptimizer,
  PowerControlValidator,
  DEFAULT_POWER_CONTROL_CONFIG,
};
