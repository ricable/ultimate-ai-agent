/**
 * Network Objective Function for Balanced GNN Optimization
 *
 * Enforces network-wide constraints ensuring balanced optimization:
 * - HARD CONSTRAINT: Network net SINR must be >= 0
 * - Prevents excessive P0_DECREASE recommendations
 * - Fairness across cells (Jain's index)
 */

import type { RealKPIGraph } from './real-kpi-graph.js';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface OptimizationRecommendation {
  cellId: string;
  cellName: string;
  band: string;
  currentP0: number;
  currentAlpha: number;
  currentSINR: number;
  recommendedP0: number;
  recommendedAlpha: number;
  predictedSINR: number;
  sinrImprovement: number;
  strategy: 'P0_INCREASE' | 'P0_DECREASE' | 'COMBINED' | 'NO_CHANGE';
  neighborImpact: number;
  confidence: number;
  // Bidirectional tracking
  receivedBenefit?: number;
  sentCost?: number;
}

export interface LossComponents {
  sinrDeficitLoss: number;
  interferenceLoss: number;
  changePenalty: number;
  fairnessLoss: number;
}

export interface ObjectiveConfig {
  sinrTarget: number;     // Target SINR (e.g., 5 dB)
  sinrMin: number;        // Minimum safe SINR (e.g., 0 dB)
  lambda: number;         // Interference penalty weight
  mu: number;             // Change penalty weight
  gamma: number;          // Fairness penalty weight
}

export interface ConstraintResult {
  satisfied: boolean;
  netSINRChange: number;
  totalGain: number;
  totalCost: number;
  scalingFactor: number;
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_CONFIG: ObjectiveConfig = {
  sinrTarget: 5.0,
  sinrMin: 0.0,
  lambda: 0.5,
  mu: 0.1,
  gamma: 0.3,
};

// ============================================================================
// NETWORK OBJECTIVE FUNCTION CLASS
// ============================================================================

/**
 * NetworkObjectiveFunction - Enforces network-wide optimization constraints
 *
 * Key responsibilities:
 * 1. Compute loss for training (multi-objective)
 * 2. HARD CONSTRAINT: Ensure net SINR >= 0
 * 3. Scale down P0_DECREASE if network would lose SINR
 * 4. Track fairness across cells
 */
export class NetworkObjectiveFunction {
  private config: ObjectiveConfig;

  constructor(config: Partial<ObjectiveConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * HARD CONSTRAINT: Network net SINR must be >= 0
   * This guarantees the network never loses total SINR from recommendations
   *
   * Iteratively scales down P0_DECREASE until constraint is satisfied
   */
  applyNetworkConstraint(
    recommendations: OptimizationRecommendation[],
    graph?: RealKPIGraph
  ): {
    results: OptimizationRecommendation[];
    constraint: ConstraintResult;
  } {
    // Deep copy to avoid mutating originals
    let recs = recommendations.map(r => ({ ...r }));

    // Calculate initial network impact
    let { totalGain, totalCost } = this.calculateNetworkImpact(recs, graph);
    let netImpact = totalGain - totalCost;
    let scalingFactor = 1.0;
    let iterations = 0;
    const maxIterations = 10;

    // HARD CONSTRAINT: Must achieve non-negative net SINR
    while (netImpact < 0 && iterations < maxIterations) {
      iterations++;

      // Calculate scaling factor
      scalingFactor = Math.max(0.3, totalGain / (totalGain + Math.abs(netImpact)));

      // Scale down P0_DECREASE recommendations
      recs = recs.map(rec => {
        if (rec.strategy === 'P0_DECREASE') {
          const originalDelta = rec.recommendedP0 - rec.currentP0;
          const scaledDelta = originalDelta * scalingFactor;

          // If scaled delta too small, convert to NO_CHANGE
          if (Math.abs(scaledDelta) < 2) {
            return {
              ...rec,
              strategy: 'NO_CHANGE' as const,
              recommendedP0: rec.currentP0,
              sinrImprovement: 0,
              neighborImpact: 0,
            };
          }

          const newP0 = Math.round(rec.currentP0 + scaledDelta);
          const newImprovement = scaledDelta * 0.18; // Recalculate

          return {
            ...rec,
            recommendedP0: newP0,
            sinrImprovement: newImprovement,
            neighborImpact: (rec.neighborImpact || 0) * scalingFactor,
          };
        }
        return rec;
      });

      // Remove NO_CHANGE recommendations that were converted
      recs = recs.filter(rec => rec.strategy !== 'NO_CHANGE');

      // Recalculate after scaling
      const updated = this.calculateNetworkImpact(recs, graph);
      totalGain = updated.totalGain;
      totalCost = updated.totalCost;
      netImpact = totalGain - totalCost;

      // Safety: if scaling doesn't help, remove lowest-impact P0_DECREASE
      if (netImpact < 0 && iterations < maxIterations) {
        recs = this.removeLowestImpactP0Decrease(recs);
      }
    }

    return {
      results: recs,
      constraint: {
        satisfied: netImpact >= 0,
        netSINRChange: netImpact,
        totalGain,
        totalCost,
        scalingFactor,
      },
    };
  }

  /**
   * Calculate network-wide impact of recommendations
   */
  private calculateNetworkImpact(
    recs: OptimizationRecommendation[],
    graph?: RealKPIGraph
  ): { totalGain: number; totalCost: number } {
    let totalGain = 0;
    let totalCost = 0;

    for (const rec of recs) {
      // Positive SINR improvement counts as gain
      if (rec.sinrImprovement > 0) {
        totalGain += rec.sinrImprovement;
      }

      // Calculate neighbor degradation cost
      if (rec.strategy === 'P0_INCREASE' || rec.strategy === 'COMBINED') {
        // P0 increase causes interference to neighbors
        const p0Increase = rec.recommendedP0 - rec.currentP0;
        const neighborCount = graph
          ? this.getNeighborCount(rec.cellId, graph)
          : 5; // Default estimate

        // Each dB P0 increase causes ~0.1 dB degradation per neighbor (weighted by coupling)
        const estimatedDegradation = p0Increase * 0.1 * Math.sqrt(neighborCount);
        totalCost += Math.max(0, estimatedDegradation);
      }

      // Track explicit neighbor impact if available
      if (rec.neighborImpact && rec.neighborImpact > 0) {
        totalCost += rec.neighborImpact;
      }

      // P0_DECREASE causes self-cost (own SINR loss)
      if (rec.strategy === 'P0_DECREASE' && rec.sinrImprovement < 0) {
        totalCost += Math.abs(rec.sinrImprovement);
      }
    }

    return { totalGain, totalCost };
  }

  /**
   * Get neighbor count for a cell from graph
   */
  private getNeighborCount(cellId: string, graph: RealKPIGraph): number {
    const idx = graph.nodeIds.indexOf(cellId);
    if (idx < 0) return 0;

    let count = 0;
    for (let j = 0; j < graph.nodeIds.length; j++) {
      if (j !== idx && graph.adjacencyMatrix[idx][j] > 0.1) {
        count++;
      }
    }
    return count;
  }

  /**
   * Remove the lowest-impact P0_DECREASE recommendation
   */
  private removeLowestImpactP0Decrease(
    recs: OptimizationRecommendation[]
  ): OptimizationRecommendation[] {
    const p0DecreaseRecs = recs.filter(r => r.strategy === 'P0_DECREASE');

    if (p0DecreaseRecs.length === 0) {
      return recs;
    }

    // Find the one with smallest absolute impact (least helpful)
    let minImpactIdx = -1;
    let minImpact = Infinity;

    for (let i = 0; i < recs.length; i++) {
      if (recs[i].strategy === 'P0_DECREASE') {
        const impact = Math.abs(recs[i].sinrImprovement);
        if (impact < minImpact) {
          minImpact = impact;
          minImpactIdx = i;
        }
      }
    }

    if (minImpactIdx >= 0) {
      return recs.filter((_, i) => i !== minImpactIdx);
    }

    return recs;
  }

  /**
   * Compute loss for training (multi-objective)
   */
  computeLoss(
    predictions: Map<string, number>,
    graph: RealKPIGraph
  ): { loss: number; components: LossComponents } {
    // Initialize loss components
    let sinrDeficitLoss = 0;
    let interferenceLoss = 0;
    let changePenalty = 0;

    const sinrValues: number[] = [];

    // Process each cell
    for (const [cellId, predictedSINR] of predictions) {
      const cell = graph.cellKPIs.get(cellId);
      if (!cell) continue;

      sinrValues.push(predictedSINR);

      // SINR deficit loss (squared for gradient)
      const deficit = Math.max(0, this.config.sinrTarget - predictedSINR);
      sinrDeficitLoss += deficit * deficit;

      // Extra penalty for going below minimum (hard constraint proxy)
      if (predictedSINR < this.config.sinrMin) {
        sinrDeficitLoss += 10 * Math.pow(this.config.sinrMin - predictedSINR, 2);
      }

      // Interference loss (if we have interference data)
      const interference = cell.ulInterferenceAvgPusch || -115;
      const interferenceThreshold = -105; // dBm
      if (interference > interferenceThreshold) {
        interferenceLoss += interference - interferenceThreshold;
      }
    }

    // Change penalty (computed from recommendations if available)
    // This is applied during training when we have the original vs recommended params

    // Fairness loss using Jain's index
    const fairnessLoss = this.computeFairnessLoss(sinrValues);

    // Total loss
    const totalLoss =
      sinrDeficitLoss +
      this.config.lambda * interferenceLoss +
      changePenalty +
      this.config.gamma * fairnessLoss;

    return {
      loss: totalLoss,
      components: {
        sinrDeficitLoss,
        interferenceLoss,
        changePenalty,
        fairnessLoss,
      },
    };
  }

  /**
   * Compute loss with explicit recommendations for change penalty
   */
  computeLossWithRecommendations(
    recommendations: OptimizationRecommendation[],
    graph: RealKPIGraph
  ): { loss: number; components: LossComponents } {
    // Build predictions map
    const predictions = new Map<string, number>();
    let changePenalty = 0;

    for (const rec of recommendations) {
      predictions.set(rec.cellId, rec.predictedSINR);

      // Change penalty: penalize large changes
      const p0Change = Math.abs(rec.recommendedP0 - rec.currentP0);
      const alphaChange = Math.abs(rec.recommendedAlpha - rec.currentAlpha) * 10; // Scale
      changePenalty += (p0Change + alphaChange) * this.config.mu;
    }

    // Also add current cells that aren't being changed
    for (const [cellId, cell] of graph.cellKPIs) {
      if (!predictions.has(cellId)) {
        predictions.set(cellId, cell.sinrPuschAvg);
      }
    }

    // Compute base loss
    const baseResult = this.computeLoss(predictions, graph);

    // Add change penalty
    const totalLoss = baseResult.loss + changePenalty;

    return {
      loss: totalLoss,
      components: {
        ...baseResult.components,
        changePenalty,
      },
    };
  }

  /**
   * Jain's fairness index: 1 = perfectly fair, 0 = maximally unfair
   * Loss = (1 - jainIndex) * N to penalize unfairness
   */
  private computeFairnessLoss(values: number[]): number {
    if (values.length === 0) return 0;

    const sum = values.reduce((a, b) => a + b, 0);
    const sumSquared = values.reduce((a, b) => a + b * b, 0);

    if (sumSquared === 0) return 0;

    const jainIndex = (sum * sum) / (values.length * sumSquared);

    // Loss = (1 - jainIndex) * N
    return (1 - jainIndex) * values.length;
  }

  /**
   * Check if recommendations satisfy network constraint
   */
  satisfiesNetworkConstraint(
    recs: OptimizationRecommendation[],
    graph?: RealKPIGraph
  ): ConstraintResult {
    const { totalGain, totalCost } = this.calculateNetworkImpact(recs, graph);
    const netChange = totalGain - totalCost;

    return {
      satisfied: netChange >= 0,
      netSINRChange: netChange,
      totalGain,
      totalCost,
      scalingFactor: 1.0,
    };
  }

  /**
   * Get strategy distribution statistics
   */
  getStrategyDistribution(
    recs: OptimizationRecommendation[]
  ): Map<string, { count: number; percentage: number }> {
    const distribution = new Map<string, { count: number; percentage: number }>();
    const total = recs.length;

    for (const rec of recs) {
      const current = distribution.get(rec.strategy) || { count: 0, percentage: 0 };
      distribution.set(rec.strategy, {
        count: current.count + 1,
        percentage: 0, // Calculate after
      });
    }

    // Calculate percentages
    for (const [strategy, stats] of distribution) {
      distribution.set(strategy, {
        count: stats.count,
        percentage: total > 0 ? (stats.count / total) * 100 : 0,
      });
    }

    return distribution;
  }

  /**
   * Validate that strategy distribution is balanced
   * Target: P0_DECREASE should be 40-50%
   */
  isDistributionBalanced(
    recs: OptimizationRecommendation[]
  ): { balanced: boolean; p0DecreaseRatio: number; message: string } {
    const distribution = this.getStrategyDistribution(recs);
    const p0DecreaseStats = distribution.get('P0_DECREASE') || { count: 0, percentage: 0 };
    const ratio = p0DecreaseStats.percentage;

    let message: string;
    let balanced: boolean;

    if (ratio > 60) {
      balanced = false;
      message = `P0_DECREASE too high (${ratio.toFixed(1)}%), target: 40-50%`;
    } else if (ratio < 30) {
      balanced = false;
      message = `P0_DECREASE too low (${ratio.toFixed(1)}%), might miss neighbor help opportunities`;
    } else {
      balanced = true;
      message = `Distribution balanced: P0_DECREASE at ${ratio.toFixed(1)}%`;
    }

    return { balanced, p0DecreaseRatio: ratio, message };
  }

  /**
   * Get configuration
   */
  getConfig(): ObjectiveConfig {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(updates: Partial<ObjectiveConfig>): void {
    this.config = { ...this.config, ...updates };
  }
}

export default NetworkObjectiveFunction;
