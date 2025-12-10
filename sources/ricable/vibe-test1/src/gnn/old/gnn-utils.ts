/**
 * GNN Utility Classes
 *
 * Migrated utilities from deprecated GNN files:
 * - PathLossAnalyzer (from uplink-power-control.ts)
 * - PowerControlValidator (from uplink-power-control.ts)
 * - SINRNeighborAnalyzer (from cell-graph.ts)
 */

import type {
  CellKPISnapshot,
  NeighborRelation,
} from '../models/ran-kpi.js';

import type { SurrogateGraph, PowerControlParams } from './network-surrogate-model.js';

// ============================================================================
// PATH LOSS DISTRIBUTION INTERFACE
// ============================================================================

export interface PathLossDistribution {
  mean: number;
  stdDev: number;
  p10: number;
  p50: number;
  p90: number;
  distribution: number[];
}

// ============================================================================
// PATH LOSS ANALYZER
// ============================================================================

/**
 * Analyzes path loss distribution from UE measurements
 * Used for uplink power control optimization
 */
export class PathLossAnalyzer {
  /**
   * Estimate path loss distribution from UE measurements
   */
  estimatePathLossDistribution(
    powerControl: CellKPISnapshot['uplinkPowerControl']
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
// POWER CONTROL VALIDATOR
// ============================================================================

/**
 * Validates proposed power control changes for safety
 */
export class PowerControlValidator {
  /**
   * Validate proposed power control changes
   */
  validateChanges(
    currentConfig: PowerControlParams,
    proposedConfig: PowerControlParams,
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
    currentConfig: PowerControlParams,
    proposedConfig: PowerControlParams,
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

// ============================================================================
// SINR NEIGHBOR ANALYZER
// ============================================================================

/**
 * Analyzes SINR relationships between neighbor cells
 * Used for handover optimization and interference detection
 */
export class SINRNeighborAnalyzer {
  /**
   * Analyze SINR relationships in the cell graph
   */
  analyzeSINRRelationships(graph: SurrogateGraph): Array<{
    sourceCellId: string;
    targetCellId: string;
    sinrDelta: number;
    recommendation: string;
    priority: 'high' | 'medium' | 'low';
  }> {
    const recommendations: Array<{
      sourceCellId: string;
      targetCellId: string;
      sinrDelta: number;
      recommendation: string;
      priority: 'high' | 'medium' | 'low';
    }> = [];

    const nodeCount = graph.nodeIds.length;

    // Analyze edge features for SINR relationships
    for (let i = 0; i < nodeCount; i++) {
      for (let j = 0; j < nodeCount; j++) {
        if (i === j) continue;

        // Check if there's an edge (adjacency > 0)
        if (graph.adjacencyMatrix[i][j] > 0 && graph.edgeFeatures[i][j].length > 0) {
          const edgeFeats = graph.edgeFeatures[i][j];

          // Edge features: [sourceSinr, targetSinr, sinrDelta, ...]
          const sourceSinr = edgeFeats[0] * 35 - 5; // Denormalize
          const targetSinr = edgeFeats[1] * 35 - 5;
          const sinrDelta = targetSinr - sourceSinr;

          const sourceCellId = graph.nodeIds[i];
          const targetCellId = graph.nodeIds[j];

          // Check for too early/late HO based on SINR
          if (sinrDelta > 10) {
            recommendations.push({
              sourceCellId,
              targetCellId,
              sinrDelta,
              recommendation: `Consider reducing A3 offset for ${sourceCellId}->${targetCellId}: target SINR is ${sinrDelta.toFixed(1)}dB better`,
              priority: sinrDelta > 15 ? 'high' : 'medium',
            });
          } else if (sinrDelta < -3 && graph.adjacencyMatrix[i][j] < 0.5) {
            recommendations.push({
              sourceCellId,
              targetCellId,
              sinrDelta,
              recommendation: `Review neighbor relation ${sourceCellId}->${targetCellId}: poor target SINR (${targetSinr.toFixed(1)}dB) causing HO failures`,
              priority: sinrDelta < -6 ? 'high' : 'medium',
            });
          }

          // Check for interference issues between neighbors
          if (sourceSinr < 3 && targetSinr < 3) {
            recommendations.push({
              sourceCellId,
              targetCellId,
              sinrDelta,
              recommendation: `Both cells ${sourceCellId} and ${targetCellId} have low SINR - check for mutual interference`,
              priority: 'high',
            });
          }
        }
      }
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { high: 0, medium: 1, low: 2 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
  }

  /**
   * Find optimal handover parameters based on SINR
   */
  optimizeHandoverParams(
    relation: NeighborRelation,
    currentSnapshot: CellKPISnapshot
  ): {
    a3Offset: number;
    hysteresis: number;
    timeToTrigger: number;
    rationale: string;
  } {
    const sinrDelta = relation.targetSinr - relation.sourceSinr;
    const currentA3 = relation.a3Offset;
    const currentHyst = relation.hysteresis;
    const currentTtt = relation.timeToTrigger;

    let newA3 = currentA3;
    let newHyst = currentHyst;
    let newTtt = currentTtt;
    let rationale = '';

    // Adjust A3 based on SINR delta and HO success rate
    if (relation.hoSuccessRate < 95) {
      if (sinrDelta > 6) {
        // Target is much better - handover is likely too late
        newA3 = Math.max(-6, currentA3 - 1);
        newTtt = Math.max(40, currentTtt - 40);
        rationale = 'Reducing A3 offset and TTT: target SINR is significantly better, HO may be too late';
      } else if (sinrDelta < 0) {
        // Target is worse - handover might be too early
        newA3 = Math.min(6, currentA3 + 1);
        newHyst = Math.min(6, currentHyst + 0.5);
        rationale = 'Increasing A3 offset and hysteresis: target SINR is worse, HO may be too early';
      }
    }

    // Handle ping-pong scenarios
    const mobility = currentSnapshot.mobility;
    if (mobility.pingPongHo > 10) {
      newHyst = Math.min(6, currentHyst + 1);
      newTtt = Math.min(1024, currentTtt + 80);
      rationale = rationale
        ? rationale + '; Also increasing hysteresis/TTT for ping-pong reduction'
        : 'Increasing hysteresis and TTT to reduce ping-pong handovers';
    }

    if (!rationale) {
      rationale = 'Current parameters appear optimal';
    }

    return {
      a3Offset: newA3,
      hysteresis: newHyst,
      timeToTrigger: newTtt,
      rationale,
    };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  PathLossAnalyzer,
  PowerControlValidator,
  SINRNeighborAnalyzer,
};
