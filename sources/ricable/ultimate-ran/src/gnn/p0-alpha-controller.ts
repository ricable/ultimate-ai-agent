/**
 * P0/Alpha Parameter Controller - TITAN Neuro-Symbolic RAN Platform
 *
 * Optimizes uplink power control parameters within 3GPP ranges:
 * - P0: -130 to -70 dBm (target received power at gNB)
 * - Alpha: 0 to 1 (pathloss compensation factor)
 *
 * Reference: 3GPP TS 38.213 Section 7.1.1
 */

import type {
  CellNode,
  ParameterOptimizationResult,
  ValidationResult,
  OptimizationOptions,
  OptimizationOutcome,
} from './types';

/**
 * P0/Alpha Parameter Controller
 */
export class P0AlphaController {
  // 3GPP parameter ranges
  private readonly P0_MIN = -130; // dBm
  private readonly P0_MAX = -70; // dBm
  private readonly ALPHA_MIN = 0.0;
  private readonly ALPHA_MAX = 1.0;

  // Learning history
  private history: Map<string, OptimizationOutcome[]> = new Map();

  /**
   * Optimize P0 parameter for a cell
   */
  optimizeP0(cell: CellNode): number {
    const currentP0 = cell.p0 || -106; // Default P0
    const [sinr, rsrp, prbUsage, cqi] = cell.features;

    // Decision logic based on network conditions
    let delta = 0;

    // Low SINR: reduce P0 to minimize interference
    if (sinr < 8) {
      delta = -3;
    }
    // High SINR with headroom: increase P0 for better coverage
    else if (sinr > 12) {
      delta = 2;
    }
    // Medium SINR: fine-tune based on load
    else {
      // High PRB usage: reduce P0 to offload users
      if (prbUsage > 80) {
        delta = -2;
      }
      // Low PRB usage: increase P0 for capacity
      else if (prbUsage < 50) {
        delta = 1;
      }
    }

    // Apply delta and clamp to 3GPP range
    const optimizedP0 = Math.max(this.P0_MIN, Math.min(this.P0_MAX, currentP0 + delta));

    return optimizedP0;
  }

  /**
   * Optimize Alpha parameter for a cell
   */
  optimizeAlpha(cell: CellNode): number {
    const currentAlpha = cell.alpha || 0.8; // Default Alpha
    const [sinr, rsrp, prbUsage, cqi] = cell.features;

    let delta = 0;

    // Cell edge users (low SINR, high pathloss): partial compensation
    if (sinr < 8 || rsrp < -100) {
      delta = -0.1; // Reduce Alpha to minimize edge interference
    }
    // Cell center users (high SINR, low pathloss): full compensation
    else if (sinr > 12 && rsrp > -90) {
      delta = 0.05; // Increase Alpha for uniform power
    }
    // Medium conditions: maintain balance
    else {
      delta = 0;
    }

    // Apply delta and clamp to valid range
    const optimizedAlpha = Math.max(this.ALPHA_MIN, Math.min(this.ALPHA_MAX, currentAlpha + delta));

    return optimizedAlpha;
  }

  /**
   * Jointly optimize P0 and Alpha
   */
  optimizeJoint(
    cell: CellNode,
    options: OptimizationOptions = {}
  ): ParameterOptimizationResult {
    const strategy = options.strategy || 'hybrid';

    let p0: number;
    let alpha: number;
    let rationale: string;

    if (strategy === 'gradient') {
      // Gradient-based optimization (simplified)
      p0 = this.optimizeP0(cell);
      alpha = this.optimizeAlpha(cell);
      rationale = 'Gradient-based optimization using SINR/PRB metrics';
    } else if (strategy === 'rules') {
      // Rule-based optimization
      const [sinr, rsrp, prbUsage, cqi] = cell.features;

      if (sinr < 7) {
        p0 = Math.max(this.P0_MIN, (cell.p0 || -106) - 4);
        alpha = Math.max(this.ALPHA_MIN, (cell.alpha || 0.8) - 0.15);
        rationale = 'Low SINR detected: reducing P0 and Alpha to minimize interference';
      } else if (sinr > 13) {
        p0 = Math.min(this.P0_MAX, (cell.p0 || -106) + 3);
        alpha = Math.min(this.ALPHA_MAX, (cell.alpha || 0.8) + 0.1);
        rationale = 'High SINR detected: increasing P0 and Alpha for better coverage';
      } else {
        p0 = cell.p0 || -106;
        alpha = cell.alpha || 0.8;
        rationale = 'SINR within acceptable range: maintaining current parameters';
      }
    } else {
      // Hybrid: combine rules + historical learning
      p0 = this.optimizeP0(cell);
      alpha = this.optimizeAlpha(cell);

      // Adjust based on historical outcomes
      const history = this.history.get(cell.cellId) || [];
      if (history.length > 0) {
        const successfulOutcomes = history.filter(h => h.sinrDelta > 0);
        if (successfulOutcomes.length > 0) {
          const avgP0 = successfulOutcomes.reduce((s, h) => s + h.p0, 0) / successfulOutcomes.length;
          const avgAlpha = successfulOutcomes.reduce((s, h) => s + h.alpha, 0) / successfulOutcomes.length;

          // Blend with historical success
          p0 = 0.7 * p0 + 0.3 * avgP0;
          alpha = 0.7 * alpha + 0.3 * avgAlpha;
        }
      }

      rationale = 'Hybrid optimization of P0/Alpha: rules + historical learning';
    }

    // Calculate confidence based on history
    const history = this.history.get(cell.cellId) || [];
    const confidence = history.length > 0
      ? Math.min(0.95, 0.5 + history.length * 0.05)
      : 0.6;

    return { p0, alpha, rationale, strategy, confidence };
  }

  /**
   * Validate parameter ranges
   */
  validateParameters(p0: number, alpha: number): ValidationResult {
    const violations: string[] = [];

    if (p0 < this.P0_MIN) violations.push('P0 below -130 dBm');
    if (p0 > this.P0_MAX) violations.push('P0 above -70 dBm');
    if (alpha < this.ALPHA_MIN) violations.push('Alpha below 0');
    if (alpha > this.ALPHA_MAX) violations.push('Alpha above 1');

    return {
      valid: violations.length === 0,
      violations,
    };
  }

  /**
   * Record optimization outcome for learning
   */
  recordOutcome(cell: CellNode, outcome: OptimizationOutcome) {
    const history = this.history.get(cell.cellId) || [];
    history.push({
      ...outcome,
      timestamp: Date.now(),
    });

    // Keep last 50 outcomes
    if (history.length > 50) {
      history.shift();
    }

    this.history.set(cell.cellId, history);
  }

  /**
   * Get optimization history for a cell
   */
  getHistory(cellId: string): OptimizationOutcome[] {
    return this.history.get(cellId) || [];
  }
}
