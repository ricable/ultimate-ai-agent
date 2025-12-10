/**
 * Lyapunov Stability Analyzer
 * TITAN Neuro-Symbolic RAN Platform
 *
 * Calculates Lyapunov exponent to detect chaos onset in parameter changes.
 * Positive exponent = mathematical signature of instability (chaotic behavior).
 *
 * Algorithm:
 * λ = lim (1/n) Σ log|δx_i+1 / δx_i|
 *
 * Where:
 * - λ > 0: Chaotic (exponential divergence of nearby trajectories)
 * - λ = 0: Neutral (periodic or quasi-periodic)
 * - λ < 0: Stable (convergence to fixed point or limit cycle)
 */

export interface LyapunovConfig {
  lyapunov_max: number;
  minSteps?: number; // Minimum simulation steps for reliable calculation
}

export interface SimulationResult {
  id: string;
  steps: SimulationStep[];
  finalState?: SimulationStep;
}

export interface SimulationStep {
  step: number;
  kpis: {
    throughput: number;
    bler: number;
    interference: number;
  };
}

export interface LyapunovResult {
  exponent: number;
  stable: boolean;
  interpretation: 'STABLE' | 'CHAOTIC';
  reliable: boolean;
  reason?: string;
}

export class LyapunovAnalyzer {
  private config: LyapunovConfig;
  private readonly MIN_STEPS = 10; // Minimum steps for reliable calculation

  constructor(config: LyapunovConfig) {
    this.config = {
      ...config,
      minSteps: config.minSteps ?? this.MIN_STEPS,
    };
  }

  /**
   * Analyze simulation for Lyapunov stability
   */
  async analyze(simulation: SimulationResult): Promise<LyapunovResult> {
    // Extract KPI time series (use throughput as primary metric)
    const states = simulation.steps.map((s) => s.kpis.throughput);

    // Check if we have enough data points
    if (states.length < this.config.minSteps!) {
      return {
        exponent: 0,
        stable: true,
        interpretation: 'STABLE',
        reliable: false,
        reason: `Insufficient simulation steps (${states.length} < ${this.config.minSteps} minimum)`,
      };
    }

    // Calculate Lyapunov exponent
    const exponent = this.calculateExponent(states);

    // Determine stability
    const stable = exponent <= this.config.lyapunov_max;
    const interpretation = exponent > 0 ? 'CHAOTIC' : 'STABLE';

    return {
      exponent,
      stable,
      interpretation,
      reliable: true,
    };
  }

  /**
   * Calculate Lyapunov exponent from state trajectory
   *
   * Simplified calculation:
   * λ = (1/N) Σ log|x_i - x_{i-1}|
   *
   * Note: This is a simplified version. Full implementation would use:
   * - Multiple nearby trajectories
   * - Renormalization to prevent overflow
   * - Multiple initial conditions
   */
  calculateExponent(states: number[]): number {
    if (states.length < 2) {
      return 0;
    }

    let sumLog = 0;
    let validDeltas = 0;

    for (let i = 1; i < states.length; i++) {
      const delta = Math.abs(states[i] - states[i - 1]);

      // Only include non-zero deltas to avoid log(0)
      if (delta > 0) {
        sumLog += Math.log(delta);
        validDeltas++;
      }
    }

    // Average the log sum
    if (validDeltas === 0) {
      return 0; // All states identical = perfectly stable
    }

    return sumLog / validDeltas;
  }

  /**
   * Calculate Lyapunov exponent with multiple metrics (multi-dimensional)
   *
   * For more robust analysis, consider multiple KPIs:
   * - Throughput
   * - BLER
   * - Interference
   */
  calculateMultiDimensionalExponent(simulation: SimulationResult): number {
    const throughputStates = simulation.steps.map((s) => s.kpis.throughput);
    const blerStates = simulation.steps.map((s) => s.kpis.bler);
    const interferenceStates = simulation.steps.map((s) => s.kpis.interference);

    const exponentThroughput = this.calculateExponent(throughputStates);
    const exponentBler = this.calculateExponent(blerStates);
    const exponentInterference = this.calculateExponent(interferenceStates);

    // Take maximum exponent (most conservative)
    return Math.max(exponentThroughput, exponentBler, exponentInterference);
  }

  /**
   * Detect if system is approaching chaos (early warning)
   *
   * Returns true if exponent is positive but still below threshold
   */
  isApproachingChaos(exponent: number): boolean {
    return exponent > 0 && exponent <= this.config.lyapunov_max;
  }

  /**
   * Classify stability level
   */
  classifyStability(exponent: number): {
    level: 'HIGHLY_STABLE' | 'STABLE' | 'NEUTRAL' | 'UNSTABLE' | 'CHAOTIC';
    description: string;
  } {
    if (exponent < -0.5) {
      return {
        level: 'HIGHLY_STABLE',
        description: 'System rapidly converges to equilibrium',
      };
    } else if (exponent < 0) {
      return {
        level: 'STABLE',
        description: 'System converges to stable state',
      };
    } else if (exponent === 0) {
      return {
        level: 'NEUTRAL',
        description: 'System is periodic or quasi-periodic',
      };
    } else if (exponent < 0.5) {
      return {
        level: 'UNSTABLE',
        description: 'System shows signs of instability',
      };
    } else {
      return {
        level: 'CHAOTIC',
        description: 'System exhibits chaotic behavior',
      };
    }
  }
}
