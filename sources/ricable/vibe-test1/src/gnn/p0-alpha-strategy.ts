/**
 * P0/Alpha Optimization Strategies
 *
 * Two main strategies:
 * 1. P0 Increase + Alpha Adaptation: For critical cells with SINR < 2 dB
 * 2. P0 Decrease: For cells generating high interference on neighbors
 */

import type { RealKPIGraph, PowerControlParams } from './real-kpi-graph.js';
import type { AggregatedCellKPI } from '../data/csv-loader.js';
import type { BidirectionalOutput } from './balanced-self-learning-gnn.js';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

export interface StrategyRecommendation {
  cellId: string;
  cellName: string;
  band: string;
  strategy: 'P0_INCREASE' | 'P0_DECREASE' | 'ALPHA_ONLY' | 'COMBINED' | 'NO_CHANGE';
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

  // Current state
  currentP0: number;
  currentAlpha: number;
  currentSINR: number;
  currentInterference: number;

  // Recommended changes
  recommendedP0: number;
  recommendedAlpha: number;
  p0Delta: number;
  alphaDelta: number;

  // Predicted outcomes
  predictedSINRImprovement: number;
  predictedInterferenceChange: number;

  // Impact analysis
  affectedNeighbors: Array<{
    cellId: string;
    currentSINR: number;
    predictedSINRChange: number;
  }>;

  // Rationale
  rationale: string[];
}

export interface StrategyConfig {
  // SINR thresholds
  sinrCritical: number;    // Below this = critical
  sinrTarget: number;      // Target SINR after optimization

  // P0 constraints
  p0Min: number;           // Minimum P0 (dBm)
  p0Max: number;           // Maximum P0 (dBm)
  p0StepSize: number;      // Step size for P0 changes

  // Alpha constraints
  alphaMin: number;        // Minimum alpha
  alphaMax: number;        // Maximum alpha
  alphaValues: number[];   // Valid alpha values

  // Neighbor protection
  maxNeighborDegradation: number;  // Max allowed SINR degradation on neighbors (dB)

  // Interference thresholds
  interferenceHigh: number;  // Above this = high interference (dBm)

  // Balanced optimization thresholds (NEW)
  benefitCostRatioThreshold: number;  // Only P0_DECREASE if benefit/cost > this
  minOwnMarginForHelping: number;     // Min SINR margin above critical to help neighbors
  overHelpThreshold: number;          // Don't help if neighbor already getting this much help
}

const DEFAULT_CONFIG: StrategyConfig = {
  sinrCritical: 1.0,
  sinrTarget: 5.0,

  p0Min: -106,
  p0Max: -74,
  p0StepSize: 2,

  alphaMin: 0.7,
  alphaMax: 1.0,
  alphaValues: [0.7, 0.8, 0.9, 1.0],

  maxNeighborDegradation: 1.5,

  interferenceHigh: -110,

  // Balanced optimization defaults
  benefitCostRatioThreshold: 1.5,  // Benefit must be 1.5x the cost
  minOwnMarginForHelping: 2.0,     // Need 2 dB margin above critical to help
  overHelpThreshold: 0.7,          // 70% of need already covered = don't help more
};

// ============================================================================
// PHYSICS-BASED MODELS
// ============================================================================

/**
 * Estimate SINR change from P0/Alpha modification
 * Based on Ericsson uplink power control model
 */
function estimateSINRChange(
  currentP0: number,
  currentAlpha: number,
  newP0: number,
  newAlpha: number,
  pathloss: number
): number {
  // P0 effect: ~0.15-0.2 dB SINR per dB P0 increase
  const p0Effect = (newP0 - currentP0) * 0.18;

  // Alpha effect: affects cell-edge UEs more
  // Higher alpha = more compensation = better SINR for high-pathloss UEs
  const alphaEffect = (newAlpha - currentAlpha) * 2.5;

  // Pathloss modulation: alpha effect is stronger for high pathloss
  const pathlossModulation = pathloss > 130 ? 1.3 : pathloss > 115 ? 1.1 : 1.0;

  return p0Effect + alphaEffect * pathlossModulation;
}

/**
 * Estimate interference impact on neighbors
 * Higher P0 = more UE TX power = more interference to neighbors
 */
function estimateInterferenceImpact(
  p0Delta: number,
  neighborDistance: number // Represented by adjacency weight
): number {
  // More P0 = more interference
  // Closer neighbors (higher adjacency) = more impact
  const baseImpact = p0Delta * 0.1;
  return baseImpact * neighborDistance;
}

// ============================================================================
// STRATEGY CLASS
// ============================================================================

export class P0AlphaStrategy {
  private config: StrategyConfig;
  private adjacencyList: Map<number, number[]> = new Map();

  constructor(config: Partial<StrategyConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Build adjacency list from edge index for O(1) neighbor lookups
   */
  private buildAdjacencyList(graph: RealKPIGraph): void {
    this.adjacencyList.clear();
    for (const [src, tgt] of graph.edgeIndex) {
      if (!this.adjacencyList.has(src)) {
        this.adjacencyList.set(src, []);
      }
      this.adjacencyList.get(src)!.push(tgt);
    }
  }

  /**
   * Analyze a cell and generate optimization recommendation
   */
  analyzeCell(
    cellId: string,
    graph: RealKPIGraph
  ): StrategyRecommendation {
    const cell = graph.cellKPIs.get(cellId)!;
    const params = graph.powerParams.get(cellId)!;
    const cellIdx = graph.nodeIds.indexOf(cellId);

    // Get neighbors
    const neighbors = this.getNeighbors(graph, cellIdx);

    // Determine if cell is critical
    const isCritical = cell.sinrPuschAvg < this.config.sinrCritical;
    const isHighInterference = cell.ulInterferenceAvgPusch > this.config.interferenceHigh;
    const hasHighP0 = params.p0 > -90;

    // Calculate optimal parameters
    let recommendation: StrategyRecommendation;

    if (isCritical) {
      // Strategy A: P0 Increase for critical cells
      recommendation = this.strategyP0Increase(cell, params, neighbors, graph);
    } else if (hasHighP0 && neighbors.length > 2) {
      // Strategy B: P0 Decrease to reduce interference on neighbors
      recommendation = this.strategyP0Decrease(cell, params, neighbors, graph);
    } else {
      // No change needed
      recommendation = this.createNoChangeRecommendation(cell, params);
    }

    return recommendation;
  }

  /**
   * Strategy A: P0 Increase for critical cells
   */
  private strategyP0Increase(
    cell: AggregatedCellKPI,
    params: PowerControlParams,
    neighbors: Array<{ cellId: string; weight: number; sinr: number }>,
    graph: RealKPIGraph
  ): StrategyRecommendation {
    const rationale: string[] = [];
    rationale.push(`Cell has critical SINR: ${cell.sinrPuschAvg.toFixed(1)} dB < ${this.config.sinrCritical} dB threshold`);

    // Calculate required P0 increase
    const sinrDeficit = this.config.sinrTarget - cell.sinrPuschAvg;
    const requiredP0Increase = Math.ceil(sinrDeficit / 0.18);

    // Constrain P0 increase
    let newP0 = Math.min(
      this.config.p0Max,
      params.p0 + Math.min(requiredP0Increase, 15)
    );

    // Round to step size
    newP0 = Math.round(newP0 / this.config.p0StepSize) * this.config.p0StepSize;

    // Determine optimal alpha based on pathloss
    let newAlpha = params.alpha;
    if (cell.pathlossDistribution.sup130 > 20) {
      // High pathloss environment: increase alpha
      newAlpha = Math.min(1.0, params.alpha + 0.1);
      rationale.push(`High pathloss detected (${cell.pathlossDistribution.sup130.toFixed(0)}% > 130dB), increasing alpha`);
    } else if (cell.pathlossDistribution.inf100 > 50) {
      // Low pathloss environment: can reduce alpha
      newAlpha = Math.max(0.7, params.alpha - 0.1);
      rationale.push(`Low pathloss environment, optimizing alpha for balanced coverage`);
    }

    // Snap to valid alpha
    newAlpha = this.snapToValidAlpha(newAlpha);

    // Calculate predicted SINR improvement
    const predictedSINRImprovement = estimateSINRChange(
      params.p0,
      params.alpha,
      newP0,
      newAlpha,
      cell.ulPathlossAvg
    );

    // Calculate neighbor impact
    const affectedNeighbors = neighbors.map(n => {
      const impact = estimateInterferenceImpact(newP0 - params.p0, n.weight);
      return {
        cellId: n.cellId,
        currentSINR: n.sinr,
        predictedSINRChange: -impact, // Negative = degradation
      };
    });

    // Check neighbor constraint
    const maxNeighborDegradation = Math.max(
      ...affectedNeighbors.map(n => -n.predictedSINRChange),
      0
    );

    if (maxNeighborDegradation > this.config.maxNeighborDegradation) {
      // Reduce P0 increase to protect neighbors
      const reduction = Math.ceil((maxNeighborDegradation - this.config.maxNeighborDegradation) / 0.1);
      newP0 = Math.max(params.p0 + 2, newP0 - reduction);
      rationale.push(`P0 increase limited to protect neighbors (max degradation: ${this.config.maxNeighborDegradation} dB)`);
    }

    rationale.push(`P0 increase: ${params.p0} → ${newP0} dBm (+${newP0 - params.p0} dB)`);
    if (newAlpha !== params.alpha) {
      rationale.push(`Alpha adjustment: ${params.alpha} → ${newAlpha}`);
    }

    return {
      cellId: cell.cellId,
      cellName: cell.cellName,
      band: cell.band,
      strategy: newAlpha !== params.alpha ? 'COMBINED' : 'P0_INCREASE',
      priority: cell.sinrPuschAvg < 0 ? 'CRITICAL' : 'HIGH',
      currentP0: params.p0,
      currentAlpha: params.alpha,
      currentSINR: cell.sinrPuschAvg,
      currentInterference: cell.ulInterferenceAvgPusch,
      recommendedP0: newP0,
      recommendedAlpha: newAlpha,
      p0Delta: newP0 - params.p0,
      alphaDelta: newAlpha - params.alpha,
      predictedSINRImprovement,
      predictedInterferenceChange: (newP0 - params.p0) * 0.3, // Estimate
      affectedNeighbors,
      rationale,
    };
  }

  /**
   * Strategy B: P0 Decrease to reduce interference on neighbors
   */
  private strategyP0Decrease(
    cell: AggregatedCellKPI,
    params: PowerControlParams,
    neighbors: Array<{ cellId: string; weight: number; sinr: number }>,
    graph: RealKPIGraph
  ): StrategyRecommendation {
    const rationale: string[] = [];
    rationale.push(`Cell has high P0 (${params.p0} dBm) with ${neighbors.length} neighbors`);

    // Find critical neighbors that would benefit from interference reduction
    const criticalNeighbors = neighbors.filter(n => n.sinr < this.config.sinrCritical);

    if (criticalNeighbors.length === 0) {
      return this.createNoChangeRecommendation(cell, params);
    }

    rationale.push(`${criticalNeighbors.length} neighbors have critical SINR and would benefit from interference reduction`);

    // Calculate safe P0 decrease (don't drop own SINR below threshold)
    const safetyMargin = cell.sinrPuschAvg - this.config.sinrCritical;
    const maxP0Decrease = Math.min(10, Math.floor(safetyMargin / 0.18));

    if (maxP0Decrease <= 0) {
      rationale.push(`Cannot decrease P0: own SINR margin insufficient`);
      return this.createNoChangeRecommendation(cell, params);
    }

    // Apply decrease
    const p0Decrease = Math.min(maxP0Decrease, 6); // Conservative decrease
    const newP0 = Math.max(this.config.p0Min, params.p0 - p0Decrease);

    // Keep alpha same for P0 decrease strategy
    const newAlpha = params.alpha;

    // Calculate impact
    const predictedSINRImprovement = estimateSINRChange(
      params.p0,
      params.alpha,
      newP0,
      newAlpha,
      cell.ulPathlossAvg
    );

    const affectedNeighbors = neighbors.map(n => {
      const benefit = estimateInterferenceImpact(params.p0 - newP0, n.weight);
      return {
        cellId: n.cellId,
        currentSINR: n.sinr,
        predictedSINRChange: benefit, // Positive = improvement
      };
    });

    rationale.push(`P0 decrease: ${params.p0} → ${newP0} dBm (-${p0Decrease} dB)`);
    rationale.push(`Predicted neighbor SINR improvement: ${affectedNeighbors.filter(n => n.predictedSINRChange > 0.2).length} cells`);

    return {
      cellId: cell.cellId,
      cellName: cell.cellName,
      band: cell.band,
      strategy: 'P0_DECREASE',
      priority: criticalNeighbors.length > 3 ? 'HIGH' : 'MEDIUM',
      currentP0: params.p0,
      currentAlpha: params.alpha,
      currentSINR: cell.sinrPuschAvg,
      currentInterference: cell.ulInterferenceAvgPusch,
      recommendedP0: newP0,
      recommendedAlpha: newAlpha,
      p0Delta: newP0 - params.p0,
      alphaDelta: 0,
      predictedSINRImprovement, // Will be negative (acceptable trade-off)
      predictedInterferenceChange: -(params.p0 - newP0) * 0.3,
      affectedNeighbors,
      rationale,
    };
  }

  /**
   * Create no-change recommendation
   */
  private createNoChangeRecommendation(
    cell: AggregatedCellKPI,
    params: PowerControlParams
  ): StrategyRecommendation {
    return {
      cellId: cell.cellId,
      cellName: cell.cellName,
      band: cell.band,
      strategy: 'NO_CHANGE',
      priority: 'LOW',
      currentP0: params.p0,
      currentAlpha: params.alpha,
      currentSINR: cell.sinrPuschAvg,
      currentInterference: cell.ulInterferenceAvgPusch,
      recommendedP0: params.p0,
      recommendedAlpha: params.alpha,
      p0Delta: 0,
      alphaDelta: 0,
      predictedSINRImprovement: 0,
      predictedInterferenceChange: 0,
      affectedNeighbors: [],
      rationale: ['Current parameters are optimal'],
    };
  }

  /**
   * Get neighbors of a cell (optimized using pre-built adjacency list)
   */
  private getNeighbors(
    graph: RealKPIGraph,
    cellIdx: number
  ): Array<{ cellId: string; weight: number; sinr: number }> {
    const neighbors: Array<{ cellId: string; weight: number; sinr: number }> = [];

    // O(1) lookup using pre-built adjacency list
    const neighborIndices = this.adjacencyList.get(cellIdx) || [];

    for (const tgt of neighborIndices) {
      const neighborId = graph.nodeIds[tgt];
      const neighborCell = graph.cellKPIs.get(neighborId);

      neighbors.push({
        cellId: neighborId,
        weight: 0.5, // Default weight
        sinr: neighborCell?.sinrPuschAvg || 0,
      });
    }

    return neighbors.sort((a, b) => b.sinr - a.sinr); // Sort by SINR for prioritization
  }

  /**
   * Snap alpha to valid 3GPP value
   */
  private snapToValidAlpha(alpha: number): number {
    return this.config.alphaValues.reduce((prev, curr) =>
      Math.abs(curr - alpha) < Math.abs(prev - alpha) ? curr : prev
    );
  }

  /**
   * Analyze entire network and generate recommendations
   */
  analyzeNetwork(graph: RealKPIGraph): StrategyRecommendation[] {
    console.log(`[P0AlphaStrategy] Building adjacency list for ${graph.metadata.numNodes} nodes...`);
    const startTime = Date.now();

    // Build adjacency list once for O(1) neighbor lookups
    this.buildAdjacencyList(graph);

    console.log(`[P0AlphaStrategy] Adjacency list built in ${Date.now() - startTime}ms`);
    console.log(`[P0AlphaStrategy] Analyzing ${graph.nodeIds.length} cells...`);

    const recommendations: StrategyRecommendation[] = [];
    let processed = 0;

    for (const cellId of graph.nodeIds) {
      const rec = this.analyzeCell(cellId, graph);
      if (rec.strategy !== 'NO_CHANGE') {
        recommendations.push(rec);
      }

      processed++;
      if (processed % 5000 === 0) {
        console.log(`[P0AlphaStrategy] Processed ${processed}/${graph.nodeIds.length} cells`);
      }
    }

    console.log(`[P0AlphaStrategy] Analysis complete in ${Date.now() - startTime}ms`);

    // Sort by priority
    const priorityOrder = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
    recommendations.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

    return recommendations;
  }

  // ============================================================================
  // BALANCED ANALYSIS METHODS (Bidirectional-Aware)
  // ============================================================================

  /**
   * Analyze a cell with bidirectional impact information from the GNN
   * Uses receivedBenefit and sentCost to make balanced decisions
   */
  analyzeCellBalanced(
    cellId: string,
    graph: RealKPIGraph,
    bipOutput: BidirectionalOutput,
    cellIdx: number
  ): StrategyRecommendation {
    const cell = graph.cellKPIs.get(cellId)!;
    const params = graph.powerParams.get(cellId)!;

    // Get bidirectional impact information
    const receivedBenefit = bipOutput.receivedBenefit[cellIdx] || 0;
    const sentCost = bipOutput.sentCost[cellIdx] || 0;
    const proposedImpact = bipOutput.proposedImpact[cellIdx] || 0;
    const feedback = bipOutput.feedback?.[cellIdx] || 0;

    // Get neighbors
    const neighbors = this.getNeighbors(graph, cellIdx);
    const isCritical = cell.sinrPuschAvg < this.config.sinrCritical;
    const ownMargin = cell.sinrPuschAvg - this.config.sinrCritical;

    if (isCritical) {
      // Critical cell - may need P0 increase, but check if already receiving help
      return this.strategyP0IncreaseBalanced(
        cell, params, neighbors, graph, receivedBenefit, cellIdx, bipOutput
      );
    } else if (ownMargin >= this.config.minOwnMarginForHelping) {
      // Non-critical with margin - may help neighbors, but check benefit/cost
      return this.strategyP0DecreaseBalanced(
        cell, params, neighbors, graph, receivedBenefit, sentCost, feedback, cellIdx, bipOutput
      );
    } else {
      // Marginal cell - don't risk own SINR
      return this.createNoChangeRecommendation(cell, params);
    }
  }

  /**
   * Strategy A (Balanced): P0 Increase for critical cells
   * Reduces P0 increase if already receiving sufficient help from neighbors
   */
  private strategyP0IncreaseBalanced(
    cell: AggregatedCellKPI,
    params: PowerControlParams,
    neighbors: Array<{ cellId: string; weight: number; sinr: number }>,
    graph: RealKPIGraph,
    receivedBenefit: number,
    cellIdx: number,
    bipOutput: BidirectionalOutput
  ): StrategyRecommendation {
    const rationale: string[] = [];
    rationale.push(`Cell has critical SINR: ${cell.sinrPuschAvg.toFixed(1)} dB`);

    // Calculate remaining SINR need after accounting for help from neighbors
    const sinrNeed = this.config.sinrTarget - cell.sinrPuschAvg;
    const effectiveHelp = receivedBenefit * 0.8; // 80% efficiency of neighbor help
    const remainingNeed = Math.max(0, sinrNeed - effectiveHelp);

    if (receivedBenefit > 0.5) {
      rationale.push(`Receiving ${receivedBenefit.toFixed(1)} dB help from neighbors`);
    }

    // If already getting enough help, only make a small adjustment
    if (remainingNeed < 1.0) {
      rationale.push(`Neighbor help sufficient, minimal P0 increase needed`);

      // Small increase or just alpha adjustment
      const smallP0Increase = Math.max(0, Math.ceil(remainingNeed / 0.18));
      const newP0 = Math.min(
        this.config.p0Max,
        params.p0 + Math.min(smallP0Increase, 4)
      );

      // Round to step size
      const roundedP0 = Math.round(newP0 / this.config.p0StepSize) * this.config.p0StepSize;

      // Determine optimal alpha
      let newAlpha = params.alpha;
      if (cell.pathlossDistribution.sup130 > 20 && params.alpha < 1.0) {
        newAlpha = Math.min(1.0, params.alpha + 0.1);
        rationale.push(`Increasing alpha for high pathloss UEs`);
      }

      newAlpha = this.snapToValidAlpha(newAlpha);

      if (roundedP0 === params.p0 && newAlpha === params.alpha) {
        return this.createNoChangeRecommendation(cell, params);
      }

      const predictedSINRImprovement = estimateSINRChange(
        params.p0, params.alpha, roundedP0, newAlpha, cell.ulPathlossAvg
      );

      return {
        cellId: cell.cellId,
        cellName: cell.cellName,
        band: cell.band,
        strategy: roundedP0 > params.p0 ? 'P0_INCREASE' : 'ALPHA_ONLY',
        priority: 'MEDIUM',
        currentP0: params.p0,
        currentAlpha: params.alpha,
        currentSINR: cell.sinrPuschAvg,
        currentInterference: cell.ulInterferenceAvgPusch,
        recommendedP0: roundedP0,
        recommendedAlpha: newAlpha,
        p0Delta: roundedP0 - params.p0,
        alphaDelta: newAlpha - params.alpha,
        predictedSINRImprovement,
        predictedInterferenceChange: (roundedP0 - params.p0) * 0.3,
        affectedNeighbors: [],
        rationale,
      };
    }

    // Full P0 increase needed
    const requiredP0Increase = Math.ceil(remainingNeed / 0.18);
    let newP0 = Math.min(
      this.config.p0Max,
      params.p0 + Math.min(requiredP0Increase, 15)
    );
    newP0 = Math.round(newP0 / this.config.p0StepSize) * this.config.p0StepSize;

    let newAlpha = params.alpha;
    if (cell.pathlossDistribution.sup130 > 20) {
      newAlpha = Math.min(1.0, params.alpha + 0.1);
      rationale.push(`High pathloss detected, increasing alpha`);
    }
    newAlpha = this.snapToValidAlpha(newAlpha);

    const predictedSINRImprovement = estimateSINRChange(
      params.p0, params.alpha, newP0, newAlpha, cell.ulPathlossAvg
    );

    // Calculate neighbor impact
    const affectedNeighbors = neighbors.slice(0, 5).map(n => ({
      cellId: n.cellId,
      currentSINR: n.sinr,
      predictedSINRChange: -estimateInterferenceImpact(newP0 - params.p0, n.weight),
    }));

    rationale.push(`P0 increase: ${params.p0} → ${newP0} dBm (+${newP0 - params.p0} dB)`);

    return {
      cellId: cell.cellId,
      cellName: cell.cellName,
      band: cell.band,
      strategy: newAlpha !== params.alpha ? 'COMBINED' : 'P0_INCREASE',
      priority: cell.sinrPuschAvg < 0 ? 'CRITICAL' : 'HIGH',
      currentP0: params.p0,
      currentAlpha: params.alpha,
      currentSINR: cell.sinrPuschAvg,
      currentInterference: cell.ulInterferenceAvgPusch,
      recommendedP0: newP0,
      recommendedAlpha: newAlpha,
      p0Delta: newP0 - params.p0,
      alphaDelta: newAlpha - params.alpha,
      predictedSINRImprovement,
      predictedInterferenceChange: (newP0 - params.p0) * 0.3,
      affectedNeighbors,
      rationale,
    };
  }

  /**
   * Strategy B (Balanced): P0 Decrease to help neighbors
   * Only recommends decrease if:
   * 1. Neighbors still need help (not already over-helped)
   * 2. Benefit/cost ratio > threshold
   * 3. Own SINR margin is sufficient
   */
  private strategyP0DecreaseBalanced(
    cell: AggregatedCellKPI,
    params: PowerControlParams,
    neighbors: Array<{ cellId: string; weight: number; sinr: number }>,
    graph: RealKPIGraph,
    receivedBenefit: number,
    sentCost: number,
    feedback: number,
    cellIdx: number,
    bipOutput: BidirectionalOutput
  ): StrategyRecommendation {
    const rationale: string[] = [];
    const ownMargin = cell.sinrPuschAvg - this.config.sinrCritical;

    // Find critical neighbors that still need help
    const criticalNeighbors = neighbors.filter(n => {
      const nIdx = graph.nodeIds.indexOf(n.cellId);
      const neighborReceivedBenefit = nIdx >= 0 ? (bipOutput.receivedBenefit[nIdx] || 0) : 0;
      const neighborSinrNeed = Math.max(0, this.config.sinrTarget - n.sinr);

      // Skip if neighbor already getting enough help
      if (neighborSinrNeed > 0 && neighborReceivedBenefit / neighborSinrNeed >= this.config.overHelpThreshold) {
        return false;
      }

      return n.sinr < this.config.sinrCritical;
    });

    if (criticalNeighbors.length === 0) {
      rationale.push(`No critical neighbors needing help`);
      return this.createNoChangeRecommendation(cell, params);
    }

    // Check if feedback indicates we should reduce helping
    if (feedback > 0.5) {
      rationale.push(`Feedback indicates neighbors being over-helped (${feedback.toFixed(1)} dB excess)`);
      return this.createNoChangeRecommendation(cell, params);
    }

    // Calculate potential benefit to neighbors
    const avgNeighborSinrNeed = criticalNeighbors.reduce(
      (sum, n) => sum + Math.max(0, this.config.sinrTarget - n.sinr), 0
    ) / criticalNeighbors.length;

    // Conservative P0 decrease
    const maxP0Decrease = Math.min(8, Math.floor(ownMargin / 0.18));
    if (maxP0Decrease <= 2) {
      rationale.push(`Own SINR margin insufficient for P0 decrease`);
      return this.createNoChangeRecommendation(cell, params);
    }

    // Estimate benefit and cost
    const avgCouplingStrength = 0.5; // Could be computed from edge features
    const potentialBenefit = maxP0Decrease * 0.15 * avgCouplingStrength * criticalNeighbors.length;
    const ownCost = maxP0Decrease * 0.18;

    // Check benefit/cost ratio
    const benefitCostRatio = potentialBenefit / ownCost;
    if (benefitCostRatio < this.config.benefitCostRatioThreshold) {
      rationale.push(`Benefit/cost ratio too low: ${benefitCostRatio.toFixed(2)} < ${this.config.benefitCostRatioThreshold}`);
      return this.createNoChangeRecommendation(cell, params);
    }

    rationale.push(`${criticalNeighbors.length} critical neighbors would benefit`);
    rationale.push(`Benefit/cost ratio: ${benefitCostRatio.toFixed(2)}`);

    // Apply moderated P0 decrease
    const moderatedDecrease = Math.min(maxP0Decrease, Math.ceil(avgNeighborSinrNeed / 0.15));
    const p0Decrease = Math.max(2, Math.min(6, moderatedDecrease));
    const newP0 = Math.max(this.config.p0Min, params.p0 - p0Decrease);

    const predictedSINRImprovement = estimateSINRChange(
      params.p0, params.alpha, newP0, params.alpha, cell.ulPathlossAvg
    );

    const affectedNeighbors = criticalNeighbors.slice(0, 5).map(n => ({
      cellId: n.cellId,
      currentSINR: n.sinr,
      predictedSINRChange: estimateInterferenceImpact(p0Decrease, n.weight),
    }));

    rationale.push(`P0 decrease: ${params.p0} → ${newP0} dBm (-${p0Decrease} dB)`);
    rationale.push(`Predicted own SINR change: ${predictedSINRImprovement.toFixed(1)} dB`);

    return {
      cellId: cell.cellId,
      cellName: cell.cellName,
      band: cell.band,
      strategy: 'P0_DECREASE',
      priority: criticalNeighbors.length > 3 ? 'HIGH' : 'MEDIUM',
      currentP0: params.p0,
      currentAlpha: params.alpha,
      currentSINR: cell.sinrPuschAvg,
      currentInterference: cell.ulInterferenceAvgPusch,
      recommendedP0: newP0,
      recommendedAlpha: params.alpha,
      p0Delta: newP0 - params.p0,
      alphaDelta: 0,
      predictedSINRImprovement,
      predictedInterferenceChange: -(p0Decrease) * 0.3,
      affectedNeighbors,
      rationale,
    };
  }

  /**
   * Analyze entire network with bidirectional impact information
   */
  analyzeNetworkBalanced(
    graph: RealKPIGraph,
    bipOutput: BidirectionalOutput
  ): StrategyRecommendation[] {
    console.log(`[P0AlphaStrategy] Balanced analysis for ${graph.metadata.numNodes} nodes...`);
    const startTime = Date.now();

    // Build adjacency list once for O(1) neighbor lookups
    this.buildAdjacencyList(graph);

    const recommendations: StrategyRecommendation[] = [];
    let processed = 0;
    let p0IncreaseCount = 0;
    let p0DecreaseCount = 0;
    let noChangeCount = 0;

    for (let cellIdx = 0; cellIdx < graph.nodeIds.length; cellIdx++) {
      const cellId = graph.nodeIds[cellIdx];
      const rec = this.analyzeCellBalanced(cellId, graph, bipOutput, cellIdx);

      if (rec.strategy === 'P0_INCREASE' || rec.strategy === 'COMBINED') {
        p0IncreaseCount++;
        recommendations.push(rec);
      } else if (rec.strategy === 'P0_DECREASE') {
        p0DecreaseCount++;
        recommendations.push(rec);
      } else {
        noChangeCount++;
      }

      processed++;
      if (processed % 5000 === 0) {
        console.log(`[P0AlphaStrategy] Processed ${processed}/${graph.nodeIds.length} cells`);
      }
    }

    console.log(`[P0AlphaStrategy] Balanced analysis complete in ${Date.now() - startTime}ms`);
    console.log(`[P0AlphaStrategy] Distribution: P0_INCREASE=${p0IncreaseCount}, P0_DECREASE=${p0DecreaseCount}, NO_CHANGE=${noChangeCount}`);

    // Sort by priority
    const priorityOrder = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3 };
    recommendations.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

    return recommendations;
  }

  /**
   * Get strategy distribution statistics
   */
  getStrategyDistribution(recommendations: StrategyRecommendation[]): {
    p0Increase: number;
    p0Decrease: number;
    alphaOnly: number;
    combined: number;
    noChange: number;
    total: number;
    p0DecreaseRatio: number;
  } {
    const counts = {
      p0Increase: 0,
      p0Decrease: 0,
      alphaOnly: 0,
      combined: 0,
      noChange: 0,
    };

    for (const rec of recommendations) {
      switch (rec.strategy) {
        case 'P0_INCREASE': counts.p0Increase++; break;
        case 'P0_DECREASE': counts.p0Decrease++; break;
        case 'ALPHA_ONLY': counts.alphaOnly++; break;
        case 'COMBINED': counts.combined++; break;
        case 'NO_CHANGE': counts.noChange++; break;
      }
    }

    const total = recommendations.length;
    return {
      ...counts,
      total,
      p0DecreaseRatio: total > 0 ? (counts.p0Decrease / total) * 100 : 0,
    };
  }
}

export default P0AlphaStrategy;
