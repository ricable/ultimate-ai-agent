/**
 * Uplink Power Control Optimizer
 *
 * Implements closed-loop Deep Reinforcement Learning for optimizing
 * P0, Alpha, PUSCH, and PUCCH parameters according to 3GPP TS 38.213.
 *
 * The Physics of Uplink Power:
 * P_PUSCH = min{P_CMAX, P_0 + α·PL + Δ_TF + f(i)}
 *
 * Key parameters:
 * - P_0: Target received power at gNodeB
 * - Alpha (α): Pathloss compensation factor (0.4-1.0)
 * - PUSCH/PUCCH power offsets
 *
 * Optimization targets:
 * - SINR distribution (10th, 50th, 90th percentiles)
 * - IoT (Interference over Thermal) minimization
 * - Edge user fairness (5th percentile throughput)
 * - BLER targets per slice type
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import {
  UplinkPowerControlParams,
  CellKPIs,
  SliceIdentity,
  Action,
  ActionRecord,
  OptimizationState,
  OptimizationAction,
  OptimizationReward
} from '../../types/index.js';
import { DecisionTransformer, TrajectorySegment } from '../../core/agentdb/index.js';

// ============================================================================
// STATE ENCODER
// ============================================================================

interface UplinkStateFeatures {
  // SINR distribution
  sinrP10: number;        // 10th percentile
  sinrP50: number;        // Median
  sinrP90: number;        // 90th percentile

  // Pathloss distribution
  pathlossP10: number;
  pathlossP50: number;
  pathlossP90: number;

  // Interference
  iotLevel: number;       // Interference over Thermal (dB)
  rtwp: number;           // Received Total Wideband Power

  // Throughput
  ulThroughputAvg: number;
  ulThroughputP5: number; // 5th percentile (edge)

  // Error rates
  blerUl: number;

  // Load
  prbUtilizationUl: number;
  activeUsers: number;

  // Current parameters
  currentP0: number;
  currentAlpha: number;

  // Neighbor state (aggregated)
  neighborAvgIoT: number;
  neighborAvgLoad: number;
}

class StateEncoder {
  private stateSize: number = 32;

  /**
   * Encode raw KPIs into normalized state vector for RL
   */
  encode(
    cellKpis: CellKPIs,
    neighborKpis: CellKPIs[],
    currentParams: UplinkPowerControlParams
  ): number[] {
    const state = new Array(this.stateSize).fill(0);

    // SINR features (normalized to -1 to 1 range)
    state[0] = this.normalize(cellKpis.avgSinrUl, -10, 30, -1, 1);

    // IoT and RTWP
    state[1] = this.normalize(cellKpis.iotUl, -6, 20, 0, 1);
    state[2] = this.normalize(cellKpis.rtwp, -120, -80, 0, 1);

    // Throughput
    state[3] = this.normalize(cellKpis.ulThroughput, 0, 100, 0, 1);
    state[4] = this.normalize(cellKpis.ulUserThroughput5Pct, 0, 50, 0, 1);

    // Error rate
    state[5] = cellKpis.blerUl;

    // Load
    state[6] = cellKpis.prbUtilizationUl / 100;
    state[7] = this.normalize(cellKpis.activeUsers, 0, 500, 0, 1);

    // Current parameters
    state[8] = this.normalize(currentParams.p0NominalPusch, -126, -60, 0, 1);
    state[9] = (currentParams.alpha - 0.4) / 0.6; // Normalize 0.4-1.0 to 0-1

    // Neighbor aggregates
    if (neighborKpis.length > 0) {
      const avgIoT = neighborKpis.reduce((sum, n) => sum + n.iotUl, 0) / neighborKpis.length;
      const avgLoad = neighborKpis.reduce((sum, n) => sum + n.prbUtilizationUl, 0) / neighborKpis.length;

      state[10] = this.normalize(avgIoT, -6, 20, 0, 1);
      state[11] = avgLoad / 100;
    }

    // Additional derived features
    state[12] = cellKpis.callDropRate;
    state[13] = this.normalize(cellKpis.avgRsrp, -120, -60, 0, 1);

    // Time features (hour of day encoded cyclically)
    const hour = cellKpis.timestamp.getHours();
    state[14] = Math.sin(2 * Math.PI * hour / 24);
    state[15] = Math.cos(2 * Math.PI * hour / 24);

    return state;
  }

  /**
   * Extract full state features for analysis
   */
  extractFeatures(
    cellKpis: CellKPIs,
    neighborKpis: CellKPIs[],
    currentParams: UplinkPowerControlParams
  ): UplinkStateFeatures {
    const neighborAvgIoT = neighborKpis.length > 0
      ? neighborKpis.reduce((sum, n) => sum + n.iotUl, 0) / neighborKpis.length
      : 0;
    const neighborAvgLoad = neighborKpis.length > 0
      ? neighborKpis.reduce((sum, n) => sum + n.prbUtilizationUl, 0) / neighborKpis.length
      : 0;

    return {
      sinrP10: cellKpis.avgSinrUl - 5,  // Approximate from average
      sinrP50: cellKpis.avgSinrUl,
      sinrP90: cellKpis.avgSinrUl + 5,
      pathlossP10: 80,                   // Would come from detailed UE reports
      pathlossP50: 100,
      pathlossP90: 120,
      iotLevel: cellKpis.iotUl,
      rtwp: cellKpis.rtwp,
      ulThroughputAvg: cellKpis.ulThroughput,
      ulThroughputP5: cellKpis.ulUserThroughput5Pct,
      blerUl: cellKpis.blerUl,
      prbUtilizationUl: cellKpis.prbUtilizationUl,
      activeUsers: cellKpis.activeUsers,
      currentP0: currentParams.p0NominalPusch,
      currentAlpha: currentParams.alpha,
      neighborAvgIoT,
      neighborAvgLoad
    };
  }

  getStateSize(): number {
    return this.stateSize;
  }

  private normalize(value: number, min: number, max: number, outMin: number, outMax: number): number {
    const clipped = Math.max(min, Math.min(max, value));
    return outMin + (clipped - min) / (max - min) * (outMax - outMin);
  }
}

// ============================================================================
// REWARD CALCULATOR
// ============================================================================

interface RewardWeights {
  throughputAvg: number;
  throughputEdge: number;
  interferenceNeighbor: number;
  blerPenalty: number;
  stabilityBonus: number;
}

class RewardCalculator {
  weights: RewardWeights;
  targetBler: number;

  constructor(weights?: Partial<RewardWeights>) {
    this.weights = {
      throughputAvg: 0.3,
      throughputEdge: 0.4,      // Higher weight for fairness
      interferenceNeighbor: -0.2,
      blerPenalty: -0.3,
      stabilityBonus: 0.1,
      ...weights
    };
    this.targetBler = 0.1;
  }

  /**
   * Calculate reward based on KPI changes
   */
  calculate(
    prevKpis: CellKPIs,
    currKpis: CellKPIs,
    neighborIoTDelta: number
  ): OptimizationReward {
    // Throughput improvements (normalized)
    const throughputAvgDelta = (currKpis.ulThroughput - prevKpis.ulThroughput) / (prevKpis.ulThroughput + 1);
    const throughputEdgeDelta = (currKpis.ulUserThroughput5Pct - prevKpis.ulUserThroughput5Pct) / (prevKpis.ulUserThroughput5Pct + 1);

    // BLER penalty (exponential if above target)
    let blerPenalty = 0;
    if (currKpis.blerUl > this.targetBler) {
      blerPenalty = Math.pow(currKpis.blerUl - this.targetBler, 2) * 10;
    }

    // Stability bonus (small changes are preferred)
    const stabilityBonus = Math.exp(-Math.abs(throughputAvgDelta) * 2);

    // Calculate total reward
    const totalReward =
      this.weights.throughputAvg * throughputAvgDelta +
      this.weights.throughputEdge * throughputEdgeDelta +
      this.weights.interferenceNeighbor * neighborIoTDelta +
      this.weights.blerPenalty * blerPenalty +
      this.weights.stabilityBonus * stabilityBonus;

    return {
      throughputAvg: throughputAvgDelta,
      throughputEdge: throughputEdgeDelta,
      neighborInterference: neighborIoTDelta,
      blerPenalty,
      totalReward: Math.max(-10, Math.min(10, totalReward)) // Clip to reasonable range
    };
  }

  /**
   * Calculate target return for Decision Transformer conditioning
   */
  calculateTargetReturn(sliceType: 'eMBB' | 'mMTC' | 'URLLC'): number {
    switch (sliceType) {
      case 'eMBB':
        // Maximize throughput
        return 5.0;
      case 'mMTC':
        // Minimize power, good enough quality
        return 2.0;
      case 'URLLC':
        // Maximize reliability (low BLER)
        return 3.0;
      default:
        return 3.0;
    }
  }
}

// ============================================================================
// ACTION SPACE
// ============================================================================

interface ActionSpace {
  p0Deltas: number[];      // Possible P0 adjustments (dB)
  alphaDeltas: number[];   // Possible Alpha adjustments
}

const DEFAULT_ACTION_SPACE: ActionSpace = {
  p0Deltas: [-2, -1, 0, 1, 2],         // ±2 dB in 1 dB steps
  alphaDeltas: [-0.1, 0, 0.1]          // ±0.1 steps
};

class ActionDecoder {
  actionSpace: ActionSpace;
  actionSize: number;

  constructor(actionSpace: ActionSpace = DEFAULT_ACTION_SPACE) {
    this.actionSpace = actionSpace;
    this.actionSize = actionSpace.p0Deltas.length * actionSpace.alphaDeltas.length;
  }

  /**
   * Decode continuous action output to discrete action
   */
  decode(actionVector: number[]): OptimizationAction {
    // Find closest match in action space
    let bestP0Delta = 0;
    let bestAlphaDelta = 0;
    let minDist = Infinity;

    for (const p0Delta of this.actionSpace.p0Deltas) {
      for (const alphaDelta of this.actionSpace.alphaDeltas) {
        const dist = Math.pow(actionVector[0] - p0Delta, 2) +
                     Math.pow(actionVector[1] * 10 - alphaDelta * 10, 2);
        if (dist < minDist) {
          minDist = dist;
          bestP0Delta = p0Delta;
          bestAlphaDelta = alphaDelta;
        }
      }
    }

    return {
      p0Delta: bestP0Delta,
      alphaDelta: bestAlphaDelta
    };
  }

  /**
   * Encode action to vector for training
   */
  encode(action: OptimizationAction): number[] {
    return [action.p0Delta, action.alphaDelta];
  }

  getActionSize(): number {
    return 2; // [p0Delta, alphaDelta]
  }
}

// ============================================================================
// SAFETY GUARDRAILS
// ============================================================================

interface SafetyLimits {
  p0Min: number;
  p0Max: number;
  alphaMin: number;
  alphaMax: number;
  maxDeltaP0PerStep: number;
  maxDeltaAlphaPerStep: number;
  minTimeBetweenChanges: number; // ms
}

class SafetyGuardrails {
  limits: SafetyLimits;
  lastChangeTime: Map<string, number> = new Map();

  constructor(limits?: Partial<SafetyLimits>) {
    this.limits = {
      p0Min: -126,
      p0Max: -60,
      alphaMin: 0.4,
      alphaMax: 1.0,
      maxDeltaP0PerStep: 3,
      maxDeltaAlphaPerStep: 0.2,
      minTimeBetweenChanges: 60000, // 1 minute
      ...limits
    };
  }

  /**
   * Validate and clip proposed action
   */
  validate(
    cellId: string,
    currentParams: UplinkPowerControlParams,
    proposedAction: OptimizationAction
  ): { valid: boolean; clippedAction: OptimizationAction; violations: string[] } {
    const violations: string[] = [];
    const clippedAction = { ...proposedAction };

    // Check time since last change
    const lastChange = this.lastChangeTime.get(cellId) || 0;
    if (Date.now() - lastChange < this.limits.minTimeBetweenChanges) {
      violations.push(`Rate limit: ${this.limits.minTimeBetweenChanges}ms between changes`);
    }

    // Clip P0 delta
    if (Math.abs(proposedAction.p0Delta) > this.limits.maxDeltaP0PerStep) {
      clippedAction.p0Delta = Math.sign(proposedAction.p0Delta) * this.limits.maxDeltaP0PerStep;
      violations.push(`P0 delta clipped from ${proposedAction.p0Delta} to ${clippedAction.p0Delta}`);
    }

    // Check P0 range
    const newP0 = currentParams.p0NominalPusch + clippedAction.p0Delta;
    if (newP0 < this.limits.p0Min || newP0 > this.limits.p0Max) {
      const clampedP0 = Math.max(this.limits.p0Min, Math.min(this.limits.p0Max, newP0));
      clippedAction.p0Delta = clampedP0 - currentParams.p0NominalPusch;
      violations.push(`P0 would exceed limits, clamped to ${clampedP0}`);
    }

    // Clip Alpha delta
    if (Math.abs(proposedAction.alphaDelta) > this.limits.maxDeltaAlphaPerStep) {
      clippedAction.alphaDelta = Math.sign(proposedAction.alphaDelta) * this.limits.maxDeltaAlphaPerStep;
      violations.push(`Alpha delta clipped from ${proposedAction.alphaDelta} to ${clippedAction.alphaDelta}`);
    }

    // Check Alpha range
    const newAlpha = currentParams.alpha + clippedAction.alphaDelta;
    if (newAlpha < this.limits.alphaMin || newAlpha > this.limits.alphaMax) {
      const clampedAlpha = Math.max(this.limits.alphaMin, Math.min(this.limits.alphaMax, newAlpha));
      clippedAction.alphaDelta = clampedAlpha - currentParams.alpha;
      violations.push(`Alpha would exceed limits, clamped to ${clampedAlpha}`);
    }

    return {
      valid: violations.length === 0,
      clippedAction,
      violations
    };
  }

  /**
   * Record parameter change
   */
  recordChange(cellId: string): void {
    this.lastChangeTime.set(cellId, Date.now());
  }
}

// ============================================================================
// MAIN UPLINK OPTIMIZER CLASS
// ============================================================================

export interface UplinkOptimizerConfig {
  cellId: string;
  learningEnabled: boolean;
  decisionTransformerConfig: {
    contextLength: number;
    hiddenSize: number;
    numHeads: number;
    numLayers: number;
  };
  rewardWeights: Partial<RewardWeights>;
  safetyLimits: Partial<SafetyLimits>;
}

const DEFAULT_OPTIMIZER_CONFIG: UplinkOptimizerConfig = {
  cellId: 'default-cell',
  learningEnabled: true,
  decisionTransformerConfig: {
    contextLength: 20,
    hiddenSize: 256,
    numHeads: 4,
    numLayers: 3
  },
  rewardWeights: {},
  safetyLimits: {}
};

export class UplinkOptimizer extends EventEmitter {
  config: UplinkOptimizerConfig;
  stateEncoder: StateEncoder;
  rewardCalculator: RewardCalculator;
  actionDecoder: ActionDecoder;
  safetyGuardrails: SafetyGuardrails;
  decisionTransformer: DecisionTransformer;

  // History for context
  private stateHistory: number[][] = [];
  private actionHistory: number[][] = [];
  private rewardHistory: number[] = [];
  private returnHistory: number[] = [];

  // Statistics
  private actionsExecuted: number = 0;
  private actionsBlocked: number = 0;
  private totalReward: number = 0;

  constructor(config: Partial<UplinkOptimizerConfig> = {}) {
    super();
    this.config = { ...DEFAULT_OPTIMIZER_CONFIG, ...config };

    this.stateEncoder = new StateEncoder();
    this.rewardCalculator = new RewardCalculator(this.config.rewardWeights);
    this.actionDecoder = new ActionDecoder();
    this.safetyGuardrails = new SafetyGuardrails(this.config.safetyLimits);

    // Initialize Decision Transformer
    this.decisionTransformer = new DecisionTransformer({
      stateSize: this.stateEncoder.getStateSize(),
      actionSize: this.actionDecoder.getActionSize(),
      hiddenSize: this.config.decisionTransformerConfig.hiddenSize,
      numHeads: this.config.decisionTransformerConfig.numHeads,
      numLayers: this.config.decisionTransformerConfig.numLayers,
      contextLength: this.config.decisionTransformerConfig.contextLength,
      maxEpisodeLength: 1000,
      maxReturnToGo: 10
    });
  }

  /**
   * Optimize uplink parameters for a cell
   */
  optimize(
    cellKpis: CellKPIs,
    neighborKpis: CellKPIs[],
    currentParams: UplinkPowerControlParams,
    sliceType: 'eMBB' | 'mMTC' | 'URLLC' = 'eMBB'
  ): ActionRecord {
    // Encode current state
    const state = this.stateEncoder.encode(cellKpis, neighborKpis, currentParams);

    // Get target return based on slice type
    const targetReturn = this.rewardCalculator.calculateTargetReturn(sliceType);

    // Generate action using Decision Transformer
    const actionVector = this.decisionTransformer.predict(
      this.stateHistory.slice(-this.config.decisionTransformerConfig.contextLength),
      this.actionHistory.slice(-this.config.decisionTransformerConfig.contextLength),
      this.returnHistory.slice(-this.config.decisionTransformerConfig.contextLength),
      targetReturn
    );

    // Decode to discrete action
    const proposedAction = this.actionDecoder.decode(actionVector);

    // Validate through safety guardrails
    const { valid, clippedAction, violations } = this.safetyGuardrails.validate(
      this.config.cellId,
      currentParams,
      proposedAction
    );

    // Create action record
    const actionRecord: ActionRecord = {
      id: uuidv4(),
      type: 'ADJUST_P0',
      targetCgi: this.config.cellId,
      parameters: {
        p0Delta: clippedAction.p0Delta,
        alphaDelta: clippedAction.alphaDelta,
        newP0: currentParams.p0NominalPusch + clippedAction.p0Delta,
        newAlpha: currentParams.alpha + clippedAction.alphaDelta
      },
      timestamp: new Date(),
      source: 'rl',
      confidence: valid ? 0.9 : 0.5,
      executed: false,
      blocked: !valid && violations.some(v => v.includes('Rate limit')),
      blockReason: violations.length > 0 ? violations.join('; ') : undefined
    };

    // Update history
    this.stateHistory.push(state);
    this.actionHistory.push(this.actionDecoder.encode(clippedAction));

    // Track statistics
    if (actionRecord.blocked) {
      this.actionsBlocked++;
    }

    this.emit('optimization-proposed', {
      cellId: this.config.cellId,
      action: actionRecord,
      stateFeatures: this.stateEncoder.extractFeatures(cellKpis, neighborKpis, currentParams)
    });

    return actionRecord;
  }

  /**
   * Execute action and record outcome
   */
  executeAction(
    actionRecord: ActionRecord,
    prevKpis: CellKPIs,
    currKpis: CellKPIs,
    neighborIoTDelta: number
  ): void {
    if (actionRecord.blocked) return;

    // Calculate reward
    const reward = this.rewardCalculator.calculate(prevKpis, currKpis, neighborIoTDelta);

    // Update histories
    this.rewardHistory.push(reward.totalReward);
    this.totalReward += reward.totalReward;
    this.actionsExecuted++;

    // Calculate return-to-go for this step
    const returnToGo = reward.totalReward; // Simplified - would sum future rewards
    this.returnHistory.push(returnToGo);

    // Record change time
    this.safetyGuardrails.recordChange(this.config.cellId);

    // Update action record with outcome
    actionRecord.executed = true;
    actionRecord.outcome = {
      preState: {
        ulThroughput: prevKpis.ulThroughput,
        ulUserThroughput5Pct: prevKpis.ulUserThroughput5Pct,
        blerUl: prevKpis.blerUl,
        iotUl: prevKpis.iotUl
      },
      postState: {
        ulThroughput: currKpis.ulThroughput,
        ulUserThroughput5Pct: currKpis.ulUserThroughput5Pct,
        blerUl: currKpis.blerUl,
        iotUl: currKpis.iotUl
      },
      deltaMetrics: {
        throughputAvg: reward.throughputAvg,
        throughputEdge: reward.throughputEdge,
        interferenceNeighbor: reward.neighborInterference,
        reward: reward.totalReward
      }
    };

    this.emit('action-executed', {
      cellId: this.config.cellId,
      action: actionRecord,
      reward
    });
  }

  /**
   * Add trajectory for offline training
   */
  addTrajectory(trajectory: TrajectorySegment): void {
    if (this.config.learningEnabled) {
      this.decisionTransformer.addTrajectory(trajectory);
    }
  }

  /**
   * Train the Decision Transformer
   */
  train(batchSize: number = 64, learningRate: number = 0.0001): number {
    if (!this.config.learningEnabled) return 0;

    const batch = this.decisionTransformer.sampleBatch(batchSize);
    if (batch.length < batchSize) return 0;

    const loss = this.decisionTransformer.trainStep(batch, learningRate);
    this.emit('training-step', { loss, step: this.decisionTransformer.trainingStep });

    return loss;
  }

  /**
   * Export model weights for federation
   */
  exportModelWeights(): ArrayBuffer {
    return this.decisionTransformer.exportWeights();
  }

  /**
   * Get optimizer statistics
   */
  getStats(): {
    cellId: string;
    actionsExecuted: number;
    actionsBlocked: number;
    avgReward: number;
    modelStats: ReturnType<DecisionTransformer['getStats']>;
    historyLength: number;
  } {
    return {
      cellId: this.config.cellId,
      actionsExecuted: this.actionsExecuted,
      actionsBlocked: this.actionsBlocked,
      avgReward: this.actionsExecuted > 0 ? this.totalReward / this.actionsExecuted : 0,
      modelStats: this.decisionTransformer.getStats(),
      historyLength: this.stateHistory.length
    };
  }

  /**
   * Reset history (e.g., at start of new episode)
   */
  resetHistory(): void {
    this.stateHistory = [];
    this.actionHistory = [];
    this.rewardHistory = [];
    this.returnHistory = [];
  }
}

// ============================================================================
// SLICE-AWARE OPTIMIZER
// ============================================================================

interface SliceConfig {
  sliceId: SliceIdentity;
  targetBler: number;
  priorityWeight: number;
  p0Offset: number;          // Per-slice P0 offset
  alphaPreference: number;   // Preferred alpha for this slice
}

export class SliceAwareOptimizer extends EventEmitter {
  private sliceOptimizers: Map<string, UplinkOptimizer> = new Map();
  private sliceConfigs: Map<string, SliceConfig> = new Map();

  constructor(cellId: string) {
    super();

    // Initialize with default slice configurations
    this.configureSlice({
      sliceId: { sst: 1 }, // eMBB
      targetBler: 0.1,
      priorityWeight: 1.0,
      p0Offset: 0,
      alphaPreference: 0.8
    }, cellId);

    this.configureSlice({
      sliceId: { sst: 2 }, // URLLC
      targetBler: 0.00001, // 10^-5 for ultra-reliability
      priorityWeight: 2.0, // Higher priority
      p0Offset: 3,         // Higher power for reliability
      alphaPreference: 1.0 // Full pathloss compensation
    }, cellId);

    this.configureSlice({
      sliceId: { sst: 3 }, // mMTC
      targetBler: 0.1,
      priorityWeight: 0.5, // Lower priority
      p0Offset: -3,        // Lower power to save battery
      alphaPreference: 0.6 // Partial compensation
    }, cellId);
  }

  /**
   * Configure a network slice
   */
  configureSlice(config: SliceConfig, cellId: string): void {
    const sliceKey = `${config.sliceId.sst}-${config.sliceId.sd || 'default'}`;
    this.sliceConfigs.set(sliceKey, config);

    // Create optimizer for this slice
    const optimizer = new UplinkOptimizer({
      cellId: `${cellId}-slice-${sliceKey}`,
      rewardWeights: {
        throughputEdge: config.sliceId.sst === 2 ? 0.6 : 0.4, // URLLC prioritizes edge
        blerPenalty: config.sliceId.sst === 2 ? -0.5 : -0.3   // URLLC stricter on BLER
      }
    });

    this.sliceOptimizers.set(sliceKey, optimizer);
  }

  /**
   * Optimize for specific slice
   */
  optimizeSlice(
    sliceId: SliceIdentity,
    cellKpis: CellKPIs,
    neighborKpis: CellKPIs[],
    currentParams: UplinkPowerControlParams
  ): ActionRecord | null {
    const sliceKey = `${sliceId.sst}-${sliceId.sd || 'default'}`;
    const optimizer = this.sliceOptimizers.get(sliceKey);
    const config = this.sliceConfigs.get(sliceKey);

    if (!optimizer || !config) return null;

    // Apply slice-specific parameter offsets
    const sliceParams: UplinkPowerControlParams = {
      ...currentParams,
      p0NominalPusch: currentParams.p0NominalPusch + config.p0Offset,
      alpha: config.alphaPreference
    };

    const sliceType = sliceId.sst === 1 ? 'eMBB' :
                      sliceId.sst === 2 ? 'URLLC' : 'mMTC';

    return optimizer.optimize(cellKpis, neighborKpis, sliceParams, sliceType);
  }

  /**
   * Get all slice statistics
   */
  getSliceStats(): Map<string, ReturnType<UplinkOptimizer['getStats']>> {
    const stats = new Map();
    for (const [key, optimizer] of this.sliceOptimizers) {
      stats.set(key, optimizer.getStats());
    }
    return stats;
  }
}
