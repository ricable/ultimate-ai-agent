/**
 * Fault Management (FM) Module
 *
 * Transcends simple alarm monitoring to implement:
 * - Automated Root Cause Analysis (RCA) via Causal Graphs
 * - Problem Classification (Uplink Interference, Pilot Pollution, etc.)
 * - Self-Healing Workflows with compensation actions
 * - Predictive maintenance
 *
 * Problem Categories:
 * - UPLINK_INTERFERENCE: High IoT, Low SINR
 * - PILOT_POLLUTION: Low SINR, High RSRP, Many neighbors
 * - COVERAGE_HOLE: Low RSRP, Low SINR
 * - OVERSHOOTING: High handover failures, interference complaints
 * - BACKHAUL_CONGESTION: High latency, low throughput, good radio
 * - SLEEPING_CELL: Sudden drop in traffic, no handovers
 * - PCI_CONFLICT: High HO failures, localized interference
 * - ANR_ISSUE: Missing neighbor relations
 * - CAPACITY_SATURATION: High PRB utilization
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import {
  CellKPIs,
  Anomaly,
  Problem,
  ProblemCategory,
  Action,
  ActionType,
  CausalEdge
} from '../../types/index.js';
import { CausalGraphEngine } from '../../core/agentdb/index.js';

// ============================================================================
// PROBLEM CLASSIFIER
// ============================================================================

interface ProblemSignature {
  category: ProblemCategory;
  indicators: {
    metric: string;
    condition: 'high' | 'low' | 'abnormal';
    threshold?: number;
    weight: number;
  }[];
  minScore: number;
}

const PROBLEM_SIGNATURES: ProblemSignature[] = [
  {
    category: 'UPLINK_INTERFERENCE',
    indicators: [
      { metric: 'iotUl', condition: 'high', threshold: 10, weight: 0.4 },
      { metric: 'avgSinrUl', condition: 'low', threshold: 5, weight: 0.3 },
      { metric: 'blerUl', condition: 'high', threshold: 0.15, weight: 0.2 },
      { metric: 'ulThroughput', condition: 'low', weight: 0.1 }
    ],
    minScore: 0.6
  },
  {
    category: 'PILOT_POLLUTION',
    indicators: [
      { metric: 'avgSinrDl', condition: 'low', threshold: 3, weight: 0.3 },
      { metric: 'avgRsrp', condition: 'high', threshold: -80, weight: 0.2 },
      { metric: 'neighborCount', condition: 'high', threshold: 8, weight: 0.3 },
      { metric: 'hoSuccessRate', condition: 'low', threshold: 95, weight: 0.2 }
    ],
    minScore: 0.5
  },
  {
    category: 'COVERAGE_HOLE',
    indicators: [
      { metric: 'avgRsrp', condition: 'low', threshold: -110, weight: 0.4 },
      { metric: 'avgSinrDl', condition: 'low', threshold: 0, weight: 0.3 },
      { metric: 'callDropRate', condition: 'high', threshold: 1, weight: 0.2 },
      { metric: 'rrcSetupSuccessRate', condition: 'low', threshold: 95, weight: 0.1 }
    ],
    minScore: 0.6
  },
  {
    category: 'OVERSHOOTING',
    indicators: [
      { metric: 'hoFailures', condition: 'high', weight: 0.3 },
      { metric: 'pingPongRate', condition: 'high', threshold: 5, weight: 0.3 },
      { metric: 'distantNeighborHO', condition: 'high', weight: 0.2 },
      { metric: 'tiltAngle', condition: 'low', weight: 0.2 }
    ],
    minScore: 0.5
  },
  {
    category: 'BACKHAUL_CONGESTION',
    indicators: [
      { metric: 'dlLatency', condition: 'high', threshold: 50, weight: 0.3 },
      { metric: 'dlThroughput', condition: 'low', weight: 0.3 },
      { metric: 'avgSinrDl', condition: 'high', weight: 0.2 }, // Good radio
      { metric: 'packetLoss', condition: 'high', threshold: 1, weight: 0.2 }
    ],
    minScore: 0.6
  },
  {
    category: 'SLEEPING_CELL',
    indicators: [
      { metric: 'activeUsers', condition: 'low', threshold: 1, weight: 0.3 },
      { metric: 'hoAttempts', condition: 'low', threshold: 1, weight: 0.3 },
      { metric: 'prbUtilizationDl', condition: 'low', threshold: 5, weight: 0.2 },
      { metric: 'dataVolumeDl', condition: 'low', weight: 0.2 }
    ],
    minScore: 0.7
  },
  {
    category: 'PCI_CONFLICT',
    indicators: [
      { metric: 'hoSuccessRate', condition: 'low', threshold: 90, weight: 0.3 },
      { metric: 'localizedInterference', condition: 'high', weight: 0.3 },
      { metric: 'pciCollisionDetected', condition: 'high', weight: 0.3 },
      { metric: 'rlcRetransmissions', condition: 'high', weight: 0.1 }
    ],
    minScore: 0.6
  },
  {
    category: 'ANR_ISSUE',
    indicators: [
      { metric: 'missingNeighborHO', condition: 'high', weight: 0.4 },
      { metric: 'hoFailures', condition: 'high', weight: 0.3 },
      { metric: 'unknownCellReports', condition: 'high', weight: 0.3 }
    ],
    minScore: 0.5
  },
  {
    category: 'CAPACITY_SATURATION',
    indicators: [
      { metric: 'prbUtilizationDl', condition: 'high', threshold: 85, weight: 0.3 },
      { metric: 'prbUtilizationUl', condition: 'high', threshold: 85, weight: 0.2 },
      { metric: 'activeUsers', condition: 'high', threshold: 400, weight: 0.2 },
      { metric: 'callDropRate', condition: 'high', weight: 0.15 },
      { metric: 'congestionDrops', condition: 'high', weight: 0.15 }
    ],
    minScore: 0.6
  }
];

class ProblemClassifier {
  /**
   * Classify anomaly into problem category
   */
  classify(
    anomaly: Anomaly,
    kpis: CellKPIs,
    neighborKpis: CellKPIs[]
  ): { category: ProblemCategory; confidence: number }[] {
    const results: { category: ProblemCategory; confidence: number }[] = [];

    for (const signature of PROBLEM_SIGNATURES) {
      const score = this.computeSignatureScore(signature, kpis, neighborKpis);

      if (score >= signature.minScore) {
        results.push({
          category: signature.category,
          confidence: score
        });
      }
    }

    // Sort by confidence descending
    results.sort((a, b) => b.confidence - a.confidence);
    return results;
  }

  private computeSignatureScore(
    signature: ProblemSignature,
    kpis: CellKPIs,
    neighborKpis: CellKPIs[]
  ): number {
    let totalWeight = 0;
    let matchedWeight = 0;

    for (const indicator of signature.indicators) {
      totalWeight += indicator.weight;

      const value = this.getMetricValue(indicator.metric, kpis, neighborKpis);
      if (value === null) continue;

      let matches = false;

      switch (indicator.condition) {
        case 'high':
          if (indicator.threshold !== undefined) {
            matches = value > indicator.threshold;
          } else {
            matches = value > this.getBaselineHigh(indicator.metric);
          }
          break;
        case 'low':
          if (indicator.threshold !== undefined) {
            matches = value < indicator.threshold;
          } else {
            matches = value < this.getBaselineLow(indicator.metric);
          }
          break;
        case 'abnormal':
          matches = Math.abs(value) > this.getAbnormalThreshold(indicator.metric);
          break;
      }

      if (matches) {
        matchedWeight += indicator.weight;
      }
    }

    return totalWeight > 0 ? matchedWeight / totalWeight : 0;
  }

  private getMetricValue(
    metric: string,
    kpis: CellKPIs,
    neighborKpis: CellKPIs[]
  ): number | null {
    // Special computed metrics
    if (metric === 'neighborCount') {
      return neighborKpis.length;
    }
    if (metric === 'distantNeighborHO') {
      // Would be computed from handover stats
      return 0;
    }

    // Direct metrics
    const value = (kpis as any)[metric];
    return typeof value === 'number' ? value : null;
  }

  private getBaselineHigh(metric: string): number {
    const baselines: Record<string, number> = {
      iotUl: 10,
      activeUsers: 300,
      hoFailures: 10,
      pingPongRate: 3
    };
    return baselines[metric] || 0;
  }

  private getBaselineLow(metric: string): number {
    const baselines: Record<string, number> = {
      ulThroughput: 10,
      dlThroughput: 50,
      hoSuccessRate: 95,
      rrcSetupSuccessRate: 98
    };
    return baselines[metric] || 0;
  }

  private getAbnormalThreshold(metric: string): number {
    return 2; // 2 standard deviations
  }
}

// ============================================================================
// ROOT CAUSE ANALYZER
// ============================================================================

interface RootCauseAnalysis {
  problem: Problem;
  causalChain: CausalEdge[];
  rootCause: string;
  contributingFactors: string[];
  confidence: number;
}

class RootCauseAnalyzer {
  private causalGraph: CausalGraphEngine;
  private predefinedCauses: Map<ProblemCategory, string[]>;

  constructor() {
    this.causalGraph = new CausalGraphEngine();
    this.initializePredefinedCauses();
    this.initializeCausalGraph();
  }

  private initializePredefinedCauses(): void {
    this.predefinedCauses = new Map([
      ['UPLINK_INTERFERENCE', [
        'neighbor_high_power',
        'p0_too_high',
        'alpha_too_high',
        'external_interference',
        'tdd_cross_link'
      ]],
      ['PILOT_POLLUTION', [
        'overlapping_coverage',
        'excessive_power',
        'improper_tilt',
        'missing_neighbor_relations'
      ]],
      ['COVERAGE_HOLE', [
        'physical_obstruction',
        'insufficient_power',
        'excessive_downtilt',
        'hardware_degradation'
      ]],
      ['OVERSHOOTING', [
        'insufficient_downtilt',
        'excessive_power',
        'antenna_height',
        'propagation_anomaly'
      ]],
      ['BACKHAUL_CONGESTION', [
        'transport_capacity',
        'routing_issue',
        'packet_loss_transport',
        'latency_hop_count'
      ]],
      ['SLEEPING_CELL', [
        'hardware_failure',
        'software_crash',
        'configuration_error',
        'transmission_issue'
      ]],
      ['PCI_CONFLICT', [
        'pci_reuse_distance',
        'pci_mod3_issue',
        'pci_planning_error'
      ]],
      ['ANR_ISSUE', [
        'anr_disabled',
        'blacklist_misconfigured',
        'x2_interface_issue'
      ]],
      ['CAPACITY_SATURATION', [
        'traffic_growth',
        'event_spike',
        'neighbor_failure',
        'spectrum_limitation'
      ]]
    ]);
  }

  private initializeCausalGraph(): void {
    // Initialize with known causal relationships
    // P(effect | do(cause))

    // Uplink interference causes
    this.causalGraph.recordObservation('neighbor_high_power', 'high_iot', 0.8);
    this.causalGraph.recordObservation('high_iot', 'low_sinr_ul', 0.9);
    this.causalGraph.recordObservation('low_sinr_ul', 'high_bler_ul', 0.85);
    this.causalGraph.recordObservation('high_bler_ul', 'low_throughput_ul', 0.8);

    // Coverage issues
    this.causalGraph.recordObservation('excessive_downtilt', 'coverage_shrinkage', 0.7);
    this.causalGraph.recordObservation('coverage_shrinkage', 'low_rsrp', 0.85);
    this.causalGraph.recordObservation('low_rsrp', 'call_drops', 0.6);

    // Mobility issues
    this.causalGraph.recordObservation('missing_neighbor', 'ho_failure', 0.7);
    this.causalGraph.recordObservation('pci_conflict', 'ho_failure', 0.8);
    this.causalGraph.recordObservation('ho_failure', 'call_drops', 0.5);

    // Capacity issues
    this.causalGraph.recordObservation('traffic_spike', 'high_prb_utilization', 0.9);
    this.causalGraph.recordObservation('high_prb_utilization', 'congestion_drops', 0.7);
    this.causalGraph.recordObservation('neighbor_failure', 'traffic_spike', 0.6);
  }

  /**
   * Perform root cause analysis on a problem
   */
  analyze(problem: Problem, kpis: CellKPIs, context: any): RootCauseAnalysis {
    // Get predefined causes for this problem category
    const possibleCauses = this.predefinedCauses.get(problem.category) || [];

    // Query causal graph for most likely root cause
    const symptom = this.problemToSymptom(problem.category);
    const causalChain = this.causalGraph.findRootCause(symptom);

    // Score each possible cause based on context
    const scoredCauses = possibleCauses.map(cause => ({
      cause,
      score: this.scoreCauseGivenContext(cause, kpis, context)
    }));

    scoredCauses.sort((a, b) => b.score - a.score);

    const rootCause = causalChain.length > 0
      ? causalChain[0].cause
      : (scoredCauses[0]?.cause || 'unknown');

    const contributingFactors = scoredCauses
      .filter(c => c.score > 0.3)
      .slice(1, 4)
      .map(c => c.cause);

    return {
      problem,
      causalChain,
      rootCause,
      contributingFactors,
      confidence: scoredCauses[0]?.score || 0.5
    };
  }

  /**
   * Learn from confirmed root cause
   */
  learnFromConfirmation(symptom: string, confirmedCause: string): void {
    this.causalGraph.recordObservation(confirmedCause, symptom, 1.0);
  }

  private problemToSymptom(category: ProblemCategory): string {
    const mapping: Record<ProblemCategory, string> = {
      UPLINK_INTERFERENCE: 'high_iot',
      PILOT_POLLUTION: 'low_sinr_dl',
      COVERAGE_HOLE: 'low_rsrp',
      OVERSHOOTING: 'distant_ho',
      BACKHAUL_CONGESTION: 'high_latency',
      SLEEPING_CELL: 'zero_traffic',
      PCI_CONFLICT: 'ho_failure',
      ANR_ISSUE: 'ho_failure',
      CAPACITY_SATURATION: 'high_prb_utilization'
    };
    return mapping[category] || 'unknown';
  }

  private scoreCauseGivenContext(cause: string, kpis: CellKPIs, context: any): number {
    // Score based on supporting evidence in context
    let score = 0.5; // Base score

    // Example contextual scoring
    switch (cause) {
      case 'neighbor_high_power':
        // Check if neighbors have higher power
        if (context.neighborPowers?.some((p: number) => p > 43)) {
          score += 0.3;
        }
        break;
      case 'traffic_spike':
        if (kpis.activeUsers > 400) score += 0.3;
        break;
      case 'hardware_degradation':
        if (context.alarms?.includes('RF_ALARM')) score += 0.4;
        break;
      case 'event_spike':
        if (context.nearbyEvents?.length > 0) score += 0.4;
        break;
    }

    return Math.min(1, score);
  }

  getCausalGraph(): CausalGraphEngine {
    return this.causalGraph;
  }
}

// ============================================================================
// SELF-HEALING ENGINE
// ============================================================================

interface HealingWorkflow {
  id: string;
  problem: Problem;
  rootCause: string;
  actions: Action[];
  compensationActions: Action[];  // Actions for neighbors while healing
  status: 'pending' | 'executing' | 'completed' | 'failed' | 'rolled_back';
  startTime?: Date;
  endTime?: Date;
  result?: {
    success: boolean;
    kpiDelta: Record<string, number>;
    message: string;
  };
}

interface HealingRule {
  problemCategory: ProblemCategory;
  rootCauses: string[];
  actions: (context: any) => Action[];
  compensations?: (context: any) => Action[];
  requiresApproval: boolean;
}

class SelfHealingEngine extends EventEmitter {
  private workflows: Map<string, HealingWorkflow> = new Map();
  private healingRules: HealingRule[] = [];
  private autoHealEnabled: boolean = true;

  constructor() {
    super();
    this.initializeHealingRules();
  }

  private initializeHealingRules(): void {
    this.healingRules = [
      {
        problemCategory: 'SLEEPING_CELL',
        rootCauses: ['hardware_failure', 'software_crash'],
        actions: (ctx) => [{
          type: 'RESTART_CELL' as ActionType,
          targetCgi: ctx.cellId,
          parameters: { softRestart: true },
          timestamp: new Date(),
          source: 'rl' as const,
          confidence: 0.9
        }],
        compensations: (ctx) => ctx.neighbors.map((n: string) => ({
          type: 'ADJUST_TILT' as ActionType,
          targetCgi: n,
          parameters: { tiltDelta: -1 }, // Increase coverage temporarily
          timestamp: new Date(),
          source: 'rl' as const,
          confidence: 0.8
        })),
        requiresApproval: false
      },
      {
        problemCategory: 'PCI_CONFLICT',
        rootCauses: ['pci_reuse_distance', 'pci_mod3_issue'],
        actions: (ctx) => [{
          type: 'CHANGE_PCI' as ActionType,
          targetCgi: ctx.cellId,
          parameters: { newPci: ctx.availablePci },
          timestamp: new Date(),
          source: 'rl' as const,
          confidence: 0.85
        }],
        requiresApproval: true // PCI changes need verification
      },
      {
        problemCategory: 'UPLINK_INTERFERENCE',
        rootCauses: ['p0_too_high', 'alpha_too_high'],
        actions: (ctx) => [{
          type: 'ADJUST_P0' as ActionType,
          targetCgi: ctx.cellId,
          parameters: { p0Delta: -2, alphaDelta: -0.1 },
          timestamp: new Date(),
          source: 'rl' as const,
          confidence: 0.8
        }],
        requiresApproval: false
      },
      {
        problemCategory: 'COVERAGE_HOLE',
        rootCauses: ['excessive_downtilt', 'insufficient_power'],
        actions: (ctx) => [
          {
            type: 'ADJUST_TILT' as ActionType,
            targetCgi: ctx.cellId,
            parameters: { tiltDelta: -1 },
            timestamp: new Date(),
            source: 'rl' as const,
            confidence: 0.7
          },
          {
            type: 'ADJUST_POWER' as ActionType,
            targetCgi: ctx.cellId,
            parameters: { powerDelta: 1 },
            timestamp: new Date(),
            source: 'rl' as const,
            confidence: 0.7
          }
        ],
        requiresApproval: false
      },
      {
        problemCategory: 'ANR_ISSUE',
        rootCauses: ['missing_neighbor', 'anr_disabled'],
        actions: (ctx) => [{
          type: 'UPDATE_ANR' as ActionType,
          targetCgi: ctx.cellId,
          parameters: {
            action: 'add',
            neighborCgi: ctx.missingNeighbor
          },
          timestamp: new Date(),
          source: 'rl' as const,
          confidence: 0.85
        }],
        requiresApproval: false
      },
      {
        problemCategory: 'OVERSHOOTING',
        rootCauses: ['insufficient_downtilt', 'excessive_power'],
        actions: (ctx) => [
          {
            type: 'ADJUST_TILT' as ActionType,
            targetCgi: ctx.cellId,
            parameters: { tiltDelta: 1 },
            timestamp: new Date(),
            source: 'rl' as const,
            confidence: 0.75
          }
        ],
        requiresApproval: false
      },
      {
        problemCategory: 'CAPACITY_SATURATION',
        rootCauses: ['traffic_growth', 'event_spike'],
        actions: (ctx) => [
          {
            type: 'ACTIVATE_CARRIER' as ActionType,
            targetCgi: ctx.cellId,
            parameters: { carrier: ctx.availableCarrier || 'CA_SCC1' },
            timestamp: new Date(),
            source: 'rl' as const,
            confidence: 0.8
          }
        ],
        compensations: (ctx) => [], // Load balance to neighbors
        requiresApproval: false
      }
    ];
  }

  /**
   * Create healing workflow for a problem
   */
  createWorkflow(
    problem: Problem,
    rootCause: string,
    context: any
  ): HealingWorkflow | null {
    // Find matching rule
    const rule = this.healingRules.find(r =>
      r.problemCategory === problem.category &&
      r.rootCauses.includes(rootCause)
    );

    if (!rule) {
      this.emit('no-healing-rule', { problem, rootCause });
      return null;
    }

    const workflow: HealingWorkflow = {
      id: uuidv4(),
      problem,
      rootCause,
      actions: rule.actions(context),
      compensationActions: rule.compensations?.(context) || [],
      status: rule.requiresApproval ? 'pending' : 'executing'
    };

    this.workflows.set(workflow.id, workflow);

    if (!rule.requiresApproval && this.autoHealEnabled) {
      this.executeWorkflow(workflow.id);
    } else {
      this.emit('workflow-pending-approval', { workflowId: workflow.id, problem });
    }

    return workflow;
  }

  /**
   * Execute a healing workflow
   */
  async executeWorkflow(workflowId: string): Promise<void> {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) return;

    workflow.status = 'executing';
    workflow.startTime = new Date();

    this.emit('workflow-started', { workflowId });

    try {
      // Execute compensation actions first (for neighbors)
      for (const action of workflow.compensationActions) {
        this.emit('executing-compensation', { action });
        // In real system, would send to O-AM interface
        await this.simulateActionExecution(action);
      }

      // Execute main healing actions
      for (const action of workflow.actions) {
        this.emit('executing-action', { action });
        await this.simulateActionExecution(action);
      }

      workflow.status = 'completed';
      workflow.endTime = new Date();
      workflow.result = {
        success: true,
        kpiDelta: {},
        message: 'Healing actions executed successfully'
      };

      this.emit('workflow-completed', { workflowId, result: workflow.result });

    } catch (error) {
      workflow.status = 'failed';
      workflow.endTime = new Date();
      workflow.result = {
        success: false,
        kpiDelta: {},
        message: `Healing failed: ${error}`
      };

      this.emit('workflow-failed', { workflowId, error });

      // Attempt rollback of compensation actions
      await this.rollbackCompensations(workflow);
    }
  }

  /**
   * Approve a pending workflow
   */
  approveWorkflow(workflowId: string): void {
    const workflow = this.workflows.get(workflowId);
    if (workflow && workflow.status === 'pending') {
      this.executeWorkflow(workflowId);
    }
  }

  /**
   * Reject and remove a pending workflow
   */
  rejectWorkflow(workflowId: string, reason: string): void {
    const workflow = this.workflows.get(workflowId);
    if (workflow && workflow.status === 'pending') {
      workflow.status = 'failed';
      workflow.result = {
        success: false,
        kpiDelta: {},
        message: `Rejected: ${reason}`
      };
      this.emit('workflow-rejected', { workflowId, reason });
    }
  }

  /**
   * Enable/disable auto-healing
   */
  setAutoHeal(enabled: boolean): void {
    this.autoHealEnabled = enabled;
    this.emit('auto-heal-changed', { enabled });
  }

  private async simulateActionExecution(action: Action): Promise<void> {
    // Simulate action execution delay
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  private async rollbackCompensations(workflow: HealingWorkflow): Promise<void> {
    workflow.status = 'rolled_back';

    for (const action of workflow.compensationActions) {
      // Reverse the compensation action
      const rollbackAction: Action = {
        ...action,
        parameters: this.reverseParameters(action.parameters)
      };
      this.emit('rolling-back', { action: rollbackAction });
      await this.simulateActionExecution(rollbackAction);
    }
  }

  private reverseParameters(params: Record<string, any>): Record<string, any> {
    const reversed: Record<string, any> = {};
    for (const [key, value] of Object.entries(params)) {
      if (key.includes('Delta') && typeof value === 'number') {
        reversed[key] = -value;
      } else {
        reversed[key] = value;
      }
    }
    return reversed;
  }

  getWorkflow(workflowId: string): HealingWorkflow | undefined {
    return this.workflows.get(workflowId);
  }

  getActiveWorkflows(): HealingWorkflow[] {
    return Array.from(this.workflows.values())
      .filter(w => w.status === 'executing' || w.status === 'pending');
  }

  getStats(): {
    totalWorkflows: number;
    completed: number;
    failed: number;
    pending: number;
    autoHealEnabled: boolean;
  } {
    const all = Array.from(this.workflows.values());
    return {
      totalWorkflows: all.length,
      completed: all.filter(w => w.status === 'completed').length,
      failed: all.filter(w => w.status === 'failed').length,
      pending: all.filter(w => w.status === 'pending').length,
      autoHealEnabled: this.autoHealEnabled
    };
  }
}

// ============================================================================
// MAIN FAULT MANAGER CLASS
// ============================================================================

export interface FaultManagerConfig {
  cellId: string;
  enableAutoHealing: boolean;
  enablePredictive: boolean;
  minConfidenceForHealing: number;
}

const DEFAULT_FM_CONFIG: FaultManagerConfig = {
  cellId: 'default-cell',
  enableAutoHealing: true,
  enablePredictive: true,
  minConfidenceForHealing: 0.7
};

export class FaultManager extends EventEmitter {
  config: FaultManagerConfig;
  classifier: ProblemClassifier;
  rcaEngine: RootCauseAnalyzer;
  healingEngine: SelfHealingEngine;

  // Active problems
  private activeProblems: Map<string, Problem> = new Map();

  // Statistics
  private problemsDetected: number = 0;
  private problemsResolved: number = 0;
  private healingAttempts: number = 0;

  constructor(config: Partial<FaultManagerConfig> = {}) {
    super();
    this.config = { ...DEFAULT_FM_CONFIG, ...config };

    this.classifier = new ProblemClassifier();
    this.rcaEngine = new RootCauseAnalyzer();
    this.healingEngine = new SelfHealingEngine();

    // Wire up healing engine events
    this.healingEngine.on('workflow-completed', (data) => {
      this.handleHealingComplete(data.workflowId);
    });

    this.healingEngine.on('workflow-failed', (data) => {
      this.emit('healing-failed', data);
    });

    this.healingEngine.setAutoHeal(this.config.enableAutoHealing);
  }

  /**
   * Process an anomaly and create problem if warranted
   */
  processAnomaly(
    anomaly: Anomaly,
    kpis: CellKPIs,
    neighborKpis: CellKPIs[],
    context: any = {}
  ): Problem | null {
    // Classify the anomaly
    const classifications = this.classifier.classify(anomaly, kpis, neighborKpis);

    if (classifications.length === 0) {
      return null;
    }

    const topClassification = classifications[0];

    // Create problem
    const problem: Problem = {
      id: uuidv4(),
      timestamp: new Date(),
      affectedCells: [anomaly.cgi],
      category: topClassification.category,
      confidence: topClassification.confidence,
      symptoms: [anomaly],
      suggestedActions: [],
      automatedRecovery: false
    };

    // Perform root cause analysis
    const rca = this.rcaEngine.analyze(problem, kpis, context);
    problem.rootCause = rca.rootCause;
    problem.causalChain = rca.causalChain;

    // Store active problem
    this.activeProblems.set(problem.id, problem);
    this.problemsDetected++;

    this.emit('problem-detected', {
      problem,
      rca
    });

    // Trigger self-healing if confidence is sufficient
    if (
      this.config.enableAutoHealing &&
      rca.confidence >= this.config.minConfidenceForHealing
    ) {
      this.triggerHealing(problem, rca, context);
    }

    return problem;
  }

  /**
   * Trigger self-healing for a problem
   */
  triggerHealing(problem: Problem, rca: RootCauseAnalysis, context: any): void {
    const workflow = this.healingEngine.createWorkflow(
      problem,
      rca.rootCause,
      {
        ...context,
        cellId: this.config.cellId,
        neighbors: context.neighbors || [],
        problem
      }
    );

    if (workflow) {
      problem.automatedRecovery = true;
      problem.recoveryStatus = workflow.status;
      problem.suggestedActions = workflow.actions;
      this.healingAttempts++;

      this.emit('healing-triggered', {
        problemId: problem.id,
        workflowId: workflow.id
      });
    }
  }

  /**
   * Manually resolve a problem
   */
  resolveProblem(problemId: string, resolution: string): void {
    const problem = this.activeProblems.get(problemId);
    if (problem) {
      problem.recoveryStatus = 'completed';
      this.activeProblems.delete(problemId);
      this.problemsResolved++;

      // Learn from resolution
      if (problem.rootCause) {
        this.rcaEngine.learnFromConfirmation(
          problem.category,
          problem.rootCause
        );
      }

      this.emit('problem-resolved', { problemId, resolution });
    }
  }

  /**
   * Get all active problems
   */
  getActiveProblems(): Problem[] {
    return Array.from(this.activeProblems.values());
  }

  /**
   * Get problem by ID
   */
  getProblem(problemId: string): Problem | undefined {
    return this.activeProblems.get(problemId);
  }

  /**
   * Approve pending healing workflow
   */
  approveHealing(problemId: string): void {
    const problem = this.activeProblems.get(problemId);
    if (problem && problem.recoveryStatus === 'pending') {
      // Find associated workflow
      const workflows = this.healingEngine.getActiveWorkflows();
      const workflow = workflows.find(w => w.problem.id === problemId);
      if (workflow) {
        this.healingEngine.approveWorkflow(workflow.id);
      }
    }
  }

  /**
   * Enable/disable auto-healing
   */
  setAutoHealing(enabled: boolean): void {
    this.config.enableAutoHealing = enabled;
    this.healingEngine.setAutoHeal(enabled);
  }

  private handleHealingComplete(workflowId: string): void {
    const workflow = this.healingEngine.getWorkflow(workflowId);
    if (workflow && workflow.result?.success) {
      const problem = this.activeProblems.get(workflow.problem.id);
      if (problem) {
        problem.recoveryStatus = 'completed';
        this.activeProblems.delete(problem.id);
        this.problemsResolved++;

        this.emit('problem-resolved', {
          problemId: problem.id,
          resolution: 'auto-healed'
        });
      }
    }
  }

  /**
   * Get fault manager statistics
   */
  getStats(): {
    cellId: string;
    problemsDetected: number;
    problemsResolved: number;
    activeProblems: number;
    healingAttempts: number;
    healingStats: ReturnType<SelfHealingEngine['getStats']>;
  } {
    return {
      cellId: this.config.cellId,
      problemsDetected: this.problemsDetected,
      problemsResolved: this.problemsResolved,
      activeProblems: this.activeProblems.size,
      healingAttempts: this.healingAttempts,
      healingStats: this.healingEngine.getStats()
    };
  }
}
