/**
 * Neuro-Symbolic Root Cause Analyzer
 * Combines symbolic rules with neural context for intelligent fault diagnosis
 * Implements psycho-symbolic-reasoner patterns
 */

import {
  Anomaly,
  AnomalyType,
  Alarm,
  RootCauseAnalysis,
  ProbableCause,
  RecommendedAction,
  ReasoningStep,
  TaskPriority,
  CellMetrics,
} from '../core/types.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('NeuroSymbolicRCA');

/**
 * External context for neural reasoning
 */
export interface NeuralContext {
  weather?: 'clear' | 'rain' | 'snow' | 'storm';
  timeOfDay?: 'morning' | 'afternoon' | 'evening' | 'night';
  dayOfWeek?: 'weekday' | 'weekend';
  specialEvent?: string;
  recentChanges?: string[];
  neighboringCellStates?: Map<string, CellMetrics>;
  historicalPatterns?: string[];
}

/**
 * Symbolic rule definition
 */
interface SymbolicRule {
  id: string;
  conditions: RuleCondition[];
  conclusion: string;
  confidence: number;
  priority: TaskPriority;
}

interface RuleCondition {
  type: 'alarm' | 'anomaly' | 'metric' | 'context';
  field: string;
  operator: 'eq' | 'gt' | 'lt' | 'contains' | 'exists';
  value: unknown;
}

/**
 * Neuro-Symbolic Root Cause Analyzer
 */
export class NeuroSymbolicRCA {
  private symbolicRules: SymbolicRule[];
  private neuralPatterns: Map<string, PatternSignature>;
  private reasoningHistory: ReasoningSession[];

  constructor() {
    this.symbolicRules = this.initializeSymbolicRules();
    this.neuralPatterns = this.initializeNeuralPatterns();
    this.reasoningHistory = [];

    logger.info('Neuro-Symbolic RCA initialized', {
      ruleCount: this.symbolicRules.length,
      patternCount: this.neuralPatterns.size,
    });
  }

  /**
   * Analyze an anomaly and determine root cause
   */
  async analyze(
    anomaly: Anomaly,
    alarms: Alarm[],
    metrics: CellMetrics,
    context: NeuralContext
  ): Promise<RootCauseAnalysis> {
    const startTime = performance.now();
    const reasoningChain: ReasoningStep[] = [];

    logger.info('Starting RCA', {
      anomalyId: anomaly.id,
      type: anomaly.type,
      cellId: anomaly.cellId,
    });

    // Step 1: Symbolic reasoning - match rules
    const symbolicResults = this.performSymbolicReasoning(
      anomaly,
      alarms,
      metrics,
      reasoningChain
    );

    // Step 2: Neural reasoning - pattern matching and context analysis
    const neuralResults = this.performNeuralReasoning(
      anomaly,
      metrics,
      context,
      reasoningChain
    );

    // Step 3: Hybrid fusion - combine symbolic and neural insights
    const fusedResults = this.fuseReasoningResults(
      symbolicResults,
      neuralResults,
      reasoningChain
    );

    // Step 4: Generate recommended actions
    const recommendedActions = this.generateRecommendations(
      fusedResults,
      anomaly,
      context
    );

    const executionTime = performance.now() - startTime;

    // Record session for learning
    this.reasoningHistory.push({
      timestamp: Date.now(),
      anomalyId: anomaly.id,
      causes: fusedResults,
      executionTimeMs: executionTime,
    });

    logger.info('RCA complete', {
      anomalyId: anomaly.id,
      topCause: fusedResults[0]?.cause,
      confidence: fusedResults[0]?.probability,
      executionTimeMs: executionTime.toFixed(2),
    });

    return {
      anomalyId: anomaly.id,
      probableCauses: fusedResults,
      recommendedActions,
      reasoningChain,
      confidence: fusedResults[0]?.probability || 0,
    };
  }

  /**
   * Symbolic reasoning using rule matching
   */
  private performSymbolicReasoning(
    anomaly: Anomaly,
    alarms: Alarm[],
    metrics: CellMetrics,
    chain: ReasoningStep[]
  ): ProbableCause[] {
    const matchedRules: { rule: SymbolicRule; matchScore: number }[] = [];

    for (const rule of this.symbolicRules) {
      const matchScore = this.evaluateRule(rule, anomaly, alarms, metrics);
      if (matchScore > 0) {
        matchedRules.push({ rule, matchScore });

        chain.push({
          type: 'symbolic',
          premise: `Rule ${rule.id}: ${rule.conditions.map((c) => `${c.field} ${c.operator} ${c.value}`).join(' AND ')}`,
          conclusion: rule.conclusion,
          confidence: matchScore * rule.confidence,
        });
      }
    }

    // Sort by match score
    matchedRules.sort((a, b) => b.matchScore - a.matchScore);

    return matchedRules.map(({ rule, matchScore }) => ({
      cause: rule.conclusion,
      probability: matchScore * rule.confidence,
      evidence: [`Matched rule: ${rule.id}`],
    }));
  }

  /**
   * Neural reasoning using pattern matching and context
   */
  private performNeuralReasoning(
    anomaly: Anomaly,
    metrics: CellMetrics,
    context: NeuralContext,
    chain: ReasoningStep[]
  ): ProbableCause[] {
    const causes: ProbableCause[] = [];

    // Pattern matching based on anomaly type and context
    const signature = this.createSignature(anomaly, metrics, context);
    const matchedPatterns = this.matchPatterns(signature);

    for (const match of matchedPatterns) {
      chain.push({
        type: 'neural',
        premise: `Pattern match: ${match.patternId} (similarity: ${match.similarity.toFixed(2)})`,
        conclusion: match.cause,
        confidence: match.similarity,
      });

      causes.push({
        cause: match.cause,
        probability: match.similarity,
        evidence: match.evidence,
      });
    }

    // Context-aware reasoning
    if (context.weather) {
      const weatherCause = this.reasonAboutWeather(anomaly, context.weather);
      if (weatherCause) {
        chain.push({
          type: 'neural',
          premise: `Weather context: ${context.weather}`,
          conclusion: weatherCause.cause,
          confidence: weatherCause.probability,
        });
        causes.push(weatherCause);
      }
    }

    // Temporal reasoning
    if (context.timeOfDay) {
      const temporalCause = this.reasonAboutTime(anomaly, context.timeOfDay, context.dayOfWeek);
      if (temporalCause) {
        chain.push({
          type: 'neural',
          premise: `Temporal context: ${context.timeOfDay}, ${context.dayOfWeek || 'unknown'}`,
          conclusion: temporalCause.cause,
          confidence: temporalCause.probability,
        });
        causes.push(temporalCause);
      }
    }

    return causes;
  }

  /**
   * Fuse symbolic and neural reasoning results
   */
  private fuseReasoningResults(
    symbolic: ProbableCause[],
    neural: ProbableCause[],
    chain: ReasoningStep[]
  ): ProbableCause[] {
    const causeMap = new Map<string, ProbableCause>();

    // Combine causes, merging evidence for same root cause
    for (const cause of [...symbolic, ...neural]) {
      const existing = causeMap.get(cause.cause);
      if (existing) {
        // Bayesian-like update
        existing.probability = 1 - (1 - existing.probability) * (1 - cause.probability);
        existing.evidence.push(...cause.evidence);
      } else {
        causeMap.set(cause.cause, { ...cause, evidence: [...cause.evidence] });
      }
    }

    // Sort by probability
    const fused = Array.from(causeMap.values()).sort((a, b) => b.probability - a.probability);

    // Record fusion step
    if (fused.length > 0) {
      chain.push({
        type: 'hybrid',
        premise: `Fused ${symbolic.length} symbolic + ${neural.length} neural causes`,
        conclusion: `Top cause: ${fused[0].cause} (p=${fused[0].probability.toFixed(2)})`,
        confidence: fused[0].probability,
      });
    }

    return fused.slice(0, 5); // Return top 5 causes
  }

  /**
   * Generate recommended actions based on root causes
   */
  private generateRecommendations(
    causes: ProbableCause[],
    anomaly: Anomaly,
    context: NeuralContext
  ): RecommendedAction[] {
    const actions: RecommendedAction[] = [];

    for (const cause of causes.slice(0, 3)) {
      const action = this.mapCauseToAction(cause, anomaly);
      if (action) {
        actions.push(action);
      }
    }

    // Add generic diagnostic action if confidence is low
    if (causes[0]?.probability < 0.6) {
      actions.push({
        action: 'Run extended diagnostics',
        priority: 'medium',
        risk: 'low',
        expectedImpact: 'Gather more data for better diagnosis',
      });
    }

    return actions;
  }

  /**
   * Map a root cause to recommended action
   */
  private mapCauseToAction(cause: ProbableCause, anomaly: Anomaly): RecommendedAction | null {
    const actionMap: Record<string, RecommendedAction> = {
      'water_ingress_connector': {
        action: 'Monitor for 24h; schedule inspection if persists after dry weather',
        priority: 'low',
        risk: 'low',
        expectedImpact: 'Avoid unnecessary truck roll; connector may self-dry',
      },
      'hardware_failure_antenna': {
        action: 'Dispatch technician to replace antenna/connector',
        priority: 'high',
        risk: 'medium',
        expectedImpact: 'Restore cell coverage and capacity',
      },
      'external_interference': {
        action: 'Analyze interference pattern; adjust PCI or antenna parameters',
        priority: 'medium',
        risk: 'low',
        expectedImpact: 'Reduce interference by 20-30%',
      },
      'configuration_drift': {
        action: 'Restore parameters to golden configuration',
        priority: 'high',
        risk: 'medium',
        expectedImpact: 'Restore normal cell operation',
      },
      'traffic_overload': {
        action: 'Activate MLB to offload traffic to neighbors; consider capacity expansion',
        priority: 'medium',
        risk: 'low',
        expectedImpact: 'Reduce congestion by 30-40%',
      },
      'neighbor_cell_issue': {
        action: 'Coordinate with neighboring cell optimization; check handover parameters',
        priority: 'medium',
        risk: 'low',
        expectedImpact: 'Improve handover success rate',
      },
      'software_fault': {
        action: 'Attempt remote restart; escalate to vendor if persists',
        priority: 'high',
        risk: 'medium',
        expectedImpact: 'Clear software fault and restore service',
      },
      'sleeping_cell_lockup': {
        action: 'Perform controlled cell restart during maintenance window',
        priority: 'critical',
        risk: 'high',
        expectedImpact: 'Restore cell to serving traffic',
      },
    };

    return actionMap[cause.cause] || null;
  }

  /**
   * Evaluate a symbolic rule against current state
   */
  private evaluateRule(
    rule: SymbolicRule,
    anomaly: Anomaly,
    alarms: Alarm[],
    metrics: CellMetrics
  ): number {
    let matchedConditions = 0;

    for (const condition of rule.conditions) {
      let value: unknown;

      switch (condition.type) {
        case 'anomaly':
          value = this.getAnomalyField(anomaly, condition.field);
          break;
        case 'alarm':
          value = this.checkAlarms(alarms, condition.field);
          break;
        case 'metric':
          value = this.getMetricField(metrics, condition.field);
          break;
        default:
          continue;
      }

      if (this.evaluateCondition(value, condition.operator, condition.value)) {
        matchedConditions++;
      }
    }

    return rule.conditions.length > 0 ? matchedConditions / rule.conditions.length : 0;
  }

  /**
   * Evaluate a single condition
   */
  private evaluateCondition(
    actual: unknown,
    operator: string,
    expected: unknown
  ): boolean {
    switch (operator) {
      case 'eq':
        return actual === expected;
      case 'gt':
        return (actual as number) > (expected as number);
      case 'lt':
        return (actual as number) < (expected as number);
      case 'contains':
        return String(actual).includes(String(expected));
      case 'exists':
        return actual !== undefined && actual !== null;
      default:
        return false;
    }
  }

  /**
   * Get field from anomaly
   */
  private getAnomalyField(anomaly: Anomaly, field: string): unknown {
    const fields: Record<string, unknown> = {
      type: anomaly.type,
      severity: anomaly.severity,
      deviation: anomaly.metrics.deviation,
      trend: anomaly.metrics.trend,
    };
    return fields[field];
  }

  /**
   * Check alarms for a condition
   */
  private checkAlarms(alarms: Alarm[], field: string): unknown {
    // Check if any alarm matches the field as alarm code
    return alarms.some((a) => a.alarmCode === field || a.alarmCode.includes(field));
  }

  /**
   * Get field from metrics
   */
  private getMetricField(metrics: CellMetrics, field: string): unknown {
    return (metrics as Record<string, unknown>)[field];
  }

  /**
   * Create pattern signature from current state
   */
  private createSignature(
    anomaly: Anomaly,
    metrics: CellMetrics,
    context: NeuralContext
  ): PatternSignature {
    return {
      anomalyType: anomaly.type,
      severity: anomaly.severity,
      metrics: {
        rsrp: metrics.rsrp,
        sinr: metrics.sinr,
        prbUtil: metrics.prbUtilizationDl,
        interference: metrics.interferenceLevel,
      },
      context: {
        weather: context.weather,
        timeOfDay: context.timeOfDay,
      },
    };
  }

  /**
   * Match patterns against known signatures
   */
  private matchPatterns(signature: PatternSignature): PatternMatch[] {
    const matches: PatternMatch[] = [];

    for (const [patternId, pattern] of this.neuralPatterns) {
      const similarity = this.calculateSimilarity(signature, pattern);
      if (similarity > 0.5) {
        matches.push({
          patternId,
          cause: pattern.rootCause,
          similarity,
          evidence: pattern.evidence,
        });
      }
    }

    return matches.sort((a, b) => b.similarity - a.similarity);
  }

  /**
   * Calculate similarity between signatures
   */
  private calculateSimilarity(a: PatternSignature, b: PatternSignature): number {
    let score = 0;
    let total = 0;

    // Anomaly type match (weight: 0.3)
    if (a.anomalyType === b.anomalyType) score += 0.3;
    total += 0.3;

    // Severity proximity (weight: 0.2)
    const sevDiff = Math.abs(a.severity - b.severity);
    score += 0.2 * (1 - sevDiff);
    total += 0.2;

    // Context match (weight: 0.2)
    if (a.context.weather === b.context.weather) score += 0.1;
    if (a.context.timeOfDay === b.context.timeOfDay) score += 0.1;
    total += 0.2;

    // Metric proximity (weight: 0.3)
    const metricSim = this.calculateMetricSimilarity(a.metrics, b.metrics);
    score += 0.3 * metricSim;
    total += 0.3;

    return score / total;
  }

  /**
   * Calculate metric similarity
   */
  private calculateMetricSimilarity(
    a: PatternSignature['metrics'],
    b: PatternSignature['metrics']
  ): number {
    const fields = ['rsrp', 'sinr', 'prbUtil', 'interference'];
    let similarity = 0;

    for (const field of fields) {
      const aVal = a[field as keyof typeof a] as number;
      const bVal = b[field as keyof typeof b] as number;
      if (aVal !== undefined && bVal !== undefined) {
        const maxVal = Math.max(Math.abs(aVal), Math.abs(bVal), 1);
        similarity += 1 - Math.abs(aVal - bVal) / maxVal;
      }
    }

    return similarity / fields.length;
  }

  /**
   * Weather-based reasoning
   */
  private reasonAboutWeather(
    anomaly: Anomaly,
    weather: string
  ): ProbableCause | null {
    if (anomaly.type === 'vswr_high' && weather === 'rain') {
      return {
        cause: 'water_ingress_connector',
        probability: 0.75,
        evidence: ['VSWR anomaly during rain', 'Common water ingress pattern'],
      };
    }

    if (anomaly.type === 'rssi_drop' && (weather === 'rain' || weather === 'storm')) {
      return {
        cause: 'atmospheric_attenuation',
        probability: 0.6,
        evidence: ['Signal degradation during precipitation', 'Rain fade effect'],
      };
    }

    return null;
  }

  /**
   * Temporal reasoning
   */
  private reasonAboutTime(
    anomaly: Anomaly,
    timeOfDay: string,
    dayOfWeek?: string
  ): ProbableCause | null {
    if (anomaly.type === 'traffic_spike') {
      if (timeOfDay === 'morning' && dayOfWeek === 'weekday') {
        return {
          cause: 'rush_hour_congestion',
          probability: 0.7,
          evidence: ['Weekday morning traffic pattern', 'Commuter activity'],
        };
      }
      if (timeOfDay === 'evening' && dayOfWeek === 'weekend') {
        return {
          cause: 'entertainment_district_load',
          probability: 0.65,
          evidence: ['Weekend evening activity', 'Nightlife pattern'],
        };
      }
    }

    return null;
  }

  /**
   * Initialize symbolic rules
   */
  private initializeSymbolicRules(): SymbolicRule[] {
    return [
      {
        id: 'VSWR_RAIN',
        conditions: [
          { type: 'anomaly', field: 'type', operator: 'eq', value: 'vswr_high' },
        ],
        conclusion: 'water_ingress_connector',
        confidence: 0.8,
        priority: 'low',
      },
      {
        id: 'SLEEPING_CELL',
        conditions: [
          { type: 'anomaly', field: 'type', operator: 'eq', value: 'sleeping_cell' },
          { type: 'metric', field: 'activeUesDl', operator: 'eq', value: 0 },
        ],
        conclusion: 'sleeping_cell_lockup',
        confidence: 0.9,
        priority: 'critical',
      },
      {
        id: 'INTERFERENCE_HIGH',
        conditions: [
          { type: 'anomaly', field: 'type', operator: 'eq', value: 'interference_spike' },
          { type: 'metric', field: 'sinr', operator: 'lt', value: 0 },
        ],
        conclusion: 'external_interference',
        confidence: 0.75,
        priority: 'medium',
      },
      {
        id: 'RSSI_DROP_HW',
        conditions: [
          { type: 'anomaly', field: 'type', operator: 'eq', value: 'rssi_drop' },
          { type: 'anomaly', field: 'severity', operator: 'gt', value: 0.8 },
        ],
        conclusion: 'hardware_failure_antenna',
        confidence: 0.85,
        priority: 'high',
      },
      {
        id: 'TRAFFIC_OVERLOAD',
        conditions: [
          { type: 'anomaly', field: 'type', operator: 'eq', value: 'traffic_spike' },
          { type: 'metric', field: 'prbUtilizationDl', operator: 'gt', value: 0.9 },
        ],
        conclusion: 'traffic_overload',
        confidence: 0.8,
        priority: 'medium',
      },
    ];
  }

  /**
   * Initialize neural patterns
   */
  private initializeNeuralPatterns(): Map<string, PatternSignature> {
    const patterns = new Map<string, PatternSignature>();

    patterns.set('PATTERN_WATER_INGRESS', {
      anomalyType: 'vswr_high',
      severity: 0.7,
      metrics: { rsrp: -85, sinr: 5, prbUtil: 0.3, interference: -100 },
      context: { weather: 'rain' },
      rootCause: 'water_ingress_connector',
      evidence: ['Historical water ingress pattern', 'Weather correlation'],
    });

    patterns.set('PATTERN_HW_FAILURE', {
      anomalyType: 'rssi_drop',
      severity: 0.9,
      metrics: { rsrp: -110, sinr: -5, prbUtil: 0.1, interference: -95 },
      context: {},
      rootCause: 'hardware_failure_antenna',
      evidence: ['Severe signal degradation', 'Hardware fault signature'],
    });

    patterns.set('PATTERN_INTERFERENCE', {
      anomalyType: 'interference_spike',
      severity: 0.6,
      metrics: { rsrp: -90, sinr: -2, prbUtil: 0.4, interference: -85 },
      context: {},
      rootCause: 'external_interference',
      evidence: ['Elevated noise floor', 'External source likely'],
    });

    return patterns;
  }

  /**
   * Get reasoning history
   */
  getReasoningHistory(): ReasoningSession[] {
    return [...this.reasoningHistory];
  }

  /**
   * Clear reasoning history
   */
  clearHistory(): void {
    this.reasoningHistory = [];
  }
}

/**
 * Pattern signature for neural matching
 */
interface PatternSignature {
  anomalyType: AnomalyType;
  severity: number;
  metrics: {
    rsrp: number;
    sinr: number;
    prbUtil: number;
    interference: number;
  };
  context: {
    weather?: string;
    timeOfDay?: string;
  };
  rootCause?: string;
  evidence?: string[];
}

/**
 * Pattern match result
 */
interface PatternMatch {
  patternId: string;
  cause: string;
  similarity: number;
  evidence: string[];
}

/**
 * Reasoning session record
 */
interface ReasoningSession {
  timestamp: number;
  anomalyId: string;
  causes: ProbableCause[];
  executionTimeMs: number;
}

/**
 * Create a configured Neuro-Symbolic RCA instance
 */
export function createNeuroSymbolicRCA(): NeuroSymbolicRCA {
  return new NeuroSymbolicRCA();
}
