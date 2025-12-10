/**
 * Adaptive Coordinator Types
 * Core type definitions for adaptive swarm coordination system
 */

export interface Agent {
  id: string;
  type: AgentType;
  capabilities: AgentCapability[];
  status: AgentStatus;
  performance: AgentPerformance;
  resources: AgentResources;
  location: AgentLocation;
  metadata: Record<string, any>;
}

export type AgentType =
  | 'coordinator'
  | 'researcher'
  | 'coder'
  | 'analyst'
  | 'optimizer'
  | 'monitor'
  | 'tester'
  | 'reviewer'
  | 'specialist'
  | 'architect'
  | 'task-orchestrator'
  | 'code-analyzer'
  | 'perf-analyzer'
  | 'api-docs'
  | 'performance-benchmarker'
  | 'system-architect';

export interface AgentCapability {
  name: string;
  level: number; // 1-10 proficiency level
  specializations: string[];
  performanceHistory: CapabilityPerformance[];
}

export interface CapabilityPerformance {
  timestamp: Date;
  taskType: string;
  success: boolean;
  executionTime: number;
  quality: number; // 0-1 quality score
  resourceUsage: number; // 0-1 resource utilization
}

export type AgentStatus =
  | 'idle'
  | 'active'
  | 'busy'
  | 'overloaded'
  | 'recovering'
  | 'offline';

export interface AgentPerformance {
  currentLoad: number; // 0-1 current workload
  averageResponseTime: number; // milliseconds
  successRate: number; // 0-1 task success rate
  qualityScore: number; // 0-1 overall quality rating
  efficiency: number; // 0-1 resource efficiency
  reliability: number; // 0-1 reliability score
}

export interface AgentResources {
  cpuUsage: number; // 0-1 CPU utilization
  memoryUsage: number; // 0-1 memory utilization
  networkUsage: number; // 0-1 network utilization
  availableCapacity: number; // 0-1 available capacity
  resourceScore: number; // 0-1 overall resource health
}

export interface AgentLocation {
  nodeId?: string;
  region?: string;
  datacenter?: string;
  networkSegment?: string;
  coordinates?: { x: number; y: number };
}

export interface AgentAssignment {
  agentId: string;
  newRole: string;
  newCapabilities: AgentCapability[];
  location?: AgentLocation;
  migrationPlan?: MigrationPlan;
}

export interface MigrationPlan {
  steps: MigrationStep[];
  estimatedTime: number; // milliseconds
  rollbackAvailable: boolean;
  validationRequired: boolean;
}

export interface MigrationStep {
  stepId: string;
  action: string;
  targetAgent: string;
  dependencies: string[];
  estimatedDuration: number; // milliseconds
  rollbackAction?: string;
}

export interface ValidationStep {
  stepId: string;
  validationType: 'connectivity' | 'performance' | 'functionality' | 'consensus';
  criteria: ValidationCriteria;
  timeout: number; // milliseconds
  critical: boolean;
}

export interface ValidationCriteria {
  successThreshold: number; // Minimum success rate (0-1)
  performanceThreshold: number; // Maximum acceptable response time (ms)
  qualityThreshold: number; // Minimum quality score (0-1)
  customChecks?: Record<string, any>;
}

export interface WorkloadPattern {
  patternId: string;
  patternType: WorkloadType;
  characteristics: WorkloadCharacteristics;
  seasonalFactors: SeasonalFactor[];
  predictions: WorkloadPrediction[];
}

export type WorkloadType =
  | 'cpu-intensive'
  | 'memory-intensive'
  | 'network-intensive'
  | 'io-intensive'
  | 'mixed'
  | 'burst'
  | 'steady'
  | 'periodic';

export interface WorkloadCharacteristics {
  complexity: number; // 0-1 complexity score
  parallelizability: number; // 0-1 parallel execution capability
  interdependencies: number; // 0-1 interdependency score
  resourceRequirements: ResourceRequirements;
  timeSensitivity: number; // 0-1 urgency level
  predictability: number; // 0-1 predictability score
}

export interface ResourceRequirements {
  minCpuCores: number;
  minMemoryGB: number;
  minNetworkMbps: number;
  minStorageGB: number;
  estimatedDuration: number; // milliseconds
  scalability: 'horizontal' | 'vertical' | 'both';
}

export interface SeasonalFactor {
  factor: string;
  impact: number; // -1 to 1 impact on workload
  timeWindow: TimeWindow;
  confidence: number; // 0-1 confidence in prediction
}

export interface TimeWindow {
  start: Date;
  end: Date;
  recurring: boolean;
  frequency?: 'hourly' | 'daily' | 'weekly' | 'monthly';
}

export interface WorkloadPrediction {
  timestamp: Date;
  predictedLoad: number; // 0-1 predicted system load
  confidence: number; // 0-1 prediction confidence
  resourceNeeds: ResourceRequirements;
  recommendedTopology: string;
  recommendedScale: number;
}

export interface ConsensusParameters {
  algorithm: string;
  timeout: number; // milliseconds
  requiredAgreement: number; // 0-1 minimum agreement
  votingMethod: VotingMethod;
  faultTolerance: FaultToleranceConfig;
}

export type VotingMethod =
  | 'simple-majority'
  | 'supermajority'
  | 'unanimous'
  | 'weighted'
  | 'delegated'
  | 'randomized';

export interface FaultToleranceConfig {
  maxFaultyNodes: number;
  byzantineFaults: boolean;
  crashFaults: boolean;
  networkPartitions: boolean;
  recoveryStrategy: 'automatic' | 'manual' | 'hybrid';
}

export interface PerformanceBaseline {
  metric: string;
  baselineValue: number;
  acceptableVariance: number; // ± percentage variance
  measurementFrequency: number; // milliseconds
  alertThreshold: number; // Alert when deviation exceeds this
}

export interface OptimizationTarget {
  metric: string;
  currentValue: number;
  targetValue: number;
  priority: 'low' | 'medium' | 'high' | 'critical';
  optimizationMethod: OptimizationMethod;
  constraints: OptimizationConstraint[];
}

export type OptimizationMethod =
  | 'gradient-descent'
  | 'genetic-algorithm'
  | 'particle-swarm'
  | 'simulated-annealing'
  | 'reinforcement-learning'
  | 'bayesian-optimization'
  | 'heuristic'
  | 'rule-based';

export interface OptimizationConstraint {
  type: 'upper-bound' | 'lower-bound' | 'equality' | 'inequality';
  parameter: string;
  value: number;
  strictness: 'soft' | 'hard';
}

export interface CognitivePattern {
  patternId: string;
  patternType: PatternType;
  characteristics: PatternCharacteristics;
  effectiveness: PatternEffectiveness;
  applications: PatternApplication[];
}

export type PatternType =
  | 'coordination'
  | 'optimization'
  | 'resource-allocation'
  | 'topology-adaptation'
  | 'consensus-building'
  | 'performance-tuning'
  | 'error-recovery'
  | 'learning-adaptation';

export interface PatternCharacteristics {
  complexity: number; // 0-1 complexity score
  applicability: number; // 0-1 range of applicability
  successRate: number; // 0-1 historical success rate
  resourceEfficiency: number; // 0-1 resource efficiency
  learningCurve: number; // 0-1 learning difficulty
}

export interface PatternEffectiveness {
  performanceImprovement: number; // 0-1 improvement score
  resourceSavings: number; // 0-1 resource savings
  errorReduction: number; // 0-1 error reduction
  adaptationSpeed: number; // 0-1 speed of adaptation
  scalability: number; // 0-1 scalability potential
}

export interface PatternApplication {
  timestamp: Date;
  context: Record<string, any>;
  outcome: ApplicationOutcome;
  metrics: Record<string, number>;
  feedback: string;
}

export interface ApplicationOutcome {
  success: boolean;
  effectiveness: number; // 0-1 effectiveness score
  sideEffects: string[];
  improvementArea: string[];
  lessons: string[];
}

export interface AdaptationStrategy {
  strategyId: string;
  name: string;
  description: string;
  conditions: AdaptationCondition[];
  actions: AdaptationAction[];
  expectedOutcome: ExpectedOutcome;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  rollbackPlan: RollbackPlan;
}

export interface AdaptationCondition {
  parameter: string;
  operator: '>' | '<' | '=' | '>=' | '<=' | '!=' | 'contains' | 'matches';
  threshold: number | string;
  duration: number; // milliseconds condition must persist
  confidence: number; // 0-1 confidence in condition detection
}

export interface AdaptationAction {
  actionType: ActionType;
  parameters: Record<string, any>;
  executionOrder: number;
  dependencies: string[];
  timeout: number; // milliseconds
  validationRequired: boolean;
}

export type ActionType =
  | 'topology-change'
  | 'scale-up'
  | 'scale-down'
  | 'rebalance'
  | 'consensus-update'
  | 'parameter-tuning'
  | 'agent-reassignment'
  | 'resource-reallocation'
  | 'cognitive-update';

export interface ExpectedOutcome {
  performanceGain: number; // 0-1 expected performance improvement
  resourceEfficiency: number; // 0-1 expected resource efficiency
  riskMitigation: number; // 0-1 expected risk mitigation
  adaptationSpeed: number; // 0-1 expected adaptation speed
  confidence: number; // 0-1 confidence in outcome prediction
}

export interface RollbackPlan {
  automaticRollback: boolean;
  rollbackTriggers: RollbackTrigger[];
  rollbackSteps: RollbackStep[];
  maxRollbackTime: number; // milliseconds
  dataConsistencyGuarantee: boolean;
}

export interface RollbackTrigger {
  metric: string;
  threshold: number;
  operator: '>' | '<' | '=';
  evaluationWindow: number; // milliseconds
  consecutiveViolations: number;
}

export interface RollbackStep {
  stepId: string;
  action: string;
  parameters: Record<string, any>;
  executionOrder: number;
  validationStep?: string;
}