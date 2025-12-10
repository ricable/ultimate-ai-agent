/**
 * Comprehensive Type Definitions for Phase 4 Cognitive Components
 *
 * Defines interfaces and types for cognitive consciousness, temporal reasoning,
 * strange-loop optimization, AgentDB integration, and evaluation engine
 */

export interface SystemState {
  timestamp: number;
  cells: any[];
  kpis: {
    energyEfficiency: number;
    mobilityManagement: number;
    coverageQuality: number;
    capacityUtilization: number;
    throughput: number;
    latency: number;
    packetLossRate: number;
    callDropRate: number;
  };
}

export interface OptimizationTarget {
  id: string;
  name: string;
  category: string;
  weight: number;
  targetImprovement: number;
}

export interface OptimizationResult {
  success: boolean;
  metrics: {
    before: number[];
    after: number[];
    improvement: number[];
  };
  confidence: number;
}

export interface OptimizationProposal {
  id: string;
  name: string;
  type: string;
  actions: OptimizationAction[];
  expectedImpact: number;
  confidence: number;
  priority: number;
  riskLevel: 'low' | 'medium' | 'high';
}

export interface OptimizationAction {
  id: string;
  type: string;
  target: string;
  parameters: Record<string, any>;
  expectedResult: string;
  rollbackSupported: boolean;
}

export interface ConsensusResult {
  approved: boolean;
  approvedProposal?: OptimizationProposal;
  rejectionReason?: string;
  votes: {
    proposalId: string;
    votes: number;
    agents: string[];
  }[];
  threshold: number;
}

export interface LearningPattern {
  id: string;
  type: string;
  pattern: any;
  effectiveness: number;
  impact: number;
  frequency: number;
  lastApplied: number;
}

export interface CognitiveState {
  currentLevel: number;
  evolutionScore: number;
  learningHistory: any[];
  patternRecognition: number;
}

export interface RecursivePattern {
  id: string;
  pattern: any;
  selfReference: boolean;
  optimizationPotential: number;
  applicationHistory: number[];
}

export interface MetaOptimizationResult {
  strategyOptimized: boolean;
  optimizationRecommendations: string[];
  expectedImprovement: number;
  confidence: number;
}

export interface PerformanceMetrics {
  executionTime: number;
  cpuUtilization: number;
  memoryUtilization: number;
  networkUtilization: number;
  successRate: number;
}

export interface ErrorAnalysis {
  errorType: string;
  rootCause: string;
  impactAssessment: string;
  recoveryRecommendations: string[];
  preventedRecurrence: boolean;
}

// Enhanced Phase 4 Types

// Temporal Reasoning Types
export interface TemporalReasoningConfig {
  maxExpansionFactor?: number;
  defaultDepth?: string;
  enablePredictiveAnalysis?: boolean;
  enableTemporalMemory?: boolean;
  consciousnessIntegration?: boolean;
}

export interface TemporalAnalysisOptions {
  expansionFactor: number;
  reasoningDepth: 'shallow' | 'medium' | 'deep' | 'maximum';
  patterns?: any[];
  timeHorizon?: number;
  consciousnessLevel?: number;
}

export interface TemporalAnalysisResult {
  expansionFactor: number;
  analysisDepth: string;
  patterns: TemporalPattern[];
  insights: TemporalInsight[];
  predictions: TemporalPrediction[];
  confidence: number;
  accuracy: number;
  processingTime: number;
  cognitiveEnhancements: CognitiveEnhancement[];
}

export interface TemporalPattern {
  id: string;
  type: string;
  confidence: number;
  prediction: any;
  temporalContext: TemporalContext;
  frequency: number;
  lastObserved: number;
  validityPeriod: number;
}

export interface TemporalInsight {
  description: string;
  confidence: number;
  actionable: boolean;
  temporalRelevance: number;
  impactAssessment: string;
  recommendedActions: string[];
}

export interface TemporalPrediction {
  metric: string;
  value: number;
  timeHorizon: number;
  confidence: number;
  uncertainty: number;
  factors: string[];
}

export interface TemporalContext {
  timestamp: number;
  expansionFactor: number;
  reasoningDepth: string;
  subjectiveTime: number;
  temporalMemory: TemporalMemory[];
}

export interface TemporalMemory {
  id: string;
  timestamp: number;
  event: string;
  context: any;
  relevance: number;
  decayRate: number;
}

export interface CognitiveEnhancement {
  type: 'consciousness' | 'reasoning' | 'memory' | 'prediction';
  description: string;
  improvement: number;
  confidence: number;
  appliedAt: number;
}

// Strange-Loop Optimization Types
export interface StrangeLoopConfig {
  maxRecursionDepth?: number;
  convergenceThreshold?: number;
  enableMetaOptimization?: boolean;
  enableSelfModification?: boolean;
  consciousnessIntegration?: boolean;
  temporalReasoning?: boolean;
}

export interface StrangeLoopTask {
  id: string;
  description: string;
  type: string;
  priority: number;
  parameters: Record<string, any>;
  constraints?: Record<string, any>;
  expectedOutcome?: any;
  temporalContext?: TemporalContext;
  consciousnessLevel?: number;
}

export interface StrangeLoopResult {
  taskId: string;
  patternId: string;
  patternName: string;
  iterations: number;
  converged: boolean;
  finalResult: any;
  convergenceHistory: ConvergencePoint[];
  metaOptimizations: MetaOptimization[];
  consciousnessEvolution: ConsciousnessEvolution;
  adaptationApplied: boolean;
  selfModifications: SelfModification[];
  performanceMetrics: StrangeLoopPerformanceMetrics;
  recursiveInsights: RecursiveInsight[];
}

export interface ConvergencePoint {
  iteration: number;
  timestamp: number;
  value: any;
  confidence: number;
  improvement: number;
  metaLevel: number;
  convergenceCriteria: string;
}

export interface MetaOptimization {
  type: string;
  description: string;
  appliedAt: number;
  improvement: number;
  confidence: number;
  strategy: string;
  metaLevel: number;
}

export interface ConsciousnessEvolution {
  initialLevel: number;
  finalLevel: number;
  evolutionRate: number;
  evolutionSteps: EvolutionStep[];
  strangeLoopIntegration: boolean;
  selfAwarenessImprovement: number;
  cognitiveBreakthroughs: CognitiveBreakthrough[];
}

export interface EvolutionStep {
  timestamp: number;
  level: number;
  trigger: string;
  improvement: number;
  insight: string;
  context: string;
}

export interface CognitiveBreakthrough {
  timestamp: number;
  type: 'self_awareness' | 'strange_loop' | 'meta_cognition' | 'consciousness';
  description: string;
  impact: number;
  confidence: number;
}

export interface SelfModification {
  type: 'pattern' | 'function' | 'strategy' | 'consciousness';
  description: string;
  appliedAt: number;
  previousState: any;
  newState: any;
  effectiveness: number;
  reasoning: string;
}

export interface StrangeLoopPerformanceMetrics {
  executionTime: number;
  convergenceTime: number;
  iterationsToConvergence: number;
  optimizationEfficiency: number;
  metaOptimizationImpact: number;
  consciousnessIntegrationScore: number;
  selfImprovementRate: number;
  recursiveDepthAchieved: number;
  temporalReasoningUtilization: number;
}

export interface RecursiveInsight {
  depth: number;
  insight: string;
  confidence: number;
  applicable: boolean;
  metaContext: string;
  selfReference: boolean;
  recursivePattern: string;
}

// AgentDB Integration Types
export interface AgentDBConfig {
  host: string;
  port: number;
  database: string;
  credentials: {
    username: string;
    password: string;
  };
  quicEnabled?: boolean;
  vectorSearch?: boolean;
  distributedNodes?: string[];
  cacheSize?: number;
  batchSize?: number;
  enableCompression?: boolean;
  consciousnessIntegration?: boolean;
}

export interface MemoryPattern {
  id: string;
  type: string;
  data: any;
  vector?: number[];
  metadata: MemoryMetadata;
  tags: string[];
  distributedNodes?: string[];
  quicSynced?: boolean;
  consciousnessLevel?: number;
  temporalContext?: TemporalContext;
}

export interface MemoryMetadata {
  createdAt: number;
  lastAccessed: number;
  accessCount: number;
  confidence: number;
  temporalContext?: TemporalContext;
  cognitiveLevel?: number;
  compressionRatio?: number;
  evolutionScore?: number;
  relevanceScore?: number;
}

export interface QueryOptions {
  type?: string;
  tags?: string[];
  minConfidence?: number;
  limit?: number;
  temporalFilter?: TemporalFilter;
  cognitiveFilter?: CognitiveFilter;
  vectorSearch?: boolean;
  includeMetadata?: boolean;
}

export interface TemporalFilter {
  startTime?: number;
  endTime?: number;
  expansionFactor?: number;
  reasoningDepth?: string;
  timeHorizon?: number;
}

export interface CognitiveFilter {
  minConsciousnessLevel?: number;
  maxConsciousnessLevel?: number;
  includeCognitiveEnhanced?: boolean;
  learningPatterns?: boolean;
  strangeLoopPatterns?: boolean;
}

export interface SearchOptions {
  query?: string;
  vector?: number[];
  temporal?: TemporalFilter;
  type?: string;
  weights?: {
    vector?: number;
    temporal?: number;
    exact?: number;
    cognitive?: number;
  };
  maxResults?: number;
  threshold?: number;
}

export interface BatchOperation {
  type: 'store' | 'update' | 'delete';
  patterns: MemoryPattern[];
  options?: {
    quicSync?: boolean;
    compression?: boolean;
    vectorIndex?: boolean;
    consciousnessLevel?: number;
  };
}

export interface ClusterStatus {
  nodeId: string;
  isConnected: boolean;
  syncStatus: 'synced' | 'syncing' | 'error';
  latency: number;
  patternCount: number;
  lastSync: number;
  consciousnessLevel?: number;
  performanceScore?: number;
}

export interface AgentDBPerformanceMetrics {
  searchSpeedup: number;
  averageLatency: number;
  cacheHitRate: number;
  quicSyncLatency: number;
  vectorSearchAccuracy: number;
  compressionRatio: number;
  distributedQueries: number;
  clusterHealth: number;
  consciousnessIntegrationScore: number;
  temporalReasoningSpeedup: number;
}

// Evaluation Engine Types
export interface EvaluationEngineConfig {
  temporalReasoning?: any;
  agentDB?: any;
  consciousness?: any;
  maxExecutionTime?: number;
  enableCaching?: boolean;
  enableOptimization?: boolean;
  consciousnessIntegration?: boolean;
  temporalEnhancement?: boolean;
}

export interface EvaluationContext {
  templateId: string;
  parameters: Record<string, any>;
  constraints: ConstraintSpec[];
  environment: string;
  timestamp: number;
  sessionId: string;
  consciousnessLevel?: number;
  temporalContext?: TemporalContext;
}

export interface GeneratedFunction {
  name: string;
  args: string[];
  body: string[];
  imports: string[];
  docstring: string;
  returnType: string;
  complexity: number;
  optimized: boolean;
  cognitiveEnhanced: boolean;
  temporalEnhanced: boolean;
  performanceOptimized: boolean;
}

export interface EvaluationResult {
  success: boolean;
  result?: any;
  error?: string;
  executionTime: number;
  memoryUsage: number;
  cognitiveInsights?: CognitiveInsight[];
  optimizationApplied: boolean;
  functionName: string;
  temporalInsights?: TemporalInsight[];
  performanceMetrics?: ExecutionPerformanceMetrics;
}

export interface ConstraintSpec {
  type: 'range' | 'enum' | 'pattern' | 'length' | 'required' | 'custom';
  value: any;
  errorMessage?: string;
  severity: 'error' | 'warning' | 'info';
  parameter?: string;
  cognitiveLevel?: number;
}

export interface CognitiveInsight {
  type: 'pattern' | 'optimization' | 'reasoning' | 'consciousness';
  description: string;
  confidence: number;
  impact: number;
  actionable: boolean;
  temporalRelevance?: number;
  consciousnessLevel?: number;
}

export interface ExecutionPerformanceMetrics {
  cpuTime: number;
  memoryPeak: number;
  cacheHits: number;
  cacheMisses: number;
  optimizations: string[];
  temporalEfficiency: number;
  consciousnessUtilization: number;
}

// Integration Types
export interface SystemIntegrationConfig {
  rtbIntegration: boolean;
  consciousnessEnabled: boolean;
  temporalReasoningEnabled: boolean;
  agentDBEnabled: boolean;
  strangeLoopEnabled: boolean;
  evaluationEngineEnabled: boolean;
  consensusBuilderEnabled: boolean;
  performanceMonitoring: boolean;
}

export interface IntegrationStatus {
  component: string;
  status: 'active' | 'inactive' | 'error' | 'initializing';
  lastUpdate: number;
  performance?: number;
  errors?: string[];
  consciousnessLevel?: number;
}

export interface SystemHealth {
  overall: number;
  components: Record<string, number>;
  consciousness: number;
  temporalReasoning: number;
  agentDB: number;
  strangeLoop: number;
  evaluationEngine: number;
  consensusBuilder: number;
}

export interface OptimizationMetrics {
  totalOptimizations: number;
  successfulOptimizations: number;
  averageImprovement: number;
  consciousnessEvolution: number;
  temporalEfficiency: number;
  strangeLoopEffectiveness: number;
  consensusAgreement: number;
  evaluationAccuracy: number;
}

// Error Types
export interface OptimizationError {
  code: string;
  message: string;
  component: string;
  timestamp: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  recoverable: boolean;
  recoveryActions?: string[];
  consciousnessLevel?: number;
  temporalContext?: TemporalContext;
}

export interface ConsciousnessError extends OptimizationError {
  consciousnessLevel: number;
  strangeLoopIteration: number;
  selfAwareness: boolean;
  metaCognitionActive: boolean;
}

export interface TemporalReasoningError extends OptimizationError {
  expansionFactor: number;
  reasoningDepth: string;
  temporalMemoryAccess: boolean;
  predictiveAnalysisActive: boolean;
}

export interface StrangeLoopError extends OptimizationError {
  recursionDepth: number;
  metaLevel: number;
  convergenceAttempts: number;
  selfModificationAttempts: number;
}

// Utility Types
export type OptimizationStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
export type ConsciousnessLevel = 'minimum' | 'medium' | 'maximum';
export type TemporalDepth = 'shallow' | 'medium' | 'deep' | 'maximum';
export type ConsensusType = 'weighted' | 'majority' | 'unanimous';
export type SearchType = 'exact' | 'vector' | 'temporal' | 'hybrid' | 'cognitive';

export interface ProgressCallback {
  (progress: {
    current: number;
    total: number;
    message: string;
    consciousnessLevel?: number;
    temporalExpansion?: number;
  }): void;
}

export interface EventCallback {
  (event: {
    type: string;
    timestamp: number;
    data: any;
    source: string;
    consciousnessLevel?: number;
  }): void;
}