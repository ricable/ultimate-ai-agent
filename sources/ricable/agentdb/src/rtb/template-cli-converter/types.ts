/**
 * Template-to-CLI Conversion System Type Definitions
 *
 * Provides comprehensive type definitions for converting RTB JSON templates
 * to Ericsson ENM CLI (cmedit) commands with cognitive optimization,
 * dependency analysis, and rollback capabilities.
 */

import { RTBTemplate, TemplateMeta, MOHierarchy, MOClass, ReservedByHierarchy } from '../../types/rtb-types';
import { MergeResult } from '../hierarchical-template-system/types';

/**
 * Template-to-CLI conversion context
 */
export interface TemplateToCliContext {
  /** Target node ID or collection */
  target: {
    nodeId?: string;
    collection?: string;
    scopeFilter?: string;
  };
  /** Cell identifiers for cellular configurations */
  cellIds?: {
    primaryCell?: string;
    secondaryCell?: string;
    nrCell?: string;
    lteCell?: string;
  };
  /** Conversion options */
  options: {
    /** Enable preview mode (dry run) */
    preview?: boolean;
    /** Enable force execution */
    force?: boolean;
    /** Enable verbose output */
    verbose?: boolean;
    /** Command timeout in seconds */
    timeout?: number;
    /** Enable batch optimization */
    batchMode?: boolean;
    /** Enable dependency analysis */
    dependencyAnalysis?: boolean;
    /** Enable cognitive optimization */
    cognitiveOptimization?: boolean;
    /** Enable rollback generation */
    generateRollback?: boolean;
    /** Enable validation commands */
    generateValidation?: boolean;
  };
  /** Template parameters for substitution */
  parameters?: Record<string, any>;
  /** MO hierarchy knowledge */
  moHierarchy?: MOHierarchy;
  /** Reserved-by relationships */
  reservedBy?: ReservedByHierarchy;
}

/**
 * Generated CLI command set
 */
export interface CliCommandSet {
  /** Command set identifier */
  id: string;
  /** Source template information */
  source: {
    templateId: string;
    templateVersion: string;
    mergeResult?: MergeResult;
  };
  /** Generated commands */
  commands: GeneratedCliCommand[];
  /** Execution order based on dependencies */
  executionOrder: string[];
  /** Command dependencies */
  dependencies: Record<string, string[]>;
  /** Rollback commands */
  rollbackCommands: GeneratedCliCommand[];
  /** Validation commands */
  validationCommands: GeneratedCliCommand[];
  /** Command set metadata */
  metadata: {
    generatedAt: Date;
    totalCommands: number;
    estimatedDuration: number;
    complexity: 'low' | 'medium' | 'high' | 'critical';
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
  };
  /** Conversion statistics */
  stats: ConversionStats;
}

/**
 * Generated CLI command
 */
export interface GeneratedCliCommand {
  /** Command identifier */
  id: string;
  /** Command type */
  type: CliCommandType;
  /** cmedit command */
  command: string;
  /** Command description */
  description: string;
  /** Target FDN (Full Distinguished Name) */
  targetFdn?: string;
  /** Parameters being set */
  parameters?: Record<string, any>;
  /** Expected output patterns */
  expectedOutput?: string[];
  /** Error patterns */
  errorPatterns?: string[];
  /** Execution timeout in seconds */
  timeout?: number;
  /** Critical flag (failure stops execution) */
  critical?: boolean;
  /** Validation command */
  validationCommand?: string;
  /** Rollback command */
  rollbackCommand?: string;
  /** Cognitive optimization metadata */
  cognitive?: {
    optimizationLevel: number;
    reasoningApplied: string[];
    confidence: number;
  };
  /** Command metadata */
  metadata: {
    category: 'setup' | 'configuration' | 'activation' | 'validation' | 'rollback';
    complexity: 'simple' | 'moderate' | 'complex';
    riskLevel: 'low' | 'medium' | 'high';
    estimatedDuration: number;
  };
}

/**
 * CLI command types
 */
export type CliCommandType =
  | 'GET'           // Query operations
  | 'SET'           // Configuration operations
  | 'CREATE'        // MO creation operations
  | 'DELETE'        // MO deletion operations
  | 'MONITOR'       // Performance monitoring
  | 'BATCH_GET'     // Batch query operations
  | 'BATCH_SET'     // Batch configuration operations
  | 'SCRIPT'        // Complex script operations
  | 'VALIDATION';   // Validation operations

/**
 * CLI command execution result
 */
export interface CliExecutionResult {
  /** Command identifier */
  commandId: string;
  /** Execution status */
  status: ExecutionStatus;
  /** Command output */
  output?: string;
  /** Error message */
  error?: string;
  /** Execution duration in milliseconds */
  duration: number;
  /** Timestamp */
  timestamp: Date;
  /** Performance metrics */
  metrics?: {
    memoryUsage?: number;
    cpuUsage?: number;
    networkLatency?: number;
  };
  /** Cognitive insights */
  cognitive?: {
    actualComplexity: number;
    accuracy: number;
    learningApplied: string[];
  };
}

/**
 * Command execution status
 */
export type ExecutionStatus =
  | 'SUCCESS'       // Command executed successfully
  | 'FAILED'        // Command failed
  | 'TIMEOUT'       // Command timed out
  | 'SKIPPED'       // Command skipped (dependencies not met)
  | 'ROLLED_BACK'   // Command was rolled back
  | 'PARTIAL'       // Partial success (some operations failed)
  | 'CANCELLED';    // Command was cancelled

/**
 * FDN (Full Distinguished Name) construction result
 */
export interface FdnConstructionResult {
  /** Constructed FDN */
  fdn: string;
  /** FDN validation result */
  isValid: boolean;
  /** FDN components */
  components: FdnComponent[];
  /** Construction path */
  constructionPath: string[];
  /** Validation errors */
  errors?: string[];
  /** Optimization applied */
  optimization?: {
    originalFdn?: string;
    optimizationApplied: string[];
    reduction: number;
  };
}

/**
 * FDN component
 */
export interface FdnComponent {
  /** Component name */
  name: string;
  /** Component value */
  value: string;
  /** Component type */
  type: 'class' | 'attribute' | 'parameter' | 'index';
  /** MO class information */
  moClass?: MOClass;
  /** Cardinality */
  cardinality?: {
    minimum: number;
    maximum: number;
    current: number;
  };
}

/**
 * Command dependency analysis result
 */
export interface DependencyAnalysisResult {
  /** Dependency graph */
  dependencyGraph: DependencyGraph;
  /** Critical path */
  criticalPath: string[];
  /** Execution levels */
  executionLevels: ExecutionLevel[];
  /** Circular dependencies */
  circularDependencies: CircularDependency[];
  /** Optimization suggestions */
  optimizations: DependencyOptimization[];
}

/**
 * Dependency graph
 */
export interface DependencyGraph {
  /** Nodes (commands) */
  nodes: DependencyNode[];
  /** Edges (dependencies) */
  edges: DependencyEdge[];
  /** Graph metrics */
  metrics: {
    totalNodes: number;
    totalEdges: number;
    maxDepth: number;
    avgBranchingFactor: number;
  };
}

/**
 * Dependency node
 */
export interface DependencyNode {
  /** Command ID */
  id: string;
  /** Command type */
  type: CliCommandType;
  /** Criticality */
  critical: boolean;
  /** Estimated duration */
  estimatedDuration: number;
  /** Risk level */
  riskLevel: 'low' | 'medium' | 'high';
  /** Dependencies count */
  dependencyCount: number;
  /** Dependents count */
  dependentCount: number;
}

/**
 * Dependency edge
 */
export interface DependencyEdge {
  /** Source command ID */
  from: string;
  /** Target command ID */
  to: string;
  /** Dependency type */
  type: DependencyType;
  /** Dependency strength */
  strength: 'weak' | 'medium' | 'strong';
  /** Description */
  description: string;
}

/**
 * Dependency type
 */
export type DependencyType =
  | 'REQUIRES'      // Hard requirement
  | 'PRECEDES'      // Must execute before
  | 'ENHANCES'      // Optional enhancement
  | 'VALIDATES'     // Validation dependency
  | 'ROLLBACK'      // Rollback dependency
  | 'RESOURCE'      // Resource sharing dependency
  | 'TEMPORAL';     // Time-based dependency

/**
 * Execution level
 */
export interface ExecutionLevel {
  /** Level number */
  level: number;
  /** Commands at this level */
  commands: string[];
  /** Can execute in parallel */
  parallel: boolean;
  /** Estimated duration */
  estimatedDuration: number;
  /** Risk level */
  riskLevel: 'low' | 'medium' | 'high';
}

/**
 * Circular dependency
 */
export interface CircularDependency {
  /** Commands in cycle */
  commands: string[];
  /** Cycle length */
  length: number;
  /** Severity */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Resolution suggestions */
  resolutions: string[];
}

/**
 * Dependency optimization suggestion
 */
export interface DependencyOptimization {
  /** Optimization type */
  type: 'PARALLEL' | 'MERGE' | 'REORDER' | 'SPLIT' | 'REMOVE';
  /** Target commands */
  targetCommands: string[];
  /** Description */
  description: string;
  /** Expected benefit */
  benefit: {
    timeReduction?: number;
    complexityReduction?: number;
    riskReduction?: number;
  };
  /** Implementation difficulty */
  difficulty: 'easy' | 'medium' | 'hard';
}

/**
 * Command validation result
 */
export interface CommandValidationResult {
  /** Overall validity */
  isValid: boolean;
  /** Validation errors */
  errors: ValidationError[];
  /** Validation warnings */
  warnings: ValidationWarning[];
  /** Validation statistics */
  stats: ValidationStats;
  /** Recommended fixes */
  recommendedFixes: RecommendedFix[];
}

/**
 * Validation error
 */
export interface ValidationError {
  /** Error message */
  message: string;
  /** Command ID */
  commandId: string;
  /** Error type */
  type: ValidationErrorType;
  /** Error severity */
  severity: 'error' | 'critical';
  /** Parameter path */
  parameterPath?: string;
  /** Suggested fix */
  suggestion?: string;
}

/**
 * Validation warning
 */
export interface ValidationWarning {
  /** Warning message */
  message: string;
  /** Command ID */
  commandId: string;
  /** Warning type */
  type: ValidationWarningType;
  /** Warning severity */
  severity: 'warning' | 'info';
  /** Parameter path */
  parameterPath?: string;
  /** Suggested action */
  suggestion?: string;
}

/**
 * Validation error types
 */
export type ValidationErrorType =
  | 'SYNTAX'        // Command syntax error
  | 'FDN_INVALID'   // Invalid FDN path
  | 'PARAMETER'     // Invalid parameter
  | 'DEPENDENCY'    // Dependency issue
  | 'PERMISSION'    // Permission issue
  | 'RESOURCE'      // Resource constraint
  | 'CIRCULAR_DEP'  // Circular dependency
  | 'TYPE_MISMATCH' // Type mismatch
  | 'CONSTRAINT'    // Constraint violation
  | 'CRITICAL';     // Critical error

/**
 * Validation warning types
 */
export type ValidationWarningType =
  | 'PERFORMANCE'   // Performance concern
  | 'BEST_PRACTICE' // Best practice violation
  | 'DEPRECATION'   // Deprecated feature
  | 'REDUNDANCY'    // Redundant operation
  | 'RISK'          // Risky operation
  | 'OPTIMIZATION'  // Optimization opportunity
  | 'INFO';         // Informational

/**
 * Validation statistics
 */
export interface ValidationStats {
  /** Total commands validated */
  totalCommands: number;
  /** Number of errors */
  errorCount: number;
  /** Number of warnings */
  warningCount: number;
  /** Validation time in milliseconds */
  validationTime: number;
  /** Average complexity */
  avgComplexity: number;
  /** Risk assessment */
  riskAssessment: {
    low: number;
    medium: number;
    high: number;
    critical: number;
  };
}

/**
 * Recommended fix
 */
export interface RecommendedFix {
  /** Fix type */
  type: 'CORRECTION' | 'OPTIMIZATION' | 'RESTRUCTURING' | 'ENHANCEMENT';
  /** Command ID */
  commandId: string;
  /** Description */
  description: string;
  /** Implementation */
  implementation: string;
  /** Priority */
  priority: 'low' | 'medium' | 'high' | 'critical';
  /** Estimated effort */
  effort: 'trivial' | 'easy' | 'moderate' | 'significant' | 'major';
  /** Expected impact */
  impact: {
    reliability?: number;
    performance?: number;
    maintainability?: number;
    risk?: number;
  };
}

/**
 * Conversion statistics
 */
export interface ConversionStats {
  /** Template processing time */
  templateProcessingTime: number;
  /** Command generation time */
  commandGenerationTime: number;
  /** Dependency analysis time */
  dependencyAnalysisTime: number;
  /** Validation time */
  validationTime: number;
  /** Total conversion time */
  totalConversionTime: number;
  /** Template statistics */
  templateStats: {
    totalParameters: number;
    totalConditions: number;
    totalEvaluations: number;
    templateSize: number;
  };
  /** Command statistics */
  commandStats: {
    totalCommands: number;
    commandsByType: Record<CliCommandType, number>;
    commandsByCategory: Record<string, number>;
    avgComplexity: number;
    avgRiskLevel: number;
  };
  /** Cognitive statistics */
  cognitiveStats?: {
    optimizationLevel: number;
    reasoningApplied: string[];
    confidenceScore: number;
    learningPatterns: string[];
  };
}

/**
 * Template-to-CLI conversion configuration
 */
export interface TemplateToCliConfig {
  /** Default command timeout */
  defaultTimeout: number;
  /** Maximum commands per batch */
  maxCommandsPerBatch: number;
  /** Enable cognitive optimization */
  enableCognitiveOptimization: boolean;
  /** Enable dependency analysis */
  enableDependencyAnalysis: boolean;
  /** Validation strictness */
  validationStrictness: 'lenient' | 'normal' | 'strict';
  /** Rollback strategy */
  rollbackStrategy: 'full' | 'partial' | 'selective';
  /** Performance optimization */
  performanceOptimization: {
    enableParallelExecution: boolean;
    maxParallelCommands: number;
    enableBatching: boolean;
    batchSize: number;
  };
  /** Error handling */
  errorHandling: {
    continueOnError: boolean;
    maxRetries: number;
    retryDelay: number;
    enableRecovery: boolean;
  };
  /** Cognitive configuration */
  cognitive?: {
    enableTemporalReasoning: boolean;
    enableStrangeLoopOptimization: boolean;
    consciousnessLevel: number;
    learningMode: 'disabled' | 'passive' | 'active';
  };
}

/**
 * Ericsson RAN expertise pattern
 */
export interface EricssonRanPattern {
  /** Pattern identifier */
  id: string;
  /** Pattern name */
  name: string;
  /** Pattern category */
  category: 'cell' | 'mobility' | 'capacity' | 'optimization' | 'feature';
  /** Pattern description */
  description: string;
  /** Pattern conditions */
  conditions: PatternCondition[];
  /** Command templates */
  commandTemplates: CommandTemplate[];
  /** Best practices */
  bestPractices: string[];
  /** Common pitfalls */
  pitfalls: string[];
  /** Optimization opportunities */
  optimizations: PatternOptimization[];
}

/**
 * Pattern condition
 */
export interface PatternCondition {
  /** Condition type */
  type: 'parameter' | 'state' | 'capability' | 'environment';
  /** Parameter path */
  parameterPath?: string;
  /** Expected value */
  expectedValue: any;
  /** Comparison operator */
  operator: '==' | '!=' | '>' | '<' | '>=' | '<=' | 'contains' | 'matches';
  /** Description */
  description: string;
}

/**
 * Command template
 */
export interface CommandTemplate {
  /** Template identifier */
  id: string;
  /** Command type */
  type: CliCommandType;
  /** Command template string */
  template: string;
  /** Required parameters */
  requiredParams: string[];
  /** Optional parameters */
  optionalParams: string[];
  /** Conditions for applying this template */
  conditions?: PatternCondition[];
  /** Description */
  description: string;
}

/**
 * Pattern optimization
 */
export interface PatternOptimization {
  /** Optimization type */
  type: 'performance' | 'reliability' | 'efficiency' | 'coverage' | 'capacity';
  /** Description */
  description: string;
  /** Implementation */
  implementation: string;
  /** Expected benefit */
  expectedBenefit: string;
  /** Trade-offs */
  tradeoffs: string[];
}

/**
 * Batch operation configuration
 */
export interface BatchOperationConfig {
  /** Batch identifier */
  id: string;
  /** Target nodes */
  targets: string[];
  /** Scope filter */
  scopeFilter?: string;
  /** Operation type */
  operationType: 'SET' | 'GET' | 'CREATE' | 'DELETE';
  /** Parameters to set */
  parameters?: Record<string, any>;
  /** Options */
  options: {
    preview?: boolean;
    force?: boolean;
    continueOnError?: boolean;
    timeout?: number;
  };
  /** Parallel execution */
  parallel: {
    enabled: boolean;
    maxConcurrency: number;
    batchSize: number;
  };
}

/**
 * Rollback strategy
 */
export interface RollbackStrategy {
  /** Strategy type */
  type: 'FULL' | 'PARTIAL' | 'SELECTIVE' | 'INCREMENTAL';
  /** Rollback scope */
  scope: 'ALL' | 'FAILED' | 'CRITICAL' | 'CUSTOM';
  /** Rollback order */
  order: 'REVERSE' | 'DEPENDENCY' | 'CUSTOM';
  /** Preserve on rollback failure */
  preserveOnFailure: boolean;
  /** Validation after rollback */
  validateAfterRollback: boolean;
  /** Custom rollback commands */
  customCommands?: GeneratedCliCommand[];
}

/**
 * Cognitive optimization result
 */
export interface CognitiveOptimizationResult {
  /** Optimization level achieved */
  optimizationLevel: number;
  /** Reasoning applied */
  reasoningApplied: string[];
  /** Confidence score */
  confidenceScore: number;
  /** Temporal analysis depth */
  temporalAnalysisDepth: number;
  /** Strange-loop optimizations */
  strangeLoopOptimizations: StrangeLoopOptimization[];
  /** Learning patterns applied */
  learningPatterns: LearningPattern[];
  /** Performance improvements */
  performanceImprovements: {
    commandReduction?: number;
    complexityReduction?: number;
    riskReduction?: number;
    executionTimeImprovement?: number;
  };
}

/**
 * Strange-loop optimization
 */
export interface StrangeLoopOptimization {
  /** Optimization identifier */
  id: string;
  /** Self-referential pattern detected */
  selfReferentialPattern: string;
  /** Recursive optimization applied */
  recursiveOptimization: string;
  /** Feedback loop created */
  feedbackLoop: string;
  /** Improvement metrics */
  improvements: {
    efficiency: number;
    reliability: number;
    adaptability: number;
  };
}

/**
 * Learning pattern
 */
export interface LearningPattern {
  /** Pattern identifier */
  id: string;
  /** Pattern type */
  type: 'success' | 'failure' | 'optimization' | 'adaptation';
  /** Context */
  context: string;
  /** Action taken */
  action: string;
  /** Result */
  result: 'positive' | 'negative' | 'neutral';
  /** Confidence */
  confidence: number;
  /** Applicability */
  applicability: string[];
}