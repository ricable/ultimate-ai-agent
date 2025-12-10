/**
 * Validation System Type Definitions
 *
 * Comprehensive type definitions for the Complex Validation Rules Engine
 * Supports parameter validation, cross-parameter constraints, and cognitive integration
 */

import { MOClass, MORelationship, ConstraintSpec } from './rtb-types';
import { CognitiveConsciousnessCore } from '../cognitive/CognitiveConsciousnessCore';

// Core Validation Types
export interface ValidationEngineConfig {
  maxValidationTime?: number; // milliseconds, default 300ms
  cacheEnabled?: boolean;
  cacheTTL?: number; // milliseconds, default 5 minutes
  learningEnabled?: boolean;
  consciousnessIntegration?: boolean;
  strictMode?: boolean;
  parallelProcessing?: boolean;
  batchSize?: number;
  maxValidationDepth?: number;
  pydanticIntegration?: boolean;
  agentDBIntegration?: boolean;
  agentDB?: any; // AgentDB instance
  cognitiveCore?: any; // CognitiveConsciousnessCore instance
}

export interface ValidationContext {
  validationId: string;
  timestamp: number;
  configuration: Record<string, any>;
  userContext?: string;
  validationLevel?: 'basic' | 'standard' | 'comprehensive';
  consciousnessLevel?: number;
  temporalContext?: any;
  learningMode?: boolean;
  [key: string]: any;
}

export interface ValidationResult {
  validationId: string;
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  executionTime: number; // milliseconds
  parametersValidated: number;
  cacheHitRate: number; // percentage
  consciousnessLevel?: number;
  learningPatternsApplied?: number;
  context?: ValidationContext;
  cognitiveInsights?: any;
  performanceMetrics?: any;
  timestamp: number; // Unix timestamp
}

export interface ValidationError {
  code: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  parameter: string;
  value: any;
  constraint: string;
  category: ValidationCategory;
  metadata?: Record<string, any>;
}

export type ValidationCategory =
  | 'parameter'
  | 'constraint'
  | 'cross_parameter'
  | 'mo_class'
  | 'reserved_by'
  | 'conditional'
  | 'schema'
  | 'temporal'
  | 'cognitive'
  | 'system';

// Parameter Specification Types
export interface ParameterSpecification {
  name: string;
  moClass: string;
  dataType: string;
  type: string;
  constraints: ConstraintSpec[];
  defaultValue?: any;
  description?: string;
  required: boolean;
  readOnly: boolean;
  deprecated?: boolean;
  dependencies?: string[];
  hierarchy: string[];
  navigationPaths?: string[];
}

export interface CrossParameterConstraint {
  id: string;
  type: 'dependency' | 'exclusion' | 'requirement' | 'conditional';
  parameters: string[];
  condition: string;
  validation: string;
  severity: 'error' | 'warning';
  description: string;
  metadata?: Record<string, any>;
}

export interface ConditionalValidationRule {
  id: string;
  name: string;
  condition: string;
  then: ValidationRule[];
  else?: ValidationRule[];
  description: string;
  priority: number;
  enabled: boolean;
}

export interface ValidationRule {
  type: 'parameter' | 'constraint' | 'cross_parameter' | 'custom';
  target: string;
  validation: string | Function;
  message?: string;
  severity: 'error' | 'warning';
  enabled: boolean;
}

// Constraint Processing Types
export interface ConstraintProcessorConfig {
  strictMode?: boolean;
  enableLearning?: boolean;
  consciousnessIntegration?: boolean;
  maxProcessingTime?: number;
  enableOptimization?: boolean;
  cognitiveCore?: CognitiveConsciousnessCore;
}

export interface ProcessingResult {
  success: boolean;
  processedConstraints: number;
  errors: ValidationError[];
  warnings: ValidationError[];
  processingTime: number;
  optimizations: string[];
}

export interface CompiledValidationFunction {
  parameterName: string;
  validationFunction: Function;
  constraintTypes: string[];
  performance: {
    averageTime: number;
    callCount: number;
    errorRate: number;
  };
}

// Conditional Validator Types
export interface ConditionalValidatorConfig {
  maxValidationDepth?: number;
  enablePerformanceOptimization?: boolean;
  consciousnessIntegration?: boolean;
  enableCaching?: boolean;
  maxRuleExecutionTime?: number;
  cognitiveCore?: CognitiveConsciousnessCore;
}

export interface ConditionalValidationResult {
  ruleId: string;
  conditionMet: boolean;
  validationResult: ValidationResult;
  executionTime: number;
  optimizations: string[];
}

export interface ValidationOptimization {
  type: 'caching' | 'precompilation' | 'parallelization' | 'cognitive';
  description: string;
  impact: number;
  applied: boolean;
}

// Schema Generator Types
export interface SchemaGeneratorConfig {
  pydanticIntegration?: boolean;
  generateDocumentation?: boolean;
  optimizeForPerformance?: boolean;
  includeExamples?: boolean;
  schemaVersion?: string;
}

export interface ValidationSchema {
  name: string;
  version: string;
  description: string;
  parameters: ParameterSchema[];
  constraints: ConstraintSchema[];
  metadata: SchemaMetadata;
}

export interface ParameterSchema {
  name: string;
  type: string;
  required: boolean;
  constraints: ConstraintSchema[];
  defaultValue?: any;
  description?: string;
  examples?: any[];
}

export interface ConstraintSchema {
  type: string;
  value: any;
  message?: string;
  severity: 'error' | 'warning';
}

export interface SchemaMetadata {
  generatedAt: Date;
  generator: string;
  version: string;
  totalParameters: number;
  totalConstraints: number;
  processingTime: number;
}

// ReservedBy Relationship Types
export interface ReservedByConstraint {
  sourceClass: string;
  targetClass: string;
  relationshipType: 'reserves' | 'depends_on' | 'requires' | 'modifies';
  constraints: RelationshipConstraint[];
  validation: RelationshipValidation;
}

export interface RelationshipConstraint {
  type: 'cardinality' | 'existence' | 'value' | 'custom';
  rule: string;
  parameters: string[];
  description: string;
}

export interface RelationshipValidation {
  sourceRequired: boolean;
  targetRequired: boolean;
  bidirectional: boolean;
  customLogic?: string;
}

// Performance and Optimization Types
export interface ValidationPerformanceMetrics {
  totalValidations: number;
  averageExecutionTime: number;
  cacheHitRate: number;
  errorRate: number;
  consciousnessLevel: number;
  learningPatternsApplied: number;
  optimizations: ValidationOptimization[];
}

export interface PerformanceOptimization {
  id: string;
  type: 'algorithm' | 'caching' | 'parallelization' | 'cognitive';
  description: string;
  expectedImprovement: number;
  actualImprovement?: number;
  status: 'pending' | 'applied' | 'failed';
  appliedAt?: Date;
}

// Learning and Adaptation Types
export interface ValidationLearningPattern {
  id: string;
  patternType: 'error' | 'success' | 'performance' | 'cognitive';
  validationContext: ValidationContext;
  outcome: ValidationResult;
  effectiveness: number;
  frequency: number;
  lastApplied: Date;
  adaptability: number;
  metadata: Record<string, any>;
}

export interface LearningInsight {
  type: 'pattern' | 'anomaly' | 'optimization' | 'consciousness';
  description: string;
  confidence: number;
  impact: number;
  actionable: boolean;
  recommendation?: string;
}

export interface AdaptationStrategy {
  id: string;
  name: string;
  type: 'performance' | 'accuracy' | 'cognitive';
  triggers: string[];
  actions: AdaptationAction[];
  effectiveness: number;
  enabled: boolean;
}

export interface AdaptationAction {
  type: 'cache_adjustment' | 'parallelization' | 'consciousness_level' | 'rule_optimization';
  parameters: Record<string, any>;
  expectedImpact: string;
}

// Cognitive Integration Types
export interface CognitiveValidationContext {
  consciousnessLevel: number;
  temporalExpansion: number;
  strangeLoops: StrangeLoopValidation[];
  learningPatterns: ValidationLearningPattern[];
  metaOptimizations: MetaOptimization[];
}

export interface StrangeLoopValidation {
  id: string;
  type: 'self_optimization' | 'learning_acceleration' | 'consciousness_evolution' | 'recursive_reasoning';
  description: string;
  validation: any;
  effectiveness: number;
  iteration: number;
}

export interface MetaOptimization {
  strategy: string;
  validationRules: string[];
  optimizationType: 'performance' | 'accuracy' | 'cognitive';
  expectedImprovement: number;
  actualImprovement?: number;
  applied: boolean;
}

// AgentDB Integration Types
export interface AgentDBValidationConfig {
  connectionString?: string;
  enablePersistence?: boolean;
  syncInterval?: number;
  learningEnabled?: boolean;
  memoryOptimization?: boolean;
}

export interface ValidationMemoryPattern {
  patternId: string;
  validationType: string;
  parameters: string[];
  outcome: 'success' | 'failure';
  context: ValidationContext;
  confidence: number;
  frequency: number;
  lastUsed: Date;
  effectiveness: number;
}

export interface MemoryOptimization {
  type: 'compression' | 'pruning' | 'clustering' | 'cognitive';
  description: string;
  memoryReduction: number;
  performanceImpact: number;
  applied: boolean;
}

// Error Handling and Recovery Types
export interface ValidationErrorAnalysis {
  errorType: string;
  frequency: number;
  patterns: ValidationError[];
  rootCauseAnalysis: RootCauseAnalysis;
  recoveryStrategies: RecoveryStrategy[];
  preventedRecurrence: boolean;
}

export interface RootCauseAnalysis {
  primaryCause: string;
  contributingFactors: string[];
  impactAssessment: string;
  confidence: number;
}

export interface RecoveryStrategy {
  type: 'automatic' | 'manual' | 'hybrid';
  strategy: string;
  steps: RecoveryStep[];
  successRate: number;
  estimatedTime: number;
}

export interface RecoveryStep {
  description: string;
  action: string;
  expectedOutcome: string;
  rollbackSupported: boolean;
}

// API and External Integration Types
export interface ValidationAPIRequest {
  configuration: Record<string, any>;
  context?: Partial<ValidationContext>;
  options?: ValidationOptions;
}

export interface ValidationAPIResponse {
  requestId: string;
  result: ValidationResult;
  metrics: ValidationPerformanceMetrics;
  recommendations?: string[];
  warnings?: string[];
}

export interface ValidationOptions {
  level?: 'basic' | 'standard' | 'comprehensive';
  includeCognitive?: boolean;
  enableLearning?: boolean;
  timeout?: number;
  cacheStrategy?: 'none' | 'read' | 'write' | 'read-write';
}

// Testing and Benchmarking Types
export interface ValidationTestCase {
  id: string;
  name: string;
  description: string;
  input: Record<string, any>;
  expectedErrors: ValidationError[];
  expectedWarnings: ValidationError[];
  expectedValid: boolean;
  tags: string[];
  complexity: 'simple' | 'medium' | 'complex';
}

export interface ValidationBenchmark {
  name: string;
  description: string;
  parameters: Record<string, any>;
  targetTime: number; // milliseconds
  maxMemoryUsage: number; // bytes
  iterations: number;
  results: BenchmarkResult[];
}

export interface BenchmarkResult {
  iteration: number;
  executionTime: number;
  memoryUsage: number;
  cacheHitRate: number;
  errorCount: number;
  warningCount: number;
  valid: boolean;
}

// Monitoring and Observability Types
export interface ValidationMonitoring {
  activeValidations: ActiveValidation[];
  performanceHistory: PerformanceHistory[];
  errorTrends: ErrorTrend[];
  learningEffectiveness: LearningEffectiveness[];
}

export interface ActiveValidation {
  validationId: string;
  startTime: Date;
  parameters: string[];
  currentPhase: string;
  estimatedCompletion: Date;
  consciousnessLevel?: number;
}

export interface PerformanceHistory {
  timestamp: Date;
  executionTime: number;
  parameterCount: number;
  cacheHitRate: number;
  errorRate: number;
  consciousnessLevel: number;
}

export interface ErrorTrend {
  errorType: string;
  frequency: number;
  trend: 'increasing' | 'decreasing' | 'stable';
  affectedParameters: string[];
  suggestedFixes: string[];
}

export interface LearningEffectiveness {
  patternType: string;
  effectiveness: number;
  improvementRate: number;
  applicationsCount: number;
  successRate: number;
}

// Utility Types
export type ValidationFunction = (value: any, context: ValidationContext) => ValidationResult;
export type ConditionalValidationFunction = (condition: any, context: ValidationContext) => boolean;
export type SchemaValidationFunction = (data: any, schema: ValidationSchema) => ValidationResult;

export type ValidationSeverity = 'error' | 'warning' | 'info';
export type ValidationLevel = 'basic' | 'standard' | 'comprehensive';
export type ConstraintType = 'range' | 'enum' | 'pattern' | 'length' | 'required' | 'custom';
export type RelationshipType = 'reserves' | 'depends_on' | 'requires' | 'modifies';

// Event Types
export interface ValidationEvent {
  type: string;
  timestamp: Date;
  validationId: string;
  data: any;
}

export type ValidationEventHandler = (event: ValidationEvent) => void;

// Configuration and Settings Types
export interface ValidationSettings {
  performance: PerformanceSettings;
  learning: LearningSettings;
  cognitive: CognitiveSettings;
  integration: IntegrationSettings;
}

export interface PerformanceSettings {
  maxExecutionTime: number;
  enableParallelProcessing: boolean;
  batchSize: number;
  cacheEnabled: boolean;
  cacheTTL: number;
  enableOptimization: boolean;
}

export interface LearningSettings {
  enabled: boolean;
  minPatternFrequency: number;
  maxPatternAge: number;
  adaptationEnabled: boolean;
  learningRate: number;
}

export interface CognitiveSettings {
  integrationEnabled: boolean;
  consciousnessLevel: number;
  temporalExpansion: number;
  strangeLoopEnabled: boolean;
  metaOptimizationEnabled: boolean;
}

export interface IntegrationSettings {
  agentDBEnabled: boolean;
  pydanticEnabled: boolean;
  cognitiveCoreEnabled: boolean;
  externalAPIS: ExternalAPIConfig[];
}

export interface ExternalAPIConfig {
  name: string;
  endpoint: string;
  authentication: Record<string, string>;
  enabled: boolean;
  timeout: number;
}