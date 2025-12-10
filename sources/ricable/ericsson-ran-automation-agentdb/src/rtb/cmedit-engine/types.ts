/**
 * Core cmedit Command Generation Engine Type Definitions
 *
 * Provides intelligent command generation for Ericsson RAN ENM CLI integration
 * with cognitive optimization and RAN expertise patterns.
 */

export interface CmeditCommand {
  /** Command type */
  type: CmeditCommandType;
  /** Target FDN path */
  target: string;
  /** Parameters to set */
  parameters?: Record<string, any>;
  /** Command options */
  options?: CmeditCommandOptions;
  /** Generated command string */
  command: string;
  /** Context for command generation */
  context: CommandContext;
  /** Validation result */
  validation?: CommandValidation;
}

export type CmeditCommandType =
  | 'get'
  | 'set'
  | 'create'
  | 'delete'
  | 'mon'
  | 'unmon'
  | 'sync'
  | 'preview';

export interface CmeditCommandOptions {
  /** Preview mode - don't execute */
  preview?: boolean;
  /** Force execution */
  force?: boolean;
  /** Table format output */
  table?: boolean;
  /** Detailed output */
  detailed?: boolean;
  /** Collection name for batch operations */
  collection?: string;
  /** Scope filter for operations */
  scopeFilter?: string;
  /** Attribute filter */
  attributes?: string[];
  /** Batch size for operations */
  batchSize?: number;
  /** Timeout for operations */
  timeout?: number;
  /** Dry run mode */
  dryRun?: boolean;
  /** Recursive operations */
  recursive?: boolean;
}

export interface CommandContext {
  /** Source template */
  sourceTemplate?: string;
  /** MO classes involved */
  moClasses: string[];
  /** Operation purpose */
  purpose: OperationPurpose;
  /** Network context */
  networkContext: NetworkContext;
  /** Cognitive optimization level */
  cognitiveLevel: CognitiveLevel;
  /** Ericsson expertise applied */
  expertisePatterns: string[];
  /** Timestamp of generation */
  generatedAt: Date;
  /** Priority level */
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export type OperationPurpose =
  | 'cell_optimization'
  | 'mobility_management'
  | 'capacity_expansion'
  | 'feature_activation'
  | 'performance_monitoring'
  | 'fault_resolution'
  | 'configuration_backup'
  | 'network_deployment';

export interface NetworkContext {
  /** Network technology */
  technology: '4G' | '5G' | '4G5G' | 'dual_mode';
  /** Network environment */
  environment: 'urban_dense' | 'urban_medium' | 'suburban' | 'rural';
  /** Vendor information */
  vendor: VendorInfo;
  /** Network topology */
  topology: NetworkTopology;
  /** Traffic patterns */
  trafficProfile?: TrafficProfile;
}

export interface VendorInfo {
  /** Primary vendor */
  primary: 'ericsson' | 'huawei' | 'nokia' | 'samsung' | 'zte';
  /** Multi-vendor setup */
  multiVendor: boolean;
  /** Secondary vendors */
  secondary?: string[];
  /** Compatibility mode */
  compatibilityMode: boolean;
}

export interface NetworkTopology {
  /** Cell count */
  cellCount: number;
  /** Site count */
  siteCount: number;
  /** Frequency bands */
  frequencyBands: string[];
  /** Carrier aggregation */
  carrierAggregation: boolean;
  /** Network sharing */
  networkSharing: boolean;
}

export interface TrafficProfile {
  /** Traffic density */
  density: 'low' | 'medium' | 'high' | 'very_high';
  /** Mobility patterns */
  mobility: 'low' | 'medium' | 'high';
  /** Service types */
  services: ('voice' | 'data' | 'video' | 'iot' | 'emergency')[];
  /** Peak hours */
  peakHours: string[];
}

export type CognitiveLevel =
  | 'basic'        // Standard command generation
  | 'enhanced'     // With RAN expertise patterns
  | 'cognitive'    // With temporal reasoning
  | 'autonomous'   // Self-optimizing with learning
  | 'conscious';   // Full cognitive consciousness

export interface FDNPath {
  /** Complete FDN path */
  path: string;
  /** Path components */
  components: FDNComponent[];
  /** MO class hierarchy */
  moHierarchy: string[];
  /** Path validation result */
  validation: FDNValidation;
  /** Alternative paths */
  alternatives: string[];
  /** Navigation complexity */
  complexity: PathComplexity;
}

export interface FDNComponent {
  /** Component name */
  name: string;
  /** MO class */
  moClass: string;
  /** Component value */
  value: string;
  /** Component type */
  type: 'class' | 'attribute' | 'index' | 'wildcard';
  /** Optional component */
  optional: boolean;
  /** Cardinality */
  cardinality: CardinalityInfo;
}

export interface CardinalityInfo {
  /** Minimum occurrences */
  minimum: number;
  /** Maximum occurrences */
  maximum: number;
  /** Current count */
  current: number;
  /** Cardinality type */
  type: 'single' | 'multiple' | 'optional';
}

export interface FDNValidation {
  /** Path is valid */
  isValid: boolean;
  /** Validation errors */
  errors: ValidationError[];
  /** Validation warnings */
  warnings: ValidationWarning[];
  /** Compliance level */
  complianceLevel: ComplianceLevel;
  /** LDN pattern match */
  ldnPatternMatch?: string;
}

export interface ValidationError {
  /** Error message */
  message: string;
  /** Component with error */
  component: string;
  /** Error type */
  type: 'syntax' | 'semantic' | 'cardinality' | 'dependency' | 'ldn';
  /** Severity */
  severity: 'error' | 'warning' | 'info';
}

export interface ValidationWarning extends ValidationError {
  severity: 'warning';
}

export type ComplianceLevel =
  | 'full'      // Fully compliant with standards
  | 'partial'   // Mostly compliant with minor issues
  | 'minimal'   // Basic compliance
  | 'custom';   // Custom/vendor-specific

export interface PathComplexity {
  /** Complexity score (0-100) */
  score: number;
  /** Navigation depth */
  depth: number;
  /** Number of components */
  componentCount: number;
  /** Wildcard usage */
  wildcardCount: number;
  /** Estimated execution time */
  estimatedTime: number;
  /** Processing difficulty */
  difficulty: 'trivial' | 'simple' | 'moderate' | 'complex' | 'very_complex';
}

export interface CommandValidation {
  /** Command is valid */
  isValid: boolean;
  /** Syntax validation */
  syntax: SyntaxValidation;
  /** Semantic validation */
  semantic: SemanticValidation;
  /** Parameter validation */
  parameters: ParameterValidation;
  /** Dependency validation */
  dependencies: DependencyValidation;
  /** Overall score */
  score: number;
  /** Recommendations */
  recommendations: string[];
}

export interface SyntaxValidation {
  /** Syntax is correct */
  isCorrect: boolean;
  /** Syntax errors */
  errors: string[];
  /** Command structure */
  structure: CommandStructure;
}

export interface CommandStructure {
  /** Command parts */
  parts: string[];
  /** Argument count */
  argCount: number;
  /** Expected pattern */
  expectedPattern: string;
  /** Actual pattern */
  actualPattern: string;
}

export interface SemanticValidation {
  /** Semantics are correct */
  isCorrect: boolean;
  /** MO class validation */
  moClasses: MOClassValidation[];
  /** Operation validation */
  operation: OperationValidation;
}

export interface MOClassValidation {
  /** MO class name */
  className: string;
  /** Class exists */
  exists: boolean;
  /** Class is supported */
  supported: boolean;
  /** Required attributes */
  requiredAttributes: string[];
  /** Optional attributes */
  optionalAttributes: string[];
}

export interface OperationValidation {
  /** Operation is supported */
  isSupported: boolean;
  /** Operation constraints */
  constraints: OperationConstraint[];
  /** Required permissions */
  permissions: string[];
}

export interface OperationConstraint {
  /** Constraint type */
  type: 'cardinality' | 'dependency' | 'state' | 'permission';
  /** Constraint description */
  description: string;
  /** Constraint value */
  value: any;
  /** Is satisfied */
  satisfied: boolean;
}

export interface ParameterValidation {
  /** Parameters are valid */
  isValid: boolean;
  /** Parameter errors */
  errors: ParameterError[];
  /** Parameter warnings */
  warnings: ParameterWarning[];
  /** Value conversions */
  conversions: ParameterConversion[];
}

export interface ParameterError {
  /** Parameter name */
  parameter: string;
  /** Error message */
  message: string;
  /** Expected type */
  expectedType: string;
  /** Actual value */
  actualValue: any;
  /** Constraint violated */
  constraint?: string;
}

export interface ParameterWarning {
  /** Parameter name */
  parameter: string;
  /** Warning message */
  message: string;
  /** Recommendation */
  recommendation?: string;
}

export interface ParameterConversion {
  /** Source parameter */
  source: string;
  /** Target type */
  targetType: string;
  /** Conversion function */
  conversion: string;
  /** Original value */
  originalValue: any;
  /** Converted value */
  convertedValue: any;
}

export interface DependencyValidation {
  /** Dependencies are satisfied */
  isSatisfied: boolean;
  /** Unresolved dependencies */
  unresolved: Dependency[];
  /** Circular dependencies */
  circular: CircularDependency[];
  /** Dependency graph */
  graph: DependencyGraph;
}

export interface Dependency {
  /** Source MO */
  source: string;
  /** Target MO */
  target: string;
  /** Dependency type */
  type: DependencyType;
  /** Description */
  description: string;
  /** Is resolved */
  resolved: boolean;
}

export type DependencyType =
  | 'requires'     // Source requires target
  | 'modifies'     // Source modifies target
  | 'reserves'     // Source reserves target
  | 'conflicts'    // Source conflicts with target
  | 'depends_on';  // Source depends on target

export interface CircularDependency {
  /** Cycle path */
  path: string[];
  /** Cycle length */
  length: number;
  /** Severity */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Resolution suggestion */
  resolution: string;
}

export interface DependencyGraph {
  /** Graph nodes */
  nodes: DependencyNode[];
  /** Graph edges */
  edges: DependencyEdge[];
  /** Connected components */
  components: string[][];
  /** Has cycles */
  hasCycles: boolean;
}

export interface DependencyNode {
  /** Node identifier */
  id: string;
  /** MO class */
  moClass: string;
  /** Node type */
  type: 'source' | 'target' | 'both';
  /** Dependencies count */
  dependencyCount: number;
}

export interface DependencyEdge {
  /** Source node */
  source: string;
  /** Target node */
  target: string;
  /** Edge type */
  type: DependencyType;
  /** Edge weight */
  weight: number;
}

// Ericsson RAN Expertise Patterns

export interface EricssonExpertisePattern {
  /** Pattern identifier */
  id: string;
  /** Pattern name */
  name: string;
  /** Pattern category */
  category: ExpertiseCategory;
  /** Pattern description */
  description: string;
  /** Pattern conditions */
  conditions: PatternCondition[];
  /** Pattern actions */
  actions: PatternAction[];
  /** Success metrics */
  successMetrics: SuccessMetric[];
  /** Cognitive enhancement */
  cognitiveEnhancement?: CognitiveEnhancement;
}

export type ExpertiseCategory =
  | 'cell_optimization'
  | 'mobility_management'
  | 'capacity_management'
  | 'energy_efficiency'
  | 'quality_assurance'
  | 'fault_management'
  | 'performance_monitoring'
  | 'feature_activation';

export interface PatternCondition {
  /** Condition type */
  type: 'network_type' | 'traffic_load' | 'environment' | 'vendor' | 'technology';
  /** Condition operator */
  operator: 'equals' | 'contains' | 'greater_than' | 'less_than' | 'between';
  /** Condition value */
  value: any;
  /** Required for pattern */
  required: boolean;
}

export interface PatternAction {
  /** Action type */
  type: 'parameter_adjustment' | 'feature_activation' | 'neighbor_addition' | 'capacity_scaling';
  /** Action target */
  target: string;
  /** Action parameters */
  parameters: Record<string, any>;
  /** Action priority */
  priority: number;
  /** Expected outcome */
  expectedOutcome: string;
}

export interface SuccessMetric {
  /** Metric name */
  name: string;
  /** Target value */
  target: number;
  /** Measurement unit */
  unit: string;
  /** Current value */
  current?: number;
  /** Achievement status */
  achieved?: boolean;
}

export interface CognitiveEnhancement {
  /** Temporal reasoning level */
  temporalLevel: number;
  /** Learning capability */
  learningEnabled: boolean;
  /** Adaptation strategy */
  adaptationStrategy: 'conservative' | 'balanced' | 'aggressive';
  /** Memory integration */
  memoryIntegration: boolean;
}

// Command Generation Results

export interface CommandGenerationResult {
  /** Generated commands */
  commands: CmeditCommand[];
  /** Generation statistics */
  stats: GenerationStats;
  /** Validation results */
  validation: CommandValidation;
  /** Optimization results */
  optimization: OptimizationResult;
  /** Expertise patterns applied */
  patternsApplied: EricssonExpertisePattern[];
  /** Execution plan */
  executionPlan: ExecutionPlan;
}

export interface GenerationStats {
  /** Total commands generated */
  totalCommands: number;
  /** Commands by type */
  commandsByType: Record<CmeditCommandType, number>;
  /** Generation time */
  generationTime: number;
  /** Memory usage */
  memoryUsage: number;
  /** Cache hits */
  cacheHits: number;
  /** Template conversions */
  templateConversions: number;
  /** FDN paths generated */
  fdnPathsGenerated: number;
}

export interface OptimizationResult {
  /** Optimization applied */
  applied: boolean;
  /** Optimization score */
  score: number;
  /** Optimizations made */
  optimizations: Optimization[];
  /** Performance improvement */
  performanceImprovement: PerformanceImprovement;
  /** Cognitive insights */
  cognitiveInsights: CognitiveInsight[];
}

export interface Optimization {
  /** Optimization type */
  type: 'path_optimization' | 'parameter_optimization' | 'batch_optimization' | 'dependency_optimization';
  /** Description */
  description: string;
  /** Impact score */
  impact: number;
  /** Applied successfully */
  applied: boolean;
}

export interface PerformanceImprovement {
  /** Execution time improvement */
  executionTime: number;
  /** Memory usage improvement */
  memoryUsage: number;
  /** Network efficiency improvement */
  networkEfficiency: number;
  /** Command success rate improvement */
  successRate: number;
}

export interface CognitiveInsight {
  /** Insight type */
  type: 'pattern_recognition' | 'anomaly_detection' | 'optimization_opportunity' | 'risk_assessment';
  /** Insight message */
  message: string;
  /** Confidence level */
  confidence: number;
  /** Recommended action */
  recommendedAction: string;
  /** Supporting data */
  supportingData: Record<string, any>;
}

export interface ExecutionPlan {
  /** Plan phases */
  phases: ExecutionPhase[];
  /** Total estimated time */
  estimatedTime: number;
  /** Risk assessment */
  riskAssessment: RiskAssessment;
  /** Rollback plan */
  rollbackPlan: RollbackPlan;
}

export interface ExecutionPhase {
  /** Phase identifier */
  id: string;
  /** Phase name */
  name: string;
  /** Commands in phase */
  commands: string[];
  /** Dependencies */
  dependencies: string[];
  /** Estimated time */
  estimatedTime: number;
  /** Parallel execution allowed */
  parallelAllowed: boolean;
}

export interface RiskAssessment {
  /** Overall risk level */
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  /** Risk factors */
  riskFactors: RiskFactor[];
  /** Mitigation strategies */
  mitigationStrategies: string[];
  /** Pre-execution checks */
  preChecks: string[];
}

export interface RiskFactor {
  /** Factor name */
  name: string;
  /** Risk score */
  score: number;
  /** Impact description */
  impact: string;
  /** Mitigation */
  mitigation: string;
}

export interface RollbackPlan {
  /** Rollback possible */
  possible: boolean;
  /** Rollback commands */
  commands: string[];
  /** Rollback time */
  estimatedTime: number;
  /** Data backup required */
  backupRequired: boolean;
}