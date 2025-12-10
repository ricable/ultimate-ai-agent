/**
 * Hierarchical Template System - Phase 2 Architecture
 *
 * Implements priority-based template inheritance engine for RTB configuration system.
 * Provides foundation for template variants, frequency relations, and conflict resolution.
 *
 * Priority Levels (lower number = higher priority):
 * - agentdb: 0     - RAN-AUTOMATION-AGENTDB overrides (highest)
 * - base: 9        - Non-variant parameters (foundation)
 * - urban: 20      - Urban/UAL high capacity variants
 * - mobility: 30   - High mobility (fast train/motorways)
 * - sleep: 40      - Sleep mode night optimization
 * - frequency_4g4g: 50  - 4G4G frequency relations
 * - frequency_4g5g: 60  - 4G5G frequency relations (EN-DC)
 * - frequency_5g5g: 70  - 5G5G frequency relations (NR-NR DC)
 * - frequency_5g4g: 80  - 5G4G frequency relations (fallback)
 */

import {
  RTBTemplate,
  TemplateMeta,
  CustomFunction,
  ConditionOperator,
  EvaluationOperator,
  RTBParameter,
  ConstraintSpec,
  MOHierarchy,
  ReservedByHierarchy
} from '../types/rtb-types';

// ============================================================================
// CORE INTERFACES
// ============================================================================

/**
 * Template priority levels for inheritance resolution
 */
export enum TemplatePriority {
  AGENTDB = 0,         // Highest priority - runtime overrides
  BASE = 9,            // Foundation parameters
  URBAN = 20,          // Urban/UAL high capacity
  MOBILITY = 30,       // High mobility scenarios
  SLEEP = 40,          // Sleep mode optimization
  FREQUENCY_4G4G = 50, // 4G4G frequency relations
  FREQUENCY_4G5G = 60, // 4G5G EN-DC relations
  FREQUENCY_5G5G = 70, // 5G5G NR-NR DC relations
  FREQUENCY_5G4G = 80  // 5G4G fallback relations
}

/**
 * Enhanced template with priority information
 */
export interface PriorityTemplate extends RTBTemplate {
  meta: EnhancedTemplateMeta;
  priority: TemplatePriority;
  inheritanceChain?: string[];
  conflictResolution?: ConflictResolutionStrategy;
  validationRules?: TemplateValidationRule[];
}

/**
 * Enhanced template metadata with priority and inheritance information
 */
export interface EnhancedTemplateMeta extends TemplateMeta {
  priority: TemplatePriority;
  variantType?: TemplateVariantType;
  frequencyBand?: FrequencyBand;
  inherits_from?: string | string[];
  conflictStrategy?: ConflictResolutionStrategy;
  generationContext?: TemplateGenerationContext;
  performanceHints?: TemplatePerformanceHints;
}

/**
 * Template variant types for specialized configurations
 */
export enum TemplateVariantType {
  BASE = 'base',
  URBAN = 'urban',
  UAL_HIGH_CAPACITY = 'ual_high_capacity',
  HIGH_MOBILITY = 'high_mobility',
  SLEEP_MODE = 'sleep_mode',
  COASTAL = 'coastal',
  RURAL = 'rural',
  DENSE_URBAN = 'dense_urban',
  SUBURBAN = 'suburban'
}

/**
 * Frequency band configurations
 */
export enum FrequencyBand {
  LTE_800 = 'lte_800',
  LTE_1800 = 'lte_1800',
  LTE_2100 = 'lte_2100',
  LTE_2600 = 'lte_2600',
  NR_700 = 'nr_700',
  NR_3500 = 'nr_3500',
  NR_26000 = 'nr_26000',
  NR_28000 = 'nr_28000'
}

/**
 * Template generation context information
 */
export interface TemplateGenerationContext {
  source: 'xml_extraction' | 'manual_creation' | 'ai_generation' | 'variant_derivation';
  sourceFile?: string;
  generationTimestamp: Date;
  parentTemplates?: string[];
  derivationRules?: string[];
  confidence?: number;
}

/**
 * Performance optimization hints for template processing
 */
export interface TemplatePerformanceHints {
  expectedParameterCount: number;
  expectedComplexity: 'low' | 'medium' | 'high' | 'extreme';
  memoryOptimization: boolean;
  streamingCapable: boolean;
  cacheable: boolean;
  batchProcessing: boolean;
}

/**
 * Template validation rules
 */
export interface TemplateValidationRule {
  ruleId: string;
  type: 'parameter_constraint' | 'dependency_check' | 'consistency_check' | 'performance_check';
  condition: string;
  action: 'error' | 'warning' | 'info';
  message: string;
  enabled: boolean;
}

/**
 * Template inheritance chain information
 */
export interface TemplateInheritanceChain {
  templateId: string;
  chain: TemplateChainLink[];
  resolvedTemplate: PriorityTemplate;
  conflicts: TemplateConflict[];
  warnings: TemplateWarning[];
  processingTime: number;
}

/**
 * Individual link in inheritance chain
 */
export interface TemplateChainLink {
  templateId: string;
  priority: TemplatePriority;
  appliedAt: Date;
  appliedParameters: string[];
  overriddenParameters: string[];
  conflicts: string[];
}

/**
 * Template conflict information
 */
export interface TemplateConflict {
  parameterPath: string;
  conflictingTemplates: Array<{
    templateId: string;
    priority: TemplatePriority;
    value: any;
  }>;
  resolutionStrategy: ConflictResolutionStrategy;
  resolvedValue?: any;
  resolutionReason?: string;
}

/**
 * Template processing warnings
 */
export interface TemplateWarning {
  warningId: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  parameterPath?: string;
  templateId?: string;
  suggestion?: string;
}

/**
 * Conflict resolution strategies
 */
export enum ConflictResolutionStrategy {
  HIGHEST_PRIORITY_WINS = 'highest_priority_wins',
  LOWEST_PRIORITY_WINS = 'lowest_priority_wins',
  MERGE_WITH_WARNING = 'merge_with_warning',
  CUSTOM_FUNCTION = 'custom_function',
  FAIL_ON_CONFLICT = 'fail_on_conflict',
  CONFLICT_LOGGING = 'conflict_logging'
}

/**
 * Template variant generation configuration
 */
export interface VariantGenerationConfig {
  baseTemplateId: string;
  variantType: TemplateVariantType;
  targetScenarios: string[];
  parameterOverrides: Record<string, any>;
  conditionalLogic: Record<string, ConditionOperator>;
  customFunctions: CustomFunction[];
  validationRules: TemplateValidationRule[];
}

/**
 * Frequency relation configuration
 */
export interface FrequencyRelationConfig {
  sourceBand: FrequencyBand;
  targetBand: FrequencyBand;
  relationType: '4G4G' | '4G5G' | '5G5G' | '5G4G';
  parameters: Record<string, any>;
  neighborRelations: NeighborRelationConfig[];
  handoverParameters: HandoverParameterConfig[];
  capacityParameters: CapacityParameterConfig;
}

/**
 * Neighbor relation configuration for frequency templates
 */
export interface NeighborRelationConfig {
  relationId: string;
  targetCellPattern: string;
  parameters: Record<string, any>;
  priority: number;
  conditions?: ConditionOperator[];
}

/**
 * Handover parameter configuration
 */
export interface HandoverParameterConfig {
  hysteresis: number;
  a3Offset: number;
  a5Offset1: number;
  a5Offset2: number;
  timeToTrigger: number;
  parameters: Record<string, any>;
}

/**
 * Capacity parameter configuration
 */
export interface CapacityParameterConfig {
  loadBalancing: boolean;
  cellIndividualOffset: number;
  qOffset: Record<string, number>;
  parameters: Record<string, any>;
}

// ============================================================================
// MAIN ENGINE INTERFACES
// ============================================================================

/**
 * Core priority template engine interface
 */
export interface IPriorityTemplateEngine {
  /**
   * Register a template with the engine
   */
  registerTemplate(template: PriorityTemplate): Promise<void>;

  /**
   * Resolve template inheritance chain
   */
  resolveInheritance(templateId: string): Promise<TemplateInheritanceChain>;

  /**
   * Merge multiple templates with conflict resolution
   */
  mergeTemplates(templateIds: string[], strategy?: ConflictResolutionStrategy): Promise<PriorityTemplate>;

  /**
   * Generate variant from base template
   */
  generateVariant(config: VariantGenerationConfig): Promise<PriorityTemplate>;

  /**
   * Validate template against constraints
   */
  validateTemplate(template: PriorityTemplate): Promise<TemplateValidationResult>;

  /**
   * Get template by ID
   */
  getTemplate(templateId: string): Promise<PriorityTemplate | null>;

  /**
   * List templates by priority or variant type
   */
  listTemplates(filter?: TemplateFilter): Promise<PriorityTemplate[]>;

  /**
   * Delete template
   */
  deleteTemplate(templateId: string): Promise<boolean>;
}

/**
 * Template variant generator interface
 */
export interface ITemplateVariantGenerator {
  /**
   * Generate urban variant template
   */
  generateUrbanVariant(baseTemplate: PriorityTemplate, config?: UrbanConfig): Promise<PriorityTemplate>;

  /**
   * Generate high mobility variant template
   */
  generateMobilityVariant(baseTemplate: PriorityTemplate, config?: MobilityConfig): Promise<PriorityTemplate>;

  /**
   * Generate sleep mode variant template
   */
  generateSleepVariant(baseTemplate: PriorityTemplate, config?: SleepConfig): Promise<PriorityTemplate>;

  /**
   * Generate custom variant
   */
  generateCustomVariant(baseTemplate: PriorityTemplate, config: VariantGenerationConfig): Promise<PriorityTemplate>;
}

/**
 * Frequency relation manager interface
 */
export interface IFrequencyRelationManager {
  /**
   * Generate 4G4G frequency relation template
   */
  generate4G4GRelation(config: FrequencyRelationConfig): Promise<PriorityTemplate>;

  /**
   * Generate 4G5G EN-DC relation template
   */
  generate4G5GRelation(config: FrequencyRelationConfig): Promise<PriorityTemplate>;

  /**
   * Generate 5G5G NR-NR DC relation template
   */
  generate5G5GRelation(config: FrequencyRelationConfig): Promise<PriorityTemplate>;

  /**
   * Generate 5G4G fallback relation template
   */
  generate5G4GRelation(config: FrequencyRelationConfig): Promise<PriorityTemplate>;

  /**
   * Get frequency compatibility matrix
   */
  getCompatibilityMatrix(): Promise<FrequencyCompatibilityMatrix>;
}

/**
 * Base template auto-generator interface
 */
export interface IBaseTemplateAutoGenerator {
  /**
   * Generate base template from XML constraints
   */
  generateFromXML(xmlFilePath: string, moHierarchy: MOHierarchy): Promise<PriorityTemplate>;

  /**
   * Extract parameters from XML schema
   */
  extractParameters(xmlFilePath: string): Promise<RTBParameter[]>;

  /**
   * Generate validation rules from constraints
   */
  generateValidationRules(parameters: RTBParameter[]): Promise<TemplateValidationRule[]>;

  /**
   * Create template structure from MO hierarchy
   */
  generateTemplateStructure(moHierarchy: MOHierarchy): Promise<Record<string, any>>;
}

/**
 * Template merger interface
 */
export interface ITemplateMerger {
  /**
   * Merge templates with priority resolution
   */
  merge(templates: PriorityTemplate[], strategy?: ConflictResolutionStrategy): Promise<PriorityTemplate>;

  /**
   * Resolve parameter conflicts
   */
  resolveConflicts(conflicts: TemplateConflict[]): Promise<TemplateConflict[]>;

  /**
   * Merge custom functions
   */
  mergeCustomFunctions(functions: CustomFunction[][]): Promise<CustomFunction[]>;

  /**
   * Merge conditional logic
   */
  mergeConditions(conditions: Record<string, ConditionOperator>[]): Promise<Record<string, ConditionOperator>>;
}

/**
 * Template conflict resolver interface
 */
export interface ITemplateConflictResolver {
  /**
   * Detect conflicts between templates
   */
  detectConflicts(templates: PriorityTemplate[]): Promise<TemplateConflict[]>;

  /**
   * Resolve conflicts using specified strategy
   */
  resolveConflict(conflict: TemplateConflict, strategy: ConflictResolutionStrategy): Promise<TemplateConflict>;

  /**
   * Get conflict resolution suggestions
   */
  getResolutionSuggestions(conflict: TemplateConflict): Promise<ConflictResolutionSuggestion[]>;

  /**
   * Log conflict for analysis
   */
  logConflict(conflict: TemplateConflict, resolution: TemplateConflict): Promise<void>;
}

// ============================================================================
// SUPPORTING INTERFACES
// ============================================================================

/**
 * Template filter options
 */
export interface TemplateFilter {
  priority?: TemplatePriority;
  variantType?: TemplateVariantType;
  frequencyBand?: FrequencyBand;
  tags?: string[];
  author?: string;
  dateRange?: { start: Date; end: Date };
}

/**
 * Template validation result
 */
export interface TemplateValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  parameterCount: number;
  constraintViolations: ConstraintViolation[];
  performanceMetrics: ValidationPerformanceMetrics;
}

/**
 * Validation error
 */
export interface ValidationError {
  parameterPath: string;
  errorType: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  constraint?: ConstraintSpec;
}

/**
 * Validation warning
 */
export interface ValidationWarning {
  parameterPath: string;
  warningType: string;
  message: string;
  suggestion?: string;
}

/**
 * Constraint violation
 */
export interface ConstraintViolation {
  parameterPath: string;
  constraint: ConstraintSpec;
  actualValue: any;
  expectedValue: any;
  severity: 'error' | 'warning';
}

/**
 * Validation performance metrics
 */
export interface ValidationPerformanceMetrics {
  validationTime: number;
  memoryUsage: number;
  parameterCount: number;
  constraintCount: number;
}

/**
 * Urban configuration for variant generation
 */
export interface UrbanConfig {
  cellDensity: 'low' | 'medium' | 'high' | 'ultra_high';
  trafficProfile: 'residential' | 'business' | 'mixed' | 'stadium';
  capacityOptimization: boolean;
  interferenceManagement: boolean;
  parameters?: Record<string, any>;
}

/**
 * Mobility configuration for variant generation
 */
export interface MobilityConfig {
  mobilityType: 'pedestrian' | 'vehicular' | 'railway' | 'highway';
  speedProfile: 'low' | 'medium' | 'high' | 'very_high';
  handoverOptimization: boolean;
  signalStabilityPriority: boolean;
  parameters?: Record<string, any>;
}

/**
 * Sleep configuration for variant generation
 */
export interface SleepConfig {
  energySavingLevel: 'conservative' | 'moderate' | 'aggressive' | 'maximum';
  trafficThreshold: number;
  activationTime: string;
  deactivationTime: string;
  servicePreservation: boolean;
  parameters?: Record<string, any>;
}

/**
 * Frequency compatibility matrix
 */
export interface FrequencyCompatibilityMatrix {
  matrix: Record<string, Record<string, boolean>>;
  recommendations: FrequencyRecommendation[];
  constraints: FrequencyConstraint[];
}

/**
 * Frequency recommendation
 */
export interface FrequencyRecommendation {
  sourceBand: FrequencyBand;
  targetBand: FrequencyBand;
  recommendation: 'recommended' | 'compatible' | 'not_recommended' | 'incompatible';
  reason: string;
  useCase: string[];
}

/**
 * Frequency constraint
 */
export interface FrequencyConstraint {
  bands: [FrequencyBand, FrequencyBand];
  constraint: string;
  parameters: Record<string, any>;
  description: string;
}

/**
 * Conflict resolution suggestion
 */
export interface ConflictResolutionSuggestion {
  strategy: ConflictResolutionStrategy;
  confidence: number;
  reasoning: string;
  expectedOutcome: string;
  risks: string[];
}

/**
 * Template processing metrics
 */
export interface TemplateProcessingMetrics {
  templateId: string;
  processingTime: number;
  memoryUsage: number;
  parameterCount: number;
  conflictCount: number;
  warningCount: number;
  cacheHits: number;
  cacheMisses: number;
}

/**
 * Engine configuration options
 */
export interface HierarchicalTemplateEngineConfig {
  cachingEnabled: boolean;
  maxCacheSize: number;
  defaultConflictStrategy: ConflictResolutionStrategy;
  parallelProcessing: boolean;
  maxConcurrentOperations: number;
  validationStrictness: 'lenient' | 'strict' | 'very_strict';
  performanceMonitoring: boolean;
  detailedLogging: boolean;
}

// ============================================================================
// EVENT INTERFACES
// ============================================================================

/**
 * Template processing events
 */
export interface TemplateProcessingEvent {
  eventType: 'template_registered' | 'template_resolved' | 'template_merged' | 'template_validated' | 'conflict_resolved';
  templateId: string;
  timestamp: Date;
  data: any;
  processingTime?: number;
  error?: Error;
}

/**
 * Template event listener
 */
export interface ITemplateEventListener {
  onEvent(event: TemplateProcessingEvent): Promise<void>;
}

/**
 * Event bus for template processing events
 */
export interface ITemplateEventBus {
  subscribe(eventType: string, listener: ITemplateEventListener): void;
  unsubscribe(eventType: string, listener: ITemplateEventListener): void;
  publish(event: TemplateProcessingEvent): Promise<void>;
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/**
 * Base template system error
 */
export class TemplateSystemError extends Error {
  constructor(
    message: string,
    public code: string,
    public templateId?: string,
    public parameterPath?: string
  ) {
    super(message);
    this.name = 'TemplateSystemError';
  }
}

/**
 * Template inheritance error
 */
export class TemplateInheritanceError extends TemplateSystemError {
  constructor(
    message: string,
    public templateId: string,
    public inheritanceChain: string[],
    public conflictingTemplates: string[]
  ) {
    super(message, 'INHERITANCE_ERROR', templateId);
    this.name = 'TemplateInheritanceError';
  }
}

/**
 * Template validation error
 */
export class TemplateValidationError extends TemplateSystemError {
  constructor(
    message: string,
    templateId: string,
    public validationErrors: ValidationError[]
  ) {
    super(message, 'VALIDATION_ERROR', templateId);
    this.name = 'TemplateValidationError';
  }
}

/**
 * Template conflict error
 */
export class TemplateConflictError extends TemplateSystemError {
  constructor(
    message: string,
    public conflicts: TemplateConflict[]
  ) {
    super(message, 'CONFLICT_ERROR', conflicts[0]?.conflictingTemplates[0]?.templateId);
    this.name = 'TemplateConflictError';
  }
}