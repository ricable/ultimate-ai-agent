/**
 * Template Merger System Type Definitions
 */

import { RTBTemplate, TemplateMeta, CustomFunction } from '../../types/rtb-types';

// Re-export types needed for test files
export { RTBTemplate, TemplateMeta, CustomFunction };

// Define MetaConfig if it doesn't exist in the original file
export interface MetaConfig {
  version: string;
  author?: string[];
  description?: string;
  tags?: string[];
  priority?: number;
  environment?: string;
}

export type ConflictType = 'value' | 'type' | 'structure' | 'conditional' | 'function' | 'metadata';
export type ResolutionStrategyType = 'highest_priority' | 'merge' | 'conditional' | 'custom' | 'interactive';

export interface TemplateConflict {
  /** The parameter path that has a conflict */
  parameter: string;
  /** Templates that are causing the conflict */
  templates: string[];
  /** Priority levels of conflicting templates */
  priorities: number[];
  /** Type of conflict */
  conflictType: ConflictType;
  /** Current values from each template */
  values: any[];
  /** Resolution strategy to apply */
  resolution: ConflictResolution;
  /** Whether the conflict has been resolved */
  resolved: boolean;
  /** Index of the template whose value was used in resolution */
  resolvedTemplateIndex?: number;
  /** Additional context about the conflict */
  context?: ConflictContext;
  /** Severity level of the conflict */
  severity: 'low' | 'medium' | 'high' | 'critical';
  /** Whether user intervention is required */
  requiresManualIntervention: boolean;
}

export interface ConflictResolution {
  /** Strategy used for resolution */
  strategy: ResolutionStrategyType;
  /** Reasoning behind the resolution choice */
  reasoning: string;
  /** Custom resolver function name (if applicable) */
  customResolver?: string;
  /** The resolved value */
  value?: any;
  /** Whether the resolution was successful */
  resolved: boolean;
  /** Additional metadata about the resolution */
  metadata?: Record<string, any>;
}

export interface ConflictContext {
  /** Inheritance depth where conflict occurred */
  inheritanceDepth: number;
  /** Template inheritance path */
  inheritancePath: string[];
  /** Parameter type information */
  parameterType?: string;
  /** Parameter constraints */
  constraints?: any[];
  /** Similar conflicts that have occurred before */
  historicalConflicts?: number;
  /** Recommended resolution based on ML predictions */
  recommendedResolution?: ResolutionStrategyType;
}

export interface MergeResult {
  /** The merged template */
  template: RTBTemplate;
  /** All detected conflicts */
  conflicts: TemplateConflict[];
  /** Successfully resolved conflicts */
  resolvedConflicts: TemplateConflict[];
  /** Unresolved conflicts */
  unresolvedConflicts: TemplateConflict[];
  /** Statistics about the merge operation */
  mergeStats: MergeStats;
  /** Inheritance chain information */
  inheritanceChain: InheritanceChain;
  /** Validation result of the merged template */
  validationResult?: ValidationResult;
}

export interface MergeStats {
  /** Total number of templates merged */
  totalTemplates: number;
  /** Total number of conflicts detected */
  conflictsDetected: number;
  /** Total number of conflicts resolved */
  conflictsResolved: number;
  /** List of applied resolutions */
  resolutionsApplied: ResolutionApplied[];
  /** Total processing time in milliseconds */
  processingTime: number;
  /** Memory usage during merge */
  memoryUsage?: number;
  /** Cache hit rate */
  cacheHitRate?: number;
}

export interface ResolutionApplied {
  /** Parameter that was resolved */
  conflict: string;
  /** Strategy used */
  strategy: ResolutionStrategyType;
  /** Template that provided the winning value */
  template: string;
  /** Time taken to resolve */
  resolutionTime: number;
}

export interface InheritanceChain {
  /** Templates in inheritance order (parent to child) */
  templates: RTBTemplate[];
  /** Priority levels for each template */
  priorities: number[];
  /** Maximum inheritance depth */
  inheritanceDepth: number;
  /** Whether circular dependency was detected */
  hasCircularDependency: boolean;
  /** Path of circular dependency if detected */
  circularPath?: string[];
  /** Template inheritance tree structure */
  inheritanceTree?: InheritanceTree;
}

export interface InheritanceTree {
  /** Template identifier */
  id: string;
  /** Template metadata */
  meta: TemplateMeta;
  /** Parent templates */
  parents: InheritanceTree[];
  /** Child templates */
  children: InheritanceTree[];
  /** Priority level */
  priority: number;
  /** Depth in inheritance tree */
  depth: number;
}

export interface MergeContext {
  /** Templates being merged */
  templates: RTBTemplate[];
  /** Merge options */
  options: MergeOptions;
  /** Start time of merge operation */
  startTime: number;
  /** Statistics tracking */
  mergeStats: MergeStats;
  /** User interactions during merge */
  interactions?: MergeInteraction[];
}

export interface MergeOptions {
  /** Strategy for handling conflicts */
  conflictResolution: 'auto' | 'interactive' | 'strict';
  /** Preserve template metadata during merge */
  preserveMetadata: boolean;
  /** Validate merged template */
  validateResult: boolean;
  /** Enable deep merging for nested objects */
  deepMerge: boolean;
  /** Custom resolvers for specific conflicts */
  customResolvers: Record<string, (conflict: TemplateConflict) => any>;
  /** Performance optimization for large template sets */
  batchMode: boolean;
  /** Enable caching for repeated merges */
  enableCache: boolean;
  /** Maximum inheritance depth allowed */
  maxInheritanceDepth?: number;
  /** Whether to allow circular dependencies */
  allowCircularDependencies?: boolean;
  /** Logging level */
  logLevel?: 'debug' | 'info' | 'warn' | 'error';
}

export interface ValidationResult {
  /** Whether the merged template is valid */
  isValid: boolean;
  /** List of validation errors */
  errors: ValidationError[];
  /** List of validation warnings */
  warnings: ValidationWarning[];
  /** Validation statistics */
  stats: ValidationStats;
}

export interface ValidationError {
  /** Error message */
  message: string;
  /** Parameter path where error occurred */
  parameter: string;
  /** Error severity */
  severity: 'error' | 'warning' | 'info';
  /** Error type */
  type: 'schema' | 'constraint' | 'dependency' | 'type' | 'circular';
  /** Template source of error */
  source?: string;
  /** Suggested fix */
  suggestion?: string;
}

export interface ValidationWarning extends ValidationError {
  severity: 'warning';
}

export interface ValidationStats {
  /** Total parameters validated */
  totalParameters: number;
  /** Number of errors found */
  errorCount: number;
  /** Number of warnings found */
  warningCount: number;
  /** Validation time in milliseconds */
  validationTime: number;
}

export interface MergeInteraction {
  /** Type of interaction */
  type: 'conflict_resolution' | 'validation_error' | 'user_choice';
  /** Parameter involved */
  parameter: string;
  /** User decision */
  decision: any;
  /** Timestamp of interaction */
  timestamp: number;
  /** Context of interaction */
  context: string;
}

export interface ConflictPattern {
  /** Pattern of conflict type */
  pattern: string;
  /** Frequency of this pattern */
  frequency: number;
  /** Common resolution strategy */
  commonResolution: ResolutionStrategyType;
  /** Success rate of common resolution */
  successRate: number;
}

export interface MergePerformanceMetrics {
  /** Time taken for each merge phase */
  phaseTimes: {
    conflictDetection: number;
    conflictResolution: number;
    templateMerging: number;
    validation: number;
  };
  /** Memory usage by phase */
  memoryUsage: {
    peak: number;
    average: number;
    byPhase: Record<string, number>;
  };
  /** Cache performance */
  cachePerformance: {
    hits: number;
    misses: number;
    hitRate: number;
  };
  /** Template processing statistics */
  templateStats: {
    averageSize: number;
    largestTemplate: string;
    smallestTemplate: string;
    totalSize: number;
  };
}