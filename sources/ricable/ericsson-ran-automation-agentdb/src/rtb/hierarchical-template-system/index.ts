/**
 * RTB Hierarchical Template System - Complete Priority-Based Template Management
 *
 * A comprehensive system for managing RTB (RAN Template Builder) templates with
 * priority-based inheritance, validation, performance optimization, and integration
 * with existing RTB processor components.
 *
 * Features:
 * - Priority-based template inheritance (0-80 priority levels)
 * - Complex inheritance chain resolution with conflict management
 * - Comprehensive template validation and error handling
 * - Performance optimization with caching and batch processing
 * - Integration with MO hierarchy and schema validation
 * - AgentDB learned pattern integration
 * - Custom function evaluation and conditional logic
 * - Template search and filtering capabilities
 */

// Legacy template merger exports (for backward compatibility)
export { TemplateMerger } from './template-merger';
export { ConflictDetector } from './conflict-detector';
export { ResolutionEngine, CustomResolver } from './resolution-engine';
export { MergeValidator, ValidationRule } from './merge-validator';

// New priority-based hierarchical template system exports
export {
  PriorityTemplateEngine,
  TemplatePriority,
  TemplatePriorityInfo,
  TemplateInheritanceChain,
  ParameterConflict,
  TemplateResolutionContext,
  TemplateValidationResult,
  ValidationError,
  ValidationWarning,
  CircularDependency,
  InheritanceNode,
  MergeConflictRule,
  ParameterMergeContext,
  InheritanceAnalysisResult
} from './priority-engine';

export {
  TemplateRegistry,
  TemplateSearchFilter,
  TemplateSearchResult,
  RegistryTemplateMeta,
  TemplateRegistryConfig,
  TemplateRegistryStats
} from './template-registry';

export {
  InheritanceResolver,
  InheritanceStrategy,
  InheritanceGraphNode,
  InheritanceResolutionOptions,
  ResolverInheritanceAnalysisResult
} from './inheritance-resolver';

export {
  PriorityManager,
  OverrideStrategy,
  PriorityAdjustmentRule,
  ParameterOverrideRule,
  PriorityRuleResult,
  ParameterOverrideResult,
  ConflictResolutionStrategy,
  PriorityCacheEntry
} from './priority-manager';

export {
  TemplateValidator,
  ValidationSeverity,
  ValidationRule as PriorityValidationRule,
  ValidationContext,
  ValidationResult as PriorityValidationResult,
  TemplateValidationSummary,
  ValidationConfig
} from './template-validator';

export {
  PerformanceOptimizer,
  OptimizationStrategy,
  CacheConfig,
  BatchConfig,
  IndexConfig,
  PerformanceMetrics,
  CacheEntry,
  BatchOperationResult,
  MemoryPool,
  OptimizationOptions
} from './performance-optimizer';

export {
  IntegratedTemplateSystem,
  IntegratedSystemConfig,
  TemplateProcessingResult,
  MOTemplateContext,
  SchemaValidationResult
} from './integrated-template-system';

export type {
  // Core types
  TemplateConflict,
  ConflictResolution,
  MergeResult,
  MergeStats,
  ResolutionApplied,
  InheritanceChain,
  InheritanceTree,
  MergeContext,
  MergeOptions,
  ValidationResult,
  ValidationError,
  ValidationWarning,
  ValidationStats,
  MergeInteraction,
  ConflictPattern,
  MergePerformanceMetrics,

  // Conflict types
  ConflictType,
  ResolutionStrategyType,

  // Conflict detection types
  ConflictDetectionOptions,
  ConflictDetectionResult,
  ConflictDetectionStats,

  // Resolution engine types
  ResolutionEngineOptions,
  ResolutionResult,
  CustomResolver as ICustomResolver,

  // Validation types
  ValidationOptions,
  ValidationRuleContext,
  ValidationReport,
  ValidationPerformanceMetrics,
  ValidationRecommendation
} from './types';

/**
 * Factory function to create a complete template merger system
 */
export function createTemplateMerger(options?: {
  merger?: Partial<import('./template-merger').MergeOptions>;
  conflictDetector?: Partial<import('./conflict-detector').ConflictDetectionOptions>;
  resolutionEngine?: Partial<import('./resolution-engine').ResolutionEngineOptions>;
  validator?: Partial<import('./merge-validator').ValidationOptions>;
}) {
  const templateMerger = new TemplateMerger(options?.merger);
  const conflictDetector = new ConflictDetector(options?.conflictDetector);
  const resolutionEngine = new ResolutionEngine(options?.resolutionEngine);
  const mergeValidator = new MergeValidator(options?.validator);

  return {
    templateMerger,
    conflictDetector,
    resolutionEngine,
    mergeValidator,

    // Convenience method for complete merge workflow
    async mergeTemplates(templates: import('../types/rtb-types').RTBTemplate[]) {
      return await templateMerger.mergeTemplates(templates);
    },

    // Statistics and diagnostics
    getSystemStats() {
      return {
        merger: templateMerger.getCacheStats(),
        conflictDetector: conflictDetector.getConflictPatterns(),
        resolutionEngine: resolutionEngine.getResolutionStats(),
        validator: mergeValidator.getRegisteredValidators()
      };
    },

    // Cache management
    clearAllCaches() {
      templateMerger.clearCache();
      conflictDetector.clearPatterns();
      resolutionEngine.clearCache();
    }
  };
}

/**
 * Default export for easy usage
 */
export default createTemplateMerger;