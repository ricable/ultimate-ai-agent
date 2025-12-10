/**
 * Hierarchical Template System - Template Merger & Conflict Resolution
 *
 * A sophisticated template merging and conflict resolution system for RTB configurations
 * with intelligent inheritance resolution, advanced conflict handling, and comprehensive validation.
 */

// Core classes
export { TemplateMerger } from './template-merger';
export { ConflictDetector } from './conflict-detector';
export { ResolutionEngine, CustomResolver } from './resolution-engine';
export { MergeValidator, ValidationRule } from './merge-validator';

// Type definitions
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
    async mergeTemplates(templates: import('../../types/rtb-types').RTBTemplate[]) {
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