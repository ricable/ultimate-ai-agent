/**
 * Hierarchical Template System Components - Module Exports
 *
 * Exports all template system components for easy importing and usage.
 * Provides a comprehensive set of tools for template inheritance,
 * variant generation, conflict resolution, and validation.
 */

// Core Components
export { PriorityTemplateEngine } from '../priority-template-engine';

// Template Processing Components
export { TemplateVariantGenerator } from './template-variant-generator';
export { TemplateMerger } from './template-merger';
export { TemplateConflictResolver } from './template-conflict-resolver';
export { TemplateValidator } from './template-validator';
export { FrequencyRelationManager } from './frequency-relation-manager';

// Event System
export { TemplateEventBus } from './template-event-bus';

// Component Interfaces
export type {
  ITemplateVariantGenerator,
  ITemplateMerger,
  ITemplateConflictResolver,
  ITemplateValidator,
  IFrequencyRelationManager,
  ITemplateEventBus,
  ITemplateEventListener
} from '../interfaces';

// Re-export main interfaces for convenience
export type {
  PriorityTemplate,
  TemplatePriority,
  TemplateInheritanceChain,
  TemplateConflict,
  ConflictResolutionStrategy,
  VariantGenerationConfig,
  TemplateValidationResult,
  TemplateFilter,
  HierarchicalTemplateEngineConfig,
  TemplateVariantType,
  FrequencyBand,
  UrbanConfig,
  MobilityConfig,
  SleepConfig,
  FrequencyRelationConfig,
  NeighborRelationConfig,
  HandoverParameterConfig,
  CapacityParameterConfig
} from '../interfaces';

// Re-export error types
export {
  TemplateSystemError,
  TemplateInheritanceError,
  TemplateValidationError,
  TemplateConflictError
} from '../interfaces';