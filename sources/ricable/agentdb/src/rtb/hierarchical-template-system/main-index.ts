/**
 * Hierarchical Template System - Phase 2 Architecture
 *
 * Complete implementation of priority-based template inheritance engine
 * for RTB configuration system with comprehensive variant generation,
 * conflict resolution, and validation capabilities.
 *
 * Main Features:
 * - Priority-based template inheritance (0-80 priority levels)
 * - Specialized variant generation (urban, mobility, sleep mode)
 * - Frequency relation management (4G4G, 4G5G, 5G5G, 5G4G)
 * - Intelligent conflict resolution with multiple strategies
 * - Comprehensive validation against XML constraints
 * - Performance optimization and caching
 * - Event-driven processing with metrics
 *
 * Usage Example:
 * ```typescript
 * import { PriorityTemplateEngine, TemplatePriority } from './hierarchical-template-system';
 *
 * // Initialize the engine
 * const engine = new PriorityTemplateEngine({
 *   cachingEnabled: true,
 *   defaultConflictStrategy: ConflictResolutionStrategy.HIGHEST_PRIORITY_WINS,
 *   performanceMonitoring: true
 * });
 *
 * // Register templates
 * await engine.registerTemplate(baseTemplate);
 * await engine.registerTemplate(urbanVariant);
 *
 * // Resolve inheritance chain
 * const chain = await engine.resolveInheritance('urban_variant');
 *
 * // Merge templates
 * const merged = await engine.mergeTemplates(['base', 'urban', 'mobility']);
 * ```
 */

// Core Engine
export { PriorityTemplateEngine } from './priority-template-engine';

// All Components
export * from './components';

// Main Interfaces and Types
export * from './interfaces';

// Re-export for convenience
export {
  // Engine Configuration
  type HierarchicalTemplateEngineConfig,

  // Template Types
  type PriorityTemplate,
  type TemplatePriority,
  type TemplateVariantType,
  type FrequencyBand,

  // Processing Results
  type TemplateInheritanceChain,
  type TemplateValidationResult,
  type TemplateConflict,

  // Configuration Types
  type VariantGenerationConfig,
  type UrbanConfig,
  type MobilityConfig,
  type SleepConfig,
  type FrequencyRelationConfig,

  // Utility Types
  type TemplateFilter,
  type ConflictResolutionStrategy,

  // Error Types
  TemplateSystemError,
  TemplateInheritanceError,
  TemplateValidationError,
  TemplateConflictError
} from './interfaces';

// Default configuration for common use cases
export const DEFAULT_ENGINE_CONFIG: HierarchicalTemplateEngineConfig = {
  cachingEnabled: true,
  maxCacheSize: 1000,
  defaultConflictStrategy: 'highest_priority_wins' as any,
  parallelProcessing: true,
  maxConcurrentOperations: 10,
  validationStrictness: 'strict',
  performanceMonitoring: true,
  detailedLogging: false
};

// Priority level constants for easy reference
export const TEMPLATE_PRIORITIES = {
  AGENTDB: 0,
  BASE: 9,
  URBAN: 20,
  MOBILITY: 30,
  SLEEP: 40,
  FREQUENCY_4G4G: 50,
  FREQUENCY_4G5G: 60,
  FREQUENCY_5G5G: 70,
  FREQUENCY_5G4G: 80
} as const;

// Template variant type constants
export const TEMPLATE_VARIANTS = {
  BASE: 'base',
  URBAN: 'urban',
  UAL_HIGH_CAPACITY: 'ual_high_capacity',
  HIGH_MOBILITY: 'high_mobility',
  SLEEP_MODE: 'sleep_mode',
  COASTAL: 'coastal',
  RURAL: 'rural',
  DENSE_URBAN: 'dense_urban',
  SUBURBAN: 'suburban'
} as const;

// Frequency band constants
export const FREQUENCY_BANDS = {
  LTE_800: 'lte_800',
  LTE_1800: 'lte_1800',
  LTE_2100: 'lte_2100',
  LTE_2600: 'lte_2600',
  NR_700: 'nr_700',
  NR_3500: 'nr_3500',
  NR_26000: 'nr_26000',
  NR_28000: 'nr_28000'
} as const;

// Utility function to create a template engine with sensible defaults
export function createTemplateEngine(config?: Partial<HierarchicalTemplateEngineConfig>): PriorityTemplateEngine {
  const finalConfig = { ...DEFAULT_ENGINE_CONFIG, ...config };
  return new PriorityTemplateEngine(finalConfig);
}

// Version information
export const HIERARCHICAL_TEMPLATE_SYSTEM_VERSION = '2.0.0';

// System capabilities
export const SYSTEM_CAPABILITIES = {
  maxTemplates: 10000,
  maxInheritanceDepth: 10,
  maxParametersPerTemplate: 5000,
  maxCustomFunctions: 100,
  supportedConflictStrategies: [
    'highest_priority_wins',
    'lowest_priority_wins',
    'merge_with_warning',
    'custom_function',
    'fail_on_conflict',
    'conflict_logging'
  ],
  supportedVariantTypes: Object.values(TEMPLATE_VARIANTS),
  supportedFrequencyBands: Object.values(FREQUENCY_BANDS)
} as const;