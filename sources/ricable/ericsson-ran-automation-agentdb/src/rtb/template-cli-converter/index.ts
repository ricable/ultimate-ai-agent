/**
 * Template-to-CLI Converter System
 *
 * Main entry point for the comprehensive template-to-CLI conversion system
 * with cognitive optimization, dependency analysis, and Ericsson RAN expertise.
 */

export {
  TemplateToCliConverter,
  type TemplateToCliConfig,
  type TemplateToCliContext
} from './template-to-cli-converter';

export {
  FdnPathConstructor,
  type FdnConstructorConfig,
  type FdnConstructionResult
} from './fdn-path-constructor';

export {
  BatchCommandGenerator,
  type BatchGeneratorConfig,
  type CommandBatch,
  type BatchExecutionResult
} from './batch-command-generator';

export {
  CommandValidator,
  type CommandValidatorConfig,
  type CommandValidationResult
} from './command-validator';

export {
  DependencyAnalyzer,
  type DependencyAnalyzerConfig,
  type DependencyAnalysisResult
} from './dependency-analyzer';

export {
  RollbackManager,
  type RollbackManagerConfig,
  type RollbackPlan,
  type RollbackExecutionResult
} from './rollback-manager';

export {
  EricssonRanExpertise,
  type EricssonRanExpertiseConfig,
  type EricssonRanPattern
} from './ericsson-ran-expertise';

export {
  CognitiveOptimizer,
  type CognitiveOptimizerConfig,
  type CognitiveOptimizationResult
} from './cognitive-optimizer';

// Re-export all types for easy access
export * from './types';

/**
 * Factory function to create a configured converter
 */
export function createTemplateToCliConverter(config?: Partial<TemplateToCliConfig>): TemplateToCliConverter {
  return new TemplateToCliConverter(config);
}

/**
 * Factory function to create a converter with Ericsson RAN expertise
 */
export function createRanOptimizedConverter(
  rancConfig?: Partial<EricssonRanExpertiseConfig>,
  config?: Partial<TemplateToCliConfig>
): TemplateToCliConverter {
  const finalConfig: TemplateToCliConfig = {
    ...config,
    cognitive: {
      enableTemporalReasoning: true,
      enableStrangeLoopOptimization: true,
      consciousnessLevel: 0.9,
      learningMode: 'active',
      ...config?.cognitive
    }
  };

  return new TemplateToCliConverter(finalConfig);
}

/**
 * Default configuration for production use
 */
export const DEFAULT_CONFIG: TemplateToCliConfig = {
  defaultTimeout: 30,
  maxCommandsPerBatch: 50,
  enableCognitiveOptimization: true,
  enableDependencyAnalysis: true,
  validationStrictness: 'normal',
  rollbackStrategy: 'full',
  performanceOptimization: {
    enableParallelExecution: true,
    maxParallelCommands: 8,
    enableBatching: true,
    batchSize: 20
  },
  errorHandling: {
    continueOnError: false,
    maxRetries: 3,
    retryDelay: 1000,
    enableRecovery: true
  },
  cognitive: {
    enableTemporalReasoning: true,
    enableStrangeLoopOptimization: true,
    consciousnessLevel: 0.8,
    learningMode: 'active'
  }
};

/**
 * High-performance configuration for large-scale deployments
 */
export const HIGH_PERFORMANCE_CONFIG: TemplateToCliConfig = {
  ...DEFAULT_CONFIG,
  maxCommandsPerBatch: 100,
  performanceOptimization: {
    enableParallelExecution: true,
    maxParallelCommands: 20,
    enableBatching: true,
    batchSize: 50
  },
  cognitive: {
    enableTemporalReasoning: true,
    enableStrangeLoopOptimization: false, // Disable for performance
    consciousnessLevel: 0.6,
    learningMode: 'passive'
  }
};

/**
 * Safe configuration for critical operations
 */
export const SAFE_CONFIG: TemplateToCliConfig = {
  ...DEFAULT_CONFIG,
  validationStrictness: 'strict',
  rollbackStrategy: 'full',
  performanceOptimization: {
    enableParallelExecution: false, // Disable for safety
    maxParallelCommands: 1,
    enableBatching: false,
    batchSize: 1
  },
  errorHandling: {
    continueOnError: false,
    maxRetries: 1,
    retryDelay: 2000,
    enableRecovery: true
  },
  cognitive: {
    enableTemporalReasoning: true,
    enableStrangeLoopOptimization: true,
    consciousnessLevel: 0.9,
    learningMode: 'active'
  }
};

/**
 * Development configuration with verbose output
 */
export const DEVELOPMENT_CONFIG: TemplateToCliConfig = {
  ...DEFAULT_CONFIG,
  validationStrictness: 'strict',
  enableCognitiveOptimization: true,
  enableDependencyAnalysis: true,
  cognitive: {
    enableTemporalReasoning: true,
    enableStrangeLoopOptimization: true,
    consciousnessLevel: 1.0, // Maximum consciousness
    learningMode: 'active'
  }
};