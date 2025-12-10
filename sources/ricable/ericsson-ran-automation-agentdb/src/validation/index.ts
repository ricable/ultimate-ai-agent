/**
 * Complex Validation Rules Engine - Phase 5 Implementation
 * Main integration and public API for the validation system
 *
 * Comprehensive validation system with:
 * - CSV parameter specification processing (~19,000 parameters)
 * - Cross-parameter validation with conditional logic
 * - Real-time validation performance optimization (<300ms)
 * - Integration with cognitive consciousness system
 * - AgentDB memory pattern integration
 * - ReservedBy relationship constraint validation
 * - Pydantic schema generation
 */

export { ValidationEngine } from './validation-engine';
export { ConstraintProcessor } from './constraint-processor';
export { ConditionalValidator } from './conditional-validator';
export { ValidationSchemaGenerator } from './schema-generator';

// Export all types
export * from '../types/validation-types';

// Create and export the main validation factory
import { ValidationEngine } from './validation-engine';
import { ValidationEngineConfig } from '../types/validation-types';
import { CognitiveConsciousnessCore } from '../cognitive/CognitiveConsciousnessCore';

/**
 * Validation Factory - Creates configured validation instances
 */
export class ValidationFactory {
  /**
   * Create a complete validation engine with all components
   */
  static async createValidationEngine(config?: ValidationEngineConfig): Promise<ValidationEngine> {
    const defaultConfig: ValidationEngineConfig = {
      maxValidationTime: 300, // 300ms target
      cacheEnabled: true,
      cacheTTL: 300000, // 5 minutes
      learningEnabled: true,
      consciousnessIntegration: true,
      strictMode: false,
      parallelProcessing: true,
      batchSize: 100,
      maxValidationDepth: 10,
      pydanticIntegration: false,
      agentDBIntegration: false
    };

    const finalConfig = { ...defaultConfig, ...config };

    const validationEngine = new ValidationEngine(finalConfig);
    await validationEngine.initialize();

    return validationEngine;
  }

  /**
   * Create validation engine with cognitive consciousness integration
   */
  static async createCognitiveValidationEngine(
    cognitiveCore: CognitiveConsciousnessCore,
    config?: ValidationEngineConfig
  ): Promise<ValidationEngine> {
    const cognitiveConfig: ValidationEngineConfig = {
      ...config,
      consciousnessIntegration: true,
      cognitiveCore,
      learningEnabled: true,
      cacheEnabled: true
    };

    return this.createValidationEngine(cognitiveConfig);
  }

  /**
   * Create validation engine for production use
   */
  static async createProductionValidationEngine(config?: ValidationEngineConfig): Promise<ValidationEngine> {
    const productionConfig: ValidationEngineConfig = {
      maxValidationTime: 250, // Stricter 250ms target for production
      cacheEnabled: true,
      cacheTTL: 600000, // 10 minutes for production
      learningEnabled: true,
      consciousnessIntegration: false, // Disabled for production stability
      strictMode: true,
      parallelProcessing: true,
      batchSize: 200,
      maxValidationDepth: 8,
      pydanticIntegration: true,
      agentDBIntegration: true,
      ...config
    };

    return this.createValidationEngine(productionConfig);
  }

  /**
   * Create validation engine for development/testing
   */
  static async createDevelopmentValidationEngine(config?: ValidationEngineConfig): Promise<ValidationEngine> {
    const developmentConfig: ValidationEngineConfig = {
      maxValidationTime: 500, // More relaxed for development
      cacheEnabled: true,
      cacheTTL: 60000, // 1 minute for development
      learningEnabled: true,
      consciousnessIntegration: true,
      strictMode: false,
      parallelProcessing: true,
      batchSize: 50,
      maxValidationDepth: 15,
      pydanticIntegration: false,
      agentDBIntegration: false,
      ...config
    };

    return this.createValidationEngine(developmentConfig);
  }
}

/**
 * Validation utilities and helpers
 */
export class ValidationUtils {
  /**
   * Validate configuration with minimal setup
   */
  static async validateConfiguration(
    configuration: Record<string, any>,
    options?: {
      level?: 'basic' | 'standard' | 'comprehensive';
      strictMode?: boolean;
      enableCaching?: boolean;
    }
  ): Promise<any> {
    const validationEngine = await ValidationFactory.createValidationEngine({
      strictMode: options?.strictMode || false,
      cacheEnabled: options?.enableCaching !== false
    });

    const result = await validationEngine.validateConfiguration(configuration, {
      validationId: `quick_${Date.now()}`,
      timestamp: Date.now(),
      configuration,
      validationLevel: options?.level || 'standard'
    });

    await validationEngine.shutdown();
    return result;
  }

  /**
   * Quick parameter validation
   */
  static async validateParameter(
    parameterName: string,
    value: any,
    constraints?: any[]
  ): Promise<boolean> {
    try {
      // Simple validation logic for basic use cases
      if (!constraints || constraints.length === 0) {
        return true;
      }

      for (const constraint of constraints) {
        if (!this.validateSingleConstraint(constraint, value)) {
          return false;
        }
      }

      return true;
    } catch (error) {
      console.warn(`Parameter validation failed for ${parameterName}:`, error);
      return false;
    }
  }

  /**
   * Validate single constraint
   */
  private static validateSingleConstraint(constraint: any, value: any): boolean {
    switch (constraint.type) {
      case 'required':
        return value !== undefined && value !== null && value !== '';
      case 'range':
        if (typeof value === 'number') {
          const range = constraint.value;
          return (range.min === undefined || value >= range.min) &&
                 (range.max === undefined || value <= range.max);
        }
        return false;
      case 'enum':
        return constraint.value.includes(String(value));
      case 'pattern':
        if (typeof value === 'string') {
          const regex = new RegExp(constraint.value);
          return regex.test(value);
        }
        return false;
      default:
        return true;
    }
  }

  /**
   * Extract validation errors by category
   */
  static extractErrorsByCategory(errors: any[], category: string): any[] {
    return errors.filter(error => error.category === category);
  }

  /**
   * Extract validation errors by severity
   */
  static extractErrorsBySeverity(errors: any[], severity: string): any[] {
    return errors.filter(error => error.severity === severity);
  }

  /**
   * Format validation results for display
   */
  static formatValidationResults(result: any): string {
    let output = `Validation Results (ID: ${result.validationId})\n`;
    output += `Status: ${result.valid ? '✅ VALID' : '❌ INVALID'}\n`;
    output += `Execution Time: ${result.executionTime}ms\n`;
    output += `Parameters Validated: ${result.parametersValidated}\n`;
    output += `Cache Hit Rate: ${result.cacheHitRate.toFixed(1)}%\n`;

    if (result.errors.length > 0) {
      output += `\nErrors (${result.errors.length}):\n`;
      result.errors.forEach((error: any, index: number) => {
        output += `  ${index + 1}. [${error.code}] ${error.message}\n`;
        output += `     Parameter: ${error.parameter} | Category: ${error.category}\n`;
      });
    }

    if (result.warnings.length > 0) {
      output += `\nWarnings (${result.warnings.length}):\n`;
      result.warnings.forEach((warning: any, index: number) => {
        output += `  ${index + 1}. [${warning.code}] ${warning.message}\n`;
        output += `     Parameter: ${warning.parameter} | Category: ${warning.category}\n`;
      });
    }

    if (result.cognitiveInsights) {
      output += `\nCognitive Insights: ${result.cognitiveInsights.insights?.length || 0}\n`;
    }

    return output;
  }

  /**
   * Generate validation summary statistics
   */
  static generateValidationSummary(results: any[]): any {
    if (results.length === 0) {
      return {
        totalValidations: 0,
        successRate: 0,
        averageExecutionTime: 0,
        errorCategories: {},
        mostCommonErrors: []
      };
    }

    const successful = results.filter(r => r.valid).length;
    const totalExecutionTime = results.reduce((sum, r) => sum + r.executionTime, 0);
    const allErrors = results.flatMap(r => r.errors);

    // Count errors by category
    const errorCategories: Record<string, number> = {};
    allErrors.forEach(error => {
      errorCategories[error.category] = (errorCategories[error.category] || 0) + 1;
    });

    // Count most common errors
    const errorCounts: Record<string, number> = {};
    allErrors.forEach(error => {
      const key = `${error.code}:${error.parameter}`;
      errorCounts[key] = (errorCounts[key] || 0) + 1;
    });

    const mostCommonErrors = Object.entries(errorCounts)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([error, count]) => ({ error, count }));

    return {
      totalValidations: results.length,
      successRate: (successful / results.length) * 100,
      averageExecutionTime: totalExecutionTime / results.length,
      errorCategories,
      mostCommonErrors,
      totalErrors: allErrors.length,
      totalWarnings: results.reduce((sum, r) => sum + r.warnings.length, 0)
    };
  }
}

/**
 * Performance monitoring for validation system
 */
export class ValidationPerformanceMonitor {
  private static instance: ValidationPerformanceMonitor;
  private metrics: Map<string, any[]> = new Map();
  private startTimes: Map<string, number> = new Map();

  static getInstance(): ValidationPerformanceMonitor {
    if (!ValidationPerformanceMonitor.instance) {
      ValidationPerformanceMonitor.instance = new ValidationPerformanceMonitor();
    }
    return ValidationPerformanceMonitor.instance;
  }

  /**
   * Start timing a validation operation
   */
  startTiming(operationId: string): void {
    this.startTimes.set(operationId, Date.now());
  }

  /**
   * End timing and record metrics
   */
  endTiming(operationId: string, additionalMetrics?: any): void {
    const startTime = this.startTimes.get(operationId);
    if (!startTime) {
      console.warn(`No start time found for operation: ${operationId}`);
      return;
    }

    const executionTime = Date.now() - startTime;
    const metrics = {
      executionTime,
      timestamp: Date.now(),
      ...additionalMetrics
    };

    if (!this.metrics.has(operationId)) {
      this.metrics.set(operationId, []);
    }

    this.metrics.get(operationId)!.push(metrics);
    this.startTimes.delete(operationId);

    // Keep only last 100 metrics per operation
    const operationMetrics = this.metrics.get(operationId)!;
    if (operationMetrics.length > 100) {
      this.metrics.set(operationId, operationMetrics.slice(-100));
    }
  }

  /**
   * Get performance metrics for an operation
   */
  getMetrics(operationId: string): any {
    const operationMetrics = this.metrics.get(operationId) || [];

    if (operationMetrics.length === 0) {
      return null;
    }

    const executionTimes = operationMetrics.map(m => m.executionTime);
    const averageTime = executionTimes.reduce((sum, time) => sum + time, 0) / executionTimes.length;
    const minTime = Math.min(...executionTimes);
    const maxTime = Math.max(...executionTimes);

    return {
      operationId,
      totalOperations: operationMetrics.length,
      averageExecutionTime: averageTime,
      minExecutionTime: minTime,
      maxExecutionTime: maxTime,
      lastExecution: operationMetrics[operationMetrics.length - 1]?.timestamp,
      metrics: operationMetrics
    };
  }

  /**
   * Get all performance metrics
   */
  getAllMetrics(): Record<string, any> {
    const allMetrics: Record<string, any> = {};

    for (const [operationId] of this.metrics) {
      allMetrics[operationId] = this.getMetrics(operationId);
    }

    return allMetrics;
  }

  /**
   * Clear metrics for an operation
   */
  clearMetrics(operationId?: string): void {
    if (operationId) {
      this.metrics.delete(operationId);
      this.startTimes.delete(operationId);
    } else {
      this.metrics.clear();
      this.startTimes.clear();
    }
  }
}

/**
 * Validation constants and defaults
 */
export const ValidationConstants = {
  // Performance targets
  MAX_VALIDATION_TIME: 300, // milliseconds
  TARGET_CACHE_HIT_RATE: 80, // percentage
  TARGET_VALIDATION_COVERAGE: 99.9, // percentage

  // Default configuration
  DEFAULT_BATCH_SIZE: 100,
  DEFAULT_CACHE_TTL: 300000, // 5 minutes
  DEFAULT_MAX_VALIDATION_DEPTH: 10,

  // Error codes
  ERROR_CODES: {
    UNKNOWN_PARAMETER: 'UNKNOWN_PARAMETER',
    CONSTRAINT_VIOLATION: 'CONSTRAINT_VIOLATION',
    CROSS_PARAMETER_VIOLATION: 'CROSS_PARAMETER_VIOLATION',
    CONDITIONAL_VALIDATION_ERROR: 'CONDITIONAL_VALIDATION_ERROR',
    SCHEMA_VALIDATION_ERROR: 'SCHEMA_VALIDATION_ERROR',
    COGNITIVE_VALIDATION_ERROR: 'COGNITIVE_VALIDATION_ERROR'
  },

  // Validation categories
  CATEGORIES: {
    PARAMETER: 'parameter',
    CONSTRAINT: 'constraint',
    CROSS_PARAMETER: 'cross_parameter',
    MO_CLASS: 'mo_class',
    RESERVED_BY: 'reserved_by',
    CONDITIONAL: 'conditional',
    SCHEMA: 'schema',
    TEMPORAL: 'temporal',
    COGNITIVE: 'cognitive',
    SYSTEM: 'system'
  }
};

// Re-export for convenience
export { ValidationEngine as default } from './validation-engine';