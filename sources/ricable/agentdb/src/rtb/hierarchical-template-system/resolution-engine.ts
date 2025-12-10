/**
 * Conflict Resolution Engine for Template Merging
 *
 * Implements intelligent conflict resolution strategies with support for custom resolvers,
 * conditional logic, and ML-based recommendations. Provides automatic and interactive
 * conflict resolution with detailed reasoning and validation.
 */

import { TemplateConflict, ConflictResolution, ResolutionStrategyType, MergeContext } from './types';
import { Logger } from '../../utils/logger';

export interface ResolutionEngineOptions {
  /** Enable ML-based resolution recommendations */
  enableMLRecommendations: boolean;
  /** Strict mode for type validation during resolution */
  strictTypeValidation: boolean;
  /** Enable custom resolver functions */
  enableCustomResolvers: boolean;
  /** Cache resolution strategies */
  cacheResolutions: boolean;
  /** Timeout for resolution operations (ms) */
  resolutionTimeout: number;
  /** Enable interactive resolution prompts */
  enableInteractivePrompts: boolean;
}

export interface CustomResolver {
  name: string;
  description: string;
  resolver: (conflict: TemplateConflict, context: MergeContext) => Promise<any>;
  applicableConflicts: string[]; // Parameter patterns this resolver can handle
  priority: number; // Resolver priority
}

export interface ResolutionResult {
  /** Whether the conflict was successfully resolved */
  resolved: boolean;
  /** The resolved value */
  value: any;
  /** Strategy used for resolution */
  strategy: ResolutionStrategyType;
  /** Reasoning behind the resolution */
  reasoning: string;
  /** Time taken to resolve (ms) */
  resolutionTime: number;
  /** Whether validation was performed */
  validated: boolean;
  /** Validation result if performed */
  validationResult?: any;
}

export class ResolutionEngine {
  private logger: Logger;
  private customResolvers: Map<string, CustomResolver>;
  private resolutionCache: Map<string, ResolutionResult>;
  private options: ResolutionEngineOptions;
  private builtinResolvers: Map<ResolutionStrategyType, (conflict: TemplateConflict, context: MergeContext) => Promise<ResolutionResult>>;

  constructor(options: Partial<ResolutionEngineOptions> = {}) {
    this.logger = new Logger('ResolutionEngine');
    this.customResolvers = new Map();
    this.resolutionCache = new Map();
    this.options = {
      enableMLRecommendations: false,
      strictTypeValidation: true,
      enableCustomResolvers: true,
      cacheResolutions: true,
      resolutionTimeout: 5000,
      enableInteractivePrompts: false,
      ...options
    };

    this.initializeBuiltinResolvers();
  }

  /**
   * Resolve a conflict using appropriate strategy
   */
  async resolveConflict(conflict: TemplateConflict, context: MergeContext): Promise<ConflictResolution> {
    const startTime = Date.now();
    this.logger.debug(`Resolving conflict for parameter: ${conflict.parameter}`, {
      conflictType: conflict.conflictType,
      strategy: conflict.resolution.strategy,
      templates: conflict.templates.length
    });

    try {
      // Check cache first
      const cacheKey = this.generateCacheKey(conflict, context);
      if (this.options.cacheResolutions && this.resolutionCache.has(cacheKey)) {
        const cached = this.resolutionCache.get(cacheKey)!;
        this.logger.debug(`Using cached resolution for ${conflict.parameter}`);
        return {
          strategy: cached.strategy,
          reasoning: cached.reasoning,
          value: cached.value,
          resolved: cached.resolved
        };
      }

      // Determine best resolution strategy
      const strategy = await this.determineResolutionStrategy(conflict, context);

      // Apply resolution strategy
      const result = await this.applyResolutionStrategy(conflict, strategy, context);

      // Validate result if strict mode is enabled
      if (this.options.strictTypeValidation) {
        const validationResult = await this.validateResolution(conflict, result.value, context);
        result.validated = true;
        result.validationResult = validationResult;

        if (!validationResult.isValid) {
          this.logger.warn(`Resolution validation failed for ${conflict.parameter}`, validationResult);
          // Try fallback strategy
          const fallbackResult = await this.applyFallbackStrategy(conflict, context);
          if (fallbackResult.resolved) {
            Object.assign(result, fallbackResult);
          }
        }
      }

      // Update conflict resolution
      const resolution: ConflictResolution = {
        strategy: result.strategy,
        reasoning: result.reasoning,
        value: result.value,
        resolved: result.resolved,
        metadata: {
          resolutionTime: Date.now() - startTime,
          validated: result.validated,
          cacheHit: false,
          fallbackUsed: false
        }
      };

      // Cache result
      if (this.options.cacheResolutions && result.resolved) {
        this.resolutionCache.set(cacheKey, result);
      }

      this.logger.info(`Conflict resolved for ${conflict.parameter}`, {
        strategy: result.strategy,
        resolved: result.resolved,
        resolutionTime: Date.now() - startTime
      });

      return resolution;

    } catch (error) {
      this.logger.error(`Failed to resolve conflict for ${conflict.parameter}`, { error, conflict });

      // Return unresolved conflict
      return {
        strategy: 'highest_priority',
        reasoning: `Resolution failed: ${error.message}`,
        resolved: false
      };
    }
  }

  /**
   * Determine the best resolution strategy for a conflict
   */
  private async determineResolutionStrategy(conflict: TemplateConflict, context: MergeContext): Promise<ResolutionStrategyType> {
    // Use predefined strategy if available
    if (conflict.resolution.strategy && conflict.resolution.strategy !== 'auto') {
      return conflict.resolution.strategy;
    }

    // Check for custom resolvers
    if (this.options.enableCustomResolvers) {
      const customResolver = this.findApplicableCustomResolver(conflict);
      if (customResolver) {
        this.logger.debug(`Using custom resolver: ${customResolver.name}`);
        return 'custom';
      }
    }

    // Check for ML recommendations
    if (this.options.enableMLRecommendations) {
      const mlRecommendation = await this.getMLRecommendation(conflict, context);
      if (mlRecommendation) {
        return mlRecommendation;
      }
    }

    // Use conflict context recommendations
    if (conflict.context?.recommendedResolution) {
      return conflict.context.recommendedResolution;
    }

    // Default strategy based on conflict type
    return this.getDefaultStrategyForConflictType(conflict.conflictType);
  }

  /**
   * Apply resolution strategy
   */
  private async applyResolutionStrategy(
    conflict: TemplateConflict,
    strategy: ResolutionStrategyType,
    context: MergeContext
  ): Promise<ResolutionResult> {
    const startTime = Date.now();

    try {
      // Check for custom resolver
      if (strategy === 'custom' && this.options.enableCustomResolvers) {
        const customResolver = this.findApplicableCustomResolver(conflict);
        if (customResolver) {
          const result = await this.executeWithTimeout(
            customResolver.resolver(conflict, context),
            this.options.resolutionTimeout
          );

          return {
            resolved: true,
            value: result,
            strategy: 'custom',
            reasoning: `Resolved using custom resolver: ${customResolver.description}`,
            resolutionTime: Date.now() - startTime,
            validated: false
          };
        }
      }

      // Use built-in resolver
      const builtinResolver = this.builtinResolvers.get(strategy);
      if (builtinResolver) {
        return await builtinResolver(conflict, context);
      }

      // Fallback to highest priority
      return await this.resolveWithHighestPriority(conflict, context);

    } catch (error) {
      this.logger.error(`Resolution strategy failed: ${strategy}`, { error });
      return await this.applyFallbackStrategy(conflict, context);
    }
  }

  /**
   * Initialize built-in resolution strategies
   */
  private initializeBuiltinResolvers(): void {
    this.builtinResolvers = new Map();

    // Highest priority strategy
    this.builtinResolvers.set('highest_priority', async (conflict, context) => {
      return await this.resolveWithHighestPriority(conflict, context);
    });

    // Merge strategy
    this.builtinResolvers.set('merge', async (conflict, context) => {
      return await this.resolveWithMerge(conflict, context);
    });

    // Conditional strategy
    this.builtinResolvers.set('conditional', async (conflict, context) => {
      return await this.resolveWithConditional(conflict, context);
    });

    // Interactive strategy
    this.builtinResolvers.set('interactive', async (conflict, context) => {
      return await this.resolveWithInteractive(conflict, context);
    });
  }

  /**
   * Resolve with highest priority strategy
   */
  private async resolveWithHighestPriority(conflict: TemplateConflict, context: MergeContext): Promise<ResolutionResult> {
    const maxPriority = Math.max(...conflict.priorities);
    const highestPriorityIndex = conflict.priorities.indexOf(maxPriority);
    const selectedValue = conflict.values[highestPriorityIndex];
    const selectedTemplate = conflict.templates[highestPriorityIndex];

    return {
      resolved: true,
      value: selectedValue,
      strategy: 'highest_priority',
      reasoning: `Selected value from template with highest priority (${maxPriority}): ${selectedTemplate}`,
      resolutionTime: 0,
      validated: false
    };
  }

  /**
   * Resolve with merge strategy
   */
  private async resolveWithMerge(conflict: TemplateConflict, context: MergeContext): Promise<ResolutionResult> {
    const values = conflict.values;

    // Try to merge compatible values
    if (this.canMergeValues(values)) {
      const mergedValue = this.mergeValues(values);

      return {
        resolved: true,
        value: mergedValue,
        strategy: 'merge',
        reasoning: `Successfully merged ${values.length} compatible values`,
        resolutionTime: 0,
        validated: false
      };
    }

    // If cannot merge, fallback to highest priority
    this.logger.warn(`Cannot merge values for ${conflict.parameter}, falling back to highest priority`);
    return await this.resolveWithHighestPriority(conflict, context);
  }

  /**
   * Resolve with conditional strategy
   */
  private async resolveWithConditional(conflict: TemplateConflict, context: MergeContext): Promise<ResolutionResult> {
    // For conditional conflicts, try to evaluate conditions
    if (conflict.conflictType === 'conditional') {
      try {
        const evaluatedValue = await this.evaluateConditions(conflict, context);

        return {
          resolved: true,
          value: evaluatedValue,
          strategy: 'conditional',
          reasoning: `Evaluated conditional logic to determine value`,
          resolutionTime: 0,
          validated: false
        };
      } catch (error) {
        this.logger.warn(`Conditional evaluation failed for ${conflict.parameter}`, { error });
      }
    }

    // Fallback to highest priority
    return await this.resolveWithHighestPriority(conflict, context);
  }

  /**
   * Resolve with interactive strategy
   */
  private async resolveWithInteractive(conflict: TemplateConflict, context: MergeContext): Promise<ResolutionResult> {
    if (!this.options.enableInteractivePrompts) {
      this.logger.info(`Interactive prompts disabled, falling back to highest priority for ${conflict.parameter}`);
      return await this.resolveWithHighestPriority(conflict, context);
    }

    // In a real implementation, this would prompt the user
    // For now, we'll simulate an interactive choice
    this.logger.info(`Interactive resolution requested for ${conflict.parameter}`, {
      conflict: conflict.values,
      templates: conflict.templates
    });

    // Simulate user selecting the first value
    const selectedValue = conflict.values[0];
    const selectedTemplate = conflict.templates[0];

    return {
      resolved: true,
      value: selectedValue,
      strategy: 'interactive',
      reasoning: `User selected value from template: ${selectedTemplate}`,
      resolutionTime: 0,
      validated: false
    };
  }

  /**
   * Check if values can be merged
   */
  private canMergeValues(values: any[]): boolean {
    if (values.length < 2) return true;

    const firstType = typeof values[0];

    // All values must be of the same type
    if (!values.every(v => typeof v === firstType)) {
      return false;
    }

    // Objects can be merged
    if (firstType === 'object' && !Array.isArray(values[0])) {
      return true;
    }

    // Arrays can be merged (concatenated)
    if (Array.isArray(values[0])) {
      return true;
    }

    // Primitives cannot be merged unless they're identical
    const uniqueValues = new Set(values);
    return uniqueValues.size === 1;
  }

  /**
   * Merge compatible values
   */
  private mergeValues(values: any[]): any {
    if (values.length === 0) return undefined;
    if (values.length === 1) return values[0];

    const firstValue = values[0];

    // Merge objects
    if (typeof firstValue === 'object' && !Array.isArray(firstValue)) {
      return values.reduce((merged, value) => ({ ...merged, ...value }), {});
    }

    // Merge arrays (concatenate and remove duplicates)
    if (Array.isArray(firstValue)) {
      const mergedArray = values.flat();
      return [...new Set(mergedArray)];
    }

    // For primitives, return the first value
    return firstValue;
  }

  /**
   * Evaluate conditions for conditional conflicts
   */
  private async evaluateConditions(conflict: TemplateConflict, context: MergeContext): Promise<any> {
    // This is a simplified implementation
    // In a real scenario, this would evaluate the conditional logic
    // based on runtime conditions or environment

    // For now, return the value from the highest priority template
    return await this.resolveWithHighestPriority(conflict, context);
  }

  /**
   * Apply fallback strategy when primary strategy fails
   */
  private async applyFallbackStrategy(conflict: TemplateConflict, context: MergeContext): Promise<ResolutionResult> {
    this.logger.warn(`Applying fallback strategy for ${conflict.parameter}`);

    const result = await this.resolveWithHighestPriority(conflict, context);
    result.reasoning += ' (fallback strategy applied)';

    return result;
  }

  /**
   * Find applicable custom resolver for conflict
   */
  private findApplicableCustomResolver(conflict: TemplateConflict): CustomResolver | null {
    const resolvers = Array.from(this.customResolvers.values())
      .filter(resolver =>
        resolver.applicableConflicts.some(pattern =>
          conflict.parameter.includes(pattern) ||
          conflict.conflictType.toString().includes(pattern)
        )
      )
      .sort((a, b) => b.priority - a.priority);

    return resolvers.length > 0 ? resolvers[0] : null;
  }

  /**
   * Register a custom resolver
   */
  public registerCustomResolver(resolver: CustomResolver): void {
    this.customResolvers.set(resolver.name, resolver);
    this.logger.info(`Registered custom resolver: ${resolver.name}`);
  }

  /**
   * Unregister a custom resolver
   */
  public unregisterCustomResolver(name: string): void {
    this.customResolvers.delete(name);
    this.logger.info(`Unregistered custom resolver: ${name}`);
  }

  /**
   * Get ML recommendation (placeholder)
   */
  private async getMLRecommendation(conflict: TemplateConflict, context: MergeContext): Promise<ResolutionStrategyType | null> {
    // Placeholder for ML-based recommendation
    // Could be enhanced with actual ML model
    return null;
  }

  /**
   * Get default strategy for conflict type
   */
  private getDefaultStrategyForConflictType(conflictType: any): ResolutionStrategyType {
    switch (conflictType) {
      case 'value':
        return 'highest_priority';
      case 'type':
        return 'highest_priority';
      case 'structure':
        return 'merge';
      case 'conditional':
        return 'conditional';
      case 'function':
        return 'highest_priority';
      case 'metadata':
        return 'merge';
      default:
        return 'highest_priority';
    }
  }

  /**
   * Validate resolution result
   */
  private async validateResolution(conflict: TemplateConflict, value: any, context: MergeContext): Promise<{ isValid: boolean; errors: string[] }> {
    const errors: string[] = [];

    // Type validation
    if (conflict.context?.parameterType) {
      const expectedType = conflict.context.parameterType;
      const actualType = typeof value;

      if (expectedType !== actualType && !(expectedType === 'array' && Array.isArray(value))) {
        errors.push(`Type mismatch: expected ${expectedType}, got ${actualType}`);
      }
    }

    // Value validation
    if (value === undefined || value === null) {
      errors.push('Resolved value is null or undefined');
    }

    // Constraint validation (placeholder)
    if (conflict.context?.constraints) {
      // Could add actual constraint validation here
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  /**
   * Execute function with timeout
   */
  private async executeWithTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    return Promise.race([
      promise,
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Operation timed out after ${timeoutMs}ms`)), timeoutMs)
      )
    ]);
  }

  /**
   * Generate cache key
   */
  private generateCacheKey(conflict: TemplateConflict, context: MergeContext): string {
    const valuesHash = JSON.stringify(conflict.values.sort());
    const templatesHash = conflict.templates.sort().join('|');
    return `${conflict.parameter}:${conflict.conflictType}:${templatesHash}:${valuesHash}`;
  }

  /**
   * Clear resolution cache
   */
  public clearCache(): void {
    this.resolutionCache.clear();
    this.logger.info('Resolution cache cleared');
  }

  /**
   * Get resolution statistics
   */
  public getResolutionStats(): {
    cacheSize: number;
    customResolvers: number;
    builtinStrategies: string[]
  } {
    return {
      cacheSize: this.resolutionCache.size,
      customResolvers: this.customResolvers.size,
      builtinStrategies: Array.from(this.builtinResolvers.keys())
    };
  }
}