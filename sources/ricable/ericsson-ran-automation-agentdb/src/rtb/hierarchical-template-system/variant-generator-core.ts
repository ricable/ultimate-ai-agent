/**
 * Core Template Variant Generator Engine
 *
 * This module provides the foundation for generating specialized RTB template variants
 * for different RAN deployment scenarios. It handles template inheritance, merging,
 * and variant-specific optimizations.
 */

import { RTBTemplate, TemplateMeta, CustomFunction, ConditionOperator, EvaluationOperator } from '../../types/rtb-types';

export interface VariantConfig {
  variantType: 'urban' | 'mobility' | 'sleep';
  priority: number;
  baseTemplates: string[];
  optimizations: VariantOptimization[];
  customLogic: CustomFunction[];
  conditions: Record<string, ConditionOperator>;
  evaluations: Record<string, EvaluationOperator>;
}

export interface VariantOptimization {
  parameter: string;
  value: any;
  context: string;
  description: string;
  priority: number;
  conditions?: string[];
}

export interface VariantGenerationOptions {
  targetEnvironment?: string;
  cellCount?: number;
  trafficProfile?: 'high' | 'medium' | 'low';
  energyMode?: 'performance' | 'balanced' | 'energy_saving';
  customOverrides?: Record<string, any>;
}

export class VariantGeneratorCore {
  private baseTemplates: Map<string, RTBTemplate> = new Map();
  private variantConfigs: Map<string, VariantConfig> = new Map();
  private optimizationHistory: Map<string, VariantOptimization[]> = new Map();

  constructor() {
    this.initializeDefaultVariantConfigs();
  }

  /**
   * Register a base template for variant generation
   */
  registerBaseTemplate(name: string, template: RTBTemplate): void {
    this.baseTemplates.set(name, template);
  }

  /**
   * Register a variant configuration
   */
  registerVariantConfig(variantType: string, config: VariantConfig): void {
    this.variantConfigs.set(variantType, config);
  }

  /**
   * Generate a variant template based on configuration
   */
  generateVariant(
    variantType: string,
    baseTemplateName: string,
    options: VariantGenerationOptions = {}
  ): RTBTemplate {
    const baseTemplate = this.baseTemplates.get(baseTemplateName);
    if (!baseTemplate) {
      throw new Error(`Base template '${baseTemplateName}' not found`);
    }

    const variantConfig = this.variantConfigs.get(variantType);
    if (!variantConfig) {
      throw new Error(`Variant configuration for '${variantType}' not found`);
    }

    // Start with base template
    let variantTemplate: RTBTemplate = JSON.parse(JSON.stringify(baseTemplate));

    // Apply variant-specific transformations
    variantTemplate = this.applyVariantMeta(variantTemplate, variantConfig, options);
    variantTemplate = this.applyVariantOptimizations(variantTemplate, variantConfig, options);
    variantTemplate = this.mergeVariantLogic(variantTemplate, variantConfig);
    variantTemplate = this.applyContextualConditions(variantTemplate, variantConfig, options);

    // Store optimization history for tracking
    this.recordOptimizationHistory(variantType, variantConfig.optimizations);

    return variantTemplate;
  }

  /**
   * Apply variant-specific metadata
   */
  private applyVariantMeta(
    template: RTBTemplate,
    config: VariantConfig,
    options: VariantGenerationOptions
  ): RTBTemplate {
    if (!template.meta) {
      template.meta = {
        version: '1.0.0',
        author: ['VariantGenerator'],
        description: '',
        priority: config.priority
      };
    }

    template.meta.priority = config.priority;
    template.meta.tags = [
      ...(template.meta.tags || []),
      `variant-${config.variantType}`,
      `priority-${config.priority}`,
      `auto-generated`
    ];

    if (options.targetEnvironment) {
      template.meta.environment = options.targetEnvironment;
    }

    template.meta.description = `${config.variantType} variant template with ${config.optimizations.length} optimizations`;
    template.meta.inherits_from = config.baseTemplates;

    return template;
  }

  /**
   * Apply variant-specific optimizations
   */
  private applyVariantOptimizations(
    template: RTBTemplate,
    config: VariantConfig,
    options: VariantGenerationOptions
  ): RTBTemplate {
    config.optimizations.forEach(optimization => {
      if (this.shouldApplyOptimization(optimization, options)) {
        this.setNestedProperty(template.configuration, optimization.parameter, optimization.value);
      }
    });

    // Apply custom overrides if provided
    if (options.customOverrides) {
      Object.entries(options.customOverrides).forEach(([param, value]) => {
        this.setNestedProperty(template.configuration, param, value);
      });
    }

    return template;
  }

  /**
   * Merge variant-specific custom functions and logic
   */
  private mergeVariantLogic(template: RTBTemplate, config: VariantConfig): RTBTemplate {
    // Merge custom functions
    if (!template.custom) {
      template.custom = [];
    }

    config.customLogic.forEach(func => {
      const existingIndex = template.custom!.findIndex(f => f.name === func.name);
      if (existingIndex >= 0) {
        template.custom![existingIndex] = func;
      } else {
        template.custom!.push(func);
      }
    });

    // Merge conditions
    if (!template.conditions) {
      template.conditions = {};
    }

    Object.entries(config.conditions).forEach(([key, condition]) => {
      template.conditions![key] = condition;
    });

    // Merge evaluations
    if (!template.evaluations) {
      template.evaluations = {};
    }

    Object.entries(config.evaluations).forEach(([key, evaluation]) => {
      template.evaluations![key] = evaluation;
    });

    return template;
  }

  /**
   * Apply contextual conditions based on options
   */
  private applyContextualConditions(
    template: RTBTemplate,
    config: VariantConfig,
    options: VariantGenerationOptions
  ): RTBTemplate {
    // Add contextual conditions based on options
    if (options.cellCount) {
      template.conditions![`cellCount_${config.variantType}`] = {
        if: `cellCount > ${options.cellCount * 0.8}`,
        then: { scalabilityMode: 'high' },
        else: { scalabilityMode: 'standard' }
      };
    }

    if (options.trafficProfile) {
      template.evaluations![`trafficProfile_${config.variantType}`] = {
        eval: `adjustTrafficProfile('${options.trafficProfile}')`,
        args: [options.trafficProfile]
      };
    }

    if (options.energyMode) {
      template.conditions![`energyMode_${config.variantType}`] = {
        if: `energyMode === '${options.energyMode}'`,
        then: { powerOptimization: true },
        else: { powerOptimization: false }
      };
    }

    return template;
  }

  /**
   * Check if an optimization should be applied based on conditions
   */
  private shouldApplyOptimization(optimization: VariantOptimization, options: VariantGenerationOptions): boolean {
    if (!optimization.conditions || optimization.conditions.length === 0) {
      return true;
    }

    return optimization.conditions.every(condition => {
      // Simple condition evaluation - can be extended for complex logic
      if (condition.includes('trafficProfile')) {
        return options.trafficProfile && condition.includes(options.trafficProfile);
      }
      if (condition.includes('energyMode')) {
        return options.energyMode && condition.includes(options.energyMode);
      }
      if (condition.includes('cellCount')) {
        const match = condition.match(/cellCount\s*([><=]+)\s*(\d+)/);
        if (match && options.cellCount) {
          const operator = match[1];
          const value = parseInt(match[2]);
          switch (operator) {
            case '>': return options.cellCount > value;
            case '<': return options.cellCount < value;
            case '>=': return options.cellCount >= value;
            case '<=': return options.cellCount <= value;
            case '==': return options.cellCount === value;
            default: return false;
          }
        }
      }
      return true; // Default to apply if condition can't be evaluated
    });
  }

  /**
   * Set nested property using dot notation
   */
  private setNestedProperty(obj: any, path: string, value: any): void {
    const keys = path.split('.');
    let current = obj;

    for (let i = 0; i < keys.length - 1; i++) {
      if (!(keys[i] in current)) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }

    current[keys[keys.length - 1]] = value;
  }

  /**
   * Record optimization history for tracking and learning
   */
  private recordOptimizationHistory(variantType: string, optimizations: VariantOptimization[]): void {
    if (!this.optimizationHistory.has(variantType)) {
      this.optimizationHistory.set(variantType, []);
    }

    const history = this.optimizationHistory.get(variantType)!;
    history.push(...optimizations);

    // Keep only last 1000 optimizations to prevent memory bloat
    if (history.length > 1000) {
      history.splice(0, history.length - 1000);
    }
  }

  /**
   * Get optimization statistics for learning
   */
  getOptimizationStats(variantType?: string): Record<string, any> {
    const stats: Record<string, any> = {};

    if (variantType) {
      const optimizations = this.optimizationHistory.get(variantType) || [];
      stats[variantType] = this.calculateStats(optimizations);
    } else {
      this.optimizationHistory.forEach((optimizations, type) => {
        stats[type] = this.calculateStats(optimizations);
      });
    }

    return stats;
  }

  private calculateStats(optimizations: VariantOptimization[]): Record<string, any> {
    const paramFrequency: Record<string, number> = {};
    const contextFrequency: Record<string, number> = {};

    optimizations.forEach(opt => {
      paramFrequency[opt.parameter] = (paramFrequency[opt.parameter] || 0) + 1;
      contextFrequency[opt.context] = (contextFrequency[opt.context] || 0) + 1;
    });

    return {
      totalOptimizations: optimizations.length,
      mostUsedParameters: Object.entries(paramFrequency)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 10)
        .map(([param, count]) => ({ parameter: param, frequency: count })),
      mostCommonContexts: Object.entries(contextFrequency)
        .sort(([,a], [,b]) => b - a)
        .slice(0, 5)
        .map(([context, count]) => ({ context, frequency: count }))
    };
  }

  /**
   * Initialize default variant configurations
   */
  private initializeDefaultVariantConfigs(): void {
    // Placeholder - will be overridden by specific variant configs
    this.variantConfigs.set('urban', {
      variantType: 'urban',
      priority: 20,
      baseTemplates: ['base_4g', 'base_5g'],
      optimizations: [],
      customLogic: [],
      conditions: {},
      evaluations: {}
    });

    this.variantConfigs.set('mobility', {
      variantType: 'mobility',
      priority: 30,
      baseTemplates: ['base_4g', 'base_5g'],
      optimizations: [],
      customLogic: [],
      conditions: {},
      evaluations: {}
    });

    this.variantConfigs.set('sleep', {
      variantType: 'sleep',
      priority: 40,
      baseTemplates: ['base_4g', 'base_5g'],
      optimizations: [],
      customLogic: [],
      conditions: {},
      evaluations: {}
    });
  }

  /**
   * Generate batch variants for multiple scenarios
   */
  generateBatchVariants(
    variantTypes: string[],
    baseTemplateName: string,
    options?: VariantGenerationOptions
  ): Record<string, RTBTemplate> {
    const results: Record<string, RTBTemplate> = {};

    variantTypes.forEach(variantType => {
      try {
        results[variantType] = this.generateVariant(variantType, baseTemplateName, options);
      } catch (error) {
        console.error(`Failed to generate ${variantType} variant:`, error);
      }
    });

    return results;
  }

  /**
   * Validate generated variant template
   */
  validateVariant(template: RTBTemplate): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check required metadata
    if (!template.meta) {
      errors.push('Missing template metadata');
    } else {
      if (!template.meta.priority) {
        errors.push('Missing template priority');
      }
      if (!template.meta.description) {
        errors.push('Missing template description');
      }
    }

    // Check configuration structure
    if (!template.configuration || Object.keys(template.configuration).length === 0) {
      errors.push('Template configuration is empty');
    }

    // Validate custom functions
    if (template.custom) {
      template.custom.forEach((func, index) => {
        if (!func.name || !func.body || func.body.length === 0) {
          errors.push(`Invalid custom function at index ${index}`);
        }
      });
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}