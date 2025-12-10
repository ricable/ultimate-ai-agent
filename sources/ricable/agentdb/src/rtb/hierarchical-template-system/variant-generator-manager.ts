/**
 * Template Variant Generator Manager
 *
 * Central orchestrator for all template variant generators. Provides unified interface
 * for generating, managing, and optimizing RAN deployment variants across different
 * scenarios and contexts.
 */

import { VariantGeneratorCore, VariantGenerationOptions } from './variant-generator-core';
import { UrbanVariantGenerator, UrbanDeploymentContext } from './variant-generators/urban-variant';
import { MobilityVariantGenerator, MobilityDeploymentContext } from './variant-generators/mobility-variant';
import { SleepVariantGenerator, SleepModeContext } from './variant-generators/sleep-variant';
import { RTBTemplate } from '../../types/rtb-types';

export interface CombinedDeploymentContext {
  primaryScenario: 'urban' | 'mobility' | 'sleep' | 'hybrid';
  urbanContext?: UrbanDeploymentContext;
  mobilityContext?: MobilityDeploymentContext;
  sleepContext?: SleepModeContext;
  hybridConfig?: {
    scenarios: string[];
    weights: Record<string, number>;
    transitionRules: Record<string, string>;
  };
  globalSettings: {
    cellCount: number;
    trafficProfile: 'low' | 'medium' | 'high';
    energyMode: 'performance' | 'balanced' | 'energy_saving';
    targetEnvironment: string;
    customOverrides: Record<string, any>;
  };
}

export interface VariantGenerationResult {
  template: RTBTemplate;
  metadata: {
    variantType: string;
    priority: number;
    generationTime: number;
    optimizationsApplied: number;
    customFunctionsGenerated: number;
    conditionsAdded: number;
  };
  validation: {
    valid: boolean;
    errors: string[];
    warnings: string[];
  };
  performance: {
    estimatedCapacityImprovement: number;
    estimatedEnergySavings: number;
    estimatedLatencyImprovement: number;
  };
}

export class VariantGeneratorManager {
  private urbanGenerator: UrbanVariantGenerator;
  private mobilityGenerator: MobilityVariantGenerator;
  private sleepGenerator: SleepVariantGenerator;
  private coreGenerator: VariantGeneratorCore;

  // Performance tracking
  private generationHistory: Array<{
    timestamp: Date;
    variantType: string;
    context: any;
    generationTime: number;
    success: boolean;
  }> = [];

  constructor() {
    this.coreGenerator = new VariantGeneratorCore();
    this.urbanGenerator = new UrbanVariantGenerator();
    this.mobilityGenerator = new MobilityVariantGenerator();
    this.sleepGenerator = new SleepVariantGenerator();
  }

  /**
   * Generate a single variant based on deployment context
   */
  async generateVariant(
    baseTemplateName: string,
    context: CombinedDeploymentContext
  ): Promise<VariantGenerationResult> {
    const startTime = Date.now();
    let template: RTBTemplate;
    let variantType: string;

    try {
      switch (context.primaryScenario) {
        case 'urban':
          if (!context.urbanContext) {
            throw new Error('Urban context required for urban variant generation');
          }
          template = this.urbanGenerator.generateUrbanVariant(
            baseTemplateName,
            context.urbanContext,
            context.globalSettings
          );
          variantType = 'urban';
          break;

        case 'mobility':
          if (!context.mobilityContext) {
            throw new Error('Mobility context required for mobility variant generation');
          }
          template = this.mobilityGenerator.generateMobilityVariant(
            baseTemplateName,
            context.mobilityContext,
            context.globalSettings
          );
          variantType = 'mobility';
          break;

        case 'sleep':
          if (!context.sleepContext) {
            throw new Error('Sleep context required for sleep variant generation');
          }
          template = this.sleepGenerator.generateSleepVariant(
            baseTemplateName,
            context.sleepContext,
            context.globalSettings
          );
          variantType = 'sleep';
          break;

        case 'hybrid':
          if (!context.hybridConfig) {
            throw new Error('Hybrid configuration required for hybrid variant generation');
          }
          template = await this.generateHybridVariant(baseTemplateName, context);
          variantType = 'hybrid';
          break;

        default:
          throw new Error(`Unknown scenario type: ${context.primaryScenario}`);
      }

      const generationTime = Date.now() - startTime;
      const validation = this.validateTemplate(template);
      const performance = this.estimatePerformance(template, context);

      // Record generation history
      this.generationHistory.push({
        timestamp: new Date(),
        variantType,
        context,
        generationTime,
        success: validation.valid
      });

      return {
        template,
        metadata: {
          variantType,
          priority: template.meta?.priority || 0,
          generationTime,
          optimizationsApplied: this.countOptimizations(template),
          customFunctionsGenerated: template.custom?.length || 0,
          conditionsAdded: Object.keys(template.conditions || {}).length
        },
        validation,
        performance
      };

    } catch (error) {
      const generationTime = Date.now() - startTime;
      this.generationHistory.push({
        timestamp: new Date(),
        variantType: context.primaryScenario,
        context,
        generationTime,
        success: false
      });

      throw error;
    }
  }

  /**
   * Generate hybrid variant combining multiple scenarios
   */
  private async generateHybridVariant(
    baseTemplateName: string,
    context: CombinedDeploymentContext
  ): Promise<RTBTemplate> {
    if (!context.hybridConfig) {
      throw new Error('Hybrid configuration required');
    }

    const hybridTemplate: RTBTemplate = {
      meta: {
        version: '1.0.0',
        author: ['VariantGeneratorManager'],
        description: `Hybrid variant combining ${context.hybridConfig.scenarios.join(', ')}`,
        tags: ['hybrid', ...context.hybridConfig.scenarios],
        priority: 25, // Mid-priority for hybrid variants
        inherits_from: context.hybridConfig.scenarios
      },
      custom: [],
      configuration: {},
      conditions: {},
      evaluations: {}
    };

    // Generate individual variants
    const individualVariants: RTBTemplate[] = [];
    for (const scenario of context.hybridConfig.scenarios) {
      let variantTemplate: RTBTemplate;

      switch (scenario) {
        case 'urban':
          if (!context.urbanContext) continue;
          variantTemplate = this.urbanGenerator.generateUrbanVariant(
            baseTemplateName,
            context.urbanContext,
            context.globalSettings
          );
          break;

        case 'mobility':
          if (!context.mobilityContext) continue;
          variantTemplate = this.mobilityGenerator.generateMobilityVariant(
            baseTemplateName,
            context.mobilityContext,
            context.globalSettings
          );
          break;

        case 'sleep':
          if (!context.sleepContext) continue;
          variantTemplate = this.sleepGenerator.generateSleepVariant(
            baseTemplateName,
            context.sleepContext,
            context.globalSettings
          );
          break;

        default:
          continue;
      }

      individualVariants.push(variantTemplate);
    }

    // Merge variants based on weights
    return this.mergeVariants(individualVariants, context.hybridConfig.weights, hybridTemplate);
  }

  /**
   * Merge multiple variants with weighted configuration
   */
  private mergeVariants(
    variants: RTBTemplate[],
    weights: Record<string, number>,
    baseTemplate: RTBTemplate
  ): RTBTemplate {
    const mergedTemplate: RTBTemplate = { ...baseTemplate };

    // Merge configurations with weights
    const mergedConfiguration: Record<string, any> = {};
    const weightSum = Object.values(weights).reduce((sum, weight) => sum + weight, 0);

    variants.forEach((variant, index) => {
      const scenarioType = Object.keys(weights)[index] || 'default';
      const weight = weights[scenarioType] / weightSum;

      this.mergeConfigurationWeighted(
        mergedConfiguration,
        variant.configuration,
        weight
      );

      // Merge custom functions
      if (variant.custom) {
        mergedTemplate.custom = [
          ...(mergedTemplate.custom || []),
          ...variant.custom.map(func => ({
            ...func,
            name: `${func.name}_${scenarioType}`
          }))
        ];
      }

      // Merge conditions
      if (variant.conditions) {
        Object.entries(variant.conditions).forEach(([key, condition]) => {
          const weightedKey = `${key}_${scenarioType}`;
          mergedTemplate.conditions![weightedKey] = condition;
        });
      }

      // Merge evaluations
      if (variant.evaluations) {
        Object.entries(variant.evaluations).forEach(([key, evaluation]) => {
          const weightedKey = `${key}_${scenarioType}`;
          mergedTemplate.evaluations![weightedKey] = evaluation;
        });
      }
    });

    mergedTemplate.configuration = mergedConfiguration;

    return mergedTemplate;
  }

  /**
   * Merge configuration objects with weights
   */
  private mergeConfigurationWeighted(
    target: Record<string, any>,
    source: Record<string, any>,
    weight: number
  ): void {
    Object.entries(source).forEach(([key, value]) => {
      if (typeof value === 'number' && typeof target[key] === 'number') {
        // Weighted average for numeric values
        target[key] = target[key] * (1 - weight) + value * weight;
      } else if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        // Recursive merge for nested objects
        if (!target[key] || typeof target[key] !== 'object') {
          target[key] = {};
        }
        this.mergeConfigurationWeighted(target[key], value, weight);
      } else if (weight > 0.5) {
        // For non-numeric values, use the higher weight value
        target[key] = value;
      } else if (!target[key]) {
        // Use lower weight value only if target doesn't exist
        target[key] = value;
      }
    });
  }

  /**
   * Generate batch variants for multiple scenarios
   */
  async generateBatchVariants(
    baseTemplateName: string,
    contexts: CombinedDeploymentContext[]
  ): Promise<Record<string, VariantGenerationResult>> {
    const results: Record<string, VariantGenerationResult> = {};

    const generationPromises = contexts.map(async (context, index) => {
      const scenarioKey = `${context.primaryScenario}_${index}`;
      try {
        results[scenarioKey] = await this.generateVariant(baseTemplateName, context);
      } catch (error) {
        results[scenarioKey] = {
          template: {} as RTBTemplate,
          metadata: {
            variantType: context.primaryScenario,
            priority: 0,
            generationTime: 0,
            optimizationsApplied: 0,
            customFunctionsGenerated: 0,
            conditionsAdded: 0
          },
          validation: {
            valid: false,
            errors: [error instanceof Error ? error.message : 'Unknown error'],
            warnings: []
          },
          performance: {
            estimatedCapacityImprovement: 0,
            estimatedEnergySavings: 0,
            estimatedLatencyImprovement: 0
          }
        };
      }
    });

    await Promise.all(generationPromises);
    return results;
  }

  /**
   * Validate generated template
   */
  private validateTemplate(template: RTBTemplate): { valid: boolean; errors: string[]; warnings: string[] } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Basic validation
    if (!template.meta) {
      errors.push('Missing template metadata');
    } else {
      if (!template.meta.version) {
        errors.push('Missing template version');
      }
      if (!template.meta.description) {
        warnings.push('Missing template description');
      }
      if (!template.meta.priority && template.meta.priority !== 0) {
        warnings.push('Missing template priority');
      }
    }

    // Configuration validation
    if (!template.configuration || Object.keys(template.configuration).length === 0) {
      errors.push('Template configuration is empty');
    }

    // Custom function validation
    if (template.custom) {
      template.custom.forEach((func, index) => {
        if (!func.name) {
          errors.push(`Custom function at index ${index} missing name`);
        }
        if (!func.body || func.body.length === 0) {
          errors.push(`Custom function '${func.name}' has empty body`);
        }
      });
    }

    // Condition validation
    if (template.conditions) {
      Object.entries(template.conditions).forEach(([key, condition]) => {
        if (!condition.if) {
          errors.push(`Condition '${key}' missing 'if' clause`);
        }
        if (!condition.then) {
          errors.push(`Condition '${key}' missing 'then' clause`);
        }
      });
    }

    // Evaluation validation
    if (template.evaluations) {
      Object.entries(template.evaluations).forEach(([key, evaluation]) => {
        if (!evaluation.eval) {
          errors.push(`Evaluation '${key}' missing 'eval' expression`);
        }
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Estimate performance improvements
   */
  private estimatePerformance(
    template: RTBTemplate,
    context: CombinedDeploymentContext
  ): {
    estimatedCapacityImprovement: number;
    estimatedEnergySavings: number;
    estimatedLatencyImprovement: number;
  } {
    let capacityImprovement = 0;
    let energySavings = 0;
    let latencyImprovement = 0;

    switch (context.primaryScenario) {
      case 'urban':
        capacityImprovement = 150; // 150% capacity improvement
        energySavings = -20; // 20% more energy consumption
        latencyImprovement = 30; // 30% latency improvement
        break;

      case 'mobility':
        capacityImprovement = 20;
        energySavings = 10;
        latencyImprovement = 40;
        break;

      case 'sleep':
        capacityImprovement = -60; // 60% capacity reduction
        energySavings = 70; // 70% energy savings
        latencyImprovement = -20; // 20% latency increase
        break;

      case 'hybrid':
        // Weighted average based on hybrid configuration
        if (context.hybridConfig) {
          const weights = context.hybridConfig.weights;
          const totalWeight = Object.values(weights).reduce((sum, w) => sum + w, 0);

          Object.entries(weights).forEach(([scenario, weight]) => {
            const normalizedWeight = weight / totalWeight;
            switch (scenario) {
              case 'urban':
                capacityImprovement += 150 * normalizedWeight;
                energySavings += -20 * normalizedWeight;
                latencyImprovement += 30 * normalizedWeight;
                break;
              case 'mobility':
                capacityImprovement += 20 * normalizedWeight;
                energySavings += 10 * normalizedWeight;
                latencyImprovement += 40 * normalizedWeight;
                break;
              case 'sleep':
                capacityImprovement += -60 * normalizedWeight;
                energySavings += 70 * normalizedWeight;
                latencyImprovement += -20 * normalizedWeight;
                break;
            }
          });
        }
        break;
    }

    return {
      estimatedCapacityImprovement: Math.round(capacityImprovement),
      estimatedEnergySavings: Math.round(energySavings),
      estimatedLatencyImprovement: Math.round(latencyImprovement)
    };
  }

  /**
   * Count optimizations in template
   */
  private countOptimizations(template: RTBTemplate): number {
    let count = 0;

    // Count configuration parameters
    if (template.configuration) {
      count += this.countNestedProperties(template.configuration);
    }

    // Count custom functions
    if (template.custom) {
      count += template.custom.length;
    }

    // Count conditions
    if (template.conditions) {
      count += Object.keys(template.conditions).length;
    }

    // Count evaluations
    if (template.evaluations) {
      count += Object.keys(template.evaluations).length;
    }

    return count;
  }

  /**
   * Count nested properties recursively
   */
  private countNestedProperties(obj: any): number {
    let count = 0;
    for (const key in obj) {
      count++;
      if (typeof obj[key] === 'object' && obj[key] !== null && !Array.isArray(obj[key])) {
        count += this.countNestedProperties(obj[key]);
      }
    }
    return count;
  }

  /**
   * Get generation statistics
   */
  getGenerationStatistics(): {
    totalGenerations: number;
    successRate: number;
    averageGenerationTime: number;
    variantTypeDistribution: Record<string, number>;
    recentGenerations: Array<{
      timestamp: Date;
      variantType: string;
      generationTime: number;
      success: boolean;
    }>;
  } {
    const totalGenerations = this.generationHistory.length;
    const successfulGenerations = this.generationHistory.filter(g => g.success).length;
    const successRate = totalGenerations > 0 ? (successfulGenerations / totalGenerations) * 100 : 0;

    const averageGenerationTime = totalGenerations > 0
      ? this.generationHistory.reduce((sum, g) => sum + g.generationTime, 0) / totalGenerations
      : 0;

    const variantTypeDistribution: Record<string, number> = {};
    this.generationHistory.forEach(g => {
      variantTypeDistribution[g.variantType] = (variantTypeDistribution[g.variantType] || 0) + 1;
    });

    const recentGenerations = this.generationHistory
      .slice(-10)
      .map(g => ({
        timestamp: g.timestamp,
        variantType: g.variantType,
        generationTime: g.generationTime,
        success: g.success
      }));

    return {
      totalGenerations,
      successRate: Math.round(successRate * 100) / 100,
      averageGenerationTime: Math.round(averageGenerationTime * 100) / 100,
      variantTypeDistribution,
      recentGenerations
    };
  }

  /**
   * Get recommendations for variant selection
   */
  getRecommendations(context: Partial<CombinedDeploymentContext>): string[] {
    const recommendations: string[] = [];

    // Time-based recommendations
    const currentHour = new Date().getHours();
    if (currentHour >= 1 && currentHour <= 5) {
      recommendations.push('Consider sleep mode variant for night-time energy savings');
    } else if (currentHour >= 7 && currentHour <= 9 || currentHour >= 17 && currentHour <= 19) {
      recommendations.push('Consider mobility variant for peak traffic hours');
    }

    // Traffic-based recommendations
    if (context.globalSettings?.trafficProfile === 'high') {
      recommendations.push('Urban variant recommended for high traffic density');
    } else if (context.globalSettings?.trafficProfile === 'low') {
      recommendations.push('Sleep mode variant recommended for low traffic periods');
    }

    // Energy-based recommendations
    if (context.globalSettings?.energyMode === 'energy_saving') {
      recommendations.push('Sleep mode variant recommended for energy optimization');
    } else if (context.globalSettings?.energyMode === 'performance') {
      recommendations.push('Urban variant recommended for maximum performance');
    }

    // Cell count recommendations
    if (context.globalSettings?.cellCount && context.globalSettings.cellCount > 100) {
      recommendations.push('Urban variant with massive MIMO recommended for large deployments');
    } else if (context.globalSettings?.cellCount && context.globalSettings.cellCount < 20) {
      recommendations.push('Consider hybrid variant for small deployments');
    }

    return recommendations;
  }

  /**
   * Export generation history for analysis
   */
  exportGenerationHistory(): Array<{
    timestamp: Date;
    variantType: string;
    generationTime: number;
    success: boolean;
  }> {
    return [...this.generationHistory];
  }

  /**
   * Clear generation history
   */
  clearGenerationHistory(): void {
    this.generationHistory = [];
  }
}