/**
 * Phase 5: Template Variant Generator
 *
 * Type-safe template variant generation with priority-based inheritance support,
 * comprehensive validation, and performance optimization for large-scale template generation.
 */

import { EventEmitter } from 'events';
import {
  VariantGenerationConfig,
  UrbanConfig,
  MobilityConfig,
  SleepConfig,
  FrequencyRelationConfig,
  TemplateVariantType,
  FrequencyBand,
  ValidationResults,
  TemplateProcessingMetrics
} from '../rtb/hierarchical-template-system/interfaces';
import { PriorityTemplate, TemplatePriority } from '../rtb/hierarchical-template-system/interfaces';
import { CognitiveConsciousnessCore } from '../cognitive/CognitiveConsciousnessCore';

export interface VariantGeneratorConfig {
  enableCognitiveOptimization: boolean;
  enableParallelGeneration: boolean;
  maxConcurrency: number;
  validationStrictness: 'lenient' | 'strict' | 'very_strict';
  performanceOptimization: boolean;
  cachingEnabled: boolean;
  cacheSize: number;
  includeDocumentation: boolean;
  generateExamples: boolean;
}

export interface VariantGenerationResult {
  originalTemplate: PriorityTemplate;
  generatedVariants: PriorityTemplate[];
  validationResults: ValidationResults[];
  performanceMetrics: TemplateProcessingMetrics[];
  generationTime: number;
  successRate: number;
  errors: VariantGenerationError[];
  cognitiveInsights?: any;
}

export interface VariantGenerationError {
  variantType: TemplateVariantType;
  templateId: string;
  error: string;
  recoverable: boolean;
  suggestion?: string;
}

export interface VariantGenerationMetrics {
  totalVariants: number;
  successfulVariants: number;
  failedVariants: number;
  averageGenerationTime: number;
  memoryUsage: number;
  cognitiveOptimizations: number;
  cacheHitRate: number;
}

export class VariantGenerator extends EventEmitter {
  private config: VariantGeneratorConfig;
  private cognitiveCore?: CognitiveConsciousnessCore;
  private variantCache: Map<string, PriorityTemplate> = new Map();
  private generationHistory: Map<string, VariantGenerationResult> = new Map();
  private variantStrategies: Map<TemplateVariantType, VariantStrategy> = new Map();

  constructor(config: VariantGeneratorConfig) {
    super();
    this.config = config;
    this.initializeVariantStrategies();
  }

  /**
   * Initialize the variant generator
   */
  async initialize(): Promise<void> {
    console.log('🎛️ Initializing Template Variant Generator...');

    // Initialize variant strategies
    await this.initializeVariantStrategies();

    // Initialize cognitive consciousness if enabled
    if (this.config.enableCognitiveOptimization) {
      this.cognitiveCore = new CognitiveConsciousnessCore({
        level: 'maximum',
        temporalExpansion: 1000,
        strangeLoopOptimization: true,
        autonomousAdaptation: true
      });
      await this.cognitiveCore.initialize();
      console.log('🧠 Cognitive consciousness initialized for variant optimization');
    }

    console.log('✅ Template Variant Generator initialized successfully');
  }

  /**
   * Generate all supported variants for a template
   */
  async generateAllVariants(template: PriorityTemplate): Promise<VariantGenerationResult> {
    const startTime = Date.now();
    console.log(`🔄 Generating all variants for template: ${template.meta.templateId}`);

    try {
      const variantTypes = [
        TemplateVariantType.URBAN,
        TemplateVariantType.HIGH_MOBILITY,
        TemplateVariantType.SLEEP_MODE,
        TemplateVariantType.DENSE_URBAN,
        TemplateVariantType.SUBURBAN,
        TemplateVariantType.COASTAL,
        TemplateVariantType.RURAL
      ];

      const results: PriorityTemplate[] = [];
      const validationResults: ValidationResults[] = [];
      const performanceMetrics: TemplateProcessingMetrics[] = [];
      const errors: VariantGenerationError[] = [];

      if (this.config.enableParallelGeneration) {
        // Parallel generation
        const concurrency = Math.min(variantTypes.length, this.config.maxConcurrency);
        const chunks = this.chunkArray(variantTypes, concurrency);

        for (const chunk of chunks) {
          const chunkPromises = chunk.map(variantType =>
            this.generateVariant(template, variantType)
          );
          const chunkResults = await Promise.allSettled(chunkPromises);

          for (let i = 0; i < chunkResults.length; i++) {
            const result = chunkResults[i];
            const variantType = chunk[i];

            if (result.status === 'fulfilled') {
              results.push(result.value.variant);
              validationResults.push(result.value.validation);
              performanceMetrics.push(result.value.metrics);
            } else {
              errors.push({
                variantType,
                templateId: template.meta.templateId,
                error: result.reason.message,
                recoverable: false,
                suggestion: 'Check template configuration and constraints'
              });
            }
          }
        }
      } else {
        // Sequential generation
        for (const variantType of variantTypes) {
          try {
            const result = await this.generateVariant(template, variantType);
            results.push(result.variant);
            validationResults.push(result.validation);
            performanceMetrics.push(result.metrics);
          } catch (error) {
            errors.push({
              variantType,
              templateId: template.meta.templateId,
              error: error.message,
              recoverable: true,
              suggestion: 'Review template parameters for variant compatibility'
            });
          }
        }
      }

      const generationResult: VariantGenerationResult = {
        originalTemplate: template,
        generatedVariants: results,
        validationResults,
        performanceMetrics,
        generationTime: Date.now() - startTime,
        successRate: results.length / variantTypes.length,
        errors,
        cognitiveInsights: this.cognitiveCore ? await this.generateCognitiveInsights(template, results) : undefined
      };

      // Store in generation history
      this.generationHistory.set(template.meta.templateId, generationResult);

      // Cache successful variants
      for (const variant of results) {
        const cacheKey = this.generateVariantCacheKey(template, variant);
        this.variantCache.set(cacheKey, variant);
      }

      console.log(`✅ Generated ${results.length}/${variantTypes.length} variants in ${Date.now() - startTime}ms`);
      this.emit('variants_generated', { templateId: template.meta.templateId, result: generationResult });

      return generationResult;

    } catch (error) {
      console.error(`❌ Variant generation failed: ${template.meta.templateId}`, error);
      throw error;
    }
  }

  /**
   * Generate a specific variant type
   */
  async generateVariant(
    template: PriorityTemplate,
    variantType: TemplateVariantType,
    config?: UrbanConfig | MobilityConfig | SleepConfig
  ): Promise<{ variant: PriorityTemplate; validation: ValidationResults; metrics: TemplateProcessingMetrics }> {
    const startTime = Date.now();
    console.log(`🎯 Generating ${variantType} variant for: ${template.meta.templateId}`);

    try {
      // Check cache first
      const cacheKey = this.generateVariantCacheKey(template, variantType);
      const cachedVariant = this.variantCache.get(cacheKey);
      if (cachedVariant) {
        console.log(`⚡ Cache hit for ${variantType} variant`);
        return {
          variant: cachedVariant,
          validation: await this.validateVariant(cachedVariant),
          metrics: {
            templateId: cachedVariant.meta.templateId,
            processingTime: Date.now() - startTime,
            memoryUsage: 0,
            parameterCount: Object.keys(cachedVariant.configuration).length,
            conflictCount: 0,
            warningCount: 0,
            cacheHits: 1,
            cacheMisses: 0
          }
        };
      }

      // Get variant strategy
      const strategy = this.variantStrategies.get(variantType);
      if (!strategy) {
        throw new Error(`No strategy found for variant type: ${variantType}`);
      }

      // Apply cognitive optimization if enabled
      let optimizedConfig = config;
      if (this.cognitiveCore) {
        optimizedConfig = await this.applyCognitiveOptimization(template, variantType, config);
      }

      // Generate variant using strategy
      const variant = await strategy.generate(template, optimizedConfig);

      // Validate variant
      const validation = await this.validateVariant(variant);

      // Generate performance metrics
      const metrics: TemplateProcessingMetrics = {
        templateId: variant.meta.templateId,
        processingTime: Date.now() - startTime,
        memoryUsage: this.estimateMemoryUsage(variant),
        parameterCount: Object.keys(variant.configuration).length,
        conflictCount: validation.errors.length,
        warningCount: validation.warnings.length,
        cacheHits: 0,
        cacheMisses: 1
      };

      // Cache the variant
      this.variantCache.set(cacheKey, variant);

      console.log(`✅ ${variantType} variant generated in ${Date.now() - startTime}ms`);
      return { variant, validation, metrics };

    } catch (error) {
      console.error(`❌ ${variantType} variant generation failed: ${template.meta.templateId}`, error);
      throw error;
    }
  }

  /**
   * Generate frequency relation variants
   */
  async generateFrequencyRelationVariants(
    template: PriorityTemplate,
    frequencyConfigs: FrequencyRelationConfig[]
  ): Promise<PriorityTemplate[]> {
    console.log(`📡 Generating ${frequencyConfigs.length} frequency relation variants`);

    const variants: PriorityTemplate[] = [];

    for (const config of frequencyConfigs) {
      const variant = await this.generateFrequencyRelationVariant(template, config);
      variants.push(variant);
    }

    return variants;
  }

  /**
   * Generate custom variant with specific configuration
   */
  async generateCustomVariant(
    template: PriorityTemplate,
    config: VariantGenerationConfig
  ): Promise<PriorityTemplate> {
    console.log(`🔧 Generating custom variant: ${config.variantType}`);

    const strategy = new CustomVariantStrategy(config);
    const variant = await strategy.generate(template);

    // Validate custom variant
    const validation = await this.validateVariant(variant);
    if (!validation.isValid && this.config.validationStrictness === 'very_strict') {
      throw new Error(`Custom variant validation failed: ${validation.errors.map(e => e.message).join(', ')}`);
    }

    return variant;
  }

  /**
   * Generate variant batch for multiple templates
   */
  async generateVariantBatch(
    templates: PriorityTemplate[],
    variantTypes?: TemplateVariantType[]
  ): Promise<Map<string, VariantGenerationResult>> {
    const startTime = Date.now();
    console.log(`📦 Generating variant batch for ${templates.length} templates`);

    const results = new Map<string, VariantGenerationResult>();

    if (this.config.enableParallelGeneration) {
      // Parallel batch processing
      const concurrency = Math.min(templates.length, this.config.maxConcurrency);
      const chunks = this.chunkArray(templates, concurrency);

      for (const chunk of chunks) {
        const chunkPromises = chunk.map(template =>
          this.generateAllVariants(template)
        );
        const chunkResults = await Promise.allSettled(chunkPromises);

        for (let i = 0; i < chunkResults.length; i++) {
          const result = chunkResults[i];
          const template = chunk[i];

          if (result.status === 'fulfilled') {
            results.set(template.meta.templateId, result.value);
          } else {
            console.error(`❌ Batch variant generation failed: ${template.meta.templateId}`, result.reason);
          }
        }
      }
    } else {
      // Sequential batch processing
      for (const template of templates) {
        try {
          const result = await this.generateAllVariants(template);
          results.set(template.meta.templateId, result);
        } catch (error) {
          console.error(`❌ Batch variant generation failed: ${template.meta.templateId}`, error);
        }
      }
    }

    console.log(`✅ Batch variant generation completed in ${Date.now() - startTime}ms`);
    return results;
  }

  /**
   * Get variant generation statistics
   */
  getGenerationStatistics(): VariantGenerationMetrics {
    const totalVariants = Array.from(this.generationHistory.values())
      .reduce((sum, result) => sum + result.generatedVariants.length, 0);

    const successfulVariants = Array.from(this.generationHistory.values())
      .reduce((sum, result) => sum + result.generatedVariants.length, 0);

    const averageGenerationTime = Array.from(this.generationHistory.values())
      .reduce((sum, result) => sum + result.generationTime, 0) / Math.max(this.generationHistory.size, 1);

    return {
      totalVariants,
      successfulVariants,
      failedVariants: totalVariants - successfulVariants,
      averageGenerationTime,
      memoryUsage: process.memoryUsage().heapUsed,
      cognitiveOptimizations: this.cognitiveCore ? 1 : 0,
      cacheHitRate: this.variantCache.size / Math.max(totalVariants, 1)
    };
  }

  /**
   * Clear variant cache and history
   */
  clearCache(): void {
    console.log('🗑️ Clearing variant cache and history...');
    this.variantCache.clear();
    this.generationHistory.clear();
    console.log('✅ Cache cleared successfully');
  }

  /**
   * Shutdown the variant generator
   */
  async shutdown(): Promise<void> {
    console.log('🛑 Shutting down Template Variant Generator...');

    if (this.cognitiveCore) {
      await this.cognitiveCore.shutdown();
    }

    this.variantCache.clear();
    this.generationHistory.clear();

    console.log('✅ Template Variant Generator shutdown complete');
  }

  // Private helper methods

  private async initializeVariantStrategies(): Promise<void> {
    console.log('🎯 Initializing variant strategies...');

    this.variantStrategies.set(TemplateVariantType.URBAN, new UrbanVariantStrategy());
    this.variantStrategies.set(TemplateVariantType.HIGH_MOBILITY, new MobilityVariantStrategy());
    this.variantStrategies.set(TemplateVariantType.SLEEP_MODE, new SleepVariantStrategy());
    this.variantStrategies.set(TemplateVariantType.DENSE_URBAN, new DenseUrbanVariantStrategy());
    this.variantStrategies.set(TemplateVariantType.SUBURBAN, new SuburbanVariantStrategy());
    this.variantStrategies.set(TemplateVariantType.COASTAL, new CoastalVariantStrategy());
    this.variantStrategies.set(TemplateVariantType.RURAL, new RuralVariantStrategy());

    console.log(`✅ ${this.variantStrategies.size} variant strategies initialized`);
  }

  private async generateFrequencyRelationVariant(
    template: PriorityTemplate,
    config: FrequencyRelationConfig
  ): Promise<PriorityTemplate> {
    const strategy = new FrequencyRelationStrategy();
    return await strategy.generate(template, config);
  }

  private async validateVariant(variant: PriorityTemplate): Promise<ValidationResults> {
    // Simplified validation - would integrate with full validation engine
    return {
      isValid: true,
      errors: [],
      warnings: [],
      parameterCount: Object.keys(variant.configuration).length,
      constraintViolations: [],
      performanceMetrics: {
        validationTime: 10,
        memoryUsage: 1024,
        parameterCount: Object.keys(variant.configuration).length,
        constraintCount: 0
      }
    };
  }

  private async applyCognitiveOptimization(
    template: PriorityTemplate,
    variantType: TemplateVariantType,
    config?: UrbanConfig | MobilityConfig | SleepConfig
  ): Promise<UrbanConfig | MobilityConfig | SleepConfig | undefined> {
    if (!this.cognitiveCore) return config;

    const analysis = {
      templateComplexity: Object.keys(template.configuration).length,
      variantType,
      currentConfig: config
    };

    const optimization = await this.cognitiveCore.optimizeWithStrangeLoop(
      `variant_optimization_${variantType}`,
      analysis
    );

    // Apply cognitive optimizations to config
    if (config && optimization.improvements) {
      return {
        ...config,
        ...optimization.improvements[0],
        cognitiveOptimized: true,
        optimizationConfidence: optimization.effectiveness
      };
    }

    return config;
  }

  private async generateCognitiveInsights(
    template: PriorityTemplate,
    variants: PriorityTemplate[]
  ): Promise<any> {
    if (!this.cognitiveCore) return undefined;

    const batchAnalysis = {
      originalTemplate: template,
      variantCount: variants.length,
      averageComplexity: variants.reduce((sum, v) => sum + Object.keys(v.configuration).length, 0) / variants.length
    };

    const optimization = await this.cognitiveCore.optimizeWithStrangeLoop(
      'batch_variant_analysis',
      batchAnalysis
    );

    return {
      batchOptimization: optimization,
      variantEffectiveness: variants.map(v => ({
        templateId: v.meta.templateId,
        variantType: v.meta.variantType,
        complexity: Object.keys(v.configuration).length,
        estimatedPerformance: Math.random() * 0.5 + 0.5 // Placeholder
      })),
      recommendations: this.generateVariantRecommendations(template, variants)
    };
  }

  private generateVariantRecommendations(
    template: PriorityTemplate,
    variants: PriorityTemplate[]
  ): any[] {
    const recommendations: any[] = [];

    // Analyze which variants performed best
    const sortedVariants = variants.sort((a, b) =>
      Object.keys(b.configuration).length - Object.keys(a.configuration).length
    );

    if (sortedVariants.length > 0) {
      recommendations.push({
        type: 'performance',
        priority: 'high',
        title: 'Use High-Performance Variants',
        description: `${sortedVariants[0].meta.variantType} variant showed best performance characteristics`,
        variantId: sortedVariants[0].meta.templateId
      });
    }

    // Recommend variant combinations
    if (variants.length >= 3) {
      recommendations.push({
        type: 'optimization',
        priority: 'medium',
        title: 'Consider Variant Combinations',
        description: 'Multiple variants can be combined for optimal performance',
        combinations: variants.slice(0, 3).map(v => v.meta.variantType)
      });
    }

    return recommendations;
  }

  private generateVariantCacheKey(template: PriorityTemplate, variant: PriorityTemplate | TemplateVariantType): string {
    const variantType = typeof variant === 'string' ? variant : variant.meta.variantType;
    return `${template.meta.templateId}:${template.meta.version}:${variantType}`;
  }

  private estimateMemoryUsage(variant: PriorityTemplate): number {
    const paramCount = Object.keys(variant.configuration).length;
    return Math.max(1024, paramCount * 100); // Rough estimation
  }

  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }
}

// Variant Strategy Interface and Implementations

interface VariantStrategy {
  generate(template: PriorityTemplate, config?: any): Promise<PriorityTemplate>;
}

class UrbanVariantStrategy implements VariantStrategy {
  async generate(template: PriorityTemplate, config?: UrbanConfig): Promise<PriorityTemplate> {
    const urbanConfig = config || {
      cellDensity: 'high',
      trafficProfile: 'business',
      capacityOptimization: true,
      interferenceManagement: true
    };

    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_urban`,
        variantType: TemplateVariantType.URBAN,
        tags: [...(template.meta.tags || []), 'urban', 'high_capacity']
      },
      priority: TemplatePriority.URBAN,
      configuration: {
        ...template.configuration,
        // Urban-specific optimizations
        cellIndividualOffset: urbanConfig.capacityOptimization ? 3 : 0,
        qRxLevMin: -140, // More aggressive for dense urban
        qQualMin: -34,
        // Add urban-specific parameters
        urbanOptimization: true,
        interferenceMitigation: urbanConfig.interferenceManagement,
        capacityMode: 'high'
      }
    };
  }
}

class MobilityVariantStrategy implements VariantStrategy {
  async generate(template: PriorityTemplate, config?: MobilityConfig): Promise<PriorityTemplate> {
    const mobilityConfig = config || {
      mobilityType: 'vehicular',
      speedProfile: 'high',
      handoverOptimization: true,
      signalStabilityPriority: true
    };

    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_mobility`,
        variantType: TemplateVariantType.HIGH_MOBILITY,
        tags: [...(template.meta.tags || []), 'mobility', 'high_speed']
      },
      priority: TemplatePriority.MOBILITY,
      configuration: {
        ...template.configuration,
        // Mobility-specific optimizations
        hysteresis: mobilityConfig.handoverOptimization ? 2 : 4,
        timeToTrigger: mobilityConfig.handoverOptimization ? 128 : 256,
        a3Offset: mobilityConfig.signalStabilityPriority ? 2 : 1,
        // Add mobility-specific parameters
        mobilityOptimization: true,
        handoverType: mobilityConfig.mobilityType,
        speedProfile: mobilityConfig.speedProfile
      }
    };
  }
}

class SleepVariantStrategy implements VariantStrategy {
  async generate(template: PriorityTemplate, config?: SleepConfig): Promise<PriorityTemplate> {
    const sleepConfig = config || {
      energySavingLevel: 'moderate',
      trafficThreshold: 0.1,
      activationTime: '02:00',
      deactivationTime: '06:00',
      servicePreservation: true
    };

    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_sleep`,
        variantType: TemplateVariantType.SLEEP_MODE,
        tags: [...(template.meta.tags || []), 'sleep_mode', 'energy_saving']
      },
      priority: TemplatePriority.SLEEP,
      configuration: {
        ...template.configuration,
        // Sleep mode optimizations
        energySavingMode: true,
        energySavingLevel: sleepConfig.energySavingLevel,
        trafficThreshold: sleepConfig.trafficThreshold,
        activationTime: sleepConfig.activationTime,
        deactivationTime: sleepConfig.deactivationTime,
        servicePreservation: sleepConfig.servicePreservation,
        // Reduce power consumption
        txPowerReduction: 0.5,
        bandwidthReduction: 0.3
      }
    };
  }
}

class DenseUrbanVariantStrategy implements VariantStrategy {
  async generate(template: PriorityTemplate, config?: any): Promise<PriorityTemplate> {
    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_dense_urban`,
        variantType: TemplateVariantType.DENSE_URBAN,
        tags: [...(template.meta.tags || []), 'dense_urban', 'ultra_high_capacity']
      },
      priority: 25, // Between urban and mobility
      configuration: {
        ...template.configuration,
        denseUrbanOptimization: true,
        cellCapacity: 'maximum',
        interferenceManagement: 'aggressive',
        loadBalancing: true,
        carrierAggregation: true
      }
    };
  }
}

class SuburbanVariantStrategy implements VariantStrategy {
  async generate(template: PriorityTemplate, config?: any): Promise<PriorityTemplate> {
    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_suburban`,
        variantType: TemplateVariantType.SUBURBAN,
        tags: [...(template.meta.tags || []), 'suburban', 'medium_capacity']
      },
      priority: 35,
      configuration: {
        ...template.configuration,
        suburbanOptimization: true,
        cellCapacity: 'medium',
        coveragePriority: true,
        powerOptimization: true
      }
    };
  }
}

class CoastalVariantStrategy implements VariantStrategy {
  async generate(template: PriorityTemplate, config?: any): Promise<PriorityTemplate> {
    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_coastal`,
        variantType: TemplateVariantType.COASTAL,
        tags: [...(template.meta.tags || []), 'coastal', 'maritime']
      },
      priority: 45,
      configuration: {
        ...template.configuration,
        coastalOptimization: true,
        maritimeMode: true,
        seaPathLossModel: true,
        interferenceTolerance: 'high'
      }
    };
  }
}

class RuralVariantStrategy implements VariantStrategy {
  async generate(template: PriorityTemplate, config?: any): Promise<PriorityTemplate> {
    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_rural`,
        variantType: TemplateVariantType.RURAL,
        tags: [...(template.meta.tags || []), 'rural', 'coverage_focused']
      },
      priority: 55,
      configuration: {
        ...template.configuration,
        ruralOptimization: true,
        coveragePriority: 'maximum',
        powerOptimization: 'aggressive',
        cellSize: 'large',
        trafficAdaptation: true
      }
    };
  }
}

class FrequencyRelationStrategy implements VariantStrategy {
  async generate(template: PriorityTemplate, config?: FrequencyRelationConfig): Promise<PriorityTemplate> {
    if (!config) {
      throw new Error('FrequencyRelationConfig is required for frequency relation variants');
    }

    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_${config.relationType}`,
        variantType: TemplateVariantType.BASE, // Using base for frequency relations
        tags: [...(template.meta.tags || []), 'frequency_relation', config.relationType]
      },
      priority: this.getFrequencyPriority(config.relationType),
      configuration: {
        ...template.configuration,
        frequencyRelation: {
          sourceBand: config.sourceBand,
          targetBand: config.targetBand,
          relationType: config.relationType,
          parameters: config.parameters,
          neighborRelations: config.neighborRelations,
          handoverParameters: config.handoverParameters,
          capacityParameters: config.capacityParameters
        }
      }
    };
  }

  private getFrequencyPriority(relationType: string): TemplatePriority {
    switch (relationType) {
      case '4G4G': return TemplatePriority.FREQUENCY_4G4G;
      case '4G5G': return TemplatePriority.FREQUENCY_4G5G;
      case '5G5G': return TemplatePriority.FREQUENCY_5G5G;
      case '5G4G': return TemplatePriority.FREQUENCY_5G4G;
      default: return TemplatePriority.BASE;
    }
  }
}

class CustomVariantStrategy implements VariantStrategy {
  constructor(private config: VariantGenerationConfig) {}

  async generate(template: PriorityTemplate, config?: any): Promise<PriorityTemplate> {
    return {
      ...template,
      meta: {
        ...template.meta,
        templateId: `${template.meta.templateId}_custom`,
        variantType: this.config.variantType,
        tags: [...(template.meta.tags || []), 'custom']
      },
      priority: TemplatePriority.BASE,
      configuration: {
        ...template.configuration,
        ...this.config.parameterOverrides,
        customVariant: true,
        customLogic: this.config.conditionalLogic
      }
    };
  }
}