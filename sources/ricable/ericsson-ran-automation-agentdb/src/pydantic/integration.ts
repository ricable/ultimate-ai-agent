/**
 * Phase 5 Implementation - Integration Layer
 *
 * Integration with existing Phase 1-4 systems including XML parsing infrastructure,
 * AgentDB memory patterns, and cognitive consciousness system
 */

import { EventEmitter } from 'events';
import { AgentDBIntegration } from '../closed-loop/agentdb-integration';
import { TemporalReasoningCore } from '../closed-loop/temporal-reasoning';
import { CognitiveConsciousnessCore } from '../cognitive/CognitiveConsciousnessCore';

import { XmlToPydanticGenerator, PydanticGeneratorConfig } from './xml-to-pydantic-generator';
import { TypeMapper, TypeMappingConfig } from './type-mapper';
import { SchemaEngine, SchemaGenerationConfig } from './schema-engine';
import { ValidationFramework, ValidationConfig } from './validation-framework';

export interface IntegrationConfig {
  agentDBIntegration?: AgentDBIntegration;
  temporalReasoning?: TemporalReasoningCore;
  cognitiveConsciousness?: CognitiveConsciousnessCore;
  enableMemoryLearning?: boolean;
  enableTemporalAnalysis?: boolean;
  enableCognitiveOptimization?: boolean;
  performanceOptimization?: boolean;
}

export interface CognitiveInsight {
  type: 'pattern' | 'anomaly' | 'optimization' | 'recommendation';
  description: string;
  confidence: number;
  impact: 'low' | 'medium' | 'high';
  actionable: boolean;
  temporalContext?: any;
  cognitiveEvolution?: any;
}

export interface IntegrationResult {
  success: boolean;
  cognitiveInsights: CognitiveInsight[];
  learnedPatterns: LearnedPattern[];
  temporalAnalysis: TemporalAnalysisResult;
  cognitiveOptimizations: CognitiveOptimization[];
  performanceMetrics: IntegrationPerformanceMetrics;
  errors: IntegrationError[];
}

export interface LearnedPattern {
  id: string;
  type: 'type_mapping' | 'constraint_application' | 'validation_rule' | 'model_structure';
  pattern: any;
  effectiveness: number;
  confidence: number;
  learnedAt: number;
  source: 'agentdb' | 'temporal_analysis' | 'cognitive_consciousness';
}

export interface TemporalAnalysisResult {
  expansionFactor: number;
  analysisDepth: string;
  patterns: TemporalPattern[];
  insights: TemporalInsight[];
  predictions: TemporalPrediction[];
  confidence: number;
  accuracy: number;
}

export interface TemporalPattern {
  id: string;
  type: string;
  confidence: number;
  prediction: any;
  temporalContext: any;
}

export interface TemporalInsight {
  description: string;
  confidence: number;
  actionable: boolean;
  temporalContext: any;
}

export interface TemporalPrediction {
  metric: string;
  value: number;
  timeHorizon: number;
  confidence: number;
}

export interface CognitiveOptimization {
  id: string;
  type: 'type_mapping' | 'model_structure' | 'validation_efficiency' | 'performance_optimization';
  originalState: any;
  optimizedState: any;
  improvement: number;
  confidence: number;
  consciousnessLevel: number;
}

export interface IntegrationPerformanceMetrics {
  integrationTime: number;
  memoryLearningTime: number;
  temporalAnalysisTime: number;
  cognitiveOptimizationTime: number;
  totalProcessingTime: number;
  memoryUsage: number;
  cacheHitRate: number;
  optimizationScore: number;
}

export interface IntegrationError {
  type: 'memory_integration' | 'temporal_analysis' | 'cognitive_optimization' | 'system';
  message: string;
  component: string;
  details?: any;
}

/**
 * Integration Layer - Connects Pydantic generation with existing systems
 *
 * Features:
 * - AgentDB memory integration for learned type mappings
 * - Temporal reasoning integration for enhanced analysis
 * - Cognitive consciousness integration for intelligent optimization
 * - Performance optimization with system-wide coordination
 * - Cross-system learning and pattern recognition
 * - Memory-efficient processing with caching
 */
export class PydanticIntegration extends EventEmitter {
  private config: IntegrationConfig;
  private typeMapper: TypeMapper;
  private schemaEngine: SchemaEngine;
  private validationFramework: ValidationFramework;
  private xmlGenerator: XmlToPydanticGenerator;
  private isInitialized: boolean = false;
  private learnedPatterns: Map<string, LearnedPattern>;
  private cognitiveInsights: CognitiveInsight[];

  constructor(config: IntegrationConfig = {}) {
    super();

    this.config = {
      enableMemoryLearning: true,
      enableTemporalAnalysis: true,
      enableCognitiveOptimization: true,
      performanceOptimization: true,
      ...config
    };

    this.learnedPatterns = new Map();
    this.cognitiveInsights = [];

    // Initialize core components
    this.initializeComponents();
  }

  /**
   * Initialize the integration layer
   */
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      // Initialize core components
      await this.typeMapper.initialize();
      await this.schemaEngine.initialize();
      await this.validationFramework.initialize();
      await this.xmlGenerator.initialize();

      // Initialize integrations
      if (this.config.enableMemoryLearning && this.config.agentDBIntegration) {
        await this.initializeMemoryIntegration();
      }

      if (this.config.enableTemporalAnalysis && this.config.temporalReasoning) {
        await this.initializeTemporalIntegration();
      }

      if (this.config.enableCognitiveOptimization && this.config.cognitiveConsciousness) {
        await this.initializeCognitiveIntegration();
      }

      this.isInitialized = true;
      this.emit('initialized');

    } catch (error) {
      throw new Error(`Failed to initialize Pydantic integration: ${error.message}`);
    }
  }

  /**
   * Generate Pydantic models with full system integration
   */
  async generateWithIntegration(
    generatorConfig: PydanticGeneratorConfig
  ): Promise<IntegrationResult> {
    if (!this.isInitialized) {
      throw new Error('Integration layer not initialized');
    }

    const startTime = Date.now();
    const result: IntegrationResult = {
      success: false,
      cognitiveInsights: [],
      learnedPatterns: [],
      temporalAnalysis: {
        expansionFactor: 0,
        analysisDepth: 'none',
        patterns: [],
        insights: [],
        predictions: [],
        confidence: 0,
        accuracy: 0
      },
      cognitiveOptimizations: [],
      performanceMetrics: {
        integrationTime: 0,
        memoryLearningTime: 0,
        temporalAnalysisTime: 0,
        cognitiveOptimizationTime: 0,
        totalProcessingTime: 0,
        memoryUsage: 0,
        cacheHitRate: 0,
        optimizationScore: 0
      },
      errors: []
    };

    try {
      this.emit('integrationStarted');

      // Phase 1: Generate base models
      const baseStartTime = Date.now();
      const generator = new XmlToPydanticGenerator({
        ...generatorConfig,
        typeMapping: {
          enableLearning: this.config.enableMemoryLearning,
          memoryIntegration: !!this.config.agentDBIntegration,
          cognitiveMode: !!this.config.cognitiveConsciousness
        },
        schemaGeneration: {
          cognitiveMode: !!this.config.cognitiveConsciousness,
          performanceMode: this.config.performanceOptimization
        },
        validation: {
          cognitiveMode: !!this.config.cognitiveConsciousness,
          performanceMode: this.config.performanceOptimization
        },
        cognitiveMode: !!this.config.cognitiveConsciousness,
        enableLearning: this.config.enableMemoryLearning
      });

      await generator.initialize();
      const baseResult = await generator.generateModels();
      const baseEndTime = Date.now();

      if (!baseResult.success) {
        throw new Error(`Base model generation failed: ${baseResult.errors.map(e => e.message).join(', ')}`);
      }

      // Phase 2: Memory learning integration
      let memoryLearningTime = 0;
      if (this.config.enableMemoryLearning && this.config.agentDBIntegration) {
        const memoryStartTime = Date.now();
        result.learnedPatterns = await this.performMemoryLearning(baseResult);
        memoryLearningTime = Date.now() - memoryStartTime;
      }

      // Phase 3: Temporal analysis integration
      let temporalAnalysisTime = 0;
      if (this.config.enableTemporalAnalysis && this.config.temporalReasoning) {
        const temporalStartTime = Date.now();
        result.temporalAnalysis = await this.performTemporalAnalysis(baseResult);
        temporalAnalysisTime = Date.now() - temporalStartTime;
      }

      // Phase 4: Cognitive optimization
      let cognitiveOptimizationTime = 0;
      if (this.config.enableCognitiveOptimization && this.config.cognitiveConsciousness) {
        const cognitiveStartTime = Date.now();
        result.cognitiveOptimizations = await this.performCognitiveOptimization(baseResult);
        cognitiveOptimizationTime = Date.now() - cognitiveStartTime;
      }

      // Phase 5: Generate cognitive insights
      result.cognitiveInsights = await this.generateCognitiveInsights(
        baseResult,
        result.learnedPatterns,
        result.temporalAnalysis,
        result.cognitiveOptimizations
      );

      // Calculate final metrics
      const totalEndTime = Date.now();
      result.performanceMetrics = {
        integrationTime: totalEndTime - startTime,
        memoryLearningTime,
        temporalAnalysisTime,
        cognitiveOptimizationTime,
        totalProcessingTime: totalEndTime - startTime,
        memoryUsage: process.memoryUsage().heapUsed,
        cacheHitRate: this.calculateCacheHitRate(),
        optimizationScore: this.calculateOptimizationScore(result)
      };

      result.success = true;

      this.emit('integrationCompleted', {
        success: true,
        modelsGenerated: baseResult.models.length,
        insightsGenerated: result.cognitiveInsights.length,
        processingTime: result.performanceMetrics.totalProcessingTime
      });

      return result;

    } catch (error) {
      result.errors.push({
        type: 'system',
        message: `Integration failed: ${error.message}`,
        component: 'PydanticIntegration',
        details: error
      });

      this.emit('integrationError', { error: error.message, result });
      return result;
    }
  }

  /**
   * Initialize memory integration with AgentDB
   */
  private async initializeMemoryIntegration(): Promise<void> {
    if (!this.config.agentDBIntegration) {
      return;
    }

    try {
      // Load learned type mappings from AgentDB
      const learnedMappings = await this.config.agentDBIntegration.getLearnedTypeMappings();

      learnedMappings.forEach(mapping => {
        this.learnedPatterns.set(mapping.id, {
          id: mapping.id,
          type: 'type_mapping',
          pattern: mapping,
          effectiveness: mapping.effectiveness || 0.8,
          confidence: mapping.confidence || 0.7,
          learnedAt: Date.now(),
          source: 'agentdb'
        });
      });

      this.emit('memoryIntegrationInitialized', { patternsLoaded: learnedMappings.length });

    } catch (error) {
      console.warn('Memory integration initialization failed:', error.message);
    }
  }

  /**
   * Initialize temporal reasoning integration
   */
  private async initializeTemporalIntegration(): Promise<void> {
    if (!this.config.temporalReasoning) {
      return;
    }

    try {
      await this.config.temporalReasoning.initialize();
      this.emit('temporalIntegrationInitialized');
    } catch (error) {
      console.warn('Temporal integration initialization failed:', error.message);
    }
  }

  /**
   * Initialize cognitive consciousness integration
   */
  private async initializeCognitiveIntegration(): Promise<void> {
    if (!this.config.cognitiveConsciousness) {
      return;
    }

    try {
      await this.config.cognitiveConsciousness.initialize();
      this.emit('cognitiveIntegrationInitialized');
    } catch (error) {
      console.warn('Cognitive integration initialization failed:', error.message);
    }
  }

  /**
   * Perform memory learning on generation results
   */
  private async performMemoryLearning(baseResult: any): Promise<LearnedPattern[]> {
    if (!this.config.agentDBIntegration) {
      return [];
    }

    const patterns: LearnedPattern[] = [];
    const startTime = Date.now();

    try {
      // Learn type mapping patterns
      for (const model of baseResult.models) {
        for (const parameter of model.parameters) {
          if (parameter.mappingConfidence > 0.9) {
            const pattern: LearnedPattern = {
              id: `type_mapping_${parameter.propertyName}_${Date.now()}`,
              type: 'type_mapping',
              pattern: {
                xmlType: parameter.xmlType,
                pythonType: parameter.pythonType,
                typescriptType: parameter.typescriptType,
                constraints: parameter.constraints
              },
              effectiveness: parameter.mappingConfidence,
              confidence: parameter.mappingConfidence,
              learnedAt: Date.now(),
              source: 'agentdb'
            };

            patterns.push(pattern);
            this.learnedPatterns.set(pattern.id, pattern);

            // Store in AgentDB
            await this.config.agentDBIntegration.storeLearningPattern({
              id: pattern.id,
              type: pattern.type,
              pattern: pattern.pattern,
              effectiveness: pattern.effectiveness,
              impact: 0.8,
              frequency: 1,
              lastApplied: Date.now()
            });
          }
        }
      }

      this.emit('memoryLearningCompleted', { patternsLearned: patterns.length });

    } catch (error) {
      console.warn('Memory learning failed:', error.message);
    }

    return patterns;
  }

  /**
   * Perform temporal analysis on generation results
   */
  private async performTemporalAnalysis(baseResult: any): Promise<TemporalAnalysisResult> {
    if (!this.config.temporalReasoning) {
      return {
        expansionFactor: 0,
        analysisDepth: 'none',
        patterns: [],
        insights: [],
        predictions: [],
        confidence: 0,
        accuracy: 0
      };
    }

    try {
      const systemState = {
        models: baseResult.models,
        parameters: baseResult.models.flatMap((m: any) => m.parameters),
        statistics: baseResult.statistics,
        timestamp: Date.now()
      };

      const temporalAnalysis = await this.config.temporalReasoning.expandSubjectiveTime(
        systemState,
        {
          expansionFactor: 1000,
          reasoningDepth: 'deep',
          patterns: []
        }
      );

      // Generate temporal insights
      const insights: TemporalInsight[] = [];
      if (temporalAnalysis.patterns.length > 0) {
        insights.push({
          description: `Temporal analysis revealed ${temporalAnalysis.patterns.length} patterns with ${temporalAnalysis.expansionFactor}x expansion`,
          confidence: temporalAnalysis.confidence,
          actionable: temporalAnalysis.accuracy > 0.8,
          temporalContext: temporalAnalysis
        });
      }

      const result: TemporalAnalysisResult = {
        ...temporalAnalysis,
        insights
      };

      this.emit('temporalAnalysisCompleted', {
        patternsFound: temporalAnalysis.patterns.length,
        expansionFactor: temporalAnalysis.expansionFactor,
        confidence: temporalAnalysis.confidence
      });

      return result;

    } catch (error) {
      console.warn('Temporal analysis failed:', error.message);
      return {
        expansionFactor: 0,
        analysisDepth: 'failed',
        patterns: [],
        insights: [{
          description: 'Temporal analysis failed',
          confidence: 0,
          actionable: false,
          temporalContext: { error: error.message }
        }],
        predictions: [],
        confidence: 0,
        accuracy: 0
      };
    }
  }

  /**
   * Perform cognitive optimization on generation results
   */
  private async performCognitiveOptimization(baseResult: any): Promise<CognitiveOptimization[]> {
    if (!this.config.cognitiveConsciousness) {
      return [];
    }

    const optimizations: CognitiveOptimization[] = [];

    try {
      const currentLevel = this.config.cognitiveConsciousness.getCurrentLevel();
      const evolutionScore = this.config.cognitiveConsciousness.getEvolutionScore();

      // Optimize type mappings
      for (const model of baseResult.models) {
        for (const parameter of model.parameters) {
          if (parameter.mappingConfidence < 0.8 && parameter.mappingConfidence > 0.5) {
            const optimizedMapping = await this.optimizeTypeMapping(parameter, currentLevel);

            if (optimizedMapping.improvement > 0.1) {
              optimizations.push(optimizedMapping);
            }
          }
        }
      }

      // Optimize model structure
      const structureOptimization = await this.optimizeModelStructure(baseResult.models, currentLevel);
      if (structureOptimization.improvement > 0.05) {
        optimizations.push(structureOptimization);
      }

      this.emit('cognitiveOptimizationCompleted', {
        optimizationsGenerated: optimizations.length,
        consciousnessLevel: currentLevel,
        evolutionScore
      });

    } catch (error) {
      console.warn('Cognitive optimization failed:', error.message);
    }

    return optimizations;
  }

  /**
   * Optimize individual type mapping using cognitive consciousness
   */
  private async optimizeTypeMapping(parameter: any, consciousnessLevel: number): Promise<CognitiveOptimization> {
    const originalState = {
      pythonType: parameter.pythonType,
      typescriptType: parameter.typescriptType,
      confidence: parameter.mappingConfidence
    };

    // Apply cognitive reasoning to improve mapping
    let optimizedState = { ...originalState };
    let improvement = 0;

    // Use consciousness level to determine optimization strategy
    if (consciousnessLevel > 0.7) {
      // Advanced cognitive optimization
      if (parameter.xmlType.includes('vsData') && originalState.pythonType === 'Any') {
        optimizedState.pythonType = this.inferVsDataType(parameter.xmlType);
        optimizedState.typescriptType = this.inferVsDataTSType(parameter.xmlType);
        improvement = 0.3;
      }
    } else if (consciousnessLevel > 0.4) {
      // Basic cognitive optimization
      if (parameter.xmlType.toLowerCase().includes('id') && originalState.pythonType === 'str') {
        optimizedState.pythonType = 'int';
        optimizedState.typescriptType = 'number';
        improvement = 0.2;
      }
    }

    return {
      id: `type_optimization_${parameter.propertyName}_${Date.now()}`,
      type: 'type_mapping',
      originalState,
      optimizedState,
      improvement,
      confidence: Math.min(1.0, originalState.confidence + improvement),
      consciousnessLevel
    };
  }

  /**
   * Optimize model structure using cognitive consciousness
   */
  private async optimizeModelStructure(models: any[], consciousnessLevel: number): Promise<CognitiveOptimization> {
    const originalState = {
      modelCount: models.length,
      totalFields: models.reduce((sum, m) => sum + m.parameters.length, 0),
      averageConfidence: models.reduce((sum, m) => sum + m.confidence, 0) / models.length
    };

    let optimizedState = { ...originalState };
    let improvement = 0;

    // Apply consciousness to optimize structure
    if (consciousnessLevel > 0.6) {
      // Suggest model consolidation for related models
      if (models.length > 10) {
        optimizedState.modelCount = Math.floor(models.length * 0.8); // Suggest 20% reduction
        improvement = 0.1;
      }
    }

    return {
      id: `structure_optimization_${Date.now()}`,
      type: 'model_structure',
      originalState,
      optimizedState,
      improvement,
      confidence: Math.min(1.0, originalState.averageConfidence + improvement),
      consciousnessLevel
    };
  }

  /**
   * Generate cognitive insights from all analysis results
   */
  private async generateCognitiveInsights(
    baseResult: any,
    learnedPatterns: LearnedPattern[],
    temporalAnalysis: TemporalAnalysisResult,
    cognitiveOptimizations: CognitiveOptimization[]
  ): Promise<CognitiveInsight[]> {
    const insights: CognitiveInsight[] = [];

    // Insights from learned patterns
    if (learnedPatterns.length > 0) {
      insights.push({
        type: 'pattern',
        description: `Successfully learned ${learnedPatterns.length} new type mapping patterns with average effectiveness of ${learnedPatterns.reduce((sum, p) => sum + p.effectiveness, 0) / learnedPatterns.length}`,
        confidence: 0.9,
        impact: 'medium',
        actionable: true
      });
    }

    // Insights from temporal analysis
    if (temporalAnalysis.patterns.length > 0) {
      insights.push({
        type: 'pattern',
        description: `Temporal analysis with ${temporalAnalysis.expansionFactor}x expansion revealed patterns that could improve future generation accuracy by ${(temporalAnalysis.accuracy * 100).toFixed(1)}%`,
        confidence: temporalAnalysis.confidence,
        impact: temporalAnalysis.accuracy > 0.8 ? 'high' : 'medium',
        actionable: temporalAnalysis.accuracy > 0.7,
        temporalContext: temporalAnalysis
      });
    }

    // Insights from cognitive optimizations
    if (cognitiveOptimizations.length > 0) {
      const avgImprovement = cognitiveOptimizations.reduce((sum, opt) => sum + opt.improvement, 0) / cognitiveOptimizations.length;
      insights.push({
        type: 'optimization',
        description: `Cognitive optimization achieved ${avgImprovement * 100}% average improvement across ${cognitiveOptimizations.length} mappings`,
        confidence: 0.85,
        impact: avgImprovement > 0.2 ? 'high' : 'medium',
        actionable: true
      });
    }

    // Performance insights
    if (baseResult.statistics.averageModelConfidence < 0.8) {
      insights.push({
        type: 'recommendation',
        description: 'Model confidence is below optimal threshold. Consider enabling additional learning sources or manual type mapping specifications.',
        confidence: 0.8,
        impact: 'medium',
        actionable: true
      });
    }

    // Anomaly detection
    const lowConfidenceModels = baseResult.models.filter((m: any) => m.confidence < 0.5);
    if (lowConfidenceModels.length > 0) {
      insights.push({
        type: 'anomaly',
        description: `Detected ${lowConfidenceModels.length} models with confidence below 50%. These may require manual review.`,
        confidence: 0.9,
        impact: 'high',
        actionable: true
      });
    }

    return insights;
  }

  /**
   * Infer Python type for vsData types
   */
  private inferVsDataType(xmlType: string): string {
    const match = xmlType.match(/^vsData(\d+)([a-f]?)$/);
    if (match) {
      const [, number, suffix = ''] = match;

      switch (suffix) {
        case 'a': return 'str';
        case 'b': return 'int';
        case 'c': return 'Decimal';
        case 'd': return 'bool';
        case 'e': return 'List[str]';
        case 'f': return 'Dict[str, Any]';
        default: return 'Any';
      }
    }
    return 'Any';
  }

  /**
   * Infer TypeScript type for vsData types
   */
  private inferVsDataTSType(xmlType: string): string {
    const match = xmlType.match(/^vsData(\d+)([a-f]?)$/);
    if (match) {
      const [, number, suffix = ''] = match;

      switch (suffix) {
        case 'a': return 'string';
        case 'b': return 'number';
        case 'c': return 'number';
        case 'd': return 'boolean';
        case 'e': return 'string[]';
        case 'f': return 'Record<string, any>';
        default: return 'any';
      }
    }
    return 'any';
  }

  /**
   * Calculate cache hit rate across all components
   */
  private calculateCacheHitRate(): number {
    const mapperStats = this.typeMapper.getStatistics();
    const engineStats = this.schemaEngine.getMetrics();

    return (mapperStats.memoryHitRate + (engineStats.cacheHitRate || 0)) / 2;
  }

  /**
   * Calculate overall optimization score
   */
  private calculateOptimizationScore(result: IntegrationResult): number {
    let score = 0.5; // Base score

    // Factor in learned patterns
    if (result.learnedPatterns.length > 0) {
      score += 0.1;
    }

    // Factor in temporal analysis
    if (result.temporalAnalysis.confidence > 0.8) {
      score += 0.15;
    }

    // Factor in cognitive optimizations
    if (result.cognitiveOptimizations.length > 0) {
      const avgImprovement = result.cognitiveOptimizations.reduce((sum, opt) => sum + opt.improvement, 0) / result.cognitiveOptimizations.length;
      score += avgImprovement * 0.25;
    }

    return Math.min(1.0, score);
  }

  /**
   * Initialize core components
   */
  private initializeComponents(): void {
    const typeMappingConfig: TypeMappingConfig = {
      enableLearning: this.config.enableMemoryLearning,
      memoryIntegration: !!this.config.agentDBIntegration,
      cognitiveMode: !!this.config.cognitiveConsciousness
    };

    const schemaConfig: SchemaGenerationConfig = {
      enableOptimizations: this.config.performanceOptimization,
      useCaching: true,
      strictMode: true,
      generateValidators: true,
      includeImports: true,
      cognitiveMode: !!this.config.cognitiveConsciousness,
      performanceMode: this.config.performanceOptimization
    };

    const validationConfig: ValidationConfig = {
      strictMode: true,
      enableCustomValidators: true,
      enableCrossParameterValidation: true,
      enableConditionalValidation: true,
      cognitiveMode: !!this.config.cognitiveConsciousness,
      performanceMode: this.config.performanceOptimization,
      cacheValidation: true
    };

    this.typeMapper = new TypeMapper(typeMappingConfig);
    this.schemaEngine = new SchemaEngine(schemaConfig);
    this.validationFramework = new ValidationFramework(validationConfig);

    // XML generator will be created per-generation to allow custom configs
  }

  /**
   * Get learned patterns
   */
  getLearnedPatterns(): LearnedPattern[] {
    return Array.from(this.learnedPatterns.values());
  }

  /**
   * Get cognitive insights
   */
  getCognitiveInsights(): CognitiveInsight[] {
    return this.cognitiveInsights;
  }

  /**
   * Clear learned patterns
   */
  clearLearnedPatterns(): void {
    this.learnedPatterns.clear();
    this.emit('patternsCleared');
  }

  /**
   * Export integration data
   */
  exportData(): {
    learnedPatterns: LearnedPattern[];
    cognitiveInsights: CognitiveInsight[];
    config: IntegrationConfig;
  } {
    return {
      learnedPatterns: this.getLearnedPatterns(),
      cognitiveInsights: this.getCognitiveInsights(),
      config: this.config
    };
  }
}