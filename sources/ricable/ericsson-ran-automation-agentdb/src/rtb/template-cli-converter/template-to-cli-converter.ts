/**
 * Template-to-CLI Converter - Core Implementation
 *
 * Converts RTB JSON templates to Ericsson ENM CLI (cmedit) commands with
 * cognitive optimization, dependency analysis, and intelligent FDN construction.
 */

import {
  TemplateToCliContext,
  CliCommandSet,
  GeneratedCliCommand,
  CliCommandType,
  FdnConstructionResult,
  DependencyAnalysisResult,
  CommandValidationResult,
  CognitiveOptimizationResult,
  TemplateToCliConfig,
  ConversionStats
} from './types';

import {
  RTBTemplate,
  MOHierarchy,
  MOClass,
  ReservedByHierarchy
} from '../../types/rtb-types';

import {
  FdnPathConstructor
} from './fdn-path-constructor';

import {
  BatchCommandGenerator
} from './batch-command-generator';

import {
  CommandValidator
} from './command-validator';

import {
  DependencyAnalyzer
} from './dependency-analyzer';

import {
  CognitiveOptimizer
} from './cognitive-optimizer';

import {
  EricssonRanExpertise
} from './ericsson-ran-expertise';

import {
  RollbackManager
} from './rollback-manager';

/**
 * Template-to-CLI Converter Class
 *
 * Main orchestrator for converting RTB templates to cmedit commands
 * with cognitive optimization and dependency analysis.
 */
export class TemplateToCliConverter {
  private config: TemplateToCliConfig;
  private fdnConstructor: FdnPathConstructor;
  private batchGenerator: BatchCommandGenerator;
  private validator: CommandValidator;
  private dependencyAnalyzer: DependencyAnalyzer;
  private cognitiveOptimizer: CognitiveOptimizer;
  private ranExpertise: EricssonRanExpertise;
  private rollbackManager: RollbackManager;
  private conversionHistory: Map<string, CliCommandSet> = new Map();
  private executionHistory: Map<string, any[]> = new Map();

  constructor(config?: Partial<TemplateToCliConfig>) {
    this.config = {
      defaultTimeout: 30,
      maxCommandsPerBatch: 50,
      enableCognitiveOptimization: true,
      enableDependencyAnalysis: true,
      validationStrictness: 'normal',
      rollbackStrategy: 'full',
      performanceOptimization: {
        enableParallelExecution: true,
        maxParallelCommands: 10,
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
      },
      ...config
    };

    this.initializeComponents();
  }

  /**
   * Initialize converter components
   */
  private initializeComponents(): void {
    this.fdnConstructor = new FdnPathConstructor({
      enableOptimization: true,
      validateSyntax: true,
      applyHierarchyKnowledge: true
    });

    this.batchGenerator = new BatchCommandGenerator({
      maxCommandsPerBatch: this.config.maxCommandsPerBatch,
      enableParallelExecution: this.config.performanceOptimization.enableParallelExecution,
      maxConcurrency: this.config.performanceOptimization.maxParallelCommands
    });

    this.validator = new CommandValidator({
      strictness: this.config.validationStrictness,
      enableSyntaxValidation: true,
      enableSemanticValidation: true
    });

    this.dependencyAnalyzer = new DependencyAnalyzer({
      enableCircularDependencyDetection: true,
      enableCriticalPathAnalysis: true,
      enableOptimizationSuggestions: true
    });

    this.cognitiveOptimizer = new CognitiveOptimizer({
      enableTemporalReasoning: this.config.cognitive?.enableTemporalReasoning ?? true,
      enableStrangeLoopOptimization: this.config.cognitive?.enableStrangeLoopOptimization ?? true,
      consciousnessLevel: this.config.cognitive?.consciousnessLevel ?? 0.8,
      learningMode: this.config.cognitive?.learningMode ?? 'active'
    });

    this.ranExpertise = new EricssonRanExpertise({
      enablePatternMatching: true,
      enableBestPractices: true,
      enableOptimizationSuggestions: true
    });

    this.rollbackManager = new RollbackManager({
      strategy: this.config.rollbackStrategy,
      enableValidation: true,
      enableSelectiveRollback: true
    });
  }

  /**
   * Convert RTB template to CLI commands
   */
  public async convertTemplate(
    template: RTBTemplate,
    context: TemplateToCliContext
  ): Promise<CliCommandSet> {
    const startTime = Date.now();
    const templateId = template.meta?.version || 'unknown';
    const commandSetId = `${templateId}_${Date.now()}`;

    console.log(`Converting template ${templateId} to CLI commands...`);

    try {
      // Phase 1: Template analysis and preprocessing
      const analysisResult = await this.analyzeTemplate(template, context);

      // Phase 2: Generate initial commands
      const initialCommands = await this.generateInitialCommands(template, context, analysisResult);

      // Phase 3: Apply Ericsson RAN expertise
      const optimizedCommands = await this.applyRanExpertise(initialCommands, context);

      // Phase 4: Cognitive optimization (if enabled)
      const cognitiveCommands = this.config.enableCognitiveOptimization
        ? await this.applyCognitiveOptimization(optimizedCommands, context)
        : optimizedCommands;

      // Phase 5: Dependency analysis and ordering
      const dependencyResult = this.config.enableDependencyAnalysis
        ? await this.analyzeDependencies(cognitiveCommands, context)
        : this.createBasicDependencyGraph(cognitiveCommands);

      // Phase 6: Command validation
      const validationResult = await this.validateCommands(cognitiveCommands, context);

      // Phase 7: Generate rollback commands
      const rollbackCommands = await this.generateRollbackCommands(cognitiveCommands, context);

      // Phase 8: Generate validation commands
      const validationCommands = await this.generateValidationCommands(cognitiveCommands, context);

      // Phase 9: Batch optimization
      const finalCommands = await this.optimizeCommandBatches(cognitiveCommands, context);

      // Phase 10: Create command set
      const commandSet = await this.createCommandSet(
        commandSetId,
        template,
        finalCommands,
        dependencyResult,
        validationResult,
        rollbackCommands,
        validationCommands,
        context,
        Date.now() - startTime
      );

      // Store conversion history
      this.conversionHistory.set(commandSetId, commandSet);

      console.log(`Template conversion completed: ${commandSet.commands.length} commands generated`);
      return commandSet;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`Template conversion failed: ${errorMessage}`);
      throw new Error(`Template conversion failed: ${errorMessage}`);
    }
  }

  /**
   * Analyze template structure and characteristics
   */
  private async analyzeTemplate(
    template: RTBTemplate,
    context: TemplateToCliContext
  ): Promise<TemplateAnalysisResult> {
    const startTime = Date.now();

    const analysis: TemplateAnalysisResult = {
      complexity: this.calculateTemplateComplexity(template),
      parameterCount: Object.keys(template.configuration || {}).length,
      hasConditions: Object.keys(template.conditions || {}).length > 0,
      hasEvaluations: Object.keys(template.evaluations || {}).length > 0,
      hasCustomFunctions: (template.custom || []).length > 0,
      targetMoClasses: this.identifyTargetMoClasses(template, context),
      estimatedCommands: this.estimateCommandCount(template),
      riskLevel: this.assessRiskLevel(template),
      categories: this.categorizeTemplateContent(template)
    };

    console.log(`Template analysis completed in ${Date.now() - startTime}ms`);
    return analysis;
  }

  /**
   * Generate initial commands from template
   */
  private async generateInitialCommands(
    template: RTBTemplate,
    context: TemplateToCliContext,
    analysis: TemplateAnalysisResult
  ): Promise<GeneratedCliCommand[]> {
    const commands: GeneratedCliCommand[] = [];
    const { configuration, conditions, evaluations } = template;

    // Generate configuration commands
    if (configuration) {
      const configCommands = await this.generateConfigurationCommands(
        configuration,
        context,
        analysis
      );
      commands.push(...configCommands);
    }

    // Generate conditional commands
    if (conditions) {
      const conditionalCommands = await this.generateConditionalCommands(
        conditions,
        context,
        analysis
      );
      commands.push(...conditionalCommands);
    }

    // Generate evaluation commands
    if (evaluations) {
      const evaluationCommands = await this.generateEvaluationCommands(
        evaluations,
        context,
        analysis
      );
      commands.push(...evaluationCommands);
    }

    return commands;
  }

  /**
   * Generate configuration commands
   */
  private async generateConfigurationCommands(
    configuration: Record<string, any>,
    context: TemplateToCliContext,
    analysis: TemplateAnalysisResult
  ): Promise<GeneratedCliCommand[]> {
    const commands: GeneratedCliCommand[] = [];
    let commandIndex = 0;

    for (const [parameterPath, value] of Object.entries(configuration)) {
      const fdnResult = await this.constructFdnPath(parameterPath, context);

      if (!fdnResult.isValid) {
        console.warn(`Invalid FDN path: ${parameterPath}`, fdnResult.errors);
        continue;
      }

      const command = this.createSetCommand(
        `config_${commandIndex++}`,
        fdnResult.fdn,
        { [this.extractParameterName(parameterPath)]: value },
        `Set ${parameterPath} to ${value}`,
        context
      );

      commands.push(command);
    }

    return commands;
  }

  /**
   * Generate conditional commands
   */
  private async generateConditionalCommands(
    conditions: Record<string, any>,
    context: TemplateToCliContext,
    analysis: TemplateAnalysisResult
  ): Promise<GeneratedCliCommand[]> {
    const commands: GeneratedCliCommand[] = [];
    let commandIndex = 0;

    for (const [conditionId, condition] of Object.entries(conditions)) {
      // For now, we generate both branches (conditional execution will be handled later)
      const conditionCommand = this.createScriptCommand(
        `condition_${commandIndex++}`,
        this.generateConditionScript(condition, context),
        `Execute conditional logic for ${conditionId}`,
        context
      );

      commands.push(conditionCommand);
    }

    return commands;
  }

  /**
   * Generate evaluation commands
   */
  private async generateEvaluationCommands(
    evaluations: Record<string, any>,
    context: TemplateToCliContext,
    analysis: TemplateAnalysisResult
  ): Promise<GeneratedCliCommand[]> {
    const commands: GeneratedCliCommand[] = [];
    let commandIndex = 0;

    for (const [evalId, evaluation] of Object.entries(evaluations)) {
      const evalCommand = this.createScriptCommand(
        `eval_${commandIndex++}`,
        this.generateEvaluationScript(evaluation, context),
        `Execute evaluation for ${evalId}`,
        context
      );

      commands.push(evalCommand);
    }

    return commands;
  }

  /**
   * Apply Ericsson RAN expertise patterns
   */
  private async applyRanExpertise(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<GeneratedCliCommand[]> {
    console.log('Applying Ericsson RAN expertise patterns...');

    const enhancedCommands: GeneratedCliCommand[] = [];

    for (const command of commands) {
      const enhancedCommand = await this.ranExpertise.enhanceCommand(command, context);
      enhancedCommands.push(enhancedCommand);
    }

    // Apply RAN-specific optimizations
    const optimizations = await this.ranExpertise.generateOptimizations(commands, context);
    enhancedCommands.push(...optimizations);

    return enhancedCommands;
  }

  /**
   * Apply cognitive optimization
   */
  private async applyCognitiveOptimization(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<GeneratedCliCommand[]> {
    console.log('Applying cognitive optimization...');

    const optimizationResult = await this.cognitiveOptimizer.optimize(commands, context);

    // Apply optimizations to commands
    const optimizedCommands = commands.map((command, index) => {
      const optimization = optimizationResult.commandOptimizations[index];
      if (optimization) {
        return {
          ...command,
          cognitive: {
            optimizationLevel: optimization.level,
            reasoningApplied: optimization.reasoning,
            confidence: optimization.confidence
          }
        };
      }
      return command;
    });

    return optimizedCommands;
  }

  /**
   * Analyze command dependencies
   */
  private async analyzeDependencies(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<DependencyAnalysisResult> {
    console.log('Analyzing command dependencies...');

    return await this.dependencyAnalyzer.analyze(commands, context);
  }

  /**
   * Create basic dependency graph (when analysis is disabled)
   */
  private createBasicDependencyGraph(commands: GeneratedCliCommand[]): DependencyAnalysisResult {
    return {
      dependencyGraph: {
        nodes: commands.map(cmd => ({
          id: cmd.id,
          type: cmd.type,
          critical: cmd.critical || false,
          estimatedDuration: cmd.metadata.estimatedDuration,
          riskLevel: cmd.metadata.riskLevel,
          dependencyCount: 0,
          dependentCount: 0
        })),
        edges: [],
        metrics: {
          totalNodes: commands.length,
          totalEdges: 0,
          maxDepth: 1,
          avgBranchingFactor: 0
        }
      },
      criticalPath: commands.map(cmd => cmd.id),
      executionLevels: [{
        level: 0,
        commands: commands.map(cmd => cmd.id),
        parallel: true,
        estimatedDuration: commands.reduce((sum, cmd) => sum + cmd.metadata.estimatedDuration, 0),
        riskLevel: 'medium'
      }],
      circularDependencies: [],
      optimizations: []
    };
  }

  /**
   * Validate commands
   */
  private async validateCommands(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<CommandValidationResult> {
    console.log('Validating generated commands...');

    return await this.validator.validate(commands, context);
  }

  /**
   * Generate rollback commands
   */
  private async generateRollbackCommands(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<GeneratedCliCommand[]> {
    if (!context.options.generateRollback) {
      return [];
    }

    console.log('Generating rollback commands...');

    return await this.rollbackManager.generateRollbackCommands(commands, context);
  }

  /**
   * Generate validation commands
   */
  private async generateValidationCommands(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<GeneratedCliCommand[]> {
    if (!context.options.generateValidation) {
      return [];
    }

    console.log('Generating validation commands...');

    const validationCommands: GeneratedCliCommand[] = [];
    let commandIndex = 0;

    for (const command of commands) {
      if (command.type === 'SET' || command.type === 'CREATE') {
        const validationCommand = this.createValidationCommand(
          `validate_${commandIndex++}`,
          command,
          context
        );
        validationCommands.push(validationCommand);
      }
    }

    return validationCommands;
  }

  /**
   * Optimize command batches
   */
  private async optimizeCommandBatches(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<GeneratedCliCommand[]> {
    if (!context.options.batchMode) {
      return commands;
    }

    console.log('Optimizing command batches...');

    return await this.batchGenerator.optimizeBatches(commands, context);
  }

  /**
   * Create final command set
   */
  private async createCommandSet(
    id: string,
    template: RTBTemplate,
    commands: GeneratedCliCommand[],
    dependencyResult: DependencyAnalysisResult,
    validationResult: CommandValidationResult,
    rollbackCommands: GeneratedCliCommand[],
    validationCommands: GeneratedCliCommand[],
    context: TemplateToCliContext,
    totalConversionTime: number
  ): Promise<CliCommandSet> {
    const executionOrder = dependencyResult.criticalPath;
    const dependencies = this.extractDependencies(dependencyResult);

    // Calculate metadata
    const metadata = {
      generatedAt: new Date(),
      totalCommands: commands.length,
      estimatedDuration: commands.reduce((sum, cmd) => sum + cmd.metadata.estimatedDuration, 0),
      complexity: this.calculateOverallComplexity(commands),
      riskLevel: this.calculateOverallRiskLevel(commands)
    };

    // Create conversion statistics
    const stats: ConversionStats = {
      templateProcessingTime: totalConversionTime * 0.2,
      commandGenerationTime: totalConversionTime * 0.3,
      dependencyAnalysisTime: totalConversionTime * 0.2,
      validationTime: totalConversionTime * 0.3,
      totalConversionTime,
      templateStats: {
        totalParameters: Object.keys(template.configuration || {}).length,
        totalConditions: Object.keys(template.conditions || {}).length,
        totalEvaluations: Object.keys(template.evaluations || {}).length,
        templateSize: JSON.stringify(template).length
      },
      commandStats: {
        totalCommands: commands.length,
        commandsByType: this.groupCommandsByType(commands),
        commandsByCategory: this.groupCommandsByCategory(commands),
        avgComplexity: this.calculateAverageComplexity(commands),
        avgRiskLevel: this.calculateAverageRiskLevel(commands)
      }
    };

    return {
      id,
      source: {
        templateId: template.meta?.version || 'unknown',
        templateVersion: template.meta?.version || '1.0.0'
      },
      commands,
      executionOrder,
      dependencies,
      rollbackCommands,
      validationCommands,
      metadata,
      stats
    };
  }

  /**
   * Construct FDN path for parameter
   */
  private async constructFdnPath(
    parameterPath: string,
    context: TemplateToCliContext
  ): Promise<FdnConstructionResult> {
    return await this.fdnConstructor.construct(parameterPath, context);
  }

  /**
   * Create SET command
   */
  private createSetCommand(
    id: string,
    fdn: string,
    parameters: Record<string, any>,
    description: string,
    context: TemplateToCliContext
  ): GeneratedCliCommand {
    const paramList = Object.entries(parameters)
      .map(([key, value]) => `${key}=${value}`)
      .join(',');

    const command = `cmedit set ${context.target.nodeId} ${fdn} ${paramList}`;

    return {
      id,
      type: 'SET',
      command,
      description,
      targetFdn: fdn,
      parameters,
      timeout: context.options.timeout || this.config.defaultTimeout,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    };
  }

  /**
   * Create SCRIPT command
   */
  private createScriptCommand(
    id: string,
    script: string,
    description: string,
    context: TemplateToCliContext
  ): GeneratedCliCommand {
    return {
      id,
      type: 'SCRIPT',
      command: script,
      description,
      timeout: context.options.timeout || this.config.defaultTimeout,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'moderate',
        riskLevel: 'medium',
        estimatedDuration: 2000
      }
    };
  }

  /**
   * Create validation command
   */
  private createValidationCommand(
    id: string,
    sourceCommand: GeneratedCliCommand,
    context: TemplateToCliContext
  ): GeneratedCliCommand {
    const validationCommand = `cmedit get ${context.target.nodeId} ${sourceCommand.targetFdn || ''} -s`;

    return {
      id,
      type: 'VALIDATION',
      command: validationCommand,
      description: `Validate ${sourceCommand.description}`,
      targetFdn: sourceCommand.targetFdn,
      expectedOutput: ['syncStatus=SYNCHRONIZED', 'operState=ENABLED'],
      timeout: 30,
      critical: false,
      metadata: {
        category: 'validation',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 500
      }
    };
  }

  /**
   * Generate condition script
   */
  private generateConditionScript(condition: any, context: TemplateToCliContext): string {
    // Simplified condition script generation
    return `# Conditional logic execution\n# Condition: ${JSON.stringify(condition)}\necho "Conditional execution placeholder"`;
  }

  /**
   * Generate evaluation script
   */
  private generateEvaluationScript(evaluation: any, context: TemplateToCliContext): string {
    // Simplified evaluation script generation
    return `# Evaluation execution\n# Evaluation: ${JSON.stringify(evaluation)}\necho "Evaluation placeholder"`;
  }

  /**
   * Extract parameter name from path
   */
  private extractParameterName(parameterPath: string): string {
    const parts = parameterPath.split('.');
    return parts[parts.length - 1];
  }

  /**
   * Extract dependencies from analysis result
   */
  private extractDependencies(dependencyResult: DependencyAnalysisResult): Record<string, string[]> {
    const dependencies: Record<string, string[]> = {};

    for (const edge of dependencyResult.dependencyGraph.edges) {
      if (!dependencies[edge.to]) {
        dependencies[edge.to] = [];
      }
      dependencies[edge.to].push(edge.from);
    }

    return dependencies;
  }

  /**
   * Calculate template complexity
   */
  private calculateTemplateComplexity(template: RTBTemplate): 'low' | 'medium' | 'high' | 'critical' {
    const paramCount = Object.keys(template.configuration || {}).length;
    const conditionCount = Object.keys(template.conditions || {}).length;
    const evalCount = Object.keys(template.evaluations || {}).length;
    const customCount = (template.custom || []).length;

    const complexity = paramCount + (conditionCount * 2) + (evalCount * 3) + (customCount * 2);

    if (complexity < 10) return 'low';
    if (complexity < 25) return 'medium';
    if (complexity < 50) return 'high';
    return 'critical';
  }

  /**
   * Identify target MO classes
   */
  private identifyTargetMoClasses(template: RTBTemplate, context: TemplateToCliContext): string[] {
    // Simplified MO class identification
    const moClasses = new Set<string>();

    // Extract from configuration paths
    for (const path of Object.keys(template.configuration || {})) {
      const parts = path.split('.');
      if (parts.length > 1) {
        moClasses.add(parts[0]);
      }
    }

    return Array.from(moClasses);
  }

  /**
   * Estimate command count
   */
  private estimateCommandCount(template: RTBTemplate): number {
    const configCommands = Object.keys(template.configuration || {}).length;
    const conditionCommands = Object.keys(template.conditions || {}).length;
    const evalCommands = Object.keys(template.evaluations || {}).length;

    return configCommands + conditionCommands + evalCommands + 5; // +5 for setup/validation
  }

  /**
   * Assess risk level
   */
  private assessRiskLevel(template: RTBTemplate): 'low' | 'medium' | 'high' | 'critical' {
    const complexity = this.calculateTemplateComplexity(template);
    const hasCriticalParams = this.hasCriticalParameters(template);

    if (complexity === 'critical' || hasCriticalParams) return 'critical';
    if (complexity === 'high') return 'high';
    if (complexity === 'medium') return 'medium';
    return 'low';
  }

  /**
   * Check for critical parameters
   */
  private hasCriticalParameters(template: RTBTemplate): boolean {
    const criticalPatterns = ['adminState', 'operState', 'syncStatus', 'enabled', 'active'];

    for (const path of Object.keys(template.configuration || {})) {
      if (criticalPatterns.some(pattern => path.toLowerCase().includes(pattern.toLowerCase()))) {
        return true;
      }
    }

    return false;
  }

  /**
   * Categorize template content
   */
  private categorizeTemplateContent(template: RTBTemplate): string[] {
    const categories = new Set<string>();

    // Analyze configuration content
    for (const path of Object.keys(template.configuration || {})) {
      if (path.toLowerCase().includes('cell')) categories.add('cell');
      if (path.toLowerCase().includes('mobility')) categories.add('mobility');
      if (path.toLowerCase().includes('capacity')) categories.add('capacity');
      if (path.toLowerCase().includes('feature')) categories.add('feature');
      if (path.toLowerCase().includes('energy')) categories.add('energy');
    }

    return Array.from(categories);
  }

  /**
   * Group commands by type
   */
  private groupCommandsByType(commands: GeneratedCliCommand[]): Record<CliCommandType, number> {
    const groups: Record<string, number> = {};

    for (const command of commands) {
      groups[command.type] = (groups[command.type] || 0) + 1;
    }

    return groups as Record<CliCommandType, number>;
  }

  /**
   * Group commands by category
   */
  private groupCommandsByCategory(commands: GeneratedCliCommand[]): Record<string, number> {
    const groups: Record<string, number> = {};

    for (const command of commands) {
      const category = command.metadata.category;
      groups[category] = (groups[category] || 0) + 1;
    }

    return groups;
  }

  /**
   * Calculate average complexity
   */
  private calculateAverageComplexity(commands: GeneratedCliCommand[]): number {
    const complexityMap = { simple: 1, moderate: 2, complex: 3 };
    const totalComplexity = commands.reduce((sum, cmd) =>
      sum + (complexityMap[cmd.metadata.complexity] || 1), 0);
    return totalComplexity / commands.length;
  }

  /**
   * Calculate average risk level
   */
  private calculateAverageRiskLevel(commands: GeneratedCliCommand[]): number {
    const riskMap = { low: 1, medium: 2, high: 3 };
    const totalRisk = commands.reduce((sum, cmd) =>
      sum + (riskMap[cmd.metadata.riskLevel] || 1), 0);
    return totalRisk / commands.length;
  }

  /**
   * Calculate overall complexity
   */
  private calculateOverallComplexity(commands: GeneratedCliCommand[]): 'low' | 'medium' | 'high' | 'critical' {
    const avgComplexity = this.calculateAverageComplexity(commands);

    if (avgComplexity < 1.5) return 'low';
    if (avgComplexity < 2.5) return 'medium';
    if (avgComplexity < 3.5) return 'high';
    return 'critical';
  }

  /**
   * Calculate overall risk level
   */
  private calculateOverallRiskLevel(commands: GeneratedCliCommand[]): 'low' | 'medium' | 'high' | 'critical' {
    const avgRisk = this.calculateAverageRiskLevel(commands);
    const hasCritical = commands.some(cmd => cmd.metadata.riskLevel === 'high' && cmd.critical);

    if (hasCritical || avgRisk > 2.5) return 'critical';
    if (avgRisk > 2.0) return 'high';
    if (avgRisk > 1.5) return 'medium';
    return 'low';
  }

  /**
   * Get conversion history
   */
  public getConversionHistory(): Map<string, CliCommandSet> {
    return new Map(this.conversionHistory);
  }

  /**
   * Get execution history
   */
  public getExecutionHistory(commandSetId: string): any[] {
    return this.executionHistory.get(commandSetId) || [];
  }

  /**
   * Clear history
   */
  public clearHistory(): void {
    this.conversionHistory.clear();
    this.executionHistory.clear();
  }
}

/**
 * Template analysis result
 */
interface TemplateAnalysisResult {
  complexity: 'low' | 'medium' | 'high' | 'critical';
  parameterCount: number;
  hasConditions: boolean;
  hasEvaluations: boolean;
  hasCustomFunctions: boolean;
  targetMoClasses: string[];
  estimatedCommands: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  categories: string[];
}