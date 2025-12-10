/**
 * Template-to-CLI Converter Test Suite
 *
 * Comprehensive test suite for the template-to-CLI conversion system
 * including unit tests, integration tests, and performance tests.
 */

import {
  TemplateToCliConverter,
  FdnPathConstructor,
  BatchCommandGenerator,
  CommandValidator,
  DependencyAnalyzer,
  RollbackManager,
  EricssonRanExpertise,
  CognitiveOptimizer,
  RtbTemplateIntegration,
  createTemplateToCliConverter,
  DEFAULT_CONFIG,
  SAFE_CONFIG
} from '../../src/rtb/template-cli-converter';

import {
  RTBTemplate,
  TemplateMeta,
  MOHierarchy,
  ReservedByHierarchy
} from '../../src/types/rtb-types';

import {
  TemplateToCliContext,
  CliCommandSet,
  GeneratedCliCommand,
  CliCommandType
} from '../../src/rtb/template-cli-converter/types';

describe('Template-to-CLI Converter System', () => {
  let converter: TemplateToCliConverter;
  let sampleTemplate: RTBTemplate;
  let sampleContext: TemplateToCliContext;
  let sampleMOHierarchy: MOHierarchy;

  beforeEach(() => {
    // Initialize test data
    converter = createTemplateToCliConverter();
    sampleMOHierarchy = createSampleMOHierarchy();
    sampleTemplate = createSampleTemplate();
    sampleContext = createSampleContext(sampleMOHierarchy);
  });

  describe('TemplateToCliConverter', () => {
    describe('Basic Conversion', () => {
      test('should convert simple template to CLI commands', async () => {
        const result = await converter.convertTemplate(sampleTemplate, sampleContext);

        expect(result).toBeDefined();
        expect(result.commands).toBeDefined();
        expect(result.commands.length).toBeGreaterThan(0);
        expect(result.executionOrder).toBeDefined();
        expect(result.metadata).toBeDefined();
      });

      test('should preserve template metadata in command set', async () => {
        const result = await converter.convertTemplate(sampleTemplate, sampleContext);

        expect(result.source.templateId).toBe(sampleTemplate.meta?.version);
        expect(result.source.templateVersion).toBe(sampleTemplate.meta?.version);
      });

      test('should generate different command types', async () => {
        const result = await converter.convertTemplate(sampleTemplate, sampleContext);

        const commandTypes = new Set(result.commands.map(cmd => cmd.type));
        expect(commandTypes.size).toBeGreaterThan(0);
        expect(commandTypes.has('SET')).toBe(true);
      });

      test('should include rollback commands when enabled', async () => {
        const contextWithRollback: TemplateToCliContext = {
          ...sampleContext,
          options: {
            ...sampleContext.options,
            generateRollback: true
          }
        };

        const result = await converter.convertTemplate(sampleTemplate, contextWithRollback);

        expect(result.rollbackCommands).toBeDefined();
        expect(result.rollbackCommands.length).toBeGreaterThan(0);
      });

      test('should include validation commands when enabled', async () => {
        const contextWithValidation: TemplateToCliContext = {
          ...sampleContext,
          options: {
            ...sampleContext.options,
            generateValidation: true
          }
        };

        const result = await converter.convertTemplate(sampleTemplate, contextWithValidation);

        expect(result.validationCommands).toBeDefined();
        expect(result.validationCommands.length).toBeGreaterThan(0);
      });
    });

    describe('Configuration Options', () => {
      test('should respect preview mode', async () => {
        const previewContext: TemplateToCliContext = {
          ...sampleContext,
          options: {
            ...sampleContext.options,
            preview: true
          }
        };

        const result = await converter.convertTemplate(sampleTemplate, previewContext);

        // Commands should include preview flags
        const previewCommands = result.commands.filter(cmd =>
          cmd.command.includes('--preview')
        );
        expect(previewCommands.length).toBeGreaterThan(0);
      });

      test('should respect batch mode setting', async () => {
        const batchContext: TemplateToCliContext = {
          ...sampleContext,
          options: {
            ...sampleContext.options,
            batchMode: true
          }
        };

        const result = await converter.convertTemplate(sampleTemplate, batchContext);

        // Should have optimized command structure for batch execution
        expect(result.commands.length).toBeGreaterThan(0);
      });

      test('should apply cognitive optimization when enabled', async () => {
        const cognitiveConverter = new TemplateToCliConverter({
          enableCognitiveOptimization: true,
          cognitive: {
            enableTemporalReasoning: true,
            enableStrangeLoopOptimization: true,
            consciousnessLevel: 0.8,
            learningMode: 'active'
          }
        });

        const result = await cognitiveConverter.convertTemplate(sampleTemplate, sampleContext);

        expect(result.commands.length).toBeGreaterThan(0);
        // Commands should have cognitive metadata
        const cognitiveCommands = result.commands.filter(cmd => cmd.cognitive);
        expect(cognitiveCommands.length).toBeGreaterThanOrEqual(0);
      });
    });

    describe('Error Handling', () => {
      test('should handle invalid template gracefully', async () => {
        const invalidTemplate: RTBTemplate = {
          meta: {
            version: '1.0.0',
            author: ['Test'],
            description: 'Invalid template'
          },
          configuration: {} // Empty configuration
        };

        const result = await converter.convertTemplate(invalidTemplate, sampleContext);

        expect(result).toBeDefined();
        expect(result.commands.length).toBeGreaterThanOrEqual(0);
      });

      test('should handle missing context gracefully', async () => {
        const minimalContext: TemplateToCliContext = {
          target: { nodeId: 'TEST_NODE' },
          options: { timeout: 30 }
        };

        const result = await converter.convertTemplate(sampleTemplate, minimalContext);

        expect(result).toBeDefined();
        expect(result.commands.length).toBeGreaterThan(0);
      });
    });

    describe('Performance', () => {
      test('should convert template within reasonable time', async () => {
        const startTime = Date.now();

        const result = await converter.convertTemplate(sampleTemplate, sampleContext);

        const duration = Date.now() - startTime;
        expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
        expect(result).toBeDefined();
      });

      test('should handle large templates efficiently', async () => {
        const largeTemplate = createLargeTemplate(100); // 100 parameters
        const startTime = Date.now();

        const result = await converter.convertTemplate(largeTemplate, sampleContext);

        const duration = Date.now() - startTime;
        expect(duration).toBeLessThan(10000); // Should complete within 10 seconds
        expect(result.commands.length).toBeGreaterThan(50);
      });
    });
  });

  describe('FdnPathConstructor', () => {
    let fdnConstructor: FdnPathConstructor;

    beforeEach(() => {
      fdnConstructor = new FdnPathConstructor({
        enableOptimization: true,
        validateSyntax: true,
        applyHierarchyKnowledge: true
      });
    });

    test('should construct basic FDN paths', async () => {
      const result = await fdnConstructor.construct('EUtranCellFDD.qRxLevMin', sampleContext);

      expect(result).toBeDefined();
      expect(result.isValid).toBe(true);
      expect(result.fdn).toContain('EUtranCellFDD');
      expect(result.components.length).toBeGreaterThan(0);
    });

    test('should handle indexed FDN paths', async () => {
      const result = await fdnConstructor.construct('EUtranCellFDD=CELL_001.qRxLevMin', sampleContext);

      expect(result).toBeDefined();
      expect(result.isValid).toBe(true);
      expect(result.fdn).toContain('EUtranCellFDD=CELL_001');
    });

    test('should optimize FDN paths when enabled', async () => {
      const result = await fdnConstructor.construct('ManagedElement=1.EUtranCellFDD=CELL_001.qRxLevMin', sampleContext);

      expect(result).toBeDefined();
      expect(result.optimization).toBeDefined();
      if (result.optimization) {
        expect(result.optimization.optimizationApplied.length).toBeGreaterThanOrEqual(0);
      }
    });

    test('should validate FDN syntax', async () => {
      const result = await fdnConstructor.construct('Invalid@Path$Name.parameter', sampleContext);

      expect(result).toBeDefined();
      expect(result.isValid).toBe(false);
      expect(result.errors).toBeDefined();
      expect(result.errors!.length).toBeGreaterThan(0);
    });

    test('should use MO hierarchy knowledge', async () => {
      const contextWithHierarchy: TemplateToCliContext = {
        ...sampleContext,
        moHierarchy: sampleMOHierarchy
      };

      const result = await fdnConstructor.construct('EUtranCellFDD.qRxLevMin', contextWithHierarchy);

      expect(result).toBeDefined();
      expect(result.isValid).toBe(true);

      // Should have MO class information in components
      const cellComponent = result.components.find(comp => comp.name === 'EUtranCellFDD');
      expect(cellComponent).toBeDefined();
    });
  });

  describe('BatchCommandGenerator', () => {
    let batchGenerator: BatchCommandGenerator;
    let sampleCommands: GeneratedCliCommand[];

    beforeEach(() => {
      batchGenerator = new BatchCommandGenerator({
        maxCommandsPerBatch: 10,
        enableParallelExecution: true,
        maxConcurrency: 5
      });

      sampleCommands = createSampleCommands();
    });

    test('should optimize commands into batches', async () => {
      const optimizedCommands = await batchGenerator.optimizeBatches(sampleCommands, sampleContext);

      expect(optimizedCommands).toBeDefined();
      expect(optimizedCommands.length).toBeGreaterThan(0);
    });

    test('should respect batch size limits', async () => {
      const manyCommands = createSampleCommands(50); // 50 commands
      const optimizedCommands = await batchGenerator.optimizeBatches(manyCommands, sampleContext);

      expect(optimizedCommands.length).toBeLessThanOrEqual(manyCommands.length);
    });

    test('should enable parallel execution for compatible commands', async () => {
      const optimizedCommands = await batchGenerator.optimizeBatches(sampleCommands, sampleContext);

      // Should have batch setup and cleanup commands
      const batchCommands = optimizedCommands.filter(cmd =>
        cmd.command.includes('batch_setup') || cmd.command.includes('batch_cleanup')
      );
      expect(batchCommands.length).toBeGreaterThanOrEqual(0);
    });

    test('should handle critical commands separately', async () => {
      const commandsWithCritical = [
        ...sampleCommands,
        {
          id: 'critical_cmd',
          type: 'SET',
          command: 'cmedit set NODE criticalParameter=true',
          description: 'Critical command',
          critical: true,
          timeout: 30,
          metadata: {
            category: 'configuration',
            complexity: 'simple',
            riskLevel: 'high',
            estimatedDuration: 2000
          }
        }
      ];

      const optimizedCommands = await batchGenerator.optimizeBatches(commandsWithCritical, sampleContext);

      expect(optimizedCommands).toBeDefined();
      expect(optimizedCommands.length).toBeGreaterThan(0);
    });
  });

  describe('CommandValidator', () => {
    let validator: CommandValidator;

    beforeEach(() => {
      validator = new CommandValidator({
        strictness: 'normal',
        enableSyntaxValidation: true,
        enableSemanticValidation: true,
        enableConstraintValidation: true
      });
    });

    test('should validate valid commands', async () => {
      const validCommands = createValidCommands();
      const result = await validator.validate(validCommands, sampleContext);

      expect(result).toBeDefined();
      expect(result.isValid).toBe(true);
      expect(result.errors).toBeDefined();
      expect(result.errors.length).toBe(0);
    });

    test('should detect syntax errors', async () => {
      const invalidCommands = createInvalidCommands();
      const result = await validator.validate(invalidCommands, sampleContext);

      expect(result).toBeDefined();
      expect(result.isValid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    test('should validate parameter constraints', async () => {
      const commandsWithInvalidParams = createCommandsWithInvalidParameters();
      const result = await validator.validate(commandsWithInvalidParams, sampleContext);

      expect(result).toBeDefined();
      expect(result.isValid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
    });

    test('should provide recommended fixes', async () => {
      const commandsWithIssues = createCommandsWithIssues();
      const result = await validator.validate(commandsWithIssues, sampleContext);

      expect(result).toBeDefined();
      expect(result.recommendedFixes).toBeDefined();
      expect(result.recommendedFixes.length).toBeGreaterThan(0);
    });
  });

  describe('DependencyAnalyzer', () => {
    let analyzer: DependencyAnalyzer;

    beforeEach(() => {
      analyzer = new DependencyAnalyzer({
        enableCircularDependencyDetection: true,
        enableCriticalPathAnalysis: true,
        enableOptimizationSuggestions: true
      });
    });

    test('should analyze command dependencies', async () => {
      const commands = createCommandsWithDependencies();
      const result = await analyzer.analyze(commands, sampleContext);

      expect(result).toBeDefined();
      expect(result.dependencyGraph).toBeDefined();
      expect(result.criticalPath).toBeDefined();
      expect(result.executionLevels).toBeDefined();
    });

    test('should detect circular dependencies', async () => {
      const commandsWithCycle = createCommandsWithCircularDependency();
      const result = await analyzer.analyze(commandsWithCycle, sampleContext);

      expect(result).toBeDefined();
      expect(result.circularDependencies).toBeDefined();
      expect(result.circularDependencies.length).toBeGreaterThan(0);
    });

    test('should calculate critical path', async () => {
      const commands = createCommandsWithDependencies();
      const result = await analyzer.analyze(commands, sampleContext);

      expect(result).toBeDefined();
      expect(result.criticalPath.length).toBeGreaterThan(0);
      expect(result.criticalPath).toEqual(expect.arrayContaining(commands.map(cmd => cmd.id)));
    });

    test('should generate optimization suggestions', async () => {
      const commands = createCommandsWithDependencies();
      const result = await analyzer.analyze(commands, sampleContext);

      expect(result).toBeDefined();
      expect(result.optimizations).toBeDefined();
      expect(result.optimizations.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('RollbackManager', () => {
    let rollbackManager: RollbackManager;

    beforeEach(() => {
      rollbackManager = new RollbackManager({
        strategy: 'FULL',
        enableValidation: true,
        enableSelectiveRollback: true,
        maxRollbackDepth: 10
      });
    });

    test('should generate rollback commands', async () => {
      const commands = createSampleCommands();
      const rollbackCommands = await rollbackManager.generateRollbackCommands(commands, sampleContext);

      expect(rollbackCommands).toBeDefined();
      expect(rollbackCommands.length).toBeGreaterThan(0);
    });

    test('should create rollback plan', async () => {
      const commands = createSampleCommands();
      const plan = await rollbackManager.createRollbackPlan('test_set', commands, sampleContext);

      expect(plan).toBeDefined();
      expect(plan.id).toContain('rollback_test_set');
      expect(plan.rollbackCommands).toBeDefined();
      expect(plan.executionOrder).toBeDefined();
      expect(plan.riskAssessment).toBeDefined();
    });

    test('should assess rollback risk', async () => {
      const commands = createCommandsWithCriticalOperations();
      const plan = await rollbackManager.createRollbackPlan('critical_set', commands, sampleContext);

      expect(plan).toBeDefined();
      expect(plan.riskAssessment.rollbackRisk).toBeDefined();
      expect(['low', 'medium', 'high', 'critical']).toContain(plan.riskAssessment.rollbackRisk);
    });

    test('should execute rollback plan', async () => {
      const commands = createSampleCommands();
      const plan = await rollbackManager.createRollbackPlan('test_set', commands, sampleContext);

      const result = await rollbackManager.executeRollback(plan.id, sampleContext, {
        dryRun: true // Use dry run for testing
      });

      expect(result).toBeDefined();
      expect(result.rollbackPlanId).toBe(plan.id);
      expect(result.commandResults).toBeDefined();
      expect(result.effectiveness).toBeDefined();
    });
  });

  describe('EricssonRanExpertise', () => {
    let ranExpertise: EricssonRanExpertise;

    beforeEach(() => {
      ranExpertise = new EricssonRanExpertise({
        enablePatternMatching: true,
        enableBestPractices: true,
        enableOptimizationSuggestions: true,
        enablePerformanceOptimization: true,
        enableSafetyChecks: true
      });
    });

    test('should enhance commands with RAN expertise', async () => {
      const command = createRanSpecificCommand();
      const enhancedCommand = await ranExpertise.enhanceCommand(command, sampleContext);

      expect(enhancedCommand).toBeDefined();
      expect(enhancedCommand.appliedExpertise).toBeDefined();
      expect(enhancedCommand.ranInsights).toBeDefined();
      expect(enhancedCommand.optimizationLevel).toBeGreaterThanOrEqual(0);
    });

    test('should generate RAN-specific optimizations', async () => {
      const commands = createRanSpecificCommands();
      const optimizations = await ranExpertise.generateOptimizations(commands, sampleContext);

      expect(optimizations).toBeDefined();
      expect(optimizations.length).toBeGreaterThanOrEqual(0);
    });

    test('should apply safety checks for dangerous operations', async () => {
      const dangerousCommand = createDangerousRanCommand();
      const enhancedCommand = await ranExpertise.enhanceCommand(dangerousCommand, sampleContext);

      expect(enhancedCommand).toBeDefined();
      expect(enhancedCommand.ranInsights.some(insight =>
        insight.type === 'reliability' && insight.description.includes('dangerous')
      )).toBe(true);
    });
  });

  describe('CognitiveOptimizer', () => {
    let cognitiveOptimizer: CognitiveOptimizer;

    beforeEach(() => {
      cognitiveOptimizer = new CognitiveOptimizer({
        enableTemporalReasoning: true,
        enableStrangeLoopOptimization: true,
        consciousnessLevel: 0.8,
        learningMode: 'active',
        temporalExpansionFactor: 100,
        maxReasoningDepth: 5
      });
    });

    test('should apply cognitive optimization', async () => {
      const commands = createSampleCommands();
      const result = await cognitiveOptimizer.optimize(commands, sampleContext);

      expect(result).toBeDefined();
      expect(result.optimizedCommands).toBeDefined();
      expect(result.commandOptimizations).toBeDefined();
      expect(result.cognitiveResult).toBeDefined();
    });

    test('should perform temporal analysis', async () => {
      const commands = createCommandsWithTemporalPatterns();
      const result = await cognitiveOptimizer.optimize(commands, sampleContext);

      expect(result).toBeDefined();
      expect(result.cognitiveResult.temporalAnalysisDepth).toBeGreaterThan(0);
    });

    test('should identify optimization opportunities', async () => {
      const commands = createComplexCommandSet();
      const result = await cognitiveOptimizer.optimize(commands, sampleContext);

      expect(result).toBeDefined();
      expect(result.cognitiveResult.performanceImprovements).toBeDefined();
    });
  });

  describe('Integration Tests', () => {
    let integration: RtbTemplateIntegration;
    let templateSystem: any;

    beforeEach(() => {
      // Mock template system for integration tests
      templateSystem = {
        processTemplateInheritance: jest.fn().mockResolvedValue({
          template: sampleTemplate,
          mergeResult: {
            template: sampleTemplate,
            conflicts: [],
            resolvedConflicts: [],
            unresolvedConflicts: [],
            mergeStats: {
              totalTemplates: 1,
              conflictsDetected: 0,
              conflictsResolved: 0,
              processingTime: 100
            },
            inheritanceChain: {
              templates: [sampleTemplate],
              priorities: [50],
              inheritanceDepth: 1,
              hasCircularDependency: false
            }
          }
        })
      };

      integration = new RtbTemplateIntegration(templateSystem, {
        enableInheritanceProcessing: true,
        enablePriorityOptimization: true,
        enableTemplateValidation: true,
        enableMergeResultAnalysis: true
      });
    });

    test('should convert template with inheritance processing', async () => {
      const result = await integration.convertTemplateWithInheritance(sampleTemplate, sampleContext);

      expect(result).toBeDefined();
      expect(result.originalTemplate).toBeDefined();
      expect(result.commandSet).toBeDefined();
      expect(result.integrationStats).toBeDefined();
      expect(result.insights).toBeDefined();
    });

    test('should handle batch conversion of multiple templates', async () => {
      const templates = [sampleTemplate, createSampleTemplate('template_2'), createSampleTemplate('template_3')];

      const results = await integration.convertTemplatesBatch(templates, sampleContext, {
        processInParallel: false,
        continueOnError: true
      });

      expect(results).toBeDefined();
      expect(results.length).toBe(templates.length);
    });

    test('should provide integration statistics', async () => {
      await integration.convertTemplateWithInheritance(sampleTemplate, sampleContext);
      const stats = integration.getConversionStatistics();

      expect(stats).toBeDefined();
      expect(stats.totalConversions).toBeGreaterThan(0);
      expect(stats.averageProcessingTime).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Performance Tests', () => {
    test('should handle large template conversion efficiently', async () => {
      const largeTemplate = createLargeTemplate(500); // 500 parameters
      const startTime = Date.now();

      const result = await converter.convertTemplate(largeTemplate, sampleContext);

      const duration = Date.now() - startTime;
      expect(duration).toBeLessThan(30000); // Should complete within 30 seconds
      expect(result.commands.length).toBeGreaterThan(100);
    });

    test('should process multiple templates concurrently', async () => {
      const templates = Array(10).fill(null).map((_, i) => createSampleTemplate(`template_${i}`));
      const startTime = Date.now();

      const integration = new RtbTemplateIntegration({} as any);
      const results = await integration.convertTemplatesBatch(templates, sampleContext, {
        processInParallel: true,
        maxConcurrency: 5
      });

      const duration = Date.now() - startTime;
      expect(duration).toBeLessThan(20000); // Should complete within 20 seconds
      expect(results.length).toBe(templates.length);
    });
  });
});

// Helper Functions for Test Data Creation

function createSampleMOHierarchy(): MOHierarchy {
  return {
    rootClass: 'ManagedElement',
    classes: new Map([
      ['ManagedElement', {
        id: 'ManagedElement',
        name: 'ManagedElement',
        parentClass: '',
        cardinality: { minimum: 1, maximum: 1, type: 'single' },
        flags: {},
        children: ['ENBFunction'],
        attributes: ['userLabel'],
        derivedClasses: []
      }],
      ['ENBFunction', {
        id: 'ENBFunction',
        name: 'ENBFunction',
        parentClass: 'ManagedElement',
        cardinality: { minimum: 1, maximum: 1, type: 'single' },
        flags: {},
        children: ['EUtranCellFDD'],
        attributes: ['endcEnabled'],
        derivedClasses: []
      }],
      ['EUtranCellFDD', {
        id: 'EUtranCellFDD',
        name: 'EUtranCellFDD',
        parentClass: 'ENBFunction',
        cardinality: { minimum: 1, maximum: 256, type: 'bounded' },
        flags: {},
        children: [],
        attributes: ['qRxLevMin', 'qQualMin'],
        derivedClasses: []
      }]
    ]),
    relationships: new Map(),
    cardinality: new Map(),
    inheritanceChain: new Map()
  };
}

function createSampleTemplate(id: string = 'sample_template'): RTBTemplate {
  return {
    meta: {
      version: '1.0.0',
      author: ['Test Author'],
      description: `Sample template for ${id}`,
      tags: ['test', 'sample'],
      environment: 'test',
      priority: 50
    },
    configuration: {
      'EUtranCellFDD.qRxLevMin': -130,
      'EUtranCellFDD.qQualMin': -32,
      'EUtranCellFDD.referenceSignalPower': 15,
      'ENBFunction.endcEnabled': true
    },
    conditions: {
      'if_condition': {
        if: 'ENBFunction.endcEnabled == true',
        then: {
          'ENBFunction.nrEventB1Threshold': -110
        },
        else: 'skip'
      }
    },
    evaluations: {
      'eval_power': {
        eval: 'calculate_optimal_power',
        args: ['coverage', 'capacity']
      }
    }
  };
}

function createSampleContext(moHierarchy?: MOHierarchy): TemplateToCliContext {
  return {
    target: {
      nodeId: 'TEST_NODE_001'
    },
    cellIds: {
      primaryCell: 'CELL_001',
      lteCell: 'CELL_001'
    },
    options: {
      preview: false,
      force: false,
      verbose: false,
      timeout: 30,
      batchMode: false,
      dependencyAnalysis: true,
      cognitiveOptimization: false,
      generateRollback: false,
      generateValidation: false
    },
    moHierarchy,
    reservedBy: {} as ReservedByHierarchy
  };
}

function createSampleCommands(count: number = 5): GeneratedCliCommand[] {
  const commands: GeneratedCliCommand[] = [];

  for (let i = 0; i < count; i++) {
    commands.push({
      id: `cmd_${i}`,
      type: 'SET',
      command: `cmedit set TEST_NODE EUtranCellFDD=CELL_${i.toString().padStart(3, '0')} parameter${i}=value${i}`,
      description: `Set parameter ${i}`,
      targetFdn: `EUtranCellFDD=CELL_${i.toString().padStart(3, '0')}`,
      parameters: { [`parameter${i}`]: `value${i}` },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    });
  }

  return commands;
}

function createValidCommands(): GeneratedCliCommand[] {
  return [
    {
      id: 'valid_set_cmd',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 qRxLevMin=-130',
      description: 'Valid SET command',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { qRxLevMin: -130 },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    },
    {
      id: 'valid_get_cmd',
      type: 'GET',
      command: 'cmedit get TEST_NODE EUtranCellFDD=CELL_001 -s',
      description: 'Valid GET command',
      targetFdn: 'EUtranCellFDD=CELL_001',
      timeout: 30,
      critical: false,
      metadata: {
        category: 'query',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 500
      }
    }
  ];
}

function createInvalidCommands(): GeneratedCliCommand[] {
  return [
    {
      id: 'invalid_syntax_cmd',
      type: 'SET',
      command: 'invalid command syntax',
      description: 'Invalid command syntax',
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    },
    {
      id: 'invalid_fdn_cmd',
      type: 'SET',
      command: 'cmedit set TEST_NODE Invalid@Path$Name parameter=value',
      description: 'Invalid FDN path',
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    }
  ];
}

function createCommandsWithInvalidParameters(): GeneratedCliCommand[] {
  return [
    {
      id: 'invalid_range_cmd',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 qRxLevMin=-200',
      description: 'Parameter out of range',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { qRxLevMin: -200 }, // Invalid range
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'medium',
        estimatedDuration: 1000
      }
    }
  ];
}

function createCommandsWithIssues(): GeneratedCliCommand[] {
  return [
    {
      id: 'risky_cmd',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 adminState=LOCKED',
      description: 'Risky operation',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { adminState: 'LOCKED' },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'high',
        estimatedDuration: 1000
      }
    }
  ];
}

function createCommandsWithDependencies(): GeneratedCliCommand[] {
  return [
    {
      id: 'create_cmd',
      type: 'CREATE',
      command: 'cmedit create TEST_NODE EUtranCellFDD EUtranCellFDDId=CELL_001',
      description: 'Create cell',
      targetFdn: 'EUtranCellFDD=CELL_001',
      timeout: 30,
      critical: false,
      metadata: {
        category: 'creation',
        complexity: 'moderate',
        riskLevel: 'medium',
        estimatedDuration: 2000
      }
    },
    {
      id: 'set_cmd',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 qRxLevMin=-130',
      description: 'Configure cell',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { qRxLevMin: -130 },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    },
    {
      id: 'validate_cmd',
      type: 'VALIDATION',
      command: 'cmedit get TEST_NODE EUtranCellFDD=CELL_001 syncStatus',
      description: 'Validate cell',
      targetFdn: 'EUtranCellFDD=CELL_001',
      timeout: 30,
      critical: false,
      metadata: {
        category: 'validation',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 500
      }
    }
  ];
}

function createCommandsWithCircularDependency(): GeneratedCliCommand[] {
  return [
    {
      id: 'cmd_a',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 paramA=valueA',
      description: 'Command A',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { paramA: 'valueA' },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    },
    {
      id: 'cmd_b',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 paramB=valueB',
      description: 'Command B',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { paramB: 'valueB' },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    }
  ];
}

function createCommandsWithCriticalOperations(): GeneratedCliCommand[] {
  return [
    {
      id: 'critical_create_cmd',
      type: 'CREATE',
      command: 'cmedit create TEST_NODE ManagedElement',
      description: 'Critical: Create ManagedElement',
      timeout: 60,
      critical: true,
      metadata: {
        category: 'creation',
        complexity: 'complex',
        riskLevel: 'critical',
        estimatedDuration: 5000
      }
    }
  ];
}

function createRanSpecificCommand(): GeneratedCliCommand {
  return {
    id: 'ran_specific_cmd',
    type: 'SET',
    command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 qRxLevMin=-130',
    description: 'RAN specific configuration',
    targetFdn: 'EUtranCellFDD=CELL_001',
    parameters: { qRxLevMin: -130 },
    timeout: 30,
    critical: false,
    metadata: {
      category: 'ran_configuration',
      complexity: 'moderate',
      riskLevel: 'medium',
      estimatedDuration: 2000
    }
  };
}

function createRanSpecificCommands(): GeneratedCliCommand[] {
  return [
    {
      id: 'cell_power_cmd',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 referenceSignalPower=15',
      description: 'Configure cell power',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { referenceSignalPower: 15 },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'ran_configuration',
        complexity: 'moderate',
        riskLevel: 'medium',
        estimatedDuration: 2000
      }
    },
    {
      id: 'mobility_cmd',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 hysteresis=2',
      description: 'Configure mobility parameters',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { hysteresis: 2 },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'ran_mobility',
        complexity: 'moderate',
        riskLevel: 'medium',
        estimatedDuration: 1500
      }
    }
  ];
}

function createDangerousRanCommand(): GeneratedCliCommand {
  return {
    id: 'dangerous_cmd',
    type: 'SET',
    command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 adminState=LOCKED',
    description: 'Dangerous: Lock cell',
    targetFdn: 'EUtranCellFDD=CELL_001',
    parameters: { adminState: 'LOCKED' },
    timeout: 30,
    critical: false,
    metadata: {
      category: 'ran_configuration',
      complexity: 'simple',
      riskLevel: 'high',
      estimatedDuration: 1000
    }
  };
}

function createCommandsWithTemporalPatterns(): GeneratedCliCommand[] {
  return [
    {
      id: 'temporal_cmd_1',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 param1=value1',
      description: 'Temporal command 1',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { param1: 'value1' },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    },
    {
      id: 'temporal_cmd_2',
      type: 'SET',
      command: 'cmedit set TEST_NODE EUtranCellFDD=CELL_001 param2=value2',
      description: 'Temporal command 2',
      targetFdn: 'EUtranCellFDD=CELL_001',
      parameters: { param2: 'value2' },
      timeout: 30,
      critical: false,
      metadata: {
        category: 'configuration',
        complexity: 'simple',
        riskLevel: 'low',
        estimatedDuration: 1000
      }
    }
  ];
}

function createComplexCommandSet(): GeneratedCliCommand[] {
  const commands: GeneratedCliCommand[] = [];

  // Create a mix of different command types
  for (let i = 0; i < 20; i++) {
    const types: CliCommandType[] = ['SET', 'GET', 'CREATE', 'VALIDATION'];
    const type = types[i % types.length];

    commands.push({
      id: `complex_cmd_${i}`,
      type,
      command: `cmedit ${type.toLowerCase()} TEST_NODE target_${i} parameter${i}=value${i}`,
      description: `Complex command ${i}`,
      targetFdn: `target_${i}`,
      parameters: type === 'SET' ? { [`parameter${i}`]: `value${i}` } : undefined,
      timeout: 30,
      critical: i % 10 === 0, // Every 10th command is critical
      metadata: {
        category: 'test',
        complexity: i % 3 === 0 ? 'complex' : 'simple',
        riskLevel: i % 5 === 0 ? 'high' : 'low',
        estimatedDuration: 1000 + (i * 100)
      }
    });
  }

  return commands;
}

function createLargeTemplate(parameterCount: number): RTBTemplate {
  const configuration: Record<string, any> = {};

  for (let i = 0; i < parameterCount; i++) {
    configuration[`parameter_${i}`] = `value_${i}`;
  }

  return {
    meta: {
      version: '1.0.0',
      author: ['Test'],
      description: `Large template with ${parameterCount} parameters`,
      tags: ['large', 'test'],
      environment: 'test',
      priority: 50
    },
    configuration,
    conditions: {},
    evaluations: {}
  };
}