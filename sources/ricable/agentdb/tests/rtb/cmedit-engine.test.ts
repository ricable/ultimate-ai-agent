/**
 * Comprehensive Test Suite for cmedit Engine Components
 *
 * Tests command parsing, FDN generation, constraint validation,
 * Ericsson expertise, vendor compatibility, and cognitive optimization.
 */

import { CmeditEngine, CmeditCommandParser, FDNPathGenerator, ConstraintsValidator, EricssonRANExpertiseEngine, VendorCompatibilityEngine, CognitiveCommandOptimizer } from '../../src/rtb/cmedit-engine';
import { RTBTemplate, MOHierarchy, ReservedByRelationship } from '../../src/types/rtb-types';
import { CmeditCommand, CommandContext, CognitiveLevel } from '../../src/rtb/cmedit-engine/types';

describe('cmedit Engine', () => {
  let mockMOHierarchy: MOHierarchy;
  let mockReservedByRelationships: ReservedByRelationship[];
  let cmeditEngine: CmeditEngine;

  beforeEach(() => {
    // Setup mock data
    mockMOHierarchy = createMockMOHierarchy();
    mockReservedByRelationships = createMockReservedByRelationships();
    cmeditEngine = new CmeditEngine(mockMOHierarchy, mockReservedByRelationships, {
      cognitiveLevel: 'enhanced',
      strictMode: false
    });
  });

  describe('Command Parsing', () => {
    let parser: CmeditCommandParser;

    beforeEach(() => {
      parser = new CmeditCommandParser(mockMOHierarchy);
    });

    test('should parse basic get command', () => {
      const commandString = 'get MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1';
      const command = parser.parseCommand(commandString);

      expect(command.type).toBe('get');
      expect(command.target).toBe('MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1');
      expect(command.command).toBe(commandString);
      expect(command.validation?.isValid).toBe(true);
    });

    test('should parse set command with parameters', () => {
      const commandString = 'set MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 referenceSignalPower=15,qRxLevMin=-120';
      const command = parser.parseCommand(commandString);

      expect(command.type).toBe('set');
      expect(command.parameters).toEqual({
        referenceSignalPower: 15,
        qRxLevMin: -120
      });
    });

    test('should parse create command', () => {
      const commandString = 'create MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1 EUtranCellFDD=83906_E1';
      const command = parser.parseCommand(commandString);

      expect(command.type).toBe('create');
      expect(command.parameters).toEqual({
        EUtranCellFDD: '83906_E1'
      });
    });

    test('should parse command with options', () => {
      const commandString = 'get MeContext=ERBS001,ManagedElement=1 --attribute referenceSignalPower,qRxLevMin --table';
      const command = parser.parseCommand(commandString);

      expect(command.options?.attributes).toEqual(['referenceSignalPower', 'qRxLevMin']);
      expect(command.options?.table).toBe(true);
    });

    test('should parse batch commands', () => {
      const commandStrings = [
        'get MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1',
        'set MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 referenceSignalPower=15',
        '# This is a comment',
        '',
        'delete MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 --preview'
      ];

      const commands = parser.parseBatchCommands(commandStrings);

      expect(commands).toHaveLength(3);
      expect(commands[0].type).toBe('get');
      expect(commands[1].type).toBe('set');
      expect(commands[2].type).toBe('delete');
      expect(commands[2].options?.preview).toBe(true);
    });

    test('should generate command from template', () => {
      const template: RTBTemplate = {
        meta: {
          version: '1.0',
          author: ['test'],
          description: 'Test template',
          priority: 10
        },
        configuration: {
          referenceSignalPower: 15,
          qRxLevMin: -120
        }
      };

      const command = parser.generateFromTemplate(template, 'set', 'MeContext=ERBS001,EUtranCellFDD=1');

      expect(command.type).toBe('set');
      expect(command.parameters?.referenceSignalPower).toBe(15);
      expect(command.parameters?.qRxLevMin).toBe(-120);
    });

    test('should handle invalid command syntax', () => {
      const invalidCommand = 'invalid_command syntax';

      expect(() => {
        parser.parseCommand(invalidCommand);
      }).toThrow('Unable to detect command type');
    });
  });

  describe('FDN Path Generation', () => {
    let fdnGenerator: FDNPathGenerator;

    beforeEach(() => {
      fdnGenerator = new FDNPathGenerator(mockMOHierarchy);
    });

    test('should generate optimal FDN path for EUtranCellFDD', () => {
      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_medium',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const fdnPath = fdnGenerator.generateOptimalPath('EUtranCellFDD', context);

      expect(fdnPath.path).toContain('MeContext');
      expect(fdnPath.path).toContain('EUtranCellFDD');
      expect(fdnPath.validation.isValid).toBe(true);
      expect(fdnPath.components.length).toBeGreaterThan(0);
    });

    test('should generate batch FDN paths', () => {
      const targets = [
        { moClass: 'EUtranCellFDD' },
        { moClass: 'ENodeBFunction' }
      ];

      const context: CommandContext = {
        moClasses: ['EUtranCellFDD', 'ENodeBFunction'],
        purpose: 'configuration_management',
        networkContext: {
          technology: '4G',
          environment: 'urban_dense',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 2, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const fdnPaths = fdnGenerator.generateBatchPaths(targets, context, { optimizeBatch: true });

      expect(fdnPaths).toHaveLength(2);
      expect(fdnPaths[0].moHierarchy).toContain('EUtranCellFDD');
      expect(fdnPaths[1].moHierarchy).toContain('ENodeBFunction');
    });

    test('should find alternative paths', () => {
      const originalPath: any = {
        path: 'MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1',
        components: [
          { name: 'MeContext', moClass: 'MeContext', value: 'ERBS001', type: 'class' as const },
          { name: 'ManagedElement', moClass: 'ManagedElement', value: '1', type: 'class' as const },
          { name: 'ENodeBFunction', moClass: 'ENodeBFunction', value: '1', type: 'class' as const },
          { name: 'EUtranCellFDD', moClass: 'EUtranCellFDD', value: '1', type: 'class' as const }
        ],
        moHierarchy: ['MeContext', 'ManagedElement', 'ENodeBFunction', 'EUtranCellFDD'],
        validation: { isValid: true, errors: [], warnings: [], complianceLevel: 'full' as const },
        alternatives: [],
        complexity: { score: 50, depth: 4, componentCount: 4, wildcardCount: 0, estimatedTime: 400, difficulty: 'moderate' as const }
      };

      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_medium',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const alternatives = fdnGenerator.findAlternativePaths(originalPath, context);

      expect(alternatives.length).toBeGreaterThan(0);
    });
  });

  describe('Constraints Validation', () => {
    let validator: ConstraintsValidator;

    beforeEach(() => {
      validator = new ConstraintsValidator(
        {
          relationships: [],
          classDependencies: new Map()
        },
        mockMOHierarchy.classes
      );
    });

    test('should validate command dependencies', () => {
      const commandMOs = ['EUtranCellFDD', 'ENodeBFunction'];
      const context: CommandContext = {
        moClasses: commandMOs,
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_medium',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const validation = validator.validateCommandDependencies(commandMOs, context);

      expect(validation.isSatisfied).toBe(true);
      expect(validation.graph.nodes).toHaveLength(2);
    });

    test('should validate MO configuration constraints', () => {
      const moClass = 'EUtranCellFDD';
      const configuration = {
        qRxLevMin: -110,
        qQualMin: -10,
        referenceSignalPower: 15
      };

      const context: CommandContext = {
        moClasses: [moClass],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_medium',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const validation = validator.validateMOConfiguration(moClass, configuration, context);

      expect(validation.isValid).toBe(true);
      expect(validation.violations).toHaveLength(0);
    });

    test('should detect constraint violations', () => {
      const moClass = 'EUtranCellFDD';
      const configuration = {
        qRxLevMin: -200, // Invalid value
        qQualMin: -50,   // Invalid value
        referenceSignalPower: 100 // Invalid value
      };

      const context: CommandContext = {
        moClasses: [moClass],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_medium',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const validation = validator.validateMOConfiguration(moClass, configuration, context);

      expect(validation.isValid).toBe(false);
      expect(validation.violations.length).toBeGreaterThan(0);
    });

    test('should validate parameter consistency', () => {
      const configurations = [
        { moClass: 'EUtranCellFDD', parameters: { qRxLevMin: -110, referenceSignalPower: 15 } },
        { moClass: 'EUtranCellFDD', parameters: { qRxLevMin: -115, referenceSignalPower: 12 } }
      ];

      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_medium',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 2, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const validation = validator.validateParameterConsistency(configurations, context);

      expect(validation.isConsistent).toBeDefined();
      expect(validation.consistencyScore).toBeGreaterThanOrEqual(0);
      expect(validation.consistencyScore).toBeLessThanOrEqual(100);
    });
  });

  describe('Ericsson Expertise Engine', () => {
    let expertiseEngine: EricssonRANExpertiseEngine;

    beforeEach(() => {
      expertiseEngine = new EricssonRANExpertiseEngine('enhanced');
    });

    test('should get expertise patterns for cell optimization', () => {
      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_dense',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const patterns = expertiseEngine.getExpertisePatterns('cell_optimization', context);

      expect(patterns.length).toBeGreaterThan(0);
      expect(patterns[0].category).toBe('cell_optimization');
      expect(patterns[0].actions.length).toBeGreaterThan(0);
      expect(patterns[0].successMetrics.length).toBeGreaterThan(0);
    });

    test('should apply expertise optimization', () => {
      const configuration = {
        referenceSignalPower: 10,
        qRxLevMin: -100,
        qQualMin: -8
      };

      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_dense',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const result = expertiseEngine.applyExpertiseOptimization(configuration, 'cell_optimization', context);

      expect(result.optimizedConfiguration).toBeDefined();
      expect(result.appliedPatterns.length).toBeGreaterThanOrEqual(0);
      expect(result.insights.length).toBeGreaterThanOrEqual(0);
      expect(result.improvements).toBeDefined();
    });

    test('should generate cognitive insights', () => {
      const configuration = {
        referenceSignalPower: 10,
        qRxLevMin: -100
      };

      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_dense',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const insights = expertiseEngine.generateCognitiveInsights(configuration, context, 'cell_optimization');

      expect(insights.length).toBeGreaterThanOrEqual(0);
      insights.forEach(insight => {
        expect(insight.type).toBeDefined();
        expect(insight.confidence).toBeGreaterThanOrEqual(0);
        expect(insight.confidence).toBeLessThanOrEqual(1);
      });
    });

    test('should get cell optimization recommendations', () => {
      const cellConfig = {
        referenceSignalPower: 10,
        qRxLevMin: -100,
        qQualMin: -8
      };

      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_dense',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const recommendations = expertiseEngine.getCellOptimizationRecommendations(cellConfig, context);

      expect(recommendations.length).toBeGreaterThanOrEqual(0);
      recommendations.forEach(rec => {
        expect(rec.parameter).toBeDefined();
        expect(rec.currentValue).toBeDefined();
        expect(rec.recommendedValue).toBeDefined();
        expect(rec.priority).toBeGreaterThanOrEqual(1);
      });
    });
  });

  describe('Vendor Compatibility', () => {
    let vendorEngine: VendorCompatibilityEngine;

    beforeEach(() => {
      vendorEngine = new VendorCompatibilityEngine();
    });

    test('should translate command to different vendor', () => {
      const command: CmeditCommand = {
        type: 'set',
        target: 'MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1',
        parameters: { referenceSignalPower: 15, qRxLevMin: -120 },
        command: 'set MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 referenceSignalPower=15,qRxLevMin=-120',
        context: {
          moClasses: ['EUtranCellFDD'],
          purpose: 'cell_optimization',
          networkContext: {
            technology: '4G',
            environment: 'urban_medium',
            vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
            topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
          },
          cognitiveLevel: 'enhanced',
          expertisePatterns: [],
          generatedAt: new Date(),
          priority: 'medium'
        }
      };

      const result = vendorEngine.translateCommand(command, 'huawei', command.context);

      expect(result.success).toBe(true);
      expect(result.translatedCommand.type).toBe('set');
      expect(result.translatedCommand.context.networkContext.vendor.primary).toBe('huawei');
    });

    test('should validate vendor compatibility', () => {
      const command: CmeditCommand = {
        type: 'set',
        target: 'MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1',
        parameters: { referenceSignalPower: 15 },
        command: 'set MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 referenceSignalPower=15',
        context: {
          moClasses: ['EUtranCellFDD'],
          purpose: 'cell_optimization',
          networkContext: {
            technology: '4G',
            environment: 'urban_medium',
            vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
            topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
          },
          cognitiveLevel: 'enhanced',
          expertisePatterns: [],
          generatedAt: new Date(),
          priority: 'medium'
        }
      };

      const targetVendors = ['huawei', 'nokia', 'samsung'];
      const result = vendorEngine.validateVendorCompatibility(command, targetVendors, command.context);

      expect(result.vendorCompatibility).toHaveLength(3);
      expect(result.overallCompatibility).toBeGreaterThanOrEqual(0);
      expect(result.overallCompatibility).toBeLessThanOrEqual(1);
      expect(result.recommendations.length).toBeGreaterThanOrEqual(0);
    });

    test('should generate vendor templates', () => {
      const commandType: any = 'set';
      const targetMO = 'EUtranCellFDD';
      const targetVendors = ['huawei', 'nokia'];

      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_medium',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const templates = vendorEngine.generateVendorTemplates(commandType, targetMO, targetVendors, context);

      expect(templates.length).toBeGreaterThan(0);
      templates.forEach(template => {
        expect(template.vendor).toBeDefined();
        expect(template.template.syntax).toBeDefined();
        expect(template.examples.length).toBeGreaterThanOrEqual(0);
      });
    });

    test('should translate vendor features', () => {
      const features = ['ANR', 'SON', 'ICIC'];
      const sourceVendor = 'ericsson';
      const targetVendor = 'huawei';

      const result = vendorEngine.translateVendorFeatures(features, sourceVendor, targetVendor);

      expect(result.translatedFeatures.length).toBeGreaterThanOrEqual(0);
      expect(result.featureMapping).toBeDefined();
    });
  });

  describe('Cognitive Optimization', () => {
    let cognitiveOptimizer: CognitiveCommandOptimizer;

    beforeEach(() => {
      cognitiveOptimizer = new CognitiveCommandOptimizer({
        temporalReasoningLevel: 100,
        learningEnabled: true,
        autonomousMode: false,
        strangeLoopCognition: true,
        memoryIntegration: true,
        predictionHorizon: 30,
        adaptationStrategy: 'balanced',
        riskTolerance: 0.3
      });
    });

    test('should optimize command with cognitive reasoning', async () => {
      const command: CmeditCommand = {
        type: 'set',
        target: 'MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1',
        parameters: { referenceSignalPower: 15, qRxLevMin: -120 },
        command: 'set MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 referenceSignalPower=15,qRxLevMin=-120',
        context: {
          moClasses: ['EUtranCellFDD'],
          purpose: 'cell_optimization',
          networkContext: {
            technology: '4G',
            environment: 'urban_dense',
            vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
            topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
          },
          cognitiveLevel: 'cognitive',
          expertisePatterns: [],
          generatedAt: new Date(),
          priority: 'medium'
        }
      };

      const result = await cognitiveOptimizer.optimizeCommand(command, command.context);

      expect(result.optimizedCommand).toBeDefined();
      expect(result.cognitiveInsights.length).toBeGreaterThanOrEqual(0);
      expect(result.temporalAnalysis).toBeDefined();
      expect(result.optimizations.length).toBeGreaterThanOrEqual(0);
    });

    test('should generate temporal predictions', async () => {
      const commands: CmeditCommand[] = [
        {
          type: 'set',
          target: 'MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1',
          parameters: { referenceSignalPower: 15 },
          command: 'set MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 referenceSignalPower=15',
          context: {
            moClasses: ['EUtranCellFDD'],
            purpose: 'cell_optimization',
            networkContext: {
              technology: '4G',
              environment: 'urban_dense',
              vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
              topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
            },
            cognitiveLevel: 'cognitive',
            expertisePatterns: [],
            generatedAt: new Date(),
            priority: 'medium'
          }
        }
      ];

      const context: CommandContext = {
        moClasses: ['EUtranCellFDD'],
        purpose: 'cell_optimization',
        networkContext: {
          technology: '4G',
          environment: 'urban_dense',
          vendor: { primary: 'ericsson', multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: [], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'cognitive',
        expertisePatterns: [],
        generatedAt: new Date(),
        priority: 'medium'
      };

      const result = await cognitiveOptimizer.generateTemporalPredictions(commands, context, 60);

      expect(result.predictions.length).toBeGreaterThan(0);
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
      expect(result.riskFactors).toBeDefined();
      expect(result.recommendations.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Integration Tests', () => {
    test('should generate commands from template with full optimization', async () => {
      const template: RTBTemplate = {
        meta: {
          version: '1.0',
          author: ['test'],
          description: 'Test template for cell optimization',
          priority: 10,
          tags: ['optimization', 'coverage']
        },
        custom: [],
        configuration: {
          referenceSignalPower: 12,
          qRxLevMin: -110,
          qQualMin: -10,
          cellIndividualOffset: 0
        },
        conditions: {},
        evaluations: {}
      };

      const context = {
        purpose: 'cell_optimization' as const,
        networkContext: {
          technology: '4G' as const,
          environment: 'urban_dense' as const,
          vendor: { primary: 'ericsson' as const, multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: ['1800'], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced' as const,
        priority: 'high' as const
      };

      const options = {
        enableExpertiseOptimization: true,
        enableCognitiveOptimization: true,
        aggressiveOptimization: false
      };

      const result = await cmeditEngine.generateFromTemplate(template, 'set', context, options);

      expect(result.commands.length).toBeGreaterThan(0);
      expect(result.stats.totalCommands).toBe(result.commands.length);
      expect(result.validation.isValid).toBe(true);
      expect(result.optimization.applied).toBeDefined();
      expect(result.patternsApplied.length).toBeGreaterThanOrEqual(0);
      expect(result.executionPlan.phases.length).toBeGreaterThanOrEqual(0);
    });

    test('should handle batch command generation', async () => {
      const operations = [
        {
          template: {
            meta: { version: '1.0', author: ['test'], description: 'Template 1', priority: 10 },
            configuration: { referenceSignalPower: 15 }
          },
          commandType: 'set' as const,
          targetIdentifier: 'CELL001'
        },
        {
          template: {
            meta: { version: '1.0', author: ['test'], description: 'Template 2', priority: 10 },
            configuration: { qRxLevMin: -120 }
          },
          commandType: 'set' as const,
          targetIdentifier: 'CELL001'
        }
      ];

      const context = {
        purpose: 'cell_optimization' as const,
        networkContext: {
          technology: '4G' as const,
          environment: 'urban_medium' as const,
          vendor: { primary: 'ericsson' as const, multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 2, siteCount: 1, frequencyBands: ['1800'], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced' as const
      };

      const options = {
        batchMode: true,
        parallelExecution: false
      };

      const result = await cmeditEngine.generateBatchCommands(operations, context, options);

      expect(result.commands.length).toBeGreaterThan(0);
      expect(result.stats.totalCommands).toBe(result.commands.length);
      expect(result.executionPlan.phases.length).toBeGreaterThan(0);
    });

    test('should parse and validate existing commands', () => {
      const commandStrings = [
        'get MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 --attribute referenceSignalPower',
        'set MeContext=ERBS001,ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=1 referenceSignalPower=15 --preview',
        'create MeContext=ERBS002,ManagedElement=1,ENodeBFunction=1 EUtranCellFDD=83906_E2'
      ];

      const context = {
        purpose: 'configuration_management' as const,
        networkContext: {
          technology: '4G' as const,
          environment: 'urban_medium' as const,
          vendor: { primary: 'ericsson' as const, multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 2, siteCount: 1, frequencyBands: ['1800'], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'basic' as const
      };

      const result = cmeditEngine.parseAndValidateCommands(commandStrings, context);

      expect(result.commands.length).toBe(3);
      expect(result.validation.isValid).toBe(true);
      expect(result.stats.totalCommands).toBe(3);
    });

    test('should generate optimization recommendations', () => {
      const currentConfiguration = {
        referenceSignalPower: 10,
        qRxLevMin: -100,
        qQualMin: -8,
        cellIndividualOffset: 3
      };

      const context = {
        purpose: 'cell_optimization' as const,
        networkContext: {
          technology: '4G' as const,
          environment: 'urban_dense' as const,
          vendor: { primary: 'ericsson' as const, multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 1, siteCount: 1, frequencyBands: ['1800'], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'enhanced' as const
      };

      const result = cmeditEngine.generateOptimizationRecommendations(currentConfiguration, context);

      expect(result.recommendations.length).toBeGreaterThanOrEqual(0);
      expect(result.insights.length).toBeGreaterThanOrEqual(0);
      expect(result.potentialImprovements).toBeDefined();

      result.recommendations.forEach(rec => {
        expect(rec.parameter).toBeDefined();
        expect(rec.currentValue).toBeDefined();
        expect(rec.recommendedValue).toBeDefined();
        expect(rec.command).toBeDefined();
      });
    });
  });

  // Performance Tests
  describe('Performance Tests', () => {
    test('should handle large batch operations efficiently', async () => {
      const largeOperations = Array.from({ length: 100 }, (_, i) => ({
        template: {
          meta: { version: '1.0', author: ['test'], description: `Template ${i}`, priority: 10 },
          configuration: { referenceSignalPower: 10 + i }
        },
        commandType: 'set' as const,
        targetIdentifier: `CELL${i.toString().padStart(3, '0')}`
      }));

      const context = {
        purpose: 'cell_optimization' as const,
        networkContext: {
          technology: '4G' as const,
          environment: 'urban_dense' as const,
          vendor: { primary: 'ericsson' as const, multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 100, siteCount: 10, frequencyBands: ['1800'], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'basic' as const
      };

      const startTime = Date.now();
      const result = await cmeditEngine.generateBatchCommands(largeOperations, context, {
        batchMode: true,
        maxBatchSize: 10
      });
      const endTime = Date.now();

      expect(result.commands.length).toBe(100);
      expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
      expect(result.stats.generationTime).toBeLessThan(5000);
    });

    test('should maintain reasonable memory usage', async () => {
      const memoryBefore = process.memoryUsage();

      // Generate many commands
      const operations = Array.from({ length: 50 }, (_, i) => ({
        template: {
          meta: { version: '1.0', author: ['test'], description: `Template ${i}`, priority: 10 },
          configuration: { [`param${i}`]: i }
        },
        commandType: 'set' as const
      }));

      const context = {
        purpose: 'configuration_management' as const,
        networkContext: {
          technology: '4G' as const,
          environment: 'urban_medium' as const,
          vendor: { primary: 'ericsson' as const, multiVendor: false, compatibilityMode: false },
          topology: { cellCount: 50, siteCount: 5, frequencyBands: ['1800'], carrierAggregation: false, networkSharing: false }
        },
        cognitiveLevel: 'basic' as const
      };

      await cmeditEngine.generateBatchCommands(operations, context);

      const memoryAfter = process.memoryUsage();
      const memoryIncrease = memoryAfter.heapUsed - memoryBefore.heapUsed;

      // Memory increase should be reasonable (less than 50MB for 50 commands)
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
    });
  });
});

// Helper Functions

function createMockMOHierarchy(): MOHierarchy {
  const euTranCellFDD: MOClass = {
    id: 'EUtranCellFDD',
    name: 'EUtranCellFDD',
    parentClass: 'ENodeBFunction',
    cardinality: { minimum: 0, maximum: 65535, type: 'unbounded' },
    flags: {},
    children: [],
    attributes: ['referenceSignalPower', 'qRxLevMin', 'qQualMin', 'cellIndividualOffset'],
    derivedClasses: []
  };

  const eNodeBFunction: MOClass = {
    id: 'ENodeBFunction',
    name: 'ENodeBFunction',
    parentClass: 'ManagedElement',
    cardinality: { minimum: 1, maximum: 1, type: 'single' },
    flags: {},
    children: ['EUtranCellFDD'],
    attributes: ['eNodeBPlmnId'],
    derivedClasses: []
  };

  const managedElement: MOClass = {
    id: 'ManagedElement',
    name: 'ManagedElement',
    parentClass: 'MeContext',
    cardinality: { minimum: 1, maximum: 1, type: 'single' },
    flags: {},
    children: ['ENodeBFunction'],
    attributes: [],
    derivedClasses: []
  };

  const meContext: MOClass = {
    id: 'MeContext',
    name: 'MeContext',
    parentClass: '',
    cardinality: { minimum: 1, maximum: 1, type: 'single' },
    flags: {},
    children: ['ManagedElement'],
    attributes: ['userLabel'],
    derivedClasses: []
  };

  const classes = new Map<string, MOClass>();
  classes.set('EUtranCellFDD', euTranCellFDD);
  classes.set('ENodeBFunction', eNodeBFunction);
  classes.set('ManagedElement', managedElement);
  classes.set('MeContext', meContext);

  const relationships = new Map<string, any>();
  const cardinality = new Map<string, any>();
  const inheritanceChain = new Map<string, string[]>();

  return {
    rootClass: 'MeContext',
    classes,
    relationships,
    cardinality,
    inheritanceChain
  };
}

function createMockReservedByRelationships(): ReservedByRelationship[] {
  return [
    {
      sourceClass: 'EUtranCellFDD',
      targetClass: 'ENodeBFunction',
      relationshipType: 'requires',
      cardinality: { minimum: 1, maximum: 1, type: 'single' },
      constraints: {},
      description: 'Cell requires eNodeB function'
    },
    {
      sourceClass: 'ENodeBFunction',
      targetClass: 'ManagedElement',
      relationshipType: 'requires',
      cardinality: { minimum: 1, maximum: 1, type: 'single' },
      constraints: {},
      description: 'eNodeB function requires managed element'
    }
  ];
}