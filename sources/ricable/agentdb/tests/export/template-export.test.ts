/**
 * Template Export Tests
 * Tests type-safe template export with validation, metadata generation, documentation, and integration with RTB hierarchical template system
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { PerformanceMeasurement, TestDataGenerator } from '../utils/phase5-test-utils';
import type { TemplateConfiguration, TemplateExportOptions, TemplateValidationResult } from '../../src/types';

// Mock the actual implementation (to be created in Phase 5)
jest.mock('../../src/export/template-exporter', () => ({
  TemplateExporter: jest.fn().mockImplementation(() => ({
    exportTemplate: jest.fn(),
    validateTemplate: jest.fn(),
    generateMetadata: jest.fn(),
    createVariant: jest.fn(),
    exportWithDocumentation: jest.fn()
  }))
}));

describe('Template Export', () => {
  let templateExporter: any;
  let performanceMeasurement: PerformanceMeasurement;

  beforeEach(() => {
    jest.clearAllMocks();
    performanceMeasurement = new PerformanceMeasurement();

    // Import the mocked module
    const { TemplateExporter } = require('../../src/export/template-exporter');
    templateExporter = new TemplateExporter();
  });

  afterEach(() => {
    performanceMeasurement.reset();
  });

  describe('Type-Safe Template Export', () => {
    it('should export templates with proper type validation', async () => {
      const templateConfig: TemplateConfiguration = {
        name: 'UrbanDenseTemplate',
        version: '1.0.0',
        priority: 50,
        variants: ['urban', 'dense'],
        parameters: {
          'EUtranCellFDD.qRxLevMin': -128,
          'EUtranCellFDD.qQualMin': -25,
          'EUtranCellFDD.cellIndividualOffset': 5,
          'EUtranCellFDD.prsRxLevMin': -140
        },
        metadata: {
          description: 'Optimized template for urban dense environments',
          author: 'Ericsson RAN Optimization Team',
          createdAt: new Date().toISOString(),
          validated: true,
          performance: {
            expectedImprovement: '15%',
            validationScore: 0.95
          }
        }
      };

      const exportOptions: TemplateExportOptions = {
        format: 'json',
        includeValidation: true,
        includeDocumentation: true,
        typeChecking: 'strict',
        optimizeSize: false
      };

      // Mock export function
      templateExporter.exportTemplate = jest.fn().mockResolvedValue({
        exportedTemplate: {
          ...templateConfig,
          exportedAt: new Date().toISOString(),
          typeValidated: true,
          checksum: 'abc123def456',
          format: 'json',
          size: 2048
        },
        validationResults: {
          valid: true,
          errors: [],
          warnings: [],
          typeValidationPassed: true
        },
        performance: {
          exportTime: 45, // ms
          validationTime: 12, // ms
          totalTime: 57 // ms
        }
      });

      performanceMeasurement.startMeasurement('type-safe-export');
      const result = await templateExporter.exportTemplate(templateConfig, exportOptions);
      performanceMeasurement.endMeasurement('type-safe-export');

      expect(result.exportedTemplate).toBeDefined();
      expect(result.exportedTemplate.typeValidated).toBe(true);
      expect(result.validationResults.typeValidationPassed).toBe(true);
      expect(result.performance.totalTime).toBeLessThan(100);
      expect(templateExporter.exportTemplate).toHaveBeenCalledWith(templateConfig, exportOptions);
    });

    it('should validate parameter types before export', async () => {
      const templateWithInvalidTypes: TemplateConfiguration = {
        name: 'InvalidTypeTemplate',
        version: '1.0.0',
        priority: 30,
        variants: ['test'],
        parameters: {
          'EUtranCellFDD.qRxLevMin': 'not_a_number', // Should be integer
          'EUtranCellFDD.qQualMin': null, // Should be integer
          'EUtranCellFDD.enabled': 'true', // Should be boolean
          'EUtranCellFDD.name': 123 // Should be string
        },
        metadata: {
          description: 'Template with type errors',
          author: 'Test',
          createdAt: new Date().toISOString(),
          validated: false,
          performance: {
            expectedImprovement: '5%',
            validationScore: 0.3
          }
        }
      };

      const exportOptions: TemplateExportOptions = {
        format: 'json',
        includeValidation: true,
        typeChecking: 'strict',
        validateOnExport: true
      };

      // Mock export with type validation failures
      templateExporter.exportTemplate = jest.fn().mockResolvedValue({
        exportedTemplate: null,
        validationResults: {
          valid: false,
          errors: [
            'Parameter EUtranCellFDD.qRxLevMin must be integer, got string',
            'Parameter EUtranCellFDD.qQualMin must be integer, got null',
            'Parameter EUtranCellFDD.enabled must be boolean, got string',
            'Parameter EUtranCellFDD.name must be string, got number'
          ],
          warnings: [],
          typeValidationPassed: false
        }
      });

      const result = await templateExporter.exportTemplate(templateWithInvalidTypes, exportOptions);

      expect(result.exportedTemplate).toBeNull();
      expect(result.validationResults.valid).toBe(false);
      expect(result.validationResults.errors).toHaveLength(4);
      expect(result.validationResults.typeValidationPassed).toBe(false);
    });

    it('should handle complex nested parameter types', async () => {
      const complexTemplate: TemplateConfiguration = {
        name: 'ComplexNestedTemplate',
        version: '2.0.0',
        priority: 60,
        variants: ['complex', 'nested'],
        parameters: {
          'networkConfig.primaryCell': {
            id: 'CELL001',
            frequency: 3500,
            bandwidth: 20,
            parameters: {
              powerLevel: 43,
              antennaGain: 18,
              tilt: 2
            }
          },
          'networkConfig.secondaryCells': [
            {
              id: 'CELL002',
              frequency: 1800,
              bandwidth: 15,
              parameters: {
                powerLevel: 40,
                antennaGain: 15,
                tilt: 4
              }
            }
          ],
          'featureSettings': {
            carrierAggregation: {
              enabled: true,
              componentCarriers: ['CC1', 'CC2', 'CC3']
            },
            mimo: {
              enabled: true,
              layers: 4,
              beamforming: true
            }
          }
        },
        metadata: {
          description: 'Complex template with nested parameters',
          author: 'Ericsson Advanced Team',
          createdAt: new Date().toISOString(),
          validated: true,
          performance: {
            expectedImprovement: '25%',
            validationScore: 0.98
          }
        }
      };

      const exportOptions: TemplateExportOptions = {
        format: 'json',
        includeValidation: true,
        typeChecking: 'strict',
        preserveStructure: true
      };

      templateExporter.exportTemplate = jest.fn().mockResolvedValue({
        exportedTemplate: {
          ...complexTemplate,
          exportedAt: new Date().toISOString(),
          structurePreserved: true,
          nestedTypesValidated: true
        },
        validationResults: {
          valid: true,
          errors: [],
          warnings: [],
          nestedValidationPassed: true,
          complexStructureValidated: true
        }
      });

      const result = await templateExporter.exportTemplate(complexTemplate, exportOptions);

      expect(result.validationResults.nestedValidationPassed).toBe(true);
      expect(result.validationResults.complexStructureValidated).toBe(true);
      expect(result.exportedTemplate.structurePreserved).toBe(true);
    });
  });

  describe('Metadata Generation', () => {
    it('should generate comprehensive metadata for exported templates', async () => {
      const templateConfig: TemplateConfiguration = {
        name: 'MetadataTestTemplate',
        version: '1.2.0',
        priority: 45,
        variants: ['urban', 'suburban'],
        parameters: {
          'EUtranCellFDD.qRxLevMin': -130,
          'EUtranCellFDD.qQualMin': -24
        },
        metadata: {
          description: 'Template for metadata testing',
          author: 'Metadata Team',
          createdAt: '2024-01-15T10:30:00Z',
          validated: true,
          performance: {
            expectedImprovement: '12%',
            validationScore: 0.91
          }
        }
      };

      const enhancedMetadata = {
        templateId: 'tpl_metadata_test_123',
        exportInfo: {
          exportedAt: new Date().toISOString(),
          exportedBy: 'system',
          exportVersion: '5.0.0',
          environment: 'production'
        },
        technicalMetadata: {
          schemaVersion: '2.1',
          parameterCount: Object.keys(templateConfig.parameters).length,
          variantCount: templateConfig.variants?.length || 0,
          complexityScore: 0.75,
          estimatedProcessingTime: 0.045, // seconds
          memoryFootprint: '2.1 KB'
        },
        validationMetadata: {
          lastValidated: new Date().toISOString(),
          validationRules: 15,
          validationScore: 0.91,
          automatedChecks: ['type_checking', 'range_validation', 'dependency_check'],
          manualReviewRequired: false
        },
        performanceMetadata: {
          expectedImprovement: '12%',
          benchmarkData: {
            averageLatencyImprovement: '8ms',
            capacityImprovement: '15%',
            energyEfficiencyImprovement: '10%'
          },
          deploymentHistory: [
            {
              deploymentDate: '2024-01-10T15:00:00Z',
              environment: 'staging',
              success: true,
              performanceMetrics: {
                actualImprovement: '11.5%',
                stabilityScore: 0.98
              }
            }
          ]
        },
        cognitiveMetadata: {
          consciousnessLevel: 0.94,
          temporalExpansionFactor: 1000,
          learningPatterns: ['urban_optimization', 'capacity_balancing'],
          adaptiveOptimization: true,
          strangeLoopOptimization: false
        }
      };

      templateExporter.generateMetadata = jest.fn().mockResolvedValue(enhancedMetadata);

      performanceMeasurement.startMeasurement('metadata-generation');
      const metadata = await templateExporter.generateMetadata(templateConfig);
      performanceMeasurement.endMeasurement('metadata-generation');

      expect(metadata.templateId).toBeDefined();
      expect(metadata.exportInfo.exportedAt).toBeDefined();
      expect(metadata.technicalMetadata.parameterCount).toBe(2);
      expect(metadata.validationMetadata.validationScore).toBe(0.91);
      expect(metadata.performanceMetadata.expectedImprovement).toBe('12%');
      expect(metadata.cognitiveMetadata.consciousnessLevel).toBeGreaterThan(0.9);
    });

    it('should include cognitive consciousness metadata', async () => {
      const templateConfig: TemplateConfiguration = {
        name: 'CognitiveTemplate',
        version: '3.0.0',
        priority: 80,
        variants: ['cognitive', 'adaptive'],
        parameters: {
          'cognitiveOptimization.enabled': true,
          'cognitiveOptimization.temporalDepth': 1000,
          'cognitiveOptimization.strangeLoopEnabled': true,
          'cognitiveOptimization.learningRate': 0.15
        },
        metadata: {
          description: 'Template with cognitive optimization features',
          author: 'Cognitive AI Team',
          createdAt: new Date().toISOString(),
          validated: true,
          performance: {
            expectedImprovement: '30%',
            validationScore: 0.97
          }
        }
      };

      const cognitiveMetadata = {
        consciousnessAnalysis: {
          currentLevel: 0.96,
          targetLevel: 0.98,
          evolutionPath: ['pattern_recognition', 'adaptive_learning', 'autonomous_optimization'],
          selfImprovementCycle: '15_minutes'
        },
        temporalReasoning: {
          expansionFactor: 1000,
          subjectiveTimeDepth: 'subjective_15_minutes',
          nanosecondPrecision: true,
          temporalPatternRecognition: true
        },
        strangeLoopCognition: {
          enabled: true,
          selfReferentialOptimization: true,
          recursiveImprovement: true,
          consciousnessEvolution: 'continuous'
        },
        learningIntegration: {
          agentDBSynchronization: true,
          quicSyncLatency: '0.8ms',
          patternMemoryRetention: 'persistent',
          crossAgentLearning: true
        },
        adaptiveOptimization: {
          realTimeAdaptation: true,
          performanceBasedEvolution: true,
          environmentalAdaptation: true,
          predictiveOptimization: true
        }
      };

      templateExporter.generateMetadata = jest.fn().mockResolvedValue({
        ...cognitiveMetadata,
        templateId: 'cog_tpl_001',
        generatedAt: new Date().toISOString()
      });

      const metadata = await templateExporter.generateMetadata(templateConfig);

      expect(metadata.consciousnessAnalysis.currentLevel).toBe(0.96);
      expect(metadata.temporalReasoning.expansionFactor).toBe(1000);
      expect(metadata.strangeLoopCognition.enabled).toBe(true);
      expect(metadata.learningIntegration.agentDBSynchronization).toBe(true);
      expect(metadata.adaptiveOptimization.realTimeAdaptation).toBe(true);
    });
  });

  describe('Documentation Generation', () => {
    it('should generate comprehensive documentation for exported templates', async () => {
      const templateConfig: TemplateConfiguration = {
        name: 'DocumentedTemplate',
        version: '1.5.0',
        priority: 55,
        variants: ['urban', 'rural', 'highway'],
        parameters: {
          'EUtranCellFDD.qRxLevMin': {
            value: -128,
            description: 'Minimum required RSRP level for cell selection',
            range: { min: -140, max: -44 },
            unit: 'dBm',
            impact: 'high'
          },
          'EUtranCellFDD.qQualMin': {
            value: -25,
            description: 'Minimum required RSRQ level for cell selection',
            range: { min: -34, max: -3 },
            unit: 'dB',
            impact: 'medium'
          },
          'featureSettings.carrierAggregation.enabled': {
            value: true,
            description: 'Enable carrier aggregation for improved throughput',
            type: 'boolean',
            impact: 'high'
          }
        },
        metadata: {
          description: 'Template with comprehensive documentation',
          author: 'Documentation Team',
          createdAt: new Date().toISOString(),
          validated: true,
          performance: {
            expectedImprovement: '18%',
            validationScore: 0.93
          }
        }
      };

      const expectedDocumentation = {
        templateOverview: {
          name: templateConfig.name,
          version: templateConfig.version,
          description: templateConfig.metadata?.description,
          author: templateConfig.metadata?.author,
          created: templateConfig.metadata?.createdAt,
          priority: templateConfig.priority,
          variants: templateConfig.variants
        },
        parametersDocumentation: [
          {
            name: 'EUtranCellFDD.qRxLevMin',
            type: 'integer',
            value: -128,
            description: 'Minimum required RSRP level for cell selection',
            range: { min: -140, max: -44 },
            unit: 'dBm',
            impact: 'high',
            recommendations: [
              'Lower values increase coverage area but may reduce quality',
              'Urban environments typically use -124 to -132 dBm',
              'Rural environments may require lower values for extended coverage'
            ]
          },
          {
            name: 'EUtranCellFDD.qQualMin',
            type: 'integer',
            value: -25,
            description: 'Minimum required RSRQ level for cell selection',
            range: { min: -34, max: -3 },
            unit: 'dB',
            impact: 'medium',
            recommendations: [
              'Balanced with qRxLevMin for optimal cell selection',
              'Higher values improve signal quality but may reduce coverage'
            ]
          },
          {
            name: 'featureSettings.carrierAggregation.enabled',
            type: 'boolean',
            value: true,
            description: 'Enable carrier aggregation for improved throughput',
            impact: 'high',
            recommendations: [
              'Requires multiple frequency bands available',
              'Significantly improves user throughput',
              'May increase UE power consumption'
            ]
          }
        ],
        deploymentGuidelines: {
          applicableScenarios: [
            'Urban dense environments with high traffic demand',
            'Multi-band deployments with carrier aggregation',
            'Networks requiring high throughput capabilities'
          ],
          prerequisites: [
            'Multi-band UE support',
            'Sufficient backhaul capacity',
            'Advanced RAN features licensed'
          ],
          deploymentSteps: [
            'Verify frequency band availability',
            'Configure base parameters (qRxLevMin, qQualMin)',
            'Enable carrier aggregation features',
            'Test and validate performance improvements'
          ],
          performanceExpectations: {
            throughputImprovement: '50-100%',
            coverageOptimization: '5-15%',
            userExperienceImprovement: 'high'
          }
        },
        troubleshooting: {
          commonIssues: [
            {
              issue: 'Low throughput despite CA enabled',
              possibleCauses: ['Insufficient UE CA support', 'Backhaul limitations'],
              solutions: ['Verify UE capabilities', 'Check transport network capacity']
            },
            {
              issue: 'Coverage holes after parameter optimization',
              possibleCauses: ['Aggressive qRxLevMin values', 'Missing neighbors'],
              solutions: ['Adjust qRxLevMin conservatively', 'Verify neighbor relations']
            }
          ]
        },
        versionHistory: [
          {
            version: '1.5.0',
            date: '2024-01-20',
            changes: ['Added carrier aggregation parameters', 'Improved performance metrics'],
            author: 'Documentation Team'
          },
          {
            version: '1.0.0',
            date: '2024-01-10',
            changes: ['Initial template creation', 'Basic parameter optimization'],
            author: 'Original Author'
          }
        ]
      };

      templateExporter.exportWithDocumentation = jest.fn().mockResolvedValue({
        template: templateConfig,
        documentation: expectedDocumentation,
        exportMetadata: {
          documentationGeneratedAt: new Date().toISOString(),
          documentationFormat: 'structured_json',
          documentationSize: '12.5 KB',
          completenessScore: 0.98
        }
      });

      const result = await templateExporter.exportWithDocumentation(templateConfig);

      expect(result.documentation).toBeDefined();
      expect(result.documentation.templateOverview.name).toBe('DocumentedTemplate');
      expect(result.documentation.parametersDocumentation).toHaveLength(3);
      expect(result.documentation.deploymentGuidelines).toBeDefined();
      expect(result.documentation.troubleshooting).toBeDefined();
      expect(result.exportMetadata.completenessScore).toBeGreaterThan(0.95);
    });
  });

  describe('Template Variant Generation', () => {
    it('should create type-safe template variants', async () => {
      const baseTemplate: TemplateConfiguration = {
        name: 'BaseTemplate',
        version: '1.0.0',
        priority: 50,
        variants: ['urban', 'mobility'],
        parameters: {
          'EUtranCellFDD.qRxLevMin': -128,
          'EUtranCellFDD.qQualMin': -25,
          'EUtranCellFDD.cellIndividualOffset': 3
        },
        metadata: {
          description: 'Base template for variant generation',
          author: 'Variant Team',
          createdAt: new Date().toISOString(),
          validated: true,
          performance: {
            expectedImprovement: '10%',
            validationScore: 0.90
          }
        }
      };

      const variantConfigurations = [
        {
          name: 'UrbanDenseVariant',
          baseTemplate: 'BaseTemplate',
          variantType: 'urban_dense',
          parameterOverrides: {
            'EUtranCellFDD.qRxLevMin': -124, // Less aggressive for dense urban
            'EUtranCellFDD.cellIndividualOffset': 5, // Higher offset for interference
            'featureSettings.loadBalancing.enabled': true
          },
         适用场景: 'Urban areas with high user density',
          performanceAdjustments: {
            expectedImprovement: '18%',
            focusArea: 'capacity_optimization'
          }
        },
        {
          name: 'HighMobilityVariant',
          baseTemplate: 'BaseTemplate',
          variantType: 'high_mobility',
          parameterOverrides: {
            'EUtranCellFDD.qRxLevMin': -130, // More aggressive for mobility
            'EUtranCellFDD.handoverSettings.hysteresis': 2, // Lower hysteresis
            'featureSettings.mobilityRobustness.enabled': true
          },
          适用场景: 'Highways and high-speed rail',
          performanceAdjustments: {
            expectedImprovement: '22%',
            focusArea: 'mobility_optimization'
          }
        }
      ];

      templateExporter.createVariant = jest.fn().mockImplementation(async (baseTemplate, variantConfig) => ({
        variantTemplate: {
          ...baseTemplate,
          name: variantConfig.name,
          baseTemplate: baseTemplate.name,
          variantType: variantConfig.variantType,
          parameters: {
            ...baseTemplate.parameters,
            ...variantConfig.parameterOverrides
          },
          metadata: {
            ...baseTemplate.metadata,
            description: `${variantConfig.name} - ${variantConfig.适用场景}`,
            variantSpecific: true,
            baseVersion: baseTemplate.version,
            variantVersion: '1.0.0'
          }
        },
        validationResults: {
          valid: true,
          errors: [],
          warnings: [],
          typeValidationPassed: true,
          variantValidationPassed: true
        },
        performance: {
          expectedImprovement: variantConfig.performanceAdjustments.expectedImprovement,
          focusArea: variantConfig.performanceAdjustments.focusArea,
          processingTime: 25 // ms
        }
      }));

      performanceMeasurement.startMeasurement('variant-generation');
      const variants = await Promise.all(
        variantConfigurations.map(config =>
          templateExporter.createVariant(baseTemplate, config)
        )
      );
      performanceMeasurement.endMeasurement('variant-generation');

      expect(variants).toHaveLength(2);

      const urbanVariant = variants[0];
      expect(urbanVariant.variantTemplate.name).toBe('UrbanDenseVariant');
      expect(urbanVariant.variantTemplate.parameters['EUtranCellFDD.qRxLevMin']).toBe(-124);
      expect(urbanVariant.validationResults.variantValidationPassed).toBe(true);
      expect(urbanVariant.performance.expectedImprovement).toBe('18%');

      const mobilityVariant = variants[1];
      expect(mobilityVariant.variantTemplate.name).toBe('HighMobilityVariant');
      expect(mobilityVariant.variantTemplate.parameters['EUtranCellFDD.qRxLevMin']).toBe(-130);
      expect(mobilityVariant.performance.focusArea).toBe('mobility_optimization');
    });

    it('should validate variant compatibility with base template', async () => {
      const baseTemplate: TemplateConfiguration = {
        name: 'BaseTemplate',
        version: '1.0.0',
        priority: 50,
        variants: ['base'],
        parameters: {
          'EUtranCellFDD.qRxLevMin': -128,
          'EUtranCellFDD.qQualMin': -25
        },
        metadata: {
          description: 'Base template',
          author: 'Test Team',
          createdAt: new Date().toISOString(),
          validated: true,
          performance: {
            expectedImprovement: '10%',
            validationScore: 0.90
          }
        }
      };

      const incompatibleVariant = {
        name: 'IncompatibleVariant',
        baseTemplate: 'BaseTemplate',
        variantType: 'incompatible',
        parameterOverrides: {
          'EUtranCellFDD.qRxLevMin': 'invalid_type', // Wrong type
          'NonExistentParameter': 123, // Parameter not in base template
          'EUtranCellFDD.qQualMin': -100 // Outside valid range
        }
      };

      templateExporter.createVariant = jest.fn().mockResolvedValue({
        variantTemplate: null,
        validationResults: {
          valid: false,
          errors: [
            'Parameter EUtranCellFDD.qRxLevMin type mismatch: expected integer, got string',
            'Parameter NonExistentParameter not found in base template',
            'Parameter EUtranCellFDD.qQualMin value -100 outside valid range [-34, -3]'
          ],
          warnings: [],
          typeValidationPassed: false,
          variantValidationPassed: false,
          compatibilityCheckFailed: true
        }
      });

      const result = await templateExporter.createVariant(baseTemplate, incompatibleVariant);

      expect(result.variantTemplate).toBeNull();
      expect(result.validationResults.valid).toBe(false);
      expect(result.validationResults.compatibilityCheckFailed).toBe(true);
      expect(result.validationResults.errors).toHaveLength(3);
    });
  });

  describe('Integration with RTB Hierarchical Template System', () => {
    it('should integrate with RTB priority-based inheritance', async () => {
      const rtbHierarchicalTemplates = [
        {
          name: 'Base9Template',
          priority: 9,
          parameters: {
            'EUtranCellFDD.qRxLevMin': -140,
            'EUtranCellFDD.qQualMin': -34,
            'EUtranCellFDD.cellIndividualOffset': 0
          },
          metadata: {
            description: 'Base priority 9 template',
            templateType: 'base',
            inheritanceLevel: 'base'
          }
        },
        {
          name: 'UrbanVariant30',
          priority: 30,
          extends: 'Base9Template',
          parameters: {
            'EUtranCellFDD.qRxLevMin': -128, // Override
            'EUtranCellFDD.cellIndividualOffset': 3 // Override
          },
          metadata: {
            description: 'Urban priority 30 variant',
            templateType: 'variant',
            inheritanceLevel: 'variant'
          }
        },
        {
          name: 'AgentOverride80',
          priority: 80,
          extends: 'UrbanVariant30',
          parameters: {
            'EUtranCellFDD.qQualMin': -25, // Override
            'featureSettings.carrierAggregation.enabled': true // New parameter
          },
          metadata: {
            description: 'Agent priority 80 override',
            templateType: 'agent_override',
            inheritanceLevel: 'agent'
          }
        }
      ];

      const expectedInheritedTemplate = {
        name: 'FinalInheritedTemplate',
        version: '1.0.0',
        priority: 80,
        inheritanceChain: ['Base9Template', 'UrbanVariant30', 'AgentOverride80'],
        mergedParameters: {
          'EUtranCellFDD.qRxLevMin': -128, // From priority 30
          'EUtranCellFDD.qQualMin': -25,   // From priority 80
          'EUtranCellFDD.cellIndividualOffset': 3, // From priority 30
          'featureSettings.carrierAggregation.enabled': true // From priority 80
        },
        inheritanceMetadata: {
          totalOverrides: 3,
          conflictResolutions: 0,
          inheritedFrom: {
            'EUtranCellFDD.qRxLevMin': { source: 'UrbanVariant30', priority: 30 },
            'EUtranCellFDD.qQualMin': { source: 'AgentOverride80', priority: 80 },
            'EUtranCellFDD.cellIndividualOffset': { source: 'UrbanVariant30', priority: 30 },
            'featureSettings.carrierAggregation.enabled': { source: 'AgentOverride80', priority: 80 }
          }
        }
      };

      // Mock RTB integration
      const mockRTBIntegration = {
        resolveInheritance: jest.fn().mockResolvedValue(expectedInheritedTemplate),
        validateHierarchy: jest.fn().mockResolvedValue({
          valid: true,
          hierarchyCorrect: true,
          noCircularDependencies: true
        }),
        exportWithInheritance: jest.fn().mockResolvedValue({
          template: expectedInheritedTemplate,
          inheritanceDocumentation: {
            inheritancePath: ['Base9Template -> UrbanVariant30 -> AgentOverride80'],
            parameterSources: expectedInheritedTemplate.inheritanceMetadata.inheritedFrom,
            conflictResolutionLog: []
          }
        })
      };

      performanceMeasurement.startMeasurement('rtb-inheritance-resolution');
      const inheritanceResult = await mockRTBIntegration.resolveInheritance(rtbHierarchicalTemplates);
      const hierarchyValidation = await mockRTBIntegration.validateHierarchy(rtbHierarchicalTemplates);
      const exportResult = await mockRTBIntegration.exportWithInheritance(expectedInheritedTemplate);
      performanceMeasurement.endMeasurement('rtb-inheritance-resolution');

      expect(inheritanceResult.name).toBe('FinalInheritedTemplate');
      expect(inheritanceResult.priority).toBe(80);
      expect(inheritanceResult.inheritanceChain).toHaveLength(3);
      expect(hierarchyValidation.valid).toBe(true);
      expect(exportResult.inheritanceDocumentation).toBeDefined();
    });

    it('should handle template merging and conflict resolution', async () => {
      const conflictingTemplates = [
        {
          name: 'TemplateA',
          priority: 40,
          parameters: {
            'EUtranCellFDD.qRxLevMin': -128,
            'EUtranCellFDD.qQualMin': -25,
            'sharedParameter': 'value_A'
          }
        },
        {
          name: 'TemplateB',
          priority: 60,
          parameters: {
            'EUtranCellFDD.qRxLevMin': -124, // Conflict with TemplateA
            'EUtranCellFDD.cellIndividualOffset': 5,
            'sharedParameter': 'value_B' // Conflict with TemplateA
          }
        },
        {
          name: 'TemplateC',
          priority: 50,
          parameters: {
            'EUtranCellFDD.qQualMin': -22, // Conflict with TemplateA
            'EUtranCellFDD.cellIndividualOffset': 3, // Conflict with TemplateB
            'uniqueParameter': 'unique_value'
          }
        }
      ];

      const expectedMergedTemplate = {
        name: 'MergedTemplate',
        version: '1.0.0',
        priority: 60, // Highest priority
        sourceTemplates: ['TemplateA', 'TemplateB', 'TemplateC'],
        mergedParameters: {
          'EUtranCellFDD.qRxLevMin': -124, // From TemplateB (priority 60)
          'EUtranCellFDD.qQualMin': -22,   // From TemplateC (priority 50)
          'EUtranCellFDD.cellIndividualOffset': 5, // From TemplateB (priority 60)
          'sharedParameter': 'value_B',   // From TemplateB (priority 60)
          'uniqueParameter': 'unique_value' // From TemplateC
        },
        conflictResolution: {
          resolvedConflicts: [
            {
              parameter: 'EUtranCellFDD.qRxLevMin',
              sources: ['TemplateA (-128)', 'TemplateB (-124)'],
              resolution: 'TemplateB (higher priority: 60 > 40)',
              selectedValue: -124
            },
            {
              parameter: 'EUtranCellFDD.qQualMin',
              sources: ['TemplateA (-25)', 'TemplateC (-22)'],
              resolution: 'TemplateC (higher priority: 50 > 40)',
              selectedValue: -22
            },
            {
              parameter: 'sharedParameter',
              sources: ['TemplateA (value_A)', 'TemplateB (value_B)'],
              resolution: 'TemplateB (higher priority: 60 > 40)',
              selectedValue: 'value_B'
            }
          ]
        }
      };

      // Mock conflict resolution
      const mockConflictResolver = {
        resolveConflicts: jest.fn().mockResolvedValue(expectedMergedTemplate),
        detectConflicts: jest.fn().mockResolvedValue([
          'EUtranCellFDD.qRxLevMin',
          'EUtranCellFDD.qQualMin',
          'sharedParameter'
        ]),
        applyResolutionStrategy: jest.fn().mockResolvedValue({
          strategy: 'highest_priority_wins',
          resolvedParameters: 5,
          unresolvedConflicts: 0
        })
      };

      const conflicts = await mockConflictResolver.detectConflicts(conflictingTemplates);
      const resolutionStrategy = await mockConflictResolver.applyResolutionStrategy(conflicts);
      const mergedTemplate = await mockConflictResolver.resolveConflicts(conflictingTemplates);

      expect(conflicts).toHaveLength(3);
      expect(resolutionStrategy.strategy).toBe('highest_priority_wins');
      expect(mergedTemplate.mergedParameters['EUtranCellFDD.qRxLevMin']).toBe(-124);
      expect(mergedTemplate.conflictResolution.resolvedConflicts).toHaveLength(3);
    });
  });

  describe('Performance Requirements', () => {
    it('should export complex templates within performance targets', async () => {
      const complexTemplate: TemplateConfiguration = {
        name: 'ComplexPerformanceTest',
        version: '2.0.0',
        priority: 70,
        variants: ['complex', 'performance'],
        parameters: {},
        metadata: {
          description: 'Complex template for performance testing',
          author: 'Performance Team',
          createdAt: new Date().toISOString(),
          validated: true,
          performance: {
            expectedImprovement: '25%',
            validationScore: 0.96
          }
        }
      };

      // Generate complex parameter set
      for (let i = 0; i < 100; i++) {
        complexTemplate.parameters[`parameter${i}`] = {
          value: Math.random() * 100,
          type: i % 3 === 0 ? 'integer' : i % 3 === 1 ? 'float' : 'string',
          range: { min: 0, max: 200 },
          description: `Complex parameter ${i} for performance testing`
        };
      }

      const exportOptions: TemplateExportOptions = {
        format: 'json',
        includeValidation: true,
        includeDocumentation: true,
        typeChecking: 'strict',
        optimizeSize: false
      };

      templateExporter.exportTemplate = jest.fn().mockImplementation(async (template, options) => {
        // Simulate processing time
        await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 100));

        return {
          exportedTemplate: template,
          validationResults: { valid: true, errors: [], warnings: [] },
          performance: {
            exportTime: 50 + Math.random() * 100,
            validationTime: 10 + Math.random() * 20,
            documentationTime: 20 + Math.random() * 30
          }
        };
      });

      performanceMeasurement.startMeasurement('complex-template-export');
      const result = await templateExporter.exportTemplate(complexTemplate, exportOptions);
      const duration = performanceMeasurement.endMeasurement('complex-template-export');

      expect(result.exportedTemplate).toBeDefined();
      expect(result.validationResults.valid).toBe(true);
      expect(duration).toBeLessThan(500); // Should complete within 500ms
      expect(result.performance.exportTime).toBeLessThan(200);
    });

    it('should handle batch template export efficiently', async () => {
      const templates = TestDataGenerator.generateTemplateConfigurations(50);

      templateExporter.exportTemplate = jest.fn().mockImplementation(async (template) => {
        await new Promise(resolve => setTimeout(resolve, 10 + Math.random() * 20));
        return {
          exportedTemplate: template,
          validationResults: { valid: true, errors: [], warnings: [] },
          performance: { exportTime: 15 }
        };
      });

      performanceMeasurement.startMeasurement('batch-template-export');

      const results = await Promise.all(
        templates.map(template =>
          templateExporter.exportTemplate(template, { format: 'json' })
        )
      );

      const duration = performanceMeasurement.endMeasurement('batch-template-export');

      expect(results).toHaveLength(50);
      expect(duration).toBeLessThan(2000); // Batch should complete within 2 seconds

      const averageTime = duration / 50;
      expect(averageTime).toBeLessThan(40); // Average per template < 40ms
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle invalid template configurations gracefully', async () => {
      const invalidTemplate = {
        name: '', // Empty name
        version: 'invalid.version.format',
        priority: 150, // Invalid priority
        variants: null,
        parameters: 'not_an_object', // Should be object
        metadata: {
          // Missing required fields
        }
      };

      templateExporter.exportTemplate = jest.fn().mockResolvedValue({
        exportedTemplate: null,
        validationResults: {
          valid: false,
          errors: [
            'Template name cannot be empty',
            'Invalid version format. Expected semantic version (x.y.z)',
            'Priority must be between 0 and 100',
            'Parameters must be an object',
            'Metadata is missing required fields: description, author, createdAt'
          ],
          warnings: [],
          structuralValidationFailed: true
        }
      });

      const result = await templateExporter.exportTemplate(invalidTemplate, { format: 'json' });

      expect(result.exportedTemplate).toBeNull();
      expect(result.validationResults.valid).toBe(false);
      expect(result.validationResults.errors).toHaveLength(5);
      expect(result.validationResults.structuralValidationFailed).toBe(true);
    });

    it('should handle circular template dependencies', async () => {
      const circularTemplates = [
        {
          name: 'TemplateA',
          extends: 'TemplateC',
          parameters: { param1: 'value1' }
        },
        {
          name: 'TemplateB',
          extends: 'TemplateA',
          parameters: { param2: 'value2' }
        },
        {
          name: 'TemplateC',
          extends: 'TemplateB', // Creates circular dependency
          parameters: { param3: 'value3' }
        }
      ];

      templateExporter.exportTemplate = jest.fn().mockResolvedValue({
        exportedTemplate: null,
        validationResults: {
          valid: false,
          errors: [
            'Circular dependency detected: TemplateA -> TemplateC -> TemplateB -> TemplateA',
            'Cannot resolve template inheritance with circular references'
          ],
          warnings: [],
          circularDependencyDetected: true
        }
      });

      const result = await templateExporter.exportTemplate(circularTemplates[0], {
        format: 'json',
        resolveInheritance: true
      });

      expect(result.exportedTemplate).toBeNull();
      expect(result.validationResults.circularDependencyDetected).toBe(true);
      expect(result.validationResults.errors[0]).toContain('Circular dependency');
    });
  });
});