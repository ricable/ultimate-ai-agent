import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import {
  MockPriorityBasedInheritanceEngine,
  MockTemplateMerger,
  MockUrbanVariantGenerator,
  MockMobilityVariantGenerator,
  MockAgentDBVariantGenerator,
  MockFrequencyRelationGenerator,
  MockBaseTemplateGenerator
} from './integration-mocks';
import {
  baseTemplate,
  urbanVariantTemplate,
  mobilityVariantTemplate,
  agentdbVariantTemplate,
  frequencyRelation4G4G,
  frequencyRelation4G5G,
  mockXMLSchema,
  generateLargeTemplateSet
} from '../test-data/mock-templates';
import { RTBTemplate } from '../../../src/types/rtb-types';

// Mock the entire hierarchical template system for integration testing
class MockHierarchicalTemplateSystem {
  private inheritanceEngine: MockPriorityBasedInheritanceEngine;
  private templateMerger: MockTemplateMerger;
  private urbanGenerator: MockUrbanVariantGenerator;
  private mobilityGenerator: MockMobilityVariantGenerator;
  private agentdbGenerator: MockAgentDBVariantGenerator;
  private frequencyGenerator: MockFrequencyRelationGenerator;
  private baseGenerator: MockBaseTemplateGenerator;

  constructor() {
    this.inheritanceEngine = new MockPriorityBasedInheritanceEngine();
    this.templateMerger = new MockTemplateMerger();
    this.urbanGenerator = new MockUrbanVariantGenerator();
    this.mobilityGenerator = new MockMobilityVariantGenerator();
    this.agentdbGenerator = new MockAgentDBVariantGenerator();
    this.frequencyGenerator = new MockFrequencyRelationGenerator();
    this.baseGenerator = new MockBaseTemplateGenerator(mockXMLSchema);
  }

  // Complete end-to-end workflow: XML -> Base -> Variants -> Final Template
  async generateCompleteTemplateSet(
    moClasses: string[],
    scenarios: {
      urban?: 'dense' | 'suburban' | 'rural';
      mobility?: 'highway' | 'urban' | 'pedestrian';
      agentdb?: 'basic' | 'advanced' | 'maximum';
    } = {},
    frequencyPairs?: Array<{source: string, target: string, scenario?: string}>
  ): Promise<{
    baseTemplate: RTBTemplate;
    urbanVariant?: RTBTemplate;
    mobilityVariant?: RTBTemplate;
    agentdbVariant?: RTBTemplate;
    frequencyRelations: RTBTemplate[];
    finalMergedTemplate: RTBTemplate;
    validationResults: any[];
    performanceMetrics: any;
  }> {
    const startTime = performance.now();

    // Step 1: Generate base template from XML schema
    const baseTemplate = this.baseGenerator.generateBaseTemplate(moClasses, {
      optimizationLevel: 'standard',
      includeConditions: true,
      includeEvaluations: true
    });

    // Step 2: Generate urban variant (if MO classes support it)
    let urbanVariant: RTBTemplate | undefined;
    if (moClasses.includes('EUtranCellFDD') && scenarios.urban) {
      urbanVariant = this.urbanGenerator.generateUrbanVariant(baseTemplate, scenarios.urban);
    }

    // Step 3: Generate mobility variant (if urban variant exists)
    let mobilityVariant: RTBTemplate | undefined;
    if (urbanVariant && scenarios.mobility) {
      mobilityVariant = this.mobilityGenerator.generateMobilityVariant(urbanVariant, scenarios.mobility);
    }

    // Step 4: Generate AgentDB variant (if mobility variant exists)
    let agentdbVariant: RTBTemplate | undefined;
    if (mobilityVariant && scenarios.agentdb) {
      agentdbVariant = this.agentdbGenerator.generateAgentDBVariant(mobilityVariant, scenarios.agentdb);
    }

    // Step 5: Generate frequency relation templates
    const frequencyRelations: RTBTemplate[] = [];
    if (frequencyPairs) {
      const batchRelations = this.frequencyGenerator.generateBatchFrequencyRelations(frequencyPairs);
      frequencyRelations.push(...batchRelations);
    }

    // Step 6: Create final merged template
    const templatesToMerge = [baseTemplate];
    if (urbanVariant) templatesToMerge.push(urbanVariant);
    if (mobilityVariant) templatesToMerge.push(mobilityVariant);
    if (agentdbVariant) templatesToMerge.push(agentdbVariant);
    templatesToMerge.push(...frequencyRelations);

    const finalMergedTemplate = this.templateMerger.mergeTemplates(templatesToMerge);

    // Step 7: Validate all templates
    const validationResults = [
      this.baseGenerator.validateBaseTemplate(baseTemplate),
      ...(urbanVariant ? [this.urbanGenerator.validateUrbanVariant(urbanVariant)] : []),
      ...(mobilityVariant ? [this.mobilityGenerator.validateMobilityVariant(mobilityVariant)] : []),
      ...(agentdbVariant ? [this.agentdbGenerator.validateAgentDBVariant(agentdbVariant)] : []),
      ...frequencyRelations.map(rel => this.frequencyGenerator.validateFrequencyRelationTemplate(rel))
    ];

    // Step 8: Calculate performance metrics
    const endTime = performance.now();
    const performanceMetrics = {
      totalGenerationTime: endTime - startTime,
      templateCount: templatesToMerge.length,
      validationErrors: validationResults.filter(r => !r.isValid).length,
      mergeStats: this.templateMerger.getMergeStats()
    };

    return {
      baseTemplate,
      urbanVariant,
      mobilityVariant,
      agentdbVariant,
      frequencyRelations,
      finalMergedTemplate,
      validationResults,
      performanceMetrics
    };
  }

  // Validate complete template inheritance chain
  validateInheritanceChain(template: RTBTemplate): {
    isValid: boolean;
    chain: string[];
    conflicts: string[];
    missingDependencies: string[];
  } {
    const chain: string[] = [];
    const conflicts: string[] = [];
    const missingDependencies: string[] = [];

    const processInheritance = (tmpl: RTBTemplate, visited: Set<string> = new Set()): void => {
      const templateKey = tmpl.meta?.description || 'unknown';

      if (visited.has(templateKey)) {
        conflicts.push(`Circular dependency detected: ${templateKey}`);
        return;
      }

      visited.add(templateKey);
      chain.push(templateKey);

      if (tmpl.meta?.inherits_from) {
        const parents = Array.isArray(tmpl.meta.inherits_from)
          ? tmpl.meta.inherits_from
          : [tmpl.meta.inherits_from];

        for (const parent of parents) {
          // In a real system, we'd load the parent template here
          // For testing, we'll just validate the format
          if (typeof parent !== 'string') {
            missingDependencies.push(`Invalid parent reference: ${parent}`);
          }
        }
      }
    };

    processInheritance(template);

    return {
      isValid: conflicts.length === 0 && missingDependencies.length === 0,
      chain,
      conflicts,
      missingDependencies
    };
  }

  // Simulate template deployment and validation
  async simulateDeployment(template: RTBTemplate): Promise<{
    deploymentSuccess: boolean;
    deploymentTime: number;
    configurationErrors: string[];
    runtimeValidation: any;
  }> {
    const startTime = performance.now();

    // Simulate configuration validation
    const configurationErrors: string[] = [];

    // Check for required configurations
    if (!template.configuration) {
      configurationErrors.push('Template configuration is missing');
    }

    // Validate each MO configuration
    for (const [moName, moConfig] of Object.entries(template.configuration || {})) {
      if (!moConfig || typeof moConfig !== 'object') {
        configurationErrors.push(`Invalid configuration for MO: ${moName}`);
        continue;
      }

      // Simulate parameter validation
      for (const [paramName, paramValue] of Object.entries(moConfig as any)) {
        if (paramValue === undefined || paramValue === null) {
          configurationErrors.push(`Parameter ${paramName} in ${moName} is undefined or null`);
        }
      }
    }

    // Simulate runtime validation
    const runtimeValidation = {
      customFunctionCount: template.custom?.length || 0,
      conditionCount: Object.keys(template.conditions || {}).length,
      evaluationCount: Object.keys(template.evaluations || {}).length,
      moCount: Object.keys(template.configuration || {}).length,
      totalParameters: Object.values(template.configuration || {})
        .reduce((count, moConfig) => count + Object.keys(moConfig as any).length, 0)
    };

    const endTime = performance.now();
    const deploymentTime = endTime - startTime;

    return {
      deploymentSuccess: configurationErrors.length === 0,
      deploymentTime,
      configurationErrors,
      runtimeValidation
    };
  }

  // Generate performance benchmark report
  generateBenchmarkReport(testResults: any[]): {
    summary: any;
    performanceTargets: any;
    recommendations: string[];
  } {
    const summary = {
      totalTests: testResults.length,
      successfulTests: testResults.filter(r => r.success).length,
      averageGenerationTime: testResults.reduce((sum, r) => sum + r.generationTime, 0) / testResults.length,
      averageDeploymentTime: testResults.reduce((sum, r) => sum + r.deploymentTime, 0) / testResults.length,
      averageValidationErrors: testResults.reduce((sum, r) => sum + r.validationErrors, 0) / testResults.length
    };

    const performanceTargets = {
      generationTimeTarget: 1000, // < 1 second
      deploymentTimeTarget: 500,   // < 500ms
      validationErrorTarget: 0,    // No validation errors
      successRateTarget: 0.95      // > 95% success rate
    };

    const recommendations: string[] = [];

    if (summary.averageGenerationTime > performanceTargets.generationTimeTarget) {
      recommendations.push('Optimize template generation performance - current average exceeds target');
    }

    if (summary.averageDeploymentTime > performanceTargets.deploymentTimeTarget) {
      recommendations.push('Improve deployment validation performance');
    }

    if (summary.averageValidationErrors > performanceTargets.validationErrorTarget) {
      recommendations.push('Address validation errors in template generation');
    }

    const successRate = summary.successfulTests / summary.totalTests;
    if (successRate < performanceTargets.successRateTarget) {
      recommendations.push(`Improve overall success rate - currently ${(successRate * 100).toFixed(1)}%`);
    }

    return {
      summary,
      performanceTargets,
      recommendations
    };
  }
}

describe('End-to-End Integration Tests - RTB Hierarchical Template System', () => {
  let system: MockHierarchicalTemplateSystem;

  beforeEach(() => {
    system = new MockHierarchicalTemplateSystem();
  });

  describe('Complete Template Generation Workflow', () => {
    test('should generate complete template set for dense urban scenario', async () => {
      const result = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction', 'FeatureState'],
        {
          urban: 'dense',
          mobility: 'highway',
          agentdb: 'maximum'
        },
        [
          { source: 'LTE1800', target: 'LTE2600', scenario: 'dense' },
          { source: 'LTE2600', target: 'NR3500', scenario: 'dense' }
        ]
      );

      // Check base template
      expect(result.baseTemplate).toBeDefined();
      expect(result.baseTemplate.meta?.tags).toContain('base');
      expect(result.baseTemplate.configuration['EUtranCellFDD']).toBeDefined();

      // Check variant templates
      expect(result.urbanVariant).toBeDefined();
      expect(result.urbanVariant?.meta?.tags).toContain('urban');
      expect(result.urbanVariant?.meta?.tags).toContain('dense');

      expect(result.mobilityVariant).toBeDefined();
      expect(result.mobilityVariant?.meta?.tags).toContain('mobility');
      expect(result.mobilityVariant?.meta?.tags).toContain('highway');

      expect(result.agentdbVariant).toBeDefined();
      expect(result.agentdbVariant?.meta?.tags).toContain('agentdb');
      expect(result.agentdbVariant?.meta?.tags).toContain('maximum');

      // Check frequency relations
      expect(result.frequencyRelations).toHaveLength(2);
      expect(result.frequencyRelations[0].meta?.tags).toContain('4g4g');
      expect(result.frequencyRelations[1].meta?.tags).toContain('4g5g');

      // Check final merged template
      expect(result.finalMergedTemplate).toBeDefined();
      expect(result.finalMergedTemplate.configuration).toBeDefined();

      // Should include configurations from all templates
      expect(result.finalMergedTemplate.configuration['EUtranCellFDD']).toBeDefined();
      expect(result.finalMergedTemplate.configuration['AnrFunction']).toBeDefined();
      expect(result.finalMergedTemplate.configuration['FeatureState']).toBeDefined();
      expect(result.finalMergedTemplate.configuration['EutranFreqRelation']).toBeDefined();
      expect(result.finalMergedTemplate.configuration['NrFreqRelation']).toBeDefined();

      // Check validation results
      expect(result.validationResults.every(r => r.isValid)).toBe(true);

      // Check performance metrics
      expect(result.performanceMetrics.totalGenerationTime).toBeLessThan(2000); // < 2 seconds
      expect(result.performanceMetrics.validationErrors).toBe(0);
    });

    test('should handle minimal scenario with only base template', async () => {
      const result = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD'],
        {},
        []
      );

      expect(result.baseTemplate).toBeDefined();
      expect(result.urbanVariant).toBeUndefined();
      expect(result.mobilityVariant).toBeUndefined();
      expect(result.agentdbVariant).toBeUndefined();
      expect(result.frequencyRelations).toHaveLength(0);

      // Final template should be the same as base template
      expect(result.finalMergedTemplate.configuration).toEqual(result.baseTemplate.configuration);

      expect(result.validationResults.every(r => r.isValid)).toBe(true);
    });

    test('should handle partial variant generation', async () => {
      const result = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction'],
        {
          urban: 'suburban',
          // No mobility or agentdb scenarios
        },
        [
          { source: 'LTE800', target: 'LTE1800', scenario: 'normal' }
        ]
      );

      expect(result.baseTemplate).toBeDefined();
      expect(result.urbanVariant).toBeDefined();
      expect(result.mobilityVariant).toBeUndefined();
      expect(result.agentdbVariant).toBeUndefined();
      expect(result.frequencyRelations).toHaveLength(1);

      // Should still have valid merged template
      expect(result.finalMergedTemplate).toBeDefined();
      expect(result.validationResults.every(r => r.isValid)).toBe(true);
    });
  });

  describe('Inheritance Chain Validation', () => {
    test('should validate correct inheritance chain', () => {
      const template = {
        meta: {
          description: 'Final template',
          inherits_from: ['Urban variant', 'Mobility variant']
        }
      } as RTBTemplate;

      const validation = system.validateInheritanceChain(template);

      expect(validation.isValid).toBe(true);
      expect(validation.chain).toContain('Final template');
      expect(validation.conflicts).toHaveLength(0);
      expect(validation.missingDependencies).toHaveLength(0);
    });

    test('should detect circular dependencies', () => {
      // Simulate a template that references itself
      const template = {
        meta: {
          description: 'Circular template',
          inherits_from: 'Circular template'
        }
      } as RTBTemplate;

      const validation = system.validateInheritanceChain(template);

      expect(validation.isValid).toBe(false);
      expect(validation.conflicts.length).toBeGreaterThan(0);
      expect(validation.conflicts[0]).toContain('Circular dependency');
    });

    test('should detect invalid inheritance references', () => {
      const template = {
        meta: {
          description: 'Template with invalid parent',
          inherits_from: [123, null, ''] as any // Invalid references
        }
      } as RTBTemplate;

      const validation = system.validateInheritanceChain(template);

      expect(validation.isValid).toBe(false);
      expect(validation.missingDependencies.length).toBeGreaterThan(0);
    });
  });

  describe('Template Deployment Simulation', () => {
    test('should simulate successful deployment', async () => {
      const validTemplate: RTBTemplate = {
        meta: {
          version: '1.0.0',
          description: 'Valid template for deployment'
        },
        custom: [{ name: 'testFunc', args: [], body: ['return true;'] }],
        configuration: {
          'EUtranCellFDD': {
            'qRxLevMin': -120,
            'referenceSignalPower': 15
          }
        },
        conditions: {
          'testCondition': {
            if: 'true',
            then: { 'param': 'value' },
            else: { 'param': 'default' }
          }
        },
        evaluations: {
          'testEvaluation': {
            eval: '1 + 1',
            args: []
          }
        }
      };

      const deployment = await system.simulateDeployment(validTemplate);

      expect(deployment.deploymentSuccess).toBe(true);
      expect(deployment.configurationErrors).toHaveLength(0);
      expect(deployment.deploymentTime).toBeLessThan(100); // < 100ms

      expect(deployment.runtimeValidation.customFunctionCount).toBe(1);
      expect(deployment.runtimeValidation.conditionCount).toBe(1);
      expect(deployment.runtimeValidation.evaluationCount).toBe(1);
      expect(deployment.runtimeValidation.moCount).toBe(1);
      expect(deployment.runtimeValidation.totalParameters).toBe(2);
    });

    test('should detect deployment errors', async () => {
      const invalidTemplate: RTBTemplate = {
        meta: {
          version: '1.0.0',
          description: 'Invalid template for deployment'
        },
        custom: [],
        configuration: {
          'EUtranCellFDD': {
            'qRxLevMin': undefined, // Invalid parameter
            'referenceSignalPower': null // Invalid parameter
          },
          'InvalidMO': null // Invalid MO configuration
        }
      };

      const deployment = await system.simulateDeployment(invalidTemplate);

      expect(deployment.deploymentSuccess).toBe(false);
      expect(deployment.configurationErrors.length).toBeGreaterThan(0);
      expect(deployment.configurationErrors.some(e => e.includes('undefined or null'))).toBe(true);
    });
  });

  describe('Complex Integration Scenarios', () => {
    test('should handle multi-cell scenario with multiple frequency relations', async () => {
      const result = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction', 'FeatureState', 'CapacityFunction'],
        {
          urban: 'dense',
          mobility: 'urban',
          agentdb: 'advanced'
        },
        [
          { source: 'LTE700', target: 'LTE800', scenario: 'dense' },
          { source: 'LTE800', target: 'LTE1800', scenario: 'dense' },
          { source: 'LTE1800', target: 'LTE2600', scenario: 'dense' },
          { source: 'LTE2600', target: 'NR3500', scenario: 'dense' },
          { source: 'LTE1800', target: 'NR28GHz', scenario: 'dense' }
        ]
      );

      // Should handle complex scenario efficiently
      expect(result.performanceMetrics.totalGenerationTime).toBeLessThan(3000); // < 3 seconds
      expect(result.frequencyRelations).toHaveLength(5);

      // All templates should be valid
      expect(result.validationResults.every(r => r.isValid)).toBe(true);

      // Final template should include all frequency relations
      expect(result.finalMergedTemplate.configuration['EutranFreqRelation']).toBeDefined();
      expect(result.finalMergedTemplate.configuration['NrFreqRelation']).toBeDefined();

      // Simulate deployment
      const deployment = await system.simulateDeployment(result.finalMergedTemplate);
      expect(deployment.deploymentSuccess).toBe(true);
      expect(deployment.runtimeValidation.moCount).toBeGreaterThan(5);
    });

    test('should handle error recovery scenarios', async () => {
      // Test with invalid frequency pairs
      const result = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD'],
        { urban: 'dense' },
        [
          { source: 'LTE1800', target: 'LTE2600', scenario: 'dense' }, // Valid
          { source: 'INVALID', target: 'LTE2600', scenario: 'dense' }, // Invalid
          { source: 'LTE1800', target: 'NR3500', scenario: 'dense' }  // Valid
        ]
      );

      // Should handle invalid pairs gracefully
      expect(result.frequencyRelations.length).toBeLessThan(3); // Invalid pair filtered out
      expect(result.frequencyRelations.length).toBeGreaterThan(0); // Valid pairs included

      // Valid templates should still be generated
      expect(result.baseTemplate).toBeDefined();
      expect(result.urbanVariant).toBeDefined();
      expect(result.finalMergedTemplate).toBeDefined();
    });

    test('should maintain consistency across complex inheritance chains', async () => {
      const result = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction', 'CapacityFunction', 'MobilityFunction'],
        {
          urban: 'dense',
          mobility: 'highway',
          agentdb: 'maximum'
        },
        [
          { source: 'LTE1800', target: 'LTE2600', scenario: 'dense' }
        ]
      );

      const finalTemplate = result.finalMergedTemplate;

      // Validate inheritance chain
      const inheritanceValidation = system.validateInheritanceChain(finalTemplate);
      expect(inheritanceValidation.isValid).toBe(true);

      // Check that configurations are consistent
      const cellConfig = finalTemplate.configuration['EUtranCellFDD'];
      expect(cellConfig).toBeDefined();

      // Should have values from all inheritance levels
      expect(cellConfig.qRxLevMin).toBeDefined(); // From AgentDB (highest priority)
      expect(cellConfig.referenceSignalPower).toBeDefined(); // Should be optimized
      expect(cellConfig.handoverHysteresis).toBeDefined(); // From mobility
      expect(cellConfig.ul256qamEnabled).toBeDefined(); // From urban
      expect(cellConfig.pb).toBeDefined(); // From base

      // All custom functions should be present
      expect(finalTemplate.custom?.length).toBeGreaterThan(0);

      // Check that conditions from all levels are preserved
      expect(Object.keys(finalTemplate.conditions || {}).length).toBeGreaterThan(0);
    });
  });

  describe('Performance and Scalability Integration', () => {
    test('should handle concurrent template generation', async () => {
      const scenarios = [
        { moClasses: ['EUtranCellFDD'], urban: 'dense', mobility: 'highway', agentdb: 'maximum' },
        { moClasses: ['EUtranCellFDD', 'AnrFunction'], urban: 'suburban', mobility: 'urban', agentdb: 'advanced' },
        { moClasses: ['EUtranCellFDD', 'AnrFunction', 'FeatureState'], urban: 'rural', mobility: 'pedestrian', agentdb: 'basic' }
      ];

      const startTime = performance.now();

      // Generate templates concurrently
      const promises = scenarios.map(scenario =>
        system.generateCompleteTemplateSet(
          scenario.moClasses,
          { urban: scenario.urban, mobility: scenario.mobility, agentdb: scenario.agentdb },
          [{ source: 'LTE1800', target: 'LTE2600', scenario: scenario.urban }]
        )
      );

      const results = await Promise.all(promises);
      const endTime = performance.now();

      expect(endTime - startTime).toBeLessThan(4000); // < 4 seconds for concurrent generation
      expect(results).toHaveLength(3);

      // All results should be valid
      for (const result of results) {
        expect(result.validationResults.every(r => r.isValid)).toBe(true);
        expect(result.finalMergedTemplate).toBeDefined();
      }
    });

    test('should generate comprehensive benchmark report', async () => {
      // Generate multiple test results
      const testResults = [];
      const scenarios = ['dense', 'suburban', 'rural'];

      for (let i = 0; i < 10; i++) {
        const scenario = scenarios[i % scenarios.length];
        const result = await system.generateCompleteTemplateSet(
          ['EUtranCellFDD', 'AnrFunction'],
          { urban: scenario as any },
          [{ source: 'LTE1800', target: 'LTE2600', scenario: 'normal' }]
        );

        const deployment = await system.simulateDeployment(result.finalMergedTemplate);

        testResults.push({
          success: result.validationResults.every(r => r.isValid) && deployment.deploymentSuccess,
          generationTime: result.performanceMetrics.totalGenerationTime,
          deploymentTime: deployment.deploymentTime,
          validationErrors: result.validationResults.filter(r => !r.isValid).length
        });
      }

      const benchmarkReport = system.generateBenchmarkReport(testResults);

      expect(benchmarkReport.summary.totalTests).toBe(10);
      expect(benchmarkReport.summary.successfulTests).toBeGreaterThan(8); // At least 80% success rate
      expect(benchmarkReport.summary.averageGenerationTime).toBeLessThan(2000);
      expect(benchmarkReport.summary.averageDeploymentTime).toBeLessThan(200);

      expect(benchmarkReport.performanceTargets).toBeDefined();
      expect(benchmarkReport.recommendations).toBeDefined();
      expect(Array.isArray(benchmarkReport.recommendations)).toBe(true);
    });

    test('should handle stress testing scenarios', async () => {
      // Large stress test
      const largeMoSet = Array.from({ length: 10 }, (_, i) => `MO${i}`);
      const manyFrequencyPairs = Array.from({ length: 15 }, (_, i) => ({
        source: ['LTE700', 'LTE800', 'LTE1800'][i % 3],
        target: ['LTE2600', 'NR3500'][i % 2],
        scenario: ['dense', 'normal', 'sparse'][i % 3]
      }));

      const result = await system.generateCompleteTemplateSet(
        largeMoSet,
        {
          urban: 'dense',
          mobility: 'highway',
          agentdb: 'maximum'
        },
        manyFrequencyPairs
      );

      // Should handle large scenarios within reasonable time
      expect(result.performanceMetrics.totalGenerationTime).toBeLessThan(5000); // < 5 seconds

      // Should still be valid
      expect(result.validationResults.every(r => r.isValid)).toBe(true);

      // Final template should be comprehensive
      expect(Object.keys(result.finalMergedTemplate.configuration).length).toBeGreaterThan(10);
      expect(result.finalMergedTemplate.custom?.length).toBeGreaterThan(0);
    });
  });

  describe('Real-World Scenario Simulations', () => {
    test('should simulate metropolitan network deployment', async () => {
      const result = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction', 'FeatureState', 'CapacityFunction', 'MobilityFunction'],
        {
          urban: 'dense',
          mobility: 'urban',
          agentdb: 'advanced'
        },
        [
          // Dense urban frequency relations
          { source: 'LTE1800', target: 'LTE2600', scenario: 'dense' },
          { source: 'LTE800', target: 'LTE1800', scenario: 'dense' },
          // 5G interworking
          { source: 'LTE2600', target: 'NR3500', scenario: 'dense' },
          { source: 'LTE1800', target: 'NR3500', scenario: 'dense' },
          { source: 'LTE2100', target: 'NR28GHz', scenario: 'dense' }
        ]
      );

      // Validate metropolitan scenario characteristics
      const finalTemplate = result.finalMergedTemplate;
      const cellConfig = finalTemplate.configuration['EUtranCellFDD'];

      // Dense urban should have aggressive parameters
      expect(cellConfig.qRxLevMin).toBeGreaterThan(-125);
      expect(cellConfig.referenceSignalPower).toBeGreaterThanOrEqual(18);

      // Should have capacity and mobility optimizations
      expect(finalTemplate.configuration['CapacityFunction']).toBeDefined();
      expect(finalTemplate.configuration['MobilityFunction']).toBeDefined();

      // Should include AgentDB cognitive features
      expect(finalTemplate.configuration['AgentDBFunction']).toBeDefined();
      expect(finalTemplate.configuration['CognitiveFunction']).toBeDefined();

      // Simulate deployment and validation
      const deployment = await system.simulateDeployment(finalTemplate);
      expect(deployment.deploymentSuccess).toBe(true);

      // Performance should meet metropolitan requirements
      expect(result.performanceMetrics.totalGenerationTime).toBeLessThan(3000);
    });

    test('should simulate rural network deployment', async () => {
      const result = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction'],
        {
          urban: 'rural'
          // No mobility or agentdb for basic rural deployment
        },
        [
          // Simple rural frequency relations
          { source: 'LTE700', target: 'LTE800', scenario: 'sparse' },
          { source: 'LTE800', target: 'LTE1800', scenario: 'sparse' }
        ]
      );

      const finalTemplate = result.finalMergedTemplate;
      const cellConfig = finalTemplate.configuration['EUtranCellFDD'];

      // Rural should have conservative parameters
      expect(cellConfig.qRxLevMin).toBeLessThanOrEqual(-130);
      expect(cellConfig.referenceSignalPower).toBeLessThanOrEqual(16);

      // Should have basic features only
      expect(finalTemplate.configuration['CapacityFunction']).toBeDefined();
      expect(finalTemplate.configuration['MobilityFunction']).toBeUndefined();
      expect(finalTemplate.configuration['AgentDBFunction']).toBeUndefined();

      // Should be more efficient due to simplicity
      expect(result.performanceMetrics.totalGenerationTime).toBeLessThan(1500);
    });

    test('should simulate mixed-environment network deployment', async () => {
      // Generate templates for different environments
      const denseUrban = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction'],
        { urban: 'dense', mobility: 'highway', agentdb: 'maximum' },
        [{ source: 'LTE1800', target: 'LTE2600', scenario: 'dense' }]
      );

      const suburban = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction'],
        { urban: 'suburban', mobility: 'urban', agentdb: 'advanced' },
        [{ source: 'LTE800', target: 'LTE1800', scenario: 'normal' }]
      );

      const rural = await system.generateCompleteTemplateSet(
        ['EUtranCellFDD', 'AnrFunction'],
        { urban: 'rural' },
        [{ source: 'LTE700', target: 'LTE800', scenario: 'sparse' }]
      );

      // Validate that different environments produce different configurations
      const denseConfig = denseUrban.finalMergedTemplate.configuration['EUtranCellFDD'];
      const suburbanConfig = suburban.finalMergedTemplate.configuration['EUtranCellFDD'];
      const ruralConfig = rural.finalMergedTemplate.configuration['EUtranCellFDD'];

      // Parameters should vary by environment
      expect(denseConfig.qRxLevMin).toBeGreaterThan(suburbanConfig.qRxLevMin);
      expect(suburbanConfig.qRxLevMin).toBeGreaterThan(ruralConfig.qRxLevMin);

      // Complex environments should have more features
      expect(denseUrban.finalMergedTemplate.custom?.length).toBeGreaterThan(rural.finalMergedTemplate.custom?.length);

      // All should be valid and deployable
      expect(denseUrban.validationResults.every(r => r.isValid)).toBe(true);
      expect(suburban.validationResults.every(r => r.isValid)).toBe(true);
      expect(rural.validationResults.every(r => r.isValid)).toBe(true);

      const denseDeployment = await system.simulateDeployment(denseUrban.finalMergedTemplate);
      const suburbanDeployment = await system.simulateDeployment(suburban.finalMergedTemplate);
      const ruralDeployment = await system.simulateDeployment(rural.finalMergedTemplate);

      expect(denseDeployment.deploymentSuccess).toBe(true);
      expect(suburbanDeployment.deploymentSuccess).toBe(true);
      expect(ruralDeployment.deploymentSuccess).toBe(true);
    });
  });
});