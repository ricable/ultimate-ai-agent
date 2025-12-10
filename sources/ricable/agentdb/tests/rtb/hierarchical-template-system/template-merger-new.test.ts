/**
 * Template Merger Tests for Hierarchical Template System
 */

import { describe, test, expect, beforeEach, jest } from '@jest/globals';
import { TemplateMerger } from '../../../src/rtb/hierarchical-template-system/template-merger';
import { RTBTemplate, TemplateMeta, CustomFunction } from '../../../src/types/rtb-types';

// Helper function to create test templates
const createTestTemplate = (
  name: string,
  config: any,
  priority: number = 0,
  inherits?: string | string[]
): RTBTemplate => ({
  meta: {
    version: '1.0.0',
    author: ['test'],
    description: name,
    priority,
    inherits_from: inherits,
    source: name
  },
  configuration: config,
  custom: [],
  conditions: {},
  evaluations: {}
});

describe('TemplateMerger', () => {
  let templateMerger: TemplateMerger;

  beforeEach(() => {
    templateMerger = new TemplateMerger();
  });

  describe('Basic Template Merging', () => {
    test('should merge two simple templates without conflicts', async () => {
      const template1 = createTestTemplate('template1', {
        param1: 'value1',
        param2: 'value2'
      }, 1);

      const template2 = createTestTemplate('template2', {
        param3: 'value3',
        param4: 'value4'
      }, 2);

      const result = await templateMerger.mergeTemplates([template1, template2]);

      expect(result.resolvedConflicts).toHaveLength(0);
      expect(result.unresolvedConflicts).toHaveLength(0);
      expect(result.template.configuration).toEqual({
        param1: 'value1',
        param2: 'value2',
        param3: 'value3',
        param4: 'value4'
      });
    });

    test('should handle parameter conflicts using highest priority', async () => {
      const template1 = createTestTemplate('template1', {
        sharedParam: 'value1',
        uniqueParam1: 'unique1'
      }, 1);

      const template2 = createTestTemplate('template2', {
        sharedParam: 'value2',
        uniqueParam2: 'unique2'
      }, 2);

      const result = await templateMerger.mergeTemplates([template1, template2]);

      expect(result.resolvedConflicts).toHaveLength(1);
      expect(result.unresolvedConflicts).toHaveLength(0);
      expect(result.template.configuration.sharedParam).toBe('value2'); // Higher priority wins
      expect(result.template.configuration.uniqueParam1).toBe('unique1');
      expect(result.template.configuration.uniqueParam2).toBe('unique2');
    });

    test('should merge nested objects with deep merge enabled', async () => {
      const template1 = createTestTemplate('template1', {
        network: {
          cell1: {
            param1: 'value1',
            param2: 'value2'
          }
        }
      }, 1);

      const template2 = createTestTemplate('template2', {
        network: {
          cell1: {
            param3: 'value3'
          },
          cell2: {
            param1: 'value4'
          }
        }
      }, 2);

      const result = await templateMerger.mergeTemplates([template1, template2]);

      expect(result.template.configuration.network).toEqual({
        cell1: {
          param1: 'value3', // Overridden by higher priority
          param2: 'value2',
          param3: 'value3'
        },
        cell2: {
          param1: 'value4'
        }
      });
    });
  });

  describe('Inheritance Chain Handling', () => {
    test('should build correct inheritance chain', async () => {
      const baseTemplate = createTestTemplate('base', {
        baseParam: 'baseValue',
        sharedParam: 'baseShared'
      }, 1);

      const urbanTemplate = createTestTemplate('urban', {
        urbanParam: 'urbanValue',
        sharedParam: 'urbanShared'
      }, 2, 'base');

      const mobilityTemplate = createTestTemplate('mobility', {
        mobilityParam: 'mobilityValue',
        sharedParam: 'mobilityShared'
      }, 3, ['base', 'urban']);

      const result = await templateMerger.mergeTemplates([baseTemplate, urbanTemplate, mobilityTemplate]);

      expect(result.inheritanceChain.templates).toHaveLength(3);
      expect(result.template.configuration.baseParam).toBe('baseValue');
      expect(result.template.configuration.urbanParam).toBe('urbanValue');
      expect(result.template.configuration.mobilityParam).toBe('mobilityValue');
      expect(result.template.configuration.sharedParam).toBe('mobilityShared'); // Highest priority
    });

    test('should handle circular dependencies', async () => {
      const template1 = createTestTemplate('template1', { param1: 'value1' }, 1, 'template2');
      const template2 = createTestTemplate('template2', { param2: 'value2' }, 2, 'template1');

      await expect(templateMerger.mergeTemplates([template1, template2]))
        .rejects.toThrow('Circular dependency detected');
    });
  });

  describe('Custom Functions Merging', () => {
    test('should merge custom functions with priority override', async () => {
      const template1 = createTestTemplate('template1', { param1: 'value1' }, 1);
      template1.custom = [
        {
          name: 'func1',
          args: ['arg1'],
          body: ['return arg1 * 2;']
        },
        {
          name: 'sharedFunc',
          args: ['arg1'],
          body: ['return arg1 + 1;']
        }
      ];

      const template2 = createTestTemplate('template2', { param2: 'value2' }, 2);
      template2.custom = [
        {
          name: 'func2',
          args: ['arg1'],
          body: ['return arg1 / 2;']
        },
        {
          name: 'sharedFunc',
          args: ['arg1'],
          body: ['return arg1 - 1;'] // This should override template1's version
        }
      ];

      const result = await templateMerger.mergeTemplates([template1, template2]);

      expect(result.template.custom).toHaveLength(3);
      expect(result.template.custom?.find(f => f.name === 'sharedFunc')?.body).toEqual(['return arg1 - 1;']);
    });
  });

  describe('Conditions and Evaluations Merging', () => {
    test('should merge conditions correctly', async () => {
      const template1 = createTestTemplate('template1', { param1: 'value1' }, 1);
      template1.conditions = {
        condition1: {
          if: 'param1 == "value1"',
          then: { result1: 'success1' },
          else: 'default1'
        }
      };

      const template2 = createTestTemplate('template2', { param2: 'value2' }, 2);
      template2.conditions = {
        condition2: {
          if: 'param2 == "value2"',
          then: { result2: 'success2' },
          else: 'default2'
        }
      };

      const result = await templateMerger.mergeTemplates([template1, template2]);

      expect(result.template.conditions).toEqual({
        condition1: {
          if: 'param1 == "value1"',
          then: { result1: 'success1' },
          else: 'default1'
        },
        condition2: {
          if: 'param2 == "value2"',
          then: { result2: 'success2' },
          else: 'default2'
        }
      });
    });

    test('should merge evaluations correctly', async () => {
      const template1 = createTestTemplate('template1', { param1: 'value1' }, 1);
      template1.evaluations = {
        eval1: {
          eval: 'param1 * 2',
          args: []
        }
      };

      const template2 = createTestTemplate('template2', { param2: 'value2' }, 2);
      template2.evaluations = {
        eval2: {
          eval: 'param2 + 1',
          args: []
        }
      };

      const result = await templateMerger.mergeTemplates([template1, template2]);

      expect(result.template.evaluations).toEqual({
        eval1: {
          eval: 'param1 * 2',
          args: []
        },
        eval2: {
          eval: 'param2 + 1',
          args: []
        }
      });
    });
  });

  describe('Metadata Merging', () => {
    test('should merge metadata correctly', async () => {
      const template1: RTBTemplate = {
        meta: {
          version: '1.0.0',
          author: ['author1'],
          description: 'Template 1',
          tags: ['tag1', 'tag2'],
          priority: 1,
          source: 'template1'
        },
        configuration: { param1: 'value1' },
        custom: [],
        conditions: {},
        evaluations: {}
      };

      const template2: RTBTemplate = {
        meta: {
          version: '2.0.0',
          author: ['author2'],
          description: 'Template 2',
          tags: ['tag2', 'tag3'],
          priority: 2,
          source: 'template2',
          environment: 'production'
        },
        configuration: { param2: 'value2' },
        custom: [],
        conditions: {},
        evaluations: {}
      };

      const result = await templateMerger.mergeTemplates([template1, template2]);

      expect(result.template.meta).toEqual({
        version: '2.0.0', // Higher priority wins
        author: ['author1', 'author2'], // Merged
        description: 'Template 2', // Higher priority wins
        tags: ['tag1', 'tag2', 'tag3'], // Merged and deduplicated
        priority: 2, // Higher priority wins
        source: 'template2', // Higher priority wins
        environment: 'production' // Only in template2
      });
    });
  });

  describe('Performance and Caching', () => {
    test('should cache merge results', async () => {
      const template1 = createTestTemplate('template1', { param1: 'value1' }, 1);
      const template2 = createTestTemplate('template2', { param2: 'value2' }, 2);

      // First merge
      const startTime1 = Date.now();
      const result1 = await templateMerger.mergeTemplates([template1, template2], { enableCache: true });
      const time1 = Date.now() - startTime1;

      // Second merge (should use cache)
      const startTime2 = Date.now();
      const result2 = await templateMerger.mergeTemplates([template1, template2], { enableCache: true });
      const time2 = Date.now() - startTime2;

      expect(result1.template.configuration).toEqual(result2.template.configuration);
      expect(time2).toBeLessThan(time1 + 50); // Cached version should be faster (with some tolerance)

      const cacheStats = templateMerger.getCacheStats();
      expect(cacheStats.size).toBeGreaterThan(0);
    });

    test('should clear cache', async () => {
      const template1 = createTestTemplate('template1', { param1: 'value1' }, 1);
      const template2 = createTestTemplate('template2', { param2: 'value2' }, 2);

      await templateMerger.mergeTemplates([template1, template2], { enableCache: true });

      let cacheStats = templateMerger.getCacheStats();
      expect(cacheStats.size).toBeGreaterThan(0);

      templateMerger.clearCache();

      cacheStats = templateMerger.getCacheStats();
      expect(cacheStats.size).toBe(0);
    });
  });

  describe('Error Handling', () => {
    test('should handle empty template list', async () => {
      await expect(templateMerger.mergeTemplates([]))
        .rejects.toThrow('No templates to merge');
    });

    test('should handle single template', async () => {
      const template = createTestTemplate('single', { param1: 'value1' }, 1);
      const result = await templateMerger.mergeTemplates([template]);

      expect(result.template).toEqual(template);
      expect(result.resolvedConflicts).toHaveLength(0);
      expect(result.unresolvedConflicts).toHaveLength(0);
    });
  });

  describe('Complex RAN Template Scenarios', () => {
    test('should handle complex RAN configuration merging', async () => {
      const baseTemplate: RTBTemplate = {
        meta: {
          version: '1.0.0',
          author: ['RAN Team'],
          description: 'Base RAN Configuration',
          priority: 1,
          source: 'base-ran'
        },
        configuration: {
          'EUtranCellFDD': {
            'qRxLevMin': -140,
            'qQualMin': -18,
            'referenceSignalPower': 15,
            'pb': 0,
            'prachRootSequenceIndex': 0
          },
          'AnrFunction': {
            'anrFunctionEnabled': true,
            'isHoAllowed': true
          }
        },
        custom: [
          {
            name: 'calculateCoverage',
            args: ['power', 'gain'],
            body: ['return power * gain - 120;']
          }
        ],
        conditions: {
          coverageCondition: {
            if: 'coverage > threshold',
            then: { optimizePower: true },
            else: 'default'
          }
        },
        evaluations: {
          coverageEval: {
            eval: 'calculateCoverage(referenceSignalPower, antennaGain)',
            args: []
          }
        }
      };

      const urbanTemplate: RTBTemplate = {
        meta: {
          version: '1.0.0',
          author: ['RAN Team'],
          description: 'Urban RAN Configuration',
          priority: 2,
          inherits_from: 'base-ran',
          source: 'urban-ran'
        },
        configuration: {
          'EUtranCellFDD': {
            'qRxLevMin': -125, // Urban override
            'qQualMin': -15,   // Urban override
            'referenceSignalPower': 18, // Urban override
            'ul256qamEnabled': true // Urban addition
          },
          'CapacityFunction': {
            'capacityOptimizationEnabled': true,
            'loadBalancingEnabled': true
          }
        },
        custom: [
          {
            name: 'optimizeCapacity',
            args: ['load', 'capacity'],
            body: ['return load / capacity;']
          }
        ],
        conditions: {
          highCapacityCondition: {
            if: 'users > threshold',
            then: { enableLoadBalancing: true },
            else: 'normal'
          }
        }
      };

      const result = await templateMerger.mergeTemplates([baseTemplate, urbanTemplate]);

      // Verify inheritance and merging
      expect(result.template.configuration['EUtranCellFDD']).toEqual({
        'qRxLevMin': -125, // Urban override
        'qQualMin': -15,   // Urban override
        'referenceSignalPower': 18, // Urban override
        'pb': 0, // From base
        'prachRootSequenceIndex': 0, // From base
        'ul256qamEnabled': true // From urban
      });

      expect(result.template.configuration['AnrFunction']).toBeDefined();
      expect(result.template.configuration['CapacityFunction']).toBeDefined();

      expect(result.template.custom).toHaveLength(2);
      const functionNames = result.template.custom?.map(f => f.name);
      expect(functionNames).toEqual(['calculateCoverage', 'optimizeCapacity']);

      expect(result.template.conditions).toHaveProperty('coverageCondition');
      expect(result.template.conditions).toHaveProperty('highCapacityCondition');

      expect(result.template.evaluations).toHaveProperty('coverageEval');
    });

    test('should handle multi-level inheritance with conflicts', async () => {
      const baseTemplate = createTestTemplate('base', {
        'EUtranCellFDD': {
          'qRxLevMin': -140,
          'qQualMin': -18,
          'referenceSignalPower': 15
        }
      }, 1);

      const urbanTemplate = createTestTemplate('urban', {
        'EUtranCellFDD': {
          'qRxLevMin': -125, // Conflict with base
          'ul256qamEnabled': true
        }
      }, 2, 'base');

      const mobilityTemplate = createTestTemplate('mobility', {
        'EUtranCellFDD': {
          'qQualMin': -15, // Conflict with base
          'handoverHysteresis': 4
        }
      }, 3, ['base', 'urban']);

      const result = await templateMerger.mergeTemplates([baseTemplate, urbanTemplate, mobilityTemplate]);

      // Highest priority should win for conflicts
      expect(result.template.configuration['EUtranCellFDD']).toEqual({
        'qRxLevMin': -125, // From urban (higher than base)
        'qQualMin': -15,   // From mobility (higher than urban)
        'referenceSignalPower': 15, // From base (no conflicts)
        'ul256qamEnabled': true, // From urban
        'handoverHysteresis': 4   // From mobility
      });

      expect(result.resolvedConflicts.length).toBeGreaterThan(0);
    });

    test('should handle frequency relation template merging', async () => {
      const freq4G4G = createTestTemplate('freq-4g4g', {
        'EutranFreqRelation': {
          'eutranFreqRelationId': 'LTE1800',
          'priority': 1,
          'threshHigh': 12,
          'threshLow': 8
        }
      }, 2, ['base']);

      const freq4G5G = createTestTemplate('freq-4g5g', {
        'NrFreqRelation': {
          'nrFreqRelationId': 'NR3500',
          'priority': 2,
          'threshHighX': 10,
          'threshLowX': 6
        }
      }, 3, ['base']);

      const result = await templateMerger.mergeTemplates([freq4G4G, freq4G5G]);

      expect(result.template.configuration['EutranFreqRelation']).toBeDefined();
      expect(result.template.configuration['NrFreqRelation']).toBeDefined();
      expect(result.template.configuration['EutranFreqRelation']).toEqual({
        'eutranFreqRelationId': 'LTE1800',
        'priority': 1,
        'threshHigh': 12,
        'threshLow': 8
      });
    });
  });

  describe('Validation Integration', () => {
    test('should validate merged template when enabled', async () => {
      const template1 = createTestTemplate('template1', { param1: 'value1' }, 1);
      const template2 = createTestTemplate('template2', { param2: 'value2' }, 2);

      // Should pass validation
      const result = await templateMerger.mergeTemplates([template1, template2], { validateResult: true });
      expect(result.validationResult?.isValid).toBe(true);

      // Test with invalid template (missing required meta)
      const invalidTemplate: RTBTemplate = {
        configuration: { param1: 'value1' },
        custom: [],
        conditions: {},
        evaluations: {}
      };

      // Should fail validation when strict mode is enabled
      const result2 = await templateMerger.mergeTemplates([invalidTemplate], { validateResult: true });
      expect(result2.validationResult?.isValid).toBe(false);
      expect(result2.validationResult?.errors.length).toBeGreaterThan(0);
    });
  });
});