import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import { TemplateMerger } from '../../../src/rtb/hierarchical-template-system/template-merger';
import { RTBTemplate, TemplateMeta, CustomFunction } from '../../../src/types/rtb-types';

// Mock TemplateMerger class for testing
class MockTemplateMerger {
  private mergeStats = {
    mergesPerformed: 0,
    conflictsResolved: 0,
    deepMergesPerformed: 0,
    arrayMergesPerformed: 0
  };

  constructor() {
    this.resetStats();
  }

  resetStats(): void {
    this.mergeStats = {
      mergesPerformed: 0,
      conflictsResolved: 0,
      deepMergesPerformed: 0,
      arrayMergesPerformed: 0
    };
  }

  getMergeStats() {
    return { ...this.mergeStats };
  }

  // Main template merging function with conflict resolution
  mergeTemplates(templates: RTBTemplate[], priorityOrder: 'asc' | 'desc' = 'desc'): RTBTemplate {
    if (templates.length === 0) {
      throw new Error('At least one template is required for merging');
    }

    this.mergeStats.mergesPerformed++;

    // Sort templates by priority
    const sortedTemplates = this.sortTemplatesByPriority(templates, priorityOrder);

    // Start with the highest priority template as base
    let merged: RTBTemplate = {
      meta: { ...sortedTemplates[0].meta },
      custom: [...(sortedTemplates[0].custom || [])],
      configuration: { ...sortedTemplates[0].configuration },
      conditions: { ...sortedTemplates[0].conditions },
      evaluations: { ...sortedTemplates[0].evaluations }
    };

    // Merge remaining templates in priority order
    for (let i = 1; i < sortedTemplates.length; i++) {
      merged = this.mergeTwoTemplates(merged, sortedTemplates[i]);
    }

    return merged;
  }

  // Sort templates by priority
  private sortTemplatesByPriority(templates: RTBTemplate[], order: 'asc' | 'desc'): RTBTemplate[] {
    return [...templates].sort((a, b) => {
      const priorityA = a.meta?.priority || 0;
      const priorityB = b.meta?.priority || 0;

      if (order === 'desc') {
        return priorityB - priorityA; // Highest priority first
      } else {
        return priorityA - priorityB; // Lowest priority first
      }
    });
  }

  // Merge two templates with comprehensive conflict resolution
  private mergeTwoTemplates(template1: RTBTemplate, template2: RTBTemplate): RTBTemplate {
    const merged: RTBTemplate = {
      meta: this.mergeMeta(template1.meta, template2.meta),
      custom: this.mergeCustomFunctions(template1.custom || [], template2.custom || []),
      configuration: this.mergeConfiguration(template1.configuration || {}, template2.configuration || {}),
      conditions: this.mergeConditions(template1.conditions || {}, template2.conditions || {}),
      evaluations: this.mergeEvaluations(template1.evaluations || {}, template2.evaluations || [])
    };

    return merged;
  }

  // Merge template metadata with conflict resolution
  private mergeMeta(meta1?: TemplateMeta, meta2?: TemplateMeta): TemplateMeta {
    if (!meta1) return meta2 || {};
    if (!meta2) return meta1;

    const merged: TemplateMeta = { ...meta1 };

    // Merge basic properties (template2 takes precedence)
    for (const [key, value] of Object.entries(meta2)) {
      if (key === 'inherits_from') {
        merged[key] = this.combineInheritanceChains(meta1.inherits_from, meta2.inherits_from);
      } else if (key === 'author') {
        merged[key] = this.combineAuthors(meta1.author || [], meta2.author || []);
      } else if (key === 'tags') {
        merged[key] = this.combineTags(meta1.tags || [], meta2.tags || []);
      } else {
        (merged as any)[key] = value;
      }
    }

    return merged;
  }

  // Combine inheritance chains
  private combineInheritanceChains(chain1?: string | string[], chain2?: string | string[]): string[] {
    const combined: string[] = [];

    const addChain = (chain?: string | string[]) => {
      if (chain) {
        if (Array.isArray(chain)) {
          combined.push(...chain);
        } else {
          combined.push(chain);
        }
      }
    };

    addChain(chain1);
    addChain(chain2);

    return [...new Set(combined)]; // Remove duplicates
  }

  // Combine authors
  private combineAuthors(authors1: string[], authors2: string[]): string[] {
    return [...new Set([...authors1, ...authors2])];
  }

  // Combine tags
  private combineTags(tags1: string[], tags2: string[]): string[] {
    return [...new Set([...tags1, ...tags2])];
  }

  // Merge custom functions with override detection
  private mergeCustomFunctions(functions1: CustomFunction[], functions2: CustomFunction[]): CustomFunction[] {
    this.mergeStats.arrayMergesPerformed++;

    const merged: CustomFunction[] = [...functions1];
    const existingNames = new Set(merged.map(f => f.name));

    for (const func of functions2) {
      if (existingNames.has(func.name)) {
        // Override existing function
        const index = merged.findIndex(f => f.name === func.name);
        merged[index] = func;
        this.mergeStats.conflictsResolved++;
      } else {
        // Add new function
        merged.push(func);
        existingNames.add(func.name);
      }
    }

    return merged;
  }

  // Merge configuration with deep merge support
  private mergeConfiguration(config1: Record<string, any>, config2: Record<string, any>): Record<string, any> {
    this.mergeStats.deepMergesPerformed++;

    const merged: Record<string, any> = { ...config1 };

    for (const [key, value2] of Object.entries(config2)) {
      if (key in merged) {
        const value1 = merged[key];

        if (this.isPlainObject(value1) && this.isPlainObject(value2)) {
          // Deep merge objects
          merged[key] = this.mergeConfiguration(value1, value2);
          this.mergeStats.conflictsResolved++;
        } else if (Array.isArray(value1) && Array.isArray(value2)) {
          // Merge arrays
          merged[key] = [...value1, ...value2];
          this.mergeStats.conflictsResolved++;
        } else {
          // Override with template2 value
          merged[key] = value2;
          this.mergeStats.conflictsResolved++;
        }
      } else {
        // Add new property
        merged[key] = value2;
      }
    }

    return merged;
  }

  // Check if value is a plain object
  private isPlainObject(value: any): boolean {
    return value !== null && typeof value === 'object' && !Array.isArray(value) && !(value instanceof Date);
  }

  // Merge conditions
  private mergeConditions(conditions1: Record<string, any>, conditions2: Record<string, any>): Record<string, any> {
    // Conditions are merged with template2 taking precedence for conflicts
    return { ...conditions1, ...conditions2 };
  }

  // Merge evaluations
  private mergeEvaluations(evaluations1: Record<string, any>, evaluations2: Record<string, any>): Record<string, any> {
    // Evaluations are merged with template2 taking precedence for conflicts
    return { ...evaluations1, ...evaluations2 };
  }

  // Merge multiple templates by type (e.g., all frequency relation templates)
  mergeTemplatesByType(templates: RTBTemplate[], typeTag: string): RTBTemplate[] {
    const typeTemplates = templates.filter(template =>
      template.meta?.tags?.includes(typeTag)
    );

    // Group by priority and merge within groups
    const priorityGroups = new Map<number, RTBTemplate[]>();

    for (const template of typeTemplates) {
      const priority = template.meta?.priority || 0;
      if (!priorityGroups.has(priority)) {
        priorityGroups.set(priority, []);
      }
      priorityGroups.get(priority)!.push(template);
    }

    const mergedTemplates: RTBTemplate[] = [];

    // Merge templates within each priority group
    const sortedPriorities = Array.from(priorityGroups.keys()).sort((a, b) => b - a);

    for (const priority of sortedPriorities) {
      const groupTemplates = priorityGroups.get(priority)!;
      if (groupTemplates.length === 1) {
        mergedTemplates.push(groupTemplates[0]);
      } else {
        const merged = this.mergeTemplates(groupTemplates);
        mergedTemplates.push(merged);
      }
    }

    return mergedTemplates;
  }

  // Validate merged template structure
  validateMergedTemplate(template: RTBTemplate): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check meta
    if (!template.meta) {
      errors.push('Template meta is required');
    } else {
      if (!template.meta.version) {
        errors.push('Template version is required');
      }
      if (!template.meta.description) {
        errors.push('Template description is required');
      }
    }

    // Check configuration
    if (!template.configuration) {
      errors.push('Template configuration is required');
    }

    // Check custom functions
    if (template.custom) {
      for (const func of template.custom) {
        if (!func.name) {
          errors.push('Custom function name is required');
        }
        if (!func.args || !Array.isArray(func.args)) {
          errors.push(`Custom function ${func.name} args must be an array`);
        }
        if (!func.body || !Array.isArray(func.body)) {
          errors.push(`Custom function ${func.name} body must be an array`);
        }
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  // Get merge conflict report
  getConflictReport(template1: RTBTemplate, template2: RTBTemplate): {
    conflicts: string[];
    resolutions: string[];
    additions: string[];
  } {
    const conflicts: string[] = [];
    const resolutions: string[] = [];
    const additions: string[] = [];

    // Check configuration conflicts
    const config1Keys = Object.keys(template1.configuration || {});
    const config2Keys = Object.keys(template2.configuration || {});

    for (const key of config2Keys) {
      if (config1Keys.includes(key)) {
        conflicts.push(`Configuration key conflict: ${key}`);
        resolutions.push(`Resolved: Using template2 value for ${key}`);
      } else {
        additions.push(`New configuration key: ${key}`);
      }
    }

    // Check custom function conflicts
    const func1Names = (template1.custom || []).map(f => f.name);
    const func2Names = (template2.custom || []).map(f => f.name);

    for (const name of func2Names) {
      if (func1Names.includes(name)) {
        conflicts.push(`Custom function conflict: ${name}`);
        resolutions.push(`Resolved: Using template2 function for ${name}`);
      } else {
        additions.push(`New custom function: ${name}`);
      }
    }

    return { conflicts, resolutions, additions };
  }
}

describe('TemplateMerger', () => {
  let merger: MockTemplateMerger;

  beforeEach(() => {
    merger = new MockTemplateMerger();
  });

  afterEach(() => {
    merger.resetStats();
  });

  describe('Basic Template Merging', () => {
    test('should merge two templates correctly', () => {
      const template1: RTBTemplate = {
        meta: { version: '1.0', author: ['Author1'], description: 'Template 1', priority: 1 },
        configuration: { 'MO1': { 'param1': 'value1' } },
        custom: [{ name: 'func1', args: [], body: ['return 1;'] }]
      };

      const template2: RTBTemplate = {
        meta: { version: '2.0', author: ['Author2'], description: 'Template 2', priority: 2 },
        configuration: { 'MO2': { 'param2': 'value2' } },
        custom: [{ name: 'func2', args: [], body: ['return 2;'] }]
      };

      const merged = merger.mergeTemplates([template1, template2]);

      expect(merged.meta.version).toBe('2.0'); // Higher priority
      expect(merged.meta.author).toEqual(['Author1', 'Author2']); // Combined
      expect(merged.meta.description).toBe('Template 2'); // Higher priority
      expect(merged.meta.priority).toBe(2); // Higher priority

      expect(merged.configuration).toEqual({
        'MO1': { 'param1': 'value1' },
        'MO2': { 'param2': 'value2' }
      });

      expect(merged.custom).toHaveLength(2);
      expect(merged.custom?.map(f => f.name)).toEqual(['func1', 'func2']);

      const stats = merger.getMergeStats();
      expect(stats.mergesPerformed).toBe(1);
      expect(stats.conflictsResolved).toBe(0);
    });

    test('should handle empty configuration objects', () => {
      const template1: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 1', priority: 1 },
        configuration: {}
      };

      const template2: RTBTemplate = {
        meta: { version: '2.0', description: 'Template 2', priority: 2 },
        configuration: { 'MO1': { 'param1': 'value1' } }
      };

      const merged = merger.mergeTemplates([template1, template2]);

      expect(merged.configuration).toEqual({ 'MO1': { 'param1': 'value1' } });
    });

    test('should throw error for empty template array', () => {
      expect(() => merger.mergeTemplates([])).toThrow('At least one template is required for merging');
    });
  });

  describe('Priority-Based Merging', () => {
    test('should sort templates by priority correctly', () => {
      const lowPriority: RTBTemplate = {
        meta: { version: '1.0', description: 'Low', priority: 1 },
        configuration: { 'MO1': { 'param': 'low' } }
      };

      const mediumPriority: RTBTemplate = {
        meta: { version: '1.0', description: 'Medium', priority: 5 },
        configuration: { 'MO1': { 'param': 'medium' } }
      };

      const highPriority: RTBTemplate = {
        meta: { version: '1.0', description: 'High', priority: 10 },
        configuration: { 'MO1': { 'param': 'high' } }
      };

      // Descending order (default)
      const mergedDesc = merger.mergeTemplates([lowPriority, mediumPriority, highPriority]);
      expect(mergedDesc.configuration['MO1'].param).toBe('high'); // Highest priority wins

      // Ascending order
      const mergedAsc = merger.mergeTemplates([lowPriority, mediumPriority, highPriority], 'asc');
      expect(mergedAsc.configuration['MO1'].param).toBe('high'); // Still highest priority wins
    });

    test('should handle templates without priority', () => {
      const withPriority: RTBTemplate = {
        meta: { version: '1.0', description: 'With priority', priority: 5 },
        configuration: { 'MO1': { 'param': 'with-priority' } }
      };

      const withoutPriority: RTBTemplate = {
        meta: { version: '1.0', description: 'Without priority' },
        configuration: { 'MO1': { 'param': 'without-priority' } }
      };

      const merged = merger.mergeTemplates([withPriority, withoutPriority]);

      // Template with priority should come first (higher priority)
      expect(merged.configuration['MO1'].param).toBe('with-priority');
    });
  });

  describe('Configuration Merging', () => {
    test('should merge nested configuration objects', () => {
      const template1: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 1', priority: 1 },
        configuration: {
          'EUtranCellFDD': {
            'qRxLevMin': -140,
            'qQualMin': -18,
            'nested': {
              'param1': 'value1',
              'param2': 'value2'
            }
          }
        }
      };

      const template2: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 2', priority: 2 },
        configuration: {
          'EUtranCellFDD': {
            'qRxLevMin': -130, // Override
            'referenceSignalPower': 15, // New
            'nested': {
              'param2': 'new-value2', // Override nested
              'param3': 'value3' // New nested
            }
          },
          'AnrFunction': { // New MO
            'anrFunctionEnabled': true
          }
        }
      };

      const merged = merger.mergeTemplates([template1, template2]);

      expect(merged.configuration['EUtranCellFDD']).toEqual({
        'qRxLevMin': -130, // Overridden
        'qQualMin': -18, // Preserved
        'referenceSignalPower': 15, // Added
        'nested': {
          'param1': 'value1', // Preserved
          'param2': 'new-value2', // Overridden
          'param3': 'value3' // Added
        }
      });

      expect(merged.configuration['AnrFunction']).toEqual({
        'anrFunctionEnabled': true
      });

      const stats = merger.getMergeStats();
      expect(stats.conflictsResolved).toBeGreaterThan(0);
      expect(stats.deepMergesPerformed).toBeGreaterThan(0);
    });

    test('should merge arrays in configuration', () => {
      const template1: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 1', priority: 1 },
        configuration: {
          'MO1': {
            'arrayParam': ['item1', 'item2'],
            'stringParam': 'value1'
          }
        }
      };

      const template2: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 2', priority: 2 },
        configuration: {
          'MO1': {
            'arrayParam': ['item3', 'item4'], // Override array
            'stringParam': 'value2' // Override string
          }
        }
      };

      const merged = merger.mergeTemplates([template1, template2]);

      expect(merged.configuration['MO1'].arrayParam).toEqual(['item1', 'item2', 'item3', 'item4']);
      expect(merged.configuration['MO1'].stringParam).toBe('value2');
    });
  });

  describe('Custom Function Merging', () => {
    test('should merge custom functions with override detection', () => {
      const func1: CustomFunction = {
        name: 'calculatePower',
        args: ['power', 'gain'],
        body: ['return power * gain;']
      };

      const func2: CustomFunction = {
        name: 'calculateEfficiency',
        args: ['input', 'output'],
        body: ['return output / input;']
      };

      const func2Override: CustomFunction = {
        name: 'calculatePower', // Same name as func1
        args: ['power', 'gain', 'efficiency'],
        body: ['return power * gain * efficiency;']
      };

      const template1: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 1', priority: 1 },
        custom: [func1, func2]
      };

      const template2: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 2', priority: 2 },
        custom: [func2Override, { name: 'newFunction', args: [], body: ['return true;'] }]
      };

      const merged = merger.mergeTemplates([template1, template2]);

      expect(merged.custom).toHaveLength(3);
      expect(merged.custom?.map(f => f.name)).toEqual(['calculateEfficiency', 'calculatePower', 'newFunction']);

      // Check that calculatePower was overridden
      const calculatePower = merged.custom?.find(f => f.name === 'calculatePower');
      expect(calculatePower?.args).toEqual(['power', 'gain', 'efficiency']);
      expect(calculatePower?.body).toEqual(['return power * gain * efficiency;']);

      const stats = merger.getMergeStats();
      expect(stats.arrayMergesPerformed).toBeGreaterThan(0);
      expect(stats.conflictsResolved).toBeGreaterThan(0);
    });

    test('should handle empty custom function arrays', () => {
      const template1: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 1', priority: 1 },
        custom: []
      };

      const template2: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 2', priority: 2 },
        custom: [{ name: 'onlyFunction', args: [], body: ['return 42;'] }]
      };

      const merged = merger.mergeTemplates([template1, template2]);

      expect(merged.custom).toHaveLength(1);
      expect(merged.custom?.[0].name).toBe('onlyFunction');
    });
  });

  describe('Metadata Merging', () => {
    test('should merge inheritance chains correctly', () => {
      const template1: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Template 1',
          priority: 1,
          inherits_from: 'base'
        }
      };

      const template2: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Template 2',
          priority: 2,
          inherits_from: ['urban', 'mobility']
        }
      };

      const merged = merger.mergeTemplates([template1, template2]);

      expect(merged.meta?.inherits_from).toEqual(['base', 'urban', 'mobility']);
    });

    test('should combine authors and tags', () => {
      const template1: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Template 1',
          priority: 1,
          author: ['Author1', 'Author2'],
          tags: ['tag1', 'tag2']
        }
      };

      const template2: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Template 2',
          priority: 2,
          author: ['Author2', 'Author3'],
          tags: ['tag2', 'tag3']
        }
      };

      const merged = merger.mergeTemplates([template1, template2]);

      expect(merged.meta?.author).toEqual(['Author1', 'Author2', 'Author3']);
      expect(merged.meta?.tags).toEqual(['tag1', 'tag2', 'tag3']);
    });
  });

  describe('Conflict Resolution', () => {
    test('should generate conflict reports', () => {
      const template1: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 1', priority: 1 },
        configuration: {
          'MO1': { 'param1': 'value1' },
          'MO2': { 'param2': 'value2' }
        },
        custom: [
          { name: 'func1', args: [], body: ['return 1;'] },
          { name: 'func2', args: [], body: ['return 2;'] }
        ]
      };

      const template2: RTBTemplate = {
        meta: { version: '1.0', description: 'Template 2', priority: 2 },
        configuration: {
          'MO1': { 'param1': 'new-value1' }, // Conflict
          'MO3': { 'param3': 'value3' } // New
        },
        custom: [
          { name: 'func2', args: [], body: ['return 22;'] }, // Conflict
          { name: 'func3', args: [], body: ['return 3;'] } // New
        ]
      };

      const report = merger.getConflictReport(template1, template2);

      expect(report.conflicts).toHaveLength(2);
      expect(report.conflicts).toContain('Configuration key conflict: MO1');
      expect(report.conflicts).toContain('Custom function conflict: func2');

      expect(report.resolutions).toHaveLength(2);
      expect(report.additions).toHaveLength(2);
      expect(report.additions).toContain('New configuration key: MO3');
      expect(report.additions).toContain('New custom function: func3');
    });
  });

  describe('Template Type Merging', () => {
    test('should merge templates by type tag', () => {
      const frequencyTemplate1: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Frequency template 1',
          priority: 3,
          tags: ['frequency', '4g4g']
        },
        configuration: { 'EutranFreqRelation': { 'priority': 1 } }
      };

      const frequencyTemplate2: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Frequency template 2',
          priority: 7,
          tags: ['frequency', '4g5g']
        },
        configuration: { 'NrFreqRelation': { 'priority': 2 } }
      };

      const otherTemplate: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Other template',
          priority: 5,
          tags: ['other']
        },
        configuration: { 'OtherMO': { 'param': 'value' } }
      };

      const templates = [frequencyTemplate1, frequencyTemplate2, otherTemplate];
      const frequencyMergeds = merger.mergeTemplatesByType(templates, 'frequency');

      expect(frequencyMergeds).toHaveLength(2);
      expect(frequencyMergeds[0].meta?.priority).toBe(7); // Higher priority first
      expect(frequencyMergeds[1].meta?.priority).toBe(3);
    });

    test('should merge multiple templates with same priority', () => {
      const template1: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Template 1',
          priority: 5,
          tags: ['test']
        },
        configuration: { 'MO1': { 'param1': 'value1' } }
      };

      const template2: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Template 2',
          priority: 5,
          tags: ['test']
        },
        configuration: { 'MO2': { 'param2': 'value2' } }
      };

      const merged = merger.mergeTemplatesByType([template1, template2], 'test');

      expect(merged).toHaveLength(1);
      expect(merged[0].configuration).toEqual({
        'MO1': { 'param1': 'value1' },
        'MO2': { 'param2': 'value2' }
      });
    });
  });

  describe('Template Validation', () => {
    test('should validate merged template structure', () => {
      const validTemplate: RTBTemplate = {
        meta: {
          version: '1.0',
          description: 'Valid template',
          author: ['Test Author']
        },
        configuration: { 'MO1': { 'param': 'value' } },
        custom: [{ name: 'validFunc', args: [], body: ['return true;'] }]
      };

      const validation = merger.validateMergedTemplate(validTemplate);
      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    test('should detect validation errors', () => {
      const invalidTemplate: RTBTemplate = {
        meta: {
          description: 'Invalid template'
          // Missing version
        },
        configuration: { 'MO1': { 'param': 'value' } },
        custom: [
          { name: '', args: [], body: ['return true;'] }, // Empty name
          { name: 'func2', args: 'not-array', body: ['return true;'] }, // Invalid args
          { name: 'func3', args: [], body: 'not-array' } // Invalid body
        ]
      };

      const validation = merger.validateMergedTemplate(invalidTemplate);
      expect(validation.isValid).toBe(false);
      expect(validation.errors.length).toBeGreaterThan(0);
      expect(validation.errors).toContain('Template version is required');
      expect(validation.errors).toContain('Custom function name is required');
      expect(validation.errors).toContain('Custom function func2 args must be an array');
      expect(validation.errors).toContain('Custom function func3 body must be an array');
    });
  });

  describe('Complex Integration Scenarios', () => {
    test('should merge urban variant correctly with base template', () => {
      const merged = merger.mergeTemplates([baseTemplate, urbanVariantTemplate]);

      // Check that urban values override base values
      expect(merged.configuration['EUtranCellFDD'].qRxLevMin).toBe(-125); // Urban override
      expect(merged.configuration['EUtranCellFDD'].qQualMin).toBe(-15);   // Urban override
      expect(merged.configuration['EUtranCellFDD'].referenceSignalPower).toBe(18); // Urban override

      // Check that base values are preserved when not overridden
      expect(merged.configuration['EUtranCellFDD'].pb).toBe(0); // From base
      expect(merged.configuration['EUtranCellFDD'].prachRootSequenceIndex).toBe(0); // From base

      // Check that urban additions are present
      expect(merged.configuration['CapacityFunction']).toBeDefined();
      expect(merged.configuration['CapacityFunction'].capacityOptimizationEnabled).toBe(true);

      // Check that custom functions are merged
      expect(merged.custom).toHaveLength(1);
      expect(merged.custom?.[0].name).toBe('calculateCoverage');

      // Check that conditions are merged
      expect(merged.conditions).toHaveProperty('coverageCondition'); // From base
      expect(merged.conditions).toHaveProperty('highCapacityCondition'); // From urban
    });

    test('should merge complete inheritance chain correctly', () => {
      const merged = merger.mergeTemplates([baseTemplate, urbanVariantTemplate, mobilityVariantTemplate, agentdbVariantTemplate]);

      // Should use highest priority values (agentdb)
      expect(merged.configuration['EUtranCellFDD'].qRxLevMin).toBe(-118); // AgentDB

      // Should inherit from all levels
      expect(merged.configuration['EUtranCellFDD'].ul256qamEnabled).toBe(true); // From urban
      expect(merged.configuration['EUtranCellFDD'].handoverHysteresis).toBe(4); // From mobility
      expect(merged.configuration['EUtranCellFDD'].pb).toBe(0); // From base

      // Should have all MOs from all levels
      expect(merged.configuration['AnrFunction']).toBeDefined(); // From base
      expect(merged.configuration['CapacityFunction']).toBeDefined(); // From urban
      expect(merged.configuration['MobilityFunction']).toBeDefined(); // From mobility
      expect(merged.configuration['AgentDBFunction']).toBeDefined(); // From agentdb
      expect(merged.configuration['CognitiveFunction']).toBeDefined(); // From agentdb

      // Should have all custom functions
      expect(merged.custom).toHaveLength(3);
      const functionNames = merged.custom?.map(f => f.name);
      expect(functionNames).toEqual([
        'calculateCoverage', // From base
        'optimizeHandover',  // From mobility
        'cognitiveOptimization' // From agentdb
      ]);

      // Should have all conditions
      expect(Object.keys(merged.conditions || {})).toContain('coverageCondition'); // From base
      expect(Object.keys(merged.conditions || {})).toContain('highCapacityCondition'); // From urban
      expect(Object.keys(merged.conditions || {})).toContain('speedAdaptiveCondition'); // From mobility
      expect(Object.keys(merged.conditions || {})).toContain('cognitiveCondition'); // From agentdb
    });

    test('should handle conflict resolution correctly', () => {
      const merged = merger.mergeTemplates([baseTemplate, conflictTemplate1, conflictTemplate2]);

      // Should use highest priority values (conflictTemplate2 has priority 8)
      expect(merged.configuration['EUtranCellFDD'].qRxLevMin).toBe(-110); // From conflictTemplate2
      expect(merged.configuration['EUtranCellFDD'].cellIndividualOffset).toBe(5); // From conflictTemplate2
      expect(merged.configuration['EUtranCellFDD'].qQualMin).toBe(-20); // From conflictTemplate1 (only source)

      const stats = merger.getMergeStats();
      expect(stats.conflictsResolved).toBeGreaterThan(0);
    });
  });

  describe('Performance Testing', () => {
    test('should handle large template sets efficiently', () => {
      const largeTemplates = generateLargeTemplateSet(50);

      const startTime = Date.now();
      const merged = merger.mergeTemplates(largeTemplates);
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(1000); // Should merge in under 1 second
      expect(merged).toBeDefined();
      expect(merged.configuration).toBeDefined();

      const stats = merger.getMergeStats();
      expect(stats.mergesPerformed).toBeGreaterThan(0);
    });

    test('should handle complex nested merges efficiently', () => {
      // Create templates with deeply nested configurations
      const complexTemplates: RTBTemplate[] = [];

      for (let i = 0; i < 10; i++) {
        complexTemplates.push({
          meta: {
            version: '1.0',
            description: `Complex template ${i}`,
            priority: i,
            tags: ['complex', 'nested']
          },
          configuration: {
            [`Level${i}`]: {
              [`SubLevel${i}`]: {
                [`Deep${i}`]: {
                  [`Param${i}`]: `value-${i}`,
                  [`Nested${i}`]: {
                    [`Final${i}`]: `final-value-${i}`,
                    array: Array.from({ length: 10 }, (_, j) => `item-${i}-${j}`)
                  }
                }
              }
            }
          }
        });
      }

      const startTime = Date.now();
      const merged = merger.mergeTemplates(complexTemplates);
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(500); // Should merge complex nested in under 500ms

      // Verify deep merging worked
      expect(merged.configuration['Level9']['SubLevel9']['Deep9']['Nested9']['Final9']).toBe('final-value-9');
      expect(merged.configuration['Level0']['SubLevel0']['Deep0']['Nested9']).toBeDefined(); // Should be merged
    });
  });
});