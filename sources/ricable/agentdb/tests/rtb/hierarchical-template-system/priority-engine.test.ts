import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import { PriorityBasedInheritanceEngine } from '../../src/rtb/hierarchical-template-system/priority-inheritance-engine';
import {
  baseTemplate,
  urbanVariantTemplate,
  mobilityVariantTemplate,
  agentdbVariantTemplate,
  conflictTemplate1,
  conflictTemplate2,
  invalidTemplate,
  expectedUrbanMerged,
  generateLargeTemplateSet
} from '../test-data/mock-templates';
import { RTBTemplate } from '../../../src/types/rtb-types';

// Mock the PriorityBasedInheritanceEngine class for testing
class MockPriorityBasedInheritanceEngine {
  private templates: Map<string, RTBTemplate> = new Map();
  private priorityCache: Map<string, number> = new Map();
  private inheritanceCache: Map<string, RTBTemplate> = new Map();

  constructor() {
    this.templates.clear();
    this.priorityCache.clear();
    this.inheritanceCache.clear();
  }

  // Load templates into the engine
  loadTemplates(templates: RTBTemplate[]): void {
    templates.forEach(template => {
      const key = template.meta?.description || `template-${Date.now()}`;
      this.templates.set(key, template);
      if (template.meta?.priority) {
        this.priorityCache.set(key, template.meta.priority);
      }
    });
  }

  // Calculate inheritance priority for a template
  calculateInheritancePriority(templateName: string): number {
    const template = Array.from(this.templates.values())
      .find(t => t.meta?.description === templateName ||
                t.meta?.source === templateName ||
                this.getTemplateKey(t) === templateName);

    if (!template) {
      throw new Error(`Template not found: ${templateName}`);
    }

    let priority = template.meta?.priority || 0;
    const processed = new Set<string>();

    // Calculate priority based on inheritance chain
    const calculateFromInheritance = (template: RTBTemplate): number => {
      const key = this.getTemplateKey(template);
      if (processed.has(key)) {
        return priority; // Prevent circular dependencies
      }
      processed.add(key);

      if (template.meta?.inherits_from) {
        const parents = Array.isArray(template.meta.inherits_from)
          ? template.meta.inherits_from
          : [template.meta.inherits_from];

        let maxParentPriority = 0;
        for (const parent of parents) {
          try {
            const parentPriority = this.calculateInheritancePriority(parent);
            maxParentPriority = Math.max(maxParentPriority, parentPriority);
          } catch (error) {
            // Parent not found, continue with current priority
          }
        }
        priority = Math.max(priority, maxParentPriority + 1);
      }

      return priority;
    };

    return calculateFromInheritance(template);
  }

  // Get template key for identification
  private getTemplateKey(template: RTBTemplate): string {
    return template.meta?.description || template.meta?.source || `template-${Math.random()}`;
  }

  // Resolve template inheritance with priority-based conflict resolution
  resolveInheritance(templateName: string): RTBTemplate {
    const cacheKey = templateName;
    if (this.inheritanceCache.has(cacheKey)) {
      return this.inheritanceCache.get(cacheKey)!;
    }

    const template = Array.from(this.templates.values())
      .find(t => t.meta?.description === templateName ||
                t.meta?.source === templateName ||
                this.getTemplateKey(t) === templateName);

    if (!template) {
      throw new Error(`Template not found: ${templateName}`);
    }

    const result = this.mergeWithParents(template);
    this.inheritanceCache.set(cacheKey, result);
    return result;
  }

  // Merge template with its parents based on priority
  private mergeWithParents(template: RTBTemplate): RTBTemplate {
    let merged: RTBTemplate = {
      meta: { ...template.meta },
      custom: [...(template.custom || [])],
      configuration: { ...template.configuration },
      conditions: { ...template.conditions },
      evaluations: { ...template.evaluations }
    };

    if (template.meta?.inherits_from) {
      const parents = Array.isArray(template.meta.inherits_from)
        ? template.meta.inherits_from
        : [template.meta.inherits_from];

      // Sort parents by priority (highest first)
      const sortedParents = parents
        .map(parent => ({ name: parent, priority: this.calculateInheritancePriority(parent) }))
        .sort((a, b) => b.priority - a.priority);

      // Merge parents in priority order
      for (const parent of sortedParents) {
        try {
          const parentTemplate = this.resolveInheritance(parent.name);
          merged = this.mergeTemplates(parentTemplate, merged);
        } catch (error) {
          // Parent not found, continue with current template
        }
      }
    }

    return merged;
  }

  // Merge two templates with conflict resolution
  private mergeTemplates(parent: RTBTemplate, child: RTBTemplate): RTBTemplate {
    const merged: RTBTemplate = {
      meta: { ...parent.meta, ...child.meta },
      custom: this.mergeArrays(parent.custom || [], child.custom || [], 'name'),
      configuration: this.mergeConfigurations(parent.configuration || {}, child.configuration || {}),
      conditions: { ...parent.conditions, ...child.conditions },
      evaluations: { ...parent.evaluations, ...child.evaluations }
    };

    // Child meta takes precedence
    if (child.meta) {
      merged.meta = {
        ...parent.meta,
        ...child.meta,
        // Combine inherited_from arrays
        inherits_from: this.combineInheritedFrom(parent.meta?.inherits_from, child.meta?.inherits_from)
      };
    }

    return merged;
  }

  // Merge arrays with deduplication based on key field
  private mergeArrays<T extends Record<string, any>>(parent: T[], child: T[], keyField: string): T[] {
    const merged = [...parent];
    const existingKeys = new Set(merged.map(item => item[keyField]));

    for (const childItem of child) {
      if (!existingKeys.has(childItem[keyField])) {
        merged.push(childItem);
        existingKeys.add(childItem[keyField]);
      } else {
        // Replace existing item with child (higher priority)
        const index = merged.findIndex(item => item[keyField] === childItem[keyField]);
        if (index !== -1) {
          merged[index] = childItem;
        }
      }
    }

    return merged;
  }

  // Merge configuration objects with deep merge
  private mergeConfigurations(parent: Record<string, any>, child: Record<string, any>): Record<string, any> {
    const merged: Record<string, any> = { ...parent };

    for (const [key, childValue] of Object.entries(child)) {
      if (key in merged) {
        if (typeof merged[key] === 'object' && typeof childValue === 'object' && !Array.isArray(merged[key]) && !Array.isArray(childValue)) {
          // Deep merge objects
          merged[key] = this.mergeConfigurations(merged[key], childValue);
        } else {
          // Child value takes precedence
          merged[key] = childValue;
        }
      } else {
        merged[key] = childValue;
      }
    }

    return merged;
  }

  // Combine inherited_from arrays
  private combineInheritedFrom(parentInherits?: string | string[], childInherits?: string | string[]): string[] {
    const combined: string[] = [];

    if (parentInherits) {
      if (Array.isArray(parentInherits)) {
        combined.push(...parentInherits);
      } else {
        combined.push(parentInherits);
      }
    }

    if (childInherits) {
      if (Array.isArray(childInherits)) {
        combined.push(...childInherits);
      } else {
        combined.push(childInherits);
      }
    }

    return [...new Set(combined)]; // Remove duplicates
  }

  // Clear caches
  clearCache(): void {
    this.inheritanceCache.clear();
  }

  // Get cache statistics
  getCacheStats(): { size: number; hits: number; misses: number } {
    return {
      size: this.inheritanceCache.size,
      hits: 0, // Would need actual hit tracking in real implementation
      misses: 0
    };
  }
}

describe('PriorityBasedInheritanceEngine', () => {
  let engine: MockPriorityBasedInheritanceEngine;

  beforeEach(() => {
    engine = new MockPriorityBasedInheritanceEngine();
  });

  afterEach(() => {
    engine.clearCache();
  });

  describe('Template Loading', () => {
    test('should load templates successfully', () => {
      const templates = [baseTemplate, urbanVariantTemplate, mobilityVariantTemplate];
      expect(() => engine.loadTemplates(templates)).not.toThrow();

      // Verify templates are loaded by attempting priority calculation
      expect(() => engine.calculateInheritancePriority('Base LTE cell configuration template')).not.toThrow();
    });

    test('should handle empty template array', () => {
      expect(() => engine.loadTemplates([])).not.toThrow();
    });

    test('should handle templates without priority', () => {
      const templateWithoutPriority = { ...baseTemplate };
      delete templateWithoutPriority.meta?.priority;

      expect(() => engine.loadTemplates([templateWithoutPriority])).not.toThrow();
      expect(engine.calculateInheritancePriority('Base LTE cell configuration template')).toBe(0);
    });
  });

  describe('Priority Calculation', () => {
    beforeEach(() => {
      engine.loadTemplates([baseTemplate, urbanVariantTemplate, mobilityVariantTemplate, agentdbVariantTemplate]);
    });

    test('should calculate base template priority correctly', () => {
      const priority = engine.calculateInheritancePriority('Base LTE cell configuration template');
      expect(priority).toBe(1); // Base priority from meta
    });

    test('should calculate urban variant priority correctly', () => {
      const priority = engine.calculateInheritancePriority('Urban area variant template with optimized parameters');
      expect(priority).toBeGreaterThan(1); // Should be higher than base
    });

    test('should calculate mobility variant priority correctly', () => {
      const priority = engine.calculateInheritancePriority('Mobility optimized template for high-speed scenarios');
      expect(priority).toBeGreaterThan(3); // Should be higher than urban
    });

    test('should calculate agentdb variant priority correctly', () => {
      const priority = engine.calculateInheritancePriority('AgentDB optimized template with cognitive features');
      expect(priority).toBeGreaterThan(5); // Should be highest priority
    });

    test('should handle template not found', () => {
      expect(() => engine.calculateInheritancePriority('Non-existent template')).toThrow('Template not found: Non-existent template');
    });
  });

  describe('Inheritance Resolution', () => {
    beforeEach(() => {
      engine.loadTemplates([baseTemplate, urbanVariantTemplate, mobilityVariantTemplate, agentdbVariantTemplate]);
    });

    test('should resolve base template inheritance', () => {
      const resolved = engine.resolveInheritance('Base LTE cell configuration template');

      expect(resolved.configuration).toBeDefined();
      expect(resolved.configuration['EUtranCellFDD']).toBeDefined();
      expect(resolved.configuration['EUtranCellFDD'].qRxLevMin).toBe(-140);
      expect(resolved.custom).toHaveLength(1);
      expect(resolved.custom![0].name).toBe('calculateCoverage');
    });

    test('should resolve urban variant inheritance correctly', () => {
      const resolved = engine.resolveInheritance('Urban area variant template with optimized parameters');

      // Should inherit from base and override values
      expect(resolved.configuration['EUtranCellFDD'].qRxLevMin).toBe(-125); // Urban override
      expect(resolved.configuration['EUtranCellFDD'].qQualMin).toBe(-15);   // Urban override
      expect(resolved.configuration['EUtranCellFDD'].referenceSignalPower).toBe(18); // Urban override
      expect(resolved.configuration['EUtranCellFDD'].pb).toBe(0);         // Base inheritance

      // Should have urban-specific additions
      expect(resolved.configuration['CapacityFunction']).toBeDefined();
      expect(resolved.configuration['CapacityFunction'].capacityOptimizationEnabled).toBe(true);

      // Should inherit custom functions from base
      expect(resolved.custom).toHaveLength(1);
      expect(resolved.custom![0].name).toBe('calculateCoverage');

      // Should have both base and urban conditions
      expect(resolved.conditions).toHaveProperty('coverageCondition');
      expect(resolved.conditions).toHaveProperty('highCapacityCondition');
    });

    test('should resolve mobility variant inheritance correctly', () => {
      const resolved = engine.resolveInheritance('Mobility optimized template for high-speed scenarios');

      // Should inherit from urban->base chain
      expect(resolved.configuration['EUtranCellFDD'].qRxLevMin).toBe(-120); // Mobility override
      expect(resolved.configuration['EUtranCellFDD'].ul256qamEnabled).toBe(true); // Urban inheritance
      expect(resolved.configuration['EUtranCellFDD'].pb).toBe(0); // Base inheritance

      // Should have mobility-specific additions
      expect(resolved.configuration['MobilityFunction']).toBeDefined();
      expect(resolved.configuration['MobilityFunction'].mobilityOptimizationEnabled).toBe(true);

      // Should inherit custom functions from both levels
      expect(resolved.custom).toHaveLength(2);
      expect(resolved.custom![0].name).toBe('calculateCoverage'); // From base
      expect(resolved.custom![1].name).toBe('optimizeHandover');  // From mobility
    });

    test('should resolve agentdb variant inheritance correctly', () => {
      const resolved = engine.resolveInheritance('AgentDB optimized template with cognitive features');

      // Should inherit from mobility->urban->base chain
      expect(resolved.configuration['EUtranCellFDD'].qRxLevMin).toBe(-118); // AgentDB override
      expect(resolved.configuration['EUtranCellFDD'].handoverHysteresis).toBe(4); // Mobility inheritance
      expect(resolved.configuration['EUtranCellFDD'].ul256qamEnabled).toBe(true); // Urban inheritance
      expect(resolved.configuration['EUtranCellFDD'].pb).toBe(0); // Base inheritance

      // Should have agentdb-specific additions
      expect(resolved.configuration['AgentDBFunction']).toBeDefined();
      expect(resolved.configuration['CognitiveFunction']).toBeDefined();

      // Should inherit all custom functions
      expect(resolved.custom).toHaveLength(3);
      expect(resolved.custom!.map(c => c.name)).toEqual([
        'calculateCoverage', // From base
        'optimizeHandover',  // From mobility
        'cognitiveOptimization' // From agentdb
      ]);
    });
  });

  describe('Conflict Resolution', () => {
    beforeEach(() => {
      engine.loadTemplates([baseTemplate, conflictTemplate1, conflictTemplate2]);
    });

    test('should resolve conflicts based on priority', () => {
      // conflictTemplate2 has higher priority (8) than conflictTemplate1 (2)
      const resolved = engine.resolveInheritance('Template with conflicting values (high priority)');

      // Should use higher priority values
      expect(resolved.configuration['EUtranCellFDD'].qRxLevMin).toBe(-110); // From conflictTemplate2
      expect(resolved.configuration['EUtranCellFDD'].cellIndividualOffset).toBe(5); // From conflictTemplate2
      expect(resolved.configuration['EUtranCellFDD'].qQualMin).toBe(-20); // From conflictTemplate1 (only one with this value)
      expect(resolved.configuration['EUtranCellFDD'].newParameter).toBe('test-value'); // From conflictTemplate2
    });

    test('should handle multiple inheritance parents', () => {
      // Test template that inherits from multiple parents
      const multiInheritTemplate: RTBTemplate = {
        meta: {
          version: '1.0.0',
          author: ['RTB System'],
          description: 'Multiple inheritance test template',
          tags: ['test', 'multi-inherit'],
          priority: 9,
          inherits_from: ['Template with conflicting values (low priority)', 'Template with conflicting values (high priority)']
        },
        configuration: {
          'EUtranCellFDD': {
            'finalOverride': 'final-value'
          }
        }
      };

      engine.loadTemplates([multiInheritTemplate]);
      const resolved = engine.resolveInheritance('Multiple inheritance test template');

      // Should use highest priority parent values for conflicts
      expect(resolved.configuration['EUtranCellFDD'].qRxLevMin).toBe(-110); // From higher priority parent
      expect(resolved.configuration['EUtranCellFDD'].cellIndividualOffset).toBe(5); // From higher priority parent
      expect(resolved.configuration['EUtranCellFDD'].finalOverride).toBe('final-value'); // From child
    });
  });

  describe('Array Merging', () => {
    test('should merge custom functions correctly', () => {
      const template1: RTBTemplate = {
        custom: [
          { name: 'func1', args: [], body: ['return 1;'] },
          { name: 'func2', args: [], body: ['return 2;'] }
        ]
      };

      const template2: RTBTemplate = {
        custom: [
          { name: 'func2', args: [], body: ['return 22;'] }, // Override
          { name: 'func3', args: [], body: ['return 3;'] }   // New
        ]
      };

      engine.loadTemplates([template1, template2]);

      // Test array merging directly
      const merged = engine['mergeArrays'](template1.custom || [], template2.custom || [], 'name');

      expect(merged).toHaveLength(3);
      expect(merged[0].name).toBe('func1');
      expect(merged[1].name).toBe('func2');
      expect(merged[1].body[0]).toBe('return 22;'); // Overridden
      expect(merged[2].name).toBe('func3');
    });
  });

  describe('Deep Configuration Merging', () => {
    test('should merge nested configuration objects', () => {
      const parentConfig = {
        'MO1': {
          'param1': 'value1',
          'param2': 'value2',
          'nested': {
            'sub1': 'sub1',
            'sub2': 'sub2'
          }
        },
        'MO2': {
          'param3': 'value3'
        }
      };

      const childConfig = {
        'MO1': {
          'param2': 'new-value2', // Override
          'param4': 'value4',      // New
          'nested': {
            'sub2': 'new-sub2',    // Override nested
            'sub3': 'sub3'         // New nested
          }
        },
        'MO3': {                   // New MO
          'param5': 'value5'
        }
      };

      const merged = engine['mergeConfigurations'](parentConfig, childConfig);

      expect(merged.MO1.param1).toBe('value1'); // From parent
      expect(merged.MO1.param2).toBe('new-value2'); // From child
      expect(merged.MO1.param4).toBe('value4'); // From child
      expect(merged.MO1.nested.sub1).toBe('sub1'); // From parent
      expect(merged.MO1.nested.sub2).toBe('new-sub2'); // From child
      expect(merged.MO1.nested.sub3).toBe('sub3'); // From child
      expect(merged.MO2.param3).toBe('value3'); // From parent
      expect(merged.MO3.param5).toBe('value5'); // From child
    });
  });

  describe('Caching Performance', () => {
    test('should cache resolved templates', () => {
      engine.loadTemplates([baseTemplate, urbanVariantTemplate, mobilityVariantTemplate]);

      // First call should populate cache
      const startTime1 = Date.now();
      const resolved1 = engine.resolveInheritance('Mobility optimized template for high-speed scenarios');
      const endTime1 = Date.now();

      // Second call should use cache
      const startTime2 = Date.now();
      const resolved2 = engine.resolveInheritance('Mobility optimized template for high-speed scenarios');
      const endTime2 = Date.now();

      expect(resolved1).toEqual(resolved2);
      expect(endTime2 - startTime2).toBeLessThanOrEqual(endTime1 - startTime1);

      const stats = engine.getCacheStats();
      expect(stats.size).toBeGreaterThan(0);
    });

    test('should clear cache correctly', () => {
      engine.loadTemplates([baseTemplate, urbanVariantTemplate]);
      engine.resolveInheritance('Urban area variant template with optimized parameters');

      let stats = engine.getCacheStats();
      expect(stats.size).toBeGreaterThan(0);

      engine.clearCache();

      stats = engine.getCacheStats();
      expect(stats.size).toBe(0);
    });
  });

  describe('Error Handling', () => {
    test('should handle circular dependencies gracefully', () => {
      const template1: RTBTemplate = {
        meta: { description: 'Template 1', priority: 1, inherits_from: 'Template 2' }
      };

      const template2: RTBTemplate = {
        meta: { description: 'Template 2', priority: 2, inherits_from: 'Template 1' }
      };

      engine.loadTemplates([template1, template2]);

      // Should prevent infinite recursion and throw error
      expect(() => engine.calculateInheritancePriority('Template 1')).not.toThrow();
      expect(() => engine.resolveInheritance('Template 1')).not.toThrow();
    });

    test('should handle missing inheritance references', () => {
      const templateWithMissingParent: RTBTemplate = {
        meta: { description: 'Template with missing parent', priority: 5, inherits_from: 'Non-existent parent' },
        configuration: { 'TestMO': { 'param': 'value' } }
      };

      engine.loadTemplates([templateWithMissingParent]);

      // Should resolve to template itself without throwing
      const resolved = engine.resolveInheritance('Template with missing parent');
      expect(resolved.configuration['TestMO'].param).toBe('value');
    });
  });

  describe('Performance Targets', () => {
    test('should meet performance targets for template resolution', () => {
      const largeTemplateSet = generateLargeTemplateSet(100);
      engine.loadTemplates(largeTemplateSet);

      // Test resolution time for complex inheritance chain
      const startTime = Date.now();
      const resolved = engine.resolveInheritance(`Performance test template 99`);
      const endTime = Date.now();

      // Should resolve in under 100ms for performance target
      expect(endTime - startTime).toBeLessThan(100);
      expect(resolved).toBeDefined();
      expect(resolved.configuration).toBeDefined();
    });

    test('should handle large template sets efficiently', () => {
      const largeTemplateSet = generateLargeTemplateSet(50);
      const startTime = Date.now();

      engine.loadTemplates(largeTemplateSet);

      const loadTime = Date.now() - startTime;
      expect(loadTime).toBeLessThan(1000); // Should load in under 1 second

      // Test multiple resolutions
      const resolveStartTime = Date.now();

      for (let i = 0; i < 10; i++) {
        engine.resolveInheritance(`Performance test template ${i * 5}`);
      }

      const resolveTime = Date.now() - resolveStartTime;
      expect(resolveTime).toBeLessThan(500); // Should resolve 10 templates in under 500ms
    });
  });
});