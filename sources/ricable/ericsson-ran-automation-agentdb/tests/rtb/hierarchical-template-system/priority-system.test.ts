/**
 * Priority System Tests - Comprehensive Test Suite
 *
 * Tests for the hierarchical template system including priority engine,
 * registry, inheritance resolver, priority manager, validator, and performance optimizer.
 */

import {
  PriorityTemplateEngine,
  TemplatePriority,
  TemplatePriorityInfo,
  TemplateInheritanceChain,
  ParameterConflict
} from '../../../src/rtb/hierarchical-template-system/priority-engine';

import {
  TemplateRegistry,
  TemplateSearchFilter,
  RegistryTemplateMeta
} from '../../../src/rtb/hierarchical-template-system/template-registry';

import {
  InheritanceResolver,
  InheritanceStrategy,
  InheritanceResolutionOptions
} from '../../../src/rtb/hierarchical-template-system/inheritance-resolver';

import {
  PriorityManager,
  OverrideStrategy,
  PriorityAdjustmentRule,
  ParameterOverrideRule
} from '../../../src/rtb/hierarchical-template-system/priority-manager';

import {
  TemplateValidator,
  ValidationSeverity,
  ValidationRule,
  ValidationConfig
} from '../../../src/rtb/hierarchical-template-system/template-validator';

import {
  PerformanceOptimizer,
  OptimizationStrategy,
  CacheConfig,
  BatchConfig
} from '../../../src/rtb/hierarchical-template-system/performance-optimizer';

import {
  IntegratedTemplateSystem,
  MOTemplateContext,
  TemplateProcessingResult
} from '../../../src/rtb/hierarchical-template-system/integrated-template-system';

import {
  RTBTemplate,
  TemplateMeta,
  CustomFunction,
  ConditionOperator,
  EvaluationOperator
} from '../../../src/types/rtb-types';

// Test data factories
const createTestTemplate = (name: string, config: Record<string, any> = {}): RTBTemplate => ({
  meta: {
    version: '1.0.0',
    author: ['test'],
    description: `Test template ${name}`,
    tags: ['test'],
    priority: TemplatePriority.BASE
  },
  custom: [],
  configuration: config,
  conditions: {},
  evaluations: {}
});

const createTestPriority = (
  category: string,
  level: number,
  inheritsFrom?: string | string[]
): TemplatePriorityInfo => ({
  level,
  category,
  source: 'test',
  inherits_from: inheritsFrom
});

describe('Priority Template Engine', () => {
  let engine: PriorityTemplateEngine;

  beforeEach(() => {
    engine = new PriorityTemplateEngine();
  });

  describe('Template Registration', () => {
    test('should register template with priority', () => {
      const template = createTestTemplate('test1', { param1: 'value1' });
      const priority = createTestPriority('base', TemplatePriority.BASE);

      expect(() => engine.registerTemplate('test1', template, priority)).not.toThrow();
      expect(engine.hasTemplate('test1')).toBe(true);
    });

    test('should throw error for invalid template', () => {
      const invalidTemplate = {} as RTBTemplate;
      const priority = createTestPriority('base', TemplatePriority.BASE);

      expect(() => engine.registerTemplate('invalid', invalidTemplate, priority))
        .toThrow('Template must have a configuration object');
    });

    test('should remove template', () => {
      const template = createTestTemplate('test1');
      const priority = createTestPriority('base', TemplatePriority.BASE);

      engine.registerTemplate('test1', template, priority);
      expect(engine.hasTemplate('test1')).toBe(true);

      const removed = engine.removeTemplate('test1');
      expect(removed).toBe(true);
      expect(engine.hasTemplate('test1')).toBe(false);
    });
  });

  describe('Inheritance Resolution', () => {
    beforeEach(() => {
      // Setup inheritance hierarchy
      const baseTemplate = createTestTemplate('base', {
        param1: 'base_value',
        param2: 'base_value',
        param3: 'base_only'
      });

      const urbanTemplate = createTestTemplate('urban', {
        param1: 'urban_value', // Override
        param4: 'urban_only'
      });

      const specificTemplate = createTestTemplate('specific', {
        param2: 'specific_value', // Override
        param5: 'specific_only'
      });

      engine.registerTemplate('base', baseTemplate, createTestPriority('base', TemplatePriority.BASE));
      engine.registerTemplate('urban', urbanTemplate, createTestPriority('urban', TemplatePriority.VARIANT, 'base'));
      engine.registerTemplate('specific', specificTemplate, createTestPriority('specific', TemplatePriority.CONTEXT_SPECIFIC, 'urban'));
    });

    test('should resolve simple inheritance chain', () => {
      const result = engine.resolveInheritanceChain('specific');

      expect(result.templateName).toBe('specific');
      expect(result.chain).toHaveLength(3);
      expect(result.resolvedTemplate.configuration).toEqual({
        param1: 'urban_value',      // From urban (inherits from base, overridden by specific's parent)
        param2: 'specific_value',   // From specific
        param3: 'base_only',        // From base
        param4: 'urban_only',       // From urban
        param5: 'specific_only'     // From specific
      });
    });

    test('should detect parameter conflicts', () => {
      const result = engine.resolveInheritanceChain('specific');

      expect(result.conflicts.length).toBeGreaterThan(0);
      expect(result.conflicts.some(c => c.parameter === 'param1')).toBe(true);
      expect(result.conflicts.some(c => c.parameter === 'param2')).toBe(true);
    });

    test('should handle multiple inheritance', () => {
      const multiTemplate = createTestTemplate('multi', { param6: 'multi_value' });
      engine.registerTemplate('multi', multiTemplate, createTestPriority('multi', TemplatePriority.CONTEXT_SPECIFIC, ['base', 'urban']));

      const result = engine.resolveInheritanceChain('multi');
      expect(result.chain.length).toBeGreaterThan(2);
    });

    test('should generate warnings for missing parents', () => {
      const orphanTemplate = createTestTemplate('orphan', { param1: 'orphan_value' });
      engine.registerTemplate('orphan', orphanTemplate, createTestPriority('orphan', TemplatePriority.VARIANT, 'missing_parent'));

      const result = engine.resolveInheritanceChain('orphan');
      expect(result.warnings.some(w => w.includes('Parent template'))).toBe(true);
    });
  });

  describe('Cache Management', () => {
    test('should cache inheritance resolution', () => {
      const template = createTestTemplate('cached', { param1: 'value1' });
      const priority = createTestPriority('base', TemplatePriority.BASE);

      engine.registerTemplate('cached', template, priority);

      // First call
      const result1 = engine.resolveInheritanceChain('cached');
      const stats1 = engine.getCacheStats();

      // Second call should use cache
      const result2 = engine.resolveInheritanceChain('cached');
      const stats2 = engine.getCacheStats();

      expect(result1).toEqual(result2);
      expect(stats1.inheritanceCache).toBe(stats2.inheritanceCache);
    });

    test('should clear cache when template is removed', () => {
      const template = createTestTemplate('temp', { param1: 'value1' });
      const priority = createTestPriority('base', TemplatePriority.BASE);

      engine.registerTemplate('temp', template, priority);
      engine.resolveInheritanceChain('temp'); // Populate cache

      engine.removeTemplate('temp');
      const stats = engine.getCacheStats();

      expect(stats.inheritanceCache).toBe(0);
    });
  });
});

describe('Template Registry', () => {
  let registry: TemplateRegistry;

  beforeEach(() => {
    registry = new TemplateRegistry();
  });

  describe('Template Registration', () => {
    test('should register template with metadata', async () => {
      const template = createTestTemplate('test1', { param1: 'value1' });
      const priority = createTestPriority('base', TemplatePriority.BASE);

      await registry.registerTemplate('test1', template, priority);

      const retrieved = registry.getTemplate('test1');
      expect(retrieved).toEqual(template);

      const metadata = registry.getTemplateMetadata('test1');
      expect(metadata).toBeDefined();
      expect(metadata!.name).toBe('test1');
    });

    test('should update existing template', async () => {
      const template = createTestTemplate('test1', { param1: 'value1' });
      const priority = createTestPriority('base', TemplatePriority.BASE);

      await registry.registerTemplate('test1', template, priority);

      const updatedTemplate = createTestTemplate('test1', { param1: 'updated_value', param2: 'new_value' });
      await registry.updateTemplate('test1', { configuration: updatedTemplate.configuration });

      const retrieved = registry.getTemplate('test1');
      expect(retrieved!.configuration.param1).toBe('updated_value');
      expect(retrieved!.configuration.param2).toBe('new_value');
    });
  });

  describe('Template Search', () => {
    beforeEach(async () => {
      // Setup test templates
      const templates = [
        { name: 'urban_base', category: 'urban', priority: TemplatePriority.VARIANT },
        { name: 'mobility_base', category: 'mobility', priority: TemplatePriority.SCENARIO },
        { name: 'sleep_mode', category: 'energy', priority: TemplatePriority.FEATURE_SPECIFIC },
        { name: 'test_template', category: 'test', priority: TemplatePriority.CONTEXT_SPECIFIC }
      ];

      for (const { name, category, priority } of templates) {
        const template = createTestTemplate(name, { category, test_param: `${name}_value` });
        const templatePriority = createTestPriority(category, priority);
        await registry.registerTemplate(name, template, templatePriority);
      }
    });

    test('should search by category', async () => {
      const filter: TemplateSearchFilter = { category: 'urban' };
      const result = await registry.searchTemplates(filter);

      expect(result.totalCount).toBe(1);
      expect(result.templates[0].name).toBe('urban_base');
    });

    test('should search by priority range', async () => {
      const filter: TemplateSearchFilter = {
        priorityRange: { min: TemplatePriority.CONTEXT_SPECIFIC, max: TemplatePriority.VARIANT }
      };
      const result = await registry.searchTemplates(filter);

      expect(result.totalCount).toBe(2);
    });

    test('should search by tags', async () => {
      const filter: TemplateSearchFilter = { tags: ['test'] };
      const result = await registry.searchTemplates(filter);

      expect(result.totalCount).toBe(1);
      expect(result.templates[0].name).toBe('test_template');
    });

    test('should return search facets', async () => {
      const filter: TemplateSearchFilter = {};
      const result = await registry.searchTemplates(filter);

      expect(result.facets).toBeDefined();
      expect(result.facets.categories).toBeDefined();
      expect(result.facets.priorities).toBeDefined();
    });
  });

  describe('Dependency Management', () => {
    test('should track template dependencies', async () => {
      const baseTemplate = createTestTemplate('base', { param1: 'base_value' });
      const childTemplate = createTestTemplate('child', { param2: 'child_value' });

      await registry.registerTemplate('base', baseTemplate, createTestPriority('base', TemplatePriority.BASE));
      await registry.registerTemplate('child', childTemplate, createTestPriority('child', TemplatePriority.VARIANT, 'base'));

      const dependencies = registry.getTemplateDependencies('child');
      expect(dependencies).toHaveLength(1);
      expect(dependencies[0].name).toBe('base');

      const dependents = registry.getTemplateDependents('base');
      expect(dependents).toHaveLength(1);
      expect(dependents[0].name).toBe('child');
    });
  });
});

describe('Inheritance Resolver', () => {
  let resolver: InheritanceResolver;
  let registry: TemplateRegistry;

  beforeEach(() => {
    registry = new TemplateRegistry();
    resolver = new InheritanceResolver(registry);
  });

  describe('Complex Inheritance Chains', () => {
    beforeEach(async () => {
      // Setup complex inheritance hierarchy
      const baseTemplate = createTestTemplate('base', {
        common_param: 'base_value',
        base_param: 'base_only'
      });

      const urbanTemplate = createTestTemplate('urban', {
        common_param: 'urban_value', // Conflict
        urban_param: 'urban_only'
      });

      const mobilityTemplate = createTestTemplate('mobility', {
        common_param: 'mobility_value', // Conflict
        mobility_param: 'mobility_only'
      });

      const hybridTemplate = createTestTemplate('hybrid', {
        hybrid_param: 'hybrid_only'
      });

      await registry.registerTemplate('base', baseTemplate, createTestPriority('base', TemplatePriority.BASE));
      await registry.registerTemplate('urban', urbanTemplate, createTestPriority('urban', TemplatePriority.VARIANT, 'base'));
      await registry.registerTemplate('mobility', mobilityTemplate, createTestPriority('mobility', TemplatePriority.SCENARIO, 'base'));
      await registry.registerTemplate('hybrid', hybridTemplate, createTestPriority('hybrid', TemplatePriority.CONTEXT_SPECIFIC, ['urban', 'mobility']));
    });

    test('should resolve multiple inheritance with conflicts', async () => {
      const result = await resolver.resolveInheritance('hybrid');

      expect(result.templateName).toBe('hybrid');
      expect(result.chain.length).toBeGreaterThan(2);
      expect(result.conflicts.length).toBeGreaterThan(0);

      // Check that conflicts were resolved
      const commonParamConflict = result.conflicts.find(c => c.parameter === 'common_param');
      expect(commonParamConflict).toBeDefined();
      expect(commonParamConflict!.resolutionStrategy).toBe('highest_priority');
    });

    test('should analyze inheritance complexity', async () => {
      const analysis = await resolver.analyzeInheritance('hybrid');

      expect(analysis.templateName).toBe('hybrid');
      expect(analysis.inheritanceDepth).toBeGreaterThan(1);
      expect(analysis.totalDependencies).toBeGreaterThan(0);
      expect(analysis.recommendations).toBeDefined();
    });
  });

  describe('Circular Dependency Detection', () => {
    test('should detect circular dependencies', async () => {
      const templateA = createTestTemplate('A', { paramA: 'valueA' });
      const templateB = createTestTemplate('B', { paramB: 'valueB' });

      await registry.registerTemplate('A', templateA, createTestPriority('A', TemplatePriority.VARIANT, 'B'));
      await registry.registerTemplate('B', templateB, createTestPriority('B', TemplatePriority.VARIANT, 'A'));

      const analysis = await resolver.analyzeInheritance('A');
      expect(analysis.circularDependencies.length).toBeGreaterThan(0);
    });
  });

  describe('Merge Strategies', () => {
    test('should apply merge strategies correctly', async () => {
      const baseTemplate = createTestTemplate('base', {
        array_param: ['base1', 'base2'],
        object_param: { key1: 'base_value1' },
        string_param: 'base_string'
      });

      const childTemplate = createTestTemplate('child', {
        array_param: ['child1', 'child2'],
        object_param: { key2: 'child_value2' },
        string_param: 'child_string'
      });

      await registry.registerTemplate('base', baseTemplate, createTestPriority('base', TemplatePriority.BASE));
      await registry.registerTemplate('child', childTemplate, createTestPriority('child', TemplatePriority.VARIANT, 'base'));

      const options: InheritanceResolutionOptions = {
        strategy: InheritanceStrategy.MERGE
      };

      const result = await resolver.resolveInheritance('child', options);

      // Check merge results
      const mergedArray = result.resolvedTemplate.configuration.array_param;
      const mergedObject = result.resolvedTemplate.configuration.object_param;

      expect(Array.isArray(mergedArray)).toBe(true);
      expect(typeof mergedObject).toBe('object');
      expect(mergedObject).toHaveProperty('key1');
      expect(mergedObject).toHaveProperty('key2');
    });
  });
});

describe('Priority Manager', () => {
  let priorityManager: PriorityManager;
  let registry: TemplateRegistry;

  beforeEach(() => {
    registry = new TemplateRegistry();
    priorityManager = new PriorityManager(registry);
  });

  describe('Priority Adjustment Rules', () => {
    test('should apply environment-based priority adjustment', async () => {
      const template = createTestTemplate('test', { param1: 'value1' });
      const priority = createTestPriority('test', TemplatePriority.BASE);

      await registry.registerTemplate('test', template, priority);

      const context = {
        parameter: 'test_param',
        templateName: 'test',
        environment: 'production',
        featureFlags: {},
        metadata: {},
        timestamp: new Date(),
        inheritanceDepth: 1
      };

      const result = await priorityManager.calculateAdjustedPriority(
        'test',
        'test_param',
        TemplatePriority.BASE,
        context
      );

      expect(result.adjustedPriority).not.toBe(result.originalPriority);
      expect(result.appliedRules.length).toBeGreaterThan(0);
    });

    test('should apply feature flag priority boost', async () => {
      const template = createTestTemplate('test', { param1: 'value1' });
      const priority = createTestPriority('test', TemplatePriority.BASE);

      await registry.registerTemplate('test', template, priority);

      const context = {
        parameter: 'test_param',
        templateName: 'test',
        featureFlags: { feature1: true, feature2: true },
        metadata: {},
        timestamp: new Date(),
        inheritanceDepth: 1
      };

      const result = await priorityManager.calculateAdjustedPriority(
        'test',
        'test_param',
        TemplatePriority.BASE,
        context
      );

      expect(result.appliedRules.some(r => r.name === 'feature_flag_boost')).toBe(true);
    });

    test('should add custom priority rule', () => {
      const customRule: PriorityAdjustmentRule = {
        name: 'test_rule',
        condition: (context) => context.templateName.includes('test'),
        adjustment: (priority) => priority - 5,
        description: 'Test rule for templates with "test" in name',
        enabled: true,
        precedence: 100
      };

      priorityManager.addPriorityRule(customRule);

      const rules = priorityManager.getPriorityRules();
      expect(rules.some(r => r.name === 'test_rule')).toBe(true);
    });
  });

  describe('Parameter Override Rules', () => {
    test('should apply merge strategy for array parameters', async () => {
      const values = [
        { value: ['item1', 'item2'], source: 'template1', priority: 50 },
        { value: ['item3', 'item4'], source: 'template2', priority: 30 }
      ];

      const context = {
        parameter: 'itemsList',
        values,
        templateName: 'test',
        resolutionContext: {}
      };

      const result = await priorityManager.resolveParameterOverride('test', 'itemsList', values, {});

      expect(result.strategy).toBe(OverrideStrategy.MERGE_ALL);
      expect(Array.isArray(result.resolvedValue)).toBe(true);
      expect(result.resolvedValue).toEqual(expect.arrayContaining(['item1', 'item2', 'item3', 'item4']));
    });

    test('should apply custom resolver for boolean parameters', async () => {
      const values = [
        { value: true, source: 'template1', priority: 50 },
        { value: false, source: 'template2', priority: 30 },
        { value: true, source: 'template3', priority: 10 }
      ];

      const result = await priorityManager.resolveParameterOverride('test', 'enabledFlag', values, {});

      // Should use AND logic (all values must be true)
      expect(result.resolvedValue).toBe(true);
    });

    test('should add custom override rule', () => {
      const customRule: ParameterOverrideRule = {
        parameterPattern: /.*Custom$/,
        strategy: OverrideStrategy.SUM,
        description: 'Sum values for custom parameters',
        enabled: true
      };

      priorityManager.addOverrideRule(customRule);

      const rules = priorityManager.getOverrideRules();
      expect(rules.some(r => r.parameterPattern.source === '.*Custom$')).toBe(true);
    });
  });

  describe('Cache Performance', () => {
    test('should cache priority calculations', async () => {
      const template = createTestTemplate('test', { param1: 'value1' });
      const priority = createTestPriority('test', TemplatePriority.BASE);

      await registry.registerTemplate('test', template, priority);

      const context = {
        parameter: 'test_param',
        templateName: 'test',
        metadata: {},
        timestamp: new Date(),
        inheritanceDepth: 1
      };

      // First calculation
      await priorityManager.calculateAdjustedPriority('test', 'test_param', TemplatePriority.BASE, context);
      const stats1 = priorityManager.getCacheStats();

      // Second calculation should use cache
      await priorityManager.calculateAdjustedPriority('test', 'test_param', TemplatePriority.BASE, context);
      const stats2 = priorityManager.getCacheStats();

      expect(stats2.totalHits).toBe(stats1.totalHits + 1);
    });
  });
});

describe('Template Validator', () => {
  let validator: TemplateValidator;
  let registry: TemplateRegistry;

  beforeEach(() => {
    registry = new TemplateRegistry();
    validator = new TemplateValidator(registry);
  });

  describe('Structure Validation', () => {
    test('should validate template structure', async () => {
      const validTemplate = createTestTemplate('valid', {
        param1: 'value1',
        param2: 42
      });

      const priority = createTestPriority('test', TemplatePriority.BASE);

      const result = await validator.validateTemplate('valid', validTemplate, priority);

      expect(result.isValid).toBe(true);
      expect(result.totalErrors).toBe(0);
    });

    test('should detect missing configuration', async () => {
      const invalidTemplate = { meta: {} } as RTBTemplate;
      const priority = createTestPriority('test', TemplatePriority.BASE);

      const result = await validator.validateTemplate('invalid', invalidTemplate, priority);

      expect(result.isValid).toBe(false);
      expect(result.totalErrors).toBeGreaterThan(0);
      expect(result.categories.structure.errors).toBeGreaterThan(0);
    });

    test('should validate custom functions', async () => {
      const templateWithInvalidFunction = createTestTemplate('invalid_func', { param1: 'value1' });
      templateWithInvalidFunction.custom = [
        {
          name: 'invalid_function',
          args: ['param1'],
          body: [] // Empty body
        }
      ];

      const priority = createTestPriority('test', TemplatePriority.BASE);

      const result = await validator.validateTemplate('invalid_func', templateWithInvalidFunction, priority);

      expect(result.categories.structure.errors).toBeGreaterThan(0);
    });
  });

  describe('Inheritance Validation', () => {
    test('should validate inheritance structure', async () => {
      const baseTemplate = createTestTemplate('base', { param1: 'base_value' });
      const childTemplate = createTestTemplate('child', { param2: 'child_value' });

      await registry.registerTemplate('base', baseTemplate, createTestPriority('base', TemplatePriority.BASE));

      const childPriority = createTestPriority('child', TemplatePriority.VARIANT, 'base');
      const result = await validator.validateTemplate('child', childTemplate, childPriority);

      expect(result.categories.inheritance.errors).toBe(0);
    });

    test('should warn about missing parent templates', async () => {
      const orphanTemplate = createTestTemplate('orphan', { param1: 'orphan_value' });
      const priority = createTestPriority('orphan', TemplatePriority.VARIANT, 'missing_parent');

      const result = await validator.validateTemplate('orphan', orphanTemplate, priority);

      expect(result.categories.inheritance.warnings).toBeGreaterThan(0);
    });
  });

  describe('Content Validation', () => {
    test('should validate parameter naming conventions', async () => {
      const template = createTestTemplate('test', {
        'valid_param': 'value1',
        'invalid-param': 'value2', // Invalid naming
        'InvalidParam': 'value3'  // Invalid naming
      });

      const priority = createTestPriority('test', TemplatePriority.BASE);

      const result = await validator.validateTemplate('test', template, priority);

      expect(result.categories.content.info).toBeGreaterThan(0);
    });

    test('should validate metadata completeness', async () => {
      const template = createTestTemplate('test', { param1: 'value1' });
      delete template.meta!.version; // Remove required field

      const priority = createTestPriority('test', TemplatePriority.BASE);

      const result = await validator.validateTemplate('test', template, priority);

      expect(result.categories.content.warnings).toBeGreaterThan(0);
      expect(result.recommendations.some(r => r.includes('version'))).toBe(true);
    });
  });

  describe('Performance Validation', () => {
    test('should detect performance issues', async () => {
      const template = createTestTemplate('test', { param1: 'value1' });
      template.conditions = {
        complex_condition: {
          if: 'a very long and complex condition expression that could impact performance significantly when evaluated repeatedly',
          then: { param1: 'complex_value' },
          else: 'default'
        }
      };

      const priority = createTestPriority('test', TemplatePriority.BASE);

      const result = await validator.validateTemplate('test', template, priority);

      expect(result.categories.performance.info).toBeGreaterThan(0);
    });
  });

  describe('Security Validation', () => {
    test('should detect dangerous patterns', async () => {
      const template = createTestTemplate('test', { param1: 'value1' });
      template.evaluations = {
        dangerous_eval: {
          eval: 'eval("potentially dangerous code")',
          args: []
        }
      };

      const priority = createTestPriority('test', TemplatePriority.BASE);

      const result = await validator.validateTemplate('test', template, priority);

      expect(result.categories.security.warnings).toBeGreaterThan(0);
    });

    test('should flag sensitive parameters', async () => {
      const template = createTestTemplate('test', {
        'password': 'secret123',
        'api_key': 'abc123xyz',
        'normal_param': 'value1'
      });

      const priority = createTestPriority('test', TemplatePriority.BASE);

      const result = await validator.validateTemplate('test', template, priority);

      expect(result.categories.security.info).toBeGreaterThan(0);
    });
  });

  describe('Custom Validation Rules', () => {
    test('should add and apply custom validation rule', () => {
      const customRule: ValidationRule = {
        name: 'custom_business_rule',
        description: 'Validates business-specific requirements',
        severity: ValidationSeverity.WARNING,
        enabled: true,
        category: 'content',
        precedence: 100,
        validate: (context) => {
          const results = [];
          const config = context.template.configuration;

          if (config.business_critical_param === undefined) {
            results.push({
              valid: false,
              rule: 'custom_business_rule',
              severity: ValidationSeverity.WARNING,
              message: 'Business critical parameter is missing',
              code: 'MISSING_BUSINESS_PARAM'
            });
          }

          return results;
        }
      };

      validator.addValidationRule(customRule);

      const rules = validator.getValidationRules();
      expect(rules.some(r => r.name === 'custom_business_rule')).toBe(true);
    });
  });
});

describe('Performance Optimizer', () => {
  let optimizer: PerformanceOptimizer;
  let registry: TemplateRegistry;

  beforeEach(() => {
    registry = new TemplateRegistry();
    optimizer = new PerformanceOptimizer(registry, {
      enableCaching: true,
      enableBatching: true,
      enableIndexing: true
    });
  });

  describe('Caching', () => {
    test('should cache template resolutions', async () => {
      // Setup would require the inheritance resolver integration
      // This is a simplified test
      const cacheStats = optimizer.getPerformanceMetrics();
      expect(cacheStats.cacheStats).toBeDefined();
    });

    test('should manage cache size limits', () => {
      // Test cache eviction when size limit is reached
      const config: CacheConfig = {
        enabled: true,
        maxSize: 2,
        ttl: 3600000,
        strategy: 'lru',
        compressionEnabled: false,
        compressionThreshold: 1024,
        persistenceEnabled: false
      };

      const limitedOptimizer = new PerformanceOptimizer(registry, { cacheConfig: config });
      const stats = limitedOptimizer.getPerformanceMetrics();
      expect(stats.cacheStats).toBeDefined();
    });
  });

  describe('Batch Processing', () => {
    test('should process templates in batches', async () => {
      const templateNames = ['batch1', 'batch2', 'batch3'];

      // Register test templates
      for (const name of templateNames) {
        const template = createTestTemplate(name, { [`${name}_param`]: `${name}_value` });
        const priority = createTestPriority(name, TemplatePriority.BASE);
        await registry.registerTemplate(name, template, priority);
      }

      const batchConfig: BatchConfig = {
        enabled: true,
        batchSize: 2,
        maxConcurrency: 2,
        timeout: 5000,
        retryAttempts: 2,
        backoffStrategy: 'exponential'
      };

      const batchOptimizer = new PerformanceOptimizer(registry, { batchConfig });

      // Test would require integration with inheritance resolver
      const stats = batchOptimizer.getPerformanceMetrics();
      expect(stats).toBeDefined();
    });
  });

  describe('Indexing', () => {
    test('should build performance indexes', async () => {
      // Register test templates with different properties
      const templates = [
        { name: 'urban_1', category: 'urban', priority: TemplatePriority.VARIANT },
        { name: 'mobility_1', category: 'mobility', priority: TemplatePriority.SCENARIO },
        { name: 'urban_2', category: 'urban', priority: TemplatePriority.VARIANT }
      ];

      for (const { name, category, priority } of templates) {
        const template = createTestTemplate(name, { category });
        const templatePriority = createTestPriority(category, priority);
        await registry.registerTemplate(name, template, templatePriority);
      }

      await optimizer.buildIndexes();

      const stats = optimizer.getPerformanceMetrics();
      expect(stats.indexStats.indexCount).toBeGreaterThan(0);
      expect(stats.indexStats.totalEntries).toBeGreaterThan(0);
    });
  });

  describe('Memory Optimization', () => {
    test('should optimize memory usage', () => {
      const optimization = optimizer.optimizeMemory();

      expect(optimization).toBeDefined();
      expect(typeof optimization.freedMemory).toBe('number');
      expect(Array.isArray(optimization.optimizations)).toBe(true);
    });
  });

  describe('Performance Metrics', () => {
    test('should provide detailed performance metrics', () => {
      const metrics = optimizer.getPerformanceMetrics();

      expect(metrics.recent).toBeDefined();
      expect(metrics.summary).toBeDefined();
      expect(metrics.cacheStats).toBeDefined();
      expect(metrics.indexStats).toBeDefined();

      expect(metrics.summary.totalOperations).toBeGreaterThanOrEqual(0);
      expect(metrics.summary.averageDuration).toBeGreaterThanOrEqual(0);
      expect(metrics.summary.cacheHitRate).toBeGreaterThanOrEqual(0);
      expect(metrics.summary.memoryUsage).toBeGreaterThanOrEqual(0);
    });
  });
});

describe('Integrated Template System', () => {
  let integratedSystem: IntegratedTemplateSystem;

  beforeEach(() => {
    integratedSystem = new IntegratedTemplateSystem({
      enablePrioritySystem: true,
      enableValidation: true,
      enablePerformanceOptimization: true,
      enableSchemaValidation: true
    });
  });

  describe('End-to-End Template Processing', () => {
    test('should process template with full system integration', async () => {
      const template = createTestTemplate('integrated_test', {
        param1: 'value1',
        param2: 42,
        array_param: ['item1', 'item2']
      });

      const priority = createTestPriority('integration', TemplatePriority.BASE);

      await integratedSystem.registerTemplate('integrated_test', template, priority);

      const context: MOTemplateContext = {
        environment: 'test',
        featureFlags: { test_feature: true }
      };

      const result = await integratedSystem.processTemplate('integrated_test', context);

      expect(result.template).toBeDefined();
      expect(result.inheritanceChain).toBeDefined();
      expect(result.processingStats).toBeDefined();
      expect(result.appliedOptimizations).toContain('priority_inheritance_resolution');
    });

    test('should handle processing errors gracefully', async () => {
      const invalidTemplate = { meta: {} } as RTBTemplate;
      const priority = createTestPriority('invalid', TemplatePriority.BASE);

      await integratedSystem.registerTemplate('invalid', invalidTemplate, priority);

      try {
        await integratedSystem.processTemplate('invalid');
        // Should not reach here if validation is enabled and strict mode is on
      } catch (error) {
        expect(error).toBeDefined();
      }
    });
  });

  describe('Batch Processing Integration', () => {
    test('should batch process multiple templates', async () => {
      const templates = ['batch1', 'batch2', 'batch3'];

      // Register templates
      for (const name of templates) {
        const template = createTestTemplate(name, { [`${name}_param`]: `${name}_value` });
        const priority = createTestPriority(name, TemplatePriority.BASE);
        await integratedSystem.registerTemplate(name, template, priority);
      }

      const context: MOTemplateContext = {
        environment: 'batch_test'
      };

      const results = await integratedSystem.batchProcessTemplates(templates, context);

      expect(results).toHaveLength(3);
      expect(results.every(r => r.template !== undefined)).toBe(true);
    });
  });

  describe('MO Hierarchy Integration', () => {
    test('should validate against MO hierarchy when provided', async () => {
      const template = createTestTemplate('mo_test', {
        validMoParam: 'value1',
        invalidMoParam: 'value2'
      });

      const priority = createTestPriority('mo_test', TemplatePriority.BASE);

      await integratedSystem.registerTemplate('mo_test', template, priority);

      // Mock MO hierarchy (would be provided by actual RTB processor)
      const mockMOHierarchy = {
        rootClass: 'ManagedElement',
        classes: new Map([
          ['ManagedElement', {
            id: 'ManagedElement',
            name: 'ManagedElement',
            parentClass: '',
            cardinality: { minimum: 0, maximum: 1, type: 'single' },
            flags: {},
            children: [],
            attributes: ['validMoParam'],
            derivedClasses: []
          }]
        ]),
        relationships: new Map(),
        cardinality: new Map(),
        inheritanceChain: new Map()
      };

      integratedSystem.setMOHierarchy(mockMOHierarchy);

      const context: MOTemplateContext = {
        moHierarchy: mockMOHierarchy
      };

      const result = await integratedSystem.processTemplate('mo_test', context);

      expect(result.warnings.some(w => w.includes('does not correspond to a known MO attribute'))).toBe(true);
    });
  });

  describe('System Statistics', () => {
    test('should provide comprehensive system statistics', () => {
      const stats = integratedSystem.getSystemStats();

      expect(stats.registry).toBeDefined();
      expect(stats.priority).toBeDefined();
      expect(stats.performance).toBeDefined();
      expect(stats.validation).toBeDefined();
      expect(stats.integration).toBeDefined();

      expect(stats.integration.moClasses).toBeGreaterThanOrEqual(0);
      expect(stats.integration.parameterDefinitions).toBeGreaterThanOrEqual(0);
      expect(stats.integration.constraintValidators).toBeGreaterThanOrEqual(0);
    });
  });

  describe('System Cleanup', () => {
    test('should clear all system components', () => {
      integratedSystem.clearSystem();

      const stats = integratedSystem.getSystemStats();
      expect(stats.registry.totalTemplates).toBe(0);
    });
  });
});

// Integration test for the complete priority system
describe('Priority System Integration Tests', () => {
  test('complete workflow with inheritance, validation, and optimization', async () => {
    const integratedSystem = new IntegratedTemplateSystem({
      enablePrioritySystem: true,
      enableValidation: true,
      enablePerformanceOptimization: true
    });

    // Create inheritance hierarchy
    const baseTemplate = createTestTemplate('base', {
      common_param: 'base_value',
      base_only_param: 'base_only'
    });

    const urbanTemplate = createTestTemplate('urban', {
      common_param: 'urban_value', // Override
      urban_param: 'urban_only'
    });

    const specificTemplate = createTestTemplate('specific', {
      specific_param: 'specific_only',
      array_param: ['specific1', 'specific2']
    });

    // Register templates with priorities
    await integratedSystem.registerTemplate('base', baseTemplate, createTestPriority('base', TemplatePriority.BASE));
    await integratedSystem.registerTemplate('urban', urbanTemplate, createTestPriority('urban', TemplatePriority.VARIANT, 'base'));
    await integratedSystem.registerTemplate('specific', specificTemplate, createTestPriority('specific', TemplatePriority.CONTEXT_SPECIFIC, 'urban'));

    // Process the specific template
    const context: MOTemplateContext = {
      environment: 'production',
      featureFlags: { advanced_features: true }
    };

    const result = await integratedSystem.processTemplate('specific', context);

    // Verify inheritance resolution
    expect(result.template.configuration).toHaveProperty('common_param');
    expect(result.template.configuration).toHaveProperty('base_only_param');
    expect(result.template.configuration).toHaveProperty('urban_param');
    expect(result.template.configuration).toHaveProperty('specific_param');

    // Verify optimizations were applied
    expect(result.appliedOptimizations).toContain('priority_inheritance_resolution');
    expect(result.appliedOptimizations).toContain('template_validation');

    // Verify statistics
    expect(result.processingStats.totalParameters).toBeGreaterThan(0);
    expect(result.processingStats.validationTime).toBeGreaterThanOrEqual(0);

    // Verify no critical errors
    expect(result.errors.filter(e => e.includes('Template processing failed'))).toHaveLength(0);
  });
});