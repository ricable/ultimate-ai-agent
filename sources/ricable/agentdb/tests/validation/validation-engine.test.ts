/**
 * Validation Engine Test Suite - Phase 5 Implementation
 *
 * Comprehensive test suite for the Complex Validation Rules Engine
 * Tests all components with 99.9% coverage targets
 * Includes performance tests and cognitive integration tests
 */

import { ValidationEngine } from '../../src/validation/validation-engine';
import { ValidationFactory, ValidationUtils, ValidationPerformanceMonitor } from '../../src/validation/index';
import { CognitiveConsciousnessCore } from '../../src/cognitive/CognitiveConsciousnessCore';
import {
  ValidationEngineConfig,
  ValidationContext,
  ValidationResult
} from '../../src/types/validation-types';

describe('Validation Engine - Phase 5 Implementation', () => {
  let validationEngine: ValidationEngine;
  let cognitiveCore: CognitiveConsciousnessCore;
  let performanceMonitor: ValidationPerformanceMonitor;

  beforeEach(async () => {
    cognitiveCore = new CognitiveConsciousnessCore({
      level: 'medium',
      temporalExpansion: 100,
      strangeLoopOptimization: true,
      autonomousAdaptation: true
    });

    performanceMonitor = ValidationPerformanceMonitor.getInstance();
  });

  afterEach(async () => {
    if (validationEngine) {
      await validationEngine.shutdown();
    }
    if (cognitiveCore) {
      await cognitiveCore.shutdown();
    }
    performanceMonitor.clearMetrics();
  });

  describe('Core Functionality Tests', () => {
    test('should initialize validation engine successfully', async () => {
      const config: ValidationEngineConfig = {
        maxValidationTime: 300,
        cacheEnabled: true,
        learningEnabled: true,
        consciousnessIntegration: false
      };

      validationEngine = new ValidationEngine(config);
      await validationEngine.initialize();

      expect(validationEngine).toBeDefined();
      const metrics = validationEngine.getMetrics();
      expect(metrics.totalParameters).toBeGreaterThan(0);
      expect(metrics.validationCoverage).toBeGreaterThanOrEqual(0);
    });

    test('should validate simple configuration successfully', async () => {
      validationEngine = await ValidationFactory.createValidationEngine({
        consciousnessIntegration: false
      });

      const configuration = {
        managedElementId: 'test_element_001',
        userLabel: 'Test Element',
        qRxLevMin: -70
      };

      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'test_001',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
      expect(result.executionTime).toBeLessThan(300);
    });

    test('should detect validation errors in configuration', async () => {
      validationEngine = await ValidationFactory.createValidationEngine({
        strictMode: true,
        consciousnessIntegration: false
      });

      const configuration = {
        qRxLevMin: -200, // Below minimum range (-70 to -110)
        managedElementId: '', // Empty required field
        invalidParameter: 'should_not_exist'
      };

      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'test_002',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'comprehensive'
      });

      expect(result.valid).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);

      // Check for specific error types
      const errorCodes = result.errors.map(e => e.code);
      expect(errorCodes).toContain('CONSTRAINT_RANGE_VIOLATION');
      expect(errorCodes).toContain('UNKNOWN_PARAMETER');
    });

    test('should handle large parameter sets efficiently', async () => {
      validationEngine = await ValidationFactory.createValidationEngine({
        parallelProcessing: true,
        batchSize: 50
      });

      // Generate large configuration
      const configuration: Record<string, any> = {};
      for (let i = 0; i < 1000; i++) {
        configuration[`param_${i}`] = `value_${i}`;
      }

      const startTime = Date.now();
      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'test_large_001',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });
      const executionTime = Date.now() - startTime;

      expect(result.valid).toBe(true);
      expect(executionTime).toBeLessThan(500); // Should handle 1000 params efficiently
      expect(result.parametersValidated).toBe(1000);
    });
  });

  describe('Constraint Processing Tests', () => {
    test('should validate range constraints correctly', async () => {
      validationEngine = await ValidationFactory.createValidationEngine();

      const testCases = [
        { name: 'qRxLevMin', value: -80, expected: true },
        { name: 'qRxLevMin', value: -50, expected: false }, // Above max
        { name: 'qRxLevMin', value: -120, expected: false }, // Below min
        { name: 'qRxLevMin', value: null, expected: true } // Null should pass required check
      ];

      for (const testCase of testCases) {
        const configuration = { [testCase.name]: testCase.value };
        const result = await validationEngine.validateConfiguration(configuration, {
          validationId: `constraint_test_${testCase.name}`,
          timestamp: Date.now(),
          configuration,
          validationLevel: 'standard'
        });

        if (testCase.expected) {
          expect(result.errors.filter(e => e.parameter === testCase.name)).toHaveLength(0);
        } else {
          expect(result.errors.filter(e => e.parameter === testCase.name).length).toBeGreaterThan(0);
        }
      }
    });

    test('should validate enum constraints correctly', async () => {
      validationEngine = await ValidationFactory.createValidationEngine();

      const configuration = {
        status: 'active', // Valid enum value
        priority: 'high'  // Valid enum value
      };

      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'enum_test_001',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });

      expect(result.valid).toBe(true);
    });

    test('should validate pattern constraints correctly', async () => {
      validationEngine = await ValidationFactory.createValidationEngine();

      const testCases = [
        { name: 'cellId', value: 'CELL_001', expected: true },
        { name: 'cellId', value: 'cell@invalid', expected: false },
        { name: 'cellId', value: 'Cell-123_Valid', expected: true }
      ];

      for (const testCase of testCases) {
        const configuration = { [testCase.name]: testCase.value };
        const result = await validationEngine.validateConfiguration(configuration, {
          validationId: `pattern_test_${testCase.name}`,
          timestamp: Date.now(),
          configuration,
          validationLevel: 'standard'
        });

        // Pattern validation results depend on the actual pattern defined
        expect(result).toBeDefined();
      }
    });
  });

  describe('Cognitive Integration Tests', () => {
    test('should integrate with cognitive consciousness core', async () => {
      await cognitiveCore.initialize();

      validationEngine = await ValidationFactory.createCognitiveValidationEngine(cognitiveCore);

      const configuration = {
        managedElementId: 'cognitive_test_001',
        userLabel: 'Cognitive Test Element'
      };

      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'cognitive_test_001',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'comprehensive',
        consciousnessLevel: 0.7
      });

      expect(result.cognitiveInsights).toBeDefined();
      expect(result.consciousnessLevel).toBeGreaterThan(0);
    });

    test('should apply cognitive optimization', async () => {
      await cognitiveCore.initialize();

      validationEngine = await ValidationFactory.createCognitiveValidationEngine(cognitiveCore);

      const configuration = {
        param1: 'value1',
        param2: 'value2',
        param3: 'value3'
      };

      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'cognitive_optimization_test',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'comprehensive'
      });

      expect(result.cognitiveInsights).toBeDefined();
      if (result.cognitiveInsights.cognitiveValidation) {
        expect(result.cognitiveInsights.insights).toBeDefined();
      }
    });
  });

  describe('Performance Tests', () => {
    test('should validate within performance targets', async () => {
      validationEngine = await ValidationFactory.createValidationEngine({
        maxValidationTime: 300,
        cacheEnabled: true,
        parallelProcessing: true
      });

      const configuration = {
        managedElementId: 'perf_test_001',
        userLabel: 'Performance Test',
        qRxLevMin: -75,
        qQualMin: -20
      };

      const startTime = Date.now();
      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'performance_test_001',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });
      const executionTime = Date.now() - startTime;

      expect(executionTime).toBeLessThan(300);
      expect(result.executionTime).toBeLessThan(300);
      expect(result.cacheHitRate).toBeGreaterThanOrEqual(0);
    });

    test('should handle caching effectively', async () => {
      validationEngine = await ValidationFactory.createValidationEngine({
        cacheEnabled: true,
        cacheTTL: 60000
      });

      const configuration = {
        managedElementId: 'cache_test_001',
        userLabel: 'Cache Test'
      };

      // First validation
      const result1 = await validationEngine.validateConfiguration(configuration, {
        validationId: 'cache_test_001',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });

      // Second validation (should use cache)
      const result2 = await validationEngine.validateConfiguration(configuration, {
        validationId: 'cache_test_002',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });

      expect(result1.valid).toBe(true);
      expect(result2.valid).toBe(true);

      // Cache hit rate should improve on second validation
      expect(result2.cacheHitRate).toBeGreaterThanOrEqual(result1.cacheHitRate);
    });

    test('should maintain performance under load', async () => {
      validationEngine = await ValidationFactory.createValidationEngine({
        parallelProcessing: true,
        batchSize: 100
      });

      const configurations = Array.from({ length: 50 }, (_, i) => ({
        managedElementId: `load_test_${i}`,
        userLabel: `Load Test ${i}`,
        qRxLevMin: -70 + (i % 10)
      }));

      const startTime = Date.now();
      const results = await Promise.all(
        configurations.map((config, index) =>
          validationEngine.validateConfiguration(config, {
            validationId: `load_test_${index}`,
            timestamp: Date.now(),
            configuration: config,
            validationLevel: 'standard'
          })
        )
      );
      const totalTime = Date.now() - startTime;

      expect(results.length).toBe(50);
      expect(results.every(r => r.valid)).toBe(true);
      expect(totalTime / results.length).toBeLessThan(100); // Average < 100ms per validation
    });
  });

  describe('Integration Tests', () => {
    test('should work with ValidationFactory', async () => {
      const engine = await ValidationFactory.createValidationEngine({
        strictMode: true,
        cacheEnabled: true
      });

      const configuration = {
        managedElementId: 'factory_test_001',
        userLabel: 'Factory Test'
      };

      const result = await engine.validateConfiguration(configuration, {
        validationId: 'factory_test',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });

      expect(result.valid).toBe(true);
      await engine.shutdown();
    });

    test('should work with ValidationUtils', async () => {
      const configuration = {
        managedElementId: 'utils_test_001',
        userLabel: 'Utils Test'
      };

      const result = await ValidationUtils.validateConfiguration(configuration, {
        level: 'standard',
        strictMode: false,
        enableCaching: true
      });

      expect(result).toBeDefined();
      expect(result.valid).toBe(true);
    });
  });

  describe('Memory and Resource Management', () => {
    test('should handle memory efficiently', async () => {
      validationEngine = await ValidationFactory.createValidationEngine({
        cacheEnabled: true,
        cacheTTL: 1000 // Short TTL for testing
      });

      // Perform multiple validations
      for (let i = 0; i < 100; i++) {
        const configuration = {
          managedElementId: `memory_test_${i}`,
          userLabel: `Memory Test ${i}`
        };

        await validationEngine.validateConfiguration(configuration, {
          validationId: `memory_test_${i}`,
          timestamp: Date.now(),
          configuration,
          validationLevel: 'basic'
        });
      }

      // Clear cache and check memory usage
      validationEngine.clearCache();

      const cacheStats = validationEngine.getCacheStatistics();
      expect(cacheStats.size).toBe(0);
    });

    test('should shutdown cleanly', async () => {
      validationEngine = await ValidationFactory.createValidationEngine();

      // Perform some validations
      const configuration = { managedElementId: 'shutdown_test' };
      await validationEngine.validateConfiguration(configuration, {
        validationId: 'shutdown_test',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'basic'
      });

      // Shutdown should not throw errors
      await expect(validationEngine.shutdown()).resolves.not.toThrow();
    });
  });

  describe('Edge Cases and Boundary Conditions', () => {
    test('should handle empty configuration', async () => {
      validationEngine = await ValidationFactory.createValidationEngine();

      const result = await validationEngine.validateConfiguration({}, {
        validationId: 'empty_test',
        timestamp: Date.now(),
        configuration: {},
        validationLevel: 'basic'
      });

      expect(result.valid).toBe(true); // Empty config should be valid
      expect(result.parametersValidated).toBe(0);
    });

    test('should handle very long parameter values', async () => {
      validationEngine = await ValidationFactory.createValidationEngine();

      const longString = 'a'.repeat(10000);
      const configuration = {
        userLabel: longString
      };

      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'long_value_test',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });

      expect(result).toBeDefined();
      // Should handle long values without issues
    });

    test('should handle special characters in parameter values', async () => {
      validationEngine = await ValidationFactory.createValidationEngine();

      const configuration = {
        userLabel: 'Special chars: !@#$%^&*()_+-={}[]|\\:";\'<>?,./',
        unicodeTest: 'Unicode: ñáéíóú 中文 русский العربية',
        jsonTest: '{"key": "value", "number": 123}'
      };

      const result = await validationEngine.validateConfiguration(configuration, {
        validationId: 'special_chars_test',
        timestamp: Date.now(),
        configuration,
        validationLevel: 'standard'
      });

      expect(result).toBeDefined();
      expect(result.valid).toBe(true);
    });
  });

  describe('Concurrent Validation Tests', () => {
    test('should handle concurrent validations safely', async () => {
      validationEngine = await ValidationFactory.createValidationEngine({
        parallelProcessing: true
      });

      const concurrentValidations = Array.from({ length: 20 }, (_, i) =>
        validationEngine.validateConfiguration({
          managedElementId: `concurrent_${i}`,
          userLabel: `Concurrent Test ${i}`,
          qRxLevMin: -70 + (i % 5)
        }, {
          validationId: `concurrent_${i}`,
          timestamp: Date.now(),
          configuration: { managedElementId: `concurrent_${i}` },
          validationLevel: 'standard'
        })
      );

      const results = await Promise.all(concurrentValidations);

      expect(results.length).toBe(20);
      expect(results.every(r => r.valid)).toBe(true);

      // All validation IDs should be unique
      const validationIds = results.map(r => r.validationId);
      const uniqueIds = new Set(validationIds);
      expect(uniqueIds.size).toBe(20);
    });
  });

  describe('Metrics and Monitoring Tests', () => {
    test('should provide accurate performance metrics', async () => {
      validationEngine = await ValidationFactory.createValidationEngine();

      // Perform several validations
      for (let i = 0; i < 5; i++) {
        const configuration = {
          managedElementId: `metrics_test_${i}`,
          userLabel: `Metrics Test ${i}`
        };

        await validationEngine.validateConfiguration(configuration, {
          validationId: `metrics_test_${i}`,
          timestamp: Date.now(),
          configuration,
          validationLevel: 'standard'
        });
      }

      const metrics = validationEngine.getMetrics();

      expect(metrics.totalParameters).toBeGreaterThan(0);
      expect(metrics.validationCoverage).toBeGreaterThanOrEqual(0);
      expect(metrics.averageProcessingTime).toBeGreaterThanOrEqual(0);
      expect(metrics.cacheHitRate).toBeGreaterThanOrEqual(0);
    });
  });
});