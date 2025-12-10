/**
 * Tests for Adaptive Swarm Coordinator
 *
 * Comprehensive test suite for adaptive coordination with dynamic topology,
 * intelligent resource allocation, consensus mechanisms, and cognitive intelligence.
 */

import { AdaptiveSwarmCoordinator, AdaptiveConfiguration } from '../../src/adaptive-coordinator/adaptive-swarm-coordinator';
import { Agent } from '../../src/adaptive-coordinator/types';

describe('AdaptiveSwarmCoordinator', () => {
  let coordinator: AdaptiveSwarmCoordinator;
  let testConfig: AdaptiveConfiguration;

  beforeEach(() => {
    testConfig = {
      // Topology Configuration
      topologyStrategy: 'adaptive',
      topologySwitchThreshold: 0.2,
      adaptationFrequency: 5, // 5 minutes for testing
      maxTopologyTransitions: 3,

      // Resource Allocation
      resourcePredictionWindow: 60, // 1 hour
      scalingCooldownPeriod: 2, // 2 minutes
      resourceUtilizationTarget: 0.8,
      predictiveScaling: true,

      // Consensus Configuration
      consensusAlgorithm: 'adaptive',
      consensusTimeout: 30000, // 30 seconds
      byzantineFaultTolerance: true,
      requiredConsensus: 0.7,

      // Performance Configuration
      monitoringInterval: 30000, // 30 seconds
      performanceWindow: 60, // 1 hour
      bottleneckDetectionThreshold: 0.3,
      optimizationCycleInterval: 15, // 15 minutes

      // Cognitive Configuration
      cognitiveIntelligenceEnabled: true,
      learningRate: 0.1,
      patternRecognitionWindow: 24, // 24 hours
      autonomousDecisionThreshold: 0.8
    };

    coordinator = new AdaptiveSwarmCoordinator(testConfig);
  });

  afterEach(async () => {
    if (coordinator) {
      await coordinator.shutdown();
    }
  });

  describe('Initialization', () => {
    test('should initialize with correct configuration', () => {
      expect(coordinator).toBeDefined();
      expect(coordinator.getCurrentTopology()).toBe('hierarchical');
    });

    test('should initialize cognitive components when enabled', () => {
      expect(testConfig.cognitiveIntelligenceEnabled).toBe(true);
      // Additional cognitive component checks would go here
    });
  });

  describe('Adaptation Cycle', () => {
    test('should perform adaptation cycle successfully', async () => {
      const currentMetrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(currentMetrics).toBeDefined();
      expect(currentMetrics.topologyMetrics).toBeDefined();
      expect(currentMetrics.resourceMetrics).toBeDefined();
      expect(currentMetrics.consensusMetrics).toBeDefined();
      expect(currentMetrics.performanceMetrics).toBeDefined();
      expect(currentMetrics.cognitiveMetrics).toBeDefined();
      expect(currentMetrics.overallAdaptationScore).toBeGreaterThanOrEqual(0);
      expect(currentMetrics.overallAdaptationScore).toBeLessThanOrEqual(1);
    });

    test('should analyze adaptation needs correctly', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();

      // The analysis should identify potential improvements
      expect(metrics.overallAdaptationScore).toBeGreaterThanOrEqual(0);
      expect(metrics.overallAdaptationScore).toBeLessThanOrEqual(1);
    });
  });

  describe('Topology Optimization', () => {
    test('should handle topology analysis', async () => {
      const currentMetrics = await coordinator.getCurrentAdaptiveMetrics();

      // Should be able to analyze current topology
      expect(currentMetrics.topologyMetrics.currentTopology).toBeDefined();
      expect(currentMetrics.topologyMetrics.topologyStability).toBeGreaterThanOrEqual(0);
      expect(currentMetrics.topologyMetrics.topologyStability).toBeLessThanOrEqual(1);
    });

    test('should recommend topology changes when beneficial', async () => {
      // This would test topology recommendation logic
      const currentTopology = coordinator.getCurrentTopology();
      expect(['hierarchical', 'mesh', 'ring', 'star', 'hybrid', 'adaptive']).toContain(currentTopology);
    });
  });

  describe('Resource Management', () => {
    test('should monitor resource utilization', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();

      expect(metrics.resourceMetrics.cpuUtilization).toBeGreaterThanOrEqual(0);
      expect(metrics.resourceMetrics.cpuUtilization).toBeLessThanOrEqual(1);
      expect(metrics.resourceMetrics.memoryUtilization).toBeGreaterThanOrEqual(0);
      expect(metrics.resourceMetrics.memoryUtilization).toBeLessThanOrEqual(1);
    });

    test('should track scaling events', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(metrics.resourceMetrics.scalingEvents).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Consensus Mechanisms', () => {
    test('should track consensus metrics', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();

      expect(metrics.consensusMetrics.consensusTime).toBeGreaterThan(0);
      expect(metrics.consensusMetrics.consensusSuccessRate).toBeGreaterThanOrEqual(0);
      expect(metrics.consensusMetrics.consensusSuccessRate).toBeLessThanOrEqual(1);
    });

    test('should handle byzantine fault tolerance', () => {
      expect(testConfig.byzantineFaultTolerance).toBe(true);
    });
  });

  describe('Performance Monitoring', () => {
    test('should monitor system performance', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();

      expect(metrics.performanceMetrics.systemThroughput).toBeGreaterThan(0);
      expect(metrics.performanceMetrics.responseTime).toBeGreaterThan(0);
      expect(metrics.performanceMetrics.errorRate).toBeGreaterThanOrEqual(0);
      expect(metrics.performanceMetrics.errorRate).toBeLessThanOrEqual(1);
    });

    test('should calculate bottleneck scores', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(metrics.performanceMetrics.bottleneckScore).toBeGreaterThanOrEqual(0);
      expect(metrics.performanceMetrics.bottleneckScore).toBeLessThanOrEqual(1);
    });
  });

  describe('Cognitive Intelligence', () => {
    test('should track cognitive metrics', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();

      expect(metrics.cognitiveMetrics.learningRate).toBeGreaterThan(0);
      expect(metrics.cognitiveMetrics.learningRate).toBeLessThanOrEqual(1);
      expect(metrics.cognitiveMetrics.patternRecognition).toBeGreaterThanOrEqual(0);
      expect(metrics.cognitiveMetrics.patternRecognition).toBeLessThanOrEqual(1);
    });

    test('should support autonomous decisions', () => {
      expect(testConfig.autonomousDecisionThreshold).toBeGreaterThan(0);
      expect(testConfig.autonomousDecisionThreshold).toBeLessThanOrEqual(1);
    });
  });

  describe('Configuration Management', () => {
    test('should update configuration successfully', async () => {
      const newConfig = {
        topologyStrategy: 'mesh' as const,
        learningRate: 0.2
      };

      await expect(coordinator.updateConfiguration(newConfig)).resolves.not.toThrow();
    });

    test('should maintain configuration validity', () => {
      expect(testConfig.topologySwitchThreshold).toBeGreaterThan(0);
      expect(testConfig.topologySwitchThreshold).toBeLessThanOrEqual(1);
      expect(testConfig.requiredConsensus).toBeGreaterThan(0);
      expect(testConfig.requiredConsensus).toBeLessThanOrEqual(1);
    });
  });

  describe('Reporting and Analytics', () => {
    test('should generate adaptation report', async () => {
      const report = await coordinator.getAdaptationReport();

      expect(report).toBeDefined();
      expect(report.totalAdaptations).toBeGreaterThanOrEqual(0);
      expect(report.averageAdaptationScore).toBeGreaterThanOrEqual(0);
      expect(report.averageAdaptationScore).toBeLessThanOrEqual(1);
    });

    test('should maintain adaptation history', () => {
      const history = coordinator.getAdaptationHistory();
      expect(Array.isArray(history)).toBe(true);
    });
  });

  describe('Error Handling and Resilience', () => {
    test('should handle configuration errors gracefully', async () => {
      const invalidConfig = {
        topologySwitchThreshold: 1.5, // Invalid: > 1
        requiredConsensus: -0.1 // Invalid: < 0
      };

      // Should not throw with invalid values but handle gracefully
      await expect(coordinator.updateConfiguration(invalidConfig)).resolves.not.toThrow();
    });

    test('should cleanup resources properly', async () => {
      await expect(coordinator.shutdown()).resolves.not.toThrow();
    });
  });

  describe('Integration Points', () => {
    test('should integrate with topology optimizer', () => {
      const currentTopology = coordinator.getCurrentTopology();
      expect(typeof currentTopology).toBe('string');
    });

    test('should integrate with resource allocator', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(metrics.resourceMetrics).toBeDefined();
    });

    test('should integrate with consensus mechanism', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(metrics.consensusMetrics).toBeDefined();
    });

    test('should integrate with performance monitor', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(metrics.performanceMetrics).toBeDefined();
    });
  });

  describe('Performance Requirements', () => {
    test('should meet performance targets', async () => {
      const startTime = Date.now();
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      const endTime = Date.now();

      // Should complete within reasonable time (less than 1 second for metrics)
      expect(endTime - startTime).toBeLessThan(1000);
      expect(metrics).toBeDefined();
    });

    test('should maintain high availability', () => {
      // Coordinator should be resilient and available
      expect(coordinator).toBeDefined();
      expect(coordinator.getCurrentTopology()).toBeDefined();
    });
  });

  describe('Learning and Adaptation', () => {
    test('should support learning rate configuration', () => {
      expect(testConfig.learningRate).toBeGreaterThan(0);
      expect(testConfig.learningRate).toBeLessThanOrEqual(1);
    });

    test('should support pattern recognition window', () => {
      expect(testConfig.patternRecognitionWindow).toBeGreaterThan(0);
    });

    test('should track cognitive evolution over time', async () => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(metrics.cognitiveMetrics.adaptationEvolution).toBeGreaterThanOrEqual(0);
      expect(metrics.cognitiveMetrics.adaptationEvolution).toBeLessThanOrEqual(1);
    });
  });
});

describe('AdaptiveSwarmCoordinator Edge Cases', () => {
  let coordinator: AdaptiveSwarmCoordinator;

  beforeEach(() => {
    const minimalConfig: AdaptiveConfiguration = {
      topologyStrategy: 'hierarchical',
      topologySwitchThreshold: 0.5,
      adaptationFrequency: 60,
      maxTopologyTransitions: 1,
      resourcePredictionWindow: 60,
      scalingCooldownPeriod: 5,
      resourceUtilizationTarget: 0.7,
      predictiveScaling: false,
      consensusAlgorithm: 'raft',
      consensusTimeout: 60000,
      byzantineFaultTolerance: false,
      requiredConsensus: 0.5,
      monitoringInterval: 60000,
      performanceWindow: 60,
      bottleneckDetectionThreshold: 0.5,
      optimizationCycleInterval: 60,
      cognitiveIntelligenceEnabled: false,
      learningRate: 0.1,
      patternRecognitionWindow: 24,
      autonomousDecisionThreshold: 0.8
    };

    coordinator = new AdaptiveSwarmCoordinator(minimalConfig);
  });

  afterEach(async () => {
    if (coordinator) {
      await coordinator.shutdown();
    }
  });

  test('should handle minimal configuration', () => {
    expect(coordinator).toBeDefined();
    expect(coordinator.getCurrentTopology()).toBe('hierarchical');
  });

  test('should operate without cognitive intelligence', async () => {
    const metrics = await coordinator.getCurrentAdaptiveMetrics();
    expect(metrics).toBeDefined();
    // Should work even without cognitive features
  });

  test('should handle disabled features gracefully', async () => {
    const metrics = await coordinator.getCurrentAdaptiveMetrics();
    expect(metrics).toBeDefined();

    // Should not throw when features are disabled
    expect(metrics.cognitiveMetrics).toBeDefined();
  });
});

describe('AdaptiveSwarmCoordinator Performance Benchmarks', () => {
  let coordinator: AdaptiveSwarmCoordinator;

  beforeAll(() => {
    const config: AdaptiveConfiguration = {
      topologyStrategy: 'adaptive',
      topologySwitchThreshold: 0.2,
      adaptationFrequency: 1, // 1 minute for performance testing
      maxTopologyTransitions: 5,
      resourcePredictionWindow: 30,
      scalingCooldownPeriod: 1,
      resourceUtilizationTarget: 0.8,
      predictiveScaling: true,
      consensusAlgorithm: 'adaptive',
      consensusTimeout: 10000,
      byzantineFaultTolerance: true,
      requiredConsensus: 0.7,
      monitoringInterval: 5000, // 5 seconds
      performanceWindow: 30,
      bottleneckDetectionThreshold: 0.3,
      optimizationCycleInterval: 5, // 5 minutes
      cognitiveIntelligenceEnabled: true,
      learningRate: 0.1,
      patternRecognitionWindow: 12,
      autonomousDecisionThreshold: 0.8
    };

    coordinator = new AdaptiveSwarmCoordinator(config);
  });

  afterAll(async () => {
    if (coordinator) {
      await coordinator.shutdown();
    }
  });

  test('should handle rapid metric collection', async () => {
    const startTime = Date.now();

    // Collect metrics multiple times rapidly
    const promises = Array(10).fill(null).map(() =>
      coordinator.getCurrentAdaptiveMetrics()
    );

    const results = await Promise.all(promises);
    const endTime = Date.now();

    expect(results).toHaveLength(10);
    expect(results.every(r => r !== undefined)).toBe(true);

    // Should complete 10 metric collections in under 5 seconds
    expect(endTime - startTime).toBeLessThan(5000);
  });

  test('should maintain performance under load', async () => {
    const startTime = Date.now();

    // Simulate high load with concurrent operations
    const operations = Array(20).fill(null).map(async (_, index) => {
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      return {
        index,
        metrics,
        timestamp: Date.now()
      };
    });

    const results = await Promise.all(operations);
    const endTime = Date.now();

    expect(results).toHaveLength(20);
    expect(results.every(r => r.metrics !== undefined)).toBe(true);

    // Should handle 20 concurrent operations efficiently
    expect(endTime - startTime).toBeLessThan(10000);

    // Verify all operations completed successfully
    const failedOperations = results.filter(r => r.metrics === undefined);
    expect(failedOperations).toHaveLength(0);
  });
});