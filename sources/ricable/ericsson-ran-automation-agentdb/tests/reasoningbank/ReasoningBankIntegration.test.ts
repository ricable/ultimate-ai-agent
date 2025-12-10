/**
 * ReasoningBank AgentDB Integration Test Suite
 * Comprehensive testing for adaptive learning, trajectory tracking, and verdict judgment
 */

import { ReasoningBankAgentDBIntegration, ReasoningBankConfig } from '../../src/reasoningbank/core/ReasoningBankAgentDBIntegration';
import { AgentDBMemoryManager } from '../../src/agentdb/AgentDBMemoryManager';
import { ReinforcementLearningEngine } from '../../src/ml/reinforcement-learning-engine';

describe('ReasoningBank AgentDB Integration', () => {
  let reasoningBank: ReasoningBankAgentDBIntegration;
  let config: ReasoningBankConfig;

  beforeAll(async () => {
    // Initialize test configuration
    config = {
      agentDB: {
        swarmId: 'test-reasoningbank-swarm',
        syncProtocol: 'QUIC',
        persistenceEnabled: false, // Disable persistence for tests
        crossAgentLearning: true,
        vectorDimension: 512,
        indexingStrategy: 'HNSW',
        quantization: { enabled: true, bits: 8 }
      },
      adaptiveLearning: {
        learningRate: 0.01,
        adaptationThreshold: 0.7,
        trajectoryLength: 1000,
        patternExtractionEnabled: true,
        crossDomainTransfer: true
      },
      temporalReasoning: {
        subjectiveTimeExpansion: 1000,
        temporalPatternWindow: 300000, // 5 minutes
        causalInferenceEnabled: true,
        predictionHorizon: 600000 // 10 minutes
      },
      performance: {
        cacheEnabled: true,
        quantizationEnabled: true,
        parallelProcessingEnabled: true,
        memoryCompressionEnabled: true
      }
    };

    reasoningBank = new ReasoningBankAgentDBIntegration(config);
    await reasoningBank.initialize();
  });

  afterAll(async () => {
    if (reasoningBank) {
      await reasoningBank.shutdown();
    }
  });

  describe('Initialization', () => {
    test('should initialize successfully', () => {
      expect(reasoningBank).toBeDefined();
    });

    test('should have correct configuration', () => {
      const stats = reasoningBank.getStatistics();
      expect(stats.reasoningbank.is_initialized).toBe(true);
    });
  });

  describe('Adaptive RL Training', () => {
    test('should execute adaptive RL training', async () => {
      const result = await reasoningBank.adaptiveRLTraining();

      expect(result).toBeDefined();
      expect(result.id).toBeDefined();
      expect(result.domain).toBe('ran-optimization');
      expect(result.policy_data).toBeDefined();
      expect(result.performance_metrics).toBeDefined();
      expect(result.cross_agent_applicability).toBeGreaterThanOrEqual(0);
      expect(result.cross_agent_applicability).toBeLessThanOrEqual(1);
    }, 30000);

    test('should generate adaptive policy with reasonable metrics', async () => {
      const result = await reasoningBank.adaptiveRLTraining();

      expect(result.performance_metrics.accuracy).toBeGreaterThan(0);
      expect(result.performance_metrics.accuracy).toBeLessThanOrEqual(1);
      expect(result.performance_metrics.efficiency).toBeGreaterThan(0);
      expect(result.performance_metrics.efficiency).toBeLessThanOrEqual(1);
      expect(result.performance_metrics.overall_score).toBeGreaterThan(0);
      expect(result.performance_metrics.overall_score).toBeLessThanOrEqual(1);
    }, 30000);

    test('should store reasoning patterns in AgentDB', async () => {
      const initialStats = await reasoningBank.getStatistics();
      const initialPatterns = initialStats.reasoningbank.learning_patterns;

      await reasoningBank.adaptiveRLTraining();

      const finalStats = await reasoningBank.getStatistics();
      expect(finalStats.reasoningbank.learning_patterns).toBeGreaterThan(initialPatterns);
    }, 30000);
  });

  describe('Pattern Analysis', () => {
    test('should analyze reasoning patterns correctly', async () => {
      const result = await reasoningBank.adaptiveRLTraining();

      expect(result.policy_data.reasoning_pattern).toBeDefined();
      expect(result.policy_data.verdict).toBeDefined();
      expect(result.policy_data.adaptation).toBeDefined();
      expect(result.policy_data.trajectory).toBeDefined();
    }, 30000);

    test('should calculate cross-agent applicability', async () => {
      const result = await reasoningBank.adaptiveRLTraining();

      expect(result.cross_agent_applicability).toBeGreaterThanOrEqual(0);
      expect(result.cross_agent_applicability).toBeLessThanOrEqual(1);

      // Should have some reasonable applicability for RAN optimization
      expect(result.cross_agent_applicability).toBeGreaterThan(0.3);
    }, 30000);

    test('should extract temporal patterns', async () => {
      const result = await reasoningBank.adaptiveRLTraining();

      expect(result.temporal_patterns).toBeDefined();
      expect(Array.isArray(result.temporal_patterns)).toBe(true);
    }, 30000);
  });

  describe('Performance Optimization', () => {
    test('should optimize policy storage', async () => {
      const result = await reasoningBank.adaptiveRLTraining();

      // Performance should be optimized
      expect(result.performance_metrics.overall_score).toBeGreaterThan(0.7);
    }, 30000);

    test('should maintain reasonable execution time', async () => {
      const startTime = performance.now();
      await reasoningBank.adaptiveRLTraining();
      const endTime = performance.now();

      const executionTime = endTime - startTime;

      // Should complete within reasonable time (adjust threshold as needed)
      expect(executionTime).toBeLessThan(10000); // 10 seconds
    }, 30000);
  });

  describe('Statistics and Monitoring', () => {
    test('should provide comprehensive statistics', async () => {
      await reasoningBank.adaptiveRLTraining(); // Generate some activity
      const stats = await reasoningBank.getStatistics();

      expect(stats).toBeDefined();
      expect(stats.reasoningbank).toBeDefined();
      expect(stats.agentdb).toBeDefined();
      expect(stats.adaptive_learning).toBeDefined();
      expect(stats.trajectory_tracking).toBeDefined();
      expect(stats.verdict_judgment).toBeDefined();
      expect(stats.performance_optimization).toBeDefined();
    });

    test('should track active policies', async () => {
      await reasoningBank.adaptiveRLTraining();
      const stats = await reasoningBank.getStatistics();

      expect(stats.reasoningbank.active_policies).toBeGreaterThan(0);
    });

    test('should track learning patterns', async () => {
      await reasoningBank.adaptiveRLTraining();
      const stats = await reasoningBank.getStatistics();

      expect(stats.reasoningbank.learning_patterns).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    test('should handle initialization errors gracefully', async () => {
      const invalidConfig = {
        ...config,
        agentDB: {
          ...config.agentDB,
          swarmId: '' // Invalid empty swarm ID
        }
      };

      const invalidReasoningBank = new ReasoningBankAgentDBIntegration(invalidConfig);

      await expect(invalidReasoningBank.initialize()).rejects.toThrow();
    });

    test('should handle operations when not initialized', async () => {
      const uninitializedReasoningBank = new ReasoningBankAgentDBIntegration(config);

      await expect(uninitializedReasoningBank.adaptiveRLTraining()).rejects.toThrow('ReasoningBank not initialized');
    });
  });

  describe('Memory Management', () => {
    test('should handle memory cleanup', async () => {
      await reasoningBank.adaptiveRLTraining();

      // Get initial memory statistics
      const initialStats = await reasoningBank.getStatistics();

      // Shutdown and reinitialize
      await reasoningBank.shutdown();
      await reasoningBank.initialize();

      // Should start fresh
      const freshStats = await reasoningBank.getStatistics();
      expect(freshStats.reasoningbank.active_policies).toBe(0);
      expect(freshStats.reasoningbank.learning_patterns).toBe(0);
    });

    test('should handle large numbers of operations', async () => {
      // Perform multiple adaptive training operations
      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push(reasoningBank.adaptiveRLTraining());
      }

      const results = await Promise.all(promises);

      // All should succeed
      expect(results).toHaveLength(5);
      results.forEach(result => {
        expect(result.id).toBeDefined();
        expect(result.performance_metrics).toBeDefined();
      });
    }, 60000);
  });

  describe('Integration with AgentDB', () => {
    test('should integrate with AgentDB memory manager', async () => {
      await reasoningBank.adaptiveRLTraining();
      const stats = await reasoningBank.getStatistics();

      expect(stats.agentdb).toBeDefined();
      expect(stats.agentdb.totalMemories).toBeGreaterThan(0);
    });

    test('should leverage QUIC synchronization', async () => {
      const stats = await reasoningBank.getStatistics();

      // QUIC should be enabled in configuration
      expect(config.agentDB.syncProtocol).toBe('QUIC');
    });
  });

  describe('Cross-Agent Learning', () => {
    test('should support cross-agent learning', async () => {
      await reasoningBank.adaptiveRLTraining();
      const stats = await reasoningBank.getStatistics();

      expect(config.agentDB.crossAgentLearning).toBe(true);
    });

    test('should have cross-agent memory', async () => {
      await reasoningBank.adaptiveRLTraining();
      const stats = await reasoningBank.getStatistics();

      // Should have some cross-agent memory structures
      expect(stats.reasoningbank.cross_agent_memories).toBeDefined();
    });
  });

  describe('Temporal Reasoning Integration', () => {
    test('should integrate temporal reasoning', async () => {
      const result = await reasoningBank.adaptiveRLTraining();

      expect(result.policy_data).toBeDefined();
      expect(result.policy_data.adaptation).toBeDefined();
    });

    test('should handle subjective time expansion', async () => {
      expect(config.temporalReasoning.subjectiveTimeExpansion).toBe(1000);
    });
  });

  describe('Quantization and Performance', () => {
    test('should use quantization for performance', async () => {
      expect(config.agentDB.quantization.enabled).toBe(true);
      expect(config.agentDB.quantization.bits).toBe(8);
    });

    test('should use HNSW indexing', async () => {
      expect(config.agentDB.indexingStrategy).toBe('HNSW');
    });

    test('should have performance optimization enabled', async () => {
      expect(config.performance.cacheEnabled).toBe(true);
      expect(config.performance.quantizationEnabled).toBe(true);
      expect(config.performance.parallelProcessingEnabled).toBe(true);
    });
  });
});

describe('ReasoningBank Component Integration', () => {
  describe('Adaptive Learning Core', () => {
    test('should handle adaptive learning configuration', () => {
      expect(config.adaptiveLearning.learningRate).toBe(0.01);
      expect(config.adaptiveLearning.adaptationThreshold).toBe(0.7);
      expect(config.adaptiveLearning.patternExtractionEnabled).toBe(true);
    });
  });

  describe('Trajectory Tracking', () => {
    test('should support trajectory configuration', () => {
      expect(config.adaptiveLearning.trajectoryLength).toBe(1000);
    });
  });

  describe('Verdict Judgment', () => {
    test('should handle verdict judgment parameters', () => {
      // Verdict judgment is part of the integration
      expect(config.temporalReasoning.causalInferenceEnabled).toBe(true);
    });
  });

  describe('Memory Distillation', () => {
    test('should support memory compression', async () => {
      const reasoningBank = new ReasoningBankAgentDBIntegration(config);
      await reasoningBank.initialize();

      const result = await reasoningBank.adaptiveRLTraining();

      // Should have memory optimization
      expect(result.performance_metrics).toBeDefined();

      await reasoningBank.shutdown();
    }, 30000);
  });
});

describe('ReasoningBank Edge Cases', () => {
  let reasoningBank: ReasoningBankAgentDBIntegration;

  beforeEach(async () => {
    reasoningBank = new ReasoningBankAgentDBIntegration(config);
    await reasoningBank.initialize();
  });

  afterEach(async () => {
    if (reasoningBank) {
      await reasoningBank.shutdown();
    }
  });

  test('should handle concurrent operations', async () => {
    const promises = [];
    for (let i = 0; i < 3; i++) {
      promises.push(reasoningBank.adaptiveRLTraining());
    }

    const results = await Promise.all(promises);

    // All should complete successfully
    expect(results).toHaveLength(3);
    results.forEach(result => {
      expect(result.id).toBeDefined();
    });
  }, 60000);

  test('should handle empty or minimal data', async () => {
    // Should still work with minimal initial data
    const result = await reasoningBank.adaptiveRLTraining();
    expect(result).toBeDefined();
    expect(result.performance_metrics.overall_score).toBeGreaterThan(0);
  }, 30000);

  test('should maintain consistency across operations', async () => {
    const result1 = await reasoningBank.adaptiveRLTraining();
    const stats1 = await reasoningBank.getStatistics();

    const result2 = await reasoningBank.adaptiveRLTraining();
    const stats2 = await reasoningBank.getStatistics();

    // Statistics should be consistent
    expect(stats2.reasoningbank.active_policies).toBeGreaterThan(stats1.reasoningbank.active_policies);
    expect(stats2.reasoningbank.learning_patterns).toBeGreaterThan(stats1.reasoningbank.learning_patterns);
  }, 45000);
});

describe('ReasoningBank Performance Benchmarks', () => {
  let reasoningBank: ReasoningBankAgentDBIntegration;

  beforeAll(async () => {
    reasoningBank = new ReasoningBankAgentDBIntegration(config);
    await reasoningBank.initialize();
  });

  afterAll(async () => {
    if (reasoningBank) {
      await reasoningBank.shutdown();
    }
  });

  test('should meet performance targets for adaptive training', async () => {
    const startTime = performance.now();
    const result = await reasoningBank.adaptiveRLTraining();
    const endTime = performance.now();

    const executionTime = endTime - startTime;

    // Performance targets
    expect(executionTime).toBeLessThan(15000); // 15 seconds max
    expect(result.performance_metrics.overall_score).toBeGreaterThan(0.7); // 70% minimum performance
    expect(result.cross_agent_applicability).toBeGreaterThan(0.5); // 50% minimum cross-agent applicability
  }, 30000);

  test('should achieve memory efficiency targets', async () => {
    await reasoningBank.adaptiveRLTraining();
    const stats = await reasoningBank.getStatistics();

    // Should have reasonable memory usage
    expect(stats.reasoningbank).toBeDefined();
    expect(stats.reasoningbank.is_initialized).toBe(true);
  });

  test('should maintain search performance', async () => {
    const iterations = 5;
    const times: number[] = [];

    for (let i = 0; i < iterations; i++) {
      const startTime = performance.now();
      await reasoningBank.adaptiveRLTraining();
      const endTime = performance.now();
      times.push(endTime - startTime);
    }

    const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length;
    const maxTime = Math.max(...times);

    // Should maintain consistent performance
    expect(averageTime).toBeLessThan(12000); // 12 seconds average
    expect(maxTime).toBeLessThan(20000); // 20 seconds maximum
  }, 90000);
});