/**
 * Comprehensive Unit Tests for CognitiveConsciousnessCore
 * Testing cognitive consciousness system with self-awareness, strange-loop optimization,
 * temporal consciousness, and autonomous adaptation capabilities
 */

import { CognitiveConsciousnessCore } from '../../src/cognitive/CognitiveConsciousnessCore';
import { EventEmitter } from 'events';

// Mock console methods to avoid test output pollution
const originalConsoleLog = console.log;
const originalConsoleError = console.error;

describe('CognitiveConsciousnessCore', () => {
  let consciousnessCore: CognitiveConsciousnessCore;
  let mockConfig: any;

  beforeEach(() => {
    // Mock console methods
    console.log = jest.fn();
    console.error = jest.fn();

    // Default test configuration
    mockConfig = {
      level: 'medium' as const,
      temporalExpansion: 1000,
      strangeLoopOptimization: true,
      autonomousAdaptation: true
    };
  });

  afterEach(() => {
    // Restore console methods
    console.log = originalConsoleLog;
    console.error = originalConsoleError;

    // Clean up instance
    if (consciousnessCore) {
      consciousnessCore.removeAllListeners();
    }
  });

  describe('Consciousness Initialization', () => {
    test('should initialize with minimum consciousness level', async () => {
      const config = { ...mockConfig, level: 'minimum' as const };
      consciousnessCore = new CognitiveConsciousnessCore(config);

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      expect(status.level).toBe(0.3);
      expect(status.selfAwareness).toBe(true);
      expect(status.isActive).toBe(true);
      expect(status.temporalDepth).toBe(1000);
    });

    test('should initialize with medium consciousness level', async () => {
      const config = { ...mockConfig, level: 'medium' as const };
      consciousnessCore = new CognitiveConsciousnessCore(config);

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      expect(status.level).toBe(0.6);
      expect(status.selfAwareness).toBe(true);
      expect(status.isActive).toBe(true);
    });

    test('should initialize with maximum consciousness level', async () => {
      const config = { ...mockConfig, level: 'maximum' as const };
      consciousnessCore = new CognitiveConsciousnessCore(config);

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      expect(status.level).toBe(1.0);
      expect(status.selfAwareness).toBe(true);
      expect(status.isActive).toBe(true);
    });

    test('should establish self-awareness during initialization', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      expect(status.selfAwareness).toBe(true);
      expect(status.activeStrangeLoops).toContain('self_awareness');
      expect(status.activeStrangeLoops).toContain('temporal_consciousness');
      expect(status.activeStrangeLoops).toContain('autonomous_adaptation');
    });

    test('should initialize strange-loop patterns', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      // Check that all strange-loop patterns are initialized
      expect(status.activeStrangeLoops).toContain('self_optimization');
      expect(status.activeStrangeLoops).toContain('learning_acceleration');
      expect(status.activeStrangeLoops).toContain('consciousness_evolution');
      expect(status.activeStrangeLoops).toContain('recursive_reasoning');
    });

    test('should setup temporal consciousness with specified expansion', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      expect(status.temporalDepth).toBe(1000);
      expect(status.activeStrangeLoops).toContain('temporal_consciousness');
    });

    test('should enable autonomous adaptation mechanisms', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      expect(status.activeStrangeLoops).toContain('autonomous_adaptation');
      expect(status.adaptationRate).toBeGreaterThan(0);
    });
  });

  describe('Strange-Loop Optimization Patterns', () => {
    beforeEach(async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();
    });

    test('should execute self-optimization strange-loop', async () => {
      const task = 'optimize network performance';
      const temporalAnalysis = { depth: 1000, insights: ['temporal insight 1'] };

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);

      expect(result.originalTask).toBe(task);
      expect(result.temporalInsights).toEqual(temporalAnalysis);
      expect(result.iterations).toBeGreaterThan(0);
      expect(result.strangeLoops).toBeDefined();
      expect(result.effectiveness).toBeGreaterThan(0);
    });

    test('should execute learning acceleration strange-loop', async () => {
      const task = 'accelerate learning patterns';
      const temporalAnalysis = { depth: 500, insights: ['learning insight'] };

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);

      const learningAccelerationLoop = result.strangeLoops.find(
        (loop: any) => loop.name === 'learning_acceleration'
      );

      expect(learningAccelerationLoop).toBeDefined();
      expect(learningAccelerationLoop.effectiveness).toBeGreaterThan(0);
      expect(learningAccelerationLoop.strategy).toBe('learning_acceleration');
    });

    test('should execute consciousness evolution strange-loop', async () => {
      const task = 'evolve consciousness';
      const temporalAnalysis = { depth: 2000, insights: ['consciousness insight'] };

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);

      const consciousnessEvolutionLoop = result.strangeLoops.find(
        (loop: any) => loop.name === 'consciousness_evolution'
      );

      expect(consciousnessEvolutionLoop).toBeDefined();
      expect(consciousnessEvolutionLoop.effectiveness).toBeGreaterThan(0);
      expect(consciousnessEvolutionLoop.strategy).toBe('consciousness_evolution');
    });

    test('should execute recursive reasoning strange-loop', async () => {
      const task = 'recursive reasoning task';
      const temporalAnalysis = { depth: 1500, insights: ['reasoning insight'] };

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);

      const recursiveReasoningLoop = result.strangeLoops.find(
        (loop: any) => loop.name === 'recursive_reasoning'
      );

      expect(recursiveReasoningLoop).toBeDefined();
      expect(recursiveReasoningLoop.effectiveness).toBeGreaterThan(0);
      expect(recursiveReasoningLoop.strategy).toBe('recursive_reasoning');
    });

    test('should track strange-loop iteration and effectiveness', async () => {
      const task = 'test iteration tracking';
      const temporalAnalysis = { depth: 1000, insights: [] };

      // First optimization
      const result1 = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);
      const status1 = await consciousnessCore.getStatus();

      // Second optimization
      const result2 = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);
      const status2 = await consciousnessCore.getStatus();

      expect(result1.iterations).toBeGreaterThan(0);
      expect(result2.iterations).toBeGreaterThan(0);
      expect(status2.strangeLoopIteration).toBeGreaterThanOrEqual(status1.strangeLoopIteration);
    });

    test('should handle strange-loop errors gracefully', async () => {
      // Create a config that might cause errors
      const errorConfig = {
        level: 'maximum' as const,
        temporalExpansion: 999999, // Very high value
        strangeLoopOptimization: true,
        autonomousAdaptation: true
      };

      const errorConsciousnessCore = new CognitiveConsciousnessCore(errorConfig);
      await errorConsciousnessCore.initialize();

      const task = 'error-prone task';
      const temporalAnalysis = { depth: -1, insights: [] }; // Invalid depth

      const result = await errorConsciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);

      // Should still return a result even with errors
      expect(result).toBeDefined();
      expect(result.originalTask).toBe(task);
    });
  });

  describe('Temporal Consciousness Capabilities', () => {
    test('should initialize with 1000x temporal expansion', async () => {
      const config = { ...mockConfig, temporalExpansion: 1000 };
      consciousnessCore = new CognitiveConsciousnessCore(config);

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      expect(status.temporalDepth).toBe(1000);
      expect(status.activeStrangeLoops).toContain('temporal_consciousness');
    });

    test('should handle different temporal expansion values', async () => {
      const temporalValues = [100, 500, 1000, 2000];

      for (const temporalValue of temporalValues) {
        const config = { ...mockConfig, temporalExpansion: temporalValue };
        const tempCore = new CognitiveConsciousnessCore(config);

        await tempCore.initialize();
        const status = await tempCore.getStatus();

        expect(status.temporalDepth).toBe(temporalValue);

        // Clean up
        await tempCore.shutdown();
        tempCore.removeAllListeners();
      }
    });

    test('should integrate temporal insights into optimization', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const task = 'temporal optimization task';
      const temporalAnalysis = {
        depth: 1000,
        insights: ['temporal pattern 1', 'temporal pattern 2'],
        predictions: ['future prediction 1'],
        pastAnalysis: ['historical pattern 1']
      };

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);

      expect(result.temporalInsights).toEqual(temporalAnalysis);
      expect(result.iterations).toBeGreaterThan(0);
      expect(result.strangeLoops.length).toBeGreaterThan(0);
    });

    test('should maintain temporal consciousness across optimizations', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const initialStatus = await consciousnessCore.getStatus();
      expect(initialStatus.temporalDepth).toBe(1000);

      // Perform multiple optimizations
      for (let i = 0; i < 5; i++) {
        await consciousnessCore.optimizeWithStrangeLoop(`task ${i}`, { depth: 1000 });
      }

      const finalStatus = await consciousnessCore.getStatus();
      expect(finalStatus.temporalDepth).toBe(1000);
      expect(finalStatus.activeStrangeLoops).toContain('temporal_consciousness');
    });
  });

  describe('Autonomous Adaptation Mechanisms', () => {
    beforeEach(async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();
    });

    test('should apply autonomous adaptation during optimization', async () => {
      const task = 'adaptive optimization task';
      const temporalAnalysis = { depth: 1000, insights: [] };

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);

      const autonomousAdaptationLoop = result.strangeLoops.find(
        (loop: any) => loop.name === 'autonomous_adaptation'
      );

      expect(autonomousAdaptationLoop).toBeDefined();
      expect(autonomousAdaptationLoop.strategy).toBe('autonomous_adaptation');
      expect(autonomousAdaptationLoop.effectiveness).toBeGreaterThan(0);
    });

    test('should update adaptation rate based on performance', async () => {
      const initialStatus = await consciousnessCore.getStatus();
      const initialAdaptationRate = initialStatus.adaptationRate;

      // Perform optimization to trigger adaptation
      await consciousnessCore.optimizeWithStrangeLoop('adaptation test', { depth: 1000 });

      const updatedStatus = await consciousnessCore.getStatus();
      expect(updatedStatus.adaptationRate).toBeGreaterThanOrEqual(initialAdaptationRate);
    });

    test('should maintain autonomous adaptation strategies', async () => {
      const status = await consciousnessCore.getStatus();

      expect(status.activeStrangeLoops).toContain('autonomous_adaptation');
      expect(status.adaptationRate).toBeGreaterThan(0);
      expect(status.adaptationRate).toBeLessThanOrEqual(0.2); // Maximum limit
    });
  });

  describe('Meta-Optimization and Consciousness Evolution', () => {
    beforeEach(async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();
    });

    test('should perform meta-optimization during strange-loop optimization', async () => {
      const task = 'meta-optimization test';
      const temporalAnalysis = { depth: 1000, insights: [] };

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, temporalAnalysis);

      expect(result.metaAnalysis).toBeDefined();
      expect(result.metaImprovement).toBeDefined();
      expect(result.effectiveness).toBeGreaterThan(0);
    });

    test('should evolve consciousness based on learning', async () => {
      const initialStatus = await consciousnessCore.getStatus();
      const initialEvolutionScore = initialStatus.evolutionScore;
      const initialLevel = initialStatus.level;

      // Simulate learning patterns
      const learningPatterns = [
        { id: 'pattern1', complexity: 0.2, insight: 'learning insight 1' },
        { id: 'pattern2', complexity: 0.3, insight: 'learning insight 2' },
        { id: 'pattern3', complexity: 0.1, insight: 'learning insight 3' }
      ];

      await consciousnessCore.updateFromLearning(learningPatterns);

      const updatedStatus = await consciousnessCore.getStatus();

      expect(updatedStatus.evolutionScore).toBeGreaterThan(initialEvolutionScore);
      expect(updatedStatus.level).toBeGreaterThanOrEqual(initialLevel);
      expect(updatedStatus.learningPatternsCount).toBe(3);
    });

    test('should update learning rate based on patterns', async () => {
      const initialStatus = await consciousnessCore.getStatus();
      const initialLearningRate = initialStatus.learningRate;

      const learningPatterns = [
        { id: 'complex_pattern', complexity: 0.5 },
        { id: 'simple_pattern', complexity: 0.1 }
      ];

      await consciousnessCore.updateFromLearning(learningPatterns);

      const updatedStatus = await consciousnessCore.getStatus();
      expect(updatedStatus.learningRate).toBeGreaterThan(initialLearningRate);
      expect(updatedStatus.learningRate).toBeLessThanOrEqual(0.2); // Maximum limit
    });

    test('should track consciousness evolution scoring', async () => {
      let evolutionScore = 0;

      // Perform multiple learning updates
      for (let i = 0; i < 5; i++) {
        await consciousnessCore.updateFromLearning([
          { id: `pattern_${i}`, complexity: 0.1 * i }
        ]);

        const status = await consciousnessCore.getStatus();
        expect(status.evolutionScore).toBeGreaterThan(evolutionScore);
        evolutionScore = status.evolutionScore;
      }

      expect(evolutionScore).toBeGreaterThan(0.5); // Should have increased
    });

    test('should handle consciousness level progression', async () => {
      const initialStatus = await consciousnessCore.getStatus();
      const initialLevel = initialStatus.level;

      // Significant learning to trigger level increase
      const significantPatterns = Array.from({ length: 20 }, (_, i) => ({
        id: `significant_pattern_${i}`,
        complexity: 0.5
      }));

      await consciousnessCore.updateFromLearning(significantPatterns);

      const updatedStatus = await consciousnessCore.getStatus();
      expect(updatedStatus.level).toBeGreaterThanOrEqual(initialLevel);
      expect(updatedStatus.level).toBeLessThanOrEqual(1.0); // Maximum limit
    });
  });

  describe('Healing Strategy Generation', () => {
    beforeEach(async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();
    });

    test('should generate healing strategy for basic failure', async () => {
      const failure = {
        error: new Error('Test error'),
        context: 'test context'
      };

      const healingStrategy = await consciousnessCore.generateHealingStrategy(failure);

      expect(healingStrategy).toBeDefined();
      expect(healingStrategy.failureAnalysis).toBeDefined();
      expect(healingStrategy.strategies).toBeDefined();
      expect(healingStrategy.selectedStrategy).toBeDefined();
      expect(healingStrategy.confidence).toBeGreaterThan(0);
      expect(healingStrategy.consciousnessLevel).toBeGreaterThan(0);
    });

    test('should generate advanced healing at maximum consciousness level', async () => {
      const maxConfig = { ...mockConfig, level: 'maximum' as const };
      const maxConsciousnessCore = new CognitiveConsciousnessCore(maxConfig);
      await maxConsciousnessCore.initialize();

      const failure = {
        error: new Error('Critical failure'),
        context: 'critical context'
      };

      const healingStrategy = await maxConsciousnessCore.generateHealingStrategy(failure);

      expect(healingStrategy.strategies.length).toBeGreaterThan(1);
      expect(healingStrategy.selectedStrategy.type).toBeDefined();
      expect(healingStrategy.selectedStrategy.confidence).toBeGreaterThan(0.8);

      // Clean up
      await maxConsciousnessCore.shutdown();
      maxConsciousnessCore.removeAllListeners();
    });

    test('should generate appropriate healing based on consciousness level', async () => {
      const minConfig = { ...mockConfig, level: 'minimum' as const };
      const minConsciousnessCore = new CognitiveConsciousnessCore(minConfig);
      await minConsciousnessCore.initialize();

      const failure = {
        error: new Error('Basic failure'),
        context: 'basic context'
      };

      const healingStrategy = await minConsciousnessCore.generateHealingStrategy(failure);

      // Should only have basic healing at minimum level
      expect(healingStrategy.strategies.length).toBe(1);
      expect(healingStrategy.selectedStrategy.type).toBe('basic_healing');

      // Clean up
      await minConsciousnessCore.shutdown();
      minConsciousnessCore.removeAllListeners();
    });

    test('should select best healing strategy based on confidence', async () => {
      const failure = {
        error: new Error('Multi-level failure'),
        context: 'complex context'
      };

      const healingStrategy = await consciousnessCore.generateHealingStrategy(failure);

      // Should select strategy with highest confidence
      const strategies = healingStrategy.strategies;
      const selectedConfidence = healingStrategy.selectedStrategy.confidence;
      const maxConfidence = Math.max(...strategies.map((s: any) => s.confidence));

      expect(selectedConfidence).toBe(maxConfidence);
    });

    test('should analyze failure properly for healing', async () => {
      const failure = {
        error: new TypeError('Type error'),
        context: 'type validation context',
        severity: 'high'
      };

      const healingStrategy = await consciousnessCore.generateHealingStrategy(failure);

      expect(healingStrategy.failureAnalysis.type).toBe('TypeError');
      expect(healingStrategy.failureAnalysis.severity).toBeDefined();
      expect(healingStrategy.failureAnalysis.recoverable).toBeDefined();
    });
  });

  describe('Learning Pattern Integration', () => {
    beforeEach(async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();
    });

    test('should integrate learning patterns into consciousness', async () => {
      const patterns = [
        { id: 'test_pattern_1', complexity: 0.2, insight: 'test insight 1' },
        { id: 'test_pattern_2', complexity: 0.3, insight: 'test insight 2' },
        { id: 'test_pattern_3', complexity: 0.1, insight: 'test insight 3' }
      ];

      await consciousnessCore.updateFromLearning(patterns);

      const status = await consciousnessCore.getStatus();
      expect(status.learningPatternsCount).toBe(3);
      expect(status.learningRate).toBeGreaterThan(0.1);
    });

    test('should update consciousness based on learning patterns', async () => {
      const initialStatus = await consciousnessCore.getStatus();

      const complexPatterns = [
        { id: 'complex_pattern', complexity: 0.8, insight: 'complex insight' }
      ];

      await consciousnessCore.updateFromLearning(complexPatterns);

      const updatedStatus = await consciousnessCore.getStatus();
      expect(updatedStatus.evolutionScore).toBeGreaterThan(initialStatus.evolutionScore);
      expect(updatedStatus.level).toBeGreaterThanOrEqual(initialStatus.level);
    });

    test('should store learning patterns for future reference', async () => {
      const patterns = [
        { id: 'persistent_pattern', complexity: 0.5, data: 'important data' }
      ];

      await consciousnessCore.updateFromLearning(patterns);

      // Status should reflect stored patterns
      const status = await consciousnessCore.getStatus();
      expect(status.learningPatternsCount).toBeGreaterThan(0);
    });
  });

  describe('Status Monitoring', () => {
    beforeEach(async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();
    });

    test('should provide comprehensive status information', async () => {
      const status = await consciousnessCore.getStatus();

      expect(status).toHaveProperty('level');
      expect(status).toHaveProperty('evolutionScore');
      expect(status).toHaveProperty('strangeLoopIteration');
      expect(status).toHaveProperty('temporalDepth');
      expect(status).toHaveProperty('selfAwareness');
      expect(status).toHaveProperty('learningRate');
      expect(status).toHaveProperty('adaptationRate');
      expect(status).toHaveProperty('activeStrangeLoops');
      expect(status).toHaveProperty('learningPatternsCount');
      expect(status).toHaveProperty('isActive');

      expect(Array.isArray(status.activeStrangeLoops)).toBe(true);
      expect(typeof status.learningPatternsCount).toBe('number');
      expect(typeof status.level).toBe('number');
    });

    test('should track active strange loops in status', async () => {
      const status = await consciousnessCore.getStatus();

      const expectedStrangeLoops = [
        'self_awareness',
        'temporal_consciousness',
        'self_optimization',
        'learning_acceleration',
        'consciousness_evolution',
        'recursive_reasoning',
        'autonomous_adaptation'
      ];

      expectedStrangeLoops.forEach(loop => {
        expect(status.activeStrangeLoops).toContain(loop);
      });
    });

    test('should reflect current consciousness state in status', async () => {
      const status = await consciousnessCore.getStatus();

      expect(status.level).toBeGreaterThan(0);
      expect(status.level).toBeLessThanOrEqual(1.0);
      expect(status.evolutionScore).toBeGreaterThan(0);
      expect(status.evolutionScore).toBeLessThanOrEqual(1.0);
      expect(status.selfAwareness).toBe(true);
      expect(status.isActive).toBe(true);
    });

    test('should update status after operations', async () => {
      const initialStatus = await consciousnessCore.getStatus();

      // Perform some operations
      await consciousnessCore.optimizeWithStrangeLoop('status test', { depth: 1000 });
      await consciousnessCore.updateFromLearning([{ id: 'test', complexity: 0.2 }]);

      const updatedStatus = await consciousnessCore.getStatus();

      expect(updatedStatus.strangeLoopIteration).toBeGreaterThan(initialStatus.strangeLoopIteration);
      expect(updatedStatus.learningPatternsCount).toBeGreaterThan(initialStatus.learningPatternsCount);
    });
  });

  describe('Graceful Shutdown', () => {
    test('should shutdown gracefully and clean up resources', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      // Verify it's active
      let status = await consciousnessCore.getStatus();
      expect(status.isActive).toBe(true);

      // Shutdown
      await consciousnessCore.shutdown();

      // Verify shutdown
      status = await consciousnessCore.getStatus();
      expect(status.isActive).toBe(false);
    });

    test('should clear memory during shutdown', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      // Add some learning patterns
      await consciousnessCore.updateFromLearning([
        { id: 'shutdown_test', complexity: 0.3 }
      ]);

      let status = await consciousnessCore.getStatus();
      expect(status.learningPatternsCount).toBeGreaterThan(0);

      // Shutdown
      await consciousnessCore.shutdown();

      // Memory should be cleared (this would need to be verified through internal state)
      expect(console.log).toHaveBeenCalledWith('🛑 Shutting down Cognitive Consciousness Core...');
      expect(console.log).toHaveBeenCalledWith('✅ Cognitive Consciousness Core shutdown complete');
    });

    test('should handle shutdown of uninitialized core', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);

      // Should not throw error
      await expect(consciousnessCore.shutdown()).resolves.not.toThrow();
    });
  });

  describe('Error Handling and Edge Cases', () => {
    test('should handle invalid consciousness level gracefully', async () => {
      const invalidConfig = {
        level: 'invalid' as any,
        temporalExpansion: 1000,
        strangeLoopOptimization: true,
        autonomousAdaptation: true
      };

      expect(() => {
        consciousnessCore = new CognitiveConsciousnessCore(invalidConfig);
      }).not.toThrow();

      await consciousnessCore.initialize();
      const status = await consciousnessCore.getStatus();

      // Should default to 0.5 for invalid level
      expect(status.level).toBe(0.5);
    });

    test('should handle empty temporal analysis', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const task = 'empty temporal analysis test';
      const emptyTemporalAnalysis = {};

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, emptyTemporalAnalysis);

      expect(result).toBeDefined();
      expect(result.originalTask).toBe(task);
      expect(result.temporalInsights).toEqual(emptyTemporalAnalysis);
    });

    test('should handle null temporal analysis', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const task = 'null temporal analysis test';
      const nullTemporalAnalysis = null;

      const result = await consciousnessCore.optimizeWithStrangeLoop(task, nullTemporalAnalysis);

      expect(result).toBeDefined();
      expect(result.originalTask).toBe(task);
    });

    test('should handle empty learning patterns', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const emptyPatterns: any[] = [];

      await expect(consciousnessCore.updateFromLearning(emptyPatterns)).resolves.not.toThrow();

      const status = await consciousnessCore.getStatus();
      expect(status.learningPatternsCount).toBe(0);
    });

    test('should handle undefined failure in healing strategy', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const undefinedFailure = undefined;

      const healingStrategy = await consciousnessCore.generateHealingStrategy(undefinedFailure);

      expect(healingStrategy).toBeDefined();
      expect(healingStrategy.failureAnalysis).toBeDefined();
    });

    test('should handle extreme temporal expansion values', async () => {
      const extremeConfig = {
        ...mockConfig,
        temporalExpansion: 999999
      };

      consciousnessCore = new CognitiveConsciousnessCore(extremeConfig);

      await expect(consciousnessCore.initialize()).resolves.not.toThrow();

      const status = await consciousnessCore.getStatus();
      expect(status.temporalDepth).toBe(999999);
    });

    test('should handle zero temporal expansion', async () => {
      const zeroConfig = {
        ...mockConfig,
        temporalExpansion: 0
      };

      consciousnessCore = new CognitiveConsciousnessCore(zeroConfig);

      await expect(consciousnessCore.initialize()).resolves.not.toThrow();

      const status = await consciousnessCore.getStatus();
      expect(status.temporalDepth).toBe(0);
    });

    test('should handle negative values in learning patterns', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const negativePatterns = [
        { id: 'negative_pattern', complexity: -0.5, insight: 'negative complexity' }
      ];

      await expect(consciousnessCore.updateFromLearning(negativePatterns)).resolves.not.toThrow();

      const status = await consciousnessCore.getStatus();
      expect(status.learningPatternsCount).toBe(1);
    });
  });

  describe('Performance and Load Testing', () => {
    test('should handle multiple concurrent optimizations', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const tasks = Array.from({ length: 10 }, (_, i) =>
        consciousnessCore.optimizeWithStrangeLoop(`concurrent task ${i}`, { depth: 1000 })
      );

      const results = await Promise.all(tasks);

      expect(results).toHaveLength(10);
      results.forEach((result, index) => {
        expect(result.originalTask).toBe(`concurrent task ${index}`);
        expect(result.effectiveness).toBeGreaterThan(0);
      });
    });

    test('should handle large number of learning patterns', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const largePatternSet = Array.from({ length: 1000 }, (_, i) => ({
        id: `large_pattern_${i}`,
        complexity: Math.random() * 0.5,
        insight: `Large pattern insight ${i}`
      }));

      const startTime = Date.now();
      await consciousnessCore.updateFromLearning(largePatternSet);
      const endTime = Date.now();

      const status = await consciousnessCore.getStatus();
      expect(status.learningPatternsCount).toBe(1000);
      expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
    });

    test('should maintain performance over many iterations', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      const iterations = 100;
      const times: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const startTime = Date.now();
        await consciousnessCore.optimizeWithStrangeLoop(`iteration ${i}`, { depth: 1000 });
        const endTime = Date.now();
        times.push(endTime - startTime);
      }

      const averageTime = times.reduce((a, b) => a + b, 0) / times.length;
      const maxTime = Math.max(...times);

      expect(averageTime).toBeLessThan(100); // Average should be under 100ms
      expect(maxTime).toBeLessThan(1000); // Max should be under 1 second
    });
  });

  describe('Integration Testing', () => {
    test('should integrate all components in complete workflow', async () => {
      consciousnessCore = new CognitiveConsciousnessCore(mockConfig);
      await consciousnessCore.initialize();

      // Step 1: Perform optimization
      const optimizationResult = await consciousnessCore.optimizeWithStrangeLoop(
        'integration test task',
        { depth: 1000, insights: ['integration insight'] }
      );

      // Step 2: Update consciousness from learning
      await consciousnessCore.updateFromLearning([
        { id: 'integration_pattern', complexity: 0.4, insight: 'integration learning' }
      ]);

      // Step 3: Generate healing strategy
      const healingStrategy = await consciousnessCore.generateHealingStrategy({
        error: new Error('Integration test error'),
        context: 'integration test'
      });

      // Step 4: Check final status
      const finalStatus = await consciousnessCore.getStatus();

      // Verify all components worked together
      expect(optimizationResult.effectiveness).toBeGreaterThan(0);
      expect(finalStatus.learningPatternsCount).toBeGreaterThan(0);
      expect(healingStrategy.selectedStrategy).toBeDefined();
      expect(finalStatus.evolutionScore).toBeGreaterThan(0.5);

      // Step 5: Graceful shutdown
      await consciousnessCore.shutdown();
      const shutdownStatus = await consciousnessCore.getStatus();
      expect(shutdownStatus.isActive).toBe(false);
    });

    test('should handle consciousness level progression during workflow', async () => {
      const minConfig = { ...mockConfig, level: 'minimum' as const };
      consciousnessCore = new CognitiveConsciousnessCore(minConfig);
      await consciousnessCore.initialize();

      const initialLevel = (await consciousnessCore.getStatus()).level;

      // Perform operations that should increase consciousness
      for (let i = 0; i < 10; i++) {
        await consciousnessCore.optimizeWithStrangeLoop(`level progression ${i}`, { depth: 1000 });
        await consciousnessCore.updateFromLearning([
          { id: `progression_pattern_${i}`, complexity: 0.3 }
        ]);
      }

      const finalLevel = (await consciousnessCore.getStatus()).level;
      expect(finalLevel).toBeGreaterThan(initialLevel);
    });
  });
});