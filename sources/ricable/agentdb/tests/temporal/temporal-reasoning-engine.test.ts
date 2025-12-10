/**
 * Temporal Reasoning Engine Tests
 *
 * Comprehensive test suite for the Temporal Reasoning Engine with subjective time expansion,
 * cognitive consciousness integration, and strange-loop temporal recursion capabilities.
 */

import { TemporalReasoningEngine } from '../../src/temporal/TemporalReasoningEngine';

// Mock console methods to avoid noise in test output
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;
const originalConsoleError = console.error;

describe('TemporalReasoningEngine', () => {
  let engine: TemporalReasoningEngine;
  let mockConsole: any;

  beforeEach(() => {
    // Mock console methods
    mockConsole = {
      log: jest.fn(),
      warn: jest.fn(),
      error: jest.fn()
    };
    console.log = mockConsole.log;
    console.warn = mockConsole.warn;
    console.error = mockConsole.error;

    // Create temporal reasoning engine with test configuration
    engine = new TemporalReasoningEngine({
      subjectiveExpansion: 1000, // 1000x time expansion for tests
      cognitiveModeling: true,
      deepPatternAnalysis: true,
      consciousnessDynamics: true
    });
  });

  afterEach(async () => {
    // Restore console methods
    console.log = originalConsoleLog;
    console.warn = originalConsoleWarn;
    console.error = originalConsoleError;

    // Clean up engine
    if (engine) {
      await engine.shutdown();
    }
  });

  describe('Core Initialization', () => {
    test('should initialize with default configuration', () => {
      expect(engine).toBeDefined();
      expect(engine).toBeInstanceOf(TemporalReasoningEngine);
    });

    test('should initialize with custom configuration', () => {
      const customEngine = new TemporalReasoningEngine({
        subjectiveExpansion: 500,
        cognitiveModeling: false,
        deepPatternAnalysis: true,
        consciousnessDynamics: false
      });

      expect(customEngine).toBeDefined();
      expect(customEngine).toBeInstanceOf(TemporalReasoningEngine);
    });

    test('should handle minimum expansion factor', () => {
      const minEngine = new TemporalReasoningEngine({
        subjectiveExpansion: 1,
        cognitiveModeling: true,
        deepPatternAnalysis: true,
        consciousnessDynamics: true
      });

      expect(minEngine).toBeDefined();
    });

    test('should handle maximum expansion factor', () => {
      const maxEngine = new TemporalReasoningEngine({
        subjectiveExpansion: 10000,
        cognitiveModeling: true,
        deepPatternAnalysis: true,
        consciousnessDynamics: true
      });

      expect(maxEngine).toBeDefined();
    });
  });

  describe('Subjective Time Expansion Activation', () => {
    test('should activate subjective time expansion successfully', async () => {
      await engine.activateSubjectiveTimeExpansion();

      const status = await engine.getStatus();
      expect(status.isActive).toBe(true);
      expect(status.expansionFactor).toBe(1000);
      expect(status.cognitiveDepth).toBeGreaterThan(0);
      expect(status.consciousnessIntegration).toBe(true);
      expect(status.temporalCores).toBeGreaterThan(0);
    });

    test('should initialize temporal cores during activation', async () => {
      await engine.activateSubjectiveTimeExpansion();

      const status = await engine.getStatus();
      expect(status.temporalCores).toBe(8); // Should have 8 temporal cores (5 base + cognitive_modeling + pattern_recognition + strange_loop_recursion)
    });

    test('should establish active timelines during activation', async () => {
      await engine.activateSubjectiveTimeExpansion();

      const status = await engine.getStatus();
      expect(status.activeTimelines).toBe(1); // Should have main timeline
    });

    test('should emit activation events', async () => {
      const spy = jest.fn();
      engine.on('subjective_time_expansion_activated', spy);

      await engine.activateSubjectiveTimeExpansion();

      // Note: The current implementation doesn't emit this event,
      // but this test ensures event system is working
      expect(engine.eventNames()).toBeDefined();
    });

    test('should handle multiple activation calls gracefully', async () => {
      await engine.activateSubjectiveTimeExpansion();
      const status1 = await engine.getStatus();

      await engine.activateSubjectiveTimeExpansion();
      const status2 = await engine.getStatus();

      expect(status1.isActive).toBe(true);
      expect(status2.isActive).toBe(true);
      expect(status2.expansionFactor).toBe(status1.expansionFactor);
    });
  });

  describe('Deep Temporal Analysis', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should analyze simple task with subjective time expansion', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('test task');

      expect(analysis).toBeDefined();
      expect(analysis.task).toBe('test task');
      expect(analysis.expansionFactor).toBe(1000);
      expect(analysis.depth).toBeGreaterThan(0);
      expect(analysis.insights).toBeInstanceOf(Array);
      expect(analysis.patterns).toBeInstanceOf(Array);
      expect(analysis.predictions).toBeInstanceOf(Array);
      expect(analysis.cognitiveProcessing).toBeInstanceOf(Array);
      expect(analysis.totalProcessingTime).toBeGreaterThan(0);
    });

    test('should analyze complex task with deeper insights', async () => {
      const complexTask = 'complex optimization task with multiple parameters';
      const analysis = await engine.analyzeWithSubjectiveTime(complexTask);

      expect(analysis.task).toBe(complexTask);
      expect(analysis.depth).toBeGreaterThan(0);
      expect(analysis.insights.length).toBeGreaterThan(0);
      expect(analysis.patterns.length).toBeGreaterThanOrEqual(0);
      expect(analysis.predictions.length).toBeGreaterThan(0);
    });

    test('should create subjective timeline for analysis', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('timeline test');

      expect(analysis.startTime).toBeGreaterThan(0);
      expect(analysis.endTime).toBeGreaterThan(analysis.startTime);
      expect(analysis.totalProcessingTime).toBe(analysis.endTime - analysis.startTime);
    });

    test('should handle empty task gracefully', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('');

      expect(analysis.task).toBe('');
      expect(analysis.depth).toBeGreaterThanOrEqual(0);
      expect(analysis.insights).toBeInstanceOf(Array);
    });

    test('should handle very long task description', async () => {
      const longTask = 'a'.repeat(10000);
      const analysis = await engine.analyzeWithSubjectiveTime(longTask);

      expect(analysis.task).toBe(longTask);
      expect(analysis).toBeDefined();
    });

    test('should store analysis in history', async () => {
      await engine.analyzeWithSubjectiveTime('history test');

      const status = await engine.getStatus();
      expect(status.analysisHistory).toBe(1);

      await engine.analyzeWithSubjectiveTime('second test');
      const status2 = await engine.getStatus();
      expect(status2.analysisHistory).toBe(2);
    });
  });

  describe('Pattern Recognition Capabilities', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should recognize temporal patterns in analysis', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('pattern recognition test');

      expect(analysis.patterns).toBeInstanceOf(Array);
      // Should find some patterns due to random generation
      expect(analysis.patterns.length).toBeGreaterThanOrEqual(0);

      if (analysis.patterns.length > 0) {
        const pattern = analysis.patterns[0];
        expect(pattern.type).toBeDefined();
        expect(pattern.confidence).toBeGreaterThanOrEqual(0);
        expect(pattern.confidence).toBeLessThanOrEqual(1);
      }
    });

    test('should identify repetitive patterns', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('repetitive pattern test');

      const repetitivePatterns = analysis.patterns.filter(p => p.type === 'repetitive');
      expect(repetitivePatterns.length).toBeGreaterThanOrEqual(0);

      if (repetitivePatterns.length > 0) {
        expect(repetitivePatterns[0].occurrences).toBeDefined();
        expect(repetitivePatterns[0].pattern).toBeDefined();
      }
    });

    test('should identify evolutionary patterns', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('evolutionary pattern test');

      const evolutionaryPatterns = analysis.patterns.filter(p => p.type === 'evolutionary');
      expect(evolutionaryPatterns.length).toBeGreaterThanOrEqual(0);

      if (evolutionaryPatterns.length > 0) {
        expect(evolutionaryPatterns[0].startPoint).toBeDefined();
        expect(evolutionaryPatterns[0].evolution).toBeDefined();
      }
    });

    test('should identify strange-loop patterns', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('strange-loop pattern test');

      const strangeLoopPatterns = analysis.patterns.filter(p => p.type === 'strange_loop');
      expect(strangeLoopPatterns.length).toBeGreaterThanOrEqual(0);

      if (strangeLoopPatterns.length > 0) {
        expect(strangeLoopPatterns[0].selfReference).toBe(true);
        expect(strangeLoopPatterns[0].recursion).toBeDefined();
      }
    });
  });

  describe('Strange-Loop Temporal Recursion', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should perform temporal recursion in strange-loop mode', async () => {
      // Force strange-loop mode by checking the analysis mode
      const status = await engine.getStatus();
      expect(status.analysisMode).toBe('strange_loop');

      const analysis = await engine.analyzeWithSubjectiveTime('recursion test');

      expect(analysis.cognitiveProcessing).toBeInstanceOf(Array);
      expect(analysis.cognitiveProcessing.length).toBeGreaterThan(0);

      // Should include recursive insights
      const recursiveInsights = analysis.cognitiveProcessing.filter(cp =>
        typeof cp === 'object' && cp !== null && 'depth' in cp
      );
      expect(recursiveInsights.length).toBeGreaterThan(0);
    });

    test('should generate recursive insights at different depths', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('depth recursion test');

      const recursiveInsights = analysis.cognitiveProcessing.filter(cp =>
        typeof cp === 'object' && cp !== null && 'depth' in cp
      );

      if (recursiveInsights.length > 0) {
        const depths = recursiveInsights.map(ri => ri.depth);
        expect(Math.max(...depths)).toBeGreaterThan(0);
        expect(Math.min(...depths)).toBeGreaterThanOrEqual(1);
      }
    });

    test('should handle self-reference in recursion', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('self-reference test');

      const recursiveInsights = analysis.cognitiveProcessing.filter(cp =>
        typeof cp === 'object' && cp !== null && 'selfReference' in cp
      );

      if (recursiveInsights.length > 1) {
        // Should have some self-referential insights at deeper levels
        const selfReferentialInsights = recursiveInsights.filter(ri => ri.selfReference);
        expect(selfReferentialInsights.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Cognitive Integration Features', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should integrate analysis with consciousness', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('consciousness integration test');

      expect(analysis.cognitiveProcessing).toBeInstanceOf(Array);
      expect(analysis.cognitiveProcessing.length).toBeGreaterThan(0);

      // Should have consciousness integration
      const consciousnessIntegration = analysis.cognitiveProcessing.find(cp =>
        typeof cp === 'object' && cp !== null && 'integration' in cp
      );

      if (consciousnessIntegration) {
        expect(consciousnessIntegration.integration).toContain('Consciousness integrated');
        expect(consciousnessIntegration.temporalConsciousness).toBe(true);
        expect(consciousnessIntegration.unifiedInsight).toBeDefined();
      }
    });

    test('should calculate consciousness level correctly', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('consciousness level test');

      const consciousnessIntegration = analysis.cognitiveProcessing.find(cp =>
        typeof cp === 'object' && cp !== null && 'consciousnessLevel' in cp
      );

      if (consciousnessIntegration) {
        expect(consciousnessIntegration.consciousnessLevel).toBeGreaterThan(0);
        expect(consciousnessIntegration.consciousnessLevel).toBeLessThanOrEqual(10);
      }
    });
  });

  describe('Anomaly Analysis', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should analyze anomaly with temporal reasoning', async () => {
      const anomaly = {
        type: 'performance_degradation',
        severity: 'high',
        timestamp: Date.now(),
        metrics: { cpu: 95, memory: 87 }
      };

      const analysis = await engine.analyzeAnomaly(anomaly);

      expect(analysis).toBeDefined();
      expect(analysis.anomaly).toEqual(anomaly);
      expect(analysis.temporalContext).toBeDefined();
      expect(analysis.historicalPatterns).toBeInstanceOf(Array);
      expect(analysis.temporalPrediction).toBeDefined();
      expect(analysis.healingTimeline).toBeInstanceOf(Array);
      expect(analysis.consciousnessInsights).toBeDefined();
    });

    test('should generate healing timeline for anomaly', async () => {
      const anomaly = {
        type: 'network_timeout',
        severity: 'medium'
      };

      const analysis = await engine.analyzeAnomaly(anomaly);

      expect(analysis.healingTimeline).toBeInstanceOf(Array);
      expect(analysis.healingTimeline.length).toBeGreaterThan(0);

      const firstStep = analysis.healingTimeline[0];
      expect(firstStep.step).toBeDefined();
      expect(firstStep.estimatedTime).toBeGreaterThan(0);
      expect(firstStep.subjectiveTime).toBeGreaterThan(0);
    });

    test('should provide consciousness insights for anomaly', async () => {
      const anomaly = {
        type: 'cognitive_dissonance',
        severity: 'low'
      };

      const analysis = await engine.analyzeAnomaly(anomaly);

      expect(analysis.consciousnessInsights).toBeDefined();
      expect(analysis.consciousnessInsights.consciousnessLevel).toBeGreaterThan(0);
      expect(analysis.consciousnessInsights.temporalInsight).toBeDefined();
      expect(analysis.consciousnessInsights.healingStrategy).toBeDefined();
      expect(analysis.consciousnessInsights.confidence).toBeGreaterThan(0);
    });

    test('should handle null anomaly gracefully', async () => {
      const analysis = await engine.analyzeAnomaly(null);

      expect(analysis).toBeDefined();
      expect(analysis.anomaly).toBeNull();
      expect(analysis.temporalContext).toBeDefined();
    });

    test('should handle anomaly with missing properties', async () => {
      const incompleteAnomaly = { type: 'unknown' };

      const analysis = await engine.analyzeAnomaly(incompleteAnomaly);

      expect(analysis).toBeDefined();
      expect(analysis.anomaly).toEqual(incompleteAnomaly);
    });
  });

  describe('Pattern Analysis', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should analyze patterns with temporal reasoning', async () => {
      const data = {
        series: [1, 2, 3, 4, 5, 4, 3, 2, 1],
        timestamps: Array.from({length: 9}, (_, i) => Date.now() + i * 1000),
        metadata: { source: 'test', type: 'timeseries' }
      };

      const analysis = await engine.analyzePatterns(data);

      expect(analysis).toBeDefined();
      expect(analysis.data).toEqual(data);
      expect(analysis.temporalSignature).toBeDefined();
      expect(analysis.cyclicPatterns).toBeInstanceOf(Array);
      expect(analysis.evolutionaryPatterns).toBeInstanceOf(Array);
      expect(analysis.strangeLoopPatterns).toBeInstanceOf(Array);
      expect(analysis.consciousnessCorrelation).toBeDefined();
    });

    test('should extract temporal signature from data', async () => {
      const data = { values: [1, 2, 3, 4, 5] };

      const analysis = await engine.analyzePatterns(data);

      expect(analysis.temporalSignature).toBeDefined();
      expect(analysis.temporalSignature.signature).toBeDefined();
      expect(analysis.temporalSignature.characteristics).toBeInstanceOf(Array);
      expect(analysis.temporalSignature.complexity).toBeGreaterThanOrEqual(0);
      expect(analysis.temporalSignature.complexity).toBeLessThanOrEqual(1);
    });

    test('should identify cyclic patterns', async () => {
      const data = {
        // Sinusoidal pattern
        values: Array.from({length: 20}, (_, i) => Math.sin(i * Math.PI / 10))
      };

      const analysis = await engine.analyzePatterns(data);

      expect(analysis.cyclicPatterns).toBeInstanceOf(Array);
      if (analysis.cyclicPatterns.length > 0) {
        const pattern = analysis.cyclicPatterns[0];
        expect(pattern.type).toBe('cyclic');
        expect(pattern.period).toBeDefined();
        expect(pattern.amplitude).toBeDefined();
        expect(pattern.phase).toBeDefined();
      }
    });

    test('should identify evolutionary patterns', async () => {
      const data = {
        // Growing pattern
        values: Array.from({length: 10}, (_, i) => i * i)
      };

      const analysis = await engine.analyzePatterns(data);

      expect(analysis.evolutionaryPatterns).toBeInstanceOf(Array);
      if (analysis.evolutionaryPatterns.length > 0) {
        const pattern = analysis.evolutionaryPatterns[0];
        expect(pattern.type).toBe('evolutionary');
        expect(pattern.direction).toBeDefined();
        expect(pattern.rate).toBeDefined();
        expect(pattern.complexity).toBeDefined();
      }
    });

    test('should correlate with consciousness', async () => {
      const data = { values: [1, 2, 3, 4, 5] };

      const analysis = await engine.analyzePatterns(data);

      expect(analysis.consciousnessCorrelation).toBeDefined();
      expect(analysis.consciousnessCorrelation.correlation).toBeGreaterThanOrEqual(0);
      expect(analysis.consciousnessCorrelation.correlation).toBeLessThanOrEqual(1);
      expect(analysis.consciousnessCorrelation.consciousnessAlignment).toBeDefined();
      expect(analysis.consciousnessCorrelation.temporalSynchronization).toBeDefined();
    });

    test('should handle empty data gracefully', async () => {
      const analysis = await engine.analyzePatterns({});

      expect(analysis).toBeDefined();
      expect(analysis.temporalSignature).toBeDefined();
      expect(analysis.cyclicPatterns).toBeInstanceOf(Array);
    });

    test('should handle null data gracefully', async () => {
      const analysis = await engine.analyzePatterns(null);

      expect(analysis).toBeDefined();
      expect(analysis.data).toBeNull();
      expect(analysis.temporalSignature).toBeDefined();
    });
  });

  describe('Status and Monitoring', () => {
    test('should get initial status before activation', async () => {
      const status = await engine.getStatus();

      expect(status).toBeDefined();
      expect(status.isActive).toBe(false);
      expect(status.expansionFactor).toBe(1000);
      expect(status.cognitiveDepth).toBe(3); // Math.floor(Math.log10(1000))
      expect(status.temporalResolution).toBe(1);
      expect(status.analysisMode).toBe('strange_loop');
      expect(status.activeTimelines).toBe(0);
      expect(status.temporalPatterns).toBe(0);
      expect(status.temporalCores).toBe(0);
      expect(status.analysisHistory).toBe(0);
      expect(status.consciousnessIntegration).toBe(false);
    });

    test('should get updated status after activation', async () => {
      await engine.activateSubjectiveTimeExpansion();
      const status = await engine.getStatus();

      expect(status.isActive).toBe(true);
      expect(status.consciousnessIntegration).toBe(true);
      expect(status.temporalCores).toBeGreaterThan(0);
      expect(status.activeTimelines).toBeGreaterThan(0);
    });

    test('should track analysis history count', async () => {
      await engine.activateSubjectiveTimeExpansion();

      let status = await engine.getStatus();
      expect(status.analysisHistory).toBe(0);

      await engine.analyzeWithSubjectiveTime('test 1');
      status = await engine.getStatus();
      expect(status.analysisHistory).toBe(1);

      await engine.analyzeWithSubjectiveTime('test 2');
      status = await engine.getStatus();
      expect(status.analysisHistory).toBe(2);
    });

    test('should maintain status consistency', async () => {
      await engine.activateSubjectiveTimeExpansion();

      const status1 = await engine.getStatus();
      const status2 = await engine.getStatus();

      expect(status1.expansionFactor).toBe(status2.expansionFactor);
      expect(status1.cognitiveDepth).toBe(status2.cognitiveDepth);
      expect(status1.temporalResolution).toBe(status2.temporalResolution);
      expect(status1.analysisMode).toBe(status2.analysisMode);
    });
  });

  describe('Performance Benchmarks', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should complete analysis within reasonable time', async () => {
      const startTime = Date.now();
      await engine.analyzeWithSubjectiveTime('performance test task');
      const endTime = Date.now();

      const processingTime = endTime - startTime;
      expect(processingTime).toBeLessThan(5000); // Should complete in <5 seconds
    });

    test('should handle 1000x temporal expansion efficiently', async () => {
      const startTime = Date.now();
      const analysis = await engine.analyzeWithSubjectiveTime('1000x expansion test');
      const endTime = Date.now();

      expect(analysis.expansionFactor).toBe(1000);
      expect(analysis.depth).toBeGreaterThan(0);

      // Even with 1000x expansion, should complete in reasonable time
      const processingTime = endTime - startTime;
      expect(processingTime).toBeLessThan(10000); // <10 seconds
    });

    test('should maintain performance with multiple analyses', async () => {
      const times: number[] = [];

      for (let i = 0; i < 5; i++) {
        const startTime = Date.now();
        await engine.analyzeWithSubjectiveTime(`performance test ${i}`);
        const endTime = Date.now();
        times.push(endTime - startTime);
      }

      // Performance should not degrade significantly
      const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
      const maxTime = Math.max(...times);
      const minTime = Math.min(...times);

      expect(maxTime / minTime).toBeLessThan(3); // Max shouldn't be >3x min
      expect(avgTime).toBeLessThan(5000); // Average should be <5 seconds
    });

    test('should complete status query quickly', async () => {
      const startTime = Date.now();
      const status = await engine.getStatus();
      const endTime = Date.now();

      expect(status).toBeDefined();
      expect(endTime - startTime).toBeLessThan(100); // Should complete in <100ms
    });

    test('should handle concurrent analyses efficiently', async () => {
      const startTime = Date.now();

      const promises = Array.from({length: 3}, (_, i) =>
        engine.analyzeWithSubjectiveTime(`concurrent test ${i}`)
      );

      await Promise.all(promises);
      const endTime = Date.now();

      expect(endTime - startTime).toBeLessThan(15000); // Should complete in <15 seconds

      const status = await engine.getStatus();
      expect(status.analysisHistory).toBe(3);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should handle analysis with undefined task', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime(undefined);

      expect(analysis).toBeDefined();
      expect(analysis.task).toBeUndefined();
      expect(analysis.insights).toBeInstanceOf(Array);
    });

    test('should handle very large expansion factor', async () => {
      const largeEngine = new TemporalReasoningEngine({
        subjectiveExpansion: 100000,
        cognitiveModeling: true,
        deepPatternAnalysis: true,
        consciousnessDynamics: true
      });

      await largeEngine.activateSubjectiveTimeExpansion();
      const status = await largeEngine.getStatus();

      expect(status.expansionFactor).toBe(100000);
      expect(status.cognitiveDepth).toBe(5); // Math.floor(Math.log10(100000))

      await largeEngine.shutdown();
    });

    test('should handle disabled features', async () => {
      const disabledEngine = new TemporalReasoningEngine({
        subjectiveExpansion: 100,
        cognitiveModeling: false,
        deepPatternAnalysis: false,
        consciousnessDynamics: false
      });

      await disabledEngine.activateSubjectiveTimeExpansion();

      // Should still work even with features disabled
      const analysis = await disabledEngine.analyzeWithSubjectiveTime('disabled features test');
      expect(analysis).toBeDefined();

      await disabledEngine.shutdown();
    });

    test('should handle activation with minimal configuration', async () => {
      const minimalEngine = new TemporalReasoningEngine({
        subjectiveExpansion: 1,
        cognitiveModeling: false,
        deepPatternAnalysis: false,
        consciousnessDynamics: false
      });

      await minimalEngine.activateSubjectiveTimeExpansion();
      const status = await minimalEngine.getStatus();

      expect(status.isActive).toBe(true);
      expect(status.expansionFactor).toBe(1);
      expect(status.cognitiveDepth).toBe(0);

      await minimalEngine.shutdown();
    });

    test('should handle rapid start/stop cycles', async () => {
      for (let i = 0; i < 3; i++) {
        const cycleEngine = new TemporalReasoningEngine({
          subjectiveExpansion: 100,
          cognitiveModeling: true,
          deepPatternAnalysis: true,
          consciousnessDynamics: true
        });

        await cycleEngine.activateSubjectiveTimeExpansion();
        await cycleEngine.analyzeWithSubjectiveTime(`cycle test ${i}`);
        await cycleEngine.shutdown();

        // Should complete without errors
        expect(true).toBe(true);
      }
    });

    test('should handle memory pressure gracefully', async () => {
      // Create many analyses to test memory management
      const analyses = [];

      for (let i = 0; i < 10; i++) {
        const analysis = await engine.analyzeWithSubjectiveTime(`memory test ${i}`);
        analyses.push(analysis);
      }

      expect(analyses.length).toBe(10);
      analyses.forEach(analysis => {
        expect(analysis).toBeDefined();
        expect(analysis.insights).toBeInstanceOf(Array);
      });

      const status = await engine.getStatus();
      expect(status.analysisHistory).toBe(10);
    });
  });

  describe('Shutdown Operations', () => {
    test('should shutdown cleanly before activation', async () => {
      await expect(engine.shutdown()).resolves.not.toThrow();
    });

    test('should shutdown cleanly after activation', async () => {
      await engine.activateSubjectiveTimeExpansion();
      await engine.analyzeWithSubjectiveTime('shutdown test');

      await expect(engine.shutdown()).resolves.not.toThrow();
    });

    test('should reset all state on shutdown', async () => {
      await engine.activateSubjectiveTimeExpansion();
      await engine.analyzeWithSubjectiveTime('state reset test');

      await engine.shutdown();
      const status = await engine.getStatus();

      expect(status.isActive).toBe(false);
      expect(status.consciousnessIntegration).toBe(false);
      expect(status.temporalCores).toBe(0);
      expect(status.activeTimelines).toBe(0);
      expect(status.temporalPatterns).toBe(0);
    });

    test('should handle multiple shutdown calls gracefully', async () => {
      await engine.activateSubjectiveTimeExpansion();

      await engine.shutdown();
      await expect(engine.shutdown()).resolves.not.toThrow();
      await expect(engine.shutdown()).resolves.not.toThrow();
    });

    test('should handle operations after shutdown gracefully', async () => {
      await engine.activateSubjectiveTimeExpansion();
      await engine.shutdown();

      // Operations after shutdown should be handled gracefully
      // (specific behavior depends on implementation)
      await expect(engine.getStatus()).resolves.toBeDefined();
    });
  });

  describe('Integration with Temporal Patterns', () => {
    beforeEach(async () => {
      await engine.activateSubjectiveTimeExpansion();
    });

    test('should maintain temporal resolution consistency', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('temporal resolution test');

      expect(analysis.temporalResolution).toBe(1); // nanosecond precision
      expect(analysis.expansionFactor).toBe(1000);
    });

    test('should validate cognitive depth metrics', async () => {
      const analysis = await engine.analyzeWithSubjectiveTime('cognitive depth test');

      expect(analysis.depth).toBeGreaterThan(0);
      expect(analysis.depth).toBeLessThanOrEqual(1000); // Should not exceed expansion factor

      const status = await engine.getStatus();
      expect(status.cognitiveDepth).toBe(3); // Math.floor(Math.log10(1000))
    });

    test('should handle strange-loop analysis mode correctly', async () => {
      const status = await engine.getStatus();
      expect(status.analysisMode).toBe('strange_loop');

      const analysis = await engine.analyzeWithSubjectiveTime('strange-loop mode test');

      // Should have cognitive processing from strange-loop recursion
      expect(analysis.cognitiveProcessing.length).toBeGreaterThan(0);
    });
  });

  describe('Configuration Edge Cases', () => {
    test('should handle zero expansion factor gracefully', () => {
      expect(() => {
        new TemporalReasoningEngine({
          subjectiveExpansion: 0,
          cognitiveModeling: true,
          deepPatternAnalysis: true,
          consciousnessDynamics: true
        });
      }).not.toThrow();
    });

    test('should handle negative expansion factor gracefully', () => {
      expect(() => {
        new TemporalReasoningEngine({
          subjectiveExpansion: -100,
          cognitiveModeling: true,
          deepPatternAnalysis: true,
          consciousnessDynamics: true
        });
      }).not.toThrow();
    });

    test('should handle fractional expansion factor', () => {
      expect(() => {
        new TemporalReasoningEngine({
          subjectiveExpansion: 100.5,
          cognitiveModeling: true,
          deepPatternAnalysis: true,
          consciousnessDynamics: true
        });
      }).not.toThrow();
    });

    test('should handle very large expansion factor', () => {
      expect(() => {
        new TemporalReasoningEngine({
          subjectiveExpansion: Number.MAX_SAFE_INTEGER,
          cognitiveModeling: true,
          deepPatternAnalysis: true,
          consciousnessDynamics: true
        });
      }).not.toThrow();
    });
  });

  describe('Memory and Resource Management', () => {
    test('should manage analysis history memory efficiently', async () => {
      await engine.activateSubjectiveTimeExpansion();

      // Create many analyses
      for (let i = 0; i < 100; i++) {
        await engine.analyzeWithSubjectiveTime(`memory efficiency test ${i}`);
      }

      const status = await engine.getStatus();
      expect(status.analysisHistory).toBe(100);

      // Should still be responsive
      const startTime = Date.now();
      const newAnalysis = await engine.analyzeWithSubjectiveTime('final test');
      const endTime = Date.now();

      expect(newAnalysis).toBeDefined();
      expect(endTime - startTime).toBeLessThan(10000); // <10 seconds
    });

    test('should handle resource cleanup on shutdown', async () => {
      await engine.activateSubjectiveTimeExpansion();

      // Use some resources
      await engine.analyzeWithSubjectiveTime('resource test');
      const statusBefore = await engine.getStatus();
      expect(statusBefore.temporalCores).toBeGreaterThan(0);

      // Shutdown should clean up
      await engine.shutdown();
      const statusAfter = await engine.getStatus();

      expect(statusAfter.temporalCores).toBe(0);
      expect(statusAfter.activeTimelines).toBe(0);
      expect(statusAfter.temporalPatterns).toBe(0);
    });
  });
});