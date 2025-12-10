/**
 * Comprehensive Unit Tests for Temporal Reasoning Core
 * Tests 1000x subjective time expansion and temporal pattern analysis
 */

import { TemporalReasoningCore, TemporalPattern, TemporalState } from '../../src/closed-loop/temporal-reasoning';

describe('TemporalReasoningCore', () => {
  let temporalReasoning: TemporalReasoningCore;

  beforeEach(() => {
    temporalReasoning = new TemporalReasoningCore();
  });

  afterEach(async () => {
    if (temporalReasoning) {
      await temporalReasoning.shutdown();
    }
  });

  describe('Initialization and Configuration', () => {
    test('should initialize with default configuration', () => {
      const state = temporalReasoning.getCurrentState();

      expect(state.expansionFactor).toBe(1);
      expect(state.reasoningDepth).toBe(10);
      expect(state.patterns).toEqual([]);
      expect(state.subjectTime).toBe(0);
    });

    test('should initialize temporal reasoning successfully', async () => {
      await expect(temporalReasoning.initialize()).resolves.not.toThrow();

      const state = temporalReasoning.getCurrentState();
      expect(state.reasoningDepth).toBe(10);
    });

    test('should handle shutdown gracefully', async () => {
      await temporalReasoning.initialize();
      await expect(temporalReasoning.shutdown()).resolves.not.toThrow();

      const state = temporalReasoning.getCurrentState();
      expect(state.patterns).toEqual([]);
      expect(state.expansionFactor).toBe(1);
    });
  });

  describe('Subjective Time Expansion', () => {
    test('should expand subjective time with default factor', async () => {
      const testData = {
        timestamp: Date.now(),
        value: 100,
        kpis: { energy: 85, mobility: 92 }
      };

      const result = await temporalReasoning.expandSubjectiveTime(testData);

      expect(result.expansionFactor).toBe(1000);
      expect(result.analysisDepth).toBe('deep');
      expect(result.patterns).toBeDefined();
      expect(result.insights).toBeDefined();
      expect(result.predictions).toBeDefined();
      expect(result.confidence).toBe(0.95);
      expect(result.accuracy).toBe(0.9);
    });

    test('should expand subjective time with custom factor', async () => {
      const testData = { value: 50 };
      const customFactor = 500;

      const result = await temporalReasoning.expandSubjectiveTime(testData, {
        expansionFactor: customFactor
      });

      expect(result.expansionFactor).toBe(customFactor);
    });

    test('should limit expansion factor to maximum', async () => {
      const testData = { value: 75 };
      const excessiveFactor = 2000;

      const result = await temporalReasoning.expandSubjectiveTime(testData, {
        expansionFactor: excessiveFactor
      });

      expect(result.expansionFactor).toBe(1000); // Should be limited to max
    });

    test('should update state during expansion', async () => {
      const testData = { timestamp: Date.now(), value: 200 };
      const initialTime = Date.now();

      await temporalReasoning.expandSubjectiveTime(testData, {
        expansionFactor: 100
      });

      const state = temporalReasoning.getCurrentState();
      expect(state.expansionFactor).toBe(100);
      expect(state.subjectTime).toBeGreaterThan(initialTime);
      expect(state.reasoningDepth).toBeGreaterThan(10);
    });

    test('should handle custom reasoning depth', async () => {
      const testData = { value: 150 };

      const result = await temporalReasoning.expandSubjectiveTime(testData, {
        reasoningDepth: 'comprehensive'
      });

      expect(result.analysisDepth).toBe('comprehensive');
    });

    test('should handle null and undefined data gracefully', async () => {
      const result1 = await temporalReasoning.expandSubjectiveTime(null);
      expect(result1).toBeDefined();
      expect(result1.patterns).toBeDefined();

      const result2 = await temporalReasoning.expandSubjectiveTime(undefined);
      expect(result2).toBeDefined();
      expect(result2.patterns).toBeDefined();
    });
  });

  describe('Temporal Pattern Analysis', () => {
    test('should analyze temporal patterns from data array', () => {
      const testData = [
        { timestamp: Date.now(), value: 100, anomaly: true },
        { timestamp: Date.now() + 1000, value: 150, optimize: true },
        { timestamp: Date.now() + 2000, value: 75 }
      ];

      const patterns = temporalReasoning.analyzeTemporalPatterns(testData);

      expect(patterns).toHaveLength(3);
      expect(patterns[0]).toHaveProperty('id');
      expect(patterns[0]).toHaveProperty('pattern');
      expect(patterns[0]).toHaveProperty('conditions');
      expect(patterns[0]).toHaveProperty('actions');
      expect(patterns[0]).toHaveProperty('effectiveness');
      expect(patterns[0]).toHaveProperty('createdAt');
      expect(patterns[0]).toHaveProperty('applicationCount');
    });

    test('should generate appropriate patterns for temporal data', () => {
      const temporalData = {
        timestamp: Date.now(),
        value: 200
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([temporalData]);

      expect(patterns[0].pattern).toContain('Temporal spike detected');
      expect(patterns[0].conditions).toContain('High value threshold');
      expect(patterns[0].conditions).toContain('Valid timestamp');
    });

    test('should extract conditions from temporal data', () => {
      const highValueData = {
        timestamp: Date.now(),
        value: 150
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([highValueData]);

      expect(patterns[0].conditions).toContain('High value threshold');
      expect(patterns[0].conditions).toContain('Valid timestamp');
    });

    test('should extract actions from temporal data', () => {
      const anomalyData = {
        timestamp: Date.now(),
        value: 100,
        anomaly: true,
        optimize: true
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([anomalyData]);

      expect(patterns[0].actions).toContain('Trigger anomaly alert');
      expect(patterns[0].actions).toContain('Apply optimization');
    });

    test('should handle empty data array', () => {
      const patterns = temporalReasoning.analyzeTemporalPatterns([]);
      expect(patterns).toEqual([]);
    });

    test('should update state with new patterns', () => {
      const testData = [{ timestamp: Date.now(), value: 100 }];

      temporalReasoning.analyzeTemporalPatterns(testData);

      const state = temporalReasoning.getCurrentState();
      expect(state.patterns.length).toBeGreaterThan(0);
    });
  });

  describe('Temporal Insights Generation', () => {
    test('should generate insights from temporal data', async () => {
      const testData = {
        timestamp: Date.now(),
        value: 120,
        kpis: {
          energy: 90,
          mobility: 85,
          coverage: 95
        }
      };

      const result = await temporalReasoning.expandSubjectiveTime(testData);

      expect(result.insights.length).toBeGreaterThan(0);

      const temporalInsight = result.insights.find((insight: any) => insight.type === 'temporal_pattern');
      expect(temporalInsight).toBeDefined();
      expect(temporalInsight.description).toContain('Value trend detected');
      expect(temporalInsight.confidence).toBe(0.85);
      expect(temporalInsight.actionable).toBe(true);
    });

    test('should generate performance insights from KPIs', async () => {
      const kpiData = {
        timestamp: Date.now(),
        value: 100,
        kpis: {
          energy: 95,
          mobility: 92,
          coverage: 88
        }
      };

      const result = await temporalReasoning.expandSubjectiveTime(kpiData);

      const performanceInsights = result.insights.filter((insight: any) => insight.type === 'high_performance');
      expect(performanceInsights.length).toBeGreaterThan(0);

      performanceInsights.forEach((insight: any) => {
        expect(insight.description).toContain('is performing well');
        expect(insight.confidence).toBe(0.9);
        expect(insight.actionable).toBe(false);
      });
    });

    test('should handle data without KPIs', async () => {
      const noKpiData = {
        timestamp: Date.now(),
        value: 75
      };

      const result = await temporalReasoning.expandSubjectiveTime(noKpiData);

      const performanceInsights = result.insights.filter((insight: any) => insight.type === 'high_performance');
      expect(performanceInsights.length).toBe(0);

      const temporalInsights = result.insights.filter((insight: any) => insight.type === 'temporal_pattern');
      expect(temporalInsights.length).toBe(1);
    });
  });

  describe('Temporal Predictions Generation', () => {
    test('should generate predictions from KPI data', async () => {
      const kpiData = {
        timestamp: Date.now(),
        value: 100,
        kpis: {
          energy: 85,
          mobility: 90,
          coverage: 88
        }
      };

      const result = await temporalReasoning.expandSubjectiveTime(kpiData);

      expect(result.predictions.length).toBe(3);

      result.predictions.forEach((prediction: any) => {
        expect(prediction).toHaveProperty('metric');
        expect(prediction).toHaveProperty('value');
        expect(prediction).toHaveProperty('timeHorizon');
        expect(prediction).toHaveProperty('confidence');

        expect(prediction.value).toBeGreaterThan(0);
        expect(prediction.timeHorizon).toBe(3600000); // 1 hour
        expect(prediction.confidence).toBe(0.75);
      });
    });

    test('should predict improvement over current values', async () => {
      const kpiData = {
        kpis: {
          energy: 80,
          mobility: 85
        }
      };

      const result = await temporalReasoning.expandSubjectiveTime(kpiData);

      result.predictions.forEach((prediction: any) => {
        const originalValue = kpiData.kpis[prediction.metric];
        expect(prediction.value).toBeGreaterThan(originalValue);

        // Should be 5% improvement
        const expectedValue = originalValue * 1.05;
        expect(prediction.value).toBeCloseTo(expectedValue, 2);
      });
    });

    test('should handle data without KPIs', async () => {
      const noKpiData = {
        timestamp: Date.now(),
        value: 100
      };

      const result = await temporalReasoning.expandSubjectiveTime(noKpiData);
      expect(result.predictions).toEqual([]);
    });
  });

  describe('State Management', () => {
    test('should get current temporal state', () => {
      const state = temporalReasoning.getCurrentState();

      expect(state).toHaveProperty('timestamp');
      expect(state).toHaveProperty('subjectTime');
      expect(state).toHaveProperty('expansionFactor');
      expect(state).toHaveProperty('patterns');
      expect(state).toHaveProperty('reasoningDepth');

      expect(typeof state.timestamp).toBe('number');
      expect(typeof state.subjectTime).toBe('number');
      expect(typeof state.expansionFactor).toBe('number');
      expect(Array.isArray(state.patterns)).toBe(true);
      expect(typeof state.reasoningDepth).toBe('number');
    });

    test('should update temporal state', () => {
      const newState = {
        expansionFactor: 500,
        reasoningDepth: 15,
        subjectTime: Date.now() * 500
      };

      temporalReasoning.updateState(newState);

      const updatedState = temporalReasoning.getCurrentState();
      expect(updatedState.expansionFactor).toBe(500);
      expect(updatedState.reasoningDepth).toBe(15);
      expect(updatedState.subjectTime).toBe(newState.subjectTime);
    });

    test('should maintain state consistency', async () => {
      const initialSubjectTime = temporalReasoning.getCurrentState().subjectTime;

      await temporalReasoning.expandSubjectiveTime({ value: 100 }, {
        expansionFactor: 200
      });

      const updatedState = temporalReasoning.getCurrentState();
      expect(updatedState.subjectTime).toBeGreaterThan(initialSubjectTime);
      expect(updatedState.expansionFactor).toBe(200);
    });

    test('should handle partial state updates', () => {
      const initialState = temporalReasoning.getCurrentState();

      temporalReasoning.updateState({
        expansionFactor: 750
      });

      const updatedState = temporalReasoning.getCurrentState();
      expect(updatedState.expansionFactor).toBe(750);
      expect(updatedState.reasoningDepth).toBe(initialState.reasoningDepth);
      expect(updatedState.timestamp).toBe(initialState.timestamp);
    });
  });

  describe('Pattern Recognition', () => {
    test('should recognize temporal spike patterns', () => {
      const spikeData = {
        timestamp: Date.now(),
        value: 250 // High value spike
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([spikeData]);

      expect(patterns[0].pattern).toContain('spike detected');
      expect(patterns[0].effectiveness).toBeGreaterThan(0);
    });

    test('should recognize optimization opportunities', () => {
      const optimizationData = {
        timestamp: Date.now(),
        value: 80,
        optimize: true
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([optimizationData]);

      expect(patterns[0].actions).toContain('Apply optimization');
    });

    test('should recognize anomaly patterns', () => {
      const anomalyData = {
        timestamp: Date.now(),
        value: 60,
        anomaly: true
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([anomalyData]);

      expect(patterns[0].actions).toContain('Trigger anomaly alert');
    });

    test('should generate unique pattern IDs', () => {
      const data1 = { timestamp: Date.now(), value: 100 };
      const data2 = { timestamp: Date.now() + 1, value: 150 };

      const patterns1 = temporalReasoning.analyzeTemporalPatterns([data1]);
      const patterns2 = temporalReasoning.analyzeTemporalPatterns([data2]);

      expect(patterns1[0].id).not.toBe(patterns2[0].id);
    });
  });

  describe('Performance and Scalability', () => {
    test('should handle large datasets efficiently', async () => {
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        timestamp: Date.now() + i * 1000,
        value: Math.random() * 200,
        kpis: {
          energy: 70 + Math.random() * 30,
          mobility: 80 + Math.random() * 20
        }
      }));

      const startTime = Date.now();
      const patterns = temporalReasoning.analyzeTemporalPatterns(largeDataset);
      const endTime = Date.now();

      expect(patterns).toHaveLength(1000);
      expect(endTime - startTime).toBeLessThan(1000); // Should complete within 1 second
    });

    test('should handle concurrent expansions', async () => {
      const testData = { value: 100, kpis: { energy: 85 } };
      const concurrentTasks = Array.from({ length: 10 }, (_, i) =>
        temporalReasoning.expandSubjectiveTime(testData, { expansionFactor: 100 + i * 100 })
      );

      const results = await Promise.all(concurrentTasks);

      expect(results).toHaveLength(10);
      results.forEach((result, index) => {
        expect(result.expansionFactor).toBe(100 + index * 100);
        expect(result.insights).toBeDefined();
        expect(result.predictions).toBeDefined();
      });
    });

    test('should maintain accuracy with high expansion factors', async () => {
      const testData = {
        timestamp: Date.now(),
        value: 150,
        kpis: { energy: 90, mobility: 88 }
      };

      const result = await temporalReasoning.expandSubjectiveTime(testData, {
        expansionFactor: 1000
      });

      expect(result.accuracy).toBe(0.9);
      expect(result.confidence).toBe(0.95);
    });
  });

  describe('Edge Cases and Error Handling', () => {
    test('should handle invalid expansion factors', async () => {
      const testData = { value: 100 };

      const result1 = await temporalReasoning.expandSubjectiveTime(testData, {
        expansionFactor: -100
      });
      expect(result1.expansionFactor).toBe(1000); // Should default to max

      const result2 = await temporalReasoning.expandSubjectiveTime(testData, {
        expansionFactor: 0
      });
      expect(result2.expansionFactor).toBe(0);
    });

    test('should handle extremely large values', async () => {
      const extremeData = {
        timestamp: Date.now(),
        value: Number.MAX_SAFE_INTEGER
      };

      const result = await temporalReasoning.expandSubjectiveTime(extremeData);
      expect(result).toBeDefined();
      expect(result.patterns.length).toBeGreaterThan(0);
    });

    test('should handle circular object references', async () => {
      const circularData: any = { value: 100 };
      circularData.self = circularData;

      const result = await temporalReasoning.expandSubjectiveTime(circularData);
      expect(result).toBeDefined();
    });

    test('should handle malformed timestamp data', async () => {
      const malformedData = {
        timestamp: 'invalid',
        value: 100
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([malformedData]);
      expect(patterns).toHaveLength(1);
      expect(patterns[0].conditions).not.toContain('Valid timestamp');
    });

    test('should handle negative values', async () => {
      const negativeData = {
        timestamp: Date.now(),
        value: -50
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([negativeData]);
      expect(patterns).toHaveLength(1);
      expect(patterns[0].conditions).not.toContain('High value threshold');
    });

    test('should handle zero values', async () => {
      const zeroData = {
        timestamp: Date.now(),
        value: 0
      };

      const patterns = temporalReasoning.analyzeTemporalPatterns([zeroData]);
      expect(patterns).toHaveLength(1);
    });
  });

  describe('Integration with Other Components', () => {
    test('should integrate with optimization workflow', async () => {
      const optimizationData = {
        timestamp: Date.now(),
        value: 120,
        kpis: {
          energy: 85,
          mobility: 90,
          coverage: 88,
          capacity: 75
        },
        optimize: true,
        anomaly: false
      };

      const result = await temporalReasoning.expandSubjectiveTime(optimizationData, {
        expansionFactor: 1000,
        reasoningDepth: 'deep'
      });

      expect(result.patterns.length).toBeGreaterThan(0);
      expect(result.insights.length).toBeGreaterThan(0);
      expect(result.predictions.length).toBeGreaterThan(0);

      // Verify optimization-related insights
      const optimizationActions = result.patterns[0].actions;
      expect(optimizationActions).toContain('Apply optimization');
    });

    test('should support temporal reasoning for anomaly detection', async () => {
      const anomalyData = {
        timestamp: Date.now(),
        value: 30, // Low value indicating potential anomaly
        kpis: {
          energy: 45, // Low performance
          mobility: 60
        },
        anomaly: true
      };

      const result = await temporalReasoning.expandSubjectiveTime(anomalyData);

      expect(result.patterns[0].actions).toContain('Trigger anomaly alert');

      // Should generate insights about low performance
      const lowPerformanceInsights = result.insights.filter((insight: any) =>
        insight.description.includes('performing well')
      );
      expect(lowPerformanceInsights.length).toBe(0);
    });
  });

  describe('Memory and Resource Management', () => {
    test('should manage pattern storage efficiently', () => {
      const initialPatterns = temporalReasoning.getCurrentState().patterns.length;

      // Add many patterns
      for (let i = 0; i < 100; i++) {
        temporalReasoning.analyzeTemporalPatterns([{
          timestamp: Date.now() + i,
          value: Math.random() * 200
        }]);
      }

      const finalPatterns = temporalReasoning.getCurrentState().patterns.length;
      expect(finalPatterns).toBe(initialPatterns + 100);
    });

    test('should clear patterns on shutdown', async () => {
      // Add some patterns
      temporalReasoning.analyzeTemporalPatterns([{
        timestamp: Date.now(),
        value: 100
      }]);

      expect(temporalReasoning.getCurrentState().patterns.length).toBeGreaterThan(0);

      await temporalReasoning.shutdown();

      expect(temporalReasoning.getCurrentState().patterns.length).toBe(0);
    });

    test('should handle memory pressure gracefully', async () => {
      const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
        timestamp: Date.now() + i,
        value: Math.random() * 1000,
        kpis: {
          energy: Math.random() * 100,
          mobility: Math.random() * 100,
          coverage: Math.random() * 100,
          capacity: Math.random() * 100,
          additional_kpi: Array.from({ length: 10 }, (_, j) => Math.random() * 100)
        }
      }));

      // Should not cause memory issues
      await expect(temporalReasoning.expandSubjectiveTime(largeDataset)).resolves.toBeDefined();
    });
  });

  describe('Temporal Reasoning Accuracy', () => {
    test('should maintain high accuracy for temporal predictions', async () => {
      const historicalData = {
        timestamp: Date.now(),
        value: 100,
        kpis: {
          energy: 85,
          mobility: 92,
          coverage: 88
        }
      };

      const result = await temporalReasoning.expandSubjectiveTime(historicalData);

      expect(result.accuracy).toBeGreaterThanOrEqual(0.9);
      expect(result.confidence).toBeGreaterThanOrEqual(0.95);
    });

    test('should provide confidence scoring for insights', async () => {
      const testData = {
        timestamp: Date.now(),
        value: 150,
        kpis: { energy: 95 }
      };

      const result = await temporalReasoning.expandSubjectiveTime(testData);

      result.insights.forEach((insight: any) => {
        expect(insight.confidence).toBeGreaterThan(0);
        expect(insight.confidence).toBeLessThanOrEqual(1);
        expect(typeof insight.actionable).toBe('boolean');
      });
    });

    test('should validate temporal analysis completeness', async () => {
      const completeData = {
        timestamp: Date.now(),
        value: 120,
        kpis: {
          energy: 88,
          mobility: 91,
          coverage: 89,
          capacity: 85
        }
      };

      const result = await temporalReasoning.expandSubjectiveTime(completeData);

      expect(result.patterns.length).toBeGreaterThan(0);
      expect(result.insights.length).toBeGreaterThan(0);
      expect(result.predictions.length).toBeGreaterThan(0);
      expect(result.expansionFactor).toBeGreaterThan(0);
      expect(result.accuracy).toBeGreaterThan(0);
      expect(result.confidence).toBeGreaterThan(0);
    });
  });
});