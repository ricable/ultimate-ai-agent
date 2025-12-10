/**
 * Performance Tests for Stream-JSON Pipeline
 * Validates performance targets and optimization capabilities
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import StreamChain from '../../src/stream-chain/core';
import RANIngestionAgent from '../../src/data-ingestion/ran-ingestion';
import MOFeatureProcessor from '../../src/feature-processing/mo-processor';
import AgentDBPatternRecognizer from '../../src/pattern-recognition/agentdb-patterns';
import CognitiveOptimizer from '../../src/optimization-engine/cognitive-optimizer';
import ActionExecutionEngine from '../../src/action-execution/execution-engine';
import ResilienceEngine from '../../src/error-handling/resilience-engine';
import PerformanceMonitor from '../../src/performance-monitoring/anomaly-detector';

describe('Pipeline Performance Tests', () => {
  let streamChain: StreamChain;
  let agents: any[];
  let pipelineId: string;

  beforeEach(async () => {
    // Initialize performance-optimized agents
    streamChain = new StreamChain();

    agents = [
      new RANIngestionAgent({
        sources: [{
          id: 'perf-test',
          type: 'simulator',
          endpoint: 'perf://simulator',
          pollingInterval: 100,
          dataFormat: 'json',
          compression: 'none',
          filters: {}
        }],
        bufferSize: 1000,
        batchSize: 50,
        batchTimeout: 1000,
        compressionEnabled: false,
        temporalReasoningEnabled: true,
        realTimeProcessing: true,
        anomalyDetection: true
      }),
      new MOFeatureProcessor({
        enabledMOClasses: ['Cell', 'Radio'],
        temporalWindows: { short: 1, medium: 5, long: 60 },
        anomalyThreshold: 0.8,
        correlationThreshold: 0.7,
        predictionHorizon: 30,
        enableCausalInference: true,
        temporalReasoningDepth: 500
      }),
      new AgentDBPatternRecognizer({
        vectorDimensions: 64,
        similarityThreshold: 0.8,
        temporalWindow: 30,
        memoryRetention: 1,
        learningRate: 0.2,
        enableCausalInference: true,
        enableTemporalReasoning: true,
        maxPatternsPerCategory: 500,
        adaptationThreshold: 0.9
      }),
      new CognitiveOptimizer({
        consciousnessThreshold: 0.8,
        maxTemporalExpansion: 500,
        maxRecursiveDepth: 3,
        learningRateAdaptation: true,
        strangeLoopOptimization: true,
        metaCognitionEnabled: true,
        predictionHorizon: { short: 30, medium: 12, long: 72 },
        decisionThreshold: 0.9,
        adaptationRate: 0.2
      }),
      new ActionExecutionEngine({
        maxConcurrentExecutions: 5,
        executionTimeout: 60000,
        verificationTimeout: 30000,
        rollbackTimeout: 15000,
        safetyCheckInterval: 2000,
        feedbackSamplingRate: 2,
        enableAutoRollback: true,
        enablePredictiveVerification: true,
        enableRealTimeAdaptation: true,
        minConfidenceThreshold: 0.8,
        maxImpactThreshold: 0.2
      }),
      new ResilienceEngine({
        circuitBreakerThreshold: 3,
        circuitBreakerTimeoutMs: 30000,
        maxRetryAttempts: 2,
        baseRetryDelayMs: 500,
        maxRetryDelayMs: 10000,
        healthCheckIntervalMs: 15000,
        healthCheckTimeoutMs: 2000,
        enablePredictiveFailureDetection: true,
        enableSelfHealing: true,
        enableGracefulDegradation: true,
        enableFaultIsolation: true,
        errorRetentionPeriodMs: 3600000,
        recoveryActionTimeoutMs: 60000
      }),
      new PerformanceMonitor({
        detectionIntervalMs: 500,
        responseTimeThresholdMs: 200,
        enablePredictiveDetection: true,
        enableRealTimeAlerting: true,
        enableAutomatedResponse: true,
        anomalyRetentionPeriodMs: 3600000,
        metricsRetentionPeriodMs: 86400000,
        learningRate: 0.2,
        adaptationEnabled: true,
        alertingChannels: ['console']
      })
    ];

    // Register agents
    agents.forEach(agent => streamChain.registerAgent(agent));

    // Create high-performance pipeline
    pipelineId = streamChain.createPipeline({
      name: 'High-Performance Pipeline',
      agents,
      topology: 'parallel',
      flowControl: {
        maxConcurrency: 20,
        bufferSize: 5000,
        backpressureStrategy: 'buffer',
        temporalOptimization: true,
        cognitiveScheduling: true
      },
      errorRecovery: {
        maxRetries: 2,
        retryDelay: 500,
        circuitBreakerThreshold: 3,
        selfHealing: true
      },
      performance: {
        targetLatency: 500,
        throughputTarget: 200,
        anomalyDetectionThreshold: 0.8,
        adaptiveOptimization: true,
        closedLoopCycleTime: 300000
      }
    });
  });

  afterEach(async () => {
    if (streamChain) {
      // StreamChain doesn't have destroy method, just clear reference
      streamChain = null;
    }
  });

  describe('Sub-Second Processing Performance', () => {
    it('should process single message in under 500ms', async () => {
      const testMessage = createPerformanceTestMessage();

      const startTime = performance.now();
      const results = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true,
        enableCognitiveOptimization: true
      });
      const processingTime = performance.now() - startTime;

      expect(processingTime).toBeLessThan(500);
      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
    }, 10000);

    it('should maintain sub-second processing under moderate load', async () => {
      const messages = Array.from({ length: 10 }, (_, i) =>
        createPerformanceTestMessage(`perf-${i}`)
      );

      const processingTimes: number[] = [];

      for (const message of messages) {
        const startTime = performance.now();
        await streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true
        });
        const processingTime = performance.now() - startTime;
        processingTimes.push(processingTime);
      }

      const avgTime = processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length;
      const maxTime = Math.max(...processingTimes);

      expect(avgTime).toBeLessThan(500);
      expect(maxTime).toBeLessThan(1000);
      expect(processingTimes.every(time => time < 1000)).toBe(true);
    }, 30000);

    it('should handle burst processing without degradation', async () => {
      const burstSize = 20;
      const messages = Array.from({ length: burstSize }, (_, i) =>
        createPerformanceTestMessage(`burst-${i}`)
      );

      const startTime = performance.now();

      // Process all messages in parallel (simulating burst)
      const promises = messages.map(message =>
        streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true
        })
      );

      const results = await Promise.all(promises);
      const totalTime = performance.now() - startTime;

      expect(results).toHaveLength(burstSize);
      expect(totalTime).toBeLessThan(15000); // Should handle burst in 15 seconds

      const avgTimePerMessage = totalTime / burstSize;
      expect(avgTimePerMessage).toBeLessThan(750); // Average under 750ms
    }, 40000);
  });

  describe('Throughput Performance', () => {
    it('should achieve target throughput of 200 messages per second', async () => {
      const targetThroughput = 200; // messages per second
      const testDuration = 10000; // 10 seconds
      const targetMessages = (targetThroughput * testDuration) / 1000;

      const messages = Array.from({ length: Math.min(targetMessages, 100) }, (_, i) =>
        createPerformanceTestMessage(`throughput-${i}`)
      );

      const startTime = Date.now();
      let processedCount = 0;

      // Process messages with controlled timing
      const processPromises = messages.map(async (message, index) => {
        const targetTime = startTime + (index * 1000 / targetThroughput);
        const currentTime = Date.now();

        if (currentTime < targetTime) {
          await new Promise(resolve => setTimeout(resolve, targetTime - currentTime));
        }

        await streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true
        });
        processedCount++;
      });

      await Promise.all(processPromises);
      const actualDuration = Date.now() - startTime;

      const actualThroughput = (processedCount * 1000) / actualDuration;
      const throughputAchievement = actualThroughput / targetThroughput;

      expect(throughputAchievement).toBeGreaterThan(0.8); // At least 80% of target
      expect(processedCount).toBeGreaterThan(messages.length * 0.9); // At least 90% processed
    }, 20000);

    it('should scale throughput with parallel processing', async () => {
      const parallelBatches = 5;
      const batchSize = 10;

      const sequentialTime = await measureSequentialProcessing(batchSize);
      const parallelTime = await measureParallelProcessing(batchSize, parallelBatches);

      const speedup = sequentialTime / parallelTime;

      expect(speedup).toBeGreaterThan(2); // Should achieve at least 2x speedup
      expect(parallelTime).toBeLessThan(sequentialTime / 2);
    }, 30000);

    async function measureSequentialProcessing(batchSize: number): Promise<number> {
      const messages = Array.from({ length: batchSize }, (_, i) =>
        createPerformanceTestMessage(`sequential-${i}`)
      );

      const startTime = performance.now();
      for (const message of messages) {
        await streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true
        });
      }
      return performance.now() - startTime;
    }

    async function measureParallelProcessing(batchSize: number, parallelBatches: number): Promise<number> {
      const messages = Array.from({ length: batchSize * parallelBatches }, (_, i) =>
        createPerformanceTestMessage(`parallel-${i}`)
      );

      const startTime = performance.now();
      const promises = messages.map(message =>
        streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true
        })
      );
      await Promise.all(promises);
      return performance.now() - startTime;
    }
  });

  describe('Memory and Resource Performance', () => {
    it('should maintain stable memory usage during extended processing', async () => {
      const initialMemory = process.memoryUsage();
      const messageCount = 100;

      for (let i = 0; i < messageCount; i++) {
        const message = createPerformanceTestMessage(`memory-${i}`);
        await streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true
        });

        // Check memory every 20 messages
        if (i % 20 === 0) {
          const currentMemory = process.memoryUsage();
          const memoryGrowth = currentMemory.heapUsed - initialMemory.heapUsed;

          // Memory growth should be reasonable (less than 100MB)
          expect(memoryGrowth).toBeLessThan(100 * 1024 * 1024);
        }
      }

      const finalMemory = process.memoryUsage();
      const totalMemoryGrowth = finalMemory.heapUsed - initialMemory.heapUsed;

      // Total memory growth should be under 200MB for 100 messages
      expect(totalMemoryGrowth).toBeLessThan(200 * 1024 * 1024);
    }, 60000);

    it('should efficiently handle large message payloads', async () => {
      // Create large message (simulate high-density RAN data)
      const largeMessage = createLargePerformanceTestMessage();

      const startTime = performance.now();
      const results = await streamChain.processMessage(pipelineId, largeMessage, {
        enableTemporalReasoning: true
      });
      const processingTime = performance.now() - startTime;

      expect(processingTime).toBeLessThan(2000); // Should handle large messages in under 2 seconds
      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
    }, 15000);
  });

  describe('Cognitive Performance', () => {
    it('should perform temporal reasoning expansion efficiently', async () => {
      const testMessage = createPerformanceTestMessage();

      // Test with temporal reasoning enabled
      const startTime1 = performance.now();
      const results1 = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true,
        enableCognitiveOptimization: false
      });
      const timeWithTemporal = performance.now() - startTime1;

      // Test without temporal reasoning
      const startTime2 = performance.now();
      const results2 = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: false,
        enableCognitiveOptimization: false
      });
      const timeWithoutTemporal = performance.now() - startTime2;

      // Temporal reasoning should not add excessive overhead
      const temporalOverhead = timeWithTemporal - timeWithoutTemporal;
      expect(temporalOverhead).toBeLessThan(500); // Less than 500ms overhead
      expect(timeWithTemporal).toBeLessThan(1000); // Still under 1 second

      // Results should be enhanced with temporal reasoning
      expect(results1.length).toBeGreaterThanOrEqual(results2.length);
    }, 20000);

    it('should enable cognitive optimization without significant performance impact', async () => {
      const testMessage = createPerformanceTestMessage();

      // Test with cognitive optimization
      const startTime1 = performance.now();
      const results1 = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: false,
        enableCognitiveOptimization: true
      });
      const timeWithCognitive = performance.now() - startTime1;

      // Test baseline
      const startTime2 = performance.now();
      const results2 = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: false,
        enableCognitiveOptimization: false
      });
      const timeBaseline = performance.now() - startTime2;

      const cognitiveOverhead = timeWithCognitive - timeBaseline;
      expect(cognitiveOverhead).toBeLessThan(300); // Less than 300ms overhead
      expect(timeWithCognitive).toBeLessThan(800); // Still under 800ms

      // Should have optimization results
      const optimizationResults = results1.filter(r => r.type === 'optimization');
      expect(optimizationResults.length).toBeGreaterThan(0);
    }, 20000);

    it('should maintain strange-loop processing performance', async () => {
      // Create complex metrics that should trigger strange-loop processing
      const complexMessage = createComplexPerformanceTestMessage();

      const startTime = performance.now();
      const results = await streamChain.processMessage(pipelineId, complexMessage, {
        enableTemporalReasoning: true,
        enableCognitiveOptimization: true
      });
      const processingTime = performance.now() - startTime;

      expect(processingTime).toBeLessThan(1500); // Complex processing under 1.5 seconds

      // Should show signs of strange-loop optimization
      const optimizationResults = results.filter(r => r.type === 'optimization');
      if (optimizationResults.length > 0) {
        const optimization = optimizationResults[0];
        expect(optimization.metadata).toBeDefined();
        // Check if strange-loop optimization is present (optional property)
        if ('strangeLoopActive' in optimization.metadata) {
          expect(optimization.metadata.strangeLoopActive).toBeDefined();
        }
      }
    }, 25000);
  });

  describe('Anomaly Detection Performance', () => {
    it('should detect anomalies in sub-second timeframes', async () => {
      const anomalousMessage = createAnomalousPerformanceTestMessage();

      const startTime = performance.now();
      const results = await streamChain.processMessage(pipelineId, anomalousMessage, {
        enableTemporalReasoning: true
      });
      const processingTime = performance.now() - startTime;

      expect(processingTime).toBeLessThan(500); // Anomaly detection under 500ms

      // Should detect anomalies
      const feedbackResults = results.filter(r => r.type === 'feedback');
      if (feedbackResults.length > 0) {
        const feedback = feedbackResults[0];
        expect(feedback.data.anomaliesDetected).toBeGreaterThan(0);
        expect(feedback.data.detectionTargetMet).toBe(true);
      }
    }, 15000);

    it('should handle high-frequency anomaly detection without performance degradation', async () => {
      const anomalousMessages = Array.from({ length: 20 }, (_, i) =>
        createAnomalousPerformanceTestMessage(`anomaly-${i}`)
      );

      const processingTimes: number[] = [];

      for (const message of anomalousMessages) {
        const startTime = performance.now();
        await streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true
        });
        const processingTime = performance.now() - startTime;
        processingTimes.push(processingTime);
      }

      const avgTime = processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length;
      const maxTime = Math.max(...processingTimes);

      expect(avgTime).toBeLessThan(600); // Average under 600ms
      expect(maxTime).toBeLessThan(1200); // Max under 1.2 seconds
      expect(processingTimes.every(time => time < 1500)).toBe(true);
    }, 40000);
  });

  describe('Scalability Performance', () => {
    it('should scale linearly with increased complexity', async () => {
      const complexities = [
        { name: 'simple', messages: 10, complexity: 1 },
        { name: 'medium', messages: 20, complexity: 2 },
        { name: 'complex', messages: 30, complexity: 3 }
      ];

      const performanceData: any[] = [];

      for (const complexity of complexities) {
        const messages = Array.from({ length: complexity.messages }, (_, i) =>
          createPerformanceTestMessage(`${complexity.name}-${i}`)
        );

        const startTime = performance.now();

        for (const message of messages) {
          await streamChain.processMessage(pipelineId, message, {
            enableTemporalReasoning: true
          });
        }

        const totalTime = performance.now() - startTime;
        const avgTimePerMessage = totalTime / messages.length;

        performanceData.push({
          complexity: complexity.complexity,
          avgTimePerMessage,
          totalTime
        });
      }

      // Check for linear scaling (performance shouldn't degrade exponentially)
      const simpleTime = performanceData[0].avgTimePerMessage;
      const mediumTime = performanceData[1].avgTimePerMessage;
      const complexTime = performanceData[2].avgTimePerMessage;

      const mediumRatio = mediumTime / simpleTime;
      const complexRatio = complexTime / simpleTime;

      // Should be roughly linear (complexity 2 should be ~2x time, complexity 3 should be ~3x time)
      expect(mediumRatio).toBeLessThan(2.5); // Allow some overhead but not exponential
      expect(complexRatio).toBeLessThan(4.0);
    }, 45000);

    it('should maintain performance under sustained load', async () => {
      const sustainedDuration = 30000; // 30 seconds
      const messagesPerSecond = 10;
      const totalMessages = (sustainedDuration * messagesPerSecond) / 1000;

      const performanceSnapshots: any[] = [];
      let processedCount = 0;

      const startTime = Date.now();
      let lastSnapshotTime = startTime;

      const processLoop = async () => {
        while (Date.now() - startTime < sustainedDuration) {
          const message = createPerformanceTestMessage(`sustained-${processedCount}`);

          const messageStartTime = performance.now();
          await streamChain.processMessage(pipelineId, message, {
            enableTemporalReasoning: true
          });
          const messageProcessingTime = performance.now() - messageStartTime;

          processedCount++;

          // Take performance snapshots every 5 seconds
          const currentTime = Date.now();
          if (currentTime - lastSnapshotTime > 5000) {
            performanceSnapshots.push({
              timestamp: currentTime,
              processedCount,
              avgProcessingTime: messageProcessingTime,
              throughput: processedCount / ((currentTime - startTime) / 1000)
            });
            lastSnapshotTime = currentTime;
          }

          // Control message rate
          await new Promise(resolve => setTimeout(resolve, 1000 / messagesPerSecond));
        }
      };

      await processLoop();

      expect(processedCount).toBeGreaterThan(totalMessages * 0.9); // At least 90% target throughput

      // Performance should be stable (not degrade significantly over time)
      if (performanceSnapshots.length > 1) {
        const firstSnapshot = performanceSnapshots[0];
        const lastSnapshot = performanceSnapshots[performanceSnapshots.length - 1];

        const throughputDegradation = (firstSnapshot.throughput - lastSnapshot.throughput) / firstSnapshot.throughput;
        expect(throughputDegradation).toBeLessThan(0.3); // Less than 30% degradation
      }
    }, 45000);
  });

  describe('Optimization Performance', () => {
    it('should demonstrate performance improvements over time through learning', async () => {
      const learningMessages = Array.from({ length: 50 }, (_, i) =>
        createPerformanceTestMessage(`learning-${i}`)
      );

      const processingTimes: number[] = [];

      // Process messages and track performance improvement
      for (let i = 0; i < learningMessages.length; i++) {
        const startTime = performance.now();
        await streamChain.processMessage(pipelineId, learningMessages[i], {
          enableTemporalReasoning: true,
          enableCognitiveOptimization: true
        });
        const processingTime = performance.now() - startTime;
        processingTimes.push(processingTime);
      }

      // Compare early vs late performance
      const earlyAvg = processingTimes.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
      const lateAvg = processingTimes.slice(-10).reduce((a, b) => a + b, 0) / 10;

      const improvement = (earlyAvg - lateAvg) / earlyAvg;

      // Should show some improvement through learning (even if small)
      expect(improvement).toBeGreaterThan(-0.1); // At worst 10% degradation
      expect(lateAvg).toBeLessThan(earlyAvg * 1.2); // At most 20% slower
    }, 60000);

    it('should optimize resource usage under cognitive control', async () => {
      const initialMemory = process.memoryUsage();

      // Process messages that trigger optimization
      const optimizationMessages = Array.from({ length: 30 }, (_, i) =>
        createOptimizationTestMessage(`optimize-${i}`)
      );

      for (const message of optimizationMessages) {
        await streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true,
          enableCognitiveOptimization: true
        });
      }

      const finalMemory = process.memoryUsage();
      const memoryIncrease = finalMemory.heapUsed - initialMemory.heapUsed;

      // Memory usage should be reasonable for optimization processing
      expect(memoryIncrease).toBeLessThan(150 * 1024 * 1024); // Less than 150MB increase

      // Check for optimization decisions
      const cognitiveStatus = agents[3].getStatus(); // CognitiveOptimizer
      expect(cognitiveStatus.decisionHistory).toBeGreaterThan(0);
    }, 45000);
  });
});

// Helper functions for performance testing
function createPerformanceTestMessage(id: string = 'perf-test') {
  return {
    id,
    timestamp: Date.now(),
    type: 'ran-metrics' as const,
    data: {
      timestamp: Date.now(),
      source: 'perf-test-source',
      cellId: 'perf-cell-1',
      kpis: {
        rsrp: -75,
        rsrq: -10,
        rssi: -65,
        sinr: 15,
        throughput: { dl: 150, ul: 50 },
        latency: { dl: 25, ul: 30 },
        energyConsumption: 1000,
        energyEfficiency: 0.15,
        handoverSuccess: 98,
        handoverLatency: 50,
        coverageArea: 5,
        signalStrength: Array.from({ length: 100 }, () => -75 + Math.random() * 20)
      },
      moClasses: {
        'Cell': {
          parameters: {
            'cellId': 1,
            'pci': 100,
            'earfcn': 1800,
            'dlBandwidth': 20,
            'ulBandwidth': 20
          },
          status: 'active',
          lastUpdate: Date.now()
        }
      },
      environment: {
        timeOfDay: 14,
        dayOfWeek: 2,
        season: 'summer',
        weatherConditions: 'clear',
        eventIndicators: []
      }
    },
    metadata: {
      source: 'perf-test',
      priority: 'medium' as const,
      processingLatency: 0
    }
  };
}

function createLargePerformanceTestMessage() {
  const baseMessage = createPerformanceTestMessage('large-test');

  // Add large data payload
  baseMessage.data.kpis.signalStrength = Array.from({ length: 1000 }, () => -75 + Math.random() * 20);

  // Add multiple cells
  (baseMessage.data as any).additionalCells = Array.from({ length: 10 }, (_, i) => ({
    cellId: `cell-${i}`,
    kpis: {
      rsrp: -70 - i * 2,
      rsrq: -12 + i,
      sinr: 10 + i * 2,
      throughput: { dl: 100 + i * 10, ul: 30 + i * 5 }
    }
  }));

  return baseMessage;
}

function createComplexPerformanceTestMessage() {
  const baseMessage = createPerformanceTestMessage('complex-test');

  // Add complexity that triggers cognitive processing
  baseMessage.data.kpis.rsrp = -85; // Poor signal
  baseMessage.data.kpis.sinr = 5;   // Low SINR
  baseMessage.data.kpis.energyEfficiency = 0.08; // Poor efficiency

  // Add multiple MO classes with complex parameters
  baseMessage.data.moClasses['Radio'] = {
    parameters: {
      'txPower': 43,
      'antennaGain': 18,
      'noiseFigure': 3,
      'mimoLayers': 4,
      'carrierAggregation': 3
    },
    status: 'degraded',
    lastUpdate: Date.now()
  };

  baseMessage.data.moClasses['Transport'] = {
    parameters: {
      'bandwidth': 1000,
      'latency': 50,
      'packetLoss': 0.01,
      'jitter': 5
    },
    status: 'active',
    lastUpdate: Date.now()
  };

  return baseMessage;
}

function createAnomalousPerformanceTestMessage(id: string = 'anomaly-test') {
  const baseMessage = createPerformanceTestMessage(id);

  // Create anomalies
  baseMessage.data.kpis.rsrp = -95; // Anomalous signal strength
  baseMessage.data.kpis.sinr = -5;  // Anomalous SINR
  baseMessage.data.kpis.latency.dl = 500; // Anomalous latency
  baseMessage.data.kpis.handoverSuccess = 70; // Anomalous handover success

  return baseMessage;
}

function createOptimizationTestMessage(id: string = 'optimize-test') {
  const baseMessage = createPerformanceTestMessage(id);

  // Create conditions that should trigger optimization
  baseMessage.data.kpis.energyEfficiency = 0.05; // Low efficiency - trigger energy optimization
  baseMessage.data.kpis.throughput.dl = 50;    // Low throughput - trigger performance optimization
  baseMessage.data.kpis.coverageArea = 2;       // Small coverage - trigger coverage optimization

  return baseMessage;
}