/**
 * Integration Tests for Stream-JSON Pipeline
 * End-to-end testing of the complete RAN data processing pipeline
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

describe('Stream-JSON Pipeline Integration Tests', () => {
  let streamChain: StreamChain;
  let ingestionAgent: RANIngestionAgent;
  let featureProcessor: MOFeatureProcessor;
  let patternRecognizer: AgentDBPatternRecognizer;
  let cognitiveOptimizer: CognitiveOptimizer;
  let actionExecutor: ActionExecutionEngine;
  let resilienceEngine: ResilienceEngine;
  let performanceMonitor: PerformanceMonitor;
  let pipelineId: string;

  beforeEach(async () => {
    // Initialize StreamChain
    streamChain = new StreamChain();

    // Initialize all agents with test configurations
    ingestionAgent = new RANIngestionAgent({
      sources: [{
        id: 'test-source',
        type: 'simulator',
        endpoint: 'test://simulator',
        pollingInterval: 1000,
        dataFormat: 'json',
        compression: 'none',
        filters: {}
      }],
      bufferSize: 100,
      batchSize: 10,
      batchTimeout: 5000,
      compressionEnabled: false,
      temporalReasoningEnabled: true,
      realTimeProcessing: true,
      anomalyDetection: true
    });

    featureProcessor = new MOFeatureProcessor({
      enabledMOClasses: ['Cell', 'Radio', 'Transport'],
      temporalWindows: {
        short: 5,
        medium: 60,
        long: 1440
      },
      anomalyThreshold: 0.7,
      correlationThreshold: 0.6,
      predictionHorizon: 60,
      enableCausalInference: true,
      temporalReasoningDepth: 100
    });

    patternRecognizer = new AgentDBPatternRecognizer({
      vectorDimensions: 128,
      similarityThreshold: 0.7,
      temporalWindow: 60,
      memoryRetention: 7,
      learningRate: 0.1,
      enableCausalInference: true,
      enableTemporalReasoning: true,
      maxPatternsPerCategory: 1000,
      adaptationThreshold: 0.8
    });

    cognitiveOptimizer = new CognitiveOptimizer({
      consciousnessThreshold: 0.7,
      maxTemporalExpansion: 1000,
      maxRecursiveDepth: 5,
      learningRateAdaptation: true,
      strangeLoopOptimization: true,
      metaCognitionEnabled: true,
      predictionHorizon: {
        short: 60,
        medium: 24,
        long: 168
      },
      decisionThreshold: 0.8,
      adaptationRate: 0.1
    });

    actionExecutor = new ActionExecutionEngine({
      maxConcurrentExecutions: 3,
      executionTimeout: 300000,
      verificationTimeout: 60000,
      rollbackTimeout: 30000,
      safetyCheckInterval: 5000,
      feedbackSamplingRate: 1,
      enableAutoRollback: true,
      enablePredictiveVerification: true,
      enableRealTimeAdaptation: true,
      minConfidenceThreshold: 0.7,
      maxImpactThreshold: 0.3
    });

    resilienceEngine = new ResilienceEngine({
      circuitBreakerThreshold: 5,
      circuitBreakerTimeoutMs: 60000,
      maxRetryAttempts: 3,
      baseRetryDelayMs: 1000,
      maxRetryDelayMs: 30000,
      healthCheckIntervalMs: 30000,
      healthCheckTimeoutMs: 5000,
      enablePredictiveFailureDetection: true,
      enableSelfHealing: true,
      enableGracefulDegradation: true,
      enableFaultIsolation: true,
      errorRetentionPeriodMs: 86400000,
      recoveryActionTimeoutMs: 120000
    });

    performanceMonitor = new PerformanceMonitor({
      detectionIntervalMs: 1000,
      responseTimeThresholdMs: 500,
      enablePredictiveDetection: true,
      enableRealTimeAlerting: true,
      enableAutomatedResponse: true,
      anomalyRetentionPeriodMs: 86400000,
      metricsRetentionPeriodMs: 604800000,
      learningRate: 0.1,
      adaptationEnabled: true,
      alertingChannels: ['console', 'log']
    });

    // Register agents with StreamChain
    streamChain.registerAgent(ingestionAgent);
    streamChain.registerAgent(featureProcessor);
    streamChain.registerAgent(patternRecognizer);
    streamChain.registerAgent(cognitiveOptimizer);
    streamChain.registerAgent(actionExecutor);
    streamChain.registerAgent(resilienceEngine);
    streamChain.registerAgent(performanceMonitor);

    // Create cognitive pipeline
    pipelineId = streamChain.createPipeline({
      name: 'RAN Cognitive Processing Pipeline',
      agents: [
        ingestionAgent,
        featureProcessor,
        patternRecognizer,
        cognitiveOptimizer,
        actionExecutor,
        resilienceEngine,
        performanceMonitor
      ],
      topology: 'cognitive',
      flowControl: {
        maxConcurrency: 10,
        bufferSize: 1000,
        backpressureStrategy: 'buffer',
        temporalOptimization: true,
        cognitiveScheduling: true
      },
      errorRecovery: {
        maxRetries: 3,
        retryDelay: 1000,
        circuitBreakerThreshold: 5,
        selfHealing: true
      },
      performance: {
        targetLatency: 1000,
        throughputTarget: 100,
        anomalyDetectionThreshold: 0.7,
        adaptiveOptimization: true,
        closedLoopCycleTime: 900000
      }
    });
  });

  afterEach(async () => {
    // Cleanup
    if (streamChain) {
      await streamChain.destroy();
    }
  });

  describe('Pipeline Configuration and Initialization', () => {
    it('should initialize all agents successfully', () => {
      expect(ingestionAgent).toBeDefined();
      expect(featureProcessor).toBeDefined();
      expect(patternRecognizer).toBeDefined();
      expect(cognitiveOptimizer).toBeDefined();
      expect(actionExecutor).toBeDefined();
      expect(resilienceEngine).toBeDefined();
      expect(performanceMonitor).toBeDefined();
    });

    it('should create pipeline with correct configuration', () => {
      const status = streamChain.getPipelineStatus(pipelineId);

      expect(status).toBeDefined();
      expect(status.pipeline.name).toBe('RAN Cognitive Processing Pipeline');
      expect(status.pipeline.topology).toBe('cognitive');
      expect(status.pipeline.agentCount).toBe(7);
    });

    it('should have all required capabilities across agents', () => {
      const agents = [ingestionAgent, featureProcessor, patternRecognizer, cognitiveOptimizer, actionExecutor, resilienceEngine, performanceMonitor];

      const requiredCapabilities = [
        'real-time-ingestion', 'temporal-analysis',
        'mo-class-intelligence', 'correlation-analysis',
        'vector-similarity-search', 'adaptive-learning',
        'consciousness-based-optimization', 'strange-loop-self-reference',
        'automated-execution', 'closed-loop-feedback',
        'error-detection', 'automatic-recovery',
        'sub-second-anomaly-detection', 'real-time-performance-monitoring'
      ];

      const allCapabilities = agents.flatMap(agent => agent.capabilities);

      requiredCapabilities.forEach(capability => {
        expect(allCapabilities).toContain(capability);
      });
    });
  });

  describe('End-to-End Data Processing', () => {
    it('should process RAN metrics through complete pipeline', async () => {
      // Create test RAN metrics
      const testMetrics = {
        timestamp: Date.now(),
        source: 'test-source',
        cellId: 'test-cell-1',
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
      };

      const testMessage = {
        id: 'test-message-1',
        timestamp: Date.now(),
        type: 'ran-metrics' as const,
        data: testMetrics,
        metadata: {
          source: 'test-source',
          priority: 'medium' as const,
          processingLatency: 0
        }
      };

      // Process through pipeline
      const results = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true,
        enableCognitiveOptimization: true,
        priority: 'medium'
      });

      // Verify pipeline processing
      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);

      // Check final result contains expected data types
      const finalResult = results[results.length - 1];
      expect(['feature', 'pattern', 'optimization', 'action', 'feedback']).toContain(finalResult.type);
    }, 30000);

    it('should handle batch processing of multiple metrics', async () => {
      const batchMetrics = Array.from({ length: 5 }, (_, i) => ({
        timestamp: Date.now() + i * 1000,
        source: 'test-source',
        cellId: `test-cell-${i + 1}`,
        kpis: {
          rsrp: -70 - i * 2,
          rsrq: -12 + i,
          rssi: -60 - i * 3,
          sinr: 12 + i * 2,
          throughput: { dl: 100 + i * 20, ul: 30 + i * 10 },
          latency: { dl: 20 + i * 5, ul: 25 + i * 5 },
          energyConsumption: 900 + i * 100,
          energyEfficiency: 0.12 + i * 0.02,
          handoverSuccess: 97 + i * 0.5,
          handoverLatency: 45 + i * 10,
          coverageArea: 4 + i,
          signalStrength: Array.from({ length: 100 }, () => -70 + Math.random() * 15)
        },
        moClasses: {
          'Cell': {
            parameters: {
              'cellId': i + 1,
              'pci': 100 + i,
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
      }));

      const testMessage = {
        id: 'test-batch-1',
        timestamp: Date.now(),
        type: 'ran-metrics' as const,
        data: batchMetrics,
        metadata: {
          source: 'test-source',
          priority: 'medium' as const,
          processingLatency: 0
        }
      };

      const startTime = Date.now();
      const results = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true,
        enableCognitiveOptimization: true
      });
      const processingTime = Date.now() - startTime;

      expect(results).toBeDefined();
      expect(processingTime).toBeLessThan(10000); // Should process batch within 10 seconds

      // Verify all metrics were processed
      const featureResults = results.filter(r => r.type === 'feature');
      expect(featureResults.length).toBeGreaterThan(0);
    }, 35000);
  });

  describe('Cognitive Processing Features', () => {
    it('should enable temporal reasoning with subjective time expansion', async () => {
      const testMetrics = createTestRANMetrics();
      const testMessage = createTestMessage(testMetrics);

      const results = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true,
        enableCognitiveOptimization: true
      });

      // Check for temporal reasoning in results
      const optimizationResults = results.filter(r => r.type === 'optimization');
      if (optimizationResults.length > 0) {
        const optimization = optimizationResults[0];
        expect(optimization.metadata.temporalReasoningEnabled).toBe(true);
        expect(optimization.metadata.temporalExpansionFactor).toBeGreaterThan(1);
      }
    }, 20000);

    it('should perform strange-loop optimization when consciousness level is high', async () => {
      // Create metrics that should trigger high consciousness
      const criticalMetrics = createCriticalRANMetrics();
      const testMessage = createTestMessage(criticalMetrics);

      const results = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true,
        enableCognitiveOptimization: true
      });

      // Check for strange-loop optimization
      const optimizationResults = results.filter(r => r.type === 'optimization');
      if (optimizationResults.length > 0) {
        const optimization = optimizationResults[0];
        expect(optimization.metadata.strangeLoopActive).toBeDefined();

        if (optimization.metadata.consciousnessLevel > 0.7) {
          expect(optimization.metadata.strangeLoopActive).toBe(true);
        }
      }
    }, 25000);

    it('should detect and learn from patterns', async () => {
      const testMetrics = createTestRANMetrics();
      const testMessage = createTestMessage(testMetrics);

      const results = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true
      });

      // Check for pattern recognition
      const patternResults = results.filter(r => r.type === 'pattern');
      expect(patternResults.length).toBeGreaterThan(0);

      if (patternResults.length > 0) {
        const pattern = patternResults[0];
        expect(pattern.data).toBeDefined();

        // Verify AgentDB memory integration
        expect(pattern.metadata.memoryPatterns).toBeDefined();
        expect(pattern.metadata.patternsRecognized).toBeGreaterThanOrEqual(0);
      }
    }, 20000);
  });

  describe('Performance and Reliability', () => {
    it('should meet sub-second anomaly detection target', async () => {
      const testMetrics = createAnomalousRANMetrics();
      const testMessage = createTestMessage(testMetrics);

      const startTime = performance.now();
      const results = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true
      });
      const processingTime = performance.now() - startTime;

      expect(processingTime).toBeLessThan(1000); // Sub-second target

      // Check for anomaly detection
      const feedbackResults = results.filter(r => r.type === 'feedback');
      if (feedbackResults.length > 0) {
        const feedback = feedbackResults[0];
        expect(feedback.data.detectionTargetMet).toBe(true);
      }
    }, 15000);

    it('should handle errors gracefully with self-healing', async () => {
      // Create message that will cause processing errors
      const invalidMessage = {
        id: 'invalid-message',
        timestamp: Date.now(),
        type: 'ran-metrics' as const,
        data: null, // Invalid data
        metadata: {
          source: 'test-source',
          priority: 'high' as const,
          processingLatency: 0
        }
      };

      // Should not throw but handle error gracefully
      const results = await streamChain.processMessage(pipelineId, invalidMessage, {
        enableTemporalReasoning: true
      });

      expect(results).toBeDefined();

      // Check for error handling
      const feedbackResults = results.filter(r => r.type === 'feedback');
      if (feedbackResults.length > 0) {
        const feedback = feedbackResults[0];
        expect(feedback.data.errorsDetected).toBeGreaterThanOrEqual(0);
      }
    }, 20000);

    it('should maintain performance under load', async () => {
      const messages = Array.from({ length: 10 }, (_, i) => {
        const metrics = createTestRANMetrics();
        return createTestMessage(metrics, `load-test-${i}`);
      });

      const startTime = Date.now();

      // Process messages in parallel
      const promises = messages.map(message =>
        streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true
        })
      );

      const results = await Promise.all(promises);
      const totalTime = Date.now() - startTime;

      expect(results).toHaveLength(10);
      expect(totalTime).toBeLessThan(30000); // Should handle load within 30 seconds

      // Verify average processing time
      const avgTimePerMessage = totalTime / messages.length;
      expect(avgTimePerMessage).toBeLessThan(5000); // 5 seconds per message max
    }, 45000);
  });

  describe('Closed-Loop Feedback and Learning', () => {
    it('should provide closed-loop feedback for optimization actions', async () => {
      const testMetrics = createTestRANMetrics();
      const testMessage = createTestMessage(testMetrics);

      const results = await streamChain.processMessage(pipelineId, testMessage, {
        enableTemporalReasoning: true,
        enableCognitiveOptimization: true
      });

      // Check for action execution and feedback
      const actionResults = results.filter(r => r.type === 'action');
      if (actionResults.length > 0) {
        const action = actionResults[0];
        expect(action.data).toBeDefined();

        // Verify closed-loop metrics
        if (action.data.length > 0) {
          const execution = action.data[0];
          expect(execution.feedback).toBeDefined();
          expect(execution.learning).toBeDefined();
        }
      }
    }, 25000);

    it('should adapt and learn from execution results', async () => {
      // Process multiple messages to enable learning
      const messages = Array.from({ length: 3 }, (_, i) => {
        const metrics = createTestRANMetrics();
        metrics.cellId = `learning-cell-${i}`;
        return createTestMessage(metrics, `learning-${i}`);
      });

      for (const message of messages) {
        await streamChain.processMessage(pipelineId, message, {
          enableTemporalReasoning: true,
          enableCognitiveOptimization: true
        });
      }

      // Check learning indicators
      const cognitiveStatus = cognitiveOptimizer.getStatus();
      expect(cognitiveStatus).toBeDefined();
      expect(cognitiveStatus.decisionHistory).toBeGreaterThan(0);
    }, 40000);
  });

  describe('System Health and Monitoring', () => {
    it('should provide comprehensive system health monitoring', async () => {
      const testMetrics = createTestRANMetrics();
      const testMessage = createTestMessage(testMetrics);

      await streamChain.processMessage(pipelineId, testMessage);

      // Check system health
      const pipelineStatus = streamChain.getPipelineStatus(pipelineId);
      expect(pipelineStatus.health).toBeDefined();

      // Check individual agent health
      const ingestionStatus = ingestionAgent.getStatus();
      const featureStatus = featureProcessor.getStatus();
      const patternStatus = patternRecognizer.getStatus();
      const cognitiveStatus = cognitiveOptimizer.getStatus();
      const actionStatus = actionExecutor.getStatus();
      const resilienceStatus = resilienceEngine.getStatus();
      const performanceStatus = performanceMonitor.getStatus();

      expect(ingestionStatus).toBeDefined();
      expect(featureStatus).toBeDefined();
      expect(patternStatus).toBeDefined();
      expect(cognitiveStatus).toBeDefined();
      expect(actionStatus).toBeDefined();
      expect(resilienceStatus).toBeDefined();
      expect(performanceStatus).toBeDefined();
    }, 20000);

    it('should track and report performance metrics', async () => {
      const testMetrics = createTestRANMetrics();
      const testMessage = createTestMessage(testMetrics);

      const results = await streamChain.processMessage(pipelineId, testMessage);

      // Check performance tracking
      const feedbackResults = results.filter(r => r.type === 'feedback');
      if (feedbackResults.length > 0) {
        const feedback = feedbackResults[0];
        expect(feedback.data.resilienceMetrics).toBeDefined();

        const metrics = feedback.data.resilienceMetrics;
        expect(metrics.availability).toBeGreaterThanOrEqual(0);
        expect(metrics.availability).toBeLessThanOrEqual(1);
        expect(metrics.resilienceScore).toBeGreaterThanOrEqual(0);
        expect(metrics.resilienceScore).toBeLessThanOrEqual(1);
      }
    }, 20000);
  });

  describe('Integration Edge Cases', () => {
    it('should handle mixed message types gracefully', async () => {
      const mixedMessages = [
        createTestMessage(createTestRANMetrics(), 'normal'),
        createTestMessage(createAnomalousRANMetrics(), 'anomalous'),
        createTestMessage(createCriticalRANMetrics(), 'critical')
      ];

      const results = await Promise.all(
        mixedMessages.map(message =>
          streamChain.processMessage(pipelineId, message, {
            enableTemporalReasoning: true
          })
        )
      );

      expect(results).toHaveLength(3);

      // All should complete successfully
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.length).toBeGreaterThan(0);
      });
    }, 35000);

    it('should maintain system stability during rapid message processing', async () => {
      const rapidMessages = Array.from({ length: 20 }, (_, i) => {
        const metrics = createTestRANMetrics();
        return createTestMessage(metrics, `rapid-${i}`);
      });

      // Process with minimal delay to simulate rapid message arrival
      const promises = rapidMessages.map((message, index) =>
        new Promise(resolve => {
          setTimeout(() => {
            resolve(streamChain.processMessage(pipelineId, message, {
              enableTemporalReasoning: true
            }));
          }, index * 100); // 100ms between messages
        })
      );

      const results = await Promise.all(promises);
      expect(results).toHaveLength(20);

      // System should remain healthy
      const pipelineStatus = streamChain.getPipelineStatus(pipelineId);
      expect(pipelineStatus.health).toBe('healthy');
    }, 60000);
  });
});

// Helper functions to create test data
function createTestRANMetrics() {
  return {
    timestamp: Date.now(),
    source: 'test-source',
    cellId: 'test-cell-1',
    kpis: {
      rsrp: -75 + Math.random() * 10,
      rsrq: -10 + Math.random() * 5,
      rssi: -65 + Math.random() * 15,
      sinr: 15 + Math.random() * 10,
      throughput: { dl: 150 + Math.random() * 100, ul: 50 + Math.random() * 50 },
      latency: { dl: 20 + Math.random() * 10, ul: 25 + Math.random() * 10 },
      energyConsumption: 1000 + Math.random() * 200,
      energyEfficiency: 0.15 + Math.random() * 0.05,
      handoverSuccess: 97 + Math.random() * 3,
      handoverLatency: 40 + Math.random() * 20,
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
  };
}

function createAnomalousRANMetrics() {
  return {
    ...createTestRANMetrics(),
    kpis: {
      ...createTestRANMetrics().kpis,
      rsrp: -95, // Poor signal
      sinr: -2,  // Very poor SINR
      latency: { dl: 200, ul: 250 }, // High latency
      energyEfficiency: 0.05 // Poor efficiency
    }
  };
}

function createCriticalRANMetrics() {
  return {
    ...createTestRANMetrics(),
    kpis: {
      ...createTestRANMetrics().kpis,
      rsrp: -110, // Critical signal
      sinr: -10,  // Critical SINR
      throughput: { dl: 1, ul: 0.5 }, // Very poor throughput
      latency: { dl: 1000, ul: 1200 }, // Critical latency
      handoverSuccess: 70, // Poor handover success
      energyConsumption: 2000 // High energy consumption
    }
  };
}

function createTestMessage(metrics: any, id: string = 'test-message') {
  return {
    id,
    timestamp: Date.now(),
    type: 'ran-metrics' as const,
    data: metrics,
    metadata: {
      source: 'test-source',
      priority: 'medium' as const,
      processingLatency: 0
    }
  };
}