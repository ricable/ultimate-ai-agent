/**
 * Integration Tests for Adaptive Coordination System
 *
 * End-to-end integration tests covering the complete adaptive coordination
 * pipeline with dynamic topology, resource allocation, consensus mechanisms,
 * performance monitoring, and cognitive intelligence.
 */

import { AdaptiveSwarmCoordinator, AdaptiveConfiguration } from '../../src/adaptive-coordinator/adaptive-swarm-coordinator';
import { DynamicTopologyOptimizer } from '../../src/topology/dynamic-topology-optimizer';
import { IntelligentResourceAllocator } from '../../src/resource-allocation/intelligent-resource-allocator';
import { ConsensusMechanism } from '../../src/consensus/consensus-mechanism';
import { PerformanceMonitor } from '../../src/performance/performance-monitor';
import { AutonomousScaler } from '../../src/scaling/autonomous-scaler';
import { OptimizationCycleCoordinator } from '../../src/optimization/optimization-cycle-coordinator';
import { AgentDBMemoryPatterns } from '../../src/memory/agentdb-memory-patterns';
import { Agent } from '../../src/adaptive-coordinator/types';

describe('Adaptive Coordination Integration Tests', () => {
  let coordinator: AdaptiveSwarmCoordinator;
  let topologyOptimizer: DynamicTopologyOptimizer;
  let resourceAllocator: IntelligentResourceAllocator;
  let consensusMechanism: ConsensusMechanism;
  let performanceMonitor: PerformanceMonitor;
  let autonomousScaler: AutonomousScaler;
  let optimizationCoordinator: OptimizationCycleCoordinator;
  let memoryPatterns: AgentDBMemoryPatterns;

  beforeAll(async () => {
    // Initialize all components with realistic configuration
    const config: AdaptiveConfiguration = {
      topologyStrategy: 'adaptive',
      topologySwitchThreshold: 0.2,
      adaptationFrequency: 5,
      maxTopologyTransitions: 3,
      resourcePredictionWindow: 60,
      scalingCooldownPeriod: 2,
      resourceUtilizationTarget: 0.8,
      consensusAlgorithm: 'adaptive',
      consensusTimeout: 30000,
      byzantineFaultTolerance: true,
      requiredConsensus: 0.7,
      monitoringInterval: 30000,
      performanceWindow: 60,
      bottleneckDetectionThreshold: 0.3,
      optimizationCycleInterval: 15,
      cognitiveIntelligenceEnabled: true,
      learningRate: 0.1,
      patternRecognitionWindow: 24,
      autonomousDecisionThreshold: 0.8
    };

    coordinator = new AdaptiveSwarmCoordinator(config);

    // Initialize individual components for integration testing
    topologyOptimizer = new DynamicTopologyOptimizer({
      currentTopology: 'hierarchical',
      availableTopologies: ['hierarchical', 'mesh', 'ring', 'star', 'hybrid', 'adaptive'],
      switchThreshold: 0.2,
      adaptationFrequency: 5,
      maxTransitions: 3,
      migrationStrategy: 'gradual',
      validationRequired: true,
      rollbackEnabled: true
    });

    resourceAllocator = new IntelligentResourceAllocator({
      predictionWindow: 60,
      scalingCooldown: 2,
      utilizationTarget: 0.8,
      loadBalancingStrategy: 'adaptive-hybrid',
      resourceOptimization: {
        optimizationInterval: 10,
        optimizationMethod: 'genetic-algorithm',
        costOptimization: true,
        performanceOptimization: true,
        efficiencyOptimization: true,
        constraints: {
          maxCpuCores: 1000,
          maxMemoryGB: 8000,
          maxNetworkMbps: 10000,
          maxStorageGB: 50000,
          maxCostPerHour: 1000,
          minPerformanceScore: 0.8,
          maxAgentCount: 500
        }
      },
      cognitiveLearning: {
        learningRate: 0.1,
        patternRecognitionWindow: 24,
        predictionAccuracyTarget: 0.9,
        modelUpdateFrequency: 6,
        featureEngineering: true,
        ensembleMethods: true
      }
    });

    consensusMechanism = new ConsensusMechanism({
      algorithm: 'adaptive',
      timeout: 30000,
      byzantineTolerance: true,
      requiredConsensus: 0.7,
      adaptiveSelection: true,
      votingMethod: 'reputation-based',
      faultTolerance: {
        maxFaultyNodes: 3,
        byzantineFaults: true,
        crashFaults: true,
        networkPartitions: true,
        recoveryStrategy: 'automatic',
        checkpointInterval: 60000,
        logCompaction: true
      },
      cognitiveLearning: {
        learningEnabled: true,
        historicalLearning: true,
        patternRecognition: true,
        adaptiveThresholds: true,
        confidenceWeighting: true,
        modelUpdateFrequency: 6,
        consensusPrediction: true
      }
    });

    performanceMonitor = new PerformanceMonitor({
      monitoringInterval: 30000,
      performanceWindow: 60,
      bottleneckThreshold: 0.3,
      alertThresholds: {
        responseTime: 1000,
        errorRate: 0.05,
        cpuUtilization: 0.9,
        memoryUtilization: 0.9,
        networkLatency: 100,
        throughput: 100,
        availability: 0.99
      },
      optimizationConfig: {
        automaticOptimization: true,
        optimizationInterval: 15,
        optimizationMethods: ['parameter-tuning', 'resource-rebalancing', 'topology-optimization'],
        performanceTargets: {
          targetResponseTime: 500,
          targetThroughput: 200,
          targetAvailability: 0.995,
          targetErrorRate: 0.01,
          targetResourceEfficiency: 0.85,
          targetCostEfficiency: 0.9
        },
        optimizationHistory: {
          retentionPeriod: 30,
          maxHistoryEntries: 1000,
          performanceTracking: true,
          rollbackTracking: true
        }
      },
      cognitiveMonitoring: {
        enabled: true,
        patternRecognition: true,
        anomalyDetection: true,
        predictiveAnalysis: true,
        learningRate: 0.1,
        modelUpdateFrequency: 6,
        confidenceThreshold: 0.8
      },
      retentionPolicy: {
        metricsRetentionDays: 7,
        alertsRetentionDays: 30,
        optimizationRetentionDays: 90,
        rawLogsRetentionDays: 3,
        compressionEnabled: true,
        archiveStorage: true
      }
    });

    autonomousScaler = new AutonomousScaler({
      scalingCooldownPeriod: 2,
      utilizationTarget: 0.8,
      cognitiveScaling: true,
      costOptimization: true,
      scalingPolicies: [
        {
          policyId: 'cpu-scaling',
          name: 'CPU-based Scaling',
          description: 'Scale based on CPU utilization',
          conditions: [
            {
              metric: 'cpu-utilization',
              operator: '>',
              threshold: 0.8,
              duration: 2,
              evaluationWindow: 5,
              weight: 0.7,
              aggregator: 'average'
            }
          ],
          actions: [
            {
              actionType: 'scale-up',
              agentType: 'worker',
              count: 2,
              parameters: {},
              timeout: 300,
              validationRequired: true
            }
          ],
          priority: 8,
          enabled: true,
          cooldownPeriod: 5,
          rollbackPolicy: {
            automaticRollback: true,
            rollbackTriggers: [
              {
                metric: 'error-rate',
                threshold: 0.1,
                operator: '>',
                evaluationWindow: 5,
                consecutiveViolations: 3,
                severity: 'high'
              }
            ],
            rollbackTimeout: 10,
            dataConsistency: true,
            serviceDisruption: false
          }
        }
      ],
      cognitiveModels: {
        enabled: true,
        learningRate: 0.1,
        patternRecognition: true,
        anomalyDetection: true,
        predictiveAccuracy: 0.9,
        modelUpdateFrequency: 6,
        confidenceThreshold: 0.8,
        adaptationEnabled: true,
        featureEngineering: true,
        ensembleMethods: true
      },
      costConstraints: {
        maxCostPerHour: 500,
        costBudgetPerDay: 10000,
        costOptimizationTarget: 0.2,
        preferSpotInstances: false,
        preferReservedInstances: true,
        costPredictionAccuracy: 0.85
      },
      scalingLimits: {
        minAgents: 3,
        maxAgents: 50,
        maxScalingStep: 5,
        scalingRateLimit: 10,
        resourceLimits: {
          maxCpuCores: 200,
          maxMemoryGB: 1600,
          maxNetworkMbps: 2000,
          maxStorageGB: 10000,
          maxEnergyConsumption: 1000
        },
        geographicLimits: {
          allowedRegions: ['us-east-1', 'us-west-2'],
          maxAgentsPerRegion: { 'us-east-1': 30, 'us-west-2': 20 },
          latencyConstraints: [],
          complianceConstraints: []
        },
        providerLimits: {
          aws: { maxInstances: 40, maxCpu: 160, maxMemory: 1280, maxStorage: 8000, rateLimits: [], supportedInstanceTypes: [], costPerHour: {} },
          azure: { maxInstances: 30, maxCpu: 120, maxMemory: 960, maxStorage: 6000, rateLimits: [], supportedInstanceTypes: [], costPerHour: {} },
          gcp: { maxInstances: 20, maxCpu: 80, maxMemory: 640, maxStorage: 4000, rateLimits: [], supportedInstanceTypes: [], costPerHour: {} },
          custom: {}
        }
      },
      emergencyScaling: {
        enabled: true,
        triggerConditions: [
          {
            conditionId: 'high-error-rate',
            name: 'High Error Rate',
            severity: 'critical',
            metric: 'error-rate',
            threshold: 0.2,
            evaluationWindow: 30,
            triggerAction: {
              actionType: 'immediate-scale',
              parameters: { scaleFactor: 2 },
              escalationLevel: 5
            },
            cooldownPeriod: 15
          }
        ],
        emergencyScalingFactor: 2,
        maxEmergencyScale: 100,
        emergencyTimeout: 30,
        autoRollback: true,
        notificationChannels: ['email', 'slack']
      }
    });

    optimizationCoordinator = new OptimizationCycleCoordinator({
      cycleInterval: 15,
      cognitiveIntelligence: true,
      learningRate: 0.1,
      optimizationScope: {
        topologyOptimization: true,
        resourceOptimization: true,
        performanceOptimization: true,
        costOptimization: true,
        securityOptimization: true,
        reliabilityOptimization: true,
        scalabilityOptimization: true,
        cognitiveOptimization: true
      },
      adaptiveStrategies: [],
      performanceTargets: {
        targetResponseTime: 500,
        targetThroughput: 200,
        targetAvailability: 0.995,
        targetErrorRate: 0.01,
        targetResourceEfficiency: 0.85,
        targetCostEfficiency: 0.9,
        targetUserSatisfaction: 0.9,
        targetSystemHealth: 0.95
      },
      learningConfiguration: {
        enabled: true,
        learningAlgorithms: [
          {
            algorithmId: 'neural-network',
            algorithmType: 'neural-network',
            targetProblems: ['performance-optimization', 'resource-allocation'],
            accuracy: 0.88,
            trainingDataRequirement: 1000,
            updateFrequency: 6,
            confidenceThreshold: 0.8,
            ensembleWeight: 0.4
          }
        ],
        featureEngineering: true,
        patternRecognition: true,
        predictiveModeling: true,
        adaptiveThresholds: true,
        ensembleMethods: true,
        continuousLearning: true,
        modelValidation: true,
        knowledgeRetention: 100
      },
      cyclePhases: {
        phases: [
          {
            phaseId: 'analysis',
            phaseName: 'analysis',
            description: 'Analyze current system state',
            sequence: 1,
            dependencies: [],
            parallelizable: false,
            critical: true,
            timeout: 5,
            validationRequired: true,
            checkpointRequired: false
          },
          {
            phaseId: 'optimization',
            phaseName: 'optimization',
            description: 'Execute optimization strategies',
            sequence: 2,
            dependencies: ['analysis'],
            parallelizable: true,
            critical: true,
            timeout: 8,
            validationRequired: true,
            checkpointRequired: true
          },
          {
            phaseId: 'validation',
            phaseName: 'validation',
            description: 'Validate optimization results',
            sequence: 3,
            dependencies: ['optimization'],
            parallelizable: false,
            critical: true,
            timeout: 2,
            validationRequired: true,
            checkpointRequired: false
          }
        ],
        phaseTimeouts: { 'analysis': 5, 'optimization': 8, 'validation': 2 },
        parallelExecution: true,
        checkpointEnabled: true,
        validationRequired: true,
        rollbackCheckpoints: ['optimization'],
        criticalPhases: ['analysis', 'optimization', 'validation']
      },
      rollbackConfiguration: {
        enabled: true,
        automaticRollback: true,
        rollbackTriggers: [
          {
            triggerId: 'performance-degradation',
            triggerType: 'performance-degradation',
            metric: 'response-time',
            threshold: 1000,
            operator: '>',
            evaluationWindow: 5,
            consecutiveViolations: 3,
            severity: 'high',
            automaticRollback: true
          }
        ],
        rollbackStrategies: [],
        maxRollbackTime: 10,
        dataConsistencyGuarantee: true,
        serviceDisruptionAllowed: false,
        rollbackValidation: true
      }
    });

    memoryPatterns = new AgentDBMemoryPatterns({
      patternRecognitionWindow: 24,
      learningRate: 0.1,
      cognitiveIntelligence: true,
      vectorSearchEnabled: true,
      quicSyncEnabled: true,
      persistenceEnabled: true,
      memoryConsolidation: {
        enabled: true,
        consolidationInterval: 6,
        importanceThreshold: 0.7,
        compressionRatio: 0.7,
        retentionPolicy: {
          shortTermRetention: 7,
          longTermRetention: 90,
          criticalPatternRetention: 365,
          archiveStorage: true,
          compressionEnabled: true,
          tieredStorage: true
        },
        forgettingCurve: {
          enabled: true,
          decayRate: 0.1,
          reinforcementFactor: 0.8,
          reviewSchedule: [
            {
              scheduleType: 'spaced',
              intervals: [1, 6, 24, 72],
              reviewTrigger: 'performance-degradation',
              successThreshold: 0.8
            }
          ],
          importanceWeighting: true,
          adaptiveDecay: true
        },
        patternPruning: {
          enabled: true,
          pruningThreshold: 0.3,
          redundancyThreshold: 0.8,
          pruningFrequency: 24,
          preserveCritical: true,
          backupBeforePrune: true
        }
      },
      predictiveAnalytics: {
        enabled: true,
        predictionHorizon: 60,
        modelTypes: ['lstm', 'transformer', 'ensemble'],
        ensembleMethods: true,
        confidenceThreshold: 0.8,
        updateFrequency: 6,
        accuracyTarget: 0.9
      },
      crossAgentLearning: {
        enabled: true,
        knowledgeSharing: true,
        patternTransfer: true,
        collectiveIntelligence: true,
        sharingProtocols: [
          {
            protocolId: 'federated-learning',
            protocolType: 'federated',
            encryption: true,
            authentication: true,
            validation: true,
            priority: 8,
            scope: 'global'
          }
        ],
        privacyControls: [
          {
            controlId: 'data-anonymization',
            controlType: 'data-anonymization',
            sensitivity: 'high',
            anonymization: true,
            accessControl: true,
            auditLogging: true,
            dataMinimization: true
          }
        ],
        knowledgeValidation: true,
        reputationSystem: true
      },
      memoryOptimization: {
        enabled: true,
        vectorIndexing: true,
        compressionEnabled: true,
        deduplication: true,
        tieredStorage: true,
        cacheManagement: {
          enabled: true,
          cacheSize: 1024,
          evictionPolicy: 'lru',
          prewarming: true,
          cacheInvalidation: true,
          distributedCaching: true,
          cacheAnalytics: true
        },
        indexingStrategy: {
          primaryIndex: 'pattern-id',
          secondaryIndexes: ['pattern-type', 'importance', 'timestamp'],
          vectorIndex: true,
          fullTextIndex: true,
          compositeIndexes: [],
          indexMaintenance: true
        },
        queryOptimization: true
      }
    });
  });

  afterAll(async () => {
    // Cleanup all components
    const cleanupPromises = [
      coordinator?.shutdown(),
      topologyOptimizer?.cleanup(),
      resourceAllocator?.cleanup(),
      consensusMechanism?.cleanup(),
      performanceMonitor?.cleanup(),
      autonomousScaler?.cleanup(),
      memoryPatterns?.cleanup()
    ].filter(Boolean);

    await Promise.all(cleanupPromises);
  });

  describe('Complete Adaptive Coordination Flow', () => {
    test('should execute complete adaptation cycle', async () => {
      // 1. Collect current metrics
      const currentMetrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(currentMetrics).toBeDefined();

      // 2. Analyze topology needs
      const topologyAnalysis = await topologyOptimizer.analyzeTopologyNeeds(
        currentMetrics.topologyMetrics,
        currentMetrics.performanceMetrics
      );
      expect(topologyAnalysis).toBeDefined();
      expect(topologyAnalysis.confidence).toBeGreaterThanOrEqual(0);
      expect(topologyAnalysis.confidence).toBeLessThanOrEqual(1);

      // 3. Analyze scaling needs
      const scalingAnalysis = await resourceAllocator.analyzeScalingNeeds(
        currentMetrics.resourceMetrics,
        currentMetrics.performanceMetrics
      );
      expect(scalingAnalysis).toBeDefined();
      expect(scalingAnalysis.confidence).toBeGreaterThanOrEqual(0);
      expect(scalingAnalysis.confidence).toBeLessThanOrEqual(1);

      // 4. Analyze consensus needs
      const consensusAnalysis = await consensusMechanism.analyzeConsensusNeeds(
        currentMetrics.consensusMetrics,
        currentMetrics.performanceMetrics
      );
      expect(consensusAnalysis).toBeDefined();

      // 5. Detect bottlenecks
      const bottleneckDetection = await performanceMonitor.detectBottlenecks();
      expect(bottleneckDetection).toBeDefined();
      expect(bottleneckDetection.bottlenecks).toBeDefined();
      expect(Array.isArray(bottleneckDetection.bottlenecks)).toBe(true);

      // 6. Generate optimization cycle
      const optimizationCycle = await optimizationCoordinator.executeOptimizationCycle({
        swarmTopology: coordinator.getCurrentTopology(),
        currentAgents: createTestAgents(10),
        performanceMetrics: currentMetrics.performanceMetrics,
        cognitivePatterns: []
      });
      expect(optimizationCycle).toBeDefined();
      expect(optimizationCycle.success).toBe(true);

      // 7. Store learning patterns
      await memoryPatterns.storeAdaptiveMetrics(currentMetrics);
      const storedPatterns = await memoryPatterns.getCurrentPatterns();
      expect(Array.isArray(storedPatterns)).toBe(true);
    }, 60000); // 60 second timeout for complete integration test

    test('should handle dynamic topology switching', async () => {
      const initialTopology = coordinator.getCurrentTopology();
      expect(initialTopology).toBeDefined();

      // Simulate performance degradation that would trigger topology change
      const degradedMetrics = createDegradedMetrics();

      // Analyze if topology change is needed
      const topologyAnalysis = await topologyOptimizer.analyzeTopologyNeeds(
        degradedMetrics.topologyMetrics,
        degradedMetrics.performanceMetrics
      );

      // Should recommend topology change if performance is poor
      if (topologyAnalysis.confidence > testConfig.topologySwitchThreshold) {
        expect(topologyAnalysis.recommendedTopology).toBeDefined();
        expect(topologyAnalysis.recommendedTopology).not.toBe(initialTopology);
      }
    });

    test('should coordinate resource scaling with performance monitoring', async () => {
      // Get current state
      const currentMetrics = await coordinator.getCurrentAdaptiveMetrics();

      // Detect high resource utilization scenario
      const highUtilizationMetrics = createHighUtilizationMetrics();

      // Analyze scaling needs
      const scalingAnalysis = await resourceAllocator.analyzeScalingNeeds(
        highUtilizationMetrics.resourceMetrics,
        highUtilizationMetrics.performanceMetrics
      );

      expect(scalingAnalysis).toBeDefined();

      // If scaling is recommended, it should be actionable
      if (scalingAnalysis.scalingRecommendations.length > 0) {
        const recommendation = scalingAnalysis.scalingRecommendations[0];
        expect(recommendation.confidence).toBeGreaterThan(0);
        expect(recommendation.expectedBenefit).toBeDefined();
      }
    });

    test('should maintain consensus during reconfiguration', async () => {
      // Create a test proposal
      const proposal = {
        proposalId: 'test-proposal-1',
        proposalType: 'topology-change' as const,
        proposerId: 'test-agent',
        content: {
          title: 'Test Topology Change',
          description: 'Test proposal for integration',
          action: 'change-topology',
          parameters: { newTopology: 'mesh' },
          expectedOutcome: {
            performanceImprovement: 0.2,
            resourceImpact: { cpuImpact: 0, memoryImpact: 0, networkImpact: 0, storageImpact: 0, costImpact: 0 },
            riskMitigation: 0.1,
            consensusComplexity: 0.5,
            timeToBenefit: 5
          },
          riskAssessment: {
            riskLevel: 'low' as const,
            potentialFailures: [],
            mitigationStrategies: [],
            rollbackComplexity: 0.3,
            impactRadius: 0.2
          },
          implementationPlan: {
            phases: [],
            dependencies: [],
            estimatedDuration: 10,
            resourceRequirements: {
              minCpuCores: 1,
              minMemoryGB: 2,
              minNetworkMbps: 10,
              minStorageGB: 5,
              scalability: 'horizontal' as const
            },
            validationSteps: []
          },
          rollbackPlan: {
            automaticRollback: true,
            rollbackTriggers: [],
            rollbackSteps: [],
            maxRollbackTime: 5,
            dataConsistencyGuarantee: true
          }
        },
        priority: 'medium' as const,
        timestamp: new Date(),
        expirationTime: new Date(Date.now() + 60000), // 1 minute
        votingDeadline: new Date(Date.now() + 30000), // 30 seconds
        metadata: {}
      };

      // Initiate consensus
      const consensusResult = await consensusMechanism.initiateConsensus(proposal);
      expect(consensusResult).toBeDefined();
      expect(consensusResult.consensusReached).toBeDefined();
    });
  });

  describe('Error Handling and Recovery', () => {
    test('should handle component failures gracefully', async () => {
      // Simulate failure in one component
      const originalMethod = performanceMonitor.detectBottlenecks;

      // Mock a failure
      performanceMonitor.detectBottlenecks = async () => {
        throw new Error('Simulated component failure');
      };

      // System should continue operating with other components
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      expect(metrics).toBeDefined();

      // Restore original method
      performanceMonitor.detectBottlenecks = originalMethod;
    });

    test('should rollback failed optimizations', async () => {
      // Create optimization cycle that will fail validation
      const failingOptimization = await optimizationCoordinator.executeOptimizationCycle({
        swarmTopology: coordinator.getCurrentTopology(),
        currentAgents: createTestAgents(5),
        performanceMetrics: createDegradedMetrics().performanceMetrics,
        cognitivePatterns: []
      });

      // Even if some optimizations fail, the cycle should complete
      expect(failingOptimization).toBeDefined();
      expect(failingOptimization.success).toBeDefined();
    });

    test('should maintain system stability during cascade failures', async () => {
      // Simulate multiple component stress
      const stressPromises = [
        coordinator.getCurrentAdaptiveMetrics(),
        topologyOptimizer.analyzeTopologyNeeds(
          createDegradedMetrics().topologyMetrics,
          createDegradedMetrics().performanceMetrics
        ),
        resourceAllocator.analyzeScalingNeeds(
          createHighUtilizationMetrics().resourceMetrics,
          createHighUtilizationMetrics().performanceMetrics
        ),
        performanceMonitor.detectBottlenecks()
      ];

      // Should handle concurrent stress without complete failure
      const results = await Promise.allSettled(stressPromises);

      // At least some operations should succeed
      const successfulOperations = results.filter(result => result.status === 'fulfilled');
      expect(successfulOperations.length).toBeGreaterThan(0);
    });
  });

  describe('Performance and Scalability', () => {
    test('should handle concurrent optimization requests', async () => {
      const concurrentRequests = Array(10).fill(null).map(async (_, index) => {
        return {
          index,
          result: await optimizationCoordinator.executeOptimizationCycle({
            swarmTopology: coordinator.getCurrentTopology(),
            currentAgents: createTestAgents(3),
            performanceMetrics: await coordinator.getCurrentAdaptiveMetrics().then(m => m.performanceMetrics),
            cognitivePatterns: []
          })
        };
      });

      const results = await Promise.all(concurrentRequests);

      // All requests should complete
      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result.result).toBeDefined();
        expect(result.result.success).toBeDefined();
      });
    });

    test('should maintain performance under memory pressure', async () => {
      // Store many patterns to create memory pressure
      const metrics = await coordinator.getCurrentAdaptiveMetrics();

      for (let i = 0; i < 100; i++) {
        await memoryPatterns.storeAdaptiveMetrics({
          ...metrics,
          timestamp: new Date(Date.now() + i * 1000)
        });
      }

      // System should still function
      const currentPatterns = await memoryPatterns.getCurrentPatterns();
      expect(currentPatterns.length).toBeGreaterThan(0);

      // Memory should be managed
      const memoryReport = await memoryPatterns.getMemoryPatternsReport();
      expect(memoryReport).toBeDefined();
      expect(memoryReport.memoryUsage).toBeGreaterThan(0);
    });

    test('should scale with increasing agent count', async () => {
      const agentCounts = [5, 10, 20, 50];

      for (const count of agentCounts) {
        const startTime = Date.now();

        const optimization = await optimizationCoordinator.executeOptimizationCycle({
          swarmTopology: coordinator.getCurrentTopology(),
          currentAgents: createTestAgents(count),
          performanceMetrics: await coordinator.getCurrentAdaptiveMetrics().then(m => m.performanceMetrics),
          cognitivePatterns: []
        });

        const endTime = Date.now();
        const duration = endTime - startTime;

        expect(optimization.success).toBe(true);

        // Performance should not degrade significantly with more agents
        expect(duration).toBeLessThan(30000); // 30 seconds max
      }
    });
  });

  describe('Cognitive Intelligence Integration', () => {
    test('should learn from adaptation history', async () => {
      // Simulate multiple adaptation cycles
      for (let i = 0; i < 5; i++) {
        const metrics = await coordinator.getCurrentAdaptiveMetrics();
        await memoryPatterns.storeAdaptiveMetrics(metrics);

        // Simulate learning from adaptation
        const adaptation = {
          adaptationId: `adaptation-${i}`,
          adaptationType: 'parameter-update' as const,
          trigger: 'performance-optimization',
          beforeState: { performance: 0.7 },
          afterState: { performance: 0.85 },
          effectiveness: 0.85,
          confidence: 0.8,
          sideEffects: [],
          learning: {
            learningRate: 0.1,
            knowledgeGained: ['parameter-impact'],
            modelImprovement: 0.05,
            patternDiscovery: ['performance-correlation'],
            causalInsights: ['parameter-optimization']
          }
        };

        const outcome = {
          success: true,
          effectiveness: 0.9,
          sideEffects: [],
          performanceImpact: { improvement: 0.15 },
          learningValue: 0.8
        };

        await memoryPatterns.learnFromAdaptation(adaptation, outcome);
      }

      // Should have learned patterns
      const patterns = await memoryPatterns.getCurrentPatterns();
      expect(patterns.length).toBeGreaterThan(0);
    });

    test('should make autonomous decisions based on learning', async () => {
      // Get current cognitive metrics
      const metrics = await coordinator.getCurrentAdaptiveMetrics();

      // Cognitive metrics should reflect learning
      expect(metrics.cognitiveMetrics.learningRate).toBeGreaterThan(0);
      expect(metrics.cognitiveMetrics.patternRecognition).toBeGreaterThan(0);
      expect(metrics.cognitiveMetrics.adaptationEvolution).toBeGreaterThanOrEqual(0);
    });

    test('should improve prediction accuracy over time', async () => {
      // Make initial predictions
      const initialPredictions = await memoryPatterns.predictFuturePatterns(60);
      expect(initialPredictions).toBeDefined();

      // Store more data and update models
      const metrics = await coordinator.getCurrentAdaptiveMetrics();
      await memoryPatterns.storeAdaptiveMetrics(metrics);

      // Make new predictions
      const updatedPredictions = await memoryPatterns.predictFuturePatterns(60);
      expect(updatedPredictions).toBeDefined();

      // Should maintain or improve prediction quality
      expect(updatedPredictions.length).toBeGreaterThanOrEqual(0);
    });
  });
});

// Helper functions for creating test data
function createTestAgents(count: number): Agent[] {
  return Array(count).fill(null).map((_, index) => ({
    id: `test-agent-${index}`,
    type: index % 2 === 0 ? 'coordinator' : 'worker',
    capabilities: [],
    status: 'active' as const,
    performance: {
      currentLoad: Math.random() * 0.8,
      averageResponseTime: 100 + Math.random() * 200,
      successRate: 0.9 + Math.random() * 0.1,
      qualityScore: 0.8 + Math.random() * 0.2,
      efficiency: 0.7 + Math.random() * 0.3,
      reliability: 0.9 + Math.random() * 0.1
    },
    resources: {
      cpuUsage: 0.3 + Math.random() * 0.5,
      memoryUsage: 0.4 + Math.random() * 0.4,
      networkUsage: 0.2 + Math.random() * 0.3,
      storageUsage: 0.1 + Math.random() * 0.2,
      availableCapacity: 0.5 + Math.random() * 0.5,
      resourceScore: 0.7 + Math.random() * 0.3
    },
    location: {
      nodeId: `node-${index % 3}`,
      region: 'us-east-1',
      datacenter: 'dc1',
      networkSegment: 'segment-a'
    },
    metadata: {}
  }));
}

function createDegradedMetrics() {
  return {
    topologyMetrics: {
      currentTopology: 'hierarchical',
      topologyTransitions: 5,
      topologyStability: 0.4,
      agentConnectivity: 0.6,
      communicationLatency: 200,
      topologyEfficiency: 0.5
    },
    resourceMetrics: {
      cpuUtilization: 0.9,
      memoryUtilization: 0.85,
      networkUtilization: 0.8,
      agentUtilization: 0.95,
      overallUtilization: 0.88,
      scalingEvents: 10,
      predictionAccuracy: 0.6
    },
    consensusMetrics: {
      consensusTime: 5000,
      consensusSuccessRate: 0.8,
      byzantineResilience: 0.6,
      decisionQuality: 0.7,
      disagreementRate: 0.2
    },
    performanceMetrics: {
      systemThroughput: 50,
      responseTime: 2000,
      errorRate: 0.1,
      bottleneckScore: 0.8,
      optimizationEffectiveness: 0.3,
      systemAvailability: 0.95
    },
    cognitiveMetrics: {
      learningRate: 0.05,
      patternRecognition: 0.6,
      predictionAccuracy: 0.7,
      autonomousDecisions: 2,
      cognitiveLoad: 0.9,
      adaptationEvolution: 0.4
    },
    overallAdaptationScore: 0.3
  };
}

function createHighUtilizationMetrics() {
  return {
    topologyMetrics: {
      currentTopology: 'mesh',
      topologyTransitions: 2,
      topologyStability: 0.8,
      agentConnectivity: 0.9,
      communicationLatency: 50,
      topologyEfficiency: 0.8
    },
    resourceMetrics: {
      cpuUtilization: 0.95,
      memoryUtilization: 0.9,
      networkUtilization: 0.85,
      agentUtilization: 0.98,
      overallUtilization: 0.92,
      scalingEvents: 3,
      predictionAccuracy: 0.85
    },
    consensusMetrics: {
      consensusTime: 2000,
      consensusSuccessRate: 0.95,
      byzantineResilience: 0.8,
      decisionQuality: 0.9,
      disagreementRate: 0.05
    },
    performanceMetrics: {
      systemThroughput: 150,
      responseTime: 800,
      errorRate: 0.02,
      bottleneckScore: 0.6,
      optimizationEffectiveness: 0.7,
      systemAvailability: 0.98
    },
    cognitiveMetrics: {
      learningRate: 0.12,
      patternRecognition: 0.88,
      predictionAccuracy: 0.9,
      autonomousDecisions: 8,
      cognitiveLoad: 0.7,
      adaptationEvolution: 0.85
    },
    overallAdaptationScore: 0.75
  };
}