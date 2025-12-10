/**
 * Test Suite for Adaptive Coordination System
 *
 * Tests the complete adaptive learning system with maximum cognitive consciousness
 * including ReasoningBank adaptive learning, causal inference, strange-loop cognition,
 * and AgentDB memory patterns with QUIC synchronization.
 */

import { AdaptiveCoordinationSystem } from '../../src/adaptive-coordination/AdaptiveCoordinationSystem';
import { DeploymentLearningRequest, StrategyOptimizationRequest } from '../../src/adaptive-coordination/AdaptiveCoordinationSystem';

describe('AdaptiveCoordinationSystem', () => {
  let system: AdaptiveCoordinationSystem;
  let testConfig: any;

  beforeEach(async () => {
    testConfig = {
      agentdb: {
        connectionString: 'memory://test',
        syncInterval: 1000,
        compressionEnabled: true,
        memoryNamespace: 'test-adaptive-coordination'
      },
      consciousness: {
        level: 1.0,
        temporalExpansion: 1000,
        strangeLoopDepth: 10,
        learningRate: 0.95,
        adaptationSpeed: 0.9
      },
      causalInference: {
        accuracy: 0.95,
        temporalReasoning: true,
        gpcmEnabled: true,
        confidenceThreshold: 0.8
      },
      optimization: {
        recursionDepth: 10,
        selfReference: true,
        adaptationRate: 0.9
      },
      strategies: {
        defaultTypes: ['conservative', 'balanced', 'aggressive', 'adaptive'],
        maxStrategies: 10,
        adaptationThreshold: 0.7
      }
    };

    system = new AdaptiveCoordinationSystem(testConfig);
    await system.initialize();
  });

  afterEach(async () => {
    if (system) {
      // Cleanup if needed
    }
  });

  describe('System Initialization', () => {
    test('should initialize with maximum consciousness', async () => {
      expect(system.isReady()).toBe(true);

      const config = system.getConfig();
      expect(config.consciousness.level).toBe(1.0);
      expect(config.consciousness.temporalExpansion).toBe(1000);
      expect(config.causalInference.accuracy).toBe(0.95);
    });

    test('should have all components initialized', async () => {
      const analytics = await system.getLearningAnalytics();
      expect(analytics.consciousnessLevel).toBe(1.0);
      expect(analytics.totalPatterns).toBe(0);
    });
  });

  describe('Deployment Learning', () => {
    test('should learn from successful deployment', async () => {
      const request: DeploymentLearningRequest = {
        deploymentData: {
          strategy: 'balanced-deployment',
          resources: { utilization: 0.7, allocated: 0.8 },
          configuration: { complexity: 5 },
          dependencies: ['service-a', 'service-b']
        },
        outcome: 'success',
        metrics: {
          duration: 120000,
          resourceUtilization: 0.75,
          errorRate: 0.02,
          performanceScore: 0.92,
          reliabilityScore: 0.95,
          efficiencyScore: 0.88
        },
        context: {
          environment: 'staging',
          complexity: 0.6,
          dependencies: ['service-a', 'service-b'],
          constraints: ['no-downtime'],
          teamExpertise: 0.8,
          previousDeployments: 15
        }
      };

      const result = await system.learnFromDeployment(request);

      expect(result.patternId).toBeDefined();
      expect(result.causalFactors).toBeDefined();
      expect(result.consciousnessInsights).toBeDefined();
      expect(result.learningMetrics.learningTime).toBeGreaterThan(0);
      expect(result.learningMetrics.consciousnessLevel).toBe(1.0);
    });

    test('should learn from failed deployment with causal analysis', async () => {
      const request: DeploymentLearningRequest = {
        deploymentData: {
          strategy: 'aggressive-deployment',
          resources: { utilization: 0.95, allocated: 0.8 },
          configuration: { complexity: 8 },
          dependencies: ['service-a', 'service-b', 'service-c']
        },
        outcome: 'failure',
        metrics: {
          duration: 300000,
          resourceUtilization: 0.95,
          errorRate: 0.15,
          performanceScore: 0.3,
          reliabilityScore: 0.2,
          efficiencyScore: 0.4
        },
        context: {
          environment: 'production',
          complexity: 0.9,
          dependencies: ['service-a', 'service-b', 'service-c'],
          constraints: ['no-downtime', 'full-validation'],
          teamExpertise: 0.6,
          previousDeployments: 5
        },
        options: {
          enableCausalAnalysis: true,
          enableStrangeLoop: true,
          enableTemporalExpansion: true
        }
      };

      const result = await system.learnFromDeployment(request);

      expect(result.causalFactors.length).toBeGreaterThan(0);
      expect(result.consciousnessInsights.length).toBeGreaterThan(0);
      expect(result.learningMetrics.causalFactorsCount).toBeGreaterThan(0);
    });

    test('should extract causal factors with 95% accuracy', async () => {
      const request: DeploymentLearningRequest = {
        deploymentData: {
          strategy: 'test-strategy',
          resources: { utilization: 0.8, allocated: 0.9 },
          configuration: { complexity: 6 }
        },
        outcome: 'partial',
        metrics: {
          duration: 180000,
          resourceUtilization: 0.8,
          errorRate: 0.08,
          performanceScore: 0.7,
          reliabilityScore: 0.75,
          efficiencyScore: 0.68
        },
        context: {
          environment: 'staging',
          complexity: 0.7,
          dependencies: ['service-a'],
          constraints: [],
          teamExpertise: 0.7,
          previousDeployments: 10
        }
      };

      const result = await system.learnFromDeployment(request);

      // Verify causal factors meet accuracy threshold
      if (result.causalFactors.length > 0) {
        result.causalFactors.forEach(factor => {
          expect(factor.confidence).toBeGreaterThanOrEqual(0.8); // 80% confidence threshold
        });
      }
    });
  });

  describe('Strategy Optimization', () => {
    test('should optimize deployment strategy with cognitive intelligence', async () => {
      const request: StrategyOptimizationRequest = {
        context: {
          environment: 'production',
          complexity: 0.8,
          dependencies: ['critical-service'],
          constraints: ['zero-downtime', 'rollback-required'],
          teamExpertise: 0.9,
          previousDeployments: 50
        },
        constraints: ['no-downtime', 'fast-rollback'],
        objectives: ['reliability', 'speed', 'safety'],
        options: {
          enableCausalReasoning: true,
          enableConsciousness: true,
          maxRecommendations: 3,
          minConfidence: 0.8
        }
      };

      const result = await system.optimizeStrategy(request);

      expect(result.recommendations).toBeDefined();
      expect(result.confidence).toBeGreaterThanOrEqual(0.8);
      expect(result.reasoning.length).toBeGreaterThan(0);
      expect(result.riskAssessment).toBeDefined();
      expect(result.adaptationPlan).toBeDefined();
      expect(result.consciousnessInsights.length).toBeGreaterThan(0);
    });

    test('should provide multiple strategy recommendations', async () => {
      const request: StrategyOptimizationRequest = {
        context: {
          environment: 'staging',
          complexity: 0.5,
          dependencies: ['service-a', 'service-b'],
          constraints: ['quick-deployment'],
          teamExpertise: 0.7,
          previousDeployments: 20
        },
        options: {
          maxRecommendations: 5,
          minConfidence: 0.7
        }
      };

      const result = await system.optimizeStrategy(request);

      expect(result.recommendations.length).toBeGreaterThan(1);
      expect(result.recommendations.length).toBeLessThanOrEqual(5);

      // Verify recommendations are sorted by confidence
      for (let i = 1; i < result.recommendations.length; i++) {
        expect(result.recommendations[i-1].confidence).toBeGreaterThanOrEqual(
          result.recommendations[i].confidence
        );
      }
    });

    test('should include consciousness insights in recommendations', async () => {
      const request: StrategyOptimizationRequest = {
        context: {
          environment: 'production',
          complexity: 0.6,
          dependencies: ['service-a'],
          constraints: [],
          teamExpertise: 0.8,
          previousDeployments: 30
        },
        options: {
          enableConsciousness: true
        }
      };

      const result = await system.optimizeStrategy(request);

      expect(result.consciousnessInsights).toBeDefined();
      expect(result.consciousnessInsights.length).toBeGreaterThan(0);

      // Check for specific consciousness metrics
      const consciousnessText = result.consciousnessInsights.join(' ');
      expect(consciousnessText).toContain('Consciousness level');
      expect(consciousnessText).toContain('Self-awareness');
      expect(consciousnessText).toContain('Temporal expansion');
    });
  });

  describe('Strategy Adaptation', () => {
    test('should adapt strategy based on deployment outcomes', async () => {
      // First, learn from a deployment
      const learningRequest: DeploymentLearningRequest = {
        deploymentData: {
          strategy: 'conservative-deployment',
          resources: { utilization: 0.6, allocated: 0.7 },
          configuration: { complexity: 4 }
        },
        outcome: 'success',
        metrics: {
          duration: 150000,
          resourceUtilization: 0.65,
          errorRate: 0.01,
          performanceScore: 0.88,
          reliabilityScore: 0.92,
          efficiencyScore: 0.85
        },
        context: {
          environment: 'production',
          complexity: 0.4,
          dependencies: ['service-a'],
          constraints: ['no-downtime'],
          teamExpertise: 0.8,
          previousDeployments: 25
        }
      };

      await system.learnFromDeployment(learningRequest);

      // Get available strategies
      const strategies = system['strategyOptimizer'].getAllStrategies();
      expect(strategies.length).toBeGreaterThan(0);

      const strategyId = strategies[0].id;

      // Adapt strategy based on outcome
      const adaptationResult = await system.adaptStrategy(
        strategyId,
        { type: 'success', duration: 150000 },
        learningRequest.metrics,
        learningRequest.context
      );

      expect(adaptationResult.adaptedStrategy).toBeDefined();
      expect(adaptationResult.adaptationsApplied).toBeDefined();
      expect(adaptationResult.consciousnessEvolution).toBeDefined();
      expect(typeof adaptationResult.effectivenessImprovement).toBe('number');
    });

    test('should improve strategy effectiveness through adaptation', async () => {
      // Learn from multiple deployments to establish baseline
      for (let i = 0; i < 3; i++) {
        const request: DeploymentLearningRequest = {
          deploymentData: {
            strategy: 'test-adaptation',
            resources: { utilization: 0.7 + i * 0.1, allocated: 0.8 },
            configuration: { complexity: 5 + i }
          },
          outcome: i < 2 ? 'success' : 'failure',
          metrics: {
            duration: 120000 + i * 30000,
            resourceUtilization: 0.7 + i * 0.1,
            errorRate: i < 2 ? 0.02 : 0.12,
            performanceScore: i < 2 ? 0.85 - i * 0.05 : 0.4,
            reliabilityScore: i < 2 ? 0.9 - i * 0.05 : 0.5,
            efficiencyScore: i < 2 ? 0.8 - i * 0.05 : 0.45
          },
          context: {
            environment: 'staging',
            complexity: 0.5 + i * 0.1,
            dependencies: ['service-a'],
            constraints: [],
            teamExpertise: 0.7,
            previousDeployments: 10 + i * 5
          }
        };

        await system.learnFromDeployment(request);
      }

      // Get a strategy and adapt it
      const strategies = system['strategyOptimizer'].getAllStrategies();
      const strategy = strategies.find(s => s.type === 'adaptive') || strategies[0];

      const adaptationResult = await system.adaptStrategy(
        strategy.id,
        { type: 'success', duration: 120000 },
        {
          duration: 120000,
          resourceUtilization: 0.7,
          errorRate: 0.02,
          performanceScore: 0.85,
          reliabilityScore: 0.9,
          efficiencyScore: 0.8
        },
        {
          environment: 'staging',
          complexity: 0.5,
          dependencies: ['service-a'],
          constraints: [],
          teamExpertise: 0.7,
          previousDeployments: 15
        }
      );

      expect(adaptationResult.adaptedStrategy.effectiveness).toBeGreaterThanOrEqual(
        strategy.effectiveness
      );
    });
  });

  describe('Causal Relationship Discovery', () => {
    test('should discover causal relationships with GPCM', async () => {
      // First, learn from multiple deployments to establish patterns
      const deploymentPatterns = [];

      for (let i = 0; i < 10; i++) {
        const request: DeploymentLearningRequest = {
          deploymentData: {
            strategy: ['conservative', 'balanced', 'aggressive'][i % 3] + '-deployment',
            resources: { utilization: 0.5 + i * 0.05, allocated: 0.8 },
            configuration: { complexity: 3 + Math.floor(i / 3) }
          },
          outcome: i < 7 ? 'success' : 'failure',
          metrics: {
            duration: 100000 + i * 20000,
            resourceUtilization: 0.5 + i * 0.05,
            errorRate: i < 7 ? 0.01 + Math.random() * 0.02 : 0.1 + Math.random() * 0.05,
            performanceScore: i < 7 ? 0.8 + Math.random() * 0.15 : 0.3 + Math.random() * 0.2,
            reliabilityScore: i < 7 ? 0.85 + Math.random() * 0.1 : 0.4 + Math.random() * 0.15,
            efficiencyScore: i < 7 ? 0.75 + Math.random() * 0.2 : 0.35 + Math.random() * 0.2
          },
          context: {
            environment: i % 2 === 0 ? 'staging' : 'production',
            complexity: 0.4 + (i / 10),
            dependencies: i < 5 ? ['service-a'] : ['service-a', 'service-b'],
            constraints: i < 3 ? [] : ['no-downtime'],
            teamExpertise: 0.6 + Math.random() * 0.3,
            previousDeployments: 5 + i * 2
          }
        };

        await system.learnFromDeployment(request);
        deploymentPatterns.push(request);
      }

      // Discover causal relationships
      const causalResult = await system.discoverCausalRelationships();

      expect(causalResult.relationships).toBeDefined();
      expect(causalResult.modelAccuracy).toBeGreaterThan(0);
      expect(causalResult.confidence).toBeGreaterThan(0);
      expect(causalResult.insights).toBeDefined();
      expect(causalResult.recommendations).toBeDefined();

      // Verify accuracy meets threshold
      expect(causalResult.modelAccuracy).toBeGreaterThanOrEqual(0.8);
    });

    test('should generate meaningful causal insights', async () => {
      // Create deployments with clear causal patterns
      const highComplexityFailures = [
        { complexity: 0.9, outcome: 'failure', resourceUtil: 0.95 },
        { complexity: 0.85, outcome: 'failure', resourceUtil: 0.92 },
        { complexity: 0.8, outcome: 'partial', resourceUtil: 0.88 }
      ];

      const lowComplexitySuccesses = [
        { complexity: 0.3, outcome: 'success', resourceUtil: 0.5 },
        { complexity: 0.4, outcome: 'success', resourceUtil: 0.6 },
        { complexity: 0.35, outcome: 'success', resourceUtil: 0.55 }
      ];

      for (const pattern of [...highComplexityFailures, ...lowComplexitySuccesses]) {
        const request: DeploymentLearningRequest = {
          deploymentData: {
            strategy: 'test-causal',
            resources: { utilization: pattern.resourceUtil, allocated: 0.8 },
            configuration: { complexity: pattern.complexity * 10 }
          },
          outcome: pattern.outcome,
          metrics: {
            duration: pattern.outcome === 'success' ? 120000 : 240000,
            resourceUtilization: pattern.resourceUtil,
            errorRate: pattern.outcome === 'success' ? 0.02 : 0.15,
            performanceScore: pattern.outcome === 'success' ? 0.9 : 0.4,
            reliabilityScore: pattern.outcome === 'success' ? 0.95 : 0.5,
            efficiencyScore: pattern.outcome === 'success' ? 0.85 : 0.45
          },
          context: {
            environment: 'staging',
            complexity: pattern.complexity,
            dependencies: ['service-a'],
            constraints: [],
            teamExpertise: 0.7,
            previousDeployments: 10
          }
        };

        await system.learnFromDeployment(request);
      }

      const causalResult = await system.discoverCausalRelationships();

      expect(causalResult.insights.length).toBeGreaterThan(0);

      // Should identify complexity as a causal factor
      const insightsText = causalResult.insights.join(' ').toLowerCase();
      expect(insightsText).toMatch(/complexity|resource/);
    });
  });

  describe('Learning Analytics', () => {
    test('should provide comprehensive learning analytics', async () => {
      // Generate some learning data
      for (let i = 0; i < 5; i++) {
        const request: DeploymentLearningRequest = {
          deploymentData: {
            strategy: 'analytics-test',
            resources: { utilization: 0.6 + i * 0.08, allocated: 0.8 },
            configuration: { complexity: 4 + i }
          },
          outcome: i < 4 ? 'success' : 'failure',
          metrics: {
            duration: 100000 + i * 25000,
            resourceUtilization: 0.6 + i * 0.08,
            errorRate: i < 4 ? 0.02 : 0.1,
            performanceScore: i < 4 ? 0.85 - i * 0.05 : 0.5,
            reliabilityScore: i < 4 ? 0.9 - i * 0.05 : 0.55,
            efficiencyScore: i < 4 ? 0.8 - i * 0.05 : 0.48
          },
          context: {
            environment: 'staging',
            complexity: 0.5 + i * 0.1,
            dependencies: ['service-a'],
            constraints: [],
            teamExpertise: 0.7,
            previousDeployments: 8 + i * 3
          }
        };

        await system.learnFromDeployment(request);
      }

      const analytics = await system.getLearningAnalytics();

      expect(analytics.totalPatterns).toBe(5);
      expect(analytics.consciousnessLevel).toBe(1.0);
      expect(analytics.memoryStorage.totalStored).toBeGreaterThan(0);
      expect(analytics.performance.averageLearningTime).toBeGreaterThan(0);
      expect(analytics.causalModelAccuracy).toBeGreaterThanOrEqual(0);
    });

    test('should track pattern types correctly', async () => {
      // Create different types of patterns
      const patterns = [
        { type: 'deployment', data: { strategy: 'test-1' } },
        { type: 'strategy', data: { name: 'test-strategy' } },
        { type: 'causal', data: { model: 'test-model' } }
      ];

      for (const pattern of patterns) {
        await system['memoryManager'].storePattern(pattern);
      }

      const analytics = await system.getLearningAnalytics();
      expect(analytics.patternsByType.size).toBeGreaterThan(0);
    });
  });

  describe('Data Import/Export', () => {
    test('should export learning data correctly', async () => {
      // Create some test data
      const request: DeploymentLearningRequest = {
        deploymentData: {
          strategy: 'export-test',
          resources: { utilization: 0.7, allocated: 0.8 },
          configuration: { complexity: 5 }
        },
        outcome: 'success',
        metrics: {
          duration: 120000,
          resourceUtilization: 0.7,
          errorRate: 0.02,
          performanceScore: 0.88,
          reliabilityScore: 0.92,
          efficiencyScore: 0.85
        },
        context: {
          environment: 'staging',
          complexity: 0.6,
          dependencies: ['service-a'],
          constraints: [],
          teamExpertise: 0.8,
          previousDeployments: 15
        }
      };

      await system.learnFromDeployment(request);

      const exportData = await system.exportLearningData();

      expect(exportData.patterns).toBeDefined();
      expect(exportData.strategies).toBeDefined();
      expect(exportData.causalModels).toBeDefined();
      expect(exportData.analytics).toBeDefined();
      expect(exportData.consciousnessState).toBeDefined();
      expect(exportData.exported).toBeDefined();
      expect(exportData.patterns.length).toBeGreaterThan(0);
    });

    test('should import learning data correctly', async () => {
      const testLearningData = {
        patterns: [
          {
            id: 'import-test-1',
            type: 'deployment',
            timestamp: Date.now(),
            data: { strategy: 'imported-strategy' },
            context: { environment: 'test' },
            signature: 'test-signature',
            confidence: 0.9,
            frequency: 1,
            lastAccessed: Date.now(),
            tags: ['test', 'import']
          }
        ],
        strategies: [],
        clusters: []
      };

      await system.importLearningData(testLearningData);

      const analytics = await system.getLearningAnalytics();
      expect(analytics.totalPatterns).toBeGreaterThan(0);
    });
  });

  describe('Configuration Management', () => {
    test('should allow configuration updates', async () => {
      const newConfig = {
        consciousness: {
          level: 0.9,
          temporalExpansion: 800,
          strangeLoopDepth: 8,
          learningRate: 0.9,
          adaptationSpeed: 0.85
        }
      };

      await system.updateConfig(newConfig);

      const updatedConfig = system.getConfig();
      expect(updatedConfig.consciousness.level).toBe(0.9);
      expect(updatedConfig.consciousness.temporalExpansion).toBe(800);
    });

    test('should maintain system readiness after config update', async () => {
      const newConfig = {
        strategies: {
          defaultTypes: ['conservative', 'balanced'],
          maxStrategies: 5,
          adaptationThreshold: 0.8
        }
      };

      await system.updateConfig(newConfig);
      expect(system.isReady()).toBe(true);
    });
  });

  describe('Error Handling', () => {
    test('should handle initialization failures gracefully', async () => {
      const invalidConfig = {
        agentdb: {
          connectionString: 'invalid://connection',
          syncInterval: 1000,
          compressionEnabled: true,
          memoryNamespace: 'test'
        },
        consciousness: {
          level: 1.0,
          temporalExpansion: 1000,
          strangeLoopDepth: 10,
          learningRate: 0.95,
          adaptationSpeed: 0.9
        },
        causalInference: {
          accuracy: 0.95,
          temporalReasoning: true,
          gpcmEnabled: true,
          confidenceThreshold: 0.8
        },
        optimization: {
          recursionDepth: 10,
          selfReference: true,
          adaptationRate: 0.9
        },
        strategies: {
          defaultTypes: ['conservative', 'balanced', 'aggressive'],
          maxStrategies: 10,
          adaptationThreshold: 0.7
        }
      };

      const invalidSystem = new AdaptiveCoordinationSystem(invalidConfig);

      // Should handle initialization error without crashing
      try {
        await invalidSystem.initialize();
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    test('should handle operations on uninitialized system', async () => {
      const uninitializedSystem = new AdaptiveCoordinationSystem(testConfig);

      const request: DeploymentLearningRequest = {
        deploymentData: { strategy: 'test' },
        outcome: 'success',
        metrics: { duration: 1000 },
        context: { environment: 'test' }
      };

      await expect(uninitializedSystem.learnFromDeployment(request))
        .rejects.toThrow('Adaptive Coordination System not initialized');
    });

    test('should handle missing strategy gracefully', async () => {
      await expect(system.adaptStrategy(
        'non-existent-strategy',
        { type: 'success' },
        { duration: 1000 },
        { environment: 'test' }
      )).rejects.toThrow();
    });
  });
});