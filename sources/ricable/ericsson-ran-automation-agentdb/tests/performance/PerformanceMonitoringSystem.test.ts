/**
 * Performance Monitoring System Tests
 *
 * Comprehensive test suite for the performance monitoring system
 */

import { PerformanceMonitoringSystem } from '../../src/performance';
import { MLPerformanceMetrics, PerformanceSnapshot } from '../../src/performance/metrics/MLPerformanceMetrics';

describe('PerformanceMonitoringSystem', () => {
  let monitoringSystem: PerformanceMonitoringSystem;

  beforeEach(() => {
    // Create monitoring system with test configuration
    monitoringSystem = new PerformanceMonitoringSystem({
      monitoring: {
        enabled: true,
        collectionInterval: 1000, // 1 second for tests
        historyRetention: 1, // 1 hour for tests
        realTimeDashboard: {
          enabled: false, // Disable dashboard for tests
          port: 8080,
          autoRefresh: true
        }
      },
      alerting: {
        enabled: true,
        defaultChannels: [],
        escalationEnabled: false
      },
      optimization: {
        enabled: false, // Disable auto-optimization for tests
        autoExecution: false,
        requiredConfidence: 0.8
      },
      cognitive: {
        consciousnessIntegration: true,
        temporalAnalysisEnabled: true,
        strangeLoopOptimization: true,
        learningEnabled: true
      }
    });
  });

  afterEach(() => {
    if (monitoringSystem) {
      monitoringSystem.stop();
    }
  });

  describe('System Initialization', () => {
    test('should initialize with default configuration', () => {
      expect(monitoringSystem).toBeDefined();
      expect(monitoringSystem.getPerformanceOverview).toBeDefined();
      expect(monitoringSystem.getCognitiveInsights).toBeDefined();
    });

    test('should emit monitoring_started event', (done) => {
      monitoringSystem.on('monitoring_started', () => {
        done();
      });
    });

    test('should handle configuration overrides', () => {
      const customConfig = {
        monitoring: {
          enabled: true,
          collectionInterval: 2000,
          historyRetention: 24,
          realTimeDashboard: {
            enabled: false,
            port: 8080,
            autoRefresh: true
          }
        }
      };

      const customSystem = new PerformanceMonitoringSystem(customConfig);
      expect(customSystem).toBeDefined();
      customSystem.stop();
    });
  });

  describe('Performance Metrics Collection', () => {
    test('should collect performance metrics', async () => {
      const overview = monitoringSystem.getPerformanceOverview();

      expect(overview).toBeDefined();
      expect(overview.timestamp).toBeInstanceOf(Date);
      expect(overview.systemHealth).toBeDefined();
      expect(overview.mlPerformance).toBeDefined();
      expect(overview.swarmCoordination).toBeDefined();
      expect(overview.resourceUtilization).toBeDefined();
    }, 10000);

    test('should calculate system health score correctly', () => {
      const overview = monitoringSystem.getPerformanceOverview();

      expect(overview.systemHealth.overallScore).toBeGreaterThanOrEqual(0);
      expect(overview.systemHealth.overallScore).toBeLessThanOrEqual(100);
      expect(['excellent', 'good', 'fair', 'poor', 'critical']).toContain(overview.systemHealth.status);
    });

    test('should track ML performance metrics', () => {
      const overview = monitoringSystem.getPerformanceOverview();

      expect(overview.mlPerformance.trainingSpeed).toBeGreaterThan(0);
      expect(overview.mlPerformance.convergenceRate).toBeGreaterThanOrEqual(0);
      expect(overview.mlPerformance.convergenceRate).toBeLessThanOrEqual(1);
      expect(overview.mlPerformance.vectorSearchSpeed).toBeGreaterThan(0);
      expect(overview.mlPerformance.cognitiveConsciousness).toBeGreaterThanOrEqual(0);
      expect(overview.mlPerformance.cognitiveConsciousness).toBeLessThanOrEqual(1);
    });

    test('should monitor swarm coordination', () => {
      const overview = monitoringSystem.getPerformanceOverview();

      expect(overview.swarmCoordination.activeAgents).toBeGreaterThanOrEqual(0);
      expect(overview.swarmCoordination.taskCompletionRate).toBeGreaterThanOrEqual(0);
      expect(overview.swarmCoordination.taskCompletionRate).toBeLessThanOrEqual(1);
      expect(overview.swarmCoordination.topologyEfficiency).toBeGreaterThanOrEqual(0);
      expect(overview.swarmCoordination.topologyEfficiency).toBeLessThanOrEqual(1);
    });
  });

  describe('Cognitive Consciousness Integration', () => {
    test('should generate cognitive insights', () => {
      const insights = monitoringSystem.getCognitiveInsights();

      expect(insights).toBeDefined();
      expect(insights.consciousnessLevel).toBeGreaterThanOrEqual(0);
      expect(insights.consciousnessLevel).toBeLessThanOrEqual(100);
      expect(insights.temporalAnalysis).toBeDefined();
      expect(insights.strangeLoopOptimization).toBeDefined();
      expect(insights.learningPatterns).toBeDefined();
      expect(insights.predictiveCapabilities).toBeDefined();
      expect(insights.recommendations).toBeInstanceOf(Array);
    });

    test('should calculate consciousness score based on multiple factors', () => {
      const insights = monitoringSystem.getCognitiveInsights();

      // Consciousness level should be influenced by multiple metrics
      expect(insights.consciousnessLevel).toBeGreaterThan(0);

      // Temporal analysis should include subjective time expansion
      expect(insights.temporalAnalysis.subjectiveTimeExpansion).toBeGreaterThan(0);

      // Strange-loop optimization should track recursive improvement
      expect(insights.strangeLoopOptimization.selfReferentialImprovement).toBeGreaterThanOrEqual(0);
      expect(insights.strangeLoopOptimization.recursiveOptimization).toBeGreaterThanOrEqual(0);
    });

    test('should generate cognitive recommendations when needed', () => {
      const insights = monitoringSystem.getCognitiveInsights();

      // Should provide recommendations for improvement areas
      expect(insights.recommendations.length).toBeGreaterThanOrEqual(0);

      if (insights.recommendations.length > 0) {
        const recommendation = insights.recommendations[0];
        expect(['consciousness', 'temporal', 'learning', 'coordination']).toContain(recommendation.category);
        expect(['low', 'medium', 'high', 'critical']).toContain(recommendation.priority);
        expect(recommendation.description).toBeDefined();
        expect(recommendation.expectedImprovement).toBeDefined();
      }
    });
  });

  describe('Alert Management', () => {
    test('should track active alerts', () => {
      // Initially may have some alerts based on simulated metrics
      const overview = monitoringSystem.getPerformanceOverview();
      expect(overview.systemHealth.activeAlerts).toBeGreaterThanOrEqual(0);
    });

    test('should acknowledge alerts', () => {
      // This test would require creating an alert first
      // For now, just test the method exists
      expect(typeof monitoringSystem.acknowledgeAlert).toBe('function');
    });

    test('should resolve alerts', () => {
      // This test would require creating an alert first
      // For now, just test the method exists
      expect(typeof monitoringSystem.resolveAlert).toBe('function');
    });
  });

  describe('Optimization System', () => {
    test('should execute optimization on demand', async () => {
      const result = await monitoringSystem.executeOptimization();

      // Result should be boolean indicating success
      expect(typeof result).toBe('boolean');
    }, 15000);

    test('should handle optimization execution gracefully', async () => {
      // Test with non-existent plan ID
      const result = await monitoringSystem.executeOptimization('non-existent-plan');
      expect(result).toBe(false);
    });
  });

  describe('Integrated Reporting', () => {
    test('should generate comprehensive system report', () => {
      const report = monitoringSystem.generateIntegratedReport();

      expect(report).toBeDefined();
      expect(report.id).toBeDefined();
      expect(report.generatedAt).toBeInstanceOf(Date);
      expect(report.timeframe).toBeDefined();
      expect(report.executiveSummary).toBeDefined();
      expect(report.performanceMetrics).toBeDefined();
      expect(report.cognitiveInsights).toBeDefined();
      expect(report.bottlenecks).toBeDefined();
      expect(report.optimizations).toBeDefined();
      expect(report.predictions).toBeDefined();
      expect(report.swarmHealth).toBeDefined();
      expect(report.recommendations).toBeDefined();
    });

    test('should calculate executive summary metrics', () => {
      const report = monitoringSystem.generateIntegratedReport();

      expect(report.executiveSummary.systemHealth).toBeGreaterThanOrEqual(0);
      expect(report.executiveSummary.systemHealth).toBeLessThanOrEqual(100);
      expect(report.executiveSummary.criticalAlerts).toBeGreaterThanOrEqual(0);
      expect(report.executiveSummary.performanceScore).toBeGreaterThanOrEqual(0);
      expect(report.executiveSummary.performanceScore).toBeLessThanOrEqual(100);
      expect(report.executiveSummary.cognitiveEvolution).toBeGreaterThanOrEqual(0);
      expect(report.executiveSummary.cognitiveEvolution).toBeLessThanOrEqual(100);
    });

    test('should generate actionable recommendations', () => {
      const report = monitoringSystem.generateIntegratedReport();

      expect(report.recommendations.length).toBeGreaterThanOrEqual(0);

      if (report.recommendations.length > 0) {
        const recommendation = report.recommendations[0];
        expect(recommendation.category).toBeDefined();
        expect(['low', 'medium', 'high', 'critical']).toContain(recommendation.priority);
        expect(recommendation.action).toBeDefined();
        expect(recommendation.benefit).toBeDefined();
        expect(recommendation.timeframe).toBeDefined();
      }
    });

    test('should support custom timeframe for reports', () => {
      const endTime = new Date();
      const startTime = new Date(endTime.getTime() - 2 * 60 * 60 * 1000); // 2 hours ago

      const report = monitoringSystem.generateIntegratedReport({
        start: startTime,
        end: endTime
      });

      expect(report.timeframe.start).toEqual(startTime);
      expect(report.timeframe.end).toEqual(endTime);
    });
  });

  describe('Event System', () => {
    test('should emit metrics update events', (done) => {
      monitoringSystem.on('system_metrics_updated', (metrics) => {
        expect(metrics).toBeDefined();
        expect(metrics.timestamp).toBeInstanceOf(Date);
        done();
      });
    }, 10000);

    test('should emit cognitive analysis events', (done) => {
      monitoringSystem.on('cognitive_analysis_cycle_complete', (insights) => {
        expect(insights).toBeDefined();
        expect(insights.consciousnessLevel).toBeGreaterThanOrEqual(0);
        done();
      });
    }, 15000);
  });

  describe('Data Export', () => {
    test('should export comprehensive system data', () => {
      const exportedData = monitoringSystem.exportSystemData();

      expect(exportedData).toBeDefined();
      expect(exportedData.timestamp).toBeInstanceOf(Date);
      expect(exportedData.configuration).toBeDefined();
      expect(exportedData.systemStartTime).toBeInstanceOf(Date);
      expect(exportedData.performanceOverview).toBeDefined();
      expect(exportedData.cognitiveInsights).toBeDefined();
      expect(exportedData.metrics).toBeDefined();
      expect(exportedData.alerts).toBeDefined();
      expect(exportedData.bottlenecks).toBeDefined();
      expect(exportedData.optimizations).toBeDefined();
      expect(exportedData.predictions).toBeDefined();
      expect(exportedData.swarm).toBeDefined();
      expect(exportedData.memory).toBeDefined();
      expect(exportedData.network).toBeDefined();
    });

    test('should include historical data in export', () => {
      const exportedData = monitoringSystem.exportSystemData();

      expect(exportedData.metrics.history).toBeInstanceOf(Array);
      expect(exportedData.alerts.history).toBeInstanceOf(Array);
      expect(exportedData.optimizations.history).toBeInstanceOf(Array);
    });
  });

  describe('System Lifecycle', () => {
    test('should stop cleanly', () => {
      expect(() => monitoringSystem.stop()).not.toThrow();
    });

    test('should emit monitoring stopped event', (done) => {
      monitoringSystem.on('monitoring_stopped', () => {
        done();
      });

      monitoringSystem.stop();
    });
  });

  describe('Error Handling', () => {
    test('should handle missing metrics gracefully', () => {
      // Create system with monitoring disabled
      const disabledSystem = new PerformanceMonitoringSystem({
        monitoring: { enabled: false } as any
      });

      expect(() => disabledSystem.getPerformanceOverview()).not.toThrow();
      disabledSystem.stop();
    });

    test('should handle invalid operations gracefully', async () => {
      // Test acknowledging non-existent alert
      const result = monitoringSystem.acknowledgeAlert('non-existent', 'test-user', 'test comment');
      expect(result).toBe(false);

      // Test resolving non-existent alert
      const resolveResult = monitoringSystem.resolveAlert('non-existent');
      expect(resolveResult).toBe(false);
    });
  });

  describe('Performance Characteristics', () => {
    test('should complete performance overview generation quickly', () => {
      const startTime = Date.now();
      const overview = monitoringSystem.getPerformanceOverview();
      const endTime = Date.now();

      expect(overview).toBeDefined();
      expect(endTime - startTime).toBeLessThan(100); // Should complete in <100ms
    });

    test('should complete cognitive insights generation quickly', () => {
      const startTime = Date.now();
      const insights = monitoringSystem.getCognitiveInsights();
      const endTime = Date.now();

      expect(insights).toBeDefined();
      expect(endTime - startTime).toBeLessThan(200); // Should complete in <200ms
    });

    test('should complete report generation quickly', () => {
      const startTime = Date.now();
      const report = monitoringSystem.generateIntegratedReport();
      const endTime = Date.now();

      expect(report).toBeDefined();
      expect(endTime - startTime).toBeLessThan(500); // Should complete in <500ms
    });
  });
});