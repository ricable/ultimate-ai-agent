/**
 * Cognitive RAN Performance Monitoring Orchestrator
 * Coordinates all performance monitoring components and provides unified interface
 */

import { EventEmitter } from 'events';
import { PerformanceCollector } from '../monitoring/PerformanceCollector';
import { BottleneckDetector } from '../bottlenecks/BottleneckDetector';
import { RealTimeDashboard } from '../dashboard/RealTimeDashboard';
import { CognitiveAnalytics } from '../analytics/CognitiveAnalytics';
import { PerformanceReporter } from '../reporting/PerformanceReporter';
import { AgentDBMonitor } from '../../integration/agentdb/AgentDBMonitor';
import {
  SystemMetrics,
  CognitiveMetrics,
  AgentDBMetrics,
  PerformanceAnomaly,
  Bottleneck,
  SystemHealth
} from '../../types/performance';

export class PerformanceOrchestrator extends EventEmitter {
  private components: {
    collector: PerformanceCollector;
    bottleneckDetector: BottleneckDetector;
    dashboard: RealTimeDashboard;
    cognitiveAnalytics: CognitiveAnalytics;
    reporter: PerformanceReporter;
    agentdbMonitor: AgentDBMonitor;
  };

  private isRunning = false;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private coordinationInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.components = this.initializeComponents();
    this.setupComponentCoordination();
  }

  /**
   * Initialize all monitoring components
   */
  private initializeComponents(): any {
    return {
      collector: new PerformanceCollector(),
      bottleneckDetector: new BottleneckDetector(),
      dashboard: new RealTimeDashboard(),
      cognitiveAnalytics: new CognitiveAnalytics(),
      reporter: new PerformanceReporter(),
      agentdbMonitor: new AgentDBMonitor()
    };
  }

  /**
   * Setup coordination between components
   */
  private setupComponentCoordination(): void {
    // Collector -> Detector (metrics flow)
    this.components.collector.on('metrics:collected', (metrics) => {
      this.components.bottleneckDetector.updateMetrics('system', metrics.system);
      this.components.bottleneckDetector.updateMetrics('cognitive', metrics.cognitive);
      this.components.bottleneckDetector.updateMetrics('agentdb', metrics.agentdb);

      // Update cognitive analytics
      if (metrics.cognitive) {
        this.components.cognitiveAnalytics.updateCognitiveMetrics(metrics.cognitive);
      }
    });

    // Detector -> Dashboard (alerts flow)
    this.components.bottleneckDetector.on('bottleneck:detected', (bottleneck) => {
      this.emit('bottleneck:detected', bottleneck);
    });

    this.components.bottleneckDetector.on('anomaly:detected', (anomaly) => {
      this.emit('anomaly:detected', anomaly);
    });

    // AgentDB Monitor -> Dashboard (AgentDB metrics flow)
    this.components.agentdbMonitor.on('metrics:collected', (metrics) => {
      this.emit('agentdb:metrics', metrics);
    });

    this.components.agentdbMonitor.on('alert', (alert) => {
      this.emit('agentdb:alert', alert);
    });

    this.components.agentdbMonitor.on('quic:health_issue', (issue) => {
      this.emit('quic:health_issue', issue);
    });

    // Cognitive Analytics -> Dashboard (insights flow)
    this.components.cognitiveAnalytics.on('analysis:completed', (analysis) => {
      this.emit('cognitive:analysis', analysis);
    });

    // Reporter -> Orchestrator (reports flow)
    this.components.reporter.on('report:generated', (report) => {
      this.emit('report:generated', report);
    });

    // Component error handling
    Object.values(this.components).forEach(component => {
      component.on('error', (error) => {
        this.emit('component:error', { component: component.constructor.name, error });
      });
    });
  }

  /**
   * Start the complete performance monitoring system
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.log('⚠️ Performance monitoring already running');
      return;
    }

    console.log('🚀 Starting Cognitive RAN Performance Monitoring System...');

    try {
      // Start all components
      await Promise.all([
        this.components.collector.start(),
        this.components.bottleneckDetector.start(),
        this.components.dashboard.start(),
        this.components.cognitiveAnalytics.start(),
        this.components.reporter.start(),
        this.components.agentdbMonitor.start()
      ]);

      // Start health checks
      this.startHealthChecks();

      // Start coordination tasks
      this.startCoordinationTasks();

      this.isRunning = true;
      this.emit('started');

      console.log('✅ Cognitive RAN Performance Monitoring System started successfully');
      console.log('📊 Real-time dashboard active with <1s updates');
      console.log('🔍 Bottleneck detection active with AgentDB patterns');
      console.log('🧠 Cognitive analytics monitoring temporal reasoning');
      console.log('📈 Automated reporting scheduled');
      console.log('🗄️ AgentDB QUIC sync monitoring active (<1ms target)');

    } catch (error) {
      console.error('❌ Failed to start performance monitoring:', error);
      this.emit('error', error);
      throw error;
    }
  }

  /**
   * Stop the performance monitoring system
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      console.log('⚠️ Performance monitoring not running');
      return;
    }

    console.log('🛑 Stopping Cognitive RAN Performance Monitoring System...');

    try {
      // Stop health checks
      if (this.healthCheckInterval) {
        clearInterval(this.healthCheckInterval);
        this.healthCheckInterval = null;
      }

      // Stop coordination tasks
      if (this.coordinationInterval) {
        clearInterval(this.coordinationInterval);
        this.coordinationInterval = null;
      }

      // Stop all components
      await Promise.all([
        this.components.collector.stop(),
        this.components.bottleneckDetector.stop(),
        this.components.dashboard.stop(),
        this.components.cognitiveAnalytics.stop(),
        this.components.reporter.stop(),
        this.components.agentdbMonitor.stop()
      ]);

      this.isRunning = false;
      this.emit('stopped');

      console.log('✅ Cognitive RAN Performance Monitoring System stopped');

    } catch (error) {
      console.error('❌ Error stopping performance monitoring:', error);
      this.emit('error', error);
    }
  }

  /**
   * Start health checks for all components
   */
  private startHealthChecks(): void {
    this.healthCheckInterval = setInterval(() => {
      this.performSystemHealthCheck();
    }, 30000); // Every 30 seconds
  }

  /**
   * Start coordination tasks
   */
  private startCoordinationTasks(): void {
    this.coordinationInterval = setInterval(() => {
      this.performCoordinationTasks();
    }, 60000); // Every minute
  }

  /**
   * Perform system health check
   */
  private async performSystemHealthCheck(): Promise<void> {
    try {
      const health = await this.getSystemHealth();
      this.emit('health:check', health);

      // Check for critical issues
      if (health.overall === 'critical' || health.overall === 'down') {
        this.emit('health:critical', health);
        console.error('🚨 CRITICAL: System health is', health.overall);
      }

    } catch (error) {
      console.error('❌ Error during health check:', error);
    }
  }

  /**
   * Perform coordination tasks between components
   */
  private async performCoordinationTasks(): Promise<void> {
    try {
      // Sync metrics between components
      await this.syncComponentMetrics();

      // Trigger analysis based on current state
      await this.triggerConditionalAnalysis();

      // Optimize component performance
      await this.optimizeComponentPerformance();

    } catch (error) {
      console.error('❌ Error during coordination tasks:', error);
    }
  }

  /**
   * Sync metrics between components
   */
  private async syncComponentMetrics(): Promise<void> {
    // Get latest metrics from collector
    const systemMetrics = this.components.collector.getMetrics('system', 1)[0];
    const cognitiveMetrics = this.components.collector.getMetrics('cognitive', 1)[0];
    const agentdbMetrics = this.components.collector.getMetrics('agentdb', 1)[0];

    // Update bottleneck detector with AgentDB metrics from monitor
    const agentdbMonitorMetrics = this.components.agentdbMonitor.getCurrentMetrics();
    if (agentdbMonitorMetrics) {
      this.components.bottleneckDetector.updateMetrics('agentdb', agentdbMonitorMetrics);
    }

    // Update cognitive analytics with latest data
    if (cognitiveMetrics) {
      this.components.cognitiveAnalytics.updateCognitiveMetrics(cognitiveMetrics);
    }

    // Emit coordinated metrics
    this.emit('metrics:coordinated', {
      system: systemMetrics,
      cognitive: cognitiveMetrics,
      agentdb: agentdbMetrics,
      agentdbMonitor: agentdbMonitorMetrics,
      timestamp: new Date()
    });
  }

  /**
   * Trigger conditional analysis based on system state
   */
  private async triggerConditionalAnalysis(): Promise<void> {
    const health = await this.getSystemHealth();

    // Trigger deep analysis if health is degraded
    if (health.overall === 'degraded' || health.overall === 'critical') {
      this.emit('analysis:triggered', { reason: 'health_degraded', health });

      // Generate incident report if critical
      if (health.overall === 'critical') {
        const incidentReport = await this.components.reporter.generateReport('incident');
        this.emit('incident:report', incidentReport);
      }
    }

    // Check for active bottlenecks and trigger optimization
    const activeBottlenecks = this.components.bottleneckDetector.getActiveBottlenecks();
    if (activeBottlenecks.length > 0) {
      this.emit('bottlenecks:active', activeBottlenecks);
    }
  }

  /**
   * Optimize component performance
   */
  private async optimizeComponentPerformance(): Promise<void> {
    // Get performance insights from cognitive analytics
    const cognitiveReport = this.components.cognitiveAnalytics.getCognitiveReport();

    // Apply optimizations based on insights
    if (cognitiveReport.summary.overallCognitiveScore < 70) {
      this.emit('optimization:recommended', {
        area: 'cognitive_performance',
        score: cognitiveReport.summary.overallCognitiveScore,
        recommendations: cognitiveReport.summary.recommendations
      });
    }

    // Check AgentDB performance and optimize if needed
    const agentdbHealth = this.components.agentdbMonitor.getHealthReport();
    if (agentdbHealth.healthScore < 80) {
      this.emit('optimization:recommended', {
        area: 'agentdb_performance',
        score: agentdbHealth.healthScore,
        recommendations: agentdbHealth.recommendations
      });
    }
  }

  /**
   * Get comprehensive system health
   */
  public async getSystemHealth(): Promise<SystemHealth> {
    try {
      // Get health from individual components
      const collectorHealth = this.components.collector.getHealthSummary();
      const agentdbHealth = this.components.agentdbMonitor.getHealthReport();
      const cognitiveReport = this.components.cognitiveAnalytics.getCognitiveReport();

      // Combine health assessments
      let overallScore = 0;
      let componentCount = 0;

      if (collectorHealth && typeof collectorHealth.score === 'number') {
        overallScore += collectorHealth.score;
        componentCount++;
      }

      if (agentdbHealth && typeof agentdbHealth.healthScore === 'number') {
        overallScore += agentdbHealth.healthScore;
        componentCount++;
      }

      if (cognitiveReport && cognitiveReport.summary && typeof cognitiveReport.summary.overallCognitiveScore === 'number') {
        overallScore += cognitiveReport.summary.overallCognitiveScore;
        componentCount++;
      }

      const finalScore = componentCount > 0 ? overallScore / componentCount : 0;

      let overall: 'healthy' | 'degraded' | 'critical' | 'down' = 'healthy';
      if (finalScore < 30) overall = 'critical';
      else if (finalScore < 60) overall = 'degraded';

      return {
        overall,
        score: Math.round(finalScore),
        checks: [
          {
            component: 'Performance Collector',
            status: collectorHealth?.score > 70 ? 'healthy' : 'warning',
            lastCheck: new Date(),
            responseTime: 50,
            details: collectorHealth || {},
            dependencies: ['system-metrics']
          },
          {
            component: 'AgentDB Monitor',
            status: agentdbHealth?.healthScore > 70 ? 'healthy' : 'warning',
            lastCheck: new Date(),
            responseTime: 30,
            details: agentdbHealth || {},
            dependencies: ['agentdb', 'quic-sync']
          },
          {
            component: 'Cognitive Analytics',
            status: cognitiveReport?.summary?.overallCognitiveScore > 70 ? 'healthy' : 'warning',
            lastCheck: new Date(),
            responseTime: 100,
            details: cognitiveReport?.summary || {},
            dependencies: ['temporal-reasoning', 'learning-patterns']
          }
        ],
        incidents: [], // Would be populated from actual incident tracking
        lastUpdated: new Date()
      };

    } catch (error) {
      console.error('❌ Error getting system health:', error);
      return {
        overall: 'down',
        score: 0,
        checks: [],
        incidents: [],
        lastUpdated: new Date()
      };
    }
  }

  /**
   * Get comprehensive performance overview
   */
  public async getPerformanceOverview(): Promise<any> {
    try {
      const [
        systemHealth,
        agentdbHealth,
        cognitiveReport,
        latestMetrics
      ] = await Promise.all([
        this.getSystemHealth(),
        this.components.agentdbMonitor.getHealthReport(),
        this.components.cognitiveAnalytics.getCognitiveReport(),
        this.components.collector.getMetrics('system', 1)[0]
      ]);

      const activeBottlenecks = this.components.bottleneckDetector.getActiveBottlenecks();
      const agentdbAnomalies = this.components.agentdbMonitor.detectAnomalies();

      return {
        systemHealth,
        agentdbPerformance: agentdbHealth,
        cognitiveIntelligence: cognitiveReport,
        systemMetrics: latestMetrics,
        activeIssues: {
          bottlenecks: activeBottlenecks,
          anomalies: agentdbAnomalies,
          total: activeBottlenecks.length + agentdbAnomalies.length
        },
        performanceTargets: {
          quicSyncLatency: { current: agentdbHealth?.currentMetrics?.quicSyncLatency || 0, target: 1.0, unit: 'ms' },
          vectorSearchLatency: { current: agentdbHealth?.currentMetrics?.vectorSearchLatency || 0, target: 1.0, unit: 'ms' },
          systemAvailability: { current: systemHealth.score, target: 99.9, unit: '%' },
          cognitiveEfficiency: { current: cognitiveReport?.summary?.overallCognitiveScore || 0, target: 85, unit: '%' }
        },
        lastUpdated: new Date(),
        isOptimal: systemHealth.score > 80 &&
                  (agentdbHealth?.healthScore || 0) > 80 &&
                  (cognitiveReport?.summary?.overallCognitiveScore || 0) > 80
      };

    } catch (error) {
      console.error('❌ Error getting performance overview:', error);
      return {
        error: 'Failed to generate performance overview',
        lastUpdated: new Date()
      };
    }
  }

  /**
   * Generate on-demand performance report
   */
  public async generatePerformanceReport(type: 'executive' | 'technical' | 'trend' | 'incident' = 'executive'): Promise<any> {
    try {
      const report = await this.components.reporter.generateReport(type);

      // Enhance report with additional orchestration data
      const systemHealth = await this.getSystemHealth();
      const performanceOverview = await this.getPerformanceOverview();

      return {
        ...report,
        orchestrationData: {
          systemHealth,
          performanceOverview,
          componentStatus: {
            collector: this.components.collector ? 'active' : 'inactive',
            bottleneckDetector: this.components.bottleneckDetector ? 'active' : 'inactive',
            dashboard: this.components.dashboard ? 'active' : 'inactive',
            cognitiveAnalytics: this.components.cognitiveAnalytics ? 'active' : 'inactive',
            reporter: this.components.reporter ? 'active' : 'inactive',
            agentdbMonitor: this.components.agentdbMonitor ? 'active' : 'inactive'
          }
        }
      };

    } catch (error) {
      console.error('❌ Error generating performance report:', error);
      throw error;
    }
  }

  /**
   * Get component status
   */
  public getComponentStatus(): any {
    return {
      isRunning: this.isRunning,
      components: {
        collector: this.components.collector ? 'initialized' : 'not_initialized',
        bottleneckDetector: this.components.bottleneckDetector ? 'initialized' : 'not_initialized',
        dashboard: this.components.dashboard ? 'initialized' : 'not_initialized',
        cognitiveAnalytics: this.components.cognitiveAnalytics ? 'initialized' : 'not_initialized',
        reporter: this.components.reporter ? 'initialized' : 'not_initialized',
        agentdbMonitor: this.components.agentdbMonitor ? 'initialized' : 'not_initialized'
      },
      healthChecks: {
        active: this.healthCheckInterval !== null,
        interval: 30000
      },
      coordination: {
        active: this.coordinationInterval !== null,
        interval: 60000
      }
    };
  }

  /**
   * Export all performance data
   */
  public exportAllData(format: 'json' | 'csv' = 'json'): string {
    const data = {
      systemHealth: this.isRunning ? 'active' : 'inactive',
      componentStatus: this.getComponentStatus(),
      collectorMetrics: this.components.collector.exportMetrics(undefined, format),
      agentdbMetrics: this.components.agentdbMonitor.exportMetrics(format),
      cognitiveReport: this.components.cognitiveAnalytics.getCognitiveReport(),
      recentReports: this.components.reporter.getReports(undefined, new Date(Date.now() - 24 * 60 * 60 * 1000), new Date()),
      exportedAt: new Date()
    };

    return JSON.stringify(data, null, 2);
  }

  /**
   * Get monitoring statistics
   */
  public getMonitoringStatistics(): any {
    return {
      uptime: this.isRunning ? Date.now() - (this as any).startTime : 0,
      components: {
        collector: this.components.collector?.getMetricsHistory?.()?.length || 0,
        bottleneckDetector: this.components.bottleneckDetector?.getActiveBottlenecks?.()?.length || 0,
        reporter: this.components.reporter.getReportingStats()
      },
      performance: {
        healthScore: 0, // Would be calculated from current health
        activeIssues: 0,
        lastHealthCheck: new Date()
      },
      dataPoints: {
        total: this.components.collector?.getMetricsHistory?.()?.length || 0,
        last24Hours: 0 // Would be calculated
      }
    };
  }
}