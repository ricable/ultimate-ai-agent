/**
 * Deployment Metrics Monitoring System
 * Real-time tracking of deployment progress, success rates, and performance indicators
 */

import { EventEmitter } from 'events';
import { createLogger, Logger } from '../../utils/logger';
import { MetricsCollector } from './metrics-collector';
import { AlertManager } from './alert-manager';
import { DashboardManager } from './dashboard-manager';

export interface DeploymentMetrics {
  deploymentId: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'rolled_back';
  startTime: Date;
  endTime?: Date;
  duration?: number;
  successRate: number;
  rollbackCount: number;
  errorCount: number;
  performanceImpact: {
    cpuImpact: number;
    memoryImpact: number;
    networkImpact: number;
    responseTimeImpact: number;
  };
  healthChecks: {
    total: number;
    passed: number;
    failed: number;
    pending: number;
  };
  rolloutProgress: {
    current: number;
    total: number;
    percentage: number;
  };
}

export interface DeploymentHealthCheck {
  id: string;
  name: string;
  type: 'api' | 'database' | 'service' | 'network' | 'custom';
  status: 'pending' | 'passed' | 'failed' | 'skipped';
  responseTime?: number;
  error?: string;
  retryCount: number;
  maxRetries: number;
}

export class DeploymentMonitor extends EventEmitter {
  private logger: Logger;
  private metricsCollector: MetricsCollector;
  private alertManager: AlertManager;
  private dashboardManager: DashboardManager;
  private activeDeployments: Map<string, DeploymentMetrics> = new Map();
  private deploymentHistory: DeploymentMetrics[] = [];
  private healthChecks: Map<string, DeploymentHealthCheck[]> = new Map();

  constructor() {
    super();
    this.logger = createLogger('DeploymentMonitor');
    this.metricsCollector = new MetricsCollector('deployment');
    this.alertManager = new AlertManager();
    this.dashboardManager = new DashboardManager();
    this.initializeMonitoring();
  }

  private initializeMonitoring(): void {
    // Setup monitoring intervals
    setInterval(() => this.collectMetrics(), 5000); // Every 5 seconds
    setInterval(() => this.analyzePerformance(), 30000); // Every 30 seconds
    setInterval(() => this.generateReports(), 300000); // Every 5 minutes

    this.logger.info('Deployment monitoring initialized');
    this.emit('monitoring:initialized');
  }

  /**
   * Start monitoring a new deployment
   */
  async startDeploymentMonitoring(deploymentId: string, config: any): Promise<void> {
    const metrics: DeploymentMetrics = {
      deploymentId,
      status: 'pending',
      startTime: new Date(),
      successRate: 0,
      rollbackCount: 0,
      errorCount: 0,
      performanceImpact: {
        cpuImpact: 0,
        memoryImpact: 0,
        networkImpact: 0,
        responseTimeImpact: 0
      },
      healthChecks: {
        total: 0,
        passed: 0,
        failed: 0,
        pending: 0
      },
      rolloutProgress: {
        current: 0,
        total: config.totalInstances || 1,
        percentage: 0
      }
    };

    this.activeDeployments.set(deploymentId, metrics);

    // Setup health checks
    await this.setupHealthChecks(deploymentId, config.healthChecks || []);

    this.logger.info(`Started monitoring deployment: ${deploymentId}`);
    this.emit('deployment:started', { deploymentId, metrics });

    // Send alerts for critical deployments
    if (config.critical) {
      await this.alertManager.sendAlert({
        level: 'info',
        title: `Critical Deployment Started: ${deploymentId}`,
        message: `Deployment ${deploymentId} has been initiated`,
        deploymentId,
        timestamp: new Date()
      });
    }
  }

  /**
   * Update deployment progress
   */
  async updateDeploymentProgress(deploymentId: string, progress: number, status?: DeploymentMetrics['status']): Promise<void> {
    const metrics = this.activeDeployments.get(deploymentId);
    if (!metrics) {
      this.logger.warn(`Deployment ${deploymentId} not found for progress update`);
      return;
    }

    metrics.rolloutProgress.current = progress;
    metrics.rolloutProgress.percentage = (progress / metrics.rolloutProgress.total) * 100;

    if (status) {
      metrics.status = status;
      if (status === 'completed') {
        metrics.endTime = new Date();
        metrics.duration = metrics.endTime.getTime() - metrics.startTime.getTime();
      }
    }

    // Calculate success rate based on health checks
    const healthChecks = this.healthChecks.get(deploymentId) || [];
    const completedChecks = healthChecks.filter(hc => hc.status !== 'pending');
    if (completedChecks.length > 0) {
      metrics.successRate = (completedChecks.filter(hc => hc.status === 'passed').length / completedChecks.length) * 100;
    }

    // Collect performance impact
    await this.collectPerformanceImpact(deploymentId);

    // Check for alerts
    await this.checkDeploymentAlerts(deploymentId);

    this.emit('deployment:updated', { deploymentId, metrics });
  }

  /**
   * Setup health checks for deployment
   */
  private async setupHealthChecks(deploymentId: string, healthCheckConfigs: any[]): Promise<void> {
    const healthChecks: DeploymentHealthCheck[] = healthCheckConfigs.map(config => ({
      id: `${deploymentId}-${config.name}`,
      name: config.name,
      type: config.type,
      status: 'pending',
      retryCount: 0,
      maxRetries: config.maxRetries || 3
    }));

    this.healthChecks.set(deploymentId, healthChecks);

    const metrics = this.activeDeployments.get(deploymentId);
    if (metrics) {
      metrics.healthChecks.total = healthChecks.length;
      metrics.healthChecks.pending = healthChecks.length;
    }

    // Start health check execution
    this.executeHealthChecks(deploymentId);
  }

  /**
   * Execute health checks
   */
  private async executeHealthChecks(deploymentId: string): Promise<void> {
    const healthChecks = this.healthChecks.get(deploymentId);
    if (!healthChecks) return;

    const metrics = this.activeDeployments.get(deploymentId);

    for (const healthCheck of healthChecks) {
      try {
        const startTime = Date.now();
        const result = await this.executeHealthCheck(healthCheck);
        const responseTime = Date.now() - startTime;

        healthCheck.responseTime = responseTime;
        healthCheck.status = result.success ? 'passed' : 'failed';
        healthCheck.error = result.error;

        // Update metrics
        if (metrics) {
          if (result.success) {
            metrics.healthChecks.passed++;
            metrics.healthChecks.pending--;
          } else {
            metrics.healthChecks.failed++;
            metrics.healthChecks.pending--;
          }
        }

        this.logger.debug(`Health check ${healthCheck.name} ${healthCheck.status} in ${responseTime}ms`);

      } catch (error) {
        healthCheck.status = 'failed';
        healthCheck.error = error instanceof Error ? error.message : 'Unknown error';
        healthCheck.retryCount++;

        if (metrics) {
          metrics.healthChecks.failed++;
          metrics.healthChecks.pending--;
        }

        // Retry logic
        if (healthCheck.retryCount < healthCheck.maxRetries) {
          this.logger.info(`Retrying health check ${healthCheck.name} (${healthCheck.retryCount}/${healthCheck.maxRetries})`);
          setTimeout(() => this.executeHealthChecks(deploymentId), 5000 * healthCheck.retryCount);
        }
      }
    }

    this.emit('health_checks:completed', { deploymentId, healthChecks });
  }

  /**
   * Execute individual health check
   */
  private async executeHealthCheck(healthCheck: DeploymentHealthCheck): Promise<{ success: boolean; error?: string }> {
    // Implementation would depend on the type of health check
    switch (healthCheck.type) {
      case 'api':
        return this.executeApiHealthCheck(healthCheck);
      case 'database':
        return this.executeDatabaseHealthCheck(healthCheck);
      case 'service':
        return this.executeServiceHealthCheck(healthCheck);
      case 'network':
        return this.executeNetworkHealthCheck(healthCheck);
      default:
        return { success: false, error: `Unknown health check type: ${healthCheck.type}` };
    }
  }

  /**
   * Execute API health check
   */
  private async executeApiHealthCheck(healthCheck: DeploymentHealthCheck): Promise<{ success: boolean; error?: string }> {
    try {
      // Example API health check implementation
      const response = await fetch(`http://localhost:3000/health`, {
        method: 'GET',
        timeout: 10000
      });

      if (response.ok) {
        return { success: true };
      } else {
        return { success: false, error: `API returned status ${response.status}` };
      }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'API check failed' };
    }
  }

  /**
   * Execute database health check
   */
  private async executeDatabaseHealthCheck(healthCheck: DeploymentHealthCheck): Promise<{ success: boolean; error?: string }> {
    try {
      // Example database health check
      // Would implement actual database connection test
      return { success: true };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Database check failed' };
    }
  }

  /**
   * Execute service health check
   */
  private async executeServiceHealthCheck(healthCheck: DeploymentHealthCheck): Promise<{ success: boolean; error?: string }> {
    try {
      // Example service health check
      return { success: true };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Service check failed' };
    }
  }

  /**
   * Execute network health check
   */
  private async executeNetworkHealthCheck(healthCheck: DeploymentHealthCheck): Promise<{ success: boolean; error?: string }> {
    try {
      // Example network health check
      return { success: true };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Network check failed' };
    }
  }

  /**
   * Collect performance impact of deployment
   */
  private async collectPerformanceImpact(deploymentId: string): Promise<void> {
    const metrics = this.activeDeployments.get(deploymentId);
    if (!metrics) return;

    try {
      // Collect current system metrics
      const currentMetrics = await this.metricsCollector.collectSystemMetrics();

      // Get baseline metrics (from before deployment)
      const baselineMetrics = await this.metricsCollector.getBaselineMetrics(deploymentId);

      if (baselineMetrics) {
        // Calculate impact
        metrics.performanceImpact.cpuImpact = ((currentMetrics.cpu - baselineMetrics.cpu) / baselineMetrics.cpu) * 100;
        metrics.performanceImpact.memoryImpact = ((currentMetrics.memory - baselineMetrics.memory) / baselineMetrics.memory) * 100;
        metrics.performanceImpact.networkImpact = ((currentMetrics.network - baselineMetrics.network) / baselineMetrics.network) * 100;
        metrics.performanceImpact.responseTimeImpact = ((currentMetrics.responseTime - baselineMetrics.responseTime) / baselineMetrics.responseTime) * 100;
      }

    } catch (error) {
      this.logger.error(`Failed to collect performance impact for deployment ${deploymentId}:`, error);
    }
  }

  /**
   * Check for deployment alerts
   */
  private async checkDeploymentAlerts(deploymentId: string): Promise<void> {
    const metrics = this.activeDeployments.get(deploymentId);
    if (!metrics) return;

    // Check for high failure rates
    if (metrics.healthChecks.failed > metrics.healthChecks.total * 0.3) {
      await this.alertManager.sendAlert({
        level: 'warning',
        title: `High Failure Rate in Deployment: ${deploymentId}`,
        message: `${metrics.healthChecks.failed}/${metrics.healthChecks.total} health checks failed`,
        deploymentId,
        timestamp: new Date()
      });
    }

    // Check for performance degradation
    const maxAcceptableImpact = 20; // 20% impact threshold
    if (Math.abs(metrics.performanceImpact.responseTimeImpact) > maxAcceptableImpact) {
      await this.alertManager.sendAlert({
        level: 'critical',
        title: `Performance Degradation in Deployment: ${deploymentId}`,
        message: `Response time impact: ${metrics.performanceImpact.responseTimeImpact.toFixed(2)}%`,
        deploymentId,
        timestamp: new Date()
      });
    }

    // Check for stalled deployment
    const stallThreshold = 10 * 60 * 1000; // 10 minutes
    if (metrics.status === 'in_progress' && (Date.now() - metrics.startTime.getTime()) > stallThreshold) {
      await this.alertManager.sendAlert({
        level: 'warning',
        title: `Deployment Stalled: ${deploymentId}`,
        message: `Deployment has been in progress for ${Math.round((Date.now() - metrics.startTime.getTime()) / 60000)} minutes`,
        deploymentId,
        timestamp: new Date()
      });
    }
  }

  /**
   * Collect deployment metrics
   */
  private async collectMetrics(): Promise<void> {
    for (const [deploymentId, metrics] of this.activeDeployments) {
      try {
        await this.metricsCollector.recordMetric('deployment_status', metrics.status, { deploymentId });
        await this.metricsCollector.recordMetric('deployment_success_rate', metrics.successRate, { deploymentId });
        await this.metricsCollector.recordMetric('deployment_progress', metrics.rolloutProgress.percentage, { deploymentId });

        // Record health check metrics
        await this.metricsCollector.recordMetric('health_checks_total', metrics.healthChecks.total, { deploymentId });
        await this.metricsCollector.recordMetric('health_checks_passed', metrics.healthChecks.passed, { deploymentId });
        await this.metricsCollector.recordMetric('health_checks_failed', metrics.healthChecks.failed, { deploymentId });

        // Record performance impact
        await this.metricsCollector.recordMetric('cpu_impact', metrics.performanceImpact.cpuImpact, { deploymentId });
        await this.metricsCollector.recordMetric('memory_impact', metrics.performanceImpact.memoryImpact, { deploymentId });
        await this.metricsCollector.recordMetric('response_time_impact', metrics.performanceImpact.responseTimeImpact, { deploymentId });

      } catch (error) {
        this.logger.error(`Failed to collect metrics for deployment ${deploymentId}:`, error);
      }
    }
  }

  /**
   * Analyze performance patterns
   */
  private async analyzePerformance(): Promise<void> {
    for (const [deploymentId, metrics] of this.activeDeployments) {
      try {
        // Analyze deployment patterns
        if (metrics.status === 'failed') {
          await this.analyzeFailurePatterns(deploymentId);
        }

        if (metrics.status === 'completed' && metrics.successRate < 95) {
          await this.analyzePartialSuccessPatterns(deploymentId);
        }

      } catch (error) {
        this.logger.error(`Failed to analyze performance for deployment ${deploymentId}:`, error);
      }
    }
  }

  /**
   * Analyze failure patterns
   */
  private async analyzeFailurePatterns(deploymentId: string): Promise<void> {
    const metrics = this.activeDeployments.get(deploymentId);
    if (!metrics) return;

    const healthChecks = this.healthChecks.get(deploymentId) || [];
    const failedChecks = healthChecks.filter(hc => hc.status === 'failed');

    // Group failures by type
    const failuresByType = failedChecks.reduce((acc, check) => {
      acc[check.type] = (acc[check.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    this.logger.info(`Failure analysis for ${deploymentId}:`, failuresByType);

    this.emit('deployment:failure_analysis', { deploymentId, failuresByType, metrics });
  }

  /**
   * Analyze partial success patterns
   */
  private async analyzePartialSuccessPatterns(deploymentId: string): Promise<void> {
    const metrics = this.activeDeployments.get(deploymentId);
    if (!metrics) return;

    this.logger.info(`Partial success analysis for ${deploymentId}: Success rate ${metrics.successRate.toFixed(2)}%`);

    this.emit('deployment:partial_success_analysis', { deploymentId, metrics });
  }

  /**
   * Generate monitoring reports
   */
  private async generateReports(): Promise<void> {
    try {
      const report = await this.generateDeploymentReport();
      await this.dashboardManager.updateDashboard('deployment', report);

      this.emit('deployment:report_generated', report);

    } catch (error) {
      this.logger.error('Failed to generate deployment report:', error);
    }
  }

  /**
   * Generate deployment report
   */
  private async generateDeploymentReport(): Promise<any> {
    const activeDeployments = Array.from(this.activeDeployments.values());
    const recentDeployments = this.deploymentHistory.slice(-10);

    return {
      timestamp: new Date(),
      summary: {
        activeDeployments: activeDeployments.length,
        completedToday: recentDeployments.filter(d => d.status === 'completed').length,
        failedToday: recentDeployments.filter(d => d.status === 'failed').length,
        averageSuccessRate: this.calculateAverageSuccessRate(activeDeployments),
        averageDuration: this.calculateAverageDeploymentDuration(activeDeployments)
      },
      activeDeployments: activeDeployments.map(d => ({
        deploymentId: d.deploymentId,
        status: d.status,
        progress: d.rolloutProgress.percentage,
        successRate: d.successRate,
        duration: d.duration,
        healthCheckStatus: `${d.healthChecks.passed}/${d.healthChecks.total} passed`
      })),
      performanceImpacts: this.calculatePerformanceImpacts(activeDeployments),
      alerts: await this.alertManager.getRecentAlerts('deployment')
    };
  }

  /**
   * Calculate average success rate
   */
  private calculateAverageSuccessRate(deployments: DeploymentMetrics[]): number {
    if (deployments.length === 0) return 0;
    const total = deployments.reduce((sum, d) => sum + d.successRate, 0);
    return total / deployments.length;
  }

  /**
   * Calculate average deployment duration
   */
  private calculateAverageDeploymentDuration(deployments: DeploymentMetrics[]): number {
    const completedDeployments = deployments.filter(d => d.duration !== undefined);
    if (completedDeployments.length === 0) return 0;
    const total = completedDeployments.reduce((sum, d) => sum + (d.duration || 0), 0);
    return total / completedDeployments.length;
  }

  /**
   * Calculate performance impacts
   */
  private calculatePerformanceImpacts(deployments: DeploymentMetrics[]): any {
    if (deployments.length === 0) return { cpu: 0, memory: 0, network: 0, responseTime: 0 };

    return {
      cpu: deployments.reduce((sum, d) => sum + d.performanceImpact.cpuImpact, 0) / deployments.length,
      memory: deployments.reduce((sum, d) => sum + d.performanceImpact.memoryImpact, 0) / deployments.length,
      network: deployments.reduce((sum, d) => sum + d.performanceImpact.networkImpact, 0) / deployments.length,
      responseTime: deployments.reduce((sum, d) => sum + d.performanceImpact.responseTimeImpact, 0) / deployments.length
    };
  }

  /**
   * Get deployment metrics
   */
  getDeploymentMetrics(deploymentId?: string): DeploymentMetrics | DeploymentMetrics[] {
    if (deploymentId) {
      return this.activeDeployments.get(deploymentId) || this.deploymentHistory.find(d => d.deploymentId === deploymentId);
    }
    return Array.from(this.activeDeployments.values());
  }

  /**
   * Complete deployment monitoring
   */
  async completeDeploymentMonitoring(deploymentId: string, status: 'completed' | 'failed'): Promise<void> {
    const metrics = this.activeDeployments.get(deploymentId);
    if (!metrics) return;

    metrics.status = status;
    metrics.endTime = new Date();
    metrics.duration = metrics.endTime.getTime() - metrics.startTime.getTime();

    // Move to history
    this.deploymentHistory.push(metrics);
    this.activeDeployments.delete(deploymentId);

    this.logger.info(`Completed monitoring deployment ${deploymentId} with status: ${status}`);
    this.emit('deployment:completed', { deploymentId, metrics });

    // Send completion alert
    await this.alertManager.sendAlert({
      level: status === 'completed' ? 'info' : 'error',
      title: `Deployment ${status}: ${deploymentId}`,
      message: `Deployment ${deploymentId} ${status} after ${Math.round(metrics.duration / 60000)} minutes`,
      deploymentId,
      timestamp: new Date()
    });
  }
}