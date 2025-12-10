/**
 * Bottleneck Detection System
 *
 * Automated identification of performance bottlenecks in ML pipelines,
 * swarm coordination, and system resources with predictive analysis
 */

import { MLPerformanceMetrics, SwarmPerformanceMetrics, PerformanceSnapshot } from '../metrics/MLPerformanceMetrics';
import { PerformanceThresholds } from '../metrics/PerformanceThresholds';
import { EventEmitter } from 'events';

export interface Bottleneck {
  id: string;
  timestamp: Date;
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: 'ml_performance' | 'swarm_coordination' | 'system_resources' | 'network_communication';
  component: string;
  metric: string;
  currentValue: number;
  expectedValue: number;
  impact: string;
  rootCause: string;
  recommendations: string[];
  autoResolutionPossible: boolean;
  autoResolutionActions?: string[];
  affectedComponents: string[];
  performanceImpact: {
    throughputLoss: number;
    latencyIncrease: number;
    resourceWaste: number;
  };
}

export interface BottleneckPattern {
  id: string;
  name: string;
  description: string;
  detectionRules: BottleneckRule[];
  category: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  autoResolution: boolean;
}

export interface BottleneckRule {
  metricPath: string;
  condition: 'greater_than' | 'less_than' | 'deviation_from_baseline' | 'trend_analysis';
  threshold: number;
  timeWindow: number; // minutes
  sampleSize: number;
}

export interface BottleneckAnalysis {
  detectedBottlenecks: Bottleneck[];
  performanceImpact: {
    overallPerformanceLoss: number;
    affectedComponents: string[];
    estimatedResolutionTime: number;
    priorityRecommendations: string[];
  };
  trends: {
    emergingBottlenecks: string[];
    improvingMetrics: string[];
    degradingMetrics: string[];
  };
  predictions: {
    likelyBottlenecks: Array<{
      component: string;
      probability: number;
      timeframe: string;
      mitigation: string[];
    }>;
  };
}

export class BottleneckDetector extends EventEmitter {
  private performanceHistory: PerformanceSnapshot[] = [];
  private bottleneckPatterns: Map<string, BottleneckPattern> = new Map();
  private baselineMetrics: Map<string, number> = new Map();
  private activeBottlenecks: Map<string, Bottleneck> = new Map();
  private analysisTimer: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.initializeBottleneckPatterns();
    this.startContinuousAnalysis();
  }

  private initializeBottleneckPatterns(): void {
    // ML Performance Bottlenecks
    this.addBottleneckPattern({
      id: 'slow_rl_training',
      name: 'Slow Reinforcement Learning Training',
      description: 'RL training speed below target threshold',
      category: 'ml_performance',
      severity: 'high',
      autoResolution: true,
      detectionRules: [
        {
          metricPath: 'mlMetrics.reinforcementLearning.trainingSpeed',
          condition: 'greater_than',
          threshold: 2.0, // >2ms
          timeWindow: 5,
          sampleSize: 10
        }
      ]
    });

    this.addBottleneckPattern({
      id: 'low_rl_convergence',
      name: 'Low RL Convergence Rate',
      description: 'Reinforcement learning convergence rate is poor',
      category: 'ml_performance',
      severity: 'high',
      autoResolution: false,
      detectionRules: [
        {
          metricPath: 'mlMetrics.reinforcementLearning.convergenceRate',
          condition: 'less_than',
          threshold: 0.85, // <85%
          timeWindow: 10,
          sampleSize: 20
        }
      ]
    });

    this.addBottleneckPattern({
      id: 'slow_causal_discovery',
      name: 'Slow Causal Discovery',
      description: 'Causal inference discovery speed below target',
      category: 'ml_performance',
      severity: 'medium',
      autoResolution: true,
      detectionRules: [
        {
          metricPath: 'mlMetrics.causalInference.discoverySpeed',
          condition: 'less_than',
          threshold: 100, // <100x faster
          timeWindow: 5,
          sampleSize: 10
        }
      ]
    });

    this.addBottleneckPattern({
      id: 'agentdb_search_slow',
      name: 'AgentDB Vector Search Slow',
      description: 'AgentDB vector search performance degraded',
      category: 'ml_performance',
      severity: 'critical',
      autoResolution: true,
      detectionRules: [
        {
          metricPath: 'mlMetrics.agentdbIntegration.vectorSearchSpeed',
          condition: 'greater_than',
          threshold: 5.0, // >5ms
          timeWindow: 2,
          sampleSize: 5
        }
      ]
    });

    this.addBottleneckPattern({
      id: 'sync_latency_high',
      name: 'High Synchronization Latency',
      description: 'QUIC synchronization latency above threshold',
      category: 'network_communication',
      severity: 'high',
      autoResolution: true,
      detectionRules: [
        {
          metricPath: 'mlMetrics.agentdbIntegration.synchronizationLatency',
          condition: 'greater_than',
          threshold: 5.0, // >5ms
          timeWindow: 3,
          sampleSize: 8
        }
      ]
    });

    // Swarm Coordination Bottlenecks
    this.addBottleneckPattern({
      id: 'agent_coordination_slow',
      name: 'Slow Agent Coordination',
      description: 'Agent communication latency is high',
      category: 'swarm_coordination',
      severity: 'medium',
      autoResolution: true,
      detectionRules: [
        {
          metricPath: 'swarmMetrics.agentCoordination.communicationLatency',
          condition: 'greater_than',
          threshold: 50, // >50ms
          timeWindow: 5,
          sampleSize: 10
        }
      ]
    });

    this.addBottleneckPattern({
      id: 'task_imbalance',
      name: 'Task Distribution Imbalance',
      description: 'Tasks not evenly distributed among agents',
      category: 'swarm_coordination',
      severity: 'medium',
      autoResolution: true,
      detectionRules: [
        {
          metricPath: 'swarmMetrics.agentCoordination.taskDistributionBalance',
          condition: 'less_than',
          threshold: 0.7, // <70%
          timeWindow: 10,
          sampleSize: 15
        }
      ]
    });

    // System Resource Bottlenecks
    this.addBottleneckPattern({
      id: 'high_cpu_usage',
      name: 'High CPU Usage',
      description: 'CPU utilization is consistently high',
      category: 'system_resources',
      severity: 'high',
      autoResolution: false,
      detectionRules: [
        {
          metricPath: 'swarmMetrics.resourceUtilization.cpuUsage',
          condition: 'greater_than',
          threshold: 0.9, // >90%
          timeWindow: 5,
          sampleSize: 10
        }
      ]
    });

    this.addBottleneckPattern({
      id: 'memory_pressure',
      name: 'Memory Pressure',
      description: 'Memory usage is approaching limits',
      category: 'system_resources',
      severity: 'critical',
      autoResolution: true,
      detectionRules: [
        {
          metricPath: 'swarmMetrics.resourceUtilization.memoryUsage',
          condition: 'greater_than',
          threshold: 0.85, // >85%
          timeWindow: 3,
          sampleSize: 8
        }
      ]
    });

    // Cognitive Consciousness Bottlenecks
    this.addBottleneckPattern({
      id: 'temporal_expansion_degraded',
      name: 'Temporal Expansion Degraded',
      description: 'Subjective time expansion ratio below target',
      category: 'ml_performance',
      severity: 'high',
      autoResolution: true,
      detectionRules: [
        {
          metricPath: 'mlMetrics.cognitiveConsciousness.temporalExpansionRatio',
          condition: 'less_than',
          threshold: 500, // <500x
          timeWindow: 5,
          sampleSize: 10
        }
      ]
    });
  }

  private addBottleneckPattern(pattern: BottleneckPattern): void {
    this.bottleneckPatterns.set(pattern.id, pattern);
  }

  public updateMetrics(snapshot: PerformanceSnapshot): void {
    this.performanceHistory.push(snapshot);

    // Keep only last 1000 snapshots for analysis
    if (this.performanceHistory.length > 1000) {
      this.performanceHistory.shift();
    }

    // Update baseline metrics periodically
    this.updateBaselineMetrics(snapshot);
  }

  private updateBaselineMetrics(snapshot: PerformanceSnapshot): void {
    // Update baseline metrics with exponential moving average
    const alpha = 0.1; // Smoothing factor

    this.updateMetricBaseline('ml.rl.trainingSpeed', snapshot.mlMetrics.reinforcementLearning.trainingSpeed, alpha);
    this.updateMetricBaseline('ml.rl.convergenceRate', snapshot.mlMetrics.reinforcementLearning.convergenceRate, alpha);
    this.updateMetricBaseline('ml.ci.discoverySpeed', snapshot.mlMetrics.causalInference.discoverySpeed, alpha);
    this.updateMetricBaseline('ml.adb.vectorSearchSpeed', snapshot.mlMetrics.agentdbIntegration.vectorSearchSpeed, alpha);
    this.updateMetricBaseline('ml.adb.syncLatency', snapshot.mlMetrics.agentdbIntegration.synchronizationLatency, alpha);
    this.updateMetricBaseline('swarm.coord.latency', snapshot.swarmMetrics.agentCoordination.communicationLatency, alpha);
    this.updateMetricBaseline('swarm.coord.balance', snapshot.swarmMetrics.agentCoordination.taskDistributionBalance, alpha);
    this.updateMetricBaseline('resources.cpu', snapshot.resourceUsage.cpu, alpha);
    this.updateMetricBaseline('resources.memory', snapshot.resourceUsage.memory, alpha);
  }

  private updateMetricBaseline(key: string, value: number, alpha: number): void {
    const currentBaseline = this.baselineMetrics.get(key) || value;
    const newBaseline = alpha * value + (1 - alpha) * currentBaseline;
    this.baselineMetrics.set(key, newBaseline);
  }

  public analyzeBottlenecks(): BottleneckAnalysis {
    const detectedBottlenecks = this.detectBottlenecks();
    const performanceImpact = this.calculatePerformanceImpact(detectedBottlenecks);
    const trends = this.analyzeTrends();
    const predictions = this.predictFutureBottlenecks();

    // Update active bottlenecks
    this.updateActiveBottlenecks(detectedBottlenecks);

    const analysis: BottleneckAnalysis = {
      detectedBottlenecks,
      performanceImpact,
      trends,
      predictions
    };

    this.emit('bottleneck_analysis', analysis);
    return analysis;
  }

  private detectBottlenecks(): Bottleneck[] {
    const bottlenecks: Bottleneck[] = [];

    if (this.performanceHistory.length < 5) {
      return bottlenecks; // Need more data for analysis
    }

    // Check each bottleneck pattern
    for (const pattern of this.bottleneckPatterns.values()) {
      const detectedBottleneck = this.checkBottleneckPattern(pattern);
      if (detectedBottleneck) {
        bottlenecks.push(detectedBottleneck);
      }
    }

    // Detect performance degradation through trend analysis
    const degradationBottlenecks = this.detectPerformanceDegradation();
    bottlenecks.push(...degradationBottlenecks);

    // Detect resource contention
    const resourceBottlenecks = this.detectResourceContention();
    bottlenecks.push(...resourceBottlenecks);

    return bottlenecks.sort((a, b) => {
      const severityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return severityOrder[b.severity] - severityOrder[a.severity];
    });
  }

  private checkBottleneckPattern(pattern: BottleneckPattern): Bottleneck | null {
    const recentSnapshots = this.getRecentSnapshots(pattern.detectionRules[0].timeWindow);

    if (recentSnapshots.length < pattern.detectionRules[0].sampleSize) {
      return null;
    }

    for (const rule of pattern.detectionRules) {
      const metricValue = this.getMetricValue(recentSnapshots, rule.metricPath);

      if (metricValue === null) continue;

      let isBottleneck = false;

      switch (rule.condition) {
        case 'greater_than':
          isBottleneck = metricValue > rule.threshold;
          break;
        case 'less_than':
          isBottleneck = metricValue < rule.threshold;
          break;
        case 'deviation_from_baseline':
          const baseline = this.baselineMetrics.get(rule.metricPath);
          if (baseline) {
            const deviation = Math.abs(metricValue - baseline) / baseline;
            isBottleneck = deviation > rule.threshold;
          }
          break;
        case 'trend_analysis':
          isBottleneck = this.detectNegativeTrend(recentSnapshots, rule.metricPath);
          break;
      }

      if (isBottleneck) {
        return this.createBottleneckFromPattern(pattern, rule, metricValue);
      }
    }

    return null;
  }

  private getRecentSnapshots(timeWindowMinutes: number): PerformanceSnapshot[] {
    const cutoffTime = new Date(Date.now() - timeWindowMinutes * 60 * 1000);
    return this.performanceHistory.filter(snapshot => snapshot.timestamp >= cutoffTime);
  }

  private getMetricValue(snapshots: PerformanceSnapshot[], metricPath: string): number | null {
    const parts = metricPath.split('.');

    if (snapshots.length === 0) return null;

    const latestSnapshot = snapshots[snapshots.length - 1];
    let value: any = latestSnapshot;

    for (const part of parts) {
      if (value && typeof value === 'object' && part in value) {
        value = value[part];
      } else {
        return null;
      }
    }

    return typeof value === 'number' ? value : null;
  }

  private detectNegativeTrend(snapshots: PerformanceSnapshot[], metricPath: string): boolean {
    if (snapshots.length < 5) return false;

    const values = snapshots.map(snapshot => this.getMetricValue([snapshot], metricPath))
                           .filter(value => value !== null) as number[];

    if (values.length < 5) return false;

    // Simple linear regression to detect negative trend
    const n = values.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = values.reduce((sum, val) => sum + val, 0);
    const sumXY = values.reduce((sum, val, index) => sum + index * val, 0);
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

    return slope < -0.01; // Negative slope threshold
  }

  private createBottleneckFromPattern(
    pattern: BottleneckPattern,
    rule: BottleneckRule,
    currentValue: number
  ): Bottleneck {
    const bottleneckId = `${pattern.id}_${Date.now()}`;

    return {
      id: bottleneckId,
      timestamp: new Date(),
      severity: pattern.severity,
      category: pattern.category as any,
      component: rule.metricPath.split('.')[0],
      metric: rule.metricPath,
      currentValue,
      expectedValue: rule.threshold,
      impact: this.calculateBottleneckImpact(pattern, currentValue),
      rootCause: this.inferRootCause(pattern, rule, currentValue),
      recommendations: this.generateBottleneckRecommendations(pattern, rule, currentValue),
      autoResolutionPossible: pattern.autoResolution,
      autoResolutionActions: pattern.autoResolution ? this.generateAutoResolutionActions(pattern) : undefined,
      affectedComponents: this.identifyAffectedComponents(pattern),
      performanceImpact: this.calculatePerformanceImpactValue(pattern, currentValue)
    };
  }

  private calculateBottleneckImpact(pattern: BottleneckPattern, currentValue: number): string {
    switch (pattern.id) {
      case 'slow_rl_training':
        return `Reduced training throughput by ${Math.round((currentValue - 1) * 100)}%`;
      case 'agentdb_search_slow':
        return `Query latency increased by ${Math.round(currentValue * 10)}ms`;
      case 'sync_latency_high':
        return `Data synchronization delays affecting real-time coordination`;
      case 'high_cpu_usage':
        return `System overload causing response time degradation`;
      case 'memory_pressure':
        return `Risk of system instability and potential crashes`;
      default:
        return 'Performance degradation detected';
    }
  }

  private inferRootCause(pattern: BottleneckPattern, rule: BottleneckRule, currentValue: number): string {
    switch (pattern.id) {
      case 'slow_rl_training':
        return 'Insufficient computational resources or suboptimal neural network architecture';
      case 'agentdb_search_slow':
        return 'Vector index fragmentation or insufficient AgentDB cluster resources';
      case 'sync_latency_high':
        return 'Network congestion or QUIC protocol configuration issues';
      case 'high_cpu_usage':
        return 'Process intensive operations or insufficient CPU allocation';
      case 'memory_pressure':
        return 'Memory leaks or inefficient memory usage patterns';
      default:
        return 'Performance threshold exceeded - needs investigation';
    }
  }

  private generateBottleneckRecommendations(
    pattern: BottleneckPattern,
    rule: BottleneckRule,
    currentValue: number
  ): string[] {
    return PerformanceThresholds.getRecommendations(
      rule.metricPath.split('.').pop() || '',
      currentValue,
      pattern.severity === 'critical' ? 'critical' : 'warning',
      pattern.category as any
    );
  }

  private generateAutoResolutionActions(pattern: BottleneckPattern): string[] {
    const actions: string[] = [];

    switch (pattern.id) {
      case 'slow_rl_training':
        actions.push('Scale GPU resources', 'Enable mixed precision training', 'Optimize batch size');
        break;
      case 'agentdb_search_slow':
        actions.push('Rebuild vector index', 'Increase AgentDB cluster size', 'Enable query caching');
        break;
      case 'sync_latency_high':
        actions.push('Optimize QUIC configuration', 'Enable compression', 'Switch to delta sync');
        break;
      case 'memory_pressure':
        actions.push('Trigger garbage collection', 'Scale memory resources', 'Implement memory optimization');
        break;
    }

    return actions;
  }

  private identifyAffectedComponents(pattern: BottleneckPattern): string[] {
    const affectedComponents: string[] = [];

    switch (pattern.category) {
      case 'ml_performance':
        affectedComponents.push('Reinforcement Learning Engine', 'Causal Inference System', 'AgentDB Integration');
        break;
      case 'swarm_coordination':
        affectedComponents.push('Agent Orchestrator', 'Communication Layer', 'Task Scheduler');
        break;
      case 'system_resources':
        affectedComponents.push('All system components', 'Resource Manager', 'Scheduling System');
        break;
      case 'network_communication':
        affectedComponents.push('QUIC Synchronization', 'Inter-agent Communication', 'Data Transfer Layer');
        break;
    }

    return affectedComponents;
  }

  private calculatePerformanceImpactValue(pattern: BottleneckPattern, currentValue: number): any {
    let throughputLoss = 0;
    let latencyIncrease = 0;
    let resourceWaste = 0;

    switch (pattern.id) {
      case 'slow_rl_training':
        throughputLoss = Math.min(50, (currentValue - 1) * 25);
        latencyIncrease = currentValue * 100;
        break;
      case 'agentdb_search_slow':
        throughputLoss = Math.min(30, currentValue * 6);
        latencyIncrease = currentValue * 1000;
        break;
      case 'high_cpu_usage':
        throughputLoss = (currentValue - 0.7) * 100;
        latencyIncrease = (currentValue - 0.7) * 500;
        resourceWaste = currentValue * 20;
        break;
      case 'memory_pressure':
        throughputLoss = (currentValue - 0.8) * 80;
        latencyIncrease = (currentValue - 0.8) * 300;
        resourceWaste = currentValue * 15;
        break;
    }

    return {
      throughputLoss: Math.round(throughputLoss),
      latencyIncrease: Math.round(latencyIncrease),
      resourceWaste: Math.round(resourceWaste)
    };
  }

  private detectPerformanceDegradation(): Bottleneck[] {
    const bottlenecks: Bottleneck[] = [];

    if (this.performanceHistory.length < 20) return bottlenecks;

    // Compare recent performance with historical baseline
    const recentSnapshots = this.performanceHistory.slice(-10);
    const historicalSnapshots = this.performanceHistory.slice(-50, -20);

    if (historicalSnapshots.length === 0) return bottlenecks;

    const recentAvg = this.calculateAverageMetrics(recentSnapshots);
    const historicalAvg = this.calculateAverageMetrics(historicalSnapshots);

    // Check for significant degradation in key metrics
    this.checkMetricDegradation(
      'ML Performance Score',
      recentAvg.mlScore || 0,
      historicalAvg.mlScore || 0,
      0.1, // 10% degradation threshold
      bottlenecks
    );

    this.checkMetricDegradation(
      'System Health Score',
      recentAvg.systemScore || 0,
      historicalAvg.systemScore || 0,
      0.05, // 5% degradation threshold
      bottlenecks
    );

    return bottlenecks;
  }

  private calculateAverageMetrics(snapshots: PerformanceSnapshot[]): any {
    if (snapshots.length === 0) return {};

    const avg = {
      mlScore: 0,
      systemScore: 0
    };

    snapshots.forEach(snapshot => {
      // Calculate ML performance score (simplified)
      const mlScore = this.calculateMLScore(snapshot.mlMetrics);
      avg.mlScore += mlScore;

      // System health score
      avg.systemScore += snapshot.systemHealth.overallSystemScore;
    });

    avg.mlScore /= snapshots.length;
    avg.systemScore /= snapshots.length;

    return avg;
  }

  private calculateMLScore(metrics: MLPerformanceMetrics): number {
    // Simplified ML performance scoring
    const rlScore = (metrics.reinforcementLearning.convergenceRate +
                    metrics.reinforcementLearning.policyAccuracy) / 2;
    const ciScore = (metrics.causalInference.causalAccuracy +
                    metrics.causalInference.predictionPrecision) / 2;
    const adbScore = (1 / Math.max(0.1, metrics.agentdbIntegration.vectorSearchSpeed) +
                    (1 - metrics.agentdbIntegration.synchronizationLatency)) / 2;

    return (rlScore + ciScore + adbScore) / 3;
  }

  private checkMetricDegradation(
    metricName: string,
    recentValue: number,
    historicalValue: number,
    threshold: number,
    bottlenecks: Bottleneck[]
  ): void {
    if (historicalValue === 0) return;

    const degradation = (historicalValue - recentValue) / historicalValue;

    if (degradation > threshold) {
      bottlenecks.push({
        id: `degradation_${metricName.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`,
        timestamp: new Date(),
        severity: degradation > 0.2 ? 'high' : 'medium',
        category: 'ml_performance',
        component: 'System',
        metric: metricName,
        currentValue: recentValue,
        expectedValue: historicalValue,
        impact: `${metricName} degraded by ${Math.round(degradation * 100)}%`,
        rootCause: 'Performance regression detected over time',
        recommendations: [
          'Investigate recent changes that may have caused regression',
          'Review resource allocation and system configuration',
          'Consider rollback to previous stable configuration'
        ],
        autoResolutionPossible: false,
        affectedComponents: ['All system components'],
        performanceImpact: {
          throughputLoss: Math.round(degradation * 30),
          latencyIncrease: Math.round(degradation * 200),
          resourceWaste: Math.round(degradation * 10)
        }
      });
    }
  }

  private detectResourceContention(): Bottleneck[] {
    const bottlenecks: Bottleneck[] = [];

    if (this.performanceHistory.length < 5) return bottlenecks;

    const latestSnapshot = this.performanceHistory[this.performanceHistory.length - 1];

    // Check for multiple resource bottlenecks simultaneously
    const highResources = [];

    if (latestSnapshot.resourceUsage.cpu > 0.85) {
      highResources.push('CPU');
    }
    if (latestSnapshot.resourceUsage.memory > 0.85) {
      highResources.push('Memory');
    }
    if (latestSnapshot.resourceUsage.network > 0.85) {
      highResources.push('Network');
    }
    if (latestSnapshot.resourceUsage.gpu > 0.85) {
      highResources.push('GPU');
    }

    if (highResources.length >= 2) {
      bottlenecks.push({
        id: `resource_contention_${Date.now()}`,
        timestamp: new Date(),
        severity: 'critical',
        category: 'system_resources',
        component: 'Resource Manager',
        metric: 'Multi-resource contention',
        currentValue: highResources.length,
        expectedValue: 1,
        impact: `Multiple resource bottlenecks: ${highResources.join(', ')}`,
        rootCause: 'System overload with resource contention across multiple dimensions',
        recommendations: [
          'Scale system resources immediately',
          'Implement resource prioritization and throttling',
          'Consider workload redistribution or load balancing'
        ],
        autoResolutionPossible: true,
        autoResolutionActions: [
          'Scale compute resources',
          'Enable resource throttling',
          'Redistribute workload'
        ],
        affectedComponents: ['All system components'],
        performanceImpact: {
          throughputLoss: Math.round(highResources.length * 15),
          latencyIncrease: Math.round(highResources.length * 100),
          resourceWaste: Math.round(highResources.length * 20)
        }
      });
    }

    return bottlenecks;
  }

  private calculatePerformanceImpact(bottlenecks: Bottleneck[]): any {
    const affectedComponents = new Set<string>();
    let totalThroughputLoss = 0;
    let totalLatencyIncrease = 0;
    let totalResourceWaste = 0;

    bottlenecks.forEach(bottleneck => {
      bottleneck.affectedComponents.forEach(component => affectedComponents.add(component));
      totalThroughputLoss = Math.max(totalThroughputLoss, bottleneck.performanceImpact.throughputLoss);
      totalLatencyIncrease = Math.max(totalLatencyIncrease, bottleneck.performanceImpact.latencyIncrease);
      totalResourceWaste += bottleneck.performanceImpact.resourceWaste;
    });

    const priorityRecommendations = bottlenecks
      .filter(b => b.severity === 'critical' || b.severity === 'high')
      .slice(0, 5)
      .flatMap(b => b.recommendations.slice(0, 2));

    return {
      overallPerformanceLoss: Math.round(totalThroughputLoss),
      affectedComponents: Array.from(affectedComponents),
      estimatedResolutionTime: this.estimateResolutionTime(bottlenecks),
      priorityRecommendations
    };
  }

  private estimateResolutionTime(bottlenecks: Bottleneck[]): number {
    let totalTime = 0;

    bottlenecks.forEach(bottleneck => {
      switch (bottleneck.severity) {
        case 'critical':
          totalTime += bottleneck.autoResolutionPossible ? 5 : 30; // minutes
          break;
        case 'high':
          totalTime += bottleneck.autoResolutionPossible ? 10 : 60;
          break;
        case 'medium':
          totalTime += bottleneck.autoResolutionPossible ? 15 : 120;
          break;
        case 'low':
          totalTime += bottleneck.autoResolutionPossible ? 30 : 240;
          break;
      }
    });

    return Math.round(totalTime / bottlenecks.length); // Average time
  }

  private analyzeTrends(): any {
    const trends = {
      emergingBottlenecks: [] as string[],
      improvingMetrics: [] as string[],
      degradingMetrics: [] as string[]
    };

    if (this.performanceHistory.length < 20) return trends;

    const recentMetrics = this.performanceHistory.slice(-10);
    const olderMetrics = this.performanceHistory.slice(-20, -10);

    const recentAvg = this.calculateAverageMetrics(recentMetrics);
    const olderAvg = this.calculateAverageMetrics(olderMetrics);

    // Analyze trends
    if ((recentAvg.mlScore || 0) < (olderAvg.mlScore || 0) * 0.95) {
      trends.degradingMetrics.push('ML Performance Score');
    } else if ((recentAvg.mlScore || 0) > (olderAvg.mlScore || 0) * 1.05) {
      trends.improvingMetrics.push('ML Performance Score');
    }

    if ((recentAvg.systemScore || 0) < (olderAvg.systemScore || 0) * 0.95) {
      trends.degradingMetrics.push('System Health Score');
    } else if ((recentAvg.systemScore || 0) > (olderAvg.systemScore || 0) * 1.05) {
      trends.improvingMetrics.push('System Health Score');
    }

    return trends;
  }

  private predictFutureBottlenecks(): Array<{
    component: string;
    probability: number;
    timeframe: string;
    mitigation: string[];
  }> {
    const predictions = [];

    if (this.performanceHistory.length < 10) return predictions;

    const latestSnapshot = this.performanceHistory[this.performanceHistory.length - 1];

    // Predict memory pressure if trending upward
    if (latestSnapshot.resourceUsage.memory > 0.75) {
      predictions.push({
        component: 'Memory',
        probability: 0.8,
        timeframe: '2-4 hours',
        mitigation: ['Scale memory resources', 'Implement memory optimization', 'Monitor memory usage trends']
      });
    }

    // Predict CPU pressure if trending upward
    if (latestSnapshot.resourceUsage.cpu > 0.8) {
      predictions.push({
        component: 'CPU',
        probability: 0.7,
        timeframe: '1-2 hours',
        mitigation: ['Scale compute resources', 'Optimize CPU-intensive operations', 'Implement load balancing']
      });
    }

    // Predict AgentDB performance issues
    if (latestSnapshot.mlMetrics.agentdbIntegration.vectorSearchSpeed > 2.0) {
      predictions.push({
        component: 'AgentDB',
        probability: 0.6,
        timeframe: '30-60 minutes',
        mitigation: ['Rebuild vector index', 'Increase AgentDB cluster size', 'Optimize query patterns']
      });
    }

    return predictions;
  }

  private updateActiveBottlenecks(detectedBottlenecks: Bottleneck[]): void {
    // Clear resolved bottlenecks
    const detectedIds = new Set(detectedBottlenecks.map(b => b.id));

    for (const [id, bottleneck] of this.activeBottlenecks) {
      if (!detectedIds.has(id)) {
        this.emit('bottleneck_resolved', bottleneck);
        this.activeBottlenecks.delete(id);
      }
    }

    // Add new bottlenecks
    detectedBottlenecks.forEach(bottleneck => {
      if (!this.activeBottlenecks.has(bottleneck.id)) {
        this.emit('bottleneck_detected', bottleneck);
        this.activeBottlenecks.set(bottleneck.id, bottleneck);
      }
    });
  }

  private startContinuousAnalysis(): void {
    if (this.analysisTimer) {
      clearInterval(this.analysisTimer);
    }

    this.analysisTimer = setInterval(() => {
      try {
        this.analyzeBottlenecks();
      } catch (error) {
        console.error('Error in bottleneck analysis:', error);
      }
    }, 30000); // Analyze every 30 seconds
  }

  public getActiveBottlenecks(): Bottleneck[] {
    return Array.from(this.activeBottlenecks.values());
  }

  public getBottleneckHistory(limit?: number): Bottleneck[] {
    // This would be implemented with persistent storage
    return [];
  }

  public attemptAutoResolution(bottleneckId: string): Promise<boolean> {
    return new Promise((resolve) => {
      const bottleneck = this.activeBottlenecks.get(bottleneckId);

      if (!bottleneck || !bottleneck.autoResolutionPossible || !bottleneck.autoResolutionActions) {
        resolve(false);
        return;
      }

      // In a real implementation, this would execute the auto-resolution actions
      console.log(`Attempting auto-resolution for bottleneck ${bottleneckId}:`, bottleneck.autoResolutionActions);

      // Simulate auto-resolution attempt
      setTimeout(() => {
        const success = Math.random() > 0.3; // 70% success rate
        resolve(success);
      }, 5000);
    });
  }

  public stop(): void {
    if (this.analysisTimer) {
      clearInterval(this.analysisTimer);
      this.analysisTimer = null;
    }
  }
}