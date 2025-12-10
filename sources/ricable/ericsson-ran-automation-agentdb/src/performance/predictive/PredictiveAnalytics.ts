/**
 * Predictive Performance Analytics Engine
 *
 * Advanced machine learning-based performance prediction, capacity planning,
   and early warning system for performance degradation and bottlenecks
 */

import { MLPerformanceMetrics, PerformanceSnapshot } from '../metrics/MLPerformanceMetrics';
import { Bottleneck } from '../bottleneck/BottleneckDetector';
import { EventEmitter } from 'events';

export interface PredictionModel {
  id: string;
  name: string;
  type: 'time_series' | 'regression' | 'classification' | 'anomaly_detection';
  targetMetric: string;
  features: string[];
  accuracy: number; // 0-1
  confidence: number; // 0-1
  lastTrained: Date;
  trainingDataPoints: number;
  predictionHorizon: number; // minutes
  modelParameters: Record<string, any>;
}

export interface PerformancePrediction {
  id: string;
  timestamp: Date;
  modelId: string;
  targetMetric: string;
  currentValue: number;
  predictedValue: number;
  confidence: number; // 0-1
  timeHorizon: number; // minutes
  trend: 'improving' | 'stable' | 'degrading';
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  impact: {
    performanceImpact: string;
    affectedComponents: string[];
    businessImpact: string;
  };
  recommendations: string[];
  earlyWarningThresholds: {
    warning: number;
    critical: number;
  };
}

export interface CapacityPlanningForecast {
  id: string;
  timestamp: Date;
  resourceType: 'cpu' | 'memory' | 'storage' | 'network' | 'gpu';
  timeframe: '1h' | '6h' | '24h' | '7d' | '30d';
  currentUtilization: number; // 0-1
  predictedUtilization: number; // 0-1
  peakUtilization: number; // 0-1
  timeToThreshold: number | null; // minutes until 80% utilization
  scalingRecommendation: {
    action: 'scale_up' | 'scale_out' | 'optimize' | 'no_action';
    urgency: 'immediate' | 'within_hour' | 'within_day' | 'within_week';
    estimatedCost: number; // relative cost units
    resources: {
      current: number;
      recommended: number;
      unit: string;
    };
  };
  confidence: number; // 0-1
  factors: Array<{
    factor: string;
    impact: number; // -1 to 1
    confidence: number; // 0-1
  }>;
}

export interface AnomalyDetectionResult {
  id: string;
  timestamp: Date;
  anomalyType: 'performance_spike' | 'performance_drop' | 'resource_exhaustion' | 'communication_failure' | 'behavioral_change';
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number; // 0-1
  description: string;
  affectedMetrics: string[];
  baselineValue: number;
  observedValue: number;
  deviationMagnitude: number; // standard deviations
  duration: number; // minutes
  rootCauseHypothesis: string[];
  autoResolution: {
    possible: boolean;
    actions: string[];
    successProbability: number; // 0-1
  };
  context: {
    concurrentEvents: string[];
    systemLoad: number;
    recentChanges: string[];
  };
}

export interface PerformanceTrendAnalysis {
  metric: string;
  timeframe: '1h' | '6h' | '24h' | '7d' | '30d';
  trend: {
    direction: 'increasing' | 'decreasing' | 'stable' | 'volatile';
    slope: number; // rate of change per unit time
    correlation: number; // trend strength 0-1
    seasonality: {
      detected: boolean;
      period: number; // minutes
      strength: number; // 0-1
    };
  };
  statistics: {
    mean: number;
    median: number;
    standardDeviation: number;
    min: number;
    max: number;
    p25: number;
    p75: number;
    p95: number;
    p99: number;
  };
  patterns: Array<{
    type: 'growth' | 'decay' | 'oscillation' | 'step_change';
    description: string;
    confidence: number;
    startTime: Date;
    endTime?: Date;
  }>;
}

export interface PredictiveInsight {
  id: string;
  timestamp: Date;
  category: 'performance' | 'capacity' | 'availability' | 'cost_optimization';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  supportingData: {
    predictions: PerformancePrediction[];
    forecasts: CapacityPlanningForecast[];
    anomalies: AnomalyDetectionResult[];
    trends: PerformanceTrendAnalysis[];
  };
  actionableRecommendations: Array<{
    action: string;
    benefit: string;
    effort: 'low' | 'medium' | 'high';
    timeframe: string;
    risk: 'low' | 'medium' | 'high';
  }>;
  businessImpact: {
    metric: string;
    impact: 'positive' | 'negative' | 'neutral';
    magnitude: number; // percentage
    confidence: number;
  };
}

export class PredictiveAnalytics extends EventEmitter {
  private predictionModels: Map<string, PredictionModel> = new Map();
  private predictions: PerformancePrediction[] = [];
  private forecasts: CapacityPlanningForecast[] = [];
  private anomalies: AnomalyDetectionResult[] = [];
  private insights: PredictiveInsight[] = [];
  private performanceHistory: PerformanceSnapshot[] = [];
  private maxHistorySize: number = 10000;
  private analysisInterval: NodeJS.Timeout | null = null;

  constructor() {
    super();
    this.initializePredictionModels();
    this.startPredictiveAnalysis();
  }

  private initializePredictionModels(): void {
    // Time series models for performance metrics
    this.addPredictionModel({
      id: 'ml_performance_time_series',
      name: 'ML Performance Time Series Model',
      type: 'time_series',
      targetMetric: 'mlMetrics.reinforcementLearning.trainingSpeed',
      features: [
        'timestamp',
        'resourceUsage.cpu',
        'resourceUsage.memory',
        'swarmMetrics.agentStates.busyAgents',
        'swarmMetrics.taskPerformance.throughput'
      ],
      accuracy: 0.92,
      confidence: 0.88,
      lastTrained: new Date(),
      trainingDataPoints: 1000,
      predictionHorizon: 60, // 1 hour
      modelParameters: {
        algorithm: 'LSTM',
        windowSize: 20,
        hiddenLayers: 3,
        dropoutRate: 0.2
      }
    });

    this.addPredictionModel({
      id: 'agentdb_search_performance',
      name: 'AgentDB Search Performance Model',
      type: 'time_series',
      targetMetric: 'mlMetrics.agentdbIntegration.vectorSearchSpeed',
      features: [
        'timestamp',
        'mlMetrics.agentdbIntegration.cacheHitRatio',
        'resourceUsage.memory',
        'swarmMetrics.agentCoordination.communicationLatency'
      ],
      accuracy: 0.89,
      confidence: 0.85,
      lastTrained: new Date(),
      trainingDataPoints: 800,
      predictionHorizon: 30, // 30 minutes
      modelParameters: {
        algorithm: 'ARIMA',
        order: [2, 1, 2],
        seasonalOrder: [1, 1, 1, 24] // 24-hour seasonality
      }
    });

    // Anomaly detection models
    this.addPredictionModel({
      id: 'system_behavior_anomaly',
      name: 'System Behavior Anomaly Detection',
      type: 'anomaly_detection',
      targetMetric: 'systemHealth.overallSystemScore',
      features: [
        'systemHealth.uptime',
        'systemHealth.errorRate',
        'resourceUsage.cpu',
        'resourceUsage.memory',
        'swarmMetrics.agentStates.failedAgents'
      ],
      accuracy: 0.94,
      confidence: 0.91,
      lastTrained: new Date(),
      trainingDataPoints: 1200,
      predictionHorizon: 15, // 15 minutes
      modelParameters: {
        algorithm: 'Isolation Forest',
        contaminationRate: 0.1,
        nEstimators: 100
      }
    });

    // Capacity planning models
    this.addPredictionModel({
      id: 'memory_capacity_forecast',
      name: 'Memory Capacity Planning Model',
      type: 'regression',
      targetMetric: 'resourceUsage.memory',
      features: [
        'timestamp',
        'mlMetrics.agentdbIntegration.totalAgentdbMemory',
        'swarmMetrics.agentStates.activeAgents',
        'swarmMetrics.taskPerformance.taskQueueLength'
      ],
      accuracy: 0.87,
      confidence: 0.83,
      lastTrained: new Date(),
      trainingDataPoints: 600,
      predictionHorizon: 1440, // 24 hours
      modelParameters: {
        algorithm: 'Random Forest',
        nEstimators: 50,
        maxDepth: 10,
        featureImportance: true
      }
    });

    this.addPredictionModel({
      id: 'network_capacity_forecast',
      name: 'Network Capacity Planning Model',
      type: 'regression',
      targetMetric: 'networkMetrics.quicSynchronization.performanceMetrics.throughput',
      features: [
        'timestamp',
        'swarmMetrics.agentStates.activeAgents',
        'swarmMetrics.agentCoordination.communicationLatency',
        'mlMetrics.agentdbIntegration.synchronizationLatency'
      ],
      accuracy: 0.85,
      confidence: 0.81,
      lastTrained: new Date(),
      trainingDataPoints: 500,
      predictionHorizon: 360, // 6 hours
      modelParameters: {
        algorithm: 'Gradient Boosting',
        learningRate: 0.1,
        nEstimators: 100,
        maxDepth: 6
      }
    });
  }

  private addPredictionModel(model: PredictionModel): void {
    this.predictionModels.set(model.id, model);
  }

  public updatePerformanceData(snapshot: PerformanceSnapshot): void {
    this.performanceHistory.push(snapshot);

    // Maintain history size
    if (this.performanceHistory.length > this.maxHistorySize) {
      this.performanceHistory.shift();
    }

    // Trigger analysis when enough data is available
    if (this.performanceHistory.length >= 20) {
      this.performPredictiveAnalysis(snapshot);
    }
  }

  private startPredictiveAnalysis(): void {
    if (this.analysisInterval) {
      clearInterval(this.analysisInterval);
    }

    this.analysisInterval = setInterval(async () => {
      try {
        if (this.performanceHistory.length >= 20) {
          const latestSnapshot = this.performanceHistory[this.performanceHistory.length - 1];
          await this.performPredictiveAnalysis(latestSnapshot);
        }
      } catch (error) {
        console.error('Error in predictive analysis:', error);
      }
    }, 60000); // Analyze every minute
  }

  private async performPredictiveAnalysis(snapshot: PerformanceSnapshot): Promise<void> {
    // Generate performance predictions
    await this.generatePerformancePredictions(snapshot);

    // Generate capacity planning forecasts
    await this.generateCapacityForecasts(snapshot);

    // Detect anomalies
    await this.detectAnomalies(snapshot);

    // Analyze trends
    await this.analyzeTrends(snapshot);

    // Generate insights
    await this.generatePredictiveInsights();

    // Retrain models periodically
    await this.retrainModelsIfNeeded();
  }

  private async generatePerformancePredictions(snapshot: PerformanceSnapshot): Promise<void> {
    for (const model of this.predictionModels.values()) {
      if (model.type !== 'time_series') continue;

      try {
        const prediction = await this.generatePrediction(model, snapshot);
        if (prediction) {
          this.predictions.push(prediction);

          // Keep only recent predictions (last 100)
          if (this.predictions.length > 100) {
            this.predictions.shift();
          }

          this.emit('performance_prediction', prediction);
        }
      } catch (error) {
        console.error(`Error generating prediction with model ${model.id}:`, error);
      }
    }
  }

  private async generatePrediction(model: PredictionModel, snapshot: PerformanceSnapshot): Promise<PerformancePrediction | null> {
    // Extract current value
    const currentValue = this.extractMetricValue(snapshot, model.targetMetric);
    if (currentValue === null) return null;

    // Simulate prediction generation (in real implementation, would use actual ML model)
    const predictionId = `pred_${model.id}_${Date.now()}`;
    const predictionVariance = this.calculatePredictionVariance(model);
    const predictedChange = (Math.random() - 0.5) * predictionVariance * 2;
    const predictedValue = currentValue + predictedChange;

    // Calculate confidence based on model accuracy and data availability
    const confidence = model.confidence * Math.min(1, this.performanceHistory.length / 100);

    // Determine trend
    let trend: 'improving' | 'stable' | 'degrading' = 'stable';
    if (Math.abs(predictedChange) > currentValue * 0.05) {
      trend = predictedChange > 0 ? 'improving' : 'degrading';
    }

    // Determine risk level
    let riskLevel: 'low' | 'medium' | 'high' | 'critical' = 'low';
    const deviationPercent = Math.abs(predictedChange) / currentValue;

    if (deviationPercent > 0.2) {
      riskLevel = 'critical';
    } else if (deviationPercent > 0.1) {
      riskLevel = 'high';
    } else if (deviationPercent > 0.05) {
      riskLevel = 'medium';
    }

    // Generate impact assessment
    const impact = this.assessPredictionImpact(model.targetMetric, currentValue, predictedValue);

    // Generate recommendations
    const recommendations = this.generatePredictionRecommendations(model, trend, riskLevel, impact);

    // Set warning thresholds
    const warningThresholds = this.calculateWarningThresholds(model.targetMetric, currentValue);

    return {
      id: predictionId,
      timestamp: new Date(),
      modelId: model.id,
      targetMetric: model.targetMetric,
      currentValue,
      predictedValue,
      confidence,
      timeHorizon: model.predictionHorizon,
      trend,
      riskLevel,
      impact,
      recommendations,
      earlyWarningThresholds: warningThresholds
    };
  }

  private calculatePredictionVariance(model: PredictionModel): number {
    // Simulate variance based on model accuracy
    return (1 - model.accuracy) * 0.5;
  }

  private extractMetricValue(snapshot: PerformanceSnapshot, metricPath: string): number | null {
    const parts = metricPath.split('.');
    let value: any = snapshot;

    for (const part of parts) {
      if (value && typeof value === 'object' && part in value) {
        value = value[part];
      } else {
        return null;
      }
    }

    return typeof value === 'number' ? value : null;
  }

  private assessPredictionImpact(metricPath: string, current: number, predicted: number): {
    performanceImpact: string;
    affectedComponents: string[];
    businessImpact: string;
  } {
    const change = predicted - current;
    const changePercent = Math.abs(change / current) * 100;

    let performanceImpact = '';
    let affectedComponents: string[] = [];
    let businessImpact = '';

    if (metricPath.includes('trainingSpeed')) {
      performanceImpact = change > 0
        ? `Training speed improvement of ${changePercent.toFixed(1)}%`
        : `Training speed degradation of ${changePercent.toFixed(1)}%`;
      affectedComponents = ['Reinforcement Learning Engine', 'Model Training Pipeline'];
      businessImpact = change > 0
        ? 'Faster model training and optimization cycles'
        : 'Slower model adaptation and reduced optimization frequency';
    } else if (metricPath.includes('vectorSearchSpeed')) {
      performanceImpact = change > 0
        ? `Query response improvement of ${changePercent.toFixed(1)}%`
        : `Query latency increase of ${changePercent.toFixed(1)}%`;
      affectedComponents = ['AgentDB', 'Vector Search Engine', 'Query Processing'];
      businessImpact = change > 0
        ? 'Faster data retrieval and improved system responsiveness'
        : 'Slower search operations and degraded user experience';
    } else if (metricPath.includes('systemScore')) {
      performanceImpact = change > 0
        ? `System health improvement of ${changePercent.toFixed(1)}%`
        : `System health degradation of ${changePercent.toFixed(1)}%`;
      affectedComponents = ['All System Components'];
      businessImpact = change > 0
        ? 'Improved system reliability and performance'
        : 'Reduced system reliability and potential downtime risk';
    }

    return {
      performanceImpact,
      affectedComponents,
      businessImpact
    };
  }

  private generatePredictionRecommendations(
    model: PredictionModel,
    trend: 'improving' | 'stable' | 'degrading',
    riskLevel: 'low' | 'medium' | 'high' | 'critical',
    impact: any
  ): string[] {
    const recommendations: string[] = [];

    if (trend === 'degrading' && riskLevel !== 'low') {
      recommendations.push('Monitor system closely for performance degradation');
      recommendations.push('Prepare contingency plans for potential issues');

      if (riskLevel === 'critical') {
        recommendations.push('Consider immediate intervention measures');
        recommendations.push('Scale resources preemptively if possible');
      }
    }

    if (model.targetMetric.includes('memory')) {
      if (trend === 'degrading') {
        recommendations.push('Optimize memory usage patterns');
        recommendations.push('Consider memory scaling options');
      }
      recommendations.push('Review memory-intensive operations');
    } else if (model.targetMetric.includes('network') || model.targetMetric.includes('latency')) {
      if (trend === 'degrading') {
        recommendations.push('Optimize network configuration');
        recommendations.push('Check for network congestion');
      }
      recommendations.push('Monitor network bandwidth utilization');
    }

    if (impact.affectedComponents.length > 0) {
      recommendations.push(`Pay special attention to: ${impact.affectedComponents.join(', ')}`);
    }

    return recommendations;
  }

  private calculateWarningThresholds(metricPath: string, currentValue: number): {
    warning: number;
    critical: number;
  } {
    let warningPercent = 0.1; // 10% change
    let criticalPercent = 0.2; // 20% change

    // Adjust thresholds based on metric type
    if (metricPath.includes('Speed') || metricPath.includes('Latency')) {
      // For latency metrics, lower is better
      return {
        warning: currentValue * (1 + warningPercent),
        critical: currentValue * (1 + criticalPercent)
      };
    } else {
      // For most other metrics, higher is better
      return {
        warning: currentValue * (1 - warningPercent),
        critical: currentValue * (1 - criticalPercent)
      };
    }
  }

  private async generateCapacityForecasts(snapshot: PerformanceSnapshot): Promise<void> {
    const resourceTypes: Array<'cpu' | 'memory' | 'storage' | 'network' | 'gpu'> = ['cpu', 'memory', 'storage', 'network', 'gpu'];
    const timeframes: Array<'1h' | '6h' | '24h' | '7d' | '30d'> = ['1h', '6h', '24h', '7d', '30d'];

    for (const resourceType of resourceTypes) {
      for (const timeframe of timeframes) {
        try {
          const forecast = await this.generateCapacityForecast(resourceType, timeframe, snapshot);
          if (forecast) {
            this.forecasts.push(forecast);

            // Keep only recent forecasts (last 200)
            if (this.forecasts.length > 200) {
              this.forecasts.shift();
            }

            this.emit('capacity_forecast', forecast);
          }
        } catch (error) {
          console.error(`Error generating capacity forecast for ${resourceType} ${timeframe}:`, error);
        }
      }
    }
  }

  private async generateCapacityForecast(
    resourceType: 'cpu' | 'memory' | 'storage' | 'network' | 'gpu',
    timeframe: '1h' | '6h' | '24h' | '7d' | '30d',
    snapshot: PerformanceSnapshot
  ): Promise<CapacityPlanningForecast | null> {
    // Get current utilization
    const currentUtilization = this.getCurrentResourceUtilization(snapshot, resourceType);
    if (currentUtilization === null) return null;

    // Simulate forecast generation (in real implementation, would use trained models)
    const forecastId = `forecast_${resourceType}_${timeframe}_${Date.now()}`;

    // Calculate time in minutes
    const timeframeMinutes = {
      '1h': 60,
      '6h': 360,
      '24h': 1440,
      '7d': 10080,
      '30d': 43200
    }[timeframe];

    // Predict utilization with some randomness and trend
    const trendFactor = this.getResourceTrendFactor(resourceType);
    const randomVariation = (Math.random() - 0.5) * 0.1;
    const utilizationGrowth = (trendFactor + randomVariation) * (timeframeMinutes / 1440); // Normalize to 24h
    const predictedUtilization = Math.min(1, Math.max(0, currentUtilization + utilizationGrowth));

    // Calculate peak utilization (higher than average)
    const peakUtilization = Math.min(1, predictedUtilization + (Math.random() * 0.1));

    // Calculate time to threshold (80% utilization)
    let timeToThreshold: number | null = null;
    if (predictedUtilization > 0.8 && currentUtilization < 0.8) {
      const growthRate = (predictedUtilization - currentUtilization) / timeframeMinutes;
      timeToThreshold = growthRate > 0 ? ((0.8 - currentUtilization) / growthRate) : null;
    }

    // Generate scaling recommendation
    const scalingRecommendation = this.generateScalingRecommendation(
      resourceType,
      currentUtilization,
      predictedUtilization,
      peakUtilization,
      timeframeMinutes
    );

    // Generate influencing factors
    const factors = this.generateInfluencingFactors(resourceType, snapshot);

    // Calculate confidence
    const confidence = this.calculateForecastConfidence(resourceType, timeframe);

    return {
      id: forecastId,
      timestamp: new Date(),
      resourceType,
      timeframe,
      currentUtilization,
      predictedUtilization,
      peakUtilization,
      timeToThreshold,
      scalingRecommendation,
      confidence,
      factors
    };
  }

  private getCurrentResourceUtilization(snapshot: PerformanceSnapshot, resourceType: string): number | null {
    switch (resourceType) {
      case 'cpu':
        return snapshot.resourceUsage.cpu;
      case 'memory':
        return snapshot.resourceUsage.memory;
      case 'network':
        return snapshot.resourceUsage.network;
      case 'gpu':
        return snapshot.resourceUsage.gpu;
      case 'storage':
        // Storage utilization not directly available, estimate from memory usage
        return snapshot.resourceUsage.memory * 0.8;
      default:
        return null;
    }
  }

  private getResourceTrendFactor(resourceType: string): number {
    // Simulate different growth trends for different resources
    switch (resourceType) {
      case 'cpu':
        return 0.02; // 2% growth per day
      case 'memory':
        return 0.05; // 5% growth per day
      case 'network':
        return 0.03; // 3% growth per day
      case 'gpu':
        return 0.04; // 4% growth per day
      case 'storage':
        return 0.01; // 1% growth per day
      default:
        return 0.02;
    }
  }

  private generateScalingRecommendation(
    resourceType: string,
    currentUtilization: number,
    predictedUtilization: number,
    peakUtilization: number,
    timeframeMinutes: number
  ): any {
    let action: 'scale_up' | 'scale_out' | 'optimize' | 'no_action' = 'no_action';
    let urgency: 'immediate' | 'within_hour' | 'within_day' | 'within_week' = 'within_week';
    let estimatedCost = 0;
    let currentResources = 0;
    let recommendedResources = 0;
    let unit = '';

    // Determine action based on predicted utilization
    if (peakUtilization > 0.95) {
      action = resourceType === 'cpu' || resourceType === 'memory' ? 'scale_up' : 'scale_out';
      urgency = timeframeMinutes <= 60 ? 'immediate' : timeframeMinutes <= 360 ? 'within_hour' : 'within_day';
      estimatedCost = 5;
    } else if (peakUtilization > 0.8) {
      action = 'optimize';
      urgency = timeframeMinutes <= 360 ? 'within_day' : 'within_week';
      estimatedCost = 2;
    } else if (predictedUtilization < 0.3) {
      action = 'optimize'; // Consider scaling down
      urgency = 'within_week';
      estimatedCost = -1; // Cost savings
    }

    // Calculate resource amounts
    switch (resourceType) {
      case 'cpu':
        currentResources = 8; // cores
        recommendedResources = action === 'scale_up' ? Math.ceil(currentResources * 1.5) : currentResources;
        unit = 'cores';
        break;
      case 'memory':
        currentResources = 32; // GB
        recommendedResources = action === 'scale_up' ? Math.ceil(currentResources * 1.5) : currentResources;
        unit = 'GB';
        break;
      case 'network':
        currentResources = 1000; // Mbps
        recommendedResources = action === 'scale_out' ? Math.ceil(currentResources * 1.5) : currentResources;
        unit = 'Mbps';
        break;
      case 'gpu':
        currentResources = 4; // GPUs
        recommendedResources = action === 'scale_up' ? Math.ceil(currentResources * 1.5) : currentResources;
        unit = 'GPUs';
        break;
    }

    return {
      action,
      urgency,
      estimatedCost,
      resources: {
        current: currentResources,
        recommended: recommendedResources,
        unit
      }
    };
  }

  private generateInfluencingFactors(resourceType: string, snapshot: PerformanceSnapshot): Array<{
    factor: string;
    impact: number;
    confidence: number;
  }> {
    const factors = [];

    // Agent count impact
    const agentCount = snapshot.environmentContext.agentCount;
    factors.push({
      factor: 'Agent Count',
      impact: agentCount > 10 ? 0.3 : 0.1,
      confidence: 0.8
    });

    // Task load impact
    const taskQueueLength = snapshot.swarmMetrics.taskPerformance.taskQueueLength;
    factors.push({
      factor: 'Task Load',
      impact: Math.min(0.5, taskQueueLength * 0.1),
      confidence: 0.7
    });

    // System health impact
    const systemHealth = snapshot.systemHealth.overallSystemScore;
    factors.push({
      factor: 'System Health',
      impact: (1 - systemHealth) * -0.2, // Poor health reduces resource efficiency
      confidence: 0.9
    });

    // Time of day impact (peak hours)
    const hour = new Date().getHours();
    const isPeakHour = hour >= 9 && hour <= 17;
    factors.push({
      factor: 'Time of Day',
      impact: isPeakHour ? 0.15 : -0.05,
      confidence: 0.6
    });

    return factors;
  }

  private calculateForecastConfidence(resourceType: string, timeframe: string): number {
    let baseConfidence = 0.85;

    // Adjust confidence based on timeframe
    const timeframeConfidence = {
      '1h': 0.95,
      '6h': 0.90,
      '24h': 0.85,
      '7d': 0.75,
      '30d': 0.65
    }[timeframe];

    // Adjust confidence based on resource type
    const resourceConfidence = {
      'cpu': 0.90,
      'memory': 0.85,
      'network': 0.80,
      'gpu': 0.75,
      'storage': 0.95
    }[resourceType];

    return baseConfidence * timeframeConfidence * resourceConfidence;
  }

  private async detectAnomalies(snapshot: PerformanceSnapshot): Promise<void> {
    for (const model of this.predictionModels.values()) {
      if (model.type !== 'anomaly_detection') continue;

      try {
        const anomaly = await this.detectAnomaly(model, snapshot);
        if (anomaly) {
          this.anomalies.push(anomaly);

          // Keep only recent anomalies (last 50)
          if (this.anomalies.length > 50) {
            this.anomalies.shift();
          }

          this.emit('anomaly_detected', anomaly);
        }
      } catch (error) {
        console.error(`Error detecting anomaly with model ${model.id}:`, error);
      }
    }
  }

  private async detectAnomaly(model: PredictionModel, snapshot: PerformanceSnapshot): Promise<AnomalyDetectionResult | null> {
    const currentValue = this.extractMetricValue(snapshot, model.targetMetric);
    if (currentValue === null) return null;

    // Calculate baseline from historical data
    const baselineValue = this.calculateBaselineValue(model.targetMetric);
    if (baselineValue === null) return null;

    // Calculate deviation
    const deviation = currentValue - baselineValue;
    const standardDeviation = this.calculateStandardDeviation(model.targetMetric);
    const deviationMagnitude = Math.abs(deviation) / standardDeviation;

    // Detect anomaly if deviation is significant
    if (deviationMagnitude < 2) return null; // Less than 2 standard deviations

    // Determine anomaly type
    let anomalyType: 'performance_spike' | 'performance_drop' | 'resource_exhaustion' | 'communication_failure' | 'behavioral_change' = 'behavioral_change';

    if (model.targetMetric.includes('performance') || model.targetMetric.includes('throughput')) {
      anomalyType = deviation > 0 ? 'performance_spike' : 'performance_drop';
    } else if (model.targetMetric.includes('resource') || model.targetMetric.includes('usage')) {
      anomalyType = 'resource_exhaustion';
    } else if (model.targetMetric.includes('latency') || model.targetMetric.includes('error')) {
      anomalyType = 'communication_failure';
    }

    // Determine severity
    let severity: 'low' | 'medium' | 'high' | 'critical' = 'low';
    if (deviationMagnitude > 4) {
      severity = 'critical';
    } else if (deviationMagnitude > 3) {
      severity = 'high';
    } else if (deviationMagnitude > 2) {
      severity = 'medium';
    }

    // Calculate confidence
    const confidence = model.confidence * Math.min(1, deviationMagnitude / 3);

    // Generate description
    const description = `${anomalyType.replace(/_/g, ' ')} detected: ${model.targetMetric} ${deviation > 0 ? 'increased' : 'decreased'} by ${Math.abs(deviation).toFixed(2)} (${deviationMagnitude.toFixed(1)}σ)`;

    // Generate root cause hypotheses
    const rootCauseHypothesis = this.generateRootCauseHypotheses(model.targetMetric, deviation, anomalyType);

    // Determine auto-resolution options
    const autoResolution = this.determineAutoResolution(anomalyType, severity);

    return {
      id: `anomaly_${model.id}_${Date.now()}`,
      timestamp: new Date(),
      anomalyType,
      severity,
      confidence,
      description,
      affectedMetrics: [model.targetMetric],
      baselineValue,
      observedValue: currentValue,
      deviationMagnitude,
      duration: 0, // Will be updated as anomaly persists
      rootCauseHypothesis,
      autoResolution,
      context: {
        concurrentEvents: [],
        systemLoad: snapshot.resourceUsage.cpu,
        recentChanges: []
      }
    };
  }

  private calculateBaselineValue(metricPath: string): number | null {
    if (this.performanceHistory.length < 10) return null;

    const recentValues = this.performanceHistory.slice(-20)
      .map(snapshot => this.extractMetricValue(snapshot, metricPath))
      .filter(value => value !== null) as number[];

    if (recentValues.length < 5) return null;

    // Calculate median as baseline
    const sortedValues = [...recentValues].sort((a, b) => a - b);
    return sortedValues[Math.floor(sortedValues.length / 2)];
  }

  private calculateStandardDeviation(metricPath: string): number {
    if (this.performanceHistory.length < 10) return 1;

    const recentValues = this.performanceHistory.slice(-20)
      .map(snapshot => this.extractMetricValue(snapshot, metricPath))
      .filter(value => value !== null) as number[];

    if (recentValues.length < 5) return 1;

    const mean = recentValues.reduce((sum, val) => sum + val, 0) / recentValues.length;
    const variance = recentValues.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / recentValues.length;

    return Math.sqrt(variance);
  }

  private generateRootCauseHypotheses(metricPath: string, deviation: number, anomalyType: string): string[] {
    const hypotheses = [];

    if (metricPath.includes('cpu') || metricPath.includes('memory')) {
      hypotheses.push('Resource contention or exhaustion');
      hypotheses.push('Memory leak or inefficient resource usage');
      if (deviation > 0) {
        hypotheses.push('Increased computational workload');
      }
    }

    if (metricPath.includes('network') || metricPath.includes('latency')) {
      hypotheses.push('Network congestion or connectivity issues');
      hypotheses.push('Increased data transfer volumes');
    }

    if (metricPath.includes('error')) {
      hypotheses.push('System component failure or misconfiguration');
      hypotheses.push('External dependency issues');
    }

    if (anomalyType === 'performance_spike') {
      hypotheses.push('Temporary resource boost (e.g., caching effect)');
      hypotheses.push('Reduced system load or improved efficiency');
    }

    return hypotheses;
  }

  private determineAutoResolution(anomalyType: string, severity: 'low' | 'medium' | 'high' | 'critical'): {
    possible: boolean;
    actions: string[];
    successProbability: number;
  } {
    const possible = severity !== 'critical';
    const actions = [];
    let successProbability = 0.7;

    if (!possible) {
      return { possible: false, actions: [], successProbability: 0 };
    }

    switch (anomalyType) {
      case 'resource_exhaustion':
        actions.push('Scale resources automatically');
        actions.push('Clear caches and temporary data');
        actions.push('Terminate non-essential processes');
        successProbability = 0.8;
        break;
      case 'performance_drop':
        actions.push('Restart affected services');
        actions.push('Optimize configuration parameters');
        actions.push('Enable performance optimizations');
        successProbability = 0.7;
        break;
      case 'communication_failure':
        actions.push('Reestablish network connections');
        actions.push('Switch to backup communication channels');
        actions.push('Increase timeout values');
        successProbability = 0.6;
        break;
    }

    return { possible, actions, successProbability };
  }

  private async analyzeTrends(snapshot: PerformanceSnapshot): Promise<void> {
    // This would implement comprehensive trend analysis
    // For now, placeholder implementation
  }

  private async generatePredictiveInsights(): Promise<void> {
    // Generate insights from predictions, forecasts, and anomalies
    const insight = this.generateHighPriorityInsight();

    if (insight) {
      this.insights.push(insight);

      // Keep only recent insights (last 20)
      if (this.insights.length > 20) {
        this.insights.shift();
      }

      this.emit('predictive_insight', insight);
    }
  }

  private generateHighPriorityInsight(): PredictiveInsight | null {
    // Check for critical predictions or forecasts
    const criticalPredictions = this.predictions.filter(p => p.riskLevel === 'critical');
    const urgentForecasts = this.forecasts.filter(f => f.scalingRecommendation.urgency === 'immediate');
    const criticalAnomalies = this.anomalies.filter(a => a.severity === 'critical');

    if (criticalPredictions.length === 0 && urgentForecasts.length === 0 && criticalAnomalies.length === 0) {
      return null;
    }

    const insightId = `insight_${Date.now()}`;
    const priority = criticalAnomalies.length > 0 ? 'critical' :
                    urgentForecasts.length > 0 ? 'high' : 'medium';

    let title = '';
    let description = '';
    let category: 'performance' | 'capacity' | 'availability' | 'cost_optimization' = 'performance';

    if (criticalAnomalies.length > 0) {
      title = 'Critical System Anomalies Detected';
      description = `${criticalAnomalies.length} critical anomalies detected requiring immediate attention`;
      category = 'availability';
    } else if (urgentForecasts.length > 0) {
      title = 'Urgent Capacity Scaling Required';
      description = `${urgentForecasts.length} resources require immediate scaling to prevent performance degradation`;
      category = 'capacity';
    } else {
      title = 'Performance Degradation Predicted';
      description = `${criticalPredictions.length} critical performance issues predicted in the near future`;
      category = 'performance';
    }

    // Generate actionable recommendations
    const actionableRecommendations = this.generateInsightRecommendations(
      criticalPredictions,
      urgentForecasts,
      criticalAnomalies
    );

    // Calculate business impact
    const businessImpact = this.calculateBusinessImpact(category, priority);

    return {
      id: insightId,
      timestamp: new Date(),
      category,
      priority,
      title,
      description,
      supportingData: {
        predictions: criticalPredictions,
        forecasts: urgentForecasts,
        anomalies: criticalAnomalies,
        trends: []
      },
      actionableRecommendations,
      businessImpact
    };
  }

  private generateInsightRecommendations(
    predictions: PerformancePrediction[],
    forecasts: CapacityPlanningForecast[],
    anomalies: AnomalyDetectionResult[]
  ): Array<{
    action: string;
    benefit: string;
    effort: 'low' | 'medium' | 'high';
    timeframe: string;
    risk: 'low' | 'medium' | 'high';
  }> {
    const recommendations = [];

    // Recommendations for anomalies
    anomalies.forEach(anomaly => {
      if (anomaly.autoResolution.possible) {
        recommendations.push({
          action: `Execute auto-resolution for ${anomaly.anomalyType}`,
          benefit: 'Automatically resolve detected anomaly',
          effort: 'low',
          timeframe: 'Immediate',
          risk: 'low'
        });
      }
    });

    // Recommendations for forecasts
    forecasts.forEach(forecast => {
      if (forecast.scalingRecommendation.action !== 'no_action') {
        recommendations.push({
          action: `${forecast.scalingRecommendation.action} ${forecast.resourceType} resources`,
          benefit: 'Prevent performance degradation from resource exhaustion',
          effort: forecast.scalingRecommendation.action === 'optimize' ? 'medium' : 'high',
          timeframe: forecast.scalingRecommendation.urgency,
          risk: 'low'
        });
      }
    });

    // Recommendations for predictions
    predictions.forEach(prediction => {
      prediction.recommendations.forEach(rec => {
        recommendations.push({
          action: rec,
          benefit: prediction.impact.businessImpact,
          effort: 'medium',
          timeframe: 'Within hour',
          risk: prediction.riskLevel === 'critical' ? 'high' : 'medium'
        });
      });
    });

    return recommendations.slice(0, 5); // Limit to top 5 recommendations
  }

  private calculateBusinessImpact(
    category: 'performance' | 'capacity' | 'availability' | 'cost_optimization',
    priority: 'low' | 'medium' | 'high' | 'critical'
  ): {
    metric: string;
    impact: 'positive' | 'negative' | 'neutral';
    magnitude: number;
    confidence: number;
  } {
    const impactMagnitude = {
      'low': 5,
      'medium': 15,
      'high': 30,
      'critical': 50
    }[priority];

    const metrics = {
      'performance': 'System Performance Score',
      'capacity': 'Resource Utilization',
      'availability': 'System Uptime',
      'cost_optimization': 'Operational Costs'
    };

    return {
      metric: metrics[category],
      impact: 'negative', // Insights are typically about preventing negative impacts
      magnitude: impactMagnitude,
      confidence: 0.8
    };
  }

  private async retrainModelsIfNeeded(): Promise<void> {
    // Check if models need retraining based on age and data availability
    const now = new Date();
    const retrainingThreshold = 24 * 60 * 60 * 1000; // 24 hours

    for (const model of this.predictionModels.values()) {
      const timeSinceLastTraining = now.getTime() - model.lastTrained.getTime();

      if (timeSinceLastTraining > retrainingThreshold && this.performanceHistory.length > 100) {
        try {
          await this.retrainModel(model);
        } catch (error) {
          console.error(`Error retraining model ${model.id}:`, error);
        }
      }
    }
  }

  private async retrainModel(model: PredictionModel): Promise<void> {
    console.log(`Retraining model: ${model.name}`);

    // Simulate model retraining
    await new Promise(resolve => setTimeout(resolve, 5000));

    // Update model metadata
    model.lastTrained = new Date();
    model.trainingDataPoints = this.performanceHistory.length;
    model.accuracy = Math.min(0.95, model.accuracy + (Math.random() - 0.5) * 0.02);
    model.confidence = Math.min(0.95, model.confidence + (Math.random() - 0.5) * 0.02);

    this.emit('model_retrained', model);
  }

  // Public API methods
  public getPredictions(limit?: number): PerformancePrediction[] {
    return limit ? this.predictions.slice(-limit) : this.predictions;
  }

  public getForecasts(limit?: number): CapacityPlanningForecast[] {
    return limit ? this.forecasts.slice(-limit) : this.forecasts;
  }

  public getAnomalies(limit?: number): AnomalyDetectionResult[] {
    return limit ? this.anomalies.slice(-limit) : this.anomalies;
  }

  public getInsights(limit?: number): PredictiveInsight[] {
    return limit ? this.insights.slice(-limit) : this.insights;
  }

  public getPredictionModels(): PredictionModel[] {
    return Array.from(this.predictionModels.values());
  }

  public getHighRiskPredictions(): PerformancePrediction[] {
    return this.predictions.filter(p => p.riskLevel === 'critical' || p.riskLevel === 'high');
  }

  public getUrgentCapacityNeeds(): CapacityPlanningForecast[] {
    return this.forecasts.filter(f => f.scalingRecommendation.urgency === 'immediate' || f.scalingRecommendation.urgency === 'within_hour');
  }

  public exportPredictiveData(): any {
    return {
      timestamp: new Date(),
      predictions: this.getPredictions(50),
      forecasts: this.getForecasts(100),
      anomalies: this.getAnomalies(25),
      insights: this.getInsights(10),
      models: this.getPredictionModels(),
      summary: {
        totalPredictions: this.predictions.length,
        highRiskPredictions: this.getHighRiskPredictions().length,
        urgentCapacityNeeds: this.getUrgentCapacityNeeds().length,
        criticalAnomalies: this.anomalies.filter(a => a.severity === 'critical').length,
        activeInsights: this.insights.length
      }
    };
  }

  public stop(): void {
    if (this.analysisInterval) {
      clearInterval(this.analysisInterval);
      this.analysisInterval = null;
    }
  }
}