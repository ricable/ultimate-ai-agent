/**
 * Performance Management (PM) Module
 *
 * The sensory system of the swarm that transforms raw counter data
 * into actionable insights. Implements multi-granularity analysis:
 * - 15-min (ROP): Immediate anomaly detection
 * - Hourly: Trend analysis & forecasting
 * - Daily: Pattern recognition & seasonality
 * - Weekly: Capacity planning
 *
 * Uses:
 * - Attractor dynamics for contextual anomaly detection
 * - TCN/N-BEATS for predictive resource management
 * - Multi-dimensional KPI correlation analysis
 */

import { EventEmitter } from 'eventemitter3';
import {
  CellKPIs,
  SliceKPIs,
  TimeSeries,
  Anomaly,
  AnomalyType,
  CellGlobalIdentity
} from '../../types/index.js';
import { MidstreamerEngine, DynamicTimeWarping, ChaosAnalyzer } from '../../core/midstreamer/index.js';

// ============================================================================
// KPI AGGREGATOR
// ============================================================================

interface AggregatedKPIs {
  granularity: '15min' | '1hour' | '1day' | '1week';
  timestamp: Date;
  cellId: string;

  // Aggregated metrics
  metrics: {
    name: string;
    min: number;
    max: number;
    avg: number;
    p5: number;
    p50: number;
    p95: number;
    std: number;
    sum: number;
    count: number;
  }[];

  // Derived metrics
  availability: number;       // % of samples with good coverage
  congestionIndex: number;    // 0-1, based on PRB utilization
  qualityScore: number;       // 0-100, composite quality metric
}

class KPIAggregator {
  /**
   * Aggregate raw KPIs to specified granularity
   */
  aggregate(
    rawKpis: CellKPIs[],
    targetGranularity: '15min' | '1hour' | '1day' | '1week'
  ): AggregatedKPIs[] {
    if (rawKpis.length === 0) return [];

    // Group by time bucket
    const buckets = this.groupByTimeBucket(rawKpis, targetGranularity);
    const results: AggregatedKPIs[] = [];

    for (const [bucketKey, kpis] of buckets) {
      const aggregated = this.aggregateBucket(kpis, targetGranularity, bucketKey);
      results.push(aggregated);
    }

    return results.sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());
  }

  private groupByTimeBucket(
    kpis: CellKPIs[],
    granularity: '15min' | '1hour' | '1day' | '1week'
  ): Map<string, CellKPIs[]> {
    const buckets = new Map<string, CellKPIs[]>();

    for (const kpi of kpis) {
      const bucketKey = this.getBucketKey(kpi.timestamp, granularity);
      if (!buckets.has(bucketKey)) {
        buckets.set(bucketKey, []);
      }
      buckets.get(bucketKey)!.push(kpi);
    }

    return buckets;
  }

  private getBucketKey(timestamp: Date, granularity: string): string {
    const d = new Date(timestamp);

    switch (granularity) {
      case '15min':
        d.setMinutes(Math.floor(d.getMinutes() / 15) * 15, 0, 0);
        break;
      case '1hour':
        d.setMinutes(0, 0, 0);
        break;
      case '1day':
        d.setHours(0, 0, 0, 0);
        break;
      case '1week':
        const day = d.getDay();
        d.setDate(d.getDate() - day);
        d.setHours(0, 0, 0, 0);
        break;
    }

    return d.toISOString();
  }

  private aggregateBucket(
    kpis: CellKPIs[],
    granularity: '15min' | '1hour' | '1day' | '1week',
    bucketKey: string
  ): AggregatedKPIs {
    const metricNames = [
      'dlThroughput', 'ulThroughput', 'callDropRate', 'hoSuccessRate',
      'prbUtilizationDl', 'prbUtilizationUl', 'avgSinrDl', 'avgSinrUl',
      'iotUl', 'activeUsers', 'blerDl', 'blerUl'
    ];

    const metrics = metricNames.map(name => {
      const values = kpis.map(k => (k as any)[name]).filter(v => typeof v === 'number');
      return this.computeStats(name, values);
    });

    // Compute derived metrics
    const availability = kpis.filter(k => k.avgRsrp > -110).length / kpis.length;
    const avgPrbUtil = kpis.reduce((sum, k) => sum + k.prbUtilizationDl, 0) / kpis.length;
    const congestionIndex = Math.min(1, avgPrbUtil / 80); // 80% threshold

    const qualityScore = this.computeQualityScore(kpis);

    return {
      granularity,
      timestamp: new Date(bucketKey),
      cellId: kpis[0]?.cgi ? this.cgiToString(kpis[0].cgi) : 'unknown',
      metrics,
      availability,
      congestionIndex,
      qualityScore
    };
  }

  private computeStats(name: string, values: number[]): AggregatedKPIs['metrics'][0] {
    if (values.length === 0) {
      return { name, min: 0, max: 0, avg: 0, p5: 0, p50: 0, p95: 0, std: 0, sum: 0, count: 0 };
    }

    const sorted = [...values].sort((a, b) => a - b);
    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / values.length;
    const variance = values.reduce((acc, v) => acc + Math.pow(v - avg, 2), 0) / values.length;

    return {
      name,
      min: sorted[0],
      max: sorted[sorted.length - 1],
      avg,
      p5: sorted[Math.floor(values.length * 0.05)],
      p50: sorted[Math.floor(values.length * 0.50)],
      p95: sorted[Math.floor(values.length * 0.95)],
      std: Math.sqrt(variance),
      sum,
      count: values.length
    };
  }

  private computeQualityScore(kpis: CellKPIs[]): number {
    if (kpis.length === 0) return 0;

    let score = 100;

    // Penalize for high drop rate
    const avgDropRate = kpis.reduce((s, k) => s + k.callDropRate, 0) / kpis.length;
    score -= avgDropRate * 100; // 1% drop rate = -1 point

    // Penalize for low HO success
    const avgHoSuccess = kpis.reduce((s, k) => s + k.hoSuccessRate, 0) / kpis.length;
    score -= (100 - avgHoSuccess);

    // Penalize for high BLER
    const avgBler = kpis.reduce((s, k) => s + k.blerDl, 0) / kpis.length;
    score -= avgBler * 50;

    // Penalize for congestion
    const avgPrbUtil = kpis.reduce((s, k) => s + k.prbUtilizationDl, 0) / kpis.length;
    if (avgPrbUtil > 80) {
      score -= (avgPrbUtil - 80);
    }

    return Math.max(0, Math.min(100, score));
  }

  private cgiToString(cgi: CellGlobalIdentity): string {
    return `${cgi.mcc}-${cgi.mnc}-${cgi.gnbId}-${cgi.cellId}`;
  }
}

// ============================================================================
// TREND ANALYZER
// ============================================================================

interface TrendAnalysis {
  metric: string;
  direction: 'improving' | 'degrading' | 'stable';
  slope: number;           // Change per hour
  confidence: number;
  forecast: number[];      // Predicted values
  breachTime?: Date;       // When threshold will be breached
}

class TrendAnalyzer {
  /**
   * Analyze trends in time series data
   */
  analyzeTrend(
    timeSeries: TimeSeries,
    threshold?: number
  ): TrendAnalysis {
    const { values, timestamps } = timeSeries;

    if (values.length < 3) {
      return {
        metric: timeSeries.metric,
        direction: 'stable',
        slope: 0,
        confidence: 0,
        forecast: []
      };
    }

    // Linear regression for slope
    const { slope, intercept, r2 } = this.linearRegression(values);

    // Determine direction
    const stdDev = this.standardDeviation(values);
    const significantSlope = Math.abs(slope) > stdDev / values.length;

    let direction: 'improving' | 'degrading' | 'stable' = 'stable';
    if (significantSlope) {
      // For metrics where higher is better (throughput)
      direction = slope > 0 ? 'improving' : 'degrading';
    }

    // Generate forecast (next 8 periods)
    const forecast: number[] = [];
    for (let i = 1; i <= 8; i++) {
      const predictedValue = intercept + slope * (values.length + i);
      forecast.push(predictedValue);
    }

    // Calculate breach time if threshold provided
    let breachTime: Date | undefined;
    if (threshold !== undefined && slope !== 0) {
      const periodsToThreshold = (threshold - (intercept + slope * values.length)) / slope;
      if (periodsToThreshold > 0 && periodsToThreshold < 1000) {
        const msPerPeriod = this.getMsPerPeriod(timeSeries.granularity);
        breachTime = new Date(timestamps[timestamps.length - 1].getTime() + periodsToThreshold * msPerPeriod);
      }
    }

    return {
      metric: timeSeries.metric,
      direction,
      slope,
      confidence: r2,
      forecast,
      breachTime
    };
  }

  private linearRegression(values: number[]): { slope: number; intercept: number; r2: number } {
    const n = values.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += values[i];
      sumXY += i * values[i];
      sumX2 += i * i;
      sumY2 += values[i] * values[i];
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    // R-squared
    const yMean = sumY / n;
    let ssTotal = 0, ssRes = 0;
    for (let i = 0; i < n; i++) {
      const predicted = intercept + slope * i;
      ssTotal += Math.pow(values[i] - yMean, 2);
      ssRes += Math.pow(values[i] - predicted, 2);
    }
    const r2 = ssTotal > 0 ? 1 - ssRes / ssTotal : 0;

    return { slope, intercept, r2 };
  }

  private standardDeviation(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  private getMsPerPeriod(granularity: string): number {
    switch (granularity) {
      case '15min': return 15 * 60 * 1000;
      case '1hour': return 60 * 60 * 1000;
      case '1day': return 24 * 60 * 60 * 1000;
      case '1week': return 7 * 24 * 60 * 60 * 1000;
      default: return 15 * 60 * 1000;
    }
  }
}

// ============================================================================
// CORRELATION ANALYZER
// ============================================================================

interface CorrelationResult {
  metric1: string;
  metric2: string;
  correlation: number;      // Pearson correlation coefficient
  significance: number;     // p-value approximation
  isAnomalous: boolean;     // Correlation break detected
}

class CorrelationAnalyzer {
  // Expected correlations based on network physics
  private expectedCorrelations: Map<string, Map<string, number>> = new Map([
    ['activeUsers', new Map([
      ['prbUtilizationDl', 0.8],
      ['dlThroughput', 0.7],
      ['congestionIndex', 0.6]
    ])],
    ['iotUl', new Map([
      ['blerUl', 0.5],
      ['avgSinrUl', -0.7]
    ])],
    ['prbUtilizationDl', new Map([
      ['congestionIndex', 0.9],
      ['callDropRate', 0.4]
    ])]
  ]);

  /**
   * Compute correlations between all metric pairs
   */
  analyzeCorrelations(kpis: CellKPIs[]): CorrelationResult[] {
    const metricNames = [
      'dlThroughput', 'ulThroughput', 'callDropRate', 'activeUsers',
      'prbUtilizationDl', 'iotUl', 'avgSinrUl', 'blerUl'
    ];

    const results: CorrelationResult[] = [];

    for (let i = 0; i < metricNames.length; i++) {
      for (let j = i + 1; j < metricNames.length; j++) {
        const m1 = metricNames[i];
        const m2 = metricNames[j];

        const values1 = kpis.map(k => (k as any)[m1]).filter(v => typeof v === 'number');
        const values2 = kpis.map(k => (k as any)[m2]).filter(v => typeof v === 'number');

        if (values1.length !== values2.length || values1.length < 3) continue;

        const correlation = this.pearsonCorrelation(values1, values2);
        const significance = this.approximateSignificance(correlation, values1.length);

        // Check if correlation breaks expected pattern
        const expected = this.expectedCorrelations.get(m1)?.get(m2) ||
                        this.expectedCorrelations.get(m2)?.get(m1);
        const isAnomalous = expected !== undefined &&
                           Math.abs(correlation - expected) > 0.4;

        results.push({
          metric1: m1,
          metric2: m2,
          correlation,
          significance,
          isAnomalous
        });
      }
    }

    return results;
  }

  private pearsonCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;

    for (let i = 0; i < n; i++) {
      sumX += x[i];
      sumY += y[i];
      sumXY += x[i] * y[i];
      sumX2 += x[i] * x[i];
      sumY2 += y[i] * y[i];
    }

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator > 0 ? numerator / denominator : 0;
  }

  private approximateSignificance(r: number, n: number): number {
    // T-test approximation for correlation significance
    if (Math.abs(r) >= 1) return 0;
    const t = r * Math.sqrt((n - 2) / (1 - r * r));
    // Approximate p-value (simplified)
    return Math.exp(-0.5 * t * t) * 2;
  }
}

// ============================================================================
// CAPACITY PLANNER
// ============================================================================

interface CapacityForecast {
  cellId: string;
  metric: string;
  currentValue: number;
  projectedValue: number;
  daysToCapacity: number;     // Days until 100% capacity
  recommendation: string;
}

class CapacityPlanner {
  private thresholds = {
    prbUtilizationDl: 80,
    prbUtilizationUl: 80,
    activeUsers: 450,
    congestionIndex: 0.8
  };

  /**
   * Project capacity needs based on trends
   */
  projectCapacity(
    historicalData: AggregatedKPIs[],
    trendAnalyzer: TrendAnalyzer
  ): CapacityForecast[] {
    const forecasts: CapacityForecast[] = [];

    if (historicalData.length < 7) return forecasts;

    const cellId = historicalData[0].cellId;

    for (const [metric, threshold] of Object.entries(this.thresholds)) {
      const metricData = historicalData.map(d => {
        const m = d.metrics.find(m => m.name === metric);
        return m?.avg || 0;
      });

      const timestamps = historicalData.map(d => d.timestamp);

      const timeSeries: TimeSeries = {
        timestamps,
        values: metricData,
        granularity: historicalData[0].granularity,
        metric
      };

      const trend = trendAnalyzer.analyzeTrend(timeSeries, threshold);
      const currentValue = metricData[metricData.length - 1];

      let daysToCapacity = Infinity;
      if (trend.slope > 0) {
        const remaining = threshold - currentValue;
        const slopePerDay = trend.slope * this.getPeriodsPerDay(timeSeries.granularity);
        daysToCapacity = remaining / slopePerDay;
      }

      let recommendation = '';
      if (daysToCapacity < 7) {
        recommendation = `URGENT: ${metric} will reach capacity in ${Math.round(daysToCapacity)} days. Immediate action required.`;
      } else if (daysToCapacity < 30) {
        recommendation = `WARNING: ${metric} trending towards capacity. Plan expansion within ${Math.round(daysToCapacity)} days.`;
      } else if (daysToCapacity < 90) {
        recommendation = `MONITOR: ${metric} growing steadily. Review in ${Math.round(daysToCapacity / 2)} days.`;
      } else {
        recommendation = `OK: ${metric} stable with sufficient headroom.`;
      }

      forecasts.push({
        cellId,
        metric,
        currentValue,
        projectedValue: trend.forecast[7] || currentValue,
        daysToCapacity: Math.min(daysToCapacity, 365),
        recommendation
      });
    }

    return forecasts;
  }

  private getPeriodsPerDay(granularity: string): number {
    switch (granularity) {
      case '15min': return 96;
      case '1hour': return 24;
      case '1day': return 1;
      case '1week': return 1 / 7;
      default: return 96;
    }
  }
}

// ============================================================================
// MAIN PERFORMANCE MANAGER CLASS
// ============================================================================

export interface PerformanceManagerConfig {
  cellId: string;
  enableAnomalyDetection: boolean;
  enableTrendAnalysis: boolean;
  enableCapacityPlanning: boolean;
  anomalyThresholds: {
    dropRateWarning: number;
    dropRateCritical: number;
    blerWarning: number;
    blerCritical: number;
    congestionWarning: number;
    congestionCritical: number;
  };
}

const DEFAULT_PM_CONFIG: PerformanceManagerConfig = {
  cellId: 'default-cell',
  enableAnomalyDetection: true,
  enableTrendAnalysis: true,
  enableCapacityPlanning: true,
  anomalyThresholds: {
    dropRateWarning: 0.5,
    dropRateCritical: 1.0,
    blerWarning: 0.1,
    blerCritical: 0.2,
    congestionWarning: 0.7,
    congestionCritical: 0.9
  }
};

export class PerformanceManager extends EventEmitter {
  config: PerformanceManagerConfig;
  midstreamer: MidstreamerEngine;
  kpiAggregator: KPIAggregator;
  trendAnalyzer: TrendAnalyzer;
  correlationAnalyzer: CorrelationAnalyzer;
  capacityPlanner: CapacityPlanner;

  // Data storage
  private rawKpiBuffer: CellKPIs[] = [];
  private aggregatedCache: Map<string, AggregatedKPIs[]> = new Map();
  private maxBufferSize: number = 10000;

  // Statistics
  private anomaliesDetected: number = 0;
  private trendsAnalyzed: number = 0;
  private capacityForecastsGenerated: number = 0;

  constructor(config: Partial<PerformanceManagerConfig> = {}) {
    super();
    this.config = { ...DEFAULT_PM_CONFIG, ...config };

    this.midstreamer = new MidstreamerEngine();
    this.kpiAggregator = new KPIAggregator();
    this.trendAnalyzer = new TrendAnalyzer();
    this.correlationAnalyzer = new CorrelationAnalyzer();
    this.capacityPlanner = new CapacityPlanner();
  }

  /**
   * Ingest new KPI data point
   */
  ingestKPI(kpi: CellKPIs): void {
    this.rawKpiBuffer.push(kpi);

    // Enforce buffer limit
    if (this.rawKpiBuffer.length > this.maxBufferSize) {
      this.rawKpiBuffer = this.rawKpiBuffer.slice(-this.maxBufferSize / 2);
    }

    // Real-time anomaly detection
    if (this.config.enableAnomalyDetection) {
      this.detectRealTimeAnomalies(kpi);
    }

    this.emit('kpi-ingested', { cellId: this.config.cellId, timestamp: kpi.timestamp });
  }

  /**
   * Process KPIs at specified granularity
   */
  processGranularity(granularity: '15min' | '1hour' | '1day' | '1week'): {
    aggregated: AggregatedKPIs[];
    trends: TrendAnalysis[];
    anomalies: Anomaly[];
    correlations: CorrelationResult[];
  } {
    // Aggregate KPIs
    const aggregated = this.kpiAggregator.aggregate(this.rawKpiBuffer, granularity);
    this.aggregatedCache.set(granularity, aggregated);

    // Analyze trends
    const trends: TrendAnalysis[] = [];
    if (this.config.enableTrendAnalysis && aggregated.length > 3) {
      for (const metric of aggregated[0].metrics) {
        const values = aggregated.map(a => {
          const m = a.metrics.find(m => m.name === metric.name);
          return m?.avg || 0;
        });

        const timeSeries: TimeSeries = {
          timestamps: aggregated.map(a => a.timestamp),
          values,
          granularity,
          metric: metric.name
        };

        const trend = this.trendAnalyzer.analyzeTrend(timeSeries);
        trends.push(trend);
        this.trendsAnalyzed++;
      }
    }

    // Detect anomalies using Midstreamer
    const anomalies: Anomaly[] = [];
    if (this.config.enableAnomalyDetection) {
      for (const metric of aggregated[0]?.metrics || []) {
        const values = aggregated.map(a => {
          const m = a.metrics.find(m => m.name === metric.name);
          return m?.avg || 0;
        });

        const anomaly = this.midstreamer.processTimeSeries(
          `${this.config.cellId}:${metric.name}`,
          values,
          granularity
        );

        if (anomaly) {
          anomalies.push(anomaly);
          this.anomaliesDetected++;
        }
      }
    }

    // Correlation analysis
    const correlations = this.correlationAnalyzer.analyzeCorrelations(this.rawKpiBuffer);

    return { aggregated, trends, anomalies, correlations };
  }

  /**
   * Generate capacity forecasts
   */
  generateCapacityForecasts(): CapacityForecast[] {
    const weeklyData = this.aggregatedCache.get('1day') || [];

    if (weeklyData.length < 7) {
      return [];
    }

    const forecasts = this.capacityPlanner.projectCapacity(weeklyData, this.trendAnalyzer);
    this.capacityForecastsGenerated += forecasts.length;

    // Emit alerts for urgent forecasts
    for (const forecast of forecasts) {
      if (forecast.daysToCapacity < 7) {
        this.emit('capacity-alert', {
          level: 'critical',
          forecast
        });
      } else if (forecast.daysToCapacity < 30) {
        this.emit('capacity-alert', {
          level: 'warning',
          forecast
        });
      }
    }

    return forecasts;
  }

  /**
   * Get current cell health score
   */
  getCellHealth(): {
    score: number;
    factors: { name: string; score: number; weight: number }[];
    status: 'healthy' | 'degraded' | 'critical';
  } {
    const recentKpis = this.rawKpiBuffer.slice(-10);
    if (recentKpis.length === 0) {
      return { score: 100, factors: [], status: 'healthy' };
    }

    const factors = [
      {
        name: 'Accessibility',
        score: this.computeAccessibilityScore(recentKpis),
        weight: 0.25
      },
      {
        name: 'Retainability',
        score: this.computeRetainabilityScore(recentKpis),
        weight: 0.25
      },
      {
        name: 'Integrity',
        score: this.computeIntegrityScore(recentKpis),
        weight: 0.25
      },
      {
        name: 'Mobility',
        score: this.computeMobilityScore(recentKpis),
        weight: 0.25
      }
    ];

    const score = factors.reduce((sum, f) => sum + f.score * f.weight, 0);

    let status: 'healthy' | 'degraded' | 'critical' = 'healthy';
    if (score < 50) status = 'critical';
    else if (score < 80) status = 'degraded';

    return { score, factors, status };
  }

  /**
   * Run full multi-granularity analysis
   */
  runFullAnalysis(): {
    granularities: {
      [K in '15min' | '1hour' | '1day' | '1week']?: ReturnType<PerformanceManager['processGranularity']>;
    };
    health: ReturnType<PerformanceManager['getCellHealth']>;
    capacity: CapacityForecast[];
    multiGranularity: ReturnType<MidstreamerEngine['analyzeMultiGranularity']> | null;
  } {
    const granularities: any = {};

    for (const g of ['15min', '1hour', '1day', '1week'] as const) {
      granularities[g] = this.processGranularity(g);
    }

    // Multi-granularity temporal analysis
    const raw15Min = this.rawKpiBuffer.map(k => k.dlThroughput);
    const multiGranularity = raw15Min.length >= 96
      ? this.midstreamer.analyzeMultiGranularity(raw15Min, this.config.cellId)
      : null;

    return {
      granularities,
      health: this.getCellHealth(),
      capacity: this.generateCapacityForecasts(),
      multiGranularity
    };
  }

  private detectRealTimeAnomalies(kpi: CellKPIs): void {
    const t = this.config.anomalyThresholds;

    // Drop rate check
    if (kpi.callDropRate >= t.dropRateCritical) {
      this.emitAnomaly('THRESHOLD_BREACH', 'callDropRate', kpi, 'critical');
    } else if (kpi.callDropRate >= t.dropRateWarning) {
      this.emitAnomaly('THRESHOLD_BREACH', 'callDropRate', kpi, 'high');
    }

    // BLER check
    if (kpi.blerDl >= t.blerCritical || kpi.blerUl >= t.blerCritical) {
      this.emitAnomaly('THRESHOLD_BREACH', 'bler', kpi, 'critical');
    } else if (kpi.blerDl >= t.blerWarning || kpi.blerUl >= t.blerWarning) {
      this.emitAnomaly('THRESHOLD_BREACH', 'bler', kpi, 'high');
    }

    // Congestion check
    const congestion = Math.max(kpi.prbUtilizationDl, kpi.prbUtilizationUl) / 100;
    if (congestion >= t.congestionCritical) {
      this.emitAnomaly('THRESHOLD_BREACH', 'congestion', kpi, 'critical');
    } else if (congestion >= t.congestionWarning) {
      this.emitAnomaly('THRESHOLD_BREACH', 'congestion', kpi, 'medium');
    }
  }

  private emitAnomaly(
    type: AnomalyType,
    metric: string,
    kpi: CellKPIs,
    severity: 'low' | 'medium' | 'high' | 'critical'
  ): void {
    const anomaly: Anomaly = {
      id: `anomaly-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: kpi.timestamp,
      cgi: this.config.cellId,
      type,
      severity,
      anomalyVector: [(kpi as any)[metric] || 0],
      affectedMetrics: [metric],
      confidence: 0.9,
      context: {
        timeOfDay: kpi.timestamp.getHours(),
        dayOfWeek: kpi.timestamp.getDay(),
        isWeekend: kpi.timestamp.getDay() === 0 || kpi.timestamp.getDay() === 6
      }
    };

    this.anomaliesDetected++;
    this.emit('anomaly-detected', anomaly);
  }

  private computeAccessibilityScore(kpis: CellKPIs[]): number {
    const avgSetupSuccess = kpis.reduce((s, k) => s + k.rrcSetupSuccessRate, 0) / kpis.length;
    return avgSetupSuccess;
  }

  private computeRetainabilityScore(kpis: CellKPIs[]): number {
    const avgDropRate = kpis.reduce((s, k) => s + k.callDropRate, 0) / kpis.length;
    return Math.max(0, 100 - avgDropRate * 10);
  }

  private computeIntegrityScore(kpis: CellKPIs[]): number {
    const avgBler = kpis.reduce((s, k) => s + (k.blerDl + k.blerUl) / 2, 0) / kpis.length;
    return Math.max(0, 100 - avgBler * 200);
  }

  private computeMobilityScore(kpis: CellKPIs[]): number {
    const avgHoSuccess = kpis.reduce((s, k) => s + k.hoSuccessRate, 0) / kpis.length;
    return avgHoSuccess;
  }

  /**
   * Get performance manager statistics
   */
  getStats(): {
    cellId: string;
    bufferSize: number;
    anomaliesDetected: number;
    trendsAnalyzed: number;
    capacityForecastsGenerated: number;
    midstreamerStats: ReturnType<MidstreamerEngine['getStats']>;
  } {
    return {
      cellId: this.config.cellId,
      bufferSize: this.rawKpiBuffer.length,
      anomaliesDetected: this.anomaliesDetected,
      trendsAnalyzed: this.trendsAnalyzed,
      capacityForecastsGenerated: this.capacityForecastsGenerated,
      midstreamerStats: this.midstreamer.getStats()
    };
  }
}
