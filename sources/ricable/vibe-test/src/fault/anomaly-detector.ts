/**
 * Anomaly Detector - Fast-path signal anomaly detection
 * Implements aimds-detection patterns for real-time monitoring
 */

import {
  Anomaly,
  AnomalyType,
  AnomalyMetrics,
  CellMetrics,
  Alarm,
  AlarmSeverity,
} from '../core/types.js';
import { v4 as uuidv4 } from 'uuid';
import { exponentialMovingAverage, standardDeviation, percentile } from '../utils/math.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('AnomalyDetector');

/**
 * Anomaly detection configuration
 */
export interface AnomalyDetectorConfig {
  rssiDropThreshold: number;        // dB drop to trigger
  interferenceThreshold: number;     // Interference level threshold
  sleepingCellThreshold: number;     // PRB utilization threshold
  handoverStormThreshold: number;    // Handovers per minute
  windowSize: number;                // Sliding window size
  sensitivityLevel: 'low' | 'medium' | 'high';
  baselineWindow: number;            // Baseline calculation window
}

const DEFAULT_CONFIG: AnomalyDetectorConfig = {
  rssiDropThreshold: 6,              // 6dB drop
  interferenceThreshold: -85,        // dBm
  sleepingCellThreshold: 0.05,       // 5% PRB utilization
  handoverStormThreshold: 100,       // HOs per minute
  windowSize: 60,                    // 60 samples
  sensitivityLevel: 'medium',
  baselineWindow: 300,               // 5 minutes
};

/**
 * Statistical baseline for a metric
 */
interface MetricBaseline {
  mean: number;
  stdDev: number;
  p5: number;
  p95: number;
  lastUpdated: number;
}

/**
 * Fast-path Anomaly Detector
 * Designed for microsecond-level detection on the fast path
 */
export class AnomalyDetector {
  private config: AnomalyDetectorConfig;
  private baselines: Map<string, MetricBaseline>;
  private metricHistory: Map<string, number[]>;
  private activeAnomalies: Map<string, Anomaly>;
  private alertThresholds: Map<AnomalyType, number>;

  constructor(config: Partial<AnomalyDetectorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.baselines = new Map();
    this.metricHistory = new Map();
    this.activeAnomalies = new Map();

    // Set thresholds based on sensitivity
    this.alertThresholds = this.initializeThresholds();

    logger.info('Anomaly detector initialized', {
      sensitivity: this.config.sensitivityLevel,
    });
  }

  /**
   * Initialize detection thresholds based on sensitivity
   */
  private initializeThresholds(): Map<AnomalyType, number> {
    const multiplier = {
      low: 1.5,
      medium: 1.0,
      high: 0.7,
    }[this.config.sensitivityLevel];

    return new Map([
      ['rssi_drop', 3.0 * multiplier],
      ['jamming', 2.5 * multiplier],
      ['sleeping_cell', 3.5 * multiplier],
      ['handover_storm', 2.0 * multiplier],
      ['traffic_spike', 2.5 * multiplier],
      ['interference_spike', 2.5 * multiplier],
      ['vswr_high', 2.0 * multiplier],
      ['coverage_hole', 3.0 * multiplier],
    ]);
  }

  /**
   * Real-time anomaly detection on incoming metrics
   * Designed for fast-path processing (microseconds)
   */
  detect(cellId: string, metrics: CellMetrics): Anomaly[] {
    const detected: Anomaly[] = [];

    // Update baselines
    this.updateBaselines(cellId, metrics);

    // Run all detectors in parallel (conceptually)
    const rssiAnomaly = this.detectRSSIDrop(cellId, metrics);
    if (rssiAnomaly) detected.push(rssiAnomaly);

    const jammingAnomaly = this.detectJamming(cellId, metrics);
    if (jammingAnomaly) detected.push(jammingAnomaly);

    const sleepingCellAnomaly = this.detectSleepingCell(cellId, metrics);
    if (sleepingCellAnomaly) detected.push(sleepingCellAnomaly);

    const interferenceAnomaly = this.detectInterferenceSpike(cellId, metrics);
    if (interferenceAnomaly) detected.push(interferenceAnomaly);

    const trafficAnomaly = this.detectTrafficSpike(cellId, metrics);
    if (trafficAnomaly) detected.push(trafficAnomaly);

    // Log detected anomalies
    if (detected.length > 0) {
      logger.warn('Anomalies detected', {
        cellId,
        count: detected.length,
        types: detected.map((a) => a.type),
      });
    }

    return detected;
  }

  /**
   * Batch detection across multiple cells
   */
  detectBatch(cellsMetrics: Map<string, CellMetrics>): Map<string, Anomaly[]> {
    const results = new Map<string, Anomaly[]>();

    for (const [cellId, metrics] of cellsMetrics) {
      const anomalies = this.detect(cellId, metrics);
      if (anomalies.length > 0) {
        results.set(cellId, anomalies);
      }
    }

    return results;
  }

  /**
   * Detect RSSI drop anomaly (signal degradation)
   */
  private detectRSSIDrop(cellId: string, metrics: CellMetrics): Anomaly | null {
    const key = `${cellId}:rssi`;
    const history = this.getHistory(key);
    history.push(metrics.rssiUl);

    if (history.length < 10) return null;

    const baseline = this.baselines.get(key);
    if (!baseline) return null;

    // Detect sudden drop
    const recent = history.slice(-5);
    const recentMean = recent.reduce((a, b) => a + b, 0) / recent.length;
    const drop = baseline.mean - recentMean;

    if (drop > this.config.rssiDropThreshold) {
      const zScore = (recentMean - baseline.mean) / (baseline.stdDev || 1);

      return this.createAnomaly(
        'rssi_drop',
        cellId,
        Math.min(1, Math.abs(zScore) / 5),
        {
          baseline: baseline.mean,
          observed: recentMean,
          deviation: drop,
          trend: this.calculateTrend(recent),
        }
      );
    }

    return null;
  }

  /**
   * Detect jamming (elevated noise floor)
   */
  private detectJamming(cellId: string, metrics: CellMetrics): Anomaly | null {
    const key = `${cellId}:interference`;
    const history = this.getHistory(key);
    history.push(metrics.interferenceLevel);

    if (history.length < 5) return null;

    const baseline = this.baselines.get(key);
    if (!baseline) return null;

    // Detect elevated interference across all PRBs
    const recent = history.slice(-3);
    const recentMean = recent.reduce((a, b) => a + b, 0) / recent.length;

    // Jamming: sudden, sustained increase in interference
    if (recentMean > baseline.mean + 2 * baseline.stdDev &&
        recentMean > this.config.interferenceThreshold) {
      return this.createAnomaly(
        'jamming',
        cellId,
        0.9,
        {
          baseline: baseline.mean,
          observed: recentMean,
          deviation: recentMean - baseline.mean,
          trend: 'increasing',
        }
      );
    }

    return null;
  }

  /**
   * Detect sleeping cell (zero or minimal traffic despite coverage)
   */
  private detectSleepingCell(cellId: string, metrics: CellMetrics): Anomaly | null {
    const key = `${cellId}:prb`;
    const history = this.getHistory(key);
    history.push(metrics.prbUtilizationDl);

    if (history.length < 10) return null;

    // Check for sustained low utilization
    const recent = history.slice(-10);
    const allBelowThreshold = recent.every((v) => v < this.config.sleepingCellThreshold);

    // Also check that RSRP is reasonable (coverage exists)
    const hasCoverage = metrics.rsrp > -110;

    if (allBelowThreshold && hasCoverage && metrics.activeUesDl === 0) {
      return this.createAnomaly(
        'sleeping_cell',
        cellId,
        0.95,
        {
          baseline: 0.3, // Expected normal utilization
          observed: recent.reduce((a, b) => a + b, 0) / recent.length,
          deviation: 0.3,
          trend: 'stable',
        }
      );
    }

    return null;
  }

  /**
   * Detect interference spike
   */
  private detectInterferenceSpike(cellId: string, metrics: CellMetrics): Anomaly | null {
    const key = `${cellId}:interference`;
    const baseline = this.baselines.get(key);
    if (!baseline) return null;

    const threshold = this.alertThresholds.get('interference_spike') || 2.5;
    const zScore = (metrics.interferenceLevel - baseline.mean) / (baseline.stdDev || 1);

    if (Math.abs(zScore) > threshold) {
      return this.createAnomaly(
        'interference_spike',
        cellId,
        Math.min(1, Math.abs(zScore) / 5),
        {
          baseline: baseline.mean,
          observed: metrics.interferenceLevel,
          deviation: metrics.interferenceLevel - baseline.mean,
          trend: zScore > 0 ? 'increasing' : 'decreasing',
        }
      );
    }

    return null;
  }

  /**
   * Detect traffic spike
   */
  private detectTrafficSpike(cellId: string, metrics: CellMetrics): Anomaly | null {
    const key = `${cellId}:throughput`;
    const history = this.getHistory(key);
    history.push(metrics.throughputDl);

    if (history.length < 10) return null;

    const baseline = this.baselines.get(key);
    if (!baseline) return null;

    const threshold = this.alertThresholds.get('traffic_spike') || 2.5;
    const zScore = (metrics.throughputDl - baseline.mean) / (baseline.stdDev || 1);

    if (Math.abs(zScore) > threshold) {
      return this.createAnomaly(
        'traffic_spike',
        cellId,
        Math.min(1, Math.abs(zScore) / 5),
        {
          baseline: baseline.mean,
          observed: metrics.throughputDl,
          deviation: metrics.throughputDl - baseline.mean,
          trend: zScore > 0 ? 'increasing' : 'decreasing',
        }
      );
    }

    return null;
  }

  /**
   * Process alarm from Ericsson network element
   */
  processAlarm(alarm: Alarm): Anomaly | null {
    // Map alarm codes to anomaly types
    const alarmMapping: Record<string, AnomalyType> = {
      'VSWR_HIGH': 'vswr_high',
      'LOSS_OF_SIGNAL': 'rssi_drop',
      'HIGH_INTERFERENCE': 'interference_spike',
      'TX_DIVERSITY_FAULT': 'coverage_hole',
    };

    const anomalyType = alarmMapping[alarm.alarmCode];
    if (!anomalyType) return null;

    const severity = {
      critical: 1.0,
      major: 0.8,
      minor: 0.5,
      warning: 0.3,
      cleared: 0,
    }[alarm.severity];

    if (severity === 0) {
      // Clear existing anomaly
      this.activeAnomalies.delete(`${alarm.cellId}:${anomalyType}`);
      return null;
    }

    const anomaly = this.createAnomaly(
      anomalyType,
      alarm.cellId,
      severity,
      {
        baseline: 0,
        observed: 0,
        deviation: 0,
        trend: 'stable',
      }
    );

    this.activeAnomalies.set(`${alarm.cellId}:${anomalyType}`, anomaly);

    return anomaly;
  }

  /**
   * Update baselines for a cell
   */
  private updateBaselines(cellId: string, metrics: CellMetrics): void {
    const metricsToTrack = [
      { key: `${cellId}:rssi`, value: metrics.rssiUl },
      { key: `${cellId}:interference`, value: metrics.interferenceLevel },
      { key: `${cellId}:prb`, value: metrics.prbUtilizationDl },
      { key: `${cellId}:throughput`, value: metrics.throughputDl },
      { key: `${cellId}:sinr`, value: metrics.sinr },
    ];

    for (const { key, value } of metricsToTrack) {
      const history = this.getHistory(key);
      history.push(value);

      // Maintain window size
      while (history.length > this.config.baselineWindow) {
        history.shift();
      }

      // Update baseline if enough data
      if (history.length >= 30) {
        const baseline: MetricBaseline = {
          mean: history.reduce((a, b) => a + b, 0) / history.length,
          stdDev: standardDeviation(history),
          p5: percentile(history, 5),
          p95: percentile(history, 95),
          lastUpdated: Date.now(),
        };
        this.baselines.set(key, baseline);
      }
    }
  }

  /**
   * Get or create history array for a metric
   */
  private getHistory(key: string): number[] {
    let history = this.metricHistory.get(key);
    if (!history) {
      history = [];
      this.metricHistory.set(key, history);
    }
    return history;
  }

  /**
   * Calculate trend from recent values
   */
  private calculateTrend(values: number[]): 'increasing' | 'decreasing' | 'stable' {
    if (values.length < 2) return 'stable';

    const firstHalf = values.slice(0, Math.floor(values.length / 2));
    const secondHalf = values.slice(Math.floor(values.length / 2));

    const firstMean = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondMean = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;

    const change = (secondMean - firstMean) / (firstMean || 1);

    if (change > 0.1) return 'increasing';
    if (change < -0.1) return 'decreasing';
    return 'stable';
  }

  /**
   * Create an anomaly object
   */
  private createAnomaly(
    type: AnomalyType,
    cellId: string,
    severity: number,
    metrics: AnomalyMetrics
  ): Anomaly {
    return {
      id: uuidv4(),
      type,
      cellId,
      detectedAt: Date.now(),
      severity,
      metrics,
      resolved: false,
    };
  }

  /**
   * Get active anomalies for a cell
   */
  getActiveAnomalies(cellId?: string): Anomaly[] {
    const anomalies = Array.from(this.activeAnomalies.values());
    if (cellId) {
      return anomalies.filter((a) => a.cellId === cellId);
    }
    return anomalies;
  }

  /**
   * Mark an anomaly as resolved
   */
  resolveAnomaly(anomalyId: string): boolean {
    for (const [key, anomaly] of this.activeAnomalies) {
      if (anomaly.id === anomalyId) {
        anomaly.resolved = true;
        this.activeAnomalies.delete(key);
        return true;
      }
    }
    return false;
  }

  /**
   * Get baseline for a metric
   */
  getBaseline(cellId: string, metric: string): MetricBaseline | undefined {
    return this.baselines.get(`${cellId}:${metric}`);
  }

  /**
   * Reset detector state
   */
  reset(): void {
    this.baselines.clear();
    this.metricHistory.clear();
    this.activeAnomalies.clear();
  }
}

/**
 * Create a configured anomaly detector
 */
export function createAnomalyDetector(
  config?: Partial<AnomalyDetectorConfig>
): AnomalyDetector {
  return new AnomalyDetector(config);
}
