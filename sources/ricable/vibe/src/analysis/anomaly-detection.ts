/**
 * Anomaly Detection System for RAN KPIs
 * Multiple detection algorithms: Statistical, ML-based, and domain-specific
 */

import { v4 as uuidv4 } from 'uuid';
import * as ss from 'simple-statistics';
import type {
  DetectedAnomaly,
  AnomalyType,
  AnomalySeverity,
  KPITimeSeries,
} from '../models/ran-kpi.js';
import {
  computeTimeSeriesStats,
  exponentialMovingAverage,
  analyzeTrend,
  decomposeTimeSeries,
  analyzeSeasonality,
} from './time-series.js';

// ============================================================================
// ANOMALY DETECTION CONFIG
// ============================================================================

export interface AnomalyDetectionConfig {
  // Z-score threshold for statistical detection
  zScoreThreshold: number;
  // IQR multiplier for outlier detection
  iqrMultiplier: number;
  // Window size for rolling statistics
  rollingWindow: number;
  // Minimum number of points for collective anomaly
  minCollectiveSize: number;
  // Sensitivity (0-1, higher = more sensitive)
  sensitivity: number;
  // Domain-specific thresholds
  domainThresholds: DomainThresholds;
}

export interface DomainThresholds {
  accessibility: {
    rrcSetupSuccessRateMin: number;
    erabSetupSuccessRateMin: number;
  };
  retainability: {
    dropRateMax: number;
    erabDropRateMax: number;
  };
  radioQuality: {
    dlCqiMin: number;
    ulSinrMin: number;
    rsrpMin: number;
  };
  mobility: {
    hoSuccessRateMin: number;
    pingPongRateMax: number;
  };
  uplinkInterference: {
    iotMax: number;
    highInterferencePrbRatioMax: number;
  };
  uplinkPowerControl: {
    powerLimitedUeRatioMax: number;
    negativePhrRatioMax: number;
  };
}

export const DEFAULT_CONFIG: AnomalyDetectionConfig = {
  zScoreThreshold: 3,
  iqrMultiplier: 1.5,
  rollingWindow: 12,
  minCollectiveSize: 3,
  sensitivity: 0.7,
  domainThresholds: {
    accessibility: {
      rrcSetupSuccessRateMin: 98,
      erabSetupSuccessRateMin: 98,
    },
    retainability: {
      dropRateMax: 1,
      erabDropRateMax: 0.5,
    },
    radioQuality: {
      dlCqiMin: 7,
      ulSinrMin: 5,
      rsrpMin: -110,
    },
    mobility: {
      hoSuccessRateMin: 95,
      pingPongRateMax: 5,
    },
    uplinkInterference: {
      iotMax: 10,
      highInterferencePrbRatioMax: 20,
    },
    uplinkPowerControl: {
      powerLimitedUeRatioMax: 10,
      negativePhrRatioMax: 15,
    },
  },
};

// ============================================================================
// STATISTICAL ANOMALY DETECTION
// ============================================================================

export class StatisticalAnomalyDetector {
  private config: AnomalyDetectionConfig;

  constructor(config: Partial<AnomalyDetectionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Z-Score based anomaly detection
   */
  detectZScoreAnomalies(
    values: number[],
    timestamps: Date[],
    cellId: string,
    kpiName: string,
    domain: string
  ): DetectedAnomaly[] {
    const anomalies: DetectedAnomaly[] = [];
    const stats = computeTimeSeriesStats(values);

    for (let i = 0; i < values.length; i++) {
      const zScore = (values[i] - stats.mean) / stats.stdDev;

      if (Math.abs(zScore) > this.config.zScoreThreshold) {
        const anomalyType: AnomalyType = zScore > 0 ? 'spike' : 'dip';
        const severity = this.calculateSeverity(Math.abs(zScore));

        anomalies.push({
          id: uuidv4(),
          cellId,
          kpiName,
          domain,
          timestamp: timestamps[i],
          anomalyType,
          severity,
          observedValue: values[i],
          expectedValue: stats.mean,
          deviation: zScore,
          confidence: this.zScoreToConfidence(Math.abs(zScore)),
        });
      }
    }

    return anomalies;
  }

  /**
   * IQR-based outlier detection (Tukey's method)
   */
  detectIQRAnomalies(
    values: number[],
    timestamps: Date[],
    cellId: string,
    kpiName: string,
    domain: string
  ): DetectedAnomaly[] {
    const anomalies: DetectedAnomaly[] = [];
    const sorted = [...values].sort((a, b) => a - b);
    const q1 = ss.quantile(sorted, 0.25);
    const q3 = ss.quantile(sorted, 0.75);
    const iqr = q3 - q1;

    const lowerBound = q1 - this.config.iqrMultiplier * iqr;
    const upperBound = q3 + this.config.iqrMultiplier * iqr;

    for (let i = 0; i < values.length; i++) {
      if (values[i] < lowerBound || values[i] > upperBound) {
        const median = ss.median(values);
        const deviation = (values[i] - median) / iqr;

        anomalies.push({
          id: uuidv4(),
          cellId,
          kpiName,
          domain,
          timestamp: timestamps[i],
          anomalyType: 'outlier',
          severity: this.calculateSeverity(Math.abs(deviation)),
          observedValue: values[i],
          expectedValue: median,
          deviation: deviation,
          confidence: Math.min(0.95, 0.7 + Math.abs(deviation) * 0.05),
        });
      }
    }

    return anomalies;
  }

  /**
   * Rolling window anomaly detection
   */
  detectRollingAnomalies(
    values: number[],
    timestamps: Date[],
    cellId: string,
    kpiName: string,
    domain: string
  ): DetectedAnomaly[] {
    const anomalies: DetectedAnomaly[] = [];
    const window = this.config.rollingWindow;

    for (let i = window; i < values.length; i++) {
      const windowValues = values.slice(i - window, i);
      const windowMean = ss.mean(windowValues);
      const windowStd = ss.standardDeviation(windowValues);

      const zScore = windowStd > 0 ? (values[i] - windowMean) / windowStd : 0;

      if (Math.abs(zScore) > this.config.zScoreThreshold * (1 - this.config.sensitivity * 0.3)) {
        anomalies.push({
          id: uuidv4(),
          cellId,
          kpiName,
          domain,
          timestamp: timestamps[i],
          anomalyType: zScore > 0 ? 'spike' : 'dip',
          severity: this.calculateSeverity(Math.abs(zScore)),
          observedValue: values[i],
          expectedValue: windowMean,
          deviation: zScore,
          confidence: this.zScoreToConfidence(Math.abs(zScore)),
        });
      }
    }

    return anomalies;
  }

  private calculateSeverity(deviation: number): AnomalySeverity {
    if (deviation > 5) return 'critical';
    if (deviation > 4) return 'high';
    if (deviation > 3) return 'medium';
    return 'low';
  }

  private zScoreToConfidence(zScore: number): number {
    // Convert z-score to confidence using normal CDF approximation
    return Math.min(0.99, 1 - 2 * (1 - this.normalCDF(zScore)));
  }

  private normalCDF(x: number): number {
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x) / Math.sqrt(2);

    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return 0.5 * (1.0 + sign * y);
  }
}

// ============================================================================
// TREND-BASED ANOMALY DETECTION
// ============================================================================

export class TrendAnomalyDetector {
  private config: AnomalyDetectionConfig;

  constructor(config: Partial<AnomalyDetectionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Detect level shifts (sudden baseline changes)
   */
  detectLevelShifts(
    values: number[],
    timestamps: Date[],
    cellId: string,
    kpiName: string,
    domain: string
  ): DetectedAnomaly[] {
    const anomalies: DetectedAnomaly[] = [];
    const window = this.config.rollingWindow;

    for (let i = window * 2; i < values.length; i++) {
      const prevWindow = values.slice(i - window * 2, i - window);
      const currWindow = values.slice(i - window, i);

      const prevMean = ss.mean(prevWindow);
      const currMean = ss.mean(currWindow);
      const pooledStd = Math.sqrt(
        (ss.variance(prevWindow) + ss.variance(currWindow)) / 2
      );

      if (pooledStd > 0) {
        const tScore = Math.abs(currMean - prevMean) / (pooledStd * Math.sqrt(2 / window));

        if (tScore > 2.5) {
          anomalies.push({
            id: uuidv4(),
            cellId,
            kpiName,
            domain,
            timestamp: timestamps[i - window],
            anomalyType: 'level_shift',
            severity: this.tScoreToSeverity(tScore),
            observedValue: currMean,
            expectedValue: prevMean,
            deviation: tScore,
            confidence: Math.min(0.95, 0.7 + tScore * 0.05),
            duration: window,
          });
        }
      }
    }

    return anomalies;
  }

  /**
   * Detect trend shifts (changes in slope)
   */
  detectTrendShifts(
    values: number[],
    timestamps: Date[],
    cellId: string,
    kpiName: string,
    domain: string
  ): DetectedAnomaly[] {
    const anomalies: DetectedAnomaly[] = [];
    const trendAnalysis = analyzeTrend(values);

    for (const changePoint of trendAnalysis.changePoints) {
      if (changePoint > 0 && changePoint < values.length - 1) {
        const beforeTrend = analyzeTrend(values.slice(0, changePoint));
        const afterTrend = analyzeTrend(values.slice(changePoint));

        const slopeChange = Math.abs(afterTrend.slope - beforeTrend.slope);

        if (slopeChange > 0.1) {
          anomalies.push({
            id: uuidv4(),
            cellId,
            kpiName,
            domain,
            timestamp: timestamps[changePoint],
            anomalyType: 'trend_shift',
            severity: this.slopeChangeToSeverity(slopeChange),
            observedValue: afterTrend.slope,
            expectedValue: beforeTrend.slope,
            deviation: slopeChange,
            confidence: Math.min(0.9, trendAnalysis.rSquared),
          });
        }
      }
    }

    return anomalies;
  }

  /**
   * Detect variance changes
   */
  detectVarianceChanges(
    values: number[],
    timestamps: Date[],
    cellId: string,
    kpiName: string,
    domain: string
  ): DetectedAnomaly[] {
    const anomalies: DetectedAnomaly[] = [];
    const window = this.config.rollingWindow;

    for (let i = window * 2; i < values.length; i++) {
      const prevWindow = values.slice(i - window * 2, i - window);
      const currWindow = values.slice(i - window, i);

      const prevVar = ss.variance(prevWindow);
      const currVar = ss.variance(currWindow);

      // F-test for variance change
      const fRatio = currVar > prevVar ? currVar / prevVar : prevVar / currVar;

      if (fRatio > 3) {
        anomalies.push({
          id: uuidv4(),
          cellId,
          kpiName,
          domain,
          timestamp: timestamps[i - window],
          anomalyType: 'variance_change',
          severity: fRatio > 5 ? 'high' : fRatio > 4 ? 'medium' : 'low',
          observedValue: currVar,
          expectedValue: prevVar,
          deviation: fRatio,
          confidence: Math.min(0.9, 0.5 + fRatio * 0.05),
          duration: window,
        });
      }
    }

    return anomalies;
  }

  private tScoreToSeverity(tScore: number): AnomalySeverity {
    if (tScore > 5) return 'critical';
    if (tScore > 4) return 'high';
    if (tScore > 3) return 'medium';
    return 'low';
  }

  private slopeChangeToSeverity(change: number): AnomalySeverity {
    if (change > 1) return 'critical';
    if (change > 0.5) return 'high';
    if (change > 0.2) return 'medium';
    return 'low';
  }
}

// ============================================================================
// SEASONAL ANOMALY DETECTION
// ============================================================================

export class SeasonalAnomalyDetector {
  private config: AnomalyDetectionConfig;

  constructor(config: Partial<AnomalyDetectionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Detect deviations from seasonal patterns
   */
  detectSeasonalAnomalies(
    values: number[],
    timestamps: Date[],
    cellId: string,
    kpiName: string,
    domain: string,
    expectedPeriod: number = 96 // Default 15-min granularity, 96 per day
  ): DetectedAnomaly[] {
    const anomalies: DetectedAnomaly[] = [];

    // Decompose time series
    const seasonality = analyzeSeasonality(values, expectedPeriod);

    if (!seasonality.seasonalPattern) {
      return anomalies;
    }

    const decomposed = decomposeTimeSeries(values, expectedPeriod);

    // Detect anomalies in residuals
    const residualStats = computeTimeSeriesStats(decomposed.residual);

    for (let i = 0; i < decomposed.residual.length; i++) {
      const zScore = Math.abs(decomposed.residual[i]) / residualStats.stdDev;

      if (zScore > this.config.zScoreThreshold) {
        const expectedValue = decomposed.trend[i] + decomposed.seasonal[i];

        anomalies.push({
          id: uuidv4(),
          cellId,
          kpiName,
          domain,
          timestamp: timestamps[i],
          anomalyType: 'seasonality_break',
          severity: this.calculateSeverity(zScore),
          observedValue: values[i],
          expectedValue,
          deviation: zScore,
          confidence: Math.min(0.95, seasonality.seasonalStrength * (0.7 + zScore * 0.05)),
        });
      }
    }

    return anomalies;
  }

  private calculateSeverity(deviation: number): AnomalySeverity {
    if (deviation > 5) return 'critical';
    if (deviation > 4) return 'high';
    if (deviation > 3) return 'medium';
    return 'low';
  }
}

// ============================================================================
// COLLECTIVE ANOMALY DETECTION
// ============================================================================

export class CollectiveAnomalyDetector {
  private config: AnomalyDetectionConfig;

  constructor(config: Partial<AnomalyDetectionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Detect collective anomalies (groups of points that together are anomalous)
   */
  detectCollectiveAnomalies(
    values: number[],
    timestamps: Date[],
    cellId: string,
    kpiName: string,
    domain: string
  ): DetectedAnomaly[] {
    const anomalies: DetectedAnomaly[] = [];
    const stats = computeTimeSeriesStats(values);
    const threshold = stats.mean + stats.stdDev * (this.config.zScoreThreshold - 1);

    // Find runs of elevated/depressed values
    let runStart = -1;
    let runType: 'high' | 'low' | null = null;

    for (let i = 0; i < values.length; i++) {
      const isHigh = values[i] > threshold;
      const isLow = values[i] < stats.mean - stats.stdDev * (this.config.zScoreThreshold - 1);

      if ((isHigh || isLow) && runStart === -1) {
        runStart = i;
        runType = isHigh ? 'high' : 'low';
      } else if ((!isHigh && !isLow) || (isHigh && runType === 'low') || (isLow && runType === 'high')) {
        if (runStart !== -1 && i - runStart >= this.config.minCollectiveSize) {
          const runValues = values.slice(runStart, i);
          const runMean = ss.mean(runValues);

          anomalies.push({
            id: uuidv4(),
            cellId,
            kpiName,
            domain,
            timestamp: timestamps[runStart],
            anomalyType: 'collective',
            severity: this.calculateCollectiveSeverity(i - runStart, Math.abs(runMean - stats.mean) / stats.stdDev),
            observedValue: runMean,
            expectedValue: stats.mean,
            deviation: (runMean - stats.mean) / stats.stdDev,
            confidence: Math.min(0.95, 0.6 + (i - runStart) * 0.02),
            duration: i - runStart,
          });
        }
        runStart = (isHigh || isLow) ? i : -1;
        runType = isHigh ? 'high' : isLow ? 'low' : null;
      }
    }

    // Check for run at the end
    if (runStart !== -1 && values.length - runStart >= this.config.minCollectiveSize) {
      const runValues = values.slice(runStart);
      const runMean = ss.mean(runValues);

      anomalies.push({
        id: uuidv4(),
        cellId,
        kpiName,
        domain,
        timestamp: timestamps[runStart],
        anomalyType: 'collective',
        severity: this.calculateCollectiveSeverity(values.length - runStart, Math.abs(runMean - stats.mean) / stats.stdDev),
        observedValue: runMean,
        expectedValue: stats.mean,
        deviation: (runMean - stats.mean) / stats.stdDev,
        confidence: Math.min(0.95, 0.6 + (values.length - runStart) * 0.02),
        duration: values.length - runStart,
      });
    }

    return anomalies;
  }

  private calculateCollectiveSeverity(duration: number, deviation: number): AnomalySeverity {
    const score = duration * 0.3 + deviation;
    if (score > 10) return 'critical';
    if (score > 6) return 'high';
    if (score > 3) return 'medium';
    return 'low';
  }
}

// ============================================================================
// DOMAIN-SPECIFIC THRESHOLD DETECTION
// ============================================================================

export class DomainAnomalyDetector {
  private config: AnomalyDetectionConfig;

  constructor(config: Partial<AnomalyDetectionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Detect violations of domain-specific thresholds
   */
  detectDomainViolations(
    value: number,
    timestamp: Date,
    cellId: string,
    kpiName: string,
    domain: string
  ): DetectedAnomaly | null {
    const thresholds = this.config.domainThresholds;
    let violated = false;
    let expectedValue = 0;
    let severity: AnomalySeverity = 'low';

    switch (domain) {
      case 'accessibility':
        if (kpiName.includes('rrcSetupSuccessRate') && value < thresholds.accessibility.rrcSetupSuccessRateMin) {
          violated = true;
          expectedValue = thresholds.accessibility.rrcSetupSuccessRateMin;
          severity = value < 95 ? 'critical' : value < 97 ? 'high' : 'medium';
        }
        if (kpiName.includes('erabSetupSuccessRate') && value < thresholds.accessibility.erabSetupSuccessRateMin) {
          violated = true;
          expectedValue = thresholds.accessibility.erabSetupSuccessRateMin;
          severity = value < 95 ? 'critical' : value < 97 ? 'high' : 'medium';
        }
        break;

      case 'retainability':
        if (kpiName.includes('dropRate') && value > thresholds.retainability.dropRateMax) {
          violated = true;
          expectedValue = thresholds.retainability.dropRateMax;
          severity = value > 3 ? 'critical' : value > 2 ? 'high' : 'medium';
        }
        break;

      case 'radioQuality':
        if (kpiName.includes('dlAvgCqi') && value < thresholds.radioQuality.dlCqiMin) {
          violated = true;
          expectedValue = thresholds.radioQuality.dlCqiMin;
          severity = value < 4 ? 'critical' : value < 6 ? 'high' : 'medium';
        }
        if (kpiName.includes('ulSinrAvg') && value < thresholds.radioQuality.ulSinrMin) {
          violated = true;
          expectedValue = thresholds.radioQuality.ulSinrMin;
          severity = value < 0 ? 'critical' : value < 3 ? 'high' : 'medium';
        }
        break;

      case 'mobility':
        if (kpiName.includes('HoSuccessRate') && value < thresholds.mobility.hoSuccessRateMin) {
          violated = true;
          expectedValue = thresholds.mobility.hoSuccessRateMin;
          severity = value < 90 ? 'critical' : value < 93 ? 'high' : 'medium';
        }
        break;

      case 'uplinkInterference':
        if (kpiName.includes('iotAvg') && value > thresholds.uplinkInterference.iotMax) {
          violated = true;
          expectedValue = thresholds.uplinkInterference.iotMax;
          severity = value > 15 ? 'critical' : value > 12 ? 'high' : 'medium';
        }
        break;

      case 'uplinkPowerControl':
        if (kpiName.includes('powerLimitedUeRatio') && value > thresholds.uplinkPowerControl.powerLimitedUeRatioMax) {
          violated = true;
          expectedValue = thresholds.uplinkPowerControl.powerLimitedUeRatioMax;
          severity = value > 25 ? 'critical' : value > 15 ? 'high' : 'medium';
        }
        break;
    }

    if (violated) {
      return {
        id: uuidv4(),
        cellId,
        kpiName,
        domain,
        timestamp,
        anomalyType: 'outlier',
        severity,
        observedValue: value,
        expectedValue,
        deviation: Math.abs(value - expectedValue),
        confidence: 0.95,
        possibleCauses: this.getPossibleCauses(domain, kpiName),
      };
    }

    return null;
  }

  private getPossibleCauses(domain: string, kpiName: string): string[] {
    const causes: Record<string, Record<string, string[]>> = {
      accessibility: {
        default: ['High load', 'Hardware issue', 'Parameter misconfiguration', 'Interference'],
        rrc: ['RACH congestion', 'Preamble collision', 'Coverage issue'],
        erab: ['S1 congestion', 'MME overload', 'Resource exhaustion'],
      },
      retainability: {
        default: ['UL interference', 'Poor coverage', 'Mobility issues', 'Backhaul congestion'],
      },
      radioQuality: {
        default: ['Interference', 'Coverage hole', 'Antenna issue', 'Feeder loss'],
        cqi: ['DL interference', 'Power control issue', 'MCS adaptation'],
        sinr: ['UL interference', 'Power control misconfiguration', 'External interference'],
      },
      mobility: {
        default: ['Wrong neighbor relations', 'Parameter misconfiguration', 'Coverage overlap issue'],
        ho: ['A3 offset too high/low', 'Hysteresis issue', 'Missing neighbors'],
      },
      uplinkInterference: {
        default: ['External interference', 'PIM', 'Cross-border interference', 'Passive intermodulation'],
      },
      uplinkPowerControl: {
        default: ['P0 too low', 'Alpha misconfiguration', 'Coverage limitation', 'High path loss'],
      },
    };

    const domainCauses = causes[domain] || { default: ['Unknown'] };
    const specificKey = Object.keys(domainCauses).find(k => k !== 'default' && kpiName.toLowerCase().includes(k));

    return specificKey ? domainCauses[specificKey] : domainCauses.default;
  }
}

// ============================================================================
// UNIFIED ANOMALY DETECTOR
// ============================================================================

export class UnifiedAnomalyDetector {
  private statisticalDetector: StatisticalAnomalyDetector;
  private trendDetector: TrendAnomalyDetector;
  private seasonalDetector: SeasonalAnomalyDetector;
  private collectiveDetector: CollectiveAnomalyDetector;
  private domainDetector: DomainAnomalyDetector;
  private config: AnomalyDetectionConfig;

  constructor(config: Partial<AnomalyDetectionConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.statisticalDetector = new StatisticalAnomalyDetector(this.config);
    this.trendDetector = new TrendAnomalyDetector(this.config);
    this.seasonalDetector = new SeasonalAnomalyDetector(this.config);
    this.collectiveDetector = new CollectiveAnomalyDetector(this.config);
    this.domainDetector = new DomainAnomalyDetector(this.config);
  }

  /**
   * Run all anomaly detection methods on a time series
   */
  detectAllAnomalies(timeSeries: KPITimeSeries): DetectedAnomaly[] {
    const values = timeSeries.dataPoints.map(p => p.value);
    const timestamps = timeSeries.dataPoints.map(p => p.timestamp);
    const { cellId, kpiName, domain } = timeSeries;

    const allAnomalies: DetectedAnomaly[] = [];

    // Statistical detection
    allAnomalies.push(
      ...this.statisticalDetector.detectZScoreAnomalies(values, timestamps, cellId, kpiName, domain),
      ...this.statisticalDetector.detectIQRAnomalies(values, timestamps, cellId, kpiName, domain),
      ...this.statisticalDetector.detectRollingAnomalies(values, timestamps, cellId, kpiName, domain)
    );

    // Trend detection
    allAnomalies.push(
      ...this.trendDetector.detectLevelShifts(values, timestamps, cellId, kpiName, domain),
      ...this.trendDetector.detectTrendShifts(values, timestamps, cellId, kpiName, domain),
      ...this.trendDetector.detectVarianceChanges(values, timestamps, cellId, kpiName, domain)
    );

    // Seasonal detection
    allAnomalies.push(
      ...this.seasonalDetector.detectSeasonalAnomalies(values, timestamps, cellId, kpiName, domain)
    );

    // Collective detection
    allAnomalies.push(
      ...this.collectiveDetector.detectCollectiveAnomalies(values, timestamps, cellId, kpiName, domain)
    );

    // Domain-specific threshold violations
    for (let i = 0; i < values.length; i++) {
      const violation = this.domainDetector.detectDomainViolations(
        values[i],
        timestamps[i],
        cellId,
        kpiName,
        domain
      );
      if (violation) {
        allAnomalies.push(violation);
      }
    }

    // Deduplicate and merge overlapping anomalies
    return this.deduplicateAnomalies(allAnomalies);
  }

  /**
   * Deduplicate anomalies that overlap in time
   */
  private deduplicateAnomalies(anomalies: DetectedAnomaly[]): DetectedAnomaly[] {
    if (anomalies.length === 0) return anomalies;

    // Sort by timestamp
    const sorted = [...anomalies].sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime());

    const deduplicated: DetectedAnomaly[] = [];
    const timeWindow = 5 * 60 * 1000; // 5 minutes

    for (const anomaly of sorted) {
      const existing = deduplicated.find(
        a =>
          a.cellId === anomaly.cellId &&
          a.kpiName === anomaly.kpiName &&
          Math.abs(a.timestamp.getTime() - anomaly.timestamp.getTime()) < timeWindow
      );

      if (existing) {
        // Keep the one with higher severity/confidence
        if (
          this.severityToNumber(anomaly.severity) > this.severityToNumber(existing.severity) ||
          anomaly.confidence > existing.confidence
        ) {
          const idx = deduplicated.indexOf(existing);
          deduplicated[idx] = anomaly;
        }
      } else {
        deduplicated.push(anomaly);
      }
    }

    return deduplicated;
  }

  private severityToNumber(severity: AnomalySeverity): number {
    const map: Record<AnomalySeverity, number> = {
      low: 1,
      medium: 2,
      high: 3,
      critical: 4,
    };
    return map[severity];
  }
}

export default {
  StatisticalAnomalyDetector,
  TrendAnomalyDetector,
  SeasonalAnomalyDetector,
  CollectiveAnomalyDetector,
  DomainAnomalyDetector,
  UnifiedAnomalyDetector,
  DEFAULT_CONFIG,
};
