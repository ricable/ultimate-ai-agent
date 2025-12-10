/**
 * Time Series Analysis for RAN KPIs
 * Implements statistical analysis, decomposition, and forecasting
 */

import * as ss from 'simple-statistics';
import type { KPITimeSeries, TimeSeriesPoint } from '../models/ran-kpi.js';

// ============================================================================
// TIME SERIES STATISTICS
// ============================================================================

export interface TimeSeriesStats {
  mean: number;
  median: number;
  stdDev: number;
  variance: number;
  min: number;
  max: number;
  range: number;
  skewness: number;
  kurtosis: number;
  q1: number;
  q3: number;
  iqr: number;
  count: number;
  // Trend indicators
  trendSlope: number;
  trendIntercept: number;
  autocorrelation: number[];
}

export function computeTimeSeriesStats(values: number[]): TimeSeriesStats {
  if (values.length === 0) {
    throw new Error('Cannot compute stats on empty array');
  }

  const sorted = [...values].sort((a, b) => a - b);
  const mean = ss.mean(values);
  const stdDev = ss.standardDeviation(values);
  const variance = ss.variance(values);

  // Compute autocorrelation for first 5 lags
  const autocorrelation = computeAutocorrelation(values, 5);

  // Linear trend
  const indices = values.map((_, i) => i);
  const regression = ss.linearRegression(indices.map((x, i) => [x, values[i]]));

  return {
    mean,
    median: ss.median(values),
    stdDev,
    variance,
    min: ss.min(values),
    max: ss.max(values),
    range: ss.max(values) - ss.min(values),
    skewness: computeSkewness(values, mean, stdDev),
    kurtosis: computeKurtosis(values, mean, stdDev),
    q1: ss.quantile(sorted, 0.25),
    q3: ss.quantile(sorted, 0.75),
    iqr: ss.interquartileRange(values),
    count: values.length,
    trendSlope: regression.m,
    trendIntercept: regression.b,
    autocorrelation,
  };
}

function computeSkewness(values: number[], mean: number, stdDev: number): number {
  if (stdDev === 0) return 0;
  const n = values.length;
  const sum = values.reduce((acc, v) => acc + Math.pow((v - mean) / stdDev, 3), 0);
  return (n / ((n - 1) * (n - 2))) * sum;
}

function computeKurtosis(values: number[], mean: number, stdDev: number): number {
  if (stdDev === 0) return 0;
  const n = values.length;
  const sum = values.reduce((acc, v) => acc + Math.pow((v - mean) / stdDev, 4), 0);
  return ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum - (3 * (n - 1) * (n - 1)) / ((n - 2) * (n - 3));
}

function computeAutocorrelation(values: number[], maxLag: number): number[] {
  const mean = ss.mean(values);
  const variance = ss.variance(values);
  const n = values.length;
  const result: number[] = [];

  for (let lag = 1; lag <= Math.min(maxLag, n - 1); lag++) {
    let sum = 0;
    for (let i = 0; i < n - lag; i++) {
      sum += (values[i] - mean) * (values[i + lag] - mean);
    }
    result.push(sum / ((n - lag) * variance));
  }

  return result;
}

// ============================================================================
// MOVING AVERAGES
// ============================================================================

export function simpleMovingAverage(values: number[], window: number): number[] {
  if (window > values.length) {
    throw new Error('Window size cannot exceed array length');
  }

  const result: number[] = [];
  for (let i = window - 1; i < values.length; i++) {
    const windowSlice = values.slice(i - window + 1, i + 1);
    result.push(ss.mean(windowSlice));
  }
  return result;
}

export function exponentialMovingAverage(values: number[], alpha: number): number[] {
  if (alpha < 0 || alpha > 1) {
    throw new Error('Alpha must be between 0 and 1');
  }

  const result: number[] = [values[0]];
  for (let i = 1; i < values.length; i++) {
    result.push(alpha * values[i] + (1 - alpha) * result[i - 1]);
  }
  return result;
}

export function weightedMovingAverage(values: number[], weights: number[]): number[] {
  const window = weights.length;
  if (window > values.length) {
    throw new Error('Window size cannot exceed array length');
  }

  const weightSum = weights.reduce((a, b) => a + b, 0);
  const normalizedWeights = weights.map(w => w / weightSum);

  const result: number[] = [];
  for (let i = window - 1; i < values.length; i++) {
    let sum = 0;
    for (let j = 0; j < window; j++) {
      sum += values[i - window + 1 + j] * normalizedWeights[j];
    }
    result.push(sum);
  }
  return result;
}

// ============================================================================
// TREND DETECTION
// ============================================================================

export interface TrendAnalysis {
  direction: 'increasing' | 'decreasing' | 'stable';
  slope: number;
  rSquared: number;
  significance: number;
  changePoints: number[]; // Indices of significant change points
}

export function analyzeTrend(values: number[]): TrendAnalysis {
  const n = values.length;
  const indices = values.map((_, i) => i);
  const pairs = indices.map((x, i) => [x, values[i]] as [number, number]);

  const regression = ss.linearRegression(pairs);
  const predicted = indices.map(x => regression.m * x + regression.b);

  // Calculate R-squared
  const meanY = ss.mean(values);
  const ssTot = values.reduce((acc, v) => acc + Math.pow(v - meanY, 2), 0);
  const ssRes = values.reduce((acc, v, i) => acc + Math.pow(v - predicted[i], 2), 0);
  const rSquared = ssTot === 0 ? 0 : 1 - ssRes / ssTot;

  // Mann-Kendall significance test approximation
  let s = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      s += Math.sign(values[j] - values[i]);
    }
  }
  const varS = (n * (n - 1) * (2 * n + 5)) / 18;
  const z = s > 0 ? (s - 1) / Math.sqrt(varS) : s < 0 ? (s + 1) / Math.sqrt(varS) : 0;
  const significance = 2 * (1 - normalCDF(Math.abs(z)));

  // Detect change points using CUSUM
  const changePoints = detectChangePoints(values);

  return {
    direction: Math.abs(regression.m) < 0.001 ? 'stable' : regression.m > 0 ? 'increasing' : 'decreasing',
    slope: regression.m,
    rSquared,
    significance,
    changePoints,
  };
}

function normalCDF(x: number): number {
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

function detectChangePoints(values: number[], threshold: number = 2): number[] {
  const mean = ss.mean(values);
  const stdDev = ss.standardDeviation(values);
  const changePoints: number[] = [];

  let cusum = 0;
  let cusumNeg = 0;

  for (let i = 0; i < values.length; i++) {
    const normalized = (values[i] - mean) / stdDev;
    cusum = Math.max(0, cusum + normalized - 0.5);
    cusumNeg = Math.min(0, cusumNeg + normalized + 0.5);

    if (cusum > threshold || cusumNeg < -threshold) {
      changePoints.push(i);
      cusum = 0;
      cusumNeg = 0;
    }
  }

  return changePoints;
}

// ============================================================================
// SEASONALITY DETECTION
// ============================================================================

export interface SeasonalityAnalysis {
  hasDailySeasonality: boolean;
  hasWeeklySeasonality: boolean;
  dominantPeriod: number | null;
  seasonalStrength: number;
  seasonalPattern: number[] | null;
}

export function analyzeSeasonality(values: number[], samplesPerDay: number = 96): SeasonalityAnalysis {
  // Simple FFT-based seasonality detection
  const fft = computeFFT(values);
  const magnitudes = fft.map(c => Math.sqrt(c.real * c.real + c.imag * c.imag));

  // Find dominant frequency (excluding DC component)
  let maxMag = 0;
  let dominantIdx = 0;
  for (let i = 1; i < magnitudes.length / 2; i++) {
    if (magnitudes[i] > maxMag) {
      maxMag = magnitudes[i];
      dominantIdx = i;
    }
  }

  const dominantPeriod = dominantIdx > 0 ? values.length / dominantIdx : null;

  // Check for daily seasonality (period = samplesPerDay)
  const dailyIdx = Math.round(values.length / samplesPerDay);
  const hasDailySeasonality = dailyIdx > 0 && dailyIdx < magnitudes.length && magnitudes[dailyIdx] > magnitudes[0] * 0.1;

  // Check for weekly seasonality (period = 7 * samplesPerDay)
  const weeklyIdx = Math.round(values.length / (7 * samplesPerDay));
  const hasWeeklySeasonality = weeklyIdx > 0 && weeklyIdx < magnitudes.length && magnitudes[weeklyIdx] > magnitudes[0] * 0.1;

  // Calculate seasonal strength
  const seasonalStrength = maxMag / (ss.sum(magnitudes) || 1);

  // Extract seasonal pattern if strong seasonality exists
  let seasonalPattern: number[] | null = null;
  if (dominantPeriod && seasonalStrength > 0.2) {
    const period = Math.round(dominantPeriod);
    seasonalPattern = new Array(period).fill(0);
    const counts = new Array(period).fill(0);

    for (let i = 0; i < values.length; i++) {
      const idx = i % period;
      seasonalPattern[idx] += values[i];
      counts[idx]++;
    }

    seasonalPattern = seasonalPattern.map((sum, i) => sum / counts[i]);
  }

  return {
    hasDailySeasonality,
    hasWeeklySeasonality,
    dominantPeriod,
    seasonalStrength,
    seasonalPattern,
  };
}

interface Complex {
  real: number;
  imag: number;
}

function computeFFT(values: number[]): Complex[] {
  // Pad to power of 2
  const n = Math.pow(2, Math.ceil(Math.log2(values.length)));
  const padded = [...values, ...new Array(n - values.length).fill(0)];

  // Simple DFT (for production, use proper FFT library)
  const result: Complex[] = [];
  for (let k = 0; k < n; k++) {
    let real = 0;
    let imag = 0;
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * k * t) / n;
      real += padded[t] * Math.cos(angle);
      imag -= padded[t] * Math.sin(angle);
    }
    result.push({ real, imag });
  }
  return result;
}

// ============================================================================
// DECOMPOSITION
// ============================================================================

export interface DecompositionResult {
  trend: number[];
  seasonal: number[];
  residual: number[];
}

export function decomposeTimeSeries(values: number[], period: number): DecompositionResult {
  // Trend extraction using centered moving average
  const halfPeriod = Math.floor(period / 2);
  const trend: number[] = [];

  for (let i = 0; i < values.length; i++) {
    if (i < halfPeriod || i >= values.length - halfPeriod) {
      // Extend trend at edges using linear regression
      const windowStart = Math.max(0, i - halfPeriod);
      const windowEnd = Math.min(values.length, i + halfPeriod + 1);
      trend.push(ss.mean(values.slice(windowStart, windowEnd)));
    } else {
      trend.push(ss.mean(values.slice(i - halfPeriod, i + halfPeriod + 1)));
    }
  }

  // Detrended series
  const detrended = values.map((v, i) => v - trend[i]);

  // Seasonal component - average by position in period
  const seasonalAvg: number[] = new Array(period).fill(0);
  const counts: number[] = new Array(period).fill(0);

  for (let i = 0; i < detrended.length; i++) {
    const pos = i % period;
    seasonalAvg[pos] += detrended[i];
    counts[pos]++;
  }

  for (let i = 0; i < period; i++) {
    seasonalAvg[i] /= counts[i];
  }

  // Normalize seasonal to sum to zero
  const seasonalMean = ss.mean(seasonalAvg);
  const normalizedSeasonal = seasonalAvg.map(s => s - seasonalMean);

  // Apply seasonal pattern
  const seasonal = values.map((_, i) => normalizedSeasonal[i % period]);

  // Residual
  const residual = values.map((v, i) => v - trend[i] - seasonal[i]);

  return { trend, seasonal, residual };
}

// ============================================================================
// FORECASTING
// ============================================================================

export interface ForecastResult {
  values: number[];
  lowerBound: number[];
  upperBound: number[];
  confidence: number;
}

export function forecast(timeSeries: KPITimeSeries, horizonPoints: number, confidenceLevel: number = 0.95): ForecastResult {
  const values = timeSeries.dataPoints.map(p => p.value);
  const stats = computeTimeSeriesStats(values);

  // Use Holt-Winters exponential smoothing
  const alpha = 0.3; // Level
  const beta = 0.1; // Trend
  const gamma = 0.1; // Seasonal

  // Determine seasonality period
  const seasonality = analyzeSeasonality(values);
  const period = seasonality.dominantPeriod ? Math.round(seasonality.dominantPeriod) : 1;

  // Initialize
  const level: number[] = [values[0]];
  const trend: number[] = [values.length > 1 ? values[1] - values[0] : 0];
  const seasonal: number[] = seasonality.seasonalPattern || new Array(period).fill(0);

  // Fit model
  for (let i = 1; i < values.length; i++) {
    const s = period > 1 ? seasonal[i % period] : 0;

    const newLevel = alpha * (values[i] - s) + (1 - alpha) * (level[i - 1] + trend[i - 1]);
    const newTrend = beta * (newLevel - level[i - 1]) + (1 - beta) * trend[i - 1];

    if (period > 1) {
      seasonal[i % period] = gamma * (values[i] - newLevel) + (1 - gamma) * seasonal[i % period];
    }

    level.push(newLevel);
    trend.push(newTrend);
  }

  // Forecast
  const lastLevel = level[level.length - 1];
  const lastTrend = trend[trend.length - 1];

  const forecastValues: number[] = [];
  for (let h = 1; h <= horizonPoints; h++) {
    const s = period > 1 ? seasonal[(values.length + h - 1) % period] : 0;
    forecastValues.push(lastLevel + h * lastTrend + s);
  }

  // Confidence intervals
  const zScore = 1.96; // ~95% confidence
  const stdError = stats.stdDev;

  const lowerBound = forecastValues.map((v, i) => v - zScore * stdError * Math.sqrt(1 + (i + 1) / values.length));
  const upperBound = forecastValues.map((v, i) => v + zScore * stdError * Math.sqrt(1 + (i + 1) / values.length));

  return {
    values: forecastValues,
    lowerBound,
    upperBound,
    confidence: confidenceLevel,
  };
}

// ============================================================================
// CROSS-CORRELATION (FOR MULTI-KPI ANALYSIS)
// ============================================================================

export function crossCorrelation(seriesA: number[], seriesB: number[], maxLag: number): number[] {
  const meanA = ss.mean(seriesA);
  const meanB = ss.mean(seriesB);
  const stdA = ss.standardDeviation(seriesA);
  const stdB = ss.standardDeviation(seriesB);
  const n = Math.min(seriesA.length, seriesB.length);

  const result: number[] = [];

  for (let lag = -maxLag; lag <= maxLag; lag++) {
    let sum = 0;
    let count = 0;

    for (let i = Math.max(0, lag); i < Math.min(n, n + lag); i++) {
      const j = i - lag;
      if (j >= 0 && j < seriesB.length) {
        sum += (seriesA[i] - meanA) * (seriesB[j] - meanB);
        count++;
      }
    }

    result.push(count > 0 ? sum / (count * stdA * stdB) : 0);
  }

  return result;
}

export default {
  computeTimeSeriesStats,
  simpleMovingAverage,
  exponentialMovingAverage,
  weightedMovingAverage,
  analyzeTrend,
  analyzeSeasonality,
  decomposeTimeSeries,
  forecast,
  crossCorrelation,
};
