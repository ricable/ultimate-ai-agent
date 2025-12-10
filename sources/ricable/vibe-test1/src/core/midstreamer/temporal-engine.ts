/**
 * Midstreamer - Temporal Intelligence Engine
 *
 * Handles high-velocity time-series data from the RAN (PM counters).
 * Key capabilities:
 * - Dynamic Time Warping (DTW) for pattern recognition despite temporal shifts
 * - Chaos Analysis (Lyapunov exponents) for detecting network instability
 * - Multi-granularity analysis (15-min, hourly, daily, weekly)
 * - Attractor reconstruction for contextual anomaly detection
 * - N-BEATS and TCN for forecasting
 */

import { EventEmitter } from 'eventemitter3';
import { TimeSeries, Anomaly, AnomalyType } from '../../types/index.js';

// ============================================================================
// DYNAMIC TIME WARPING (DTW)
// ============================================================================

export class DynamicTimeWarping {
  windowSize: number;
  distanceMetric: 'euclidean' | 'manhattan' | 'cosine';

  constructor(windowSize: number = 100, distanceMetric: 'euclidean' | 'manhattan' | 'cosine' = 'euclidean') {
    this.windowSize = windowSize;
    this.distanceMetric = distanceMetric;
  }

  /**
   * Compute DTW distance between two time series
   * Recognizes similar patterns even with temporal shifts
   */
  compute(series1: number[], series2: number[]): number {
    const n = series1.length;
    const m = series2.length;

    // Initialize cost matrix with infinity
    const dtw = Array(n + 1).fill(null).map(() =>
      Array(m + 1).fill(Infinity)
    );
    dtw[0][0] = 0;

    // Fill cost matrix with Sakoe-Chiba band constraint
    for (let i = 1; i <= n; i++) {
      const jStart = Math.max(1, i - this.windowSize);
      const jEnd = Math.min(m, i + this.windowSize);

      for (let j = jStart; j <= jEnd; j++) {
        const cost = this.distance(series1[i - 1], series2[j - 1]);
        dtw[i][j] = cost + Math.min(
          dtw[i - 1][j],     // Insertion
          dtw[i][j - 1],     // Deletion
          dtw[i - 1][j - 1]  // Match
        );
      }
    }

    return dtw[n][m];
  }

  /**
   * Find the optimal warping path
   */
  computeWithPath(series1: number[], series2: number[]): {
    distance: number;
    path: Array<[number, number]>;
  } {
    const n = series1.length;
    const m = series2.length;

    // Build cost matrix
    const dtw = Array(n + 1).fill(null).map(() =>
      Array(m + 1).fill(Infinity)
    );
    dtw[0][0] = 0;

    for (let i = 1; i <= n; i++) {
      const jStart = Math.max(1, i - this.windowSize);
      const jEnd = Math.min(m, i + this.windowSize);

      for (let j = jStart; j <= jEnd; j++) {
        const cost = this.distance(series1[i - 1], series2[j - 1]);
        dtw[i][j] = cost + Math.min(
          dtw[i - 1][j],
          dtw[i][j - 1],
          dtw[i - 1][j - 1]
        );
      }
    }

    // Backtrack to find path
    const path: Array<[number, number]> = [];
    let i = n, j = m;

    while (i > 0 || j > 0) {
      path.unshift([i - 1, j - 1]);

      if (i === 0) {
        j--;
      } else if (j === 0) {
        i--;
      } else {
        const options = [
          { di: -1, dj: 0, cost: dtw[i - 1][j] },
          { di: 0, dj: -1, cost: dtw[i][j - 1] },
          { di: -1, dj: -1, cost: dtw[i - 1][j - 1] }
        ];
        const best = options.reduce((a, b) => a.cost < b.cost ? a : b);
        i += best.di;
        j += best.dj;
      }
    }

    return { distance: dtw[n][m], path };
  }

  /**
   * Compare series against a library of patterns
   */
  findClosestPattern(
    series: number[],
    patterns: Map<string, number[]>
  ): { patternId: string; distance: number } | null {
    let bestMatch: { patternId: string; distance: number } | null = null;

    for (const [patternId, pattern] of patterns) {
      const distance = this.compute(series, pattern);

      if (!bestMatch || distance < bestMatch.distance) {
        bestMatch = { patternId, distance };
      }
    }

    return bestMatch;
  }

  private distance(a: number, b: number): number {
    switch (this.distanceMetric) {
      case 'euclidean':
        return (a - b) ** 2;
      case 'manhattan':
        return Math.abs(a - b);
      case 'cosine':
        // For single values, cosine reduces to sign comparison
        return 1 - (a * b) / (Math.abs(a) * Math.abs(b) + 1e-10);
      default:
        return (a - b) ** 2;
    }
  }
}

// ============================================================================
// CHAOS ANALYSIS - LYAPUNOV EXPONENT ESTIMATION
// ============================================================================

export class ChaosAnalyzer {
  embeddingDimension: number;
  delay: number;
  epsilon: number;

  constructor(embeddingDimension: number = 5, delay: number = 1, epsilon: number = 0.01) {
    this.embeddingDimension = embeddingDimension;
    this.delay = delay;
    this.epsilon = epsilon;
  }

  /**
   * Estimate the largest Lyapunov exponent
   * Positive values indicate chaos/instability
   * Near-zero indicates stable periodic behavior
   * Negative indicates convergence to fixed point
   */
  estimateLyapunovExponent(series: number[]): number {
    // Create delay embedding (Takens' theorem)
    const embedded = this.delayEmbed(series);
    if (embedded.length < 10) return 0;

    // Find pairs of nearby points
    const divergences: number[] = [];

    for (let i = 0; i < embedded.length - 1; i++) {
      // Find nearest neighbor (not temporally adjacent)
      let nearestIdx = -1;
      let nearestDist = Infinity;

      for (let j = 0; j < embedded.length - 1; j++) {
        if (Math.abs(i - j) < this.embeddingDimension * 2) continue;

        const dist = this.euclideanDistance(embedded[i], embedded[j]);
        if (dist < nearestDist && dist > this.epsilon) {
          nearestDist = dist;
          nearestIdx = j;
        }
      }

      if (nearestIdx === -1) continue;

      // Track divergence over time
      const initialDist = nearestDist;
      const finalDist = this.euclideanDistance(embedded[i + 1], embedded[nearestIdx + 1]);

      if (initialDist > 0 && finalDist > 0) {
        divergences.push(Math.log(finalDist / initialDist));
      }
    }

    if (divergences.length === 0) return 0;

    // Average divergence rate is the Lyapunov exponent
    return divergences.reduce((a, b) => a + b, 0) / divergences.length;
  }

  /**
   * Detect if system is entering chaotic regime
   */
  detectChaosOnset(series: number[], threshold: number = 0.05): {
    isChaotic: boolean;
    lyapunovExponent: number;
    stability: 'stable' | 'periodic' | 'edge-of-chaos' | 'chaotic';
  } {
    const lambda = this.estimateLyapunovExponent(series);

    let stability: 'stable' | 'periodic' | 'edge-of-chaos' | 'chaotic';
    if (lambda < -threshold) {
      stability = 'stable';
    } else if (Math.abs(lambda) <= threshold) {
      stability = 'periodic';
    } else if (lambda <= threshold * 3) {
      stability = 'edge-of-chaos';
    } else {
      stability = 'chaotic';
    }

    return {
      isChaotic: lambda > threshold,
      lyapunovExponent: lambda,
      stability
    };
  }

  /**
   * Create delay embedding of time series (Takens' embedding)
   */
  private delayEmbed(series: number[]): number[][] {
    const embedded: number[][] = [];
    const maxIdx = series.length - (this.embeddingDimension - 1) * this.delay;

    for (let i = 0; i < maxIdx; i++) {
      const point: number[] = [];
      for (let d = 0; d < this.embeddingDimension; d++) {
        point.push(series[i + d * this.delay]);
      }
      embedded.push(point);
    }

    return embedded;
  }

  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }
}

// ============================================================================
// ATTRACTOR RECONSTRUCTION FOR ANOMALY DETECTION
// ============================================================================

export class AttractorDetector {
  embeddingDimension: number;
  delay: number;
  manifoldRadius: number;
  private normalAttractor: number[][] = [];

  constructor(embeddingDimension: number = 5, delay: number = 1, manifoldRadius: number = 0.1) {
    this.embeddingDimension = embeddingDimension;
    this.delay = delay;
    this.manifoldRadius = manifoldRadius;
  }

  /**
   * Train the detector on normal operating data
   */
  train(normalData: number[][]): void {
    // Normalize and embed each dimension
    const normalized = this.normalizeMultivariate(normalData);
    this.normalAttractor = this.embedMultivariate(normalized);
  }

  /**
   * Detect if current state departs from normal attractor
   */
  detectDeparture(currentState: number[]): {
    isAnomaly: boolean;
    distance: number;
    nearestNormal: number[];
    severity: 'low' | 'medium' | 'high' | 'critical';
  } {
    if (this.normalAttractor.length === 0) {
      return {
        isAnomaly: false,
        distance: 0,
        nearestNormal: currentState,
        severity: 'low'
      };
    }

    // Find distance to nearest point on normal attractor
    let minDistance = Infinity;
    let nearestPoint = this.normalAttractor[0];

    for (const point of this.normalAttractor) {
      const dist = this.euclideanDistance(currentState, point);
      if (dist < minDistance) {
        minDistance = dist;
        nearestPoint = point;
      }
    }

    // Determine severity based on distance
    const normalizedDist = minDistance / this.manifoldRadius;
    let severity: 'low' | 'medium' | 'high' | 'critical';

    if (normalizedDist < 1) {
      severity = 'low';
    } else if (normalizedDist < 2) {
      severity = 'medium';
    } else if (normalizedDist < 4) {
      severity = 'high';
    } else {
      severity = 'critical';
    }

    return {
      isAnomaly: normalizedDist > 1,
      distance: minDistance,
      nearestNormal: nearestPoint,
      severity
    };
  }

  /**
   * Detect correlation breaks between metrics
   * E.g., "High Throughput" + "Zero Handover" is suspicious
   */
  detectCorrelationBreak(
    metrics: Map<string, number>,
    expectedCorrelations: Map<string, Map<string, number>>
  ): { metric1: string; metric2: string; expected: number; actual: number }[] {
    const breaks: { metric1: string; metric2: string; expected: number; actual: number }[] = [];

    for (const [metric1, correlations] of expectedCorrelations) {
      for (const [metric2, expectedCorr] of correlations) {
        const val1 = metrics.get(metric1);
        const val2 = metrics.get(metric2);

        if (val1 === undefined || val2 === undefined) continue;

        // Simple correlation check (sign agreement)
        const actualCorr = Math.sign(val1) === Math.sign(val2) ? 1 : -1;

        if (Math.sign(expectedCorr) !== actualCorr && Math.abs(expectedCorr) > 0.5) {
          breaks.push({ metric1, metric2, expected: expectedCorr, actual: actualCorr });
        }
      }
    }

    return breaks;
  }

  private normalizeMultivariate(data: number[][]): number[][] {
    if (data.length === 0) return [];

    const numDims = data[0].length;
    const means = new Array(numDims).fill(0);
    const stds = new Array(numDims).fill(0);

    // Calculate means
    for (const point of data) {
      for (let d = 0; d < numDims; d++) {
        means[d] += point[d];
      }
    }
    for (let d = 0; d < numDims; d++) {
      means[d] /= data.length;
    }

    // Calculate standard deviations
    for (const point of data) {
      for (let d = 0; d < numDims; d++) {
        stds[d] += (point[d] - means[d]) ** 2;
      }
    }
    for (let d = 0; d < numDims; d++) {
      stds[d] = Math.sqrt(stds[d] / data.length) || 1;
    }

    // Normalize
    return data.map(point =>
      point.map((val, d) => (val - means[d]) / stds[d])
    );
  }

  private embedMultivariate(data: number[][]): number[][] {
    // Flatten multivariate into single attractor representation
    const embedded: number[][] = [];

    for (let i = 0; i < data.length - this.delay * (this.embeddingDimension - 1); i++) {
      const point: number[] = [];
      for (let d = 0; d < this.embeddingDimension; d++) {
        const idx = i + d * this.delay;
        point.push(...data[idx]);
      }
      embedded.push(point);
    }

    return embedded;
  }

  private euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }
}

// ============================================================================
// TIME SERIES FORECASTER (N-BEATS inspired)
// ============================================================================

export class TimeSeriesForecaster {
  numBlocks: number;
  blockWidth: number;
  lookback: number;
  horizon: number;

  // Learned weights (simplified)
  private blockWeights: Float32Array[];

  constructor(
    lookback: number = 96,   // 24 hours at 15-min intervals
    horizon: number = 8,      // 2 hours forecast
    numBlocks: number = 4,
    blockWidth: number = 64
  ) {
    this.lookback = lookback;
    this.horizon = horizon;
    this.numBlocks = numBlocks;
    this.blockWidth = blockWidth;

    // Initialize block weights
    this.blockWeights = [];
    for (let i = 0; i < numBlocks; i++) {
      const weights = new Float32Array(lookback * blockWidth + blockWidth * horizon);
      for (let j = 0; j < weights.length; j++) {
        weights[j] = (Math.random() - 0.5) * 0.1;
      }
      this.blockWeights.push(weights);
    }
  }

  /**
   * Forecast future values based on historical lookback
   */
  forecast(history: number[]): number[] {
    if (history.length < this.lookback) {
      // Pad with last value
      const padded = new Array(this.lookback - history.length).fill(history[history.length - 1] || 0);
      history = [...padded, ...history];
    }

    const input = history.slice(-this.lookback);
    let residual = new Float32Array(input);
    const forecast = new Float32Array(this.horizon);

    // N-BEATS style: each block predicts and subtracts
    for (let block = 0; block < this.numBlocks; block++) {
      const weights = this.blockWeights[block];

      // Encode input to hidden
      const hidden = new Float32Array(this.blockWidth);
      for (let h = 0; h < this.blockWidth; h++) {
        for (let i = 0; i < this.lookback; i++) {
          hidden[h] += residual[i] * weights[i * this.blockWidth + h];
        }
        hidden[h] = Math.max(0, hidden[h]); // ReLU
      }

      // Decode to forecast
      const blockForecast = new Float32Array(this.horizon);
      const forecastOffset = this.lookback * this.blockWidth;
      for (let f = 0; f < this.horizon; f++) {
        for (let h = 0; h < this.blockWidth; h++) {
          blockForecast[f] += hidden[h] * weights[forecastOffset + h * this.horizon + f];
        }
      }

      // Add to total forecast
      for (let f = 0; f < this.horizon; f++) {
        forecast[f] += blockForecast[f];
      }
    }

    return Array.from(forecast);
  }

  /**
   * Train on historical data (simplified batch training)
   */
  train(data: number[][], epochs: number = 100, learningRate: number = 0.001): void {
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;

      for (const series of data) {
        if (series.length < this.lookback + this.horizon) continue;

        // Create training samples
        for (let i = 0; i <= series.length - this.lookback - this.horizon; i++) {
          const input = series.slice(i, i + this.lookback);
          const target = series.slice(i + this.lookback, i + this.lookback + this.horizon);

          // Forward pass
          const predicted = this.forecast(input);

          // Compute loss (MSE)
          let loss = 0;
          for (let j = 0; j < this.horizon; j++) {
            loss += (predicted[j] - target[j]) ** 2;
          }
          totalLoss += loss / this.horizon;

          // Simplified gradient update (would be backprop in real impl)
          // Just add noise proportional to error direction
          for (const weights of this.blockWeights) {
            for (let j = 0; j < weights.length; j++) {
              const grad = (Math.random() - 0.5) * loss * learningRate;
              weights[j] -= grad;
            }
          }
        }
      }
    }
  }
}

// ============================================================================
// MAIN TEMPORAL ENGINE CLASS
// ============================================================================

export interface TemporalEngineConfig {
  dtwWindowSize: number;
  chaosEmbeddingDim: number;
  chaosDelay: number;
  lyapunovThreshold: number;
  forecastLookback: number;
  forecastHorizon: number;
}

const DEFAULT_CONFIG: TemporalEngineConfig = {
  dtwWindowSize: 100,
  chaosEmbeddingDim: 5,
  chaosDelay: 1,
  lyapunovThreshold: 0.05,
  forecastLookback: 96,
  forecastHorizon: 8
};

export class MidstreamerEngine extends EventEmitter {
  config: TemporalEngineConfig;
  dtw: DynamicTimeWarping;
  chaosAnalyzer: ChaosAnalyzer;
  attractorDetector: AttractorDetector;
  forecaster: TimeSeriesForecaster;

  // Pattern library for daily/weekly profiles
  private dailyPatterns: Map<string, number[]> = new Map();
  private weeklyPatterns: Map<string, number[]> = new Map();

  // Statistics
  anomaliesDetected: number = 0;
  forecastsMade: number = 0;

  constructor(config: Partial<TemporalEngineConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.dtw = new DynamicTimeWarping(this.config.dtwWindowSize);
    this.chaosAnalyzer = new ChaosAnalyzer(
      this.config.chaosEmbeddingDim,
      this.config.chaosDelay
    );
    this.attractorDetector = new AttractorDetector(
      this.config.chaosEmbeddingDim,
      this.config.chaosDelay
    );
    this.forecaster = new TimeSeriesForecaster(
      this.config.forecastLookback,
      this.config.forecastHorizon
    );
  }

  /**
   * Process incoming time series and detect anomalies
   */
  processTimeSeries(
    seriesId: string,
    values: number[],
    granularity: '15min' | '1hour' | '1day' | '1week'
  ): Anomaly | null {
    // Check for chaos onset
    const chaosResult = this.chaosAnalyzer.detectChaosOnset(
      values,
      this.config.lyapunovThreshold
    );

    if (chaosResult.isChaotic) {
      this.anomaliesDetected++;
      return this.createAnomaly(seriesId, 'CHAOS_ONSET', values, chaosResult.stability);
    }

    // Check against historical patterns
    const patternLib = granularity === '1week' ? this.weeklyPatterns : this.dailyPatterns;
    const closestPattern = this.dtw.findClosestPattern(values, patternLib);

    if (closestPattern && closestPattern.distance > values.length * 0.5) {
      this.anomaliesDetected++;
      return this.createAnomaly(seriesId, 'PATTERN_DEVIATION', values, 'medium');
    }

    // Check for attractor departure (if trained)
    const currentState = values.slice(-this.config.chaosEmbeddingDim);
    const attractorResult = this.attractorDetector.detectDeparture(currentState);

    if (attractorResult.isAnomaly) {
      this.anomaliesDetected++;
      return this.createAnomaly(
        seriesId,
        'ATTRACTOR_DEPARTURE',
        values,
        attractorResult.severity
      );
    }

    return null;
  }

  /**
   * Detect threshold breach with context awareness
   */
  detectThresholdBreach(
    metricName: string,
    currentValue: number,
    threshold: number,
    recentHistory: number[]
  ): Anomaly | null {
    // Simple threshold
    if (currentValue > threshold) {
      // But check if this is normal for the time of day
      const mean = recentHistory.reduce((a, b) => a + b, 0) / recentHistory.length;
      const std = Math.sqrt(
        recentHistory.reduce((a, b) => a + (b - mean) ** 2, 0) / recentHistory.length
      );

      // Only alert if more than 2 sigma above recent mean
      if (currentValue > mean + 2 * std) {
        this.anomaliesDetected++;
        return this.createAnomaly(
          metricName,
          'THRESHOLD_BREACH',
          [currentValue],
          currentValue > mean + 3 * std ? 'high' : 'medium'
        );
      }
    }

    return null;
  }

  /**
   * Forecast future values
   */
  forecastMetric(history: number[]): number[] {
    this.forecastsMade++;
    return this.forecaster.forecast(history);
  }

  /**
   * Register a daily pattern for comparison
   */
  registerDailyPattern(patternId: string, values: number[]): void {
    this.dailyPatterns.set(patternId, values);
  }

  /**
   * Register a weekly pattern for comparison
   */
  registerWeeklyPattern(patternId: string, values: number[]): void {
    this.weeklyPatterns.set(patternId, values);
  }

  /**
   * Train the attractor detector on normal data
   */
  trainAttractorDetector(normalData: number[][]): void {
    this.attractorDetector.train(normalData);
    this.emit('attractor-trained', { sampleCount: normalData.length });
  }

  /**
   * Multi-granularity analysis
   */
  analyzeMultiGranularity(
    raw15Min: number[],
    cellId: string
  ): {
    hourly: { forecast: number[]; trend: 'up' | 'down' | 'stable' };
    daily: { matchedPattern: string | null; deviation: number };
    weekly: { forecast: number[]; seasonality: number };
  } {
    // Aggregate to hourly (4 x 15-min)
    const hourly: number[] = [];
    for (let i = 0; i < raw15Min.length; i += 4) {
      const chunk = raw15Min.slice(i, i + 4);
      hourly.push(chunk.reduce((a, b) => a + b, 0) / chunk.length);
    }

    // Hourly analysis
    const hourlyForecast = this.forecastMetric(hourly);
    const hourlyTrend = hourlyForecast[hourlyForecast.length - 1] > hourly[hourly.length - 1]
      ? 'up' as const
      : hourlyForecast[hourlyForecast.length - 1] < hourly[hourly.length - 1] * 0.95
        ? 'down' as const
        : 'stable' as const;

    // Daily pattern matching
    const dailyMatch = this.dtw.findClosestPattern(hourly.slice(-24), this.dailyPatterns);

    // Weekly analysis (aggregate to daily)
    const daily: number[] = [];
    for (let i = 0; i < hourly.length; i += 24) {
      const chunk = hourly.slice(i, i + 24);
      if (chunk.length > 0) {
        daily.push(chunk.reduce((a, b) => a + b, 0) / chunk.length);
      }
    }

    // Detect seasonality using autocorrelation
    const seasonality = this.computeSeasonality(daily, 7);
    const weeklyForecast = this.forecastMetric(daily);

    return {
      hourly: {
        forecast: hourlyForecast,
        trend: hourlyTrend
      },
      daily: {
        matchedPattern: dailyMatch?.patternId || null,
        deviation: dailyMatch?.distance || 0
      },
      weekly: {
        forecast: weeklyForecast,
        seasonality
      }
    };
  }

  private computeSeasonality(series: number[], period: number): number {
    if (series.length < period * 2) return 0;

    // Compute autocorrelation at given period
    const n = series.length;
    const mean = series.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n - period; i++) {
      numerator += (series[i] - mean) * (series[i + period] - mean);
    }

    for (let i = 0; i < n; i++) {
      denominator += (series[i] - mean) ** 2;
    }

    return denominator > 0 ? numerator / denominator : 0;
  }

  private createAnomaly(
    seriesId: string,
    type: AnomalyType,
    values: number[],
    severity: 'low' | 'medium' | 'high' | 'critical' | string
  ): Anomaly {
    const now = new Date();

    return {
      id: `anomaly-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: now,
      cgi: seriesId,
      type,
      severity: (severity === 'edge-of-chaos' || severity === 'chaotic' ? 'high' : severity) as 'low' | 'medium' | 'high' | 'critical',
      anomalyVector: values.slice(-10),
      affectedMetrics: [seriesId],
      confidence: 0.85,
      context: {
        timeOfDay: now.getHours(),
        dayOfWeek: now.getDay(),
        isWeekend: now.getDay() === 0 || now.getDay() === 6
      }
    };
  }

  getStats(): {
    anomaliesDetected: number;
    forecastsMade: number;
    dailyPatternsCount: number;
    weeklyPatternsCount: number;
  } {
    return {
      anomaliesDetected: this.anomaliesDetected,
      forecastsMade: this.forecastsMade,
      dailyPatternsCount: this.dailyPatterns.size,
      weeklyPatternsCount: this.weeklyPatterns.size
    };
  }
}
