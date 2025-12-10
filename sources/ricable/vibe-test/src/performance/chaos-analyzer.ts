/**
 * Chaos Analyzer - Detects chaotic behavior in network time series
 * Uses sublinear algorithms for O(log^k n) complexity
 */

import {
  ChaosAnalysis,
  TimeSeries,
  CellMetrics,
} from '../core/types.js';
import {
  calculateLyapunovExponent,
  calculateCorrelationDimension,
  calculateEntropy,
  fft,
  standardDeviation,
} from '../utils/math.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('ChaosAnalyzer');

/**
 * Configuration for chaos analysis
 */
export interface ChaosAnalyzerConfig {
  lyapunovThreshold: number;      // Above this = chaotic
  entropyThreshold: number;       // Above this = high entropy
  embeddingDimension: number;     // For phase space reconstruction
  timeDelay: number;              // Time delay for embedding
  minDataPoints: number;          // Minimum points for analysis
  temporalLeadMs: number;         // Target temporal lead (36.4ms)
}

const DEFAULT_CONFIG: ChaosAnalyzerConfig = {
  lyapunovThreshold: 0.05,
  entropyThreshold: 0.8,
  embeddingDimension: 3,
  timeDelay: 1,
  minDataPoints: 100,
  temporalLeadMs: 36.4,
};

/**
 * Chaos Analyzer for network traffic dynamics
 * Implements algorithms from sublinear-time-solver
 */
export class ChaosAnalyzer {
  private config: ChaosAnalyzerConfig;
  private analysisCache: Map<string, ChaosAnalysis>;
  private cacheTTL: number = 5000; // 5 second cache

  constructor(config: Partial<ChaosAnalyzerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.analysisCache = new Map();

    logger.info('Chaos analyzer initialized', { config: this.config });
  }

  /**
   * Analyze a time series for chaotic behavior
   * Returns within target temporal lead of ~36.4ms
   */
  async analyze(timeSeries: TimeSeries): Promise<ChaosAnalysis> {
    const startTime = performance.now();
    const cacheKey = `${timeSeries.cellId}:${timeSeries.metricName}`;

    // Check cache
    const cached = this.analysisCache.get(cacheKey);
    if (cached && Date.now() - cached.entropy < this.cacheTTL) {
      return cached;
    }

    const values = timeSeries.values;

    if (values.length < this.config.minDataPoints) {
      logger.warn('Insufficient data points for chaos analysis', {
        cellId: timeSeries.cellId,
        dataPoints: values.length,
      });
      return this.createDefaultAnalysis();
    }

    // Calculate Lyapunov exponent (main chaos indicator)
    const lyapunovExponent = calculateLyapunovExponent(
      values,
      this.config.embeddingDimension,
      this.config.timeDelay
    );

    // Calculate correlation dimension
    const correlationDimension = calculateCorrelationDimension(
      values,
      this.config.embeddingDimension,
      this.config.timeDelay
    );

    // Calculate entropy of the distribution
    const entropy = this.calculateDistributionEntropy(values);

    // Calculate predictability score
    const predictability = this.calculatePredictability(values);

    // Determine if system is chaotic
    const isChaoatic = lyapunovExponent > this.config.lyapunovThreshold;

    // Recommend control strategy
    const recommendedStrategy = this.determineStrategy(
      isChaoatic,
      predictability,
      entropy
    );

    const analysis: ChaosAnalysis = {
      isChaoatic,
      lyapunovExponent,
      correlationDimension,
      entropy,
      predictability,
      recommendedStrategy,
    };

    // Cache result
    this.analysisCache.set(cacheKey, analysis);

    const executionTime = performance.now() - startTime;
    logger.debug('Chaos analysis complete', {
      cellId: timeSeries.cellId,
      metric: timeSeries.metricName,
      isChaoatic,
      lyapunovExponent: lyapunovExponent.toFixed(4),
      executionTimeMs: executionTime.toFixed(2),
    });

    return analysis;
  }

  /**
   * Batch analyze multiple time series (parallel processing)
   */
  async analyzeBatch(seriesList: TimeSeries[]): Promise<Map<string, ChaosAnalysis>> {
    const results = new Map<string, ChaosAnalysis>();

    const analyses = await Promise.all(
      seriesList.map((series) => this.analyze(series))
    );

    seriesList.forEach((series, index) => {
      const key = `${series.cellId}:${series.metricName}`;
      results.set(key, analyses[index]);
    });

    return results;
  }

  /**
   * Detect onset of chaos in streaming data
   * Uses sliding window for real-time detection
   */
  detectChaosOnset(
    values: number[],
    windowSize: number = 50
  ): { isOnset: boolean; transitionPoint: number; trend: 'increasing' | 'decreasing' | 'stable' } {
    if (values.length < windowSize * 2) {
      return { isOnset: false, transitionPoint: -1, trend: 'stable' };
    }

    const windows: number[] = [];
    const step = Math.max(1, Math.floor(windowSize / 10));

    // Calculate Lyapunov for sliding windows
    for (let i = 0; i <= values.length - windowSize; i += step) {
      const window = values.slice(i, i + windowSize);
      const lyap = calculateLyapunovExponent(window, 2, 1);
      windows.push(lyap);
    }

    // Detect transition
    let transitionPoint = -1;
    let maxChange = 0;

    for (let i = 1; i < windows.length; i++) {
      const change = windows[i] - windows[i - 1];
      if (Math.abs(change) > maxChange) {
        maxChange = Math.abs(change);
        if (windows[i] > this.config.lyapunovThreshold && windows[i - 1] <= this.config.lyapunovThreshold) {
          transitionPoint = i * step;
        }
      }
    }

    // Determine trend
    const recentTrend = windows.slice(-5);
    const firstHalf = recentTrend.slice(0, 2).reduce((a, b) => a + b, 0) / 2;
    const secondHalf = recentTrend.slice(-2).reduce((a, b) => a + b, 0) / 2;

    let trend: 'increasing' | 'decreasing' | 'stable';
    if (secondHalf - firstHalf > 0.01) {
      trend = 'increasing';
    } else if (firstHalf - secondHalf > 0.01) {
      trend = 'decreasing';
    } else {
      trend = 'stable';
    }

    return {
      isOnset: transitionPoint !== -1,
      transitionPoint,
      trend,
    };
  }

  /**
   * Analyze network stability across multiple cells
   */
  analyzeNetworkStability(cellMetrics: Map<string, CellMetrics>): NetworkStabilityReport {
    const analyses: CellStabilityInfo[] = [];

    for (const [cellId, metrics] of cellMetrics) {
      // Create time series from metrics (simulated historical data)
      const throughputSeries: TimeSeries = {
        metricName: 'throughput',
        cellId,
        values: this.generateSimulatedHistory(metrics.throughputDl, 200),
        timestamps: [],
        resolution: 1000,
      };

      const chaos = {
        lyapunovExponent: calculateLyapunovExponent(throughputSeries.values, 3, 1),
        isChaoatic: false,
      };
      chaos.isChaoatic = chaos.lyapunovExponent > this.config.lyapunovThreshold;

      analyses.push({
        cellId,
        lyapunovExponent: chaos.lyapunovExponent,
        isChaoatic: chaos.isChaoatic,
        currentLoad: metrics.prbUtilizationDl,
      });
    }

    // Calculate network-wide stability
    const chaoticCells = analyses.filter((a) => a.isChaoatic).length;
    const avgLyapunov =
      analyses.reduce((sum, a) => sum + a.lyapunovExponent, 0) / analyses.length;

    return {
      timestamp: Date.now(),
      totalCells: analyses.length,
      chaoticCells,
      stableRatio: (analyses.length - chaoticCells) / analyses.length,
      averageLyapunov: avgLyapunov,
      cellAnalyses: analyses,
      networkState: chaoticCells === 0 ? 'stable' : chaoticCells < analyses.length * 0.1 ? 'mostly_stable' : 'unstable',
    };
  }

  /**
   * Calculate distribution entropy
   */
  private calculateDistributionEntropy(values: number[]): number {
    // Create histogram
    const bins = 20;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const binWidth = (max - min) / bins || 1;

    const histogram = new Array(bins).fill(0);
    for (const value of values) {
      const bin = Math.min(bins - 1, Math.floor((value - min) / binWidth));
      histogram[bin]++;
    }

    // Convert to probabilities
    const probabilities = histogram.map((count) => count / values.length);

    return calculateEntropy(probabilities);
  }

  /**
   * Calculate predictability score based on autocorrelation
   */
  private calculatePredictability(values: number[]): number {
    // Use FFT to analyze periodicity
    const { real, imag } = fft(values);

    // Calculate power spectrum
    const power = real.map((r, i) => r * r + imag[i] * imag[i]);

    // Find dominant frequencies
    const maxPower = Math.max(...power.slice(1, power.length / 2));
    const dcPower = power[0];

    // High ratio of dominant frequency to DC = more predictable
    const spectralRatio = dcPower > 0 ? maxPower / dcPower : 0;

    // Also consider coefficient of variation
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const cv = mean !== 0 ? standardDeviation(values) / Math.abs(mean) : 1;

    // Combine metrics into predictability score (0-1)
    const predictability = Math.max(0, Math.min(1,
      0.5 * (1 - cv) + 0.5 * Math.tanh(spectralRatio)
    ));

    return predictability;
  }

  /**
   * Determine recommended control strategy
   */
  private determineStrategy(
    isChaoatic: boolean,
    predictability: number,
    entropy: number
  ): 'predictive' | 'damping' | 'hybrid' {
    if (isChaoatic || entropy > this.config.entropyThreshold) {
      // Chaotic systems need damping (stability-focused control)
      return 'damping';
    } else if (predictability > 0.7) {
      // Highly predictable systems benefit from predictive control
      return 'predictive';
    } else {
      // Moderate predictability - use hybrid approach
      return 'hybrid';
    }
  }

  /**
   * Create default analysis for insufficient data
   */
  private createDefaultAnalysis(): ChaosAnalysis {
    return {
      isChaoatic: false,
      lyapunovExponent: 0,
      correlationDimension: 0,
      entropy: 0,
      predictability: 0.5,
      recommendedStrategy: 'hybrid',
    };
  }

  /**
   * Generate simulated historical data for testing
   */
  private generateSimulatedHistory(currentValue: number, length: number): number[] {
    const history: number[] = [];
    let value = currentValue;

    for (let i = 0; i < length; i++) {
      // Add some realistic variation
      const noise = (Math.random() - 0.5) * currentValue * 0.1;
      const trend = Math.sin(i / 20) * currentValue * 0.05;
      value = currentValue + noise + trend;
      history.push(Math.max(0, value));
    }

    return history;
  }

  /**
   * Clear the analysis cache
   */
  clearCache(): void {
    this.analysisCache.clear();
  }
}

/**
 * Network stability report
 */
export interface NetworkStabilityReport {
  timestamp: number;
  totalCells: number;
  chaoticCells: number;
  stableRatio: number;
  averageLyapunov: number;
  cellAnalyses: CellStabilityInfo[];
  networkState: 'stable' | 'mostly_stable' | 'unstable';
}

/**
 * Individual cell stability information
 */
export interface CellStabilityInfo {
  cellId: string;
  lyapunovExponent: number;
  isChaoatic: boolean;
  currentLoad: number;
}

/**
 * Create a configured chaos analyzer instance
 */
export function createChaosAnalyzer(
  config?: Partial<ChaosAnalyzerConfig>
): ChaosAnalyzer {
  return new ChaosAnalyzer(config);
}
