/**
 * Statistical analysis utilities for benchmark results
 * Provides functions for calculating mean, median, percentiles, and speedup metrics
 */

/**
 * Calculate mean (average) of an array of numbers
 */
export function mean(values) {
  if (!values || values.length === 0) return 0;
  return values.reduce((sum, val) => sum + val, 0) / values.length;
}

/**
 * Calculate median of an array of numbers
 */
export function median(values) {
  if (!values || values.length === 0) return 0;

  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  } else {
    return sorted[mid];
  }
}

/**
 * Calculate percentile (p50, p95, p99, etc.)
 */
export function percentile(values, p) {
  if (!values || values.length === 0) return 0;
  if (p < 0 || p > 100) throw new Error('Percentile must be between 0 and 100');

  const sorted = [...values].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index % 1;

  if (lower === upper) {
    return sorted[lower];
  } else {
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }
}

/**
 * Calculate standard deviation
 */
export function standardDeviation(values) {
  if (!values || values.length === 0) return 0;

  const avg = mean(values);
  const squareDiffs = values.map(value => Math.pow(value - avg, 2));
  const avgSquareDiff = mean(squareDiffs);

  return Math.sqrt(avgSquareDiff);
}

/**
 * Calculate relative standard error (RSE)
 */
export function relativeStandardError(values) {
  if (!values || values.length === 0) return 0;

  const avg = mean(values);
  const stdDev = standardDeviation(values);

  return (stdDev / avg) * 100;
}

/**
 * Calculate speedup between two implementations
 */
export function speedup(baseline, optimized) {
  if (!baseline || !optimized) return null;
  return baseline / optimized;
}

/**
 * Calculate performance improvement percentage
 */
export function improvementPercentage(baseline, optimized) {
  if (!baseline || !optimized) return null;
  return ((baseline - optimized) / baseline) * 100;
}

/**
 * Calculate confidence interval (95% by default)
 */
export function confidenceInterval(values, confidence = 0.95) {
  if (!values || values.length < 2) return { lower: 0, upper: 0 };

  const avg = mean(values);
  const stdDev = standardDeviation(values);
  const n = values.length;

  // t-score for 95% confidence (approximation for large samples)
  const tScore = 1.96;

  const margin = tScore * (stdDev / Math.sqrt(n));

  return {
    lower: avg - margin,
    upper: avg + margin,
    margin
  };
}

/**
 * Detect performance regression
 */
export function detectRegression(baseline, current, threshold = 0.05) {
  if (!baseline || !current) return null;

  const change = (current - baseline) / baseline;

  return {
    hasRegression: change > threshold,
    percentChange: change * 100,
    isImprovement: change < 0,
    isSignificant: Math.abs(change) > threshold
  };
}

/**
 * Calculate throughput (ops/sec) from execution time (ms)
 */
export function throughput(executionTimeMs) {
  if (!executionTimeMs || executionTimeMs === 0) return 0;
  return 1000 / executionTimeMs;
}

/**
 * Calculate latency percentiles (p50, p95, p99)
 */
export function latencyPercentiles(values) {
  return {
    p50: percentile(values, 50),
    p90: percentile(values, 90),
    p95: percentile(values, 95),
    p99: percentile(values, 99),
    p999: percentile(values, 99.9)
  };
}

/**
 * Analyze benchmark samples
 */
export function analyzeSamples(samples) {
  if (!samples || samples.length === 0) {
    return null;
  }

  const stats = {
    count: samples.length,
    mean: mean(samples),
    median: median(samples),
    min: Math.min(...samples),
    max: Math.max(...samples),
    stdDev: standardDeviation(samples),
    rse: relativeStandardError(samples),
    percentiles: latencyPercentiles(samples),
    confidenceInterval: confidenceInterval(samples)
  };

  return stats;
}

/**
 * Compare two benchmark results
 */
export function compareBenchmarks(baseline, current) {
  if (!baseline || !current) return null;

  const baselineStats = analyzeSamples(baseline.samples || [baseline.mean]);
  const currentStats = analyzeSamples(current.samples || [current.mean]);

  const speedupValue = speedup(baselineStats.mean, currentStats.mean);
  const improvement = improvementPercentage(baselineStats.mean, currentStats.mean);
  const regression = detectRegression(baselineStats.mean, currentStats.mean);

  return {
    baseline: baselineStats,
    current: currentStats,
    speedup: speedupValue,
    improvement,
    regression,
    verdict: improvement > 0 ? 'faster' : improvement < 0 ? 'slower' : 'no change'
  };
}

/**
 * Format number with appropriate units
 */
export function formatDuration(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(2)}µs`;
  if (ms < 1000) return `${ms.toFixed(2)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
  return `${(ms / 60000).toFixed(2)}m`;
}

/**
 * Format throughput with appropriate units
 */
export function formatThroughput(opsPerSec) {
  if (opsPerSec >= 1000000) return `${(opsPerSec / 1000000).toFixed(2)}M ops/sec`;
  if (opsPerSec >= 1000) return `${(opsPerSec / 1000).toFixed(2)}K ops/sec`;
  return `${opsPerSec.toFixed(2)} ops/sec`;
}

/**
 * Format bytes with appropriate units
 */
export function formatBytes(bytes) {
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`;
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(2)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${bytes} bytes`;
}

/**
 * Calculate Amdahl's Law speedup limit
 */
export function amdahlSpeedup(parallelFraction, processorCount) {
  return 1 / ((1 - parallelFraction) + (parallelFraction / processorCount));
}

/**
 * Calculate Gustafson's Law scaled speedup
 */
export function gustafsonSpeedup(parallelFraction, processorCount) {
  return (1 - parallelFraction) + parallelFraction * processorCount;
}

export default {
  mean,
  median,
  percentile,
  standardDeviation,
  relativeStandardError,
  speedup,
  improvementPercentage,
  confidenceInterval,
  detectRegression,
  throughput,
  latencyPercentiles,
  analyzeSamples,
  compareBenchmarks,
  formatDuration,
  formatThroughput,
  formatBytes,
  amdahlSpeedup,
  gustafsonSpeedup
};
