/**
 * Midstreamer - Temporal Intelligence Engine for RAN Optimization
 *
 * Midstreamer v0.2.3 handles high-velocity time-series data with:
 * - Dynamic Time Warping for pattern recognition
 * - Chaos Analysis (Lyapunov exponents) for instability detection
 * - Attractor reconstruction for contextual anomaly detection
 * - N-BEATS style forecasting for predictive resource management
 * - Multi-granularity analysis (15-min, hourly, daily, weekly)
 */

export {
  MidstreamerEngine,
  DynamicTimeWarping,
  ChaosAnalyzer,
  AttractorDetector,
  TimeSeriesForecaster
} from './temporal-engine.js';

export type { TemporalEngineConfig } from './temporal-engine.js';
