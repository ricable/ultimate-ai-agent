// Claude Skill: Time Series Analysis
// Provides specialized time series analysis capabilities as a reusable skill

import { MultiVariatePoint, PredictionResult } from '../types/ran-models.js';
import { logger } from '../core/logger.js';

export interface TimeSeriesAnalysisSkillInput {
  data: MultiVariatePoint[];
  analysisType: 'trend' | 'seasonality' | 'forecast' | 'anomaly' | 'correlation';
  options?: {
    forecastHorizon?: number;
    seasonalPeriod?: number;
    anomalyThreshold?: number;
  };
}

export interface TimeSeriesAnalysisSkillOutput {
  success: boolean;
  analysisType: string;
  results: any;
  metadata: {
    dataPoints: number;
    processingTimeMs: number;
    model?: string;
  };
}

/**
 * Claude Skill for Time Series Analysis
 * Can be invoked as a subagent or standalone skill
 */
export class TimeSeriesAnalysisSkill {
  private name: string = 'TimeSeriesAnalysis';

  async execute(input: TimeSeriesAnalysisSkillInput): Promise<TimeSeriesAnalysisSkillOutput> {
    const startTime = Date.now();
    logger.info(`[Skill:${this.name}] Executing ${input.analysisType} analysis`, {
      dataPoints: input.data.length,
    });

    try {
      let results: any;

      switch (input.analysisType) {
        case 'trend':
          results = await this.analyzeTrend(input.data);
          break;
        case 'seasonality':
          results = await this.detectSeasonality(input.data, input.options?.seasonalPeriod);
          break;
        case 'forecast':
          results = await this.forecast(input.data, input.options?.forecastHorizon || 24);
          break;
        case 'anomaly':
          results = await this.detectAnomalies(input.data, input.options?.anomalyThreshold || 3.0);
          break;
        case 'correlation':
          results = await this.analyzeCorrelations(input.data);
          break;
        default:
          throw new Error(`Unknown analysis type: ${input.analysisType}`);
      }

      const processingTimeMs = Date.now() - startTime;

      return {
        success: true,
        analysisType: input.analysisType,
        results,
        metadata: {
          dataPoints: input.data.length,
          processingTimeMs,
        },
      };
    } catch (error) {
      logger.error(`[Skill:${this.name}] Analysis failed`, { error });
      throw error;
    }
  }

  private async analyzeTrend(data: MultiVariatePoint[]): Promise<any> {
    const trends: Record<string, any> = {};

    if (data.length < 2) {
      return trends;
    }

    // Get all feature names
    const featureNames = Object.keys(data[0].features);

    for (const featureName of featureNames) {
      const values = data
        .map(point => point.features[featureName])
        .filter(v => v !== undefined);

      if (values.length < 2) continue;

      // Simple linear regression
      const n = values.length;
      const indices = Array.from({ length: n }, (_, i) => i);
      const sumX = indices.reduce((a, b) => a + b, 0);
      const sumY = values.reduce((a, b) => a + b, 0);
      const sumXY = indices.reduce((acc, x, i) => acc + x * values[i], 0);
      const sumXX = indices.reduce((acc, x) => acc + x * x, 0);

      const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
      const intercept = (sumY - slope * sumX) / n;

      const mean = sumY / n;
      const relativeSlope = mean !== 0 ? (slope / mean) * 100 : 0;

      trends[featureName] = {
        slope,
        intercept,
        relativeSlope: relativeSlope.toFixed(4),
        direction: slope > 0.01 ? 'increasing' : slope < -0.01 ? 'decreasing' : 'stable',
        confidence: this.calculateTrendConfidence(values, slope, intercept),
      };
    }

    return trends;
  }

  private calculateTrendConfidence(values: number[], slope: number, intercept: number): number {
    if (values.length < 2) return 0;

    const n = values.length;
    const predicted = values.map((_, i) => slope * i + intercept);
    const residuals = values.map((y, i) => y - predicted[i]);
    const rss = residuals.reduce((acc, r) => acc + r * r, 0);
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const tss = values.reduce((acc, y) => acc + (y - mean) ** 2, 0);
    const rSquared = 1 - rss / tss;

    return Math.max(0, Math.min(1, rSquared));
  }

  private async detectSeasonality(data: MultiVariatePoint[], period?: number): Promise<any> {
    const seasonality: Record<string, any> = {};
    const defaultPeriod = period || 24; // Default: daily seasonality for hourly data

    const featureNames = Object.keys(data[0]?.features || {});

    for (const featureName of featureNames) {
      const values = data
        .map(point => point.features[featureName])
        .filter(v => v !== undefined);

      if (values.length < defaultPeriod * 2) continue;

      // Calculate autocorrelation at seasonal lag
      const autocorr = this.calculateAutocorrelation(values, defaultPeriod);

      seasonality[featureName] = {
        period: defaultPeriod,
        autocorrelation: autocorr,
        hasSeasonality: autocorr > 0.6,
        strength: autocorr > 0.8 ? 'strong' : autocorr > 0.6 ? 'moderate' : 'weak',
      };
    }

    return seasonality;
  }

  private calculateAutocorrelation(values: number[], lag: number): number {
    if (values.length < lag + 1) return 0;

    const n = values.length - lag;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;

    let numerator = 0;
    let denominator = 0;

    for (let i = 0; i < n; i++) {
      numerator += (values[i] - mean) * (values[i + lag] - mean);
    }

    for (let i = 0; i < values.length; i++) {
      denominator += (values[i] - mean) ** 2;
    }

    return denominator !== 0 ? numerator / denominator : 0;
  }

  private async forecast(data: MultiVariatePoint[], horizon: number): Promise<PredictionResult[]> {
    const predictions: PredictionResult[] = [];

    if (data.length === 0) return predictions;

    const lastTimestamp = data[data.length - 1].timestamp;
    const featureNames = Object.keys(data[0].features);

    // Simple forecasting: use recent average
    const recentData = data.slice(-24); // Last 24 points

    for (let h = 0; h < horizon; h++) {
      const timestamp = lastTimestamp + (h + 1) * 3600; // Hourly
      const features: Record<string, number> = {};

      for (const featureName of featureNames) {
        const recentValues = recentData
          .map(p => p.features[featureName])
          .filter(v => v !== undefined);

        const mean = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
        features[featureName] = mean;
      }

      // Use first feature as the predicted value
      const predictedValue = features[featureNames[0]] || 0;
      const stdDev = this.calculateStdDev(
        recentData.map(p => p.features[featureNames[0]]).filter(v => v !== undefined)
      );

      predictions.push({
        timestamp,
        predictedValue,
        confidenceInterval: [predictedValue - 2 * stdDev, predictedValue + 2 * stdDev],
        anomalyScore: 0,
        featureImportance: {},
      });
    }

    return predictions;
  }

  private calculateStdDev(values: number[]): number {
    if (values.length === 0) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((acc, v) => acc + (v - mean) ** 2, 0) / values.length;
    return Math.sqrt(variance);
  }

  private async detectAnomalies(data: MultiVariatePoint[], threshold: number): Promise<any[]> {
    const anomalies: any[] = [];

    if (data.length < 10) return anomalies;

    const featureNames = Object.keys(data[0]?.features || {});

    for (const featureName of featureNames) {
      const values = data
        .map(point => point.features[featureName])
        .filter(v => v !== undefined);

      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const stdDev = this.calculateStdDev(values);

      data.forEach((point, idx) => {
        const value = point.features[featureName];
        if (value === undefined) return;

        const zScore = stdDev !== 0 ? Math.abs(value - mean) / stdDev : 0;

        if (zScore > threshold) {
          anomalies.push({
            timestamp: point.timestamp,
            featureName,
            value,
            expectedValue: mean,
            zScore,
            severity: zScore > 5 ? 'critical' : zScore > 4 ? 'major' : 'minor',
          });
        }
      });
    }

    return anomalies.sort((a, b) => b.zScore - a.zScore);
  }

  private async analyzeCorrelations(data: MultiVariatePoint[]): Promise<any[]> {
    const correlations: any[] = [];

    if (data.length < 2) return correlations;

    const featureNames = Object.keys(data[0]?.features || {});

    for (let i = 0; i < featureNames.length; i++) {
      for (let j = i + 1; j < featureNames.length; j++) {
        const feature1 = featureNames[i];
        const feature2 = featureNames[j];

        const values1 = data.map(p => p.features[feature1]).filter(v => v !== undefined);
        const values2 = data.map(p => p.features[feature2]).filter(v => v !== undefined);

        const minLength = Math.min(values1.length, values2.length);
        if (minLength < 2) continue;

        const corr = this.calculateCorrelation(
          values1.slice(0, minLength),
          values2.slice(0, minLength)
        );

        if (Math.abs(corr) > 0.5) {
          correlations.push({
            feature1,
            feature2,
            correlation: corr,
            strength: Math.abs(corr) > 0.8 ? 'strong' : 'moderate',
            direction: corr > 0 ? 'positive' : 'negative',
          });
        }
      }
    }

    return correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
  }

  private calculateCorrelation(values1: number[], values2: number[]): number {
    const n = values1.length;
    if (n !== values2.length || n < 2) return 0;

    const mean1 = values1.reduce((a, b) => a + b, 0) / n;
    const mean2 = values2.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let sum1 = 0;
    let sum2 = 0;

    for (let i = 0; i < n; i++) {
      const diff1 = values1[i] - mean1;
      const diff2 = values2[i] - mean2;
      numerator += diff1 * diff2;
      sum1 += diff1 * diff1;
      sum2 += diff2 * diff2;
    }

    const denominator = Math.sqrt(sum1 * sum2);
    return denominator !== 0 ? numerator / denominator : 0;
  }
}

export default TimeSeriesAnalysisSkill;
