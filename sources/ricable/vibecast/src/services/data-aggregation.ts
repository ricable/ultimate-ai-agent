// Data Aggregation Service
// Handles multi-granularity time series aggregation (hourly, daily, weekly)

import { MultiVariatePoint, Granularity, KpiMeasurement } from '../types/ran-models.js';
import { logger } from '../core/logger.js';

export interface AggregationOptions {
  granularities: Granularity[];
  aggregationMethods: ('mean' | 'sum' | 'max' | 'min' | 'count')[];
  timeRangeHours?: number;
}

export interface AggregatedData {
  granularity: Granularity;
  data: MultiVariatePoint[];
  metadata: {
    originalPoints: number;
    aggregatedPoints: number;
    startTime: number;
    endTime: number;
  };
}

export class DataAggregationService {
  /**
   * Aggregate data to multiple granularities
   */
  async aggregateMultiGranularity(
    rawData: MultiVariatePoint[],
    options: AggregationOptions
  ): Promise<Map<Granularity, AggregatedData>> {
    logger.info('Starting multi-granularity aggregation', {
      rawDataPoints: rawData.length,
      granularities: options.granularities,
    });

    const results = new Map<Granularity, AggregatedData>();

    for (const granularity of options.granularities) {
      const aggregated = await this.aggregateToGranularity(
        rawData,
        granularity,
        options.aggregationMethods
      );
      results.set(granularity, aggregated);
    }

    return results;
  }

  /**
   * Aggregate to a specific granularity
   */
  private async aggregateToGranularity(
    rawData: MultiVariatePoint[],
    granularity: Granularity,
    methods: string[]
  ): Promise<AggregatedData> {
    if (rawData.length === 0) {
      return {
        granularity,
        data: [],
        metadata: {
          originalPoints: 0,
          aggregatedPoints: 0,
          startTime: 0,
          endTime: 0,
        },
      };
    }

    const bucketSizeSeconds = this.getGranularitySeconds(granularity);
    const buckets = this.createTimeBuckets(rawData, bucketSizeSeconds);
    const aggregatedData = this.aggregateBuckets(buckets, methods);

    const startTime = rawData[0].timestamp;
    const endTime = rawData[rawData.length - 1].timestamp;

    return {
      granularity,
      data: aggregatedData,
      metadata: {
        originalPoints: rawData.length,
        aggregatedPoints: aggregatedData.length,
        startTime,
        endTime,
      },
    };
  }

  /**
   * Get bucket size in seconds for granularity
   */
  private getGranularitySeconds(granularity: Granularity): number {
    switch (granularity) {
      case 'Hourly':
        return 3600; // 1 hour
      case 'Daily':
        return 86400; // 24 hours
      case 'Weekly':
        return 604800; // 7 days
      default:
        return 3600;
    }
  }

  /**
   * Create time buckets for aggregation
   */
  private createTimeBuckets(
    data: MultiVariatePoint[],
    bucketSizeSeconds: number
  ): Map<number, MultiVariatePoint[]> {
    const buckets = new Map<number, MultiVariatePoint[]>();

    for (const point of data) {
      const bucketKey = Math.floor(point.timestamp / bucketSizeSeconds) * bucketSizeSeconds;

      if (!buckets.has(bucketKey)) {
        buckets.set(bucketKey, []);
      }

      buckets.get(bucketKey)!.push(point);
    }

    return buckets;
  }

  /**
   * Aggregate data within buckets
   */
  private aggregateBuckets(
    buckets: Map<number, MultiVariatePoint[]>,
    methods: string[]
  ): MultiVariatePoint[] {
    const aggregated: MultiVariatePoint[] = [];

    for (const [timestamp, points] of buckets) {
      if (points.length === 0) continue;

      // Get all feature names
      const featureNames = new Set<string>();
      points.forEach(p => Object.keys(p.features).forEach(k => featureNames.add(k)));

      // Aggregate each feature
      const aggregatedFeatures: Record<string, number> = {};

      for (const featureName of featureNames) {
        const values = points
          .map(p => p.features[featureName])
          .filter(v => v !== undefined);

        if (values.length === 0) continue;

        // Use first method as primary aggregation
        const method = methods[0] || 'mean';
        aggregatedFeatures[featureName] = this.applyAggregationMethod(values, method);

        // Add additional aggregation methods as separate features
        for (let i = 1; i < methods.length; i++) {
          const additionalMethod = methods[i];
          const additionalFeatureName = `${featureName}_${additionalMethod}`;
          aggregatedFeatures[additionalFeatureName] = this.applyAggregationMethod(
            values,
            additionalMethod
          );
        }
      }

      // Merge labels
      const mergedLabels: Record<string, string> = {};
      points.forEach(p => Object.assign(mergedLabels, p.labels));

      aggregated.push({
        timestamp,
        features: aggregatedFeatures,
        labels: mergedLabels,
      });
    }

    return aggregated.sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Apply aggregation method to values
   */
  private applyAggregationMethod(values: number[], method: string): number {
    switch (method) {
      case 'mean':
        return values.reduce((a, b) => a + b, 0) / values.length;
      case 'sum':
        return values.reduce((a, b) => a + b, 0);
      case 'max':
        return Math.max(...values);
      case 'min':
        return Math.min(...values);
      case 'count':
        return values.length;
      default:
        return values.reduce((a, b) => a + b, 0) / values.length;
    }
  }

  /**
   * Convert KPI measurements to MultiVariatePoint format
   */
  kpiMeasurementsToMultivariate(measurements: KpiMeasurement[]): MultiVariatePoint[] {
    const grouped = new Map<number, Map<string, number>>();
    const labels = new Map<number, Record<string, string>>();

    for (const m of measurements) {
      if (!grouped.has(m.timestamp)) {
        grouped.set(m.timestamp, new Map());
        labels.set(m.timestamp, {});
      }

      grouped.get(m.timestamp)!.set(m.kpiName, m.value);

      const pointLabels = labels.get(m.timestamp)!;
      pointLabels['nodeId'] = m.nodeId;
      if (m.cellId) pointLabels['cellId'] = m.cellId;
      pointLabels['granularity'] = m.granularity;
    }

    const points: MultiVariatePoint[] = [];
    for (const [timestamp, features] of grouped) {
      points.push({
        timestamp,
        features: Object.fromEntries(features),
        labels: labels.get(timestamp) || {},
      });
    }

    return points.sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Downsample data by selecting every Nth point
   */
  downsample(data: MultiVariatePoint[], factor: number): MultiVariatePoint[] {
    if (factor <= 1) return data;
    return data.filter((_, idx) => idx % factor === 0);
  }

  /**
   * Fill missing time points with interpolation
   */
  fillMissingPoints(
    data: MultiVariatePoint[],
    expectedIntervalSeconds: number
  ): MultiVariatePoint[] {
    if (data.length < 2) return data;

    const filled: MultiVariatePoint[] = [];
    filled.push(data[0]);

    for (let i = 1; i < data.length; i++) {
      const prev = data[i - 1];
      const curr = data[i];
      const gap = curr.timestamp - prev.timestamp;
      const expectedGaps = Math.floor(gap / expectedIntervalSeconds);

      // If gap is larger than expected, fill with interpolated points
      if (expectedGaps > 1) {
        for (let j = 1; j < expectedGaps; j++) {
          const ratio = j / expectedGaps;
          const interpolatedTimestamp = prev.timestamp + j * expectedIntervalSeconds;
          const interpolatedFeatures: Record<string, number> = {};

          for (const featureName of Object.keys(prev.features)) {
            const prevValue = prev.features[featureName];
            const currValue = curr.features[featureName];

            if (prevValue !== undefined && currValue !== undefined) {
              interpolatedFeatures[featureName] = prevValue + (currValue - prevValue) * ratio;
            } else if (prevValue !== undefined) {
              interpolatedFeatures[featureName] = prevValue;
            } else if (currValue !== undefined) {
              interpolatedFeatures[featureName] = currValue;
            }
          }

          filled.push({
            timestamp: interpolatedTimestamp,
            features: interpolatedFeatures,
            labels: { ...prev.labels, interpolated: 'true' },
          });
        }
      }

      filled.push(curr);
    }

    return filled;
  }
}

export default DataAggregationService;
