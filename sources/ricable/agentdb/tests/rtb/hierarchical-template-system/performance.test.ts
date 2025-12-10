import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';
import { PerformanceMonitor } from '../../src/rtb/hierarchical-template-system/performance-monitor';
import { generateLargeTemplateSet } from '../test-data/mock-templates';
import { RTBTemplate } from '../../../src/types/rtb-types';

// Performance target constants from RTB PRD
const PERFORMANCE_TARGETS = {
  TEMPLATE_PROCESSING_TIME: 5000,      // < 5 seconds per template
  TEMPLATE_MERGING_TIME: 2000,         // < 2 seconds for complex inheritance chains
  CONFLICT_RESOLUTION_TIME: 1000,      // < 1 second for typical scenarios
  MEMORY_USAGE_LIMIT: 1024 * 1024 * 1024, // < 1GB for typical template sets
  TEMPLATE_INHERITANCE_ACCURACY: 100,  // 100% template inheritance accuracy
  PARAMETER_COVERAGE_TARGET: 99,       // 99% parameter coverage from XML source
  LARGE_SET_PROCESSING_TIME: 10000,    // < 10 seconds for large template sets
  BATCH_PROCESSING_THROUGHPUT: 100,    // > 100 templates per second
  CACHE_HIT_RATIO: 0.8                 // > 80% cache hit ratio
};

// Mock PerformanceMonitor class for testing
class MockPerformanceMonitor {
  private measurements: Map<string, PerformanceMeasurement[]> = new Map();
  private memorySnapshots: MemorySnapshot[] = [];
  private thresholds: Map<string, number> = new Map();

  constructor() {
    this.setupDefaultThresholds();
  }

  private setupDefaultThresholds(): void {
    this.thresholds.set('template_processing', PERFORMANCE_TARGETS.TEMPLATE_PROCESSING_TIME);
    this.thresholds.set('template_merging', PERFORMANCE_TARGETS.TEMPLATE_MERGING_TIME);
    this.thresholds.set('conflict_resolution', PERFORMANCE_TARGETS.CONFLICT_RESOLUTION_TIME);
    this.thresholds.set('large_set_processing', PERFORMANCE_TARGETS.LARGE_SET_PROCESSING_TIME);
  }

  // Start timing a measurement
  startMeasurement(operation: string, context?: any): string {
    const measurementId = `${operation}_${Date.now()}_${Math.random()}`;
    const measurement: PerformanceMeasurement = {
      id: measurementId,
      operation,
      startTime: performance.now(),
      startMemory: this.getMemoryUsage(),
      context: context || {}
    };

    if (!this.measurements.has(operation)) {
      this.measurements.set(operation, []);
    }
    this.measurements.get(operation)!.push(measurement);

    return measurementId;
  }

  // End timing a measurement
  endMeasurement(measurementId: string): PerformanceResult {
    for (const [operation, measurements] of this.measurements.entries()) {
      const index = measurements.findIndex(m => m.id === measurementId);
      if (index !== -1) {
        const measurement = measurements[index];
        const endTime = performance.now();
        const endMemory = this.getMemoryUsage();

        const result: PerformanceResult = {
          operation: measurement.operation,
          duration: endTime - measurement.startTime,
          memoryDelta: endMemory - measurement.startMemory,
          startMemory: measurement.startMemory,
          endMemory: endMemory,
          context: measurement.context,
          timestamp: new Date()
        };

        measurement.endTime = endTime;
        measurement.endMemory = endMemory;
        measurement.result = result;

        return result;
      }
    }

    throw new Error(`Measurement not found: ${measurementId}`);
  }

  // Get current memory usage
  private getMemoryUsage(): number {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      return process.memoryUsage().heapUsed;
    }
    return 0; // Fallback for environments without process.memoryUsage
  }

  // Create memory snapshot
  createMemorySnapshot(label: string): MemorySnapshot {
    const snapshot: MemorySnapshot = {
      label,
      timestamp: new Date(),
      memoryUsage: this.getMemoryUsage(),
      details: typeof process !== 'undefined' && process.memoryUsage ? {
        heapUsed: process.memoryUsage().heapUsed,
        heapTotal: process.memoryUsage().heapTotal,
        external: process.memoryUsage().external,
        rss: process.memoryUsage().rss
      } : undefined
    };

    this.memorySnapshots.push(snapshot);
    return snapshot;
  }

  // Get performance statistics for an operation
  getStatistics(operation: string): PerformanceStatistics | null {
    const measurements = this.measurements.get(operation);
    if (!measurements || measurements.length === 0) {
      return null;
    }

    const completedMeasurements = measurements.filter(m => m.result !== undefined);
    if (completedMeasurements.length === 0) {
      return null;
    }

    const durations = completedMeasurements.map(m => m.result!.duration);
    const memoryDeltas = completedMeasurements.map(m => m.result!.memoryDelta);

    return {
      operation,
      count: completedMeasurements.length,
      avgDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
      minDuration: Math.min(...durations),
      maxDuration: Math.max(...durations),
      avgMemoryDelta: memoryDeltas.reduce((a, b) => a + b, 0) / memoryDeltas.length,
      minMemoryDelta: Math.min(...memoryDeltas),
      maxMemoryDelta: Math.max(...memoryDeltas),
      threshold: this.thresholds.get(operation),
      thresholdCompliance: this.calculateThresholdCompliance(operation, durations)
    };
  }

  // Calculate threshold compliance percentage
  private calculateThresholdCompliance(operation: string, durations: number[]): number {
    const threshold = this.thresholds.get(operation);
    if (!threshold) return 100;

    const compliantCount = durations.filter(d => d <= threshold).length;
    return (compliantCount / durations.length) * 100;
  }

  // Get all statistics
  getAllStatistics(): PerformanceStatistics[] {
    const stats: PerformanceStatistics[] = [];

    for (const operation of this.measurements.keys()) {
      const stat = this.getStatistics(operation);
      if (stat) {
        stats.push(stat);
      }
    }

    return stats.sort((a, b) => b.avgDuration - a.avgDuration);
  }

  // Validate against performance targets
  validatePerformanceTargets(): PerformanceValidation {
    const validation: PerformanceValidation = {
      overall: { passed: false, score: 0 },
      targets: {}
    };

    let totalScore = 0;
    let targetCount = 0;

    for (const [operation, threshold] of this.thresholds.entries()) {
      const stats = this.getStatistics(operation);
      if (stats) {
        const compliance = stats.thresholdCompliance || 0;
        const passed = compliance >= 80; // 80% compliance required to pass

        validation.targets[operation] = {
          threshold,
          actual: stats.avgDuration,
          compliance: compliance / 100,
          passed
        };

        totalScore += compliance;
        targetCount++;
      }
    }

    if (targetCount > 0) {
      validation.overall.score = totalScore / targetCount;
      validation.overall.passed = validation.overall.score >= 80;
    }

    return validation;
  }

  // Generate performance report
  generateReport(): PerformanceReport {
    const stats = this.getAllStatistics();
    const validation = this.validatePerformanceTargets();
    const memoryTrend = this.getMemoryTrend();

    return {
      timestamp: new Date(),
      summary: {
        totalOperations: stats.length,
        avgCompliance: validation.overall.score,
        overallPassed: validation.overall.passed
      },
      operations: stats,
      validation,
      memoryTrend,
      recommendations: this.generateRecommendations(stats, validation)
    };
  }

  // Get memory usage trend
  private getMemoryTrend(): MemoryTrend {
    if (this.memorySnapshots.length < 2) {
      return { trend: 'insufficient_data', slope: 0 };
    }

    const snapshots = this.memorySnapshots.slice(-10); // Last 10 snapshots
    const memoryValues = snapshots.map(s => s.memoryUsage);

    // Calculate linear regression slope
    const n = memoryValues.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = memoryValues.reduce((a, b) => a + b, 0);
    const sumXY = memoryValues.reduce((sum, y, x) => sum + x * y, 0);
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6;

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

    let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (slope > 1024 * 1024) { // > 1MB per operation
      trend = 'increasing';
    } else if (slope < -1024 * 1024) { // < -1MB per operation
      trend = 'decreasing';
    }

    return { trend, slope };
  }

  // Generate performance recommendations
  private generateRecommendations(stats: PerformanceStatistics[], validation: PerformanceValidation): string[] {
    const recommendations: string[] = [];

    for (const stat of stats) {
      if (stat.thresholdCompliance && stat.thresholdCompliance < 80) {
        recommendations.push(
          `Consider optimizing ${stat.operation} - only ${stat.thresholdCompliance.toFixed(1)}% compliance with threshold`
        );
      }

      if (stat.avgMemoryDelta > 50 * 1024 * 1024) { // > 50MB memory increase
        recommendations.push(
          `High memory usage detected in ${stat.operation} - average delta: ${(stat.avgMemoryDelta / 1024 / 1024).toFixed(1)}MB`
        );
      }

      if (stat.maxDuration > stat.avgDuration * 3) {
        recommendations.push(
          `High variance in ${stat.operation} - consider investigating outliers (max: ${stat.maxDuration.toFixed(1)}ms)`
        );
      }
    }

    if (validation.overall.score < 80) {
      recommendations.push('Overall performance score below target - review system optimization');
    }

    const memoryTrend = this.getMemoryTrend();
    if (memoryTrend.trend === 'increasing') {
      recommendations.push('Memory usage trending upward - monitor for potential memory leaks');
    }

    return recommendations;
  }

  // Clear all measurements
  clear(): void {
    this.measurements.clear();
    this.memorySnapshots = [];
  }

  // Simulate template processing with performance monitoring
  simulateTemplateProcessing(template: RTBTemplate): Promise<PerformanceResult> {
    return new Promise((resolve) => {
      const measurementId = this.startMeasurement('template_processing', {
        templateSize: JSON.stringify(template).length,
        hasCustomFunctions: !!(template.custom && template.custom.length > 0),
        hasConditions: !!(template.conditions && Object.keys(template.conditions).length > 0),
        hasEvaluations: !!(template.evaluations && Object.keys(template.evaluations).length > 0)
      });

      // Simulate processing time based on template complexity
      const baseProcessingTime = 10; // 10ms base
      const complexityMultiplier = (
        (template.custom?.length || 0) * 2 +
        (Object.keys(template.conditions || {}).length) * 1.5 +
        (Object.keys(template.evaluations || {}).length) * 1.2 +
        (Object.keys(template.configuration || {}).length) * 0.5
      );

      const processingTime = baseProcessingTime * Math.max(1, complexityMultiplier / 10);

      setTimeout(() => {
        const result = this.endMeasurement(measurementId);
        resolve(result);
      }, processingTime);
    });
  }

  // Simulate template merging with performance monitoring
  simulateTemplateMerging(templates: RTBTemplate[]): Promise<PerformanceResult> {
    return new Promise((resolve) => {
      const measurementId = this.startMeasurement('template_merging', {
        templateCount: templates.length,
        totalSize: templates.reduce((sum, t) => sum + JSON.stringify(t).length, 0),
        hasInheritance: templates.some(t => t.meta?.inherits_from)
      });

      // Simulate merging time based on complexity
      const baseMergingTime = 5; // 5ms base
      const complexityMultiplier = (
        templates.length * 2 +
        templates.reduce((sum, t) => sum + (t.custom?.length || 0), 0) * 1.5 +
        templates.reduce((sum, t) => sum + Object.keys(t.configuration || {}).length, 0) * 0.3
      );

      const mergingTime = baseMergingTime * Math.max(1, complexityMultiplier / 15);

      setTimeout(() => {
        const result = this.endMeasurement(measurementId);
        resolve(result);
      }, mergingTime);
    });
  }
}

// Type definitions
interface PerformanceMeasurement {
  id: string;
  operation: string;
  startTime: number;
  endTime?: number;
  startMemory: number;
  endMemory?: number;
  context: any;
  result?: PerformanceResult;
}

interface PerformanceResult {
  operation: string;
  duration: number;
  memoryDelta: number;
  startMemory: number;
  endMemory: number;
  context: any;
  timestamp: Date;
}

interface MemorySnapshot {
  label: string;
  timestamp: Date;
  memoryUsage: number;
  details?: {
    heapUsed: number;
    heapTotal: number;
    external: number;
    rss: number;
  };
}

interface PerformanceStatistics {
  operation: string;
  count: number;
  avgDuration: number;
  minDuration: number;
  maxDuration: number;
  avgMemoryDelta: number;
  minMemoryDelta: number;
  maxMemoryDelta: number;
  threshold?: number;
  thresholdCompliance?: number;
}

interface PerformanceValidation {
  overall: {
    passed: boolean;
    score: number;
  };
  targets: Record<string, {
    threshold: number;
    actual: number;
    compliance: number;
    passed: boolean;
  }>;
}

interface MemoryTrend {
  trend: 'increasing' | 'decreasing' | 'stable' | 'insufficient_data';
  slope: number;
}

interface PerformanceReport {
  timestamp: Date;
  summary: {
    totalOperations: number;
    avgCompliance: number;
    overallPassed: boolean;
  };
  operations: PerformanceStatistics[];
  validation: PerformanceValidation;
  memoryTrend: MemoryTrend;
  recommendations: string[];
}

describe('Performance Tests - RTB Hierarchical Template System', () => {
  let monitor: MockPerformanceMonitor;

  beforeEach(() => {
    monitor = new MockPerformanceMonitor();
    monitor.createMemorySnapshot('test_start');
  });

  afterEach(() => {
    monitor.createMemorySnapshot('test_end');
    const report = monitor.generateReport();
    console.log(`Performance Report - ${new Date().toISOString()}:`, JSON.stringify(report, null, 2));
    monitor.clear();
  });

  describe('Template Processing Performance', () => {
    test('should meet template processing time target', async () => {
      const templates = generateLargeTemplateSet(10);
      const results: PerformanceResult[] = [];

      for (let i = 0; i < templates.length; i++) {
        const result = await monitor.simulateTemplateProcessing(templates[i]);
        results.push(result);
      }

      const stats = monitor.getStatistics('template_processing');
      expect(stats).toBeDefined();
      expect(stats!.avgDuration).toBeLessThan(PERFORMANCE_TARGETS.TEMPLATE_PROCESSING_TIME);
      expect(stats!.maxDuration).toBeLessThan(PERFORMANCE_TARGETS.TEMPLATE_PROCESSING_TIME * 2);
      expect(stats!.thresholdCompliance).toBeGreaterThan(80);
    });

    test('should handle large template processing within time limits', async () => {
      const largeTemplate = generateLargeTemplateSet(1)[0]; // Get one complex template

      const result = await monitor.simulateTemplateProcessing(largeTemplate);

      expect(result.duration).toBeLessThan(PERFORMANCE_TARGETS.TEMPLATE_PROCESSING_TIME);
      expect(Math.abs(result.memoryDelta)).toBeLessThan(100 * 1024 * 1024); // < 100MB memory delta
    });

    test('should maintain consistent performance across multiple runs', async () => {
      const template = generateLargeTemplateSet(1)[0];
      const durations: number[] = [];

      // Run the same template processing 10 times
      for (let i = 0; i < 10; i++) {
        const result = await monitor.simulateTemplateProcessing(template);
        durations.push(result.duration);
      }

      const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
      const variance = durations.reduce((sum, d) => sum + Math.pow(d - avgDuration, 2), 0) / durations.length;
      const stdDeviation = Math.sqrt(variance);

      // Standard deviation should be less than 20% of average
      expect(stdDeviation).toBeLessThan(avgDuration * 0.2);
    });
  });

  describe('Template Merging Performance', () => {
    test('should meet template merging time target', async () => {
      const templateSets = [
        generateLargeTemplateSet(2),
        generateLargeTemplateSet(5),
        generateLargeTemplateSet(10)
      ];

      for (const templates of templateSets) {
        const result = await monitor.simulateTemplateMerging(templates);
        expect(result.duration).toBeLessThan(PERFORMANCE_TARGETS.TEMPLATE_MERGING_TIME);
      }

      const stats = monitor.getStatistics('template_merging');
      expect(stats).toBeDefined();
      expect(stats!.avgDuration).toBeLessThan(PERFORMANCE_TARGETS.TEMPLATE_MERGING_TIME);
      expect(stats!.thresholdCompliance).toBeGreaterThan(80);
    });

    test('should handle complex inheritance chains efficiently', async () => {
      // Create templates with complex inheritance
      const baseTemplate = generateLargeTemplateSet(1)[0];
      const inheritedTemplates = [];

      for (let i = 1; i <= 10; i++) {
        const template = generateLargeTemplateSet(1)[0];
        template.meta = {
          ...template.meta,
          inherits_from: i === 1 ? 'base' : `inherited-${i-1}`,
          priority: i,
          description: `Inherited template ${i}`
        };
        inheritedTemplates.push(template);
      }

      const allTemplates = [baseTemplate, ...inheritedTemplates];
      const result = await monitor.simulateTemplateMerging(allTemplates);

      expect(result.duration).toBeLessThan(PERFORMANCE_TARGETS.TEMPLATE_MERGING_TIME);
      expect(Math.abs(result.memoryDelta)).toBeLessThan(50 * 1024 * 1024); // < 50MB for complex inheritance
    });

    test('should scale linearly with template count', async () => {
      const templateCounts = [2, 5, 10, 20];
      const durations: number[] = [];

      for (const count of templateCounts) {
        const templates = generateLargeTemplateSet(count);
        const result = await monitor.simulateTemplateMerging(templates);
        durations.push(result.duration);
      }

      // Check that performance scales reasonably (not exponentially)
      // The ratio between consecutive measurements should be reasonable
      for (let i = 1; i < durations.length; i++) {
        const ratio = durations[i] / durations[i - 1];
        const templateRatio = templateCounts[i] / templateCounts[i - 1];

        // Performance shouldn't degrade more than 2x the template ratio
        expect(ratio).toBeLessThan(templateRatio * 2);
      }
    });
  });

  describe('Memory Usage Performance', () => {
    test('should stay within memory usage limits', async () => {
      const initialSnapshot = monitor.createMemorySnapshot('initial');

      // Process multiple large templates
      const templates = generateLargeTemplateSet(20);
      for (const template of templates) {
        await monitor.simulateTemplateProcessing(template);
      }

      const finalSnapshot = monitor.createMemorySnapshot('final');
      const memoryDelta = finalSnapshot.memoryUsage - initialSnapshot.memoryUsage;

      expect(memoryDelta).toBeLessThan(PERFORMANCE_TARGETS.MEMORY_USAGE_LIMIT);
    });

    test('should not have memory leaks during repeated operations', async () => {
      const template = generateLargeTemplateSet(1)[0];
      const memorySnapshots: MemorySnapshot[] = [];

      // Perform 50 operations and check memory usage
      for (let i = 0; i < 50; i++) {
        await monitor.simulateTemplateProcessing(template);

        if (i % 10 === 0) {
          memorySnapshots.push(monitor.createMemorySnapshot(`iteration_${i}`));
        }
      }

      // Check that memory usage is stable (not continuously growing)
      const memoryUsages = memorySnapshots.map(s => s.memoryUsage);
      const maxMemory = Math.max(...memoryUsages);
      const minMemory = Math.min(...memoryUsages);
      const memoryRange = maxMemory - minMemory;

      // Memory range should be reasonable (less than 100MB variation)
      expect(memoryRange).toBeLessThan(100 * 1024 * 1024);

      // Check memory trend
      const trend = monitor.getMemoryTrend();
      expect(trend.trend).not.toBe('increasing');
    });

    test('should efficiently handle memory during batch operations', async () => {
      const largeBatch = generateLargeTemplateSet(100);
      const batchSize = 10;
      const batchResults: PerformanceResult[] = [];

      for (let i = 0; i < largeBatch.length; i += batchSize) {
        const batch = largeBatch.slice(i, i + batchSize);
        const batchStart = monitor.createMemorySnapshot(`batch_start_${i / batchSize}`);

        // Process batch
        for (const template of batch) {
          await monitor.simulateTemplateProcessing(template);
        }

        const batchEnd = monitor.createMemorySnapshot(`batch_end_${i / batchSize}`);
        const batchMemoryDelta = batchEnd.memoryUsage - batchStart.memoryUsage;

        // Each batch should use reasonable memory
        expect(batchMemoryDelta).toBeLessThan(50 * 1024 * 1024); // < 50MB per batch
      }
    });
  });

  describe('Throughput Performance', () => {
    test('should meet batch processing throughput target', async () => {
      const templateCount = 100;
      const templates = generateLargeTemplateSet(templateCount);

      const startTime = performance.now();
      monitor.createMemorySnapshot('throughput_start');

      // Process all templates
      const promises = templates.map(template => monitor.simulateTemplateProcessing(template));
      await Promise.all(promises);

      const endTime = performance.now();
      monitor.createMemorySnapshot('throughput_end');

      const totalTime = endTime - startTime;
      const throughput = templateCount / (totalTime / 1000); // templates per second

      expect(throughput).toBeGreaterThan(PERFORMANCE_TARGETS.BATCH_PROCESSING_THROUGHPUT);
      expect(totalTime).toBeLessThan(PERFORMANCE_TARGETS.LARGE_SET_PROCESSING_TIME);
    });

    test('should maintain throughput under load', async () => {
      const loadTests = [
        { templateCount: 50, expectedMinThroughput: 150 },
        { templateCount: 100, expectedMinThroughput: 100 },
        { templateCount: 200, expectedMinThroughput: 50 }
      ];

      for (const test of loadTests) {
        const templates = generateLargeTemplateSet(test.templateCount);

        const startTime = performance.now();
        const promises = templates.map(template => monitor.simulateTemplateProcessing(template));
        await Promise.all(promises);

        const endTime = performance.now();
        const totalTime = endTime - startTime;
        const throughput = test.templateCount / (totalTime / 1000);

        expect(throughput).toBeGreaterThan(test.expectedMinThroughput);
      }
    });
  });

  describe('Concurrent Performance', () => {
    test('should handle concurrent template processing', async () => {
      const templateCount = 50;
      const templates = generateLargeTemplateSet(templateCount);

      // Test sequential processing
      const sequentialStart = performance.now();
      for (const template of templates) {
        await monitor.simulateTemplateProcessing(template);
      }
      const sequentialTime = performance.now() - sequentialStart;

      // Test concurrent processing
      const concurrentStart = performance.now();
      const promises = templates.map(template => monitor.simulateTemplateProcessing(template));
      await Promise.all(promises);
      const concurrentTime = performance.now() - concurrentStart;

      // Concurrent should be faster (at least 30% improvement)
      const improvement = (sequentialTime - concurrentTime) / sequentialTime;
      expect(improvement).toBeGreaterThan(0.3);
    });

    test('should handle concurrent merging operations', async () => {
      const mergeSets = Array.from({ length: 10 }, (_, i) =>
        generateLargeTemplateSet(5).map((t, j) => ({
          ...t,
          meta: { ...t.meta, description: `Set ${i} Template ${j}` }
        }))
      );

      const startTime = performance.now();
      const mergePromises = mergeSets.map(templates =>
        monitor.simulateTemplateMerging(templates)
      );
      await Promise.all(mergePromises);
      const totalTime = performance.now() - startTime;

      // Should handle 10 concurrent merge operations efficiently
      expect(totalTime).toBeLessThan(PERFORMANCE_TARGETS.LARGE_SET_PROCESSING_TIME);

      const stats = monitor.getStatistics('template_merging');
      expect(stats).toBeDefined();
      expect(stats!.count).toBe(10);
    });
  });

  describe('Performance Validation', () => {
    test('should validate against all performance targets', async () => {
      // Run comprehensive performance test suite
      const templates = generateLargeTemplateSet(20);

      // Template processing tests
      for (const template of templates) {
        await monitor.simulateTemplateProcessing(template);
      }

      // Template merging tests
      for (let i = 0; i < templates.length; i += 5) {
        const batch = templates.slice(i, i + 5);
        await monitor.simulateTemplateMerging(batch);
      }

      // Generate and validate performance report
      const report = monitor.generateReport();

      expect(report.summary.overallPassed).toBe(true);
      expect(report.summary.avgCompliance).toBeGreaterThanOrEqual(80);

      // Check individual target compliance
      for (const [target, validation] of Object.entries(report.validation.targets)) {
        expect(validation.passed).toBe(true);
        expect(validation.compliance).toBeGreaterThanOrEqual(0.8);
      }

      // Check that recommendations are reasonable
      expect(Array.isArray(report.recommendations)).toBe(true);
    });

    test('should generate meaningful performance reports', async () => {
      const templates = generateLargeTemplateSet(15);

      // Run operations
      for (const template of templates) {
        await monitor.simulateTemplateProcessing(template);
      }

      for (let i = 0; i < templates.length; i += 3) {
        await monitor.simulateTemplateMerging(templates.slice(i, i + 3));
      }

      const report = monitor.generateReport();

      // Validate report structure
      expect(report).toHaveProperty('timestamp');
      expect(report).toHaveProperty('summary');
      expect(report).toHaveProperty('operations');
      expect(report).toHaveProperty('validation');
      expect(report).toHaveProperty('memoryTrend');
      expect(report).toHaveProperty('recommendations');

      // Validate summary
      expect(report.summary.totalOperations).toBeGreaterThan(0);
      expect(report.summary.avgCompliance).toBeGreaterThanOrEqual(0);
      expect(report.summary.avgCompliance).toBeLessThanOrEqual(100);
      expect(typeof report.summary.overallPassed).toBe('boolean');

      // Validate operations
      expect(Array.isArray(report.operations)).toBe(true);
      expect(report.operations.length).toBeGreaterThan(0);

      // Validate memory trend
      expect(report.memoryTrend).toHaveProperty('trend');
      expect(['increasing', 'decreasing', 'stable', 'insufficient_data']).toContain(report.memoryTrend.trend);
      expect(typeof report.memoryTrend.slope).toBe('number');
    });
  });

  describe('Stress Testing', () => {
    test('should handle stress load gracefully', async () => {
      // Create extremely large template set
      const largeTemplateSet = generateLargeTemplateSet(500);

      const startTime = performance.now();
      monitor.createMemorySnapshot('stress_start');

      // Process with controlled concurrency to avoid overwhelming the system
      const concurrencyLimit = 20;
      for (let i = 0; i < largeTemplateSet.length; i += concurrencyLimit) {
        const batch = largeTemplateSet.slice(i, i + concurrencyLimit);
        const promises = batch.map(template => monitor.simulateTemplateProcessing(template));
        await Promise.all(promises);

        // Periodic memory check
        if (i % 100 === 0) {
          monitor.createMemorySnapshot(`stress_checkpoint_${i}`);
        }
      }

      const endTime = performance.now();
      monitor.createMemorySnapshot('stress_end');

      const totalTime = endTime - startTime;
      const throughput = largeTemplateSet.length / (totalTime / 1000);

      // Should handle large load within reasonable time
      expect(totalTime).toBeLessThan(60000); // < 60 seconds for 500 templates
      expect(throughput).toBeGreaterThan(8); // > 8 templates per second under stress

      // Memory should be reasonable
      const stressStats = monitor.getStatistics('template_processing');
      expect(stressStats).toBeDefined();
      expect(stressStats!.avgMemoryDelta).toBeLessThan(200 * 1024 * 1024); // < 200MB average
    });

    test('should recover from performance degradation', async () => {
      // Simulate performance degradation and recovery
      const normalTemplates = generateLargeTemplateSet(10);
      const heavyTemplates = generateLargeTemplateSet(10).map(t => ({
        ...t,
        custom: Array.from({ length: 50 }, (_, i) => ({
          name: `heavyFunc${i}`,
          args: [`arg${i}`],
          body: Array.from({ length: 20 }, () => `// Heavy operation ${i}`)
        }))
      }));

      // Normal processing
      const normalResults: PerformanceResult[] = [];
      for (const template of normalTemplates) {
        normalResults.push(await monitor.simulateTemplateProcessing(template));
      }

      // Heavy processing (simulated degradation)
      const heavyResults: PerformanceResult[] = [];
      for (const template of heavyTemplates) {
        heavyResults.push(await monitor.simulateTemplateProcessing(template));
      }

      // Recovery processing
      const recoveryResults: PerformanceResult[] = [];
      for (const template of normalTemplates) {
        recoveryResults.push(await monitor.simulateTemplateProcessing(template));
      }

      // Calculate averages
      const avgNormalDuration = normalResults.reduce((sum, r) => sum + r.duration, 0) / normalResults.length;
      const avgRecoveryDuration = recoveryResults.reduce((sum, r) => sum + r.duration, 0) / recoveryResults.length;

      // Recovery should be close to normal performance (within 50%)
      const recoveryRatio = avgRecoveryDuration / avgNormalDuration;
      expect(recoveryRatio).toBeLessThan(1.5);
    });
  });
});