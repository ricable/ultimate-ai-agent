import { ProcessingStats } from '../../types/rtb-types';

export interface PerformanceMetrics {
  startTime: number;
  endTime: number;
  duration: number;
  memoryUsage: {
    peak: number;
    initial: number;
    final: number;
    delta: number;
  };
  processingStages: ProcessingStageMetrics[];
  errors: ProcessingError[];
  warnings: ProcessingWarning[];
  customMetrics: Map<string, any>;
}

export interface ProcessingStageMetrics {
  stageName: string;
  startTime: number;
  endTime: number;
  duration: number;
  memoryUsage: number;
  processedItems: number;
  itemsPerSecond: number;
  successRate: number;
  errors: string[];
  warnings: string[];
}

export interface ProcessingError {
  timestamp: number;
  stage: string;
  errorType: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  context?: any;
  recovered: boolean;
}

export interface ProcessingWarning {
  timestamp: number;
  stage: string;
  message: string;
  suggestion?: string;
  context?: any;
}

export interface MemoryProfile {
  timestamp: number;
  heapUsed: number;
  heapTotal: number;
  external: number;
  rss: number;
  arrayBuffers: number;
}

export interface PerformanceThreshold {
  metric: string;
  threshold: number;
  operator: '>' | '<' | '=' | '>=' | '<=';
  severity: 'warning' | 'error';
  action: 'log' | 'alert' | 'stop';
}

export interface OptimizationSuggestion {
  type: 'memory' | 'cpu' | 'io' | 'algorithm';
  description: string;
  expectedImprovement: string;
  implementation: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
}

export class PerformanceMonitor {
  private metrics: PerformanceMetrics;
  private stageMetrics: Map<string, ProcessingStageMetrics> = new Map();
  private memoryProfiles: MemoryProfile[] = [];
  private thresholds: PerformanceThreshold[] = [];
  private startMemory: number;
  private isMonitoring: boolean = false;

  constructor() {
    this.metrics = this.initializeMetrics();
    this.startMemory = process.memoryUsage().heapUsed;
    this.initializeThresholds();
  }

  /**
   * Start performance monitoring
   */
  startMonitoring(): void {
    this.isMonitoring = true;
    this.metrics.startTime = Date.now();
    this.metrics.memoryUsage.initial = process.memoryUsage().heapUsed;

    console.log('📊 Performance monitoring started');
    console.log(`🧠 Initial memory usage: ${(this.metrics.memoryUsage.initial / 1024 / 1024).toFixed(2)}MB`);

    // Start memory profiling
    this.startMemoryProfiling();
  }

  /**
   * Stop performance monitoring and return results
   */
  stopMonitoring(): PerformanceMetrics {
    if (!this.isMonitoring) {
      throw new Error('Performance monitoring not started');
    }

    this.isMonitoring = false;
    this.metrics.endTime = Date.now();
    this.metrics.duration = this.metrics.endTime - this.metrics.startTime;
    this.metrics.memoryUsage.final = process.memoryUsage().heapUsed;
    this.metrics.memoryUsage.delta = this.metrics.memoryUsage.final - this.metrics.memoryUsage.initial;
    this.metrics.memoryUsage.peak = Math.max(...this.memoryProfiles.map(p => p.heapUsed));

    // Stop memory profiling
    this.stopMemoryProfiling();

    // Convert stage metrics map to array
    this.metrics.processingStages = Array.from(this.stageMetrics.values());

    console.log('📊 Performance monitoring stopped');
    console.log(`⏱️  Total duration: ${(this.metrics.duration / 1000).toFixed(2)}s`);
    console.log(`🧠 Final memory usage: ${(this.metrics.memoryUsage.final / 1024 / 1024).toFixed(2)}MB`);
    console.log(`📈 Peak memory usage: ${(this.metrics.memoryUsage.peak / 1024 / 1024).toFixed(2)}MB`);
    console.log(`📊 Memory delta: ${(this.metrics.memoryUsage.delta / 1024 / 1024).toFixed(2)}MB`);

    return this.metrics;
  }

  /**
   * Start monitoring a processing stage
   */
  startStage(stageName: string): void {
    if (!this.isMonitoring) {
      throw new Error('Performance monitoring not started');
    }

    const stageMetrics: ProcessingStageMetrics = {
      stageName,
      startTime: Date.now(),
      endTime: 0,
      duration: 0,
      memoryUsage: process.memoryUsage().heapUsed,
      processedItems: 0,
      itemsPerSecond: 0,
      successRate: 1.0,
      errors: [],
      warnings: []
    };

    this.stageMetrics.set(stageName, stageMetrics);
    console.log(`🔄 Starting stage: ${stageName}`);
  }

  /**
   * Stop monitoring a processing stage
   */
  endStage(stageName: string, processedItems: number = 0): void {
    const stageMetrics = this.stageMetrics.get(stageName);
    if (!stageMetrics) {
      throw new Error(`Stage '${stageName}' not started`);
    }

    stageMetrics.endTime = Date.now();
    stageMetrics.duration = stageMetrics.endTime - stageMetrics.startTime;
    stageMetrics.processedItems = processedItems;
    stageMetrics.itemsPerSecond = processedItems > 0 ? (processedItems / (stageMetrics.duration / 1000)) : 0;
    stageMetrics.successRate = stageMetrics.errors.length === 0 ? 1.0 : (processedItems - stageMetrics.errors.length) / processedItems;

    console.log(`✅ Completed stage: ${stageName} (${(stageMetrics.duration / 1000).toFixed(2)}s, ${processedItems} items, ${stageMetrics.itemsPerSecond.toFixed(2)} items/s)`);

    // Check performance thresholds
    this.checkThresholds(stageName, stageMetrics);
  }

  /**
   * Record an error in performance monitoring
   */
  recordError(stage: string, errorType: string, message: string, severity: 'low' | 'medium' | 'high' | 'critical' = 'medium', context?: any, recovered: boolean = false): void {
    const error: ProcessingError = {
      timestamp: Date.now(),
      stage,
      errorType,
      message,
      severity,
      context,
      recovered
    };

    this.metrics.errors.push(error);

    // Add to stage metrics if stage exists
    const stageMetrics = this.stageMetrics.get(stage);
    if (stageMetrics) {
      stageMetrics.errors.push(message);
    }

    console.error(`❌ Error in ${stage} [${errorType}]: ${message}`);
  }

  /**
   * Record a warning in performance monitoring
   */
  recordWarning(stage: string, message: string, suggestion?: string, context?: any): void {
    const warning: ProcessingWarning = {
      timestamp: Date.now(),
      stage,
      message,
      suggestion,
      context
    };

    this.metrics.warnings.push(warning);

    // Add to stage metrics if stage exists
    const stageMetrics = this.stageMetrics.get(stage);
    if (stageMetrics) {
      stageMetrics.warnings.push(message);
    }

    console.warn(`⚠️  Warning in ${stage}: ${message}`);
  }

  /**
   * Record a custom metric
   */
  recordCustomMetric(name: string, value: any): void {
    this.metrics.customMetrics.set(name, value);
  }

  /**
   * Generate performance report
   */
  generateReport(): string {
    const report: string[] = [];
    report.push('# Performance Monitoring Report');
    report.push(`Generated: ${new Date(this.metrics.endTime).toISOString()}`);
    report.push('');

    // Executive Summary
    report.push('## Executive Summary');
    report.push(`- **Total Duration**: ${(this.metrics.duration / 1000).toFixed(2)} seconds`);
    report.push(`- **Peak Memory Usage**: ${(this.metrics.memoryUsage.peak / 1024 / 1024).toFixed(2)} MB`);
    report.push(`- **Memory Delta**: ${(this.metrics.memoryUsage.delta / 1024 / 1024).toFixed(2)} MB`);
    report.push(`- **Total Errors**: ${this.metrics.errors.length}`);
    report.push(`- **Total Warnings**: ${this.metrics.warnings.length}`);
    report.push(`- **Processing Stages**: ${this.metrics.processingStages.length}`);
    report.push('');

    // Stage Performance
    report.push('## Stage Performance');
    for (const stage of this.metrics.processingStages) {
      report.push(`### ${stage.stageName}`);
      report.push(`- **Duration**: ${(stage.duration / 1000).toFixed(2)} seconds`);
      report.push(`- **Items Processed**: ${stage.processedItems.toLocaleString()}`);
      report.push(`- **Throughput**: ${stage.itemsPerSecond.toFixed(2)} items/second`);
      report.push(`- **Success Rate**: ${(stage.successRate * 100).toFixed(1)}%`);
      report.push(`- **Memory Usage**: ${(stage.memoryUsage / 1024 / 1024).toFixed(2)} MB`);

      if (stage.errors.length > 0) {
        report.push(`- **Errors**: ${stage.errors.length}`);
        for (const error of stage.errors) {
          report.push(`  - ${error}`);
        }
      }

      if (stage.warnings.length > 0) {
        report.push(`- **Warnings**: ${stage.warnings.length}`);
        for (const warning of stage.warnings) {
          report.push(`  - ${warning}`);
        }
      }
      report.push('');
    }

    // Error Analysis
    if (this.metrics.errors.length > 0) {
      report.push('## Error Analysis');
      const errorsByType = new Map<string, number>();
      const errorsBySeverity = new Map<string, number>();

      for (const error of this.metrics.errors) {
        errorsByType.set(error.errorType, (errorsByType.get(error.errorType) || 0) + 1);
        errorsBySeverity.set(error.severity, (errorsBySeverity.get(error.severity) || 0) + 1);
      }

      report.push('### Errors by Type');
      for (const [type, count] of errorsByType) {
        report.push(`- **${type}**: ${count}`);
      }
      report.push('');

      report.push('### Errors by Severity');
      for (const [severity, count] of errorsBySeverity) {
        report.push(`- **${severity}**: ${count}`);
      }
      report.push('');
    }

    // Memory Analysis
    report.push('## Memory Analysis');
    report.push(`- **Initial Memory**: ${(this.metrics.memoryUsage.initial / 1024 / 1024).toFixed(2)} MB`);
    report.push(`- **Final Memory**: ${(this.metrics.memoryUsage.final / 1024 / 1024).toFixed(2)} MB`);
    report.push(`- **Peak Memory**: ${(this.metrics.memoryUsage.peak / 1024 / 1024).toFixed(2)} MB`);
    report.push(`- **Memory Growth**: ${(this.metrics.memoryUsage.delta / 1024 / 1024).toFixed(2)} MB`);
    report.push('');

    if (this.memoryProfiles.length > 1) {
      report.push('### Memory Trend');
      const startMem = this.memoryProfiles[0].heapUsed;
      const endMem = this.memoryProfiles[this.memoryProfiles.length - 1].heapUsed;
      const trend = endMem > startMem ? 'Increasing' : endMem < startMem ? 'Decreasing' : 'Stable';
      report.push(`- **Memory Trend**: ${trend}`);
      report.push(`- **Average Memory**: ${(this.memoryProfiles.reduce((sum, p) => sum + p.heapUsed, 0) / this.memoryProfiles.length / 1024 / 1024).toFixed(2)} MB`);
      report.push('');
    }

    // Optimization Suggestions
    const suggestions = this.generateOptimizationSuggestions();
    if (suggestions.length > 0) {
      report.push('## Optimization Suggestions');
      for (const suggestion of suggestions) {
        report.push(`### ${suggestion.type.toUpperCase()} Optimization (Priority: ${suggestion.priority})`);
        report.push(`**Description**: ${suggestion.description}`);
        report.push(`**Expected Improvement**: ${suggestion.expectedImprovement}`);
        report.push(`**Implementation**: ${suggestion.implementation}`);
        report.push('');
      }
    }

    // Custom Metrics
    if (this.metrics.customMetrics.size > 0) {
      report.push('## Custom Metrics');
      for (const [name, value] of this.metrics.customMetrics) {
        report.push(`- **${name}**: ${JSON.stringify(value)}`);
      }
      report.push('');
    }

    return report.join('\n');
  }

  /**
   * Generate JSON export of metrics
   */
  generateJSON(): string {
    return JSON.stringify(this.metrics, null, 2);
  }

  /**
   * Generate optimization suggestions
   */
  generateOptimizationSuggestions(): OptimizationSuggestion[] {
    const suggestions: OptimizationSuggestion[] = [];

    // Memory optimization suggestions
    const memoryGrowthMB = this.metrics.memoryUsage.delta / 1024 / 1024;
    if (memoryGrowthMB > 500) {
      suggestions.push({
        type: 'memory',
        description: 'High memory growth detected during processing',
        expectedImprovement: 'Reduce memory usage by 30-50%',
        implementation: 'Implement streaming processing for large files, increase batch size for garbage collection',
        priority: 'high'
      });
    }

    // CPU optimization suggestions
    for (const stage of this.metrics.processingStages) {
      if (stage.itemsPerSecond < 100 && stage.processedItems > 1000) {
        suggestions.push({
          type: 'cpu',
          description: `Low throughput in ${stage.stageName} stage`,
          expectedImprovement: 'Improve processing speed by 2-3x',
          implementation: 'Optimize algorithms, use parallel processing, consider worker threads',
          priority: 'medium'
        });
      }
    }

    // I/O optimization suggestions
    const slowStages = this.metrics.processingStages.filter(s => s.duration > 30000); // > 30 seconds
    if (slowStages.length > 0) {
      suggestions.push({
        type: 'io',
        description: 'Slow processing stages detected',
        expectedImprovement: 'Reduce processing time by 40-60%',
        implementation: 'Use streaming I/O, implement caching, optimize file reading/writing',
        priority: 'high'
      });
    }

    // Error rate optimization
    const totalErrors = this.metrics.errors.length;
    const totalItems = this.metrics.processingStages.reduce((sum, s) => sum + s.processedItems, 0);
    const errorRate = totalItems > 0 ? (totalErrors / totalItems) * 100 : 0;

    if (errorRate > 5) {
      suggestions.push({
        type: 'algorithm',
        description: 'High error rate during processing',
        expectedImprovement: 'Reduce error rate by 80-90%',
        implementation: 'Improve input validation, add better error handling, implement retry mechanisms',
        priority: 'critical'
      });
    }

    return suggestions;
  }

  /**
   * Get real-time status
   */
  getStatus(): {
    isMonitoring: boolean;
    currentStage?: string;
    elapsedTime: number;
    currentMemory: number;
    errorCount: number;
    warningCount: number;
  } {
    const currentStage = Array.from(this.stageMetrics.values()).find(s => s.endTime === 0);

    return {
      isMonitoring: this.isMonitoring,
      currentStage: currentStage?.stageName,
      elapsedTime: this.isMonitoring ? Date.now() - this.metrics.startTime : this.metrics.duration,
      currentMemory: process.memoryUsage().heapUsed,
      errorCount: this.metrics.errors.length,
      warningCount: this.metrics.warnings.length
    };
  }

  /**
   * Initialize metrics object
   */
  private initializeMetrics(): PerformanceMetrics {
    return {
      startTime: 0,
      endTime: 0,
      duration: 0,
      memoryUsage: {
        peak: 0,
        initial: 0,
        final: 0,
        delta: 0
      },
      processingStages: [],
      errors: [],
      warnings: [],
      customMetrics: new Map()
    };
  }

  /**
   * Initialize performance thresholds
   */
  private initializeThresholds(): void {
    this.thresholds = [
      {
        metric: 'memory_growth',
        threshold: 1024, // 1GB
        operator: '>',
        severity: 'warning',
        action: 'log'
      },
      {
        metric: 'memory_growth',
        threshold: 2048, // 2GB
        operator: '>',
        severity: 'error',
        action: 'alert'
      },
      {
        metric: 'error_rate',
        threshold: 10, // 10%
        operator: '>',
        severity: 'warning',
        action: 'log'
      },
      {
        metric: 'error_rate',
        threshold: 25, // 25%
        operator: '>',
        severity: 'error',
        action: 'stop'
      }
    ];
  }

  /**
   * Start memory profiling
   */
  private startMemoryProfiling(): void {
    // Profile memory every 5 seconds
    const profileInterval = setInterval(() => {
      if (!this.isMonitoring) {
        clearInterval(profileInterval);
        return;
      }

      const memUsage = process.memoryUsage();
      const profile: MemoryProfile = {
        timestamp: Date.now(),
        heapUsed: memUsage.heapUsed,
        heapTotal: memUsage.heapTotal,
        external: memUsage.external,
        rss: memUsage.rss,
        arrayBuffers: (memUsage as any).arrayBuffers || 0
      };

      this.memoryProfiles.push(profile);

      // Keep only last 100 profiles
      if (this.memoryProfiles.length > 100) {
        this.memoryProfiles = this.memoryProfiles.slice(-100);
      }
    }, 5000);
  }

  /**
   * Stop memory profiling
   */
  private stopMemoryProfiling(): void {
    // Profiling stops automatically when isMonitoring is set to false
  }

  /**
   * Check performance thresholds
   */
  private checkThresholds(stageName: string, stageMetrics: ProcessingStageMetrics): void {
    for (const threshold of this.thresholds) {
      let value: number;
      let meetsThreshold = false;

      switch (threshold.metric) {
        case 'memory_growth':
          value = (process.memoryUsage().heapUsed - this.startMemory) / 1024 / 1024; // MB
          meetsThreshold = this.evaluateThreshold(value, threshold);
          break;

        case 'error_rate':
          value = stageMetrics.processedItems > 0 ? (stageMetrics.errors.length / stageMetrics.processedItems) * 100 : 0;
          meetsThreshold = this.evaluateThreshold(value, threshold);
          break;

        default:
          continue;
      }

      if (meetsThreshold) {
        const message = `Performance threshold exceeded in ${stageName}: ${threshold.metric} ${threshold.operator} ${threshold.threshold} (actual: ${value.toFixed(2)})`;

        if (threshold.severity === 'error') {
          this.recordError(stageName, 'threshold_exceeded', message, 'high');
        } else {
          this.recordWarning(stageName, message, `Consider performance optimization`);
        }

        if (threshold.action === 'stop') {
          throw new Error(`Performance threshold exceeded: ${message}`);
        }
      }
    }
  }

  /**
   * Evaluate threshold condition
   */
  private evaluateThreshold(value: number, threshold: PerformanceThreshold): boolean {
    switch (threshold.operator) {
      case '>': return value > threshold.threshold;
      case '<': return value < threshold.threshold;
      case '>=': return value >= threshold.threshold;
      case '<=': return value <= threshold.threshold;
      case '=': return value === threshold.threshold;
      default: return false;
    }
  }
}

/**
 * Global performance monitor instance
 */
export const performanceMonitor = new PerformanceMonitor();