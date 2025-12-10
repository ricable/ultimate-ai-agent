/**
 * Performance Optimizer - Advanced Performance Optimization for Large Template Sets
 *
 * Provides caching, indexing, lazy loading, and batch processing optimizations
 * for handling large numbers of RTB templates efficiently.
 */

import {
  RTBTemplate,
  TemplateMeta
} from '../../types/rtb-types';
import {
  TemplatePriorityInfo,
  TemplateInheritanceChain,
  TemplateResolutionContext
} from './priority-engine';
import { TemplateRegistry } from './template-registry';

/**
 * Performance optimization strategy
 */
export enum OptimizationStrategy {
  LAZY_LOADING = 'lazy_loading',
  BATCH_PROCESSING = 'batch_processing',
  CACHING = 'caching',
  INDEXING = 'indexing',
  PARALLEL_PROCESSING = 'parallel_processing',
  MEMORY_OPTIMIZATION = 'memory_optimization',
  STREAMING = 'streaming'
}

/**
 * Cache configuration
 */
export interface CacheConfig {
  enabled: boolean;
  maxSize: number;
  ttl: number; // Time to live in milliseconds
  strategy: 'lru' | 'lfu' | 'fifo' | 'custom';
  compressionEnabled: boolean;
  compressionThreshold: number;
  persistenceEnabled: boolean;
  persistencePath?: string;
}

/**
 * Batch processing configuration
 */
export interface BatchConfig {
  enabled: boolean;
  batchSize: number;
  maxConcurrency: number;
  timeout: number;
  retryAttempts: number;
  backoffStrategy: 'linear' | 'exponential' | 'fixed';
  progressCallback?: (processed: number, total: number) => void;
}

/**
 * Index configuration
 */
export interface IndexConfig {
  enabled: boolean;
  indexTypes: string[];
  refreshInterval: number;
  indexingStrategy: 'full' | 'incremental' | 'on_demand';
  memoryLimit: number;
  diskCache: boolean;
  diskCachePath?: string;
}

/**
 * Performance metrics
 */
export interface PerformanceMetrics {
  operationType: string;
  startTime: Date;
  endTime: Date;
  duration: number;
  memoryUsage: {
    before: number;
    after: number;
    peak: number;
  };
  cacheHits: number;
  cacheMisses: number;
  templatesProcessed: number;
  errors: number;
  optimizations: string[];
}

/**
 * Cache entry
 */
export interface CacheEntry<T> {
  key: string;
  value: T;
  timestamp: Date;
  ttl: number;
  size: number;
  accessCount: number;
  lastAccessed: Date;
  compressed?: boolean;
  checksum?: string;
}

/**
 * Batch operation result
 */
export interface BatchOperationResult<T> {
  results: T[];
  errors: Array<{
    index: number;
    error: Error;
    item?: any;
  }>;
  metrics: PerformanceMetrics;
  successCount: number;
  failureCount: number;
  totalProcessed: number;
}

/**
 * Performance optimization options
 */
export interface OptimizationOptions {
  enableCaching?: boolean;
  enableBatching?: boolean;
  enableIndexing?: boolean;
  enableParallelProcessing?: boolean;
  enableMemoryOptimization?: boolean;
  cacheConfig?: Partial<CacheConfig>;
  batchConfig?: Partial<BatchConfig>;
  indexConfig?: Partial<IndexConfig>;
}

/**
 * Memory pool for reusable objects
 */
export interface MemoryPool<T> {
  acquire(): T;
  release(item: T): void;
  size(): number;
  clear(): void;
}

/**
 * Performance Optimizer
 *
 * Advanced performance optimization system for handling large template sets
 * with caching, indexing, batch processing, and memory optimization.
 */
export class PerformanceOptimizer {
  private registry: TemplateRegistry;
  private cache = new Map<string, CacheEntry<any>>();
  private indexes = new Map<string, Map<string, Set<string>>>();
  private memoryPools = new Map<string, MemoryPool<any>>();
  private config: Required<OptimizationOptions>;
  private metrics: PerformanceMetrics[] = [];
  private activeOperations = new Map<string, AbortController>();

  constructor(registry: TemplateRegistry, options: OptimizationOptions = {}) {
    this.registry = registry;
    this.config = {
      enableCaching: options.enableCaching ?? true,
      enableBatching: options.enableBatching ?? true,
      enableIndexing: options.enableIndexing ?? true,
      enableParallelProcessing: options.enableParallelProcessing ?? true,
      enableMemoryOptimization: options.enableMemoryOptimization ?? true,
      cacheConfig: {
        enabled: true,
        maxSize: 10000,
        ttl: 3600000, // 1 hour
        strategy: 'lru',
        compressionEnabled: true,
        compressionThreshold: 1024,
        persistenceEnabled: false
      },
      batchConfig: {
        enabled: true,
        batchSize: 50,
        maxConcurrency: 4,
        timeout: 30000,
        retryAttempts: 3,
        backoffStrategy: 'exponential'
      },
      indexConfig: {
        enabled: true,
        indexTypes: ['category', 'priority', 'environment', 'tags', 'parameters'],
        refreshInterval: 300000, // 5 minutes
        indexingStrategy: 'incremental',
        memoryLimit: 100 * 1024 * 1024, // 100MB
        diskCache: false
      }
    };

    // Merge user-provided options
    if (options.cacheConfig) {
      this.config.cacheConfig = { ...this.config.cacheConfig, ...options.cacheConfig };
    }
    if (options.batchConfig) {
      this.config.batchConfig = { ...this.config.batchConfig, ...options.batchConfig };
    }
    if (options.indexConfig) {
      this.config.indexConfig = { ...this.config.indexConfig, ...options.indexConfig };
    }

    this.initializeOptimizations();
  }

  /**
   * Resolve template with performance optimizations
   */
  async resolveTemplateOptimized(
    templateName: string,
    context: TemplateResolutionContext = {}
  ): Promise<TemplateInheritanceChain> {
    const operationId = `resolve:${templateName}:${Date.now()}`;
    const metrics = this.startMetrics(operationId, 'template_resolution');

    try {
      // Check cache first
      if (this.config.enableCaching) {
        const cached = this.getFromCache(this.generateCacheKey('template', templateName, context));
        if (cached) {
          metrics.cacheHits++;
          return cached;
        }
        metrics.cacheMisses++;
      }

      // Resolve template (this would use the inheritance resolver)
      const result = await this.resolveTemplateInternal(templateName, context);

      // Cache result
      if (this.config.enableCaching) {
        this.setCache(this.generateCacheKey('template', templateName, context), result);
      }

      metrics.templatesProcessed = 1;
      metrics.optimizations.push(this.getAppliedOptimizations());

      return result;
    } finally {
      this.endMetrics(operationId);
    }
  }

  /**
   * Batch resolve multiple templates
   */
  async batchResolveTemplates(
    templateNames: string[],
    context: TemplateResolutionContext = {}
  ): Promise<BatchOperationResult<TemplateInheritanceChain>> {
    const operationId = `batch_resolve:${Date.now()}`;
    const metrics = this.startMetrics(operationId, 'batch_template_resolution');

    const results: TemplateInheritanceChain[] = [];
    const errors: Array<{ index: number; error: Error; item?: string }> = [];

    try {
      if (!this.config.enableBatching) {
        // Fallback to sequential processing
        for (let i = 0; i < templateNames.length; i++) {
          try {
            const result = await this.resolveTemplateOptimized(templateNames[i], context);
            results.push(result);
          } catch (error) {
            errors.push({ index: i, error: error as Error, item: templateNames[i] });
          }
        }
      } else {
        // Batch processing with parallel execution
        const batches = this.createBatches(templateNames, this.config.batchConfig.batchSize);

        for (const batch of batches) {
          const batchResults = await this.processBatch(
            batch,
            (templateName) => this.resolveTemplateOptimized(templateName, context),
            this.config.batchConfig
          );

          results.push(...batchResults.successes);
          errors.push(...batchResults.errors);
        }
      }

      const result: BatchOperationResult<TemplateInheritanceChain> = {
        results,
        errors,
        metrics,
        successCount: results.length,
        failureCount: errors.length,
        totalProcessed: templateNames.length
      };

      metrics.templatesProcessed = templateNames.length;

      return result;
    } finally {
      this.endMetrics(operationId);
    }
  }

  /**
   * Search templates with optimized indexing
   */
  async searchTemplatesOptimized(filter: any): Promise<any> {
    const operationId = `search:${Date.now()}`;
    const metrics = this.startMetrics(operationId, 'template_search');

    try {
      // Check if we have optimized indexes
      if (this.config.enableIndexing && this.hasIndexesForFilter(filter)) {
        return this.searchWithIndexes(filter);
      }

      // Fallback to registry search
      const result = await this.registry.searchTemplates(filter);
      metrics.templatesProcessed = result.totalCount;

      return result;
    } finally {
      this.endMetrics(operationId);
    }
  }

  /**
   * Build or refresh performance indexes
   */
  async buildIndexes(): Promise<void> {
    if (!this.config.enableIndexing) return;

    const operationId = `build_indexes:${Date.now()}`;
    const metrics = this.startMetrics(operationId, 'index_building');

    try {
      // Clear existing indexes
      this.indexes.clear();

      // Get all templates from registry
      const templates = this.registry.getRegisteredTemplates();

      // Build indexes for each configured type
      for (const indexType of this.config.indexConfig.indexTypes) {
        const index = new Map<string, Set<string>>();

        for (const [templateName, priority] of templates) {
          const keys = this.extractIndexKeys(templateName, priority, indexType);
          for (const key of keys) {
            if (!index.has(key)) {
              index.set(key, new Set());
            }
            index.get(key)!.add(templateName);
          }
        }

        this.indexes.set(indexType, index);
      }

      metrics.optimizations.push('indexes_built');
    } finally {
      this.endMetrics(operationId);
    }
  }

  /**
   * Preload templates into cache
   */
  async preloadTemplates(templateNames: string[]): Promise<void> {
    if (!this.config.enableCaching) return;

    const operationId = `preload:${Date.now()}`;
    const metrics = this.startMetrics(operationId, 'template_preloading');

    try {
      const context = {}; // Default context for preloading

      // Batch preload for efficiency
      const batches = this.createBatches(templateNames, this.config.batchConfig.batchSize);

      for (const batch of batches) {
        await Promise.all(
          batch.map(name => this.resolveTemplateOptimized(name, context))
        );
      }

      metrics.optimizations.push('templates_preloaded');
      metrics.templatesProcessed = templateNames.length;
    } finally {
      this.endMetrics(operationId);
    }
  }

  /**
   * Optimize memory usage
   */
  optimizeMemory(): {
    freedMemory: number;
    optimizations: string[];
  } {
    const freedMemory: number[] = [];
    const optimizations: string[] = [];

    if (this.config.enableMemoryOptimization) {
      // Clean expired cache entries
      const beforeSize = this.getCacheSize();
      this.cleanExpiredCache();
      const afterSize = this.getCacheSize();
      freedMemory.push(beforeSize - afterSize);

      if (beforeSize > afterSize) {
        optimizations.push('expired_cache_cleaned');
      }

      // LRU cache eviction if necessary
      if (this.cache.size > this.config.cacheConfig.maxSize) {
        this.evictLRUCache();
        const finalSize = this.getCacheSize();
        freedMemory.push(afterSize - finalSize);
        optimizations.push('lru_eviction');
      }

      // Clear memory pools
      for (const [name, pool] of this.memoryPools) {
        const poolSize = pool.size();
        pool.clear();
        freedMemory.push(poolSize);
        optimizations.push(`memory_pool_cleared:${name}`);
      }
    }

    return {
      freedMemory: freedMemory.reduce((sum, size) => sum + size, 0),
      optimizations
    };
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): {
    recent: PerformanceMetrics[];
    summary: {
      totalOperations: number;
      averageDuration: number;
      cacheHitRate: number;
      memoryUsage: number;
      errorRate: number;
    };
    cacheStats: {
      size: number;
      hitRate: number;
      memoryUsage: number;
    };
    indexStats: {
      indexCount: number;
      totalEntries: number;
      memoryUsage: number;
    };
  } {
    const recent = this.metrics.slice(-100); // Last 100 operations

    const summary = {
      totalOperations: recent.length,
      averageDuration: recent.length > 0 ?
        recent.reduce((sum, m) => sum + m.duration, 0) / recent.length : 0,
      cacheHitRate: this.calculateCacheHitRate(recent),
      memoryUsage: this.getCurrentMemoryUsage(),
      errorRate: this.calculateErrorRate(recent)
    };

    const cacheStats = {
      size: this.cache.size,
      hitRate: this.calculateCacheHitRate(recent),
      memoryUsage: this.getCacheSize()
    };

    const indexStats = {
      indexCount: this.indexes.size,
      totalEntries: this.getTotalIndexEntries(),
      memoryUsage: this.getIndexSize()
    };

    return {
      recent,
      summary,
      cacheStats,
      indexStats
    };
  }

  /**
   * Clear all performance optimizations
   */
  clearOptimizations(): void {
    this.cache.clear();
    this.indexes.clear();
    this.memoryPools.clear();
    this.metrics = [];
  }

  // Private helper methods

  private initializeOptimizations(): void {
    if (this.config.enableIndexing) {
      // Build indexes in background
      setTimeout(() => this.buildIndexes(), 0);
    }

    if (this.config.enableMemoryOptimization) {
      // Setup periodic memory optimization
      setInterval(() => this.optimizeMemory(), 300000); // Every 5 minutes
    }
  }

  private async resolveTemplateInternal(
    templateName: string,
    context: TemplateResolutionContext
  ): Promise<TemplateInheritanceChain> {
    // This would integrate with the inheritance resolver
    // For now, return a placeholder implementation
    const template = this.registry.getTemplate(templateName);
    if (!template) {
      throw new Error(`Template '${templateName}' not found`);
    }

    return {
      templateName,
      chain: [],
      resolvedTemplate: template,
      conflicts: [],
      warnings: []
    };
  }

  private createBatches<T>(items: T[], batchSize: number): T[][] {
    const batches: T[][] = [];
    for (let i = 0; i < items.length; i += batchSize) {
      batches.push(items.slice(i, i + batchSize));
    }
    return batches;
  }

  private async processBatch<T, R>(
    batch: T[],
    processor: (item: T) => Promise<R>,
    config: Required<BatchConfig>
  ): Promise<{
    successes: R[];
    errors: Array<{ index: number; error: Error; item?: T }>;
  }> {
    if (!this.config.enableParallelProcessing) {
      // Sequential processing
      const successes: R[] = [];
      const errors: Array<{ index: number; error: Error; item?: T }> = [];

      for (let i = 0; i < batch.length; i++) {
        try {
          const result = await processor(batch[i]);
          successes.push(result);
        } catch (error) {
          errors.push({ index: i, error: error as Error, item: batch[i] });
        }
      }

      return { successes, errors };
    }

    // Parallel processing with concurrency limit
    const semaphore = new Semaphore(config.maxConcurrency);
    const promises = batch.map(async (item, index) => {
      await semaphore.acquire();
      try {
        const result = await processor(item);
        return { success: true, result, index };
      } catch (error) {
        return { success: false, error: error as Error, index, item };
      } finally {
        semaphore.release();
      }
    });

    const results = await Promise.all(promises);

    const successes = results
      .filter(r => r.success)
      .map(r => (r as any).result);

    const errors = results
      .filter(r => !r.success)
      .map(r => ({
        index: r.index,
        error: r.error,
        item: (r as any).item
      }));

    return { successes, errors };
  }

  private generateCacheKey(type: string, templateName: string, context: any): string {
    const contextHash = JSON.stringify(context);
    return `${type}:${templateName}:${Buffer.from(contextHash).toString('base64')}`;
  }

  private getFromCache<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Check TTL
    if (Date.now() - entry.timestamp.getTime() > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    // Update access statistics
    entry.accessCount++;
    entry.lastAccessed = new Date();

    return entry.value;
  }

  private setCache<T>(key: string, value: T): void {
    const size = this.calculateSize(value);
    const entry: CacheEntry<T> = {
      key,
      value,
      timestamp: new Date(),
      ttl: this.config.cacheConfig.ttl,
      size,
      accessCount: 1,
      lastAccessed: new Date()
    };

    // Check cache size limit
    if (this.cache.size >= this.config.cacheConfig.maxSize) {
      this.evictLRUCache();
    }

    this.cache.set(key, entry);
  }

  private cleanExpiredCache(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache) {
      if (now - entry.timestamp.getTime() > entry.ttl) {
        this.cache.delete(key);
      }
    }
  }

  private evictLRUCache(): void {
    if (this.cache.size === 0) return;

    // Find least recently used entry
    let lruKey: string | null = null;
    let lruTime = Date.now();

    for (const [key, entry] of this.cache) {
      if (entry.lastAccessed.getTime() < lruTime) {
        lruTime = entry.lastAccessed.getTime();
        lruKey = key;
      }
    }

    if (lruKey) {
      this.cache.delete(lruKey);
    }
  }

  private extractIndexKeys(templateName: string, priority: TemplatePriorityInfo, indexType: string): string[] {
    switch (indexType) {
      case 'category':
        return [priority.category];
      case 'priority':
        return [priority.level.toString()];
      case 'environment':
        return [priority.metadata?.environment || 'unknown'];
      case 'tags':
        return priority.metadata?.tags || [];
      case 'parameters':
        const template = this.registry.getTemplate(templateName);
        return template ? Object.keys(template.configuration || {}) : [];
      default:
        return [];
    }
  }

  private hasIndexesForFilter(filter: any): boolean {
    // Check if we have relevant indexes for the filter
    return this.indexes.size > 0;
  }

  private searchWithIndexes(filter: any): any {
    // Simplified index-based search
    // In a full implementation, this would use the indexes to efficiently filter results
    return {
      templates: [],
      totalCount: 0,
      facets: {},
      searchTime: 0
    };
  }

  private startMetrics(operationId: string, operationType: string): PerformanceMetrics {
    const metrics: PerformanceMetrics = {
      operationType,
      startTime: new Date(),
      endTime: new Date(),
      duration: 0,
      memoryUsage: {
        before: this.getCurrentMemoryUsage(),
        after: 0,
        peak: 0
      },
      cacheHits: 0,
      cacheMisses: 0,
      templatesProcessed: 0,
      errors: 0,
      optimizations: []
    };

    this.metrics.push(metrics);
    this.activeOperations.set(operationId, new AbortController());

    return metrics;
  }

  private endMetrics(operationId: string): void {
    const metrics = this.metrics[this.metrics.length - 1];
    if (metrics) {
      metrics.endTime = new Date();
      metrics.duration = metrics.endTime.getTime() - metrics.startTime.getTime();
      metrics.memoryUsage.after = this.getCurrentMemoryUsage();
    }

    this.activeOperations.delete(operationId);
  }

  private getAppliedOptimizations(): string {
    const optimizations: string[] = [];

    if (this.config.enableCaching) optimizations.push('caching');
    if (this.config.enableBatching) optimizations.push('batching');
    if (this.config.enableIndexing) optimizations.push('indexing');
    if (this.config.enableParallelProcessing) optimizations.push('parallel_processing');
    if (this.config.enableMemoryOptimization) optimizations.push('memory_optimization');

    return optimizations.join(',');
  }

  private calculateSize(obj: any): number {
    return JSON.stringify(obj).length;
  }

  private getCacheSize(): number {
    let size = 0;
    for (const entry of this.cache.values()) {
      size += entry.size;
    }
    return size;
  }

  private getIndexSize(): number {
    let size = 0;
    for (const index of this.indexes.values()) {
      for (const [key, values] of index) {
        size += key.length * 2 + values.size * 8; // Rough estimate
      }
    }
    return size;
  }

  private getTotalIndexEntries(): number {
    let total = 0;
    for (const index of this.indexes.values()) {
      for (const values of index.values()) {
        total += values.size;
      }
    }
    return total;
  }

  private getCurrentMemoryUsage(): number {
    // This would use actual memory monitoring in a real implementation
    return Math.floor(Math.random() * 10000000); // Placeholder
  }

  private calculateCacheHitRate(metrics: PerformanceMetrics[]): number {
    const totalRequests = metrics.reduce((sum, m) => sum + m.cacheHits + m.cacheMisses, 0);
    const totalHits = metrics.reduce((sum, m) => sum + m.cacheHits, 0);
    return totalRequests > 0 ? totalHits / totalRequests : 0;
  }

  private calculateErrorRate(metrics: PerformanceMetrics[]): number {
    const totalOperations = metrics.length;
    const totalErrors = metrics.reduce((sum, m) => sum + m.errors, 0);
    return totalOperations > 0 ? totalErrors / totalOperations : 0;
  }
}

/**
 * Simple semaphore implementation for concurrency control
 */
class Semaphore {
  private permits: number;
  private waitQueue: (() => void)[] = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }

    return new Promise<void>(resolve => {
      this.waitQueue.push(resolve);
    });
  }

  release(): void {
    this.permits++;
    if (this.waitQueue.length > 0) {
      const resolve = this.waitQueue.shift()!;
      this.permits--;
      resolve();
    }
  }
}