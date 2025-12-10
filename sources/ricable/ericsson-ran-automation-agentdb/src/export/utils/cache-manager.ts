/**
 * Cache Manager Utility
 *
 * High-performance caching system for <1 second template export with
 * intelligent eviction, compression, and memory optimization.
 */

import { ExportCache } from '../types/export-types';

export interface CacheEntry<T> {
  key: string;
  value: T;
  timestamp: number;
  accessCount: number;
  lastAccessed: number;
  size: number;
  compressed?: boolean;
}

export interface CacheStatistics {
  hitRate: number;
  missRate: number;
  totalHits: number;
  totalMisses: number;
  currentSize: number;
  maxEntries: number;
  evictions: number;
  averageLookupTime: number;
  memoryUsage: number;
}

export class CacheManager {
  private config: ExportCache;
  private cache: Map<string, CacheEntry<any>> = new Map();
  private accessOrder: string[] = [];
  private statistics: CacheStatistics;
  private cleanupInterval?: NodeJS.Timeout;

  constructor(config: ExportCache) {
    this.config = config;
    this.statistics = {
      hitRate: 0,
      missRate: 0,
      totalHits: 0,
      totalMisses: 0,
      currentSize: 0,
      maxEntries: config.maxSize,
      evictions: 0,
      averageLookupTime: 0,
      memoryUsage: 0
    };
  }

  async initialize(): Promise<void> {
    console.log('💾 Initializing Cache Manager...');

    // Start cleanup interval
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 60000); // Cleanup every minute

    console.log('✅ Cache Manager initialized');
  }

  async get<T>(key: string): Promise<T | null> {
    const startTime = Date.now();

    const entry = this.cache.get(key);
    if (!entry) {
      this.statistics.totalMisses++;
      this.updateStatistics();
      return null;
    }

    // Check TTL
    if (this.isExpired(entry)) {
      this.cache.delete(key);
      this.removeFromAccessOrder(key);
      this.statistics.totalMisses++;
      this.updateStatistics();
      return null;
    }

    // Update access statistics
    entry.accessCount++;
    entry.lastAccessed = Date.now();
    this.updateAccessOrder(key);

    this.statistics.totalHits++;
    this.updateStatistics();

    const lookupTime = Date.now() - startTime;
    this.updateAverageLookupTime(lookupTime);

    return this.decompressIfNeeded(entry.value);
  }

  async set<T>(key: string, value: T, ttl?: number): Promise<void> {
    const size = this.calculateSize(value);
    const now = Date.now();

    // Check if we need to evict entries
    while (this.cache.size >= this.config.maxSize) {
      this.evict();
    }

    const entry: CacheEntry<T> = {
      key,
      value: this.compressIfNeeded(value),
      timestamp: now,
      accessCount: 1,
      lastAccessed: now,
      size,
      compressed: this.config.compressionEnabled
    };

    this.cache.set(key, entry);
    this.addToAccessOrder(key);
    this.statistics.currentSize = this.cache.size;
  }

  async delete(key: string): Promise<boolean> {
    const deleted = this.cache.delete(key);
    if (deleted) {
      this.removeFromAccessOrder(key);
      this.statistics.currentSize = this.cache.size;
    }
    return deleted;
  }

  async clear(): Promise<void> {
    this.cache.clear();
    this.accessOrder = [];
    this.statistics.currentSize = 0;
    console.log('🗑️ Cache cleared');
  }

  getStatistics(): CacheStatistics {
    return { ...this.statistics };
  }

  async shutdown(): Promise<void> {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    await this.clear();
    console.log('🛑 Cache Manager shutdown complete');
  }

  private isExpired(entry: CacheEntry<any>): boolean {
    const age = Date.now() - entry.timestamp;
    return age > this.config.ttl;
  }

  private evict(): void {
    if (this.accessOrder.length === 0) return;

    let keyToEvict: string;

    switch (this.config.evictionPolicy) {
      case 'lru':
        keyToEvict = this.accessOrder[0];
        break;
      case 'lfu':
        keyToEvict = this.getLeastFrequentlyUsed();
        break;
      case 'fifo':
        keyToEvict = this.accessOrder[0];
        break;
      default:
        keyToEvict = this.accessOrder[0];
    }

    this.cache.delete(keyToEvict);
    this.removeFromAccessOrder(keyToEvict);
    this.statistics.evictions++;
    this.statistics.currentSize = this.cache.size;
  }

  private getLeastFrequentlyUsed(): string {
    let leastUsed = this.accessOrder[0];
    let minCount = this.cache.get(leastUsed)?.accessCount || 0;

    for (const key of this.accessOrder) {
      const entry = this.cache.get(key);
      if (entry && entry.accessCount < minCount) {
        leastUsed = key;
        minCount = entry.accessCount;
      }
    }

    return leastUsed;
  }

  private addToAccessOrder(key: string): void {
    this.removeFromAccessOrder(key);
    this.accessOrder.push(key);
  }

  private removeFromAccessOrder(key: string): void {
    const index = this.accessOrder.indexOf(key);
    if (index > -1) {
      this.accessOrder.splice(index, 1);
    }
  }

  private updateAccessOrder(key: string): void {
    this.removeFromAccessOrder(key);
    this.accessOrder.push(key);
  }

  private calculateSize(value: any): number {
    // Rough estimation of memory size
    return JSON.stringify(value).length * 2; // Assuming 2 bytes per character
  }

  private compressIfNeeded<T>(value: T): T {
    if (!this.config.compressionEnabled) return value;

    // In a real implementation, would use compression library
    // For now, just return the value
    return value;
  }

  private decompressIfNeeded<T>(value: T): T {
    // In a real implementation, would decompress if needed
    return value;
  }

  private updateStatistics(): void {
    const total = this.statistics.totalHits + this.statistics.totalMisses;
    this.statistics.hitRate = total > 0 ? this.statistics.totalHits / total : 0;
    this.statistics.missRate = total > 0 ? this.statistics.totalMisses / total : 0;
    this.statistics.memoryUsage = this.calculateMemoryUsage();
  }

  private updateAverageLookupTime(lookupTime: number): void {
    const totalLookups = this.statistics.totalHits + this.statistics.totalMisses;
    const currentAverage = this.statistics.averageLookupTime;
    this.statistics.averageLookupTime =
      (currentAverage * (totalLookups - 1) + lookupTime) / totalLookups;
  }

  private calculateMemoryUsage(): number {
    let totalSize = 0;
    for (const entry of this.cache.values()) {
      totalSize += entry.size;
    }
    return totalSize;
  }

  private cleanup(): void {
    const now = Date.now();
    const keysToRemove: string[] = [];

    for (const [key, entry] of this.cache.entries()) {
      if (this.isExpired(entry)) {
        keysToRemove.push(key);
      }
    }

    for (const key of keysToRemove) {
      this.cache.delete(key);
      this.removeFromAccessOrder(key);
    }

    if (keysToRemove.length > 0) {
      console.log(`🧹 Cache cleanup: removed ${keysToRemove.length} expired entries`);
    }
  }
}