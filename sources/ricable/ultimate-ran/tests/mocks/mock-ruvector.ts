/**
 * Mock Ruvector HNSW Client - London School TDD
 *
 * Mocks @ruvector/core for testing vector similarity search with <10ms latency
 */

import { vi } from 'vitest';
import type { Mock } from 'vitest';
import { cosineSimilarity, generateEmbedding } from '../setup';

export interface MockRuvectorConfig {
  dimension?: number;
  maxElements?: number;
  searchLatencyMs?: number;
  failureRate?: number;
}

export interface HNSWSearchResult {
  id: string;
  score: number;
  distance: number;
  metadata?: any;
}

export class MockRuvector {
  private dimension: number;
  private maxElements: number;
  private searchLatencyMs: number;
  private failureRate: number;
  private vectors: Map<string, { vector: number[]; metadata: any }>;

  // Mock methods
  public insert: Mock;
  public search: Mock;
  public delete: Mock;
  public update: Mock;
  public getStats: Mock;

  constructor(config: MockRuvectorConfig = {}) {
    this.dimension = config.dimension ?? 768;
    this.maxElements = config.maxElements ?? 100000;
    this.searchLatencyMs = config.searchLatencyMs ?? 5;
    this.failureRate = config.failureRate ?? 0;
    this.vectors = new Map();

    // Mock insert
    this.insert = vi.fn(async (id: string, vector: number[], metadata?: any) => {
      if (Math.random() < this.failureRate) {
        throw new Error('Ruvector insert failed');
      }

      if (vector.length !== this.dimension) {
        throw new Error(`Vector dimension mismatch: expected ${this.dimension}, got ${vector.length}`);
      }

      if (this.vectors.size >= this.maxElements) {
        throw new Error('Max elements reached');
      }

      this.vectors.set(id, { vector, metadata });
      return { success: true, id };
    });

    // Mock search with latency validation
    this.search = vi.fn(async (queryVector: number[], k: number = 5, filter?: any) => {
      const startTime = performance.now();

      if (queryVector.length !== this.dimension) {
        throw new Error(`Query vector dimension mismatch: expected ${this.dimension}, got ${queryVector.length}`);
      }

      // Simulate HNSW search latency
      await this._simulateLatency();

      // Calculate cosine similarity for all vectors
      const results: HNSWSearchResult[] = [];

      for (const [id, { vector, metadata }] of this.vectors.entries()) {
        // Apply filter if provided
        if (filter && metadata) {
          let matches = true;
          for (const [key, value] of Object.entries(filter)) {
            if (metadata[key] !== value) {
              matches = false;
              break;
            }
          }
          if (!matches) continue;
        }

        const similarity = cosineSimilarity(queryVector, vector);
        const distance = 1 - similarity; // Convert to distance

        results.push({
          id,
          score: similarity,
          distance,
          metadata,
        });
      }

      // Sort by similarity (descending) and take top k
      results.sort((a, b) => b.score - a.score);
      const topK = results.slice(0, k);

      const latency = performance.now() - startTime;

      return {
        results: topK,
        latency,
        count: topK.length,
      };
    });

    // Mock delete
    this.delete = vi.fn(async (id: string) => {
      const existed = this.vectors.delete(id);
      return { success: existed, id };
    });

    // Mock update
    this.update = vi.fn(async (id: string, vector: number[], metadata?: any) => {
      if (!this.vectors.has(id)) {
        throw new Error(`Vector ${id} not found`);
      }
      this.vectors.set(id, { vector, metadata });
      return { success: true, id };
    });

    // Mock stats
    this.getStats = vi.fn(async () => {
      return {
        dimension: this.dimension,
        maxElements: this.maxElements,
        currentElements: this.vectors.size,
        avgSearchLatency: this.searchLatencyMs,
        p95SearchLatency: this.searchLatencyMs * 1.5,
        p99SearchLatency: this.searchLatencyMs * 2,
      };
    });
  }

  private async _simulateLatency(): Promise<void> {
    if (this.searchLatencyMs > 0) {
      await new Promise(resolve => setTimeout(resolve, this.searchLatencyMs));
    }
  }

  /**
   * Test helper: Add vector directly
   */
  addVector(id: string, vector: number[], metadata?: any): void {
    this.vectors.set(id, { vector, metadata });
  }

  /**
   * Test helper: Clear all vectors
   */
  clear(): void {
    this.vectors.clear();
  }

  /**
   * Test helper: Get vector count
   */
  size(): number {
    return this.vectors.size;
  }

  /**
   * Test helper: Validate latency requirement
   */
  async validateLatency(queryVector: number[], maxLatencyMs: number): Promise<boolean> {
    const result = await this.search(queryVector, 5);
    return result.latency <= maxLatencyMs;
  }
}

/**
 * Factory function for creating mock Ruvector
 */
export function createMockRuvector(config?: MockRuvectorConfig): MockRuvector {
  return new MockRuvector(config);
}

/**
 * Default mock for tests with pre-populated vectors
 */
export const mockRuvector = createMockRuvector({
  dimension: 768,
  searchLatencyMs: 5, // Target <10ms, use 5ms for testing
});

// Pre-populate with some test vectors
for (let i = 0; i < 10; i++) {
  mockRuvector.addVector(
    `vec-${i}`,
    generateEmbedding(i),
    { cellId: `NRCELL_00${i}`, outcome: i % 3 === 0 ? 'SUCCESS' : 'NEUTRAL' }
  );
}
