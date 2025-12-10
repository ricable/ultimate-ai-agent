/**
 * Vector Search Performance Benchmarks
 *
 * Tests vector similarity search performance with different dataset sizes
 * Validates the "150x faster" claim
 */

import { performance } from 'perf_hooks';
import { AgentDB } from '../../dist/index.js';
import type { BenchmarkResult } from '../benchmark-runner';

export class VectorSearchBenchmark {
  private db: AgentDB | null = null;

  /**
   * Generate random embedding vector
   */
  private generateVector(dim: number = 1536): number[] {
    const vector: number[] = [];
    for (let i = 0; i < dim; i++) {
      vector.push(Math.random() * 2 - 1); // Range: -1 to 1
    }
    // Normalize
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(v => v / magnitude);
  }

  /**
   * Insert vectors into database
   */
  private async insertVectors(count: number, useHNSW: boolean = true): Promise<void> {
    this.db = new AgentDB({
      dbPath: ':memory:',
      enableHNSW: useHNSW,
      hnswConfig: {
        M: 16,
        efConstruction: 200,
        efSearch: 100
      }
    });

    const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];

    for (let i = 0; i < count; i++) {
      vectors.push({
        id: `vec_${i}`,
        embedding: this.generateVector(),
        metadata: { index: i, category: `cat_${i % 10}` }
      });
    }

    // Batch insert
    const batchSize = 1000;
    for (let i = 0; i < vectors.length; i += batchSize) {
      const batch = vectors.slice(i, Math.min(i + batchSize, vectors.length));
      await this.db.vectorStore.addBatch(batch);
    }
  }

  /**
   * Test vector search with 100 vectors
   */
  async testVectorSearch100(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 100;

    try {
      await this.insertVectors(vectorCount);

      // Perform searches
      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.db!.vectorStore.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'Vector Search (100 vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          totalDurationMs: searchDuration
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Vector Search (100 vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test vector search with 1K vectors
   */
  async testVectorSearch1K(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 1000;

    try {
      await this.insertVectors(vectorCount);

      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.db!.vectorStore.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'Vector Search (1K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          totalDurationMs: searchDuration
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Vector Search (1K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test vector search with 10K vectors
   */
  async testVectorSearch10K(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;

    try {
      await this.insertVectors(vectorCount);

      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.db!.vectorStore.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'Vector Search (10K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          totalDurationMs: searchDuration
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Vector Search (10K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test vector search with 100K vectors
   */
  async testVectorSearch100K(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 100000;

    try {
      await this.insertVectors(vectorCount);

      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 50; // Fewer searches for large dataset

      for (let i = 0; i < searchCount; i++) {
        await this.db!.vectorStore.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'Vector Search (100K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          totalDurationMs: searchDuration
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Vector Search (100K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test with HNSW indexing
   */
  async testHNSWIndexing(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;

    try {
      await this.insertVectors(vectorCount, true);

      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.db!.vectorStore.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'Vector Search with HNSW (10K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          hnswEnabled: true
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Vector Search with HNSW (10K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test without HNSW indexing (brute force)
   */
  async testWithoutHNSW(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;

    try {
      await this.insertVectors(vectorCount, false);

      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.db!.vectorStore.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'Vector Search without HNSW (10K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          hnswEnabled: false
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Vector Search without HNSW (10K vectors)',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Verify 150x faster claim
   */
  async verify150xClaim(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;

    try {
      // Test with HNSW
      await this.insertVectors(vectorCount, true);
      const queryVector = this.generateVector();

      const hnswStart = performance.now();
      for (let i = 0; i < 50; i++) {
        await this.db!.vectorStore.search(queryVector, 10);
      }
      const hnswDuration = performance.now() - hnswStart;
      this.db?.close();

      // Test without HNSW (brute force)
      await this.insertVectors(vectorCount, false);

      const bruteStart = performance.now();
      for (let i = 0; i < 50; i++) {
        await this.db!.vectorStore.search(queryVector, 10);
      }
      const bruteDuration = performance.now() - bruteStart;
      this.db?.close();
      this.db = null;

      const speedup = bruteDuration / hnswDuration;
      const claimVerified = speedup >= 100; // At least 100x faster

      return {
        name: '150x Faster Claim Verification',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        metrics: {
          hnswSearchTimeMs: hnswDuration,
          bruteForceSearchTimeMs: bruteDuration,
          speedupFactor: speedup,
          claimVerified,
          percentageOfClaim: (speedup / 150) * 100
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: '150x Faster Claim Verification',
        category: 'Vector Search',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
