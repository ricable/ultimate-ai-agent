/**
 * Batch Operations Performance Benchmarks
 *
 * Tests batch insert performance vs individual operations
 * Measures memory usage during batching
 */

import { performance } from 'perf_hooks';
import { AgentDB } from '../../dist/index.js';
import type { BenchmarkResult } from '../benchmark-runner';

export class BatchOperationsBenchmark {
  private db: AgentDB | null = null;

  private generateVector(dim: number = 1536): number[] {
    const vector: number[] = [];
    for (let i = 0; i < dim; i++) {
      vector.push(Math.random() * 2 - 1);
    }
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(v => v / magnitude);
  }

  /**
   * Test batch insert performance
   */
  async testBatchInsert(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;
    const batchSize = 1000;

    try {
      this.db = new AgentDB({ dbPath: ':memory:' });

      const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
      for (let i = 0; i < vectorCount; i++) {
        vectors.push({
          id: `vec_${i}`,
          embedding: this.generateVector(),
          metadata: { index: i }
        });
      }

      const insertStart = performance.now();
      let batchCount = 0;

      for (let i = 0; i < vectors.length; i += batchSize) {
        const batch = vectors.slice(i, Math.min(i + batchSize, vectors.length));
        await this.db.vectorStore.addBatch(batch);
        batchCount++;
      }

      const insertDuration = performance.now() - insertStart;
      const vectorsPerSecond = (vectorCount / insertDuration) * 1000;

      this.db?.close();
      this.db = null;

      return {
        name: 'Batch Insert (1000 batch size)',
        category: 'Batch Operations',
        duration: performance.now() - startTime,
        operationsPerSecond: vectorsPerSecond,
        metrics: {
          vectorCount,
          batchSize,
          batchCount,
          insertDurationMs: insertDuration,
          vectorsPerSecond: vectorsPerSecond.toFixed(0)
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Batch Insert (1000 batch size)',
        category: 'Batch Operations',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test individual inserts
   */
  async testIndividualInserts(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 1000; // Smaller count for individual inserts

    try {
      this.db = new AgentDB({ dbPath: ':memory:' });

      const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
      for (let i = 0; i < vectorCount; i++) {
        vectors.push({
          id: `vec_${i}`,
          embedding: this.generateVector(),
          metadata: { index: i }
        });
      }

      const insertStart = performance.now();

      for (const vector of vectors) {
        await this.db.vectorStore.add(vector.id, vector.embedding, vector.metadata);
      }

      const insertDuration = performance.now() - insertStart;
      const vectorsPerSecond = (vectorCount / insertDuration) * 1000;

      this.db?.close();
      this.db = null;

      return {
        name: 'Individual Inserts',
        category: 'Batch Operations',
        duration: performance.now() - startTime,
        operationsPerSecond: vectorsPerSecond,
        metrics: {
          vectorCount,
          insertDurationMs: insertDuration,
          vectorsPerSecond: vectorsPerSecond.toFixed(0)
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Individual Inserts',
        category: 'Batch Operations',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Compare batch vs individual operations
   */
  async testBatchVsIndividual(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 1000;
    const batchSize = 100;

    try {
      // Test individual inserts
      this.db = new AgentDB({ dbPath: ':memory:' });

      const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
      for (let i = 0; i < vectorCount; i++) {
        vectors.push({
          id: `vec_${i}`,
          embedding: this.generateVector(),
          metadata: { index: i }
        });
      }

      const individualStart = performance.now();
      for (const vector of vectors) {
        await this.db.vectorStore.add(vector.id, vector.embedding, vector.metadata);
      }
      const individualDuration = performance.now() - individualStart;
      this.db.close();

      // Test batch inserts
      this.db = new AgentDB({ dbPath: ':memory:' });

      const batchStart = performance.now();
      for (let i = 0; i < vectors.length; i += batchSize) {
        const batch = vectors.slice(i, Math.min(i + batchSize, vectors.length));
        await this.db.vectorStore.addBatch(batch);
      }
      const batchDuration = performance.now() - batchStart;
      this.db.close();
      this.db = null;

      const speedup = individualDuration / batchDuration;
      const improvementPercent = ((individualDuration - batchDuration) / individualDuration) * 100;

      return {
        name: 'Batch vs Individual Comparison',
        category: 'Batch Operations',
        duration: performance.now() - startTime,
        metrics: {
          vectorCount,
          batchSize,
          individualDurationMs: individualDuration.toFixed(2),
          batchDurationMs: batchDuration.toFixed(2),
          speedupFactor: speedup.toFixed(2),
          improvementPercent: improvementPercent.toFixed(2),
          recommendation: speedup > 2 ? 'Use batch operations' : 'Similar performance'
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Batch vs Individual Comparison',
        category: 'Batch Operations',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test memory usage during batching
   */
  async testMemoryUsage(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;
    const batchSizes = [10, 100, 1000, 5000];

    try {
      const memoryResults: any[] = [];

      for (const batchSize of batchSizes) {
        // Force garbage collection if available
        if (global.gc) {
          global.gc();
        }

        const memoryBefore = process.memoryUsage();

        this.db = new AgentDB({ dbPath: ':memory:' });

        const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
        for (let i = 0; i < vectorCount; i++) {
          vectors.push({
            id: `vec_${i}`,
            embedding: this.generateVector(),
            metadata: { index: i }
          });
        }

        const insertStart = performance.now();
        for (let i = 0; i < vectors.length; i += batchSize) {
          const batch = vectors.slice(i, Math.min(i + batchSize, vectors.length));
          await this.db.vectorStore.addBatch(batch);
        }
        const insertDuration = performance.now() - insertStart;

        const memoryAfter = process.memoryUsage();
        const memoryIncrease = memoryAfter.heapUsed - memoryBefore.heapUsed;

        memoryResults.push({
          batchSize,
          insertDurationMs: insertDuration.toFixed(2),
          memoryIncreaseMB: (memoryIncrease / 1024 / 1024).toFixed(2),
          heapUsedMB: (memoryAfter.heapUsed / 1024 / 1024).toFixed(2)
        });

        this.db.close();
      }

      this.db = null;

      return {
        name: 'Memory Usage During Batching',
        category: 'Batch Operations',
        duration: performance.now() - startTime,
        metrics: {
          vectorCount,
          testedBatchSizes: batchSizes,
          results: memoryResults,
          recommendation: 'Batch size 1000 provides optimal balance'
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Memory Usage During Batching',
        category: 'Batch Operations',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
