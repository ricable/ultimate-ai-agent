/**
 * Quantization Performance Benchmarks
 *
 * Tests 4-bit and 8-bit quantization performance
 * Measures memory reduction and accuracy tradeoffs
 */

import { performance } from 'perf_hooks';
import { AgentDB } from '../../dist/index.js';
import type { BenchmarkResult } from '../benchmark-runner';

export class QuantizationBenchmark {
  private db: AgentDB | null = null;

  private generateVector(dim: number = 1536): number[] {
    const vector: number[] = [];
    for (let i = 0; i < dim; i++) {
      vector.push(Math.random() * 2 - 1);
    }
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(v => v / magnitude);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let magA = 0;
    let magB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(magA) * Math.sqrt(magB));
  }

  /**
   * Test 4-bit quantization
   */
  async test4BitQuantization(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;

    try {
      this.db = new AgentDB({
        dbPath: ':memory:',
        quantization: {
          enabled: true,
          bits: 4
        }
      });

      // Insert vectors
      const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
      for (let i = 0; i < vectorCount; i++) {
        vectors.push({
          id: `vec_${i}`,
          embedding: this.generateVector(),
          metadata: { index: i }
        });
      }

      const insertStart = performance.now();
      await this.db.vectorStore.addBatch(vectors);
      const insertDuration = performance.now() - insertStart;

      // Test search performance
      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.db.vectorStore.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      // Get memory usage estimate
      const memoryUsed = process.memoryUsage().heapUsed;

      this.db?.close();
      this.db = null;

      return {
        name: '4-bit Quantization',
        category: 'Quantization',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        memoryUsed,
        metrics: {
          vectorCount,
          insertDurationMs: insertDuration,
          avgSearchTimeMs: avgSearchTime,
          quantizationBits: 4,
          theoreticalMemoryReduction: '8x'
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: '4-bit Quantization',
        category: 'Quantization',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test 8-bit quantization
   */
  async test8BitQuantization(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;

    try {
      this.db = new AgentDB({
        dbPath: ':memory:',
        quantization: {
          enabled: true,
          bits: 8
        }
      });

      // Insert vectors
      const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
      for (let i = 0; i < vectorCount; i++) {
        vectors.push({
          id: `vec_${i}`,
          embedding: this.generateVector(),
          metadata: { index: i }
        });
      }

      const insertStart = performance.now();
      await this.db.vectorStore.addBatch(vectors);
      const insertDuration = performance.now() - insertStart;

      // Test search performance
      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.db.vectorStore.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      // Get memory usage estimate
      const memoryUsed = process.memoryUsage().heapUsed;

      this.db?.close();
      this.db = null;

      return {
        name: '8-bit Quantization',
        category: 'Quantization',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        memoryUsed,
        metrics: {
          vectorCount,
          insertDurationMs: insertDuration,
          avgSearchTimeMs: avgSearchTime,
          quantizationBits: 8,
          theoreticalMemoryReduction: '4x'
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: '8-bit Quantization',
        category: 'Quantization',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test memory reduction
   */
  async testMemoryReduction(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 10000;

    try {
      // Test without quantization
      this.db = new AgentDB({
        dbPath: ':memory:',
        quantization: { enabled: false }
      });

      const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
      for (let i = 0; i < vectorCount; i++) {
        vectors.push({
          id: `vec_${i}`,
          embedding: this.generateVector(),
          metadata: { index: i }
        });
      }

      await this.db.vectorStore.addBatch(vectors);
      const memoryWithoutQuantization = process.memoryUsage().heapUsed;
      this.db.close();

      // Test with 4-bit quantization
      this.db = new AgentDB({
        dbPath: ':memory:',
        quantization: { enabled: true, bits: 4 }
      });

      await this.db.vectorStore.addBatch(vectors);
      const memoryWith4Bit = process.memoryUsage().heapUsed;
      this.db.close();

      // Test with 8-bit quantization
      this.db = new AgentDB({
        dbPath: ':memory:',
        quantization: { enabled: true, bits: 8 }
      });

      await this.db.vectorStore.addBatch(vectors);
      const memoryWith8Bit = process.memoryUsage().heapUsed;
      this.db.close();
      this.db = null;

      const reduction4Bit = ((memoryWithoutQuantization - memoryWith4Bit) / memoryWithoutQuantization) * 100;
      const reduction8Bit = ((memoryWithoutQuantization - memoryWith8Bit) / memoryWithoutQuantization) * 100;

      return {
        name: 'Memory Reduction Comparison',
        category: 'Quantization',
        duration: performance.now() - startTime,
        metrics: {
          vectorCount,
          memoryWithoutQuantizationMB: (memoryWithoutQuantization / 1024 / 1024).toFixed(2),
          memoryWith4BitMB: (memoryWith4Bit / 1024 / 1024).toFixed(2),
          memoryWith8BitMB: (memoryWith8Bit / 1024 / 1024).toFixed(2),
          reduction4BitPercent: reduction4Bit.toFixed(2),
          reduction8BitPercent: reduction8Bit.toFixed(2)
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Memory Reduction Comparison',
        category: 'Quantization',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test accuracy vs size tradeoff
   */
  async testAccuracyTradeoff(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 1000;
    const testQueries = 50;

    try {
      // Generate test data
      const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
      const queryVectors: number[][] = [];

      for (let i = 0; i < vectorCount; i++) {
        vectors.push({
          id: `vec_${i}`,
          embedding: this.generateVector(),
          metadata: { index: i }
        });
      }

      for (let i = 0; i < testQueries; i++) {
        queryVectors.push(this.generateVector());
      }

      // Test without quantization (baseline)
      this.db = new AgentDB({
        dbPath: ':memory:',
        quantization: { enabled: false }
      });
      await this.db.vectorStore.addBatch(vectors);

      const baselineResults: any[] = [];
      for (const query of queryVectors) {
        const results = await this.db.vectorStore.search(query, 10);
        baselineResults.push(results);
      }
      this.db.close();

      // Test with 4-bit quantization
      this.db = new AgentDB({
        dbPath: ':memory:',
        quantization: { enabled: true, bits: 4 }
      });
      await this.db.vectorStore.addBatch(vectors);

      const quant4Results: any[] = [];
      for (const query of queryVectors) {
        const results = await this.db.vectorStore.search(query, 10);
        quant4Results.push(results);
      }
      this.db.close();

      // Test with 8-bit quantization
      this.db = new AgentDB({
        dbPath: ':memory:',
        quantization: { enabled: true, bits: 8 }
      });
      await this.db.vectorStore.addBatch(vectors);

      const quant8Results: any[] = [];
      for (const query of queryVectors) {
        const results = await this.db.vectorStore.search(query, 10);
        quant8Results.push(results);
      }
      this.db.close();
      this.db = null;

      // Calculate accuracy (percentage of top-10 matches that are the same)
      let accuracy4Bit = 0;
      let accuracy8Bit = 0;

      for (let i = 0; i < testQueries; i++) {
        const baselineIds = new Set(baselineResults[i].map((r: any) => r.id));
        const quant4Ids = new Set(quant4Results[i].map((r: any) => r.id));
        const quant8Ids = new Set(quant8Results[i].map((r: any) => r.id));

        // Count overlapping results
        let overlap4 = 0;
        let overlap8 = 0;

        for (const id of quant4Ids) {
          if (baselineIds.has(id)) overlap4++;
        }
        for (const id of quant8Ids) {
          if (baselineIds.has(id)) overlap8++;
        }

        accuracy4Bit += overlap4 / 10;
        accuracy8Bit += overlap8 / 10;
      }

      accuracy4Bit = (accuracy4Bit / testQueries) * 100;
      accuracy8Bit = (accuracy8Bit / testQueries) * 100;

      return {
        name: 'Accuracy vs Size Tradeoff',
        category: 'Quantization',
        duration: performance.now() - startTime,
        metrics: {
          vectorCount,
          testQueries,
          accuracy4BitPercent: accuracy4Bit.toFixed(2),
          accuracy8BitPercent: accuracy8Bit.toFixed(2),
          memoryReduction4Bit: '8x theoretical',
          memoryReduction8Bit: '4x theoretical'
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Accuracy vs Size Tradeoff',
        category: 'Quantization',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
