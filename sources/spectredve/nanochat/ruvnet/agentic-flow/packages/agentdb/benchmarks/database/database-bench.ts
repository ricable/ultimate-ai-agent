/**
 * Database Backend Performance Benchmarks
 *
 * Compares better-sqlite3 vs sql.js performance
 * Tests initialization time and query performance
 */

import { performance } from 'perf_hooks';
import { AgentDB } from '../../dist/index.js';
import type { BenchmarkResult } from '../benchmark-runner';

export class DatabaseBackendBenchmark {
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
   * Test better-sqlite3 performance
   */
  async testBetterSqlite3(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 5000;

    try {
      // Initialize database
      const initStart = performance.now();
      this.db = new AgentDB({
        dbPath: ':memory:',
        useSqlJs: false // Force better-sqlite3
      });
      const initDuration = performance.now() - initStart;

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

      // Test queries
      const queryVector = this.generateVector();
      const queryStart = performance.now();
      const queryCount = 100;

      for (let i = 0; i < queryCount; i++) {
        await this.db.vectorStore.search(queryVector, 10);
      }

      const queryDuration = performance.now() - queryStart;
      const avgQueryTime = queryDuration / queryCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'better-sqlite3 Backend',
        category: 'Database Backend',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgQueryTime,
        metrics: {
          backend: 'better-sqlite3',
          initDurationMs: initDuration.toFixed(2),
          insertDurationMs: insertDuration.toFixed(2),
          avgQueryTimeMs: avgQueryTime.toFixed(2),
          vectorCount,
          queryCount
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'better-sqlite3 Backend',
        category: 'Database Backend',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test sql.js performance
   */
  async testSqlJs(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 5000;

    try {
      // Initialize database with sql.js
      const initStart = performance.now();
      this.db = new AgentDB({
        dbPath: ':memory:',
        useSqlJs: true // Force sql.js
      });
      const initDuration = performance.now() - initStart;

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

      // Test queries
      const queryVector = this.generateVector();
      const queryStart = performance.now();
      const queryCount = 100;

      for (let i = 0; i < queryCount; i++) {
        await this.db.vectorStore.search(queryVector, 10);
      }

      const queryDuration = performance.now() - queryStart;
      const avgQueryTime = queryDuration / queryCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'sql.js Backend',
        category: 'Database Backend',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgQueryTime,
        metrics: {
          backend: 'sql.js',
          initDurationMs: initDuration.toFixed(2),
          insertDurationMs: insertDuration.toFixed(2),
          avgQueryTimeMs: avgQueryTime.toFixed(2),
          vectorCount,
          queryCount
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'sql.js Backend',
        category: 'Database Backend',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Compare database backends
   */
  async compareBackends(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const vectorCount = 5000;

    try {
      // Test better-sqlite3
      let sqlite3InitTime = 0;
      let sqlite3InsertTime = 0;
      let sqlite3QueryTime = 0;

      try {
        const initStart = performance.now();
        this.db = new AgentDB({ dbPath: ':memory:', useSqlJs: false });
        sqlite3InitTime = performance.now() - initStart;

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
        sqlite3InsertTime = performance.now() - insertStart;

        const queryVector = this.generateVector();
        const queryStart = performance.now();
        for (let i = 0; i < 50; i++) {
          await this.db.vectorStore.search(queryVector, 10);
        }
        sqlite3QueryTime = performance.now() - queryStart;

        this.db.close();
      } catch (error) {
        console.warn('better-sqlite3 not available, using fallback');
      }

      // Test sql.js
      const sqlJsInitStart = performance.now();
      this.db = new AgentDB({ dbPath: ':memory:', useSqlJs: true });
      const sqlJsInitTime = performance.now() - sqlJsInitStart;

      const vectors: Array<{ id: string; embedding: number[]; metadata: any }> = [];
      for (let i = 0; i < vectorCount; i++) {
        vectors.push({
          id: `vec_${i}`,
          embedding: this.generateVector(),
          metadata: { index: i }
        });
      }

      const sqlJsInsertStart = performance.now();
      await this.db.vectorStore.addBatch(vectors);
      const sqlJsInsertTime = performance.now() - sqlJsInsertStart;

      const queryVector = this.generateVector();
      const sqlJsQueryStart = performance.now();
      for (let i = 0; i < 50; i++) {
        await this.db.vectorStore.search(queryVector, 10);
      }
      const sqlJsQueryTime = performance.now() - sqlJsQueryStart;

      this.db.close();
      this.db = null;

      const comparison: any = {
        vectorCount,
        sqlJs: {
          initTimeMs: sqlJsInitTime.toFixed(2),
          insertTimeMs: sqlJsInsertTime.toFixed(2),
          queryTimeMs: sqlJsQueryTime.toFixed(2)
        }
      };

      if (sqlite3InitTime > 0) {
        comparison.betterSqlite3 = {
          initTimeMs: sqlite3InitTime.toFixed(2),
          insertTimeMs: sqlite3InsertTime.toFixed(2),
          queryTimeMs: sqlite3QueryTime.toFixed(2)
        };

        comparison.speedup = {
          init: (sqlJsInitTime / sqlite3InitTime).toFixed(2),
          insert: (sqlJsInsertTime / sqlite3InsertTime).toFixed(2),
          query: (sqlJsQueryTime / sqlite3QueryTime).toFixed(2)
        };

        comparison.recommendation = sqlite3QueryTime < sqlJsQueryTime
          ? 'better-sqlite3 is faster (recommended for Node.js)'
          : 'sql.js is comparable (use for browser compatibility)';
      } else {
        comparison.recommendation = 'sql.js (better-sqlite3 not available)';
      }

      return {
        name: 'Backend Comparison',
        category: 'Database Backend',
        duration: performance.now() - startTime,
        metrics: comparison,
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Backend Comparison',
        category: 'Database Backend',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test initialization time
   */
  async testInitializationTime(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const iterations = 10;

    try {
      const sqlite3Times: number[] = [];
      const sqlJsTimes: number[] = [];

      // Test better-sqlite3 initialization
      for (let i = 0; i < iterations; i++) {
        try {
          const initStart = performance.now();
          this.db = new AgentDB({ dbPath: ':memory:', useSqlJs: false });
          sqlite3Times.push(performance.now() - initStart);
          this.db.close();
        } catch (error) {
          // Skip if not available
          break;
        }
      }

      // Test sql.js initialization
      for (let i = 0; i < iterations; i++) {
        const initStart = performance.now();
        this.db = new AgentDB({ dbPath: ':memory:', useSqlJs: true });
        sqlJsTimes.push(performance.now() - initStart);
        this.db.close();
      }

      this.db = null;

      const avgSqlite3 = sqlite3Times.length > 0
        ? sqlite3Times.reduce((sum, t) => sum + t, 0) / sqlite3Times.length
        : 0;
      const avgSqlJs = sqlJsTimes.reduce((sum, t) => sum + t, 0) / sqlJsTimes.length;

      return {
        name: 'Initialization Time Comparison',
        category: 'Database Backend',
        duration: performance.now() - startTime,
        metrics: {
          iterations,
          betterSqlite3AvgMs: avgSqlite3 > 0 ? avgSqlite3.toFixed(2) : 'N/A',
          sqlJsAvgMs: avgSqlJs.toFixed(2),
          speedup: avgSqlite3 > 0 ? (avgSqlJs / avgSqlite3).toFixed(2) : 'N/A',
          recommendation: avgSqlite3 > 0 && avgSqlite3 < avgSqlJs
            ? 'better-sqlite3 initializes faster'
            : 'Similar initialization performance'
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Initialization Time Comparison',
        category: 'Database Backend',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
