/**
 * HNSW Performance Benchmarks
 *
 * Comprehensive benchmarking of HNSW vs brute-force vector search.
 * Validates the "150x faster" claim with real performance data.
 */

import { performance } from 'perf_hooks';
import { HNSWIndex } from '../../dist/controllers/HNSWIndex.js';
import { WASMVectorSearch } from '../../dist/controllers/WASMVectorSearch.js';
import Database from 'better-sqlite3';
import type { BenchmarkResult } from '../benchmark-runner';

export class HNSWBenchmark {
  private db: Database | null = null;
  private hnswIndex: HNSWIndex | null = null;
  private bruteForce: WASMVectorSearch | null = null;

  /**
   * Generate random embedding vector
   */
  private generateVector(dim: number = 1536): Float32Array {
    const vector: Float32Array = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
      vector[i] = Math.random() * 2 - 1; // Range: -1 to 1
    }
    // Normalize
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(v => v / magnitude) as any as Float32Array;
  }

  /**
   * Insert vectors into database
   */
  private async insertVectors(count: number): Promise<void> {
    this.db = new Database(':memory:');

    // Create table
    this.db.exec(`
      CREATE TABLE pattern_embeddings (
        pattern_id INTEGER PRIMARY KEY,
        embedding BLOB NOT NULL,
        metadata TEXT
      )
    `);

    console.log(`  Inserting ${count} vectors...`);

    const insert = this.db.prepare(`
      INSERT INTO pattern_embeddings (pattern_id, embedding, metadata)
      VALUES (?, ?, ?)
    `);

    const insertMany = this.db.transaction((vectors: any[]) => {
      for (const vec of vectors) {
        insert.run(vec.id, vec.embedding, vec.metadata);
      }
    });

    const vectors: any[] = [];
    for (let i = 0; i < count; i++) {
      const embedding = this.generateVector();
      vectors.push({
        id: i,
        embedding: Buffer.from(embedding.buffer),
        metadata: JSON.stringify({ index: i, category: `cat_${i % 10}` })
      });
    }

    insertMany(vectors);
    console.log(`  ✅ ${count} vectors inserted`);
  }

  /**
   * Test HNSW search with 1K vectors
   */
  async testHNSW1K(): Promise<BenchmarkResult> {
    const start = performance.now();
    const vectorCount = 1000;

    try {
      await this.insertVectors(vectorCount);

      // Build HNSW index
      this.hnswIndex = new HNSWIndex(this.db!, {
        dimension: 1536,
        M: 16,
        efConstruction: 200,
        efSearch: 100,
        metric: 'cosine',
        persistIndex: false
      });

      const buildStart = performance.now();
      await this.hnswIndex.buildIndex();
      const buildTime = performance.now() - buildStart;

      // Perform searches
      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.hnswIndex.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;
      this.hnswIndex = null;

      return {
        name: 'HNSW Search (1K vectors)',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          buildTimeMs: buildTime,
          totalDurationMs: searchDuration
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'HNSW Search (1K vectors)',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test HNSW search with 10K vectors
   */
  async testHNSW10K(): Promise<BenchmarkResult> {
    const start = performance.now();
    const vectorCount = 10000;

    try {
      await this.insertVectors(vectorCount);

      this.hnswIndex = new HNSWIndex(this.db!, {
        dimension: 1536,
        M: 16,
        efConstruction: 200,
        efSearch: 100,
        metric: 'cosine',
        persistIndex: false
      });

      const buildStart = performance.now();
      await this.hnswIndex.buildIndex();
      const buildTime = performance.now() - buildStart;

      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.hnswIndex.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;
      this.hnswIndex = null;

      return {
        name: 'HNSW Search (10K vectors)',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          buildTimeMs: buildTime,
          totalDurationMs: searchDuration
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'HNSW Search (10K vectors)',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test HNSW search with 100K vectors
   */
  async testHNSW100K(): Promise<BenchmarkResult> {
    const start = performance.now();
    const vectorCount = 100000;

    try {
      await this.insertVectors(vectorCount);

      this.hnswIndex = new HNSWIndex(this.db!, {
        dimension: 1536,
        M: 16,
        efConstruction: 200,
        efSearch: 100,
        metric: 'cosine',
        persistIndex: false
      });

      const buildStart = performance.now();
      await this.hnswIndex.buildIndex();
      const buildTime = performance.now() - buildStart;

      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 50; // Fewer searches for large dataset

      for (let i = 0; i < searchCount; i++) {
        await this.hnswIndex.search(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;
      this.hnswIndex = null;

      return {
        name: 'HNSW Search (100K vectors)',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          vectorCount,
          searchCount,
          avgSearchTimeMs: avgSearchTime,
          buildTimeMs: buildTime,
          totalDurationMs: searchDuration
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'HNSW Search (100K vectors)',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Brute-force search with 10K vectors for comparison
   */
  async testBruteForce10K(): Promise<BenchmarkResult> {
    const start = performance.now();
    const vectorCount = 10000;

    try {
      await this.insertVectors(vectorCount);

      this.bruteForce = new WASMVectorSearch(this.db!, {
        enableWASM: false,
        enableSIMD: false,
        batchSize: 100,
        indexThreshold: 999999 // Disable indexing
      });

      const queryVector = this.generateVector();
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await this.bruteForce.findKNN(queryVector, 10);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      this.db?.close();
      this.db = null;
      this.bruteForce = null;

      return {
        name: 'Brute-Force Search (10K vectors)',
        category: 'HNSW Performance',
        duration: performance.now() - start,
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
        name: 'Brute-Force Search (10K vectors)',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Verify 150x faster claim by direct comparison
   */
  async verify150xClaim(): Promise<BenchmarkResult> {
    const start = performance.now();
    const vectorCount = 10000;

    try {
      console.log('\n  🔬 Verifying 150x speedup claim...');

      // Test HNSW
      await this.insertVectors(vectorCount);

      this.hnswIndex = new HNSWIndex(this.db!, {
        dimension: 1536,
        M: 16,
        efConstruction: 200,
        efSearch: 100,
        metric: 'cosine',
        persistIndex: false
      });

      await this.hnswIndex.buildIndex();

      const queryVector = this.generateVector();

      const hnswStart = performance.now();
      for (let i = 0; i < 50; i++) {
        await this.hnswIndex.search(queryVector, 10);
      }
      const hnswDuration = performance.now() - hnswStart;
      const hnswAvgMs = hnswDuration / 50;

      this.hnswIndex.clear();
      this.hnswIndex = null;

      // Test Brute-Force
      this.bruteForce = new WASMVectorSearch(this.db!, {
        enableWASM: false,
        enableSIMD: false,
        batchSize: 100,
        indexThreshold: 999999
      });

      const bruteStart = performance.now();
      for (let i = 0; i < 50; i++) {
        await this.bruteForce.findKNN(queryVector, 10);
      }
      const bruteDuration = performance.now() - bruteStart;
      const bruteAvgMs = bruteDuration / 50;

      this.db?.close();
      this.db = null;
      this.bruteForce = null;

      // Calculate speedup
      const speedup = bruteDuration / hnswDuration;
      const claimVerified = speedup >= 150;
      const percentageOfClaim = (speedup / 150) * 100;

      console.log(`\n  📊 Results:`);
      console.log(`     HNSW:        ${hnswAvgMs.toFixed(2)}ms per search`);
      console.log(`     Brute-Force: ${bruteAvgMs.toFixed(2)}ms per search`);
      console.log(`     Speedup:     ${speedup.toFixed(1)}x`);
      console.log(`     Claim:       ${claimVerified ? '✅ VERIFIED' : `⚠️  ${percentageOfClaim.toFixed(1)}% of claim`}`);

      return {
        name: '150x Faster Claim Verification',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        metrics: {
          hnswAvgSearchMs: hnswAvgMs,
          bruteForceAvgSearchMs: bruteAvgMs,
          speedupFactor: speedup,
          claimVerified,
          percentageOfClaim,
          vectorCount,
          searchesPerformed: 50
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: '150x Faster Claim Verification',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test different efSearch values for quality/speed tradeoff
   */
  async testEfSearchTradeoff(): Promise<BenchmarkResult> {
    const start = performance.now();
    const vectorCount = 10000;

    try {
      await this.insertVectors(vectorCount);

      const efValues = [10, 50, 100, 200, 400];
      const results: any = {};

      for (const ef of efValues) {
        this.hnswIndex = new HNSWIndex(this.db!, {
          dimension: 1536,
          M: 16,
          efConstruction: 200,
          efSearch: ef,
          metric: 'cosine',
          persistIndex: false
        });

        await this.hnswIndex.buildIndex();

        const queryVector = this.generateVector();
        const searchStart = performance.now();

        for (let i = 0; i < 50; i++) {
          await this.hnswIndex.search(queryVector, 10);
        }

        const searchTime = (performance.now() - searchStart) / 50;
        results[`ef${ef}`] = searchTime;

        this.hnswIndex.clear();
        this.hnswIndex = null;
      }

      this.db?.close();
      this.db = null;

      return {
        name: 'efSearch Quality/Speed Tradeoff',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        metrics: {
          ...results,
          vectorCount,
          testedEfValues: efValues
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'efSearch Quality/Speed Tradeoff',
        category: 'HNSW Performance',
        duration: performance.now() - start,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
