/**
 * AgentDB v2 Vector Search Benchmarks
 * Tests search performance across different vector counts and k values
 */

import { describe, bench, beforeAll } from 'vitest';
import { runBenchmark, generateRandomVectors, DEFAULT_CONFIG } from './runner.js';

// Mock backend interface for testing
interface VectorBackend {
  insert(id: string, vector: Float32Array): void;
  search(query: Float32Array, k: number): Array<{ id: string; distance: number }>;
  destroy(): void;
}

// RuVector backend implementation (mocked for now)
class RuVectorBackend implements VectorBackend {
  private vectors = new Map<string, Float32Array>();

  insert(id: string, vector: Float32Array): void {
    this.vectors.set(id, vector);
  }

  search(query: Float32Array, k: number): Array<{ id: string; distance: number }> {
    const results: Array<{ id: string; distance: number }> = [];

    for (const [id, vector] of this.vectors.entries()) {
      const distance = this.cosineSimilarity(query, vector);
      results.push({ id, distance });
    }

    results.sort((a, b) => b.distance - a.distance);
    return results.slice(0, k);
  }

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    let magA = 0;
    let magB = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      magA += a[i] * a[i];
      magB += b[i] * b[i];
    }

    return dot / (Math.sqrt(magA) * Math.sqrt(magB));
  }

  destroy(): void {
    this.vectors.clear();
  }
}

const DIMENSION = DEFAULT_CONFIG.dimension;
const VECTOR_COUNTS = DEFAULT_CONFIG.vectorCounts;
const K_VALUES = DEFAULT_CONFIG.kValues;

describe('Vector Search Benchmarks - RuVector', () => {
  for (const count of VECTOR_COUNTS) {
    describe(`${count.toLocaleString()} vectors`, () => {
      let backend: VectorBackend;
      let queryVectors: Float32Array[];

      beforeAll(async () => {
        backend = new RuVectorBackend();

        // Populate index
        console.log(`Populating ${count} vectors...`);
        const vectors = generateRandomVectors(count, DIMENSION);

        for (let i = 0; i < count; i++) {
          backend.insert(`vec-${i}`, vectors[i]);

          if ((i + 1) % 10000 === 0) {
            console.log(`  Inserted ${i + 1}/${count} vectors`);
          }
        }

        // Generate query vectors
        queryVectors = generateRandomVectors(10, DIMENSION);
        console.log(`Setup complete for ${count} vectors`);
      });

      for (const k of K_VALUES) {
        bench(`search k=${k}`, async () => {
          const query = queryVectors[Math.floor(Math.random() * queryVectors.length)];
          backend.search(query, k);
        }, {
          warmupIterations: 10,
          iterations: 100
        });
      }

      bench('insert single', async () => {
        const vector = new Float32Array(DIMENSION);
        for (let i = 0; i < DIMENSION; i++) {
          vector[i] = Math.random() * 2 - 1;
        }
        backend.insert(`new-vec-${Date.now()}`, vector);
      }, {
        warmupIterations: 10,
        iterations: 100
      });
    });
  }
});

describe('Batch Insert Benchmarks - RuVector', () => {
  const BATCH_SIZES = [10, 100, 1000];

  for (const batchSize of BATCH_SIZES) {
    bench(`batch insert ${batchSize} vectors`, async () => {
      const backend = new RuVectorBackend();
      const vectors = generateRandomVectors(batchSize, DIMENSION);

      for (let i = 0; i < batchSize; i++) {
        backend.insert(`batch-vec-${i}`, vectors[i]);
      }

      backend.destroy();
    }, {
      warmupIterations: 5,
      iterations: 50
    });
  }
});

describe('Search Latency Distribution - RuVector', () => {
  const COUNT = 100000;
  let backend: VectorBackend;
  let queryVector: Float32Array;

  beforeAll(async () => {
    backend = new RuVectorBackend();

    console.log('Setting up 100K vectors for latency distribution test...');
    const vectors = generateRandomVectors(COUNT, DIMENSION);

    for (let i = 0; i < COUNT; i++) {
      backend.insert(`vec-${i}`, vectors[i]);

      if ((i + 1) % 10000 === 0) {
        console.log(`  Inserted ${i + 1}/${COUNT} vectors`);
      }
    }

    queryVector = generateRandomVectors(1, DIMENSION)[0];
    console.log('Setup complete for latency distribution test');
  });

  bench('search k=10 (100K vectors)', async () => {
    backend.search(queryVector, 10);
  }, {
    warmupIterations: 20,
    iterations: 1000 // More iterations for better distribution
  });

  bench('search k=50 (100K vectors)', async () => {
    backend.search(queryVector, 50);
  }, {
    warmupIterations: 20,
    iterations: 1000
  });

  bench('search k=100 (100K vectors)', async () => {
    backend.search(queryVector, 100);
  }, {
    warmupIterations: 20,
    iterations: 1000
  });
});

describe('Concurrent Search Benchmarks - RuVector', () => {
  const COUNT = 10000;
  let backend: VectorBackend;
  let queryVectors: Float32Array[];

  beforeAll(async () => {
    backend = new RuVectorBackend();

    const vectors = generateRandomVectors(COUNT, DIMENSION);
    for (let i = 0; i < COUNT; i++) {
      backend.insert(`vec-${i}`, vectors[i]);
    }

    queryVectors = generateRandomVectors(10, DIMENSION);
  });

  bench('concurrent searches (10 queries)', async () => {
    const promises = queryVectors.map(query =>
      Promise.resolve(backend.search(query, 10))
    );
    await Promise.all(promises);
  }, {
    warmupIterations: 5,
    iterations: 50
  });
});

describe('Search Accuracy vs Performance Tradeoff', () => {
  const COUNT = 50000;
  let backend: VectorBackend;
  let queryVector: Float32Array;

  beforeAll(async () => {
    backend = new RuVectorBackend();

    const vectors = generateRandomVectors(COUNT, DIMENSION);
    for (let i = 0; i < COUNT; i++) {
      backend.insert(`vec-${i}`, vectors[i]);
    }

    queryVector = generateRandomVectors(1, DIMENSION)[0];
  });

  // Different k values to measure accuracy/performance tradeoff
  const K_VALUES_EXTENDED = [1, 5, 10, 25, 50, 100, 200];

  for (const k of K_VALUES_EXTENDED) {
    bench(`search k=${k} (${COUNT} vectors)`, async () => {
      backend.search(queryVector, k);
    }, {
      warmupIterations: 10,
      iterations: 100
    });
  }
});
