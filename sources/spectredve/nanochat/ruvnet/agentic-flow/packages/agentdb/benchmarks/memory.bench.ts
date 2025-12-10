/**
 * AgentDB v2 Memory Usage Benchmarks
 * Measures memory consumption for different vector counts and operations
 */

import { describe, test, beforeEach } from 'vitest';
import { measureMemory, generateRandomVectors, DEFAULT_CONFIG } from './runner.js';

// Mock backend for memory testing
class RuVectorBackend {
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

  getSize(): number {
    return this.vectors.size;
  }

  destroy(): void {
    this.vectors.clear();
  }
}

const DIMENSION = DEFAULT_CONFIG.dimension;

describe('Memory Usage Benchmarks - RuVector', () => {
  beforeEach(() => {
    // Force garbage collection before each test
    if (global.gc) {
      global.gc();
    }
  });

  test('memory usage - 1K vectors', async () => {
    const memory = await measureMemory(async () => {
      const backend = new RuVectorBackend();
      const vectors = generateRandomVectors(1000, DIMENSION);

      for (let i = 0; i < 1000; i++) {
        backend.insert(`vec-${i}`, vectors[i]);
      }
    });

    console.log('Memory (1K vectors):', {
      peakMB: memory.peakMB.toFixed(2),
      finalMB: memory.finalMB.toFixed(2),
      heapUsedMB: memory.heapUsedMB.toFixed(2),
      externalMB: memory.externalMB.toFixed(2)
    });

    // Expected: < 5MB for 1K vectors
    console.log('Expected: < 5MB, Actual:', memory.peakMB.toFixed(2), 'MB');
  });

  test('memory usage - 10K vectors', async () => {
    const memory = await measureMemory(async () => {
      const backend = new RuVectorBackend();
      const vectors = generateRandomVectors(10000, DIMENSION);

      for (let i = 0; i < 10000; i++) {
        backend.insert(`vec-${i}`, vectors[i]);
      }
    });

    console.log('Memory (10K vectors):', {
      peakMB: memory.peakMB.toFixed(2),
      finalMB: memory.finalMB.toFixed(2),
      heapUsedMB: memory.heapUsedMB.toFixed(2),
      externalMB: memory.externalMB.toFixed(2)
    });

    // Expected: < 20MB for 10K vectors
    console.log('Expected: < 20MB, Actual:', memory.peakMB.toFixed(2), 'MB');
  });

  test('memory usage - 100K vectors', async () => {
    const memory = await measureMemory(async () => {
      const backend = new RuVectorBackend();
      const vectors = generateRandomVectors(100000, DIMENSION);

      for (let i = 0; i < 100000; i++) {
        backend.insert(`vec-${i}`, vectors[i]);

        if ((i + 1) % 10000 === 0) {
          console.log(`  Inserted ${i + 1}/100000 vectors`);
        }
      }
    });

    console.log('Memory (100K vectors):', {
      peakMB: memory.peakMB.toFixed(2),
      finalMB: memory.finalMB.toFixed(2),
      heapUsedMB: memory.heapUsedMB.toFixed(2),
      externalMB: memory.externalMB.toFixed(2)
    });

    // Expected: < 50MB for 100K vectors (RuVector target)
    console.log('Expected: < 50MB (RuVector target), Actual:', memory.peakMB.toFixed(2), 'MB');
  });

  test('memory growth rate - incremental inserts', async () => {
    const measurements: Array<{ count: number; memoryMB: number }> = [];

    const backend = new RuVectorBackend();
    const checkpoints = [1000, 5000, 10000, 25000, 50000];

    if (global.gc) global.gc();
    const baseMemory = process.memoryUsage().heapUsed / 1024 / 1024;

    for (const checkpoint of checkpoints) {
      const currentSize = backend.getSize();
      const vectors = generateRandomVectors(checkpoint - currentSize, DIMENSION);

      for (let i = currentSize; i < checkpoint; i++) {
        backend.insert(`vec-${i}`, vectors[i - currentSize]);
      }

      if (global.gc) global.gc();
      await new Promise(resolve => setTimeout(resolve, 100));

      const currentMemory = process.memoryUsage().heapUsed / 1024 / 1024;
      measurements.push({
        count: checkpoint,
        memoryMB: currentMemory - baseMemory
      });

      console.log(`  ${checkpoint.toLocaleString()} vectors: ${(currentMemory - baseMemory).toFixed(2)} MB`);
    }

    // Calculate memory per vector
    const memoryPerVector = measurements[measurements.length - 1].memoryMB /
                           measurements[measurements.length - 1].count;
    console.log(`Average memory per vector: ${(memoryPerVector * 1024).toFixed(2)} KB`);

    backend.destroy();
  });

  test('memory after delete operations', async () => {
    const backend = new RuVectorBackend();

    // Insert vectors
    console.log('Inserting 10K vectors...');
    const vectors = generateRandomVectors(10000, DIMENSION);
    for (let i = 0; i < 10000; i++) {
      backend.insert(`vec-${i}`, vectors[i]);
    }

    if (global.gc) global.gc();
    const beforeMemory = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log('Memory before destroy:', beforeMemory.toFixed(2), 'MB');

    // Destroy backend
    backend.destroy();

    if (global.gc) global.gc();
    await new Promise(resolve => setTimeout(resolve, 100));

    const afterMemory = process.memoryUsage().heapUsed / 1024 / 1024;
    console.log('Memory after destroy:', afterMemory.toFixed(2), 'MB');
    console.log('Memory reclaimed:', (beforeMemory - afterMemory).toFixed(2), 'MB');
  });

  test('memory with concurrent operations', async () => {
    const memory = await measureMemory(async () => {
      const backend = new RuVectorBackend();
      const vectors = generateRandomVectors(10000, DIMENSION);

      // Insert vectors
      for (let i = 0; i < 10000; i++) {
        backend.insert(`vec-${i}`, vectors[i]);
      }

      // Simulate concurrent searches
      const queryVectors = generateRandomVectors(100, DIMENSION);
      const searchPromises = queryVectors.map(query =>
        Promise.resolve(backend.search(query, 10))
      );

      await Promise.all(searchPromises);
    });

    console.log('Memory (concurrent operations):', {
      peakMB: memory.peakMB.toFixed(2),
      finalMB: memory.finalMB.toFixed(2)
    });
  });

  test('memory leak detection - repeated insert/search cycles', async () => {
    const measurements: number[] = [];

    for (let cycle = 0; cycle < 5; cycle++) {
      const backend = new RuVectorBackend();

      if (global.gc) global.gc();
      const beforeCycle = process.memoryUsage().heapUsed / 1024 / 1024;

      // Insert and search
      const vectors = generateRandomVectors(5000, DIMENSION);
      for (let i = 0; i < 5000; i++) {
        backend.insert(`vec-${i}`, vectors[i]);
      }

      const query = generateRandomVectors(1, DIMENSION)[0];
      for (let i = 0; i < 100; i++) {
        backend.search(query, 10);
      }

      backend.destroy();

      if (global.gc) global.gc();
      await new Promise(resolve => setTimeout(resolve, 100));

      const afterCycle = process.memoryUsage().heapUsed / 1024 / 1024;
      measurements.push(afterCycle - beforeCycle);

      console.log(`  Cycle ${cycle + 1}: ${(afterCycle - beforeCycle).toFixed(2)} MB`);
    }

    // Check for memory leak (memory should stay relatively constant)
    const avgMemory = measurements.reduce((a, b) => a + b, 0) / measurements.length;
    const maxDeviation = Math.max(...measurements.map(m => Math.abs(m - avgMemory)));

    console.log('Average memory per cycle:', avgMemory.toFixed(2), 'MB');
    console.log('Max deviation:', maxDeviation.toFixed(2), 'MB');
    console.log('Memory leak check:', maxDeviation < avgMemory * 0.5 ? 'PASS' : 'WARN');
  });
});

describe('Memory Profiling - Different Dimensions', () => {
  const DIMENSIONS = [128, 256, 384, 512, 768, 1024];
  const VECTOR_COUNT = 1000;

  for (const dim of DIMENSIONS) {
    test(`memory usage - dimension ${dim}`, async () => {
      const memory = await measureMemory(async () => {
        const backend = new RuVectorBackend();
        const vectors = generateRandomVectors(VECTOR_COUNT, dim);

        for (let i = 0; i < VECTOR_COUNT; i++) {
          backend.insert(`vec-${i}`, vectors[i]);
        }
      });

      const bytesPerVector = (memory.peakMB * 1024 * 1024) / VECTOR_COUNT;
      console.log(`Dimension ${dim}: ${memory.peakMB.toFixed(2)} MB (${bytesPerVector.toFixed(0)} bytes/vector)`);
    });
  }
});
