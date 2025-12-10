/**
 * RuVector Backend Tests
 *
 * Tests RuVector-specific functionality including WASM acceleration,
 * SIMD optimizations, batch operations, and performance characteristics.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { WASMVectorSearch } from '../../src/controllers/WASMVectorSearch.js';

// Mock database interface
interface MockDatabase {
  prepare: (sql: string) => {
    all: (...params: any[]) => any[];
    get: (...params: any[]) => any;
  };
}

function generateVector(dimension: number): Float32Array {
  const vec = new Float32Array(dimension);
  for (let i = 0; i < dimension; i++) {
    vec[i] = Math.random() * 2 - 1;
  }
  return normalizeVector(vec);
}

function normalizeVector(vec: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) {
    norm += vec[i] * vec[i];
  }
  norm = Math.sqrt(norm);

  if (norm > 0) {
    for (let i = 0; i < vec.length; i++) {
      vec[i] /= norm;
    }
  }
  return vec;
}

interface TestVector {
  id: string;
  embedding: Float32Array;
}

function generateTestVectors(count: number, dimension: number): TestVector[] {
  return Array.from({ length: count }, (_, i) => ({
    id: `vec-${i}`,
    embedding: generateVector(dimension),
  }));
}

function createMockDatabase(vectors: TestVector[]): MockDatabase {
  const vectorMap = new Map(
    vectors.map((v, idx) => [
      idx + 1,
      {
        id: idx + 1,
        embedding: Buffer.from(v.embedding.buffer),
      },
    ])
  );

  return {
    prepare: (sql: string) => ({
      all: (...params: any[]) => {
        if (sql.includes('SELECT pattern_id')) {
          return Array.from(vectorMap.values());
        }
        return [];
      },
      get: (id: number) => {
        return vectorMap.get(id);
      },
    }),
  };
}

describe('RuVector Backend Tests', () => {
  const DIMENSION = 384;
  let wasmSearch: WASMVectorSearch;
  let testVectors: TestVector[];
  let mockDb: MockDatabase;

  beforeAll(() => {
    testVectors = generateTestVectors(1000, DIMENSION);
    mockDb = createMockDatabase(testVectors) as any;

    wasmSearch = new WASMVectorSearch(mockDb as any, {
      enableWASM: false, // Test pure JS implementation
      enableSIMD: false,
      batchSize: 100,
      indexThreshold: 500,
    });
  });

  afterAll(() => {
    wasmSearch.clearIndex();
  });

  describe('Initialization', () => {
    it('should initialize with correct configuration', () => {
      const stats = wasmSearch.getStats();

      expect(stats.wasmAvailable).toBe(false); // Disabled in config
      expect(stats.simdAvailable).toBe(false); // Disabled in config
      expect(stats.indexBuilt).toBe(false); // Not built yet
    });

    it('should create instance with custom config', () => {
      const customSearch = new WASMVectorSearch(mockDb as any, {
        enableWASM: true,
        enableSIMD: true,
        batchSize: 200,
        indexThreshold: 2000,
      });

      const stats = customSearch.getStats();
      expect(stats).toBeDefined();
    });
  });

  describe('Cosine Similarity', () => {
    it('should compute identical vector similarity as 1.0', () => {
      const vec = generateVector(DIMENSION);
      const similarity = wasmSearch.cosineSimilarity(vec, vec);

      expect(similarity).toBeCloseTo(1.0, 5);
    });

    it('should compute orthogonal vectors similarity as 0.0', () => {
      const vec1 = new Float32Array([1, 0, 0]);
      const vec2 = new Float32Array([0, 1, 0]);

      const similarity = wasmSearch.cosineSimilarity(vec1, vec2);
      expect(similarity).toBeCloseTo(0.0, 5);
    });

    it('should compute opposite vectors similarity as -1.0', () => {
      const vec1 = new Float32Array([1, 0, 0]);
      const vec2 = new Float32Array([-1, 0, 0]);

      const similarity = wasmSearch.cosineSimilarity(vec1, vec2);
      expect(similarity).toBeCloseTo(-1.0, 5);
    });

    it('should throw error for mismatched vector dimensions', () => {
      const vec1 = new Float32Array([1, 0, 0]);
      const vec2 = new Float32Array([1, 0, 0, 0]);

      expect(() => {
        wasmSearch.cosineSimilarity(vec1, vec2);
      }).toThrow('Vectors must have same length');
    });

    it('should handle normalized vectors correctly', () => {
      const unnormalized = new Float32Array([3, 4, 0]);
      const magnitude = Math.sqrt(9 + 16); // = 5

      const normalized = new Float32Array([3 / magnitude, 4 / magnitude, 0]);

      const similarity = wasmSearch.cosineSimilarity(normalized, normalized);
      expect(similarity).toBeCloseTo(1.0, 5);
    });
  });

  describe('Batch Similarity', () => {
    it('should compute batch similarities efficiently', () => {
      const query = generateVector(DIMENSION);
      const vectors = Array.from({ length: 100 }, () => generateVector(DIMENSION));

      const start = performance.now();
      const similarities = wasmSearch.batchSimilarity(query, vectors);
      const duration = performance.now() - start;

      expect(similarities.length).toBe(100);
      expect(duration).toBeLessThan(100); // Should be fast

      // All similarities should be in valid range
      for (const sim of similarities) {
        expect(sim).toBeGreaterThanOrEqual(-1);
        expect(sim).toBeLessThanOrEqual(1);
      }
    });

    it('should process large batches without errors', () => {
      const query = generateVector(DIMENSION);
      const vectors = Array.from({ length: 10000 }, () => generateVector(DIMENSION));

      const similarities = wasmSearch.batchSimilarity(query, vectors);

      expect(similarities.length).toBe(10000);
    });

    it('should handle empty vector arrays', () => {
      const query = generateVector(DIMENSION);
      const vectors: Float32Array[] = [];

      const similarities = wasmSearch.batchSimilarity(query, vectors);

      expect(similarities.length).toBe(0);
    });
  });

  describe('K-Nearest Neighbors Search', () => {
    it('should find k-nearest neighbors correctly', async () => {
      const query = testVectors[0].embedding;
      const k = 10;

      const results = await wasmSearch.findKNN(query, k);

      expect(results.length).toBeLessThanOrEqual(k);

      // Results should be sorted by similarity (descending)
      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].similarity).toBeGreaterThanOrEqual(results[i].similarity);
      }
    });

    it('should respect threshold parameter', async () => {
      const query = generateVector(DIMENSION);
      const threshold = 0.8;

      const results = await wasmSearch.findKNN(query, 20, 'pattern_embeddings', {
        threshold,
      });

      // All results should meet threshold
      for (const result of results) {
        expect(result.similarity).toBeGreaterThanOrEqual(threshold);
      }
    });

    it('should handle k larger than dataset size', async () => {
      const query = generateVector(DIMENSION);
      const k = testVectors.length * 2;

      const results = await wasmSearch.findKNN(query, k);

      expect(results.length).toBeLessThanOrEqual(testVectors.length);
    });

    it('should return correct distance and similarity', async () => {
      const query = testVectors[0].embedding;

      const results = await wasmSearch.findKNN(query, 5);

      for (const result of results) {
        // Distance should equal (1 - similarity) for cosine metric
        const expectedDistance = 1 - result.similarity;
        expect(Math.abs(result.distance - expectedDistance)).toBeLessThan(0.0001);
      }
    });

    it('should handle zero-result scenarios gracefully', async () => {
      const query = generateVector(DIMENSION);
      const threshold = 1.0; // Impossible threshold

      const results = await wasmSearch.findKNN(query, 10, 'pattern_embeddings', {
        threshold,
      });

      expect(results.length).toBe(0);
    });
  });

  describe('Index Building', () => {
    it('should skip index for small datasets', () => {
      const smallVectors = Array.from({ length: 100 }, () => generateVector(DIMENSION));
      const ids = smallVectors.map((_, i) => i);

      wasmSearch.buildIndex(smallVectors, ids);

      const stats = wasmSearch.getStats();
      expect(stats.indexBuilt).toBe(false); // Below threshold
    });

    it('should build index for large datasets', () => {
      const largeVectors = Array.from({ length: 1000 }, () => generateVector(DIMENSION));
      const ids = largeVectors.map((_, i) => i);

      wasmSearch.buildIndex(largeVectors, ids);

      const stats = wasmSearch.getStats();
      expect(stats.indexBuilt).toBe(true);
      expect(stats.indexSize).toBe(1000);
      expect(stats.lastIndexUpdate).toBeGreaterThan(0);
    });

    it('should store metadata with index', () => {
      const vectors = Array.from({ length: 1000 }, () => generateVector(DIMENSION));
      const ids = vectors.map((_, i) => i);
      const metadata = ids.map(id => ({ id, label: `item-${id}` }));

      wasmSearch.buildIndex(vectors, ids, metadata);

      const stats = wasmSearch.getStats();
      expect(stats.indexBuilt).toBe(true);
    });
  });

  describe('Index Search', () => {
    beforeAll(() => {
      // Build index for these tests
      const vectors = Array.from({ length: 1000 }, () => generateVector(DIMENSION));
      const ids = vectors.map((_, i) => i);
      wasmSearch.buildIndex(vectors, ids);
    });

    it('should search using built index', () => {
      const query = generateVector(DIMENSION);
      const k = 10;

      const results = wasmSearch.searchIndex(query, k);

      expect(results.length).toBeLessThanOrEqual(k);

      // Results should be sorted by similarity
      for (let i = 1; i < results.length; i++) {
        expect(results[i - 1].similarity).toBeGreaterThanOrEqual(results[i].similarity);
      }
    });

    it('should apply threshold in index search', () => {
      const query = generateVector(DIMENSION);
      const threshold = 0.5;

      const results = wasmSearch.searchIndex(query, 20, threshold);

      for (const result of results) {
        expect(result.similarity).toBeGreaterThanOrEqual(threshold);
      }
    });

    it('should throw error when index not built', () => {
      const newSearch = new WASMVectorSearch(mockDb as any);
      const query = generateVector(DIMENSION);

      expect(() => {
        newSearch.searchIndex(query, 10);
      }).toThrow('Index not built');
    });
  });

  describe('Performance', () => {
    it('should perform searches within time budget', async () => {
      const query = generateVector(DIMENSION);
      const iterations = 100;

      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        await wasmSearch.findKNN(query, 10);
      }
      const duration = performance.now() - start;
      const avgTime = duration / iterations;

      console.log(`[RuVector] Average search time: ${avgTime.toFixed(2)}ms`);
      expect(avgTime).toBeLessThan(50); // <50ms per search
    });

    it('should handle concurrent searches efficiently', async () => {
      const queries = Array.from({ length: 10 }, () => generateVector(DIMENSION));

      const start = performance.now();
      await Promise.all(queries.map(q => wasmSearch.findKNN(q, 10)));
      const duration = performance.now() - start;

      console.log(`[RuVector] Concurrent search time: ${duration.toFixed(2)}ms`);
      expect(duration).toBeLessThan(500); // All 10 should complete quickly
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero vectors gracefully', () => {
      const zero1 = new Float32Array(DIMENSION);
      const zero2 = new Float32Array(DIMENSION);

      const similarity = wasmSearch.cosineSimilarity(zero1, zero2);
      expect(similarity).toBe(0); // Division by zero protection
    });

    it('should handle single-element vectors', () => {
      const vec1 = new Float32Array([0.5]);
      const vec2 = new Float32Array([0.5]);

      const similarity = wasmSearch.cosineSimilarity(vec1, vec2);
      expect(similarity).toBeCloseTo(1.0, 5);
    });

    it('should handle very large dimensions', () => {
      const largeDim = 4096;
      const vec1 = generateVector(largeDim);
      const vec2 = generateVector(largeDim);

      const similarity = wasmSearch.cosineSimilarity(vec1, vec2);

      expect(similarity).toBeGreaterThanOrEqual(-1);
      expect(similarity).toBeLessThanOrEqual(1);
    });

    it('should handle NaN values safely', () => {
      const vecWithNaN = new Float32Array([1, NaN, 0]);
      const normalVec = new Float32Array([1, 0, 0]);

      const similarity = wasmSearch.cosineSimilarity(vecWithNaN, normalVec);
      expect(isNaN(similarity)).toBe(true);
    });
  });

  describe('Statistics and Reporting', () => {
    it('should report accurate statistics', () => {
      const stats = wasmSearch.getStats();

      expect(stats).toHaveProperty('wasmAvailable');
      expect(stats).toHaveProperty('simdAvailable');
      expect(stats).toHaveProperty('indexBuilt');
      expect(stats).toHaveProperty('indexSize');
      expect(stats).toHaveProperty('lastIndexUpdate');

      expect(typeof stats.wasmAvailable).toBe('boolean');
      expect(typeof stats.simdAvailable).toBe('boolean');
      expect(typeof stats.indexBuilt).toBe('boolean');
      expect(typeof stats.indexSize).toBe('number');
    });

    it('should clear index correctly', () => {
      const vectors = Array.from({ length: 1000 }, () => generateVector(DIMENSION));
      const ids = vectors.map((_, i) => i);

      wasmSearch.buildIndex(vectors, ids);
      expect(wasmSearch.getStats().indexBuilt).toBe(true);

      wasmSearch.clearIndex();
      expect(wasmSearch.getStats().indexBuilt).toBe(false);
      expect(wasmSearch.getStats().indexSize).toBe(0);
    });
  });
});
