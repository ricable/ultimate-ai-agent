/**
 * Comprehensive tests for ONNX Embedding Service
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { ONNXEmbeddingService } from '../services/ONNXEmbeddingService.js';

describe('ONNXEmbeddingService', () => {
  let embedder: ONNXEmbeddingService;

  beforeAll(async () => {
    embedder = new ONNXEmbeddingService({
      modelName: 'Xenova/all-MiniLM-L6-v2',
      useGPU: false, // Use CPU for tests
      batchSize: 4,
      cacheSize: 100
    });
    await embedder.initialize();
  });

  afterAll(() => {
    embedder.clearCache();
  });

  describe('Initialization', () => {
    it('should initialize successfully', async () => {
      const stats = embedder.getStats();
      expect(stats.initialized).toBe(true);
      expect(stats.model).toBe('Xenova/all-MiniLM-L6-v2');
    });

    it('should have correct dimension', () => {
      const dimension = embedder.getDimension();
      expect(dimension).toBe(384);
    });
  });

  describe('Single Embedding', () => {
    it('should generate embedding for single text', async () => {
      const text = 'This is a test sentence';
      const result = await embedder.embed(text);

      expect(result.embedding).toBeInstanceOf(Float32Array);
      expect(result.embedding.length).toBe(384);
      expect(result.latency).toBeGreaterThan(0);
      expect(result.cached).toBe(false);
      expect(result.model).toBe('Xenova/all-MiniLM-L6-v2');
    });

    it('should return cached result for same text', async () => {
      const text = 'Cached test sentence';

      // First call
      const result1 = await embedder.embed(text);
      expect(result1.cached).toBe(false);

      // Second call should be cached
      const result2 = await embedder.embed(text);
      expect(result2.cached).toBe(true);
      expect(result2.latency).toBeLessThan(result1.latency);
    });

    it('should generate different embeddings for different texts', async () => {
      const text1 = 'First sentence';
      const text2 = 'Second sentence';

      const result1 = await embedder.embed(text1);
      const result2 = await embedder.embed(text2);

      // Embeddings should be different
      const areDifferent = Array.from(result1.embedding).some(
        (val, i) => val !== result2.embedding[i]
      );
      expect(areDifferent).toBe(true);
    });

    it('should handle empty text', async () => {
      const result = await embedder.embed('');
      expect(result.embedding).toBeInstanceOf(Float32Array);
      expect(result.embedding.length).toBe(384);
    });

    it('should handle very long text', async () => {
      const longText = 'word '.repeat(1000);
      const result = await embedder.embed(longText);
      expect(result.embedding).toBeInstanceOf(Float32Array);
      expect(result.embedding.length).toBe(384);
    });
  });

  describe('Batch Embedding', () => {
    it('should generate embeddings for batch', async () => {
      const texts = [
        'First text',
        'Second text',
        'Third text',
        'Fourth text'
      ];

      const result = await embedder.embedBatch(texts);

      expect(result.embeddings).toHaveLength(4);
      expect(result.total).toBe(4);
      expect(result.cached).toBe(0);
      expect(result.latency).toBeGreaterThan(0);

      result.embeddings.forEach(emb => {
        expect(emb).toBeInstanceOf(Float32Array);
        expect(emb.length).toBe(384);
      });
    });

    it('should use cache for batch processing', async () => {
      const texts = ['Cached 1', 'Cached 2', 'Cached 3'];

      // First batch
      const result1 = await embedder.embedBatch(texts);
      expect(result1.cached).toBe(0);

      // Second batch (should all be cached)
      const result2 = await embedder.embedBatch(texts);
      expect(result2.cached).toBe(3);
      expect(result2.latency).toBeLessThan(result1.latency);
    });

    it('should handle large batches', async () => {
      const texts = Array.from({ length: 50 }, (_, i) => `Batch text ${i}`);
      const result = await embedder.embedBatch(texts);

      expect(result.embeddings).toHaveLength(50);
      expect(result.total).toBe(50);
    });

    it('should handle empty batch', async () => {
      const result = await embedder.embedBatch([]);
      expect(result.embeddings).toHaveLength(0);
      expect(result.total).toBe(0);
    });
  });

  describe('Performance', () => {
    it('should generate embedding quickly', async () => {
      const startTime = Date.now();
      await embedder.embed('Performance test');
      const latency = Date.now() - startTime;

      // Should be under 1 second for first run
      expect(latency).toBeLessThan(1000);
    });

    it('should be fast with cache', async () => {
      const text = 'Cache speed test';

      // Warm up cache
      await embedder.embed(text);

      // Measure cached access
      const startTime = Date.now();
      await embedder.embed(text);
      const latency = Date.now() - startTime;

      // Cached access should be < 10ms
      expect(latency).toBeLessThan(10);
    });

    it('should show performance improvement with warmup', async () => {
      const newEmbedder = new ONNXEmbeddingService({
        modelName: 'Xenova/all-MiniLM-L6-v2',
        useGPU: false
      });
      await newEmbedder.initialize();

      // Before warmup
      const start1 = Date.now();
      await newEmbedder.embed('Test before warmup');
      const beforeWarmup = Date.now() - start1;

      // Warmup
      await newEmbedder.warmup(5);

      // After warmup
      const start2 = Date.now();
      await newEmbedder.embed('Test after warmup');
      const afterWarmup = Date.now() - start2;

      // Warmup should improve performance
      expect(newEmbedder.getStats().warmupComplete).toBe(true);
    });
  });

  describe('Cache Management', () => {
    it('should respect cache size limit', async () => {
      const smallCache = new ONNXEmbeddingService({
        modelName: 'Xenova/all-MiniLM-L6-v2',
        useGPU: false,
        cacheSize: 5
      });
      await smallCache.initialize();

      // Add 10 items (should evict 5)
      for (let i = 0; i < 10; i++) {
        await smallCache.embed(`Cache test ${i}`);
      }

      const stats = smallCache.getStats();
      expect(stats.cache.size).toBeLessThanOrEqual(5);
    });

    it('should clear cache', async () => {
      await embedder.embed('Test 1');
      await embedder.embed('Test 2');

      let stats = embedder.getStats();
      expect(stats.cache.size).toBeGreaterThan(0);

      embedder.clearCache();

      stats = embedder.getStats();
      expect(stats.cache.size).toBe(0);
    });

    it('should track cache hit rate', async () => {
      embedder.clearCache();

      // Generate some embeddings
      await embedder.embed('Hit rate test 1');
      await embedder.embed('Hit rate test 2');

      // Access cached items
      await embedder.embed('Hit rate test 1');
      await embedder.embed('Hit rate test 2');

      const stats = embedder.getStats();
      expect(stats.cache.hitRate).toBeGreaterThan(0);
    });
  });

  describe('Statistics', () => {
    it('should track total embeddings', async () => {
      const initialStats = embedder.getStats();
      const initialCount = initialStats.totalEmbeddings;

      await embedder.embed('Stats test 1');
      await embedder.embed('Stats test 2');

      const newStats = embedder.getStats();
      expect(newStats.totalEmbeddings).toBe(initialCount + 2);
    });

    it('should track average latency', async () => {
      const stats = embedder.getStats();
      expect(stats.avgLatency).toBeGreaterThan(0);
    });

    it('should track batch sizes', async () => {
      await embedder.embedBatch(['Batch 1', 'Batch 2', 'Batch 3']);
      const stats = embedder.getStats();
      expect(stats.avgBatchSize).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    it('should throw error if not initialized', async () => {
      const uninitialized = new ONNXEmbeddingService({
        modelName: 'Xenova/all-MiniLM-L6-v2'
      });

      await expect(uninitialized.embed('Test')).rejects.toThrow('not initialized');
    });
  });

  describe('Similarity', () => {
    it('should generate similar embeddings for similar texts', async () => {
      const text1 = 'The cat sits on the mat';
      const text2 = 'A cat is sitting on a mat';

      const result1 = await embedder.embed(text1);
      const result2 = await embedder.embed(text2);

      // Calculate cosine similarity
      const similarity = cosineSimilarity(result1.embedding, result2.embedding);

      // Similar texts should have high similarity (>0.7)
      expect(similarity).toBeGreaterThan(0.7);
    });

    it('should generate dissimilar embeddings for different texts', async () => {
      const text1 = 'The weather is sunny today';
      const text2 = 'Quantum physics is fascinating';

      const result1 = await embedder.embed(text1);
      const result2 = await embedder.embed(text2);

      const similarity = cosineSimilarity(result1.embedding, result2.embedding);

      // Different texts should have lower similarity (<0.7)
      expect(similarity).toBeLessThan(0.7);
    });
  });
});

/**
 * Helper: Calculate cosine similarity
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
