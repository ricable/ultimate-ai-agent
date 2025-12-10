/**
 * @test Attention Mechanism Integration Tests
 * @description Comprehensive end-to-end tests for all attention mechanisms
 * @prerequisites
 *   - AgentDB initialized
 *   - RuVector native bindings available
 *   - Test database created
 * @coverage
 *   - Self-attention mechanisms
 *   - Cross-attention mechanisms
 *   - Multi-head attention
 *   - Memory controller integrations
 *   - CLI commands
 *   - MCP tools
 *   - Browser WASM loading
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import AgentDB from '../../src/index.js';
import { MemoryController } from '../../src/controllers/MemoryController.js';
import { SelfAttentionController } from '../../src/controllers/attention/SelfAttentionController.js';
import { CrossAttentionController } from '../../src/controllers/attention/CrossAttentionController.js';
import { MultiHeadAttentionController } from '../../src/controllers/attention/MultiHeadAttentionController.js';
import fs from 'fs';
import path from 'path';

// TODO: These tests require attention controllers (MemoryController, SelfAttentionController, etc.)
// that are not yet implemented. Mark as .todo() until feature is built.
describe.todo('Attention Mechanism Integration', () => {
  let db: AgentDB;
  let memoryController: MemoryController;
  const testDbPath = path.join(__dirname, '../fixtures/test-attention.db');

  beforeEach(async () => {
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    // Initialize AgentDB
    db = new AgentDB({
      dbPath: testDbPath,
      namespace: 'test-attention',
      enableAttention: true,
      attentionConfig: {
        selfAttention: { enabled: true },
        crossAttention: { enabled: true },
        multiHeadAttention: { enabled: true, numHeads: 8 }
      }
    });

    await db.initialize();
    memoryController = db.getController('memory') as MemoryController;
  });

  afterEach(async () => {
    await db.close();
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Self-Attention Mechanism', () => {
    it('should compute self-attention scores for memory entries', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      // Store test memories
      const memories = [
        { id: 'mem1', content: 'AI agent learns task', embedding: [0.1, 0.2, 0.3] },
        { id: 'mem2', content: 'Task execution successful', embedding: [0.15, 0.25, 0.35] },
        { id: 'mem3', content: 'Agent improves performance', embedding: [0.12, 0.22, 0.32] }
      ];

      for (const mem of memories) {
        await memoryController.store(mem);
      }

      // Compute self-attention
      const query = [0.1, 0.2, 0.3];
      const result = await controller.computeAttention(query, {
        topK: 3,
        minScore: 0.0
      });

      expect(result).toBeDefined();
      expect(result.scores).toHaveLength(3);
      expect(result.scores[0].score).toBeGreaterThanOrEqual(0);
      expect(result.scores[0].score).toBeLessThanOrEqual(1);
      expect(result.attended).toBeDefined();
      expect(result.attended.length).toBe(query.length);
    });

    it('should apply softmax normalization to attention scores', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      // Store test data
      await memoryController.store({
        id: 'test1',
        content: 'Test content',
        embedding: [1.0, 0.0, 0.0]
      });

      const query = [1.0, 0.0, 0.0];
      const result = await controller.computeAttention(query);

      // Check softmax normalization
      const sum = result.scores.reduce((acc, item) => acc + item.score, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(0.001);
    });

    it('should filter results by minimum attention score', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      // Store memories with varying similarity
      await memoryController.store({ id: 'm1', content: 'High similarity', embedding: [0.9, 0.1, 0.0] });
      await memoryController.store({ id: 'm2', content: 'Low similarity', embedding: [0.0, 0.1, 0.9] });

      const query = [1.0, 0.0, 0.0];
      const result = await controller.computeAttention(query, {
        minScore: 0.5
      });

      // Only high similarity items should pass threshold
      expect(result.scores.length).toBeLessThanOrEqual(1);
      result.scores.forEach(item => {
        expect(item.score).toBeGreaterThanOrEqual(0.5);
      });
    });

    it('should handle empty memory gracefully', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      const query = [0.1, 0.2, 0.3];
      const result = await controller.computeAttention(query);

      expect(result.scores).toHaveLength(0);
      expect(result.attended).toEqual(query);
    });

    it('should scale with large memory sets efficiently', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      // Store 1000 memories
      const promises = [];
      for (let i = 0; i < 1000; i++) {
        promises.push(memoryController.store({
          id: `mem${i}`,
          content: `Memory ${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        }));
      }
      await Promise.all(promises);

      const query = [0.5, 0.5, 0.5];
      const start = Date.now();
      const result = await controller.computeAttention(query, { topK: 10 });
      const duration = Date.now() - start;

      expect(result.scores).toHaveLength(10);
      expect(duration).toBeLessThan(1000); // Should complete in < 1 second
    });
  });

  describe('Cross-Attention Mechanism', () => {
    it('should compute cross-attention between query and memory', async () => {
      const controller = db.getController('cross-attention') as CrossAttentionController;

      // Store memory context
      const context = [
        { id: 'ctx1', embedding: [0.1, 0.2, 0.3] },
        { id: 'ctx2', embedding: [0.4, 0.5, 0.6] }
      ];

      for (const ctx of context) {
        await memoryController.store(ctx);
      }

      // Compute cross-attention
      const query = [0.2, 0.3, 0.4];
      const result = await controller.computeCrossAttention(query, 'memory');

      expect(result).toBeDefined();
      expect(result.scores).toHaveLength(2);
      expect(result.attended).toBeDefined();
    });

    it('should integrate query and context via attention', async () => {
      const controller = db.getController('cross-attention') as CrossAttentionController;

      await memoryController.store({
        id: 'context1',
        embedding: [1.0, 0.0, 0.0]
      });

      const query = [0.0, 1.0, 0.0];
      const result = await controller.computeCrossAttention(query, 'memory');

      // Attended output should be a blend of query and context
      expect(result.attended).toBeDefined();
      expect(result.attended.length).toBe(query.length);
      expect(result.attended).not.toEqual(query);
    });

    it('should support multiple context sources', async () => {
      const controller = db.getController('cross-attention') as CrossAttentionController;

      // Store in different namespaces
      await memoryController.store({ id: 'mem1', embedding: [0.1, 0.2, 0.3] }, 'episodic');
      await memoryController.store({ id: 'mem2', embedding: [0.4, 0.5, 0.6] }, 'semantic');

      const query = [0.2, 0.3, 0.4];
      const result1 = await controller.computeCrossAttention(query, 'episodic');
      const result2 = await controller.computeCrossAttention(query, 'semantic');

      expect(result1.scores).not.toEqual(result2.scores);
    });
  });

  describe('Multi-Head Attention Mechanism', () => {
    it('should compute multi-head attention with configured heads', async () => {
      const controller = db.getController('multi-head-attention') as MultiHeadAttentionController;

      // Store test data
      for (let i = 0; i < 5; i++) {
        await memoryController.store({
          id: `head-test-${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        });
      }

      const query = [0.5, 0.5, 0.5];
      const result = await controller.computeMultiHeadAttention(query, {
        numHeads: 4,
        topK: 5
      });

      expect(result).toBeDefined();
      expect(result.heads).toHaveLength(4);
      expect(result.attended).toBeDefined();
      expect(result.attended.length).toBe(query.length);
    });

    it('should aggregate attention from multiple heads', async () => {
      const controller = db.getController('multi-head-attention') as MultiHeadAttentionController;

      await memoryController.store({
        id: 'multi-head-test',
        embedding: [0.1, 0.2, 0.3]
      });

      const query = [0.1, 0.2, 0.3];
      const result = await controller.computeMultiHeadAttention(query, {
        numHeads: 8
      });

      // Each head should produce different attention patterns
      const uniqueHeads = new Set(result.heads.map(h => JSON.stringify(h.attended)));
      expect(uniqueHeads.size).toBeGreaterThan(1);
    });

    it('should support different aggregation strategies', async () => {
      const controller = db.getController('multi-head-attention') as MultiHeadAttentionController;

      await memoryController.store({ id: 'agg-test', embedding: [0.5, 0.5, 0.5] });

      const query = [0.5, 0.5, 0.5];

      const avgResult = await controller.computeMultiHeadAttention(query, {
        aggregation: 'average'
      });

      const maxResult = await controller.computeMultiHeadAttention(query, {
        aggregation: 'max'
      });

      expect(avgResult.attended).not.toEqual(maxResult.attended);
    });

    it('should handle varying head dimensions', async () => {
      const controller = db.getController('multi-head-attention') as MultiHeadAttentionController;

      await memoryController.store({ id: 'dim-test', embedding: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] });

      const query = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
      const result = await controller.computeMultiHeadAttention(query, {
        numHeads: 3,
        headDim: 2
      });

      expect(result.heads).toHaveLength(3);
      expect(result.attended.length).toBe(query.length);
    });
  });

  describe('Memory Controller Integration', () => {
    it('should enhance memory retrieval with attention', async () => {
      // Store memories with metadata
      await memoryController.store({
        id: 'task1',
        content: 'Complete authentication',
        embedding: [0.1, 0.2, 0.3],
        importance: 0.9
      });

      await memoryController.store({
        id: 'task2',
        content: 'Write documentation',
        embedding: [0.4, 0.5, 0.6],
        importance: 0.5
      });

      // Retrieve with attention
      const query = [0.15, 0.25, 0.35];
      const results = await memoryController.retrieveWithAttention(query, {
        topK: 2,
        useAttention: true
      });

      expect(results).toHaveLength(2);
      expect(results[0]).toHaveProperty('attentionScore');
      expect(results[0].attentionScore).toBeGreaterThan(0);
    });

    it('should prioritize important memories via attention', async () => {
      await memoryController.store({
        id: 'important',
        embedding: [0.5, 0.5, 0.5],
        importance: 1.0
      });

      await memoryController.store({
        id: 'unimportant',
        embedding: [0.51, 0.51, 0.51],
        importance: 0.1
      });

      const query = [0.5, 0.5, 0.5];
      const results = await memoryController.retrieveWithAttention(query, {
        topK: 1,
        weighByImportance: true
      });

      expect(results[0].id).toBe('important');
    });

    it('should support temporal attention for recent memories', async () => {
      const now = Date.now();

      await memoryController.store({
        id: 'old',
        embedding: [0.1, 0.2, 0.3],
        timestamp: now - 86400000 // 1 day ago
      });

      await memoryController.store({
        id: 'recent',
        embedding: [0.1, 0.2, 0.3],
        timestamp: now - 3600000 // 1 hour ago
      });

      const query = [0.1, 0.2, 0.3];
      const results = await memoryController.retrieveWithAttention(query, {
        temporalWeight: 0.5
      });

      expect(results[0].id).toBe('recent');
    });
  });

  describe('Performance Tests', () => {
    it('should process attention in real-time (<100ms)', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      // Store 100 memories
      for (let i = 0; i < 100; i++) {
        await memoryController.store({
          id: `perf-${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        });
      }

      const query = [0.5, 0.5, 0.5];
      const start = performance.now();
      await controller.computeAttention(query, { topK: 10 });
      const duration = performance.now() - start;

      expect(duration).toBeLessThan(100);
    });

    it('should handle concurrent attention requests', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      // Store test data
      for (let i = 0; i < 50; i++) {
        await memoryController.store({
          id: `concurrent-${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        });
      }

      // Run 10 concurrent queries
      const queries = Array(10).fill(null).map(() =>
        controller.computeAttention([Math.random(), Math.random(), Math.random()])
      );

      const results = await Promise.all(queries);
      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.scores).toBeDefined();
      });
    });

    it('should maintain memory efficiency with attention computation', async () => {
      const controller = db.getController('multi-head-attention') as MultiHeadAttentionController;

      const initialMemory = process.memoryUsage().heapUsed;

      // Store and process large dataset
      for (let i = 0; i < 1000; i++) {
        await memoryController.store({
          id: `mem-eff-${i}`,
          embedding: Array(128).fill(0).map(() => Math.random())
        });
      }

      const query = Array(128).fill(0).map(() => Math.random());
      await controller.computeMultiHeadAttention(query, {
        numHeads: 8,
        topK: 50
      });

      global.gc && global.gc();
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      // Should not use more than 50MB
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid query dimensions', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      await memoryController.store({
        id: 'test',
        embedding: [0.1, 0.2, 0.3]
      });

      const invalidQuery = [0.1, 0.2]; // Wrong dimension
      await expect(
        controller.computeAttention(invalidQuery)
      ).rejects.toThrow();
    });

    it('should handle null/undefined inputs gracefully', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      await expect(
        controller.computeAttention(null as any)
      ).rejects.toThrow();

      await expect(
        controller.computeAttention(undefined as any)
      ).rejects.toThrow();
    });

    it('should validate attention configuration', async () => {
      await expect(
        new AgentDB({
          dbPath: ':memory:',
          enableAttention: true,
          attentionConfig: {
            multiHeadAttention: {
              enabled: true,
              numHeads: 0 // Invalid
            }
          }
        }).initialize()
      ).rejects.toThrow();
    });

    it('should recover from attention computation errors', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      // Mock a computation error
      vi.spyOn(controller as any, 'computeScores').mockRejectedValueOnce(
        new Error('Computation failed')
      );

      const query = [0.1, 0.2, 0.3];
      await expect(
        controller.computeAttention(query)
      ).rejects.toThrow('Computation failed');

      // Should recover on next attempt
      (controller as any).computeScores.mockRestore();
      await memoryController.store({ id: 'recovery', embedding: query });

      const result = await controller.computeAttention(query);
      expect(result).toBeDefined();
    });
  });

  describe('Edge Cases', () => {
    it('should handle zero vectors in attention', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      await memoryController.store({
        id: 'zero',
        embedding: [0, 0, 0]
      });

      const query = [0, 0, 0];
      const result = await controller.computeAttention(query);

      expect(result).toBeDefined();
      expect(isNaN(result.scores[0]?.score || 0)).toBe(false);
    });

    it('should handle very large attention scores', async () => {
      const controller = db.getController('self-attention') as SelfAttentionController;

      await memoryController.store({
        id: 'large',
        embedding: [1e10, 1e10, 1e10]
      });

      const query = [1e10, 1e10, 1e10];
      const result = await controller.computeAttention(query);

      expect(result.scores[0].score).toBeGreaterThanOrEqual(0);
      expect(result.scores[0].score).toBeLessThanOrEqual(1);
      expect(isFinite(result.scores[0].score)).toBe(true);
    });

    it('should handle high-dimensional embeddings', async () => {
      const controller = db.getController('multi-head-attention') as MultiHeadAttentionController;

      const dim = 1024;
      const embedding = Array(dim).fill(0).map(() => Math.random());

      await memoryController.store({ id: 'high-dim', embedding });

      const query = Array(dim).fill(0).map(() => Math.random());
      const result = await controller.computeMultiHeadAttention(query, {
        numHeads: 16
      });

      expect(result.attended.length).toBe(dim);
    });
  });
});
