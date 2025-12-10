/**
 * Integration tests for AgentDB + ONNX
 *
 * Tests the integration between ONNXEmbeddingService and AgentDB controllers
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createONNXAgentDB } from '../index.js';

describe('AgentDB + ONNX Integration', () => {
  let agentdb: Awaited<ReturnType<typeof createONNXAgentDB>>;

  beforeAll(async () => {
    agentdb = await createONNXAgentDB({
      dbPath: ':memory:',  // Use in-memory database for tests
      modelName: 'Xenova/all-MiniLM-L6-v2',
      useGPU: false,
      batchSize: 4,
      cacheSize: 100
    });
  });

  afterAll(async () => {
    await agentdb.close();
  });

  describe('ReasoningBank', () => {
    it('should store and retrieve patterns', async () => {
      const patternId = await agentdb.reasoningBank.storePattern({
        taskType: 'debugging',
        approach: 'Binary search through code execution',
        successRate: 0.92,
        tags: ['systematic', 'efficient']
      });

      expect(patternId).toBeGreaterThan(0);

      const pattern = agentdb.reasoningBank.getPattern(patternId);
      expect(pattern).not.toBeNull();
      expect(pattern?.taskType).toBe('debugging');
      expect(pattern?.successRate).toBe(0.92);
    });

    it('should search for similar patterns with semantic matching', async () => {
      // Store multiple debugging patterns
      await agentdb.reasoningBank.storePattern({
        taskType: 'debugging',
        approach: 'Check logs first, then reproduce the issue systematically',
        successRate: 0.88
      });

      await agentdb.reasoningBank.storePattern({
        taskType: 'debugging',
        approach: 'Use debugger breakpoints and step through execution',
        successRate: 0.85
      });

      await agentdb.reasoningBank.storePattern({
        taskType: 'optimization',
        approach: 'Profile before optimizing to identify bottlenecks',
        successRate: 0.95
      });

      // Search for debugging patterns using semantic query
      const results = await agentdb.reasoningBank.searchPatterns({
        task: 'how to debug code issues',
        k: 5,
        threshold: 0.5
      });

      expect(results.length).toBeGreaterThan(0);
      results.forEach(r => {
        expect(r.similarity).toBeGreaterThanOrEqual(0.5);
        expect(r.similarity).toBeLessThanOrEqual(1.0);
      });

      // Most results should be debugging-related (semantic matching)
      const debuggingResults = results.filter(r => r.taskType === 'debugging');
      expect(debuggingResults.length).toBeGreaterThan(0);
    });

    it('should filter by task type', async () => {
      const results = await agentdb.reasoningBank.searchPatterns({
        task: 'approach for solving problems',
        k: 10,
        filters: { taskType: 'debugging' }
      });

      results.forEach(r => {
        expect(r.taskType).toBe('debugging');
      });
    });

    it('should use ONNX cache for repeated queries', async () => {
      const stats1 = agentdb.embedder.getStats();

      // First search - will generate embedding
      await agentdb.reasoningBank.searchPatterns({
        task: 'test query for caching',
        k: 5
      });

      const stats2 = agentdb.embedder.getStats();
      const embeddings1 = stats2.totalEmbeddings - stats1.totalEmbeddings;

      // Second search - should use cache
      await agentdb.reasoningBank.searchPatterns({
        task: 'test query for caching',
        k: 5
      });

      const stats3 = agentdb.embedder.getStats();
      const embeddings2 = stats3.totalEmbeddings - stats2.totalEmbeddings;

      // Second query should use cache (0 new embeddings)
      expect(embeddings2).toBe(0);
      expect(embeddings1).toBeGreaterThan(0);
    });

    it('should delete patterns', async () => {
      const id = await agentdb.reasoningBank.storePattern({
        taskType: 'test',
        approach: 'temporary pattern',
        successRate: 0.5
      });

      // Verify it exists
      let pattern = agentdb.reasoningBank.getPattern(id);
      expect(pattern).not.toBeNull();

      // Delete it
      const deleted = agentdb.reasoningBank.deletePattern(id);
      expect(deleted).toBe(true);

      // Verify it's gone
      pattern = agentdb.reasoningBank.getPattern(id);
      expect(pattern).toBeNull();
    });

    it('should record outcomes for learning', async () => {
      const id = await agentdb.reasoningBank.storePattern({
        taskType: 'testing',
        approach: 'learning pattern',
        successRate: 0.5,
        uses: 0,
        avgReward: 0
      });

      // Record successful outcome
      await agentdb.reasoningBank.recordOutcome(id, true, 0.95);

      // Verify stats updated
      const pattern = agentdb.reasoningBank.getPattern(id);
      expect(pattern?.uses).toBe(1);
      expect(pattern?.avgReward).toBeGreaterThan(0);
    });
  });

  describe('ReflexionMemory', () => {
    it('should store and retrieve episodes', async () => {
      const episodeId = await agentdb.reflexionMemory.storeEpisode({
        sessionId: 'test-session-1',
        task: 'Debug memory leak in server',
        reward: 0.95,
        success: true,
        critique: 'Profiling helped identify the leak quickly'
      });

      expect(episodeId).toBeGreaterThan(0);
    });

    it('should retrieve relevant episodes', async () => {
      // Store multiple episodes
      await agentdb.reflexionMemory.storeEpisode({
        sessionId: 'session-2',
        task: 'Optimize database queries',
        reward: 0.88,
        success: true,
        critique: 'Adding indexes improved performance significantly'
      });

      await agentdb.reflexionMemory.storeEpisode({
        sessionId: 'session-2',
        task: 'Debug connection timeout',
        reward: 0.65,
        success: false,
        critique: 'Should have checked network logs first'
      });

      await agentdb.reflexionMemory.storeEpisode({
        sessionId: 'session-2',
        task: 'Fix API response time',
        reward: 0.92,
        success: true,
        critique: 'Caching strategy worked well'
      });

      // Retrieve episodes related to performance
      const results = await agentdb.reflexionMemory.retrieveRelevant({
        task: 'performance optimization',
        k: 5
      });

      expect(results.length).toBeGreaterThan(0);
      results.forEach(r => {
        expect(r.similarity).toBeGreaterThan(0);
      });
    });

    it('should filter by success', async () => {
      const successes = await agentdb.reflexionMemory.retrieveRelevant({
        task: 'debugging approach',
        k: 10,
        onlySuccesses: true
      });

      successes.forEach(r => {
        expect(r.success).toBe(true);
      });

      const failures = await agentdb.reflexionMemory.retrieveRelevant({
        task: 'debugging approach',
        k: 10,
        onlyFailures: true
      });

      failures.forEach(r => {
        expect(r.success).toBe(false);
      });
    });

    it('should get critique summary', async () => {
      const summary = await agentdb.reflexionMemory.getCritiqueSummary({
        task: 'debugging',
        k: 5
      });

      expect(typeof summary).toBe('string');
      // Summary should contain critique content if failures exist
    });

    it('should get success strategies', async () => {
      const strategies = await agentdb.reflexionMemory.getSuccessStrategies({
        task: 'optimization',
        k: 5
      });

      expect(typeof strategies).toBe('string');
      // Strategies should contain successful approach descriptions
    });
  });

  describe('Performance', () => {
    it('should have good cache hit rate', async () => {
      // Clear cache first
      agentdb.embedder.clearCache();

      // Generate some queries
      const queries = [
        'debug memory issue',
        'optimize performance',
        'debug memory issue',  // Repeat
        'fix bug',
        'optimize performance'   // Repeat
      ];

      for (const query of queries) {
        await agentdb.reasoningBank.searchPatterns({
          task: query,
          k: 3
        });
      }

      const stats = agentdb.embedder.getStats();
      expect(stats.cache.hitRate).toBeGreaterThan(0.2); // At least 20% hit rate
    });

    it('should maintain low latency with warmup', async () => {
      const stats = agentdb.embedder.getStats();

      // After warmup, average latency should be reasonable
      expect(stats.avgLatency).toBeLessThan(200); // < 200ms average
      expect(stats.warmupComplete).toBe(true);
    });
  });

  describe('Statistics', () => {
    it('should provide comprehensive stats', async () => {
      const stats = agentdb.getStats();

      expect(stats).toHaveProperty('embedder');
      expect(stats).toHaveProperty('database');

      expect(stats.embedder).toHaveProperty('totalEmbeddings');
      expect(stats.embedder).toHaveProperty('avgLatency');
      expect(stats.embedder).toHaveProperty('cache');

      expect(stats.embedder.cache).toHaveProperty('hitRate');
      expect(stats.embedder.cache).toHaveProperty('size');
    });
  });
});
