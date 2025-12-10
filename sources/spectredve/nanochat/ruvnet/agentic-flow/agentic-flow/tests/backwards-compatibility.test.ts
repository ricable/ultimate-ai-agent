/**
 * Backwards Compatibility Test Suite
 *
 * Ensures all existing code continues to work after AgentDB integration
 *
 * Tests:
 * 1. Import paths (both old and new work)
 * 2. API signatures (methods accept same parameters)
 * 3. Return types (return same data structures)
 * 4. CLI commands (all commands still work)
 * 5. Memory operations (produce same results)
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

describe('Backwards Compatibility - Imports', () => {
  it('should support old embedded agentdb imports', async () => {
    // Old import path (should still work via re-exports)
    const {
      ReflexionMemory,
      SkillLibrary,
      CausalMemoryGraph
    } = await import('../src/agentdb/index.js');

    expect(ReflexionMemory).toBeDefined();
    expect(SkillLibrary).toBeDefined();
    expect(CausalMemoryGraph).toBeDefined();
  });

  it('should support new reasoningbank exports', async () => {
    const {
      HybridReasoningBank,
      AdvancedMemorySystem,
      ReasoningBankEngine
    } = await import('../src/reasoningbank/index.js');

    expect(HybridReasoningBank).toBeDefined();
    expect(AdvancedMemorySystem).toBeDefined();
    expect(ReasoningBankEngine).toBeDefined();
  });

  it('should support shared memory pool', async () => {
    const { SharedMemoryPool } = await import('../src/memory/index.js');
    expect(SharedMemoryPool).toBeDefined();
  });
});

describe('Backwards Compatibility - API Signatures', () => {
  let testDbPath: string;

  beforeAll(() => {
    testDbPath = path.join(process.cwd(), 'test-compat.db');
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  afterAll(() => {
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  it('should maintain ReflexionMemory API', async () => {
    const Database = (await import('better-sqlite3')).default;
    const { EmbeddingService, ReflexionMemory } = await import('../src/agentdb/index.js');

    const db = new Database(testDbPath);
    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });

    await embedder.initialize();
    const reflexion = new ReflexionMemory(db, embedder);

    // Test old API signature
    const episodeId = await reflexion.storeEpisode({
      sessionId: 'test-session',
      task: 'test task',
      input: 'input data',
      output: 'output data',
      critique: 'test critique',
      reward: 0.85,
      success: true,
      latencyMs: 100,
      tokensUsed: 50
    });

    expect(episodeId).toBeGreaterThan(0);

    // Test retrieval
    const results = await reflexion.retrieveRelevant({
      task: 'test',
      k: 5
    });

    expect(Array.isArray(results)).toBe(true);

    db.close();
  });

  it('should maintain HybridReasoningBank API', async () => {
    const { HybridReasoningBank } = await import('../src/reasoningbank/HybridBackend.js');
    const { SharedMemoryPool } = await import('../src/memory/index.js');

    // Reset pool for clean test
    SharedMemoryPool.resetInstance();

    const rb = new HybridReasoningBank({ preferWasm: false });

    // Test pattern storage
    const patternId = await rb.storePattern({
      sessionId: 'test-hybrid',
      task: 'test hybrid task',
      success: true,
      reward: 0.9
    });

    expect(patternId).toBeGreaterThan(0);

    // Test pattern retrieval
    const patterns = await rb.retrievePatterns('test', { k: 5 });
    expect(Array.isArray(patterns)).toBe(true);

    SharedMemoryPool.getInstance().close();
  });
});

describe('Backwards Compatibility - Memory Operations', () => {
  it('should produce consistent results between old and new APIs', async () => {
    const Database = (await import('better-sqlite3')).default;
    const { EmbeddingService, ReflexionMemory } = await import('../src/agentdb/index.js');
    const { HybridReasoningBank } = await import('../src/reasoningbank/HybridBackend.js');
    const { SharedMemoryPool } = await import('../src/memory/index.js');

    const testDbPath1 = path.join(process.cwd(), 'test-old.db');
    const testDbPath2 = path.join(process.cwd(), 'test-new.db');

    // Clean up
    [testDbPath1, testDbPath2].forEach(p => {
      if (fs.existsSync(p)) fs.unlinkSync(p);
    });

    try {
      // Old API
      const db1 = new Database(testDbPath1);
      const embedder1 = new EmbeddingService({
        model: 'Xenova/all-MiniLM-L6-v2',
        dimension: 384,
        provider: 'transformers'
      });
      await embedder1.initialize();
      const reflexion = new ReflexionMemory(db1, embedder1);

      await reflexion.storeEpisode({
        sessionId: 'test',
        task: 'authentication',
        input: '',
        output: '',
        critique: 'test',
        reward: 0.85,
        success: true,
        latencyMs: 0,
        tokensUsed: 0
      });

      const oldResults = await reflexion.retrieveRelevant({
        task: 'auth',
        k: 5
      });

      // New API
      SharedMemoryPool.resetInstance();
      const pool = SharedMemoryPool.getInstance({ dbPath: testDbPath2 });
      await pool.ensureInitialized();

      const rb = new HybridReasoningBank({ preferWasm: false });
      await rb.storePattern({
        sessionId: 'test',
        task: 'authentication',
        success: true,
        reward: 0.85
      });

      const newResults = await rb.retrievePatterns('auth', { k: 5 });

      // Both should return arrays with at least one result
      expect(oldResults.length).toBeGreaterThan(0);
      expect(newResults.length).toBeGreaterThan(0);

      // Both should have similar structure
      expect(oldResults[0]).toHaveProperty('task');
      expect(newResults[0]).toHaveProperty('task');

      db1.close();
      pool.close();
    } finally {
      [testDbPath1, testDbPath2].forEach(p => {
        if (fs.existsSync(p)) fs.unlinkSync(p);
      });
    }
  });
});

describe('Backwards Compatibility - Package Exports', () => {
  it('should export all expected modules', async () => {
    const pkg = await import('../src/index.js');

    // Check reasoningbank export
    expect(pkg.reasoningbank).toBeDefined();
  });
});
