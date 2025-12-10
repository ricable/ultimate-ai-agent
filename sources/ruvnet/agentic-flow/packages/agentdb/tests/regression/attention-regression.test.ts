/**
 * @test Attention Mechanism Regression Tests
 * @description Ensure attention integration doesn't break existing AgentDB functionality
 * @prerequisites
 *   - Baseline AgentDB functionality established
 *   - All existing tests passing
 * @coverage
 *   - Backward compatibility
 *   - Feature flag behavior
 *   - Existing API stability
 *   - Performance regression
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AgentDB } from '../../src/index';
import { MemoryController } from '../../src/controllers/MemoryController';
import { ReflexionMemory } from '../../src/controllers/ReflexionMemory';
import { SkillLibrary } from '../../src/controllers/SkillLibrary';
import fs from 'fs';
import path from 'path';

// TODO: These tests require MemoryController which is not yet implemented
describe.todo('Attention Mechanism Regression Tests', () => {
  let db: AgentDB;
  const testDbPath = path.join(__dirname, '../fixtures/test-regression.db');

  describe('Backward Compatibility - Attention Disabled', () => {
    beforeEach(async () => {
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }

      // Initialize WITHOUT attention features
      db = new AgentDB({
        dbPath: testDbPath,
        namespace: 'regression-test',
        enableAttention: false
      });

      await db.initialize();
    });

    afterEach(async () => {
      await db.close();
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }
    });

    it('should initialize AgentDB without attention controllers', async () => {
      const controllers = db.listControllers();

      expect(controllers).not.toContain('self-attention');
      expect(controllers).not.toContain('cross-attention');
      expect(controllers).not.toContain('multi-head-attention');
    });

    it('should store and retrieve memories normally', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      const memory = {
        id: 'test-memory',
        content: 'Test content',
        embedding: [0.1, 0.2, 0.3]
      };

      await memoryController.store(memory);
      const retrieved = await memoryController.retrieve(memory.id);

      expect(retrieved).toBeDefined();
      expect(retrieved.id).toBe(memory.id);
      expect(retrieved.content).toBe(memory.content);
    });

    it('should perform vector search without attention', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      await memoryController.store({
        id: 'm1',
        embedding: [0.1, 0.2, 0.3]
      });

      await memoryController.store({
        id: 'm2',
        embedding: [0.4, 0.5, 0.6]
      });

      const query = [0.1, 0.2, 0.3];
      const results = await memoryController.search(query, { topK: 2 });

      expect(results).toHaveLength(2);
      expect(results[0]).toHaveProperty('score');
      expect(results[0]).not.toHaveProperty('attentionScore');
    });

    it('should maintain existing ReflexionMemory functionality', async () => {
      const reflexion = db.getController('reflexion') as ReflexionMemory;

      const trajectory = {
        sessionId: 'session1',
        task: 'Complete task',
        steps: ['step1', 'step2'],
        reward: 0.8,
        success: true
      };

      await reflexion.storeTrajectory(trajectory);
      const retrieved = await reflexion.getTrajectory('session1');

      expect(retrieved).toBeDefined();
      expect(retrieved.reward).toBe(0.8);
      expect(retrieved.success).toBe(true);
    });

    it('should maintain existing SkillLibrary functionality', async () => {
      const skillLib = db.getController('skills') as SkillLibrary;

      const skill = {
        name: 'test-skill',
        description: 'Test skill',
        code: 'function test() { return true; }',
        successRate: 0.9
      };

      await skillLib.storeSkill(skill);
      const retrieved = await skillLib.getSkill('test-skill');

      expect(retrieved).toBeDefined();
      expect(retrieved.name).toBe(skill.name);
      expect(retrieved.successRate).toBe(skill.successRate);
    });

    it('should not impact database schema', async () => {
      const tables = await db.query('SELECT name FROM sqlite_master WHERE type="table"');

      // Should not have attention-specific tables when disabled
      const tableNames = tables.map(t => t.name);
      expect(tableNames).not.toContain('attention_scores');
      expect(tableNames).not.toContain('attention_cache');
    });
  });

  describe('Feature Flag Behavior - Attention Enabled', () => {
    beforeEach(async () => {
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }

      // Initialize WITH attention features
      db = new AgentDB({
        dbPath: testDbPath,
        namespace: 'feature-flag-test',
        enableAttention: true,
        attentionConfig: {
          selfAttention: { enabled: true },
          crossAttention: { enabled: true },
          multiHeadAttention: { enabled: true }
        }
      });

      await db.initialize();
    });

    afterEach(async () => {
      await db.close();
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }
    });

    it('should initialize attention controllers when enabled', async () => {
      const controllers = db.listControllers();

      expect(controllers).toContain('self-attention');
      expect(controllers).toContain('cross-attention');
      expect(controllers).toContain('multi-head-attention');
    });

    it('should enhance memory retrieval with attention scores', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      await memoryController.store({
        id: 'enhanced-mem',
        embedding: [0.1, 0.2, 0.3]
      });

      const query = [0.1, 0.2, 0.3];
      const results = await memoryController.retrieveWithAttention(query);

      expect(results[0]).toHaveProperty('attentionScore');
      expect(results[0].attentionScore).toBeGreaterThanOrEqual(0);
    });

    it('should still support legacy search API', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      await memoryController.store({
        id: 'legacy-search',
        embedding: [0.1, 0.2, 0.3]
      });

      const query = [0.1, 0.2, 0.3];
      const results = await memoryController.search(query);

      expect(results).toBeDefined();
      expect(results.length).toBeGreaterThan(0);
    });

    it('should selectively enable attention mechanisms', async () => {
      await db.close();
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }

      // Only self-attention enabled
      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true,
        attentionConfig: {
          selfAttention: { enabled: true },
          crossAttention: { enabled: false },
          multiHeadAttention: { enabled: false }
        }
      });

      await db.initialize();

      const controllers = db.listControllers();
      expect(controllers).toContain('self-attention');
      expect(controllers).not.toContain('cross-attention');
      expect(controllers).not.toContain('multi-head-attention');
    });
  });

  describe('API Stability', () => {
    beforeEach(async () => {
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }

      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true
      });

      await db.initialize();
    });

    afterEach(async () => {
      await db.close();
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }
    });

    it('should maintain stable AgentDB initialization API', async () => {
      expect(db).toBeInstanceOf(AgentDB);
      expect(db.initialize).toBeInstanceOf(Function);
      expect(db.getController).toBeInstanceOf(Function);
      expect(db.query).toBeInstanceOf(Function);
      expect(db.close).toBeInstanceOf(Function);
    });

    it('should maintain stable MemoryController API', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      expect(memoryController.store).toBeInstanceOf(Function);
      expect(memoryController.retrieve).toBeInstanceOf(Function);
      expect(memoryController.search).toBeInstanceOf(Function);
      expect(memoryController.delete).toBeInstanceOf(Function);
      expect(memoryController.update).toBeInstanceOf(Function);
    });

    it('should not break existing method signatures', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      // Test that all existing parameters still work
      const memory = {
        id: 'api-test',
        content: 'Test',
        embedding: [0.1, 0.2, 0.3],
        metadata: { key: 'value' }
      };

      await expect(memoryController.store(memory)).resolves.not.toThrow();
      await expect(memoryController.retrieve('api-test')).resolves.toBeDefined();
      await expect(memoryController.search([0.1, 0.2, 0.3])).resolves.toBeDefined();
    });

    it('should maintain backward-compatible search options', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      await memoryController.store({
        id: 'compat-test',
        embedding: [0.1, 0.2, 0.3]
      });

      // Old-style options should still work
      const results = await memoryController.search([0.1, 0.2, 0.3], {
        topK: 10,
        threshold: 0.5,
        filter: { /* metadata filter */ }
      });

      expect(results).toBeDefined();
    });
  });

  describe('Performance Regression', () => {
    beforeEach(async () => {
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }
    });

    afterEach(async () => {
      if (db) {
        await db.close();
      }
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }
    });

    it('should not slow down initialization', async () => {
      const start = performance.now();

      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true
      });
      await db.initialize();

      const duration = performance.now() - start;

      // Should initialize in under 1 second
      expect(duration).toBeLessThan(1000);
    });

    it('should not impact memory storage performance', async () => {
      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true
      });
      await db.initialize();

      const memoryController = db.getController('memory') as MemoryController;

      const start = performance.now();

      for (let i = 0; i < 100; i++) {
        await memoryController.store({
          id: `perf-test-${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        });
      }

      const duration = performance.now() - start;

      // Should store 100 items in under 2 seconds
      expect(duration).toBeLessThan(2000);
    });

    it('should not increase memory footprint significantly', async () => {
      const initialMemory = process.memoryUsage().heapUsed;

      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true
      });
      await db.initialize();

      const memoryController = db.getController('memory') as MemoryController;

      // Store 1000 items
      for (let i = 0; i < 1000; i++) {
        await memoryController.store({
          id: `mem-footprint-${i}`,
          embedding: Array(128).fill(0).map(() => Math.random())
        });
      }

      global.gc && global.gc();
      const finalMemory = process.memoryUsage().heapUsed;
      const increase = finalMemory - initialMemory;

      // Should not use more than 100MB
      expect(increase).toBeLessThan(100 * 1024 * 1024);
    });

    it('should not degrade search performance', async () => {
      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: false
      });
      await db.initialize();

      const memoryController = db.getController('memory') as MemoryController;

      // Store test data
      for (let i = 0; i < 500; i++) {
        await memoryController.store({
          id: `search-perf-${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        });
      }

      const query = [0.5, 0.5, 0.5];

      // Baseline without attention
      const start1 = performance.now();
      await memoryController.search(query);
      const baseline = performance.now() - start1;

      await db.close();
      fs.unlinkSync(testDbPath);

      // With attention enabled
      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true
      });
      await db.initialize();

      const memoryController2 = db.getController('memory') as MemoryController;

      for (let i = 0; i < 500; i++) {
        await memoryController2.store({
          id: `search-perf-${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        });
      }

      const start2 = performance.now();
      await memoryController2.search(query);
      const withAttention = performance.now() - start2;

      // With attention should not be more than 2x slower
      expect(withAttention).toBeLessThan(baseline * 2);
    });
  });

  describe('Database Migration', () => {
    it('should upgrade existing database to support attention', async () => {
      // Create old database without attention
      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: false
      });
      await db.initialize();

      const memoryController = db.getController('memory') as MemoryController;
      await memoryController.store({
        id: 'migration-test',
        embedding: [0.1, 0.2, 0.3]
      });

      await db.close();

      // Reopen with attention enabled
      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true
      });
      await db.initialize();

      // Should still be able to retrieve old data
      const memoryController2 = db.getController('memory') as MemoryController;
      const retrieved = await memoryController2.retrieve('migration-test');

      expect(retrieved).toBeDefined();
      expect(retrieved.id).toBe('migration-test');

      // Should be able to use attention features
      const results = await memoryController2.retrieveWithAttention([0.1, 0.2, 0.3]);
      expect(results).toBeDefined();
    });

    it('should preserve data integrity during migration', async () => {
      // Create database with sample data
      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: false
      });
      await db.initialize();

      const memoryController = db.getController('memory') as MemoryController;

      const memories = [];
      for (let i = 0; i < 100; i++) {
        const memory = {
          id: `integrity-${i}`,
          content: `Memory ${i}`,
          embedding: [Math.random(), Math.random(), Math.random()]
        };
        await memoryController.store(memory);
        memories.push(memory);
      }

      await db.close();

      // Reopen with attention
      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true
      });
      await db.initialize();

      const memoryController2 = db.getController('memory') as MemoryController;

      // Verify all data is intact
      for (const memory of memories) {
        const retrieved = await memoryController2.retrieve(memory.id);
        expect(retrieved).toBeDefined();
        expect(retrieved.id).toBe(memory.id);
        expect(retrieved.content).toBe(memory.content);
      }
    });
  });

  describe('Error Handling Stability', () => {
    beforeEach(async () => {
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }

      db = new AgentDB({
        dbPath: testDbPath,
        enableAttention: true
      });
      await db.initialize();
    });

    afterEach(async () => {
      await db.close();
      if (fs.existsSync(testDbPath)) {
        fs.unlinkSync(testDbPath);
      }
    });

    it('should handle missing attention controllers gracefully', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      await memoryController.store({
        id: 'error-test',
        embedding: [0.1, 0.2, 0.3]
      });

      // Even if attention fails, basic search should work
      const results = await memoryController.search([0.1, 0.2, 0.3]);
      expect(results).toBeDefined();
    });

    it('should maintain transaction integrity with attention', async () => {
      const memoryController = db.getController('memory') as MemoryController;

      await db.beginTransaction();

      try {
        await memoryController.store({
          id: 'tx-test-1',
          embedding: [0.1, 0.2, 0.3]
        });

        await memoryController.store({
          id: 'tx-test-2',
          embedding: [0.4, 0.5, 0.6]
        });

        await db.commitTransaction();
      } catch (error) {
        await db.rollbackTransaction();
        throw error;
      }

      const result1 = await memoryController.retrieve('tx-test-1');
      const result2 = await memoryController.retrieve('tx-test-2');

      expect(result1).toBeDefined();
      expect(result2).toBeDefined();
    });
  });
});
