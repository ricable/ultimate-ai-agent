/**
 * Unit Tests for BatchOperations Optimization
 *
 * Tests bulk insert, batch processing, and database optimization
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { BatchOperations } from '../../../src/optimizations/BatchOperations.js';
import { EmbeddingService } from '../../../src/controllers/EmbeddingService.js';
import { Episode } from '../../../src/controllers/ReflexionMemory.js';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DB_PATH = './tests/fixtures/test-batch.db';

describe('BatchOperations', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let batchOps: BatchOperations;

  beforeEach(async () => {
    // Clean up
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });

    // Initialize
    db = new Database(TEST_DB_PATH);
    db.pragma('journal_mode = WAL');

    // Load schemas
    const schemaPath = path.join(__dirname, '../../../src/schemas/schema.sql');
    if (fs.existsSync(schemaPath)) {
      db.exec(fs.readFileSync(schemaPath, 'utf-8'));
    }

    const frontierSchemaPath = path.join(__dirname, '../../../src/schemas/frontier-schema.sql');
    if (fs.existsSync(frontierSchemaPath)) {
      db.exec(fs.readFileSync(frontierSchemaPath, 'utf-8'));
    }

    embedder = new EmbeddingService({
      model: 'mock-model',
      dimension: 384,
      provider: 'local',
    });
    await embedder.initialize();

    batchOps = new BatchOperations(db, embedder, {
      batchSize: 10,
      parallelism: 4,
    });
  });

  afterEach(() => {
    db.close();
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
  });

  describe('insertEpisodes', () => {
    it('should insert batch of episodes', async () => {
      const episodes: Episode[] = Array.from({ length: 25 }, (_, i) => ({
        sessionId: `session-${i}`,
        task: `task ${i}`,
        reward: Math.random(),
        success: Math.random() > 0.5,
      }));

      const count = await batchOps.insertEpisodes(episodes);

      expect(count).toBe(25);

      // Verify episodes were inserted
      const dbCount = db.prepare('SELECT COUNT(*) as count FROM episodes').get() as any;
      expect(dbCount.count).toBe(25);

      // Verify embeddings were generated
      const embCount = db.prepare('SELECT COUNT(*) as count FROM episode_embeddings').get() as any;
      expect(embCount.count).toBe(25);
    }, 10000);

    it('should handle progress callback', async () => {
      const episodes: Episode[] = Array.from({ length: 30 }, (_, i) => ({
        sessionId: `session-${i}`,
        task: `task ${i}`,
        reward: Math.random(),
        success: true,
      }));

      const progressUpdates: number[] = [];

      const customBatchOps = new BatchOperations(db, embedder, {
        batchSize: 10,
        parallelism: 4,
        progressCallback: (completed, total) => {
          progressUpdates.push(completed);
        },
      });

      await customBatchOps.insertEpisodes(episodes);

      expect(progressUpdates.length).toBeGreaterThan(0);
      expect(progressUpdates[progressUpdates.length - 1]).toBe(30);
    }, 10000);

    it('should handle large batch efficiently', async () => {
      const episodes: Episode[] = Array.from({ length: 100 }, (_, i) => ({
        sessionId: `session-${i}`,
        task: `task ${i}`,
        reward: Math.random(),
        success: Math.random() > 0.5,
      }));

      const startTime = Date.now();
      const count = await batchOps.insertEpisodes(episodes);
      const duration = Date.now() - startTime;

      expect(count).toBe(100);
      expect(duration).toBeLessThan(5000); // Should complete in less than 5 seconds
    }, 15000);

    it('should handle episodes with all fields', async () => {
      const episodes: Episode[] = [{
        sessionId: 'full-episode',
        task: 'comprehensive task',
        input: 'test input',
        output: 'test output',
        critique: 'test critique',
        reward: 0.95,
        success: true,
        latencyMs: 100,
        tokensUsed: 500,
      }];

      const count = await batchOps.insertEpisodes(episodes);

      expect(count).toBe(1);

      // Verify all fields were saved
      const saved = db.prepare('SELECT * FROM episodes WHERE session_id = ?').get('full-episode') as any;
      expect(saved.task).toBe('comprehensive task');
      expect(saved.input).toBe('test input');
      expect(saved.output).toBe('test output');
      expect(saved.critique).toBe('test critique');
    });
  });

  describe('regenerateEmbeddings', () => {
    beforeEach(async () => {
      // Seed episodes without embeddings
      const stmt = db.prepare(`
        INSERT INTO episodes (session_id, task, reward, success)
        VALUES (?, ?, ?, ?)
      `);

      for (let i = 0; i < 20; i++) {
        stmt.run(`session-${i}`, `task ${i}`, Math.random(), 1);
      }
    });

    it('should regenerate all embeddings', async () => {
      const count = await batchOps.regenerateEmbeddings();

      expect(count).toBe(20);

      // Verify embeddings were created
      const embCount = db.prepare('SELECT COUNT(*) as count FROM episode_embeddings').get() as any;
      expect(embCount.count).toBe(20);
    }, 10000);

    it('should regenerate specific embeddings', async () => {
      const episodeIds = [1, 2, 3, 4, 5];
      const count = await batchOps.regenerateEmbeddings(episodeIds);

      expect(count).toBe(5);
    });
  });

  describe('processInParallel', () => {
    it('should process items in parallel', async () => {
      const items = Array.from({ length: 20 }, (_, i) => i);

      const results = await batchOps.processInParallel(items, async (item) => {
        return item * 2;
      });

      expect(results).toHaveLength(20);
      results.forEach((result, i) => {
        expect(result).toBe(i * 2);
      });
    });

    it('should handle async processing', async () => {
      const items = Array.from({ length: 10 }, (_, i) => `item-${i}`);

      const results = await batchOps.processInParallel(items, async (item) => {
        await new Promise(resolve => setTimeout(resolve, 10));
        return item.toUpperCase();
      });

      expect(results).toHaveLength(10);
      expect(results[0]).toBe('ITEM-0');
    });
  });

  describe('bulkDelete', () => {
    beforeEach(async () => {
      // Seed data
      const stmt = db.prepare(`
        INSERT INTO episodes (session_id, task, reward, success)
        VALUES (?, ?, ?, ?)
      `);

      for (let i = 0; i < 20; i++) {
        stmt.run(`session-${i % 5}`, `task ${i}`, Math.random(), i > 10 ? 1 : 0);
      }
    });

    it('should bulk delete by condition', () => {
      const deleted = batchOps.bulkDelete('episodes', { success: 0 });

      expect(deleted).toBeGreaterThan(0);

      // Verify deletions
      const remaining = db.prepare('SELECT COUNT(*) as count FROM episodes WHERE success = 0').get() as any;
      expect(remaining.count).toBe(0);
    });
  });

  describe('bulkUpdate', () => {
    beforeEach(async () => {
      // Seed data
      const stmt = db.prepare(`
        INSERT INTO episodes (session_id, task, reward, success)
        VALUES (?, ?, ?, ?)
      `);

      for (let i = 0; i < 20; i++) {
        stmt.run(`session-${i}`, `task ${i}`, Math.random(), 1);
      }
    });

    it('should bulk update by condition', () => {
      const updated = batchOps.bulkUpdate(
        'episodes',
        { success: 0 },
        { session_id: 'session-5' }
      );

      expect(updated).toBeGreaterThan(0);

      // Verify update
      const record = db.prepare('SELECT * FROM episodes WHERE session_id = ?').get('session-5') as any;
      expect(record.success).toBe(0);
    });
  });

  describe('optimize', () => {
    beforeEach(async () => {
      // Add some data
      const stmt = db.prepare(`
        INSERT INTO episodes (session_id, task, reward, success)
        VALUES (?, ?, ?, ?)
      `);

      for (let i = 0; i < 100; i++) {
        stmt.run(`session-${i}`, `task ${i}`, Math.random(), 1);
      }
    });

    it('should optimize database', () => {
      expect(() => batchOps.optimize()).not.toThrow();
    });
  });

  describe('getStats', () => {
    beforeEach(async () => {
      // Add some data
      const episodeStmt = db.prepare(`
        INSERT INTO episodes (session_id, task, reward, success)
        VALUES (?, ?, ?, ?)
      `);

      for (let i = 0; i < 50; i++) {
        episodeStmt.run(`session-${i}`, `task ${i}`, Math.random(), 1);
      }
    });

    it('should return database statistics', () => {
      const stats = batchOps.getStats();

      expect(stats).toHaveProperty('totalSize');
      expect(stats).toHaveProperty('tableStats');
      expect(stats.totalSize).toBeGreaterThan(0);
      expect(stats.tableStats).toBeInstanceOf(Array);
    });

    it('should include table row counts', () => {
      const stats = batchOps.getStats();

      const episodeStats = stats.tableStats.find(t => t.name === 'episodes');
      expect(episodeStats).toBeDefined();
      expect(episodeStats!.rows).toBe(50);
    });
  });

  describe('Performance', () => {
    it('should insert 1000 episodes efficiently', async () => {
      const episodes: Episode[] = Array.from({ length: 1000 }, (_, i) => ({
        sessionId: `perf-${i}`,
        task: `performance test ${i}`,
        reward: Math.random(),
        success: Math.random() > 0.5,
      }));

      const startTime = Date.now();
      const count = await batchOps.insertEpisodes(episodes);
      const duration = Date.now() - startTime;

      expect(count).toBe(1000);
      expect(duration).toBeLessThan(15000); // Should complete in less than 15 seconds
    }, 20000);
  });
});
