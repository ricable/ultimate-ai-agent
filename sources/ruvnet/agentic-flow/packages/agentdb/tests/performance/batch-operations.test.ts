/**
 * Performance Tests - Batch Operations
 *
 * Tests batch insert/update performance and throughput
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { BatchOperations } from '../../src/optimizations/BatchOperations.js';
import { EmbeddingService } from '../../src/controllers/EmbeddingService.js';
import { Episode } from '../../src/controllers/ReflexionMemory.js';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DB_PATH = './tests/fixtures/test-perf-batch.db';

describe('Batch Operations Performance', () => {
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
    const schemaPath = path.join(__dirname, '../../src/schemas/schema.sql');
    if (fs.existsSync(schemaPath)) {
      db.exec(fs.readFileSync(schemaPath, 'utf-8'));
    }

    const frontierSchemaPath = path.join(__dirname, '../../src/schemas/frontier-schema.sql');
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
      batchSize: 100,
      parallelism: 4,
    });
  });

  afterEach(() => {
    db.close();
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
  });

  describe('Batch Insert Performance', () => {
    it('should insert 100 episodes in <2 seconds', async () => {
      const episodes: Episode[] = Array.from({ length: 100 }, (_, i) => ({
        sessionId: `perf-${i}`,
        task: `task ${i}`,
        reward: Math.random(),
        success: Math.random() > 0.5,
      }));

      const startTime = Date.now();
      await batchOps.insertEpisodes(episodes);
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(2000);
    }, 10000);

    it('should insert 500 episodes in <8 seconds', async () => {
      const episodes: Episode[] = Array.from({ length: 500 }, (_, i) => ({
        sessionId: `perf-${i}`,
        task: `task ${i % 50}`,
        reward: Math.random(),
        success: true,
      }));

      const startTime = Date.now();
      await batchOps.insertEpisodes(episodes);
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(8000);
    }, 15000);

    it('should insert 1000 episodes in <15 seconds', async () => {
      const episodes: Episode[] = Array.from({ length: 1000 }, (_, i) => ({
        sessionId: `perf-${i}`,
        task: `task ${i % 100}`,
        reward: Math.random(),
        success: true,
      }));

      const startTime = Date.now();
      await batchOps.insertEpisodes(episodes);
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(15000);
    }, 20000);
  });

  describe('Batch Size Impact', () => {
    it('should optimize for batch size 50', async () => {
      const batchOps50 = new BatchOperations(db, embedder, {
        batchSize: 50,
        parallelism: 4,
      });

      const episodes: Episode[] = Array.from({ length: 200 }, (_, i) => ({
        sessionId: `batch50-${i}`,
        task: `task ${i}`,
        reward: Math.random(),
        success: true,
      }));

      const startTime = Date.now();
      await batchOps50.insertEpisodes(episodes);
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(5000);
    }, 10000);

    it('should optimize for batch size 200', async () => {
      const batchOps200 = new BatchOperations(db, embedder, {
        batchSize: 200,
        parallelism: 4,
      });

      const episodes: Episode[] = Array.from({ length: 400 }, (_, i) => ({
        sessionId: `batch200-${i}`,
        task: `task ${i}`,
        reward: Math.random(),
        success: true,
      }));

      const startTime = Date.now();
      await batchOps200.insertEpisodes(episodes);
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(8000);
    }, 15000);
  });

  describe('Throughput Metrics', () => {
    it('should achieve >50 inserts/second', async () => {
      const episodes: Episode[] = Array.from({ length: 500 }, (_, i) => ({
        sessionId: `throughput-${i}`,
        task: `task ${i}`,
        reward: Math.random(),
        success: true,
      }));

      const startTime = Date.now();
      await batchOps.insertEpisodes(episodes);
      const duration = Date.now() - startTime;

      const insertsPerSecond = (500 / duration) * 1000;

      expect(insertsPerSecond).toBeGreaterThan(50);
    }, 15000);
  });

  describe('Parallel Processing Performance', () => {
    it('should process 100 items in parallel efficiently', async () => {
      const items = Array.from({ length: 100 }, (_, i) => i);

      const startTime = Date.now();
      await batchOps.processInParallel(items, async (item) => {
        await new Promise(resolve => setTimeout(resolve, 10));
        return item * 2;
      });
      const duration = Date.now() - startTime;

      // With parallelism=4, should be ~4x faster than sequential
      expect(duration).toBeLessThan(400); // Sequential would be ~1000ms
    }, 5000);
  });

  describe('Regenerate Embeddings Performance', () => {
    beforeEach(async () => {
      // Seed episodes
      const stmt = db.prepare(`
        INSERT INTO episodes (session_id, task, reward, success)
        VALUES (?, ?, ?, ?)
      `);

      for (let i = 0; i < 200; i++) {
        stmt.run(`regen-${i}`, `task ${i}`, Math.random(), 1);
      }
    });

    it('should regenerate 200 embeddings in <4 seconds', async () => {
      const startTime = Date.now();
      await batchOps.regenerateEmbeddings();
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(4000);
    }, 10000);
  });
});
