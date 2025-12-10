/**
 * Performance Tests - Vector Search
 *
 * Tests vector search performance, scaling, and optimization
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { ReflexionMemory, Episode } from '../../src/controllers/ReflexionMemory.js';
import { SkillLibrary, Skill } from '../../src/controllers/SkillLibrary.js';
import { EmbeddingService } from '../../src/controllers/EmbeddingService.js';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DB_PATH = './tests/fixtures/test-perf-vector.db';

describe('Vector Search Performance', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let reflexion: ReflexionMemory;
  let skills: SkillLibrary;

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

    reflexion = new ReflexionMemory(db, embedder);
    skills = new SkillLibrary(db, embedder);
  });

  afterEach(() => {
    db.close();
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
  });

  describe('Episode Vector Search Scaling', () => {
    it('should search 100 episodes in <50ms', async () => {
      // Seed 100 episodes
      for (let i = 0; i < 100; i++) {
        await reflexion.storeEpisode({
          sessionId: `perf-${i}`,
          task: `Task variation ${i % 20}`,
          reward: Math.random(),
          success: Math.random() > 0.5,
        });
      }

      const startTime = Date.now();
      await reflexion.retrieveRelevant({ task: 'Task search', k: 10 });
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(50);
    }, 15000);

    it('should search 500 episodes in <200ms', async () => {
      // Seed 500 episodes
      for (let i = 0; i < 500; i++) {
        await reflexion.storeEpisode({
          sessionId: `perf-${i}`,
          task: `Task variation ${i % 50}`,
          reward: Math.random(),
          success: true,
        });
      }

      const startTime = Date.now();
      await reflexion.retrieveRelevant({ task: 'Task search query', k: 10 });
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(200);
    }, 30000);

    it('should search 1000 episodes in <500ms', async () => {
      // Seed 1000 episodes
      for (let i = 0; i < 1000; i++) {
        await reflexion.storeEpisode({
          sessionId: `perf-${i}`,
          task: `Task ${i % 100}`,
          reward: Math.random(),
          success: true,
        });
      }

      const startTime = Date.now();
      await reflexion.retrieveRelevant({ task: 'search', k: 10 });
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(500);
    }, 60000);
  });

  describe('Skill Vector Search Scaling', () => {
    it('should search 100 skills in <50ms', async () => {
      // Seed 100 skills
      for (let i = 0; i < 100; i++) {
        await skills.createSkill({
          name: `skill-${i}`,
          description: `Skill for ${i % 20} tasks`,
          signature: { inputs: {}, outputs: {} },
          successRate: Math.random(),
          uses: Math.floor(Math.random() * 100),
          avgReward: Math.random(),
          avgLatencyMs: Math.random() * 200,
        });
      }

      const startTime = Date.now();
      await skills.searchSkills({ task: 'task search', k: 10 });
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(50);
    }, 15000);

    it('should search 500 skills in <200ms', async () => {
      // Seed 500 skills
      for (let i = 0; i < 500; i++) {
        await skills.createSkill({
          name: `skill-${i}`,
          description: `Skill ${i}`,
          signature: { inputs: {}, outputs: {} },
          successRate: Math.random(),
          uses: Math.floor(Math.random() * 100),
          avgReward: Math.random(),
          avgLatencyMs: Math.random() * 200,
        });
      }

      const startTime = Date.now();
      await skills.searchSkills({ task: 'search', k: 10 });
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(200);
    }, 30000);
  });

  describe('Concurrent Search Performance', () => {
    beforeEach(async () => {
      // Seed episodes for concurrent tests
      for (let i = 0; i < 200; i++) {
        await reflexion.storeEpisode({
          sessionId: `concurrent-${i}`,
          task: `Task ${i % 40}`,
          reward: Math.random(),
          success: true,
        });
      }
    });

    it('should handle 10 concurrent searches efficiently', async () => {
      const queries = Array.from({ length: 10 }, (_, i) => `query ${i}`);

      const startTime = Date.now();
      await Promise.all(
        queries.map(query => reflexion.retrieveRelevant({ task: query, k: 5 }))
      );
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(300);
    }, 15000);

    it('should handle 50 concurrent searches', async () => {
      const queries = Array.from({ length: 50 }, (_, i) => `query ${i}`);

      const startTime = Date.now();
      await Promise.all(
        queries.map(query => reflexion.retrieveRelevant({ task: query, k: 5 }))
      );
      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(1000);
    }, 20000);
  });

  describe('Embedding Cache Performance', () => {
    it('should benefit from embedding cache', async () => {
      // Seed episodes
      for (let i = 0; i < 50; i++) {
        await reflexion.storeEpisode({
          sessionId: `cache-${i}`,
          task: 'repeated task',
          reward: Math.random(),
          success: true,
        });
      }

      // First search (uncached)
      const start1 = Date.now();
      await reflexion.retrieveRelevant({ task: 'search query', k: 10 });
      const uncachedDuration = Date.now() - start1;

      // Second search (cached)
      const start2 = Date.now();
      await reflexion.retrieveRelevant({ task: 'search query', k: 10 });
      const cachedDuration = Date.now() - start2;

      // Cached should be faster
      expect(cachedDuration).toBeLessThan(uncachedDuration);
    }, 15000);
  });

  describe('Throughput Metrics', () => {
    it('should achieve >20 searches/second for 100 episodes', async () => {
      // Seed episodes
      for (let i = 0; i < 100; i++) {
        await reflexion.storeEpisode({
          sessionId: `throughput-${i}`,
          task: `Task ${i % 20}`,
          reward: Math.random(),
          success: true,
        });
      }

      const searchCount = 100;
      const startTime = Date.now();

      for (let i = 0; i < searchCount; i++) {
        await reflexion.retrieveRelevant({ task: `query ${i % 10}`, k: 5 });
      }

      const duration = Date.now() - startTime;
      const searchesPerSecond = (searchCount / duration) * 1000;

      expect(searchesPerSecond).toBeGreaterThan(20);
    }, 30000);
  });
});
