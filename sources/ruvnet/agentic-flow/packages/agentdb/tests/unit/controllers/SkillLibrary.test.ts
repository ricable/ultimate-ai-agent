/**
 * Unit Tests for SkillLibrary Controller
 *
 * Tests skill management, consolidation, and pattern extraction
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { SkillLibrary, Skill } from '../../../src/controllers/SkillLibrary.js';
import { ReflexionMemory, Episode } from '../../../src/controllers/ReflexionMemory.js';
import { EmbeddingService } from '../../../src/controllers/EmbeddingService.js';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DB_PATH = './tests/fixtures/test-skills.db';

describe('SkillLibrary', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let skills: SkillLibrary;
  let reflexion: ReflexionMemory;

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

    skills = new SkillLibrary(db, embedder);
    reflexion = new ReflexionMemory(db, embedder);
  });

  afterEach(() => {
    db.close();
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
  });

  describe('createSkill', () => {
    it('should create skill with all fields', async () => {
      const skill: Skill = {
        name: 'jwt_auth',
        description: 'Generate JWT tokens with 24h expiry',
        signature: {
          inputs: { user: 'object' },
          outputs: { token: 'string' },
        },
        code: 'const jwt = require("jsonwebtoken"); return jwt.sign(payload, secret, {expiresIn: "24h"});',
        successRate: 0.95,
        uses: 50,
        avgReward: 0.92,
        avgLatencyMs: 120,
      };

      const skillId = await skills.createSkill(skill);

      expect(skillId).toBeGreaterThan(0);
      expect(typeof skillId).toBe('number');
    });

    it('should create skill with minimal fields', async () => {
      const skill: Skill = {
        name: 'simple_skill',
        description: 'A simple test skill',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.0,
        uses: 0,
        avgReward: 0.0,
        avgLatencyMs: 0.0,
      };

      const skillId = await skills.createSkill(skill);

      expect(skillId).toBeGreaterThan(0);
    });

    it('should generate and store skill embeddings', async () => {
      const skill: Skill = {
        name: 'test_skill',
        description: 'Test skill for embedding verification',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.8,
        uses: 10,
        avgReward: 0.75,
        avgLatencyMs: 100,
      };

      const skillId = await skills.createSkill(skill);

      // Verify embedding was stored
      const embedding = db.prepare('SELECT embedding FROM skill_embeddings WHERE skill_id = ?')
        .get(skillId) as any;

      expect(embedding).toBeDefined();
      expect(embedding.embedding).toBeInstanceOf(Buffer);
    });
  });

  describe('searchSkills', () => {
    beforeEach(async () => {
      // Seed test skills
      const testSkills: Skill[] = [
        {
          name: 'jwt_auth',
          description: 'JWT authentication with tokens',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.95,
          uses: 100,
          avgReward: 0.92,
          avgLatencyMs: 80,
        },
        {
          name: 'oauth2_login',
          description: 'OAuth2 login flow with PKCE',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.88,
          uses: 50,
          avgReward: 0.85,
          avgLatencyMs: 150,
        },
        {
          name: 'database_query',
          description: 'Execute database queries safely',
          signature: { inputs: {}, outputs: {} },
          successRate: 0.75,
          uses: 200,
          avgReward: 0.70,
          avgLatencyMs: 50,
        },
      ];

      for (const skill of testSkills) {
        await skills.createSkill(skill);
      }
    });

    it('should search skills by semantic similarity', async () => {
      const results = await skills.searchSkills({
        task: 'authentication security',
        k: 5,
      });

      expect(results.length).toBeGreaterThan(0);
      expect(results.length).toBeLessThanOrEqual(5);
      expect(results[0]).toHaveProperty('id');
      expect(results[0]).toHaveProperty('name');
      expect(results[0]).toHaveProperty('similarity');
    });

    it('should filter by minimum success rate', async () => {
      const results = await skills.searchSkills({
        task: 'authentication',
        k: 10,
        minSuccessRate: 0.9,
      });

      results.forEach(skill => {
        expect(skill.successRate).toBeGreaterThanOrEqual(0.9);
      });
    });

    it('should return top-k results', async () => {
      const results = await skills.searchSkills({
        task: 'database',
        k: 2,
      });

      expect(results.length).toBeLessThanOrEqual(2);
    });
  });

  describe('updateSkillStats', () => {
    it('should update skill statistics after use', async () => {
      const skillId = await skills.createSkill({
        name: 'test_skill',
        description: 'Test skill',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.8,
        uses: 10,
        avgReward: 0.75,
        avgLatencyMs: 100,
      });

      // Update with successful use
      skills.updateSkillStats(skillId, true, 0.9, 90);

      // Verify updated stats
      const updated = db.prepare('SELECT * FROM skills WHERE id = ?').get(skillId) as any;

      expect(updated.uses).toBe(11);
      expect(updated.success_rate).toBeGreaterThan(0.8);
      expect(updated.avg_reward).toBeGreaterThan(0.75);
    });

    it('should update stats with failure', async () => {
      const skillId = await skills.createSkill({
        name: 'test_skill',
        description: 'Test skill',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.9,
        uses: 10,
        avgReward: 0.85,
        avgLatencyMs: 100,
      });

      // Update with failure
      skills.updateSkillStats(skillId, false, 0.2, 200);

      // Verify updated stats
      const updated = db.prepare('SELECT * FROM skills WHERE id = ?').get(skillId) as any;

      expect(updated.uses).toBe(11);
      expect(updated.success_rate).toBeLessThan(0.9);
      expect(updated.avg_reward).toBeLessThan(0.85);
    });
  });

  describe('consolidateEpisodesIntoSkills', () => {
    beforeEach(async () => {
      // Seed successful episodes for consolidation
      for (let i = 0; i < 5; i++) {
        await reflexion.storeEpisode({
          sessionId: `consolidate-${i}`,
          task: 'implement_rest_api',
          output: 'Created REST API with proper error handling',
          reward: 0.85 + i * 0.02,
          success: true,
        });
      }

      for (let i = 0; i < 4; i++) {
        await reflexion.storeEpisode({
          sessionId: `db-${i}`,
          task: 'optimize_database',
          output: 'Added indexes and query optimization',
          reward: 0.8 + i * 0.03,
          success: true,
        });
      }
    });

    it('should consolidate high-reward episodes into skills', async () => {
      const result = await skills.consolidateEpisodesIntoSkills({
        minAttempts: 3,
        minReward: 0.7,
        timeWindowDays: 7,
      });

      expect(result.created).toBeGreaterThan(0);
      expect(typeof result.created).toBe('number');
      expect(typeof result.updated).toBe('number');
    });

    it('should extract patterns when enabled', async () => {
      const result = await skills.consolidateEpisodesIntoSkills({
        minAttempts: 3,
        minReward: 0.7,
        extractPatterns: true,
      });

      expect(result.patterns).toBeInstanceOf(Array);
      if (result.patterns.length > 0) {
        expect(result.patterns[0]).toHaveProperty('task');
        expect(result.patterns[0]).toHaveProperty('commonPatterns');
        expect(result.patterns[0]).toHaveProperty('successIndicators');
      }
    });

    it('should respect minimum attempts threshold', async () => {
      const result = await skills.consolidateEpisodesIntoSkills({
        minAttempts: 100, // Very high threshold
        minReward: 0.7,
      });

      expect(result.created).toBe(0); // Should not create any skills
    });

    it('should update existing skills instead of duplicating', async () => {
      // First consolidation
      const result1 = await skills.consolidateEpisodesIntoSkills({
        minAttempts: 3,
        minReward: 0.7,
      });

      // Second consolidation
      const result2 = await skills.consolidateEpisodesIntoSkills({
        minAttempts: 3,
        minReward: 0.7,
      });

      expect(result2.created).toBe(0);
      expect(result2.updated).toBeGreaterThan(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle skill with empty description', async () => {
      const skill: Skill = {
        name: 'no_description',
        description: '',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.8,
        uses: 10,
        avgReward: 0.75,
        avgLatencyMs: 100,
      };

      const skillId = await skills.createSkill(skill);

      expect(skillId).toBeGreaterThan(0);
    });

    it('should handle skill with complex signature', async () => {
      const skill: Skill = {
        name: 'complex_skill',
        description: 'Complex skill with nested types',
        signature: {
          inputs: {
            user: { id: 'string', name: 'string', roles: 'array' },
            options: { timeout: 'number', retries: 'number' },
          },
          outputs: {
            result: { status: 'string', data: 'any' },
          },
        },
        successRate: 0.9,
        uses: 20,
        avgReward: 0.85,
        avgLatencyMs: 150,
      };

      const skillId = await skills.createSkill(skill);

      expect(skillId).toBeGreaterThan(0);
    });

    it('should handle very long skill names', async () => {
      const skill: Skill = {
        name: 'a'.repeat(200),
        description: 'Long name skill',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.8,
        uses: 5,
        avgReward: 0.7,
        avgLatencyMs: 100,
      };

      const skillId = await skills.createSkill(skill);

      expect(skillId).toBeGreaterThan(0);
    });
  });

  describe('Performance', () => {
    it('should create 50 skills efficiently', async () => {
      const startTime = Date.now();

      for (let i = 0; i < 50; i++) {
        await skills.createSkill({
          name: `skill_${i}`,
          description: `Test skill ${i}`,
          signature: { inputs: {}, outputs: {} },
          successRate: Math.random(),
          uses: Math.floor(Math.random() * 100),
          avgReward: Math.random(),
          avgLatencyMs: Math.random() * 200,
        });
      }

      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(3000); // Should complete in less than 3 seconds
    }, 10000);

    it('should search skills efficiently', async () => {
      // Seed skills
      for (let i = 0; i < 30; i++) {
        await skills.createSkill({
          name: `skill_${i}`,
          description: `Test skill ${i} for searching`,
          signature: { inputs: {}, outputs: {} },
          successRate: Math.random(),
          uses: Math.floor(Math.random() * 100),
          avgReward: Math.random(),
          avgLatencyMs: Math.random() * 200,
        });
      }

      const startTime = Date.now();

      await skills.searchSkills({
        task: 'test search',
        k: 10,
      });

      const duration = Date.now() - startTime;

      expect(duration).toBeLessThan(200); // Should complete in less than 200ms
    });
  });
});
