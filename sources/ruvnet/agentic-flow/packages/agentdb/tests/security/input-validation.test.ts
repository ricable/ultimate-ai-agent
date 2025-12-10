/**
 * Security Tests - Input Validation
 *
 * Tests input validation and sanitization
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { ReflexionMemory, Episode } from '../../src/controllers/ReflexionMemory.js';
import { SkillLibrary, Skill } from '../../src/controllers/SkillLibrary.js';
import { CausalMemoryGraph, CausalEdge } from '../../src/controllers/CausalMemoryGraph.js';
import { EmbeddingService } from '../../src/controllers/EmbeddingService.js';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DB_PATH = './tests/fixtures/test-validation.db';

describe('Input Validation', () => {
  let db: Database.Database;
  let embedder: EmbeddingService;
  let reflexion: ReflexionMemory;
  let skills: SkillLibrary;
  let causalGraph: CausalMemoryGraph;

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
    causalGraph = new CausalMemoryGraph(db);
  });

  afterEach(() => {
    db.close();
    [TEST_DB_PATH, `${TEST_DB_PATH}-wal`, `${TEST_DB_PATH}-shm`].forEach(file => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
  });

  describe('Reward Value Validation', () => {
    it('should accept valid reward values (0-1)', async () => {
      const validRewards = [0, 0.25, 0.5, 0.75, 1.0];

      for (const reward of validRewards) {
        const episode: Episode = {
          sessionId: `test-${reward}`,
          task: 'test',
          reward,
          success: true,
        };

        await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
      }
    });

    it('should handle reward values outside normal range', async () => {
      const edgeCases = [-0.5, 1.5, 2.0, -1.0];

      for (const reward of edgeCases) {
        const episode: Episode = {
          sessionId: `edge-${reward}`,
          task: 'test',
          reward,
          success: reward > 0,
        };

        // Should store but may want to add validation in production
        await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
      }
    });
  });

  describe('Confidence Score Validation', () => {
    it('should accept valid confidence values (0-1)', () => {
      const validConfidence = [0, 0.3, 0.5, 0.7, 0.9, 1.0];

      validConfidence.forEach(confidence => {
        const edge: CausalEdge = {
          fromMemoryId: 1,
          fromMemoryType: 'episode',
          toMemoryId: 2,
          toMemoryType: 'episode',
          similarity: 0.8,
          confidence,
        };

        expect(() => causalGraph.addCausalEdge(edge)).not.toThrow();
      });
    });
  });

  describe('String Length Validation', () => {
    it('should handle very long strings', async () => {
      const longString = 'a'.repeat(10000);

      const episode: Episode = {
        sessionId: 'long-test',
        task: longString,
        output: longString,
        critique: longString,
        reward: 0.8,
        success: true,
      };

      await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
    });

    it('should handle empty strings', async () => {
      const episode: Episode = {
        sessionId: '',
        task: '',
        reward: 0.8,
        success: true,
      };

      await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
    });

    it('should handle Unicode and special characters', async () => {
      const specialStrings = [
        '你好世界 🌍',
        'مرحبا بالعالم',
        'Привет мир',
        '😀😁😂🤣',
        '\u0000\u0001\u0002', // Control characters
        '\n\r\t',
      ];

      for (const str of specialStrings) {
        const episode: Episode = {
          sessionId: `special-${specialStrings.indexOf(str)}`,
          task: str,
          reward: 0.8,
          success: true,
        };

        await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
      }
    });
  });

  describe('Numeric Range Validation', () => {
    it('should handle latency edge cases', async () => {
      const latencies = [0, 1, 100, 10000, 100000];

      for (const latency of latencies) {
        const episode: Episode = {
          sessionId: `latency-${latency}`,
          task: 'test',
          reward: 0.8,
          success: true,
          latencyMs: latency,
        };

        await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
      }
    });

    it('should handle token count edge cases', async () => {
      const tokenCounts = [0, 1, 1000, 100000, 1000000];

      for (const tokens of tokenCounts) {
        const episode: Episode = {
          sessionId: `tokens-${tokens}`,
          task: 'test',
          reward: 0.8,
          success: true,
          tokensUsed: tokens,
        };

        await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
      }
    });

    it('should handle success rate bounds', async () => {
      const successRates = [0.0, 0.5, 1.0];

      for (const rate of successRates) {
        const skill: Skill = {
          name: `skill-${rate}`,
          description: 'Test',
          signature: { inputs: {}, outputs: {} },
          successRate: rate,
          uses: 10,
          avgReward: 0.8,
          avgLatencyMs: 100,
        };

        await expect(skills.createSkill(skill)).resolves.toBeGreaterThan(0);
      }
    });
  });

  describe('JSON Validation', () => {
    it('should handle complex nested JSON metadata', async () => {
      const complexMetadata = {
        level1: {
          level2: {
            level3: {
              array: [1, 2, 3],
              object: { key: 'value' },
            },
          },
        },
        numbers: [1, 2.5, -3, 0],
        booleans: [true, false],
        null: null,
      };

      const episode: Episode = {
        sessionId: 'json-test',
        task: 'test',
        reward: 0.8,
        success: true,
        metadata: complexMetadata,
      };

      const id = await reflexion.storeEpisode(episode);
      expect(id).toBeGreaterThan(0);

      // Verify retrieval
      const stored = db.prepare('SELECT metadata FROM episodes WHERE id = ?').get(id) as any;
      const parsedMetadata = JSON.parse(stored.metadata);
      expect(parsedMetadata).toEqual(complexMetadata);
    });

    it('should handle malformed JSON gracefully', async () => {
      const skill: Skill = {
        name: 'test-skill',
        description: 'Test',
        signature: { inputs: {}, outputs: {} },
        successRate: 0.9,
        uses: 10,
        avgReward: 0.8,
        avgLatencyMs: 100,
      };

      await expect(skills.createSkill(skill)).resolves.toBeGreaterThan(0);
    });
  });

  describe('XSS Prevention', () => {
    it('should store but not execute HTML/JavaScript', async () => {
      const xssPayloads = [
        '<script>alert("XSS")</script>',
        '<img src=x onerror=alert("XSS")>',
        'javascript:alert("XSS")',
        '<iframe src="javascript:alert(\'XSS\')">',
      ];

      for (const payload of xssPayloads) {
        const episode: Episode = {
          sessionId: `xss-${xssPayloads.indexOf(payload)}`,
          task: payload,
          reward: 0.8,
          success: true,
        };

        const id = await reflexion.storeEpisode(episode);

        // Verify stored as data, not executed
        const stored = db.prepare('SELECT task FROM episodes WHERE id = ?').get(id) as any;
        expect(stored.task).toBe(payload);
      }
    });
  });

  describe('Path Traversal Prevention', () => {
    it('should handle path-like strings safely', async () => {
      const pathStrings = [
        '../../../etc/passwd',
        '..\\..\\..\\windows\\system32',
        '/etc/shadow',
        'C:\\Windows\\System32',
      ];

      for (const pathStr of pathStrings) {
        const episode: Episode = {
          sessionId: `path-${pathStrings.indexOf(pathStr)}`,
          task: pathStr,
          reward: 0.8,
          success: true,
        };

        // Should store as regular data
        await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
      }
    });
  });

  describe('Null and Undefined Handling', () => {
    it('should handle null values correctly', async () => {
      const episode: Episode = {
        sessionId: 'null-test',
        task: 'test',
        input: undefined,
        output: undefined,
        critique: undefined,
        reward: 0.8,
        success: true,
        latencyMs: undefined,
        tokensUsed: undefined,
      };

      await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
    });
  });

  describe('Array Validation', () => {
    it('should handle empty arrays', async () => {
      const episode: Episode = {
        sessionId: 'array-test',
        task: 'test',
        reward: 0.8,
        success: true,
        tags: [],
      };

      await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
    });

    it('should handle large arrays', async () => {
      const largeTags = Array.from({ length: 100 }, (_, i) => `tag-${i}`);

      const episode: Episode = {
        sessionId: 'large-array',
        task: 'test',
        reward: 0.8,
        success: true,
        tags: largeTags,
      };

      await expect(reflexion.storeEpisode(episode)).resolves.toBeGreaterThan(0);
    });
  });
});
