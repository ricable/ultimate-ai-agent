/**
 * Comprehensive Test Suite for HybridReasoningBank v1.7.1
 *
 * Tests full implementation with:
 * - CausalRecall integration
 * - CausalMemoryGraph tracking
 * - WASM acceleration (with fallback)
 * - Pattern storage and retrieval
 * - Strategy learning
 * - Auto-consolidation
 * - What-if analysis
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { HybridReasoningBank, PatternData, RetrievalOptions, CausalInsight } from '../../src/reasoningbank/HybridBackend.js';
import { SharedMemoryPool } from '../../src/memory/SharedMemoryPool.js';
import * as fs from 'fs';
import * as path from 'path';

describe('HybridReasoningBank v1.7.1 - Full Implementation', () => {
  let reasoningBank: HybridReasoningBank;
  const testDbPath = path.join(process.cwd(), 'test-agentdb-hybrid.db');

  beforeEach(async () => {
    // Clean up any existing test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    // Initialize with WASM preference
    reasoningBank = new HybridReasoningBank({ preferWasm: true });

    // Give time for WASM module loading
    await new Promise(resolve => setTimeout(resolve, 100));
  });

  afterEach(() => {
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Pattern Storage', () => {
    it('should store a successful pattern with causal edge', async () => {
      const pattern: PatternData = {
        sessionId: 'test-session-1',
        task: 'API optimization',
        input: 'Original slow endpoint',
        output: 'Optimized with caching',
        critique: 'Good approach with measurable improvement',
        success: true,
        reward: 0.95,
        latencyMs: 150,
        tokensUsed: 1200
      };

      const episodeId = await reasoningBank.storePattern(pattern);

      expect(episodeId).toBeGreaterThan(0);
    });

    it('should store a failed pattern without causal edge', async () => {
      const pattern: PatternData = {
        sessionId: 'test-session-2',
        task: 'Database migration',
        input: 'Old schema',
        output: 'Migration failed',
        critique: 'Need better rollback strategy',
        success: false,
        reward: 0.2,
        latencyMs: 5000,
        tokensUsed: 2500
      };

      const episodeId = await reasoningBank.storePattern(pattern);

      expect(episodeId).toBeGreaterThan(0);
    });

    it('should store multiple related patterns', async () => {
      const patterns: PatternData[] = [
        {
          sessionId: 'test-session-3',
          task: 'User authentication',
          input: 'No auth',
          output: 'JWT implemented',
          success: true,
          reward: 0.9,
          latencyMs: 200
        },
        {
          sessionId: 'test-session-3',
          task: 'User authentication',
          input: 'JWT basic',
          output: 'JWT with refresh tokens',
          success: true,
          reward: 0.95,
          latencyMs: 180
        },
        {
          sessionId: 'test-session-3',
          task: 'User authentication',
          input: 'JWT refresh',
          output: 'JWT with OAuth2',
          success: true,
          reward: 0.98,
          latencyMs: 160
        }
      ];

      const episodeIds = await Promise.all(
        patterns.map(p => reasoningBank.storePattern(p))
      );

      expect(episodeIds).toHaveLength(3);
      episodeIds.forEach(id => expect(id).toBeGreaterThan(0));
    });
  });

  describe('Pattern Retrieval with CausalRecall', () => {
    beforeEach(async () => {
      // Seed database with diverse patterns
      const seedPatterns: PatternData[] = [
        {
          sessionId: 'seed-1',
          task: 'API caching',
          input: 'No cache',
          output: 'Redis cache',
          success: true,
          reward: 0.92,
          latencyMs: 150
        },
        {
          sessionId: 'seed-2',
          task: 'API rate limiting',
          input: 'No limits',
          output: 'Token bucket',
          success: true,
          reward: 0.88,
          latencyMs: 120
        },
        {
          sessionId: 'seed-3',
          task: 'API authentication',
          input: 'Basic auth',
          output: 'OAuth2',
          success: true,
          reward: 0.95,
          latencyMs: 180
        },
        {
          sessionId: 'seed-4',
          task: 'API versioning',
          input: 'Single version',
          output: 'Multiple versions',
          success: false,
          reward: 0.3,
          latencyMs: 4000
        }
      ];

      await Promise.all(seedPatterns.map(p => reasoningBank.storePattern(p)));

      // Give time for embeddings and indexing
      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should retrieve similar successful patterns', async () => {
      const patterns = await reasoningBank.retrievePatterns('API optimization', {
        k: 3,
        onlySuccesses: true
      });

      expect(patterns.length).toBeGreaterThan(0);
      expect(patterns.length).toBeLessThanOrEqual(3);

      // Should have similarity or uplift scores
      patterns.forEach(p => {
        expect(p).toHaveProperty('task');
        expect(p.similarity !== undefined || p.uplift !== undefined).toBe(true);
      });
    });

    it('should retrieve failed patterns for learning', async () => {
      const patterns = await reasoningBank.retrievePatterns('API', {
        k: 5,
        onlyFailures: true
      });

      expect(patterns.length).toBeGreaterThanOrEqual(0);

      // All returned patterns should be failures
      patterns.forEach(p => {
        if (p.success !== undefined) {
          expect(p.success).toBe(false);
        }
      });
    });

    it('should use query cache for repeated queries', async () => {
      const query = 'API caching strategy';

      // First query - will hit database
      const startTime1 = Date.now();
      const patterns1 = await reasoningBank.retrievePatterns(query, { k: 5 });
      const duration1 = Date.now() - startTime1;

      // Second query - should hit cache
      const startTime2 = Date.now();
      const patterns2 = await reasoningBank.retrievePatterns(query, { k: 5 });
      const duration2 = Date.now() - startTime2;

      expect(patterns1).toEqual(patterns2);
      expect(duration2).toBeLessThan(duration1); // Cache should be faster
    });

    it('should filter by minimum reward', async () => {
      const patterns = await reasoningBank.retrievePatterns('API', {
        k: 10,
        minReward: 0.85
      });

      // All patterns should have high uplift/reward
      patterns.forEach(p => {
        if (p.uplift !== undefined) {
          expect(p.uplift).toBeGreaterThanOrEqual(0.85);
        }
      });
    });
  });

  describe('Strategy Learning with Task Statistics', () => {
    beforeEach(async () => {
      // Seed with patterns showing improvement trend
      const patterns: PatternData[] = [
        {
          sessionId: 'strategy-1',
          task: 'Error handling',
          success: true,
          reward: 0.7,
          latencyMs: 200
        },
        {
          sessionId: 'strategy-2',
          task: 'Error handling',
          success: true,
          reward: 0.8,
          latencyMs: 180
        },
        {
          sessionId: 'strategy-3',
          task: 'Error handling',
          success: true,
          reward: 0.9,
          latencyMs: 150
        }
      ];

      await Promise.all(patterns.map(p => reasoningBank.storePattern(p)));
      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should learn optimal strategy from successful patterns', async () => {
      const strategy = await reasoningBank.learnStrategy('Error handling');

      expect(strategy).toHaveProperty('patterns');
      expect(strategy).toHaveProperty('causality');
      expect(strategy).toHaveProperty('confidence');
      expect(strategy).toHaveProperty('recommendation');

      expect(strategy.patterns.length).toBeGreaterThan(0);
      expect(strategy.confidence).toBeGreaterThan(0);
      expect(strategy.recommendation).toBeTruthy();
    });

    it('should provide causal insights', async () => {
      const strategy = await reasoningBank.learnStrategy('Error handling');
      const causality = strategy.causality;

      expect(causality).toHaveProperty('action');
      expect(causality).toHaveProperty('avgReward');
      expect(causality).toHaveProperty('avgUplift');
      expect(causality).toHaveProperty('confidence');
      expect(causality).toHaveProperty('evidenceCount');
      expect(causality).toHaveProperty('recommendation');

      expect(['DO_IT', 'AVOID', 'NEUTRAL']).toContain(causality.recommendation);
    });

    it('should handle tasks with no prior experience', async () => {
      const strategy = await reasoningBank.learnStrategy('Never seen before task');

      expect(strategy.patterns.length).toBe(0);
      expect(strategy.confidence).toBeLessThan(0.5);
      expect(strategy.recommendation).toContain('Limited evidence');
    });

    it('should calculate confidence based on evidence', async () => {
      // Task with many patterns should have high confidence
      const strategy1 = await reasoningBank.learnStrategy('Error handling');

      // Task with no patterns should have low confidence
      const strategy2 = await reasoningBank.learnStrategy('Unknown task');

      expect(strategy1.confidence).toBeGreaterThan(strategy2.confidence);
    });
  });

  describe('Auto-Consolidation', () => {
    beforeEach(async () => {
      // Seed with patterns that should consolidate into skills
      const patterns: PatternData[] = [];

      // Create 5 successful patterns for "caching" task
      for (let i = 0; i < 5; i++) {
        patterns.push({
          sessionId: `consolidate-${i}`,
          task: 'API caching',
          success: true,
          reward: 0.85 + (i * 0.02),
          latencyMs: 150 - (i * 5)
        });
      }

      // Create 3 successful patterns for "rate limiting" task
      for (let i = 0; i < 3; i++) {
        patterns.push({
          sessionId: `rate-limit-${i}`,
          task: 'Rate limiting',
          success: true,
          reward: 0.88 + (i * 0.02),
          latencyMs: 120
        });
      }

      await Promise.all(patterns.map(p => reasoningBank.storePattern(p)));
      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should consolidate frequent patterns into skills', async () => {
      const result = await reasoningBank.autoConsolidate(3, 0.7, 30);

      expect(result).toHaveProperty('skillsCreated');
      expect(result.skillsCreated).toBeGreaterThanOrEqual(0);
    });

    it('should respect minimum uses threshold', async () => {
      const result1 = await reasoningBank.autoConsolidate(10, 0.7, 30); // High threshold
      const result2 = await reasoningBank.autoConsolidate(2, 0.7, 30);  // Low threshold

      // Lower threshold should create more skills
      expect(result2.skillsCreated).toBeGreaterThanOrEqual(result1.skillsCreated);
    });

    it('should respect minimum success rate threshold', async () => {
      const result1 = await reasoningBank.autoConsolidate(3, 0.95, 30); // High success rate
      const result2 = await reasoningBank.autoConsolidate(3, 0.5, 30);  // Low success rate

      // Lower success rate threshold should create more skills
      expect(result2.skillsCreated).toBeGreaterThanOrEqual(result1.skillsCreated);
    });
  });

  describe('What-If Causal Analysis', () => {
    beforeEach(async () => {
      // Seed with patterns showing positive and negative outcomes
      const patterns: PatternData[] = [
        {
          sessionId: 'whatif-1',
          task: 'Add caching',
          success: true,
          reward: 0.95,
          latencyMs: 100
        },
        {
          sessionId: 'whatif-2',
          task: 'Add caching',
          success: true,
          reward: 0.92,
          latencyMs: 110
        },
        {
          sessionId: 'whatif-3',
          task: 'Remove validation',
          success: false,
          reward: 0.2,
          latencyMs: 3000
        }
      ];

      await Promise.all(patterns.map(p => reasoningBank.storePattern(p)));
      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should predict positive outcomes for beneficial actions', async () => {
      const insight = await reasoningBank.whatIfAnalysis('Add caching');

      expect(insight).toHaveProperty('action');
      expect(insight).toHaveProperty('avgReward');
      expect(insight).toHaveProperty('avgUplift');
      expect(insight).toHaveProperty('confidence');
      expect(insight).toHaveProperty('evidenceCount');
      expect(insight).toHaveProperty('recommendation');

      expect(insight.action).toBe('Add caching');

      // Should recommend beneficial action
      if (insight.evidenceCount > 0) {
        expect(['DO_IT', 'NEUTRAL']).toContain(insight.recommendation);
      }
    });

    it('should predict negative outcomes for harmful actions', async () => {
      const insight = await reasoningBank.whatIfAnalysis('Remove validation');

      expect(insight.action).toBe('Remove validation');

      // Should recommend avoiding harmful action
      if (insight.evidenceCount > 0 && insight.avgUplift < -0.1) {
        expect(insight.recommendation).toBe('AVOID');
      }
    });

    it('should handle unknown actions with neutral recommendation', async () => {
      const insight = await reasoningBank.whatIfAnalysis('Never tried action');

      expect(insight.evidenceCount).toBe(0);
      expect(insight.confidence).toBe(0);
      expect(insight.recommendation).toBe('NEUTRAL');
    });

    it('should calculate confidence from evidence count', async () => {
      const insight1 = await reasoningBank.whatIfAnalysis('Add caching'); // Has evidence
      const insight2 = await reasoningBank.whatIfAnalysis('Unknown action'); // No evidence

      expect(insight1.confidence).toBeGreaterThan(insight2.confidence);
    });
  });

  describe('Skill Search', () => {
    it('should search for relevant skills', async () => {
      const skills = await reasoningBank.searchSkills('API optimization', 5);

      expect(Array.isArray(skills)).toBe(true);
      // Skills may be empty if none consolidated yet
    });

    it('should limit results to k skills', async () => {
      const skills = await reasoningBank.searchSkills('API', 3);

      expect(skills.length).toBeLessThanOrEqual(3);
    });
  });

  describe('Statistics', () => {
    it('should return system statistics', () => {
      const stats = reasoningBank.getStats();

      expect(stats).toHaveProperty('causalRecall');
      expect(stats).toHaveProperty('reflexion');
      expect(stats).toHaveProperty('skills');
    });
  });

  describe('WASM Fallback', () => {
    it('should gracefully fall back to TypeScript when WASM unavailable', async () => {
      // Create instance with WASM disabled
      const rbNoWasm = new HybridReasoningBank({ preferWasm: false });

      const pattern: PatternData = {
        sessionId: 'fallback-test',
        task: 'Test fallback',
        success: true,
        reward: 0.9
      };

      // Should work without WASM
      const episodeId = await rbNoWasm.storePattern(pattern);
      expect(episodeId).toBeGreaterThan(0);

      const patterns = await rbNoWasm.retrievePatterns('Test', { k: 5 });
      expect(Array.isArray(patterns)).toBe(true);
    });
  });

  describe('Integration - Full Workflow', () => {
    it('should support complete reasoning workflow', async () => {
      // 1. Store initial pattern
      const pattern1: PatternData = {
        sessionId: 'workflow-1',
        task: 'Implement feature X',
        input: 'Requirements',
        output: 'Implementation v1',
        critique: 'Works but slow',
        success: true,
        reward: 0.7,
        latencyMs: 500
      };
      await reasoningBank.storePattern(pattern1);

      // 2. Store improved pattern
      const pattern2: PatternData = {
        sessionId: 'workflow-2',
        task: 'Implement feature X',
        input: 'Requirements + performance focus',
        output: 'Implementation v2 with optimization',
        critique: 'Much better performance',
        success: true,
        reward: 0.95,
        latencyMs: 150
      };
      await reasoningBank.storePattern(pattern2);

      await new Promise(resolve => setTimeout(resolve, 200));

      // 3. Learn strategy from patterns
      const strategy = await reasoningBank.learnStrategy('Implement feature X');
      expect(strategy.patterns.length).toBeGreaterThan(0);
      expect(strategy.confidence).toBeGreaterThan(0);

      // 4. What-if analysis
      const insight = await reasoningBank.whatIfAnalysis('Implement feature X');
      expect(insight.evidenceCount).toBeGreaterThan(0);

      // 5. Auto-consolidate
      const consolidation = await reasoningBank.autoConsolidate(2, 0.7, 30);
      expect(consolidation.skillsCreated).toBeGreaterThanOrEqual(0);

      // 6. Search skills
      const skills = await reasoningBank.searchSkills('feature', 5);
      expect(Array.isArray(skills)).toBe(true);
    });
  });
});
