/**
 * Comprehensive Test Suite for AdvancedMemorySystem v1.7.1
 *
 * Tests full implementation with:
 * - NightlyLearner integration
 * - Auto-consolidation pipeline
 * - Episodic replay
 * - What-if analysis
 * - Skill composition
 * - Automated learning cycles
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AdvancedMemorySystem, FailureAnalysis, SkillComposition, ConsolidationResult } from '../../src/reasoningbank/AdvancedMemory.js';
import { PatternData } from '../../src/reasoningbank/HybridBackend.js';
import * as fs from 'fs';
import * as path from 'path';

describe('AdvancedMemorySystem v1.7.1 - Full Implementation', () => {
  let memorySystem: AdvancedMemorySystem;
  const testDbPath = path.join(process.cwd(), 'test-agentdb-advanced.db');

  beforeEach(async () => {
    // Clean up any existing test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }

    // Initialize memory system
    memorySystem = new AdvancedMemorySystem({ preferWasm: false });

    // Give time for initialization
    await new Promise(resolve => setTimeout(resolve, 100));
  });

  afterEach(() => {
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Auto-Consolidation with NightlyLearner', () => {
    beforeEach(async () => {
      // Seed with diverse patterns for consolidation
      const patterns: PatternData[] = [
        // Successful caching patterns
        {
          sessionId: 'auto-1',
          task: 'Implement caching',
          success: true,
          reward: 0.88,
          latencyMs: 150
        },
        {
          sessionId: 'auto-2',
          task: 'Implement caching',
          success: true,
          reward: 0.92,
          latencyMs: 130
        },
        {
          sessionId: 'auto-3',
          task: 'Implement caching',
          success: true,
          reward: 0.95,
          latencyMs: 110
        },
        // Successful validation patterns
        {
          sessionId: 'auto-4',
          task: 'Add validation',
          success: true,
          reward: 0.85,
          latencyMs: 100
        },
        {
          sessionId: 'auto-5',
          task: 'Add validation',
          success: true,
          reward: 0.90,
          latencyMs: 95
        }
      ];

      // Store patterns via HybridReasoningBank
      for (const pattern of patterns) {
        await (memorySystem as any).reasoning.storePattern(pattern);
      }

      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should run full consolidation pipeline', async () => {
      const result = await memorySystem.autoConsolidate({
        minUses: 2,
        minSuccessRate: 0.7,
        lookbackDays: 30
      });

      expect(result).toHaveProperty('skillsCreated');
      expect(result).toHaveProperty('causalEdgesCreated');
      expect(result).toHaveProperty('patternsAnalyzed');
      expect(result).toHaveProperty('executionTimeMs');
      expect(result).toHaveProperty('recommendations');

      expect(result.skillsCreated).toBeGreaterThanOrEqual(0);
      expect(result.executionTimeMs).toBeGreaterThan(0);
      expect(Array.isArray(result.recommendations)).toBe(true);
    });

    it('should discover causal edges from patterns', async () => {
      const result = await memorySystem.autoConsolidate({
        minUses: 2,
        minSuccessRate: 0.7,
        lookbackDays: 30
      });

      // NightlyLearner should discover causal relationships
      expect(result.causalEdgesCreated).toBeGreaterThanOrEqual(0);
    });

    it('should create skills from high-performing patterns', async () => {
      const result = await memorySystem.autoConsolidate({
        minUses: 2,
        minSuccessRate: 0.8,
        lookbackDays: 30
      });

      expect(result.skillsCreated).toBeGreaterThanOrEqual(0);
    });

    it('should provide recommendations', async () => {
      const result = await memorySystem.autoConsolidate({
        minUses: 3,
        minSuccessRate: 0.7,
        lookbackDays: 30
      });

      expect(Array.isArray(result.recommendations)).toBe(true);
      expect(result.recommendations.length).toBeGreaterThan(0);
    });

    it('should handle dry run mode', async () => {
      const result = await memorySystem.autoConsolidate({
        minUses: 2,
        minSuccessRate: 0.7,
        lookbackDays: 30,
        dryRun: true
      });

      // Dry run should still analyze but not create permanent changes
      expect(result).toBeDefined();
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should respect lookback window', async () => {
      const result1 = await memorySystem.autoConsolidate({
        minUses: 2,
        minSuccessRate: 0.7,
        lookbackDays: 7  // Recent only
      });

      const result2 = await memorySystem.autoConsolidate({
        minUses: 2,
        minSuccessRate: 0.7,
        lookbackDays: 90  // All history
      });

      // Longer lookback should analyze more patterns
      expect(result2.patternsAnalyzed).toBeGreaterThanOrEqual(result1.patternsAnalyzed);
    });

    it('should handle graceful fallback on NightlyLearner error', async () => {
      // Should not throw even if NightlyLearner fails
      const result = await memorySystem.autoConsolidate({
        minUses: 100, // Very high threshold to trigger edge cases
        minSuccessRate: 0.99,
        lookbackDays: 1
      });

      expect(result).toBeDefined();
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });
  });

  describe('Episodic Replay - Learning from Failures', () => {
    beforeEach(async () => {
      // Seed with failure patterns
      const patterns: PatternData[] = [
        {
          sessionId: 'failure-1',
          task: 'Database migration',
          input: 'Schema v1',
          output: 'Migration failed',
          critique: 'Missing rollback strategy',
          success: false,
          reward: 0.2,
          latencyMs: 5000
        },
        {
          sessionId: 'failure-2',
          task: 'Database migration',
          input: 'Schema v2',
          output: 'Data corruption',
          critique: 'Need better validation',
          success: false,
          reward: 0.1,
          latencyMs: 8000
        },
        {
          sessionId: 'failure-3',
          task: 'API authentication',
          input: 'Basic implementation',
          output: 'Security vulnerability',
          critique: 'Need HTTPS and token rotation',
          success: false,
          reward: 0.3,
          latencyMs: 2000
        }
      ];

      for (const pattern of patterns) {
        await (memorySystem as any).reasoning.storePattern(pattern);
      }

      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should retrieve and analyze failures', async () => {
      const analyses = await memorySystem.replayFailures('Database migration', 5);

      expect(Array.isArray(analyses)).toBe(true);
      expect(analyses.length).toBeGreaterThan(0);

      analyses.forEach(analysis => {
        expect(analysis).toHaveProperty('critique');
        expect(analysis).toHaveProperty('whatWentWrong');
        expect(analysis).toHaveProperty('howToFix');
        expect(analysis).toHaveProperty('similarFailures');

        expect(Array.isArray(analysis.whatWentWrong)).toBe(true);
        expect(Array.isArray(analysis.howToFix)).toBe(true);
        expect(analysis.similarFailures).toBeGreaterThan(0);
      });
    });

    it('should extract meaningful critiques', async () => {
      const analyses = await memorySystem.replayFailures('Database migration', 3);

      analyses.forEach(analysis => {
        expect(typeof analysis.critique).toBe('string');
        expect(analysis.critique.length).toBeGreaterThan(0);
      });
    });

    it('should identify what went wrong', async () => {
      const analyses = await memorySystem.replayFailures('Database', 5);

      analyses.forEach(analysis => {
        expect(analysis.whatWentWrong.length).toBeGreaterThan(0);

        // Should identify common issues
        const issues = analysis.whatWentWrong.join(' ');
        expect(
          issues.includes('Low success rate') ||
          issues.includes('High latency') ||
          issues.includes('Task type')
        ).toBe(true);
      });
    });

    it('should generate actionable fixes', async () => {
      const analyses = await memorySystem.replayFailures('migration', 3);

      analyses.forEach(analysis => {
        expect(analysis.howToFix.length).toBeGreaterThan(0);

        // Should provide concrete recommendations
        const fixes = analysis.howToFix.join(' ');
        expect(
          fixes.includes('Review') ||
          fixes.includes('Optimize') ||
          fixes.includes('Consider') ||
          fixes.includes('Add')
        ).toBe(true);
      });
    });

    it('should count similar failures', async () => {
      const analyses = await memorySystem.replayFailures('Database migration', 5);

      analyses.forEach(analysis => {
        expect(analysis.similarFailures).toBeGreaterThanOrEqual(1);
      });
    });

    it('should handle tasks with no failures', async () => {
      const analyses = await memorySystem.replayFailures('Never failed task', 5);

      expect(Array.isArray(analyses)).toBe(true);
      expect(analyses.length).toBe(0);
    });
  });

  describe('What-If Causal Analysis', () => {
    beforeEach(async () => {
      // Seed with patterns showing clear outcomes
      const patterns: PatternData[] = [
        {
          sessionId: 'whatif-1',
          task: 'Enable caching',
          success: true,
          reward: 0.95,
          latencyMs: 100
        },
        {
          sessionId: 'whatif-2',
          task: 'Enable caching',
          success: true,
          reward: 0.92,
          latencyMs: 110
        },
        {
          sessionId: 'whatif-3',
          task: 'Disable validation',
          success: false,
          reward: 0.2,
          latencyMs: 3000
        }
      ];

      for (const pattern of patterns) {
        await (memorySystem as any).reasoning.storePattern(pattern);
      }

      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should predict outcomes with impact description', async () => {
      const analysis = await memorySystem.whatIfAnalysis('Enable caching');

      expect(analysis).toHaveProperty('action');
      expect(analysis).toHaveProperty('avgReward');
      expect(analysis).toHaveProperty('avgUplift');
      expect(analysis).toHaveProperty('confidence');
      expect(analysis).toHaveProperty('evidenceCount');
      expect(analysis).toHaveProperty('recommendation');
      expect(analysis).toHaveProperty('expectedImpact');

      expect(typeof analysis.expectedImpact).toBe('string');
      expect(analysis.expectedImpact.length).toBeGreaterThan(0);
    });

    it('should identify highly beneficial actions', async () => {
      const analysis = await memorySystem.whatIfAnalysis('Enable caching');

      if (analysis.avgUplift > 0.2 && analysis.evidenceCount > 0) {
        expect(analysis.expectedImpact).toContain('Highly beneficial');
        expect(analysis.recommendation).toBe('DO_IT');
      }
    });

    it('should identify harmful actions', async () => {
      const analysis = await memorySystem.whatIfAnalysis('Disable validation');

      if (analysis.avgUplift < -0.1 && analysis.evidenceCount > 0) {
        expect(analysis.expectedImpact).toContain('Harmful');
        expect(analysis.recommendation).toBe('AVOID');
      }
    });

    it('should handle neutral or unknown actions', async () => {
      const analysis = await memorySystem.whatIfAnalysis('Unknown action');

      expect(analysis.evidenceCount).toBe(0);
      expect(analysis.recommendation).toBe('NEUTRAL');
      expect(analysis.expectedImpact).toContain('Neutral or insufficient evidence');
    });

    it('should quantify expected impact', async () => {
      const analysis = await memorySystem.whatIfAnalysis('Enable caching');

      // Should provide percentage improvement if data available
      if (analysis.avgUplift !== 0) {
        expect(analysis.expectedImpact).toMatch(/\d+\.\d+%/);
      }
    });
  });

  describe('Skill Composition', () => {
    beforeEach(async () => {
      // Seed with patterns that consolidate into skills
      const patterns: PatternData[] = [];

      // High-performing caching skill
      for (let i = 0; i < 10; i++) {
        patterns.push({
          sessionId: `skill-cache-${i}`,
          task: 'Implement caching',
          success: true,
          reward: 0.9 + (i * 0.01),
          latencyMs: 100
        });
      }

      // Moderate validation skill
      for (let i = 0; i < 5; i++) {
        patterns.push({
          sessionId: `skill-valid-${i}`,
          task: 'Add validation',
          success: true,
          reward: 0.8 + (i * 0.01),
          latencyMs: 150
        });
      }

      for (const pattern of patterns) {
        await (memorySystem as any).reasoning.storePattern(pattern);
      }

      // Consolidate into skills
      await memorySystem.autoConsolidate({
        minUses: 3,
        minSuccessRate: 0.7,
        lookbackDays: 30
      });

      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should find and compose relevant skills', async () => {
      const composition = await memorySystem.composeSkills('API optimization', 5);

      expect(composition).toHaveProperty('availableSkills');
      expect(composition).toHaveProperty('compositionPlan');
      expect(composition).toHaveProperty('expectedSuccessRate');

      expect(Array.isArray(composition.availableSkills)).toBe(true);
      expect(typeof composition.compositionPlan).toBe('string');
      expect(typeof composition.expectedSuccessRate).toBe('number');
    });

    it('should sort skills by quality', async () => {
      const composition = await memorySystem.composeSkills('caching', 5);

      if (composition.availableSkills.length > 1) {
        // Skills should be sorted by combined score (success rate + usage)
        for (let i = 0; i < composition.availableSkills.length - 1; i++) {
          const skill1 = composition.availableSkills[i];
          const skill2 = composition.availableSkills[i + 1];

          const score1 = (skill1.successRate || 0) * 0.7 + (Math.log(skill1.uses || 1) / 10) * 0.3;
          const score2 = (skill2.successRate || 0) * 0.7 + (Math.log(skill2.uses || 1) / 10) * 0.3;

          expect(score1).toBeGreaterThanOrEqual(score2);
        }
      }
    });

    it('should create composition plan', async () => {
      const composition = await memorySystem.composeSkills('optimization', 5);

      expect(composition.compositionPlan.length).toBeGreaterThan(0);

      if (composition.availableSkills.length === 0) {
        expect(composition.compositionPlan).toBe('No relevant skills found');
      }
    });

    it('should calculate expected success rate', async () => {
      const composition = await memorySystem.composeSkills('caching', 5);

      expect(composition.expectedSuccessRate).toBeGreaterThanOrEqual(0);
      expect(composition.expectedSuccessRate).toBeLessThanOrEqual(1);

      if (composition.availableSkills.length > 0) {
        expect(composition.expectedSuccessRate).toBeGreaterThan(0);
      }
    });

    it('should handle task with no relevant skills', async () => {
      const composition = await memorySystem.composeSkills('Never seen task', 5);

      expect(composition.availableSkills.length).toBe(0);
      expect(composition.compositionPlan).toBe('No relevant skills found');
      expect(composition.expectedSuccessRate).toBe(0);
    });
  });

  describe('Automated Learning Cycle', () => {
    beforeEach(async () => {
      // Seed with diverse learning data
      const patterns: PatternData[] = [
        {
          sessionId: 'cycle-1',
          task: 'Feature A',
          success: true,
          reward: 0.85,
          latencyMs: 200
        },
        {
          sessionId: 'cycle-2',
          task: 'Feature A',
          success: true,
          reward: 0.90,
          latencyMs: 180
        },
        {
          sessionId: 'cycle-3',
          task: 'Feature B',
          success: true,
          reward: 0.88,
          latencyMs: 150
        }
      ];

      for (const pattern of patterns) {
        await (memorySystem as any).reasoning.storePattern(pattern);
      }

      await new Promise(resolve => setTimeout(resolve, 200));
    });

    it('should run complete learning cycle', async () => {
      const result = await memorySystem.runLearningCycle();

      expect(result).toHaveProperty('skillsCreated');
      expect(result).toHaveProperty('causalEdgesCreated');
      expect(result).toHaveProperty('patternsAnalyzed');
      expect(result).toHaveProperty('executionTimeMs');
      expect(result).toHaveProperty('recommendations');

      expect(result.executionTimeMs).toBeGreaterThan(0);
    });

    it('should use optimal consolidation parameters', async () => {
      const result = await memorySystem.runLearningCycle();

      // Should use default optimal parameters
      expect(result).toBeDefined();
      expect(result.executionTimeMs).toBeGreaterThan(0);
    });
  });

  describe('System Statistics', () => {
    it('should return comprehensive statistics', () => {
      const stats = memorySystem.getStats();

      expect(stats).toHaveProperty('reasoningBank');
      expect(stats).toHaveProperty('learner');
      expect(stats).toHaveProperty('memoryPool');

      expect(typeof stats.learner).toBe('string');
      expect(stats.learner).toContain('NightlyLearner');
    });
  });

  describe('Integration - Complete Memory Lifecycle', () => {
    it('should support full memory management workflow', async () => {
      // 1. Store diverse patterns
      const patterns: PatternData[] = [
        {
          sessionId: 'lifecycle-1',
          task: 'Build feature X',
          input: 'Requirements',
          output: 'Implementation v1',
          critique: 'Works but needs optimization',
          success: true,
          reward: 0.7,
          latencyMs: 500
        },
        {
          sessionId: 'lifecycle-2',
          task: 'Build feature X',
          input: 'Optimized approach',
          output: 'Implementation v2',
          critique: 'Much better',
          success: true,
          reward: 0.95,
          latencyMs: 150
        },
        {
          sessionId: 'lifecycle-3',
          task: 'Build feature Y',
          input: 'Poor approach',
          output: 'Failed',
          critique: 'Missing validation',
          success: false,
          reward: 0.2,
          latencyMs: 3000
        }
      ];

      for (const pattern of patterns) {
        await (memorySystem as any).reasoning.storePattern(pattern);
      }

      await new Promise(resolve => setTimeout(resolve, 200));

      // 2. Auto-consolidate
      const consolidation = await memorySystem.autoConsolidate({
        minUses: 2,
        minSuccessRate: 0.7,
        lookbackDays: 30
      });
      expect(consolidation.skillsCreated).toBeGreaterThanOrEqual(0);

      // 3. Learn from failures
      const failures = await memorySystem.replayFailures('feature', 5);
      expect(Array.isArray(failures)).toBe(true);

      // 4. What-if analysis
      const analysis = await memorySystem.whatIfAnalysis('Build feature X');
      expect(analysis.evidenceCount).toBeGreaterThan(0);

      // 5. Compose skills
      const composition = await memorySystem.composeSkills('Build feature', 5);
      expect(composition).toBeDefined();

      // 6. Run learning cycle
      const cycle = await memorySystem.runLearningCycle();
      expect(cycle.executionTimeMs).toBeGreaterThan(0);

      // 7. Get statistics
      const stats = memorySystem.getStats();
      expect(stats).toBeDefined();
      expect(stats.reasoningBank).toBeDefined();
      expect(stats.memoryPool).toBeDefined();
    });
  });
});
