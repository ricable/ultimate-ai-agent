/**
 * Integration Test for ReasoningBank v1.7.1
 *
 * Tests the full implementation with actual agentdb integration:
 * - HybridReasoningBank with CausalRecall
 * - AdvancedMemorySystem with NightlyLearner
 * - End-to-end workflows
 */

import { describe, it, expect } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

describe('ReasoningBank v1.7.1 - Integration Tests', () => {
  const testDbPath = path.join(process.cwd(), 'test-reasoningbank-integration.db');

  // Clean up before and after
  const cleanup = () => {
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  };

  describe('Module Exports', () => {
    it('should export HybridReasoningBank', async () => {
      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      expect(HybridReasoningBank).toBeDefined();
      expect(typeof HybridReasoningBank).toBe('function');
    });

    it('should export AdvancedMemorySystem', async () => {
      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      expect(AdvancedMemorySystem).toBeDefined();
      expect(typeof AdvancedMemorySystem).toBeDefined();
    });

    it('should export type interfaces', async () => {
      const module = await import('../../src/reasoningbank/HybridBackend.js');
      expect(module).toHaveProperty('HybridReasoningBank');

      // Type interfaces are compile-time only, but we can verify the main class
      const rb = new module.HybridReasoningBank();
      expect(rb).toBeDefined();
      expect(rb).toHaveProperty('storePattern');
      expect(rb).toHaveProperty('retrievePatterns');
      expect(rb).toHaveProperty('learnStrategy');
      expect(rb).toHaveProperty('autoConsolidate');
      expect(rb).toHaveProperty('whatIfAnalysis');
      expect(rb).toHaveProperty('searchSkills');
      expect(rb).toHaveProperty('getStats');
    });
  });

  describe('HybridReasoningBank - Basic Operations', () => {
    it('should initialize with WASM preference', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: true });

      expect(rb).toBeDefined();
      expect(rb.getStats).toBeDefined();

      cleanup();
    });

    it('should store a pattern successfully', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: false });

      const pattern = {
        sessionId: 'integration-test-1',
        task: 'Test pattern storage',
        success: true,
        reward: 0.95
      };

      const episodeId = await rb.storePattern(pattern);
      expect(episodeId).toBeGreaterThan(0);

      cleanup();
    });

    it('should retrieve patterns', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: false });

      // Store a pattern first
      await rb.storePattern({
        sessionId: 'integration-test-2',
        task: 'Caching implementation',
        success: true,
        reward: 0.9
      });

      // Give time for indexing
      await new Promise(resolve => setTimeout(resolve, 100));

      // Retrieve patterns
      const patterns = await rb.retrievePatterns('caching', { k: 5 });
      expect(Array.isArray(patterns)).toBe(true);

      cleanup();
    });

    it('should perform strategy learning', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: false });

      // Store multiple patterns
      await rb.storePattern({
        sessionId: 'strategy-1',
        task: 'Error handling',
        success: true,
        reward: 0.85
      });

      await rb.storePattern({
        sessionId: 'strategy-2',
        task: 'Error handling',
        success: true,
        reward: 0.92
      });

      await new Promise(resolve => setTimeout(resolve, 100));

      const strategy = await rb.learnStrategy('Error handling');
      expect(strategy).toHaveProperty('patterns');
      expect(strategy).toHaveProperty('causality');
      expect(strategy).toHaveProperty('confidence');
      expect(strategy).toHaveProperty('recommendation');

      cleanup();
    });

    it('should perform what-if analysis', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: false });

      const insight = await rb.whatIfAnalysis('Add caching');
      expect(insight).toHaveProperty('action');
      expect(insight).toHaveProperty('avgReward');
      expect(insight).toHaveProperty('avgUplift');
      expect(insight).toHaveProperty('confidence');
      expect(insight).toHaveProperty('evidenceCount');
      expect(insight).toHaveProperty('recommendation');

      expect(['DO_IT', 'AVOID', 'NEUTRAL']).toContain(insight.recommendation);

      cleanup();
    });

    it('should auto-consolidate patterns', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: false });

      // Store multiple patterns
      for (let i = 0; i < 3; i++) {
        await rb.storePattern({
          sessionId: `consolidate-${i}`,
          task: 'API optimization',
          success: true,
          reward: 0.85 + (i * 0.05)
        });
      }

      await new Promise(resolve => setTimeout(resolve, 100));

      const result = await rb.autoConsolidate(2, 0.7, 30);
      expect(result).toHaveProperty('skillsCreated');
      expect(result.skillsCreated).toBeGreaterThanOrEqual(0);

      cleanup();
    });

    it('should search for skills', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: false });

      const skills = await rb.searchSkills('API optimization', 5);
      expect(Array.isArray(skills)).toBe(true);

      cleanup();
    });

    it('should return statistics', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: false });

      const stats = rb.getStats();
      expect(stats).toHaveProperty('causalRecall');
      expect(stats).toHaveProperty('reflexion');
      expect(stats).toHaveProperty('skills');

      cleanup();
    });
  });

  describe('AdvancedMemorySystem - High-Level Operations', () => {
    it('should initialize successfully', async () => {
      cleanup();

      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      const memory = new AdvancedMemorySystem({ preferWasm: false });

      expect(memory).toBeDefined();
      expect(memory.getStats).toBeDefined();

      cleanup();
    });

    it('should run auto-consolidation', async () => {
      cleanup();

      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      const memory = new AdvancedMemorySystem({ preferWasm: false });

      const result = await memory.autoConsolidate({
        minUses: 2,
        minSuccessRate: 0.7,
        lookbackDays: 30
      });

      expect(result).toHaveProperty('skillsCreated');
      expect(result).toHaveProperty('causalEdgesCreated');
      expect(result).toHaveProperty('patternsAnalyzed');
      expect(result).toHaveProperty('executionTimeMs');
      expect(result).toHaveProperty('recommendations');

      expect(result.executionTimeMs).toBeGreaterThan(0);
      expect(Array.isArray(result.recommendations)).toBe(true);

      cleanup();
    });

    it('should replay failures', async () => {
      cleanup();

      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      const memory = new AdvancedMemorySystem({ preferWasm: false });

      const analyses = await memory.replayFailures('test task', 5);
      expect(Array.isArray(analyses)).toBe(true);

      cleanup();
    });

    it('should perform what-if analysis', async () => {
      cleanup();

      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      const memory = new AdvancedMemorySystem({ preferWasm: false });

      const analysis = await memory.whatIfAnalysis('test action');
      expect(analysis).toHaveProperty('action');
      expect(analysis).toHaveProperty('expectedImpact');

      cleanup();
    });

    it('should compose skills', async () => {
      cleanup();

      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      const memory = new AdvancedMemorySystem({ preferWasm: false });

      const composition = await memory.composeSkills('test task', 5);
      expect(composition).toHaveProperty('availableSkills');
      expect(composition).toHaveProperty('compositionPlan');
      expect(composition).toHaveProperty('expectedSuccessRate');

      cleanup();
    });

    it('should run learning cycle', async () => {
      cleanup();

      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      const memory = new AdvancedMemorySystem({ preferWasm: false });

      const result = await memory.runLearningCycle();
      expect(result).toHaveProperty('executionTimeMs');
      expect(result.executionTimeMs).toBeGreaterThan(0);

      cleanup();
    });

    it('should return system statistics', async () => {
      cleanup();

      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      const memory = new AdvancedMemorySystem({ preferWasm: false });

      const stats = memory.getStats();
      expect(stats).toHaveProperty('reasoningBank');
      expect(stats).toHaveProperty('learner');
      expect(stats).toHaveProperty('memoryPool');

      cleanup();
    });
  });

  describe('End-to-End Workflow', () => {
    it('should support complete reasoning workflow', async () => {
      cleanup();

      const { HybridReasoningBank } = await import('../../src/reasoningbank/HybridBackend.js');
      const rb = new HybridReasoningBank({ preferWasm: false });

      // 1. Store patterns showing improvement
      await rb.storePattern({
        sessionId: 'workflow-1',
        task: 'Implement feature X',
        success: true,
        reward: 0.7,
        latencyMs: 500
      });

      await rb.storePattern({
        sessionId: 'workflow-2',
        task: 'Implement feature X',
        success: true,
        reward: 0.95,
        latencyMs: 150
      });

      await new Promise(resolve => setTimeout(resolve, 100));

      // 2. Learn strategy
      const strategy = await rb.learnStrategy('Implement feature X');
      expect(strategy.patterns.length).toBeGreaterThan(0);

      // 3. What-if analysis
      const insight = await rb.whatIfAnalysis('Implement feature X');
      expect(insight.evidenceCount).toBeGreaterThan(0);

      // 4. Auto-consolidate
      const consolidation = await rb.autoConsolidate(2, 0.7, 30);
      expect(consolidation.skillsCreated).toBeGreaterThanOrEqual(0);

      // 5. Search skills
      const skills = await rb.searchSkills('feature', 5);
      expect(Array.isArray(skills)).toBe(true);

      cleanup();
    });

    it('should support advanced memory workflow', async () => {
      cleanup();

      const { AdvancedMemorySystem } = await import('../../src/reasoningbank/AdvancedMemory.js');
      const memory = new AdvancedMemorySystem({ preferWasm: false });

      // 1. Auto-consolidate
      const consolidation = await memory.autoConsolidate({ minUses: 2, minSuccessRate: 0.7 });
      expect(consolidation.executionTimeMs).toBeGreaterThan(0);

      // 2. What-if analysis
      const analysis = await memory.whatIfAnalysis('test action');
      expect(analysis.expectedImpact).toBeDefined();

      // 3. Compose skills
      const composition = await memory.composeSkills('test', 5);
      expect(composition.compositionPlan).toBeDefined();

      // 4. Get stats
      const stats = memory.getStats();
      expect(stats.reasoningBank).toBeDefined();

      cleanup();
    });
  });
});
