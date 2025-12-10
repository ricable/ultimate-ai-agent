/**
 * Memory Systems Performance Benchmarks
 *
 * Tests causal graph, reflexion memory, and skill library performance
 */

import { performance } from 'perf_hooks';
import { AgentDB } from '../../dist/index.js';
import type { BenchmarkResult } from '../benchmark-runner';

export class MemorySystemsBenchmark {
  private db: AgentDB | null = null;

  /**
   * Test causal graph query performance
   */
  async testCausalGraphQuery(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const nodeCount = 1000;
    const edgeCount = 5000;

    try {
      this.db = new AgentDB({ dbPath: ':memory:' });

      // Insert nodes
      const insertNodesStart = performance.now();
      for (let i = 0; i < nodeCount; i++) {
        await this.db.causalGraph.addNode({
          id: `node_${i}`,
          type: i % 5 === 0 ? 'action' : 'observation',
          content: `Node ${i} content`,
          metadata: { index: i }
        });
      }
      const insertNodesDuration = performance.now() - insertNodesStart;

      // Insert edges
      const insertEdgesStart = performance.now();
      for (let i = 0; i < edgeCount; i++) {
        const fromId = `node_${Math.floor(Math.random() * nodeCount)}`;
        const toId = `node_${Math.floor(Math.random() * nodeCount)}`;

        if (fromId !== toId) {
          await this.db.causalGraph.addEdge({
            from: fromId,
            to: toId,
            type: 'causes',
            weight: Math.random()
          });
        }
      }
      const insertEdgesDuration = performance.now() - insertEdgesStart;

      // Test queries
      const queryStart = performance.now();
      const queryCount = 100;

      for (let i = 0; i < queryCount; i++) {
        const nodeId = `node_${Math.floor(Math.random() * nodeCount)}`;
        await this.db.causalGraph.getDescendants(nodeId, 2);
      }

      const queryDuration = performance.now() - queryStart;
      const avgQueryTime = queryDuration / queryCount;

      this.db?.close();
      this.db = null;

      return {
        name: 'Causal Graph Query Performance',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgQueryTime,
        metrics: {
          nodeCount,
          edgeCount,
          insertNodesDurationMs: insertNodesDuration.toFixed(2),
          insertEdgesDurationMs: insertEdgesDuration.toFixed(2),
          avgQueryTimeMs: avgQueryTime.toFixed(2),
          queryCount
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Causal Graph Query Performance',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test reflexion memory retrieval
   */
  async testReflexionRetrieval(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const episodeCount = 500;

    try {
      this.db = new AgentDB({ dbPath: ':memory:' });

      // Generate test episodes
      const generateVector = (dim: number = 1536): number[] => {
        const vector: number[] = [];
        for (let i = 0; i < dim; i++) {
          vector.push(Math.random() * 2 - 1);
        }
        const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        return vector.map(v => v / magnitude);
      };

      // Insert episodes
      const insertStart = performance.now();
      for (let i = 0; i < episodeCount; i++) {
        await this.db.reflexionMemory.store({
          task: `Task ${i}`,
          trajectory: [
            { action: 'start', observation: 'Begin task' },
            { action: 'process', observation: 'Processing...' },
            { action: 'complete', observation: 'Task done' }
          ],
          verdict: Math.random() > 0.5 ? 'success' : 'failure',
          critique: `Critique for task ${i}`,
          reward: Math.random(),
          embedding: generateVector()
        });
      }
      const insertDuration = performance.now() - insertStart;

      // Test retrieval
      const queryStart = performance.now();
      const queryCount = 100;

      for (let i = 0; i < queryCount; i++) {
        await this.db.reflexionMemory.retrieve(`Task ${Math.floor(Math.random() * episodeCount)}`, 5);
      }

      const queryDuration = performance.now() - queryStart;
      const avgQueryTime = queryDuration / queryCount;

      // Test success/failure filtering
      const filterStart = performance.now();
      await this.db.reflexionMemory.getSuccessfulEpisodes('Task', 10);
      await this.db.reflexionMemory.getFailedEpisodes('Task', 10);
      const filterDuration = performance.now() - filterStart;

      this.db?.close();
      this.db = null;

      return {
        name: 'Reflexion Memory Retrieval',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgQueryTime,
        metrics: {
          episodeCount,
          insertDurationMs: insertDuration.toFixed(2),
          avgQueryTimeMs: avgQueryTime.toFixed(2),
          filterDurationMs: filterDuration.toFixed(2),
          queryCount
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Reflexion Memory Retrieval',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test skill library search
   */
  async testSkillLibrarySearch(): Promise<BenchmarkResult> {
    const startTime = performance.now();
    const skillCount = 200;

    try {
      this.db = new AgentDB({ dbPath: ':memory:' });

      // Generate test skills
      const generateVector = (dim: number = 1536): number[] => {
        const vector: number[] = [];
        for (let i = 0; i < dim; i++) {
          vector.push(Math.random() * 2 - 1);
        }
        const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        return vector.map(v => v / magnitude);
      };

      const categories = ['coding', 'analysis', 'communication', 'planning', 'debugging'];
      const difficulties = ['beginner', 'intermediate', 'advanced', 'expert'];

      // Insert skills
      const insertStart = performance.now();
      for (let i = 0; i < skillCount; i++) {
        await this.db.skillLibrary.add({
          name: `skill_${i}`,
          description: `Description for skill ${i}`,
          code: `function skill_${i}() { return ${i}; }`,
          category: categories[i % categories.length],
          successRate: Math.random(),
          difficulty: difficulties[i % difficulties.length] as any,
          embedding: generateVector()
        });
      }
      const insertDuration = performance.now() - insertStart;

      // Test semantic search
      const semanticStart = performance.now();
      const semanticCount = 100;

      for (let i = 0; i < semanticCount; i++) {
        await this.db.skillLibrary.search('coding skill', 5);
      }

      const semanticDuration = performance.now() - semanticStart;
      const avgSemanticTime = semanticDuration / semanticCount;

      // Test category filtering
      const categoryStart = performance.now();
      for (const category of categories) {
        await this.db.skillLibrary.getByCategory(category);
      }
      const categoryDuration = performance.now() - categoryStart;

      // Test difficulty filtering
      const difficultyStart = performance.now();
      for (const difficulty of difficulties) {
        await this.db.skillLibrary.getByDifficulty(difficulty as any);
      }
      const difficultyDuration = performance.now() - difficultyStart;

      this.db?.close();
      this.db = null;

      return {
        name: 'Skill Library Search',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSemanticTime,
        metrics: {
          skillCount,
          insertDurationMs: insertDuration.toFixed(2),
          avgSemanticSearchMs: avgSemanticTime.toFixed(2),
          categoryFilterDurationMs: categoryDuration.toFixed(2),
          difficultyFilterDurationMs: difficultyDuration.toFixed(2),
          semanticSearchCount: semanticCount
        },
        timestamp: Date.now(),
        success: true
      };
    } catch (error) {
      return {
        name: 'Skill Library Search',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        metrics: {},
        timestamp: Date.now(),
        success: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }
}
