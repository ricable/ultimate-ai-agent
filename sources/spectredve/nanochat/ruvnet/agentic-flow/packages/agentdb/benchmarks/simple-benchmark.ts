#!/usr/bin/env node
/**
 * Simplified AgentDB Performance Benchmark
 *
 * Direct testing without full AgentDB wrapper
 */

import { performance } from 'perf_hooks';
import { ReflexionMemory } from '../dist/controllers/ReflexionMemory.js';
import { SkillLibrary } from '../dist/controllers/SkillLibrary.js';
import { CausalMemoryGraph } from '../dist/controllers/CausalMemoryGraph.js';
import * as fs from 'fs/promises';
import * as path from 'path';

interface BenchmarkResult {
  name: string;
  category: string;
  duration: number;
  operationsPerSecond?: number;
  metrics: Record<string, any>;
  success: boolean;
}

class SimpleBenchmark {
  private results: BenchmarkResult[] = [];

  // Helper to generate random embedding
  private generateVector(dim: number = 384): number[] {
    const vector: number[] = [];
    for (let i = 0; i < dim; i++) {
      vector.push(Math.random() * 2 - 1);
    }
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(v => v / magnitude);
  }

  /**
   * Benchmark Reflexion Memory
   */
  async benchmarkReflexionMemory(): Promise<void> {
    console.log('\n📊 Reflexion Memory Benchmarks');
    console.log('-'.repeat(80));

    const startTime = performance.now();
    const episodeCount = 500;

    try {
      const memory = new ReflexionMemory({
        dbPath: ':memory:',
        vectorDimension: 384
      });

      // Insert episodes
      const insertStart = performance.now();
      for (let i = 0; i < episodeCount; i++) {
        await memory.store({
          task: `Task ${i}`,
          trajectory: [
            { action: 'start', observation: 'Begin task' },
            { action: 'process', observation: 'Processing...' },
            { action: 'complete', observation: 'Task done' }
          ],
          verdict: Math.random() > 0.5 ? 'success' : 'failure',
          critique: `Critique for task ${i}`,
          reward: Math.random(),
          embedding: this.generateVector(384)
        });
      }
      const insertDuration = performance.now() - insertStart;

      // Test retrieval
      const queryStart = performance.now();
      const queryCount = 100;

      for (let i = 0; i < queryCount; i++) {
        await memory.retrieve(`Task ${Math.floor(Math.random() * episodeCount)}`, 5);
      }

      const queryDuration = performance.now() - queryStart;
      const avgQueryTime = queryDuration / queryCount;

      memory.close();

      const result: BenchmarkResult = {
        name: 'Reflexion Memory',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgQueryTime,
        metrics: {
          episodeCount,
          insertDurationMs: insertDuration.toFixed(2),
          avgQueryTimeMs: avgQueryTime.toFixed(2),
          totalDurationMs: (performance.now() - startTime).toFixed(2)
        },
        success: true
      };

      this.results.push(result);
      console.log(`   ✅ ${result.name}: ${result.duration.toFixed(2)}ms (${result.operationsPerSecond?.toFixed(0)} ops/sec)`);
      console.log(`      Episodes: ${episodeCount}, Insert: ${insertDuration.toFixed(2)}ms, Avg Query: ${avgQueryTime.toFixed(2)}ms`);

    } catch (error) {
      console.log(`   ❌ Reflexion Memory failed: ${error}`);
      this.results.push({
        name: 'Reflexion Memory',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        metrics: {},
        success: false
      });
    }
  }

  /**
   * Benchmark Skill Library
   */
  async benchmarkSkillLibrary(): Promise<void> {
    console.log('\n📊 Skill Library Benchmarks');
    console.log('-'.repeat(80));

    const startTime = performance.now();
    const skillCount = 200;

    try {
      const library = new SkillLibrary({
        dbPath: ':memory:',
        vectorDimension: 384
      });

      const categories = ['coding', 'analysis', 'communication', 'planning'];
      const difficulties = ['beginner', 'intermediate', 'advanced', 'expert'];

      // Insert skills
      const insertStart = performance.now();
      for (let i = 0; i < skillCount; i++) {
        await library.add({
          name: `skill_${i}`,
          description: `Description for skill ${i}`,
          code: `function skill_${i}() { return ${i}; }`,
          category: categories[i % categories.length],
          successRate: Math.random(),
          difficulty: difficulties[i % difficulties.length] as any,
          embedding: this.generateVector(384)
        });
      }
      const insertDuration = performance.now() - insertStart;

      // Test semantic search
      const searchStart = performance.now();
      const searchCount = 100;

      for (let i = 0; i < searchCount; i++) {
        await library.search('coding skill', 5);
      }

      const searchDuration = performance.now() - searchStart;
      const avgSearchTime = searchDuration / searchCount;

      library.close();

      const result: BenchmarkResult = {
        name: 'Skill Library',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgSearchTime,
        metrics: {
          skillCount,
          insertDurationMs: insertDuration.toFixed(2),
          avgSearchTimeMs: avgSearchTime.toFixed(2),
          totalDurationMs: (performance.now() - startTime).toFixed(2)
        },
        success: true
      };

      this.results.push(result);
      console.log(`   ✅ ${result.name}: ${result.duration.toFixed(2)}ms (${result.operationsPerSecond?.toFixed(0)} ops/sec)`);
      console.log(`      Skills: ${skillCount}, Insert: ${insertDuration.toFixed(2)}ms, Avg Search: ${avgSearchTime.toFixed(2)}ms`);

    } catch (error) {
      console.log(`   ❌ Skill Library failed: ${error}`);
      this.results.push({
        name: 'Skill Library',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        metrics: {},
        success: false
      });
    }
  }

  /**
   * Benchmark Causal Memory Graph
   */
  async benchmarkCausalGraph(): Promise<void> {
    console.log('\n📊 Causal Memory Graph Benchmarks');
    console.log('-'.repeat(80));

    const startTime = performance.now();
    const nodeCount = 500;
    const edgeCount = 2000;

    try {
      const graph = new CausalMemoryGraph({
        dbPath: ':memory:'
      });

      // Insert nodes
      const insertNodesStart = performance.now();
      for (let i = 0; i < nodeCount; i++) {
        await graph.addNode({
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
          await graph.addEdge({
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
        await graph.getDescendants(nodeId, 2);
      }

      const queryDuration = performance.now() - queryStart;
      const avgQueryTime = queryDuration / queryCount;

      graph.close();

      const result: BenchmarkResult = {
        name: 'Causal Memory Graph',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        operationsPerSecond: 1000 / avgQueryTime,
        metrics: {
          nodeCount,
          edgeCount,
          insertNodesDurationMs: insertNodesDuration.toFixed(2),
          insertEdgesDurationMs: insertEdgesDuration.toFixed(2),
          avgQueryTimeMs: avgQueryTime.toFixed(2),
          totalDurationMs: (performance.now() - startTime).toFixed(2)
        },
        success: true
      };

      this.results.push(result);
      console.log(`   ✅ ${result.name}: ${result.duration.toFixed(2)}ms (${result.operationsPerSecond?.toFixed(0)} ops/sec)`);
      console.log(`      Nodes: ${nodeCount}, Edges: ${edgeCount}, Avg Query: ${avgQueryTime.toFixed(2)}ms`);

    } catch (error) {
      console.log(`   ❌ Causal Memory Graph failed: ${error}`);
      this.results.push({
        name: 'Causal Memory Graph',
        category: 'Memory Systems',
        duration: performance.now() - startTime,
        metrics: {},
        success: false
      });
    }
  }

  /**
   * Generate performance report
   */
  async generateReport(): Promise<void> {
    console.log('\n' + '='.repeat(80));
    console.log('📈 Performance Report Summary');
    console.log('='.repeat(80));

    const successful = this.results.filter(r => r.success);
    const failed = this.results.filter(r => !r.success);

    console.log(`\nTotal Benchmarks: ${this.results.length}`);
    console.log(`Successful: ${successful.length}`);
    console.log(`Failed: ${failed.length}`);

    if (successful.length > 0) {
      console.log('\n✅ Successful Benchmarks:');
      for (const result of successful) {
        console.log(`   ${result.name}: ${result.duration.toFixed(2)}ms (${result.operationsPerSecond?.toFixed(0) || 'N/A'} ops/sec)`);
      }
    }

    if (failed.length > 0) {
      console.log('\n❌ Failed Benchmarks:');
      for (const result of failed) {
        console.log(`   ${result.name}`);
      }
    }

    // Generate JSON report
    const reportPath = path.join(__dirname, 'reports', 'simple-performance-report.json');
    const report = {
      generatedAt: new Date().toISOString(),
      summary: {
        total: this.results.length,
        successful: successful.length,
        failed: failed.length
      },
      results: this.results
    };

    await fs.mkdir(path.dirname(reportPath), { recursive: true });
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));

    console.log(`\n📄 Report saved to: ${reportPath}`);
  }

  /**
   * Run all benchmarks
   */
  async runAll(): Promise<void> {
    console.log('🚀 AgentDB Performance Benchmark Suite (Simplified)');
    console.log('='.repeat(80));

    const totalStart = performance.now();

    await this.benchmarkReflexionMemory();
    await this.benchmarkSkillLibrary();
    await this.benchmarkCausalGraph();

    const totalDuration = performance.now() - totalStart;

    console.log(`\n⏱️  Total benchmark duration: ${(totalDuration / 1000).toFixed(2)}s`);

    await this.generateReport();
  }
}

// Main execution
const benchmark = new SimpleBenchmark();
benchmark.runAll()
  .then(() => {
    console.log('\n✅ All benchmarks completed!');
    process.exit(0);
  })
  .catch((error) => {
    console.error('\n❌ Benchmark suite failed:', error);
    process.exit(1);
  });
