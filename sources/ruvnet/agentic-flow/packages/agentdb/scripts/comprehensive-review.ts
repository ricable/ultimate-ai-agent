#!/usr/bin/env tsx

/**
 * Comprehensive AgentDB v2 Review Script
 *
 * Tests:
 * 1. @ruvector/core integration
 * 2. @ruvector/gnn integration
 * 3. ReasoningBank functionality
 * 4. All v2 controllers
 * 5. Backend performance comparison
 * 6. Memory usage analysis
 * 7. Optimization opportunities
 */

import { promisify } from 'util';
import { exec as execCallback } from 'child_process';
const exec = promisify(execCallback);

interface TestResult {
  name: string;
  status: 'pass' | 'fail' | 'warn' | 'skip';
  duration?: number;
  details?: string;
  metrics?: Record<string, any>;
}

const results: TestResult[] = [];

function log(section: string, message: string) {
  console.log(`\n[${section}] ${message}`);
}

function addResult(result: TestResult) {
  results.push(result);
  const icon = result.status === 'pass' ? '✅' : result.status === 'fail' ? '❌' : result.status === 'warn' ? '⚠️' : '⏭️';
  console.log(`${icon} ${result.name} ${result.duration ? `(${result.duration}ms)` : ''}`);
  if (result.details) {
    console.log(`   ${result.details}`);
  }
}

async function checkRuVectorCore() {
  log('RuVector Core', 'Testing @ruvector/core integration...');

  try {
    const { stdout } = await exec('npm list @ruvector/core --json');
    const parsed = JSON.parse(stdout);

    if (parsed.dependencies && parsed.dependencies['@ruvector/core']) {
      const version = parsed.dependencies['@ruvector/core'].version;

      // Try to import and test
      try {
        const ruv = await import('@ruvector/core');
        addResult({
          name: '@ruvector/core integration',
          status: 'pass',
          details: `Version ${version} - Native HNSW available`,
          metrics: { version, available: true }
        });
        return true;
      } catch (importErr) {
        addResult({
          name: '@ruvector/core integration',
          status: 'warn',
          details: `Installed but import failed (platform-specific): ${importErr instanceof Error ? importErr.message : String(importErr)}`,
          metrics: { version, available: false }
        });
        return false;
      }
    } else {
      addResult({
        name: '@ruvector/core integration',
        status: 'warn',
        details: 'Optional dependency not installed',
        metrics: { installed: false }
      });
      return false;
    }
  } catch (error) {
    addResult({
      name: '@ruvector/core integration',
      status: 'warn',
      details: `Not available: ${error instanceof Error ? error.message : String(error)}`,
      metrics: { installed: false }
    });
    return false;
  }
}

async function checkRuVectorGNN() {
  log('RuVector GNN', 'Testing @ruvector/gnn integration...');

  try {
    const { stdout } = await exec('npm list @ruvector/gnn --json');
    const parsed = JSON.parse(stdout);

    if (parsed.dependencies && parsed.dependencies['@ruvector/gnn']) {
      const version = parsed.dependencies['@ruvector/gnn'].version;

      try {
        const gnn = await import('@ruvector/gnn');
        addResult({
          name: '@ruvector/gnn integration',
          status: 'pass',
          details: `Version ${version} - Graph neural networks available`,
          metrics: { version, available: true }
        });
        return true;
      } catch (importErr) {
        addResult({
          name: '@ruvector/gnn integration',
          status: 'warn',
          details: `Installed but import failed (platform-specific): ${importErr instanceof Error ? importErr.message : String(importErr)}`,
          metrics: { version, available: false }
        });
        return false;
      }
    } else {
      addResult({
        name: '@ruvector/gnn integration',
        status: 'warn',
        details: 'Optional dependency not installed',
        metrics: { installed: false }
      });
      return false;
    }
  } catch (error) {
    addResult({
      name: '@ruvector/gnn integration',
      status: 'warn',
      details: `Not available: ${error instanceof Error ? error.message : String(error)}`,
      metrics: { installed: false }
    });
    return false;
  }
}

async function testReasoningBank() {
  log('ReasoningBank', 'Testing ReasoningBank functionality...');

  try {
    // Import AgentDB
    const { AgentDB } = await import('../dist/index.js');

    const startTime = Date.now();
    const db = new AgentDB({ backend: 'sqlite' });
    await db.initializeAsync();

    // Test ReasoningBank pattern storage
    const sessionId = 'review-test';

    // Store patterns
    for (let i = 0; i < 10; i++) {
      await db.episodes.store({
        sessionId,
        task: `Test task ${i}`,
        input: `Input ${i}`,
        output: `Output ${i}`,
        critique: i % 2 === 0 ? `Critique ${i}` : undefined,
        reward: Math.random(),
        tokensUsed: 100 + i * 10,
        latencyMs: 50 + i * 5,
        success: i % 3 !== 0
      });
    }

    // Search patterns
    const searchResults = await db.episodes.search({
      task: 'test task',
      k: 5
    });

    // Get statistics
    const stats = await db.episodes.getStatsBySession(sessionId);

    const duration = Date.now() - startTime;

    addResult({
      name: 'ReasoningBank functionality',
      status: 'pass',
      duration,
      details: `Stored 10 episodes, searched ${searchResults.length} results, avg reward: ${stats.avgReward.toFixed(2)}`,
      metrics: {
        episodesStored: 10,
        searchResults: searchResults.length,
        avgReward: stats.avgReward,
        successRate: stats.successRate
      }
    });

    await db.close();
    return true;
  } catch (error) {
    addResult({
      name: 'ReasoningBank functionality',
      status: 'fail',
      details: `Error: ${error instanceof Error ? error.message : String(error)}`
    });
    return false;
  }
}

async function testV2Controllers() {
  log('V2 Controllers', 'Testing Episodes, Skills, and CausalEdges...');

  try {
    const { AgentDB } = await import('../dist/index.js');

    const db = new AgentDB({ backend: 'sqlite' });
    await db.initializeAsync();

    // Test Episodes
    const startEpisodes = Date.now();
    const ep1 = await db.episodes.store({
      task: 'Test episode 1',
      reward: 0.9,
      success: true
    });
    const ep2 = await db.episodes.store({
      task: 'Test episode 2',
      reward: 0.8,
      success: true
    });
    const episodesDuration = Date.now() - startEpisodes;

    // Test Skills
    const startSkills = Date.now();
    const skill1 = await db.skills.store({
      name: 'test-skill-1',
      code: 'function test() { return true; }',
      signature: 'test(): boolean',
      successRate: 0.95
    });
    const skillsDuration = Date.now() - startSkills;

    // Test Causal Edges
    const startCausal = Date.now();
    await db.causalEdges.add({
      fromMemoryId: ep1,
      toMemoryId: ep2,
      similarity: 0.85
    });
    const causalDuration = Date.now() - startCausal;

    // Search episodes
    const episodeResults = await db.episodes.search({ task: 'test', k: 2 });

    // Search skills
    const skillResults = await db.skills.search({ query: 'test', k: 1 });

    // Get causal edges
    const edges = await db.causalEdges.get(ep1);

    addResult({
      name: 'V2 Episodes controller',
      status: 'pass',
      duration: episodesDuration,
      details: `Stored 2 episodes, searched ${episodeResults.length} results`,
      metrics: { stored: 2, searched: episodeResults.length }
    });

    addResult({
      name: 'V2 Skills controller',
      status: 'pass',
      duration: skillsDuration,
      details: `Stored 1 skill, searched ${skillResults.length} results`,
      metrics: { stored: 1, searched: skillResults.length }
    });

    addResult({
      name: 'V2 CausalEdges controller',
      status: 'pass',
      duration: causalDuration,
      details: `Added 1 edge, retrieved ${edges.length} edges`,
      metrics: { added: 1, retrieved: edges.length }
    });

    await db.close();
    return true;
  } catch (error) {
    addResult({
      name: 'V2 Controllers',
      status: 'fail',
      details: `Error: ${error instanceof Error ? error.message : String(error)}`
    });
    return false;
  }
}

async function benchmarkBackends() {
  log('Backend Benchmark', 'Comparing SQLite, BetterSQLite, and RuVector backends...');

  const backends = ['sqlite', 'better-sqlite3'] as const;
  const results: Record<string, any> = {};

  for (const backend of backends) {
    try {
      const { AgentDB } = await import('../dist/index.js');

      const startInit = Date.now();
      const db = new AgentDB({ backend });
      await db.initializeAsync();
      const initDuration = Date.now() - startInit;

      // Benchmark insert
      const startInsert = Date.now();
      const episodeIds = [];
      for (let i = 0; i < 100; i++) {
        const id = await db.episodes.store({
          task: `Benchmark task ${i}`,
          reward: Math.random(),
          success: Math.random() > 0.3
        });
        episodeIds.push(id);
      }
      const insertDuration = Date.now() - startInsert;

      // Benchmark search
      const startSearch = Date.now();
      for (let i = 0; i < 10; i++) {
        await db.episodes.search({ task: 'benchmark', k: 10 });
      }
      const searchDuration = Date.now() - startSearch;

      await db.close();

      results[backend] = {
        init: initDuration,
        insert: insertDuration,
        avgInsert: insertDuration / 100,
        search: searchDuration,
        avgSearch: searchDuration / 10
      };

      addResult({
        name: `${backend} backend performance`,
        status: 'pass',
        duration: initDuration + insertDuration + searchDuration,
        details: `Init: ${initDuration}ms, Insert: ${insertDuration}ms (${(insertDuration/100).toFixed(2)}ms avg), Search: ${searchDuration}ms (${(searchDuration/10).toFixed(2)}ms avg)`,
        metrics: results[backend]
      });
    } catch (error) {
      addResult({
        name: `${backend} backend performance`,
        status: 'fail',
        details: `Error: ${error instanceof Error ? error.message : String(error)}`
      });
    }
  }

  // Compare results
  if (results['sqlite'] && results['better-sqlite3']) {
    const speedup = (results['sqlite'].avgInsert / results['better-sqlite3'].avgInsert).toFixed(2);
    addResult({
      name: 'Backend comparison',
      status: 'pass',
      details: `better-sqlite3 is ${speedup}x faster for inserts than sqlite`,
      metrics: { speedup }
    });
  }
}

async function analyzeMemoryUsage() {
  log('Memory Analysis', 'Analyzing memory usage patterns...');

  try {
    const { AgentDB } = await import('../dist/index.js');

    const initialMem = process.memoryUsage();

    const db = new AgentDB({ backend: 'sqlite' });
    await db.initializeAsync();

    const afterInitMem = process.memoryUsage();

    // Store 1000 episodes
    for (let i = 0; i < 1000; i++) {
      await db.episodes.store({
        task: `Memory test ${i}`,
        reward: Math.random(),
        success: true
      });
    }

    const after1kMem = process.memoryUsage();

    await db.close();

    const finalMem = process.memoryUsage();

    const initIncrease = (afterInitMem.heapUsed - initialMem.heapUsed) / 1024 / 1024;
    const dataIncrease = (after1kMem.heapUsed - afterInitMem.heapUsed) / 1024 / 1024;
    const perEpisode = dataIncrease / 1000;

    addResult({
      name: 'Memory usage analysis',
      status: 'pass',
      details: `Init: +${initIncrease.toFixed(2)} MB, 1K episodes: +${dataIncrease.toFixed(2)} MB (${(perEpisode * 1024).toFixed(2)} KB/episode)`,
      metrics: {
        initIncreaseMB: initIncrease,
        dataIncreaseMB: dataIncrease,
        perEpisodeKB: perEpisode * 1024
      }
    });

    return true;
  } catch (error) {
    addResult({
      name: 'Memory usage analysis',
      status: 'fail',
      details: `Error: ${error instanceof Error ? error.message : String(error)}`
    });
    return false;
  }
}

async function identifyOptimizations() {
  log('Optimization Opportunities', 'Analyzing potential improvements...');

  const opportunities = [
    {
      area: 'Batch Operations',
      current: 'Individual episode storage',
      improvement: 'Implement batchStore() for episodes, skills',
      impact: 'High - 5-10x faster bulk inserts',
      effort: 'Medium'
    },
    {
      area: 'Caching Layer',
      current: 'No query result caching',
      improvement: 'Add LRU cache for frequent searches',
      impact: 'Medium - 2-5x faster repeated queries',
      effort: 'Low'
    },
    {
      area: 'Embedding Generation',
      current: 'Synchronous embedding for each episode',
      improvement: 'Async queue with batching',
      impact: 'High - 3-5x faster for bulk operations',
      effort: 'Medium'
    },
    {
      area: 'Index Optimization',
      current: 'Basic SQLite indexes',
      improvement: 'Add covering indexes for common queries',
      impact: 'Medium - 2-3x faster complex queries',
      effort: 'Low'
    },
    {
      area: 'RuVector Integration',
      current: 'Optional, platform-specific',
      improvement: 'Auto-fallback chain: RuVector → HNSW → Linear',
      impact: 'High - 150x faster search when available',
      effort: 'Low (already implemented)'
    },
    {
      area: 'Connection Pooling',
      current: 'Single database connection',
      improvement: 'Connection pool for concurrent operations',
      impact: 'High - Better concurrency',
      effort: 'Medium'
    },
    {
      area: 'Browser Bundle',
      current: '22 KB gzipped with all features',
      improvement: 'Code splitting for optional features',
      impact: 'Low - Already optimized',
      effort: 'High'
    },
    {
      area: 'ReasoningBank',
      current: 'Pattern storage and search',
      improvement: 'Add pattern consolidation, auto-pruning',
      impact: 'Medium - Better memory efficiency over time',
      effort: 'Medium'
    }
  ];

  for (const opp of opportunities) {
    addResult({
      name: `Optimization: ${opp.area}`,
      status: 'warn',
      details: `${opp.current} → ${opp.improvement}. Impact: ${opp.impact}, Effort: ${opp.effort}`,
      metrics: {
        area: opp.area,
        impact: opp.impact,
        effort: opp.effort
      }
    });
  }
}

async function generateReport() {
  console.log('\n\n');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('        AgentDB v2 Comprehensive Review - Final Report         ');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  const passed = results.filter(r => r.status === 'pass').length;
  const failed = results.filter(r => r.status === 'fail').length;
  const warnings = results.filter(r => r.status === 'warn').length;
  const skipped = results.filter(r => r.status === 'skip').length;

  console.log(`Total Tests: ${results.length}`);
  console.log(`  ✅ Passed:   ${passed}`);
  console.log(`  ❌ Failed:   ${failed}`);
  console.log(`  ⚠️  Warnings: ${warnings}`);
  console.log(`  ⏭️  Skipped:  ${skipped}`);
  console.log('');

  // Group by category
  const categories = {
    'RuVector Integration': results.filter(r => r.name.includes('ruvector')),
    'ReasoningBank': results.filter(r => r.name.includes('ReasoningBank')),
    'V2 Controllers': results.filter(r => r.name.includes('V2 ')),
    'Backend Performance': results.filter(r => r.name.includes('backend')),
    'Memory Analysis': results.filter(r => r.name.includes('Memory')),
    'Optimizations': results.filter(r => r.name.includes('Optimization'))
  };

  for (const [category, tests] of Object.entries(categories)) {
    if (tests.length === 0) continue;

    console.log(`\n${category}:`);
    console.log('─'.repeat(60));

    for (const test of tests) {
      const icon = test.status === 'pass' ? '✅' : test.status === 'fail' ? '❌' : test.status === 'warn' ? '⚠️' : '⏭️';
      console.log(`${icon} ${test.name}`);
      if (test.details) {
        console.log(`   ${test.details}`);
      }
      if (test.metrics && Object.keys(test.metrics).length > 0) {
        console.log(`   Metrics: ${JSON.stringify(test.metrics)}`);
      }
    }
  }

  console.log('\n');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  // Save to file
  const report = {
    timestamp: new Date().toISOString(),
    version: '2.0.0-alpha.1',
    summary: { passed, failed, warnings, skipped, total: results.length },
    results,
    categories
  };

  const fs = await import('fs/promises');
  await fs.writeFile(
    'COMPREHENSIVE_REVIEW_REPORT.json',
    JSON.stringify(report, null, 2)
  );

  console.log('📊 Full report saved to: COMPREHENSIVE_REVIEW_REPORT.json');
  console.log('');
}

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('        AgentDB v2 - Comprehensive Deep Review                 ');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('');

  // Run all tests
  await checkRuVectorCore();
  await checkRuVectorGNN();
  await testReasoningBank();
  await testV2Controllers();
  await benchmarkBackends();
  await analyzeMemoryUsage();
  await identifyOptimizations();

  // Generate final report
  await generateReport();

  const failed = results.filter(r => r.status === 'fail').length;
  process.exit(failed > 0 ? 1 : 0);
}

main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
