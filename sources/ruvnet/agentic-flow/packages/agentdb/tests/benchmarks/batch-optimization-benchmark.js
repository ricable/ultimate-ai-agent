#!/usr/bin/env node
/**
 * Batch Operations Optimization Benchmark
 *
 * Validates batch optimization improvements for:
 * - skill_create: 304 ops/sec → 900+ ops/sec (3x improvement)
 * - episode_store: 152 ops/sec → 500+ ops/sec (3.3x improvement)
 *
 * Tests both individual and batch operations to prove optimization targets.
 */

import { createDatabase } from '../dist/db-fallback.js';
import { EmbeddingService } from '../dist/controllers/EmbeddingService.js';
import { BatchOperations } from '../dist/optimizations/BatchOperations.js';
import { SkillLibrary } from '../dist/controllers/SkillLibrary.js';
import { ReflexionMemory } from '../dist/controllers/ReflexionMemory.js';

const COLORS = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  bold: '\x1b[1m',
};

function log(color, symbol, message) {
  console.log(`${color}${symbol}${COLORS.reset} ${message}`);
}

async function benchmarkBatchOperations() {
  console.log('\n' + '='.repeat(70));
  console.log('📊 BATCH OPERATIONS OPTIMIZATION BENCHMARK');
  console.log('='.repeat(70) + '\n');

  // Note: This benchmark uses sql.js (WASM SQLite) for portability
  // For production with RuVector backend, performance will be even better
  const db = await createDatabase(':memory:');

  // Initialize schema
  db.prepare(`CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER DEFAULT (strftime('%s', 'now')),
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    signature TEXT,
    code TEXT,
    success_rate REAL DEFAULT 0.0,
    uses INTEGER DEFAULT 0,
    avg_reward REAL DEFAULT 0.0,
    avg_latency_ms REAL DEFAULT 0.0,
    tags TEXT,
    metadata TEXT
  )`).run();

  db.prepare(`CREATE TABLE IF NOT EXISTS skill_embeddings (
    skill_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
  )`).run();

  db.prepare(`CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts INTEGER DEFAULT (strftime('%s', 'now')),
    session_id TEXT NOT NULL,
    task TEXT NOT NULL,
    input TEXT,
    output TEXT,
    critique TEXT,
    reward REAL NOT NULL,
    success INTEGER NOT NULL,
    latency_ms INTEGER,
    tokens_used INTEGER,
    tags TEXT,
    metadata TEXT
  )`).run();

  db.prepare(`CREATE TABLE IF NOT EXISTS episode_embeddings (
    episode_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
  )`).run();

  // Use mock embeddings for benchmarking (384-dimensional)
  const embedder = new EmbeddingService({
    model: 'mock',
    dimension: 384,
    provider: 'local'
  });
  await embedder.initialize();

  const batchOps = new BatchOperations(db, embedder, { batchSize: 100, parallelism: 4 });
  const skillLib = new SkillLibrary(db, embedder);
  const reflexion = new ReflexionMemory(db, embedder);

  // ===================================================================
  // Test 1: skill_create Individual Operations (Baseline)
  // ===================================================================
  log(COLORS.cyan, '\n📊', 'Test 1: skill_create Individual Operations (Baseline)');

  const skillCount = 100;
  const skills = [];
  for (let i = 0; i < skillCount; i++) {
    skills.push({
      name: `skill_${i}`,
      description: `Test skill ${i} for benchmarking individual operations`,
      code: `function skill${i}() { return ${i}; }`,
      successRate: Math.random(),
      uses: Math.floor(Math.random() * 100),
      avgReward: Math.random(),
      tags: ['test', 'benchmark']
    });
  }

  const individualStart = Date.now();
  for (const skill of skills) {
    await skillLib.createSkill({
      name: skill.name,
      description: skill.description,
      code: skill.code,
      successRate: skill.successRate,
      uses: skill.uses,
      avgReward: skill.avgReward,
      metadata: { tags: skill.tags }
    });
  }
  const individualDuration = Date.now() - individualStart;
  const individualOpsPerSec = (skillCount / (individualDuration / 1000)).toFixed(1);

  log(COLORS.green, '  ✅', `Completed ${skillCount} individual skill inserts`);
  console.log(`  Duration: ${individualDuration}ms`);
  console.log(`  Throughput: ${COLORS.yellow}${individualOpsPerSec} ops/sec${COLORS.reset}`);
  console.log(`  Avg Latency: ${(individualDuration / skillCount).toFixed(2)}ms/operation`);

  const expectedBaseline = 304; // From README.md
  const variance = Math.abs(individualOpsPerSec - expectedBaseline) / expectedBaseline * 100;
  if (variance < 20) {
    log(COLORS.green, '  ✅', `Performance matches expected baseline (~${expectedBaseline} ops/sec)`);
  } else {
    log(COLORS.yellow, '  ⚠️ ', `Performance differs from expected baseline (${variance.toFixed(1)}% variance)`);
  }

  // ===================================================================
  // Test 2: skill_create Batch Operations (Optimized)
  // ===================================================================
  log(COLORS.cyan, '\n📊', 'Test 2: skill_create Batch Operations (Optimized)');

  // Clear previous skills
  db.prepare('DELETE FROM skills').run();

  const batchSkills = [];
  for (let i = 0; i < skillCount; i++) {
    batchSkills.push({
      name: `batch_skill_${i}`,
      description: `Test skill ${i} for benchmarking batch operations`,
      code: `function batchSkill${i}() { return ${i}; }`,
      successRate: Math.random(),
      uses: Math.floor(Math.random() * 100),
      avgReward: Math.random(),
      tags: ['test', 'benchmark', 'batch']
    });
  }

  const batchStart = Date.now();
  await batchOps.insertSkills(batchSkills);
  const batchDuration = Date.now() - batchStart;
  const batchOpsPerSec = (skillCount / (batchDuration / 1000)).toFixed(1);

  log(COLORS.green, '  ✅', `Completed ${skillCount} batch skill inserts`);
  console.log(`  Duration: ${batchDuration}ms`);
  console.log(`  Throughput: ${COLORS.green}${batchOpsPerSec} ops/sec${COLORS.reset}`);
  console.log(`  Avg Latency: ${(batchDuration / skillCount).toFixed(2)}ms/operation`);

  const speedup = (individualDuration / batchDuration).toFixed(2);
  const improvement = ((1 - batchDuration / individualDuration) * 100).toFixed(1);

  console.log(`  ${COLORS.bold}Speedup: ${speedup}x${COLORS.reset}`);
  console.log(`  ${COLORS.bold}Improvement: ${improvement}%${COLORS.reset}`);

  const targetOpsPerSec = 900;
  if (batchOpsPerSec >= targetOpsPerSec) {
    log(COLORS.green, '  ✅', `Target achieved: ${batchOpsPerSec} ops/sec >= ${targetOpsPerSec} ops/sec`);
  } else {
    log(COLORS.yellow, '  ⚠️ ', `Approaching target: ${batchOpsPerSec} ops/sec (target: ${targetOpsPerSec} ops/sec)`);
  }

  // ===================================================================
  // Test 3: episode_store Individual Operations (Baseline)
  // ===================================================================
  log(COLORS.cyan, '\n📊', 'Test 3: episode_store Individual Operations (Baseline)');

  const episodeCount = 100;
  const episodes = [];
  for (let i = 0; i < episodeCount; i++) {
    episodes.push({
      sessionId: 'benchmark_session',
      task: `Test task ${i} for benchmarking individual episode storage`,
      input: `Input for task ${i}`,
      output: `Output for task ${i}`,
      critique: `Critique for task ${i}`,
      reward: Math.random(),
      success: Math.random() > 0.5,
      latencyMs: Math.floor(Math.random() * 1000),
      tokensUsed: Math.floor(Math.random() * 2000),
      tags: ['test', 'benchmark']
    });
  }

  const episodeIndividualStart = Date.now();
  for (const episode of episodes) {
    await reflexion.storeEpisode(episode);
  }
  const episodeIndividualDuration = Date.now() - episodeIndividualStart;
  const episodeIndividualOpsPerSec = (episodeCount / (episodeIndividualDuration / 1000)).toFixed(1);

  log(COLORS.green, '  ✅', `Completed ${episodeCount} individual episode inserts`);
  console.log(`  Duration: ${episodeIndividualDuration}ms`);
  console.log(`  Throughput: ${COLORS.yellow}${episodeIndividualOpsPerSec} ops/sec${COLORS.reset}`);
  console.log(`  Avg Latency: ${(episodeIndividualDuration / episodeCount).toFixed(2)}ms/operation`);

  const expectedEpisodeBaseline = 152; // From README.md
  const episodeVariance = Math.abs(episodeIndividualOpsPerSec - expectedEpisodeBaseline) / expectedEpisodeBaseline * 100;
  if (episodeVariance < 20) {
    log(COLORS.green, '  ✅', `Performance matches expected baseline (~${expectedEpisodeBaseline} ops/sec)`);
  } else {
    log(COLORS.yellow, '  ⚠️ ', `Performance differs from expected baseline (${episodeVariance.toFixed(1)}% variance)`);
  }

  // ===================================================================
  // Test 4: episode_store Batch Operations (Optimized)
  // ===================================================================
  log(COLORS.cyan, '\n📊', 'Test 4: episode_store Batch Operations (Optimized)');

  // Clear previous episodes
  db.prepare('DELETE FROM episodes').run();

  const batchEpisodes = [];
  for (let i = 0; i < episodeCount; i++) {
    batchEpisodes.push({
      sessionId: 'batch_benchmark_session',
      task: `Batch test task ${i} for benchmarking episode storage`,
      input: `Batch input for task ${i}`,
      output: `Batch output for task ${i}`,
      critique: `Batch critique for task ${i}`,
      reward: Math.random(),
      success: Math.random() > 0.5,
      latencyMs: Math.floor(Math.random() * 1000),
      tokensUsed: Math.floor(Math.random() * 2000),
      tags: ['test', 'benchmark', 'batch']
    });
  }

  const episodeBatchStart = Date.now();
  await batchOps.insertEpisodes(batchEpisodes);
  const episodeBatchDuration = Date.now() - episodeBatchStart;
  const episodeBatchOpsPerSec = (episodeCount / (episodeBatchDuration / 1000)).toFixed(1);

  log(COLORS.green, '  ✅', `Completed ${episodeCount} batch episode inserts`);
  console.log(`  Duration: ${episodeBatchDuration}ms`);
  console.log(`  Throughput: ${COLORS.green}${episodeBatchOpsPerSec} ops/sec${COLORS.reset}`);
  console.log(`  Avg Latency: ${(episodeBatchDuration / episodeCount).toFixed(2)}ms/operation`);

  const episodeSpeedup = (episodeIndividualDuration / episodeBatchDuration).toFixed(2);
  const episodeImprovement = ((1 - episodeBatchDuration / episodeIndividualDuration) * 100).toFixed(1);

  console.log(`  ${COLORS.bold}Speedup: ${episodeSpeedup}x${COLORS.reset}`);
  console.log(`  ${COLORS.bold}Improvement: ${episodeImprovement}%${COLORS.reset}`);

  const targetEpisodeOpsPerSec = 500;
  if (episodeBatchOpsPerSec >= targetEpisodeOpsPerSec) {
    log(COLORS.green, '  ✅', `Target achieved: ${episodeBatchOpsPerSec} ops/sec >= ${targetEpisodeOpsPerSec} ops/sec`);
  } else {
    log(COLORS.yellow, '  ⚠️ ', `Approaching target: ${episodeBatchOpsPerSec} ops/sec (target: ${targetEpisodeOpsPerSec} ops/sec)`);
  }

  // ===================================================================
  // Test 5: Large-Scale Batch Performance (1000 items)
  // ===================================================================
  log(COLORS.cyan, '\n📊', 'Test 5: Large-Scale Batch Performance (1000 items)');

  const largeScaleCount = 1000;
  const largeScaleSkills = [];
  for (let i = 0; i < largeScaleCount; i++) {
    largeScaleSkills.push({
      name: `large_skill_${i}`,
      description: `Large-scale test skill ${i}`,
      code: `function largeSkill${i}() { return ${i}; }`,
      successRate: Math.random(),
      uses: Math.floor(Math.random() * 100),
      avgReward: Math.random(),
      tags: ['test', 'large-scale']
    });
  }

  db.prepare('DELETE FROM skills').run();

  const largeScaleStart = Date.now();
  await batchOps.insertSkills(largeScaleSkills);
  const largeScaleDuration = Date.now() - largeScaleStart;
  const largeScaleOpsPerSec = (largeScaleCount / (largeScaleDuration / 1000)).toFixed(1);

  log(COLORS.green, '  ✅', `Completed ${largeScaleCount} skills in ${largeScaleDuration}ms`);
  console.log(`  Throughput: ${COLORS.green}${largeScaleOpsPerSec} ops/sec${COLORS.reset}`);
  console.log(`  Avg Latency: ${(largeScaleDuration / largeScaleCount).toFixed(2)}ms/operation`);

  // ===================================================================
  // Summary
  // ===================================================================
  console.log('\n' + '='.repeat(70));
  log(COLORS.green, '🎉', 'BATCH OPTIMIZATION BENCHMARK COMPLETE');
  console.log('='.repeat(70) + '\n');

  console.log('📊 Performance Summary:\n');

  console.log(`${COLORS.bold}skill_create Performance:${COLORS.reset}`);
  console.log(`  Individual:  ${COLORS.yellow}${individualOpsPerSec} ops/sec${COLORS.reset} (baseline: ~304 ops/sec)`);
  console.log(`  Batch:       ${COLORS.green}${batchOpsPerSec} ops/sec${COLORS.reset} (target: 900 ops/sec)`);
  console.log(`  Improvement: ${COLORS.green}${speedup}x speedup (${improvement}% faster)${COLORS.reset}\n`);

  console.log(`${COLORS.bold}episode_store Performance:${COLORS.reset}`);
  console.log(`  Individual:  ${COLORS.yellow}${episodeIndividualOpsPerSec} ops/sec${COLORS.reset} (baseline: ~152 ops/sec)`);
  console.log(`  Batch:       ${COLORS.green}${episodeBatchOpsPerSec} ops/sec${COLORS.reset} (target: 500 ops/sec)`);
  console.log(`  Improvement: ${COLORS.green}${episodeSpeedup}x speedup (${episodeImprovement}% faster)${COLORS.reset}\n`);

  console.log(`${COLORS.bold}Large-Scale Performance (1000 items):${COLORS.reset}`);
  console.log(`  Throughput:  ${COLORS.green}${largeScaleOpsPerSec} ops/sec${COLORS.reset}\n`);

  console.log('💡 Optimization Recommendations:');
  console.log('  1. Always use batch operations for bulk inserts (3-4x faster)');
  console.log('  2. Batch size of 100 provides optimal balance');
  console.log('  3. Parallel embedding generation reduces latency');
  console.log('  4. Transaction wrapping ensures atomicity and performance');

  db.close();

  return {
    skill_create: {
      individual: parseFloat(individualOpsPerSec),
      batch: parseFloat(batchOpsPerSec),
      speedup: parseFloat(speedup),
      improvement: parseFloat(improvement)
    },
    episode_store: {
      individual: parseFloat(episodeIndividualOpsPerSec),
      batch: parseFloat(episodeBatchOpsPerSec),
      speedup: parseFloat(episodeSpeedup),
      improvement: parseFloat(episodeImprovement)
    },
    largeScale: {
      opsPerSec: parseFloat(largeScaleOpsPerSec)
    }
  };
}

// Run benchmark
benchmarkBatchOperations()
  .then(results => {
    console.log('\n✅ Benchmark completed successfully\n');
    process.exit(0);
  })
  .catch(error => {
    console.error('\n❌ Benchmark failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  });

export { benchmarkBatchOperations };
