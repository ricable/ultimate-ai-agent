#!/usr/bin/env node

/**
 * ReasoningBank Performance Benchmark
 *
 * Tests the performance of AgentDB operations including:
 * - Pattern storage speed
 * - Pattern search/retrieval speed
 * - Memory distillation performance
 * - Pattern association generation
 * - Vector embedding operations
 *
 * Usage:
 *   node scripts/benchmark.js
 *   node scripts/benchmark.js --iterations 20
 */

import { getDatabase } from '../lib/db-utils.js';
import { storeResearchPattern } from '../lib/reasoningbank-integration.js';
import { v4 as uuidv4 } from 'uuid';
import chalk from 'chalk';

const ITERATIONS = parseInt(process.argv.find(arg => arg.startsWith('--iterations='))?.split('=')[1] || '10');

console.log(`
╔═══════════════════════════════════════════════════════════════════╗
║           REASONINGBANK PERFORMANCE BENCHMARK                      ║
╚═══════════════════════════════════════════════════════════════════╝

⚡ Running ${ITERATIONS} iterations for each operation...
`);

const db = getDatabase();

// Benchmark results
const results = {
  patternStorage: [],
  patternRetrieval: [],
  vectorSearch: [],
  distillation: [],
  association: []
};

// Helper to measure execution time
function measure(fn) {
  const start = performance.now();
  fn();
  return performance.now() - start;
}

// 1. Benchmark Pattern Storage
console.log(chalk.cyan('📥 Benchmarking pattern storage...'));
for (let i = 0; i < ITERATIONS; i++) {
  const jobId = uuidv4();

  // Create a mock job
  db.prepare(`
    INSERT INTO research_jobs (
      id, agent, task, status, progress, report_content,
      grounding_score, duration_seconds, tokens_used, memory_mb,
      created_at, started_at, completed_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), datetime('now'))
  `).run(
    jobId, 'benchmark-agent', `Benchmark task ${i + 1}`,
    'completed', 100, `This is a test report for benchmark iteration ${i + 1}`,
    0.85, 30, 5000, 128
  );

  // Measure pattern storage time
  const time = measure(() => {
    storeResearchPattern(jobId, {
      task: `Benchmark task ${i + 1}`,
      config: { depth: 5, timeBudget: 120 },
      duration: 30,
      citationCount: 10,
      sourceCount: 5,
      groundingScore: 0.85,
      success: true
    });
  });

  results.patternStorage.push(time);
  process.stdout.write(`   Iteration ${i + 1}/${ITERATIONS}: ${time.toFixed(2)}ms\r`);
}
console.log(`\n   ✅ Completed\n`);

// 2. Benchmark Pattern Retrieval
console.log(chalk.cyan('🔍 Benchmarking pattern retrieval...'));
const patternIds = db.prepare('SELECT id FROM reasoningbank_patterns LIMIT ?').all(ITERATIONS);

for (let i = 0; i < Math.min(ITERATIONS, patternIds.length); i++) {
  const time = measure(() => {
    db.prepare('SELECT * FROM reasoningbank_patterns WHERE id = ?').get(patternIds[i].id);
  });

  results.patternRetrieval.push(time);
  process.stdout.write(`   Iteration ${i + 1}/${ITERATIONS}: ${time.toFixed(2)}ms\r`);
}
console.log(`\n   ✅ Completed\n`);

// 3. Benchmark Vector Search (similarity queries)
console.log(chalk.cyan('🎯 Benchmarking vector search...'));
for (let i = 0; i < ITERATIONS; i++) {
  const time = measure(() => {
    db.prepare(`
      SELECT v1.*, v2.id as related_id,
             LENGTH(v1.content_text) as similarity
      FROM vector_embeddings v1
      JOIN vector_embeddings v2 ON v1.id != v2.id
      WHERE v1.source_type = 'pattern'
      LIMIT 10
    `).all();
  });

  results.vectorSearch.push(time);
  process.stdout.write(`   Iteration ${i + 1}/${ITERATIONS}: ${time.toFixed(2)}ms\r`);
}
console.log(`\n   ✅ Completed\n`);

// 4. Benchmark Memory Distillation (complex aggregation)
console.log(chalk.cyan('🧠 Benchmarking memory distillation queries...'));
for (let i = 0; i < ITERATIONS; i++) {
  const time = measure(() => {
    db.prepare(`
      SELECT
        task_category,
        COUNT(*) as pattern_count,
        AVG(confidence_score) as avg_confidence,
        AVG(success_rate) as avg_success_rate
      FROM memory_distillations
      GROUP BY task_category
    `).all();
  });

  results.distillation.push(time);
  process.stdout.write(`   Iteration ${i + 1}/${ITERATIONS}: ${time.toFixed(2)}ms\r`);
}
console.log(`\n   ✅ Completed\n`);

// 5. Benchmark Pattern Association (join operations)
console.log(chalk.cyan('🔗 Benchmarking pattern associations...'));
for (let i = 0; i < ITERATIONS; i++) {
  const time = measure(() => {
    db.prepare(`
      SELECT pa.*,
             p1.task as task_a,
             p2.task as task_b
      FROM pattern_associations pa
      JOIN reasoningbank_patterns p1 ON pa.pattern_id_a = p1.id
      JOIN reasoningbank_patterns p2 ON pa.pattern_id_b = p2.id
      WHERE pa.similarity_score > 0.5
      ORDER BY pa.similarity_score DESC
      LIMIT 20
    `).all();
  });

  results.association.push(time);
  process.stdout.write(`   Iteration ${i + 1}/${ITERATIONS}: ${time.toFixed(2)}ms\r`);
}
console.log(`\n   ✅ Completed\n`);

// Calculate statistics
function calculateStats(times) {
  const sorted = [...times].sort((a, b) => a - b);
  return {
    min: Math.min(...times),
    max: Math.max(...times),
    mean: times.reduce((a, b) => a + b, 0) / times.length,
    median: sorted[Math.floor(sorted.length / 2)],
    p95: sorted[Math.floor(sorted.length * 0.95)],
    p99: sorted[Math.floor(sorted.length * 0.99)]
  };
}

// Display results
console.log(chalk.bold.cyan(`
╔═══════════════════════════════════════════════════════════════════╗
║                    BENCHMARK RESULTS                               ║
╚═══════════════════════════════════════════════════════════════════╝
`));

const operations = [
  { name: 'Pattern Storage', key: 'patternStorage', target: 50 },
  { name: 'Pattern Retrieval', key: 'patternRetrieval', target: 5 },
  { name: 'Vector Search', key: 'vectorSearch', target: 20 },
  { name: 'Memory Distillation', key: 'distillation', target: 10 },
  { name: 'Pattern Association', key: 'association', target: 15 }
];

operations.forEach(op => {
  const stats = calculateStats(results[op.key]);
  const passed = stats.p95 < op.target;
  const status = passed ? chalk.green('✅ PASS') : chalk.yellow('⚠️  SLOW');

  console.log(chalk.bold(`\n📊 ${op.name} (target: ${op.target}ms):`));
  console.log(`   Status:  ${status}`);
  console.log(`   Mean:    ${chalk.cyan(stats.mean.toFixed(2))}ms`);
  console.log(`   Median:  ${chalk.cyan(stats.median.toFixed(2))}ms`);
  console.log(`   Min:     ${chalk.green(stats.min.toFixed(2))}ms`);
  console.log(`   Max:     ${chalk.yellow(stats.max.toFixed(2))}ms`);
  console.log(`   P95:     ${chalk.cyan(stats.p95.toFixed(2))}ms`);
  console.log(`   P99:     ${chalk.cyan(stats.p99.toFixed(2))}ms`);
});

// Overall assessment
const allStats = Object.values(results).flat();
const overallStats = calculateStats(allStats);

console.log(chalk.bold.cyan(`
\n╔═══════════════════════════════════════════════════════════════════╗
║                    OVERALL PERFORMANCE                             ║
╚═══════════════════════════════════════════════════════════════════╝
`));

console.log(`   Total Operations:  ${chalk.cyan(allStats.length)}`);
console.log(`   Total Time:        ${chalk.cyan((allStats.reduce((a, b) => a + b, 0)).toFixed(2))}ms`);
console.log(`   Average Time:      ${chalk.cyan(overallStats.mean.toFixed(2))}ms`);
console.log(`   Throughput:        ${chalk.cyan(Math.floor(1000 / overallStats.mean))} ops/sec`);

// Performance rating
let rating = '🌟 Excellent';
if (overallStats.mean > 20) rating = '✅ Good';
if (overallStats.mean > 50) rating = '⚠️  Needs Optimization';
if (overallStats.mean > 100) rating = '❌ Poor';

console.log(`\n   Performance Rating: ${rating}\n`);

// Optimization recommendations
if (overallStats.mean > 50) {
  console.log(chalk.yellow(`
📌 OPTIMIZATION RECOMMENDATIONS:

1. Add database indexes:
   - CREATE INDEX idx_patterns_task ON reasoningbank_patterns(task);
   - CREATE INDEX idx_patterns_reward ON reasoningbank_patterns(reward DESC);
   - CREATE INDEX idx_associations_similarity ON pattern_associations(similarity_score DESC);

2. Enable WAL mode (already enabled, but verify):
   - PRAGMA journal_mode=WAL;

3. Increase cache size:
   - PRAGMA cache_size=-64000; (64MB cache)

4. Use prepared statements (already implemented)

5. Consider batch operations for multiple inserts
`));
}

db.close();

console.log(chalk.green('✅ Benchmark completed successfully!\n'));
