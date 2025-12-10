/**
 * ReasoningBank Performance Benchmark
 * Compares v1 vs v2 performance for pattern storage and search
 */

const { performance } = require('perf_hooks');
const { createDatabase } = require('../dist/db-fallback.js');
const { ReasoningBank } = require('../dist/controllers/ReasoningBank.js');
const { EmbeddingService } = require('../dist/controllers/EmbeddingService.js');
const fs = require('fs');

// Benchmark configuration
const PATTERNS_COUNT = 1000;
const SEARCH_ITERATIONS = 100;
const TASK_TYPES = ['coding', 'debugging', 'optimization', 'refactoring', 'testing'];

async function measureMemory() {
  const usage = process.memoryUsage();
  return {
    heapUsed: Math.round(usage.heapUsed / 1024 / 1024), // MB
    heapTotal: Math.round(usage.heapTotal / 1024 / 1024), // MB
    external: Math.round(usage.external / 1024 / 1024), // MB
    rss: Math.round(usage.rss / 1024 / 1024) // MB
  };
}

async function setupDatabase(version, useBackends = false) {
  const dbPath = version === 'v1' ? ':memory:' : ':memory:';
  const db = await createDatabase(dbPath);

  // Load schemas
  const schema = fs.readFileSync('./dist/schemas/schema.sql', 'utf-8');
  const frontierSchema = fs.readFileSync('./dist/schemas/frontier-schema.sql', 'utf-8');
  db.exec(schema);
  db.exec(frontierSchema);

  const embedder = new EmbeddingService({
    model: 'Xenova/all-MiniLM-L6-v2',
    dimension: 384,
    provider: 'transformers'
  });

  await embedder.initialize();

  // v1: Only db and embedder
  // v2: Can use vectorBackend for HNSW indexing
  let bank;
  if (version === 'v1') {
    bank = new ReasoningBank(db, embedder);
  } else {
    // v2 with optional backends
    let vectorBackend = undefined;
    if (useBackends) {
      try {
        const { detectBackend } = require('../dist/backends/detector.js');
        const detection = await detectBackend();
        if (detection.backend === 'hnswlib') {
          const { HNSWVectorBackend } = require('../dist/backends/hnswlib-backend.js');
          vectorBackend = new HNSWVectorBackend({ dimension: 384 });
        }
      } catch (error) {
        console.log(`⚠️  ${version} backend detection failed:`, error.message);
      }
    }
    bank = new ReasoningBank(db, embedder, vectorBackend);
  }

  return { db, embedder, bank };
}

function generatePattern(index) {
  const taskType = TASK_TYPES[index % TASK_TYPES.length];
  return {
    taskType,
    approach: `Approach ${index}: Use ${taskType} technique with optimization level ${index % 3}`,
    successRate: 0.7 + (Math.random() * 0.3),
    context: `Context for task ${index} with detailed information about the problem and solution`,
    outcome: `Successful outcome with ${Math.floor(Math.random() * 10)} improvements`,
    tags: [taskType, `level-${index % 3}`, `category-${Math.floor(index / 100)}`]
  };
}

async function benchmarkPatternStorage(version, useBackends = false) {
  console.log(`\n📊 Benchmarking ${version} Pattern Storage (backends: ${useBackends})...`);

  const { db, embedder, bank } = await setupDatabase(version, useBackends);
  const startMem = await measureMemory();

  const startTime = performance.now();

  for (let i = 0; i < PATTERNS_COUNT; i++) {
    const pattern = generatePattern(i);
    await bank.storePattern(pattern);

    if ((i + 1) % 100 === 0) {
      process.stdout.write(`\r  Stored ${i + 1}/${PATTERNS_COUNT} patterns...`);
    }
  }

  const endTime = performance.now();
  const endMem = await measureMemory();

  const duration = endTime - startTime;
  const throughput = (PATTERNS_COUNT / duration) * 1000; // patterns/sec

  console.log(`\n  ✅ Stored ${PATTERNS_COUNT} patterns`);
  console.log(`  ⏱️  Duration: ${duration.toFixed(2)}ms`);
  console.log(`  🚀 Throughput: ${throughput.toFixed(2)} patterns/sec`);
  console.log(`  💾 Memory used: ${endMem.heapUsed - startMem.heapUsed}MB`);
  console.log(`  📈 Heap total: ${endMem.heapTotal}MB`);

  db.close();

  return {
    version,
    useBackends,
    operation: 'storage',
    count: PATTERNS_COUNT,
    duration,
    throughput,
    memoryUsed: endMem.heapUsed - startMem.heapUsed,
    memoryTotal: endMem.heapTotal
  };
}

async function benchmarkPatternSearch(version, useBackends = false) {
  console.log(`\n📊 Benchmarking ${version} Pattern Search (backends: ${useBackends})...`);

  const { db, embedder, bank } = await setupDatabase(version, useBackends);

  // Pre-populate with patterns
  console.log(`  Populating with ${PATTERNS_COUNT} patterns...`);
  for (let i = 0; i < PATTERNS_COUNT; i++) {
    const pattern = generatePattern(i);
    await bank.storePattern(pattern);
  }

  const startMem = await measureMemory();
  const searchQueries = [
    'coding optimization strategy',
    'debugging techniques for complex systems',
    'refactoring legacy code',
    'testing automation framework',
    'performance optimization methods'
  ];

  const startTime = performance.now();
  let totalResults = 0;

  for (let i = 0; i < SEARCH_ITERATIONS; i++) {
    const query = searchQueries[i % searchQueries.length];
    const results = await bank.searchPatterns({
      task: query,
      k: 10,
      threshold: 0.5
    });
    totalResults += results.length;

    if ((i + 1) % 20 === 0) {
      process.stdout.write(`\r  Completed ${i + 1}/${SEARCH_ITERATIONS} searches...`);
    }
  }

  const endTime = performance.now();
  const endMem = await measureMemory();

  const duration = endTime - startTime;
  const throughput = (SEARCH_ITERATIONS / duration) * 1000; // searches/sec
  const avgResults = totalResults / SEARCH_ITERATIONS;

  console.log(`\n  ✅ Completed ${SEARCH_ITERATIONS} searches`);
  console.log(`  ⏱️  Duration: ${duration.toFixed(2)}ms`);
  console.log(`  🚀 Throughput: ${throughput.toFixed(2)} searches/sec`);
  console.log(`  📊 Avg results per search: ${avgResults.toFixed(2)}`);
  console.log(`  💾 Memory used: ${endMem.heapUsed - startMem.heapUsed}MB`);

  db.close();

  return {
    version,
    useBackends,
    operation: 'search',
    iterations: SEARCH_ITERATIONS,
    duration,
    throughput,
    avgResults,
    memoryUsed: endMem.heapUsed - startMem.heapUsed
  };
}

async function runBenchmarks() {
  console.log('=========================================');
  console.log('AgentDB ReasoningBank Performance Benchmark');
  console.log('=========================================');
  console.log(`Patterns: ${PATTERNS_COUNT}`);
  console.log(`Search Iterations: ${SEARCH_ITERATIONS}`);
  console.log('');

  const results = [];

  // v1 Storage Benchmark
  results.push(await benchmarkPatternStorage('v1', false));

  // v2 Storage Benchmark (no backends)
  results.push(await benchmarkPatternStorage('v2', false));

  // v2 Storage Benchmark (with backends)
  results.push(await benchmarkPatternStorage('v2', true));

  // v1 Search Benchmark
  results.push(await benchmarkPatternSearch('v1', false));

  // v2 Search Benchmark (no backends)
  results.push(await benchmarkPatternSearch('v2', false));

  // v2 Search Benchmark (with backends)
  results.push(await benchmarkPatternSearch('v2', true));

  // Summary
  console.log('\n\n=========================================');
  console.log('📊 BENCHMARK SUMMARY');
  console.log('=========================================\n');

  // Storage comparison
  const v1Storage = results.find(r => r.version === 'v1' && r.operation === 'storage');
  const v2Storage = results.find(r => r.version === 'v2' && r.operation === 'storage' && !r.useBackends);
  const v2StorageBackends = results.find(r => r.version === 'v2' && r.operation === 'storage' && r.useBackends);

  console.log('Pattern Storage Performance:');
  console.log(`  v1: ${v1Storage.throughput.toFixed(2)} patterns/sec`);
  console.log(`  v2 (no backends): ${v2Storage.throughput.toFixed(2)} patterns/sec (${((v2Storage.throughput / v1Storage.throughput) * 100).toFixed(1)}%)`);
  console.log(`  v2 (with backends): ${v2StorageBackends.throughput.toFixed(2)} patterns/sec (${((v2StorageBackends.throughput / v1Storage.throughput) * 100).toFixed(1)}%)`);
  console.log(`  🚀 Speedup: ${(v2StorageBackends.throughput / v1Storage.throughput).toFixed(2)}x`);

  // Search comparison
  const v1Search = results.find(r => r.version === 'v1' && r.operation === 'search');
  const v2Search = results.find(r => r.version === 'v2' && r.operation === 'search' && !r.useBackends);
  const v2SearchBackends = results.find(r => r.version === 'v2' && r.operation === 'search' && r.useBackends);

  console.log('\nPattern Search Performance:');
  console.log(`  v1: ${v1Search.throughput.toFixed(2)} searches/sec`);
  console.log(`  v2 (no backends): ${v2Search.throughput.toFixed(2)} searches/sec (${((v2Search.throughput / v1Search.throughput) * 100).toFixed(1)}%)`);
  console.log(`  v2 (with backends): ${v2SearchBackends.throughput.toFixed(2)} searches/sec (${((v2SearchBackends.throughput / v1Search.throughput) * 100).toFixed(1)}%)`);
  console.log(`  🚀 Speedup: ${(v2SearchBackends.throughput / v1Search.throughput).toFixed(2)}x`);

  // Memory comparison
  console.log('\nMemory Usage:');
  console.log(`  v1 storage: ${v1Storage.memoryUsed}MB`);
  console.log(`  v2 storage (no backends): ${v2Storage.memoryUsed}MB`);
  console.log(`  v2 storage (with backends): ${v2StorageBackends.memoryUsed}MB`);
  console.log(`  v1 search: ${v1Search.memoryUsed}MB`);
  console.log(`  v2 search (no backends): ${v2Search.memoryUsed}MB`);
  console.log(`  v2 search (with backends): ${v2SearchBackends.memoryUsed}MB`);

  return results;
}

// Run benchmarks
if (require.main === module) {
  runBenchmarks()
    .then(() => {
      console.log('\n✅ Benchmark completed successfully');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ Benchmark failed:', error);
      process.exit(1);
    });
}

module.exports = { runBenchmarks };
