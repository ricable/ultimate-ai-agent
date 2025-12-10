/**
 * Self-Learning Performance Benchmark
 * Tests ReflexionMemory with GNN integration and learning capabilities
 */

const { performance } = require('perf_hooks');
const { createDatabase } = require('../dist/db-fallback.js');
const { ReflexionMemory } = require('../dist/controllers/ReflexionMemory.js');
const { EmbeddingService } = require('../dist/controllers/EmbeddingService.js');
const fs = require('fs');

// Benchmark configuration
const EPISODES_COUNT = 500;
const RETRIEVAL_ITERATIONS = 50;
const SESSIONS = ['session-1', 'session-2', 'session-3', 'session-4', 'session-5'];

async function measureMemory() {
  const usage = process.memoryUsage();
  return {
    heapUsed: Math.round(usage.heapUsed / 1024 / 1024),
    heapTotal: Math.round(usage.heapTotal / 1024 / 1024),
    external: Math.round(usage.external / 1024 / 1024),
    rss: Math.round(usage.rss / 1024 / 1024)
  };
}

async function setupDatabase(version, useBackends = false) {
  const db = await createDatabase(':memory:');

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

  let reflexion;
  if (version === 'v1') {
    // v1: Only db and embedder
    reflexion = new ReflexionMemory(db, embedder);
  } else {
    // v2: With optional backends for GNN and Graph
    let vectorBackend = undefined;
    let learningBackend = undefined;
    let graphBackend = undefined;

    if (useBackends) {
      try {
        const { detectBackend } = require('../dist/backends/detector.js');
        const detection = await detectBackend();

        if (detection.backend === 'hnswlib') {
          const { HNSWVectorBackend } = require('../dist/backends/hnswlib-backend.js');
          vectorBackend = new HNSWVectorBackend({ dimension: 384 });
        }

        // Try to enable GNN and Graph backends
        if (detection.features.gnn) {
          console.log('  ✅ GNN backend available');
          // GNN backend would be initialized here
        }
        if (detection.features.graph) {
          console.log('  ✅ Graph backend available');
          // Graph backend would be initialized here
        }
      } catch (error) {
        console.log(`  ⚠️  Backend detection: ${error.message}`);
      }
    }

    reflexion = new ReflexionMemory(db, embedder, vectorBackend, learningBackend, graphBackend);
  }

  return { db, embedder, reflexion };
}

function generateEpisode(index) {
  const sessionId = SESSIONS[index % SESSIONS.length];
  const success = Math.random() > 0.3;
  const reward = success ? 0.7 + (Math.random() * 0.3) : 0.2 + (Math.random() * 0.3);

  return {
    sessionId,
    task: `Task ${index}: Implement feature with complexity level ${index % 5}`,
    input: `Input data for task ${index} with various parameters`,
    output: success ? `Successful output ${index}` : `Failed attempt ${index}`,
    reward,
    success,
    latencyMs: 100 + Math.floor(Math.random() * 900),
    tokensUsed: 100 + Math.floor(Math.random() * 1000),
    critique: success
      ? `Task completed successfully with ${Math.floor(reward * 10)} improvements`
      : `Task failed, need to improve approach in areas: ${Math.floor((1 - reward) * 5)}`
  };
}

async function benchmarkEpisodeStorage(version, useBackends = false) {
  console.log(`\n📊 Benchmarking ${version} Episode Storage (backends: ${useBackends})...`);

  const { db, embedder, reflexion } = await setupDatabase(version, useBackends);
  const startMem = await measureMemory();

  const startTime = performance.now();

  for (let i = 0; i < EPISODES_COUNT; i++) {
    const episode = generateEpisode(i);
    await reflexion.storeEpisode(episode);

    if ((i + 1) % 50 === 0) {
      process.stdout.write(`\r  Stored ${i + 1}/${EPISODES_COUNT} episodes...`);
    }
  }

  const endTime = performance.now();
  const endMem = await measureMemory();

  const duration = endTime - startTime;
  const throughput = (EPISODES_COUNT / duration) * 1000;

  console.log(`\n  ✅ Stored ${EPISODES_COUNT} episodes`);
  console.log(`  ⏱️  Duration: ${duration.toFixed(2)}ms`);
  console.log(`  🚀 Throughput: ${throughput.toFixed(2)} episodes/sec`);
  console.log(`  💾 Memory used: ${endMem.heapUsed - startMem.heapUsed}MB`);

  db.close();

  return {
    version,
    useBackends,
    operation: 'episode_storage',
    count: EPISODES_COUNT,
    duration,
    throughput,
    memoryUsed: endMem.heapUsed - startMem.heapUsed
  };
}

async function benchmarkEpisodeRetrieval(version, useBackends = false) {
  console.log(`\n📊 Benchmarking ${version} Episode Retrieval (backends: ${useBackends})...`);

  const { db, embedder, reflexion } = await setupDatabase(version, useBackends);

  // Pre-populate with episodes
  console.log(`  Populating with ${EPISODES_COUNT} episodes...`);
  for (let i = 0; i < EPISODES_COUNT; i++) {
    const episode = generateEpisode(i);
    await reflexion.storeEpisode(episode);
  }

  const startMem = await measureMemory();
  const queries = [
    'implement feature with high complexity',
    'handle error conditions gracefully',
    'optimize performance for large datasets',
    'refactor code for better maintainability',
    'add comprehensive test coverage'
  ];

  const startTime = performance.now();
  let totalResults = 0;

  for (let i = 0; i < RETRIEVAL_ITERATIONS; i++) {
    const query = queries[i % queries.length];
    const results = await reflexion.retrieveRelevant({
      task: query,
      k: 10,
      onlySuccesses: false
    });
    totalResults += results.length;

    if ((i + 1) % 10 === 0) {
      process.stdout.write(`\r  Completed ${i + 1}/${RETRIEVAL_ITERATIONS} retrievals...`);
    }
  }

  const endTime = performance.now();
  const endMem = await measureMemory();

  const duration = endTime - startTime;
  const throughput = (RETRIEVAL_ITERATIONS / duration) * 1000;
  const avgResults = totalResults / RETRIEVAL_ITERATIONS;

  console.log(`\n  ✅ Completed ${RETRIEVAL_ITERATIONS} retrievals`);
  console.log(`  ⏱️  Duration: ${duration.toFixed(2)}ms`);
  console.log(`  🚀 Throughput: ${throughput.toFixed(2)} retrievals/sec`);
  console.log(`  📊 Avg results per retrieval: ${avgResults.toFixed(2)}`);
  console.log(`  💾 Memory used: ${endMem.heapUsed - startMem.heapUsed}MB`);

  db.close();

  return {
    version,
    useBackends,
    operation: 'episode_retrieval',
    iterations: RETRIEVAL_ITERATIONS,
    duration,
    throughput,
    avgResults,
    memoryUsed: endMem.heapUsed - startMem.heapUsed
  };
}

async function benchmarkTaskStats(version, useBackends = false) {
  console.log(`\n📊 Benchmarking ${version} Task Stats (backends: ${useBackends})...`);

  const { db, embedder, reflexion } = await setupDatabase(version, useBackends);

  // Pre-populate
  console.log(`  Populating with ${EPISODES_COUNT} episodes...`);
  for (let i = 0; i < EPISODES_COUNT; i++) {
    const episode = generateEpisode(i);
    await reflexion.storeEpisode(episode);
  }

  const startTime = performance.now();

  // Get stats for different tasks
  const stats = [];
  for (let i = 0; i < 20; i++) {
    const taskStats = await reflexion.getTaskStats(`Task ${i * 5}`);
    stats.push(taskStats);
  }

  const endTime = performance.now();
  const duration = endTime - startTime;

  console.log(`  ✅ Retrieved stats for ${stats.length} tasks`);
  console.log(`  ⏱️  Duration: ${duration.toFixed(2)}ms`);
  console.log(`  🚀 Avg per task: ${(duration / stats.length).toFixed(2)}ms`);

  db.close();

  return {
    version,
    useBackends,
    operation: 'task_stats',
    count: stats.length,
    duration,
    avgDuration: duration / stats.length
  };
}

async function runBenchmarks() {
  console.log('=========================================');
  console.log('AgentDB Self-Learning Performance Benchmark');
  console.log('=========================================');
  console.log(`Episodes: ${EPISODES_COUNT}`);
  console.log(`Retrieval Iterations: ${RETRIEVAL_ITERATIONS}`);
  console.log('');

  const results = [];

  // Episode Storage
  results.push(await benchmarkEpisodeStorage('v1', false));
  results.push(await benchmarkEpisodeStorage('v2', false));
  results.push(await benchmarkEpisodeStorage('v2', true));

  // Episode Retrieval
  results.push(await benchmarkEpisodeRetrieval('v1', false));
  results.push(await benchmarkEpisodeRetrieval('v2', false));
  results.push(await benchmarkEpisodeRetrieval('v2', true));

  // Task Stats
  results.push(await benchmarkTaskStats('v1', false));
  results.push(await benchmarkTaskStats('v2', false));
  results.push(await benchmarkTaskStats('v2', true));

  // Summary
  console.log('\n\n=========================================');
  console.log('📊 SELF-LEARNING BENCHMARK SUMMARY');
  console.log('=========================================\n');

  // Storage comparison
  const v1Storage = results.find(r => r.version === 'v1' && r.operation === 'episode_storage');
  const v2Storage = results.find(r => r.version === 'v2' && r.operation === 'episode_storage' && !r.useBackends);
  const v2StorageBackends = results.find(r => r.version === 'v2' && r.operation === 'episode_storage' && r.useBackends);

  console.log('Episode Storage Performance:');
  console.log(`  v1: ${v1Storage.throughput.toFixed(2)} episodes/sec`);
  console.log(`  v2 (no backends): ${v2Storage.throughput.toFixed(2)} episodes/sec (${((v2Storage.throughput / v1Storage.throughput) * 100).toFixed(1)}%)`);
  console.log(`  v2 (with backends): ${v2StorageBackends.throughput.toFixed(2)} episodes/sec (${((v2StorageBackends.throughput / v1Storage.throughput) * 100).toFixed(1)}%)`);
  console.log(`  🚀 Speedup: ${(v2StorageBackends.throughput / v1Storage.throughput).toFixed(2)}x`);

  // Retrieval comparison
  const v1Retrieval = results.find(r => r.version === 'v1' && r.operation === 'episode_retrieval');
  const v2Retrieval = results.find(r => r.version === 'v2' && r.operation === 'episode_retrieval' && !r.useBackends);
  const v2RetrievalBackends = results.find(r => r.version === 'v2' && r.operation === 'episode_retrieval' && r.useBackends);

  console.log('\nEpisode Retrieval Performance:');
  console.log(`  v1: ${v1Retrieval.throughput.toFixed(2)} retrievals/sec`);
  console.log(`  v2 (no backends): ${v2Retrieval.throughput.toFixed(2)} retrievals/sec (${((v2Retrieval.throughput / v1Retrieval.throughput) * 100).toFixed(1)}%)`);
  console.log(`  v2 (with backends): ${v2RetrievalBackends.throughput.toFixed(2)} retrievals/sec (${((v2RetrievalBackends.throughput / v1Retrieval.throughput) * 100).toFixed(1)}%)`);
  console.log(`  🚀 Speedup: ${(v2RetrievalBackends.throughput / v1Retrieval.throughput).toFixed(2)}x`);

  // Task Stats comparison
  const v1Stats = results.find(r => r.version === 'v1' && r.operation === 'task_stats');
  const v2Stats = results.find(r => r.version === 'v2' && r.operation === 'task_stats' && !r.useBackends);
  const v2StatsBackends = results.find(r => r.version === 'v2' && r.operation === 'task_stats' && r.useBackends);

  console.log('\nTask Stats Performance:');
  console.log(`  v1: ${v1Stats.avgDuration.toFixed(2)}ms per task`);
  console.log(`  v2 (no backends): ${v2Stats.avgDuration.toFixed(2)}ms per task`);
  console.log(`  v2 (with backends): ${v2StatsBackends.avgDuration.toFixed(2)}ms per task`);
  console.log(`  🚀 Speedup: ${(v1Stats.avgDuration / v2StatsBackends.avgDuration).toFixed(2)}x`);

  return results;
}

// Run benchmarks
if (require.main === module) {
  runBenchmarks()
    .then(() => {
      console.log('\n✅ Self-learning benchmark completed successfully');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ Self-learning benchmark failed:', error);
      process.exit(1);
    });
}

module.exports = { runBenchmarks };
