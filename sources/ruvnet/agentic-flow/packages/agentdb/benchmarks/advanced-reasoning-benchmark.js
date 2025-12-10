/**
 * Advanced ReasoningBank Capabilities Benchmark
 * Tests pattern learning, similarity detection, and reasoning optimization
 */

const { performance } = require('perf_hooks');
const { createDatabase } = require('../dist/db-fallback.js');
const { ReasoningBank } = require('../dist/controllers/ReasoningBank.js');
const { EmbeddingService } = require('../dist/controllers/EmbeddingService.js');
const fs = require('fs');

// Advanced benchmark configuration
const CONFIG = {
  patterns: {
    warmup: 100,
    small: 500,
    medium: 2000,
    large: 5000
  },
  similarity: {
    queries: 200,
    threshold: 0.7
  },
  learning: {
    episodes: 1000,
    feedback_cycles: 10
  }
};

async function measureMemory() {
  if (global.gc) global.gc();
  const usage = process.memoryUsage();
  return {
    heapUsed: Math.round(usage.heapUsed / 1024 / 1024),
    heapTotal: Math.round(usage.heapTotal / 1024 / 1024),
    external: Math.round(usage.external / 1024 / 1024),
    rss: Math.round(usage.rss / 1024 / 1024)
  };
}

async function setupReasoningBank(useBackends = false) {
  const db = await createDatabase(':memory:');

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

  let vectorBackend = undefined;
  if (useBackends) {
    try {
      const { detectBackend } = require('../dist/backends/detector.js');
      const detection = await detectBackend();
      if (detection.backend === 'hnswlib') {
        const { HNSWVectorBackend } = require('../dist/backends/hnswlib-backend.js');
        vectorBackend = new HNSWVectorBackend({ dimension: 384, maxElements: 10000 });
      }
    } catch (error) {
      // Backends not available, continue with fallback
    }
  }

  const bank = new ReasoningBank(db, embedder, vectorBackend);

  return { db, embedder, bank };
}

// Generate realistic coding patterns
function generateCodingPattern(index, category) {
  const categories = {
    debugging: {
      tasks: [
        'Debug memory leak in React component',
        'Fix race condition in async handler',
        'Resolve null pointer exception',
        'Debug infinite loop in recursion',
        'Fix closure variable capture issue'
      ],
      approaches: [
        'Use memory profiler and heap snapshots',
        'Add mutex locks and synchronization',
        'Add null checks and validation',
        'Add base case and termination condition',
        'Use debugger to inspect scope chain'
      ]
    },
    optimization: {
      tasks: [
        'Optimize database query performance',
        'Reduce bundle size in production',
        'Improve API response time',
        'Optimize image loading performance',
        'Speed up data processing pipeline'
      ],
      approaches: [
        'Add indexes and optimize query plan',
        'Enable tree-shaking and code splitting',
        'Add caching and pagination',
        'Implement lazy loading and WebP format',
        'Use parallel processing and batching'
      ]
    },
    refactoring: {
      tasks: [
        'Extract reusable component logic',
        'Simplify complex conditional logic',
        'Remove code duplication',
        'Improve function naming clarity',
        'Restructure module dependencies'
      ],
      approaches: [
        'Create custom hook or utility function',
        'Use strategy pattern or lookup table',
        'Create shared utility module',
        'Follow domain-driven naming conventions',
        'Apply dependency inversion principle'
      ]
    }
  };

  const cat = categories[category];
  const taskIdx = index % cat.tasks.length;

  return {
    taskType: category,
    approach: cat.approaches[taskIdx],
    context: cat.tasks[taskIdx],
    successRate: 0.7 + (Math.random() * 0.25),
    outcome: `Successfully resolved with ${Math.floor(Math.random() * 5) + 1} iterations`,
    tags: [category, `complexity-${Math.floor(index / 100) % 3}`, `priority-${index % 3}`],
    metadata: {
      iteration: index,
      timestamp: Date.now() - (index * 1000),
      toolsUsed: ['debugger', 'profiler', 'linter'][index % 3]
    }
  };
}

// Benchmark 1: Pattern Storage Scalability
async function benchmarkPatternScalability() {
  console.log('\n' + '='.repeat(60));
  console.log('📊 BENCHMARK 1: Pattern Storage Scalability');
  console.log('='.repeat(60));

  const results = [];
  const categories = ['debugging', 'optimization', 'refactoring'];

  for (const size of ['small', 'medium', 'large']) {
    const count = CONFIG.patterns[size];
    console.log(`\n🔍 Testing ${size} dataset: ${count} patterns`);

    const { db, bank } = await setupReasoningBank(true);
    const startMem = await measureMemory();
    const startTime = performance.now();

    // Store patterns
    for (let i = 0; i < count; i++) {
      const category = categories[i % categories.length];
      const pattern = generateCodingPattern(i, category);
      await bank.storePattern(pattern);

      if ((i + 1) % 500 === 0) {
        process.stdout.write(`\r  Progress: ${i + 1}/${count}...`);
      }
    }

    const endTime = performance.now();
    const endMem = await measureMemory();

    const result = {
      size,
      count,
      duration: endTime - startTime,
      throughput: (count / (endTime - startTime)) * 1000,
      memoryUsed: endMem.heapUsed - startMem.heapUsed,
      avgPerPattern: (endTime - startTime) / count
    };

    console.log(`\n  ✅ Duration: ${result.duration.toFixed(2)}ms`);
    console.log(`  🚀 Throughput: ${result.throughput.toFixed(2)} patterns/sec`);
    console.log(`  💾 Memory: ${result.memoryUsed}MB`);
    console.log(`  ⚡ Avg per pattern: ${result.avgPerPattern.toFixed(2)}ms`);

    results.push(result);
    db.close();
  }

  return results;
}

// Benchmark 2: Pattern Similarity Detection
async function benchmarkSimilarityDetection() {
  console.log('\n' + '='.repeat(60));
  console.log('📊 BENCHMARK 2: Pattern Similarity Detection');
  console.log('='.repeat(60));

  const { db, bank } = await setupReasoningBank(true);

  // Populate with patterns
  console.log('\n🔧 Populating database with 2000 patterns...');
  const categories = ['debugging', 'optimization', 'refactoring'];
  for (let i = 0; i < 2000; i++) {
    const category = categories[i % categories.length];
    const pattern = generateCodingPattern(i, category);
    await bank.storePattern(pattern);
  }

  // Test similarity detection at different thresholds
  const queries = [
    { task: 'Debug memory leak in application', category: 'debugging' },
    { task: 'Optimize slow database queries', category: 'optimization' },
    { task: 'Refactor complex conditional code', category: 'refactoring' },
    { task: 'Fix race condition bug', category: 'debugging' },
    { task: 'Reduce application bundle size', category: 'optimization' }
  ];

  const thresholds = [0.5, 0.6, 0.7, 0.8, 0.9];
  const results = [];

  for (const threshold of thresholds) {
    console.log(`\n🎯 Testing threshold: ${threshold}`);

    let totalMatches = 0;
    let totalTime = 0;

    for (const query of queries) {
      const startTime = performance.now();
      const matches = await bank.searchPatterns({
        task: query.task,
        k: 20,
        threshold,
        filters: { taskType: query.category }
      });
      const endTime = performance.now();

      totalMatches += matches.length;
      totalTime += (endTime - startTime);
    }

    const result = {
      threshold,
      avgMatches: totalMatches / queries.length,
      avgSearchTime: totalTime / queries.length,
      totalSearchTime: totalTime
    };

    console.log(`  📊 Avg matches: ${result.avgMatches.toFixed(2)}`);
    console.log(`  ⏱️  Avg search time: ${result.avgSearchTime.toFixed(2)}ms`);

    results.push(result);
  }

  db.close();
  return results;
}

// Benchmark 3: Pattern Learning & Adaptation
async function benchmarkPatternLearning() {
  console.log('\n' + '='.repeat(60));
  console.log('📊 BENCHMARK 3: Pattern Learning & Adaptation');
  console.log('='.repeat(60));

  const { db, bank } = await setupReasoningBank(true);

  console.log('\n🧠 Simulating iterative learning cycle...');

  const learningResults = [];
  let currentSuccessRate = 0.5;

  for (let cycle = 0; cycle < CONFIG.learning.feedback_cycles; cycle++) {
    console.log(`\n📈 Learning Cycle ${cycle + 1}/${CONFIG.learning.feedback_cycles}`);

    const cycleStart = performance.now();

    // Store patterns with increasing success rate (simulating learning)
    currentSuccessRate += 0.04;
    const patternsThisCycle = 100;

    for (let i = 0; i < patternsThisCycle; i++) {
      const pattern = {
        taskType: 'learning',
        approach: `Approach iteration ${cycle * patternsThisCycle + i}`,
        context: `Learning task ${i}`,
        successRate: Math.min(currentSuccessRate + (Math.random() * 0.1), 1.0),
        outcome: `Cycle ${cycle} result`,
        tags: [`cycle-${cycle}`, `success-${currentSuccessRate.toFixed(2)}`]
      };
      await bank.storePattern(pattern);
    }

    // Test retrieval of best patterns
    const bestPatterns = await bank.searchPatterns({
      task: 'learning task',
      k: 10,
      threshold: 0.7
    });

    const avgSuccessRate = bestPatterns.reduce((sum, p) => sum + p.successRate, 0) / bestPatterns.length;
    const cycleEnd = performance.now();

    const result = {
      cycle: cycle + 1,
      currentSuccessRate,
      avgRetrievedSuccessRate: avgSuccessRate,
      patternsStored: patternsThisCycle,
      duration: cycleEnd - cycleStart,
      improvement: avgSuccessRate - currentSuccessRate
    };

    console.log(`  📊 Current success rate: ${currentSuccessRate.toFixed(3)}`);
    console.log(`  🎯 Avg retrieved success rate: ${avgSuccessRate.toFixed(3)}`);
    console.log(`  ⏱️  Cycle duration: ${result.duration.toFixed(2)}ms`);

    learningResults.push(result);
  }

  db.close();
  return learningResults;
}

// Benchmark 4: Query Optimization Analysis
async function benchmarkQueryOptimization() {
  console.log('\n' + '='.repeat(60));
  console.log('📊 BENCHMARK 4: Query Optimization Analysis');
  console.log('='.repeat(60));

  const { db, bank } = await setupReasoningBank(true);

  // Populate database
  console.log('\n🔧 Populating with 3000 patterns...');
  const categories = ['debugging', 'optimization', 'refactoring'];
  for (let i = 0; i < 3000; i++) {
    const category = categories[i % categories.length];
    const pattern = generateCodingPattern(i, category);
    await bank.storePattern(pattern);
  }

  const results = [];

  // Test 1: Simple query
  console.log('\n🔍 Test 1: Simple pattern search');
  const simple1 = performance.now();
  const simpleResults = await bank.searchPatterns({
    task: 'debug memory issue',
    k: 10
  });
  const simple2 = performance.now();

  results.push({
    type: 'simple',
    duration: simple2 - simple1,
    resultsCount: simpleResults.length
  });
  console.log(`  ⏱️  Duration: ${(simple2 - simple1).toFixed(2)}ms`);
  console.log(`  📊 Results: ${simpleResults.length}`);

  // Test 2: Filtered query
  console.log('\n🔍 Test 2: Filtered by category');
  const filter1 = performance.now();
  const filterResults = await bank.searchPatterns({
    task: 'debug memory issue',
    k: 10,
    filters: { taskType: 'debugging' }
  });
  const filter2 = performance.now();

  results.push({
    type: 'filtered',
    duration: filter2 - filter1,
    resultsCount: filterResults.length
  });
  console.log(`  ⏱️  Duration: ${(filter2 - filter1).toFixed(2)}ms`);
  console.log(`  📊 Results: ${filterResults.length}`);

  // Test 3: High-threshold query
  console.log('\n🔍 Test 3: High similarity threshold');
  const threshold1 = performance.now();
  const thresholdResults = await bank.searchPatterns({
    task: 'debug memory issue',
    k: 10,
    threshold: 0.9
  });
  const threshold2 = performance.now();

  results.push({
    type: 'high_threshold',
    duration: threshold2 - threshold1,
    resultsCount: thresholdResults.length
  });
  console.log(`  ⏱️  Duration: ${(threshold2 - threshold1).toFixed(2)}ms`);
  console.log(`  📊 Results: ${thresholdResults.length}`);

  // Test 4: Large result set
  console.log('\n🔍 Test 4: Large result set (k=100)');
  const large1 = performance.now();
  const largeResults = await bank.searchPatterns({
    task: 'debug memory issue',
    k: 100,
    threshold: 0.5
  });
  const large2 = performance.now();

  results.push({
    type: 'large_k',
    duration: large2 - large1,
    resultsCount: largeResults.length
  });
  console.log(`  ⏱️  Duration: ${(large2 - large1).toFixed(2)}ms`);
  console.log(`  📊 Results: ${largeResults.length}`);

  db.close();
  return results;
}

async function runAdvancedBenchmarks() {
  console.log('╔' + '═'.repeat(58) + '╗');
  console.log('║' + ' '.repeat(10) + 'AgentDB ReasoningBank Advanced Benchmark' + ' '.repeat(7) + '║');
  console.log('╚' + '═'.repeat(58) + '╝');
  console.log('');
  console.log('Testing: Pattern Learning, Similarity Detection, Query Optimization');
  console.log('');

  const allResults = {
    scalability: await benchmarkPatternScalability(),
    similarity: await benchmarkSimilarityDetection(),
    learning: await benchmarkPatternLearning(),
    queryOptimization: await benchmarkQueryOptimization()
  };

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 ADVANCED BENCHMARK SUMMARY');
  console.log('='.repeat(60));

  console.log('\n🔧 Scalability Results:');
  allResults.scalability.forEach(r => {
    console.log(`  ${r.size.padEnd(10)}: ${r.throughput.toFixed(2)} patterns/sec, ${r.memoryUsed}MB`);
  });

  console.log('\n🎯 Similarity Detection Optimal Threshold:');
  const bestThreshold = allResults.similarity.reduce((best, curr) =>
    curr.avgMatches > 5 && curr.avgSearchTime < best.avgSearchTime ? curr : best
  );
  console.log(`  Threshold: ${bestThreshold.threshold} (${bestThreshold.avgMatches.toFixed(1)} matches, ${bestThreshold.avgSearchTime.toFixed(2)}ms)`);

  console.log('\n🧠 Learning Improvement:');
  const firstCycle = allResults.learning[0];
  const lastCycle = allResults.learning[allResults.learning.length - 1];
  console.log(`  Initial success rate: ${firstCycle.currentSuccessRate.toFixed(3)}`);
  console.log(`  Final success rate: ${lastCycle.currentSuccessRate.toFixed(3)}`);
  console.log(`  Improvement: ${((lastCycle.currentSuccessRate - firstCycle.currentSuccessRate) * 100).toFixed(1)}%`);

  console.log('\n⚡ Query Performance:');
  allResults.queryOptimization.forEach(r => {
    console.log(`  ${r.type.padEnd(18)}: ${r.duration.toFixed(2)}ms (${r.resultsCount} results)`);
  });

  return allResults;
}

if (require.main === module) {
  runAdvancedBenchmarks()
    .then(() => {
      console.log('\n✅ Advanced benchmarks completed successfully');
      process.exit(0);
    })
    .catch(error => {
      console.error('\n❌ Benchmark failed:', error);
      process.exit(1);
    });
}

module.exports = { runAdvancedBenchmarks };
