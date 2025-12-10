#!/usr/bin/env node
/**
 * ReasoningBank Performance Benchmark Suite
 * Tests storage, query, and optimization capabilities
 */

const { execSync } = require('child_process');
const fs = require('fs');

// Utility to measure execution time
function benchmark(name, fn) {
  console.log(`\n⏱️  Benchmarking: ${name}`);
  const start = Date.now();
  try {
    const result = fn();
    const duration = Date.now() - start;
    console.log(`✅ Completed in ${duration}ms`);
    return { name, duration, success: true, result };
  } catch (error) {
    const duration = Date.now() - start;
    console.error(`❌ Failed in ${duration}ms: ${error.message}`);
    return { name, duration, success: false, error: error.message };
  }
}

// Helper to run memory commands
function memory(action, ...args) {
  const cmd = `npx claude-flow@alpha memory ${action} ${args.join(' ')}`;
  return execSync(cmd, { encoding: 'utf8' });
}

console.log(`
╔═══════════════════════════════════════════════════════════════╗
║  🚀 ReasoningBank Performance Benchmark Suite                ║
║  Testing: Storage, Query, Optimization                       ║
╚═══════════════════════════════════════════════════════════════╝
`);

const results = [];

// =============================================================================
// PHASE 1: BASELINE PERFORMANCE
// =============================================================================
console.log('\n📊 PHASE 1: Baseline Performance Testing\n');

// Test 1: Single storage operation
results.push(benchmark('Store single entry (100 bytes)', () => {
  return memory('store', 'bench_single', '"Test entry for baseline performance measurement with 100 bytes of data here"', '--namespace benchmark', '--reasoningbank');
}));

// Test 2: Query latency (cold)
results.push(benchmark('Cold query (first query)', () => {
  return memory('query', '"performance"', '--namespace benchmark', '--reasoningbank');
}));

// Test 3: Query latency (warm)
results.push(benchmark('Warm query (cached)', () => {
  return memory('query', '"performance"', '--namespace benchmark', '--reasoningbank');
}));

// Test 4: Status check
results.push(benchmark('Database status check', () => {
  return memory('status', '--reasoningbank');
}));

// =============================================================================
// PHASE 2: BULK OPERATIONS
// =============================================================================
console.log('\n\n📦 PHASE 2: Bulk Operations Testing\n');

// Test 5: Bulk storage (10 entries)
results.push(benchmark('Store 10 entries sequentially', () => {
  for (let i = 0; i < 10; i++) {
    memory('store', `bench_bulk_${i}`, `"Pattern ${i}: Testing bulk storage performance with various data"`, '--namespace benchmark', '--reasoningbank');
  }
}));

// Test 6: Query after bulk load
results.push(benchmark('Query after bulk load (10 entries)', () => {
  return memory('query', '"pattern"', '--namespace benchmark', '--reasoningbank');
}));

// =============================================================================
// PHASE 3: STRESS TESTING
// =============================================================================
console.log('\n\n💪 PHASE 3: Stress Testing\n');

// Test 7: Large entry storage (1KB)
const largeData = 'A'.repeat(1000);
results.push(benchmark('Store large entry (1KB)', () => {
  return memory('store', 'bench_large', `"${largeData}"`, '--namespace benchmark', '--reasoningbank');
}));

// Test 8: Complex query
results.push(benchmark('Complex semantic query', () => {
  return memory('query', '"advanced pattern recognition with multiple keywords and semantic meaning"', '--namespace benchmark', '--reasoningbank');
}));

// Test 9: Multiple rapid queries
results.push(benchmark('5 rapid queries (stress test)', () => {
  for (let i = 0; i < 5; i++) {
    memory('query', `"query ${i}"`, '--namespace benchmark', '--reasoningbank');
  }
}));

// =============================================================================
// PHASE 4: OPTIMIZATION TESTS
// =============================================================================
console.log('\n\n⚡ PHASE 4: Optimization Testing\n');

// Test 10: Export performance
results.push(benchmark('Export database', () => {
  return memory('export', '/tmp/benchmark-export.json');
}));

// Test 11: Import performance
results.push(benchmark('Import database', () => {
  return memory('import', '/tmp/benchmark-export.json');
}));

// Test 12: Namespace query (filtered)
results.push(benchmark('Namespace-filtered query', () => {
  return memory('query', '"pattern"', '--namespace benchmark', '--reasoningbank');
}));

// =============================================================================
// RESULTS ANALYSIS
// =============================================================================
console.log('\n\n📈 BENCHMARK RESULTS ANALYSIS\n');

const successful = results.filter(r => r.success);
const failed = results.filter(r => !r.success);

console.log('═══════════════════════════════════════════════════════════════\n');

// Display results table
console.log('┌─────────────────────────────────────────────────┬──────────────┐');
console.log('│ Operation                                       │ Duration (ms)│');
console.log('├─────────────────────────────────────────────────┼──────────────┤');

results.forEach(r => {
  const status = r.success ? '✅' : '❌';
  const name = r.name.padEnd(45);
  const duration = r.duration.toString().padStart(12);
  console.log(`│ ${status} ${name} │ ${duration} │`);
});

console.log('└─────────────────────────────────────────────────┴──────────────┘\n');

// Statistics
const durations = successful.map(r => r.duration);
const avgDuration = durations.reduce((a, b) => a + b, 0) / durations.length;
const minDuration = Math.min(...durations);
const maxDuration = Math.max(...durations);

console.log('📊 Statistics:\n');
console.log(`   Total Tests: ${results.length}`);
console.log(`   Successful: ${successful.length} ✅`);
console.log(`   Failed: ${failed.length} ❌`);
console.log(`   Success Rate: ${(successful.length / results.length * 100).toFixed(1)}%\n`);

console.log(`   Average Duration: ${avgDuration.toFixed(2)}ms`);
console.log(`   Fastest: ${minDuration}ms`);
console.log(`   Slowest: ${maxDuration}ms`);
console.log(`   Total Time: ${results.reduce((a, r) => a + r.duration, 0)}ms\n`);

// Performance grades
console.log('🏆 Performance Grades:\n');

const gradeQuery = (avg) => {
  if (avg < 100) return '⚡ EXCELLENT (sub-100ms)';
  if (avg < 500) return '✅ GOOD (sub-500ms)';
  if (avg < 1000) return '⚠️  FAIR (sub-1s)';
  return '❌ NEEDS OPTIMIZATION (>1s)';
};

console.log(`   Overall: ${gradeQuery(avgDuration)}`);

// Query performance breakdown
const queryTests = successful.filter(r => r.name.includes('query') || r.name.includes('Query'));
if (queryTests.length > 0) {
  const avgQueryTime = queryTests.reduce((a, r) => a + r.duration, 0) / queryTests.length;
  console.log(`   Query Operations: ${gradeQuery(avgQueryTime)}`);
  console.log(`     Average: ${avgQueryTime.toFixed(2)}ms`);
}

// Storage performance breakdown
const storageTests = successful.filter(r => r.name.includes('Store') || r.name.includes('store'));
if (storageTests.length > 0) {
  const avgStorageTime = storageTests.reduce((a, r) => a + r.duration, 0) / storageTests.length;
  console.log(`   Storage Operations: ${gradeQuery(avgStorageTime)}`);
  console.log(`     Average: ${avgStorageTime.toFixed(2)}ms`);
}

// =============================================================================
// OPTIMIZATION RECOMMENDATIONS
// =============================================================================
console.log('\n\n💡 OPTIMIZATION RECOMMENDATIONS:\n');

const recommendations = [];

if (avgDuration > 1000) {
  recommendations.push('⚠️  HIGH: Average operation time >1s - Consider database optimization');
}

if (maxDuration > 5000) {
  recommendations.push('⚠️  HIGH: Slowest operation >5s - Investigate bottleneck');
}

const storageAvg = storageTests.length > 0
  ? storageTests.reduce((a, r) => a + r.duration, 0) / storageTests.length
  : 0;

if (storageAvg > 2000) {
  recommendations.push('⚠️  MEDIUM: Slow storage operations - Consider batch operations');
}

const queryAvg = queryTests.length > 0
  ? queryTests.reduce((a, r) => a + r.duration, 0) / queryTests.length
  : 0;

if (queryAvg > 1000) {
  recommendations.push('⚠️  MEDIUM: Slow queries - Consider adding indexes or caching');
}

if (failed.length > 0) {
  recommendations.push('⚠️  HIGH: Some tests failed - Review error logs');
}

if (recommendations.length === 0) {
  console.log('   ✅ No optimization needed - Performance is excellent!');
} else {
  recommendations.forEach((rec, i) => {
    console.log(`   ${i + 1}. ${rec}`);
  });
}

// =============================================================================
// OPTIMIZATION STRATEGIES
// =============================================================================
console.log('\n\n🔧 OPTIMIZATION STRATEGIES:\n');

console.log(`
1. Database Optimization:
   ✅ Regular VACUUM to reclaim space
   ✅ Analyze query patterns for indexing
   ✅ Use batch operations for bulk inserts

2. Query Optimization:
   ✅ Use namespace filtering to reduce search space
   ✅ Cache frequently accessed patterns
   ✅ Limit result count with top-k retrieval

3. Storage Optimization:
   ✅ Compress large entries before storage
   ✅ Use batch operations for multiple inserts
   ✅ Export/backup regularly to maintain performance

4. Memory Management:
   ✅ Monitor database growth
   ✅ Archive old namespaces
   ✅ Use TTL for temporary patterns

5. Application-Level:
   ✅ Implement connection pooling
   ✅ Use async operations where possible
   ✅ Batch similar queries together
`);

// =============================================================================
// EXPORT RESULTS
// =============================================================================
const benchmarkResults = {
  timestamp: new Date().toISOString(),
  summary: {
    total: results.length,
    successful: successful.length,
    failed: failed.length,
    successRate: (successful.length / results.length * 100).toFixed(1) + '%',
    avgDuration: avgDuration.toFixed(2) + 'ms',
    minDuration: minDuration + 'ms',
    maxDuration: maxDuration + 'ms'
  },
  results: results,
  recommendations: recommendations
};

fs.writeFileSync('/tmp/benchmark-results.json', JSON.stringify(benchmarkResults, null, 2));

console.log('\n📁 Results exported to: /tmp/benchmark-results.json\n');

console.log(`
╔═══════════════════════════════════════════════════════════════╗
║  ✅ Benchmark Complete!                                       ║
╚═══════════════════════════════════════════════════════════════╝

🎯 Next Steps:

1. Review detailed results: cat /tmp/benchmark-results.json
2. Apply optimization recommendations above
3. Re-run benchmark after optimizations
4. Compare results to measure improvement

📊 Performance Dashboard:
   - Average latency: ${avgDuration.toFixed(2)}ms
   - Success rate: ${(successful.length / results.length * 100).toFixed(1)}%
   - Total operations: ${results.length}
`);
