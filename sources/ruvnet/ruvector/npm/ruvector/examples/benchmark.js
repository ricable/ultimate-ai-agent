/**
 * Performance benchmark example
 */

const { VectorIndex, Utils, getBackendInfo } = require('ruvector');

function formatNumber(num) {
  return num.toLocaleString();
}

function formatDuration(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(2)}ms`;
}

async function runBenchmark(dimension, numVectors, numQueries) {
  console.log(`\n📊 Benchmark: dim=${dimension}, vectors=${formatNumber(numVectors)}, queries=${numQueries}`);
  console.log('─'.repeat(70));

  // Create index
  const index = new VectorIndex({
    dimension,
    metric: 'cosine',
    indexType: 'hnsw',
    hnswConfig: { m: 16, efConstruction: 200 }
  });

  // Generate vectors
  console.log('Generating vectors...');
  const vectors = Array.from({ length: numVectors }, (_, i) => ({
    id: `vec_${i}`,
    values: Utils.randomVector(dimension),
    metadata: { index: i }
  }));

  // Benchmark insertions
  console.log('Benchmarking insertions...');
  const insertStart = performance.now();
  await index.insertBatch(vectors, { batchSize: 1000 });
  const insertDuration = performance.now() - insertStart;
  const insertThroughput = numVectors / (insertDuration / 1000);

  console.log(`  ✓ Inserted ${formatNumber(numVectors)} vectors in ${formatDuration(insertDuration)}`);
  console.log(`  ✓ Throughput: ${formatNumber(Math.round(insertThroughput))} vectors/sec`);

  // Benchmark searches
  console.log('\nBenchmarking searches...');
  const queries = Array.from({ length: numQueries }, () => Utils.randomVector(dimension));

  const searchStart = performance.now();
  const results = await Promise.all(
    queries.map(q => index.search(q, { k: 10 }))
  );
  const searchDuration = performance.now() - searchStart;
  const searchThroughput = numQueries / (searchDuration / 1000);

  console.log(`  ✓ Executed ${numQueries} searches in ${formatDuration(searchDuration)}`);
  console.log(`  ✓ Throughput: ${formatNumber(Math.round(searchThroughput))} queries/sec`);
  console.log(`  ✓ Avg latency: ${formatDuration(searchDuration / numQueries)}`);

  // Check recall (verify we get results)
  const avgResults = results.reduce((sum, r) => sum + r.length, 0) / results.length;
  console.log(`  ✓ Avg results per query: ${avgResults.toFixed(2)}`);

  // Get memory stats
  const stats = await index.stats();
  if (stats.memoryUsage) {
    const mb = (stats.memoryUsage / 1024 / 1024).toFixed(2);
    console.log(`  ✓ Memory usage: ${mb} MB`);
  }

  return {
    dimension,
    numVectors,
    insertDuration,
    insertThroughput,
    searchDuration,
    searchThroughput,
    avgLatency: searchDuration / numQueries
  };
}

async function main() {
  console.log('⚡ rUvector Performance Benchmark\n');

  const info = getBackendInfo();
  console.log(`Backend: ${info.type}`);
  console.log(`Features: ${info.features.join(', ')}`);

  // Run benchmarks with different configurations
  const configs = [
    { dimension: 128, vectors: 1000, queries: 100 },
    { dimension: 384, vectors: 5000, queries: 100 },
    { dimension: 768, vectors: 10000, queries: 100 },
    { dimension: 1536, vectors: 5000, queries: 100 }
  ];

  const results = [];
  for (const config of configs) {
    const result = await runBenchmark(config.dimension, config.vectors, config.queries);
    results.push(result);
  }

  // Summary
  console.log('\n' + '═'.repeat(70));
  console.log('Summary');
  console.log('═'.repeat(70));
  console.log('\nInsert Throughput:');
  results.forEach(r => {
    console.log(`  dim=${r.dimension}: ${formatNumber(Math.round(r.insertThroughput))} vectors/sec`);
  });

  console.log('\nSearch Throughput:');
  results.forEach(r => {
    console.log(`  dim=${r.dimension}: ${formatNumber(Math.round(r.searchThroughput))} queries/sec`);
  });

  console.log('\nSearch Latency:');
  results.forEach(r => {
    console.log(`  dim=${r.dimension}: ${formatDuration(r.avgLatency)}`);
  });
}

main().catch(console.error);
