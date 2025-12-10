/**
 * Test CLI commands with mock backend
 */

const path = require('path');
const Module = require('module');
const fs = require('fs').promises;

// Mock require
const originalRequire = Module.prototype.require;
const mockBackend = require('./test-mock-backend.js');

Module.prototype.require = function(id) {
  if (id === '@ruvector/core' || id === '@ruvector/wasm') {
    return mockBackend;
  }
  return originalRequire.apply(this, arguments);
};

async function testCLI() {
  console.log('🧪 Testing CLI Commands\n');

  try {
    // Test 1: Info command
    console.log('1. Testing info command:');
    const { getBackendInfo } = require('./dist/index.js');
    const info = getBackendInfo();
    console.log(`   ✓ Backend: ${info.type}`);
    console.log(`   ✓ Version: ${info.version}\n`);

    // Test 2: Create test vectors file
    console.log('2. Creating test vectors file:');
    const testVectors = [];
    const { Utils } = require('./dist/index.js');
    for (let i = 0; i < 50; i++) {
      testVectors.push({
        id: `test_${i}`,
        values: Utils.randomVector(128),
        metadata: { index: i, category: i % 3 === 0 ? 'A' : 'B' }
      });
    }
    await fs.writeFile('/tmp/test-vectors.json', JSON.stringify(testVectors, null, 2));
    console.log(`   ✓ Created /tmp/test-vectors.json with ${testVectors.length} vectors\n`);

    // Test 3: Index initialization
    console.log('3. Testing index operations:');
    const { VectorIndex } = require('./dist/index.js');
    const index = new VectorIndex({
      dimension: 128,
      metric: 'cosine',
      indexType: 'hnsw'
    });
    console.log('   ✓ Index created\n');

    // Test 4: Insert vectors
    console.log('4. Testing insertBatch:');
    const startInsert = Date.now();
    await index.insertBatch(testVectors, {
      batchSize: 10,
      progressCallback: (p) => {
        if (p === 1) console.log(`   Progress: 100%`);
      }
    });
    const insertTime = Date.now() - startInsert;
    console.log(`   ✓ Inserted ${testVectors.length} vectors in ${insertTime}ms\n`);

    // Test 5: Search
    console.log('5. Testing search:');
    const query = Utils.randomVector(128);
    const startSearch = Date.now();
    const results = await index.search(query, { k: 5 });
    const searchTime = Date.now() - startSearch;
    console.log(`   ✓ Found ${results.length} results in ${searchTime}ms`);
    results.slice(0, 3).forEach((r, i) => {
      console.log(`      ${i + 1}. ${r.id} (score: ${r.score.toFixed(4)})`);
    });
    console.log();

    // Test 6: Stats
    console.log('6. Testing stats:');
    const stats = await index.stats();
    console.log(`   ✓ Vectors: ${stats.vectorCount}`);
    console.log(`   ✓ Dimension: ${stats.dimension}`);
    console.log(`   ✓ Type: ${stats.indexType}`);
    console.log(`   ✓ Memory: ${(stats.memoryUsage / 1024).toFixed(2)} KB\n`);

    // Test 7: Save/Load
    console.log('7. Testing save/load:');
    await index.save('/tmp/test-index.bin');
    console.log('   ✓ Saved index');
    const loaded = await VectorIndex.load('/tmp/test-index.bin');
    console.log('   ✓ Loaded index\n');

    // Test 8: Performance
    console.log('8. Performance summary:');
    const insertThroughput = testVectors.length / (insertTime / 1000);
    const searchLatency = searchTime;
    console.log(`   Insert throughput: ${insertThroughput.toFixed(0)} vectors/sec`);
    console.log(`   Search latency: ${searchLatency.toFixed(2)}ms`);
    console.log();

    console.log('✅ All CLI tests passed!');
    return true;

  } catch (error) {
    console.error('❌ CLI test failed:', error.message);
    console.error(error.stack);
    return false;
  }
}

testCLI().then(success => {
  process.exit(success ? 0 : 1);
});
