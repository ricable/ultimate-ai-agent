/**
 * Basic test of ruvector package with mock backend
 */

const path = require('path');
const Module = require('module');

// Mock require to return our mock backend
const originalRequire = Module.prototype.require;
const mockBackend = require('./test-mock-backend.js');

Module.prototype.require = function(id) {
  if (id === '@ruvector/core' || id === '@ruvector/wasm') {
    return mockBackend;
  }
  return originalRequire.apply(this, arguments);
};

const { VectorIndex, Utils, getBackendInfo, isNativeAvailable } = require('./dist/index.js');

async function testBasicOperations() {
  console.log('🧪 Testing Basic Operations\n');

  try {
    // Test backend info
    console.log('1. Backend Info:');
    const info = getBackendInfo();
    console.log(`   Type: ${info.type}`);
    console.log(`   Version: ${info.version}`);
    console.log(`   Native Available: ${isNativeAvailable()}`);
    console.log('   ✓ Backend info works\n');

    // Test index creation
    console.log('2. Creating Index:');
    const index = new VectorIndex({
      dimension: 128,
      metric: 'cosine',
      indexType: 'hnsw'
    });
    console.log('   ✓ Index created\n');

    // Test single insert
    console.log('3. Single Insert:');
    await index.insert({
      id: 'vec1',
      values: Utils.randomVector(128),
      metadata: { test: true }
    });
    console.log('   ✓ Vector inserted\n');

    // Test batch insert
    console.log('4. Batch Insert:');
    const vectors = [];
    for (let i = 0; i < 100; i++) {
      vectors.push({
        id: `vec${i + 2}`,
        values: Utils.randomVector(128),
        metadata: { index: i }
      });
    }
    await index.insertBatch(vectors, { batchSize: 10 });
    console.log('   ✓ Batch inserted\n');

    // Test stats
    console.log('5. Stats:');
    const stats = await index.stats();
    console.log(`   Vectors: ${stats.vectorCount}`);
    console.log(`   Dimension: ${stats.dimension}`);
    console.log(`   Type: ${stats.indexType}`);
    console.log('   ✓ Stats retrieved\n');

    // Test search
    console.log('6. Search:');
    const query = Utils.randomVector(128);
    const results = await index.search(query, { k: 5 });
    console.log(`   Found ${results.length} results`);
    results.slice(0, 3).forEach((r, i) => {
      console.log(`   ${i + 1}. ${r.id} (score: ${r.score.toFixed(4)})`);
    });
    console.log('   ✓ Search works\n');

    // Test get
    console.log('7. Get by ID:');
    const retrieved = await index.get('vec1');
    console.log(`   Retrieved: ${retrieved ? retrieved.id : 'null'}`);
    console.log('   ✓ Get works\n');

    // Test delete
    console.log('8. Delete:');
    const deleted = await index.delete('vec1');
    console.log(`   Deleted: ${deleted}`);
    const statsAfter = await index.stats();
    console.log(`   Vectors remaining: ${statsAfter.vectorCount}`);
    console.log('   ✓ Delete works\n');

    // Test utilities
    console.log('9. Utilities:');
    const v1 = Utils.randomVector(128);
    const v2 = Utils.randomVector(128);
    const similarity = Utils.cosineSimilarity(v1, v2);
    const distance = Utils.euclideanDistance(v1, v2);
    const normalized = Utils.normalize(v1);
    console.log(`   Cosine similarity: ${similarity.toFixed(4)}`);
    console.log(`   Euclidean distance: ${distance.toFixed(4)}`);
    console.log(`   Normalized length: ${Math.sqrt(normalized.reduce((s, v) => s + v * v, 0)).toFixed(4)}`);
    console.log('   ✓ Utilities work\n');

    console.log('✅ All tests passed!');
    return true;
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
    return false;
  }
}

// Run tests
testBasicOperations().then(success => {
  process.exit(success ? 0 : 1);
});
