/**
 * Basic usage example for rUvector
 */

const { VectorIndex, Utils, getBackendInfo } = require('ruvector');

async function main() {
  console.log('🚀 rUvector Basic Usage Example\n');

  // Show backend info
  const info = getBackendInfo();
  console.log(`Backend: ${info.type} (${info.version})`);
  console.log(`Features: ${info.features.join(', ')}\n`);

  // Create a new index
  console.log('Creating index...');
  const index = new VectorIndex({
    dimension: 384,
    metric: 'cosine',
    indexType: 'hnsw',
    hnswConfig: {
      m: 16,
      efConstruction: 200
    }
  });

  // Insert some vectors
  console.log('Inserting vectors...');
  const vectors = [];
  for (let i = 0; i < 1000; i++) {
    vectors.push({
      id: `doc_${i}`,
      values: Utils.randomVector(384),
      metadata: {
        title: `Document ${i}`,
        category: i % 5 === 0 ? 'important' : 'normal'
      }
    });
  }

  await index.insertBatch(vectors, {
    batchSize: 100,
    progressCallback: (progress) => {
      process.stdout.write(`\rProgress: ${(progress * 100).toFixed(1)}%`);
    }
  });
  console.log('\n');

  // Get stats
  const stats = await index.stats();
  console.log('Index stats:', {
    vectors: stats.vectorCount,
    dimension: stats.dimension,
    type: stats.indexType
  });
  console.log();

  // Search
  console.log('Searching...');
  const query = Utils.randomVector(384);
  const results = await index.search(query, { k: 5 });

  console.log('\nTop 5 results:');
  results.forEach((result, i) => {
    console.log(`  ${i + 1}. ${result.id} (score: ${result.score.toFixed(4)})`);
    console.log(`     metadata: ${JSON.stringify(result.metadata)}`);
  });

  // Save index
  console.log('\nSaving index...');
  await index.save('my-index.bin');
  console.log('✓ Index saved to my-index.bin');

  // Load and verify
  console.log('\nLoading index...');
  const loadedIndex = await VectorIndex.load('my-index.bin');
  const loadedStats = await loadedIndex.stats();
  console.log('✓ Index loaded, vectors:', loadedStats.vectorCount);
}

main().catch(console.error);
