/**
 * Advanced search features example
 */

const { VectorIndex, Utils } = require('ruvector');

async function main() {
  console.log('🔍 Advanced Search Example\n');

  // Create index
  const index = new VectorIndex({
    dimension: 128,
    metric: 'cosine',
    indexType: 'hnsw'
  });

  // Insert vectors with rich metadata
  console.log('Inserting documents...');
  const documents = [
    { id: 'doc1', category: 'tech', tags: ['ai', 'ml'] },
    { id: 'doc2', category: 'tech', tags: ['web', 'javascript'] },
    { id: 'doc3', category: 'science', tags: ['physics', 'quantum'] },
    { id: 'doc4', category: 'science', tags: ['biology', 'dna'] },
    { id: 'doc5', category: 'business', tags: ['finance', 'stocks'] }
  ];

  const vectors = documents.map(doc => ({
    id: doc.id,
    values: Utils.randomVector(128),
    metadata: doc
  }));

  await index.insertBatch(vectors);

  // Perform different types of searches
  const query = Utils.randomVector(128);

  console.log('\n1. Basic search (top 3):');
  const basic = await index.search(query, { k: 3 });
  basic.forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.id} - ${r.metadata.category} (${r.score.toFixed(4)})`);
  });

  console.log('\n2. Search with HNSW tuning (higher accuracy):');
  const accurate = await index.search(query, { k: 3, ef: 100 });
  accurate.forEach((r, i) => {
    console.log(`  ${i + 1}. ${r.id} - ${r.metadata.category} (${r.score.toFixed(4)})`);
  });

  // Calculate similarities manually
  console.log('\n3. Manual similarity calculation:');
  const vec1 = Utils.randomVector(128);
  const vec2 = Utils.randomVector(128);
  const similarity = Utils.cosineSimilarity(vec1, vec2);
  const distance = Utils.euclideanDistance(vec1, vec2);
  console.log(`  Cosine similarity: ${similarity.toFixed(4)}`);
  console.log(`  Euclidean distance: ${distance.toFixed(4)}`);

  // Get specific vector
  console.log('\n4. Get vector by ID:');
  const retrieved = await index.get('doc1');
  if (retrieved) {
    console.log(`  Retrieved: ${retrieved.id}`);
    console.log(`  Metadata:`, retrieved.metadata);
    console.log(`  Vector dimension: ${retrieved.values.length}`);
  }

  // Delete and verify
  console.log('\n5. Delete operation:');
  const deleted = await index.delete('doc5');
  console.log(`  Deleted doc5: ${deleted}`);

  const statsAfter = await index.stats();
  console.log(`  Vectors remaining: ${statsAfter.vectorCount}`);
}

main().catch(console.error);
