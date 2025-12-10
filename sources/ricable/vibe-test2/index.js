const { downloadDataset } = require('./download-dataset');
const { loadGraphMLToPostgres } = require('./load-to-postgres');
require('dotenv').config();

async function main() {
  console.log('='.repeat(70));
  console.log('3GPP Knowledge Graph - RuVector/PostgreSQL Integration');
  console.log('='.repeat(70));
  console.log('');

  try {
    // Step 1: Download dataset from Hugging Face
    console.log('Step 1: Downloading dataset from Hugging Face...');
    console.log('Source: otellm/3gpp_knowledgeGraph');
    await downloadDataset();
    console.log('');

    // Step 2: Load data into PostgreSQL/RuVector
    console.log('Step 2: Loading data into RuVector graph database...');
    console.log('Creating graph: 3gpp_knowledge_graph');
    await loadGraphMLToPostgres();
    console.log('');

    console.log('='.repeat(70));
    console.log('✓ Setup complete!');
    console.log('='.repeat(70));
    console.log('');
    console.log('Next steps:');
    console.log('');
    console.log('1. Run example queries:');
    console.log('   npm run query          # Cypher queries and graph operations');
    console.log('   npm run advanced       # GNN, embeddings, attention mechanisms');
    console.log('');
    console.log('2. Query via CLI:');
    console.log('   npx @ruvector/postgres-cli graph query 3gpp_knowledge_graph "MATCH (n) RETURN n LIMIT 10"');
    console.log('   npx @ruvector/postgres-cli graph stats 3gpp_knowledge_graph');
    console.log('');
    console.log('3. Explore advanced features:');
    console.log('   - Cypher queries (Neo4j-compatible)');
    console.log('   - Graph Neural Networks (GCN, GraphSAGE, GAT)');
    console.log('   - Hyperbolic embeddings for hierarchies');
    console.log('   - 39 attention mechanisms');
    console.log('   - Vector similarity search');
    console.log('   - BM25 hybrid search');
    console.log('');
    console.log('4. Documentation:');
    console.log('   See README.md for detailed usage examples');
    console.log('');

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    console.error('\nTroubleshooting:');
    console.error('');
    console.error('1. Install RuVector PostgreSQL:');
    console.error('   npm run install-ruvector');
    console.error('   (or use existing PostgreSQL and update .env)');
    console.error('');
    console.error('2. Check database credentials in .env');
    console.error('   cp .env.example .env');
    console.error('   (then edit .env with your settings)');
    console.error('');
    console.error('3. Verify internet connection for Hugging Face download');
    console.error('');
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { main };
