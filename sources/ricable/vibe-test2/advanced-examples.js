const { execSync } = require('child_process');
require('dotenv').config();

const GRAPH_NAME = '3gpp_knowledge_graph';

// Build connection string for ruvector CLI
function getConnectionString() {
  const host = process.env.POSTGRES_HOST || 'localhost';
  const port = process.env.POSTGRES_PORT || 5432;
  const db = process.env.POSTGRES_DB || 'ruvector';
  const user = process.env.POSTGRES_USER || 'ruvector';
  const password = process.env.POSTGRES_PASSWORD || 'ruvector';

  return `postgresql://${user}:${password}@${host}:${port}/${db}`;
}

function runCommand(description, command) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`Example: ${description}`);
  console.log(`${'='.repeat(70)}`);
  console.log(`Command: ${command}\n`);

  try {
    const result = execSync(command, { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
    console.log(result);
  } catch (error) {
    console.error('Error:', error.message);
    console.error('Note: Some features require RuVector to be fully installed');
  }
}

async function main() {
  const connStr = getConnectionString();

  console.log('\n' + '='.repeat(70));
  console.log('RuVector Advanced Features - GNN, Embeddings & Graph Analytics');
  console.log('='.repeat(70));

  // ===== Graph Neural Networks (GNN) =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 1: Graph Neural Networks (GNN)');
  console.log('█'.repeat(70));

  runCommand(
    'Create a GCN (Graph Convolutional Network) layer',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" gnn create 3gpp_gcn --type gcn --input-dim 128 --hidden-dim 64 --output-dim 32`
  );

  runCommand(
    'Create a GraphSAGE layer for inductive learning',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" gnn create 3gpp_sage --type graphsage --input-dim 128 --hidden-dim 64`
  );

  runCommand(
    'Create a GAT (Graph Attention Network) layer',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" gnn create 3gpp_gat --type gat --input-dim 128 --hidden-dim 64 --num-heads 4`
  );

  // ===== Vector Operations on Graph =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 2: Vector Embeddings for Graph Nodes');
  console.log('█'.repeat(70));

  runCommand(
    'Create vector table for node embeddings (384-dim)',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" vector create ${GRAPH_NAME}_embeddings --dim 384 --index hnsw`
  );

  console.log('\nNote: After creating embeddings, you can:');
  console.log('1. Generate embeddings for your 3GPP specifications using an embedding model');
  console.log('2. Insert them into the vector table');
  console.log('3. Perform semantic similarity search on specifications');

  // ===== Hyperbolic Embeddings =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 3: Hyperbolic Embeddings (for Hierarchical Data)');
  console.log('█'.repeat(70));

  runCommand(
    'Compute Poincaré distance between two embeddings',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" hyperbolic poincare-distance --x "[0.1, 0.2, 0.3]" --y "[0.4, 0.5, 0.6]"`
  );

  runCommand(
    'Compute Lorentz distance',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" hyperbolic lorentz-distance --x "[0.1, 0.2, 0.3]" --y "[0.4, 0.5, 0.6]"`
  );

  console.log('\nNote: Hyperbolic embeddings are ideal for:');
  console.log('- Hierarchical 3GPP specification structures');
  console.log('- Protocol dependency trees');
  console.log('- Version evolution graphs');

  // ===== Attention Mechanisms =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 4: Attention Mechanisms (39 types available)');
  console.log('█'.repeat(70));

  runCommand(
    'Scaled Dot-Product Attention',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" attention scaled-dot-product --query "[1.0, 2.0, 3.0]" --key "[4.0, 5.0, 6.0]" --value "[7.0, 8.0, 9.0]"`
  );

  runCommand(
    'Multi-Head Attention',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" attention multi-head --query "[1.0, 2.0, 3.0, 4.0]" --key "[5.0, 6.0, 7.0, 8.0]" --value "[9.0, 10.0, 11.0, 12.0]" --num-heads 2`
  );

  console.log('\nAvailable Attention Types:');
  console.log('- Flash Attention (optimized)');
  console.log('- Cross Attention');
  console.log('- Self Attention');
  console.log('- Sparse Attention');
  console.log('- And 35+ more variants!');

  // ===== Graph Algorithms =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 5: Graph Analytics & Algorithms');
  console.log('█'.repeat(70));

  runCommand(
    'Get comprehensive graph statistics',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" graph stats ${GRAPH_NAME}`
  );

  console.log('\nExample: Shortest Path (requires actual node IDs)');
  console.log(`Command: npx @ruvector/postgres-cli -c "${connStr}" graph shortest-path ${GRAPH_NAME} --from "spec_38.401" --to "spec_38.331"`);

  console.log('\nExample: Graph Traversal');
  console.log(`Command: npx @ruvector/postgres-cli -c "${connStr}" graph traverse --start "spec_root" --max-depth 3`);

  // ===== Sparse Vectors & BM25 =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 6: Sparse Vectors & BM25 (Hybrid Search)');
  console.log('█'.repeat(70));

  runCommand(
    'Create sparse vector',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" sparse create --indices "[0, 5, 10, 15]" --values "[0.8, 0.6, 0.4, 0.2]" --dim 10000`
  );

  console.log('\nExample: BM25 scoring for keyword search');
  console.log(`Command: npx @ruvector/postgres-cli -c "${connStr}" sparse bm25 \\`);
  console.log(`  --query '{"indices": [1,5,10], "values": [0.8,0.5,0.3]}' \\`);
  console.log(`  --doc '{"indices": [1,5], "values": [2,1]}' \\`);
  console.log(`  --doc-len 150 --avg-doc-len 200`);

  // ===== Vector Quantization =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 7: Vector Quantization (for compression)');
  console.log('█'.repeat(70));

  runCommand(
    'Product Quantization',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" quant product --vector "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]" --num-subvectors 2 --bits 8`
  );

  runCommand(
    'Scalar Quantization',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" quant scalar --vector "[0.1, 0.2, 0.3, 0.4]" --bits 8`
  );

  console.log('\nNote: Quantization reduces storage and speeds up similarity search');

  // ===== Agent Routing (Tiny Dancer) =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 8: Agent Routing (Tiny Dancer)');
  console.log('█'.repeat(70));

  console.log('\nTiny Dancer enables intelligent agent routing based on:');
  console.log('- Semantic similarity');
  console.log('- Graph structure');
  console.log('- Custom routing logic');
  console.log('\nExample usage: Route queries to appropriate 3GPP specification experts');

  runCommand(
    'Get agent routing info',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" routing --help`
  );

  // ===== Learning & ReasoningBank =====
  console.log('\n' + '█'.repeat(70));
  console.log('SECTION 9: Self-Learning & ReasoningBank');
  console.log('█'.repeat(70));

  console.log('\nReasoningBank provides:');
  console.log('- Self-improving query patterns');
  console.log('- Learned optimizations');
  console.log('- Reasoning trace storage');
  console.log('\nIdeal for: Building intelligent 3GPP specification assistants');

  runCommand(
    'Learning operations help',
    `npx --yes @ruvector/postgres-cli -c "${connStr}" learning --help`
  );

  // ===== Practical Use Cases =====
  console.log('\n' + '█'.repeat(70));
  console.log('PRACTICAL USE CASES FOR 3GPP KNOWLEDGE GRAPH');
  console.log('█'.repeat(70));

  console.log('\n1. Semantic Specification Search:');
  console.log('   - Generate embeddings for all 3GPP specs');
  console.log('   - Use vector similarity to find related specifications');
  console.log('   - Combine with BM25 for hybrid keyword + semantic search');

  console.log('\n2. Protocol Dependency Analysis:');
  console.log('   - Use GNN to learn protocol relationships');
  console.log('   - Identify critical dependency paths');
  console.log('   - Detect circular dependencies or conflicts');

  console.log('\n3. Version Evolution Tracking:');
  console.log('   - Use graph traversal to track spec changes across releases');
  console.log('   - Hyperbolic embeddings to represent version hierarchies');
  console.log('   - Shortest path to find upgrade paths');

  console.log('\n4. Intelligent Q&A System:');
  console.log('   - Attention mechanisms for context-aware answers');
  console.log('   - Agent routing to specialist knowledge bases');
  console.log('   - ReasoningBank for continuous learning');

  console.log('\n' + '='.repeat(70));
  console.log('Advanced examples completed!');
  console.log('='.repeat(70));
  console.log('\nNext Steps:');
  console.log('1. Install RuVector: npx @ruvector/postgres-cli install');
  console.log('2. Load your 3GPP data: npm run setup');
  console.log('3. Experiment with these features on your actual graph!');
  console.log('');
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { runCommand, getConnectionString };
