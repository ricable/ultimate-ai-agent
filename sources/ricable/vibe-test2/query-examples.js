const { execSync } = require('child_process');
const { Pool } = require('pg');
require('dotenv').config();

const GRAPH_NAME = '3gpp_knowledge_graph';

// PostgreSQL connection pool
const pool = new Pool({
  host: process.env.POSTGRES_HOST || 'localhost',
  port: process.env.POSTGRES_PORT || 5432,
  database: process.env.POSTGRES_DB || 'ruvector',
  user: process.env.POSTGRES_USER || 'ruvector',
  password: process.env.POSTGRES_PASSWORD || 'ruvector'
});

// Build connection string for ruvector CLI
function getConnectionString() {
  const host = process.env.POSTGRES_HOST || 'localhost';
  const port = process.env.POSTGRES_PORT || 5432;
  const db = process.env.POSTGRES_DB || 'ruvector';
  const user = process.env.POSTGRES_USER || 'ruvector';
  const password = process.env.POSTGRES_PASSWORD || 'ruvector';

  return `postgresql://${user}:${password}@${host}:${port}/${db}`;
}

function runCypherQuery(description, query) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`Cypher Query: ${description}`);
  console.log(`${'='.repeat(70)}`);
  console.log(`Query: ${query}\n`);

  try {
    const connStr = getConnectionString();
    const result = execSync(
      `npx --yes @ruvector/postgres-cli -c "${connStr}" graph query ${GRAPH_NAME} "${query}"`,
      { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }
    );
    console.log(result);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

function runGraphCommand(description, command) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`Graph Command: ${description}`);
  console.log(`${'='.repeat(70)}`);
  console.log(`Command: ${command}\n`);

  try {
    const result = execSync(command, { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
    console.log(result);
  } catch (error) {
    console.error('Error:', error.message);
  }
}

async function runSQLQuery(description, query) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`SQL Query: ${description}`);
  console.log(`${'='.repeat(70)}`);
  console.log(`Query: ${query}\n`);

  try {
    const result = await pool.query(query);
    console.log(`Results (${result.rows.length} rows):`);
    console.table(result.rows);
    return result.rows;
  } catch (error) {
    console.error('Error:', error.message);
  }
}

async function main() {
  const connStr = getConnectionString();

  console.log('\n' + '='.repeat(70));
  console.log('3GPP Knowledge Graph - RuVector Query Examples');
  console.log('='.repeat(70));

  try {
    // Example 1: Graph Statistics
    runGraphCommand(
      'Get graph statistics',
      `npx --yes @ruvector/postgres-cli -c "${connStr}" graph stats ${GRAPH_NAME}`
    );

    // Example 2: List all graphs
    runGraphCommand(
      'List all graphs',
      `npx --yes @ruvector/postgres-cli -c "${connStr}" graph list`
    );

    // Example 3: Simple Cypher - Get all nodes (limited)
    runCypherQuery(
      'Get 10 random nodes',
      'MATCH (n) RETURN n LIMIT 10'
    );

    // Example 4: Cypher - Get nodes with specific property
    runCypherQuery(
      'Find nodes with specific properties',
      'MATCH (n) WHERE n.type IS NOT NULL RETURN n.type, count(*) as count ORDER BY count DESC LIMIT 10'
    );

    // Example 5: Cypher - Get relationships
    runCypherQuery(
      'Get sample relationships',
      'MATCH (n)-[r]->(m) RETURN type(r) as relationship_type, count(*) as count GROUP BY type(r) ORDER BY count DESC LIMIT 10'
    );

    // Example 6: Cypher - 2-hop neighborhood
    runCypherQuery(
      'Get 2-hop neighborhood of a node',
      'MATCH (n)-[r1]->(m)-[r2]->(o) RETURN n, type(r1), m, type(r2), o LIMIT 10'
    );

    // Example 7: Cypher - Pattern matching
    runCypherQuery(
      'Find triangles in the graph',
      'MATCH (a)-[]->(b)-[]->(c)-[]->(a) RETURN a, b, c LIMIT 5'
    );

    // Example 8: SQL - Direct query on graph tables
    await runSQLQuery(
      'Query nodes table directly',
      `SELECT * FROM ${GRAPH_NAME}_nodes LIMIT 10`
    );

    await runSQLQuery(
      'Query edges table directly',
      `SELECT * FROM ${GRAPH_NAME}_edges LIMIT 10`
    );

    // Example 9: Get node by ID (you'll need to replace with actual ID from your data)
    console.log('\n' + '='.repeat(70));
    console.log('Note: For shortest path, replace <node_id_1> and <node_id_2> with actual IDs');
    console.log('='.repeat(70));
    console.log('Example command:');
    console.log(`  npx @ruvector/postgres-cli -c "${connStr}" graph shortest-path ${GRAPH_NAME} --from "node1" --to "node2"`);

    // Example 10: Advanced - Search by property value
    runCypherQuery(
      'Search nodes containing specific text in properties',
      'MATCH (n) WHERE toString(n) CONTAINS "5G" RETURN n LIMIT 10'
    );

    console.log('\n' + '='.repeat(70));
    console.log('Query examples completed!');
    console.log('='.repeat(70));
    console.log('\nAdditional RuVector Features:');
    console.log('1. GNN Operations:');
    console.log('   npx @ruvector/postgres-cli gnn create my_gnn --type gcn --input-dim 128 --hidden-dim 64');
    console.log('\n2. Vector Search on Graph Nodes:');
    console.log('   npx @ruvector/postgres-cli vector create graph_embeddings --dim 384');
    console.log('\n3. Hyperbolic Embeddings:');
    console.log('   npx @ruvector/postgres-cli hyperbolic poincare-distance --x "[0.1,0.2]" --y "[0.3,0.4]"');
    console.log('\n4. Attention Mechanisms:');
    console.log('   npx @ruvector/postgres-cli attention scaled-dot-product --query "[1,2,3]" --key "[4,5,6]" --value "[7,8,9]"');
    console.log('');

  } catch (error) {
    console.error('Error running queries:', error.message);
  } finally {
    await pool.end();
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { runCypherQuery, runGraphCommand };
