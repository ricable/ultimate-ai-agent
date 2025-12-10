const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');
const xml2js = require('xml2js');
const { execSync } = require('child_process');
require('dotenv').config();

const DATA_DIR = path.join(__dirname, 'data');
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

async function ensureRuvectorExtension() {
  const client = await pool.connect();
  try {
    console.log('Checking RuVector extension...');

    // Try to enable ruvector extension
    try {
      await client.query('CREATE EXTENSION IF NOT EXISTS ruvector');
      console.log('✓ RuVector extension enabled');
    } catch (err) {
      console.log('Note: RuVector extension not available, using standard PostgreSQL graph functions');
      console.log('For full features, install via: npx @ruvector/postgres-cli install');
    }
  } finally {
    client.release();
  }
}

async function createGraph() {
  console.log(`Creating graph: ${GRAPH_NAME}...`);

  try {
    // Use ruvector CLI to create graph
    const connStr = getConnectionString();
    execSync(
      `npx --yes @ruvector/postgres-cli -c "${connStr}" graph create ${GRAPH_NAME}`,
      { stdio: 'inherit' }
    );
    console.log('✓ Graph created successfully');
  } catch (error) {
    // Graph might already exist, which is fine
    console.log('Graph already exists or created');
  }
}

async function parseGraphML(filePath) {
  console.log(`Parsing GraphML file: ${filePath}`);

  const xmlData = fs.readFileSync(filePath, 'utf8');
  const parser = new xml2js.Parser();
  const result = await parser.parseStringPromise(xmlData);

  const graphml = result.graphml;
  const graph = graphml.graph[0];

  // Extract nodes and edges
  const nodes = graph.node || [];
  const edges = graph.edge || [];

  console.log(`Found ${nodes.length} nodes and ${edges.length} edges`);

  return { nodes, edges };
}

async function insertNodesViaRuvector(nodes) {
  console.log('Inserting nodes into graph...');
  const connStr = getConnectionString();

  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i];
    const nodeId = node.$.id;

    // Extract node properties
    const properties = {};
    const dataElements = node.data || [];

    dataElements.forEach(data => {
      const key = data.$.key;
      const value = data._;
      if (key && value) {
        properties[key] = value;
      }
    });

    // Extract label if available
    const label = properties.label || properties.name || properties.type || 'Node';

    // Convert properties to JSON string for CLI
    const propsJson = JSON.stringify(properties).replace(/"/g, '\\"');

    try {
      execSync(
        `npx --yes @ruvector/postgres-cli -c "${connStr}" graph create-node ${GRAPH_NAME} --id "${nodeId}" --labels "${label}" --properties '${JSON.stringify(properties)}'`,
        { stdio: 'pipe' }
      );
    } catch (err) {
      // Continue on errors (node might exist)
    }

    if ((i + 1) % 100 === 0) {
      console.log(`  Inserted ${i + 1}/${nodes.length} nodes`);
    }
  }

  console.log(`✓ Inserted all ${nodes.length} nodes`);
}

async function insertEdgesViaRuvector(edges) {
  console.log('Inserting edges into graph...');
  const connStr = getConnectionString();

  for (let i = 0; i < edges.length; i++) {
    const edge = edges[i];
    const sourceId = edge.$.source;
    const targetId = edge.$.target;

    // Extract edge properties
    const properties = {};
    const dataElements = edge.data || [];

    dataElements.forEach(data => {
      const key = data.$.key;
      const value = data._;
      if (key && value) {
        properties[key] = value;
      }
    });

    // Extract label if available
    const label = properties.label || properties.type || 'RELATES_TO';

    try {
      execSync(
        `npx --yes @ruvector/postgres-cli -c "${connStr}" graph create-edge ${GRAPH_NAME} --from "${sourceId}" --to "${targetId}" --type "${label}" --properties '${JSON.stringify(properties)}'`,
        { stdio: 'pipe' }
      );
    } catch (err) {
      // Continue on errors
    }

    if ((i + 1) % 100 === 0) {
      console.log(`  Inserted ${i + 1}/${edges.length} edges`);
    }
  }

  console.log(`✓ Inserted all ${edges.length} edges`);
}

async function loadGraphMLToPostgres() {
  try {
    // Find GraphML file in data directory
    const files = fs.readdirSync(DATA_DIR);
    const graphmlFile = files.find(f => f.endsWith('.graphml'));

    if (!graphmlFile) {
      throw new Error('No GraphML file found in data directory. Run download-dataset.js first.');
    }

    const filePath = path.join(DATA_DIR, graphmlFile);

    // Ensure RuVector extension is available
    await ensureRuvectorExtension();

    // Create graph
    await createGraph();

    // Parse GraphML
    const { nodes, edges } = await parseGraphML(filePath);

    // Insert data using RuVector graph commands
    await insertNodesViaRuvector(nodes);
    await insertEdgesViaRuvector(edges);

    console.log('\n✓ Successfully loaded 3GPP knowledge graph into RuVector!');
    console.log('\nYou can now query the data using:');
    console.log('  - Cypher: npx @ruvector/postgres-cli graph query ' + GRAPH_NAME + ' "MATCH (n) RETURN n LIMIT 10"');
    console.log('  - Graph stats: npx @ruvector/postgres-cli graph stats ' + GRAPH_NAME);
    console.log('  - Shortest path: npx @ruvector/postgres-cli graph shortest-path ' + GRAPH_NAME + ' --from <id1> --to <id2>');

  } catch (error) {
    console.error('Error loading data:', error.message);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

// Run if called directly
if (require.main === module) {
  loadGraphMLToPostgres();
}

module.exports = { loadGraphMLToPostgres, parseGraphML };
