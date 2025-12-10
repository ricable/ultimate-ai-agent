/**
 * 3GPP Knowledge Graph GraphML Parser - Usage Examples
 *
 * Demonstrates how to:
 * 1. Load and parse GraphML knowledge graphs
 * 2. Query nodes and relationships
 * 3. Extract 3GPP entities (IEs, procedures, parameters)
 * 4. Prepare data for ruvector indexing
 * 5. Integrate with SpecMetadataStore
 *
 * @module knowledge/graphml-example
 */

import {
  ThreeGPPKnowledgeGraph,
  createKnowledgeGraph,
  loadKnowledgeGraph,
  type GraphMLNode,
  type GraphMLEdge,
  extractASN1Definition,
  extractParameterRange,
  extractProcedureStates,
  detect3GPPSeries,
} from './graphml-parser.js';

// ============================================================================
// Example 1: Create and Parse Sample GraphML
// ============================================================================

/**
 * Create a sample 3GPP knowledge graph from GraphML
 */
export async function example1_LoadGraphML() {
  console.log('\n=== Example 1: Load GraphML Knowledge Graph ===\n');

  const kg = createKnowledgeGraph();

  // Sample GraphML representing TS 38.331 RRC specification structure
  const sampleGraphML = `<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <graph id="3gpp_knowledge" edgedefault="directed" release="R18" series="38">
    <!-- Top-level Specification -->
    <node id="TS38331">
      <data key="label">TS 38.331 - NR RRC Protocol Specification</data>
      <data key="type">3gpp_spec</data>
      <data key="release">R18</data>
      <data key="workingGroup">RAN2</data>
      <data key="version">18.1.0</data>
    </node>

    <!-- Sections -->
    <node id="TS38331_5">
      <data key="label">5 Radio Resource Control procedures</data>
      <data key="type">section</data>
      <data key="sectionNumber">5</data>
    </node>

    <node id="TS38331_5_3">
      <data key="label">5.3 RRC connection control</data>
      <data key="type">section</data>
      <data key="sectionNumber">5.3</data>
    </node>

    <!-- Messages -->
    <node id="RRCSetup">
      <data key="label">RRCSetup</data>
      <data key="type">message</data>
      <data key="asn1Type">SEQUENCE</data>
      <data key="definition">Message sent by gNB to establish RRC connection</data>
    </node>

    <node id="RRCSetupComplete">
      <data key="label">RRCSetupComplete</data>
      <data key="type">message</data>
      <data key="asn1Type">SEQUENCE</data>
      <data key="definition">Message sent by UE to confirm RRC connection establishment</data>
    </node>

    <!-- Information Elements -->
    <node id="SIB1">
      <data key="label">SystemInformationBlockType1</data>
      <data key="type">ie</data>
      <data key="asn1Name">SIB1</data>
      <data key="asn1Type">SEQUENCE</data>
      <data key="asn1Definition">Contains essential system information</data>
    </node>

    <node id="RRCSetupIE">
      <data key="label">RRCSetup-IEs</data>
      <data key="type">ie</data>
      <data key="asn1Name">RRCSetup-IEs</data>
      <data key="asn1Type">SEQUENCE</data>
    </node>

    <!-- Parameters -->
    <node id="P0NominalPUSCH">
      <data key="label">p0-NominalPUSCH</data>
      <data key="type">parameter</data>
      <data key="min">-202</data>
      <data key="max">24</data>
      <data key="unit">dBm</data>
      <data key="definition">Nominal power for PUSCH transmission</data>
    </node>

    <node id="NRPCI">
      <data key="label">physCellId</data>
      <data key="type">parameter</data>
      <data key="min">0</data>
      <data key="max">1007</data>
      <data key="unit">none</data>
      <data key="definition">Physical Cell Identity for NR</data>
    </node>

    <!-- Procedures -->
    <node id="RRCEstabProc">
      <data key="label">RRC Connection Establishment</data>
      <data key="type">procedure</data>
      <data key="states">IDLE,CONNECTING,CONNECTED,SUSPENDED</data>
      <data key="definition">Procedure to establish RRC connection</data>
    </node>

    <!-- Edges (Relationships) -->
    <edge id="e1" source="TS38331" target="TS38331_5">
      <data key="type">contains</data>
    </edge>
    <edge id="e2" source="TS38331_5" target="TS38331_5_3">
      <data key="type">contains</data>
    </edge>
    <edge id="e3" source="TS38331_5_3" target="RRCSetup">
      <data key="type">defines</data>
    </edge>
    <edge id="e4" source="TS38331_5_3" target="RRCSetupComplete">
      <data key="type">defines</data>
    </edge>
    <edge id="e5" source="RRCSetup" target="RRCSetupIE">
      <data key="type">contains</data>
    </edge>
    <edge id="e6" source="TS38331" target="SIB1">
      <data key="type">defines</data>
    </edge>
    <edge id="e7" source="TS38331" target="P0NominalPUSCH">
      <data key="type">defines</data>
    </edge>
    <edge id="e8" source="TS38331" target="NRPCI">
      <data key="type">defines</data>
    </edge>
    <edge id="e9" source="TS38331_5_3" target="RRCEstabProc">
      <data key="type">defines</data>
    </edge>
    <edge id="e10" source="RRCEstabProc" target="RRCSetup">
      <data key="type">implements</data>
    </edge>

    <!-- Cross-references to other specs -->
    <node id="TS38300">
      <data key="label">TS 38.300 - NR Overall Description</data>
      <data key="type">3gpp_spec</data>
      <data key="release">R18</data>
    </node>
    <edge id="e11" source="TS38331" target="TS38300">
      <data key="type">references</data>
    </edge>
  </graph>
</graphml>`;

  // Parse the GraphML (in memory)
  const fs = await import('fs/promises');
  const tempFile = '/tmp/3gpp-sample.graphml';
  await fs.writeFile(tempFile, sampleGraphML);
  await kg.loadFromGraphML(tempFile);

  // Display statistics
  const stats = kg.getStats();
  console.log('Graph Statistics:');
  console.log(`  Nodes: ${stats.nodeCount}`);
  console.log(`  Edges: ${stats.edgeCount}`);
  console.log(`  Average Degree: ${stats.avgDegree.toFixed(2)}`);
  console.log('\nNode Type Distribution:');
  for (const [type, count] of Object.entries(stats.nodeTypeDistribution)) {
    console.log(`  ${type}: ${count}`);
  }

  return kg;
}

// ============================================================================
// Example 2: Query Knowledge Graph
// ============================================================================

/**
 * Demonstrate various query operations
 */
export async function example2_QueryGraph() {
  console.log('\n=== Example 2: Query Knowledge Graph ===\n');

  const kg = await example1_LoadGraphML();

  // Find a specific spec
  console.log('1. Find Specification:');
  const spec = kg.findSpec('TS38331');
  if (spec) {
    console.log(`  Found: ${spec.label}`);
    console.log(`  Type: ${spec.type}`);
    console.log(`  Release: ${spec.attributes.release}`);
  }

  // Find all messages
  console.log('\n2. Find All Messages:');
  const messages = kg.findByType('message');
  messages.forEach(msg => {
    console.log(`  - ${msg.label} (${msg.attributes.asn1Type})`);
  });

  // Find all parameters
  console.log('\n3. Find All Parameters:');
  const parameters = kg.findByType('parameter');
  parameters.forEach(param => {
    const range = extractParameterRange(param);
    console.log(`  - ${param.label}: [${range.min}, ${range.max}] ${range.unit}`);
  });

  // Find related nodes
  console.log('\n4. Find Related Nodes (depth=2):');
  const related = kg.findRelated('RRCSetup', 2);
  console.log(`  Found ${related.length} related nodes:`);
  related.slice(0, 5).forEach(node => {
    console.log(`  - ${node.label} (${node.type})`);
  });

  // Find path between nodes
  console.log('\n5. Find Path:');
  const path = kg.findPath('TS38331', 'RRCSetup');
  console.log(`  Path from TS38331 to RRCSetup:`);
  path.forEach((node, i) => {
    console.log(`  ${i + 1}. ${node.label}`);
  });

  // Search by label
  console.log('\n6. Search by Label:');
  const searchResults = kg.searchByLabel('RRC');
  console.log(`  Found ${searchResults.length} nodes containing 'RRC':`);
  searchResults.slice(0, 5).forEach(node => {
    console.log(`  - ${node.label}`);
  });

  return kg;
}

// ============================================================================
// Example 3: Extract 3GPP Entities
// ============================================================================

/**
 * Extract specific 3GPP entities (IEs, procedures, parameters)
 */
export async function example3_ExtractEntities() {
  console.log('\n=== Example 3: Extract 3GPP Entities ===\n');

  const kg = await example1_LoadGraphML();

  // Extract ASN.1 Information Elements
  console.log('1. ASN.1 Information Elements:');
  const ies = kg.findByType('ie');
  ies.forEach(ie => {
    const asn1 = extractASN1Definition(ie);
    if (asn1) {
      console.log(`  - ${asn1.name} (${asn1.type})`);
      console.log(`    Definition: ${asn1.definition}`);
    }
  });

  // Extract Procedures and State Machines
  console.log('\n2. Procedures and State Machines:');
  const procedures = kg.findByType('procedure');
  procedures.forEach(proc => {
    const states = extractProcedureStates(proc);
    console.log(`  - ${proc.label}`);
    console.log(`    States: ${states.join(' → ')}`);
  });

  // Extract Parameter Constraints
  console.log('\n3. Parameter Constraints (3GPP TS 28.552 / 38.331):');
  const params = kg.findByType('parameter');
  params.forEach(param => {
    const range = extractParameterRange(param);
    console.log(`  - ${param.label}`);
    console.log(`    Range: [${range.min}, ${range.max}] ${range.unit}`);
    console.log(`    Definition: ${param.attributes.definition}`);
  });

  // Detect spec series
  console.log('\n4. Detect Specification Series:');
  const specs = kg.findByType('3gpp_spec');
  specs.forEach(spec => {
    const series = detect3GPPSeries(spec.id);
    console.log(`  - ${spec.label}`);
    console.log(`    Series: ${series}`);
  });
}

// ============================================================================
// Example 4: Export for Vector Indexing (ruvector)
// ============================================================================

/**
 * Prepare knowledge graph for ruvector semantic search
 */
export async function example4_VectorIndexing() {
  console.log('\n=== Example 4: Export for Vector Indexing ===\n');

  const kg = await example1_LoadGraphML();

  // Export all nodes with embedding text
  const vectorData = kg.exportForVectorIndexing();

  console.log(`Exported ${vectorData.length} nodes for vector indexing\n`);

  // Show first few examples
  console.log('Sample embedding texts:');
  vectorData.slice(0, 3).forEach((item, i) => {
    console.log(`\n${i + 1}. Node ID: ${item.id}`);
    console.log('   Embedding Text:');
    console.log('   ' + item.text.split('\n').join('\n   '));
    console.log('   Metadata:', JSON.stringify(item.metadata, null, 2).split('\n').join('\n   '));
  });

  // In production, you would index this with ruvector:
  console.log('\n--- Integration with ruvector ---');
  console.log('// Example ruvector indexing code:');
  console.log('/*');
  console.log('import { RuvectorClient } from "ruvector";');
  console.log('const client = new RuvectorClient("./ruvector-spatial.db");');
  console.log('');
  console.log('for (const item of vectorData) {');
  console.log('  await client.index({');
  console.log('    id: item.id,');
  console.log('    text: item.text,');
  console.log('    metadata: item.metadata');
  console.log('  });');
  console.log('}');
  console.log('*/');

  return vectorData;
}

// ============================================================================
// Example 5: Subgraph Extraction
// ============================================================================

/**
 * Extract subgraphs for specific spec series
 */
export async function example5_SubgraphExtraction() {
  console.log('\n=== Example 5: Subgraph Extraction ===\n');

  const kg = await example1_LoadGraphML();

  // Extract NR/5G subgraph (TS 38.xxx)
  console.log('Extract TS 38.xxx (NR/5G) subgraph:');
  const nrSubgraph = kg.getSubgraphBySeries('38');
  const nrStats = nrSubgraph.getStats();
  console.log(`  Nodes: ${nrStats.nodeCount}`);
  console.log(`  Edges: ${nrStats.edgeCount}`);

  // List nodes in subgraph
  console.log('\n  Nodes in NR subgraph:');
  Array.from(nrSubgraph.nodes.values()).forEach(node => {
    console.log(`    - ${node.label} (${node.type})`);
  });
}

// ============================================================================
// Example 6: Integration with SpecMetadataStore
// ============================================================================

/**
 * Demonstrate integration with existing SpecMetadataStore
 */
export async function example6_IntegrationWithSpecMetadata() {
  console.log('\n=== Example 6: Integration with SpecMetadataStore ===\n');

  const kg = await example1_LoadGraphML();

  console.log('Knowledge Graph can be integrated with SpecMetadataStore:');
  console.log('');
  console.log('1. Use GraphML parser to extract spec structure and relationships');
  console.log('2. Use SpecMetadataStore for metadata storage and retrieval');
  console.log('3. Combine both for comprehensive 3GPP knowledge management');
  console.log('');
  console.log('Example workflow:');
  console.log('  a) Load GraphML knowledge graph (entities & relationships)');
  console.log('  b) Load spec metadata (versions, dependencies, content)');
  console.log('  c) Cross-reference using spec numbers');
  console.log('  d) Index both with ruvector for semantic search');
  console.log('  e) Use in LLM Council for 3GPP compliance validation');

  // Example integration pattern
  console.log('\n--- Integration Pattern ---');
  console.log('/*');
  console.log('import { SpecMetadataStore } from "./spec-metadata.js";');
  console.log('import { ThreeGPPKnowledgeGraph } from "./graphml-parser.js";');
  console.log('');
  console.log('const metadataStore = new SpecMetadataStore();');
  console.log('const knowledgeGraph = new ThreeGPPKnowledgeGraph();');
  console.log('');
  console.log('// Load both sources');
  console.log('await knowledgeGraph.loadFromGraphML("./3gpp-kg.graphml");');
  console.log('await metadataStore.loadFromDataset("./3gpp-metadata.json");');
  console.log('');
  console.log('// Cross-reference');
  console.log('const spec = knowledgeGraph.findSpec("TS38331");');
  console.log('const metadata = await metadataStore.getSpec("TS 38.331");');
  console.log('');
  console.log('// Get complete view');
  console.log('const completeView = {');
  console.log('  structure: knowledgeGraph.findRelated(spec.id, 3),');
  console.log('  metadata: metadata,');
  console.log('  sections: await metadataStore.getSections("TS 38.331")');
  console.log('};');
  console.log('*/');
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * Run all examples
 */
export async function runAllExamples() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  3GPP Knowledge Graph GraphML Parser - Examples             ║');
  console.log('║  TITAN RAN v7.0 - Neuro-Symbolic Platform                   ║');
  console.log('╚══════════════════════════════════════════════════════════════╝');

  try {
    await example1_LoadGraphML();
    await example2_QueryGraph();
    await example3_ExtractEntities();
    await example4_VectorIndexing();
    await example5_SubgraphExtraction();
    await example6_IntegrationWithSpecMetadata();

    console.log('\n✅ All examples completed successfully!\n');
  } catch (error) {
    console.error('\n❌ Error running examples:', error);
    throw error;
  }
}

// Run if executed directly (CommonJS compatible check)
const isMainModule = require.main === module || (typeof process !== 'undefined' && process.argv[1] && process.argv[1].endsWith('graphml-example.ts'));
if (isMainModule) {
  runAllExamples().catch(console.error);
}
