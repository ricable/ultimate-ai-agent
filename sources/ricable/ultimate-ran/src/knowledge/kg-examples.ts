/**
 * Knowledge Graph Query Interface - Usage Examples
 *
 * Demonstrates all query patterns with real 3GPP spec scenarios:
 * - Natural language queries
 * - Cypher-like structured queries
 * - SPARQL-like queries
 * - Graph traversal
 * - Cross-spec analysis
 *
 * @module knowledge/examples
 * @version 7.0.0-alpha.1
 */

import {
  createGraphMLKnowledgeGraph,
  type KnowledgeGraphAgent,
  type QueryResult,
  ThreeGPPKnowledgeAgent
} from './index.js';

// ============================================================================
// Example 1: Natural Language Queries
// ============================================================================

async function exampleNaturalLanguageQueries() {
  console.log('\n' + '='.repeat(80));
  console.log('Example 1: Natural Language Queries');
  console.log('='.repeat(80) + '\n');

  const kg = await createGraphMLKnowledgeGraph({ loadSampleData: true });
  await kg.initialize();

  // Query 1: Find parameters
  console.log('Query: "What parameters control uplink power in 5G NR?"\n');
  const result1 = await kg.query.query("What parameters control uplink power in 5G NR?");
  console.log('Answer:', result1.answer);
  console.log('Confidence:', result1.confidence);
  console.log('Found nodes:', result1.nodes.length);
  console.log('Sources:', result1.sourceSpecs.join(', '));
  console.log();

  // Query 2: Relationship query
  console.log('Query: "How does P0 relate to SINR?"\n');
  const result2 = await kg.query.query("How does P0 relate to SINR?");
  console.log('Answer:', result2.answer);
  console.log('Paths found:', result2.paths?.length || 0);
  if (result2.paths && result2.paths.length > 0) {
    console.log('Shortest path:');
    for (const node of result2.paths[0].nodes) {
      console.log(`  - ${node.label} (${node.type})`);
    }
  }
  console.log();

  // Query 3: List IEs
  console.log('Query: "List all IEs in RRCReconfiguration"\n');
  const result3 = await kg.query.query("List all IEs in RRCReconfiguration");
  console.log('Answer:', result3.answer);
  console.log('IEs found:', result3.nodes.filter((n: any) => n.type === 'ie').length);
  console.log();

  // Query 4: Cross-spec comparison
  console.log('Query: "Compare LTE and NR power control"\n');
  const result4 = await kg.query.query("Compare LTE and NR power control");
  console.log('Answer:', result4.answer);
  console.log('Execution time:', result4.executionTime.toFixed(2), 'ms');
  console.log();
}

// ============================================================================
// Example 2: Structured Queries (Cypher-like)
// ============================================================================

async function exampleCypherQueries() {
  console.log('\n' + '='.repeat(80));
  console.log('Example 2: Cypher-like Structured Queries');
  console.log('='.repeat(80) + '\n');

  const kg = await createGraphMLKnowledgeGraph({ loadSampleData: true });
  await kg.initialize();

  // Cypher query 1: Find parameters that control metrics
  console.log('Cypher: MATCH (p:Parameter)-[:CONTROLS]->(m:Metric) RETURN p, m\n');
  const cypher1 = await kg.query.cypher(
    "MATCH (p:Parameter)-[:CONTROLS]->(m:Metric) WHERE p.name = 'P0' RETURN p, m"
  );
  console.log('Matches:', cypher1.totalMatches);
  console.log('Execution time:', cypher1.executionTime.toFixed(2), 'ms');
  console.log();

  // Cypher query 2: Find spec contents
  console.log('Cypher: MATCH (s:Spec)-[:CONTAINS]->(sec:Section) RETURN sec\n');
  const cypher2 = await kg.query.cypher(
    "MATCH (s:Spec)-[:CONTAINS]->(sec:Section) WHERE s.id = 'TS-38.331' RETURN sec"
  );
  console.log('Sections found:', cypher2.totalMatches);
  console.log();
}

// ============================================================================
// Example 3: Graph Traversal
// ============================================================================

async function exampleGraphTraversal() {
  console.log('\n' + '='.repeat(80));
  console.log('Example 3: Graph Traversal Patterns');
  console.log('='.repeat(80) + '\n');

  const kg = await createGraphMLKnowledgeGraph({ loadSampleData: true });
  await kg.initialize();

  // Traversal 1: Outgoing edges (what does P0 control?)
  console.log('Traversal: Starting from P0-PUSCH, find what it controls\n');
  const traversal1 = await kg.query.traverse('param-p0-pusch', {
    direction: 'outgoing',
    edgeTypes: ['controls', 'affects'],
    maxDepth: 2,
    filter: (node: any) => node.type === 'parameter' || node.type === 'concept'
  });
  console.log('Nodes found:', traversal1.length);
  traversal1.slice(0, 5).forEach((node: any) => {
    console.log(`  - ${node.label} (${node.type})`);
  });
  console.log();

  // Traversal 2: Bidirectional (full context)
  console.log('Traversal: Full context around SINR (both directions)\n');
  const traversal2 = await kg.query.traverse('metric-sinr', {
    direction: 'both',
    maxDepth: 1,
    collectPaths: true
  });
  console.log('Related nodes:', traversal2.length);
  traversal2.slice(0, 5).forEach((node: any) => {
    console.log(`  - ${node.label} (${node.type}) from ${node.metadata.source}`);
  });
  console.log();

  // Traversal 3: Filtered by spec
  console.log('Traversal: Find all parameters in TS 38.213\n');
  const traversal3 = await kg.query.traverse('spec-ts38213', {
    direction: 'outgoing',
    edgeTypes: ['contains'],
    maxDepth: 3,
    filter: (node: any) => node.type === 'parameter'
  });
  console.log('Parameters found:', traversal3.length);
  console.log();
}

// ============================================================================
// Example 4: RuvLLM Integration
// ============================================================================

async function exampleRuvLLMIntegration() {
  console.log('\n' + '='.repeat(80));
  console.log('Example 4: RuvLLM Semantic Understanding');
  console.log('='.repeat(80) + '\n');

  const kg = await createGraphMLKnowledgeGraph({ loadSampleData: true });
  await kg.initialize();

  // Complex semantic query
  console.log('Complex Query: Explain the relationship between power control and interference\n');
  const result = await kg.query.query(
    "Explain the relationship between uplink power control parameters and interference management in 5G NR"
  );

  console.log('Answer:');
  console.log(result.answer);
  console.log();

  console.log('Reasoning:');
  console.log(result.reasoning);
  console.log();

  console.log('Graph Analysis:');
  console.log(`  - Found ${result.nodes.length} relevant nodes`);
  console.log(`  - ${result.edges.length} relationships`);
  console.log(`  - Confidence: ${(result.confidence * 100).toFixed(1)}%`);
  console.log(`  - Sources: ${result.sourceSpecs.join(', ')}`);
  console.log();

  // Path explanation
  if (result.paths && result.paths.length > 0) {
    console.log('Explanation of key path:');
    const path = result.paths[0];
    // Explain path would use RuvLLM in production
    console.log(`  Path with ${path.nodes.length} steps, weight: ${path.weight}`);
    console.log('  ', path.nodes.map((n: any) => n.label).join(' → '));
  }
  console.log();
}

// ============================================================================
// Example 5: Agentic-Flow Integration
// ============================================================================

async function exampleAgenticFlowIntegration() {
  console.log('\n' + '='.repeat(80));
  console.log('Example 5: Agentic-Flow Knowledge Graph Agent');
  console.log('='.repeat(80) + '\n');

  // Create knowledge graph agent using ThreeGPPKnowledgeAgent
  const agent = new ThreeGPPKnowledgeAgent('kg-specialist');

  // Listen to events
  agent.on('initialized', (stats: any) => {
    console.log('Agent initialized with stats:', stats);
  });

  agent.on('query-complete', (event: any) => {
    console.log(`Query complete: ${event.nodeCount} nodes, confidence: ${event.confidence}`);
  });

  // Initialize agent
  await agent.initialize({
    loadSampleData: true,
    specs: ['TS 38.331', 'TS 38.213', 'TS 28.552']
  });
  console.log();

  // Process queries through agent
  console.log('Agent Query: "What is P0-PUSCH?"\n');
  const response1 = await agent.processQuery("What is P0-PUSCH?");
  console.log('Answer:', response1.answer);
  console.log('Details:', response1.details);
  console.log();

  // Traversal through agent
  console.log('Agent Traversal: From P0-PUSCH\n');
  const traversalResult = await agent.traverse('param-p0-pusch', {
    direction: 'outgoing',
    maxDepth: 2
  });
  console.log('Traversal result:', traversalResult.nodeCount, 'nodes');
  console.log();

  // Get stats
  const stats = agent.getStats();
  console.log('Agent Stats:', stats);
  console.log();
}

// ============================================================================
// Example 6: Real-World Scenario - RAN Optimization
// ============================================================================

async function exampleRANOptimizationScenario() {
  console.log('\n' + '='.repeat(80));
  console.log('Example 6: Real-World RAN Optimization Scenario');
  console.log('='.repeat(80) + '\n');

  const kg = await createGraphMLKnowledgeGraph({ loadSampleData: true });
  await kg.initialize();

  console.log('Scenario: Engineer needs to optimize uplink power for poor SINR\n');

  // Step 1: Understand the problem
  console.log('Step 1: What parameters affect SINR?\n');
  const step1 = await kg.query.query("What parameters affect SINR in uplink?");
  console.log('Found parameters:', step1.nodes.filter((n: any) => n.type === 'parameter').length);
  console.log();

  // Step 2: Focus on power control
  console.log('Step 2: How does P0-PUSCH control uplink power?\n');
  const step2 = await kg.query.query("How does P0-PUSCH control uplink power?");
  console.log('Answer:', step2.answer);
  console.log();

  // Step 3: Check constraints
  console.log('Step 3: What are the constraints on P0-PUSCH?\n');
  const step3 = await kg.query.query("What are the valid range and constraints for P0-PUSCH?");
  console.log('Answer:', step3.answer);
  console.log();

  // Step 4: Find related procedures
  console.log('Step 4: What procedures use P0-PUSCH?\n');
  const step4 = await kg.query.traverse('param-p0-pusch', {
    direction: 'incoming',
    edgeTypes: ['uses', 'configures'],
    maxDepth: 2,
    filter: (node: any) => node.type === 'procedure'
  });
  console.log('Related procedures:', step4.length);
  console.log();

  // Step 5: Cross-reference with O1 interface
  console.log('Step 5: How does P0-PUSCH map to O1 MOM?\n');
  const step5 = await kg.query.query("Map P0-PUSCH to TS 28.552 managed object model");
  console.log('Answer:', step5.answer);
  console.log();

  console.log('Optimization Path:');
  console.log('  1. Identify parameter: P0-PUSCH');
  console.log('  2. Valid range: -202 to 24 dBm (from TS 38.213)');
  console.log('  3. Related procedures: Power control loop, RRC reconfiguration');
  console.log('  4. O1 mapping: GNBCUUPFunction.pZeroNominalPusch');
  console.log('  5. Action: Adjust P0 within constraints to improve SINR');
  console.log();
}

// ============================================================================
// Example 7: Export and Integration
// ============================================================================

async function exampleExportIntegration() {
  console.log('\n' + '='.repeat(80));
  console.log('Example 7: Export Knowledge Graph');
  console.log('='.repeat(80) + '\n');

  const kg = await createGraphMLKnowledgeGraph({ loadSampleData: true });
  await kg.initialize();

  // Export as JSON
  console.log('Exporting as JSON...');
  const jsonExport = await kg.exportGraph('json');
  console.log('JSON size:', jsonExport.length, 'bytes');
  console.log('Sample:', jsonExport.substring(0, 200) + '...');
  console.log();

  // Export as GraphML
  console.log('Exporting as GraphML...');
  const graphmlExport = await kg.exportGraph('graphml');
  console.log('GraphML size:', graphmlExport.length, 'bytes');
  console.log('Sample:', graphmlExport.substring(0, 200) + '...');
  console.log();

  // Export as Cypher
  console.log('Exporting as Cypher...');
  const cypherExport = await kg.exportGraph('cypher');
  console.log('Cypher statements:', cypherExport.split(';').length - 1);
  console.log('Sample:', cypherExport.split(';')[0] + '...');
  console.log();
}

// ============================================================================
// Run All Examples
// ============================================================================

export async function runAllExamples() {
  console.log('\n' + '█'.repeat(80));
  console.log('TITAN RAN - Knowledge Graph Query Interface Examples');
  console.log('█'.repeat(80));

  try {
    await exampleNaturalLanguageQueries();
    await exampleCypherQueries();
    await exampleGraphTraversal();
    await exampleRuvLLMIntegration();
    await exampleAgenticFlowIntegration();
    await exampleRANOptimizationScenario();
    await exampleExportIntegration();

    console.log('\n' + '█'.repeat(80));
    console.log('All examples completed successfully!');
    console.log('█'.repeat(80) + '\n');
  } catch (error) {
    console.error('Error running examples:', error);
    throw error;
  }
}

// ============================================================================
// Individual Example Exports
// ============================================================================

export {
  exampleNaturalLanguageQueries,
  exampleCypherQueries,
  exampleGraphTraversal,
  exampleRuvLLMIntegration,
  exampleAgenticFlowIntegration,
  exampleRANOptimizationScenario,
  exampleExportIntegration
};

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples().catch(console.error);
}
