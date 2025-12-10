/**
 * Example: 3GPP Spec Metadata Integration
 *
 * Demonstrates how to use SpecMetadataStore and DatasetLoader
 * to index and query 3GPP specifications with AgentDB.
 *
 * @module knowledge/example
 * @version 7.0.0-alpha.1
 */

import { SpecMetadataStore } from './spec-metadata.js';
import { DatasetLoader } from './dataset-loader.js';

// ============================================================================
// Example 1: Basic Setup and Loading
// ============================================================================

async function example1_basicSetup() {
  console.log('\n=== Example 1: Basic Setup ===\n');

  // Initialize the metadata store
  const store = new SpecMetadataStore('./titan-ran.db');
  await store.initialize();

  // Create dataset loader
  const loader = new DatasetLoader(store, {
    batchSize: 50,
    validate: true,
    skipInvalid: true
  });

  // Load from HuggingFace
  const stats = await loader.loadFromHuggingFace({
    id: 'OrganizedProgrammers/3GPPSpecMetadata',
    format: 'json',
    huggingFaceName: 'OrganizedProgrammers/3GPPSpecMetadata',
    split: 'train'
  });

  console.log('Dataset Statistics:', stats);
  console.log(`Loaded ${stats.uniqueSpecs} specs in ${stats.loadingTime.toFixed(2)}ms`);
}

// ============================================================================
// Example 2: Semantic Search
// ============================================================================

async function example2_semanticSearch() {
  console.log('\n=== Example 2: Semantic Search ===\n');

  const store = new SpecMetadataStore('./titan-ran.db');
  await store.initialize();

  // Find specs related to RRC configuration
  const results = await store.findRelevantSpecs(
    'Radio Resource Control configuration and procedures',
    5,
    {
      release: 'Rel-17',
      status: 'active',
      domain: 'RAN'
    }
  );

  console.log(`Found ${results.length} relevant specs:`);
  for (const result of results) {
    if (result.type === 'spec') {
      const spec = result.item;
      console.log(`- ${spec.specNumber}: ${spec.title}`);
      console.log(`  Similarity: ${result.similarity.toFixed(3)}`);
      console.log(`  Release: ${spec.release}, WG: ${spec.workingGroup}`);
      console.log();
    }
  }
}

// ============================================================================
// Example 3: Section-Level Search for RAG
// ============================================================================

async function example3_sectionSearch() {
  console.log('\n=== Example 3: Section-Level Search (RAG) ===\n');

  const store = new SpecMetadataStore('./titan-ran.db');
  await store.initialize();

  // Find specific sections about power control
  const sections = await store.findRelevantSections(
    'uplink power control P0 alpha parameters',
    'TS 38.213',
    3
  );

  console.log(`Found ${sections.length} relevant sections:`);
  for (const result of sections) {
    if (result.type === 'section') {
      const section = result.item;
      console.log(`- Section ${section.sectionNumber}: ${section.title}`);
      console.log(`  Similarity: ${result.similarity.toFixed(3)}`);
      console.log(`  Excerpt: ${result.excerpt}`);
      console.log();
    }
  }
}

// ============================================================================
// Example 4: Dependency Graph Queries
// ============================================================================

async function example4_dependencyGraph() {
  console.log('\n=== Example 4: Dependency Graph ===\n');

  const store = new SpecMetadataStore('./titan-ran.db');
  await store.initialize();

  // Get dependency tree for TS 38.331 (RRC spec)
  const dependencies = await store.getDependencyTree('TS 38.331', 2);

  console.log('Dependency tree for TS 38.331:');
  for (const spec of dependencies) {
    console.log(`- ${spec.specNumber}: ${spec.title}`);
    console.log(`  Dependencies: ${spec.dependencies.join(', ') || 'none'}`);
  }

  console.log();

  // Get specs that reference TS 38.213
  const referencing = await store.getReferencingSpecs('TS 38.213');

  console.log('\nSpecs that reference TS 38.213:');
  for (const spec of referencing) {
    console.log(`- ${spec.specNumber}: ${spec.title}`);
  }
}

// ============================================================================
// Example 5: Progress Tracking
// ============================================================================

async function example5_progressTracking() {
  console.log('\n=== Example 5: Progress Tracking ===\n');

  const store = new SpecMetadataStore('./titan-ran.db');
  await store.initialize();

  const loader = new DatasetLoader(store);

  // Listen to progress events
  loader.on('progress', (progress) => {
    console.log(
      `Progress: ${progress.processed}/${progress.total} ` +
      `(${progress.percentage.toFixed(1)}%) - ` +
      `Indexed: ${progress.indexed}, Failed: ${progress.failed}`
    );

    if (progress.estimatedTimeRemaining) {
      const eta = (progress.estimatedTimeRemaining / 1000).toFixed(1);
      console.log(`  ETA: ${eta}s`);
    }
  });

  loader.on('loading_complete', ({ stats }) => {
    console.log('\nLoading complete!');
    console.log(`Total specs: ${stats.uniqueSpecs}`);
    console.log(`Total sections: ${stats.totalSections}`);
    console.log(`Average dependencies: ${stats.avgDependencies.toFixed(1)}`);
  });

  // Load dataset
  await loader.loadFromHuggingFace({
    id: 'OrganizedProgrammers/3GPPSpecMetadata',
    format: 'json',
    huggingFaceName: 'OrganizedProgrammers/3GPPSpecMetadata'
  });
}

// ============================================================================
// Example 6: Validation and Error Handling
// ============================================================================

async function example6_validation() {
  console.log('\n=== Example 6: Validation ===\n');

  const store = new SpecMetadataStore('./titan-ran.db');
  await store.initialize();

  const loader = new DatasetLoader(store, {
    validate: true,
    skipInvalid: true
  });

  // Load dataset
  await loader.loadFromJSON(JSON.stringify([
    {
      spec_number: 'TS 38.331',
      version: '17.4.0',
      title: 'Valid spec'
    },
    {
      // Missing spec_number - will fail validation
      version: '17.4.0',
      title: 'Invalid spec'
    }
  ]));

  // Check validation errors
  const errors = loader.getValidationErrors();

  console.log(`Validation errors: ${errors.length}`);
  for (const error of errors) {
    console.log(`- [${error.severity}] ${error.recordId}.${error.field}: ${error.message}`);
  }
}

// ============================================================================
// Example 7: Bulk Indexing with Custom Data
// ============================================================================

async function example7_bulkIndexing() {
  console.log('\n=== Example 7: Bulk Indexing ===\n');

  const store = new SpecMetadataStore('./titan-ran.db');
  await store.initialize();

  // Create custom specs
  const specs = [
    {
      specNumber: 'TS 38.331',
      version: '17.4.0',
      release: 'Rel-17',
      title: 'NR; Radio Resource Control (RRC)',
      workingGroup: 'RAN2',
      status: 'active' as const,
      scope: 'RRC protocol specification',
      keywords: ['RRC', 'NR', '5G'],
      dependencies: ['TS 38.300'],
      lastUpdate: new Date()
    },
    {
      specNumber: 'TS 28.552',
      version: '17.3.0',
      release: 'Rel-17',
      title: '5G Performance Measurements',
      workingGroup: 'SA5',
      status: 'active' as const,
      scope: 'Performance measurements for 5G',
      keywords: ['KPI', 'performance'],
      dependencies: ['TS 28.550'],
      lastUpdate: new Date()
    }
  ];

  // Bulk index
  await store.bulkIndexSpecs(specs);

  console.log('Bulk indexing complete');

  // Query statistics
  const stats = store.getStats();
  console.log(`Indexed specs: ${stats.specsCount}`);
  console.log(`Average search latency: ${stats.avgSearchLatency.toFixed(2)}ms`);
}

// ============================================================================
// Example 8: Integration with Titan Council
// ============================================================================

async function example8_titanIntegration() {
  console.log('\n=== Example 8: Titan Council Integration ===\n');

  const store = new SpecMetadataStore('./titan-ran.db');
  await store.initialize();

  // Scenario: Council needs to validate a P0 optimization against 3GPP specs
  const query = 'uplink power control P0 parameter range and constraints for NR';

  const relevantSections = await store.findRelevantSections(query, undefined, 3);

  console.log('3GPP Compliance Check for P0 Optimization:');
  console.log(`Query: "${query}"`);
  console.log();

  for (const result of relevantSections) {
    if (result.type === 'section') {
      const section = result.item;
      console.log(`Relevant Spec: ${section.specNumber}`);
      console.log(`Section: ${section.sectionNumber} - ${section.title}`);
      console.log(`Confidence: ${result.similarity.toFixed(3)}`);
      console.log(`Content excerpt: ${result.excerpt}`);
      console.log();
    }
  }

  // Council can use these sections for validation
  console.log('Council Decision: Use these sections to validate P0 is within [-202, -60] dBm range');
}

// ============================================================================
// Run Examples
// ============================================================================

async function runAllExamples() {
  try {
    await example1_basicSetup();
    await example2_semanticSearch();
    await example3_sectionSearch();
    await example4_dependencyGraph();
    await example5_progressTracking();
    await example6_validation();
    await example7_bulkIndexing();
    await example8_titanIntegration();

    console.log('\n=== All Examples Completed Successfully ===\n');
  } catch (error) {
    console.error('Error running examples:', error);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAllExamples();
}

// Export for use in other modules
export {
  example1_basicSetup,
  example2_semanticSearch,
  example3_sectionSearch,
  example4_dependencyGraph,
  example5_progressTracking,
  example6_validation,
  example7_bulkIndexing,
  example8_titanIntegration
};
