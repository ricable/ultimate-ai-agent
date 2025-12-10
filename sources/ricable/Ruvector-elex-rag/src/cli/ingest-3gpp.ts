#!/usr/bin/env npx tsx
/**
 * 3GPP MOM XML Ingestion CLI
 *
 * Usage: npx tsx src/cli/ingest-3gpp.ts [directory-path]
 *
 * Ingests Ericsson 3GPP Managed Object Model XML files into the
 * self-learning RAG system.
 */

import { promises as fs } from 'fs';
import * as path from 'path';
import { RAGPipeline } from '../rag/rag-pipeline.js';
import { ThreeGPPParser } from '../parsers/threegpp-parser.js';
import { logger } from '../utils/logger.js';
import { getConfig } from '../core/config.js';

async function main() {
  const args = process.argv.slice(2);
  const config = getConfig();

  let dirPath = args[0] || config.paths.threeGppData;
  dirPath = path.isAbsolute(dirPath) ? dirPath : path.resolve(process.cwd(), dirPath);

  // Validate directory
  try {
    const stat = await fs.stat(dirPath);
    if (!stat.isDirectory()) {
      console.error(`Not a directory: ${dirPath}`);
      process.exit(1);
    }
  } catch {
    console.error(`Directory not found: ${dirPath}`);
    console.log(`
RuVector 3GPP MOM Ingestion CLI
===============================

Usage: npx tsx src/cli/ingest-3gpp.ts [directory-path]

Please provide a directory containing 3GPP MOM XML files.

Default path: ${config.paths.threeGppData}

Example:
  npx tsx src/cli/ingest-3gpp.ts ./data/3gpp
`);
    process.exit(1);
  }

  // Check for XML files
  const files = await fs.readdir(dirPath);
  const xmlFiles = files.filter((f) => f.toLowerCase().endsWith('.xml'));

  if (xmlFiles.length === 0) {
    console.error(`No XML files found in: ${dirPath}`);
    process.exit(1);
  }

  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║           RuVector 3GPP MOM XML Ingestion                        ║
╠══════════════════════════════════════════════════════════════════╣
║  Self-Learning RAG for Ericsson 3GPP Managed Object Models       ║
╚══════════════════════════════════════════════════════════════════╝
`);

  console.log(`Processing ${xmlFiles.length} XML file(s) from ${dirPath}:\n`);
  xmlFiles.slice(0, 10).forEach((f, i) => console.log(`  ${i + 1}. ${f}`));
  if (xmlFiles.length > 10) {
    console.log(`  ... and ${xmlFiles.length - 10} more files`);
  }
  console.log('');

  // Initialize RAG pipeline
  console.log('Initializing RAG pipeline...');
  const ragPipeline = new RAGPipeline();
  await ragPipeline.initialize();

  // Also parse MOMs for statistics
  const parser = new ThreeGPPParser();

  // Process files
  const startTime = Date.now();
  let totalChunks = 0;
  let totalClasses = 0;
  let totalAttributes = 0;

  try {
    // First, get MOM statistics
    const moms = await parser.parseDirectory(dirPath);
    for (const mom of moms) {
      totalClasses += mom.classes.size;
      for (const cls of mom.classes.values()) {
        totalAttributes += cls.attributes.length;
      }
    }

    // Then ingest into RAG
    totalChunks = await ragPipeline.ingest3GPP(dirPath);
  } catch (error) {
    console.error('Ingestion failed:', (error as Error).message);
    process.exit(1);
  }

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);

  // Persist
  await ragPipeline.persist();

  // Print summary
  const stats = ragPipeline.getStats();

  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                       Ingestion Complete                         ║
╠══════════════════════════════════════════════════════════════════╣
║  3GPP MOM Statistics:                                            ║
║    XML Files Processed: ${String(xmlFiles.length).padEnd(39)}║
║    MO Classes Parsed: ${String(totalClasses).padEnd(41)}║
║    Attributes Extracted: ${String(totalAttributes).padEnd(38)}║
╠══════════════════════════════════════════════════════════════════╣
║  RAG Statistics:                                                 ║
║    Chunks Created: ${String(totalChunks).padEnd(44)}║
║    Processing Time: ${String(duration + 's').padEnd(43)}║
╠══════════════════════════════════════════════════════════════════╣
║  Vector Store Statistics:                                        ║
║    Total Chunks: ${String(stats.totalChunks).padEnd(46)}║
║    By Document Type:                                             ║`);

  for (const [type, count] of Object.entries(stats.byDocumentType)) {
    console.log(`║      ${type}: ${String(count).padEnd(50 - type.length)}║`);
  }

  console.log(`╚══════════════════════════════════════════════════════════════════╝
`);

  logger.info('3GPP ingestion completed', {
    filesProcessed: xmlFiles.length,
    classesExtracted: totalClasses,
    attributesExtracted: totalAttributes,
    chunksIngested: totalChunks,
    durationSeconds: parseFloat(duration),
  });
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
