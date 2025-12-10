#!/usr/bin/env npx tsx
/**
 * ELEX Documentation Ingestion CLI
 *
 * Usage: npx tsx src/cli/ingest-elex.ts <zip-file-paths...>
 *
 * Ingests Ericsson ELEX HTML documentation from ZIP files into the
 * self-learning RAG system.
 */

import { promises as fs } from 'fs';
import * as path from 'path';
import { RAGPipeline } from '../rag/rag-pipeline.js';
import { logger } from '../utils/logger.js';
import { getConfig } from '../core/config.js';

async function main() {
  const args = process.argv.slice(2);
  const config = getConfig();

  if (args.length === 0) {
    // If no args, look for ZIP files in default ELEX data path
    const elexPath = config.paths.elexData;

    try {
      const files = await fs.readdir(elexPath);
      const zipFiles = files.filter((f) => f.endsWith('.zip')).map((f) => path.join(elexPath, f));

      if (zipFiles.length === 0) {
        console.log(`
RuVector ELEX Ingestion CLI
===========================

Usage: npx tsx src/cli/ingest-elex.ts <zip-file-paths...>

No ZIP files found in ${elexPath}

Please either:
1. Provide ZIP file paths as arguments
2. Place your ELEX ZIP files in ${elexPath}

Example:
  npx tsx src/cli/ingest-elex.ts ./data/elex/elex_vol1.zip ./data/elex/elex_vol2.zip
`);
        process.exit(1);
      }

      args.push(...zipFiles);
      console.log(`Found ${zipFiles.length} ZIP files in ${elexPath}`);
    } catch {
      console.error(`Default ELEX path not accessible: ${elexPath}`);
      process.exit(1);
    }
  }

  // Validate file paths
  const validPaths: string[] = [];
  for (const arg of args) {
    const absolutePath = path.isAbsolute(arg) ? arg : path.resolve(process.cwd(), arg);

    try {
      const stat = await fs.stat(absolutePath);
      if (!stat.isFile()) {
        console.warn(`Skipping non-file: ${arg}`);
        continue;
      }
      if (!absolutePath.endsWith('.zip')) {
        console.warn(`Skipping non-ZIP file: ${arg}`);
        continue;
      }
      validPaths.push(absolutePath);
    } catch {
      console.warn(`File not found: ${arg}`);
    }
  }

  if (validPaths.length === 0) {
    console.error('No valid ZIP files to process');
    process.exit(1);
  }

  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║           RuVector ELEX Documentation Ingestion                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Self-Learning RAG for Ericsson RAN Technical Documentation      ║
╚══════════════════════════════════════════════════════════════════╝
`);

  console.log(`Processing ${validPaths.length} ZIP file(s):\n`);
  validPaths.forEach((p, i) => console.log(`  ${i + 1}. ${path.basename(p)}`));
  console.log('');

  // Initialize RAG pipeline
  console.log('Initializing RAG pipeline...');
  const ragPipeline = new RAGPipeline();
  await ragPipeline.initialize();

  // Process files
  const startTime = Date.now();
  let totalChunks = 0;

  try {
    totalChunks = await ragPipeline.ingestELEX(validPaths);
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
║  Total Chunks Ingested: ${String(totalChunks).padEnd(38)}║
║  Total Time: ${String(duration + 's').padEnd(50)}║
║  Chunks/Second: ${String((totalChunks / parseFloat(duration)).toFixed(2)).padEnd(47)}║
╠══════════════════════════════════════════════════════════════════╣
║  Vector Store Statistics:                                        ║
║    Total Chunks: ${String(stats.totalChunks).padEnd(46)}║
║    By Document Type:                                             ║`);

  for (const [type, count] of Object.entries(stats.byDocumentType)) {
    console.log(`║      ${type}: ${String(count).padEnd(50 - type.length)}║`);
  }

  console.log(`╚══════════════════════════════════════════════════════════════════╝
`);

  logger.info('ELEX ingestion completed', {
    filesProcessed: validPaths.length,
    chunksIngested: totalChunks,
    durationSeconds: parseFloat(duration),
  });
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
