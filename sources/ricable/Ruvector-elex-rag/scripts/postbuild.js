#!/usr/bin/env node
/**
 * Post-build script for RuVector Telecom RAG
 */

import { promises as fs } from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');

async function main() {
  console.log('Running post-build tasks...');

  // Ensure data directories exist
  const dataDirs = [
    'data/elex',
    'data/3gpp',
    'data/processed',
    'data/vector_store',
    'data/graph_store',
    'logs',
  ];

  for (const dir of dataDirs) {
    const fullPath = path.join(rootDir, dir);
    await fs.mkdir(fullPath, { recursive: true });
    console.log(`  Created: ${dir}`);
  }

  // Copy .env.example if .env doesn't exist
  const envPath = path.join(rootDir, '.env');
  const envExamplePath = path.join(rootDir, '.env.example');

  try {
    await fs.access(envPath);
  } catch {
    try {
      await fs.copyFile(envExamplePath, envPath);
      console.log('  Created .env from .env.example');
    } catch {
      console.log('  Note: .env.example not found, skipping');
    }
  }

  console.log('Post-build complete!');
}

main().catch(console.error);
