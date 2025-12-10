#!/usr/bin/env node

/**
 * Test lib/index.js exports
 */

import * as exports from './lib/index.js';
import defaultExport from './lib/index.js';

console.log('🧪 Testing lib/index.js exports...\n');

console.log('✅ Named exports:', Object.keys(exports).filter(k => !k.startsWith('_') && k !== 'default').sort().join(', '));
console.log('\n✅ Default export:', Object.keys(defaultExport).sort().join(', '));
console.log('\n✅ Version:', exports.VERSION);
console.log('✅ Package:', exports.PACKAGE_NAME);

// Test key functions exist
const requiredExports = [
  'initDatabase',
  'createResearchJob',
  'getJobStatus',
  'listJobs',
  'storeResearchPattern',
  'searchSimilarPatterns',
  'searchSimilarVectors'
  // Note: startMCPServer is available via subpath: import from 'research-swarm/mcp'
];

let allPresent = true;
console.log('\n🔍 Checking required exports:');
for (const name of requiredExports) {
  const present = name in exports;
  console.log(`   ${present ? '✅' : '❌'} ${name}`);
  if (!present) allPresent = false;
}

if (allPresent) {
  console.log('\n✅ All exports validated successfully!');
  process.exit(0);
} else {
  console.log('\n❌ Some exports are missing!');
  process.exit(1);
}
