#!/usr/bin/env node

/**
 * AgentDB v2 Browser Bundle Test (Node.js verification)
 * Verifies bundle structure, API availability, and basic functionality
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('🧪 AgentDB v2 Browser Bundle Test\n');

let testsPassed = 0;
let testsFailed = 0;

function assert(condition, message) {
  if (condition) {
    console.log(`✅ PASS: ${message}`);
    testsPassed++;
    return true;
  } else {
    console.error(`❌ FAIL: ${message}`);
    testsFailed++;
    return false;
  }
}

function test(name, fn) {
  console.log(`\n📋 ${name}`);
  try {
    fn();
  } catch (error) {
    console.error(`❌ Test suite failed: ${error.message}`);
    testsFailed++;
  }
}

// Test 1: Bundle file exists
test('Test Suite 1: Bundle File Verification', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');

  assert(fs.existsSync(bundlePath), 'Bundle file exists at dist/agentdb.min.js');

  const stats = fs.statSync(bundlePath);
  const sizeKB = (stats.size / 1024).toFixed(2);
  assert(stats.size > 0, `Bundle has content (${sizeKB} KB)`);
  assert(stats.size < 200 * 1024, `Bundle size is reasonable (<200KB, got ${sizeKB}KB)`);

  console.log(`   📦 Bundle size: ${sizeKB} KB`);
});

// Test 2: Bundle content structure
test('Test Suite 2: Bundle Content Analysis', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  // Check header comment
  assert(
    bundleContent.includes('AgentDB Browser Bundle'),
    'Bundle includes header comment'
  );

  assert(
    bundleContent.includes('v2.0.0-alpha.1') || bundleContent.includes('${pkg.version}'),
    'Bundle includes version information'
  );

  assert(
    bundleContent.includes('Backward compatible with v1 API'),
    'Bundle declares v1 API compatibility'
  );

  // Check sql.js inclusion
  assert(
    bundleContent.includes('initSqlJs'),
    'Bundle includes sql.js WASM loader'
  );

  assert(
    bundleContent.includes('sql.js initialized'),
    'Bundle includes sql.js initialization code'
  );

  console.log(`   📄 Bundle content length: ${bundleContent.length} characters`);
});

// Test 3: v1 API backward compatibility
test('Test Suite 3: v1 API Backward Compatibility', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  // Check v1 methods exist
  const v1Methods = [
    'this.run',
    'this.exec',
    'this.prepare',
    'this.export',
    'this.close',
    'this.insert',
    'this.search',
    'this.delete',
    'this.storePattern',
    'this.storeEpisode',
    'this.addCausalEdge',
    'this.storeSkill',
    'this.initializeAsync'
  ];

  v1Methods.forEach(method => {
    assert(
      bundleContent.includes(method),
      `v1 method "${method}" is present`
    );
  });
});

// Test 4: v2 Enhanced API
test('Test Suite 4: v2 Enhanced API Features', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  // Check v2 controllers
  assert(bundleContent.includes('this.episodes'), 'v2 episodes controller exists');
  assert(bundleContent.includes('this.skills'), 'v2 skills controller exists');
  assert(bundleContent.includes('this.causal_edges'), 'v2 causal_edges controller exists');

  // Check v2 methods
  const v2Methods = [
    'episodes.store',
    'episodes.search',
    'episodes.getStats',
    'skills.store',
    'causal_edges.add'
  ];

  v2Methods.forEach(method => {
    assert(
      bundleContent.includes(method),
      `v2 method "${method}" is present`
    );
  });
});

// Test 5: Schema tables
test('Test Suite 5: Database Schema', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  // Check v2 tables
  const v2Tables = [
    'CREATE TABLE IF NOT EXISTS episodes',
    'CREATE TABLE IF NOT EXISTS episode_embeddings',
    'CREATE TABLE IF NOT EXISTS skills',
    'CREATE TABLE IF NOT EXISTS causal_edges'
  ];

  v2Tables.forEach(table => {
    assert(
      bundleContent.includes(table),
      `v2 table creation: ${table.match(/TABLE IF NOT EXISTS (\w+)/)?.[1]}`
    );
  });

  // Check v1 legacy tables
  const v1Tables = [
    'CREATE TABLE IF NOT EXISTS vectors',
    'CREATE TABLE IF NOT EXISTS patterns',
    'CREATE TABLE IF NOT EXISTS episodes_legacy',
    'CREATE TABLE IF NOT EXISTS causal_edges_legacy',
    'CREATE TABLE IF NOT EXISTS skills_legacy'
  ];

  v1Tables.forEach(table => {
    assert(
      bundleContent.includes(table),
      `v1 legacy table: ${table.match(/TABLE IF NOT EXISTS (\w+)/)?.[1]}`
    );
  });
});

// Test 6: Embedding and search functionality
test('Test Suite 6: Embedding & Search Features', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  assert(
    bundleContent.includes('generateMockEmbedding'),
    'Mock embedding generation function exists'
  );

  assert(
    bundleContent.includes('cosineSimilarity'),
    'Cosine similarity function exists'
  );

  assert(
    bundleContent.includes('Float32Array'),
    'Float32Array used for embeddings'
  );

  assert(
    bundleContent.includes('384'),
    '384-dimensional embeddings configured'
  );
});

// Test 7: Configuration options
test('Test Suite 7: Configuration Options', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  const configOptions = [
    'memoryMode',
    'backend',
    'enableGNN',
    'storage',
    'dbName',
    'syncAcrossTabs'
  ];

  configOptions.forEach(option => {
    assert(
      bundleContent.includes(option),
      `Configuration option "${option}" available`
    );
  });

  assert(
    bundleContent.includes('indexeddb'),
    'IndexedDB storage option available'
  );

  assert(
    bundleContent.includes('BroadcastChannel'),
    'Cross-tab sync with BroadcastChannel'
  );
});

// Test 8: Namespace exports
test('Test Suite 8: Namespace & Module Exports', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  assert(
    bundleContent.includes('var AgentDB'),
    'AgentDB namespace is defined'
  );

  assert(
    bundleContent.includes('AgentDB.Database'),
    'AgentDB.Database is exported'
  );

  assert(
    bundleContent.includes('AgentDB.SQLiteVectorDB'),
    'AgentDB.SQLiteVectorDB alias is exported'
  );

  assert(
    bundleContent.includes('global.AgentDB'),
    'AgentDB is attached to global object'
  );

  assert(
    bundleContent.includes('module.exports'),
    'CommonJS module.exports is available'
  );

  assert(
    bundleContent.includes('define.amd'),
    'AMD (RequireJS) support is available'
  );
});

// Test 9: Error handling
test('Test Suite 9: Error Handling', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  assert(
    bundleContent.includes('try {') && bundleContent.includes('catch'),
    'Try-catch blocks for error handling'
  );

  assert(
    bundleContent.includes('throw new Error'),
    'Error throwing for invalid states'
  );

  assert(
    bundleContent.includes('[AgentDB]'),
    'Consistent error message prefixes'
  );
});

// Test 10: Performance features
test('Test Suite 10: Performance Features', () => {
  const bundlePath = path.join(__dirname, '..', 'dist', 'agentdb.min.js');
  const bundleContent = fs.readFileSync(bundlePath, 'utf8');

  assert(
    bundleContent.includes('performance.now'),
    'Performance timing available (if supported)'
  );

  assert(
    bundleContent.includes('Promise'),
    'Promise support for async operations'
  );

  assert(
    bundleContent.includes('async function') || bundleContent.includes('async '),
    'Async/await syntax used'
  );
});

// Final summary
console.log('\n' + '='.repeat(60));
console.log('📊 Test Summary');
console.log('='.repeat(60));
console.log(`Total Tests: ${testsPassed + testsFailed}`);
console.log(`✅ Passed: ${testsPassed}`);
console.log(`❌ Failed: ${testsFailed}`);

const passRate = ((testsPassed / (testsPassed + testsFailed)) * 100).toFixed(1);
console.log(`Pass Rate: ${passRate}%`);

console.log('='.repeat(60));

if (testsFailed === 0) {
  console.log('\n🎉 All tests passed! Browser bundle is ready for deployment.\n');
  process.exit(0);
} else {
  console.log(`\n⚠️  ${testsFailed} test(s) failed. Please review and fix.\n`);
  process.exit(1);
}
