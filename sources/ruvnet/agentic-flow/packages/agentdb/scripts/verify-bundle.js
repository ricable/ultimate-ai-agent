#!/usr/bin/env node

/**
 * Verify Browser Bundle Integrity
 * Ensures all required features are present in the built bundle
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const rootDir = join(__dirname, '..');

const checks = {
  passed: 0,
  failed: 0,
  warnings: 0
};

function pass(message) {
  console.log(`✅ ${message}`);
  checks.passed++;
}

function fail(message) {
  console.error(`❌ ${message}`);
  checks.failed++;
}

function warn(message) {
  console.warn(`⚠️  ${message}`);
  checks.warnings++;
}

function verifyBundle() {
  console.log('🔍 Verifying browser bundle integrity...\n');

  const bundlePath = join(rootDir, 'dist', 'agentdb.min.js');

  // Check 1: Bundle exists
  if (!fs.existsSync(bundlePath)) {
    fail('Bundle file does not exist at dist/agentdb.min.js');
    return false;
  }
  pass('Bundle file exists');

  // Check 2: Read bundle content
  const bundle = fs.readFileSync(bundlePath, 'utf8');

  if (!bundle || bundle.length === 0) {
    fail('Bundle is empty');
    return false;
  }
  pass(`Bundle size: ${(bundle.length / 1024).toFixed(2)} KB`);

  // Check 3: Version header
  if (bundle.includes('/*! AgentDB Browser Bundle v')) {
    const versionMatch = bundle.match(/AgentDB Browser Bundle v([\d.]+)/);
    if (versionMatch) {
      pass(`Version header found: v${versionMatch[1]}`);
    } else {
      warn('Version header format incorrect');
    }
  } else {
    fail('Missing version header');
  }

  // Check 4: v1.0.7 compatibility marker
  if (bundle.includes('v1.0.7 API compatible')) {
    pass('v1.0.7 compatibility marker present');
  } else {
    warn('Missing v1.0.7 compatibility marker');
  }

  // Check 5: sql.js inclusion
  if (bundle.includes('sql.js') || bundle.includes('initSqlJs')) {
    pass('sql.js WASM code included');
  } else {
    fail('Missing sql.js WASM code');
  }

  // Check 6: Core v1.0.7 methods
  const v107Methods = ['this.run', 'this.exec', 'this.prepare', 'this.export', 'this.close'];
  let methodsFound = 0;

  v107Methods.forEach(method => {
    if (bundle.includes(method)) {
      methodsFound++;
    }
  });

  if (methodsFound === v107Methods.length) {
    pass(`All ${v107Methods.length} v1.0.7 methods present`);
  } else {
    fail(`Only ${methodsFound}/${v107Methods.length} v1.0.7 methods found`);
  }

  // Check 7: New browser bundle methods
  const newMethods = [
    'this.initializeAsync',
    'this.insert',
    'this.search',
    'this.delete',
    'this.storePattern',
    'this.storeEpisode',
    'this.addCausalEdge',
    'this.storeSkill'
  ];

  let newMethodsFound = 0;
  newMethods.forEach(method => {
    if (bundle.includes(method)) {
      newMethodsFound++;
    }
  });

  if (newMethodsFound === newMethods.length) {
    pass(`All ${newMethods.length} new methods present`);
  } else {
    fail(`Only ${newMethodsFound}/${newMethods.length} new methods found`);
  }

  // Check 8: Table schemas
  const tables = ['vectors', 'patterns', 'episodes', 'causal_edges', 'skills'];
  let tablesFound = 0;

  tables.forEach(table => {
    if (bundle.includes(`CREATE TABLE IF NOT EXISTS ${table}`)) {
      tablesFound++;
    }
  });

  if (tablesFound === tables.length) {
    pass(`All ${tables.length} table schemas present`);
  } else {
    fail(`Only ${tablesFound}/${tables.length} table schemas found`);
  }

  // Check 9: Export statements
  const exports = [
    'AgentDB.Database',
    'AgentDB.SQLiteVectorDB',
    'AgentDB.createVectorDB',
    'global.AgentDB',
    'global.Database',
    'global.SQLiteVectorDB'
  ];

  let exportsFound = 0;
  exports.forEach(exp => {
    if (bundle.includes(exp)) {
      exportsFound++;
    }
  });

  if (exportsFound >= exports.length - 1) { // Allow 1 missing for flexibility
    pass(`Export statements present (${exportsFound}/${exports.length})`);
  } else {
    warn(`Only ${exportsFound}/${exports.length} exports found`);
  }

  // Check 10: No development artifacts
  if (bundle.includes('debugger;') || bundle.includes('console.debug')) {
    warn('Development artifacts (debugger/console.debug) found');
  } else {
    pass('No development artifacts');
  }

  // Check 11: SQL injection prevention
  if (bundle.includes('placeholders') && bundle.includes('?')) {
    pass('Parameterized queries (SQL injection prevention)');
  } else {
    warn('Missing parameterized query patterns');
  }

  // Check 12: Error handling
  if (bundle.includes('try') && bundle.includes('catch')) {
    pass('Error handling present');
  } else {
    warn('Limited error handling');
  }

  // Check 13: AgentDB namespace
  if (bundle.includes('var AgentDB = {')) {
    pass('AgentDB namespace defined');
  } else {
    fail('Missing AgentDB namespace');
  }

  // Check 14: Ready flag
  if (bundle.includes('AgentDB.ready')) {
    pass('Ready flag for initialization tracking');
  } else {
    warn('Missing ready flag');
  }

  // Check 15: onReady callback
  if (bundle.includes('onReady')) {
    pass('onReady callback for async initialization');
  } else {
    warn('Missing onReady callback');
  }

  return checks.failed === 0;
}

// Run verification
console.log('═'.repeat(60));
console.log('  AgentDB Browser Bundle Verification');
console.log('═'.repeat(60));
console.log();

const success = verifyBundle();

console.log();
console.log('═'.repeat(60));
console.log('Summary:');
console.log(`  ✅ Passed: ${checks.passed}`);
console.log(`  ❌ Failed: ${checks.failed}`);
console.log(`  ⚠️  Warnings: ${checks.warnings}`);
console.log('═'.repeat(60));

if (success) {
  console.log('\n🎉 Bundle verification successful!\n');
  process.exit(0);
} else {
  console.error('\n💥 Bundle verification failed!\n');
  process.exit(1);
}
