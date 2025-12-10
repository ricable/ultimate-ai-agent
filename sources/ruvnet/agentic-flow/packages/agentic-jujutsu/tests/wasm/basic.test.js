#!/usr/bin/env node
/**
 * Basic WASM functionality tests for Node.js
 */

const path = require('path');
const fs = require('fs');

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
};

function log(color, symbol, message) {
  console.log(`${color}${symbol}${colors.reset} ${message}`);
}

async function runTests() {
  console.log('\n🧪 Running agentic-jujutsu WASM Tests\n');
  let passed = 0, failed = 0;

  try {
    const pkgPath = path.join(__dirname, '../../pkg/node');
    
    if (!fs.existsSync(pkgPath)) {
      throw new Error(`WASM package not found. Run 'npm run build' first.`);
    }

    const jj = require(pkgPath);
    log(colors.green, '✓', 'Module loaded successfully');
    passed++;

    // Check exports
    const exports = Object.keys(jj);
    if (exports.length > 0) {
      log(colors.green, '✓', `Found ${exports.length} exports`);
      passed++;
    }

    // Check TypeScript definitions
    const dtsPath = path.join(pkgPath, 'agentic_jujutsu.d.ts');
    if (fs.existsSync(dtsPath)) {
      log(colors.green, '✓', 'TypeScript definitions present');
      passed++;
    }

    // Check WASM binary
    const wasmPath = path.join(pkgPath, 'agentic_jujutsu_bg.wasm');
    if (fs.existsSync(wasmPath)) {
      const size = Math.round(fs.statSync(wasmPath).size / 1024);
      log(colors.green, '✓', `WASM binary: ${size}KB`);
      passed++;
    }

  } catch (error) {
    log(colors.red, '✗', `Test failed: ${error.message}`);
    failed++;
  }

  console.log(`\n📊 Results: ${colors.green}${passed} passed${colors.reset}, ${failed > 0 ? colors.red : colors.green}${failed} failed${colors.reset}\n`);
  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
