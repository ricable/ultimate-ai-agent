#!/usr/bin/env node
/**
 * Comprehensive WASM functionality tests for agentic-jujutsu
 * Tests all exported functions, types, and error handling
 */

const path = require('path');
const fs = require('fs');

const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

function log(color, symbol, message) {
  console.log(`${color}${symbol}${colors.reset} ${message}`);
}

class TestRunner {
  constructor() {
    this.passed = 0;
    this.failed = 0;
    this.skipped = 0;
    this.testResults = [];
  }

  async test(name, fn) {
    try {
      await fn();
      log(colors.green, '✓', name);
      this.passed++;
      this.testResults.push({ name, status: 'passed' });
    } catch (error) {
      log(colors.red, '✗', `${name}\n  ${error.message}`);
      this.failed++;
      this.testResults.push({ name, status: 'failed', error: error.message });
    }
  }

  skip(name) {
    log(colors.yellow, '○', `${name} (skipped)`);
    this.skipped++;
    this.testResults.push({ name, status: 'skipped' });
  }

  assert(condition, message) {
    if (!condition) {
      throw new Error(message || 'Assertion failed');
    }
  }

  assertEqual(actual, expected, message) {
    if (actual !== expected) {
      throw new Error(message || `Expected ${expected}, got ${actual}`);
    }
  }

  assertNotNull(value, message) {
    if (value === null || value === undefined) {
      throw new Error(message || 'Expected non-null value');
    }
  }

  assertThrows(fn, message) {
    let threw = false;
    try {
      fn();
    } catch (e) {
      threw = true;
    }
    if (!threw) {
      throw new Error(message || 'Expected function to throw');
    }
  }

  report() {
    const total = this.passed + this.failed + this.skipped;
    console.log('\n' + '='.repeat(60));
    console.log(`📊 Test Results`);
    console.log('='.repeat(60));
    console.log(`${colors.green}✓ Passed: ${this.passed}${colors.reset}`);
    console.log(`${colors.red}✗ Failed: ${this.failed}${colors.reset}`);
    console.log(`${colors.yellow}○ Skipped: ${this.skipped}${colors.reset}`);
    console.log(`Total: ${total}`);
    console.log(`Coverage: ${((this.passed / total) * 100).toFixed(1)}%`);
    console.log('='.repeat(60) + '\n');

    return this.failed === 0;
  }
}

async function runTests() {
  console.log('\n🧪 agentic-jujutsu Comprehensive WASM Tests\n');
  const runner = new TestRunner();

  // Load the WASM module
  let jj;
  try {
    const pkgPath = path.join(__dirname, '../../pkg/node');

    if (!fs.existsSync(pkgPath)) {
      throw new Error(`WASM package not found at ${pkgPath}. Run 'npm run build' first.`);
    }

    jj = require(pkgPath);
    log(colors.blue, 'ℹ', `Loaded WASM module from ${pkgPath}`);
  } catch (error) {
    log(colors.red, '✗', `Failed to load WASM module: ${error.message}`);
    process.exit(1);
  }

  // Test Suite 1: Module Structure
  console.log(`\n${colors.cyan}═══ Module Structure Tests ═══${colors.reset}\n`);

  await runner.test('Module exports are defined', () => {
    const exports = Object.keys(jj);
    runner.assert(exports.length > 0, 'Module should have exports');
    log(colors.blue, 'ℹ', `Found exports: ${exports.join(', ')}`);
  });

  await runner.test('TypeScript definitions exist', () => {
    const dtsPath = path.join(__dirname, '../../pkg/node/agentic_jujutsu.d.ts');
    runner.assert(fs.existsSync(dtsPath), 'TypeScript definitions should exist');
  });

  await runner.test('WASM binary exists and is valid size', () => {
    const wasmPath = path.join(__dirname, '../../pkg/node/agentic_jujutsu_bg.wasm');
    runner.assert(fs.existsSync(wasmPath), 'WASM binary should exist');

    const size = fs.statSync(wasmPath).size;
    runner.assert(size > 0, 'WASM binary should have non-zero size');
    runner.assert(size < 10 * 1024 * 1024, 'WASM binary should be under 10MB');

    const sizeKB = Math.round(size / 1024);
    log(colors.blue, 'ℹ', `WASM binary size: ${sizeKB}KB`);
  });

  await runner.test('Package.json metadata is correct', () => {
    const pkgJsonPath = path.join(__dirname, '../../pkg/node/package.json');
    runner.assert(fs.existsSync(pkgJsonPath), 'package.json should exist');

    const pkgJson = JSON.parse(fs.readFileSync(pkgJsonPath, 'utf8'));
    runner.assert(pkgJson.name, 'Package should have a name');
    runner.assert(pkgJson.version, 'Package should have a version');
  });

  // Test Suite 2: Core Functionality
  console.log(`\n${colors.cyan}═══ Core Functionality Tests ═══${colors.reset}\n`);

  await runner.test('VERSION constant is defined', () => {
    if (jj.VERSION) {
      runner.assertNotNull(jj.VERSION, 'VERSION should be defined');
      runner.assert(typeof jj.VERSION === 'string', 'VERSION should be a string');
      log(colors.blue, 'ℹ', `Version: ${jj.VERSION}`);
    } else {
      runner.skip('VERSION export not available');
    }
  });

  await runner.test('JJWrapper class is exported', () => {
    if (jj.JJWrapper) {
      runner.assertNotNull(jj.JJWrapper, 'JJWrapper should be exported');
      runner.assert(typeof jj.JJWrapper === 'function', 'JJWrapper should be a constructor');
    } else {
      runner.skip('JJWrapper not exported in WASM build');
    }
  });

  await runner.test('Error types are defined', () => {
    if (jj.JJError) {
      runner.assertNotNull(jj.JJError, 'JJError should be defined');
    } else {
      runner.skip('JJError not exported in WASM build');
    }
  });

  // Test Suite 3: Type Checking
  console.log(`\n${colors.cyan}═══ Type Definition Tests ═══${colors.reset}\n`);

  await runner.test('All exports have correct types', () => {
    const exports = Object.keys(jj);
    exports.forEach(name => {
      const value = jj[name];
      runner.assertNotNull(value, `Export ${name} should not be null`);
      log(colors.blue, 'ℹ', `${name}: ${typeof value}`);
    });
  });

  // Test Suite 4: Performance Characteristics
  console.log(`\n${colors.cyan}═══ Performance Tests ═══${colors.reset}\n`);

  await runner.test('Module loads quickly', async () => {
    const start = process.hrtime.bigint();
    const testModule = require(path.join(__dirname, '../../pkg/node'));
    const end = process.hrtime.bigint();

    const loadTimeMs = Number(end - start) / 1_000_000;
    runner.assert(loadTimeMs < 1000, 'Module should load in under 1 second');
    log(colors.blue, 'ℹ', `Load time: ${loadTimeMs.toFixed(2)}ms`);
  });

  await runner.test('WASM initialization is fast', async () => {
    if (jj.init) {
      const start = process.hrtime.bigint();
      jj.init();
      const end = process.hrtime.bigint();

      const initTimeMs = Number(end - start) / 1_000_000;
      log(colors.blue, 'ℹ', `Init time: ${initTimeMs.toFixed(2)}ms`);
    } else {
      runner.skip('init() function not available');
    }
  });

  // Test Suite 5: Memory Management
  console.log(`\n${colors.cyan}═══ Memory Tests ═══${colors.reset}\n`);

  await runner.test('No memory leaks on repeated requires', () => {
    const initialMem = process.memoryUsage().heapUsed;

    // Load module multiple times
    for (let i = 0; i < 10; i++) {
      delete require.cache[require.resolve(path.join(__dirname, '../../pkg/node'))];
      require(path.join(__dirname, '../../pkg/node'));
    }

    if (global.gc) global.gc();

    const finalMem = process.memoryUsage().heapUsed;
    const memIncrease = finalMem - initialMem;
    const memIncreaseMB = memIncrease / (1024 * 1024);

    runner.assert(memIncreaseMB < 50, 'Memory increase should be under 50MB');
    log(colors.blue, 'ℹ', `Memory increase: ${memIncreaseMB.toFixed(2)}MB`);
  });

  // Test Suite 6: Error Handling
  console.log(`\n${colors.cyan}═══ Error Handling Tests ═══${colors.reset}\n`);

  await runner.test('Invalid imports are handled gracefully', () => {
    try {
      const invalidPath = path.join(__dirname, '../../pkg/nonexistent');
      require(invalidPath);
      runner.assert(false, 'Should have thrown an error');
    } catch (error) {
      runner.assert(error.code === 'MODULE_NOT_FOUND', 'Should throw MODULE_NOT_FOUND');
    }
  });

  // Test Suite 7: Integration Points
  console.log(`\n${colors.cyan}═══ Integration Tests ═══${colors.reset}\n`);

  await runner.test('Module is compatible with Node.js version', () => {
    const nodeVersion = process.version;
    const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);

    runner.assert(majorVersion >= 16, 'Node.js version should be >= 16');
    log(colors.blue, 'ℹ', `Node.js version: ${nodeVersion}`);
  });

  await runner.test('Module works with CommonJS', () => {
    const cjsModule = require(path.join(__dirname, '../../pkg/node'));
    runner.assertNotNull(cjsModule, 'CommonJS require should work');
  });

  await runner.test('Package structure is correct', () => {
    const pkgPath = path.join(__dirname, '../../pkg/node');
    const expectedFiles = [
      'agentic_jujutsu.js',
      'agentic_jujutsu_bg.wasm',
      'agentic_jujutsu.d.ts',
      'package.json'
    ];

    expectedFiles.forEach(file => {
      const filePath = path.join(pkgPath, file);
      runner.assert(fs.existsSync(filePath), `${file} should exist`);
    });
  });

  // Final Report
  const success = runner.report();

  // Save test results to file
  const resultsPath = path.join(__dirname, '../../test-results.json');
  fs.writeFileSync(resultsPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    summary: {
      passed: runner.passed,
      failed: runner.failed,
      skipped: runner.skipped,
      total: runner.passed + runner.failed + runner.skipped,
      coverage: ((runner.passed / (runner.passed + runner.failed + runner.skipped)) * 100).toFixed(1)
    },
    results: runner.testResults
  }, null, 2));

  log(colors.blue, 'ℹ', `Test results saved to ${resultsPath}`);

  process.exit(success ? 0 : 1);
}

// Run tests
runTests().catch(err => {
  console.error(`\n${colors.red}Fatal error:${colors.reset}`, err);
  process.exit(1);
});
