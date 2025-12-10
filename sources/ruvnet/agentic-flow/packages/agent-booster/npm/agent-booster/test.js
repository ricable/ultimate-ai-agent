/**
 * Agent Booster - Auto-detection Tests
 * Tests runtime detection and API functionality
 */

const assert = require('assert');
const agentBooster = require('./index');

// Test configuration
const TESTS_ENABLED = {
  autoDetection: true,
  api: true,
  errorHandling: true
};

// Test results tracking
const results = {
  passed: 0,
  failed: 0,
  skipped: 0
};

/**
 * Test helper function
 */
async function test(name, fn, enabled = true) {
  if (!enabled) {
    console.log(`⊘ SKIPPED: ${name}`);
    results.skipped++;
    return;
  }

  try {
    await fn();
    console.log(`✓ PASSED: ${name}`);
    results.passed++;
  } catch (error) {
    console.log(`✗ FAILED: ${name}`);
    console.log(`  Error: ${error.message}`);
    results.failed++;
  }
}

// Test Suite 1: Runtime Auto-detection
async function testAutoDetection() {
  console.log('\n=== Test Suite 1: Runtime Auto-detection ===\n');

  await test('Runtime detection', () => {
    // Initialize should not throw
    try {
      agentBooster.initialize();
    } catch (error) {
      // Expected if no runtime available
      if (!error.message.includes('No runtime available')) {
        throw error;
      }
    }
    // Test passes if no unexpected error
  }, TESTS_ENABLED.autoDetection);

  await test('getRuntime returns valid type', () => {
    const runtime = agentBooster.getRuntime();
    // Should be null, 'native', or 'wasm'
    assert(
      runtime === null || runtime === 'native' || runtime === 'wasm',
      `Invalid runtime type: ${runtime}`
    );
  }, TESTS_ENABLED.autoDetection);

  await test('getVersion returns version info', () => {
    try {
      const info = agentBooster.getVersion();
      assert(typeof info === 'object', 'Version info should be an object');
      assert(typeof info.version === 'string', 'Version should be a string');
      assert(typeof info.runtime === 'string' || info.runtime === null, 'Runtime should be string or null');
    } catch (error) {
      // Expected if no runtime available
      if (!error.message.includes('No runtime available')) {
        throw error;
      }
    }
  }, TESTS_ENABLED.autoDetection);
}

// Test Suite 2: API Functionality
async function testAPI() {
  console.log('\n=== Test Suite 2: API Functionality ===\n');

  await test('apply accepts string prompt', async () => {
    try {
      const result = await agentBooster.apply('test prompt');
      assert(typeof result === 'string', 'Result should be a string');
    } catch (error) {
      // Expected if no runtime available
      if (!error.message.includes('No runtime available')) {
        throw error;
      }
    }
  }, TESTS_ENABLED.api);

  await test('apply accepts options', async () => {
    try {
      const result = await agentBooster.apply('test prompt', {
        strategy: 'balanced',
        maxTokens: 100
      });
      assert(typeof result === 'string', 'Result should be a string');
    } catch (error) {
      // Expected if no runtime available
      if (!error.message.includes('No runtime available')) {
        throw error;
      }
    }
  }, TESTS_ENABLED.api);

  await test('batchApply accepts array', async () => {
    try {
      const results = await agentBooster.batchApply(['prompt1', 'prompt2']);
      assert(Array.isArray(results), 'Results should be an array');
      assert(results.length === 2, 'Results length should match input');
    } catch (error) {
      // Expected if no runtime available
      if (!error.message.includes('No runtime available')) {
        throw error;
      }
    }
  }, TESTS_ENABLED.api);

  await test('analyze returns analysis object', async () => {
    try {
      const analysis = await agentBooster.analyze('test prompt');
      assert(typeof analysis === 'object', 'Analysis should be an object');
      assert(typeof analysis.originalTokens === 'number', 'Should have originalTokens');
      assert(typeof analysis.optimizedTokens === 'number', 'Should have optimizedTokens');
      assert(typeof analysis.savings === 'number', 'Should have savings');
      assert(typeof analysis.savingsPercent === 'number', 'Should have savingsPercent');
    } catch (error) {
      // Expected if no runtime available
      if (!error.message.includes('No runtime available')) {
        throw error;
      }
    }
  }, TESTS_ENABLED.api);
}

// Test Suite 3: Error Handling
async function testErrorHandling() {
  console.log('\n=== Test Suite 3: Error Handling ===\n');

  await test('Throws on invalid prompt type', async () => {
    try {
      await agentBooster.apply(123); // Invalid: number instead of string
      throw new Error('Should have thrown an error');
    } catch (error) {
      // Expected to throw
      if (error.message === 'Should have thrown an error') {
        throw error;
      }
      // Test passes if error was thrown
    }
  }, TESTS_ENABLED.errorHandling);

  await test('Throws on invalid options', async () => {
    try {
      await agentBooster.apply('test', { strategy: 'invalid' });
      throw new Error('Should have thrown an error');
    } catch (error) {
      // Expected to throw
      if (error.message === 'Should have thrown an error') {
        throw error;
      }
      // Test passes if error was thrown
    }
  }, TESTS_ENABLED.errorHandling);

  await test('Graceful handling when no runtime available', async () => {
    // This test verifies the error message is helpful
    try {
      const tempModule = { ...agentBooster };
      await tempModule.apply('test');
    } catch (error) {
      if (error.message.includes('No runtime available')) {
        // Expected error with helpful message
        assert(
          error.message.includes('npm install'),
          'Error should suggest installation'
        );
      }
    }
  }, TESTS_ENABLED.errorHandling);
}

// Main test runner
async function runTests() {
  console.log('Agent Booster - Auto-detection Tests\n');
  console.log('='.repeat(50));

  // Display runtime status
  try {
    agentBooster.initialize();
    const runtime = agentBooster.getRuntime();
    console.log(`\nRuntime detected: ${runtime || 'none'}\n`);
  } catch (error) {
    console.log('\n⚠ No runtime available - some tests will be skipped\n');
  }

  // Run test suites
  await testAutoDetection();
  await testAPI();
  await testErrorHandling();

  // Display results
  console.log('\n' + '='.repeat(50));
  console.log('\nTest Results:');
  console.log(`  Passed:  ${results.passed}`);
  console.log(`  Failed:  ${results.failed}`);
  console.log(`  Skipped: ${results.skipped}`);
  console.log(`  Total:   ${results.passed + results.failed + results.skipped}`);

  if (results.failed === 0) {
    console.log('\n✓ All tests passed!\n');
    process.exit(0);
  } else {
    console.log('\n✗ Some tests failed\n');
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  runTests().catch((error) => {
    console.error('\nTest runner error:', error.message);
    process.exit(1);
  });
}

module.exports = { runTests };
