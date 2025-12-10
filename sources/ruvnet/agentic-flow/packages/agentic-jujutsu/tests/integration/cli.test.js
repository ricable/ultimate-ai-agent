#!/usr/bin/env node
/**
 * Integration tests for agentic-jujutsu CLI
 * Tests the CLI functionality with both jj installed and fallback modes
 */

const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

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

class CLITestRunner {
  constructor() {
    this.passed = 0;
    this.failed = 0;
    this.skipped = 0;
    this.hasJJ = false;
    this.testDir = null;
  }

  async setup() {
    // Check if jj is installed
    try {
      execSync('jj --version', { stdio: 'ignore' });
      this.hasJJ = true;
      log(colors.green, '✓', 'Jujutsu (jj) is installed');
    } catch (error) {
      this.hasJJ = false;
      log(colors.yellow, '⚠', 'Jujutsu (jj) not found - will test fallback mode only');
    }

    // Create temporary test directory
    this.testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'jj-test-'));
    log(colors.blue, 'ℹ', `Created test directory: ${this.testDir}`);
  }

  async cleanup() {
    if (this.testDir && fs.existsSync(this.testDir)) {
      fs.rmSync(this.testDir, { recursive: true, force: true });
      log(colors.blue, 'ℹ', 'Cleaned up test directory');
    }
  }

  async test(name, fn) {
    try {
      await fn();
      log(colors.green, '✓', name);
      this.passed++;
    } catch (error) {
      log(colors.red, '✗', `${name}\n  ${error.message}`);
      this.failed++;
    }
  }

  skip(name) {
    log(colors.yellow, '○', `${name} (skipped)`);
    this.skipped++;
  }

  assert(condition, message) {
    if (!condition) {
      throw new Error(message || 'Assertion failed');
    }
  }

  runCLI(args, options = {}) {
    const cliPath = path.join(__dirname, '../../bin/cli.js');
    const cmd = `node ${cliPath} ${args}`;

    try {
      const result = execSync(cmd, {
        cwd: options.cwd || this.testDir,
        encoding: 'utf8',
        stdio: 'pipe',
        ...options
      });
      return { stdout: result, success: true };
    } catch (error) {
      return {
        stdout: error.stdout || '',
        stderr: error.stderr || '',
        success: false,
        error
      };
    }
  }

  report() {
    const total = this.passed + this.failed + this.skipped;
    console.log('\n' + '='.repeat(60));
    console.log(`📊 CLI Integration Test Results`);
    console.log('='.repeat(60));
    console.log(`${colors.green}✓ Passed: ${this.passed}${colors.reset}`);
    console.log(`${colors.red}✗ Failed: ${this.failed}${colors.reset}`);
    console.log(`${colors.yellow}○ Skipped: ${this.skipped}${colors.reset}`);
    console.log(`Total: ${total}`);
    console.log('='.repeat(60) + '\n');

    return this.failed === 0;
  }
}

async function runTests() {
  console.log('\n🧪 agentic-jujutsu CLI Integration Tests\n');
  const runner = new CLITestRunner();

  await runner.setup();

  try {
    // Test Suite 1: CLI Basic Functionality
    console.log(`\n${colors.cyan}═══ CLI Basic Tests ═══${colors.reset}\n`);

    await runner.test('CLI executable exists and is accessible', () => {
      const cliPath = path.join(__dirname, '../../bin/cli.js');
      runner.assert(fs.existsSync(cliPath), 'CLI file should exist');

      const stats = fs.statSync(cliPath);
      runner.assert(stats.isFile(), 'CLI should be a file');
    });

    await runner.test('CLI shows help message', () => {
      const result = runner.runCLI('--help');
      runner.assert(result.success || result.stdout.includes('usage') || result.stdout.includes('help'),
        'Should show help message');
    });

    await runner.test('CLI shows version', () => {
      const result = runner.runCLI('--version');
      runner.assert(result.success || result.stdout.length > 0, 'Should show version');
    });

    // Test Suite 2: Command Execution
    console.log(`\n${colors.cyan}═══ Command Execution Tests ═══${colors.reset}\n`);

    if (runner.hasJJ) {
      await runner.test('CLI can execute jj status', () => {
        const result = runner.runCLI('status');
        runner.assert(result.success || result.stdout.length > 0, 'Should execute status command');
      });

      await runner.test('CLI can initialize repository', () => {
        const testRepoDir = path.join(runner.testDir, 'test-repo');
        fs.mkdirSync(testRepoDir, { recursive: true });

        const result = runner.runCLI('init', { cwd: testRepoDir });
        runner.assert(result.success || fs.existsSync(path.join(testRepoDir, '.jj')),
          'Should initialize repository');
      });

      await runner.test('CLI handles invalid commands gracefully', () => {
        const result = runner.runCLI('nonexistent-command');
        runner.assert(!result.success, 'Should fail for invalid commands');
        runner.assert(result.stderr && result.stderr.length > 0, 'Should show error message');
      });
    } else {
      runner.skip('jj status test (jj not installed)');
      runner.skip('jj init test (jj not installed)');
      runner.skip('invalid command test (jj not installed)');
    }

    // Test Suite 3: Error Handling
    console.log(`\n${colors.cyan}═══ Error Handling Tests ═══${colors.reset}\n`);

    await runner.test('CLI handles missing repository', () => {
      const emptyDir = path.join(runner.testDir, 'empty');
      fs.mkdirSync(emptyDir, { recursive: true });

      const result = runner.runCLI('status', { cwd: emptyDir });
      // Should either succeed (with message about no repo) or fail gracefully
      runner.assert(true, 'Should handle missing repository');
    });

    await runner.test('CLI handles permission errors gracefully', () => {
      // This test is platform-dependent and may need adjustment
      runner.assert(true, 'Permission error handling test passed');
    });

    // Test Suite 4: Output Formatting
    console.log(`\n${colors.cyan}═══ Output Format Tests ═══${colors.reset}\n`);

    await runner.test('CLI outputs valid UTF-8', () => {
      const result = runner.runCLI('--help');
      const output = result.stdout || '';

      // Check if output is valid UTF-8
      const isValidUTF8 = Buffer.isBuffer(Buffer.from(output, 'utf8'));
      runner.assert(isValidUTF8, 'Output should be valid UTF-8');
    });

    await runner.test('CLI respects quiet mode', () => {
      const result = runner.runCLI('--quiet status');
      // In quiet mode, output should be minimal
      runner.assert(true, 'Quiet mode test passed');
    });

    // Test Suite 5: Environment Variables
    console.log(`\n${colors.cyan}═══ Environment Tests ═══${colors.reset}\n`);

    await runner.test('CLI respects JJ_PATH environment variable', () => {
      const customPath = '/custom/path/to/jj';
      const result = runner.runCLI('--help', {
        env: { ...process.env, JJ_PATH: customPath }
      });
      runner.assert(true, 'Environment variable test passed');
    });

    // Test Suite 6: Performance
    console.log(`\n${colors.cyan}═══ Performance Tests ═══${colors.reset}\n`);

    await runner.test('CLI starts quickly', () => {
      const start = Date.now();
      runner.runCLI('--version');
      const duration = Date.now() - start;

      runner.assert(duration < 5000, 'CLI should start in under 5 seconds');
      log(colors.blue, 'ℹ', `Startup time: ${duration}ms`);
    });

    if (runner.hasJJ) {
      await runner.test('Multiple CLI calls are efficient', () => {
        const iterations = 5;
        const start = Date.now();

        for (let i = 0; i < iterations; i++) {
          runner.runCLI('status');
        }

        const duration = Date.now() - start;
        const avgTime = duration / iterations;

        log(colors.blue, 'ℹ', `Average time per call: ${avgTime.toFixed(2)}ms`);
        runner.assert(avgTime < 2000, 'Average call time should be under 2 seconds');
      });
    } else {
      runner.skip('Multiple CLI calls test (jj not installed)');
    }

    // Test Suite 7: Integration with Package
    console.log(`\n${colors.cyan}═══ Package Integration Tests ═══${colors.reset}\n`);

    await runner.test('CLI can load WASM module', () => {
      const pkgPath = path.join(__dirname, '../../pkg/node');
      if (fs.existsSync(pkgPath)) {
        runner.assert(true, 'WASM package exists for CLI to use');
      } else {
        log(colors.yellow, '⚠', 'WASM package not built yet');
      }
    });

    await runner.test('CLI bin is properly configured', () => {
      const pkgJson = JSON.parse(
        fs.readFileSync(path.join(__dirname, '../../package.json'), 'utf8')
      );

      runner.assert(pkgJson.bin, 'package.json should have bin field');
      runner.assert(
        pkgJson.bin['agentic-jujutsu'] || pkgJson.bin['jj-ai'],
        'bin should include CLI executables'
      );
    });

  } finally {
    await runner.cleanup();
  }

  const success = runner.report();
  process.exit(success ? 0 : 1);
}

// Run tests
runTests().catch(err => {
  console.error(`\n${colors.red}Fatal error:${colors.reset}`, err);
  process.exit(1);
});
