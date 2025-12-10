#!/usr/bin/env npx ts-node

/**
 * RTB Hierarchical Template System Test Runner
 *
 * This script provides a convenient way to run specific test suites
 * for the RTB hierarchical template system with different options.
 */

import { execSync } from 'child_process';
import { existsSync } from 'fs';
import { join } from 'path';

interface TestOptions {
  coverage?: boolean;
  watch?: boolean;
  verbose?: boolean;
  suite?: 'unit' | 'integration' | 'performance' | 'all';
  pattern?: string;
  bail?: boolean;
}

const DEFAULT_OPTIONS: TestOptions = {
  coverage: false,
  watch: false,
  verbose: true,
  suite: 'all',
  bail: false
};

class RTBTestRunner {
  private projectRoot: string;

  constructor() {
    this.projectRoot = process.cwd();
  }

  /**
   * Run RTB hierarchical template system tests
   */
  async runTests(options: TestOptions = {}): Promise<void> {
    const opts = { ...DEFAULT_OPTIONS, ...options };

    console.log('🚀 RTB Hierarchical Template System Test Runner');
    console.log('=====================================================');
    console.log(`Suite: ${opts.suite}`);
    console.log(`Coverage: ${opts.coverage ? '✅' : '❌'}`);
    console.log(`Watch Mode: ${opts.watch ? '✅' : '❌'}`);
    console.log(`Verbose: ${opts.verbose ? '✅' : '❌'}`);
    console.log(`Bail on Failure: ${opts.bail ? '✅' : '❌'}`);
    console.log('');

    try {
      switch (opts.suite) {
        case 'unit':
          await this.runUnitTests(opts);
          break;
        case 'integration':
          await this.runIntegrationTests(opts);
          break;
        case 'performance':
          await this.runPerformanceTests(opts);
          break;
        case 'all':
        default:
          await this.runAllTests(opts);
          break;
      }

      console.log('\n✅ Tests completed successfully!');
    } catch (error) {
      console.error('\n❌ Tests failed!');
      process.exit(1);
    }
  }

  /**
   * Run unit tests only
   */
  private async runUnitTests(options: TestOptions): Promise<void> {
    console.log('Running Unit Tests...');

    const jestArgs = this.buildJestArgs({
      ...options,
      config: 'jest.rtb.config.js',
      selectProjects: ['Unit Tests']
    });

    this.executeJest(jestArgs);
  }

  /**
   * Run integration tests only
   */
  private async runIntegrationTests(options: TestOptions): Promise<void> {
    console.log('Running Integration Tests...');

    const jestArgs = this.buildJestArgs({
      ...options,
      config: 'jest.rtb.config.js',
      selectProjects: ['Integration Tests']
    });

    this.executeJest(jestArgs);
  }

  /**
   * Run performance tests only
   */
  private async runPerformanceTests(options: TestOptions): Promise<void> {
    console.log('Running Performance Tests...');
    console.log('⚠️  Performance tests may take several minutes to complete');

    const jestArgs = this.buildJestArgs({
      ...options,
      config: 'jest.rtb.config.js',
      selectProjects: ['Performance Tests']
    });

    this.executeJest(jestArgs);
  }

  /**
   * Run all tests
   */
  private async runAllTests(options: TestOptions): Promise<void> {
    console.log('Running All Tests...');

    const jestArgs = this.buildJestArgs({
      ...options,
      config: 'jest.rtb.config.js'
    });

    this.executeJest(jestArgs);
  }

  /**
   * Build Jest command arguments
   */
  private buildJestArgs(options: any): string[] {
    const args: string[] = [];

    // Config file
    if (options.config) {
      args.push(`--config=${options.config}`);
    }

    // Coverage
    if (options.coverage) {
      args.push('--coverage');
      args.push('--coverageReporters=text-lcov');
      args.push('--coverageReporters=html');
    }

    // Watch mode
    if (options.watch) {
      args.push('--watch');
    }

    // Verbose output
    if (options.verbose) {
      args.push('--verbose');
    }

    // Bail on first failure
    if (options.bail) {
      args.push('--bail');
    }

    // Select specific projects
    if (options.selectProjects) {
      if (Array.isArray(options.selectProjects)) {
        options.selectProjects.forEach((project: string) => {
          args.push(`--selectProjects=${project}`);
        });
      } else {
        args.push(`--selectProjects=${options.selectProjects}`);
      }
    }

    // Pattern matching
    if (options.pattern) {
      args.push(`--testNamePattern=${options.pattern}`);
    }

    return args;
  }

  /**
   * Execute Jest with provided arguments
   */
  private executeJest(args: string[]): void {
    const jestCommand = 'npx jest';
    const fullCommand = `${jestCommand} ${args.join(' ')}`;

    console.log(`Executing: ${fullCommand}`);
    console.log('');

    try {
      execSync(fullCommand, {
        stdio: 'inherit',
        cwd: this.projectRoot
      });
    } catch (error) {
      throw new Error(`Jest execution failed: ${error}`);
    }
  }

  /**
   * Generate test coverage report
   */
  async generateCoverageReport(): Promise<void> {
    console.log('Generating Coverage Report...');

    const jestArgs = this.buildJestArgs({
      config: 'jest.rtb.config.js',
      coverage: true,
      verbose: false
    });

    this.executeJest(jestArgs);

    console.log('\n📊 Coverage report generated in coverage/rtb/');
    console.log('   - HTML report: coverage/rtb/lcov-report/index.html');
    console.log('   - LCOV report: coverage/rtb/lcov.info');
  }

  /**
   * Run tests in CI mode
   */
  async runCITests(): Promise<void> {
    console.log('Running Tests in CI Mode...');

    const jestArgs = this.buildJestArgs({
      config: 'jest.rtb.config.js',
      coverage: true,
      verbose: false,
      bail: true,
      watch: false
    });

    // Add CI-specific options
    jestArgs.push('--ci');
    jestArgs.push('--reporters=default');
    jestArgs.push('--reporters=jest-junit');

    this.executeJest(jestArgs);
  }

  /**
   * Validate test configuration
   */
  validateConfiguration(): boolean {
    const requiredFiles = [
      'jest.rtb.config.js',
      'tests/rtb/hierarchical-template-system/priority-engine.test.ts',
      'tests/rtb/hierarchical-template-system/template-merger.test.ts',
      'tests/rtb/hierarchical-template-system/base-generator.test.ts',
      'tests/rtb/hierarchical-template-system/variant-generators.test.ts',
      'tests/rtb/hierarchical-template-system/frequency-relations.test.ts',
      'tests/rtb/hierarchical-template-system/integration.test.ts',
      'tests/rtb/hierarchical-template-system/performance.test.ts'
    ];

    console.log('Validating test configuration...');

    let allValid = true;

    for (const file of requiredFiles) {
      const filePath = join(this.projectRoot, file);
      if (!existsSync(filePath)) {
        console.error(`❌ Missing required file: ${file}`);
        allValid = false;
      } else {
        console.log(`✅ Found: ${file}`);
      }
    }

    return allValid;
  }
}

/**
 * Command line interface
 */
function printUsage(): void {
  console.log(`
Usage: npx ts-node scripts/run-rtb-tests.ts [options]

Options:
  --suite <type>     Test suite to run (unit|integration|performance|all) [default: all]
  --coverage         Generate coverage report
  --watch            Run tests in watch mode
  --verbose          Enable verbose output [default: true]
  --bail             Stop on first failure
  --pattern <regex>  Run tests matching pattern
  --ci               Run in CI mode (implies --coverage --bail)
  --validate         Validate test configuration only
  --help             Show this help message

Examples:
  # Run all tests
  npx ts-node scripts/run-rtb-tests.ts

  # Run only unit tests with coverage
  npx ts-node scripts/run-rtb-tests.ts --suite unit --coverage

  # Run integration tests
  npx ts-node scripts/run-rtb-tests.ts --suite integration

  # Run performance tests
  npx ts-node scripts/run-rtb-tests.ts --suite performance

  # Run tests matching a pattern
  npx ts-node scripts/run-rtb-tests.ts --pattern "priority.*inheritance"

  # Run in CI mode
  npx ts-node scripts/run-rtb-tests.ts --ci

  # Validate configuration only
  npx ts-node scripts/run-rtb-tests.ts --validate
`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const options: TestOptions = {};

  // Parse command line arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case '--suite':
        options.suite = args[++i] as any;
        break;
      case '--coverage':
        options.coverage = true;
        break;
      case '--watch':
        options.watch = true;
        break;
      case '--verbose':
        options.verbose = true;
        break;
      case '--bail':
        options.bail = true;
        break;
      case '--pattern':
        options.pattern = args[++i];
        break;
      case '--ci':
        options.coverage = true;
        options.bail = true;
        options.watch = false;
        break;
      case '--validate':
        const runner = new RTBTestRunner();
        const isValid = runner.validateConfiguration();
        if (isValid) {
          console.log('\n✅ Test configuration is valid!');
        } else {
          console.log('\n❌ Test configuration has issues!');
          process.exit(1);
        }
        return;
      case '--help':
        printUsage();
        return;
      default:
        if (arg.startsWith('--')) {
          console.error(`Unknown option: ${arg}`);
          printUsage();
          process.exit(1);
        }
    }
  }

  // Validate configuration before running tests
  const runner = new RTBTestRunner();
  if (!runner.validateConfiguration()) {
    console.log('\n❌ Please fix the configuration issues before running tests.');
    process.exit(1);
  }

  // Run tests
  if (args.includes('--ci')) {
    await runner.runCITests();
  } else {
    await runner.runTests(options);
  }
}

// Run the script
if (require.main === module) {
  main().catch(error => {
    console.error('Error running tests:', error);
    process.exit(1);
  });
}

export { RTBTestRunner };