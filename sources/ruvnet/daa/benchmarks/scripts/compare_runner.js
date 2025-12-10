#!/usr/bin/env node
/**
 * Comprehensive benchmark runner
 * Executes all benchmarks and generates comparative analysis
 */

import { spawn } from 'child_process';
import chalk from 'chalk';
import { writeFile } from 'fs/promises';

console.log(chalk.blue.bold('\n🏁 DAA Comprehensive Benchmark Suite\n'));
console.log(chalk.gray('Running all benchmarks and generating reports...\n'));

const results = {
  timestamp: new Date().toISOString(),
  platform: {
    node: process.version,
    arch: process.arch,
    platform: process.platform,
    cpus: require('os').cpus().length
  },
  benchmarks: {
    crypto: { status: 'pending', duration: 0 },
    orchestrator: { status: 'pending', duration: 0 },
    prime: { status: 'pending', duration: 0 }
  }
};

/**
 * Execute a command and capture output
 */
function execCommand(command, args = []) {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();
    console.log(chalk.cyan(`\n▶ Running: ${command} ${args.join(' ')}\n`));

    const proc = spawn(command, args, {
      stdio: 'inherit',
      shell: true
    });

    proc.on('close', (code) => {
      const duration = Date.now() - startTime;

      if (code === 0) {
        console.log(chalk.green(`\n✓ Completed in ${(duration / 1000).toFixed(2)}s\n`));
        resolve({ success: true, duration });
      } else {
        console.log(chalk.yellow(`\n⚠ Exited with code ${code}\n`));
        resolve({ success: false, duration, code });
      }
    });

    proc.on('error', (err) => {
      console.error(chalk.red(`\n✗ Error: ${err.message}\n`));
      reject(err);
    });
  });
}

/**
 * Run all benchmarks sequentially
 */
async function runAllBenchmarks() {
  console.log(chalk.magenta('═'.repeat(60)));
  console.log(chalk.magenta.bold('  CRYPTO BENCHMARKS'));
  console.log(chalk.magenta('═'.repeat(60)));

  try {
    const cryptoResult = await execCommand('npm', ['run', 'bench:crypto-compare']);
    results.benchmarks.crypto = {
      status: cryptoResult.success ? 'completed' : 'failed',
      duration: cryptoResult.duration,
      code: cryptoResult.code
    };
  } catch (error) {
    results.benchmarks.crypto = {
      status: 'error',
      error: error.message
    };
  }

  console.log(chalk.magenta('═'.repeat(60)));
  console.log(chalk.magenta.bold('  ORCHESTRATOR BENCHMARKS'));
  console.log(chalk.magenta('═'.repeat(60)));

  try {
    const orchestratorResult = await execCommand('npm', ['run', 'bench:orchestrator']);
    results.benchmarks.orchestrator = {
      status: orchestratorResult.success ? 'completed' : 'failed',
      duration: orchestratorResult.duration,
      code: orchestratorResult.code
    };
  } catch (error) {
    results.benchmarks.orchestrator = {
      status: 'error',
      error: error.message
    };
  }

  console.log(chalk.magenta('═'.repeat(60)));
  console.log(chalk.magenta.bold('  PRIME ML BENCHMARKS'));
  console.log(chalk.magenta('═'.repeat(60)));

  try {
    const primeResult = await execCommand('npm', ['run', 'bench:prime']);
    results.benchmarks.prime = {
      status: primeResult.success ? 'completed' : 'failed',
      duration: primeResult.duration,
      code: primeResult.code
    };
  } catch (error) {
    results.benchmarks.prime = {
      status: 'error',
      error: error.message
    };
  }

  // Generate summary
  console.log(chalk.blue.bold('\n📊 Benchmark Execution Summary\n'));
  console.log(chalk.cyan('Platform:'), chalk.white(`${results.platform.platform} ${results.platform.arch}`));
  console.log(chalk.cyan('Node.js:'), chalk.white(results.platform.node));
  console.log(chalk.cyan('CPUs:'), chalk.white(results.platform.cpus));
  console.log('');

  const totalDuration = Object.values(results.benchmarks).reduce((sum, b) => sum + (b.duration || 0), 0);

  Object.entries(results.benchmarks).forEach(([name, result]) => {
    const statusColor = result.status === 'completed' ? chalk.green :
      result.status === 'failed' ? chalk.yellow :
        chalk.red;

    const icon = result.status === 'completed' ? '✓' :
      result.status === 'failed' ? '⚠' : '✗';

    console.log(
      statusColor(icon),
      chalk.white(name.padEnd(15)),
      statusColor(result.status.padEnd(10)),
      chalk.gray(`(${(result.duration / 1000).toFixed(2)}s)`)
    );
  });

  console.log('');
  console.log(chalk.cyan('Total Duration:'), chalk.white(`${(totalDuration / 1000).toFixed(2)}s`));

  // Save execution summary
  await writeFile(
    'reports/execution_summary.json',
    JSON.stringify(results, null, 2)
  );

  console.log(chalk.gray('\n💾 Execution summary saved to reports/execution_summary.json\n'));
}

// Main execution
async function main() {
  const startTime = Date.now();

  try {
    await runAllBenchmarks();

    const totalTime = (Date.now() - startTime) / 1000;
    console.log(chalk.green.bold(`\n✓ All benchmarks completed in ${totalTime.toFixed(2)}s\n`));
    console.log(chalk.cyan('Next steps:'));
    console.log(chalk.gray('  1. Generate report: npm run report:generate'));
    console.log(chalk.gray('  2. Create visualizations: npm run report:visualize'));
    console.log(chalk.gray('  3. View HTML report: open reports/benchmark_report.html\n'));

  } catch (error) {
    console.error(chalk.red('\n✗ Benchmark execution failed:'), error.message);
    process.exit(1);
  }
}

main();
