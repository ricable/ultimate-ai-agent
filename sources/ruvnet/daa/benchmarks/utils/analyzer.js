#!/usr/bin/env node
/**
 * Benchmark result analyzer
 * Provides detailed statistical analysis of benchmark results
 */

import { readFile } from 'fs/promises';
import { existsSync } from 'fs';
import chalk from 'chalk';
import stats from './statistics.js';

console.log(chalk.blue.bold('\n📊 Benchmark Result Analyzer\n'));

async function analyzeCryptoResults() {
  if (!existsSync('reports/crypto_comparison.json')) {
    console.log(chalk.yellow('⚠ No crypto comparison results found\n'));
    return;
  }

  const data = JSON.parse(await readFile('reports/crypto_comparison.json', 'utf-8'));

  console.log(chalk.cyan.bold('═══ Crypto Performance Analysis ═══\n'));

  // Overall statistics
  const speedups = data.benchmarks
    .filter(b => b.speedup)
    .map(b => b.speedup);

  if (speedups.length > 0) {
    const speedupStats = stats.analyzeSamples(speedups);

    console.log(chalk.white.bold('Overall Speedup Statistics:'));
    console.log(chalk.gray('  Mean:'), chalk.white(`${speedupStats.mean.toFixed(2)}x`));
    console.log(chalk.gray('  Median:'), chalk.white(`${speedupStats.median.toFixed(2)}x`));
    console.log(chalk.gray('  Min:'), chalk.white(`${speedupStats.min.toFixed(2)}x`));
    console.log(chalk.gray('  Max:'), chalk.white(`${speedupStats.max.toFixed(2)}x`));
    console.log(chalk.gray('  Std Dev:'), chalk.white(`±${speedupStats.stdDev.toFixed(2)}x`));
    console.log(chalk.gray('  95% CI:'), chalk.white(`${speedupStats.confidenceInterval.lower.toFixed(2)}x - ${speedupStats.confidenceInterval.upper.toFixed(2)}x`));
    console.log('');
  }

  // Top performers
  const topPerformers = data.benchmarks
    .filter(b => b.speedup)
    .sort((a, b) => b.speedup - a.speedup)
    .slice(0, 5);

  if (topPerformers.length > 0) {
    console.log(chalk.white.bold('Top 5 Native Performance Gains:'));
    topPerformers.forEach((bench, i) => {
      console.log(
        chalk.green(`  ${i + 1}.`),
        chalk.white(bench.name.padEnd(40)),
        chalk.yellow(`${bench.speedup.toFixed(2)}x faster`)
      );
    });
    console.log('');
  }

  // Analyze by operation type
  const mlKemBenches = data.benchmarks.filter(b => b.name.includes('ML-KEM'));
  const mlDsaBenches = data.benchmarks.filter(b => b.name.includes('ML-DSA'));
  const blake3Benches = data.benchmarks.filter(b => b.name.includes('BLAKE3'));

  console.log(chalk.white.bold('Analysis by Operation Type:'));

  if (mlKemBenches.length > 0) {
    const mlKemSpeedups = mlKemBenches.map(b => b.speedup).filter(Boolean);
    const avgSpeedup = stats.mean(mlKemSpeedups);
    console.log(
      chalk.gray('  ML-KEM-768:'),
      chalk.white(`${avgSpeedup.toFixed(2)}x average speedup`),
      chalk.gray(`(${mlKemBenches.length} operations)`)
    );
  }

  if (mlDsaBenches.length > 0) {
    const mlDsaSpeedups = mlDsaBenches.map(b => b.speedup).filter(Boolean);
    const avgSpeedup = stats.mean(mlDsaSpeedups);
    console.log(
      chalk.gray('  ML-DSA:'),
      chalk.white(`${avgSpeedup.toFixed(2)}x average speedup`),
      chalk.gray(`(${mlDsaBenches.length} operations)`)
    );
  }

  if (blake3Benches.length > 0) {
    const blake3Speedups = blake3Benches.map(b => b.speedup).filter(Boolean);
    const avgSpeedup = stats.mean(blake3Speedups);
    console.log(
      chalk.gray('  BLAKE3:'),
      chalk.white(`${avgSpeedup.toFixed(2)}x average speedup`),
      chalk.gray(`(${blake3Benches.length} operations)`)
    );
  }

  console.log('');

  // Performance recommendations
  console.log(chalk.white.bold('Recommendations:'));

  const avgSpeedup = stats.mean(speedups);
  if (avgSpeedup >= 3) {
    console.log(chalk.green('  ✓ Native implementation shows excellent performance (3x+ speedup)'));
    console.log(chalk.green('    Strongly recommend using NAPI-rs for Node.js environments'));
  } else if (avgSpeedup >= 2) {
    console.log(chalk.cyan('  ✓ Native implementation shows good performance (2-3x speedup)'));
    console.log(chalk.cyan('    Recommend using NAPI-rs for performance-critical paths'));
  } else {
    console.log(chalk.yellow('  ⚠ Native speedup is modest (<2x)'));
    console.log(chalk.yellow('    Consider WASM for better portability'));
  }

  console.log('');
}

async function analyzeOrchestratorResults() {
  if (!existsSync('reports/orchestrator_results.json')) {
    console.log(chalk.yellow('⚠ No orchestrator results found\n'));
    return;
  }

  const data = JSON.parse(await readFile('reports/orchestrator_results.json', 'utf-8'));

  console.log(chalk.cyan.bold('═══ Orchestrator Performance Analysis ═══\n'));

  const benchmarks = data.benchmarks;

  // Overall statistics
  const throughputs = benchmarks.map(b => b.hz);
  const latencies = benchmarks.map(b => b.mean);

  const throughputStats = stats.analyzeSamples(throughputs);
  const latencyStats = stats.analyzeSamples(latencies);

  console.log(chalk.white.bold('Overall Performance:'));
  console.log(chalk.gray('  Avg Throughput:'), chalk.white(stats.formatThroughput(throughputStats.mean)));
  console.log(chalk.gray('  Avg Latency:'), chalk.white(stats.formatDuration(latencyStats.mean)));
  console.log(chalk.gray('  Best Throughput:'), chalk.white(stats.formatThroughput(throughputStats.max)));
  console.log(chalk.gray('  Best Latency:'), chalk.white(stats.formatDuration(latencyStats.min)));
  console.log('');

  // Top performers
  const topPerformers = benchmarks
    .sort((a, b) => b.hz - a.hz)
    .slice(0, 5);

  console.log(chalk.white.bold('Top 5 Fastest Operations:'));
  topPerformers.forEach((bench, i) => {
    console.log(
      chalk.green(`  ${i + 1}.`),
      chalk.white(bench.name.padEnd(40)),
      chalk.yellow(stats.formatThroughput(bench.hz))
    );
  });

  console.log('');
}

async function analyzePrimeResults() {
  if (!existsSync('reports/prime_ml_results.json')) {
    console.log(chalk.yellow('⚠ No Prime ML results found\n'));
    return;
  }

  const data = JSON.parse(await readFile('reports/prime_ml_results.json', 'utf-8'));

  console.log(chalk.cyan.bold('═══ Prime ML Performance Analysis ═══\n'));

  const benchmarks = data.benchmarks;

  // Analyze gradient aggregation scalability
  const gradientBenches = benchmarks.filter(b => b.name.includes('Federated Avg'));

  if (gradientBenches.length > 0) {
    console.log(chalk.white.bold('Gradient Aggregation Scalability:'));

    // Group by node count
    const byNodes = {};
    gradientBenches.forEach(bench => {
      const match = bench.name.match(/(\d+) nodes/);
      if (match) {
        const nodes = parseInt(match[1]);
        if (!byNodes[nodes]) byNodes[nodes] = [];
        byNodes[nodes].push(bench);
      }
    });

    Object.entries(byNodes)
      .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
      .forEach(([nodes, benches]) => {
        const avgThroughput = stats.mean(benches.map(b => b.hz));
        console.log(
          chalk.gray(`  ${nodes} nodes:`),
          chalk.white(stats.formatThroughput(avgThroughput))
        );
      });

    console.log('');

    // Calculate scalability efficiency
    const nodeKeys = Object.keys(byNodes).map(Number).sort((a, b) => a - b);
    if (nodeKeys.length >= 2) {
      const minNodes = nodeKeys[0];
      const maxNodes = nodeKeys[nodeKeys.length - 1];

      const minThroughput = stats.mean(byNodes[minNodes].map(b => b.hz));
      const maxThroughput = stats.mean(byNodes[maxNodes].map(b => b.hz));

      const idealSpeedup = maxNodes / minNodes;
      const actualSpeedup = minThroughput / maxThroughput;
      const efficiency = (actualSpeedup / idealSpeedup) * 100;

      console.log(chalk.white.bold('Scalability Efficiency:'));
      console.log(chalk.gray('  Ideal speedup:'), chalk.white(`${idealSpeedup.toFixed(2)}x`));
      console.log(chalk.gray('  Actual speedup:'), chalk.white(`${actualSpeedup.toFixed(2)}x`));
      console.log(chalk.gray('  Efficiency:'), chalk.white(`${efficiency.toFixed(1)}%`));
      console.log('');
    }
  }

  // Analyze model update performance
  const updateBenches = benchmarks.filter(b => b.name.includes('Model Update'));

  if (updateBenches.length > 0) {
    console.log(chalk.white.bold('Model Update Performance:'));

    updateBenches.forEach(bench => {
      const match = bench.name.match(/(\d+) params/);
      const params = match ? match[1] : 'N/A';
      console.log(
        chalk.gray(`  ${params} params:`),
        chalk.white(stats.formatThroughput(bench.hz)),
        chalk.gray(`(${stats.formatDuration(bench.mean)})`)
      );
    });

    console.log('');
  }
}

async function main() {
  await analyzeCryptoResults();
  await analyzeOrchestratorResults();
  await analyzePrimeResults();

  console.log(chalk.blue.bold('Analysis Complete!\n'));
  console.log(chalk.gray('For detailed visualizations, run: npm run report:visualize\n'));
}

main().catch(err => {
  console.error(chalk.red('Error:'), err);
  process.exit(1);
});
