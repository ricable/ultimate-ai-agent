#!/usr/bin/env node
/**
 * Generate performance visualization charts
 * Creates comparison graphs for benchmark results
 */

import { readFile, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import chalk from 'chalk';
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';

console.log(chalk.blue.bold('\n📈 Generating Performance Visualizations\n'));

const width = 1200;
const height = 600;
const chartCallback = (ChartJS) => {
  // Chart.js configuration
};

const chartJSNodeCanvas = new ChartJSNodeCanvas({ width, height, chartCallback });

// Load benchmark results
async function loadResults() {
  const results = {};

  try {
    if (existsSync('reports/crypto_comparison.json')) {
      results.crypto = JSON.parse(await readFile('reports/crypto_comparison.json', 'utf-8'));
      console.log(chalk.green('✓'), 'Loaded crypto comparison results');
    }

    if (existsSync('reports/orchestrator_results.json')) {
      results.orchestrator = JSON.parse(await readFile('reports/orchestrator_results.json', 'utf-8'));
      console.log(chalk.green('✓'), 'Loaded orchestrator results');
    }

    if (existsSync('reports/prime_ml_results.json')) {
      results.prime = JSON.parse(await readFile('reports/prime_ml_results.json', 'utf-8'));
      console.log(chalk.green('✓'), 'Loaded Prime ML results');
    }
  } catch (error) {
    console.error(chalk.red('Error loading results:'), error.message);
  }

  return results;
}

// Generate crypto speedup chart
async function generateCryptoSpeedupChart(data) {
  if (!data?.benchmarks?.length) return;

  const benchmarks = data.benchmarks.filter(b => b.speedup);
  const labels = benchmarks.map(b => b.name.replace(/^(ML-KEM-768|ML-DSA|BLAKE3|Quantum Fingerprint)\s*/, ''));
  const speedups = benchmarks.map(b => b.speedup);

  const configuration = {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Native Speedup vs WASM',
        data: speedups,
        backgroundColor: speedups.map(s =>
          s >= 3 ? 'rgba(40, 167, 69, 0.8)' :
            s >= 2 ? 'rgba(0, 123, 255, 0.8)' :
              'rgba(255, 193, 7, 0.8)'
        ),
        borderColor: speedups.map(s =>
          s >= 3 ? 'rgb(40, 167, 69)' :
            s >= 2 ? 'rgb(0, 123, 255)' :
              'rgb(255, 193, 7)'
        ),
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Native vs WASM Performance Speedup',
          font: { size: 24 }
        },
        legend: {
          display: true,
          position: 'bottom'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Speedup (x times faster)'
          }
        },
        x: {
          ticks: {
            maxRotation: 45,
            minRotation: 45
          }
        }
      }
    }
  };

  const image = await chartJSNodeCanvas.renderToBuffer(configuration);
  await writeFile('reports/crypto_speedup_chart.png', image);
  console.log(chalk.green('✓'), 'Generated crypto speedup chart');
}

// Generate crypto throughput comparison chart
async function generateCryptoThroughputChart(data) {
  if (!data?.benchmarks?.length) return;

  const benchmarks = data.benchmarks.slice(0, 10); // Top 10
  const labels = benchmarks.map(b => b.name.split(' ').slice(0, 3).join(' '));

  const configuration = {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Native (ops/sec)',
          data: benchmarks.map(b => b.implementations.Native?.hz || 0),
          backgroundColor: 'rgba(102, 126, 234, 0.8)',
          borderColor: 'rgb(102, 126, 234)',
          borderWidth: 2
        },
        {
          label: 'WASM (ops/sec)',
          data: benchmarks.map(b => b.implementations.WASM?.hz || 0),
          backgroundColor: 'rgba(118, 75, 162, 0.8)',
          borderColor: 'rgb(118, 75, 162)',
          borderWidth: 2
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Throughput Comparison: Native vs WASM',
          font: { size: 24 }
        },
        legend: {
          display: true,
          position: 'bottom'
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Operations per second'
          }
        },
        x: {
          ticks: {
            maxRotation: 45,
            minRotation: 45
          }
        }
      }
    }
  };

  const image = await chartJSNodeCanvas.renderToBuffer(configuration);
  await writeFile('reports/crypto_throughput_chart.png', image);
  console.log(chalk.green('✓'), 'Generated crypto throughput chart');
}

// Generate orchestrator performance chart
async function generateOrchestratorChart(data) {
  if (!data?.benchmarks?.length) return;

  const benchmarks = data.benchmarks.slice(0, 10);
  const labels = benchmarks.map(b => b.name);
  const throughputs = benchmarks.map(b => b.hz);

  const configuration = {
    type: 'horizontalBar',
    data: {
      labels,
      datasets: [{
        label: 'Throughput (ops/sec)',
        data: throughputs,
        backgroundColor: 'rgba(102, 126, 234, 0.8)',
        borderColor: 'rgb(102, 126, 234)',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Orchestrator Performance',
          font: { size: 24 }
        }
      },
      scales: {
        x: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Operations per second'
          }
        }
      }
    }
  };

  const image = await chartJSNodeCanvas.renderToBuffer(configuration);
  await writeFile('reports/orchestrator_chart.png', image);
  console.log(chalk.green('✓'), 'Generated orchestrator chart');
}

// Generate Prime ML performance chart
async function generatePrimeChart(data) {
  if (!data?.benchmarks?.length) return;

  // Group by operation type
  const gradientBenches = data.benchmarks.filter(b => b.name.includes('Federated Avg'));
  const updateBenches = data.benchmarks.filter(b => b.name.includes('Model Update'));

  const labels = gradientBenches.map(b => b.name.match(/(\d+) nodes/)?.[1] || 'N/A');
  const throughputs = gradientBenches.map(b => b.hz);

  const configuration = {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Gradient Aggregation Throughput',
        data: throughputs,
        borderColor: 'rgb(102, 126, 234)',
        backgroundColor: 'rgba(102, 126, 234, 0.2)',
        borderWidth: 3,
        tension: 0.4,
        fill: true
      }]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Prime ML: Gradient Aggregation Scalability',
          font: { size: 24 }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Operations per second'
          }
        },
        x: {
          title: {
            display: true,
            text: 'Number of Nodes'
          }
        }
      }
    }
  };

  const image = await chartJSNodeCanvas.renderToBuffer(configuration);
  await writeFile('reports/prime_ml_chart.png', image);
  console.log(chalk.green('✓'), 'Generated Prime ML chart');
}

// Main execution
async function main() {
  const results = await loadResults();

  if (!results.crypto && !results.orchestrator && !results.prime) {
    console.log(chalk.yellow('\n⚠️  No benchmark results found.'));
    console.log(chalk.gray('Run benchmarks first with:'));
    console.log(chalk.cyan('  npm run bench:all\n'));
    return;
  }

  // Generate all charts
  await generateCryptoSpeedupChart(results.crypto);
  await generateCryptoThroughputChart(results.crypto);
  await generateOrchestratorChart(results.orchestrator);
  await generatePrimeChart(results.prime);

  console.log(chalk.green('\n✓ Visualizations generated successfully!'));
  console.log(chalk.cyan('📊 Charts saved to reports/ directory\n'));
}

main().catch(err => {
  console.error(chalk.red('Error:'), err);
  process.exit(1);
});
