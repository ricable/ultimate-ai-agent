#!/usr/bin/env node
/**
 * Generate comprehensive HTML benchmark report
 * Combines results from all benchmarks with visualizations
 */

import { readFile, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import chalk from 'chalk';

console.log(chalk.blue.bold('\n📊 Generating Benchmark Report\n'));

// Load all benchmark results
async function loadResults() {
  const results = {
    crypto: null,
    cryptoComparison: null,
    orchestrator: null,
    prime: null
  };

  try {
    if (existsSync('reports/wasm_crypto_results.json')) {
      results.crypto = JSON.parse(await readFile('reports/wasm_crypto_results.json', 'utf-8'));
      console.log(chalk.green('✓'), 'Loaded crypto WASM results');
    }

    if (existsSync('reports/crypto_comparison.json')) {
      results.cryptoComparison = JSON.parse(await readFile('reports/crypto_comparison.json', 'utf-8'));
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

// Generate HTML report
function generateHTML(results) {
  const timestamp = new Date().toISOString();

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DAA Performance Benchmarks Report</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      line-height: 1.6;
      color: #333;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      padding: 20px;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      background: white;
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
      overflow: hidden;
    }

    header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 40px;
      text-align: center;
    }

    header h1 {
      font-size: 2.5rem;
      margin-bottom: 10px;
    }

    header p {
      font-size: 1.1rem;
      opacity: 0.9;
    }

    .meta {
      background: #f8f9fa;
      padding: 20px 40px;
      border-bottom: 1px solid #dee2e6;
    }

    .meta-item {
      display: inline-block;
      margin-right: 30px;
      font-size: 0.9rem;
      color: #6c757d;
    }

    .meta-item strong {
      color: #495057;
    }

    .content {
      padding: 40px;
    }

    .section {
      margin-bottom: 60px;
    }

    .section h2 {
      font-size: 2rem;
      color: #667eea;
      margin-bottom: 20px;
      padding-bottom: 10px;
      border-bottom: 3px solid #667eea;
    }

    .section h3 {
      font-size: 1.5rem;
      color: #495057;
      margin: 30px 0 15px 0;
    }

    .benchmark-table {
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      background: white;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
      border-radius: 8px;
      overflow: hidden;
    }

    .benchmark-table thead {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
    }

    .benchmark-table th,
    .benchmark-table td {
      padding: 15px;
      text-align: left;
      border-bottom: 1px solid #dee2e6;
    }

    .benchmark-table tbody tr:hover {
      background: #f8f9fa;
    }

    .benchmark-table tbody tr:last-child td {
      border-bottom: none;
    }

    .speedup {
      display: inline-block;
      padding: 4px 12px;
      border-radius: 20px;
      font-weight: bold;
      font-size: 0.9rem;
    }

    .speedup.good {
      background: #d4edda;
      color: #155724;
    }

    .speedup.excellent {
      background: #c3e6cb;
      color: #0a3d16;
    }

    .speedup.warning {
      background: #fff3cd;
      color: #856404;
    }

    .metric {
      font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
      background: #f8f9fa;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.9em;
    }

    .chart-container {
      margin: 30px 0;
      padding: 20px;
      background: #f8f9fa;
      border-radius: 8px;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 20px;
      margin: 30px 0;
    }

    .summary-card {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .summary-card h4 {
      font-size: 0.9rem;
      opacity: 0.9;
      margin-bottom: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .summary-card .value {
      font-size: 2.5rem;
      font-weight: bold;
      margin-bottom: 5px;
    }

    .summary-card .label {
      font-size: 0.9rem;
      opacity: 0.8;
    }

    footer {
      background: #f8f9fa;
      padding: 30px 40px;
      text-align: center;
      color: #6c757d;
      border-top: 1px solid #dee2e6;
    }

    .no-data {
      padding: 40px;
      text-align: center;
      color: #6c757d;
      background: #f8f9fa;
      border-radius: 8px;
      margin: 20px 0;
    }

    @media (max-width: 768px) {
      body {
        padding: 10px;
      }

      header h1 {
        font-size: 1.8rem;
      }

      .content {
        padding: 20px;
      }

      .benchmark-table {
        font-size: 0.9rem;
      }

      .summary-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🚀 DAA Performance Benchmarks</h1>
      <p>Comprehensive performance comparison of Native NAPI-rs vs WASM implementations</p>
    </header>

    <div class="meta">
      <div class="meta-item">
        <strong>Generated:</strong> ${new Date(timestamp).toLocaleString()}
      </div>
      <div class="meta-item">
        <strong>Platform:</strong> ${results.cryptoComparison?.platform?.platform || 'Unknown'} ${results.cryptoComparison?.platform?.arch || ''}
      </div>
      <div class="meta-item">
        <strong>Node.js:</strong> ${results.cryptoComparison?.platform?.node || 'N/A'}
      </div>
    </div>

    <div class="content">
      ${generateCryptoSection(results)}
      ${generateOrchestratorSection(results)}
      ${generatePrimeSection(results)}
    </div>

    <footer>
      <p>Generated by DAA Benchmarking Suite</p>
      <p>For more information, visit <a href="https://github.com/ruvnet/daa">github.com/ruvnet/daa</a></p>
    </footer>
  </div>
</body>
</html>`;
}

function generateCryptoSection(results) {
  if (!results.cryptoComparison?.benchmarks?.length) {
    return `
      <section class="section">
        <h2>🔐 Cryptographic Operations</h2>
        <div class="no-data">
          No crypto benchmark data available. Run: <code>npm run bench:crypto-compare</code>
        </div>
      </section>
    `;
  }

  const benchmarks = results.cryptoComparison.benchmarks;
  const avgSpeedup = benchmarks
    .filter(b => b.speedup)
    .reduce((sum, b) => sum + b.speedup, 0) / benchmarks.filter(b => b.speedup).length;

  const tableRows = benchmarks
    .map(bench => {
      const nativeHz = bench.implementations.Native?.hz.toFixed(2) || 'N/A';
      const wasmHz = bench.implementations.WASM?.hz.toFixed(2) || 'N/A';
      const nativeMean = bench.implementations.Native?.mean.toFixed(3) || 'N/A';
      const wasmMean = bench.implementations.WASM?.mean.toFixed(3) || 'N/A';
      const speedup = bench.speedup ? bench.speedup.toFixed(2) + 'x' : 'N/A';
      const speedupClass = bench.speedup >= 3 ? 'excellent' : bench.speedup >= 2 ? 'good' : 'warning';

      return `
        <tr>
          <td><strong>${bench.name}</strong></td>
          <td><span class="metric">${nativeHz} ops/sec</span><br><small>${nativeMean}ms</small></td>
          <td><span class="metric">${wasmHz} ops/sec</span><br><small>${wasmMean}ms</small></td>
          <td><span class="speedup ${speedupClass}">${speedup}</span></td>
        </tr>
      `;
    })
    .join('');

  return `
    <section class="section">
      <h2>🔐 Cryptographic Operations</h2>

      <div class="summary-grid">
        <div class="summary-card">
          <h4>Average Speedup</h4>
          <div class="value">${avgSpeedup.toFixed(2)}x</div>
          <div class="label">Native vs WASM</div>
        </div>
        <div class="summary-card">
          <h4>Total Benchmarks</h4>
          <div class="value">${benchmarks.length}</div>
          <div class="label">Operations Tested</div>
        </div>
        <div class="summary-card">
          <h4>Best Speedup</h4>
          <div class="value">${Math.max(...benchmarks.map(b => b.speedup || 0)).toFixed(2)}x</div>
          <div class="label">Peak Performance</div>
        </div>
      </div>

      <h3>Performance Comparison</h3>
      <table class="benchmark-table">
        <thead>
          <tr>
            <th>Operation</th>
            <th>Native (NAPI-rs)</th>
            <th>WASM</th>
            <th>Speedup</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
        </tbody>
      </table>
    </section>
  `;
}

function generateOrchestratorSection(results) {
  if (!results.orchestrator?.benchmarks?.length) {
    return `
      <section class="section">
        <h2>⚙️ Orchestrator Performance</h2>
        <div class="no-data">
          No orchestrator benchmark data available. Run: <code>npm run bench:orchestrator</code>
        </div>
      </section>
    `;
  }

  const benchmarks = results.orchestrator.benchmarks;
  const tableRows = benchmarks
    .map(bench => `
      <tr>
        <td><strong>${bench.name}</strong></td>
        <td><span class="metric">${bench.hz.toFixed(2)} ops/sec</span></td>
        <td><span class="metric">${bench.mean.toFixed(3)}ms</span></td>
        <td><span class="metric">±${bench.rme.toFixed(2)}%</span></td>
      </tr>
    `)
    .join('');

  return `
    <section class="section">
      <h2>⚙️ Orchestrator Performance</h2>

      <table class="benchmark-table">
        <thead>
          <tr>
            <th>Operation</th>
            <th>Throughput</th>
            <th>Mean Time</th>
            <th>RME</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
        </tbody>
      </table>
    </section>
  `;
}

function generatePrimeSection(results) {
  if (!results.prime?.benchmarks?.length) {
    return `
      <section class="section">
        <h2>🧠 Prime ML Performance</h2>
        <div class="no-data">
          No Prime ML benchmark data available. Run: <code>npm run bench:prime</code>
        </div>
      </section>
    `;
  }

  const benchmarks = results.prime.benchmarks;
  const tableRows = benchmarks
    .slice(0, 20) // Show top 20
    .map(bench => `
      <tr>
        <td><strong>${bench.name}</strong></td>
        <td><span class="metric">${bench.hz.toFixed(2)} ops/sec</span></td>
        <td><span class="metric">${bench.mean.toFixed(3)}ms</span></td>
        <td><span class="metric">±${bench.rme.toFixed(2)}%</span></td>
      </tr>
    `)
    .join('');

  return `
    <section class="section">
      <h2>🧠 Prime ML Performance</h2>

      <table class="benchmark-table">
        <thead>
          <tr>
            <th>Operation</th>
            <th>Throughput</th>
            <th>Mean Time</th>
            <th>RME</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
          ${benchmarks.length > 20 ? `<tr><td colspan="4" style="text-align: center; color: #6c757d;">... and ${benchmarks.length - 20} more operations</td></tr>` : ''}
        </tbody>
      </table>
    </section>
  `;
}

// Main execution
async function main() {
  const results = await loadResults();

  if (!results.crypto && !results.cryptoComparison && !results.orchestrator && !results.prime) {
    console.log(chalk.yellow('\n⚠️  No benchmark results found.'));
    console.log(chalk.gray('Run benchmarks first with:'));
    console.log(chalk.cyan('  npm run bench:all\n'));
    return;
  }

  const html = generateHTML(results);

  await writeFile('reports/benchmark_report.html', html);

  console.log(chalk.green('\n✓ Report generated successfully!'));
  console.log(chalk.cyan('📄 reports/benchmark_report.html\n'));
  console.log(chalk.gray('Open in browser to view the report.\n'));
}

main().catch(err => {
  console.error(chalk.red('Error:'), err);
  process.exit(1);
});
