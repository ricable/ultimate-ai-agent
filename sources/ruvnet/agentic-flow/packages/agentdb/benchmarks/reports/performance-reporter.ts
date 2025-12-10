/**
 * Performance Reporter
 *
 * Generates comprehensive performance reports in multiple formats
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import type { BenchmarkResult } from '../benchmark-runner';

export class PerformanceReporter {
  private readonly reportsDir = path.join(__dirname, '..');

  /**
   * Generate HTML report
   */
  async generateHTMLReport(results: BenchmarkResult[]): Promise<void> {
    const html = this.buildHTMLReport(results);
    await fs.writeFile(
      path.join(this.reportsDir, 'reports', 'performance-report.html'),
      html,
      'utf-8'
    );
  }

  /**
   * Generate JSON report
   */
  async generateJSONReport(results: BenchmarkResult[]): Promise<void> {
    const report = {
      generatedAt: new Date().toISOString(),
      summary: this.generateSummary(results),
      results: results,
      bottlenecks: this.identifyBottlenecks(results),
      recommendations: this.generateRecommendations(results)
    };

    await fs.writeFile(
      path.join(this.reportsDir, 'reports', 'performance-report.json'),
      JSON.stringify(report, null, 2),
      'utf-8'
    );
  }

  /**
   * Generate Markdown report
   */
  async generateMarkdownReport(results: BenchmarkResult[]): Promise<void> {
    const md = this.buildMarkdownReport(results);
    await fs.writeFile(
      path.join(this.reportsDir, 'reports', 'performance-report.md'),
      md,
      'utf-8'
    );
  }

  /**
   * Generate summary statistics
   */
  private generateSummary(results: BenchmarkResult[]): any {
    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);

    const categories = new Map<string, BenchmarkResult[]>();
    for (const result of successful) {
      if (!categories.has(result.category)) {
        categories.set(result.category, []);
      }
      categories.get(result.category)!.push(result);
    }

    return {
      totalBenchmarks: results.length,
      successful: successful.length,
      failed: failed.length,
      successRate: ((successful.length / results.length) * 100).toFixed(2) + '%',
      categorySummary: Array.from(categories.entries()).map(([category, items]) => ({
        category,
        count: items.length,
        avgDuration: (items.reduce((sum, r) => sum + r.duration, 0) / items.length).toFixed(2)
      }))
    };
  }

  /**
   * Identify performance bottlenecks
   */
  private identifyBottlenecks(results: BenchmarkResult[]): any[] {
    const bottlenecks: any[] = [];

    // Find slowest operations
    const sorted = results
      .filter(r => r.success)
      .sort((a, b) => b.duration - a.duration);

    if (sorted.length > 0) {
      const slowest = sorted[0];
      if (slowest.duration > 10000) {
        bottlenecks.push({
          type: 'Slow Operation',
          name: slowest.name,
          duration: slowest.duration.toFixed(2) + 'ms',
          severity: 'HIGH'
        });
      }
    }

    // Find low throughput operations
    for (const result of results) {
      if (result.success && result.operationsPerSecond && result.operationsPerSecond < 10) {
        bottlenecks.push({
          type: 'Low Throughput',
          name: result.name,
          opsPerSec: result.operationsPerSecond.toFixed(2),
          severity: 'MEDIUM'
        });
      }
    }

    return bottlenecks;
  }

  /**
   * Generate optimization recommendations
   */
  private generateRecommendations(results: BenchmarkResult[]): string[] {
    const recommendations: string[] = [];

    // Check vector search results
    const vectorResults = results.filter(r => r.category === 'Vector Search' && r.success);
    const hnswResult = vectorResults.find(r => r.name.includes('HNSW'));
    const withoutHNSW = vectorResults.find(r => r.name.includes('without HNSW'));

    if (hnswResult && withoutHNSW && hnswResult.operationsPerSecond && withoutHNSW.operationsPerSecond) {
      const speedup = withoutHNSW.duration / hnswResult.duration;
      if (speedup > 10) {
        recommendations.push(`✅ HNSW indexing provides ${speedup.toFixed(0)}x speedup - highly recommended for production`);
      }
    }

    // Check batch operations
    const batchComparison = results.find(r => r.name === 'Batch vs Individual Comparison' && r.success);
    if (batchComparison && batchComparison.metrics.speedupFactor > 2) {
      recommendations.push(`✅ Batch operations are ${batchComparison.metrics.speedupFactor}x faster - use batch inserts for bulk data`);
    }

    // Check quantization
    const quantResults = results.filter(r => r.category === 'Quantization' && r.success);
    const memoryReduction = quantResults.find(r => r.name === 'Memory Reduction Comparison');
    if (memoryReduction && memoryReduction.metrics.reduction4BitPercent) {
      const reduction = parseFloat(memoryReduction.metrics.reduction4BitPercent);
      if (reduction > 50) {
        recommendations.push(`✅ 4-bit quantization reduces memory by ${reduction.toFixed(0)}% - recommended for large datasets`);
      }
    }

    // Check database backend
    const backendComparison = results.find(r => r.name === 'Backend Comparison' && r.success);
    if (backendComparison) {
      recommendations.push(`✅ ${backendComparison.metrics.recommendation}`);
    }

    // General recommendations
    if (recommendations.length === 0) {
      recommendations.push('No specific optimizations identified - performance is within normal ranges');
    }

    return recommendations;
  }

  /**
   * Build HTML report
   */
  private buildHTMLReport(results: BenchmarkResult[]): string {
    const summary = this.generateSummary(results);
    const bottlenecks = this.identifyBottlenecks(results);
    const recommendations = this.generateRecommendations(results);

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AgentDB Performance Report</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
    h2 { color: #555; margin-top: 30px; }
    .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
    .stat-card { background: #f9f9f9; padding: 20px; border-radius: 4px; border-left: 4px solid #4CAF50; }
    .stat-card h3 { margin: 0 0 10px 0; color: #666; font-size: 14px; }
    .stat-card .value { font-size: 32px; font-weight: bold; color: #333; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #4CAF50; color: white; font-weight: 600; }
    tr:hover { background: #f5f5f5; }
    .success { color: #4CAF50; }
    .failure { color: #f44336; }
    .metric { background: #f0f0f0; padding: 4px 8px; border-radius: 4px; font-family: monospace; font-size: 12px; }
    .bottleneck { background: #fff3cd; padding: 15px; margin: 10px 0; border-left: 4px solid #ffc107; border-radius: 4px; }
    .recommendation { background: #d4edda; padding: 15px; margin: 10px 0; border-left: 4px solid #28a745; border-radius: 4px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🚀 AgentDB Performance Report</h1>
    <p>Generated: ${new Date().toISOString()}</p>

    <h2>📊 Summary</h2>
    <div class="summary">
      <div class="stat-card">
        <h3>Total Benchmarks</h3>
        <div class="value">${summary.totalBenchmarks}</div>
      </div>
      <div class="stat-card">
        <h3>Successful</h3>
        <div class="value success">${summary.successful}</div>
      </div>
      <div class="stat-card">
        <h3>Failed</h3>
        <div class="value failure">${summary.failed}</div>
      </div>
      <div class="stat-card">
        <h3>Success Rate</h3>
        <div class="value">${summary.successRate}</div>
      </div>
    </div>

    <h2>📈 Benchmark Results</h2>
    <table>
      <thead>
        <tr>
          <th>Benchmark</th>
          <th>Category</th>
          <th>Duration (ms)</th>
          <th>Ops/Sec</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        ${results.map(r => `
          <tr>
            <td>${r.name}</td>
            <td>${r.category}</td>
            <td>${r.duration.toFixed(2)}</td>
            <td>${r.operationsPerSecond ? r.operationsPerSecond.toFixed(0) : 'N/A'}</td>
            <td class="${r.success ? 'success' : 'failure'}">${r.success ? '✅ Success' : '❌ Failed'}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>

    ${bottlenecks.length > 0 ? `
      <h2>⚠️ Bottlenecks Identified</h2>
      ${bottlenecks.map(b => `
        <div class="bottleneck">
          <strong>${b.type}:</strong> ${b.name} - ${b.duration || b.opsPerSec} (${b.severity} severity)
        </div>
      `).join('')}
    ` : ''}

    <h2>💡 Recommendations</h2>
    ${recommendations.map(r => `
      <div class="recommendation">${r}</div>
    `).join('')}
  </div>
</body>
</html>`;
  }

  /**
   * Build Markdown report
   */
  private buildMarkdownReport(results: BenchmarkResult[]): string {
    const summary = this.generateSummary(results);
    const bottlenecks = this.identifyBottlenecks(results);
    const recommendations = this.generateRecommendations(results);

    let md = `# AgentDB Performance Report\n\n`;
    md += `**Generated:** ${new Date().toISOString()}\n\n`;

    md += `## Summary\n\n`;
    md += `- **Total Benchmarks:** ${summary.totalBenchmarks}\n`;
    md += `- **Successful:** ${summary.successful}\n`;
    md += `- **Failed:** ${summary.failed}\n`;
    md += `- **Success Rate:** ${summary.successRate}\n\n`;

    md += `## Benchmark Results\n\n`;
    md += `| Benchmark | Category | Duration (ms) | Ops/Sec | Status |\n`;
    md += `|-----------|----------|---------------|---------|--------|\n`;

    for (const result of results) {
      const status = result.success ? '✅ Success' : '❌ Failed';
      const opsPerSec = result.operationsPerSecond ? result.operationsPerSecond.toFixed(0) : 'N/A';
      md += `| ${result.name} | ${result.category} | ${result.duration.toFixed(2)} | ${opsPerSec} | ${status} |\n`;
    }

    if (bottlenecks.length > 0) {
      md += `\n## Bottlenecks Identified\n\n`;
      for (const bottleneck of bottlenecks) {
        md += `- **${bottleneck.type}:** ${bottleneck.name} - ${bottleneck.duration || bottleneck.opsPerSec} (${bottleneck.severity} severity)\n`;
      }
    }

    md += `\n## Recommendations\n\n`;
    for (const rec of recommendations) {
      md += `${rec}\n\n`;
    }

    return md;
  }
}
