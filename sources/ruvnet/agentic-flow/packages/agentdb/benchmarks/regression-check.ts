/**
 * AgentDB v2 Regression Detection
 * Compares current benchmark results against baseline to detect performance regressions
 */

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import type { BenchmarkResult } from './runner.js';

export interface BaselineData {
  version: string;
  timestamp: string;
  platform: string;
  node: string;
  baselines: {
    ruvector: Record<string, { p50Ms: number; p99Ms: number; memoryMB?: number }>;
    hnswlib: Record<string, { p50Ms: number; p99Ms: number; memoryMB?: number }>;
  };
  thresholds: {
    regressionPercent: number;
    criticalRegressionPercent: number;
    memoryRegressionPercent: number;
  };
}

export interface RegressionResult {
  benchmark: string;
  metric: 'p50' | 'p99' | 'memory';
  current: number;
  baseline: number;
  changePercent: number;
  changeMs?: number;
  status: 'pass' | 'improvement' | 'warning' | 'fail';
  message: string;
}

export interface RegressionReport {
  summary: {
    total: number;
    passed: number;
    improved: number;
    warnings: number;
    failed: number;
  };
  regressions: RegressionResult[];
  timestamp: string;
  platform: string;
  nodeVersion: string;
}

/**
 * Load baseline data from JSON file
 */
export function loadBaseline(baselinePath?: string): BaselineData {
  const path = baselinePath || join(__dirname, 'baseline.json');

  if (!existsSync(path)) {
    throw new Error(`Baseline file not found: ${path}`);
  }

  const content = readFileSync(path, 'utf-8');
  return JSON.parse(content) as BaselineData;
}

/**
 * Check for performance regressions
 */
export function checkRegression(
  results: BenchmarkResult[],
  backend: 'ruvector' | 'hnswlib',
  baselineData?: BaselineData
): RegressionReport {
  const baseline = baselineData || loadBaseline();
  const backendBaseline = baseline.baselines[backend];
  const regressions: RegressionResult[] = [];

  for (const result of results) {
    const base = backendBaseline[result.name];
    if (!base) {
      console.warn(`No baseline found for benchmark: ${result.name}`);
      continue;
    }

    // Check p50 latency
    const p50Regression = checkMetricRegression(
      result.name,
      'p50',
      result.p50Ms,
      base.p50Ms,
      baseline.thresholds
    );
    if (p50Regression) {
      regressions.push(p50Regression);
    }

    // Check p99 latency
    const p99Regression = checkMetricRegression(
      result.name,
      'p99',
      result.p99Ms,
      base.p99Ms,
      baseline.thresholds
    );
    if (p99Regression) {
      regressions.push(p99Regression);
    }

    // Check memory if available
    if (result.memoryMB && base.memoryMB) {
      const memoryRegression = checkMetricRegression(
        result.name,
        'memory',
        result.memoryMB,
        base.memoryMB,
        baseline.thresholds,
        true
      );
      if (memoryRegression) {
        regressions.push(memoryRegression);
      }
    }
  }

  // Calculate summary
  const summary = {
    total: regressions.length,
    passed: regressions.filter(r => r.status === 'pass').length,
    improved: regressions.filter(r => r.status === 'improvement').length,
    warnings: regressions.filter(r => r.status === 'warning').length,
    failed: regressions.filter(r => r.status === 'fail').length
  };

  return {
    summary,
    regressions,
    timestamp: new Date().toISOString(),
    platform: process.platform,
    nodeVersion: process.version
  };
}

/**
 * Check regression for a single metric
 */
function checkMetricRegression(
  benchmark: string,
  metric: 'p50' | 'p99' | 'memory',
  current: number,
  baseline: number,
  thresholds: BaselineData['thresholds'],
  isMemory = false
): RegressionResult | null {
  const changePercent = ((current - baseline) / baseline) * 100;
  const changeMs = current - baseline;

  let status: RegressionResult['status'] = 'pass';
  let message = '';

  const regressionThreshold = isMemory
    ? thresholds.memoryRegressionPercent
    : thresholds.regressionPercent;
  const criticalThreshold = thresholds.criticalRegressionPercent;

  if (changePercent < -5) {
    // Significant improvement
    status = 'improvement';
    message = `${Math.abs(changePercent).toFixed(1)}% improvement`;
  } else if (changePercent > criticalThreshold) {
    // Critical regression
    status = 'fail';
    message = `CRITICAL: ${changePercent.toFixed(1)}% regression (threshold: ${criticalThreshold}%)`;
  } else if (changePercent > regressionThreshold) {
    // Warning regression
    status = 'warning';
    message = `${changePercent.toFixed(1)}% regression (threshold: ${regressionThreshold}%)`;
  } else {
    // Within acceptable range
    status = 'pass';
    message = `${Math.abs(changePercent).toFixed(1)}% ${changePercent >= 0 ? 'slower' : 'faster'}`;
  }

  return {
    benchmark,
    metric,
    current,
    baseline,
    changePercent,
    changeMs: isMemory ? undefined : changeMs,
    status,
    message
  };
}

/**
 * Format regression report as console output
 */
export function formatRegressionReport(report: RegressionReport): string {
  const lines: string[] = [];

  lines.push('');
  lines.push('╔══════════════════════════════════════════════════════════════════════╗');
  lines.push('║                  AgentDB v2 Regression Report                        ║');
  lines.push('╠══════════════════════════════════════════════════════════════════════╣');
  lines.push(`║ Platform: ${report.platform.padEnd(12)} │ Node: ${report.nodeVersion.padEnd(12)} │ Date: ${new Date(report.timestamp).toLocaleDateString().padEnd(10)} ║`);
  lines.push('╚══════════════════════════════════════════════════════════════════════╝');
  lines.push('');

  // Summary
  lines.push('┌─────────────────────────────────────────────────────────────────────┐');
  lines.push('│ Summary                                                             │');
  lines.push('├───────────────────┬─────────────────────────────────────────────────┤');
  lines.push(`│ Total Checks      │ ${report.summary.total.toString().padStart(6)}                                       │`);
  lines.push(`│ ✅ Passed         │ ${report.summary.passed.toString().padStart(6)}                                       │`);
  lines.push(`│ ⬆️  Improved       │ ${report.summary.improved.toString().padStart(6)}                                       │`);
  lines.push(`│ ⚠️  Warnings       │ ${report.summary.warnings.toString().padStart(6)}                                       │`);
  lines.push(`│ ❌ Failed         │ ${report.summary.failed.toString().padStart(6)}                                       │`);
  lines.push('└───────────────────┴─────────────────────────────────────────────────┘');
  lines.push('');

  // Details
  if (report.regressions.length > 0) {
    lines.push('┌────────────────────────────────────────────────────────────────────────┐');
    lines.push('│ Regression Details                                                     │');
    lines.push('├─────────────────────────┬──────────┬──────────┬───────────┬───────────┤');
    lines.push('│ Benchmark               │ Metric   │ Current  │ Baseline  │ Change    │');
    lines.push('├─────────────────────────┼──────────┼──────────┼───────────┼───────────┤');

    // Sort by status priority (fail > warning > pass > improvement)
    const statusPriority = { fail: 0, warning: 1, pass: 2, improvement: 3 };
    const sorted = [...report.regressions].sort((a, b) =>
      statusPriority[a.status] - statusPriority[b.status]
    );

    for (const reg of sorted) {
      const name = reg.benchmark.substring(0, 23).padEnd(23);
      const metric = reg.metric.padEnd(8);
      const current = reg.current.toFixed(2).padStart(8);
      const baseline = reg.baseline.toFixed(2).padStart(9);

      let statusIcon = '';
      switch (reg.status) {
        case 'fail': statusIcon = '❌'; break;
        case 'warning': statusIcon = '⚠️'; break;
        case 'improvement': statusIcon = '⬆️'; break;
        case 'pass': statusIcon = '✅'; break;
      }

      const change = `${reg.changePercent >= 0 ? '+' : ''}${reg.changePercent.toFixed(1)}%`.padStart(9);

      lines.push(`│ ${name} │ ${metric} │ ${current} │ ${baseline} │ ${change} ${statusIcon} │`);
    }

    lines.push('└─────────────────────────┴──────────┴──────────┴───────────┴───────────┘');
  }

  lines.push('');

  // Overall status
  const overallStatus = report.summary.failed > 0 ? 'FAILED ❌' :
                       report.summary.warnings > 0 ? 'PASSED WITH WARNINGS ⚠️' :
                       'PASSED ✅';

  lines.push(`Overall Status: ${overallStatus}`);
  lines.push('');

  return lines.join('\n');
}

/**
 * Export regression report to JSON
 */
export function exportRegressionReport(report: RegressionReport, outputPath: string): void {
  const fs = require('fs');
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
  console.log(`Regression report exported to: ${outputPath}`);
}

/**
 * Compare two benchmark result sets
 */
export function compareBenchmarks(
  current: BenchmarkResult[],
  previous: BenchmarkResult[]
): RegressionResult[] {
  const regressions: RegressionResult[] = [];

  for (const currentResult of current) {
    const prevResult = previous.find(r => r.name === currentResult.name);
    if (!prevResult) continue;

    const p50Change = ((currentResult.p50Ms - prevResult.p50Ms) / prevResult.p50Ms) * 100;
    const p99Change = ((currentResult.p99Ms - prevResult.p99Ms) / prevResult.p99Ms) * 100;

    if (Math.abs(p50Change) > 5) {
      regressions.push({
        benchmark: currentResult.name,
        metric: 'p50',
        current: currentResult.p50Ms,
        baseline: prevResult.p50Ms,
        changePercent: p50Change,
        changeMs: currentResult.p50Ms - prevResult.p50Ms,
        status: p50Change > 10 ? 'warning' : p50Change < -5 ? 'improvement' : 'pass',
        message: `${Math.abs(p50Change).toFixed(1)}% ${p50Change >= 0 ? 'regression' : 'improvement'}`
      });
    }

    if (Math.abs(p99Change) > 5) {
      regressions.push({
        benchmark: currentResult.name,
        metric: 'p99',
        current: currentResult.p99Ms,
        baseline: prevResult.p99Ms,
        changePercent: p99Change,
        changeMs: currentResult.p99Ms - prevResult.p99Ms,
        status: p99Change > 10 ? 'warning' : p99Change < -5 ? 'improvement' : 'pass',
        message: `${Math.abs(p99Change).toFixed(1)}% ${p99Change >= 0 ? 'regression' : 'improvement'}`
      });
    }
  }

  return regressions;
}
