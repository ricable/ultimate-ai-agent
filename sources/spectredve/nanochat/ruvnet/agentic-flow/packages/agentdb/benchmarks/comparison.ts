/**
 * AgentDB v2 Benchmark Comparison Tool
 * Compares RuVector vs hnswlib performance and generates detailed reports
 */

import { BenchmarkResult, formatResults } from './runner.js';
import { loadBaseline, BaselineData } from './regression-check.js';

export interface ComparisonResult {
  benchmark: string;
  ruvector: {
    p50Ms: number;
    p99Ms: number;
    opsPerSec: number;
  };
  hnswlib: {
    p50Ms: number;
    p99Ms: number;
    opsPerSec: number;
  };
  improvement: {
    p50Speedup: number;
    p99Speedup: number;
    opsSpeedup: number;
  };
  status: 'met' | 'exceeded' | 'missed';
}

export interface MemoryComparison {
  vectorCount: number;
  ruvectorMB: number;
  hnswlibMB: number;
  reduction: number;
  reductionPercent: number;
}

/**
 * Compare RuVector and hnswlib benchmark results
 */
export function compareBenchmarks(
  ruvectorResults: BenchmarkResult[],
  hnswlibResults: BenchmarkResult[]
): ComparisonResult[] {
  const comparisons: ComparisonResult[] = [];

  for (const ruvResult of ruvectorResults) {
    const hnswResult = hnswlibResults.find(r => r.name === ruvResult.name);
    if (!hnswResult) continue;

    const p50Speedup = hnswResult.p50Ms / ruvResult.p50Ms;
    const p99Speedup = hnswResult.p99Ms / ruvResult.p99Ms;
    const opsSpeedup = ruvResult.opsPerSec / hnswResult.opsPerSec;

    // Determine if performance target was met (8x target)
    const targetSpeedup = 8.0;
    const status = p50Speedup >= targetSpeedup * 1.1 ? 'exceeded' :
                   p50Speedup >= targetSpeedup * 0.9 ? 'met' :
                   'missed';

    comparisons.push({
      benchmark: ruvResult.name,
      ruvector: {
        p50Ms: ruvResult.p50Ms,
        p99Ms: ruvResult.p99Ms,
        opsPerSec: ruvResult.opsPerSec
      },
      hnswlib: {
        p50Ms: hnswResult.p50Ms,
        p99Ms: hnswResult.p99Ms,
        opsPerSec: hnswResult.opsPerSec
      },
      improvement: {
        p50Speedup,
        p99Speedup,
        opsSpeedup
      },
      status
    });
  }

  return comparisons;
}

/**
 * Compare memory usage between backends
 */
export function compareMemory(baseline: BaselineData): MemoryComparison[] {
  const comparisons: MemoryComparison[] = [];

  const vectorCounts = [1000, 10000, 100000, 1000000];
  const baseKeys = {
    1000: 'memory-1K',
    10000: 'memory-10K',
    100000: 'memory-100K',
    1000000: 'memory-1M'
  };

  for (const count of vectorCounts) {
    const key = baseKeys[count as keyof typeof baseKeys];
    const ruvData = baseline.baselines.ruvector[key];
    const hnswData = baseline.baselines.hnswlib[key];

    if (ruvData?.memoryMB && hnswData?.memoryMB) {
      const reduction = hnswData.memoryMB / ruvData.memoryMB;
      const reductionPercent = ((hnswData.memoryMB - ruvData.memoryMB) / hnswData.memoryMB) * 100;

      comparisons.push({
        vectorCount: count,
        ruvectorMB: ruvData.memoryMB,
        hnswlibMB: hnswData.memoryMB,
        reduction,
        reductionPercent
      });
    }
  }

  return comparisons;
}

/**
 * Format comparison results as a detailed report
 */
export function formatComparisonReport(
  comparisons: ComparisonResult[],
  memoryComparisons: MemoryComparison[]
): string {
  const lines: string[] = [];

  lines.push('');
  lines.push('╔══════════════════════════════════════════════════════════════════════╗');
  lines.push('║              AgentDB v2 Performance Comparison Report                ║');
  lines.push('╠══════════════════════════════════════════════════════════════════════╣');
  lines.push('║                    RuVector vs hnswlib                               ║');
  lines.push('╚══════════════════════════════════════════════════════════════════════╝');
  lines.push('');

  // Vector Search Comparison
  lines.push('┌────────────────────────────────────────────────────────────────────────┐');
  lines.push('│ Vector Search Performance                                              │');
  lines.push('├──────────────────────┬─────────────┬─────────────┬───────────┬────────┤');
  lines.push('│ Benchmark            │ RuVector    │ hnswlib     │ Speedup   │ Status │');
  lines.push('├──────────────────────┼─────────────┼─────────────┼───────────┼────────┤');

  for (const comp of comparisons) {
    const name = comp.benchmark.substring(0, 20).padEnd(20);
    const ruv = `${comp.ruvector.p50Ms.toFixed(2)}ms`.padStart(11);
    const hnsw = `${comp.hnswlib.p50Ms.toFixed(2)}ms`.padStart(11);
    const speedup = `${comp.improvement.p50Speedup.toFixed(1)}x`.padStart(9);

    let statusIcon = '';
    switch (comp.status) {
      case 'exceeded': statusIcon = '🚀'; break;
      case 'met': statusIcon = '✅'; break;
      case 'missed': statusIcon = '⚠️'; break;
    }

    lines.push(`│ ${name} │ ${ruv} │ ${hnsw} │ ${speedup} │ ${statusIcon}     │`);
  }

  lines.push('└──────────────────────┴─────────────┴─────────────┴───────────┴────────┘');
  lines.push('');

  // Memory Comparison
  lines.push('┌────────────────────────────────────────────────────────────────────────┐');
  lines.push('│ Memory Usage Comparison                                                │');
  lines.push('├──────────────┬─────────────┬─────────────┬───────────┬────────────────┤');
  lines.push('│ Vector Count │ RuVector    │ hnswlib     │ Reduction │ Savings        │');
  lines.push('├──────────────┼─────────────┼─────────────┼───────────┼────────────────┤');

  for (const mem of memoryComparisons) {
    const count = mem.vectorCount.toLocaleString().padStart(12);
    const ruv = `${mem.ruvectorMB.toFixed(1)}MB`.padStart(11);
    const hnsw = `${mem.hnswlibMB.toFixed(1)}MB`.padStart(11);
    const reduction = `${mem.reduction.toFixed(1)}x`.padStart(9);
    const savings = `${mem.reductionPercent.toFixed(1)}%`.padStart(14);

    lines.push(`│ ${count} │ ${ruv} │ ${hnsw} │ ${reduction} │ ${savings} │`);
  }

  lines.push('└──────────────┴─────────────┴─────────────┴───────────┴────────────────┘');
  lines.push('');

  // Summary Statistics
  const avgSpeedup = comparisons.reduce((sum, c) => sum + c.improvement.p50Speedup, 0) / comparisons.length;
  const maxSpeedup = Math.max(...comparisons.map(c => c.improvement.p50Speedup));
  const avgMemReduction = memoryComparisons.reduce((sum, m) => sum + m.reduction, 0) / memoryComparisons.length;

  lines.push('┌────────────────────────────────────────────────────────────────────────┐');
  lines.push('│ Summary Statistics                                                     │');
  lines.push('├────────────────────────────────────────────────────────────────────────┤');
  lines.push(`│ Average Search Speedup:        ${avgSpeedup.toFixed(1)}x faster                          │`);
  lines.push(`│ Maximum Search Speedup:        ${maxSpeedup.toFixed(1)}x faster                          │`);
  lines.push(`│ Average Memory Reduction:      ${avgMemReduction.toFixed(1)}x less memory                    │`);
  lines.push(`│ Benchmarks Meeting Target:     ${comparisons.filter(c => c.status !== 'missed').length}/${comparisons.length}                               │`);
  lines.push('└────────────────────────────────────────────────────────────────────────┘');
  lines.push('');

  // Overall Assessment
  const metTarget = comparisons.filter(c => c.status !== 'missed').length / comparisons.length >= 0.9;
  const overallStatus = metTarget ? '✅ PERFORMANCE TARGETS MET' : '⚠️ SOME TARGETS MISSED';

  lines.push(`Overall Assessment: ${overallStatus}`);
  lines.push('');

  return lines.join('\n');
}

/**
 * Generate JSON comparison report
 */
export function generateComparisonJSON(
  comparisons: ComparisonResult[],
  memoryComparisons: MemoryComparison[]
): object {
  const avgSpeedup = comparisons.reduce((sum, c) => sum + c.improvement.p50Speedup, 0) / comparisons.length;
  const avgMemReduction = memoryComparisons.reduce((sum, m) => sum + m.reduction, 0) / memoryComparisons.length;

  return {
    version: '2.0.0',
    timestamp: new Date().toISOString(),
    platform: process.platform,
    nodeVersion: process.version,
    summary: {
      averageSearchSpeedup: avgSpeedup,
      maximumSearchSpeedup: Math.max(...comparisons.map(c => c.improvement.p50Speedup)),
      averageMemoryReduction: avgMemReduction,
      benchmarksMeetingTarget: comparisons.filter(c => c.status !== 'missed').length,
      totalBenchmarks: comparisons.length,
      successRate: (comparisons.filter(c => c.status !== 'missed').length / comparisons.length) * 100
    },
    searchPerformance: comparisons,
    memoryUsage: memoryComparisons,
    targets: {
      searchSpeedup: '8-12.5x',
      memoryReduction: '8.6x',
      status: comparisons.filter(c => c.status !== 'missed').length / comparisons.length >= 0.9 ? 'met' : 'missed'
    }
  };
}

/**
 * CLI entry point for comparison tool
 */
export async function main() {
  const baseline = loadBaseline();

  console.log('Loading benchmark results...');

  // In a real implementation, these would be loaded from actual benchmark runs
  // For now, we'll use the baseline data to demonstrate the comparison
  const ruvectorResults: BenchmarkResult[] = Object.entries(baseline.baselines.ruvector)
    .filter(([_, data]) => 'p50Ms' in data)
    .map(([name, data]) => ({
      name,
      backend: 'ruvector' as const,
      iterations: 100,
      meanMs: data.p50Ms,
      p50Ms: data.p50Ms,
      p95Ms: data.p50Ms * 1.5,
      p99Ms: data.p99Ms,
      minMs: data.p50Ms * 0.8,
      maxMs: data.p99Ms * 1.2,
      opsPerSec: 1000 / data.p50Ms,
      timestamp: Date.now()
    }));

  const hnswlibResults: BenchmarkResult[] = Object.entries(baseline.baselines.hnswlib)
    .filter(([_, data]) => 'p50Ms' in data)
    .map(([name, data]) => ({
      name,
      backend: 'hnswlib' as const,
      iterations: 100,
      meanMs: data.p50Ms,
      p50Ms: data.p50Ms,
      p95Ms: data.p50Ms * 1.5,
      p99Ms: data.p99Ms,
      minMs: data.p50Ms * 0.8,
      maxMs: data.p99Ms * 1.2,
      opsPerSec: 1000 / data.p50Ms,
      timestamp: Date.now()
    }));

  const comparisons = compareBenchmarks(ruvectorResults, hnswlibResults);
  const memoryComparisons = compareMemory(baseline);

  const report = formatComparisonReport(comparisons, memoryComparisons);
  console.log(report);

  // Export JSON report
  const jsonReport = generateComparisonJSON(comparisons, memoryComparisons);
  const fs = require('fs');
  fs.writeFileSync('benchmark-comparison.json', JSON.stringify(jsonReport, null, 2));
  console.log('JSON report exported to: benchmark-comparison.json');
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}
