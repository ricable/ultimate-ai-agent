/**
 * Hot Path Profiler for Attention Mechanisms
 * Identifies performance bottlenecks and optimization opportunities
 */

import { performance } from 'perf_hooks';
import { writeFileSync } from 'fs';
import { join } from 'path';

interface ProfiledFunction {
  name: string;
  callCount: number;
  totalTimeMs: number;
  avgTimeMs: number;
  minTimeMs: number;
  maxTimeMs: number;
  percentOfTotal: number;
}

class HotPathProfiler {
  private profiles: Map<string, number[]> = new Map();
  private callStacks: Map<string, number> = new Map();

  startProfiling(functionName: string): void {
    this.callStacks.set(functionName, performance.now());
  }

  endProfiling(functionName: string): void {
    const startTime = this.callStacks.get(functionName);
    if (!startTime) return;

    const duration = performance.now() - startTime;

    if (!this.profiles.has(functionName)) {
      this.profiles.set(functionName, []);
    }

    this.profiles.get(functionName)!.push(duration);
    this.callStacks.delete(functionName);
  }

  getHotPaths(): ProfiledFunction[] {
    const totalTime = Array.from(this.profiles.values())
      .flat()
      .reduce((sum, time) => sum + time, 0);

    const results: ProfiledFunction[] = [];

    for (const [name, times] of this.profiles) {
      const callCount = times.length;
      const totalTimeMs = times.reduce((sum, time) => sum + time, 0);
      const avgTimeMs = totalTimeMs / callCount;
      const minTimeMs = Math.min(...times);
      const maxTimeMs = Math.max(...times);
      const percentOfTotal = (totalTimeMs / totalTime) * 100;

      results.push({
        name,
        callCount,
        totalTimeMs,
        avgTimeMs,
        minTimeMs,
        maxTimeMs,
        percentOfTotal,
      });
    }

    // Sort by total time (hottest first)
    return results.sort((a, b) => b.totalTimeMs - a.totalTimeMs);
  }

  generateReport(): string {
    const hotPaths = this.getHotPaths();
    const lines: string[] = [
      '# Hot Path Profiling Report',
      '',
      `**Date**: ${new Date().toISOString()}`,
      `**Total Functions Profiled**: ${hotPaths.length}`,
      '',
      '## Top Hot Paths (by total time)',
      '',
      '| Rank | Function | Calls | Total (ms) | Avg (ms) | Min (ms) | Max (ms) | % of Total |',
      '|------|----------|-------|------------|----------|----------|----------|------------|',
    ];

    hotPaths.slice(0, 20).forEach((path, index) => {
      lines.push(
        `| ${index + 1} | ${path.name} | ${path.callCount} | ${path.totalTimeMs.toFixed(2)} | ${path.avgTimeMs.toFixed(4)} | ${path.minTimeMs.toFixed(4)} | ${path.maxTimeMs.toFixed(4)} | ${path.percentOfTotal.toFixed(2)}% |`
      );
    });

    lines.push('', '## Optimization Recommendations', '');

    // Identify optimization opportunities
    for (const path of hotPaths.slice(0, 10)) {
      if (path.percentOfTotal > 10) {
        lines.push(
          `### ⚠️ CRITICAL: ${path.name}`,
          `- **Impact**: ${path.percentOfTotal.toFixed(2)}% of total execution time`,
          `- **Recommendation**: High priority for optimization`,
          `- **Strategies**: Profile further, consider caching, algorithm optimization`,
          ''
        );
      } else if (path.percentOfTotal > 5) {
        lines.push(
          `### ⚡ ${path.name}`,
          `- **Impact**: ${path.percentOfTotal.toFixed(2)}% of total execution time`,
          `- **Recommendation**: Medium priority for optimization`,
          `- **Strategies**: Reduce allocations, use batch operations`,
          ''
        );
      }
    }

    // Identify high-variance functions
    lines.push('## High Variance Functions', '');
    const highVariance = hotPaths.filter(path => {
      const variance = path.maxTimeMs / path.avgTimeMs;
      return variance > 3;
    });

    for (const path of highVariance.slice(0, 5)) {
      lines.push(
        `### ${path.name}`,
        `- **Variance**: ${(path.maxTimeMs / path.avgTimeMs).toFixed(2)}x`,
        `- **Min/Max**: ${path.minTimeMs.toFixed(4)}ms - ${path.maxTimeMs.toFixed(4)}ms`,
        `- **Recommendation**: Investigate conditional branches or data-dependent behavior`,
        ''
      );
    }

    return lines.join('\n');
  }

  reset(): void {
    this.profiles.clear();
    this.callStacks.clear();
  }
}

// Global profiler instance
export const profiler = new HotPathProfiler();

/**
 * Decorator for automatic profiling
 */
export function profileFunction(target: any, propertyKey: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  const functionName = `${target.constructor.name}.${propertyKey}`;

  descriptor.value = async function (...args: any[]) {
    profiler.startProfiling(functionName);
    try {
      const result = await originalMethod.apply(this, args);
      profiler.endProfiling(functionName);
      return result;
    } catch (error) {
      profiler.endProfiling(functionName);
      throw error;
    }
  };

  return descriptor;
}

/**
 * Manual profiling helper
 */
export function profileSync<T>(functionName: string, operation: () => T): T {
  profiler.startProfiling(functionName);
  try {
    const result = operation();
    profiler.endProfiling(functionName);
    return result;
  } catch (error) {
    profiler.endProfiling(functionName);
    throw error;
  }
}

export async function profileAsync<T>(functionName: string, operation: () => Promise<T>): Promise<T> {
  profiler.startProfiling(functionName);
  try {
    const result = await operation();
    profiler.endProfiling(functionName);
    return result;
  } catch (error) {
    profiler.endProfiling(functionName);
    throw error;
  }
}

// Example usage for attention mechanisms
if (require.main === module) {
  console.log('🔍 Hot Path Profiler Example\n');

  // Simulate some profiled operations
  for (let i = 0; i < 1000; i++) {
    profiler.startProfiling('attention.softmax');
    const delay = Math.random() * 5;
    const start = performance.now();
    while (performance.now() - start < delay) {}
    profiler.endProfiling('attention.softmax');

    profiler.startProfiling('attention.matmul');
    const delay2 = Math.random() * 10;
    const start2 = performance.now();
    while (performance.now() - start2 < delay2) {}
    profiler.endProfiling('attention.matmul');

    profiler.startProfiling('embedding.lookup');
    const delay3 = Math.random() * 2;
    const start3 = performance.now();
    while (performance.now() - start3 < delay3) {}
    profiler.endProfiling('embedding.lookup');
  }

  const report = profiler.generateReport();
  console.log(report);

  const reportPath = join(__dirname, '../benchmarks/results/hot-paths.md');
  writeFileSync(reportPath, report);
  console.log(`\n📄 Report saved to: ${reportPath}`);
}
