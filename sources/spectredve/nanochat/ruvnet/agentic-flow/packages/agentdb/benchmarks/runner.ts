/**
 * AgentDB v2 Benchmark Runner
 * Provides core benchmarking utilities for measuring performance
 */

import { performance } from 'perf_hooks';

export interface BenchmarkResult {
  name: string;
  backend: 'ruvector' | 'hnswlib';
  iterations: number;
  meanMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  minMs: number;
  maxMs: number;
  opsPerSec: number;
  memoryMB?: number;
  timestamp: number;
}

export interface BenchmarkConfig {
  warmupIterations: number;
  iterations: number;
  dimension: number;
  vectorCounts: number[];
  kValues: number[];
}

export interface MemoryMeasurement {
  peakMB: number;
  finalMB: number;
  heapUsedMB: number;
  externalMB: number;
}

/**
 * Run a benchmark function and collect timing statistics
 */
export async function runBenchmark(
  name: string,
  fn: () => void | Promise<void>,
  config: { warmup?: number; iterations?: number; backend?: 'ruvector' | 'hnswlib' } = {}
): Promise<BenchmarkResult> {
  const warmup = config.warmup ?? 10;
  const iterations = config.iterations ?? 100;
  const backend = config.backend ?? 'ruvector';
  const times: number[] = [];

  // Warmup phase - ensure JIT compilation and cache warming
  for (let i = 0; i < warmup; i++) {
    await fn();
  }

  // Force garbage collection before benchmark if available
  if (global.gc) {
    global.gc();
  }

  // Benchmark phase
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    const end = performance.now();
    times.push(end - start);
  }

  // Sort for percentile calculation
  times.sort((a, b) => a - b);

  const mean = times.reduce((a, b) => a + b, 0) / times.length;

  return {
    name,
    backend,
    iterations,
    meanMs: mean,
    p50Ms: times[Math.floor(times.length * 0.5)],
    p95Ms: times[Math.floor(times.length * 0.95)],
    p99Ms: times[Math.floor(times.length * 0.99)],
    minMs: times[0],
    maxMs: times[times.length - 1],
    opsPerSec: 1000 / mean,
    timestamp: Date.now()
  };
}

/**
 * Measure memory usage for a benchmark operation
 */
export async function measureMemory(
  fn: () => void | Promise<void>
): Promise<MemoryMeasurement> {
  // Force GC before measurement
  if (global.gc) {
    global.gc();
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  const baseline = process.memoryUsage();

  // Execute operation
  await fn();

  // Force GC after operation
  if (global.gc) {
    global.gc();
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  const final = process.memoryUsage();

  return {
    peakMB: (final.heapUsed - baseline.heapUsed) / 1024 / 1024,
    finalMB: final.heapUsed / 1024 / 1024,
    heapUsedMB: final.heapUsed / 1024 / 1024,
    externalMB: final.external / 1024 / 1024
  };
}

/**
 * Generate random vectors for benchmarking
 */
export function generateRandomVectors(count: number, dimension: number): Float32Array[] {
  const vectors: Float32Array[] = [];

  for (let i = 0; i < count; i++) {
    const vector = new Float32Array(dimension);
    for (let j = 0; j < dimension; j++) {
      vector[j] = Math.random() * 2 - 1; // Range [-1, 1]
    }
    vectors.push(vector);
  }

  return vectors;
}

/**
 * Normalize a vector to unit length
 */
export function normalizeVector(vector: Float32Array): Float32Array {
  let magnitude = 0;
  for (let i = 0; i < vector.length; i++) {
    magnitude += vector[i] * vector[i];
  }
  magnitude = Math.sqrt(magnitude);

  const normalized = new Float32Array(vector.length);
  for (let i = 0; i < vector.length; i++) {
    normalized[i] = vector[i] / magnitude;
  }

  return normalized;
}

/**
 * Calculate improvement ratio between two benchmarks
 */
export function calculateImprovement(baseline: number, current: number): number {
  return baseline / current;
}

/**
 * Format benchmark results as a table
 */
export function formatResults(results: BenchmarkResult[]): string {
  const lines: string[] = [];

  lines.push('┌────────────────────────────────────────────────────────────────┐');
  lines.push('│ Benchmark Results                                              │');
  lines.push('├────────────────────────┬───────────┬───────────┬───────────────┤');
  lines.push('│ Name                   │ P50 (ms)  │ P99 (ms)  │ Ops/sec       │');
  lines.push('├────────────────────────┼───────────┼───────────┼───────────────┤');

  for (const result of results) {
    const name = result.name.padEnd(22);
    const p50 = result.p50Ms.toFixed(3).padStart(9);
    const p99 = result.p99Ms.toFixed(3).padStart(9);
    const opsPerSec = result.opsPerSec.toFixed(0).padStart(13);

    lines.push(`│ ${name} │ ${p50} │ ${p99} │ ${opsPerSec} │`);
  }

  lines.push('└────────────────────────┴───────────┴───────────┴───────────────┘');

  return lines.join('\n');
}

/**
 * Export results to JSON format
 */
export function exportResults(results: BenchmarkResult[], outputPath: string): void {
  const fs = require('fs');
  const output = {
    version: '2.0.0',
    timestamp: new Date().toISOString(),
    platform: process.platform,
    arch: process.arch,
    nodeVersion: process.version,
    results
  };

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));
}

/**
 * Default benchmark configuration
 */
export const DEFAULT_CONFIG: BenchmarkConfig = {
  warmupIterations: 10,
  iterations: 100,
  dimension: 384,
  vectorCounts: [1000, 10000, 100000],
  kValues: [10, 50, 100]
};
