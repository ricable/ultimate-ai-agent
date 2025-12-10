# AgentDB v2 Benchmarking Plan

## Overview

Comprehensive benchmarking strategy to validate RuVector performance claims and ensure no regressions from hnswlib baseline.

## Benchmark Categories

### 1. Vector Operations

| Benchmark | Description | Target (RuVector) | Baseline (hnswlib) |
|-----------|-------------|-------------------|-------------------|
| `insert-single` | Single vector insert | < 0.05ms | < 0.2ms |
| `insert-batch-100` | Batch insert 100 vectors | < 2ms | < 10ms |
| `insert-batch-1000` | Batch insert 1000 vectors | < 15ms | < 80ms |
| `search-k10` | Search k=10 | < 0.1ms | < 0.5ms |
| `search-k100` | Search k=100 | < 0.2ms | < 2ms |
| `search-k10-100K` | Search in 100K vectors | < 0.2ms | < 1ms |
| `search-k10-1M` | Search in 1M vectors | < 0.5ms | < 10ms |

### 2. Memory Usage

| Benchmark | Description | Target (RuVector) | Baseline (hnswlib) |
|-----------|-------------|-------------------|-------------------|
| `memory-10K` | Memory for 10K vectors | < 20MB | < 50MB |
| `memory-100K` | Memory for 100K vectors | < 50MB | < 400MB |
| `memory-100K-compressed` | With compression | < 15MB | N/A |
| `memory-1M` | Memory for 1M vectors | < 200MB | < 4GB |

### 3. Index Operations

| Benchmark | Description | Target (RuVector) | Baseline (hnswlib) |
|-----------|-------------|-------------------|-------------------|
| `build-10K` | Build index 10K vectors | < 0.5s | < 2s |
| `build-100K` | Build index 100K vectors | < 3s | < 10s |
| `save-100K` | Save index to disk | < 1s | < 2s |
| `load-100K` | Load index from disk | < 0.5s | < 1s |

### 4. GNN Operations (RuVector only)

| Benchmark | Description | Target |
|-----------|-------------|--------|
| `gnn-enhance` | Single query enhancement | < 1ms |
| `gnn-train-100` | Train on 100 samples | < 5s |
| `gnn-train-1000` | Train on 1000 samples | < 30s |

## Benchmark Implementation

### Benchmark Runner

```typescript
// packages/agentdb/benchmarks/runner.ts

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
}

export interface BenchmarkConfig {
  warmupIterations: number;
  iterations: number;
  dimension: number;
  vectorCounts: number[];
}

export async function runBenchmark(
  name: string,
  fn: () => void | Promise<void>,
  config: { warmup?: number; iterations?: number } = {}
): Promise<BenchmarkResult> {
  const warmup = config.warmup ?? 10;
  const iterations = config.iterations ?? 100;
  const times: number[] = [];

  // Warmup
  for (let i = 0; i < warmup; i++) {
    await fn();
  }

  // Benchmark
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }

  times.sort((a, b) => a - b);

  return {
    name,
    backend: 'ruvector', // Set by caller
    iterations,
    meanMs: times.reduce((a, b) => a + b, 0) / times.length,
    p50Ms: times[Math.floor(times.length * 0.5)],
    p95Ms: times[Math.floor(times.length * 0.95)],
    p99Ms: times[Math.floor(times.length * 0.99)],
    minMs: times[0],
    maxMs: times[times.length - 1],
    opsPerSec: 1000 / (times.reduce((a, b) => a + b, 0) / times.length)
  };
}
```

### Vector Search Benchmark

```typescript
// packages/agentdb/benchmarks/vector-search.bench.ts

import { describe, bench } from 'vitest';
import { createBackend } from '../src/backends/factory.js';

const DIMENSION = 384;
const VECTOR_COUNTS = [1000, 10000, 100000];
const K_VALUES = [10, 50, 100];

describe('Vector Search Benchmarks', async () => {
  for (const backend of ['ruvector', 'hnswlib'] as const) {
    describe(backend, async () => {
      for (const count of VECTOR_COUNTS) {
        describe(`${count} vectors`, async () => {
          const index = await createBackend(backend, {
            dimension: DIMENSION,
            metric: 'cosine',
            maxElements: count * 2
          });

          // Populate
          for (let i = 0; i < count; i++) {
            const emb = new Float32Array(DIMENSION);
            for (let j = 0; j < DIMENSION; j++) emb[j] = Math.random();
            index.insert(`vec-${i}`, emb);
          }

          const query = new Float32Array(DIMENSION);
          for (let j = 0; j < DIMENSION; j++) query[j] = Math.random();

          for (const k of K_VALUES) {
            bench(`search k=${k}`, () => {
              index.search(query, k);
            });
          }
        });
      }
    });
  }
});
```

### Memory Benchmark

```typescript
// packages/agentdb/benchmarks/memory.bench.ts

import { createBackend } from '../src/backends/factory.js';

async function measureMemory(
  backend: 'ruvector' | 'hnswlib',
  vectorCount: number
): Promise<{ peakMB: number; finalMB: number }> {
  global.gc?.(); // Require --expose-gc
  const baseMemory = process.memoryUsage().heapUsed;

  const index = await createBackend(backend, {
    dimension: 384,
    metric: 'cosine',
    maxElements: vectorCount * 2
  });

  for (let i = 0; i < vectorCount; i++) {
    const emb = new Float32Array(384);
    for (let j = 0; j < 384; j++) emb[j] = Math.random();
    index.insert(`vec-${i}`, emb);
  }

  global.gc?.();
  const peakMemory = process.memoryUsage().heapUsed;

  return {
    peakMB: (peakMemory - baseMemory) / 1024 / 1024,
    finalMB: (process.memoryUsage().heapUsed - baseMemory) / 1024 / 1024
  };
}

// Run with: node --expose-gc benchmarks/memory.bench.js
```

## Baseline Configuration

```json
// packages/agentdb/benchmarks/baseline.json
{
  "version": "2.0.0",
  "timestamp": "2024-01-01T00:00:00Z",
  "platform": "linux-x64",
  "node": "v20.10.0",

  "baselines": {
    "ruvector": {
      "search-k10-10K": { "p50Ms": 0.061, "p99Ms": 0.15 },
      "search-k10-100K": { "p50Ms": 0.12, "p99Ms": 0.25 },
      "search-k100-100K": { "p50Ms": 0.164, "p99Ms": 0.35 },
      "insert-single": { "p50Ms": 0.021, "p99Ms": 0.05 },
      "memory-100K-MB": 48
    },
    "hnswlib": {
      "search-k10-10K": { "p50Ms": 0.5, "p99Ms": 1.2 },
      "search-k10-100K": { "p50Ms": 1.0, "p99Ms": 2.5 },
      "search-k100-100K": { "p50Ms": 2.1, "p99Ms": 4.0 },
      "insert-single": { "p50Ms": 0.1, "p99Ms": 0.3 },
      "memory-100K-MB": 412
    }
  },

  "thresholds": {
    "regressionPercent": 10,
    "criticalRegressionPercent": 25
  }
}
```

## Regression Detection

```typescript
// packages/agentdb/benchmarks/regression-check.ts

import baseline from './baseline.json';

interface RegressionResult {
  benchmark: string;
  current: number;
  baseline: number;
  changePercent: number;
  status: 'pass' | 'warning' | 'fail';
}

export function checkRegression(
  results: BenchmarkResult[],
  backend: 'ruvector' | 'hnswlib'
): RegressionResult[] {
  const baselineData = baseline.baselines[backend];
  const regressions: RegressionResult[] = [];

  for (const result of results) {
    const base = baselineData[result.name];
    if (!base) continue;

    const changePercent = ((result.p50Ms - base.p50Ms) / base.p50Ms) * 100;

    let status: 'pass' | 'warning' | 'fail' = 'pass';
    if (changePercent > baseline.thresholds.criticalRegressionPercent) {
      status = 'fail';
    } else if (changePercent > baseline.thresholds.regressionPercent) {
      status = 'warning';
    }

    regressions.push({
      benchmark: result.name,
      current: result.p50Ms,
      baseline: base.p50Ms,
      changePercent,
      status
    });
  }

  return regressions;
}
```

## Benchmark Reports

### Console Report

```
╔══════════════════════════════════════════════════════════════════════╗
║                    AgentDB v2 Benchmark Report                        ║
╠══════════════════════════════════════════════════════════════════════╣
║ Platform: linux-x64 | Node: v20.10.0 | Date: 2024-01-15              ║
╠══════════════════════════════════════════════════════════════════════╣

┌─────────────────────────────────────────────────────────────────────┐
│ Vector Search (100K vectors, dimension=384)                          │
├───────────────────┬───────────────┬───────────────┬─────────────────┤
│ Benchmark         │ RuVector      │ hnswlib       │ Improvement     │
├───────────────────┼───────────────┼───────────────┼─────────────────┤
│ search-k10 (p50)  │ 0.12 ms       │ 1.0 ms        │ 8.3x ✅         │
│ search-k100 (p50) │ 0.16 ms       │ 2.1 ms        │ 13.1x ✅        │
│ insert (p50)      │ 0.02 ms       │ 0.1 ms        │ 5.0x ✅         │
├───────────────────┼───────────────┼───────────────┼─────────────────┤
│ Memory Usage      │ 48 MB         │ 412 MB        │ 8.6x ✅         │
└───────────────────┴───────────────┴───────────────┴─────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Regression Check                                                     │
├───────────────────┬───────────────┬───────────────┬─────────────────┤
│ Benchmark         │ Current       │ Baseline      │ Change          │
├───────────────────┼───────────────┼───────────────┼─────────────────┤
│ search-k10        │ 0.12 ms       │ 0.12 ms       │ +0.0% ✅        │
│ search-k100       │ 0.17 ms       │ 0.16 ms       │ +6.3% ⚠️        │
└───────────────────┴───────────────┴───────────────┴─────────────────┘

Summary: 8 passed, 1 warning, 0 failed
```

### JSON Report

```json
{
  "version": "2.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "platform": "linux-x64",
  "results": {
    "ruvector": [...],
    "hnswlib": [...]
  },
  "comparison": {...},
  "regressions": [...],
  "status": "pass"
}
```

## CI Integration

See [workflows/benchmarks.yml](../workflows/benchmarks.yml) for GitHub Actions integration.

## Running Benchmarks

```bash
# Full benchmark suite
npm run bench

# Specific backend
npm run bench -- --filter ruvector

# Quick smoke test
npm run bench:quick

# With memory profiling
npm run bench:memory

# Generate report
npm run bench -- --reporter json --output benchmark-report.json
```
