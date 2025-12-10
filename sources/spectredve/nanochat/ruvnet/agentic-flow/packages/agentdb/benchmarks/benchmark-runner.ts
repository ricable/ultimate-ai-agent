#!/usr/bin/env node
/**
 * AgentDB Performance Benchmark Runner
 *
 * Comprehensive performance testing suite for AgentDB
 * Tests: Vector search, Quantization, Batch operations, Database backends, Memory systems
 */

import { performance } from 'perf_hooks';
import { VectorSearchBenchmark } from './vector-search/vector-search-bench';
import { HNSWBenchmark } from './hnsw/hnsw-benchmark';
import { QuantizationBenchmark } from './quantization/quantization-bench';
import { BatchOperationsBenchmark } from './batch-ops/batch-ops-bench';
import { DatabaseBackendBenchmark } from './database/database-bench';
import { MemorySystemsBenchmark } from './memory-systems/memory-bench';
import { PerformanceReporter } from './reports/performance-reporter';
import * as fs from 'fs';
import * as path from 'path';

export interface BenchmarkResult {
  name: string;
  category: string;
  duration: number;
  operationsPerSecond?: number;
  memoryUsed?: number;
  metrics: Record<string, any>;
  timestamp: number;
  success: boolean;
  error?: string;
}

export interface BenchmarkSuite {
  name: string;
  description: string;
  benchmarks: Array<() => Promise<BenchmarkResult>>;
}

export class BenchmarkRunner {
  private results: BenchmarkResult[] = [];
  private reporter: PerformanceReporter;
  private startTime: number = 0;

  constructor() {
    this.reporter = new PerformanceReporter();
  }

  /**
   * Run all benchmarks
   */
  async runAll(): Promise<void> {
    console.log('🚀 AgentDB Performance Benchmark Suite\n');
    console.log('=' .repeat(80));
    this.startTime = performance.now();

    const suites: BenchmarkSuite[] = [
      {
        name: 'HNSW Indexing Performance',
        description: 'Test HNSW vs brute-force vector search (verify 150x speedup)',
        benchmarks: this.getHNSWBenchmarks()
      },
      {
        name: 'Vector Search Performance',
        description: 'Test vector similarity search with different dataset sizes',
        benchmarks: this.getVectorSearchBenchmarks()
      },
      {
        name: 'Quantization Performance',
        description: 'Test 4-bit and 8-bit quantization performance',
        benchmarks: this.getQuantizationBenchmarks()
      },
      {
        name: 'Batch Operations Performance',
        description: 'Test batch vs individual operations',
        benchmarks: this.getBatchOperationsBenchmarks()
      },
      {
        name: 'Database Backend Performance',
        description: 'Compare better-sqlite3 vs sql.js performance',
        benchmarks: this.getDatabaseBenchmarks()
      },
      {
        name: 'Memory Systems Performance',
        description: 'Test causal graph, reflexion, and skill library',
        benchmarks: this.getMemorySystemsBenchmarks()
      }
    ];

    for (const suite of suites) {
      await this.runSuite(suite);
    }

    await this.generateReport();
  }

  /**
   * Run a benchmark suite
   */
  private async runSuite(suite: BenchmarkSuite): Promise<void> {
    console.log(`\n📊 ${suite.name}`);
    console.log(`   ${suite.description}`);
    console.log('-'.repeat(80));

    for (const benchmark of suite.benchmarks) {
      try {
        const result = await benchmark();
        this.results.push(result);
        this.logResult(result);
      } catch (error) {
        console.error(`   ❌ Benchmark failed: ${error}`);
        this.results.push({
          name: 'Unknown',
          category: suite.name,
          duration: 0,
          metrics: {},
          timestamp: Date.now(),
          success: false,
          error: error instanceof Error ? error.message : String(error)
        });
      }
    }
  }

  /**
   * Log benchmark result
   */
  private logResult(result: BenchmarkResult): void {
    const icon = result.success ? '✅' : '❌';
    const duration = result.duration.toFixed(2);
    const opsPerSec = result.operationsPerSecond
      ? ` (${result.operationsPerSecond.toFixed(0)} ops/sec)`
      : '';

    console.log(`   ${icon} ${result.name}: ${duration}ms${opsPerSec}`);

    // Show key metrics
    if (Object.keys(result.metrics).length > 0) {
      for (const [key, value] of Object.entries(result.metrics)) {
        if (typeof value === 'number') {
          console.log(`      ${key}: ${value.toFixed(2)}`);
        } else {
          console.log(`      ${key}: ${value}`);
        }
      }
    }
  }

  /**
   * Generate comprehensive performance report
   */
  private async generateReport(): Promise<void> {
    const totalDuration = performance.now() - this.startTime;

    console.log('\n' + '='.repeat(80));
    console.log('📈 Benchmark Summary');
    console.log('='.repeat(80));
    console.log(`Total Duration: ${(totalDuration / 1000).toFixed(2)}s`);
    console.log(`Total Benchmarks: ${this.results.length}`);
    console.log(`Successful: ${this.results.filter(r => r.success).length}`);
    console.log(`Failed: ${this.results.filter(r => !r.success).length}`);

    // Generate detailed reports
    await this.reporter.generateHTMLReport(this.results);
    await this.reporter.generateJSONReport(this.results);
    await this.reporter.generateMarkdownReport(this.results);

    console.log('\n📄 Reports generated:');
    console.log(`   - HTML: benchmarks/reports/performance-report.html`);
    console.log(`   - JSON: benchmarks/reports/performance-report.json`);
    console.log(`   - Markdown: benchmarks/reports/performance-report.md`);
  }

  // Benchmark suite getters
  private getHNSWBenchmarks(): Array<() => Promise<BenchmarkResult>> {
    const bench = new HNSWBenchmark();
    return [
      () => bench.verify150xClaim(),
      () => bench.testHNSW1K(),
      () => bench.testHNSW10K(),
      () => bench.testHNSW100K(),
      () => bench.testBruteForce10K(),
      () => bench.testEfSearchTradeoff()
    ];
  }

  private getVectorSearchBenchmarks(): Array<() => Promise<BenchmarkResult>> {
    const bench = new VectorSearchBenchmark();
    return [
      () => bench.testVectorSearch100(),
      () => bench.testVectorSearch1K(),
      () => bench.testVectorSearch10K(),
      () => bench.testVectorSearch100K()
    ];
  }

  private getQuantizationBenchmarks(): Array<() => Promise<BenchmarkResult>> {
    const bench = new QuantizationBenchmark();
    return [
      () => bench.test4BitQuantization(),
      () => bench.test8BitQuantization(),
      () => bench.testMemoryReduction(),
      () => bench.testAccuracyTradeoff()
    ];
  }

  private getBatchOperationsBenchmarks(): Array<() => Promise<BenchmarkResult>> {
    const bench = new BatchOperationsBenchmark();
    return [
      () => bench.testBatchInsert(),
      () => bench.testIndividualInserts(),
      () => bench.testBatchVsIndividual(),
      () => bench.testMemoryUsage()
    ];
  }

  private getDatabaseBenchmarks(): Array<() => Promise<BenchmarkResult>> {
    const bench = new DatabaseBackendBenchmark();
    return [
      () => bench.testBetterSqlite3(),
      () => bench.testSqlJs(),
      () => bench.compareBackends(),
      () => bench.testInitializationTime()
    ];
  }

  private getMemorySystemsBenchmarks(): Array<() => Promise<BenchmarkResult>> {
    const bench = new MemorySystemsBenchmark();
    return [
      () => bench.testCausalGraphQuery(),
      () => bench.testReflexionRetrieval(),
      () => bench.testSkillLibrarySearch()
    ];
  }
}

// Main execution
if (require.main === module) {
  const runner = new BenchmarkRunner();
  runner.runAll()
    .then(() => {
      console.log('\n✅ All benchmarks completed successfully!');
      process.exit(0);
    })
    .catch((error) => {
      console.error('\n❌ Benchmark suite failed:', error);
      process.exit(1);
    });
}

export default BenchmarkRunner;
