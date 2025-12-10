/**
 * Comprehensive Performance Benchmarks for Attention Mechanisms
 * Measures latency, throughput, and memory usage across realistic workloads
 */

import { AgentDB } from '../src/core/db';
import { MultiHeadAttention } from '../src/attention/multi-head';
import { FlashAttention } from '../src/attention/flash';
import { HyperbolicAttention } from '../src/attention/hyperbolic';
import { MoEAttention } from '../src/attention/moe';
import { AttentionMetricsCollector } from '../src/utils/attention-metrics';
import { writeFileSync } from 'fs';
import { join } from 'path';

// Benchmark configuration
const WORKLOAD_SIZES = [100, 1000, 10000, 100000];
const ITERATIONS = 100;
const WARMUP_ITERATIONS = 10;

interface BenchmarkResult {
  mechanism: string;
  workloadSize: number;
  avgLatencyUs: number;
  p50LatencyUs: number;
  p95LatencyUs: number;
  p99LatencyUs: number;
  throughputOpsPerSec: number;
  memoryMB: number;
  peakMemoryMB: number;
}

class AttentionBenchmark {
  private db!: AgentDB;
  private multiHead!: MultiHeadAttention;
  private flash!: FlashAttention;
  private hyperbolic!: HyperbolicAttention;
  private moe!: MoEAttention;
  private metrics = new AttentionMetricsCollector();

  async setup(workloadSize: number): Promise<void> {
    console.log(`\n📦 Setting up benchmark with ${workloadSize} memories...`);

    // Initialize database with test data
    this.db = new AgentDB(':memory:');

    // Generate realistic test data
    const testData = this.generateTestData(workloadSize);
    for (const item of testData) {
      await this.db.addMemory({
        content: item.content,
        embedding: item.embedding,
        metadata: item.metadata,
      });
    }

    // Initialize attention mechanisms
    this.multiHead = new MultiHeadAttention({
      numHeads: 8,
      headDim: 64,
      dropout: 0.1,
    });

    this.flash = new FlashAttention({
      blockSize: 256,
      numWarps: 4,
    });

    this.hyperbolic = new HyperbolicAttention({
      curvature: 1.0,
      manifoldDim: 512,
    });

    this.moe = new MoEAttention({
      numExperts: 4,
      expertsPerToken: 2,
      expertCapacity: 128,
    });

    console.log('✅ Setup complete');
  }

  async teardown(): Promise<void> {
    if (this.db) {
      this.db.close();
    }
  }

  private generateTestData(count: number): Array<{ content: string; embedding: number[]; metadata: any }> {
    const data = [];
    for (let i = 0; i < count; i++) {
      data.push({
        content: `Test memory ${i}: This is a sample memory entry for benchmarking purposes. It contains realistic text data that would be used in production scenarios.`,
        embedding: this.generateRandomEmbedding(),
        metadata: {
          timestamp: Date.now(),
          source: 'benchmark',
          index: i,
          category: ['research', 'code', 'documentation', 'analysis'][i % 4],
        },
      });
    }
    return data;
  }

  private generateRandomEmbedding(): number[] {
    const embedding = new Array(512);
    for (let i = 0; i < 512; i++) {
      embedding[i] = Math.random() * 2 - 1;
    }
    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / norm);
  }

  async benchmarkMechanism(
    name: string,
    mechanism: any,
    workloadSize: number
  ): Promise<void> {
    console.log(`\n🔬 Benchmarking ${name}...`);

    // Warmup
    for (let i = 0; i < WARMUP_ITERATIONS; i++) {
      const query = this.generateRandomEmbedding();
      await mechanism.search(query, 10);
    }

    // Actual benchmark
    this.metrics.reset();
    for (let i = 0; i < ITERATIONS; i++) {
      this.metrics.startOperation(name);
      const startTime = performance.now();

      const query = this.generateRandomEmbedding();
      await mechanism.search(query, 10);

      this.metrics.endOperation(name, startTime);

      if ((i + 1) % 20 === 0) {
        console.log(`  Progress: ${i + 1}/${ITERATIONS} iterations`);
      }
    }

    const metrics = this.metrics.getMetrics(name);
    if (metrics) {
      console.log(`✅ ${name} completed:`);
      console.log(`   Avg: ${metrics.avgLatencyUs.toFixed(2)}µs`);
      console.log(`   P95: ${metrics.p95LatencyUs.toFixed(2)}µs`);
      console.log(`   P99: ${metrics.p99LatencyUs.toFixed(2)}µs`);
      console.log(`   Throughput: ${metrics.throughputOpsPerSec.toFixed(2)} ops/sec`);
    }
  }

  async benchmarkBaseline(workloadSize: number): Promise<void> {
    console.log(`\n🔬 Benchmarking Baseline (AgentDB v2.0.0-alpha.2.7)...`);

    // Warmup
    for (let i = 0; i < WARMUP_ITERATIONS; i++) {
      const query = this.generateRandomEmbedding();
      await this.db.search(query, 10);
    }

    // Actual benchmark
    this.metrics.reset();
    for (let i = 0; i < ITERATIONS; i++) {
      this.metrics.startOperation('Baseline');
      const startTime = performance.now();

      const query = this.generateRandomEmbedding();
      await this.db.search(query, 10);

      this.metrics.endOperation('Baseline', startTime);

      if ((i + 1) % 20 === 0) {
        console.log(`  Progress: ${i + 1}/${ITERATIONS} iterations`);
      }
    }

    const metrics = this.metrics.getMetrics('Baseline');
    if (metrics) {
      console.log(`✅ Baseline completed:`);
      console.log(`   Avg: ${metrics.avgLatencyUs.toFixed(2)}µs`);
      console.log(`   P95: ${metrics.p95LatencyUs.toFixed(2)}µs`);
      console.log(`   P99: ${metrics.p99LatencyUs.toFixed(2)}µs`);
      console.log(`   Throughput: ${metrics.throughputOpsPerSec.toFixed(2)} ops/sec`);
    }
  }

  async runFullBenchmark(): Promise<BenchmarkResult[]> {
    const results: BenchmarkResult[] = [];

    for (const workloadSize of WORKLOAD_SIZES) {
      console.log(`\n${'='.repeat(80)}`);
      console.log(`📊 BENCHMARK: ${workloadSize} memories`);
      console.log('='.repeat(80));

      await this.setup(workloadSize);

      // Benchmark baseline
      await this.benchmarkBaseline(workloadSize);
      const baselineMetrics = this.metrics.getMetrics('Baseline');
      if (baselineMetrics) {
        results.push({
          mechanism: 'Baseline',
          workloadSize,
          avgLatencyUs: baselineMetrics.avgLatencyUs,
          p50LatencyUs: baselineMetrics.p50LatencyUs,
          p95LatencyUs: baselineMetrics.p95LatencyUs,
          p99LatencyUs: baselineMetrics.p99LatencyUs,
          throughputOpsPerSec: baselineMetrics.throughputOpsPerSec,
          memoryMB: baselineMetrics.memoryUsageBytes / 1024 / 1024,
          peakMemoryMB: baselineMetrics.peakMemoryBytes / 1024 / 1024,
        });
      }

      // Benchmark each attention mechanism
      const mechanisms = [
        { name: 'MultiHeadAttention', instance: this.multiHead },
        { name: 'FlashAttention', instance: this.flash },
        { name: 'HyperbolicAttention', instance: this.hyperbolic },
        { name: 'MoEAttention', instance: this.moe },
      ];

      for (const { name, instance } of mechanisms) {
        await this.benchmarkMechanism(name, instance, workloadSize);
        const metrics = this.metrics.getMetrics(name);
        if (metrics) {
          results.push({
            mechanism: name,
            workloadSize,
            avgLatencyUs: metrics.avgLatencyUs,
            p50LatencyUs: metrics.p50LatencyUs,
            p95LatencyUs: metrics.p95LatencyUs,
            p99LatencyUs: metrics.p99LatencyUs,
            throughputOpsPerSec: metrics.throughputOpsPerSec,
            memoryMB: metrics.memoryUsageBytes / 1024 / 1024,
            peakMemoryMB: metrics.peakMemoryBytes / 1024 / 1024,
          });
        }
      }

      await this.teardown();
    }

    return results;
  }

  generateReport(results: BenchmarkResult[]): string {
    const lines: string[] = [
      '# Attention Mechanism Performance Benchmark Results',
      '',
      `**Date**: ${new Date().toISOString()}`,
      `**Platform**: Node.js ${process.version}`,
      `**Iterations**: ${ITERATIONS} (${WARMUP_ITERATIONS} warmup)`,
      '',
      '## Executive Summary',
      '',
    ];

    // Calculate speedups vs baseline
    const speedups: Record<string, number[]> = {};
    for (const result of results) {
      if (result.mechanism === 'Baseline') continue;

      const baseline = results.find(
        r => r.mechanism === 'Baseline' && r.workloadSize === result.workloadSize
      );

      if (baseline) {
        const speedup = baseline.avgLatencyUs / result.avgLatencyUs;
        if (!speedups[result.mechanism]) {
          speedups[result.mechanism] = [];
        }
        speedups[result.mechanism].push(speedup);
      }
    }

    lines.push('### Performance Improvements vs Baseline', '');
    for (const [mechanism, speeds] of Object.entries(speedups)) {
      const avgSpeedup = speeds.reduce((a, b) => a + b, 0) / speeds.length;
      const status = avgSpeedup > 1 ? '✅' : '⚠️';
      lines.push(`- **${mechanism}**: ${status} ${avgSpeedup.toFixed(2)}x average speedup`);
    }

    lines.push('', '## Detailed Results', '');

    for (const workloadSize of WORKLOAD_SIZES) {
      lines.push(`### ${workloadSize} Memories`, '');
      lines.push(
        '| Mechanism | Avg Latency | P95 | P99 | Throughput | Memory | Speedup |',
        '|-----------|-------------|-----|-----|------------|--------|---------|'
      );

      const workloadResults = results.filter(r => r.workloadSize === workloadSize);
      const baseline = workloadResults.find(r => r.mechanism === 'Baseline');

      for (const result of workloadResults) {
        const speedup = baseline ? baseline.avgLatencyUs / result.avgLatencyUs : 1.0;
        lines.push(
          `| ${result.mechanism} | ${result.avgLatencyUs.toFixed(2)}µs | ${result.p95LatencyUs.toFixed(2)}µs | ${result.p99LatencyUs.toFixed(2)}µs | ${result.throughputOpsPerSec.toFixed(0)} ops/s | ${result.memoryMB.toFixed(2)} MB | ${speedup.toFixed(2)}x |`
        );
      }

      lines.push('');
    }

    // Target metrics validation
    lines.push('## Target Metrics Validation', '');

    const targets = [
      { mechanism: 'MultiHeadAttention', targetUs: 50 },
      { mechanism: 'FlashAttention', targetSpeedup: 3 },
      { mechanism: 'HyperbolicAttention', targetUs: 100 },
      { mechanism: 'MoEAttention', targetUs: 200 },
    ];

    for (const target of targets) {
      const mechanismResults = results.filter(r => r.mechanism === target.mechanism);
      if (mechanismResults.length === 0) continue;

      if ('targetUs' in target) {
        const avgLatency = mechanismResults.reduce((sum, r) => sum + r.avgLatencyUs, 0) / mechanismResults.length;
        const status = avgLatency < target.targetUs ? '✅' : '❌';
        lines.push(`- ${status} **${target.mechanism}**: ${avgLatency.toFixed(2)}µs avg (target: <${target.targetUs}µs)`);
      } else if ('targetSpeedup' in target) {
        const avgSpeedup = speedups[target.mechanism]?.reduce((a, b) => a + b, 0) / (speedups[target.mechanism]?.length || 1);
        const status = avgSpeedup >= target.targetSpeedup ? '✅' : '❌';
        lines.push(`- ${status} **${target.mechanism}**: ${avgSpeedup.toFixed(2)}x speedup (target: ${target.targetSpeedup}x for 10K+)`);
      }
    }

    return lines.join('\n');
  }
}

// Run benchmark if executed directly
if (require.main === module) {
  (async () => {
    console.log('🚀 Starting Attention Mechanism Performance Benchmark Suite\n');

    const benchmark = new AttentionBenchmark();
    const results = await benchmark.runFullBenchmark();

    // Generate and save report
    const report = benchmark.generateReport(results);
    const reportPath = join(__dirname, 'results', 'attention-comparison.md');
    writeFileSync(reportPath, report);

    console.log(`\n${'='.repeat(80)}`);
    console.log('✅ BENCHMARK COMPLETE');
    console.log('='.repeat(80));
    console.log(`\n📄 Report saved to: ${reportPath}\n`);
    console.log(report);

    // Export JSON results
    const jsonPath = join(__dirname, 'results', 'attention-results.json');
    writeFileSync(jsonPath, JSON.stringify(results, null, 2));
    console.log(`📊 JSON results saved to: ${jsonPath}\n`);
  })();
}

export { AttentionBenchmark };
