/**
 * @benchmark Attention Mechanism Performance Benchmarks
 * @description Comprehensive performance testing for all attention mechanisms
 * @metrics
 *   - Throughput (queries/second)
 *   - Latency (p50, p95, p99)
 *   - Memory usage
 *   - NAPI vs WASM comparison
 */

import { AgentDB } from '../../src/index';
import { MemoryController } from '../../src/controllers/MemoryController';
import { SelfAttentionController } from '../../src/controllers/attention/SelfAttentionController';
import { CrossAttentionController } from '../../src/controllers/attention/CrossAttentionController';
import { MultiHeadAttentionController } from '../../src/controllers/attention/MultiHeadAttentionController';
import fs from 'fs';
import path from 'path';

interface BenchmarkResult {
  name: string;
  throughput: number; // ops/sec
  latency: {
    p50: number;
    p95: number;
    p99: number;
    mean: number;
  };
  memory: {
    initial: number;
    peak: number;
    final: number;
  };
  duration: number;
}

class AttentionBenchmark {
  private db!: AgentDB;
  private memoryController!: MemoryController;
  private readonly dbPath: string;
  private readonly results: BenchmarkResult[] = [];

  constructor() {
    this.dbPath = path.join(__dirname, '../fixtures/benchmark.db');
  }

  async setup(): Promise<void> {
    // Clean up
    if (fs.existsSync(this.dbPath)) {
      fs.unlinkSync(this.dbPath);
    }

    // Initialize with attention enabled
    this.db = new AgentDB({
      dbPath: this.dbPath,
      enableAttention: true,
      attentionConfig: {
        selfAttention: { enabled: true },
        crossAttention: { enabled: true },
        multiHeadAttention: { enabled: true, numHeads: 8 }
      }
    });

    await this.db.initialize();
    this.memoryController = this.db.getController('memory') as MemoryController;
  }

  async teardown(): Promise<void> {
    await this.db.close();
    if (fs.existsSync(this.dbPath)) {
      fs.unlinkSync(this.dbPath);
    }
  }

  private measureMemory(): number {
    return process.memoryUsage().heapUsed;
  }

  private calculatePercentiles(values: number[]): { p50: number; p95: number; p99: number; mean: number } {
    const sorted = values.sort((a, b) => a - b);
    const len = sorted.length;

    return {
      p50: sorted[Math.floor(len * 0.5)],
      p95: sorted[Math.floor(len * 0.95)],
      p99: sorted[Math.floor(len * 0.99)],
      mean: sorted.reduce((a, b) => a + b, 0) / len
    };
  }

  async benchmarkSelfAttention(
    dataSize: number,
    queryCount: number,
    embeddingDim: number = 128
  ): Promise<BenchmarkResult> {
    console.log(`\n🔄 Benchmarking Self-Attention: ${dataSize} items, ${queryCount} queries, dim=${embeddingDim}`);

    const controller = this.db.getController('self-attention') as SelfAttentionController;

    const memoryInitial = this.measureMemory();

    // Populate data
    console.log('  📝 Populating data...');
    for (let i = 0; i < dataSize; i++) {
      await this.memoryController.store({
        id: `self-attn-${i}`,
        embedding: Array(embeddingDim).fill(0).map(() => Math.random())
      });
    }

    let memoryPeak = this.measureMemory();
    const latencies: number[] = [];

    // Run queries
    console.log('  ⚡ Running queries...');
    const startTime = Date.now();

    for (let i = 0; i < queryCount; i++) {
      const query = Array(embeddingDim).fill(0).map(() => Math.random());

      const queryStart = performance.now();
      await controller.computeAttention(query, { topK: 10 });
      const queryEnd = performance.now();

      latencies.push(queryEnd - queryStart);

      const currentMemory = this.measureMemory();
      if (currentMemory > memoryPeak) {
        memoryPeak = currentMemory;
      }

      if ((i + 1) % 100 === 0) {
        console.log(`    Progress: ${i + 1}/${queryCount} queries`);
      }
    }

    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000;

    global.gc && global.gc();
    const memoryFinal = this.measureMemory();

    const result: BenchmarkResult = {
      name: `Self-Attention (data=${dataSize}, queries=${queryCount}, dim=${embeddingDim})`,
      throughput: queryCount / duration,
      latency: this.calculatePercentiles(latencies),
      memory: {
        initial: memoryInitial / (1024 * 1024),
        peak: memoryPeak / (1024 * 1024),
        final: memoryFinal / (1024 * 1024)
      },
      duration
    };

    this.results.push(result);
    return result;
  }

  async benchmarkCrossAttention(
    contextSize: number,
    queryCount: number,
    embeddingDim: number = 128
  ): Promise<BenchmarkResult> {
    console.log(`\n🔄 Benchmarking Cross-Attention: ${contextSize} context, ${queryCount} queries, dim=${embeddingDim}`);

    const controller = this.db.getController('cross-attention') as CrossAttentionController;

    const memoryInitial = this.measureMemory();

    // Populate context
    console.log('  📝 Populating context...');
    for (let i = 0; i < contextSize; i++) {
      await this.memoryController.store({
        id: `cross-attn-${i}`,
        embedding: Array(embeddingDim).fill(0).map(() => Math.random())
      });
    }

    let memoryPeak = this.measureMemory();
    const latencies: number[] = [];

    // Run queries
    console.log('  ⚡ Running queries...');
    const startTime = Date.now();

    for (let i = 0; i < queryCount; i++) {
      const query = Array(embeddingDim).fill(0).map(() => Math.random());

      const queryStart = performance.now();
      await controller.computeCrossAttention(query, 'memory');
      const queryEnd = performance.now();

      latencies.push(queryEnd - queryStart);

      const currentMemory = this.measureMemory();
      if (currentMemory > memoryPeak) {
        memoryPeak = currentMemory;
      }

      if ((i + 1) % 100 === 0) {
        console.log(`    Progress: ${i + 1}/${queryCount} queries`);
      }
    }

    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000;

    global.gc && global.gc();
    const memoryFinal = this.measureMemory();

    const result: BenchmarkResult = {
      name: `Cross-Attention (context=${contextSize}, queries=${queryCount}, dim=${embeddingDim})`,
      throughput: queryCount / duration,
      latency: this.calculatePercentiles(latencies),
      memory: {
        initial: memoryInitial / (1024 * 1024),
        peak: memoryPeak / (1024 * 1024),
        final: memoryFinal / (1024 * 1024)
      },
      duration
    };

    this.results.push(result);
    return result;
  }

  async benchmarkMultiHeadAttention(
    dataSize: number,
    queryCount: number,
    numHeads: number = 8,
    embeddingDim: number = 128
  ): Promise<BenchmarkResult> {
    console.log(`\n🔄 Benchmarking Multi-Head Attention: ${dataSize} items, ${queryCount} queries, heads=${numHeads}, dim=${embeddingDim}`);

    const controller = this.db.getController('multi-head-attention') as MultiHeadAttentionController;

    const memoryInitial = this.measureMemory();

    // Populate data
    console.log('  📝 Populating data...');
    for (let i = 0; i < dataSize; i++) {
      await this.memoryController.store({
        id: `multi-head-${i}`,
        embedding: Array(embeddingDim).fill(0).map(() => Math.random())
      });
    }

    let memoryPeak = this.measureMemory();
    const latencies: number[] = [];

    // Run queries
    console.log('  ⚡ Running queries...');
    const startTime = Date.now();

    for (let i = 0; i < queryCount; i++) {
      const query = Array(embeddingDim).fill(0).map(() => Math.random());

      const queryStart = performance.now();
      await controller.computeMultiHeadAttention(query, {
        numHeads,
        topK: 10
      });
      const queryEnd = performance.now();

      latencies.push(queryEnd - queryStart);

      const currentMemory = this.measureMemory();
      if (currentMemory > memoryPeak) {
        memoryPeak = currentMemory;
      }

      if ((i + 1) % 50 === 0) {
        console.log(`    Progress: ${i + 1}/${queryCount} queries`);
      }
    }

    const endTime = Date.now();
    const duration = (endTime - startTime) / 1000;

    global.gc && global.gc();
    const memoryFinal = this.measureMemory();

    const result: BenchmarkResult = {
      name: `Multi-Head Attention (data=${dataSize}, queries=${queryCount}, heads=${numHeads}, dim=${embeddingDim})`,
      throughput: queryCount / duration,
      latency: this.calculatePercentiles(latencies),
      memory: {
        initial: memoryInitial / (1024 * 1024),
        peak: memoryPeak / (1024 * 1024),
        final: memoryFinal / (1024 * 1024)
      },
      duration
    };

    this.results.push(result);
    return result;
  }

  async benchmarkNAPIvsWASM(): Promise<void> {
    console.log('\n🔄 Benchmarking NAPI vs WASM Performance');

    // NAPI benchmark (RuVector native)
    await this.teardown();
    await this.setup();

    console.log('\n  📊 Testing NAPI (RuVector native)...');
    const napiResult = await this.benchmarkSelfAttention(1000, 500, 128);

    // WASM benchmark (browser fallback)
    // Note: This would require browser environment setup
    console.log('\n  📊 Testing WASM (browser fallback)...');
    console.log('    ⚠️  WASM benchmarks require browser environment');
    console.log('    ℹ️  See browser/attention-browser.test.js for WASM tests');

    console.log('\n  📈 NAPI Performance:');
    console.log(`    Throughput: ${napiResult.throughput.toFixed(2)} queries/sec`);
    console.log(`    Latency P50: ${napiResult.latency.p50.toFixed(2)}ms`);
    console.log(`    Latency P95: ${napiResult.latency.p95.toFixed(2)}ms`);
    console.log(`    Memory Peak: ${napiResult.memory.peak.toFixed(2)}MB`);
  }

  async benchmarkScalability(): Promise<void> {
    console.log('\n🔄 Benchmarking Scalability');

    const dataSizes = [100, 500, 1000, 5000];
    const queryCount = 100;

    for (const size of dataSizes) {
      await this.teardown();
      await this.setup();

      const result = await this.benchmarkSelfAttention(size, queryCount, 128);

      console.log(`\n  📊 Data size ${size}:`);
      console.log(`    Throughput: ${result.throughput.toFixed(2)} queries/sec`);
      console.log(`    Latency P95: ${result.latency.p95.toFixed(2)}ms`);
      console.log(`    Memory Peak: ${result.memory.peak.toFixed(2)}MB`);
    }
  }

  async benchmarkConcurrency(): Promise<void> {
    console.log('\n🔄 Benchmarking Concurrent Queries');

    await this.teardown();
    await this.setup();

    const controller = this.db.getController('self-attention') as SelfAttentionController;

    // Populate data
    console.log('  📝 Populating data...');
    for (let i = 0; i < 1000; i++) {
      await this.memoryController.store({
        id: `concurrent-${i}`,
        embedding: Array(128).fill(0).map(() => Math.random())
      });
    }

    const concurrencyLevels = [1, 5, 10, 20, 50];

    for (const concurrency of concurrencyLevels) {
      const latencies: number[] = [];
      const iterations = 100;

      const startTime = Date.now();

      for (let i = 0; i < iterations / concurrency; i++) {
        const promises = [];

        for (let j = 0; j < concurrency; j++) {
          const query = Array(128).fill(0).map(() => Math.random());

          const queryStart = performance.now();
          const promise = controller.computeAttention(query, { topK: 10 })
            .then(() => {
              latencies.push(performance.now() - queryStart);
            });

          promises.push(promise);
        }

        await Promise.all(promises);
      }

      const endTime = Date.now();
      const duration = (endTime - startTime) / 1000;

      const stats = this.calculatePercentiles(latencies);

      console.log(`\n  📊 Concurrency ${concurrency}:`);
      console.log(`    Throughput: ${(iterations / duration).toFixed(2)} queries/sec`);
      console.log(`    Latency P50: ${stats.p50.toFixed(2)}ms`);
      console.log(`    Latency P95: ${stats.p95.toFixed(2)}ms`);
    }
  }

  printSummary(): void {
    console.log('\n' + '='.repeat(80));
    console.log('📊 BENCHMARK SUMMARY');
    console.log('='.repeat(80));

    for (const result of this.results) {
      console.log(`\n${result.name}`);
      console.log(`  Throughput: ${result.throughput.toFixed(2)} ops/sec`);
      console.log(`  Latency:`);
      console.log(`    P50: ${result.latency.p50.toFixed(2)}ms`);
      console.log(`    P95: ${result.latency.p95.toFixed(2)}ms`);
      console.log(`    P99: ${result.latency.p99.toFixed(2)}ms`);
      console.log(`    Mean: ${result.latency.mean.toFixed(2)}ms`);
      console.log(`  Memory:`);
      console.log(`    Initial: ${result.memory.initial.toFixed(2)}MB`);
      console.log(`    Peak: ${result.memory.peak.toFixed(2)}MB`);
      console.log(`    Final: ${result.memory.final.toFixed(2)}MB`);
      console.log(`  Duration: ${result.duration.toFixed(2)}s`);
    }

    console.log('\n' + '='.repeat(80));
  }

  exportResults(outputPath: string): void {
    const report = {
      timestamp: new Date().toISOString(),
      platform: process.platform,
      nodeVersion: process.version,
      results: this.results
    };

    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
    console.log(`\n💾 Results exported to: ${outputPath}`);
  }
}

// Main execution
async function main() {
  console.log('🚀 Starting Attention Mechanism Benchmarks\n');

  const benchmark = new AttentionBenchmark();

  try {
    await benchmark.setup();

    // Run comprehensive benchmarks
    console.log('Running comprehensive benchmarks...\n');

    // Self-attention benchmarks
    await benchmark.benchmarkSelfAttention(500, 200, 128);
    await benchmark.benchmarkSelfAttention(1000, 200, 128);
    await benchmark.benchmarkSelfAttention(1000, 200, 256);

    // Cross-attention benchmarks
    await benchmark.teardown();
    await benchmark.setup();
    await benchmark.benchmarkCrossAttention(500, 200, 128);

    // Multi-head attention benchmarks
    await benchmark.teardown();
    await benchmark.setup();
    await benchmark.benchmarkMultiHeadAttention(500, 100, 4, 128);
    await benchmark.teardown();
    await benchmark.setup();
    await benchmark.benchmarkMultiHeadAttention(500, 100, 8, 128);

    // NAPI vs WASM comparison
    await benchmark.benchmarkNAPIvsWASM();

    // Scalability tests
    await benchmark.benchmarkScalability();

    // Concurrency tests
    await benchmark.teardown();
    await benchmark.setup();
    await benchmark.benchmarkConcurrency();

    // Print summary
    benchmark.printSummary();

    // Export results
    const outputPath = path.join(__dirname, 'benchmark-results.json');
    benchmark.exportResults(outputPath);

  } catch (error) {
    console.error('❌ Benchmark failed:', error);
    throw error;
  } finally {
    await benchmark.teardown();
  }

  console.log('\n✅ Benchmarks complete!');
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { AttentionBenchmark, BenchmarkResult };
