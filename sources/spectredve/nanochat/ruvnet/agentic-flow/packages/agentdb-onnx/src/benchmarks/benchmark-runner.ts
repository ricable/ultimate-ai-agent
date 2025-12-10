#!/usr/bin/env tsx
/**
 * Comprehensive Performance Benchmarks for AgentDB + ONNX
 */

import { createONNXAgentDB } from '../index.js';
import { unlink } from 'fs/promises';
import chalk from 'chalk';

interface BenchmarkResult {
  name: string;
  operations: number;
  totalTime: number;
  opsPerSec: number;
  avgLatency: number;
  p50: number;
  p95: number;
  p99: number;
}

/**
 * Run a benchmark and collect statistics
 */
async function benchmark(
  name: string,
  operations: number,
  fn: () => Promise<void>
): Promise<BenchmarkResult> {
  const latencies: number[] = [];

  console.log(chalk.blue(`\n🏃 Running: ${name}`));
  console.log(chalk.gray(`   Operations: ${operations}`));

  const startTime = Date.now();

  for (let i = 0; i < operations; i++) {
    const opStart = Date.now();
    await fn();
    const opLatency = Date.now() - opStart;
    latencies.push(opLatency);

    if ((i + 1) % Math.max(1, Math.floor(operations / 10)) === 0) {
      process.stdout.write(chalk.gray(`.`));
    }
  }

  const totalTime = Date.now() - startTime;
  const opsPerSec = (operations / totalTime) * 1000;

  // Calculate percentiles
  latencies.sort((a, b) => a - b);
  const p50 = latencies[Math.floor(latencies.length * 0.5)];
  const p95 = latencies[Math.floor(latencies.length * 0.95)];
  const p99 = latencies[Math.floor(latencies.length * 0.99)];
  const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;

  console.log(); // Newline after dots

  return {
    name,
    operations,
    totalTime,
    opsPerSec,
    avgLatency,
    p50,
    p95,
    p99
  };
}

/**
 * Print benchmark results
 */
function printResult(result: BenchmarkResult) {
  console.log(chalk.green(`\n✅ ${result.name}`));
  console.log(chalk.white(`   Total time:     ${result.totalTime.toFixed(2)}ms`));
  console.log(chalk.white(`   Throughput:     ${result.opsPerSec.toFixed(2)} ops/sec`));
  console.log(chalk.white(`   Avg latency:    ${result.avgLatency.toFixed(2)}ms`));
  console.log(chalk.white(`   P50 latency:    ${result.p50.toFixed(2)}ms`));
  console.log(chalk.white(`   P95 latency:    ${result.p95.toFixed(2)}ms`));
  console.log(chalk.white(`   P99 latency:    ${result.p99.toFixed(2)}ms`));
}

/**
 * Main benchmark suite
 */
async function main() {
  console.log(chalk.bold.cyan('\n╔════════════════════════════════════════╗'));
  console.log(chalk.bold.cyan('║  AgentDB + ONNX Performance Benchmark  ║'));
  console.log(chalk.bold.cyan('╚════════════════════════════════════════╝\n'));

  const dbPath = './benchmark-agentdb-onnx.db';

  try {
    // Initialize AgentDB
    console.log(chalk.yellow('📦 Initializing AgentDB + ONNX...'));
    const agentdb = await createONNXAgentDB({
      dbPath,
      modelName: 'Xenova/all-MiniLM-L6-v2',
      useGPU: false,
      batchSize: 32,
      cacheSize: 10000
    });

    console.log(chalk.green('✅ Initialization complete\n'));

    const results: BenchmarkResult[] = [];

    // Benchmark 1: Single embedding generation
    results.push(await benchmark(
      'Single Embedding Generation',
      100,
      async () => {
        await agentdb.embedder.embed(`Test embedding ${Math.random()}`);
      }
    ));

    // Benchmark 2: Cached embedding access
    const cachedText = 'This text will be cached';
    await agentdb.embedder.embed(cachedText); // Warm up cache

    results.push(await benchmark(
      'Cached Embedding Access',
      1000,
      async () => {
        await agentdb.embedder.embed(cachedText);
      }
    ));

    // Benchmark 3: Batch embedding (10 items)
    results.push(await benchmark(
      'Batch Embedding (10 items)',
      50,
      async () => {
        const texts = Array.from({ length: 10 }, (_, i) =>
          `Batch text ${i} ${Math.random()}`
        );
        await agentdb.embedder.embedBatch(texts);
      }
    ));

    // Benchmark 4: Batch embedding (100 items)
    results.push(await benchmark(
      'Batch Embedding (100 items)',
      10,
      async () => {
        const texts = Array.from({ length: 100 }, (_, i) =>
          `Large batch text ${i} ${Math.random()}`
        );
        await agentdb.embedder.embedBatch(texts);
      }
    ));

    // Benchmark 5: Pattern storage
    results.push(await benchmark(
      'Pattern Storage (Single)',
      100,
      async () => {
        await agentdb.reasoningBank.storePattern({
          taskType: 'benchmark',
          approach: `Approach ${Math.random()}`,
          successRate: Math.random()
        });
      }
    ));

    // Benchmark 6: Pattern batch storage (using loops)
    results.push(await benchmark(
      'Pattern Storage (Batch of 10)',
      20,
      async () => {
        const patterns = Array.from({ length: 10 }, (_, i) => ({
          taskType: 'batch-benchmark',
          approach: `Batch approach ${i} ${Math.random()}`,
          successRate: Math.random()
        }));
        for (const pattern of patterns) {
          await agentdb.reasoningBank.storePattern(pattern);
        }
      }
    ));

    // Benchmark 7: Pattern search - pre-populate database
    for (let i = 0; i < 100; i++) {
      await agentdb.reasoningBank.storePattern({
        taskType: 'search-test',
        approach: `Search approach ${i}`,
        successRate: Math.random()
      });
    }

    results.push(await benchmark(
      'Pattern Search (k=10)',
      100,
      async () => {
        await agentdb.reasoningBank.searchPatterns({
          task: 'search approach',
          k: 10
        });
      }
    ));

    // Benchmark 8: Episode storage
    results.push(await benchmark(
      'Episode Storage (Single)',
      100,
      async () => {
        await agentdb.reflexionMemory.storeEpisode({
          sessionId: 'benchmark',
          task: `Task ${Math.random()}`,
          reward: Math.random(),
          success: Math.random() > 0.5,
          critique: 'Benchmark critique'
        });
      }
    ));

    // Benchmark 9: Episode batch storage (using loops)
    results.push(await benchmark(
      'Episode Storage (Batch of 10)',
      20,
      async () => {
        const episodes = Array.from({ length: 10 }, (_, i) => ({
          sessionId: 'batch-benchmark',
          task: `Batch task ${i} ${Math.random()}`,
          reward: Math.random(),
          success: Math.random() > 0.5,
          critique: `Batch critique ${i}`
        }));
        for (const episode of episodes) {
          await agentdb.reflexionMemory.storeEpisode(episode);
        }
      }
    ));

    // Benchmark 10: Episode retrieval - pre-populate database
    for (let i = 0; i < 100; i++) {
      await agentdb.reflexionMemory.storeEpisode({
        sessionId: 'retrieval-test',
        task: `Retrieval task ${i}`,
        reward: Math.random(),
        success: Math.random() > 0.5
      });
    }

    results.push(await benchmark(
      'Episode Retrieval (k=10)',
      100,
      async () => {
        await agentdb.reflexionMemory.retrieveRelevant({
          task: 'retrieval task',
          k: 10
        });
      }
    ));

    // Print all results
    console.log(chalk.bold.cyan('\n\n╔════════════════════════════════════════╗'));
    console.log(chalk.bold.cyan('║         Benchmark Results              ║'));
    console.log(chalk.bold.cyan('╚════════════════════════════════════════╝'));

    results.forEach(printResult);

    // Print summary
    console.log(chalk.bold.cyan('\n\n╔════════════════════════════════════════╗'));
    console.log(chalk.bold.cyan('║            Summary                     ║'));
    console.log(chalk.bold.cyan('╚════════════════════════════════════════╝\n'));

    const stats = agentdb.getStats();

    console.log(chalk.white('📊 Overall Statistics:'));
    console.log(chalk.white(`   Total embeddings:   ${stats.embedder.totalEmbeddings}`));
    console.log(chalk.white(`   Avg embedding time: ${stats.embedder.avgLatency.toFixed(2)}ms`));
    console.log(chalk.white(`   Cache hit rate:     ${(stats.embedder.cache.hitRate * 100).toFixed(1)}%`));
    console.log(chalk.white(`   Cache size:         ${stats.embedder.cache.size}/${stats.embedder.cache.maxSize}`));

    // Calculate speedup from batching
    const singlePattern = results.find(r => r.name === 'Pattern Storage (Single)');
    const batchPattern = results.find(r => r.name === 'Pattern Storage (Batch of 10)');

    if (singlePattern && batchPattern) {
      const speedup = (singlePattern.opsPerSec * 10) / batchPattern.opsPerSec;
      console.log(chalk.white(`\n🚀 Batch speedup:      ${speedup.toFixed(2)}x faster`));
    }

    // Cleanup
    await agentdb.close();

    console.log(chalk.green('\n✅ Benchmark complete!\n'));
  } catch (error) {
    console.error(chalk.red('\n❌ Benchmark failed:'), error);
    process.exit(1);
  } finally {
    try {
      await unlink(dbPath);
    } catch {}
  }
}

// Run benchmarks
main().catch(console.error);
