#!/usr/bin/env node
/**
 * Prime ML benchmarks for WASM/JavaScript implementation
 * Measures distributed training performance
 */

import Benchmark from 'benchmark';
import chalk from 'chalk';
import { writeFile } from 'fs/promises';

console.log(chalk.blue.bold('\n🧠 Prime ML Benchmarks\n'));
console.log(chalk.gray('Testing distributed training operations...\n'));

const suite = new Benchmark.Suite('Prime ML Benchmarks');
const results = {
  timestamp: new Date().toISOString(),
  benchmarks: []
};

// Generate random gradient data
function generateGradients(count, size) {
  return Array.from({ length: count }, () =>
    Array.from({ length: size }, () => Math.random())
  );
}

// Federated averaging
function federatedAverage(gradients) {
  const size = gradients[0].length;
  const aggregated = new Array(size).fill(0);

  for (const gradient of gradients) {
    for (let i = 0; i < size; i++) {
      aggregated[i] += gradient[i];
    }
  }

  for (let i = 0; i < size; i++) {
    aggregated[i] /= gradients.length;
  }

  return aggregated;
}

// Trimmed mean (Byzantine-tolerant)
function trimmedMean(gradients, trimRatio = 0.1) {
  const size = gradients[0].length;
  const trimCount = Math.floor(gradients.length * trimRatio);
  const aggregated = new Array(size);

  for (let i = 0; i < size; i++) {
    const values = gradients.map(g => g[i]).sort((a, b) => a - b);
    const trimmed = values.slice(trimCount, values.length - trimCount);
    aggregated[i] = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
  }

  return aggregated;
}

// Top-k sparsification
function topKSparsification(gradient, k) {
  const indices = gradient
    .map((v, i) => ({ index: i, value: Math.abs(v) }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k)
    .map(item => ({ index: item.index, value: gradient[item.index] }));

  return indices;
}

// Gradient Aggregation benchmarks
for (const nodeCount of [5, 10, 20, 50]) {
  for (const gradientSize of [1000, 10000, 100000]) {
    suite.add(`Federated Avg ${nodeCount} nodes ${gradientSize} params`, () => {
      const gradients = generateGradients(nodeCount, gradientSize);
      federatedAverage(gradients);
    });
  }
}

// Byzantine-tolerant aggregation benchmarks
for (const nodeCount of [10, 20, 50]) {
  suite.add(`Byzantine Tolerant ${nodeCount} nodes`, () => {
    const gradients = generateGradients(nodeCount, 10000);
    trimmedMean(gradients, 0.1);
  });
}

// Model update benchmarks
for (const modelSize of [1000, 10000, 100000, 1000000]) {
  suite.add(`Model Update ${modelSize} params`, () => {
    const model = new Array(modelSize).fill(0.5);
    const gradient = generateGradients(1, modelSize)[0];
    const learningRate = 0.01;

    for (let i = 0; i < modelSize; i++) {
      model[i] -= learningRate * gradient[i];
    }
  });
}

// Zero-copy operations (using TypedArrays)
for (const dataSize of [1024, 10240, 102400, 1024000]) {
  suite.add(`Zero-Copy ${dataSize} bytes`, () => {
    const data = new Uint8Array(dataSize);
    // Simulate zero-copy by using views
    const view = new DataView(data.buffer);
    let checksum = 0;
    for (let i = 0; i < data.length; i++) {
      checksum += view.getUint8(i);
    }
  });
}

// Gradient compression benchmarks
for (const gradientSize of [1000, 10000, 100000]) {
  suite.add(`Gradient Compression ${gradientSize} params`, () => {
    const gradient = generateGradients(1, gradientSize)[0];
    const k = Math.floor(gradientSize / 10); // Keep top 10%
    topKSparsification(gradient, k);
  });
}

// Model serialization benchmarks
for (const modelSize of [1000, 10000, 100000]) {
  suite.add(`Model Serialization ${modelSize} params`, () => {
    const model = new Array(modelSize).fill(0.5);

    // Serialize to Float32Array
    const buffer = new Float32Array(model);

    // Deserialize back
    const restored = Array.from(buffer);
  });
}

// Event handlers
suite.on('cycle', (event) => {
  const bench = event.target;
  const name = bench.name;
  const opsPerSec = bench.hz.toFixed(2);
  const meanMs = (bench.stats.mean * 1000).toFixed(3);
  const rme = bench.stats.rme.toFixed(2);

  console.log(chalk.green('✓'), chalk.white(name));
  console.log(chalk.gray(`  ${opsPerSec} ops/sec (${meanMs}ms/op) ±${rme}%`));

  results.benchmarks.push({
    name,
    hz: bench.hz,
    mean: bench.stats.mean * 1000,
    median: bench.stats.median * 1000,
    rme: bench.stats.rme,
    samples: bench.stats.sample.length
  });
});

suite.on('complete', async function() {
  console.log(chalk.blue.bold('\n📊 Prime ML Benchmark Summary\n'));

  // Calculate statistics
  const avgOpsPerSec = results.benchmarks.reduce((sum, b) => sum + b.hz, 0) / results.benchmarks.length;
  console.log(chalk.cyan('Average throughput:'), chalk.white(`${avgOpsPerSec.toFixed(2)} ops/sec`));

  // Find fastest operations
  const fastest = results.benchmarks
    .sort((a, b) => b.hz - a.hz)
    .slice(0, 5);

  console.log(chalk.cyan('\nTop 5 fastest operations:'));
  fastest.forEach((bench, i) => {
    console.log(chalk.white(`  ${i + 1}. ${bench.name}: ${bench.hz.toFixed(2)} ops/sec`));
  });

  // Save results
  await writeFile(
    'reports/prime_ml_results.json',
    JSON.stringify(results, null, 2)
  );

  console.log(chalk.gray('\n💾 Results saved to reports/prime_ml_results.json\n'));
});

suite.on('error', (event) => {
  console.error(chalk.red('Error:'), event.target.error);
});

suite.run({ async: false });
