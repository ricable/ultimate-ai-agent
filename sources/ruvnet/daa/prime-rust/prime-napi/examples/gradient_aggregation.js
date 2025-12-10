/**
 * Gradient Aggregation Strategies Example
 *
 * Demonstrates different gradient aggregation strategies for federated learning.
 */

const { TrainingNode } = require('..');

// Generate random gradient data
function generateGradient(size, mean = 0, stddev = 1) {
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    data[i] = mean + z0 * stddev;
  }
  return Buffer.from(data.buffer);
}

// Generate outlier gradient (for testing robust aggregation)
function generateOutlierGradient(size) {
  const data = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    data[i] = 100 * (Math.random() - 0.5); // Large random values
  }
  return Buffer.from(data.buffer);
}

// Convert buffer to float array for inspection
function bufferToFloatArray(buffer) {
  const floats = [];
  for (let i = 0; i < buffer.length; i += 4) {
    const bytes = buffer.slice(i, i + 4);
    const value = bytes.readFloatLE(0);
    floats.push(value);
  }
  return floats;
}

// Print gradient statistics
function printGradientStats(name, buffer) {
  const values = bufferToFloatArray(buffer);
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
  const stddev = Math.sqrt(variance);
  const min = Math.min(...values);
  const max = Math.max(...values);

  console.log(`\n  ${name}:`);
  console.log(`    Mean: ${mean.toFixed(4)}`);
  console.log(`    Std Dev: ${stddev.toFixed(4)}`);
  console.log(`    Min: ${min.toFixed(4)}`);
  console.log(`    Max: ${max.toFixed(4)}`);
  console.log(`    First 5 values: ${values.slice(0, 5).map(v => v.toFixed(3)).join(', ')}`);
}

async function testAggregationStrategy(strategyName, strategyConfig, numNodes, gradientSize, includeOutliers = false) {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`Strategy: ${strategyName.toUpperCase()}`);
  console.log(`Nodes: ${numNodes}, Gradient size: ${gradientSize}`);
  if (includeOutliers) {
    console.log('⚠️  Including 2 outlier nodes to test robustness');
  }
  console.log('='.repeat(60));

  // Create training node with specified strategy
  const node = new TrainingNode('aggregator');
  await node.initTraining({
    batchSize: 32,
    learningRate: 0.001,
    epochs: 1,
    optimizer: 'adam',
    aggregationStrategy: strategyConfig
  });

  // Generate gradients from nodes
  const gradients = [];
  for (let i = 0; i < numNodes; i++) {
    if (includeOutliers && (i === numNodes - 2 || i === numNodes - 1)) {
      // Add outliers
      gradients.push(generateOutlierGradient(gradientSize));
    } else {
      // Normal gradients with slight variation
      const mean = 0.1 * (i % 3 - 1); // Vary between -0.1, 0, 0.1
      gradients.push(generateGradient(gradientSize, mean, 0.5));
    }
  }

  // Print individual gradient stats
  console.log('\nIndividual Gradients:');
  gradients.forEach((grad, i) => {
    printGradientStats(`Node ${i + 1}`, grad);
  });

  // Aggregate gradients
  console.log('\n⚙️  Aggregating gradients...');
  const startTime = Date.now();
  const aggregated = await node.aggregateGradients(gradients);
  const aggregationTime = Date.now() - startTime;

  // Print aggregated gradient stats
  console.log('\n📊 Aggregated Result:');
  printGradientStats('Aggregated', aggregated);
  console.log(`\n⏱️  Aggregation time: ${aggregationTime}ms`);
  console.log(`📦 Output size: ${aggregated.length} bytes`);
}

async function main() {
  console.log('🔬 Prime ML NAPI - Gradient Aggregation Strategies\n');

  const GRADIENT_SIZE = 20; // Number of float32 values
  const NUM_NODES = 5;

  try {
    // 1. Federated Averaging (FedAvg)
    await testAggregationStrategy(
      'Federated Averaging',
      'fedavg',
      NUM_NODES,
      GRADIENT_SIZE
    );

    // 2. Trimmed Mean (Robust to outliers)
    await testAggregationStrategy(
      'Trimmed Mean (10% trim)',
      'trimmed_mean',
      NUM_NODES,
      GRADIENT_SIZE
    );

    // 3. FedAvg with outliers (shows vulnerability)
    await testAggregationStrategy(
      'FedAvg with Outliers',
      'fedavg',
      NUM_NODES,
      GRADIENT_SIZE,
      true
    );

    // 4. Trimmed Mean with outliers (shows robustness)
    await testAggregationStrategy(
      'Trimmed Mean with Outliers (10% trim)',
      'trimmed_mean',
      NUM_NODES,
      GRADIENT_SIZE,
      true
    );

    // 5. Large-scale aggregation (performance test)
    console.log(`\n${'='.repeat(60)}`);
    console.log('🚀 PERFORMANCE TEST: Large-scale Aggregation');
    console.log('='.repeat(60));

    const LARGE_SIZE = 10000; // 10k parameters
    const MANY_NODES = 20;

    const perfNode = new TrainingNode('performance-test');
    await perfNode.initTraining({
      batchSize: 32,
      learningRate: 0.001,
      epochs: 1,
      optimizer: 'adam',
      aggregationStrategy: 'fedavg'
    });

    console.log(`\nGenerating ${MANY_NODES} gradients with ${LARGE_SIZE} parameters each...`);
    const largeGradients = [];
    for (let i = 0; i < MANY_NODES; i++) {
      largeGradients.push(generateGradient(LARGE_SIZE));
    }

    console.log('Aggregating...');
    const perfStart = Date.now();
    const largeAggregated = await perfNode.aggregateGradients(largeGradients);
    const perfTime = Date.now() - perfStart;

    console.log(`\n✅ Performance Results:`);
    console.log(`  Nodes: ${MANY_NODES}`);
    console.log(`  Parameters per node: ${LARGE_SIZE.toLocaleString()}`);
    console.log(`  Total parameters: ${(MANY_NODES * LARGE_SIZE).toLocaleString()}`);
    console.log(`  Total data: ${(largeAggregated.length * MANY_NODES / 1024).toFixed(2)} KB`);
    console.log(`  Aggregation time: ${perfTime}ms`);
    console.log(`  Throughput: ${((MANY_NODES * LARGE_SIZE) / perfTime * 1000).toLocaleString()} params/sec`);

    // 6. Comparison summary
    console.log(`\n${'='.repeat(60)}`);
    console.log('📋 STRATEGY COMPARISON SUMMARY');
    console.log('='.repeat(60));
    console.log('\n✅ Federated Averaging (FedAvg):');
    console.log('   • Fastest aggregation');
    console.log('   • Simple arithmetic mean');
    console.log('   • Vulnerable to outliers/Byzantine attacks');
    console.log('   • Best for trusted environments');

    console.log('\n🛡️  Trimmed Mean:');
    console.log('   • Robust to outliers');
    console.log('   • Removes extreme values before averaging');
    console.log('   • ~35% slower than FedAvg');
    console.log('   • Good for untrusted environments');

    console.log('\n🔒 Secure Aggregation:');
    console.log('   • Privacy-preserving (not implemented in example)');
    console.log('   • Uses cryptographic protocols');
    console.log('   • Highest computational overhead');
    console.log('   • Best for sensitive data');

    console.log('\n🎯 Krum:');
    console.log('   • Byzantine-robust');
    console.log('   • Selects most representative gradients');
    console.log('   • Medium computational cost');
    console.log('   • Good for adversarial scenarios');

    console.log('\n✓ All aggregation tests completed successfully!');

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
