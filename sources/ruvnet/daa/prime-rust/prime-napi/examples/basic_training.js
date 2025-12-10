/**
 * Basic Training Example
 *
 * Demonstrates how to set up a simple training node and execute training epochs.
 */

const { TrainingNode, createDefaultTrainingConfig } = require('..');

async function main() {
  console.log('🚀 Prime ML NAPI - Basic Training Example\n');

  try {
    // Create a training node
    console.log('Creating training node...');
    const node = new TrainingNode('node-1');
    console.log(`✓ Created node: ${node.nodeId}\n`);

    // Get default configuration
    const config = createDefaultTrainingConfig();
    console.log('Training configuration:', JSON.stringify(config, null, 2));

    // Initialize training
    console.log('\nInitializing training...');
    await node.initTraining(config);
    console.log('✓ Training initialized\n');

    // Train for multiple epochs
    console.log('Starting training...\n');
    for (let epoch = 1; epoch <= 5; epoch++) {
      const metrics = await node.trainEpoch();
      console.log(`Epoch ${epoch}:`);
      console.log(`  Loss: ${metrics.loss.toFixed(4)}`);
      console.log(`  Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);
      console.log(`  Samples: ${metrics.samplesProcessed}`);
      console.log(`  Time: ${metrics.computationTimeMs}ms\n`);
    }

    // Get final status
    const status = await node.getStatus();
    console.log('Final status:', JSON.stringify(status, null, 2));

    console.log('\n✓ Training completed successfully!');

  } catch (error) {
    console.error('❌ Error:', error.message);
    process.exit(1);
  }
}

main();
