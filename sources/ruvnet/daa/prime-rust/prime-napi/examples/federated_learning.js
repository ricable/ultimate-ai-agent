/**
 * Federated Learning Example
 *
 * Demonstrates a complete federated learning setup with a coordinator
 * and multiple training nodes.
 */

const { Coordinator, TrainingNode, generateNodeId } = require('..');

// Simulate multiple training nodes
const NUM_NODES = 5;

async function createCoordinator() {
  console.log('📡 Creating coordinator...');

  const coordinator = new Coordinator('coordinator-main', {
    minNodesForRound: 3,
    heartbeatTimeoutMs: 10000,
    taskTimeoutMs: 120000,
    consensusThreshold: 0.66
  });

  await coordinator.init();
  console.log(`✓ Coordinator initialized: ${coordinator.nodeId}\n`);

  return coordinator;
}

async function createTrainingNodes(count) {
  console.log(`👥 Creating ${count} training nodes...`);

  const nodes = [];
  for (let i = 0; i < count; i++) {
    const nodeId = generateNodeId('trainer');
    const node = new TrainingNode(nodeId);

    await node.initTraining({
      batchSize: 32,
      learningRate: 0.001,
      epochs: 10,
      optimizer: 'adam',
      optimizerParams: { beta1: 0.9, beta2: 0.999 },
      aggregationStrategy: 'fedavg'
    });

    nodes.push(node);
    console.log(`  ✓ Created node ${i + 1}: ${nodeId}`);
  }

  console.log('');
  return nodes;
}

async function registerNodes(coordinator, nodes) {
  console.log('📝 Registering nodes with coordinator...');

  for (const node of nodes) {
    await coordinator.registerNode({
      nodeId: node.nodeId,
      nodeType: 'trainer',
      lastHeartbeat: Date.now(),
      reliabilityScore: 0.9 + Math.random() * 0.1
    });
    console.log(`  ✓ Registered: ${node.nodeId}`);
  }

  console.log('');
}

async function runTrainingRound(coordinator, nodes, roundNumber) {
  console.log(`\n🔄 Starting Training Round ${roundNumber}`);
  console.log('━'.repeat(50));

  // Start the round
  const round = await coordinator.startTraining();
  console.log(`Round ${round} initiated\n`);

  // Each node trains locally
  console.log('Training on nodes...');
  const gradients = [];

  for (const node of nodes) {
    const metrics = await node.trainEpoch();
    console.log(`  ${node.nodeId}:`);
    console.log(`    Loss: ${metrics.loss.toFixed(4)}, Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%`);

    // Simulate gradient generation (in real scenario, this would come from actual training)
    const gradient = Buffer.from(new Float32Array([
      Math.random(), Math.random(), Math.random(), Math.random()
    ]).buffer);
    gradients.push(gradient);
  }

  // Aggregate gradients
  console.log('\n📊 Aggregating gradients...');
  const aggregated = await nodes[0].aggregateGradients(gradients);
  console.log(`  ✓ Aggregated ${gradients.length} gradients (${aggregated.length} bytes)`);

  // Get progress
  const progress = await coordinator.getProgress();
  console.log('\nRound Progress:');
  console.log(`  Total nodes: ${progress.totalNodes}`);
  console.log(`  Completed: ${progress.completedNodes}`);
  console.log(`  Pending tasks: ${progress.pendingTasks}`);
}

async function main() {
  console.log('🌐 Prime ML NAPI - Federated Learning Example\n');
  console.log('='.repeat(50));
  console.log('');

  try {
    // Create coordinator
    const coordinator = await createCoordinator();

    // Create training nodes
    const nodes = await createTrainingNodes(NUM_NODES);

    // Register all nodes
    await registerNodes(coordinator, nodes);

    // Get initial status
    let status = await coordinator.getStatus();
    console.log('Initial Coordinator Status:');
    console.log(`  Active nodes: ${status.activeNodes}`);
    console.log(`  Current round: ${status.currentRound}`);
    console.log(`  Model version: ${status.modelVersion}`);

    // Run multiple training rounds
    const NUM_ROUNDS = 3;
    for (let round = 1; round <= NUM_ROUNDS; round++) {
      await runTrainingRound(coordinator, nodes, round);

      // Small delay between rounds
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Final status
    status = await coordinator.getStatus();
    console.log('\n📈 Final Coordinator Status:');
    console.log(`  Active nodes: ${status.activeNodes}`);
    console.log(`  Current round: ${status.currentRound}`);
    console.log(`  Model version: ${status.modelVersion}`);

    // Cleanup
    await coordinator.stop();
    console.log('\n✓ Federated learning session completed successfully!');

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
