# Prime ML NAPI - Quick Start Guide

## 📦 Installation & Build

### Prerequisites

- Node.js >= 16
- Rust >= 1.77 (for NAPI-rs build script compatibility)
- Cargo

### Building from Source

```bash
cd /home/user/daa/prime-rust/prime-napi

# Install Node.js dependencies
npm install

# Build native module for your platform
npm run build

# Or for debug build
npm run build:debug
```

The build will create a `.node` file in the project directory that contains the native bindings.

## 🚀 Quick Example

### 1. Basic Training Node

```javascript
const { TrainingNode } = require('@prime/ml-napi');

async function quickStart() {
  // Create a training node
  const node = new TrainingNode('my-node');

  // Initialize with configuration
  await node.initTraining({
    batchSize: 32,
    learningRate: 0.001,
    epochs: 10,
    optimizer: 'adam',
    aggregationStrategy: 'fedavg'
  });

  // Train for one epoch
  const metrics = await node.trainEpoch();
  console.log('Training metrics:', metrics);
  // Output: { loss: 0.5, accuracy: 0.85, samplesProcessed: 1000, computationTimeMs: 100 }
}

quickStart().catch(console.error);
```

### 2. Zero-Copy Tensor Operations

```javascript
const { tensorFromBuffer, TensorBuffer } = require('@prime/ml-napi');

// Create a tensor from Float32Array (zero-copy)
const data = new Float32Array([1, 2, 3, 4, 5, 6]);
const buffer = Buffer.from(data.buffer);
const tensor = tensorFromBuffer(buffer, [2, 3], 'f32');

console.log('Shape:', tensor.shape);        // [2, 3]
console.log('Elements:', tensor.numElements());  // 6
console.log('Bytes:', tensor.byteSize());    // 24 (6 * 4 bytes)

// Reshape (zero-copy - no data copied!)
const flat = tensor.reshape([6]);
console.log('New shape:', flat.shape);      // [6]
```

### 3. Gradient Aggregation

```javascript
const { TrainingNode } = require('@prime/ml-napi');

async function aggregateExample() {
  const node = new TrainingNode('aggregator');
  await node.initTraining({
    batchSize: 32,
    learningRate: 0.001,
    epochs: 1,
    optimizer: 'adam',
    aggregationStrategy: 'fedavg'  // or 'trimmed_mean'
  });

  // Create gradient buffers from different nodes
  const gradients = [
    Buffer.from(new Float32Array([0.1, 0.2, 0.3, 0.4]).buffer),
    Buffer.from(new Float32Array([0.15, 0.25, 0.35, 0.45]).buffer),
    Buffer.from(new Float32Array([0.12, 0.22, 0.32, 0.42]).buffer)
  ];

  // Aggregate using configured strategy
  const aggregated = await node.aggregateGradients(gradients);
  console.log('Aggregated gradient size:', aggregated.length);
}

aggregateExample().catch(console.error);
```

### 4. Federated Learning Coordinator

```javascript
const { Coordinator } = require('@prime/ml-napi');

async function coordinatorExample() {
  // Create and initialize coordinator
  const coordinator = new Coordinator('main-coordinator', {
    minNodesForRound: 3,
    heartbeatTimeoutMs: 10000,
    taskTimeoutMs: 120000,
    consensusThreshold: 0.66
  });

  await coordinator.init();

  // Register training nodes
  await coordinator.registerNode({
    nodeId: 'node-1',
    nodeType: 'trainer',
    lastHeartbeat: Date.now(),
    reliabilityScore: 0.95
  });

  // Start training round
  const roundNumber = await coordinator.startTraining();
  console.log('Started round:', roundNumber);

  // Check progress
  const progress = await coordinator.getProgress();
  console.log('Progress:', progress);

  // Get status
  const status = await coordinator.getStatus();
  console.log('Active nodes:', status.activeNodes);
  console.log('Current round:', status.currentRound);
}

coordinatorExample().catch(console.error);
```

## 📖 Running the Examples

After building, you can run the included examples:

```bash
# Basic training demonstration
node examples/basic_training.js

# Full federated learning workflow
node examples/federated_learning.js

# Zero-copy tensor operations
node examples/zero_copy_tensors.js

# Gradient aggregation strategies
node examples/gradient_aggregation.js
```

## 🧪 Running Tests

```bash
# Run integration tests
npm test

# Or directly with Node
node --test tests/integration.test.js
```

## 🎯 Common Use Cases

### Use Case 1: Training with Custom Configuration

```javascript
const { TrainingNode, createDefaultTrainingConfig } = require('@prime/ml-napi');

async function customTraining() {
  const node = new TrainingNode('custom-node');

  // Start with defaults and customize
  const config = createDefaultTrainingConfig();
  config.batchSize = 64;
  config.learningRate = 0.0001;
  config.optimizer = 'adamw';
  config.optimizerParams = {
    beta1: 0.9,
    beta2: 0.999,
    weightDecay: 0.01
  };

  await node.initTraining(config);

  // Train for multiple epochs
  for (let i = 0; i < 10; i++) {
    const metrics = await node.trainEpoch();
    console.log(`Epoch ${i + 1}: Loss ${metrics.loss.toFixed(4)}`);
  }
}

customTraining().catch(console.error);
```

### Use Case 2: Robust Aggregation with Outlier Handling

```javascript
const { TrainingNode } = require('@prime/ml-napi');

async function robustAggregation() {
  const node = new TrainingNode('aggregator');

  // Use trimmed mean for robustness against Byzantine attacks
  await node.initTraining({
    batchSize: 32,
    learningRate: 0.001,
    epochs: 1,
    optimizer: 'adam',
    aggregationStrategy: 'trimmed_mean'  // Removes top/bottom 10% before averaging
  });

  // Even with outliers, trimmed mean is robust
  const gradients = [
    // Normal gradients
    Buffer.from(new Float32Array([0.1, 0.2, 0.3]).buffer),
    Buffer.from(new Float32Array([0.11, 0.21, 0.31]).buffer),
    Buffer.from(new Float32Array([0.12, 0.22, 0.32]).buffer),
    // Outlier (Byzantine node)
    Buffer.from(new Float32Array([10.0, 20.0, 30.0]).buffer),
  ];

  const aggregated = await node.aggregateGradients(gradients);
  // Outlier influence is minimized!
}

robustAggregation().catch(console.error);
```

### Use Case 3: Multi-Node Federated Training

```javascript
const { Coordinator, TrainingNode, generateNodeId } = require('@prime/ml-napi');

async function federatedTraining() {
  // Setup coordinator
  const coordinator = new Coordinator('coordinator-1');
  await coordinator.init();

  // Create multiple training nodes
  const nodes = [];
  for (let i = 0; i < 5; i++) {
    const nodeId = generateNodeId('trainer');
    const node = new TrainingNode(nodeId);

    await node.initTraining({
      batchSize: 32,
      learningRate: 0.001,
      epochs: 10,
      optimizer: 'adam',
      aggregationStrategy: 'fedavg'
    });

    // Register with coordinator
    await coordinator.registerNode({
      nodeId,
      nodeType: 'trainer',
      lastHeartbeat: Date.now(),
      reliabilityScore: 0.9
    });

    nodes.push(node);
  }

  // Run federated training rounds
  for (let round = 0; round < 10; round++) {
    // Start round
    await coordinator.startTraining();

    // Each node trains locally
    const gradients = [];
    for (const node of nodes) {
      await node.trainEpoch();
      // In real scenario, get actual gradients from model
      const gradient = Buffer.from(new Float32Array([
        Math.random(), Math.random(), Math.random(), Math.random()
      ]).buffer);
      gradients.push(gradient);
    }

    // Aggregate gradients
    const aggregated = await nodes[0].aggregateGradients(gradients);

    // Broadcast aggregated updates back to nodes (implementation specific)
    console.log(`Round ${round + 1} complete, aggregated ${gradients.length} gradients`);
  }

  await coordinator.stop();
}

federatedTraining().catch(console.error);
```

## 🔧 Configuration Options

### Training Configuration

```typescript
interface TrainingConfig {
  batchSize: number;           // Training batch size (default: 32)
  learningRate: number;        // Learning rate (default: 0.001)
  epochs: number;              // Number of epochs (default: 10)
  optimizer: string;           // 'sgd', 'adam', or 'adamw' (default: 'adam')
  optimizerParams?: {          // Optimizer-specific parameters
    momentum?: number;         // For SGD (default: 0.0)
    beta1?: number;            // For Adam/AdamW (default: 0.9)
    beta2?: number;            // For Adam/AdamW (default: 0.999)
    weightDecay?: number;      // For AdamW (default: 0.01)
  };
  aggregationStrategy: string; // 'fedavg', 'trimmed_mean', 'krum', 'secure'
}
```

### Coordinator Configuration

```typescript
interface CoordinatorConfig {
  minNodesForRound: number;      // Min nodes to start round (default: 3)
  heartbeatTimeoutMs: number;    // Heartbeat timeout (default: 5000)
  taskTimeoutMs: number;         // Task timeout (default: 60000)
  consensusThreshold: number;    // Consensus threshold 0-1 (default: 0.66)
}
```

## 🎓 Best Practices

### 1. Always Use Zero-Copy When Possible

```javascript
// ✅ Good: Zero-copy
const buffer = Buffer.from(data.buffer);
const tensor = tensorFromBuffer(buffer, [100, 100], 'f32');
const reshaped = tensor.reshape([10000]);

// ❌ Avoid: Unnecessary copying
const array = tensor.toF32Array();  // Creates a copy
```

### 2. Choose the Right Aggregation Strategy

```javascript
// For trusted environments (fast)
aggregationStrategy: 'fedavg'

// For untrusted environments (robust)
aggregationStrategy: 'trimmed_mean'

// For adversarial scenarios (Byzantine-robust, needs implementation)
aggregationStrategy: 'krum'

// For privacy-preserving (needs implementation)
aggregationStrategy: 'secure'
```

### 3. Handle Errors Properly

```javascript
try {
  await node.initTraining(config);
  const metrics = await node.trainEpoch();
} catch (error) {
  if (error.message.includes('not initialized')) {
    console.error('Training not initialized');
  } else {
    console.error('Training error:', error);
  }
}
```

### 4. Monitor Training Metrics

```javascript
const metrics = await node.trainEpoch();

// Check for divergence
if (metrics.loss > 10.0 || isNaN(metrics.loss)) {
  console.warn('Training may be diverging!');
}

// Check for convergence
if (metrics.loss < 0.01 && metrics.accuracy > 0.99) {
  console.log('Training converged!');
}
```

## 🐛 Troubleshooting

### Build Issues

```bash
# If build fails, try cleaning first
rm -rf node_modules target
npm install
npm run build
```

### Runtime Issues

```javascript
// Issue: Training not initialized
// Solution: Call initTraining before trainEpoch
await node.initTraining(config);

// Issue: Gradient size mismatch
// Solution: Ensure all gradients have the same size
const size = gradients[0].length;
const allSameSize = gradients.every(g => g.length === size);
```

### Import Issues

```javascript
// CommonJS
const { TrainingNode } = require('@prime/ml-napi');

// ES Modules (if using .mjs)
import { TrainingNode } from '@prime/ml-napi';
```

## 📚 Additional Resources

- **Full API Documentation**: See `README.md`
- **Implementation Details**: See `IMPLEMENTATION.md`
- **Project Summary**: See `SUMMARY.md`
- **Examples**: See `examples/` directory
- **Tests**: See `tests/` directory

## 🤝 Support

For issues and questions:
- Check the examples in `examples/`
- Read the API documentation in `README.md`
- Review implementation details in `IMPLEMENTATION.md`

## 🎉 You're Ready!

You now have everything you need to:
- ✅ Build and run Prime ML NAPI
- ✅ Create training nodes
- ✅ Set up federated learning
- ✅ Use zero-copy tensor operations
- ✅ Aggregate gradients
- ✅ Coordinate distributed training

Happy federated learning! 🚀
