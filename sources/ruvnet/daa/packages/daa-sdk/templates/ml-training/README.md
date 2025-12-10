# ML Training DAA Template

Federated machine learning template with privacy-preserving training.

## Features

- **Federated Learning**: Distributed training across multiple nodes
- **Privacy-Preserving**: Differential privacy and secure aggregation
- **Multiple Architectures**: GPT-Mini, BERT-Tiny, ResNet-18
- **Flexible Aggregation**: FedAvg, FedProx, FedYogi strategies
- **Production-Ready**: Distributed inference and deployment

## Quick Start

### Installation

```bash
npm install
```

### Build

```bash
npm run build
```

### Run Examples

```bash
# Run main federated learning demo
npm start

# Run training node example
npm run train
```

## Project Structure

```
ml-training/
├── src/
│   ├── index.ts          # Main federated learning demo
│   ├── training-node.ts  # Training node implementation
│   ├── model.ts         # Model definitions
│   └── data-loader.ts   # Data loading utilities
├── package.json         # Dependencies and scripts
├── tsconfig.json        # TypeScript configuration
└── README.md           # This file
```

## Architecture

### Federated Learning Flow

```
┌──────────────┐
│ Coordinator  │  Manages global model and orchestrates training
└──────┬───────┘
       │
       ├────────────┬────────────┬────────────┐
       │            │            │            │
   ┌───▼───┐    ┌───▼───┐    ┌───▼───┐    ┌───▼───┐
   │Node 1 │    │Node 2 │    │Node 3 │    │Node N │
   │Local  │    │Local  │    │Local  │    │Local  │
   │Data   │    │Data   │    │Data   │    │Data   │
   └───────┘    └───────┘    └───────┘    └───────┘
```

### Training Process

1. **Initialization**: Coordinator broadcasts global model to all nodes
2. **Local Training**: Each node trains on local data
3. **Update Sharing**: Nodes send encrypted updates to coordinator
4. **Aggregation**: Coordinator aggregates updates (FedAvg, FedProx, etc.)
5. **Model Update**: New global model is created
6. **Repeat**: Process continues for multiple rounds

### Privacy Mechanisms

#### Differential Privacy

Adds calibrated noise to gradients to prevent model inversion attacks:

```typescript
const federatedConfig = {
  privacy: {
    differentialPrivacy: true,
    epsilon: 1.0,      // Privacy budget (lower = more private)
    delta: 1e-5,       // Failure probability
  },
};
```

#### Secure Aggregation

Encrypts local updates so coordinator only sees aggregated result:

```typescript
const federatedConfig = {
  privacy: {
    secureAggregation: true,  // Nodes encrypt their updates
  },
};
```

## Model Definitions

### GPT-Mini

Small transformer model for text generation:

```typescript
{
  layers: 6,
  hiddenSize: 512,
  attentionHeads: 8,
  vocabularySize: 50000,
  parameters: ~38M
}
```

### BERT-Tiny

Compact BERT for classification:

```typescript
{
  layers: 4,
  hiddenSize: 256,
  attentionHeads: 4,
  vocabularySize: 30000,
  parameters: ~16M
}
```

### ResNet-18

Convolutional network for images:

```typescript
{
  layers: 18,
  architecture: 'cnn',
  parameters: ~11.7M
}
```

## Aggregation Strategies

### FedAvg (Federated Averaging)

Simple weighted average of model parameters:

```
θ_global = Σ(n_i/n * θ_i)
```

Best for: IID data distribution

### FedProx

FedAvg with proximal term for heterogeneous data:

```
θ_global = Σ(n_i/n * θ_i) + μ||θ - θ_prev||²
```

Best for: Non-IID data, diverse node capabilities

### FedYogi

Adaptive aggregation with momentum:

Best for: Complex models, non-stationary data

## Training Node

Implement a custom training node:

```typescript
import { TrainingNode } from './src/training-node';

const config = {
  nodeId: 'node-001',
  coordinatorUrl: 'https://coordinator.example.com',
  localDataPath: './data/local',
  batchSize: 32,
  localEpochs: 3,
  gpuEnabled: true,
};

const node = new TrainingNode(config);
await node.initialize();
await node.register();
await node.participateInRound(1, globalModel);
```

## Data Loading

Load and preprocess training data:

```typescript
import { DataLoader } from './src/data-loader';

const loader = new DataLoader({
  dataPath: './data/training',
  batchSize: 32,
  shuffle: true,
  numWorkers: 4,
  prefetchFactor: 2,
});

await loader.loadDataset();

// Process batches
for (let epoch = 0; epoch < 100; epoch++) {
  loader.reset();
  while (true) {
    const batch = loader.getNextBatch();
    // Train on batch
    if (!batch.hasMore) break;
  }
}
```

## Data Augmentation

Apply augmentation to improve model generalization:

```typescript
import { DataAugmentation } from './src/data-loader';

// Image augmentation
const augmentedImage = DataAugmentation.augmentImage(image);

// Text augmentation
const augmentedText = DataAugmentation.augmentText(text);
```

## Preprocessing

Normalize and standardize data:

```typescript
import { DataPreprocessor } from './src/data-loader';

// Normalize to [0, 1]
const normalized = DataPreprocessor.normalize(features);

// Standardize to zero mean, unit variance
const standardized = DataPreprocessor.standardize(features);

// Tokenize text
const tokens = DataPreprocessor.tokenize(text, vocabularySize);
```

## Deployment

Deploy trained model for distributed inference:

```typescript
const daa = new DAA({
  prime: {
    enableTraining: false,  // Inference only
    gpuAcceleration: true,
  },
});

await daa.init();

// Inference configuration
const inferenceConfig = {
  nodes: 5,
  loadBalancing: 'round-robin',
  modelSharding: true,
  batchSize: 64,
};
```

## Performance Metrics

Track training progress:

```typescript
{
  totalRounds: 50,
  completedRounds: 3,
  currentLoss: 1.756,
  targetLoss: 0.8,
  improvement: 27.9,
  nodesParticipating: 10,
  samplesProcessed: 48000,
}
```

## Best Practices

1. **Data Distribution**: Ensure representative data across nodes
2. **Privacy Budget**: Choose epsilon carefully (1.0 is strong privacy)
3. **Node Selection**: Use reliable nodes with stable connections
4. **Batch Size**: Larger batches improve training stability
5. **Local Epochs**: 3-5 epochs per round is typical
6. **Aggregation**: Match strategy to data distribution
7. **Monitoring**: Track loss and accuracy across rounds

## Security

- **Quantum-Resistant**: ML-KEM and ML-DSA encryption
- **Differential Privacy**: Prevents model inversion
- **Secure Aggregation**: Coordinator can't see individual updates
- **Local Data**: Training data never leaves nodes

## Use Cases

### 1. Healthcare

Train medical models without sharing patient data:

```typescript
// Hospital A, B, C train jointly
// Each keeps patient data local
// Only encrypted updates shared
// HIPAA compliant
```

### 2. Financial Services

Fraud detection across multiple banks:

```typescript
// Banks collaborate on fraud detection
// Transaction data stays local
// Improved model accuracy
// Privacy preserved
```

### 3. IoT Devices

Edge device learning:

```typescript
// Smartphones, sensors train locally
// Bandwidth-efficient updates
// Privacy-preserving
// Personalized models
```

## Next Steps

1. Configure your model in `src/model.ts`
2. Implement data loading in `src/data-loader.ts`
3. Set up training nodes with `src/training-node.ts`
4. Customize aggregation strategy
5. Deploy to production cluster

## Resources

- [DAA SDK Documentation](https://github.com/ruvnet/daa)
- [Prime ML Framework](https://github.com/ruvnet/daa/tree/main/crates/prime)
- [Federated Learning Papers](https://arxiv.org/list/cs.LG/recent)

## License

MIT
