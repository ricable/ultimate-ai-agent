# AgentDB GNN Learning Guide

Graph Neural Network query enhancement and adaptive learning in AgentDB v2

---

## Overview

AgentDB v2 includes **optional GNN (Graph Neural Network) learning** that enhances vector search queries based on neighbor context and user feedback. This enables adaptive, self-improving search that learns from experience.

---

## What is GNN Learning?

### Traditional Vector Search

```
Query → Embedding → K-NN Search → Results
```

**Limitation**: Query embedding is fixed, doesn't learn from context

### GNN-Enhanced Search

```
Query → Embedding → GNN Enhancement → K-NN Search → Results
                      ↑
                  (Neighbor context + learned weights)
```

**Benefit**: Query enhanced based on:
- Nearby vectors in the graph
- User feedback (success/failure)
- Historical search patterns

### Key Concepts

1. **Graph Attention**: Multi-head attention over k-nearest neighbors
2. **Adaptive Weights**: Learned importance of different neighbors
3. **Contextual Enhancement**: Query modified based on local graph structure
4. **Continuous Learning**: Improves from feedback over time

---

## When to Use GNN

### ✅ Good Use Cases

- **Adaptive AI Agents**: Agents that learn from interactions
- **Personalized Search**: User-specific query preferences
- **Context-Aware Retrieval**: Queries depend on document relationships
- **Self-Learning Systems**: Continuous improvement from feedback
- **Multi-Modal Search**: Cross-modal query enhancement

### ❌ Not Ideal For

- **Static Datasets**: No feedback loop to learn from
- **Simple K-NN**: Traditional search is sufficient
- **Browser Applications**: GNN requires Node.js (no browser support yet)
- **Small Datasets**: (<1000 vectors) - not enough data to learn

---

## Installation

### Prerequisites

```bash
npm install @ruvector/core  # Required: RuVector backend
npm install @ruvector/gnn   # Required: GNN learning
```

**Note**: GNN requires RuVector Core. Not available with SQLite backends.

### Verify Installation

```bash
npx agentdb detect
```

Output should show:
```
📊 Backend Detection Results:
  Backend:     ruvector
  GNN:         ✅
```

---

## Basic Usage

### 1. Initialize with GNN

```typescript
import { createVectorDB } from 'agentdb';

const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'ruvector',

  // Enable GNN
  enableGNN: true,
  gnn: {
    inputDim: 384,
    outputDim: 384,
    heads: 4,              // Multi-head attention
    learningRate: 0.001
  }
});
```

### 2. Insert Training Data

GNN needs data to learn from:

```typescript
// Insert vectors as usual
await db.insertBatch([
  {
    text: 'How to implement OAuth2 authentication',
    metadata: { category: 'auth', success: true }
  },
  {
    text: 'JWT token validation best practices',
    metadata: { category: 'auth', success: true }
  },
  // ... more vectors
]);
```

### 3. Provide Feedback

Tell GNN which results were useful:

```typescript
const results = await db.search({
  query: 'authentication methods',
  k: 10
});

// User found result helpful
await db.learn({
  query: 'authentication methods',
  results: results.slice(0, 3),  // Top 3 were useful
  feedback: 'success'
});

// User didn't find results helpful
await db.learn({
  query: 'different search',
  results: results,
  feedback: 'failure'
});
```

### 4. Enhanced Search

After accumulating feedback (minimum 10 samples):

```typescript
const enhanced = await db.search({
  query: 'authentication',
  k: 10,
  useGNN: true  // Enable GNN enhancement
});

// GNN has modified the query embedding based on:
// - Nearby vectors in the graph
// - Past successful searches
// - Learned attention weights
```

---

## Configuration

### GNN Parameters

```typescript
interface GNNConfig {
  /** Input vector dimension (must match embeddings) */
  inputDim: number;

  /** Output vector dimension (usually same as input) */
  outputDim: number;

  /** Number of attention heads (default: 4) */
  heads?: number;

  /** Learning rate for training (default: 0.001) */
  learningRate?: number;

  /** Hidden layer dimension (default: 512) */
  hiddenDim?: number;

  /** Dropout rate (default: 0.1) */
  dropout?: number;

  /** Number of GNN layers (default: 2) */
  layers?: number;
}
```

### Recommended Settings

#### Small Models (Fast, Less Memory)

```typescript
gnn: {
  inputDim: 384,
  outputDim: 384,
  heads: 2,              // Fewer heads = faster
  hiddenDim: 256,        // Smaller hidden layer
  learningRate: 0.005,   // Higher LR for faster convergence
  layers: 1
}
```

**Use case**: Quick experiments, limited compute

#### Medium Models (Balanced)

```typescript
gnn: {
  inputDim: 384,
  outputDim: 384,
  heads: 4,
  hiddenDim: 512,
  learningRate: 0.001,
  layers: 2
}
```

**Use case**: Production, general-purpose (recommended)

#### Large Models (High Quality)

```typescript
gnn: {
  inputDim: 768,         // Larger embeddings
  outputDim: 768,
  heads: 8,              // More attention heads
  hiddenDim: 1024,
  learningRate: 0.0005,  // Lower LR for stability
  layers: 3,
  dropout: 0.2           // Prevent overfitting
}
```

**Use case**: Maximum quality, ample compute/memory

---

## Training Workflow

### Accumulate Training Samples

```typescript
// Collect feedback over time
for (const interaction of userInteractions) {
  const results = await db.search({
    query: interaction.query,
    k: 10
  });

  await db.learn({
    query: interaction.query,
    results: interaction.clickedResults,
    feedback: interaction.wasHelpful ? 'success' : 'failure'
  });
}
```

### Batch Training

Train after accumulating samples:

```typescript
// Train GNN (minimum 10 samples required)
const trainResult = await db.trainGNN({
  epochs: 100,           // Training iterations
  batchSize: 32,         // Samples per batch
  validationSplit: 0.2   // Hold out 20% for validation
});

console.log(trainResult);
// {
//   loss: 0.023,
//   accuracy: 0.94,
//   epochs: 100,
//   samplesUsed: 256
// }
```

### Incremental Training

Continue training without losing progress:

```typescript
// Initial training
await db.trainGNN({ epochs: 50 });

// ... collect more feedback ...

// Continue training (doesn't reset model)
await db.trainGNN({
  epochs: 50,
  continueTraining: true  // Resume from current weights
});
```

---

## Advanced Features

### 1. Custom Neighbor Selection

Control which neighbors influence query enhancement:

```typescript
const enhanced = await db.search({
  query: 'authentication',
  k: 10,
  useGNN: true,
  gnnNeighbors: 20,      // Use 20 nearest neighbors for context
  gnnThreshold: 0.7      // Only use neighbors with >0.7 similarity
});
```

### 2. Multi-Head Attention Visualization

Inspect which neighbors the GNN is attending to:

```typescript
const results = await db.search({
  query: 'authentication',
  k: 10,
  useGNN: true,
  returnAttention: true  // Include attention weights
});

for (const result of results) {
  console.log(`${result.text}`);
  console.log(`  Attention weights: ${result.attention}`);
  // Shows which neighbors influenced this result
}
```

### 3. Transfer Learning

Transfer GNN model between databases:

```typescript
// Save trained model
await db.saveGNNModel('./models/auth-gnn.pth');

// Load into new database
const newDB = await createVectorDB({
  path: './new-db.db',
  backend: 'ruvector',
  enableGNN: true,
  gnn: { inputDim: 384, outputDim: 384 }
});

await newDB.loadGNNModel('./models/auth-gnn.pth');
```

### 4. A/B Testing

Compare GNN vs standard search:

```typescript
// Standard search
const standard = await db.search({
  query: 'authentication',
  k: 10,
  useGNN: false
});

// GNN-enhanced search
const enhanced = await db.search({
  query: 'authentication',
  k: 10,
  useGNN: true
});

// Compare results
const improvement = compareResults(standard, enhanced);
```

---

## Performance Characteristics

### Training Performance

| Samples | Epochs | Training Time | Memory |
|---------|--------|---------------|--------|
| 10 | 50 | ~2s | +5 MB |
| 100 | 50 | ~8s | +10 MB |
| 1000 | 100 | ~45s | +25 MB |
| 10000 | 100 | ~7min | +50 MB |

### Query Enhancement Overhead

| Neighbors | Enhancement Time | Total Query Time |
|-----------|------------------|------------------|
| 5 | ~1ms | ~84µs + 1ms |
| 10 | ~2ms | ~84µs + 2ms |
| 20 | ~3ms | ~84µs + 3ms |
| 50 | ~7ms | ~84µs + 7ms |

**Recommendation**: Use 10-20 neighbors for balanced performance

### Memory Usage

- **Base GNN Model**: 5-10 MB (4 heads, 512 hidden)
- **Training Samples**: ~1 KB per sample
- **Attention Cache**: ~50 MB (for 100K vectors)

---

## Best Practices

### 1. Start Simple

```typescript
// Begin with minimal config
gnn: {
  inputDim: 384,
  outputDim: 384,
  heads: 4
}
```

### 2. Collect Diverse Feedback

```typescript
// ✅ Good: Mix of success and failure
await db.learn({ query: q1, results: r1, feedback: 'success' });
await db.learn({ query: q2, results: r2, feedback: 'failure' });
await db.learn({ query: q3, results: r3, feedback: 'success' });

// ❌ Bad: Only success cases (model won't learn what to avoid)
await db.learn({ query: q1, results: r1, feedback: 'success' });
await db.learn({ query: q2, results: r2, feedback: 'success' });
```

### 3. Wait for Sufficient Data

```typescript
const stats = await db.getGNNStats();

if (stats.sampleCount >= 10) {
  // Enough data to train
  await db.trainGNN({ epochs: 50 });
} else {
  console.log(`Need ${10 - stats.sampleCount} more samples`);
}
```

### 4. Monitor Training Progress

```typescript
const result = await db.trainGNN({
  epochs: 100,
  onEpoch: (epoch, loss) => {
    if (epoch % 10 === 0) {
      console.log(`Epoch ${epoch}: Loss = ${loss}`);
    }
  }
});
```

### 5. Evaluate Improvements

```typescript
// Compare before/after
const beforeTraining = await db.search({ query: 'test', k: 10, useGNN: false });
await db.trainGNN({ epochs: 100 });
const afterTraining = await db.search({ query: 'test', k: 10, useGNN: true });

// Measure improvement (e.g., precision@k, recall@k)
```

---

## Troubleshooting

### GNN Not Available

**Error**:
```
Error: GNN learning requires @ruvector/gnn
```

**Solution**:
```bash
npm install @ruvector/gnn
```

### Not Enough Training Samples

**Error**:
```
Error: Minimum 10 samples required for training (found: 5)
```

**Solution**: Collect more feedback before training:
```typescript
const stats = await db.getGNNStats();
console.log(`Current samples: ${stats.sampleCount}`);
```

### Training Loss Not Decreasing

**Issue**: Loss stays high or increases

**Solutions**:
1. **Lower learning rate**:
   ```typescript
   gnn: { learningRate: 0.0001 }  // Try 10x lower
   ```

2. **More training epochs**:
   ```typescript
   await db.trainGNN({ epochs: 500 });  // More iterations
   ```

3. **Check data quality**:
   ```typescript
   // Ensure feedback is meaningful
   await db.learn({
     query: 'specific query',
     results: actuallyRelevantResults,  // Not random results
     feedback: 'success'
   });
   ```

### High Memory Usage

**Issue**: GNN using too much memory

**Solutions**:
1. **Reduce model size**:
   ```typescript
   gnn: {
     heads: 2,         // Fewer heads
     hiddenDim: 256    // Smaller hidden layer
   }
   ```

2. **Limit training samples**:
   ```typescript
   await db.trainGNN({
     maxSamples: 1000  // Use only recent samples
   });
   ```

---

## Examples

### Example 1: Document Search with Learning

```typescript
import { createVectorDB } from 'agentdb';

// Initialize with GNN
const db = await createVectorDB({
  path: './docs.db',
  backend: 'ruvector',
  enableGNN: true,
  gnn: { inputDim: 384, outputDim: 384 }
});

// Insert documents
await db.insertBatch([
  { text: 'OAuth2 implementation guide', metadata: { type: 'guide' } },
  { text: 'JWT authentication tutorial', metadata: { type: 'tutorial' } },
  // ... more docs
]);

// User searches
const results = await db.search({ query: 'authentication', k: 10 });

// User clicks on result #2 (indicates relevance)
await db.learn({
  query: 'authentication',
  results: [results[1]],  // The clicked result
  feedback: 'success'
});

// After 10+ interactions, train GNN
setTimeout(async () => {
  const stats = await db.getGNNStats();

  if (stats.sampleCount >= 10) {
    await db.trainGNN({ epochs: 50 });
    console.log('GNN trained! Search quality improved.');
  }
}, 60000);  // Check after 1 minute
```

### Example 2: Personalized Search

```typescript
// Track user-specific feedback
const userFeedback = new Map();

async function personalizedSearch(userId, query) {
  // Search with user's learned preferences
  const results = await db.search({
    query,
    k: 10,
    useGNN: true,
    gnnContext: {
      userId,  // Use user-specific GNN model
      // GNN learns per-user preferences
    }
  });

  return results;
}

// User provides feedback
async function recordFeedback(userId, query, results, helpful) {
  await db.learn({
    query,
    results,
    feedback: helpful ? 'success' : 'failure',
    context: { userId }  // Associate with user
  });

  // Train user-specific model
  if (await db.getGNNStats({ userId }).sampleCount >= 10) {
    await db.trainGNN({
      userId,  // Train this user's model
      epochs: 50
    });
  }
}
```

---

## API Reference

### `enableGNN: boolean`

Enable GNN learning for the database.

### `gnn: GNNConfig`

GNN configuration (required if `enableGNN: true`).

### `db.learn(options)`

Add training sample from user feedback.

**Parameters**:
- `query: string` - Original query
- `results: SearchResult[]` - Results that were relevant/irrelevant
- `feedback: 'success' | 'failure'` - Whether results were helpful

### `db.trainGNN(options)`

Train GNN model on accumulated samples.

**Parameters**:
- `epochs?: number` - Training iterations (default: 100)
- `batchSize?: number` - Samples per batch (default: 32)
- `validationSplit?: number` - Validation set size (default: 0.2)

**Returns**: Training result with loss, accuracy, etc.

### `db.getGNNStats()`

Get GNN statistics.

**Returns**:
```typescript
{
  sampleCount: number;      // Training samples collected
  modelSize: number;        // Model memory usage (bytes)
  lastTraining: number;     // Timestamp of last training
  accuracy?: number;        // Validation accuracy
}
```

### `db.saveGNNModel(path)`

Save trained GNN model to disk.

### `db.loadGNNModel(path)`

Load pre-trained GNN model.

---

## Summary

GNN learning in AgentDB v2 enables:
- ✅ **Adaptive search** that learns from user feedback
- ✅ **Context-aware retrieval** using graph structure
- ✅ **Continuous improvement** over time
- ✅ **Personalized results** per user or use case

**Minimum requirements**:
- @ruvector/core + @ruvector/gnn installed
- 10+ training samples
- Node.js environment

**Performance impact**:
- +1-3ms query enhancement overhead
- +5-50 MB memory for GNN model
- Training: ~2s-7min depending on sample count

---

**Next**: [Troubleshooting Guide](./TROUBLESHOOTING.md) | [Backend Configuration](./BACKENDS.md)
