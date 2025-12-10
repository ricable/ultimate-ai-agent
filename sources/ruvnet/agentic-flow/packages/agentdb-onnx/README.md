# AgentDB-ONNX

> **High-Performance AI Agent Memory with ONNX Embeddings**

100% local, GPU-accelerated embeddings with AgentDB vector storage for intelligent AI agents.

[![Tests](https://img.shields.io/badge/tests-passing-green)]()
[![Performance](https://img.shields.io/badge/performance-optimized-brightgreen)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## 🚀 Features

### **100% Local Inference**
- No API calls, no cloud dependencies
- Complete data privacy
- Zero latency overhead from network requests
- Free unlimited embeddings

### **GPU Acceleration**
- ONNX Runtime with CUDA support (Linux/Windows)
- DirectML support (Windows)
- CoreML support (macOS)
- Automatic fallback to CPU

### **Performance Optimized**
- ⚡ **Batch processing**: 3-4x faster than sequential
- 💾 **LRU caching**: 80%+ hit rate for common queries
- 🔥 **Model warmup**: Pre-JIT compilation for consistent latency
- 📊 **Smart batching**: Automatic chunking for large datasets

### **Enterprise Features**
- **ReasoningBank**: Store and retrieve successful patterns
- **Reflexion Memory**: Self-improving episodic memory
- **Comprehensive metrics**: Latency, throughput, cache performance
- **Full TypeScript support**

---

## 📦 Installation

```bash
npm install agentdb-onnx
```

**Optional GPU acceleration:**

```bash
# CUDA (NVIDIA GPUs on Linux/Windows)
npm install onnxruntime-node-gpu

# DirectML (Any GPU on Windows)
# Already included in onnxruntime-node on Windows

# CoreML (Apple Silicon on macOS)
# Automatic on macOS ARM64
```

---

## ⚡ Quick Start

### Basic Usage

```typescript
import { createONNXAgentDB } from 'agentdb-onnx';

// Initialize
const agentdb = await createONNXAgentDB({
  dbPath: './my-agent-memory.db',
  modelName: 'Xenova/all-MiniLM-L6-v2', // 384 dimensions
  useGPU: true,
  batchSize: 32,
  cacheSize: 10000
});

// Store a reasoning pattern
const patternId = await agentdb.reasoningBank.storePattern({
  taskType: 'debugging',
  approach: 'Start with logs, reproduce issue, binary search for root cause',
  successRate: 0.92,
  tags: ['systematic', 'efficient']
});

// Search for similar patterns
const patterns = await agentdb.reasoningBank.searchPatterns(
  'how to debug memory leaks',
  { k: 5, threshold: 0.7 }
);

patterns.forEach(p => {
  console.log(`${p.approach} (${(p.similarity * 100).toFixed(1)}% match)`);
});
```

### Reflexion Memory (Self-Improvement)

```typescript
// Store an episode with self-critique
const episodeId = await agentdb.reflexionMemory.storeEpisode({
  sessionId: 'debug-session-1',
  task: 'Fix authentication bug',
  reward: 0.95,
  success: true,
  input: 'Users cannot log in',
  output: 'Fixed JWT token validation',
  critique: 'Should have checked token expiration first. Worked well.',
  latencyMs: 1200,
  tokensUsed: 450
});

// Learn from past experiences
const similar = await agentdb.reflexionMemory.retrieveRelevant(
  'authentication issues',
  { k: 5, onlySuccesses: true, minReward: 0.8 }
);

// Get critique summary
const critiques = await agentdb.reflexionMemory.getCritiqueSummary(
  'authentication debugging',
  10
);
```

### Batch Operations (3-4x Faster)

```typescript
// Store multiple patterns efficiently
const patterns = [
  { taskType: 'debugging', approach: 'Approach 1', successRate: 0.9 },
  { taskType: 'testing', approach: 'Approach 2', successRate: 0.85 },
  { taskType: 'optimization', approach: 'Approach 3', successRate: 0.92 }
];

const ids = await agentdb.reasoningBank.storePatternsBatch(patterns);
// 3-4x faster than storing individually
```

---

## 🎯 Available Models

| Model | Dimensions | Speed | Quality | Use Case |
|-------|------------|-------|---------|----------|
| `Xenova/all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ | **Recommended** - Best balance |
| `Xenova/all-MiniLM-L12-v2` | 384 | ⚡⚡ | ⭐⭐⭐⭐ | Higher quality |
| `Xenova/bge-small-en-v1.5` | 384 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Better accuracy |
| `Xenova/bge-base-en-v1.5` | 768 | ⚡⚡ | ⭐⭐⭐⭐⭐ | Highest quality |
| `Xenova/e5-small-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ | E5 series |
| `Xenova/e5-base-v2` | 768 | ⚡⚡ | ⭐⭐⭐⭐ | E5 series |

---

## 📊 Performance

### Benchmarks (M1 Pro CPU)

| Operation | Throughput | Latency (p50) | Latency (p95) |
|-----------|------------|---------------|---------------|
| Single embedding | 45 ops/sec | 22ms | 45ms |
| Cached embedding | 5000+ ops/sec | <1ms | 2ms |
| Batch (10 items) | 120 ops/sec | 83ms | 150ms |
| Pattern storage | 85 ops/sec | 12ms | 28ms |
| Pattern search | 110 ops/sec | 9ms | 22ms |
| Episode storage | 90 ops/sec | 11ms | 25ms |

**Cache performance:**
- Hit rate: 80-95% for common queries
- Speedup: 100-200x for cached access

---

## 🛠️ CLI Usage

```bash
# Initialize database
npx agentdb-onnx init ./my-memory.db --model Xenova/all-MiniLM-L6-v2 --gpu

# Store pattern
npx agentdb-onnx store-pattern ./my-memory.db \
  --task-type debugging \
  --approach "Check logs first" \
  --success-rate 0.92 \
  --tags "systematic,efficient"

# Search patterns
npx agentdb-onnx search-patterns ./my-memory.db "how to debug" \
  --top-k 5 \
  --threshold 0.7

# Store episode
npx agentdb-onnx store-episode ./my-memory.db \
  --session debug-1 \
  --task "Fix bug" \
  --reward 0.95 \
  --success \
  --critique "Worked well"

# Search episodes
npx agentdb-onnx search-episodes ./my-memory.db "debugging" \
  --top-k 5 \
  --only-successes

# Get statistics
npx agentdb-onnx stats ./my-memory.db

# Run benchmarks
npx agentdb-onnx benchmark
```

---

## 🧪 Testing

```bash
# Run tests
npm test

# Run specific test file
npm test onnx-embedding.test.ts

# Run with coverage
npm test -- --coverage
```

---

## 📈 Optimization Tips

### 1. **Enable GPU Acceleration**

```typescript
const agentdb = await createONNXAgentDB({
  dbPath: './db.db',
  useGPU: true  // Requires onnxruntime-node-gpu
});
```

### 2. **Increase Batch Size**

```typescript
const agentdb = await createONNXAgentDB({
  dbPath: './db.db',
  batchSize: 64  // Higher for GPU, lower for CPU
});
```

### 3. **Warm Up Model**

```typescript
await agentdb.embedder.warmup(10);  // Pre-JIT compile
```

### 4. **Increase Cache Size**

```typescript
const agentdb = await createONNXAgentDB({
  dbPath: './db.db',
  cacheSize: 50000  // More memory, better hit rate
});
```

### 5. **Use Batch Operations**

```typescript
// ✅ Good - batch insert
await agentdb.reasoningBank.storePatternsBatch(patterns);

// ❌ Slow - sequential inserts
for (const p of patterns) {
  await agentdb.reasoningBank.storePattern(p);
}
```

---

## 🎓 Examples

### Complete Workflow

```bash
npm run example
```

See [`examples/complete-workflow.ts`](examples/complete-workflow.ts) for a comprehensive demo including:
- Pattern storage and retrieval
- Episode-based learning
- Batch operations
- Performance optimization
- Real-world agent simulation

### Key Patterns

**1. Learn from Experience:**
```typescript
// Store successful approach
await agentdb.reflexionMemory.storeEpisode({
  task: 'Optimize API',
  reward: 0.95,
  success: true,
  critique: 'Database indexes were key'
});

// Later: retrieve when facing similar task
const experiences = await agentdb.reflexionMemory.retrieveRelevant(
  'slow API performance',
  { onlySuccesses: true }
);
```

**2. Build Knowledge Base:**
```typescript
// Accumulate successful patterns
await agentdb.reasoningBank.storePatternsBatch([
  { taskType: 'debugging', approach: 'Binary search', successRate: 0.92 },
  { taskType: 'testing', approach: 'TDD', successRate: 0.88 }
]);

// Query when needed
const approaches = await agentdb.reasoningBank.searchPatterns(
  'how to debug production issues',
  { k: 3 }
);
```

---

## 📚 API Reference

### `createONNXAgentDB(config)`

Creates an optimized AgentDB instance with ONNX embeddings.

**Config:**
- `dbPath: string` - Database file path
- `modelName?: string` - HuggingFace model (default: `Xenova/all-MiniLM-L6-v2`)
- `useGPU?: boolean` - Enable GPU (default: `true`)
- `batchSize?: number` - Batch size (default: `32`)
- `cacheSize?: number` - Cache size (default: `10000`)

**Returns:**
- `db` - Database instance
- `embedder` - ONNX embedding service
- `reasoningBank` - Pattern storage controller
- `reflexionMemory` - Episodic memory controller
- `close()` - Cleanup function
- `getStats()` - Performance statistics

### `ONNXEmbeddingService`

High-performance embedding generation.

**Methods:**
- `embed(text)` - Generate single embedding
- `embedBatch(texts)` - Generate multiple embeddings
- `warmup(samples)` - Pre-warm the model
- `getStats()` - Get performance metrics
- `clearCache()` - Clear embedding cache
- `getDimension()` - Get embedding dimension

### `ONNXReasoningBank`

Store and retrieve successful reasoning patterns.

**Methods:**
- `storePattern(pattern)` - Store pattern
- `storePatternsBatch(patterns)` - Batch store (3-4x faster)
- `searchPatterns(query, options)` - Semantic search
- `getPattern(id)` - Get by ID
- `updatePattern(id, updates)` - Update pattern
- `deletePattern(id)` - Delete pattern
- `getStats()` - Get statistics

### `ONNXReflexionMemory`

Episodic memory with self-critique for continuous improvement.

**Methods:**
- `storeEpisode(episode)` - Store episode
- `storeEpisodesBatch(episodes)` - Batch store (3-4x faster)
- `retrieveRelevant(task, options)` - Search similar episodes
- `getCritiqueSummary(task, k)` - Get critique summary
- `getEpisode(id)` - Get by ID
- `deleteEpisode(id)` - Delete episode
- `getTaskStats(sessionId?)` - Get statistics

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## 📄 License

MIT

---

## 🙏 Acknowledgments

- **AgentDB** - Vector database for AI agents
- **ONNX Runtime** - High-performance inference
- **Transformers.js** - Browser-compatible ML models
- **Xenova** - Optimized HuggingFace model conversions

---

## 🔗 Links

- [AgentDB](https://github.com/ruvnet/agentic-flow/tree/main/packages/agentdb)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Transformers.js](https://huggingface.co/docs/transformers.js)
- [HuggingFace Models](https://huggingface.co/models)

---

**Built with ❤️ for the agentic era**
