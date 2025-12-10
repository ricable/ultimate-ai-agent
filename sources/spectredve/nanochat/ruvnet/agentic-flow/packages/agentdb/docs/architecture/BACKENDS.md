# AgentDB Backend Configuration Guide

Complete guide to backend selection, configuration, and optimization in AgentDB v2

---

## Overview

AgentDB v2 introduces **pluggable backends** for vector storage, allowing you to choose the optimal backend for your use case. This guide covers backend selection, configuration, and performance tuning.

---

## Backend Architecture

### Design Principles

1. **Unified Interface**: All backends implement the same `VectorBackend` interface
2. **Automatic Detection**: Zero-config backend selection based on installed packages
3. **Graceful Fallback**: Automatic fallback to available backends
4. **Optional Dependencies**: Install only what you need
5. **Performance First**: Backends optimized for different use cases

### Backend Hierarchy

```
┌─────────────────────────────────────────┐
│         VectorBackend Interface         │
│  (Unified API for all backends)         │
└─────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬──────────┐
        │           │           │          │
   ┌────▼────┐ ┌───▼────┐ ┌────▼─────┐ ┌─▼────┐
   │ SQLite  │ │ better │ │ RuVector │ │ GNN  │
   │ (v1)    │ │-sqlite3│ │   Core   │ │ Ext  │
   └─────────┘ └────────┘ └──────────┘ └──────┘
```

---

## Available Backends

### 1. SQLite (Default - v1 Compatible)

**Built-in SQL-based vector storage**

**Pros**:
- ✅ Zero dependencies
- ✅ Cross-platform (Node.js, browser via sql.js)
- ✅ Backward compatible with v1
- ✅ Mature and stable

**Cons**:
- ❌ Slower vector search (~12ms per query)
- ❌ Higher memory usage
- ❌ No advanced features (GNN, compression)

**Use Cases**:
- Browser applications
- Legacy v1 compatibility
- Small datasets (<1K vectors)
- No-dependency deployments

**Installation**: Built-in (no installation required)

**Configuration**:
```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'sqlite'
});
```

**Performance**:
- Search latency: 12.5ms (1000 vectors)
- Throughput: 80 queries/second
- Memory: 147 MB (100K vectors)

---

### 2. better-sqlite3 (Fast SQLite)

**High-performance SQLite with native bindings**

**Pros**:
- ✅ 2-3x faster than sql.js
- ✅ Lower CPU usage
- ✅ Same API as SQLite backend
- ✅ Production-ready

**Cons**:
- ❌ Node.js only (no browser)
- ❌ Native compilation required
- ❌ Still slower than RuVector

**Use Cases**:
- Node.js applications
- Moderate datasets (1K-10K vectors)
- Easy migration from SQLite
- CPU-constrained environments

**Installation**:
```bash
npm install better-sqlite3
```

**Configuration**:
```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'better-sqlite3'
});
```

**Performance**:
- Search latency: 4.2ms (1000 vectors)
- Throughput: 238 queries/second
- Memory: 147 MB (100K vectors)
- **Improvement**: 2.9x faster than SQLite

---

### 3. RuVector Core (High Performance)

**Native/WASM vector search with HNSW indexing**

**Pros**:
- ✅ 150x faster than SQLite
- ✅ 8.6x less memory usage
- ✅ Native + WASM support
- ✅ SIMD acceleration
- ✅ Compression support
- ✅ Sub-millisecond search

**Cons**:
- ❌ External dependency
- ❌ Platform-specific builds (for native)
- ❌ Requires @ruvector/core package

**Use Cases**:
- Production applications
- Large datasets (10K-1M+ vectors)
- Real-time search
- Memory-constrained environments
- High-throughput workloads

**Installation**:
```bash
# WASM (cross-platform)
npm install @ruvector/core

# Native (faster, platform-specific)
npm install @ruvector/core-linux-x64       # Linux x64
npm install @ruvector/core-darwin-arm64    # macOS ARM64
npm install @ruvector/core-win32-x64       # Windows x64
```

**Configuration**:
```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'ruvector',

  // HNSW tuning
  M: 16,                  // Connections per layer (quality)
  efConstruction: 200,    // Build quality (higher = better)
  efSearch: 100,          // Search quality (higher = more results)

  // Vector config
  dimension: 384,
  metric: 'cosine',       // 'cosine', 'l2', or 'ip'
  maxElements: 100000
});
```

**Performance**:
- Search latency: 83µs (native), 95µs (WASM)
- Throughput: 12,048 queries/second (native)
- Memory: 17 MB (100K vectors)
- **Improvement**: 150x faster, 8.6x less memory than SQLite

**HNSW Parameters**:

| Parameter | Range | Impact | Recommendation |
|-----------|-------|--------|----------------|
| `M` | 4-64 | Quality vs memory | 16 (balanced), 32 (high quality) |
| `efConstruction` | 50-500 | Build time vs quality | 200 (production), 500 (max quality) |
| `efSearch` | 10-1000 | Search time vs recall | 100 (default), 200 (high recall) |

---

### 4. RuVector GNN (Adaptive Learning)

**RuVector Core + Graph Neural Network query enhancement**

**Pros**:
- ✅ All RuVector Core benefits
- ✅ Adaptive query enhancement
- ✅ Learning from user feedback
- ✅ Neighbor-aware search
- ✅ Continuous improvement

**Cons**:
- ❌ Requires both @ruvector/core and @ruvector/gnn
- ❌ Training overhead
- ❌ More complex configuration

**Use Cases**:
- Adaptive AI agents
- Self-learning systems
- Personalized search
- Context-aware retrieval
- Continuous learning applications

**Installation**:
```bash
npm install @ruvector/core @ruvector/gnn
```

**Configuration**:
```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'ruvector',

  // Enable GNN
  enableGNN: true,
  gnn: {
    inputDim: 384,
    outputDim: 384,
    heads: 4,              // Multi-head attention
    learningRate: 0.001,
    hiddenDim: 512         // Optional
  }
});
```

**Training**:
```typescript
// Train from search feedback
await db.learn({
  query: queryEmbedding,
  results: searchResults,
  feedback: 'success'  // or 'failure'
});

// After 10+ samples, GNN will enhance queries
const enhanced = await db.search({
  query: 'new query',
  k: 10,
  useGNN: true  // Enable GNN enhancement
});
```

**Performance**:
- Same search speed as RuVector Core
- Query enhancement: ~2ms overhead
- Training: ~50ms per batch (10 samples)
- Memory: +5-10 MB for GNN model

---

## Backend Detection

### Automatic Detection

AgentDB automatically detects available backends on startup:

```typescript
import { detectBackends } from 'agentdb/backends';

const detection = await detectBackends();

console.log(detection);
// {
//   available: 'ruvector',  // Best available backend
//   ruvector: {
//     core: true,
//     gnn: true,
//     graph: false,
//     native: true
//   },
//   hnswlib: true,           // Fallback
//   betterSqlite3: true,
//   sqlite: true             // Always available
// }
```

### Detection Priority

1. **RuVector GNN** (@ruvector/core + @ruvector/gnn)
2. **RuVector Core** (@ruvector/core)
3. **better-sqlite3** (better-sqlite3)
4. **SQLite** (built-in)

### Format Detection Output

```typescript
import { formatDetectionResult } from 'agentdb/backends';

const result = await detectBackends();
console.log(formatDetectionResult(result));

// Output:
// 📊 Backend Detection Results:
//
//   Backend:     ruvector
//   Platform:    linux-x64
//   Native:      ✅
//   GNN:         ✅
//   Graph:       ❌
//   Compression: ✅
//   Version:     1.0.0
//
// 💡 Tip: All advanced features available!
```

---

## Backend Selection Strategies

### Strategy 1: Auto (Recommended)

Let AgentDB choose the best backend:

```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'auto'  // Auto-detect
});
```

**When to use**: Most applications (95% of cases)

### Strategy 2: Explicit Backend

Force a specific backend:

```typescript
// Require RuVector (error if not installed)
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'ruvector'
});

// Fallback if RuVector not available
try {
  const db = await createVectorDB({ backend: 'ruvector' });
} catch (error) {
  const db = await createVectorDB({ backend: 'sqlite' });
}
```

**When to use**: Specific backend required for testing or compliance

### Strategy 3: Conditional Selection

Choose backend based on environment:

```typescript
const backend = process.env.NODE_ENV === 'production'
  ? 'ruvector'
  : 'sqlite';

const db = await createVectorDB({
  path: './agent-memory.db',
  backend
});
```

**When to use**: Different backends for dev/staging/production

### Strategy 4: Programmatic Detection

Detect and select based on capabilities:

```typescript
import { detectBackends, getRecommendedBackend } from 'agentdb/backends';

const detection = await detectBackends();

let backend: string;
if (detection.ruvector.gnn) {
  backend = 'ruvector';
  console.log('Using RuVector with GNN learning');
} else if (detection.ruvector.core) {
  backend = 'ruvector';
  console.log('Using RuVector (GNN not available)');
} else {
  backend = 'sqlite';
  console.log('Using SQLite fallback');
}

const db = await createVectorDB({ path: './db.db', backend });
```

**When to use**: Feature-dependent backend selection

---

## Configuration Reference

### Common Configuration

All backends support these options:

```typescript
interface CommonConfig {
  /** Database path (required) */
  path: string;

  /** Backend type */
  backend: 'auto' | 'ruvector' | 'better-sqlite3' | 'sqlite';

  /** Vector dimension */
  dimension?: number;  // Default: 384

  /** Distance metric */
  metric?: 'cosine' | 'l2' | 'ip';  // Default: 'cosine'

  /** Maximum vectors */
  maxElements?: number;  // Default: 100000
}
```

### RuVector-Specific Configuration

```typescript
interface RuVectorConfig extends CommonConfig {
  backend: 'ruvector';

  /** HNSW M - connections per layer */
  M?: number;  // Default: 16, Range: 4-64

  /** HNSW efConstruction - build quality */
  efConstruction?: number;  // Default: 200, Range: 50-500

  /** HNSW efSearch - search quality */
  efSearch?: number;  // Default: 100, Range: 10-1000

  /** Enable compression */
  compression?: boolean;  // Default: false

  /** Enable GNN learning */
  enableGNN?: boolean;  // Default: false

  /** GNN configuration */
  gnn?: {
    inputDim: number;
    outputDim: number;
    heads?: number;         // Default: 4
    learningRate?: number;  // Default: 0.001
    hiddenDim?: number;     // Default: 512
  };
}
```

### SQLite/better-sqlite3 Configuration

```typescript
interface SQLiteConfig extends CommonConfig {
  backend: 'sqlite' | 'better-sqlite3';

  /** WAL mode for better concurrency */
  wal?: boolean;  // Default: true

  /** Cache size in KB */
  cacheSize?: number;  // Default: 2000

  /** Enable foreign keys */
  foreignKeys?: boolean;  // Default: true
}
```

---

## Performance Tuning

### RuVector HNSW Tuning

#### For Maximum Speed

```typescript
const db = await createVectorDB({
  backend: 'ruvector',
  M: 8,              // Fewer connections = faster
  efConstruction: 100,
  efSearch: 50       // Lower recall, faster search
});
```

**Trade-off**: 2x faster, 90-95% recall

#### For Maximum Quality

```typescript
const db = await createVectorDB({
  backend: 'ruvector',
  M: 32,             // More connections = better recall
  efConstruction: 500,
  efSearch: 200      // Higher recall, slower search
});
```

**Trade-off**: 2x slower, 98-99% recall

#### Balanced (Recommended)

```typescript
const db = await createVectorDB({
  backend: 'ruvector',
  M: 16,
  efConstruction: 200,
  efSearch: 100
});
```

**Trade-off**: Optimal speed/quality balance

### Runtime Search Tuning

Override `efSearch` per query:

```typescript
// High-quality search (slower)
const highQuality = await db.search({
  query: 'important query',
  k: 10,
  efSearch: 500  // Override global efSearch
});

// Fast search (lower quality)
const fast = await db.search({
  query: 'quick search',
  k: 10,
  efSearch: 50
});
```

### Memory Optimization

#### Enable Compression (RuVector)

```typescript
const db = await createVectorDB({
  backend: 'ruvector',
  compression: true  // 4-8x memory reduction
});
```

**Trade-off**: Slower decompression (~10% overhead)

#### Limit Maximum Elements

```typescript
const db = await createVectorDB({
  backend: 'ruvector',
  maxElements: 10000  // Prevent unbounded growth
});
```

### Batch Operations

Use batch insert for optimal throughput:

```typescript
// ❌ Slow: Individual inserts
for (const item of items) {
  await db.insert(item);
}

// ✅ Fast: Batch insert (141x faster)
await db.insertBatch(items);
```

---

## Platform-Specific Considerations

### Linux (x64/ARM64)

**Recommended**: RuVector native

```bash
# x64
npm install @ruvector/core-linux-x64

# ARM64
npm install @ruvector/core-linux-arm64
```

**Performance**: Best performance with native bindings

### macOS (Intel/Apple Silicon)

**Recommended**: RuVector native

```bash
# Intel (x64)
npm install @ruvector/core-darwin-x64

# Apple Silicon (ARM64)
npm install @ruvector/core-darwin-arm64
```

**Note**: Apple Silicon shows best SIMD performance

### Windows

**Recommended**: RuVector WASM (easier install)

```bash
# Native (if needed)
npm install @ruvector/core-win32-x64

# WASM (recommended)
npm install @ruvector/core
```

**Note**: WASM has fewer build issues on Windows

### Browser/Edge Functions

**Required**: SQLite (sql.js WASM)

```typescript
const db = await createVectorDB({
  path: ':memory:',  // In-memory for browser
  backend: 'sqlite'
});
```

**Note**: Only SQLite backend supports browsers

---

## Migration Between Backends

### Export/Import Tool

```bash
# Export from SQLite
npx agentdb export ./sqlite-db.db --output=vectors.json

# Import to RuVector
npx agentdb import ./ruvector-db.db --input=vectors.json --backend=ruvector
```

### Programmatic Migration

```typescript
import { createVectorDB } from 'agentdb';

// Open source database
const sourceDB = await createVectorDB({
  path: './old-db.db',
  backend: 'sqlite'
});

// Open destination database
const destDB = await createVectorDB({
  path: './new-db.db',
  backend: 'ruvector'
});

// Migrate all vectors
const allVectors = await sourceDB.search({
  query: 'dummy',  // Get all
  k: 999999
});

await destDB.insertBatch(allVectors);
await destDB.save();

console.log(`Migrated ${allVectors.length} vectors`);
```

---

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed troubleshooting guide.

### Common Issues

**Backend not detected**:
```bash
npx agentdb detect  # Check available backends
npm install @ruvector/core  # Install missing backend
```

**Native build failed**:
- RuVector automatically falls back to WASM
- WASM is still 130x faster than SQLite

**GNN training not working**:
- Requires minimum 10 training samples
- Check `@ruvector/gnn` is installed

**Performance not as expected**:
```bash
npx agentdb benchmark --backend=ruvector  # Run benchmarks
```

---

## Best Practices

1. **Use auto-detect in development**: Let AgentDB choose
2. **Lock backend in production**: Explicit backend for consistency
3. **Benchmark your data**: Real-world performance may vary
4. **Start with RuVector Core**: Add GNN only if needed
5. **Monitor memory**: Use `db.getStats()` to track usage
6. **Batch operations**: Use `insertBatch()` for large imports
7. **Tune HNSW parameters**: Balance speed vs quality for your use case

---

## Summary

| Use Case | Recommended Backend | Why |
|----------|---------------------|-----|
| Browser apps | SQLite | Only browser-compatible backend |
| Small datasets (<1K) | SQLite / better-sqlite3 | Simplicity, no overhead |
| Medium datasets (1K-10K) | better-sqlite3 | 2-3x faster, easy install |
| Large datasets (10K+) | RuVector Core | 150x faster, 8.6x less memory |
| Learning agents | RuVector GNN | Adaptive query enhancement |
| Production | RuVector native | Maximum performance |

**Default recommendation**: Start with `backend: 'auto'` and install `@ruvector/core` for production.

---

**Next**: [GNN Learning Guide](./GNN_LEARNING.md) | [Troubleshooting](./TROUBLESHOOTING.md)
