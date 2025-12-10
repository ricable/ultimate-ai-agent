# AgentDB v2 Migration Guide

Upgrading from AgentDB v1.x to v2.0.0-alpha

---

## Overview

AgentDB v2 introduces **multi-backend architecture** with automatic backend detection, providing 8-150x performance improvements through pluggable vector backends. This guide will help you upgrade from v1.x to v2.0.0-alpha.

### What's New in v2

- **Multi-Backend Support**: Choose between SQLite (v1 compatible), better-sqlite3, RuVector Core, and RuVector GNN
- **Automatic Backend Detection**: Zero-config backend selection based on available packages
- **150x Faster Vector Search**: RuVector backends with native/WASM acceleration
- **8.6x Memory Reduction**: Optimized storage with optional quantization
- **Backward Compatible**: Existing v1 databases work without migration
- **GNN Learning** (Optional): Graph Neural Network query enhancement with @ruvector/gnn

---

## Quick Migration Checklist

- [ ] Review [Breaking Changes](#breaking-changes)
- [ ] Update package version to `2.0.0-alpha.1`
- [ ] Choose backend strategy (see [Backend Selection](#backend-selection))
- [ ] Install optional backend packages (optional)
- [ ] Test database initialization
- [ ] Run performance benchmarks
- [ ] Update configuration if using advanced features

---

## Breaking Changes

### 1. Backend Architecture

**v1.x (SQLite only)**:
```typescript
import { createVectorDB } from 'agentdb';

const db = await createVectorDB({ path: './agent-memory.db' });
```

**v2.0 (Multi-backend)**:
```typescript
import { createVectorDB } from 'agentdb';

// Auto-detect best available backend
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'auto'  // NEW: backend selection
});
```

### 2. Backend Detection

v2 automatically detects available backends on startup:

```bash
[AgentDB] Backend Detection Results:
  Backend:     ruvector
  Platform:    linux-x64
  Native:      ✅
  GNN:         ✅
  Graph:       ❌
  Compression: ✅
  Version:     1.0.0
```

### 3. Optional Dependencies

v2 uses **optional dependencies** for backends. Install only what you need:

```json
{
  "dependencies": {
    "agentdb": "^2.0.0-alpha.1"
  },
  "optionalDependencies": {
    "better-sqlite3": "^11.8.1",      // 2-3x faster SQLite
    "@ruvector/core": "^1.0.0",       // 150x faster vector search
    "@ruvector/gnn": "^1.0.0"         // GNN learning (requires core)
  }
}
```

### 4. API Compatibility

**All v1.x APIs remain backward compatible** in v2. No code changes required for basic usage.

---

## Backend Selection

### Available Backends

| Backend | Speed | Memory | Learning | Installation |
|---------|-------|--------|----------|--------------|
| **SQLite** (v1 default) | 1x | 1x | ❌ | Built-in |
| **better-sqlite3** | 2-3x | 1x | ❌ | `npm install better-sqlite3` |
| **RuVector Core** | 150x | 8.6x less | ❌ | `npm install @ruvector/core` |
| **RuVector GNN** | 150x | 8.6x less | ✅ | `npm install @ruvector/core @ruvector/gnn` |

### Backend Selection Strategies

#### 1. Auto-Detect (Recommended)

Let AgentDB choose the best available backend:

```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'auto'  // Tries: RuVector → better-sqlite3 → SQLite
});
```

**Priority**: RuVector GNN > RuVector Core > better-sqlite3 > SQLite

#### 2. Explicit Backend

Force a specific backend:

```typescript
// Require RuVector (throws if not available)
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'ruvector'
});

// Require better-sqlite3
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'better-sqlite3'
});

// Force SQLite (v1 compatibility)
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'sqlite'
});
```

#### 3. Check Available Backends

Programmatically detect backends:

```typescript
import { detectBackends } from 'agentdb/backends';

const detection = await detectBackends();

console.log(detection);
// {
//   available: 'ruvector',
//   ruvector: { core: true, gnn: true, graph: false, native: true },
//   hnswlib: true,
//   betterSqlite3: true,
//   sqlite: true
// }
```

---

## Installation Paths

### Path 1: Minimal (v1 Compatible)

**No changes required**. Use built-in SQLite backend:

```bash
npm install agentdb@2.0.0-alpha.1
```

Your existing code works unchanged.

### Path 2: Performance Boost (2-3x)

Add better-sqlite3 for moderate performance improvement:

```bash
npm install agentdb@2.0.0-alpha.1 better-sqlite3
```

Auto-detected and used automatically.

### Path 3: Maximum Performance (150x)

Add RuVector for dramatic performance gains:

```bash
npm install agentdb@2.0.0-alpha.1 @ruvector/core
```

**Note**: RuVector uses WASM by default. For native binaries, ensure platform-specific build:

```bash
# Linux x64
npm install @ruvector/core-linux-x64

# macOS ARM64
npm install @ruvector/core-darwin-arm64

# Windows x64
npm install @ruvector/core-win32-x64
```

### Path 4: Advanced Learning (150x + GNN)

Add GNN for query enhancement and adaptive learning:

```bash
npm install agentdb@2.0.0-alpha.1 @ruvector/core @ruvector/gnn
```

---

## Migration Steps

### Step 1: Update Package

```bash
npm install agentdb@2.0.0-alpha.1
```

### Step 2: Test Existing Code

Run your existing code **without changes**:

```typescript
import { createVectorDB } from 'agentdb';

// This still works exactly as before
const db = await createVectorDB({ path: './agent-memory.db' });

await db.insert({
  text: 'Hello world',
  metadata: { type: 'greeting' }
});

const results = await db.search({
  query: 'Hello',
  k: 5
});
```

### Step 3: Choose Performance Backend (Optional)

Add backend package and update config:

```bash
npm install @ruvector/core
```

```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'auto'  // Will auto-detect RuVector
});
```

### Step 4: Verify Backend

Check which backend is being used:

```typescript
const stats = await db.stats();
console.log(`Using backend: ${stats.backend}`);
// "Using backend: ruvector"
```

### Step 5: Benchmark Performance

Run built-in benchmarks:

```bash
npx agentdb benchmark --backend=auto
npx agentdb benchmark --backend=ruvector
npx agentdb benchmark --backend=sqlite
```

Compare results and choose optimal backend for your use case.

---

## Performance Comparison

### Search Performance (1000 vectors, k=10)

| Backend | Latency | Throughput | Speedup |
|---------|---------|------------|---------|
| SQLite | 12.5ms | 80 ops/s | 1x |
| better-sqlite3 | 4.2ms | 238 ops/s | 2.9x |
| RuVector WASM | 95µs | 10,526 ops/s | 131x |
| RuVector Native | 83µs | 12,048 ops/s | 150x |

### Memory Usage (100K vectors, 384 dimensions)

| Backend | Memory | Reduction |
|---------|--------|-----------|
| SQLite | 147 MB | 1x |
| better-sqlite3 | 147 MB | 1x |
| RuVector | 17 MB | 8.6x |

### Batch Insert (10K vectors)

| Backend | Duration | Throughput |
|---------|----------|------------|
| SQLite | 2.4s | 4,167 ops/s |
| better-sqlite3 | 0.85s | 11,765 ops/s |
| RuVector | 0.18s | 55,556 ops/s |

---

## Configuration Reference

### Basic Configuration

```typescript
interface VectorDBConfig {
  /** Database path (required) */
  path: string;

  /** Backend selection: 'auto' | 'ruvector' | 'better-sqlite3' | 'sqlite' */
  backend?: 'auto' | 'ruvector' | 'better-sqlite3' | 'sqlite';

  /** Embedding dimension (default: 384) */
  dimension?: number;

  /** Distance metric: 'cosine' | 'l2' | 'ip' (default: 'cosine') */
  metric?: 'cosine' | 'l2' | 'ip';

  /** Maximum vectors (default: 100000) */
  maxElements?: number;
}
```

### Advanced RuVector Configuration

```typescript
interface RuVectorConfig extends VectorDBConfig {
  backend: 'ruvector';

  /** HNSW M parameter - connections per layer (default: 16) */
  M?: number;

  /** HNSW efConstruction - build quality (default: 200) */
  efConstruction?: number;

  /** HNSW efSearch - search quality (default: 100) */
  efSearch?: number;

  /** Enable GNN learning (requires @ruvector/gnn) */
  enableGNN?: boolean;

  /** GNN configuration */
  gnn?: {
    inputDim: number;
    outputDim: number;
    heads?: number;
    learningRate?: number;
  };
}
```

### GNN Learning Configuration

```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'ruvector',
  enableGNN: true,
  gnn: {
    inputDim: 384,
    outputDim: 384,
    heads: 4,
    learningRate: 0.001
  }
});

// Train GNN from search feedback
await db.learn({
  query: queryEmbedding,
  results: searchResults,
  feedback: 'success'  // or 'failure'
});
```

---

## Troubleshooting

### Backend Not Detected

**Error**:
```
Error: No vector backend available.
Install one of:
  - npm install @ruvector/core (recommended)
  - npm install better-sqlite3
  - Built-in SQLite (fallback)
```

**Solution**: Install at least one backend package:
```bash
npm install @ruvector/core
```

### RuVector Native Build Failed

**Error**:
```
Error: RuVector native bindings not available for linux-arm64
```

**Solution**: RuVector will automatically fallback to WASM:
```
[AgentDB] Using RuVector backend (WASM)
```

WASM is still 130x faster than SQLite.

### GNN Package Missing

**Error**:
```
Error: GNN learning requires @ruvector/gnn
Install with: npm install @ruvector/gnn
```

**Solution**: Install GNN package:
```bash
npm install @ruvector/gnn
```

Or disable GNN:
```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'ruvector',
  enableGNN: false  // Disable GNN
});
```

### Database Migration

**Question**: Do I need to migrate my v1 database?

**Answer**: No. v2 is backward compatible. Your existing SQLite database works without changes. To use RuVector:

1. **Option A (Dual databases)**: Create new RuVector database alongside existing
2. **Option B (Export/Import)**: Export v1 data and import to v2 RuVector database

```bash
# Export from v1
npx agentdb export ./v1-db.db --output=vectors.json

# Import to v2 with RuVector
npx agentdb import ./v2-db.db --input=vectors.json --backend=ruvector
```

---

## CLI Changes

### v1.x Commands

```bash
agentdb init ./my-db.db
agentdb stats ./my-db.db
```

### v2.0 Commands (Backward Compatible + New)

```bash
# All v1 commands work unchanged
agentdb init ./my-db.db
agentdb stats ./my-db.db

# NEW: Backend detection
agentdb detect

# NEW: Backend-specific initialization
agentdb init ./my-db.db --backend=ruvector

# NEW: Performance benchmarks
agentdb benchmark --backend=auto

# NEW: Backend comparison
agentdb compare --backends=sqlite,ruvector
```

---

## MCP Tools (Claude Desktop)

### No Changes Required

All 29 MCP tools work unchanged in v2. Backend selection is automatic:

```json
{
  "mcpServers": {
    "agentdb": {
      "command": "npx",
      "args": ["agentdb@2.0.0-alpha.1", "mcp", "start"]
    }
  }
}
```

### Backend Detection in MCP

MCP server logs show detected backend:

```
[AgentDB MCP] Backend: ruvector (native)
[AgentDB MCP] Features: GNN ✅, Graph ❌, Compression ✅
```

---

## Best Practices

### 1. Start with Auto-Detect

Let v2 choose the best backend:

```typescript
const db = await createVectorDB({
  path: './agent-memory.db',
  backend: 'auto'
});
```

### 2. Benchmark Your Use Case

Run benchmarks with your data:

```bash
npx agentdb benchmark --vectors=10000 --dimension=384
```

### 3. Use RuVector for Production

For production workloads, install RuVector:

```bash
npm install @ruvector/core
```

150x faster with 8.6x less memory.

### 4. Enable GNN for Adaptive Systems

If building self-learning agents:

```bash
npm install @ruvector/core @ruvector/gnn
```

### 5. Test Backend Fallback

Test graceful degradation:

```typescript
try {
  const db = await createVectorDB({ backend: 'ruvector' });
} catch (error) {
  // Fallback to SQLite
  const db = await createVectorDB({ backend: 'sqlite' });
}
```

---

## Roadmap

### v2.0.0-alpha (Current)

- ✅ Multi-backend architecture
- ✅ RuVector Core integration
- ✅ RuVector GNN learning
- ✅ Automatic backend detection
- ✅ Backward compatibility

### v2.0.0-beta (Planned)

- Graph database integration (@ruvector/graph)
- Multi-database synchronization
- Distributed vector search
- Advanced quantization (4-bit, 8-bit)

### v2.0.0 (Stable)

- Production-ready RuVector backends
- Complete documentation
- Migration tools
- Enterprise features

---

## Getting Help

### Documentation

- [Main README](../README.md)
- [Backend Configuration](./BACKENDS.md)
- [GNN Learning Guide](./GNN_LEARNING.md)
- [Troubleshooting](./TROUBLESHOOTING.md)

### Community

- **Issues**: https://github.com/ruvnet/agentic-flow/issues
- **Discussions**: https://github.com/ruvnet/agentic-flow/discussions
- **RuVector**: https://github.com/ruvnet/ruvector

### Support

For migration assistance, file an issue with:
- Current version (v1.x)
- Target version (v2.0.0-alpha.1)
- Backend preference
- Use case description

---

## Summary

AgentDB v2 is **backward compatible** with v1.x. No code changes required.

**Recommended upgrade path**:

1. Install v2: `npm install agentdb@2.0.0-alpha.1`
2. Test existing code (should work unchanged)
3. Add RuVector: `npm install @ruvector/core`
4. Enable auto-detect: `backend: 'auto'`
5. Benchmark performance
6. Optional: Add GNN for learning

**Performance gains**: 8-150x faster, 8.6x less memory, zero migration.

Happy coding! 🚀
