# ADR-001: Backend Abstraction Layer Design

**Status:** Implemented
**Date:** 2025-11-28
**Author:** System Architect (AgentDB v2)

## Context

AgentDB v2 requires a unified abstraction layer for vector operations to support multiple backends (RuVector, HNSWLib) while maintaining backward compatibility and enabling advanced features like GNN learning and graph databases.

## Decision

### Core Design Principles

1. **String-based IDs**: All backends use string IDs for maximum flexibility
2. **Normalized Similarity**: All backends return similarity scores in 0-1 range (higher = more similar)
3. **Metadata Support**: First-class support for attaching metadata to vectors
4. **Save/Load with Metadata**: Persist both index structure and metadata mappings
5. **Backend-specific Optimizations**: Hide implementation details behind interface

### Interface Hierarchy

```typescript
VectorBackend (core interface)
├── RuVectorBackend (preferred, native Rust)
└── HNSWLibBackend (fallback, Node.js)

LearningBackend (optional GNN features)
└── RuVectorLearning (@ruvector/gnn)

GraphBackend (optional graph database)
└── RuVectorGraph (@ruvector/graph-node)
```

### VectorBackend Interface

```typescript
export interface VectorBackend {
  readonly name: 'ruvector' | 'hnswlib';

  // Core operations
  insert(id: string, embedding: Float32Array, metadata?: Record<string, any>): void;
  insertBatch(items: Array<{id, embedding, metadata?}>): void;
  search(query: Float32Array, k: number, options?: SearchOptions): SearchResult[];
  remove(id: string): boolean;

  // Index management
  save(path: string): Promise<void>;
  load(path: string): Promise<void>;

  // Stats and lifecycle
  getStats(): VectorStats;
  close(): void;
}
```

**Key Design Decisions:**

1. **Synchronous `insert()`**: Optimized for batch operations, async not needed
2. **Async `save()/load()`**: File I/O operations should be async
3. **SearchOptions**: Extensible options object for threshold, filters, efSearch override
4. **Readonly `name`**: Backend type identification for debugging and metrics

### SearchResult Format

```typescript
export interface SearchResult {
  id: string;           // String ID (backends handle label mapping)
  distance: number;     // Raw distance from backend
  similarity: number;   // Normalized 0-1 (higher = more similar)
  metadata?: Record<string, any>;  // Optional attached metadata
}
```

**Similarity Normalization:**
- Cosine: `similarity = 1 - distance`
- L2: `similarity = exp(-distance)`
- IP: `similarity = -distance` (higher inner product = more similar)

### Backend Detection and Factory

```typescript
export type BackendType = 'auto' | 'ruvector' | 'hnswlib';

export interface BackendDetection {
  available: 'ruvector' | 'hnswlib' | 'none';
  ruvector: {
    core: boolean;
    gnn: boolean;
    graph: boolean;
    native: boolean;
  };
  hnswlib: boolean;
}

export async function detectBackends(): Promise<BackendDetection>;
export async function createBackend(type: BackendType, config: VectorConfig): Promise<VectorBackend>;
```

**Detection Priority:**
1. Check for `@ruvector/core` (preferred)
2. Check for `@ruvector/gnn` (optional learning)
3. Check for `@ruvector/graph-node` (optional graph)
4. Fallback to `hnswlib-node` if RuVector unavailable
5. Clear error messages with installation instructions

### Optional Features

#### LearningBackend (GNN)

```typescript
export interface LearningBackend {
  enhance(query: Float32Array, neighbors: Float32Array[], weights: number[]): Float32Array;
  addSample(sample: TrainingSample): void;
  train(options?: {epochs?: number}): Promise<TrainingResult>;
  saveModel(path: string): Promise<void>;
  loadModel(path: string): Promise<void>;
  getStats(): LearningStats;
}
```

**Use Case:** Adaptive query enhancement using GNN attention mechanisms

#### GraphBackend (Property Graph)

```typescript
export interface GraphBackend {
  execute(cypher: string, params?: Record<string, any>): Promise<QueryResult>;
  createNode(labels: string[], properties: Record<string, any>): Promise<string>;
  createRelationship(from: string, to: string, type: string, properties?): Promise<string>;
  traverse(startId: string, pattern: string, options?: TraversalOptions): Promise<GraphNode[]>;
  vectorSearch(query: Float32Array, k: number, contextNodeId?: string): Promise<GraphNode[]>;
}
```

**Use Case:** Causal memory graphs with vector search integration

## Consequences

### Positive

1. **Backend Flexibility**: Easy to add new backends (e.g., Faiss, Annoy)
2. **Graceful Degradation**: Falls back to HNSWLib if RuVector unavailable
3. **Feature Detection**: Auto-detects GNN and Graph capabilities
4. **Consistent API**: Same code works across all backends
5. **Performance Transparency**: Backend name exposed for monitoring
6. **Clear Migration Path**: Existing HNSWIndex code can gradually migrate

### Negative

1. **Interface Limitations**: Must support common subset of features
2. **Performance Variations**: Different backends have different characteristics
3. **Complexity**: Multiple implementations to maintain
4. **Testing Burden**: Must test all backend combinations

### Mitigation Strategies

1. **Feature Flags**: Optional features clearly marked in detection
2. **Performance Benchmarks**: Document performance characteristics per backend
3. **Integration Tests**: Test suite runs against all backends
4. **Migration Guide**: Document how to migrate from HNSWIndex

## Implementation Status

### Completed

- ✅ `VectorBackend.ts` - Core interface definition
- ✅ `LearningBackend.ts` - GNN interface
- ✅ `GraphBackend.ts` - Graph database interface
- ✅ `detector.ts` - Auto-detection logic
- ✅ `factory.ts` - Backend creation and initialization
- ✅ `index.ts` - Public exports
- ✅ `ruvector/RuVectorBackend.ts` - RuVector implementation
- ✅ `ruvector/RuVectorLearning.ts` - GNN implementation
- ✅ `hnswlib/HNSWLibBackend.ts` - HNSWLib adapter

### Pending

- ⏳ `ruvector/RuVectorGraph.ts` - Graph implementation (planned)
- ⏳ Integration with ReasoningBank controller
- ⏳ Integration with SkillLibrary controller
- ⏳ CLI commands (`agentdb init`, `agentdb benchmark`)
- ⏳ Migration guide from HNSWIndex
- ⏳ Performance benchmarks

## Configuration

### Default Configuration

```typescript
const defaultConfig: VectorConfig = {
  dimension: 384,
  metric: 'cosine',
  maxElements: 100000,
  M: 16,
  efConstruction: 200,
  efSearch: 100
};
```

### Backend-Specific Tuning

**RuVector:**
- Native bindings for optimal performance
- WASM fallback for unsupported platforms
- GNN learning: 4 heads, 0.001 learning rate
- Tiered compression for memory efficiency

**HNSWLib:**
- Stable C++ implementation
- No GNN or Graph support
- Higher efConstruction for better quality
- Manual index rebuilding after updates

## Testing Strategy

1. **Unit Tests**: Test each backend independently
2. **Integration Tests**: Test backend swapping
3. **Parity Tests**: Ensure consistent results across backends
4. **Performance Tests**: Benchmark search speed and memory usage
5. **Compatibility Tests**: Test save/load across backends

## Related Documents

- [ARCHITECTURE.md](/workspaces/agentic-flow/plans/agentdb-v2/ARCHITECTURE.md)
- [ROADMAP.md](/workspaces/agentic-flow/plans/agentdb-v2/ROADMAP.md)

## References

- RuVector: https://github.com/ruvnet/ruvector
- HNSWLib: https://github.com/nmslib/hnswlib
- HNSW Paper: https://arxiv.org/abs/1603.09320
- Graph Attention Networks: https://arxiv.org/abs/1710.10903
