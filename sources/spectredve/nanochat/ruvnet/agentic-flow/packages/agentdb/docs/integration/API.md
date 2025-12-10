# AgentDB Attention Mechanisms API Reference

Complete API documentation for AgentDB's attention mechanisms, including RUV integration for ultra-fast WASM-accelerated vector operations.

## Table of Contents

- [AttentionService](#attentionservice)
- [Hyperbolic Memory](#hyperbolic-memory)
- [Flash Consolidation](#flash-consolidation)
- [Graph-RoPE Recall](#graph-rope-recall)
- [MoE Routing](#moe-routing)
- [Configuration](#configuration)

---

## AttentionService

The main service for managing all attention mechanisms.

### Constructor

```typescript
new AttentionService(
  db: Database.Database,
  config?: AttentionConfig
)
```

**Parameters:**
- `db`: Better-sqlite3 database instance
- `config`: Optional configuration object

**Example:**
```typescript
import Database from 'better-sqlite3';
import { AttentionService } from '@agentic/agentdb';

const db = new Database(':memory:');
const attention = new AttentionService(db, {
  enableHyperbolic: true,
  enableFlash: true,
  enableGraphRoPE: true,
  enableMoE: true,
  vectorDimension: 1536
});
```

### Methods

#### `enableFeatures(features: Partial<AttentionConfig>): void`

Enable or disable attention features at runtime.

```typescript
attention.enableFeatures({
  enableHyperbolic: true,
  enableFlash: true,
  flashWindowSize: 512
});
```

#### `getStatus(): AttentionStatus`

Get current status of all attention mechanisms.

```typescript
const status = attention.getStatus();
console.log(status);
// {
//   hyperbolic: { enabled: true, ready: true },
//   flash: { enabled: true, ready: true },
//   graphRoPE: { enabled: true, ready: true },
//   moe: { enabled: true, ready: true }
// }
```

#### `shutdown(): void`

Gracefully shutdown all attention mechanisms and cleanup resources.

```typescript
attention.shutdown();
```

---

## Hyperbolic Memory

Hierarchical memory organization using hyperbolic geometry for efficient recall.

### Class: `HyperbolicMemory`

```typescript
class HyperbolicMemory {
  constructor(db: Database.Database, config: HyperbolicConfig)

  storeWithHierarchy(
    vector: Float32Array,
    metadata: Record<string, any>,
    depth: number
  ): Promise<number>

  hierarchicalSearch(
    query: Float32Array,
    k: number,
    maxDepth?: number
  ): Promise<HyperbolicResult[]>

  updateHierarchy(id: number, newDepth: number): Promise<void>

  getHierarchyStats(): HierarchyStats
}
```

### `storeWithHierarchy()`

Store a vector with hierarchical metadata.

**Signature:**
```typescript
storeWithHierarchy(
  vector: Float32Array,
  metadata: Record<string, any>,
  depth: number
): Promise<number>
```

**Parameters:**
- `vector`: Embedding vector (Float32Array)
- `metadata`: Associated metadata object
- `depth`: Hierarchy depth (0 = root, higher = more specific)

**Returns:** Row ID of stored vector

**Example:**
```typescript
// Store high-level concept (root level)
const rootId = await hyperbolic.storeWithHierarchy(
  new Float32Array([...embeddings]),
  { type: 'category', name: 'Machine Learning' },
  0
);

// Store specific concept (child level)
const childId = await hyperbolic.storeWithHierarchy(
  new Float32Array([...embeddings]),
  { type: 'algorithm', name: 'Neural Networks', parent: rootId },
  1
);
```

### `hierarchicalSearch()`

Search with hierarchical awareness.

**Signature:**
```typescript
hierarchicalSearch(
  query: Float32Array,
  k: number,
  maxDepth?: number
): Promise<HyperbolicResult[]>
```

**Parameters:**
- `query`: Query vector
- `k`: Number of results to return
- `maxDepth`: Maximum hierarchy depth to search (optional)

**Returns:** Array of results with hyperbolic scores

**Example:**
```typescript
const results = await hyperbolic.hierarchicalSearch(
  queryVector,
  10,
  2 // Search up to depth 2
);

results.forEach(result => {
  console.log(`Depth ${result.depth}: ${result.metadata.name}`);
  console.log(`Hyperbolic score: ${result.hyperbolicScore}`);
});
```

### `getHierarchyStats()`

Get statistics about the memory hierarchy.

**Returns:**
```typescript
interface HierarchyStats {
  totalNodes: number;
  depthDistribution: Map<number, number>;
  avgDepth: number;
  maxDepth: number;
}
```

**Example:**
```typescript
const stats = hyperbolic.getHierarchyStats();
console.log(`Total nodes: ${stats.totalNodes}`);
console.log(`Average depth: ${stats.avgDepth}`);
console.log(`Max depth: ${stats.maxDepth}`);
```

---

## Flash Consolidation

Fast memory consolidation using sliding window attention.

### Class: `FlashConsolidation`

```typescript
class FlashConsolidation {
  constructor(db: Database.Database, config: FlashConfig)

  consolidateMemories(
    vectors: Float32Array[],
    windowSize?: number
  ): Promise<ConsolidatedMemory>

  queryConsolidated(
    query: Float32Array,
    k: number
  ): Promise<FlashResult[]>

  getConsolidationStats(): ConsolidationStats
}
```

### `consolidateMemories()`

Consolidate multiple memories using Flash Attention.

**Signature:**
```typescript
consolidateMemories(
  vectors: Float32Array[],
  windowSize?: number
): Promise<ConsolidatedMemory>
```

**Parameters:**
- `vectors`: Array of vectors to consolidate
- `windowSize`: Override default window size (optional)

**Returns:** Consolidated memory with metadata

**Example:**
```typescript
// Consolidate recent memories
const recentMemories = await db.getRecentVectors(100);
const consolidated = await flash.consolidateMemories(
  recentMemories,
  128 // Use 128-token window
);

console.log(`Consolidated ${consolidated.sourceCount} memories`);
console.log(`Compression ratio: ${consolidated.compressionRatio}`);
```

### `queryConsolidated()`

Query consolidated memories efficiently.

**Signature:**
```typescript
queryConsolidated(
  query: Float32Array,
  k: number
): Promise<FlashResult[]>
```

**Example:**
```typescript
const results = await flash.queryConsolidated(queryVector, 5);
results.forEach(result => {
  console.log(`Score: ${result.flashScore}`);
  console.log(`Window: ${result.windowInfo.start}-${result.windowInfo.end}`);
});
```

### Performance Characteristics

- **Complexity:** O(N) for consolidation, O(log N) for query
- **Memory:** Constant O(W) where W = window size
- **Speedup:** 2-5x faster than full attention for large memory sets

---

## Graph-RoPE Recall

Graph-enhanced rotary position encoding for contextual memory recall.

### Class: `GraphRoPERecall`

```typescript
class GraphRoPERecall {
  constructor(db: Database.Database, config: GraphRoPEConfig)

  buildMemoryGraph(
    vectors: Array<{ id: number; vector: Float32Array; metadata: any }>
  ): Promise<void>

  graphAwareSearch(
    query: Float32Array,
    k: number,
    hops?: number
  ): Promise<GraphResult[]>

  addEdge(sourceId: number, targetId: number, weight: number): Promise<void>

  getGraphStats(): GraphStats
}
```

### `buildMemoryGraph()`

Build a memory graph from vectors.

**Signature:**
```typescript
buildMemoryGraph(
  vectors: Array<{ id: number; vector: Float32Array; metadata: any }>
): Promise<void>
```

**Example:**
```typescript
const memories = await db.getAllMemories();
await graphRoPE.buildMemoryGraph(memories);

const stats = graphRoPE.getGraphStats();
console.log(`Graph built: ${stats.nodeCount} nodes, ${stats.edgeCount} edges`);
```

### `graphAwareSearch()`

Search with graph context awareness.

**Signature:**
```typescript
graphAwareSearch(
  query: Float32Array,
  k: number,
  hops?: number
): Promise<GraphResult[]>
```

**Parameters:**
- `query`: Query vector
- `k`: Number of results
- `hops`: Number of graph hops to explore (default: 2)

**Example:**
```typescript
// Find related memories through graph connections
const results = await graphRoPE.graphAwareSearch(
  queryVector,
  10,
  3 // Explore up to 3 hops
);

results.forEach(result => {
  console.log(`Memory: ${result.metadata.text}`);
  console.log(`RoPE score: ${result.ropeScore}`);
  console.log(`Graph path length: ${result.pathLength}`);
  console.log(`Connected memories: ${result.connectedIds.length}`);
});
```

### `addEdge()`

Manually add a relationship between memories.

**Signature:**
```typescript
addEdge(
  sourceId: number,
  targetId: number,
  weight: number
): Promise<void>
```

**Example:**
```typescript
// Create explicit relationship
await graphRoPE.addEdge(
  memory1.id,
  memory2.id,
  0.95 // High relationship strength
);
```

---

## MoE Routing

Mixture of Experts routing for specialized memory retrieval.

### Class: `MoERouting`

```typescript
class MoERouting {
  constructor(db: Database.Database, config: MoEConfig)

  routeQuery(
    query: Float32Array,
    k: number,
    expertCount?: number
  ): Promise<MoEResult[]>

  addExpert(
    name: string,
    specialization: string,
    vectors: Float32Array[]
  ): Promise<number>

  getExpertStats(): ExpertStats[]

  optimizeRouting(): Promise<RoutingOptimization>
}
```

### `routeQuery()`

Route a query to specialized experts.

**Signature:**
```typescript
routeQuery(
  query: Float32Array,
  k: number,
  expertCount?: number
): Promise<MoEResult[]>
```

**Parameters:**
- `query`: Query vector
- `k`: Number of results per expert
- `expertCount`: Number of experts to query (default: all)

**Returns:** Results from multiple experts with routing scores

**Example:**
```typescript
// Query routed to most relevant experts
const results = await moe.routeQuery(
  queryVector,
  5,  // 5 results per expert
  3   // Query top 3 experts
);

results.forEach(result => {
  console.log(`Expert: ${result.expertName}`);
  console.log(`Routing confidence: ${result.routingScore}`);
  console.log(`Result: ${result.metadata.text}`);
});
```

### `addExpert()`

Add a specialized expert with training data.

**Signature:**
```typescript
addExpert(
  name: string,
  specialization: string,
  vectors: Float32Array[]
): Promise<number>
```

**Example:**
```typescript
// Create expert for technical documentation
const expertId = await moe.addExpert(
  'Technical Docs Expert',
  'technical_documentation',
  technicalVectors
);

// Create expert for code snippets
await moe.addExpert(
  'Code Expert',
  'code_snippets',
  codeVectors
);
```

### `getExpertStats()`

Get performance statistics for all experts.

**Returns:**
```typescript
interface ExpertStats {
  expertId: number;
  expertName: string;
  specialization: string;
  queryCount: number;
  avgConfidence: number;
  memoryCount: number;
}
```

**Example:**
```typescript
const stats = moe.getExpertStats();
stats.forEach(expert => {
  console.log(`${expert.expertName}:`);
  console.log(`  Queries: ${expert.queryCount}`);
  console.log(`  Avg confidence: ${expert.avgConfidence}`);
  console.log(`  Memories: ${expert.memoryCount}`);
});
```

### `optimizeRouting()`

Optimize expert routing based on query patterns.

**Returns:**
```typescript
interface RoutingOptimization {
  rebalanced: number;
  merged: number;
  splitExperts: number;
  improvement: number;
}
```

**Example:**
```typescript
const optimization = await moe.optimizeRouting();
console.log(`Optimization results:`);
console.log(`  Rebalanced: ${optimization.rebalanced} experts`);
console.log(`  Performance improvement: ${optimization.improvement}%`);
```

---

## Configuration

### AttentionConfig

```typescript
interface AttentionConfig {
  enableHyperbolic?: boolean;      // Enable hyperbolic memory
  enableFlash?: boolean;            // Enable Flash consolidation
  enableGraphRoPE?: boolean;        // Enable Graph-RoPE recall
  enableMoE?: boolean;              // Enable MoE routing

  vectorDimension?: number;         // Vector dimension (default: 1536)

  // Hyperbolic settings
  hyperbolicCurvature?: number;     // Curvature parameter (default: -1.0)
  maxHierarchyDepth?: number;       // Max hierarchy depth (default: 5)

  // Flash settings
  flashWindowSize?: number;         // Window size (default: 256)
  flashHeadCount?: number;          // Number of attention heads (default: 8)

  // Graph-RoPE settings
  ropeTheta?: number;               // RoPE theta parameter (default: 10000)
  graphDensity?: number;            // Target graph density (default: 0.1)
  maxGraphHops?: number;            // Max graph traversal hops (default: 3)

  // MoE settings
  moeExpertCount?: number;          // Number of experts (default: 8)
  moeTopK?: number;                 // Top-K experts to activate (default: 2)
  moeLoadBalance?: boolean;         // Enable load balancing (default: true)
}
```

### Default Configuration

```typescript
const DEFAULT_CONFIG: AttentionConfig = {
  enableHyperbolic: true,
  enableFlash: true,
  enableGraphRoPE: true,
  enableMoE: true,

  vectorDimension: 1536,

  hyperbolicCurvature: -1.0,
  maxHierarchyDepth: 5,

  flashWindowSize: 256,
  flashHeadCount: 8,

  ropeTheta: 10000,
  graphDensity: 0.1,
  maxGraphHops: 3,

  moeExpertCount: 8,
  moeTopK: 2,
  moeLoadBalance: true
};
```

---

## Error Handling

All async methods throw typed errors:

```typescript
try {
  const results = await attention.hyperbolic.hierarchicalSearch(query, 10);
} catch (error) {
  if (error.code === 'HYPERBOLIC_NOT_ENABLED') {
    console.error('Hyperbolic memory is not enabled');
  } else if (error.code === 'INVALID_VECTOR_DIMENSION') {
    console.error('Vector dimension mismatch');
  } else {
    console.error('Unknown error:', error);
  }
}
```

### Error Codes

- `HYPERBOLIC_NOT_ENABLED`: Hyperbolic memory is disabled
- `FLASH_NOT_ENABLED`: Flash consolidation is disabled
- `GRAPHROPE_NOT_ENABLED`: Graph-RoPE is disabled
- `MOE_NOT_ENABLED`: MoE routing is disabled
- `INVALID_VECTOR_DIMENSION`: Vector dimension mismatch
- `DATABASE_ERROR`: Underlying database error
- `WASM_NOT_INITIALIZED`: RUV WASM runtime not initialized

---

## Performance Tips

1. **Batch Operations**: Use bulk methods when possible
   ```typescript
   // Good: Batch consolidation
   await flash.consolidateMemories(vectors);

   // Bad: Individual consolidation
   for (const vector of vectors) {
     await flash.consolidateMemories([vector]);
   }
   ```

2. **Configure Window Sizes**: Tune for your use case
   ```typescript
   // Smaller windows = faster, less context
   // Larger windows = slower, more context
   attention.enableFeatures({
     flashWindowSize: 128  // Faster for real-time
   });
   ```

3. **Use Feature Flags**: Enable only needed features
   ```typescript
   // Minimal configuration for speed
   const attention = new AttentionService(db, {
     enableFlash: true,
     enableHyperbolic: false,
     enableGraphRoPE: false,
     enableMoE: false
   });
   ```

4. **Optimize Expert Count**: More experts = better specialization but slower routing
   ```typescript
   attention.enableFeatures({
     moeExpertCount: 4,  // Fewer experts for speed
     moeTopK: 1          // Single expert activation
   });
   ```

---

## TypeScript Types

All types are exported from the main package:

```typescript
import type {
  AttentionConfig,
  AttentionStatus,
  HyperbolicResult,
  FlashResult,
  GraphResult,
  MoEResult,
  HierarchyStats,
  ConsolidationStats,
  GraphStats,
  ExpertStats
} from '@agentic/agentdb';
```

---

## Browser Usage

All features work in the browser with RUV WASM:

```html
<script type="module">
  import { AttentionService } from 'https://cdn.jsdelivr.net/npm/@agentic/agentdb/dist/browser.js';

  const attention = new AttentionService(db);
  // All APIs work identically in browser
</script>
```

---

## See Also

- [Getting Started Tutorial](tutorials/01-getting-started.md)
- [Migration Guide](MIGRATION.md)
- [FAQ](FAQ.md)
- [Examples](/packages/agentdb/examples/attention/)
