# RuVector Integration Guide

## Overview

This guide provides detailed integration patterns for using RuVector within agentic-flow.

## Installation in Agentic-Flow

### Add Dependencies

```bash
# Add to package.json
npm install ruvector @ruvector/core @ruvector/gnn

# Optional: Graph database support
npm install @ruvector/graph-node

# Optional: Data synthesis for training
npm install @ruvector/agentic-synth
```

### Package.json Update

```json
{
  "dependencies": {
    "ruvector": "^0.1.24",
    "@ruvector/core": "^0.1.15",
    "@ruvector/gnn": "^0.1.15"
  }
}
```

## Integration Patterns

### Pattern 1: Replace HNSWIndex

Replace the current `hnswlib-node` based implementation:

**Current Implementation** (`packages/agentdb/src/controllers/HNSWIndex.ts`):
```typescript
import hnswlibNode from 'hnswlib-node';
const { HierarchicalNSW } = hnswlibNode;
```

**RuVector Implementation**:
```typescript
// packages/agentdb/src/controllers/RuVectorIndex.ts
import { VectorDB, VectorDBConfig } from '@ruvector/core';

export interface RuVectorConfig {
  dimension: number;
  metric: 'cosine' | 'l2' | 'ip';
  maxElements?: number;
  efConstruction?: number;
  efSearch?: number;
  M?: number;
}

export class RuVectorIndex {
  private db: VectorDB;
  private config: RuVectorConfig;

  constructor(config: RuVectorConfig) {
    this.config = config;
    this.db = new VectorDB(config.dimension, {
      metric: config.metric,
      maxElements: config.maxElements || 100000,
      efConstruction: config.efConstruction || 200,
      M: config.M || 16
    });
  }

  /**
   * Insert a vector with ID
   */
  insert(id: string, embedding: Float32Array): void {
    this.db.insert(id, Array.from(embedding));
  }

  /**
   * Batch insert multiple vectors
   */
  insertBatch(items: Array<{ id: string; embedding: Float32Array }>): void {
    for (const item of items) {
      this.insert(item.id, item.embedding);
    }
  }

  /**
   * Search for k-nearest neighbors
   */
  search(query: Float32Array, k: number, options?: {
    threshold?: number;
    efSearch?: number;
  }): Array<{ id: string; distance: number; similarity: number }> {
    if (options?.efSearch) {
      this.db.setEfSearch(options.efSearch);
    }

    const results = this.db.search(Array.from(query), k);

    return results
      .map(r => ({
        id: r.id,
        distance: r.distance,
        similarity: this.distanceToSimilarity(r.distance)
      }))
      .filter(r => !options?.threshold || r.similarity >= options.threshold);
  }

  /**
   * Remove a vector
   */
  remove(id: string): void {
    this.db.remove(id);
  }

  /**
   * Get statistics
   */
  getStats(): {
    count: number;
    dimension: number;
    metric: string;
  } {
    return {
      count: this.db.count(),
      dimension: this.config.dimension,
      metric: this.config.metric
    };
  }

  private distanceToSimilarity(distance: number): number {
    switch (this.config.metric) {
      case 'cosine':
        return 1 - distance;
      case 'l2':
        return Math.exp(-distance);
      case 'ip':
        return -distance;
      default:
        return 1 - distance;
    }
  }
}
```

### Pattern 2: Enhanced ReasoningBank with GNN

Add self-learning capabilities to pattern matching:

```typescript
// packages/agentdb/src/controllers/RuVectorReasoningBank.ts
import { VectorDB } from '@ruvector/core';
import { GNNLayer, GNNConfig } from '@ruvector/gnn';
import { EmbeddingService } from './EmbeddingService.js';

export interface ReasoningPattern {
  id: string;
  taskType: string;
  approach: string;
  successRate: number;
  embedding?: Float32Array;
  metadata?: Record<string, any>;
}

export class RuVectorReasoningBank {
  private vectorDB: VectorDB;
  private gnnLayer: GNNLayer;
  private embedder: EmbeddingService;
  private patterns: Map<string, ReasoningPattern>;

  constructor(
    dimension: number,
    embedder: EmbeddingService,
    gnnConfig?: GNNConfig
  ) {
    this.vectorDB = new VectorDB(dimension, { metric: 'cosine' });
    this.gnnLayer = new GNNLayer(
      dimension,
      gnnConfig?.outputDim || dimension,
      gnnConfig?.heads || 4
    );
    this.embedder = embedder;
    this.patterns = new Map();
  }

  /**
   * Store a reasoning pattern with embedding
   */
  async storePattern(pattern: ReasoningPattern): Promise<void> {
    // Generate embedding if not provided
    if (!pattern.embedding) {
      pattern.embedding = await this.embedder.embed(
        `${pattern.taskType}: ${pattern.approach}`
      );
    }

    // Store in vector DB
    this.vectorDB.insert(pattern.id, Array.from(pattern.embedding));
    this.patterns.set(pattern.id, pattern);
  }

  /**
   * Search patterns with GNN-enhanced ranking
   */
  async searchPatterns(
    query: string,
    k: number = 10,
    useGNN: boolean = true
  ): Promise<ReasoningPattern[]> {
    // Generate query embedding
    const queryEmbedding = await this.embedder.embed(query);

    // Get initial candidates from vector search
    const candidates = this.vectorDB.search(
      Array.from(queryEmbedding),
      k * 2 // Get more candidates for GNN re-ranking
    );

    if (!useGNN || candidates.length === 0) {
      return candidates.slice(0, k).map(c => this.patterns.get(c.id)!);
    }

    // Apply GNN for enhanced ranking
    const neighborEmbeddings = candidates.map(c => {
      const pattern = this.patterns.get(c.id)!;
      return Array.from(pattern.embedding!);
    });

    const weights = candidates.map(c => 1 - c.distance);

    // GNN forward pass for query enhancement
    const enhanced = this.gnnLayer.forward(
      Array.from(queryEmbedding),
      neighborEmbeddings,
      weights
    );

    // Re-rank with enhanced query
    const reranked = this.vectorDB.search(enhanced, k);

    return reranked.map(r => this.patterns.get(r.id)!);
  }

  /**
   * Update pattern with feedback for learning
   */
  async updatePattern(
    patternId: string,
    success: boolean,
    reward: number
  ): Promise<void> {
    const pattern = this.patterns.get(patternId);
    if (!pattern) return;

    // Update success rate with exponential moving average
    const alpha = 0.1;
    pattern.successRate = (1 - alpha) * pattern.successRate + alpha * (success ? 1 : 0);

    // The GNN learns from these interactions over time
    // Training happens in background/offline
  }

  /**
   * Train GNN on accumulated patterns
   */
  async trainGNN(epochs: number = 100): Promise<void> {
    const patterns = Array.from(this.patterns.values());

    // Prepare training data
    const trainingData = patterns.map(p => ({
      embedding: Array.from(p.embedding!),
      label: p.successRate
    }));

    // Train GNN layer
    await this.gnnLayer.train(trainingData, {
      epochs,
      learningRate: 0.001,
      batchSize: 32
    });
  }
}
```

### Pattern 3: Graph-Based Agent Memory

Use Cypher queries for complex agent relationships:

```typescript
// packages/agentdb/src/controllers/RuVectorAgentGraph.ts

export class RuVectorAgentGraph {
  private graphDB: any; // @ruvector/graph-node

  constructor(dbPath: string) {
    // Initialize graph database
    // Note: Requires @ruvector/graph-node package
  }

  /**
   * Create an agent node
   */
  async createAgent(agent: {
    name: string;
    type: string;
    capabilities: string[];
    metadata?: Record<string, any>;
  }): Promise<string> {
    const cypher = `
      CREATE (a:Agent {
        name: $name,
        type: $type,
        capabilities: $capabilities,
        metadata: $metadata,
        createdAt: timestamp()
      })
      RETURN a
    `;

    return this.execute(cypher, agent);
  }

  /**
   * Create collaboration relationship
   */
  async createCollaboration(
    agent1: string,
    agent2: string,
    context: string
  ): Promise<void> {
    const cypher = `
      MATCH (a:Agent {name: $agent1}), (b:Agent {name: $agent2})
      CREATE (a)-[:COLLABORATES_WITH {
        context: $context,
        timestamp: timestamp()
      }]->(b)
    `;

    await this.execute(cypher, { agent1, agent2, context });
  }

  /**
   * Find agents for task based on relationships
   */
  async findAgentsForTask(
    taskType: string,
    requiredCapabilities: string[]
  ): Promise<string[]> {
    const cypher = `
      MATCH (a:Agent)
      WHERE a.type = $taskType
        OR any(cap IN a.capabilities WHERE cap IN $requiredCapabilities)
      WITH a, size([cap IN a.capabilities WHERE cap IN $requiredCapabilities]) as matchScore
      ORDER BY matchScore DESC
      RETURN a.name
      LIMIT 10
    `;

    return this.execute(cypher, { taskType, requiredCapabilities });
  }

  /**
   * Find collaboration chains
   */
  async findCollaborationPath(
    startAgent: string,
    endAgent: string,
    maxDepth: number = 3
  ): Promise<string[]> {
    const cypher = `
      MATCH path = shortestPath(
        (start:Agent {name: $startAgent})-[:COLLABORATES_WITH*1..${maxDepth}]-(end:Agent {name: $endAgent})
      )
      RETURN [node IN nodes(path) | node.name] as agents
    `;

    return this.execute(cypher, { startAgent, endAgent });
  }

  /**
   * Get agent's collaboration network
   */
  async getAgentNetwork(agentName: string, depth: number = 2): Promise<{
    agents: string[];
    relationships: Array<{ from: string; to: string; context: string }>;
  }> {
    const cypher = `
      MATCH (a:Agent {name: $agentName})-[r:COLLABORATES_WITH*1..${depth}]-(related:Agent)
      WITH collect(DISTINCT related.name) as agents,
           collect(DISTINCT {from: startNode(r).name, to: endNode(r).name, context: r.context}) as rels
      RETURN agents, rels as relationships
    `;

    return this.execute(cypher, { agentName });
  }

  private async execute(cypher: string, params: Record<string, any>): Promise<any> {
    // Execute Cypher query against graph database
    // Implementation depends on @ruvector/graph-node
    throw new Error('Requires @ruvector/graph-node installation');
  }
}
```

### Pattern 4: Tiered Compression for Memory Efficiency

Leverage RuVector's automatic compression tiers:

```typescript
// packages/agentdb/src/controllers/RuVectorTieredStorage.ts
import { VectorDB, CompressionTier } from '@ruvector/core';

export interface TieredStorageConfig {
  dimension: number;
  hotThreshold: number;   // Access frequency for hot tier
  warmThreshold: number;  // Access frequency for warm tier
  coolThreshold: number;  // Access frequency for cool tier
}

export class RuVectorTieredStorage {
  private db: VectorDB;
  private config: TieredStorageConfig;
  private accessCounts: Map<string, number>;

  constructor(config: TieredStorageConfig) {
    this.config = config;
    this.db = new VectorDB(config.dimension, {
      metric: 'cosine',
      enableCompression: true,
      compressionTiers: {
        hot: { threshold: config.hotThreshold, precision: 'f32' },
        warm: { threshold: config.warmThreshold, precision: 'f16' },
        cool: { threshold: config.coolThreshold, precision: 'pq8' },
        cold: { threshold: 0, precision: 'binary' }
      }
    });
    this.accessCounts = new Map();
  }

  /**
   * Store vector with automatic tier assignment
   */
  store(id: string, embedding: Float32Array, accessHint?: number): void {
    this.db.insert(id, Array.from(embedding));
    this.accessCounts.set(id, accessHint || 0);
  }

  /**
   * Search with access tracking
   */
  search(query: Float32Array, k: number): Array<{ id: string; similarity: number }> {
    const results = this.db.search(Array.from(query), k);

    // Update access counts for retrieved items
    for (const result of results) {
      const count = this.accessCounts.get(result.id) || 0;
      this.accessCounts.set(result.id, count + 1);
    }

    return results.map(r => ({
      id: r.id,
      similarity: 1 - r.distance
    }));
  }

  /**
   * Get storage statistics by tier
   */
  getStorageStats(): {
    hot: { count: number; sizeBytes: number };
    warm: { count: number; sizeBytes: number };
    cool: { count: number; sizeBytes: number };
    cold: { count: number; sizeBytes: number };
    totalCompression: string;
  } {
    return this.db.getCompressionStats();
  }

  /**
   * Force tier migration (manual optimization)
   */
  async optimizeTiers(): Promise<void> {
    await this.db.optimizeCompression();
  }
}
```

## Integration with Existing Components

### Update agentdb/index.ts

```typescript
// packages/agentdb/src/index.ts

// Existing exports
export { CausalMemoryGraph } from './controllers/CausalMemoryGraph.js';
export { ReasoningBank } from './controllers/ReasoningBank.js';
export { HNSWIndex } from './controllers/HNSWIndex.js';

// New RuVector exports
export { RuVectorIndex } from './controllers/RuVectorIndex.js';
export { RuVectorReasoningBank } from './controllers/RuVectorReasoningBank.js';
export { RuVectorAgentGraph } from './controllers/RuVectorAgentGraph.js';
export { RuVectorTieredStorage } from './controllers/RuVectorTieredStorage.js';
```

### Feature Flag for Migration

```typescript
// packages/agentdb/src/config.ts
export const AgentDBConfig = {
  // Enable RuVector as backend (set to true to use RuVector)
  USE_RUVECTOR: process.env.AGENTDB_USE_RUVECTOR === 'true',

  // RuVector-specific settings
  RUVECTOR_GNN_ENABLED: process.env.RUVECTOR_GNN_ENABLED === 'true',
  RUVECTOR_GRAPH_ENABLED: process.env.RUVECTOR_GRAPH_ENABLED === 'true',
  RUVECTOR_COMPRESSION_ENABLED: process.env.RUVECTOR_COMPRESSION === 'true',
};
```

### Factory Pattern for Backend Selection

```typescript
// packages/agentdb/src/factory.ts
import { AgentDBConfig } from './config.js';
import { HNSWIndex } from './controllers/HNSWIndex.js';
import { RuVectorIndex } from './controllers/RuVectorIndex.js';

export function createVectorIndex(config: {
  dimension: number;
  metric: 'cosine' | 'l2' | 'ip';
  maxElements?: number;
}) {
  if (AgentDBConfig.USE_RUVECTOR) {
    return new RuVectorIndex(config);
  }
  return new HNSWIndex(null, config);
}
```

## Testing the Integration

```typescript
// packages/agentdb/tests/ruvector-integration.test.ts
import { describe, it, expect, beforeAll } from 'vitest';
import { RuVectorIndex } from '../src/controllers/RuVectorIndex.js';
import { RuVectorReasoningBank } from '../src/controllers/RuVectorReasoningBank.js';

describe('RuVector Integration', () => {
  let vectorIndex: RuVectorIndex;

  beforeAll(() => {
    vectorIndex = new RuVectorIndex({
      dimension: 384,
      metric: 'cosine',
      maxElements: 10000
    });
  });

  it('should insert and search vectors', () => {
    const embedding = new Float32Array(384).fill(0.1);
    vectorIndex.insert('test-1', embedding);

    const results = vectorIndex.search(embedding, 1);
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('test-1');
    expect(results[0].similarity).toBeGreaterThan(0.99);
  });

  it('should achieve sub-millisecond search latency', async () => {
    // Insert 10k vectors
    for (let i = 0; i < 10000; i++) {
      const embedding = new Float32Array(384);
      for (let j = 0; j < 384; j++) {
        embedding[j] = Math.random();
      }
      vectorIndex.insert(`vec-${i}`, embedding);
    }

    // Measure search time
    const query = new Float32Array(384).fill(0.5);
    const start = performance.now();
    const results = vectorIndex.search(query, 10);
    const duration = performance.now() - start;

    expect(duration).toBeLessThan(1); // < 1ms
    expect(results).toHaveLength(10);
  });
});
```

## Migration Path

1. **Phase 1**: Add RuVector as optional dependency
2. **Phase 2**: Implement adapter classes
3. **Phase 3**: Enable feature flag for early adopters
4. **Phase 4**: Migrate default backend to RuVector
5. **Phase 5**: Deprecate hnswlib-node dependency

## Environment Variables

```bash
# Enable RuVector backend
export AGENTDB_USE_RUVECTOR=true

# Enable GNN self-learning
export RUVECTOR_GNN_ENABLED=true

# Enable graph database
export RUVECTOR_GRAPH_ENABLED=true

# Enable tiered compression
export RUVECTOR_COMPRESSION=true
```
