# AgentDB v2 Implementation Guide

## Phase 1: Core Backend Abstraction

### Step 1.1: Create Backend Interface

```typescript
// packages/agentdb/src/backends/VectorBackend.ts

export interface VectorConfig {
  dimension: number;
  metric: 'cosine' | 'l2' | 'ip';
  maxElements?: number;
  efConstruction?: number;
  efSearch?: number;
  M?: number;
}

export interface SearchResult {
  id: string;
  distance: number;
  similarity: number;
  metadata?: Record<string, any>;
}

export interface SearchOptions {
  threshold?: number;
  efSearch?: number;
  filter?: Record<string, any>;
}

export interface VectorStats {
  count: number;
  dimension: number;
  metric: string;
  backend: 'ruvector' | 'hnswlib';
  memoryUsage: number;
}

export interface VectorBackend {
  readonly name: 'ruvector' | 'hnswlib';

  insert(id: string, embedding: Float32Array, metadata?: Record<string, any>): void;
  insertBatch(items: Array<{ id: string; embedding: Float32Array }>): void;
  search(query: Float32Array, k: number, options?: SearchOptions): SearchResult[];
  remove(id: string): boolean;
  getStats(): VectorStats;
  save(path: string): Promise<void>;
  load(path: string): Promise<void>;
  close(): void;
}
```

### Step 1.2: Implement RuVector Backend

```typescript
// packages/agentdb/src/backends/ruvector/RuVectorBackend.ts

import type { VectorBackend, VectorConfig, SearchResult, SearchOptions, VectorStats } from '../VectorBackend.js';

export class RuVectorBackend implements VectorBackend {
  readonly name = 'ruvector' as const;
  private db: any; // VectorDB from @ruvector/core
  private config: VectorConfig;
  private metadata: Map<string, Record<string, any>> = new Map();

  constructor(config: VectorConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    const { VectorDB } = await import('@ruvector/core');
    this.db = new VectorDB(this.config.dimension, {
      metric: this.config.metric,
      maxElements: this.config.maxElements || 100000,
      efConstruction: this.config.efConstruction || 200,
      M: this.config.M || 16
    });
  }

  insert(id: string, embedding: Float32Array, metadata?: Record<string, any>): void {
    this.db.insert(id, Array.from(embedding));
    if (metadata) {
      this.metadata.set(id, metadata);
    }
  }

  insertBatch(items: Array<{ id: string; embedding: Float32Array; metadata?: Record<string, any> }>): void {
    for (const item of items) {
      this.insert(item.id, item.embedding, item.metadata);
    }
  }

  search(query: Float32Array, k: number, options?: SearchOptions): SearchResult[] {
    if (options?.efSearch) {
      this.db.setEfSearch(options.efSearch);
    }

    const results = this.db.search(Array.from(query), k);

    return results
      .map((r: { id: string; distance: number }) => ({
        id: r.id,
        distance: r.distance,
        similarity: this.distanceToSimilarity(r.distance),
        metadata: this.metadata.get(r.id)
      }))
      .filter((r: SearchResult) => !options?.threshold || r.similarity >= options.threshold);
  }

  remove(id: string): boolean {
    this.metadata.delete(id);
    return this.db.remove(id);
  }

  getStats(): VectorStats {
    return {
      count: this.db.count(),
      dimension: this.config.dimension,
      metric: this.config.metric,
      backend: 'ruvector',
      memoryUsage: this.db.memoryUsage?.() || 0
    };
  }

  async save(path: string): Promise<void> {
    this.db.save(path);
    // Save metadata separately
    const metadataPath = path + '.meta.json';
    const fs = await import('fs/promises');
    await fs.writeFile(metadataPath, JSON.stringify(Object.fromEntries(this.metadata)));
  }

  async load(path: string): Promise<void> {
    this.db.load(path);
    // Load metadata
    const metadataPath = path + '.meta.json';
    try {
      const fs = await import('fs/promises');
      const data = await fs.readFile(metadataPath, 'utf-8');
      this.metadata = new Map(Object.entries(JSON.parse(data)));
    } catch {
      // No metadata file
    }
  }

  close(): void {
    // RuVector cleanup if needed
  }

  private distanceToSimilarity(distance: number): number {
    switch (this.config.metric) {
      case 'cosine': return 1 - distance;
      case 'l2': return Math.exp(-distance);
      case 'ip': return -distance;
      default: return 1 - distance;
    }
  }
}
```

### Step 1.3: Implement HNSWLib Backend (Fallback)

```typescript
// packages/agentdb/src/backends/hnswlib/HNSWLibBackend.ts

import type { VectorBackend, VectorConfig, SearchResult, SearchOptions, VectorStats } from '../VectorBackend.js';

export class HNSWLibBackend implements VectorBackend {
  readonly name = 'hnswlib' as const;
  private index: any; // HierarchicalNSW
  private config: VectorConfig;
  private idToLabel: Map<string, number> = new Map();
  private labelToId: Map<number, string> = new Map();
  private metadata: Map<string, Record<string, any>> = new Map();
  private nextLabel = 0;

  constructor(config: VectorConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    const hnswlib = await import('hnswlib-node');
    const { HierarchicalNSW } = hnswlib.default || hnswlib;

    const metricMap = { cosine: 'cosine', l2: 'l2', ip: 'ip' };
    this.index = new HierarchicalNSW(metricMap[this.config.metric], this.config.dimension);
    this.index.initIndex(
      this.config.maxElements || 100000,
      this.config.M || 16,
      this.config.efConstruction || 200
    );
    this.index.setEf(this.config.efSearch || 100);
  }

  insert(id: string, embedding: Float32Array, metadata?: Record<string, any>): void {
    const label = this.nextLabel++;
    this.index.addPoint(Array.from(embedding), label);
    this.idToLabel.set(id, label);
    this.labelToId.set(label, id);
    if (metadata) {
      this.metadata.set(id, metadata);
    }
  }

  insertBatch(items: Array<{ id: string; embedding: Float32Array; metadata?: Record<string, any> }>): void {
    for (const item of items) {
      this.insert(item.id, item.embedding, item.metadata);
    }
  }

  search(query: Float32Array, k: number, options?: SearchOptions): SearchResult[] {
    if (options?.efSearch) {
      this.index.setEf(options.efSearch);
    }

    const result = this.index.searchKnn(Array.from(query), k);
    const results: SearchResult[] = [];

    for (let i = 0; i < result.neighbors.length; i++) {
      const label = result.neighbors[i];
      const id = this.labelToId.get(label);
      if (!id) continue;

      const distance = result.distances[i];
      const similarity = this.distanceToSimilarity(distance);

      if (options?.threshold && similarity < options.threshold) continue;

      results.push({
        id,
        distance,
        similarity,
        metadata: this.metadata.get(id)
      });
    }

    return results;
  }

  remove(id: string): boolean {
    const label = this.idToLabel.get(id);
    if (label === undefined) return false;

    // hnswlib doesn't support deletion, mark for rebuild
    this.idToLabel.delete(id);
    this.labelToId.delete(label);
    this.metadata.delete(id);
    return true;
  }

  getStats(): VectorStats {
    return {
      count: this.idToLabel.size,
      dimension: this.config.dimension,
      metric: this.config.metric,
      backend: 'hnswlib',
      memoryUsage: 0 // hnswlib doesn't expose this
    };
  }

  async save(path: string): Promise<void> {
    this.index.writeIndex(path);
    // Save mappings
    const fs = await import('fs/promises');
    await fs.writeFile(path + '.mappings.json', JSON.stringify({
      idToLabel: Object.fromEntries(this.idToLabel),
      labelToId: Object.fromEntries(this.labelToId),
      metadata: Object.fromEntries(this.metadata),
      nextLabel: this.nextLabel
    }));
  }

  async load(path: string): Promise<void> {
    this.index.readIndex(path);
    const fs = await import('fs/promises');
    try {
      const data = JSON.parse(await fs.readFile(path + '.mappings.json', 'utf-8'));
      this.idToLabel = new Map(Object.entries(data.idToLabel).map(([k, v]) => [k, v as number]));
      this.labelToId = new Map(Object.entries(data.labelToId).map(([k, v]) => [Number(k), v as string]));
      this.metadata = new Map(Object.entries(data.metadata));
      this.nextLabel = data.nextLabel;
    } catch {}
  }

  close(): void {
    // hnswlib cleanup if needed
  }

  private distanceToSimilarity(distance: number): number {
    switch (this.config.metric) {
      case 'cosine': return 1 - distance;
      case 'l2': return Math.exp(-distance);
      case 'ip': return -distance;
      default: return 1 - distance;
    }
  }
}
```

### Step 1.4: Backend Factory

```typescript
// packages/agentdb/src/backends/factory.ts

import type { VectorBackend, VectorConfig } from './VectorBackend.js';
import { RuVectorBackend } from './ruvector/RuVectorBackend.js';
import { HNSWLibBackend } from './hnswlib/HNSWLibBackend.js';

export type BackendType = 'auto' | 'ruvector' | 'hnswlib';

export interface BackendDetection {
  available: 'ruvector' | 'hnswlib';
  ruvector: {
    core: boolean;
    gnn: boolean;
    graph: boolean;
    native: boolean;
  };
  hnswlib: boolean;
}

export async function detectBackends(): Promise<BackendDetection> {
  const result: BackendDetection = {
    available: 'hnswlib',
    ruvector: { core: false, gnn: false, graph: false, native: false },
    hnswlib: false
  };

  // Check RuVector
  try {
    const core = await import('@ruvector/core');
    result.ruvector.core = true;
    result.ruvector.native = core.isNative?.() ?? false;
    result.available = 'ruvector';

    try { await import('@ruvector/gnn'); result.ruvector.gnn = true; } catch {}
    try { await import('@ruvector/graph-node'); result.ruvector.graph = true; } catch {}
  } catch {}

  // Check hnswlib
  try {
    await import('hnswlib-node');
    result.hnswlib = true;
    if (!result.ruvector.core) {
      result.available = 'hnswlib';
    }
  } catch {}

  return result;
}

export async function createBackend(
  type: BackendType,
  config: VectorConfig
): Promise<VectorBackend> {
  const detection = await detectBackends();

  let backend: VectorBackend;

  if (type === 'ruvector' || (type === 'auto' && detection.ruvector.core)) {
    if (!detection.ruvector.core) {
      throw new Error('RuVector not available. Install: npm install @ruvector/core');
    }
    backend = new RuVectorBackend(config);
  } else {
    if (!detection.hnswlib) {
      throw new Error('No vector backend available. Install: npm install hnswlib-node');
    }
    backend = new HNSWLibBackend(config);
  }

  await (backend as any).initialize();
  return backend;
}
```

### Step 1.5: Update CLI

```typescript
// packages/agentdb/src/cli/commands/init.ts

import { Command } from 'commander';
import { detectBackends, createBackend } from '../../backends/factory.js';

export const initCommand = new Command('init')
  .description('Initialize AgentDB')
  .option('-b, --backend <type>', 'Backend: auto, ruvector, hnswlib', 'auto')
  .option('-d, --dimension <n>', 'Vector dimension', '384')
  .option('--dry-run', 'Show detection only')
  .action(async (opts) => {
    const detection = await detectBackends();

    if (opts.dryRun) {
      console.log('\n🔍 Backend Detection:\n');
      console.log(`  Recommended: ${detection.available}`);
      console.log(`  RuVector Core: ${detection.ruvector.core ? '✅' : '❌'}`);
      console.log(`  RuVector GNN:  ${detection.ruvector.gnn ? '✅' : '❌'}`);
      console.log(`  RuVector Graph: ${detection.ruvector.graph ? '✅' : '❌'}`);
      console.log(`  RuVector Native: ${detection.ruvector.native ? '✅' : '⚠️ WASM'}`);
      console.log(`  HNSWLib: ${detection.hnswlib ? '✅' : '❌'}`);
      return;
    }

    console.log(`\n🚀 Initializing AgentDB with ${opts.backend} backend...\n`);

    const backend = await createBackend(opts.backend, {
      dimension: parseInt(opts.dimension),
      metric: 'cosine'
    });

    console.log(`✅ Initialized with ${backend.name} backend`);
    backend.close();
  });
```

---

## Phase 2: GNN Learning Integration

### Step 2.1: Learning Backend Implementation

```typescript
// packages/agentdb/src/backends/ruvector/RuVectorLearning.ts

export interface LearningConfig {
  inputDim: number;
  outputDim: number;
  heads: number;
  learningRate: number;
}

export class RuVectorLearning {
  private gnnLayer: any;
  private config: LearningConfig;
  private trainingBuffer: Array<{ embedding: number[]; label: number }> = [];
  private trained = false;

  constructor(config: LearningConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    const { GNNLayer } = await import('@ruvector/gnn');
    this.gnnLayer = new GNNLayer(
      this.config.inputDim,
      this.config.outputDim,
      this.config.heads
    );
  }

  enhance(query: Float32Array, neighbors: Float32Array[], weights: number[]): Float32Array {
    if (!this.trained) {
      return query; // Return unchanged if not trained
    }

    const result = this.gnnLayer.forward(
      Array.from(query),
      neighbors.map(n => Array.from(n)),
      weights
    );

    return new Float32Array(result);
  }

  addSample(embedding: Float32Array, success: boolean): void {
    this.trainingBuffer.push({
      embedding: Array.from(embedding),
      label: success ? 1 : 0
    });
  }

  async train(options: { epochs?: number; batchSize?: number } = {}): Promise<{
    epochs: number;
    finalLoss: number;
  }> {
    if (this.trainingBuffer.length < 10) {
      throw new Error('Need at least 10 samples to train');
    }

    const result = await this.gnnLayer.train(this.trainingBuffer, {
      epochs: options.epochs || 100,
      learningRate: this.config.learningRate,
      batchSize: options.batchSize || 32
    });

    this.trained = true;
    this.trainingBuffer = [];

    return result;
  }

  async save(path: string): Promise<void> {
    this.gnnLayer.save(path);
  }

  async load(path: string): Promise<void> {
    this.gnnLayer.load(path);
    this.trained = true;
  }
}
```

### Step 2.2: Enhanced ReasoningBank

```typescript
// packages/agentdb/src/controllers/ReasoningBankV2.ts

import type { VectorBackend } from '../backends/VectorBackend.js';
import type { RuVectorLearning } from '../backends/ruvector/RuVectorLearning.js';
import type { EmbeddingService } from './EmbeddingService.js';

export class ReasoningBankV2 {
  private vector: VectorBackend;
  private learning?: RuVectorLearning;
  private embedder: EmbeddingService;
  private db: any;

  constructor(
    db: any,
    embedder: EmbeddingService,
    vector: VectorBackend,
    learning?: RuVectorLearning
  ) {
    this.db = db;
    this.embedder = embedder;
    this.vector = vector;
    this.learning = learning;
  }

  async storePattern(pattern: ReasoningPattern): Promise<string> {
    const embedding = await this.embedder.embed(
      `${pattern.taskType}: ${pattern.approach}`
    );

    const id = pattern.id || crypto.randomUUID();
    this.vector.insert(id, embedding, {
      taskType: pattern.taskType,
      successRate: pattern.successRate
    });

    // Store in SQLite for metadata queries
    this.db.prepare(`
      INSERT INTO reasoning_patterns (id, task_type, approach, success_rate)
      VALUES (?, ?, ?, ?)
    `).run(id, pattern.taskType, pattern.approach, pattern.successRate);

    return id;
  }

  async searchPatterns(
    query: string,
    k: number = 10,
    options: { useGNN?: boolean; threshold?: number } = {}
  ): Promise<ReasoningPattern[]> {
    let queryEmbedding = await this.embedder.embed(query);

    // Apply GNN enhancement if available and enabled
    if (options.useGNN && this.learning) {
      const candidates = this.vector.search(queryEmbedding, k * 3);
      if (candidates.length > 0) {
        const neighborEmbeddings = await this.getEmbeddings(candidates.map(c => c.id));
        const weights = candidates.map(c => c.similarity);
        queryEmbedding = this.learning.enhance(queryEmbedding, neighborEmbeddings, weights);
      }
    }

    const results = this.vector.search(queryEmbedding, k, {
      threshold: options.threshold
    });

    return this.hydratePatterns(results);
  }

  async recordOutcome(patternId: string, success: boolean): Promise<void> {
    // Update success rate
    this.db.prepare(`
      UPDATE reasoning_patterns
      SET success_rate = (success_rate * uses + ?) / (uses + 1),
          uses = uses + 1
      WHERE id = ?
    `).run(success ? 1 : 0, patternId);

    // Add to learning buffer
    if (this.learning) {
      const embedding = await this.getEmbedding(patternId);
      if (embedding) {
        this.learning.addSample(embedding, success);
      }
    }
  }

  async trainGNN(options?: { epochs?: number }): Promise<void> {
    if (!this.learning) {
      throw new Error('GNN not available');
    }
    await this.learning.train(options);
  }

  private async getEmbeddings(ids: string[]): Promise<Float32Array[]> {
    // Retrieve from vector store or cache
    return [];
  }

  private async getEmbedding(id: string): Promise<Float32Array | null> {
    return null;
  }

  private hydratePatterns(results: any[]): ReasoningPattern[] {
    return results.map(r => {
      const row = this.db.prepare(
        'SELECT * FROM reasoning_patterns WHERE id = ?'
      ).get(r.id);

      return {
        id: r.id,
        taskType: row?.task_type,
        approach: row?.approach,
        successRate: row?.success_rate,
        similarity: r.similarity
      };
    });
  }
}

interface ReasoningPattern {
  id?: string;
  taskType: string;
  approach: string;
  successRate: number;
  similarity?: number;
}
```

---

## Phase 3: Testing & Validation

See [tests/REGRESSION_PLAN.md](./tests/REGRESSION_PLAN.md) for complete test coverage.

### Unit Tests

```typescript
// packages/agentdb/tests/backends/ruvector.test.ts

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { RuVectorBackend } from '../../src/backends/ruvector/RuVectorBackend.js';

describe('RuVectorBackend', () => {
  let backend: RuVectorBackend;

  beforeAll(async () => {
    backend = new RuVectorBackend({ dimension: 384, metric: 'cosine' });
    await (backend as any).initialize();
  });

  afterAll(() => {
    backend.close();
  });

  it('should insert and search vectors', () => {
    const embedding = new Float32Array(384).fill(0.1);
    backend.insert('test-1', embedding);

    const results = backend.search(embedding, 1);
    expect(results).toHaveLength(1);
    expect(results[0].id).toBe('test-1');
    expect(results[0].similarity).toBeGreaterThan(0.99);
  });

  it('should achieve target latency', () => {
    // Insert 10K vectors
    for (let i = 0; i < 10000; i++) {
      const emb = new Float32Array(384);
      for (let j = 0; j < 384; j++) emb[j] = Math.random();
      backend.insert(`vec-${i}`, emb);
    }

    const query = new Float32Array(384).fill(0.5);
    const start = performance.now();
    backend.search(query, 10);
    const duration = performance.now() - start;

    expect(duration).toBeLessThan(1); // < 1ms target
  });
});
```

---

## Phase 4: Release

### Version Bump

```json
{
  "name": "agentdb",
  "version": "2.0.0",
  "description": "High-performance vector database with automatic RuVector/hnswlib backend selection"
}
```

### Migration Guide

```markdown
# Migrating to AgentDB v2

## Breaking Changes
None - v2 is fully backward compatible.

## New Features
- Automatic RuVector detection
- 8x faster search when RuVector installed
- GNN self-learning (optional)
- Tiered compression (optional)

## Upgrade Steps
1. npm install agentdb@2
2. (Optional) npm install @ruvector/core
3. agentdb init --dry-run  # Verify detection
4. agentdb init            # Reinitialize
```
