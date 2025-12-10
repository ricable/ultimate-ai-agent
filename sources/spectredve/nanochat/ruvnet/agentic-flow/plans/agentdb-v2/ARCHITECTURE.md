# AgentDB v2 Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AgentDB v2 Public API                          │
├─────────────────────────────────────────────────────────────────────────┤
│  createDatabase()  │  createVectorIndex()  │  createReasoningBank()     │
└────────────────────┴───────────────────────┴────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Backend Abstraction Layer                       │
├─────────────────────────────────────────────────────────────────────────┤
│  VectorBackend    │  StorageBackend   │  LearningBackend (optional)     │
│  interface        │  interface        │  interface                       │
└────────────────────┴───────────────────┴────────────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│      RuVector Backend        │          │      HNSWLib Backend         │
│         (Default)            │          │        (Fallback)            │
├──────────────────────────────┤          ├──────────────────────────────┤
│  @ruvector/core              │          │  hnswlib-node                │
│  @ruvector/gnn (optional)    │          │                              │
│  @ruvector/graph (optional)  │          │                              │
└──────────────────────────────┘          └──────────────────────────────┘
              │                                           │
              ▼                                           ▼
┌──────────────────────────────┐          ┌──────────────────────────────┐
│     Native Rust Bindings     │          │     Native C++ Bindings      │
│     (platform-specific)      │          │     (platform-specific)      │
├──────────────────────────────┤          ├──────────────────────────────┤
│  linux-x64-gnu               │          │  linux-x64                   │
│  linux-arm64-gnu             │          │  darwin-x64                  │
│  darwin-x64                  │          │  darwin-arm64                │
│  darwin-arm64                │          │  win32-x64                   │
│  win32-x64-msvc              │          │                              │
│  WASM (fallback)             │          │                              │
└──────────────────────────────┘          └──────────────────────────────┘
```

## Core Interfaces

### VectorBackend Interface

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

export interface VectorStats {
  count: number;
  dimension: number;
  metric: string;
  backend: 'ruvector' | 'hnswlib';
  memoryUsage: number;
  indexBuilt: boolean;
}

export interface VectorBackend {
  // Core operations
  insert(id: string, embedding: Float32Array, metadata?: Record<string, any>): void;
  insertBatch(items: Array<{ id: string; embedding: Float32Array; metadata?: Record<string, any> }>): void;
  search(query: Float32Array, k: number, options?: SearchOptions): SearchResult[];
  remove(id: string): boolean;

  // Index management
  buildIndex(): Promise<void>;
  saveIndex(path: string): Promise<void>;
  loadIndex(path: string): Promise<void>;

  // Stats and config
  getStats(): VectorStats;
  setEfSearch(ef: number): void;

  // Lifecycle
  close(): void;
}

export interface SearchOptions {
  threshold?: number;
  efSearch?: number;
  filter?: Record<string, any>;
  includeMetadata?: boolean;
}
```

### LearningBackend Interface (Optional)

```typescript
// packages/agentdb/src/backends/LearningBackend.ts

export interface LearningConfig {
  enabled: boolean;
  inputDim: number;
  outputDim?: number;
  heads?: number;
  learningRate?: number;
  batchSize?: number;
}

export interface TrainingSample {
  embedding: Float32Array;
  label: number;
  weight?: number;
}

export interface LearningBackend {
  // GNN operations
  enhance(query: Float32Array, neighbors: Float32Array[], weights: number[]): Float32Array;

  // Training
  addSample(sample: TrainingSample): void;
  train(options?: { epochs?: number }): Promise<TrainingResult>;

  // Persistence
  saveModel(path: string): Promise<void>;
  loadModel(path: string): Promise<void>;

  // Stats
  getStats(): LearningStats;
}

export interface TrainingResult {
  epochs: number;
  finalLoss: number;
  improvement: number;
  duration: number;
}

export interface LearningStats {
  enabled: boolean;
  samplesCollected: number;
  lastTrainingTime: number | null;
  modelVersion: number;
}
```

### GraphBackend Interface (Optional)

```typescript
// packages/agentdb/src/backends/GraphBackend.ts

export interface GraphBackend {
  // Cypher execution
  execute(cypher: string, params?: Record<string, any>): Promise<QueryResult>;

  // Node operations
  createNode(labels: string[], properties: Record<string, any>): Promise<string>;
  getNode(id: string): Promise<GraphNode | null>;
  deleteNode(id: string): Promise<boolean>;

  // Relationship operations
  createRelationship(from: string, to: string, type: string, properties?: Record<string, any>): Promise<string>;

  // Traversal
  traverse(startId: string, pattern: string, maxDepth?: number): Promise<GraphNode[]>;

  // Stats
  getStats(): GraphStats;
}
```

## Backend Detection & Initialization

### Auto-Detection Flow

```typescript
// packages/agentdb/src/backends/detector.ts

export type BackendType = 'ruvector' | 'hnswlib' | 'auto';

export interface DetectionResult {
  backend: 'ruvector' | 'hnswlib';
  features: {
    gnn: boolean;
    graph: boolean;
    compression: boolean;
  };
  platform: string;
  native: boolean;
}

export async function detectBackend(): Promise<DetectionResult> {
  // 1. Check for RuVector
  const ruvectorAvailable = await checkRuVector();

  if (ruvectorAvailable.available) {
    return {
      backend: 'ruvector',
      features: {
        gnn: ruvectorAvailable.gnn,
        graph: ruvectorAvailable.graph,
        compression: true
      },
      platform: process.platform + '-' + process.arch,
      native: ruvectorAvailable.native
    };
  }

  // 2. Fallback to hnswlib
  return {
    backend: 'hnswlib',
    features: {
      gnn: false,
      graph: false,
      compression: false
    },
    platform: process.platform + '-' + process.arch,
    native: await checkHnswlib()
  };
}

async function checkRuVector(): Promise<{
  available: boolean;
  native: boolean;
  gnn: boolean;
  graph: boolean;
}> {
  try {
    const core = await import('@ruvector/core');
    const native = core.isNative?.() ?? false;

    let gnn = false;
    try {
      await import('@ruvector/gnn');
      gnn = true;
    } catch {}

    let graph = false;
    try {
      await import('@ruvector/graph-node');
      graph = true;
    } catch {}

    return { available: true, native, gnn, graph };
  } catch {
    return { available: false, native: false, gnn: false, graph: false };
  }
}
```

### Initialization API

```typescript
// packages/agentdb/src/init.ts

export interface InitOptions {
  backend?: BackendType;
  dimension?: number;
  dbPath?: string;

  // RuVector-specific
  enableGNN?: boolean;
  enableGraph?: boolean;
  enableCompression?: boolean;

  // Performance tuning
  efConstruction?: number;
  efSearch?: number;
  M?: number;
  maxElements?: number;
}

export interface AgentDBInstance {
  backend: DetectionResult;
  vector: VectorBackend;
  learning?: LearningBackend;
  graph?: GraphBackend;
  db: Database;

  // High-level APIs
  reasoningBank: ReasoningBank;
  skillLibrary: SkillLibrary;
  causalMemory: CausalMemoryGraph;

  close(): Promise<void>;
}

export async function init(options: InitOptions = {}): Promise<AgentDBInstance> {
  const backendType = options.backend || 'auto';

  // Detect available backend
  const detection = await detectBackend();

  // Validate requested backend
  if (backendType === 'ruvector' && detection.backend !== 'ruvector') {
    throw new Error('RuVector requested but not available. Install with: npm install @ruvector/core');
  }

  // Create backend instance
  const vector = createVectorBackend(detection, options);
  const learning = detection.features.gnn ? createLearningBackend(options) : undefined;
  const graph = detection.features.graph ? createGraphBackend(options) : undefined;

  // Create SQLite database
  const db = createDatabase(options.dbPath);

  // Create high-level controllers
  const embeddingService = new EmbeddingService({ dimension: options.dimension || 384 });
  const reasoningBank = new ReasoningBank(db, embeddingService, vector, learning);
  const skillLibrary = new SkillLibrary(db, embeddingService, vector);
  const causalMemory = new CausalMemoryGraph(db);

  console.log(`[AgentDB] Initialized with ${detection.backend} backend`);
  if (detection.features.gnn) console.log('[AgentDB] GNN learning enabled');
  if (detection.features.graph) console.log('[AgentDB] Graph queries enabled');
  if (detection.features.compression) console.log('[AgentDB] Tiered compression enabled');

  return {
    backend: detection,
    vector,
    learning,
    graph,
    db,
    reasoningBank,
    skillLibrary,
    causalMemory,

    async close() {
      vector.close();
      db.close();
    }
  };
}
```

## CLI Commands

### agentdb init

```bash
# Auto-detect (recommended)
agentdb init

# Force specific backend
agentdb init --backend=ruvector
agentdb init --backend=hnswlib

# With options
agentdb init --backend=ruvector --enable-gnn --enable-compression

# Show detection info
agentdb init --dry-run
```

### Implementation

```typescript
// packages/agentdb/src/cli/commands/init.ts

import { Command } from 'commander';
import { init, detectBackend } from '../init.js';

export const initCommand = new Command('init')
  .description('Initialize AgentDB with optimal backend')
  .option('-b, --backend <type>', 'Backend: auto, ruvector, hnswlib', 'auto')
  .option('--enable-gnn', 'Enable GNN self-learning')
  .option('--enable-graph', 'Enable graph queries')
  .option('--enable-compression', 'Enable tiered compression')
  .option('-d, --dimension <number>', 'Vector dimension', '384')
  .option('-p, --path <path>', 'Database path', './agentdb')
  .option('--dry-run', 'Show detection without initializing')
  .action(async (options) => {
    if (options.dryRun) {
      const detection = await detectBackend();
      console.log('\n📊 Backend Detection Results:\n');
      console.log(`  Backend:     ${detection.backend}`);
      console.log(`  Platform:    ${detection.platform}`);
      console.log(`  Native:      ${detection.native ? '✅' : '❌ (using WASM)'}`);
      console.log(`  GNN:         ${detection.features.gnn ? '✅' : '❌'}`);
      console.log(`  Graph:       ${detection.features.graph ? '✅' : '❌'}`);
      console.log(`  Compression: ${detection.features.compression ? '✅' : '❌'}`);
      return;
    }

    console.log('🚀 Initializing AgentDB...\n');

    const instance = await init({
      backend: options.backend,
      dimension: parseInt(options.dimension),
      dbPath: options.path,
      enableGNN: options.enableGnn,
      enableGraph: options.enableGraph,
      enableCompression: options.enableCompression
    });

    console.log('\n✅ AgentDB initialized successfully!');
    console.log(`   Backend: ${instance.backend.backend}`);
    console.log(`   Path: ${options.path}`);

    await instance.close();
  });
```

## File Structure

```
packages/agentdb/
├── src/
│   ├── backends/
│   │   ├── index.ts              # Backend exports
│   │   ├── VectorBackend.ts      # Interface definition
│   │   ├── LearningBackend.ts    # GNN interface
│   │   ├── GraphBackend.ts       # Graph interface
│   │   ├── detector.ts           # Auto-detection logic
│   │   ├── factory.ts            # Backend factory
│   │   ├── ruvector/
│   │   │   ├── RuVectorBackend.ts
│   │   │   ├── RuVectorLearning.ts
│   │   │   └── RuVectorGraph.ts
│   │   └── hnswlib/
│   │       └── HNSWLibBackend.ts
│   ├── controllers/
│   │   ├── ReasoningBank.ts      # Updated for backend abstraction
│   │   ├── SkillLibrary.ts
│   │   ├── CausalMemoryGraph.ts
│   │   └── ...
│   ├── cli/
│   │   ├── agentdb-cli.ts
│   │   └── commands/
│   │       ├── init.ts
│   │       ├── benchmark.ts
│   │       └── ...
│   ├── init.ts                   # Main initialization
│   ├── config.ts                 # Configuration
│   └── index.ts                  # Public exports
├── tests/
│   ├── backends/
│   │   ├── ruvector.test.ts
│   │   ├── hnswlib.test.ts
│   │   └── detector.test.ts
│   ├── integration/
│   │   └── backend-parity.test.ts
│   └── benchmarks/
│       └── backend-comparison.bench.ts
└── package.json
```

## Configuration

### Environment Variables

```bash
# Backend selection
AGENTDB_BACKEND=auto|ruvector|hnswlib

# Feature flags
AGENTDB_GNN_ENABLED=true|false
AGENTDB_GRAPH_ENABLED=true|false
AGENTDB_COMPRESSION_ENABLED=true|false

# Performance tuning
AGENTDB_EF_CONSTRUCTION=200
AGENTDB_EF_SEARCH=100
AGENTDB_M=16
AGENTDB_MAX_ELEMENTS=100000

# Paths
AGENTDB_PATH=./agentdb
AGENTDB_INDEX_PATH=./agentdb/index

# Logging
AGENTDB_LOG_LEVEL=info|debug|warn|error
```

### Config File (agentdb.config.json)

```json
{
  "backend": "auto",
  "dimension": 384,
  "dbPath": "./agentdb",

  "ruvector": {
    "enableGNN": true,
    "enableGraph": false,
    "enableCompression": true,
    "gnn": {
      "heads": 4,
      "learningRate": 0.001,
      "trainInterval": 3600
    },
    "compression": {
      "hotThreshold": 0.8,
      "warmThreshold": 0.4,
      "coolThreshold": 0.1
    }
  },

  "hnswlib": {
    "efConstruction": 200,
    "efSearch": 100,
    "M": 16
  },

  "logging": {
    "level": "info",
    "file": "./agentdb/logs/agentdb.log"
  }
}
```
